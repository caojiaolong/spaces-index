# 为什么Adam的Update RMS是0\.2？

> 作者：苏剑林 · 科学空间 · 2025-09-02
>
> 原文：<https://spaces.ac.cn/archives/11267>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，我们很早就开始尝试将Muon用于大规模LLM的训练。特别地，在[《Muon续集：为什么我们选择尝试Muon？》](<https://spaces.ac.cn/archives/10739>)中，我们提出了“Match Adam Update RMS”的技巧，以便快速从Adam迁移到Muon上，这个技巧同样用到了Kimi K2的训练中。该技巧是指将Muon的Update RMS统一成0\.2，这使得我们复用Adam的学习率和权重衰减率。

这一技巧的背后，是我们观察到Adam的Update RMS约等于0\.2，并且这一现象是稳定且可复现的。这便引发了一个有趣的问题：为什么Adam的Update RMS是0\.2？我们可以从理论上解释它吗？

## 问题引入

首先描述一下现象：从实验中我们观察到，大致上在Warmup结束、模型进入正式训练后，Adam的Update RMS几乎都保持在0\.2～0\.3之间，并且不同尺寸的模型也呈现出相似的规律。这些模型的共同点是都用Adam训练，参数是$\beta_1=0.9,\beta_2=0.95$。由于共性很明显，所以这大概率不是巧合，因此笔者尝试分析背后的原理。

然后我们回顾一下Adam优化器的形式：
\begin{equation}\text{Adam}\color{skyblue}{\text{W}}:=\left\{\begin{aligned}
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\
&\boldsymbol{u}_t =\hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.\\
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})
\end{aligned}\right.\end{equation}
注意：本文所有向量的乘除法，包括平方，默认都是指Hadamard积/商，即Element\-wise的乘/除。

我们要做的事情，就是证明$\Vert\boldsymbol{u}_t\Vert_{RMS}\approx 0.2$，至少在$\beta_1=0.9,\beta_2=0.95$这组设置下如此。我们假设$\epsilon$足够小，以至于可以忽略它，并且我们考虑$t\to \infty$的稳态，那么$\beta_1^t$、$\beta_2^t$都足够接近于零，所以不用区分$\boldsymbol{m}_t$和$\hat{\boldsymbol{m}}_t$、$\boldsymbol{v}_t$和$\hat{\boldsymbol{v}}_t$，由此有$\boldsymbol{u}_t =\boldsymbol{m}_t/\sqrt{\boldsymbol{v}_t}$。

对于$\boldsymbol{m}_t,\boldsymbol{v}_t$，我们可以得到展开式
\begin{equation}\boldsymbol{m}_t = (1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i}\boldsymbol{g}_i,\qquad \boldsymbol{v}_t = (1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i}\boldsymbol{g}_i^2\end{equation}

## 数值模拟

如果我们假设$\boldsymbol{g}_1,\boldsymbol{g}_2,\cdots,\boldsymbol{g}_t$都是从同一个分布采样出来的，那么我们就可以直接用数值模拟的方法估计$\Vert\boldsymbol{u}_t\Vert_{RMS}$。事不宜迟，让我们从最简单的标准正态分布$\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$进行尝试，参考代码如下：



<pre><code class="language-python">import numpy as np

N, T = 10000, 2000
beta1, beta2 = 0.9, 0.95
m, v = 0, 0
for t in range(1, T + 1):
    g = np.random.randn(N)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    u = m / v**0.5

rms = (u**2).mean()**0.5
print(rms)</code></pre>



大家猜猜结果是多少？答案大概是0\.225，居然跟实验结果惊人地相似！这反过来表明我们的模拟假设跟实际情况还是很吻合的。可能有读者觉得不对，$\boldsymbol{g}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$不是纯噪声了吗，这还能吻合？实际训练当然不可能是纯噪声，只能说单次梯度的信噪比小得可怜，因此可以用纯噪声来模拟。

读者可以自行折腾一下上述参考代码，观察Update RMS的影响变量，大体结论是：Update RMS跟$\beta_1$正相关，跟$\beta_2$似乎关系不大，如果$\boldsymbol{g}$的分布具有非零均值（相当于增大梯度的信噪比），那么Update RMS也会变大。

## 平均近似

这一节笔者尝试从理论方面推导上述模拟结果的一个近似解析解。首先，我们从RMS的定义可知，要求$\Vert\boldsymbol{u}_t\Vert_{RMS}$，需要先求$\boldsymbol{u}_t^2 = \boldsymbol{m}_t^2/\boldsymbol{v}_t$。笔者的想法是，用$\boldsymbol{u}_t^2$的期望作为它的近似，并进一步转化为平均场近似：
\begin{equation}\mathbb{E}[\boldsymbol{u}_t^2] = \mathbb{E}\left[\frac{\boldsymbol{m}_t^2}{\boldsymbol{v}_t}\right] \approx \frac{\mathbb{E}[\boldsymbol{m}_t^2]}{\mathbb{E}[\boldsymbol{v}_t]}\end{equation}
可能会有读者质疑最后一步近似的合理性。笔者的建议是，先不管这些细枝末节，就好比上一节假设$\boldsymbol{g}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$一样，先算了再说，如果结果合理那么过程必然一定程度上也是合理的。现在我们分别算分子、分母，这次我们一般地设$\mathbb{E}[\boldsymbol{g}]=\boldsymbol{\mu},\mathbb{E}[\boldsymbol{g}^2]=\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2 $，其中分母比较简单
\begin{equation}\begin{aligned}
\mathbb{E}[\boldsymbol{v}_t] =&\, (1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i}\mathbb{E}[\boldsymbol{g}_i^2] \\
=&\, (1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i}(\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2) \\
=&\, (1 - \beta_2^t) (\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2) \\[5pt]
\approx &\, \boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2 \qquad(t\to\infty)
\end{aligned}\end{equation}
至于分子，可以直接展开平方计算，也可以稍微偷懒一下：我们要求的是$\boldsymbol{m}_t$的二阶矩$\mathbb{E}[\boldsymbol{m}_t^2]$，它又等于$\mathbb{E}[\boldsymbol{m}_t]^2 + \mathbb{V}ar[\boldsymbol{m}_t]$。$\mathbb{E}[\boldsymbol{m}_t]$的计算跟$\mathbb{E}[\boldsymbol{m}_t]$类似，结果是$(1 - \beta_1^t)\boldsymbol{\mu}\approx\boldsymbol{\mu}$；至于方差，它具有平方可加性，因此
\begin{equation}\mathbb{V}ar[\boldsymbol{m}_t] = (1 - \beta_1)^2\sum_{i=1}^t \beta_1^{2(t-i)}\boldsymbol{\sigma}^2 = \frac{(1 - \beta_1)^2 (1 - \beta_1^{2t})}{1 - \beta_1^2}\boldsymbol{\sigma}^2\approx \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\sigma}^2\qquad (t\to\infty)\end{equation}
所以
\begin{equation}\mathbb{E}[\boldsymbol{u}_t^2]\approx \frac{\boldsymbol{\mu}^2 + \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\sigma}^2}{\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2}\end{equation}

## 结果分析

由于$\mathbb{E}[\boldsymbol{u}_t^2]$已经是平方后的向量，所以为了估计$\Vert\boldsymbol{u}_t\Vert_{RMS}$，我们只需要对各个分量求平均然后开平方。求平均这一步，我们不妨再来一次平均场近似（分子分母分别求平均），最终将得到
\begin{equation}\Vert\boldsymbol{u}_t\Vert_{RMS} \approx \sqrt{\frac{\Vert\boldsymbol{\mu}\Vert^2 + \frac{1 - \beta_1}{1 + \beta_1}\Vert\boldsymbol{\sigma}\Vert^2}{\Vert\boldsymbol{\mu}\Vert^2 + \Vert\boldsymbol{\sigma}\Vert^2}} = \sqrt{\frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + \frac{1 - \beta_1}{1 + \beta_1}}{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + 1}}\label{eq:mean-field}\end{equation}
它有两个影响因子：一是$\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2$，这可以看成是梯度的信噪比（SNR）；二是$\beta_1$，这是Adam的超参数之一。特别地，结果不依赖于$\beta_2$，这跟前面的模拟结果吻合。那么这个式子究竟近似得好不好呢？我们不妨考虑最简单的特例$\boldsymbol{\mu}=\boldsymbol{0}$，此时
\begin{equation}\Vert\boldsymbol{u}_t\Vert_{RMS} \approx  \sqrt{\frac{1 - \beta_1}{1 + \beta_1}}\end{equation}
代入$\beta_1=0.9$，结果是$0.2294\cdots$，跟模拟结果和实践表现居然都很吻合！进一步地，它跟模拟结果的多个对比如下：

[![模拟结果与平均场近似（不同beta1、beta2）](<https://spaces.ac.cn/usr/uploads/2025/09/1430078993.svg>)](<https://spaces.ac.cn/usr/uploads/2025/09/1430078993.svg>)

模拟结果与平均场近似（不同beta1、beta2）

应该说，整体的近似程度还是不错的，特别是$\beta_2 \geq 0.9$之后，结果几乎跟平均场近似重合了（经 [@EIFY](<https://x.com/EIFY/status/1965888629814988984>) 提醒，论文[《Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks》](<https://papers.cool/arxiv/2305.17212>)也曾得到过相同的计算结果）。

至于考虑SNR的比较结果如下：

[![模拟结果与平均场近似（不同beta1、SNR）](<https://spaces.ac.cn/usr/uploads/2025/09/1584329510.svg>)](<https://spaces.ac.cn/usr/uploads/2025/09/1584329510.svg>)

模拟结果与平均场近似（不同beta1、SNR）

当信噪比增大时，平均场近似的误差开始变大，不过仍旧能预测一个整体趋势。事实上，实际训练中梯度的信噪比很少机会能有接近1这么大，因此依然可以认为平均场是一个良好近似。

## 反向预测

如果我们已经接受平均场近似$\eqref{eq:mean-field}$，那么可以反过来用它估算梯度的信噪比：
\begin{equation}\frac{\Vert\boldsymbol{\mu}\Vert^2}{\Vert\boldsymbol{\sigma}\Vert^2} \approx \frac{\Vert\boldsymbol{u}_t\Vert_{RMS}^2 - \frac{1 - \beta_1}{1 + \beta_1}}{1 - \Vert\boldsymbol{u}_t\Vert_{RMS}^2}\end{equation}
在实际训练中，$\beta_1$是给定的，$\Vert\boldsymbol{u}_t\Vert_{RMS}$（也就是Adam的Update RMS）也是可以直接估算的，所以上式是可计算的。当然，这个式子只对Adam适用，有没有更一般的估计思路呢？还真有！别忘了前面我们估计得到
\begin{equation}\mathbb{E}[\boldsymbol{m}_t^2]\approx \boldsymbol{\mu}^2 + \frac{1 - \beta_1}{1 + \beta_1}\boldsymbol{\sigma}^2\end{equation}
那么对它的分量求和然后开平方，我们认为它会是$\Vert\boldsymbol{m}_t\Vert$的一个近似：
\begin{equation}\Vert\boldsymbol{m}_t\Vert\approx \sqrt{\Vert\boldsymbol{\mu}\Vert^2 + \frac{1 - \beta_1}{1 + \beta_1}\Vert\boldsymbol{\sigma}\Vert^2}\end{equation}
至于二阶矩是$\mathbb{E}[\boldsymbol{v}_t]\approx \boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2$，而像Muon之类的优化器并没有二阶矩可用，但是我们留意到二阶矩的结果是跟$\beta_2$无关的，所以我们不妨考虑一个最简单的特例——$\beta_2=0$——此时$\boldsymbol{v}_t=\boldsymbol{g}_t^2$。当然这可能有点勉强，但估算嘛肯定是怎么方便怎么来。这个“近似”意味着成立$\Vert\boldsymbol{g}_t\Vert^2\approx \Vert\boldsymbol{\mu}\Vert^2 + \Vert\boldsymbol{\sigma}\Vert^2$，于是我们有
\begin{equation}\frac{\Vert\boldsymbol{m}_t\Vert}{\Vert\boldsymbol{g}_t\Vert}\approx \sqrt{\frac{\Vert\boldsymbol{\mu}\Vert^2 + \frac{1 - \beta_1}{1 + \beta_1}\Vert\boldsymbol{\sigma}\Vert^2}{\Vert\boldsymbol{\mu}\Vert^2 + \Vert\boldsymbol{\sigma}\Vert^2}}\end{equation}
右端的形式跟式$\eqref{eq:mean-field}$如出一辙，所以我们可以写出
\begin{equation}\frac{\Vert\boldsymbol{\mu}\Vert^2}{\Vert\boldsymbol{\sigma}\Vert^2} \approx \frac{\Vert\boldsymbol{m}_t\Vert^2/\Vert\boldsymbol{g}_t\Vert^2 - \frac{1 - \beta_1}{1 + \beta_1}}{1 - \Vert\boldsymbol{m}_t\Vert^2/\Vert\boldsymbol{g}_t\Vert^2}\end{equation}
也就是用$\Vert\boldsymbol{m}_t\Vert/\Vert\boldsymbol{g}_t\Vert$替代$\Vert\boldsymbol{u}_t\Vert_{RMS}$，这就给出了一种带动量优化器通用的估计$\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2$的思路。可能还有读者想问动量都没有咋办？这就真没有办法了，因为这里的$\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2$属于跨优化轨迹的统计量，我们总得有些跨轨迹的统计信息，才有可能去估计它。

## 文章小结

本文主要从模拟实验和理论近似两个角度探讨了Adam的Update RMS，它可以作为我们在Muon优化器中将Update RMS对齐到0\.2的理论依据之一。
