# 生成扩散模型漫谈（十八）：得分匹配 = 条件得分匹配

> 作者：苏剑林 · 科学空间 · 2023-02-28
>
> 原文：<https://spaces.ac.cn/archives/9509>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在前面的介绍中，我们多次提及“得分匹配”和“条件得分匹配”，它们是扩散模型、能量模型等经常出现的概念，特别是很多文章直接说扩散模型的训练目标是“得分匹配”，但事实上当前主流的扩散模型如DDPM的训练目标是“条件得分匹配”才对。

那么“得分匹配”与“条件得分匹配”具体是什么关系呢？它们两者是否等价呢？本文详细讨论这个问题。

## 得分匹配

首先，得分匹配（Score Matching）是指训练目标：
\begin{equation}\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right]\label{eq:sm}\end{equation}
其中$\boldsymbol{\theta}$是训练参数。很明显，得分匹配是想学习一个模型$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$来逼近$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$，这里的$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$我们就称为“得分”。

在扩散模型场景，$p_t(\boldsymbol{x}_t)$由下式给出：
\begin{equation}p_t(\boldsymbol{x}_t) = \int p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)p_0(\boldsymbol{x}_0)d\boldsymbol{x}_0 = \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)}\left[p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]\label{eq:pt}\end{equation}
其中$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$一般都是已知概率密度解析式的简单分布（如条件正态分布），$p_0(\boldsymbol{x}_0)$也是给定的分布，但一般代表训练数据，也就是说我们只能从$p_0(\boldsymbol{x}_0)$中采样，但不知道$p_0(\boldsymbol{x}_0)$的具体表达式。

根据式$\eqref{eq:pt}$，我们可以推导得
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) =&\, \frac{\nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t)}{p_t(\boldsymbol{x}_t)} \\
=&\, \frac{\int p_0(\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)d\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)} \\
=&\, \frac{\int p_0(\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)d\boldsymbol{x}_0}{\int p_0(\boldsymbol{x}_0) p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)d\boldsymbol{x}_0} \\
=&\, \frac{\mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)}\left[\nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]}{\mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)}\left[p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]} \\
\end{aligned}\label{eq:score-1}\end{equation}
根据我们的假设，$\nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$和$p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$都是已知解析式的，因此理论上我们可以通过采样$\boldsymbol{x}_0$来估计$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$，但由于这里涉及到两个期望的除法，是一个有偏估计（参考[《简述无偏估计和有偏估计》](<https://spaces.ac.cn/archives/6747>)），因此需要采样足够多的点才能做出比较准确的估计，因此直接用式$\eqref{eq:sm}$作为训练目标的话，需要较大的batch\_size才有比较好的效果。

## 条件得分

实际上，一般扩散模型所用的训练目标是“条件得分匹配（Conditional Score Matching）”：
\begin{equation}\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t\sim p_0(\boldsymbol{x}_0)p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right]\end{equation}
根据假设，$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)$是已知解析式的，因此上述目标是直接可用的，每次采样一对$(\boldsymbol{x}_0,\boldsymbol{x}_t)$进行估算。特别地，这是一个无偏估计，这意味着它不是特别依赖于大batch\_size，因此是一个比较实用的训练目标。

为了分析“得分匹配”与“条件得分匹配”的关系，我们还需要$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$的另一个恒等式：
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) =&\, \frac{\nabla_{\boldsymbol{x}_t}p_t(\boldsymbol{x}_t)}{p_t(\boldsymbol{x}_t)} \\
=&\, \frac{\int p_0(\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t} p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)d\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)} \\
=&\, \frac{\int p_0(\boldsymbol{x}_0)p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) d\boldsymbol{x}_0}{p_t(\boldsymbol{x}_t)} \\
=&\, \int p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) d\boldsymbol{x}_0 \\
=&\, \mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\nabla_{\boldsymbol{x}_t} \log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right] \\
\end{aligned}\label{eq:score-2}\end{equation}

## 不等关系

首先，我们可以很快速地证明两者之间的第一个结果：条件得分匹配是得分匹配的一个上界。这也就意味着最小化条件得分匹配，某种程度上也在最小化得分匹配。

证明并不困难，之前我们在[《生成扩散模型漫谈（十六）：W距离 ≤ 得分匹配》](<https://spaces.ac.cn/archives/9467>)就已经证明过：
\begin{equation}\begin{aligned}
&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\
=&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right] - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\
\leq &\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\
= &\,\mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\
\end{aligned}\end{equation}
第一个等号是因为恒等式$\eqref{eq:score-2}$，第二个不等号则是因为平方平均不等式的推广或者詹森不等式，第三个等号则是贝叶斯公式了。

## 等价关系

前两天，在微信群里大家讨论到得分匹配的时候，有群友指点到：条件得分匹配与得分匹配之差是一个跟优化无关的常数，所以两者实际上是完全等价的！刚听到这个结论的时候笔者也相当惊讶，两者居然还是等价关系，而不单单是上下界关系。不仅如此，笔者尝试证明了一下后，发现证明过程居然也很简单！

首先，关于得分匹配，我们有
\begin{equation}\begin{aligned}
&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\
=&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\color{orange}{\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right\Vert^2} + \color{red}{\left\Vert\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2} - \color{green}{2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\cdot\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)}\right] \\
\end{aligned}\end{equation}
然后，关于条件得分匹配，我们有
\begin{equation}\begin{aligned}
&\,\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t\sim p_0(\boldsymbol{x}_0)p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0) - \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2\right] \\[5pt]
=&\,\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t\sim p_0(\boldsymbol{x}_0)p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2 + \left\Vert\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2 - 2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\cdot\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right] \\[5pt]
=&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t),\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2 + \left\Vert\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2 - 2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\cdot\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right] \\[5pt]
=&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[{\begin{aligned}&\color{orange}{\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right]} + \color{red}{\left\Vert\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2} \\
&\qquad\qquad- \color{green}{2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\cdot\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right]}\end{aligned}}\right] \\[5pt]
=&\,\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\color{orange}{\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right]} + \color{red}{\left\Vert\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\right\Vert^2}  - \color{green}{2\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\cdot\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)}\right] \\
\end{aligned}\end{equation}
两者作差，可以发现结果是
\begin{equation}\mathbb{E}_{\boldsymbol{x}_t\sim p_t(\boldsymbol{x}_t)}\left[\color{orange}{\mathbb{E}_{\boldsymbol{x}_0\sim p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)}\left[\left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right] - \left\Vert\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)\right\Vert^2}\right]\end{equation}
它跟参数$\boldsymbol{\theta}$无关。所以最小化得分匹配目标，跟最小化条件得分匹配，在理论上是等价的。根据群友介绍，相关结果首次出现在文章[《A Connection Between Score Matching and Denoising Autoencoders》](<https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf>)中。

既然两者理论上等价，这是不是意味着我们前面的“得分匹配”比“条件得分匹配”需要更大batch\_size的说法不成立？并不是。如果还是直接从式$\eqref{eq:score-1}$来估计$\nabla_{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t)$然后做得分匹配，那么结果确实还是有偏的，依赖于大batch\_size；而当我们将目标$\eqref{eq:sm}$展开进一步化简后，已经逐步将有偏估计转化为无偏估计了，这时候不会太依赖于batch\_size。也就是说，虽然两个目标从理论上是等价的，从统计量的角度，属于不同性质的统计量，它们的等价仅仅是在采样样本数趋于无穷时的精确等价。

## 文章小结

本文主要分析“得分匹配”和“条件得分匹配”两个训练目标之间的关联。
