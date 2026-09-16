# 为什么梯度裁剪的默认模长是1？

> 作者：苏剑林 · 科学空间 · 2025-01-02
>
> 原文：<https://spaces.ac.cn/archives/10657>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们知道，梯度裁剪（Gradient Clipping）是让模型训练更加平稳的常用技巧。常用的梯度裁剪是根据所有参数的梯度总模长来对梯度进行裁剪，其运算可以表示为
\begin{equation}\text{clip}(\boldsymbol{g},\tau)=\left\{\begin{aligned}&\boldsymbol{g}, &\Vert\boldsymbol{g}\Vert\leq \tau \\
&\frac{\tau}{\Vert\boldsymbol{g}\Vert}\boldsymbol{g},&\Vert\boldsymbol{g}\Vert > \tau
\end{aligned}\right.\end{equation}
这样一来，$\text{clip}(\boldsymbol{g},\tau)$保持跟$\boldsymbol{g}$相同的方向，但模长不超过$\tau$。注意这里的$\Vert\boldsymbol{g}\Vert$是整个模型所有的参数梯度放在一起视为单个向量所算的模长，也就是所谓的Global Gradient Norm。

不知道大家有没有留意到一个细节：不管是数百万参数还是数百亿参数的模型，$\tau$的取值在很多时候都是1。这意味着什么呢？是单纯地复用默认值，还是背后隐含着什么深刻的原理呢？

## 是什么

可能有读者觉得，默认值又不一定是最优值，有什么值得纠结的？确实，$\tau=1$未必是最优的选择，但它是很多模型的默认选择，并且在这个默认选择下表现尚可，这反过来表明$\tau=1$具有普遍的合理性。

这里的“合理性”又指什么呢？让我们回到$\text{clip}$运算上。如果$\Vert\boldsymbol{g}\Vert$总是小于$\tau$，那么$\text{clip}$就退化为恒等变换了；如果$\Vert\boldsymbol{g}\Vert$总是大于$\tau$，那么$\text{clip}$就退化成L2归一化。换句话说，$\text{clip}$之所以为$\text{clip}$，就是因为$\tau$产生了适当的区分度，使得大部分的$\Vert\boldsymbol{g}\Vert$都是小于$\tau$的，只有小部分才是大于$\tau$的，这就是$\tau$的合理性的含义。

这个当然可以举出反例，而且还不少，这里主要想强调这个现象的普遍性以及这个默认设置的普适性，所以较真的读者大可不必过于执着于个别细节。

因此，我们认为，$\tau=1$的普遍合理性的含义，就是不论模型参数量多少、怎么初始化、取何种损失函数，它的梯度总模长都能恰好大致以$1$为“异常值”的分界点，这无疑是一个非常不可思议的性质——笔者第一次意识到这个结论时的感受便是如此。

## 为什么

为什么会如此“巧合”呢？笔者的答案可能会让人有些意外：因为只有这样，模型才有稳定训练的可能。

让我们考虑损失函数$\mathcal{L}(\boldsymbol{\theta})$，优化器更新规则为$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\, \boldsymbol{u}_t$，那么损失函数的变化近似为
\begin{equation}\Delta \mathcal{L} = \mathcal{L}(\boldsymbol{\theta}_{t+1}) - \mathcal{L}(\boldsymbol{\theta}_t) \approx (\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t)\cdot\nabla_{\boldsymbol{\theta}_t}\mathcal{L}(\boldsymbol{\theta}) = -\eta\, \boldsymbol{u}_t\cdot \boldsymbol{g}_t\end{equation}
先考虑最简单的SGD，那么$\boldsymbol{u}_t = \boldsymbol{g}_t$以及$\Delta \mathcal{L}=-\eta\Vert\boldsymbol{g}_t\Vert^2$，即损失函数的变化量正比于梯度模长的平方。我们知道，不管是CV还是NLP，纯粹的SGD（不带动量）都是非常低效的优化器，训练到中后期，平均来说多数任务每步的损失下降量是远不如学习率大小的，也就是$|\Delta \mathcal{L}| < \eta$，由此推得$\Vert\boldsymbol{g}_t\Vert < 1$。这就表明了$\Vert\boldsymbol{g}_t\Vert < 1$是一个能正常收敛的模型的长期表现。

当然，训练初期模型有可能会出现$\Vert\boldsymbol{g}_t\Vert > 1$，这是正常的，但很少情况会出现$\Vert\boldsymbol{g}_t\Vert \gg 1$，或者说一个优秀的初始化应该避免出现$\Vert\boldsymbol{g}_t\Vert \gg 1$，像[DeepNorm](<https://spaces.ac.cn/archives/8978>)等的理论依据便是如此。原因是相似的，如果梯度模长太大，那么前期的学习就会过于“激进”，导致提前收敛到不好的局部解。另一个方案是缩小$\eta$，这同样能够缩小$|\Delta \mathcal{L}|$，这也就是为什么在训练初期我们通常使用Warmup。

顺便说，关于Warmup的理解大家可以参考论文[《Optimal Linear Decay Learning Rate Schedules and Further Refinements》](<https://papers.cool/arxiv/2310.07831>)，这是笔者认为对Warmup的最合理的分析。

## 怎么办

简单来说，就是由于损失函数的变化量正比于梯度模长的平方，所以训练的平稳性决定了梯度模长不能太大，并且长期表现为小于1。而初期如果出现明显大于1的梯度模长，那么通常的策略是Warmup。或者也可以考虑一个更通用的策略：设置另一个阈值$\mathcal{T}$，根据$\boldsymbol{u}_t\cdot \boldsymbol{g}_t$的值对$\eta$进行裁剪
\begin{equation}\eta_t = \left\{\begin{aligned}&\eta,& \boldsymbol{u}_t\cdot \boldsymbol{g}_t\leq \mathcal{T} \\ &\frac{\mathcal{T}}{\boldsymbol{u}_t\cdot \boldsymbol{g}_t}\eta,& \boldsymbol{u}_t\cdot \boldsymbol{g}_t > \mathcal{T}
\end{aligned}\right.\end{equation}
这样就免除了额外的Warmup设置，更加具有自适应性。

对于Adam等优化器，我们可以跟[《当Batch Size增大时，学习率该如何随之变化？》](<https://spaces.ac.cn/archives/10542>)一样，通过$\boldsymbol{u}_t=\text{sign}(\boldsymbol{g}_t)$来进行近似分析，此时
\begin{equation}\Delta \mathcal{L} = -\eta\, \text{sign}(\boldsymbol{g}_t)\cdot \boldsymbol{g}_t = -\eta\, \Vert\boldsymbol{g}_t\Vert_1\end{equation}
这里的$\Vert\Vert_1$是L1范数，即分量的绝对值之和。由于梯度分量基本都小于1，因此$\Vert\boldsymbol{g}_t\Vert_1 \gg \Vert\boldsymbol{g}_t\Vert$，因此同样出于平稳训练的需求，Adam的学习率通常要明显小于SGD的学习率。此外，上式还可以改写成
\begin{equation}\Delta \mathcal{L} = -\eta\, \text{sign}(\boldsymbol{g}_t)\cdot \boldsymbol{g}_t = -\eta\, \sqrt{N}\Vert\boldsymbol{g}_t\Vert \cos(\text{sign}(\boldsymbol{g}_t), \boldsymbol{g}_t) \end{equation}
这里假设了$\boldsymbol{g}_t$没有零分量，因此$\Vert\text{sign}(\boldsymbol{g}_t)\Vert=\sqrt{N}$，$N$是模型总参数量。实践发现$\Vert\boldsymbol{g}_t\Vert$和$\cos(\text{sign}(\boldsymbol{g}_t), \boldsymbol{g}_t)$在不同的模型尺度下都大致为常数，因此如果要维持$\Delta \mathcal{L}$不变，应该有$\eta$反比于$\sqrt{N}$，也就是说模型参数量增加到4倍，那么学习率可以考虑减半。

## 全剧终

本文对“梯度裁剪的默认模长为1”这一现象给出了自己的一些看法和思考。
