# 配置不同的学习率，LoRA还能再涨一点？

> 作者：苏剑林 · 科学空间 · 2024-02-27
>
> 原文：<https://spaces.ac.cn/archives/10001>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

LoRA（Low\-Rank Adaptation）是当前LLM的参数高效微调手段之一，此前我们在[《梯度视角下的LoRA：简介、分析、猜测及推广》](<https://spaces.ac.cn/archives/9590>)也有过简单讨论。这篇文章我们来学习LoRA的一个新结论：

> <strong>给LoRA的两个矩阵分配不同的学习率，LoRA的效果还能进一步提升。</strong>

该结论出自最近的论文[《LoRA\+: Efficient Low Rank Adaptation of Large Models》](<https://papers.cool/arxiv/2402.12354>)（下称“LoRA\+”）。咋看之下，该结论似乎没有什么特别的，因为配置不同的学习率相当于引入了新的超参数，通常来说只要引入并精调超参数都会有提升。“LoRA\+”的特别之处在于，它从理论角度肯定了这个必要性，并且断定最优解必然是右矩阵的学习率大于左矩阵的学习率。简而言之，“LoRA\+”称得上是理论指导训练并且在实践中确实有效的经典例子，值得仔细学习一番。

## 结论简析

假设预训练参数为$W_0 \in \mathbb{R}^{n\times m}$，如果使用全量参数微调，那么增量也是一个$n\times m$矩阵。为了降低参数量，LoRA将更新量约束为低秩矩阵，即设$W=W_0 + AB$，其中$A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m}$以及有$r\ll \min(n,m)$，用新的$W$替换模型原有参数，然后固定$W_0$不变，训练的时候只更新$A,B$，如下图所示：
$$\style{display: inline-block; width: 24ex; padding: 10ex 0; border: 1px solid #6C8EBF; background-color: #DAE8FC}{W_0\in\mathbb{R}^{n\times m}} \quad + \quad \style{display: inline-block; width: 8ex; padding: 10ex 0; border: 1px solid #D79B00; background-color: #FFE6CC}{A\in\mathbb{R}^{n\times r}}\quad\times\quad \style{display: inline-block; width: 24ex; padding: 3ex 0; border: 1px solid #D79B00; background-color: #FFE6CC}{B\in\mathbb{R}^{r\times m}}$$

注意LoRA通常都是用于Dense层，但原论文的分析是基于权重左乘输入的，而实现中基本上都是输入右乘权重，为了避免理解上的困难，本文的记号跟实现对齐，即假设层的输入是$X\in\mathbb{R}^{b\times n}$，层的运算是$XW = X(W_0 + AB)$。由于“LoRA\+”的结论跟预训练权重无关，因此不失一般性可以设$W_0=0$，那么层运算简化为$Y=XAB\in\mathbb{R}^{b\times m}$。

“LoRA\+”的结论是：

> 为了使LoRA的效果尽可能接近最优，权重$B$的学习率应该要大于权重$A$的学习率。

注意，为了使初始模型等价于原始预训练模型，LoRA通常会将$A,B$之一全零初始化。笔者一开始以为，该结论是由于全零初始化导致的，所以应该依赖于全零初始化的位置，但仔细阅读后发现，“LoRA\+”所声称的结论跟全零初始化无关，也就是说，表面上$A,B$是对称的，但实际上它们有着固有的不对称性，以至于不管选择$A$还是$B$来全零初始化，结论都是$B$的学习率要大于$A$。这就有意思起来了。

然而，不得不说的是“LoRA\+”原论文的讲解写得相当让人费解，所以下面都是笔者用自己的思路尽量简化后的推导。大体上，它基于两点假设：

> 1、<strong>数值稳定</strong>：模型每一层的输出值都应该是数值稳定的，跟网络宽度无关；
> 
> 
> 
> 2、<strong>贡献相当</strong>：为了使LoRA最优，$A,B$两个矩阵对效果应该有同等程度的贡献。

接下来我们逐一分析并量化这两点假设。

## 数值稳定

首先，数值稳定说的是$X,XA,XAB$的每个分量都应该是$\mathcal{O}(1)$级别的，而不依赖于网络宽度$n,m$，这里的$\mathcal{O}(1)$主要描述的是它关于网络宽度的阶是零阶，并不代表它的绝对值就接近于1。这个假设应该没有什么争议，很难想象一个数值不稳定的网络能够能有好的预测效果。不过有些读者可能会质疑“$XA$是$\mathcal{O}(1)$”的必要性，因为$X$是输入、$XAB$是输出，要求它俩的数值稳定性很合理，但$XA$只是中间变量，它也必须数值稳定吗？

单看前向传播来说，$XA$的数值稳定性确实不是必要的。但如果$XA$数值不稳定同时$XAB$数值稳定的话，那么有两种情况：$XA$数值偏大、$B$数值偏小，根据求导公式，这将导致$A$的梯度偏小、$B$的梯度偏大；反过来，$XA$数值偏小、$B$数值偏大，这将导致$A$的梯度偏大、$B$的梯度偏小。总而言之，$XA$的数值不稳定会导致$A,B$的梯度不稳定，从而增加优化难度，所以还是加上$XA$的数值稳定性为条件比较好。

这个数值稳定性条件容易让我们联想到“[LeCun初始化](<https://spaces.ac.cn/archives/8620>)”，它说的是如果$W\in\mathbb{R}^{n\times m}$是独立同分布地采样自“均值为0、方差为$1/n$”的分布，那么$XW$每个分量的数量级，大致上就跟$X$的分量相同。按照相同的策略，如果输入$X$已经是$\mathcal{O}(1)$，那么为了使得$XA,XAB$的分量数量级都是$\mathcal{O}(1)$，$A,B$应该分别用$1/n,1/r$的方差初始化（后面均值默认为0，不再重复写出）。

当然，前面说了LoRA为了保证初始化的恒等性，$A,B$之一要选择全零初始化，但这不大重要，我们只需要意识到$1/n,1/r$的方差可以让$XA,XAB$都保持数值稳定性，那么就可以猜测训练完成后的$A,B$，很可能也近似地也有$1/n,1/r$的方差。鉴于$r \ll n$，所以这等价于说$A$的分量绝对值会明显小于$B$的分量绝对值，这就是$A,B$不对称性的源头。

## 贡献相当

接着，我们来看第二个假设：$A,B$应该对效果有同等程度上的贡献，这个假设看上去也很合理，因为在LLM\+LoRA的场景，通常有$m=n$，即$A,B$的参数量相同，那么它们对效果的贡献相同是合理的，如果$m\neq n$，我们也可以进一步将这个假设推广为效果贡献正比于参数数量。衡量效果的最基本指标自然是损失函数，这里记为$\mathcal{L}$。

我们要衡量$A\to A+\Delta A,B\to B + \Delta B$时，损失函数的变化：
\begin{equation}\mathcal{L}(A+\Delta A,B+\Delta B) - \mathcal{L}(A,B)\approx \left\langle \frac{\partial\mathcal{L}}{\partial A},\Delta A\right\rangle + \left\langle \frac{\partial\mathcal{L}}{\partial B},\Delta B\right\rangle\label{eq:delta-loss}\end{equation}
这里使用了一阶线性近似，其中$\frac{\partial\mathcal{L}}{\partial A},\frac{\partial\mathcal{L}}{\partial B}$是$A,B$的梯度，$\langle\cdot,\cdot\rangle$是（Frobenius）内积运算，右端两项就可以理解为$A,B$对效果的分别贡献。但注意线性近似的有效性取决于增量$\Delta A,\Delta B$是小量，但对于训练好的权重，它对于原始权重的增量还真未必是小量。所以退而求其次，我们将“贡献相当”假设改为“$A,B$在每一步更新中应该对效果有同等程度上的贡献”，由于单步更新的量通常很小，因此线性近似能比较好地满足。

既然要考虑每一步的更新量，那么就引导我们到了优化器的方向上。当前预训练和微调的主流优化器都是Adam，那么我们就以Adam为主要分析对象。我们知道，Adam优化器有两组滑动平均状态以及对应的超参$\beta_1,\beta_2$，这使得精准的分析比较困难，但就本文的目的而言，我们只需要一个数量级估计，因此我们试图只考虑一个极端的例子，并且认为它和一般情形具有相同的数量级估计结果。这个例子就是$\beta_1=\beta_2=0$，此时Adam退化为[SignSGD](<https://spaces.ac.cn/archives/9473>)：
\begin{equation}\Delta A = -\eta_A\,\text{sign}\left(\frac{\partial\mathcal{L}}{\partial A}\right),\quad\Delta B = -\eta_B\,\text{sign}\left(\frac{\partial\mathcal{L}}{\partial B}\right)\label{eq:sign-sgd}\end{equation}
其中$\eta_A,\eta_B$是各自的学习率，“LoRA\+”的结论就是$\eta_B \gg \eta_A$。

将SignSGD的增量$\eqref{eq:sign-sgd}$代回式$\eqref{eq:delta-loss}$，那么就得到
\begin{equation}\mathcal{L}(A+\Delta A,B+\Delta B) - \mathcal{L}(A,B)\approx \underbrace{-\,\eta_A \left\Vert\frac{\partial\mathcal{L}}{\partial A}\right\Vert_1}_{\Delta \mathcal{L}_A}\,\underbrace{-\,\eta_B \left\Vert \frac{\partial\mathcal{L}}{\partial B}\right\Vert_1}_{\Delta \mathcal{L}_B}\end{equation}
这里的$\Vert\cdot\Vert_1$是$L_1$范数，即所有分量的绝对值之和。“贡献相当”即希望右端的$\Delta \mathcal{L}_A,\Delta \mathcal{L}_B$在数量级上是一致的。

## 快速推导

进一步的分析需要求出梯度的具体形式。再次设$Y=XAB$，那么可以求出：
\begin{equation}\frac{\partial \mathcal{L}}{\partial A} = X^{\top}\frac{\partial \mathcal{L}}{\partial Y}B^{\top},\quad \frac{\partial \mathcal{L}}{\partial B} = A^{\top} X^{\top}\frac{\partial \mathcal{L}}{\partial Y}\end{equation}
不了解矩阵求导的读者可能会困惑于以上结果的推导，其实笔者也不熟悉，但这里有个简单的技巧可以用。比如$\frac{\partial \mathcal{L}}{\partial A}$，我们知道它是一个$n\times r$的矩阵（跟$A$同形状），同理$\frac{\partial \mathcal{L}}{\partial Y}$是一个$b\times m$的矩阵，并且根据求导的链式法则不难知道$\frac{\partial \mathcal{L}}{\partial A}$应该是$\frac{\partial \mathcal{L}}{\partial Y}$、$X$、$B$的乘积，那么我们就按照矩阵乘法的规定去想这三个矩阵怎么相乘才能得到一个$n\times r$的矩阵就是了。

求出$\frac{\partial \mathcal{L}}{\partial A},\frac{\partial \mathcal{L}}{\partial B}$的具体形式之后，我们有一个快速的方式来理解LoRA\+。首先，$\Delta \mathcal{L}_A$正比于$\left\Vert\frac{\partial\mathcal{L}}{\partial A}\right\Vert_1$，这是$nr$个分量绝对值的和，假如每个分量相当，那么这意味着$\Delta \mathcal{L}_A$大致正比于$nr$；然后，$\frac{\partial\mathcal{L}}{\partial A}$关于$B$是一次的，可以大致认为$\frac{\partial\mathcal{L}}{\partial A}$的每个分量量级正比于$B$的分量量级，合并起来就是$\Delta \mathcal{L}_A$同时正比于$nr$和$B$的量级；同理，$\Delta \mathcal{L}_B$大致上也同时正比于$mr$和$A$的量级。前面我们在“[数值稳定](<https://spaces.ac.cn/archives/10001#%E6%95%B0%E5%80%BC%E7%A8%B3%E5%AE%9A>)”一节说了，为了前向的数值稳定性，$B$的量级应该会大于$A$的量级（正比于它们的近似标准差$\sqrt{1/r},\sqrt{1/n}$，于是为了$\Delta \mathcal{L}_A$与$\Delta \mathcal{L}_B$的大小相当，那么应该有近似：
\begin{equation}\eta_A \times nr \times \sqrt{1/r} \approx  \eta_B \times mr \times \sqrt{1/n}\quad\Rightarrow\quad \frac{\eta_B}{\eta_A} \approx \frac{n}{m}\sqrt{\frac{n}{r}}\end{equation}
考虑到实际使用时常有$m=n$且$r=\mathcal{O}(1)$，那么可以简单记为
\begin{equation}\frac{\eta_B}{\eta_A} = \mathcal{O}(\sqrt{n})\end{equation}

但是还没完，我们要检查一下结果是否自洽，因为我们用到的条件之一是“前向的数值稳定性”，至今为止还只是一个理想的假设。如何让假设尽可能成立呢？战胜一个假设的方法是引入另一个假设：

> 在Adam优化器中，如果两个参数的学习率之比是$\lambda$，那么经过长期的训练后，这两个参数的数量级之比也是$\lambda$。

根据Adam的近似式$\eqref{eq:sign-sgd}$，每步增量的数量级确实正比于学习率，但总的更新结果又不完全是每一步简单叠加，所以这个假设给人的感觉就是“看上去有点道理，但又不完全有道理”。但不要紧，假设通常都是这样子的，有点道理就行，剩下的就只能靠信仰了。在这个假设之下，如果我们用$\frac{\eta_B}{\eta_A} = \mathcal{O}(\sqrt{n})$的学习率训练，那么$B,A$两个参数的数量级之比也是$\mathcal{O}(\sqrt{n})$，而我们之前期望它们有近似的标准差$\sqrt{1/r},\sqrt{1/n}$，这两个之比正好是$\mathcal{O}(\sqrt{n})$，结果完全自洽！

原论文的结果跟上述结果略有不同，它给出的答案是$\mathcal{O}(n)$，这是因为原论文考虑的是$\Delta A,\Delta B$对$Y$有同等程度的增量，但$Y$只是模型层的输出，并不代表最终效果，因此是欠妥的。尽管原论文也试图将$Y$的增量跟$\mathcal{L}$的增量联系起来，但并没有仔细展开运算，导致计算结果出现偏差。此外，原论文的推导，原则上也只适用于$b=1,r=1,m=n$的特殊情形，$b > 1, r > 1$的一般情况是直接沿用的，这意味着分析过程其实是不够通用的。

当然，具体是$\mathcal{O}(n)$还是$\mathcal{O}(\sqrt{n})$其实不大重要，实际还是得调。但LoRA\+在各种尺寸的模型上做了实验，$r$普遍是8，$n$从768到4096不等，最后得出推荐默认的学习率比例是$2^4 = 16$，这正好跟$\sqrt{n/r}$差不多，因此最优值更接近于$\mathcal{O}(\sqrt{n})$而不是$\mathcal{O}(n)$。

## 文章小结

这篇文章中，我们介绍并推导了一个名为“LoRA\+”的结果，它支持LoRA的两个低秩矩阵$A,B$存在固有的不对称性，不管将哪个矩阵全零初始化，都应该将$B$的学习率设置得大于$A$，以达到更优的效果。
