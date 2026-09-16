# 矩阵的有效秩（Effective Rank）

> 作者：苏剑林 · 科学空间 · 2025-04-10
>
> 原文：<https://spaces.ac.cn/archives/10847>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

秩（Rank）是线性代数中的重要概念，它代表了矩阵的内在维度。然而，数学上对秩的严格定义，很多时候并不完全适用于数值计算场景，因为秩等于非零奇异值的个数，而数学上对“等于零”这件事的理解跟数值计算有所不同，数学上的“等于零”是绝对地、严格地等于零，哪怕是$10^{-100}$也是不等于零，但数值计算不一样，很多时候$10^{-10}$就可以当零看待。

因此，我们希望将秩的概念推广到更符合数值计算特性的形式，这便是有效秩（Effective Rank）概念的由来。

## 误差截断

需要指出的是，目前学术界对有效秩并没有统一的定义，接下来我们介绍的是一些从不同角度切入来定义有效秩的思路。对于实际问题，读者可以自行选择适合的定义来使用。

接下来我们主要从奇异值的角度来考虑秩。对于矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，不失一般性，设$n\leq m$，那么它的标准秩为
\begin{equation}\mathop{\text{rank}}(\boldsymbol{M}) \triangleq \max\{i\,|\,\sigma_i > 0\}\leq n\end{equation}
其中$\sigma_1\geq \sigma_2\geq \cdots\geq\sigma_n \geq 0$是$\boldsymbol{M}$的奇异值。直观上，有效秩的概念就是为了将接近于零的奇异值当成零处理，所以有效秩的一个基本属性是不超过标准秩，并且特殊情况下能退化成标准秩。满足该属性的一个简单的定义是
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M},\epsilon) \triangleq \max\{i\,|\,\sigma_i > \epsilon\}\end{equation}
然而，“大”与“小”的概念应该是相对的，1相对于0\.01大，但相对于100就小了，所以看起来除以$\sigma_1$来标准化一下更科学：
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M},\epsilon) \triangleq \max\big\{i\,\big|\,\sigma_i/\sigma_1 > \epsilon\big\}\end{equation}

除了直接对接近于零的奇异值进行截断外，我们还可以从低秩近似的角度来考虑这个问题。在[《低秩近似之路（二）：SVD》](<https://spaces.ac.cn/archives/10407>)我们证明了，一个矩阵的最优$r$秩近似就是只保留最大的$r$个奇异值的SVD结果。反过来，我们可以指定一个相对误差$\epsilon$，将有效秩定义为可以达到这个相对误差的最小秩：
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M},\epsilon) \triangleq \min\left\{ i\,\,\left|\,\,\sqrt{\left(\sum_{i=1}^r \sigma_i^2\right)\left/\left(\sum_{i=1}^n \sigma_i^2\right.\right)} \geq 1-\epsilon\right.\right\}\end{equation}
这个定义的数值意义更清晰，但它只考虑了总体误差，所以有些例子反而不大优雅。比如$n\times n$的单位阵，我们期望它的有效秩总是$n$，因为每个奇异值都是等同的1，不应有任何截断行为。但采用上述定义的话，当$n$足够大（$1/n < \epsilon$）时，有效秩就会小于$n$。

## 范数之比

上一节定义的有效秩虽然直观，但都依赖于一个超参数$\epsilon$，终究是不够简洁。现在我们的基本认知是有效秩的概念只能依赖于相对大小，所以问题等价为从$1\geq \sigma_2/\sigma_1\geq\cdots\geq\sigma_n/\sigma_1\geq 0$中构建有效秩。由于都在$[0,1]$内的特点，一个巧妙的想法是直接将它们求和
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \sum_{i=1}^n\frac{\sigma_i}{\sigma_1}\end{equation}
从[《低秩近似之路（二）：SVD》](<https://spaces.ac.cn/archives/10407>)我们知道，最大的奇异值$\sigma_1$是矩阵的[谱范数（Spectral Norm）](<https://en.wikipedia.org/wiki/Matrix_norm#Spectral_norm_%28p_=_2%29>)，记为$\Vert\boldsymbol{M}\Vert_2$，而所有奇异值之和实际上也是一个矩阵范数，称为“[核范数（Nuclear Norm）](<https://en.wikipedia.org/wiki/Matrix_norm#Schatten_norms>)”，通常记为$\Vert\boldsymbol{M}\Vert_*$，于是上式也可以简记为
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \frac{\Vert\boldsymbol{M}\Vert_*}{\Vert\boldsymbol{M}\Vert_2}\label{eq:n-2}\end{equation}
这最早的出处可能是[《An Introduction to Matrix Concentration Inequalities》](<https://papers.cool/arxiv/1501.01571>)，在里边被称为“Intrinsic Dimension”，但相关性质在[《Guaranteed Minimum\-Rank Solutions of Linear Matrix Equations via Nuclear Norm Minimization》](<https://papers.cool/arxiv/0706.4138>)已经被探讨过了。

类似地，我们也可以通过平方求和来构建有效秩：
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq \sum_{i=1}^n\frac{\sigma_i^2}{\sigma_1^2} = \frac{\Vert\boldsymbol{M}\Vert_F^2}{\Vert\boldsymbol{M}\Vert_2^2}\label{eq:f-2}\end{equation}
这里的$\Vert\cdot\Vert_F$是$F$范数。该定义的出自应该是[《Sampling from large matrices: an approach through geometric functional analysis》](<https://arxiv.org/abs/math/0503442>)，当时被称为“Numerical Rank”，如今更多称为“Stable Rank”，是比较流行的有效秩概念之一。

从计算复杂度来看，式$\eqref{eq:f-2}$比式$\eqref{eq:n-2}$更低。因为计算核范数需要全体奇异值，这意味着需要完整的SVD；而$\Vert\boldsymbol{M}\Vert_F^2$又等于矩阵所有元素的平方和，所以式$\eqref{eq:f-2}$主要的计算量是最大奇异值，这比计算全体奇异值成本更低。如果愿意，我们还可以将式$\eqref{eq:n-2}$、式$\eqref{eq:f-2}$推广到$k$次方求和，结果实际上是更一般的[Schatten范数](<https://en.wikipedia.org/wiki/Schatten_norm>)与谱范数之比。

## 分布与熵

读者如果直接搜索“Effective Rank”，大概率会搜到论文[《The Effective Rank: a Measure of Effective Dimensionality》](<https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf>)，这也是比较早探讨有效秩的文献，里边提出基于熵来定义有效秩。

首先，由于奇异值总是非负的，我们可以将它归一化得到一个概率分布：
\begin{equation}p_i = \frac{\sigma_i^{\gamma}}{\sum_{j=1}^n \sigma_j^{\gamma}}\end{equation}
其中$\gamma > 0$，从文献看$\gamma=1$和$\gamma=2$都经常有人用（我们在[Moonlight](<https://spaces.ac.cn/archives/10739>)中用了$\gamma=2$），下面我们都以$\gamma=1$为例。有了概率分布，那么就可以计算信息熵（香农熵）：
\begin{equation}H = -\sum_{i=1}^n p_i \log p_i\end{equation}
回忆一下，熵的值域是$[0, \log n]$，取指数后则有$e^H \in [1, n]$，当分布是One Hot时$e^H=1$（只有一个非零奇异值），当分布均匀时$e^H=n$（全体奇异值相等），这正好是标准秩的两个特殊例子，这启发我们可以将有效秩定义为
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq e^H = \exp\left(-\sum_{i=1}^n p_i \log p_i\right)\label{eq:h-erank}\end{equation}
代入$p_i$的定义，我们发现它可以进一步变换成
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) = \exp\left(\log\sum_{i=1}^n \sigma_i -\frac{\sum_{i=1}^n \sigma_i\log\sigma_i}{\sum_{i=1}^n \sigma_i}\right)\end{equation}
很明显括号内的第一项$\exp$后就是$\Vert\boldsymbol{M}\Vert_*$；第二项是$\log\sigma_i$的加权平均，权重是$\sigma_i$，此时$\log\sigma_i$会约等于最大的$\log\sigma_1$，取指数后即$\sigma_1=\Vert\boldsymbol{M}\Vert_2$。所以，上式总的结果将会近似于式$\eqref{eq:n-2}$，这表明了基于熵来定义有效秩看似是一条截然不同的路径，但实际上跟上一节的范数之比有异曲同工之妙。

我们知道，标准秩满足三角不等式$\mathop{\text{rank}}(\boldsymbol{A}+\boldsymbol{B})\leq \mathop{\text{rank}}(\boldsymbol{A}) + \mathop{\text{rank}}(\boldsymbol{B})$，而原论文证明了，对于（半）正定对称矩阵$\boldsymbol{A},\boldsymbol{B}$，式$\eqref{eq:h-erank}$所定义的有效秩满足$\mathop{\text{erank}}(\boldsymbol{A}+\boldsymbol{B})\leq \mathop{\text{erank}}(\boldsymbol{A}) + \mathop{\text{erank}}(\boldsymbol{B})$，尚不清楚这个不等式是否可以推广到一般矩阵。目前看来，证明有效秩能否保持标准秩的一些不等式，是一件并不容易的事情。

## 稀疏指标

从上述一系列有效秩的定义中，尤其是从奇异值到分布再到熵的转变，相信已经有读者隐约意识到，有效秩跟稀疏性有明显的共性，实际上有效秩可以理解为奇异值向量的一个稀疏性度量，跟一般的稀疏性指标不同的是，我们会将值域对齐到$1\leq \mathop{\text{erank}}(\boldsymbol{M}) \leq \mathop{\text{rank}}(\boldsymbol{M}) \leq n$，使其跟秩的概念对齐，从而更直观地感知稀疏程度。

关于稀疏性度量，我们曾在[《如何度量数据的稀疏程度？》](<https://spaces.ac.cn/archives/9595>)有过比较系统的讨论，理论上那里边的结果都可以用来构造有效秩。事实上我们也是这样做的，“[范数之比](<https://spaces.ac.cn/archives/10847#%E8%8C%83%E6%95%B0%E4%B9%8B%E6%AF%94>)”一节中我们基于Schatten范数与谱范数之比来构建有效值，其实只相当于那篇文章的公式$(1)$，我们还可以其他公式如$(16)$，它相当于将有效秩定义为核范数和$F$范数之比的平方：
\begin{equation}\mathop{\text{erank}}(\boldsymbol{M}) \triangleq  \frac{\Vert\boldsymbol{M}\Vert_*^2}{\Vert\boldsymbol{M}\Vert_F^2} = \frac{(\sum_{i=1}^n\sigma_i)^2}{\sum_{i=1}^n\sigma_i^2}\end{equation}
这同样满足$1\leq \mathop{\text{erank}}(\boldsymbol{M}) \leq \mathop{\text{rank}}(\boldsymbol{M})$，是一个可用的有效秩定义。

非常神奇，我们关于有效秩和稀疏性的认知，在无形之中达成了闭环。不得不说这是一种很美妙的体验：笔者开始学习稀疏性度量时对有效秩是一无所知的，而这几天在学习有效秩时，慢慢意识到它跟稀疏性本质是相通的，冥冥之中似乎有一种神秘力量，将我们在不同领域、不同学科中积累的知识悄然串联起来，最终汇聚在同一个正确的方向上。

## 文章小结

本文探讨了矩阵的有效秩（Effective Rank）概念，它是线性代数中矩阵的秩（Rank）概念在数值计算方面的延伸，能够更有效地度量矩阵的本质维度。
