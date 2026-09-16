# 矩阵参数的奇异值熵越高越好吗？

> 作者：苏剑林 · 科学空间 · 2026-05-29
>
> 原文：<https://spaces.ac.cn/archives/11767>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在去年的技术报告[《Muon is Scalable for LLM Training》](<https://papers.cool/arxiv/2502.16982>)中，为了对比Muon与Adam所训练出来的模型的差异，我们引入了“奇异值熵”的概念，观察到Muon训练出来的矩阵参数，其奇异值熵普遍高于Adam所训的矩阵，我们将此作为“Muon更充分利用了模型参数”的一个经验证据。

然而，奇异值熵真的越高越好吗？是否存在一个最优的奇异值熵呢？或者反过来问，如果我们必须事先限定矩阵参数的奇异值熵，那么该取一个什么值呢？

## 概念回顾

首先回顾一些概念。我们知道矩阵的奇异值都是非负的，那么我们就可以对奇异值做某种归一化，比如直接归一化/平方归一化：
\begin{equation}p_i = \frac{\sigma_i}{\sum_{j=1}^n \sigma_j}\qquad\text{或}\qquad p_i = \frac{\sigma_i^2}{\sum_{j=1}^n \sigma_j^2}\end{equation}
然后就得到一个$n$元概率分布$\boldsymbol{p} = (p_1,p_2,\cdots,p_n)$。有了概率分布后，我们就可以计算一些相关的变量，比如熵
\begin{equation}H(\boldsymbol{p}) = -\sum_{i=1}^n p_i \log p_i\end{equation}
这便是“奇异值熵”的概念。我们知道最大熵对应的是均匀分布，所以奇异值熵也是用来度量奇异值的均匀程度。不过，熵的概念并不止一种，上述对数期望形式的熵，我们称为“香农熵”，此外还有别的熵，比如[Rényi熵](<https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy>)
\begin{equation}H_q(\boldsymbol{p}) = -\frac{1}{q-1}\log \sum_{i=1}^n p_i^q \end{equation}
实际上香农熵也是它的特例，对应的是$q\to 1$。对于其他$q$值，它也是在均匀分布取到最大值$\log n$，在One Hot分布取到最小值$0$。矩阵的奇异值熵实际上跟[矩阵的有效秩（Effective Rank）](<https://spaces.ac.cn/archives/10847>)是本质等价的，它的另一个含义是矩阵奇异值的稀疏程度（参考[《如何度量数据的稀疏程度？》](<https://spaces.ac.cn/archives/9595>)）。

## 问题转化

现在的问题是，如何将“奇异值熵真的越高越好吗”这一疑问转化为可计算的命题，从而给出一个参考答案呢？

直观来想，由于当前模型普遍是过参数化的，所以存在不少冗余自由度，这允许我们给模型参数加一些约束——比如约束它的模长或奇异值熵——但不会明显损失性能。给参数施加约束后，其自由度自然会被压缩。我们可以经验地认为：剩余的自由度越大，模型的能力就越强，这便是我们寻找“最优奇异值熵”的原理。

那么，给定奇异值熵后，矩阵的自由度还剩多少呢？先来看两种极端情形：第一种是最小熵，对应的奇异值只有一个非零，此时矩阵是一个Rank\-1矩阵，自由度不超过$2n$；第二种是最大熵，对应奇异值非零且全等，此时矩阵是一个正交矩阵（的若干倍），自由度约$n^2/2$。这样看来，最大熵确实比最小熵要优，但是否存在一个更优的中间值呢？

我们也可以抛开矩阵背景，直接看奇异值分布的“熵密度”。$n$元分布的最大熵是$\log n$，此时只有均匀分布满足；最小熵是$0$，对应的分布是One Hot分布，有$n$种不同的选择。显然，这两个边界的“密度”看起来都不是很高，它们只能表示一小部分分布，因此可以猜测最优值确实位于中间位置。

这样转化下来，我们的问题变成了：如果均匀地采样$n$元分布，然后计算熵，那么熵的分布长什么样？概率密度最大的地方在哪？概率密度最大的点，意味着在那附近拥有最大数量的分布，我们就认为这是表达能力最大的熵值。

## 几何图景

可能有些读者对“均匀地采样$n$元分布”不大理解——通常情况下我们都是给定概率分布，然后采样一个数，现在我们要采样的是概率分布本身。不过，只要我们暂时抛开$\boldsymbol{p}$的概率意义，那么就可以获得一个很清晰的几何理解。

具体来说，一个$n$元概率分布$\boldsymbol{p}$，实际上是指满足如下约束的一个$n$维向量：
\begin{equation}0 \leq p_i \leq 1 \, (\forall i=1,2,\cdots,n)\qquad \sum_{i=1}^n p_i = 1\end{equation}
第一个约束，描述了一个$n$维空间中的单位超立方体，第二个约束则描述一个$n$维空间中的超平面，两者一相交，得到一块有限的子平面，这块子平面就称为“$n-1$单纯形”，记为$\Delta^{n-1}$。所谓“均匀采样$n$元分布”，实际上就是在这块子平面上均匀采样一个点，这样理解就直观多了。

不知道读者是否好奇，为什么用三角形符号$\Delta$来表示单纯形？实际上，当$n=3$时，这块子平面正是以$(1,0,0)$、$(0,1,0)$和$(0,0,1)$为顶点的等边三角形；不难想象，当$n=4$时，它构成了三维空间一个正四面体。依此类推可以发现，从几何角度看，单纯形实际上就是等边三角形在高维空间中的推广，因此用三角形$\Delta$来表示。

## 极限定理

言归正传。现在我们考虑计算熵的概率密度，它可以形式地写成
\begin{equation}\rho(h) = \frac{\int_{\boldsymbol{p}\in\Delta^{n-1}} \delta(H(\boldsymbol{p}) - h) d\boldsymbol{p}}{\int_{\boldsymbol{p}\in\Delta^{n-1}} d\boldsymbol{p}}\end{equation}
其中$\delta(\cdot)$是[Dirac函数](<https://spaces.ac.cn/archives/1870>)。该式的物理意义其实也很直观，就是对$H(\boldsymbol{p})=h$的分布进行“计数”，然后除以总数。然而，这纯粹是一个形式解，实际去解析计算几乎不可能。

考虑到实际场景下$n$通常都比较大（数百乃至数千），我们可以基于“[中心极限定理](<https://en.wikipedia.org/wiki/Central_limit_theorem>)”，假设此时$\rho(h)$近似为一个正态分布。正态分布只有均值和方差两个参数，所以我们只需计算熵的均值和方差（二阶矩），即$\mathbb{E}_{\boldsymbol{p}\sim\Delta^{n-1}}[H(\boldsymbol{p})]$和$\mathbb{E}_{\boldsymbol{p}\sim\Delta^{n-1}}[H(\boldsymbol{p})^2]$。如果我们只想找最大概率点，那么只需计算均值，因为正态分布的最大概率点就是均值。

计算均值$\mathbb{E}_{\boldsymbol{p}\sim\Delta^{n-1}}[H(\boldsymbol{p})]$的困难是如何实现$\boldsymbol{p}\sim\Delta^{n-1}$，这个采样虽然概念上是直观的，但计算中如何实现并不清晰。如果读者了解一些Dirichlet分布的背景知识，那么可以利用一些现成结论来加速理解，但这里我们刻意回避引入Dirichlet分布，降低普通读者的理解成本。

## 采样转化

幸运的是，在单纯形中均匀采样可以转化为从[指数分布](<https://en.wikipedia.org/wiki/Exponential_distribution>)$\text{Exp}(1)$独立重复采样：
\begin{equation}\boldsymbol{p}\sim\Delta^{n-1}\qquad\Leftrightarrow\qquad p_i = \frac{x_i}{\sum_{j=1}^n x_j},\quad x_1,x_2,\cdots,x_n\sim \text{Exp}(1)\end{equation}
也就是说，从指数分布$\text{Exp}(1)$独立重复采样$n$个数，然后归一化，就等价于单纯形均匀采样。怎么去理解这个事情呢？严格地证明自然是可以的，但这里笔者更想介绍一个类比各向同性的视角来理解。

先让我们换另一个问题：均匀地采样$n$维单位向量。这个不难实现，只需从标准正态分布$\mathcal{N}(0,1)$中采样$n$个数$(x_1,x_2,\cdots,x_n)=\boldsymbol{x}$，然后返回$\boldsymbol{x}/\Vert\boldsymbol{x}\Vert$。为什么能这样做？因为$\mathcal{N}(0,1)$的概率密度正比于$e^{-x^2/2}$，采样$n$次就是$e^{-\sum_{i=1}^n x_i^2/2} = e^{-\Vert\boldsymbol{x}\Vert^2/2}$，只依赖于$\boldsymbol{x}$的模长而不依赖于方向，所以从$\mathcal{N}(0,1)$采样对于方向来说是均匀的。

单位向量是“平方和等于1”的向量，现在我们要采样的概率分布则是“和等于1”的向量，这样看来，概率分布就是<u>以“和”为模长</u>的单位向量。而我们知道指数分布$\text{Exp}(1)$的概率密度是$e^{-x}$，采样$n$次就是$e^{-\sum_{i=1}^n x_i}$，只依赖于“和”模长，所以它对于这种新的“方向向量”来说是均匀的。

## 平均之场

准备了那么多，现在我们可以正式进入计算了。在新参数化下，熵$H(\boldsymbol{p})$可以变换为
\begin{equation}H(\boldsymbol{p}) = \log \sum_{i=1}^n x_i - \frac{\sum_{i=1}^n x_i \log x_i}{\sum_{i=1}^n x_i}\end{equation}
然后对两边算期望，得到一道颇为复杂的积分题。有趣的是，这道积分虽然复杂，但它是可以通过一些技巧来求得解析解的。不过，这里我们就不去深究这些细节了，而是通过平均场技巧快速获得一个近似解：
\begin{equation}\begin{aligned}
\mathbb{E}\left[\log \sum_{i=1}^n x_i - \frac{\sum_{i=1}^n x_i \log x_i}{\sum_{i=1}^n x_i}\right] \approx \log \sum_{i=1}^n \mathbb{E}[x_i] - \frac{\sum_{i=1}^n \mathbb{E}[x_i \log x_i]}{\sum_{i=1}^n \mathbb{E}[x_i]} = \log n - (1 - \gamma)
\end{aligned}\end{equation}
其中$\gamma$是[欧拉常数](<https://en.wikipedia.org/wiki/Euler%27s_constant>)。这里$\mathbb{E}[x_i]=1$是显然的，有点难度的是$\mathbb{E}[x_i \log x_i]=1-\gamma$，但这个结果也是经典的，跟[Gamma函数](<https://zh.wikipedia.org/wiki/%CE%93%E5%87%BD%E6%95%B0#%E5%AF%BC%E6%95%B0>)的导数有关，要是不知道这个结果，直接用Mathematica也可以计算出来。

我们知道，$n$元分布的最大熵是$\log n$，现在多出了一项$-(1-\gamma) = - 0.42278\cdots$，这正是我们想要的，它表明熵密度最大的地方，大概在比最大熵小$0.42$的位置，对于矩阵而言，奇异值熵在此处附近的矩阵表达能力更强，而非越大越好。如果我们需要固定奇异值熵，那么可以选择这个值。

## 一般结果

用同样的思路，我们可以计算一般Rényi熵的期望：
\begin{equation}\begin{aligned}
\mathbb{E}[H_q(\boldsymbol{p})] =&\, \frac{1}{q-1}\mathbb{E}\left[q\log \sum_{i=1}^n x_i -\log \sum_{i=1}^n x_i^q\right] \\
\approx&\, \frac{1}{q-1}\left[q\log \sum_{i=1}^n \mathbb{E}[x_i] - \log \sum_{i=1}^n \mathbb{E}[x_i^q]\right] \\
=&\, \frac{1}{q-1}\left[q \log n - \log n\Gamma(q+1)\right] \\
=&\, \log n - \frac{\log \Gamma(q+1)}{q-1}
\end{aligned}\label{eq:E-Hq}\end{equation}
如果$q\to 1$，那么对应的极限正是$\log n - (1-\gamma)$，跟上一节的结果一致。如果$q$是大于1的整数，那么直接可以写出
\begin{equation}\mathbb{E}[H_q(\boldsymbol{x})] \approx \log n - \frac{\log q\,!}{q-1}\end{equation}
如果取$q=2$，那么结果是极简的$\log (n/2)$。我们知道$\log (n/2)$是$n/2$元分布的最大熵，所以这个结果的形象理解就是：奇异值熵只要“打满”一半的维度就好了，不需要打满全部维度，否则反而会丧失多样性。

## 完整分析

上面两节我们都在集中精力估计均值，当然这对于我们的原始目标来说已经够用了。不过，为了分析的完整性，这一节我们来补充方差的估计，并补充讨论渐近正态性对$\rho(h)$为什么成立。

首先，我们将$H_q(\boldsymbol{p})$恒等变换成
\begin{equation}H_q(\boldsymbol{p}) = \log n + \frac{1}{q-1}\Bigg[q\log \underbrace{\frac{1}{n}\sum_{i=1}^n x_i}_u -\log \underbrace{\frac{1}{n}\sum_{i=1}^n x_i^q}_v\Bigg]\end{equation}
当$q\neq 1$时，$u,v$是两个不同的统计量，中心极限定理告诉我们，当$n\to\infty$时，$(u,v)$渐近服从二元正态分布。根据$x_i\sim \text{Exp}(1)$，我们可以求出$u$的均值和方差分别是$1$和$1/n$，$v$的均值和方差则分别是$\Gamma(q+1)$和$[\Gamma(2q+1) - \Gamma(q+1)^2]/n$。

由此可见，当$q$不大而$n$较大时，它们都集中在各自的均值附近，这允许我们在均值附近做泰勒展开，比如一阶近似
\begin{equation}H_q(\boldsymbol{p}) \approx \log n + \frac{1}{q-1}\Bigg[q(u-1) - \left(\log \Gamma(q+1) + \frac{v - \Gamma(q+1)}{\Gamma(q+1)}\right)\Bigg]\end{equation}
两端求期望，即可得到式$\eqref{eq:E-Hq}$，如果我们有需要，也可以展开到更多阶去求余项。由于现在$H_q(\boldsymbol{p})$关于$u,v$是线性的，我们还可以根据如下公式计算方差
\begin{equation}\mathbb{V}\mathrm{ar}[H_q(\boldsymbol{p})] \approx \mathbb{V}\mathrm{ar}\left[\frac{q u}{q-1}\right] + \mathbb{V}\mathrm{ar}\left[\frac{v}{(q-1) \Gamma(q+1)}\right] - 2\mathbb{C}\mathrm{ov}\left[\frac{q u}{q-1},\frac{v}{(q-1) \Gamma(q+1)}\right]\end{equation}
这里要注意，$u,v$通常不是独立的，所以协方差项不可忽略
\begin{equation}\mathbb{C}\mathrm{ov}[u, v] = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n \mathbb{C}\mathrm{ov}[x_i, x_j^q] = \frac{1}{n^2}\sum_{i=1}^n \mathbb{C}\mathrm{ov}[x_i, x_i^q]\end{equation}
第一个等号是根据$u,v$的定义以及$\mathbb{C}\mathrm{ov}$的双线性，第二个等号是因为$i\neq j$时$x_i,x_j$相互独立，协方差为0。接着计算得$\mathbb{C}\mathrm{ov}[x_i, x_i^q] = \mathbb{E}[x_i^{q+1}] - \mathbb{E}[x_i] \mathbb{E}[x_i^q]=\Gamma(q+2) - \Gamma(q+1) = q \Gamma(q+1)$。代入这些结果，我们得到
\begin{equation}\mathbb{V}\mathrm{ar}[H_q(\boldsymbol{p})] \approx \frac{1}{n}\left[\frac{\Gamma(2q+1)}{(q-1)^2 \Gamma(q+1)^2} - \frac{q^2+1}{(q-1)^2}\right]\end{equation}
当$q=2$时，结果是$1/n$；当$q\to 1$时，结果是$(\pi^2/3-3)/n$。

总的来说，中心极限定理其实没有直接给出熵$h$的渐近正态性，我们只能得到中间变量$u,v$的渐近正态性，然后从中估算$h$的均值和方差，显示$h$的方差也是$\mathcal{O}(1/n)$的，这表明方差会随着$n$增大而缩小，所以通过均值来找概率密度最大点是科学的。

## 文章小结

本文将“奇异值熵是否越高越好”转化为一个可定量分析的数学命题，核心假设是：固定奇异值熵后，矩阵参数剩余的自由度代表着它的表达能力。随后，我们将自由度最大的位置，理解为熵密度最大的点，通过一系列变换和近似，得到了一个最优熵值。

当然，这个结果本身并不是特别重要，本文的意义更多是为相关问题的分析提供了一个参考框架。
