# 如何度量数据的稀疏程度？

> 作者：苏剑林 · 科学空间 · 2023-05-05
>
> 原文：<https://spaces.ac.cn/archives/9595>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在机器学习中，我们经常会谈到稀疏性，比如我们经常说注意力矩阵通常是很稀疏的。然而，不知道大家发现没有，我们似乎从没有给出过度量稀疏程度的标准方法。也就是说，以往我们关于稀疏性的讨论，仅仅是直观层面的感觉，并没有过定量分析。那么问题来了，稀疏性的度量有标准方法了吗？

经过搜索，笔者发现确实是有一些可用的指标，比如$l_1/l_2$、熵等，但由于关注视角的不同，在稀疏性度量方面并没有标准答案。本文简单记录一下笔者的结果。

## 基本结果

狭义上来讲，“稀疏”就是指数据中有大量的零，所以最简单的稀疏性指标就是统计零的比例。但如果仅仅是这样的话，注意力矩阵就谈不上稀疏了，因为softmax出来的结果一定是正数。所以，有必要推广稀疏的概念。一个朴素的想法是统计绝对值不超过$\epsilon$的元素比例，但这个$\epsilon$怎么确定呢？

1相比于0\.0001很大，但是相比10000又很小，所以大和小的概念不是绝对的。直观来想，稀疏的向量存在很多接近于零的数，那么它的绝对值的平均肯定会比较小，又因为大和小的概念是相对的，我们不妨将这个平均值跟最大值做除法，以获得相对结果，所以一个看上去比较合理的指标是
\begin{equation}S_0(\boldsymbol{x})=\frac{(|x_1|+|x_2|+\cdots+|x_n|)/n}{\max(|x_1|,|x_2|,\cdots,|x_n|)}\label{eq:start}\end{equation}
其中$\boldsymbol{x}=[x_1,x_2,\cdots,x_n]\in\mathbb{R}^n$是需要评估稀疏性的向量（下面都假设$\boldsymbol{x}$并非全等向量，即至少有两个不同的元素），指标越小的向量越稀疏。不过，尽管这个指标有一定的合理性，但它却不够“光滑”，主要是$\max$这个操作极易受到异常值的影响，不能很好地反应数据统计特性。

## 光滑齐次

于是，我们按照[《寻求一个光滑的最大值函数》](<https://spaces.ac.cn/archives/3290>)的思路，将$\max$换成它的光滑近似。$\max$标准的光滑近似是$\text{logsumexp}$：
\begin{equation}\max(|x_1|,|x_2|,\cdots,|x_n|)\approx \frac{1}{k}\log\sum_{i=1}^n e^{k|x_i|}\end{equation}
然而，如果这里用$\text{logsumexp}$替代$\max$，并不能起到较好的改进效果。一是因为有$e^{k|x_i|}$的放大作用，$\text{logsumexp}$同样容易受到异常值的影响，二是因为$\text{logsumexp}$没有正齐次性，反而不美（所有$x_i$乘以正数$\alpha$，稀疏性应该不改变）。在[《寻求一个光滑的最大值函数》](<https://spaces.ac.cn/archives/3290>)中我们也给出了一个$\max$的具备正齐次性的光滑近似，它正好是$l_p$范数（$p > 1$）：
\begin{equation}\max(|x_1|,|x_2|,\cdots,|x_n|)\approx \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}\triangleq l_p(\boldsymbol{x})\end{equation}
再次观察式$\eqref{eq:start}$，我们可以发现它的分子正好是$l_1(\boldsymbol{x}) / n$，所以综合上面的结果，我们得到了度量稀疏性的一个指标为
\begin{equation}S_{1,p}(\boldsymbol{x})=\frac{l_1(\boldsymbol{x})/n}{l_p(\boldsymbol{x})}\label{eq:s1p}\end{equation}
如果只是固定维度的向量比较，那么$/n$可以略掉。常见的是取$p=2$，那么就得到稀疏性度量的$l_1/l_2$指标；如果取$p\to\infty$，那么其实就是指标$\eqref{eq:start}$。

## 理想性质

在[《“熵”不起：从熵、最大熵原理到最大熵模型（一）》](<https://spaces.ac.cn/archives/3534>)介绍“熵”的概念时，我们发现可以通过几条熵应当具备的理想性质来把熵的数学形式确定下来。那么稀疏性可否效仿这一点呢？

论文[《Comparing Measures of Sparsity》](<https://papers.cool/arxiv/0811.4706>)做了这方面的尝试，它提出稀疏性度量$S(\boldsymbol{x})$应该具备以下几点理想性质（不失一般性，这里假设$\boldsymbol{x}$是非负向量，如果不是的话，逐项取绝对值即可）：

> <strong>D1</strong>、$S([\cdots,x_i - \alpha,\cdots,x_j + \alpha,\cdots]) > S(\boldsymbol{x})$，其中$x_i > x_j$且$0 < \alpha < \frac{x_i - x_j}{2}$。这点性质说的是如果总和不变，那么越均匀越不稀疏。
> 
> 
> 
> <strong>D2</strong>、$S(\alpha\boldsymbol{x}) = S(\boldsymbol{x})$，其中$\alpha > 0$。这点容易理解，就是指稀疏是一个相对的性质，所有元素乘以一个相同的倍数，不改变相对大小，也就不改变稀疏性。
> 
> 
> 
> <strong>D3</strong>、$S(\alpha + \boldsymbol{x}) > S(\boldsymbol{x})$，其中$\alpha > 0$。这点也容易理解，全体元素加上一个正数，那么全体都更加远离了零，稀疏性自然要降低。
> 
> 
> 
> <strong>D4</strong>、$S(\boldsymbol{x}) = S(\boldsymbol{x}\circ\boldsymbol{x}) = S(\boldsymbol{x}\circ\boldsymbol{x}\circ\boldsymbol{x}) = \cdots$，这里的$\circ$指两个向量拼接。这点也不难理解，就是单纯的数据复制不改变稀疏性。
> 
> 
> 
> <strong>P1</strong>、任意给定$i\in\{1,2,\cdots,n\}$，存在$\beta_i > 0$，使得对于任意$\alpha > 0$，都有$S([\cdots,x_{i-1},x_i + \beta_i + \alpha,x_{i
+1},\cdots]) < S([\cdots,x_{i-1},x_i + \beta_i,x_{i+1},\cdots])$。这条性质想表达的是当某个元素足够大的时候，整个向量的稀疏性就由它主导了。
> 
> 
> 
> <strong>P2</strong>、$S(\boldsymbol{x}\circ[0]) < S(\boldsymbol{x})$。这个就很朴素了，给向量添加零，应该导致稀疏性增加。

原论文对稀疏性的各种常用指标做了推导，发现只有一个名为“Gini指数”的指标同时满足以上6点性质（但这个Gini指数比较复杂，这里就不介绍了）。不过要提醒读者的是，读原论文时需要仔细留意一下推导过程，因为笔者发现它证明$l_1/l_2$不满足<strong>D3</strong>的证明是错的，事实上$l_1/l_2$是满足<strong>D3</strong>的，至于其他推导笔者也没有细看，读者请在阅读过程中自行甄别正误。

## 参考证明

对于本文的两个指标，指标$\eqref{eq:start}$同时满足除<strong>D1</strong>外的另外5点性质（如果D1的$ > $改为$\geq $，那么也满足），这些性质的判断是比较平凡的，这里不详细讨论了，请读者自行完成。

至于指标$\eqref{eq:s1p}$，可以证明它同时满足除<strong>D4</strong>外的另外5点性质，其中<strong>D3</strong>、<strong>P1</strong>的证明稍微复杂一些，这里给出它们的参考证明。

> 为了证明<strong>D3</strong>，只需要证明
> \begin{equation}\frac{\left(\sum\limits_{i=1}^n (x_i + \alpha)\right)^p}{\sum\limits_{i=1}^n (x_i + \alpha)^p}\end{equation}
> 关于$\alpha > 0$是单调递增的。两边取对数，得到
> \begin{equation}p\log\sum\limits_{i=1}^n (x_i + \alpha) - \log\sum\limits_{i=1}^n (x_i + \alpha)^p\triangleq f(\alpha)\end{equation}
> 我们只需要证明$f'(\alpha) > 0$。直接求导得到
> \begin{equation}f'(\alpha) = \frac{pn}{\sum\limits_{i=1}^n (x_i + \alpha)} - \frac{p\sum\limits_{i=1}^n (x_i + \alpha)^{p-1}}{\sum\limits_{i=1}^n (x_i + \alpha)^p}\end{equation}
> $f'(\alpha) > 0$等价于
> \begin{equation}\frac{1}{n}\sum\limits_{i=1}^n (x_i + \alpha)^p > \left(\frac{1}{n}\sum\limits_{i=1}^n (x_i + \alpha)\right)\left(\frac{1}{n}\sum\limits_{i=1}^n (x_i + \alpha)^{p-1}\right)\end{equation}
> 这由[幂均值不等式](<https://en.wikipedia.org/wiki/Generalized_mean>)直接可得。

> <strong>P1</strong>的证明思路是类似的，只需要证明当$x_i$足够大的时候，
> \begin{equation}\frac{\left(x_i + \alpha + \sum\limits_{j\neq i} x_j\right)^p}{(x_i + \alpha)^p + \sum\limits_{j\neq i} x_j^p}\end{equation}
> 关于$\alpha > 0$是单调递减的。两边取对数，得到
> \begin{equation}p\log\left(x_i + \alpha + \sum\limits_{j\neq i} x_j\right) - \log\left((x_i + \alpha)^p + \sum\limits_{j\neq i} x_j^p\right)\triangleq g(\alpha)\end{equation}
> 我们只需要证明$g'(\alpha) < 0$。直接求导得到
> \begin{equation}g'(\alpha) = \frac{p}{x_i + \alpha + \sum\limits_{j\neq i} x_j} - \frac{p (x_i + \alpha)^{p-1}}{(x_i + \alpha)^p + \sum\limits_{j\neq i} x_j^p}\end{equation}
> $g'(\alpha) < 0$等价于
> \begin{equation}p \sum\limits_{j\neq i} x_j^p < p(x_i+\alpha)^{p-1}\sum\limits_{j\neq i} x_j\end{equation}
> 只要各$x_j$并非全零，那么当$x_i$足够大时，总可以使得上式对$\forall \alpha > 0$恒成立。

## 完美指标

尽管指标$\eqref{eq:s1p}$不满足<strong>D4</strong>，但我们将它简单修改为
\begin{equation}S_{1,p}^*(\boldsymbol{x})=\frac{l_1(\boldsymbol{x})/n}{l_p(\boldsymbol{x})/n^{1/p}}=n^{(1-p)/p}\frac{l_1(\boldsymbol{x})}{l_p(\boldsymbol{x})}\label{eq:s1p-plus}\end{equation}
后，它是满足<strong>D4</strong>的，并且也可以检验它也满足<strong>P2</strong>，至于其他几点性质并不涉及到维度$n$的改变，因此跟指标$\eqref{eq:s1p}$一样也满足。也就是说，$S_{1,p}^*(\boldsymbol{x})$是同时满足6点性质的“<strong>完美指标</strong>”！

由[幂均值不等式](<https://en.wikipedia.org/wiki/Generalized_mean>)可知$l_p(\boldsymbol{x})/n^{1/p} \geq l_1(\boldsymbol{x})/n$，所以$S_{1,p}^*(\boldsymbol{x})\leq 1$；此外，由于
\begin{equation}\frac{l_p(\boldsymbol{x})}{l_1(\boldsymbol{x})}=\left(\left(\frac{x_1}{l_1(\boldsymbol{x})}\right)^p+\cdots+\left(\frac{x_n}{l_1(\boldsymbol{x})}\right)^p\right)^{1/p}\leq\left(\frac{x_1}{l_1(\boldsymbol{x})}+\cdots+\frac{x_n}{l_1(\boldsymbol{x})}\right)^{1/p}=1\end{equation}
所以$S_{1,p}^*(\boldsymbol{x})\geq n^{(1-p)/p}$，综上即$S_{1,p}^*(\boldsymbol{x})\in[n^{(1-p)/p},1]$。此外，$S_{1,p}^*(\boldsymbol{x})$还可以更一般地推广为：
\begin{equation}S_{q,p}^*(\boldsymbol{x})=\frac{l_q(\boldsymbol{x})/n^{1/q}}{l_p(\boldsymbol{x})/n^{1/p}}=n^{1/p-1/q}\frac{l_q(\boldsymbol{x})}{l_p(\boldsymbol{x})}\in\left[n^{1/p-1/q},1\right]\end{equation}
只要$p > q > 0$，那么它也是符合6点性质的“完美指标”。

当$p=2,q=1$时，有
\begin{equation}S_{1,2}^*(\boldsymbol{x})=\frac{1}{\sqrt{n}}\frac{l_1(\boldsymbol{x})}{l_2(\boldsymbol{x})}\in\left[\frac{1}{\sqrt{n}},1\right]\end{equation}
这个结果可以回答关于稀疏化的一些问题。比如为什么L1正则有利于稀疏化？因为L1出现在上式的分子中，它越小越稀疏。为什么L2正则不利于稀疏化？因为L2出现在上式的分母中，它越小越不稀疏。如果要更准确地实现稀疏，应该以$S_{1,2}^*(\boldsymbol{x})$为正则项，它既最小化L1，又最大化L2，直接优化稀疏指标。

特别地，当$\boldsymbol{x}$全不为零时，我们还有关系式
\begin{equation}S_{1,2}^*(\boldsymbol{x})=\cos(\text{sign}(\boldsymbol{x}),\boldsymbol{x})\end{equation}
其中$\text{sign}$是对向量的每一个分量都取符号函数。这个式子很形象，$\text{sign}(\boldsymbol{x})$可以说是$\boldsymbol{x}$的最稠密的衍生向量，稀疏程度就是$\boldsymbol{x}$与最稠密那个衍生向量的相似度，相似度越小自然是代表越稀疏。

如果我们将$\boldsymbol{x}$的元素视为随机变量$x$的采样结果，那么还可以写出
\begin{equation}S_{1,2}^*(\boldsymbol{x})=\frac{\mathbb{E}[|x|]}{\sqrt{\mathbb{E}[x^2]}}=\frac{\mathbb{E}[|x|]}{\sqrt{\mathbb{E}[|x|]^2 + \mathbb{V}ar[|x|]}} = \frac{1}{1 + \mathbb{V}ar[|x|] / \mathbb{E}[|x|]^2}\end{equation}
单独拎出来，$\frac{\mathbb{E}[|x|]}{\sqrt{\mathbb{V}ar[|x|]}}$的平方正是$|x|$的“信噪比”（均值平方与方差之比），所以这告诉我们“$|x|$的信噪比越低，那么$x$就越稀疏”。

## 熵的联系

现在回到注意力矩阵上，它的特点是每一行对应一个概率分布，即自动满足$x_i \geq 0$且$x_1+\cdots+x_n=1$。最确定的概率分布是one hot分布，此时它也最稀疏；最不确定的分布是均匀分布，很显然此时它也最不稀疏。从这两个极端可以猜测，概率分布的稀疏性与不确定性有一定的关联。

我们知道，概率分布的不确定性一般用（香侬）熵来度量：
\begin{equation}H(\boldsymbol{x}) = -\sum_{i=1}^n x_i \log x_i\in[0,\log n] \end{equation}
而此时指标$\eqref{eq:s1p-plus}$则变为$\frac{n^{(1-p)/p}}{l_p(\boldsymbol{x})}$，它是$l_p$范数的衍生物。既然稀疏性与不确定性可能有一定的关联，那么是否意味着熵与$l_p$范数存在一定程度的相关性呢？

确实如此。事实上，基于$l_p$范数我们可以构造出[Rényi熵](<https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy>)：
\begin{equation}H_p(\boldsymbol{x}) = -\frac{1}{p-1}\log \sum_{i=1}^n x_i^p \end{equation}
可以证明$H_1(\boldsymbol{x}) = \lim\limits_{p\to 1} H_p(\boldsymbol{x}) = H(\boldsymbol{x})$，即$p\to 1$时正好对应于经典的香侬熵，而当$p \neq 1$时就是一般的Rényi熵（有些场景Rényi熵也特指$p=2$时的情形）。每种Rényi熵都可以作为不确定性的某种度量，它们的值域都是$[0,\log n]$，都是在one hot分布取的最小值、在均匀分布取得最大值。从这个意义上来说，所有的Rényi熵在一定程度上都是等价的，这就解释了熵与$l_p$范数的关联。

值得一提的是，$p \neq 1$时的Rényi熵往往对数值计算更加友好，这是因为$\sum\limits_{i=1}^n x_i^p$必然是一个正的、有界的结果，$\log$运算不用担心$\log 0$问题，而且只需要对最后的结果做一次$\log$；相反标准的香侬熵，需要对每个$x_i$都算$\log$，计算量增加而且还要特别$\text{clip}$一下以防$\log 0$的出现。

## 文章小结

本文系统整理了一下关于稀疏性的度量问题，并且讨论了它与L1、L2、熵等概念的联系。
