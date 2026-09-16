#  最小熵原理（六）：词向量的维度应该怎么选择？

> 作者：苏剑林 · 科学空间 · 2020-08-20
>
> 原文：<https://spaces.ac.cn/archives/7695>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

随着NLP的发展，像Word2Vec、Glove这样的词向量模型，正逐渐地被基于Transformer的BERT等模型代替，不过经典始终是经典，词向量模型依然在不少场景发光发热，并且仍有不少值得我们去研究的地方。本文我们来关心一个词向量模型可能有的疑惑：词向量的维度大概多少才够？

先说结论，笔者给出的估算结果是
\begin{equation}n > 8.33\log N\label{eq:final}\end{equation}
更简约的话可以直接记$n > 8\log N$，其中$N$是词表大小，$n$就是词向量维度，$\log$是自然对数。当$n$超过这个阈值时，就说明模型有足够的容量容纳这$N$个词语（当然$n$越大过拟合风险也越大）。这样一来，当$N=100000$时，得到的$n$大约是96，所以对于10万个词的词向量模型来说，维度选择96就足够了；如果要容纳500万个词，那么$n$大概就是128。

## 背景

之所以想起这个问题，是因为昨天在Arxiv上刷到了论文[《Word2vec Skip\-gram Dimensionality Selection via Sequential Normalized Maximum Likelihood》](<https://papers.cool/arxiv/2008.07720>)，遗憾的是，从这篇论文中笔者并没有找到想要的答案。顺带搜索了一下，发现也有类似文献研究同样的问题，比如[《On the Dimensionality of Word Embedding》](<https://papers.cool/arxiv/1812.04224>)，但答案依旧不是笔者想要的。

为什么这样说呢？很显然，这个问题的最标准答案应该是靠反复实验来确定最优维度，所以不能指望理论分析给出相当精确的答案。我们平时用到的词向量维度，一般有64、100、128、256、300等，不同的维度之间效果差别其实也没多少，所以笔者只希望能从最简洁直观的方式推导一下一般词向量模型所需要的维度量级，比如几十或者几百，不应该出现太过复杂的分析。

由于没有找到比较满意的现有结果，因此笔者从[最小熵原理](<https://spaces.ac.cn/tag/%E6%9C%80%E5%B0%8F%E7%86%B5/>)角度分析了一下，得到了一个接近自己心中所想的答案。

## 分析

本文要分析是基于Skip Gram思想的词向量模型，多数词向量模型其实都是它的变种，至于CBOW类的模型，在以往的实验里，它的表现其实跟Skip Gram差不多（尤其是数据量较大时），因此可以认为Skip Gram的分析结果应该是通用的。

### 最小化熵

我们的出发点是信息熵，我们知道，熵是不确定性的度量（参考[《“熵”不起：从熵、最大熵原理到最大熵模型（一）》](<https://spaces.ac.cn/archives/3534>)），语言本身具有一定的不确定性，而我们在用向量编码词语时，编码结果应该要等于甚至小于这种不确定性，才能保证这种编码是有效的、能充分保留原来语言的信息。所以，我们要消除不确定性，也就是要最小熵。

要注意的是，词向量是基于Skip Gram模型的，所以我们要计算的不是词平均熵，而是整个Skip Gram模型的平均熵，假设词对$(w_i, w_j)$的频率是$\tilde{p}(w_i, w_j)$，那么可以估算它的熵为
\begin{equation}\tilde{H}=-\sum_{i, j} \tilde{p}(w_i, w_j)\log \tilde{p}(w_i, w_j)\end{equation}
不同的词向量训练目标也有所差异，有些是在拟合联合概率$p(w_i, w_j)$，有些是在拟合条件概率$p(w_j|w_i)$，但这差别不大，前面说了，本文只是想得到一个概数。所以这里统一假设词向量模型为
\begin{equation}p(w_i, w_j) = \frac{e^{\langle\boldsymbol{u}_i, \boldsymbol{v}_j\rangle}}{Z},\quad Z = \sum_{i,j}e^{\langle\boldsymbol{u}_i, \boldsymbol{v}_j\rangle}\end{equation}
其中$\boldsymbol{u},\boldsymbol{v}$代表两套不同的词向量（中心词向量、上下文词向量），$\boldsymbol{u}_i$代表词$w_i$而$\boldsymbol{v}_j$代表词$w_j$。这时候它的信息熵是
\begin{equation}H=-\sum_{i, j} p(w_i, w_j)\log p(w_i, w_j)=\log Z-\frac{1}{Z}\sum_{i, j} e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle} \langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\end{equation}

### 采样近似

为了近似计算上式，我们将求和用采样近似，比如
\begin{equation}Z = \sum_{i, j} e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle} =  N^2\times \frac{1}{N^2}\sum_{i, j} e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle}\approx N^2\mathbb{E}_{\boldsymbol{u}_i,\boldsymbol{v}_j}\left[e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle}\right]\end{equation}
这里的$N$是词表大小。同理
\begin{equation}\sum_{i, j} e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle} \langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\approx N^2\mathbb{E}_{\boldsymbol{u}_i,\boldsymbol{v}_j}\left[e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle} \langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\right]\end{equation}
所以我们有近似
\begin{equation}H\approx\log N^2 + \log \mathbb{E}_{\boldsymbol{u}_i,\boldsymbol{v}_j}\left[e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle}\right]-\frac{\mathbb{E}_{\boldsymbol{u}_i,\boldsymbol{v}_j}\left[e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle} \langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\right]}{\mathbb{E}_{\boldsymbol{u}_i,\boldsymbol{v}_j}\left[e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle}\right]}\end{equation}

### 分布假设

观察已有的词向量模型，我们可以发现每个维度的数值有正有负，绝对值大小一般也比较均匀。在此，我们不妨假设每个元素的绝对值大概为1，那么每个词向量的模长大致就为$\sqrt{n}$（$n$是词向量的维度，也就是我们要估算的目标，如果觉得这个近似不够准确，也可以自行调整），并且进一步假设所有的词向量均匀分布在半径为$\sqrt{n}$的$n$维超球面上，那么$\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle=n\cos\theta$，$\theta$是它们的夹角，所以
\begin{equation}H\approx\log N^2 + \log \mathbb{E}_{\theta}\left[e^{n\cos\theta}\right]-\frac{\mathbb{E}_{\theta}\left[e^{n\cos\theta} n\cos\theta\right]}{\mathbb{E}_{\theta}\left[e^{n\cos\theta}\right]}\label{eq:H-theta}\end{equation}
现在$\theta$相当于$n$维空间中任意两个向量的夹角，我们在[《n维空间下两个随机向量的夹角分布》](<https://spaces.ac.cn/archives/7076>)中就求出了它的分布为
\begin{equation}p_n(\theta) = \frac{\Gamma\left(\frac{n}{2}\right)}{\Gamma\left(\frac{n-1}{2}\right)\sqrt{\pi}}\sin^{n-2} \label{eq:jqfb}\theta\end{equation}
既然概率密度函数都确定了，那么对于给定的$N$和$n$，近似式$\eqref{eq:H-theta}$是完全可以数值计算出来的，而由$\tilde{H} > H$便可以解出对应的$n$。

### 结果对比

首先我们数值计算出$h_n=\log \mathbb{E}_{\theta}\left[e^{n\cos\theta}\right]-\frac{\mathbb{E}_{\theta}\left[e^{n\cos\theta} n\cos\theta\right]}{\mathbb{E}_{\theta}\left[e^{n\cos\theta}\right]}$的一些结果：
\begin{array}{c|cccccc}
\hline
n & 32 & 64 & 96 & 128 & 256 & 512\\
\hline
h_n & -7.77471 & -15.4734 & -23.1726 & -30.8718 & -61.6692 & -123.264\\
\hline
\end{array}
那么比如$n=64,N=100000$，就有$H\approx \log 100000^2 - 15.4734 = 7.55245$。读者可能会觉得奇怪，当$n=128,N=100000$时，$H$不就是负数了？离散熵怎么可能是负数？事实上，这是因为我们在前面的推导过程中，使用了采样近似和精确积分相结合的方式，当空间维数$n$足够大时，就算你采样几十万个样本也不一定能准确估计一些统计量，所以采样近似这一步带来了误差。

不过这倒是给我们另外一个确定$n$的思路：当出现$H < 0$时，说明$N$个样本已经无法对统计量做很好地估计了，那么反过来说就是此时的$n$维空间要容纳$N$个样本是“绰绰有余”的。因此，我们可以用$H < 0$简单确定一个边界，而不需要去估算$\tilde{H}$。（或者从另外一个角度想：$\tilde{H}$一定是大于0的，因此$H < 0$是$H < \tilde{H}$的充分条件。）

最后，我们看到$h_n$关于$n$大概是线性的，$h_n/n\approx -0.24$，因此$H\approx\log N^2 -0.24n$，让它小于0我们可以解出公式$\eqref{eq:final}$了。

### 近似估计

对于$h_n/n\approx -0.24$这个结果，我们也可以从理论上得到它。根据$h_n$定义，我们有
\begin{equation}h_n = \log\frac{\int_0^{\pi} \sin^{n-2}\theta\, e^{n\cos\theta}d\theta}{\int_0^{\pi} \sin^{n-2}\theta d\theta} - n\frac{\int_0^{\pi} \sin^{n-2}\theta\, e^{n\cos\theta}\cos\theta d\theta}{\int_0^{\pi} \sin^{n-2}\theta\,e^{n\cos\theta} d\theta}\end{equation}
所以，这是几个定积分的渐近估计问题，我们主要利用拉普拉斯近似的思想，假设$n$比较大，以至于我们可以忽略$n$与$n-2$的区别，那么
\begin{equation}\log\left[\sin^{n-2}\theta\, e^{n\cos\theta}\right]=(n-2)\log \sin\theta + n\cos\theta \approx n (\log \sin\theta + \cos\theta)\end{equation}
可以算出$\log\sin\theta + \cos\theta$在$[0,\pi]$的极大值点为$\arctan\sqrt{\frac{\sqrt{5}+1}{2}}\approx 0.904557$，那么在该处展开到二阶，就得到
\begin{equation}\log \sin\theta + \cos\theta\approx 0.377428 -1.11803 (\theta - 0.904557)^2\end{equation}
从而
\begin{equation}\begin{aligned}
\int_0^{\pi} \sin^{n-2}\theta\, e^{n\cos\theta}d\theta\approx&\, \int_{-\infty}^{\infty} e^{n[0.377428 -1.11803 (\theta - 0.904557)^2]}d\theta\\
\approx&\, \frac{1.67629}{\sqrt{n}}e^{0.377428n}
\end{aligned}\end{equation}
以及注意当$n$足够大的时候，被积函数只有在$\theta=0.904557$附近才会明显不等于0（或者将正态分布的极限视为[狄拉克函数](<https://spaces.ac.cn/archives/1870>)的近似），从而有
\begin{equation}\begin{aligned}
\frac{\int_0^{\pi} \sin^{n-2}\theta\, e^{n\cos\theta}\cos\theta d\theta}{\int_0^{\pi} \sin^{n-2}\theta\, e^{n\cos\theta} d\theta}\approx&\, \frac{\int_{-\infty}^{\infty} e^{n[0.377428 -1.11803 (\theta - 0.904557)^2]}\cos\theta d\theta}{\int_{-\infty}^{\infty} e^{n[0.377428 -1.11803 (\theta - 0.904557)^2]}d\theta} \\
\approx&\, \int_{-\infty}^{\infty} \delta(\theta - 0.904557) \cos\theta d\theta\\
=&\,\cos 0.904557 \approx 0.618034
\end{aligned}\end{equation}
同样利用拉普拉斯近似，我们可以得到$\int_0^{\pi} \sin^{n-2}\theta d\theta\approx\frac{2.50663}{\sqrt{n}}$。综合所有结果，我们可以得到
\begin{equation}h_n \approx \log \frac{\frac{1.67629}{\sqrt{n}}e^{0.377428n}}{\frac{2.50663}{\sqrt{n}}}-0.618034n\approx -0.240606 n\end{equation}
这就得到了$h_n/n\approx -0.24$这个结果。上述推导过程用的都是数值值，如果全部保留解析值，那么可以得到这个系数的解析解为$\frac{1}{2}\log\frac{\sqrt{5}+1}{2}$。

## 小结

本文从最小熵原理的思想出发分析了词向量的维度选择问题，最终给出了一个近似的估算公式，计算结果表明该估算公式与我们以往的炼丹经验是相符的。
