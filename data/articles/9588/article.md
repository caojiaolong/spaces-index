# 从JL引理看熵不变性Attention

> 作者：苏剑林 · 科学空间 · 2023-04-10
>
> 原文：<https://spaces.ac.cn/archives/9588>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《从熵不变性看Attention的Scale操作》](<https://spaces.ac.cn/archives/8823>)、[《熵不变性Softmax的一个快速推导》](<https://spaces.ac.cn/archives/9034>)中笔者提出了熵不变性Softmax，简单来说就是往Softmax之前的Attention矩阵多乘上一个$\log n$，理论上有助于增强长度外推性，其中$n$是序列长度。$\log n$这个因子让笔者联系到了JL引理（[Johnson\-Lindenstrauss引理](<https://spaces.ac.cn/archives/8679>)），因为JL引理告诉我们编码$n$个向量只需要$\mathcal{O}(\log n)$的维度就行了，大家都是$\log n$，这两者有没有什么关联呢？

## 熵不变性

我们知道，熵是不确定性的度量，用在注意力机制中，我们将它作为“集中注意力的程度”。所谓熵不变性，指的是不管序列长度$n$是多少，我们都要将注意力集中在关键的几个token上，而不要太过分散。为此，我们提出的熵不变性Attention形式为
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\log_{512} n}{\sqrt{d}}QK^{\top}\right)V\label{eq:core}\end{equation}
这里$Q,K\in\mathbb{R}^{n\times d}$。跟常规的Attention相比，就是scale的因子多了个$\log_{512} n$，其中底数取512，是假设我们所有的超参数（比如$d$）都是为训练长度512调好的。当然，即便你计划中的预训练长度不是512，底数也可以直接无脑取512，结果基本不会有什么影响。

这个形式的原理也很直观，当$n$增大时，意味着有更多的token去平摊了注意力，导致注意力不集中，此时我们乘上一个关于$n$单调递增的因子，softmax之后它实际上就相当于原来概率的幂运算，由于概率都小于1，所以概率越小幂运算之后会变得更小，这样注意力重新变得集中起来。至于这个因子为什么是对数的形式，那就需要看开头文章的推导过程了。

## JL引理

JL引理，全称“Johnson\-Lindenstrauss引理”，是关于向量嵌入的一个重要结论，简单来说它就是告诉我们“要塞下$n$个向量，只需$\mathcal{O}(\log n)$维空间”（这里的$\log$没有写出底数，默认都是以自然对数$e$为底），详细介绍可以参考[《让人惊叹的Johnson\-Lindenstrauss引理：理论篇》](<https://spaces.ac.cn/archives/8679>)。

有意思的是，早在笔者知道JL引理之前，就在[《最小熵原理（六）：词向量的维度应该怎么选择？》](<https://spaces.ac.cn/archives/7695>)推导过同样的、甚至更具体的结果——嵌入$n$个词向量，大致上需要$8\log n$维空间就行了。这个估计跟实际使用的维度很接近，比如$n$等于10万时，$8\log n$算出来大概是92，而我们经常用的词向量维度也是一两百维这个量级。

另外，JL引理还可以用来解释注意力机制的多头性。如果代入$n=512$，那么$8\log n\approx 50$，这跟Attention的Q、K常用的投影维度（也就是key\_size，BERT里边是64，参考[这里](<https://spaces.ac.cn/archives/7325#Attention%E9%87%8C%E6%9C%89%E4%B8%AA%E7%93%B6%E9%A2%88>)）很接近，这就告诉我们，如果序列长度时512，那么算Attention的Q、K的维度在50这个量级就够了，没必要用全部的hidden\_size（BERT base是768），省下来的维度可以转而用来做多头注意力。

更多相关讨论可以参考[《关于维度公式“n \> 8\.33 log N”的可用性分析》](<https://spaces.ac.cn/archives/8711>)、[《让人惊叹的Johnson\-Lindenstrauss引理：应用篇》](<https://spaces.ac.cn/archives/8706>)。

## 联系起来

现在，我们就可以尝试JL引理跟熵不变性Attention联系起来了。

我们将Q、K的key\_size记为$d$，那么JL引理告诉我们，$d$的最佳选择应该是$d_n=\lambda \log n$，这里的$\lambda$是比例常数，具体是多少不重要。也就是说，理想情况下，$d$应该随着$n$的变化而变化，但很显然这样的设计并不容易实现，也不利于计算的并行化，所以实际情况下我们都只能使用固定的$d$。

假设我们选定了一个固定的$d$，并且假设这个$d$是为训练长度512设计的，那么我们可以得出$d = \lambda \log 512$，也就是$\lambda = \frac{d}{\log 512}$，以及
\begin{equation}d_n = \frac{d}{\log 512}\log n=d\log_{512} n\end{equation}
对于$n\neq 512$，理想情况下应该用$d_n$维的投影维度，但实际用了$d$维，根据内积的定义$\langle q,k\rangle = \sum\limits_{i=1}^d q_i k_i$，求和的项数正好等于维度数$d$，也就是说，理想情况下应该是$d_n$项求和，但实际上变为了$d$项求和，那么直觉上来看，如果每一项的贡献接近，那么我们将结果乘以$\frac{d_n}{d}$后，能够让结果更接近$d_n$项求和的理想情况，所以我们就得出，应当往$\langle q,k\rangle$中乘上因子
\begin{equation}\frac{d_n}{d} = \log_{512} n\end{equation}
来弥补实际情况与理想情况的差距。而常规的Scaled\-Dot Attention乘上$\log_{512} n$后，正好是熵不变性Attention，也就是式$\eqref{eq:core}$。

这样，我们就将JL引理跟熵不变性Attention联系了起来。注意这只是个直观的、定性的理解过程，很难从定量角度将它进一步严格化，事实上也没有必要进一步定量化了，因为JL引理本身更多也只是一个定性的结论。

## 文章小结

本文构建了JL引理与熵不变性Attention之间的一个简单联系。
