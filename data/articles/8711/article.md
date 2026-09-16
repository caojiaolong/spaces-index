# 关于维度公式“n \> 8\.33 log N”的可用性分析

> 作者：苏剑林 · 科学空间 · 2021-09-27
>
> 原文：<https://spaces.ac.cn/archives/8711>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在之前的文章[《最小熵原理（六）：词向量的维度应该怎么选择？》](<https://spaces.ac.cn/archives/7695>)中，我们基于最小熵思想推导出了一个词向量维度公式“$n > 8.33\log N$”，然后在[《让人惊叹的Johnson\-Lindenstrauss引理：应用篇》](<https://spaces.ac.cn/archives/8706>)中我们进一步指出，该结果与JL引理所给出的$\mathcal{O}(\log N)$是吻合的。

既然理论上看上去很完美，那么自然就有读者发问了：实验结果如何呢？8\.33这个系数是最优的吗？本文就对此问题的相关内容做一个简单汇总。

## 词向量

首先，我们可以直接，当$N$为10万时，$8.33\log N\approx 96$，当$N$为500万时，$8.33\log N\approx 128$。这说明，至少在数量级上，该公式给出的结果是很符合我们实际所用维度的，因为在词向量时代，我们自行训练的词向量维度也就是100维左右。可能有读者会质疑，目前开源的词向量多数是300维的，像BERT的Embedding层都达到了768维，这不是明显偏离了你的结果了？

事实上，像FastText之类的开源词向量是300维，也没法否定128维能够达到类似效果的可能性。至于BERT，它本身并不是一个词向量模型，所以它选多少维跟词向量维度的选择也没有直接关系，何况ALBERT已经表明，将Embedding层进行低秩分解（降到128维）几乎不会改变模型效果，因此BERT的768维Embedding多多少少是有冗余的。

关于词向量的评价，2015年有一篇比较全面的论文[《How to Generate a Good Word Embedding?》](<https://papers.cool/arxiv/1507.05523>)可以参考，文中显示其实词向量在超过50维之后的提升就比较弱了，这也算是$n > 8.33\log N$的一个佐证吧～

## 注意力

公式$n > 8.33\log N$的另一个间接的实验证明来自注意力机制。在[《让人惊叹的Johnson\-Lindenstrauss引理：应用篇》](<https://spaces.ac.cn/archives/8706>)我们分析过，Attention矩阵的计算公式跟词向量的Skip Gram模型是数学等价的，这就意味着$n > 8.33\log N$这个公式同样可以用于注意力机制的head\_size选择问题。

在注意力机制中，$N$就是要序列长度，常见的预训练长度是512，代入后得到$8.33\log 512\approx 52$，这与当前主流的head\_size大小$64$非常接近，因此这间接证明了$n > 8.33\log N$的可用性。反过来，如果承认这个公式，那么这就解释了注意力机制的head\_size为什么只需要64，也间接解释了注意力机制为什么要多个小的head而不是一个大的head的问题。

关于注意力机制的head\_size选择与表达能力问题，还可以参考[《On the Expressive Power of Self\-Attention Matrices》](<https://papers.cool/arxiv/2106.03764>)。

## 图网络

如果将每个词看成一个节点，将词与词之间的共现看成边，那么Skip Gram也可以视为一个简单的图模型，所以，关于词向量维度的选择结果，理论上也可以用于图网络的嵌入维度选择。

这方面的结果可以参考论文[《Graph Entropy Guided Node Embedding Dimension Selection for Graph Neural Networks》](<https://papers.cool/arxiv/2105.03178>)，文中同时考虑了图的特征熵和结构熵，其中特征熵跟Skip Gram类似，采用了跟[《最小熵原理（六）：词向量的维度应该怎么选择？》](<https://spaces.ac.cn/archives/7695>)同样的近似，所以这部分本质上也是公式$n > 8.33\log N$。

将特征熵和结构熵结合后，用它计算出来的结果作为图网络的嵌入维度来进行各种图任务，实验结果显示该方法确实能得到较优的维度选择结果：

[![基于熵的维度选择](<https://spaces.ac.cn/usr/uploads/2021/09/2266660168.png>)](<https://spaces.ac.cn/usr/uploads/2021/09/2266660168.png>)

基于熵的维度选择

## 小总结

本文对之前导出的维度选择公式$n > 8.33\log N$的可用性进行了分析，综合了词向量、注意力、图网络的已有的一些实验结果，显示出该公式可以得到比较合理的维度估计结果，通过也表明通过熵来进一步确定JL引理中$\log N$的常数也许是一个可行的选择。
