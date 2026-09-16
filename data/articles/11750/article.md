# DeepSeek V4的tid2eid是怎么来的？

> 作者：苏剑林 · 科学空间 · 2026-05-15
>
> 原文：<https://spaces.ac.cn/archives/11750>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

训过MoE的同学都知道，如果把整个模型的MLP部分都换成MoE，那么靠近Embedding的前几层MoE往往很难实现负载均衡。对此，[DeepSeek V3](<https://papers.cool/arxiv/2412.19437>)，包括我们的[Kimi K2](<https://papers.cool/arxiv/2507.20534>)，采取的应对策略是“first\_k\_dense”——顾名思义，就是前$k$层不用MoE而用常规的Dense型GLU。到了[DeepSeek V4](<https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf>)，这个策略改成了“first\_k\_hash”。

这里的“hash”指的是“Hash Routing”，提出自[《Hash Layers For Large Sparse Models》](<https://papers.cool/arxiv/2106.04426>)，它通过一张事先确定的映射表tid2eid来为每个Token分配Expert。本文来探讨一下如何生成这个tid2eid。

## 问题背景

首先要指出的是，对于前几层MoE，如果用笔者在[《MoE环游记：6、最优分配促均衡》](<https://spaces.ac.cn/archives/11619>)和[《MoE环游记：7、动态激活极简解》](<https://spaces.ac.cn/archives/11626>)所提的Quantile Balancing，其实已经能较大程度上缓解它们的不均衡问题。

至于V4采用Hash Routing，可以说是同一问题在不同方向的尝试，其想法是对于前几层MoE来说，上下文信息并不多，所以基于它的输入向量来选Expert，跟直接基于没有任何上下文信息的Token Id来选，也许并无太大区别。既然如此，不如干脆基于Token Id来预先写死前几层每个Token该激活的Expert，这便是“tid2eid （Token Id to Expert Id）”的由来。

那怎么生成tid2eid这个表格呢？完全随机可以吗？看上去不大行，因为每个Token的出现频率并不一样，完全随机的话反而会导致不均衡。DeepSeek并没有公布这个表格的生成细节，我们只能按照自己的思路去猜测一下。

## 数学描述

假设一共有$m$个不同的Token，第$i$个Token的出现频率是$p_i$，不失一般性，假设它们已经按从大到小排列，即$p_i\geq p_{i+1}$。我们用$x_{i,j}\in\{0, 1\}$表示Token $i$是否应该激活Expert $j$（$0$表示不激活，$1$表示激活），共有$n$个Expert，每个Token选$k$个Expert，那么我们实际上要求解方程组
\begin{equation}x_{i,j}\in\{0, 1\},\qquad\sum_{j=1}^n x_{i,j} = k,\qquad \sum_{i=1}^m p_i x_{i,j}\approx \frac{k}{n}\end{equation}
注意最后一个等式我们用了约等于$\approx$，这是因为如果改为$=$，方程组严格来讲未必有解。所以我们保留$\approx$的记号，其含义是两端应该尽可能接近。也可以考虑写成优化形式
\begin{equation}\min_{x_{i,j}\in\{0, 1\}} \sum_{j=1}^n \left(\sum_{i=1}^m p_i x_{i,j} - \frac{k}{n}\right)^2 \qquad\text{s.t.}\qquad\sum_{j=1}^n x_{i,j} = k\end{equation}
该问题的一个比较朴素的解法是贪心算法：

> 按频率从高到低逐Token处理，先统计各Expert的已分配负载，然后选择负载最轻的$k$个Expert作为当前Token的激活Expert，随后更新Expert的负载分布。

## 参考实现

贪心算法虽然原理上是贪心的，但它大多数情况下都能得到一个近乎最优的解，一个参考实现如下：



<pre><code class="language-python">import numpy as np

# 模拟分布
m, n, k = 80000, 128, 4
p = 1 / (10 + np.arange(m))
p /= p.sum()

# 贪心处理
x, f = np.zeros((m, k), dtype='int32'), np.zeros(n)
for i in range(m):
    j = f.argsort()[:k]  # 选择负载最轻的k个
    f[j] += p[i] / k  # 更新负载分布
    x[i] = j  # 记录到tid2eid中

# 评估均衡
max_vio, min_vio = f.max() * n - 1, f.min() * n - 1</code></pre>



注意，该代码没有任何随机的地方，原则上是一个确定性的算法。那如果我们前几层想要不同的tid2eid呢？可以考虑加入一些随机性，比如<code>range(m)</code>换成<code>np.random.permutation(m)</code>，即不按照频率从高到低的顺序执行，这样依然能够得到可用的解，只不过均衡性相对差点。我们可以重复跑多次，然后挑max\_vio最低的几个解。

## 极端情形

现在我们考虑一种极端的情况：$p_1 \gg k / n$，即某个Token的频率已经远超均衡水平，此时无论tid2eid怎么编排，都不可能实现负载均衡。换句话说，只根据单个Token Id来决策，是无论如何也均衡不了的。这种情况下又该如何应对呢？

答案很简单，换一种依赖于更多输入信息的Hash方法。前面我们只根据当前Token Id来决策，实际上每个Token都处于某个序列中，它是有上下文的，如果说之前的做法是“1\-gram to expert”，现在我们可以考虑跟它前面的Token一起，组成“2\-gram to expert”、“3\-gram to expert”等等。

以2\-gram $(a, b)$为例，一种朴素的Hash方式是：选大于$m$的质数$q$，计算$(aq+b)\mod n$作为激活的Expert，用不同的质数重复$k$次就得到$k$个Expert。由于增加了gram数，输入的组合数大大增加，然后映射到有限的$n$个数中，每个数大概率都能被“塞满”，所以只要Hash函数不是特别糟糕，那么结果几乎就是均衡的。

因此，它的优点是不用考虑频率，也不用事先记忆一个tid2eid。至于缺点，就是Hash函数的计算会稍微复杂一些了，而且同一个Token可能会选到重复的Expert，但这些都并不是特别严重的问题。

## 文章小结

本文简单梳理了DeepSeek V4中的Hash Routing的基本思想，并重点探讨了它的tid2eid映射表的构造原理。
