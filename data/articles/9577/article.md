# Bias项的神奇作用：RoPE \+ Bias = 更好的长度外推性

> 作者：苏剑林 · 科学空间 · 2023-04-03
>
> 原文：<https://spaces.ac.cn/archives/9577>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

万万没想到，Bias项能跟Transformer的长度外推性联系在一起！

长度外推性是我们希望Transformer具有的一个理想性质，笔者曾在[《Transformer升级之路：7、长度外推性与局部注意力》](<https://spaces.ac.cn/archives/9431>)、[《Transformer升级之路：8、长度外推性与位置鲁棒性》](<https://spaces.ac.cn/archives/9444>)系统地介绍过这一问题。至于Bias项（偏置项），目前的主流观点是当模型足够大时，Bias项不会有什么特别的作用，所以很多模型选择去掉Bias项，其中代表是Google的[T5](<https://spaces.ac.cn/archives/7867>)和[PaLM](<https://papers.cool/arxiv/2204.02311>)，我们后面做的[RoFormerV2](<https://spaces.ac.cn/archives/8998>)和[GAU\-α](<https://spaces.ac.cn/archives/9052>)也沿用了这个做法。

那么，这两个看上去“风牛马不相及”的东西，究竟是怎么联系起来的呢？Bias项真的可以增强Transformer的长度外推性？且听笔者慢慢道来。

## 隐藏彩蛋

首先，为什么会想到考察Bias项和长度外推性的联系呢？这是因为笔者前几天在重温GAU的论文[《Transformer Quality in Linear Time》](<https://papers.cool/arxiv/2202.10447>)时，发现了之前没有在意的一个“隐藏彩蛋”——加性相对位置编码，其伪代码为

[![GAU的加性相对位置编码的伪代码](<https://spaces.ac.cn/usr/uploads/2023/04/1959476500.png>)](<https://spaces.ac.cn/usr/uploads/2023/04/1959476500.png>)

GAU的加性相对位置编码的伪代码

这里我们主要看$n\geq 512$的部分，如果写成公式，大致是
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \quad\to\quad \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n+ \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b}\label{eq:rel-bias}\end{equation}
其中$\boldsymbol{\mathcal{R}}_m,\boldsymbol{\mathcal{R}}_n$是RoPE的旋转矩阵，$\boldsymbol{a},\boldsymbol{b}$是两个可学习参数。

这个加性相对位置编码其实之前也留意到了，但当时的评价只是“不理解为什么同时用几种位置编码”，而最近笔者一直在思考长度外推性问题，所以对这个形式就比较敏感了。可以证明，当$\boldsymbol{a}=\boldsymbol{b}=[\sqrt{\lambda},0,\sqrt{\lambda},0,\cdots,\sqrt{\lambda},0]^{\top}$时，结果正好是[《Transformer升级之路：7、长度外推性与局部注意力》](<https://spaces.ac.cn/archives/9431>)介绍的能改善长度外推性的Sandwich ，其原理就是$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b}$呈现出关于$|m-n|$递减的趋势，加到注意力矩阵上后，能够起到局部化注意力的作用，而根据[《Transformer升级之路：7、长度外推性与局部注意力》](<https://spaces.ac.cn/archives/9431>)，注意力局部化是语言模型外推性的关键。

所以笔者不禁猜测，难道原论文中的这个加性相对位置编码，就是用来增强长度外推性的？GAU的作者竟然如此有先见之明，早在Sandwich之前就提出了类似的想法来解决长度外推性问题？

## 换成偏置

不过，对于笔者来说，这种往Attention矩阵上额外加上一项来增强长度外推性的方案都显得不够优雅，所以不管原作者意图如何以及实际效果如何，笔者都不倾向这样做。有什么类似的但几乎“无感”的方案呢？笔者考虑到，如果$\boldsymbol{a}$、$\boldsymbol{b}$分别是$\boldsymbol{q}_m,\boldsymbol{k}_n$的Bias项，或许可以起到类似的效果，即考虑
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n \quad\to\quad (\boldsymbol{q}_m + \boldsymbol{a})^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n(\boldsymbol{k}_n + \boldsymbol{b})\end{equation}
很明显，单纯增加一个Bias项，不管从形式上还是计算量上看都几乎是“无感”的，如果这样就能增强长度外推性，无疑是一个很漂亮的方案。是否可行呢？我们先来看展开后的结果：
\begin{equation}\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n + \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n + \boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b} + \boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b} \label{eq:bias}\end{equation}
其中第一项和第四项正好对应公式$\eqref{eq:rel-bias}$，它们都是我们想要的，所以我们想看看第二项和第三项起到什么作用，如果它们不会有什么明显的效应，那么直接加上Bias项的做法，至少是“有希望”能够取得跟式$\eqref{eq:rel-bias}$或者Sandwich相似的外推效果。

笔者是这样想的：作为Attention的Query和Key，$\boldsymbol{q}_m$、$\boldsymbol{k}_n$应该是比较“各向同性”的，即它们的方向比较均匀，接近球面上均匀采样，而$\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}$只是一个正交变换，它不改变$\boldsymbol{q}_m$、$\boldsymbol{k}_n$的各向同性性质，那么$\boldsymbol{a}^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{k}_n $、$\boldsymbol{q}_m^{\top}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n\boldsymbol{b}$这两项，就相当于从各向同性分布采样出来的向量，跟一个固定向量的内积，根据我们在[《n维空间下两个随机向量的夹角分布》](<https://spaces.ac.cn/archives/7076>)中的讨论，这样的两个向量夹角应该是很接近90度的，换言之这个内积的期望应该是0，所以第二项和第三项的效应理论上没有剩余两项那么强。

当然，这仅仅是猜测，实际它会训练成怎样，只能通过实验来确定。所以事不宜迟，笔者立刻进行了实验。

## 实验结果

这次笔者选了语言模型任务进行实验，模型架构还是之前的[GAU\-α](<https://spaces.ac.cn/archives/9052>)，训练长度和batch\_size都是512，优化器是[Tiger](<https://spaces.ac.cn/archives/9512>)，两个模型的唯一差别就是Q、K的Bias是否开启（其他Bias仍被去掉）。

外推效果上的对比：
$$\begin{array}{c}
\text{不同测试长度下的LM准确率} \\
{\begin{array}{c|cccc}
\hline
& 512 & 1024 & 2048 & 4096 \\
\hline
\text{w/o Bias} & 52.37\% & 33.15\% & 22.85\% & 17.87\% \\
\text{w/ Bias} & 52.75\% & 50.99\% & 45.25\% & 39.55\% \\
\hline
\end{array}}
\end{array}$$
可以看到，Bias项确实不怎么影响训练效果（512长度），但却在长度外推性上面明显拉开了差距，看似毫无存在感的Bias项居然有此神奇作用！当然，要是重跑几次实验，外推性的结果可能会有明显的波动，毕竟长度外推性属于“赠送功能”，并不是我们主动触发的。

为了验证剩下生效机制是否如我们猜测，笔者可视化了式$\eqref{eq:bias}$的四项在某个样本某一层的变化规律：

[![加上Bias后四项内积对比](<https://spaces.ac.cn/usr/uploads/2023/04/83521782.svg>)](<https://spaces.ac.cn/usr/uploads/2023/04/83521782.svg>)

加上Bias后四项内积对比

可以看到，第4项确确实实呈现衰减趋势，并且其大小占据了主导地位，将这四项叠加起来，与没有加Bias的模型对比如下：

[![有无Bias的Attention矩阵对比](<https://spaces.ac.cn/usr/uploads/2023/04/2535762443.svg>)](<https://spaces.ac.cn/usr/uploads/2023/04/2535762443.svg>)

有无Bias的Attention矩阵对比

没有Bias的模型（蓝色），Attention在训练长度（512）范围内确实也呈现出衰减趋势，但长度增加之后就上升了，没有明显的局部性，这就是它外推性不够好的原因；相反，跟前面的猜测一致，带有Bias项的模型（橙色）的注意力矩阵呈现更明显的衰减趋势，换言之它的局部化效应更加强，从而有更好的外推性能。需要指出的是，加上Bias的模型并不是每一层的Attention都有这么明显的衰减趋势，总体来说前面的层衰减趋势更明显些，后面的层衰减趋势更弱些，说明越靠近输入的层越关注局部信息，这跟[《The Devil in Linear Transformer》](<https://papers.cool/arxiv/2210.10340>)的结论一致。

<strong>【注：后来经过反复测试发现，发现此篇文章的长度外推结果可复现性比较不稳定（可能跟模型结构、超参数等紧密相关），请自行斟酌使用。】</strong>

## 延伸思考

这时候问题就来了：之前做长度外推性的工作不是都验证了RoPE的外推性不大好了吗？难道它们都没加Bias？为此，笔者特意去考证了一下，果然”不出所料”：“开山之作”ALIBI和最近的XPOS都是没有加Bias项的，而KERPLE和Sandwich则是加了Bias项的。之前笔者在读论文的时候，就一直感觉KERPLE和Sandwich中的RoPE外推效果似乎比ALIBI和XPOS中的好，现在可以肯定这应该不是错觉了，既然KERPLE和Sandwich都加了Bias，那么根据本文的结论，RoPE是可能呈现出更好的长度外推性的。

可能有读者想起，之前不是说Attention的Key的Bias可以去掉吗？难道这里也可以去掉？关于这个问题，可以参考知乎的提问[《为什么有的 Vision Transformer 中的 key 不需要 bias ？》](<https://www.zhihu.com/question/506218961>)，事实上，“可以去掉Key的Bias”这个结论，是针对没有RoPE的Attention的，由于Softmax的存在，加上的bias可以约掉：
\begin{equation}\frac{e^{\boldsymbol{q}\cdot(\boldsymbol{k}_n + \boldsymbol{b})}}{\sum\limits_n e^{\boldsymbol{q}\cdot(\boldsymbol{k}_n + \boldsymbol{b})}} = \frac{e^{\boldsymbol{q}\cdot\boldsymbol{k}_n}e^{\boldsymbol{q}\cdot\boldsymbol{b}}}{\sum\limits_n e^{\boldsymbol{q}\cdot\boldsymbol{k}_n} e^{\boldsymbol{q}\cdot\boldsymbol{b}}}= \frac{e^{\boldsymbol{q}\cdot\boldsymbol{k}_n}}{\sum\limits_n e^{\boldsymbol{q}\cdot\boldsymbol{k}_n}}\end{equation}
然而，这个“可以约掉”依赖于$\boldsymbol{b}$跟$n$无关，但从式$\eqref{eq:bias}$我们就知道，经过RoPE后，$\boldsymbol{b}$也算是$m,n$的函数了，实际上是无法约掉的，因此对于加了RoPE的模型，Bias项去掉前后会有不一样的效果。

还有一个问题，就是为什么要费力探索长度外推性呢？直接在更长的样本下微调模型不行吗？事实上，即便是对于抱有这样想法的读者，长度外推性也是有好处的。抛开算力不说，更好的长度外推性意味着在微调的时候与预训练差距更小，于是微调更不容易发生灾难性遗忘，这对于当前的LLM更为重要了。当然，还可以发散一下，最理想的结果是：在短文本学习的模型，能够切换到长文本场景而无损效果甚至效果更优。

## 文章小结

本文分享了笔者发现的一个“万万没想到”的有趣结论：Bias项能增强RoPE模型的长度外推性！看上去毫无存在感的Bias项，居然能跟Transformer的长度外推性联系在一起，让人不得不感叹细节的重要性——细枝末节有时候也能发挥关键作用。
