# Transformer升级之路：19、第二类旋转位置编码

> 作者：苏剑林 · 科学空间 · 2025-04-18
>
> 原文：<https://spaces.ac.cn/archives/10862>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

持续将“Transformer升级之路”系列关注到本篇的读者，想必都已经对[旋转位置编码（RoPE）](<https://spaces.ac.cn/archives/8265>)有所了解。简单来说，RoPE是施加在Attention的Query（$\boldsymbol{Q}$）和Key（$\boldsymbol{K}$）上的旋转变换，形式上属于绝对位置编码，但结合Attention的内积（Dot\-Product）特性，能够自动实现相对位置的效果。

那么，RoPE可以加在Value（$\boldsymbol{V}$）上吗？看上去不可以，因为对$\boldsymbol{V}$旋转后就不是相对位置编码了。然而事情并没有那么绝对，本文就来讨论加在$\boldsymbol{V}$上RoPE，我们可以称之为“第二类旋转位置编码”。

## 基础回顾

我们将[Dot\-Product Attention](<https://spaces.ac.cn/archives/4765>)分解为
\begin{equation}\boldsymbol{o}_i = \sum_j a_{i,j}\boldsymbol{v}_j,\qquad a_{i,j} = \frac{e^{s_{i,j}}}{\sum\limits_j e^{s_{i,j}}},\qquad s_{i,j} = \boldsymbol{q}_i^{\top}\boldsymbol{k}_j\end{equation}
简单起见，这里省去了$s_{i,j}$的缩放因子。RoPE应用在$\boldsymbol{q}_i,\boldsymbol{k}_j$上：
\begin{equation}\boldsymbol{q}_i \to \boldsymbol{\mathcal{R}}_i\boldsymbol{q}_i,\qquad \boldsymbol{k}_j \to \boldsymbol{\mathcal{R}}_j\boldsymbol{k}_j\end{equation}
这将导致Attention Logits也就是$s_{i,j}$变成
\begin{equation}s_{i,j} = (\boldsymbol{\mathcal{R}}_i\boldsymbol{q}_i)^{\top} (\boldsymbol{\mathcal{R}}_j\boldsymbol{k}_j) = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}_i^{\top}\boldsymbol{\mathcal{R}}_j\boldsymbol{k}_j=\boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{k}_j\end{equation}
也就是说$s_{i,j}$只依赖于相对位置$j-i$，从而通过绝对位置形式达到相对位置的效果。这个变换过程利用了旋转矩阵的特性$\boldsymbol{\mathcal{R}}_i^{\top}\boldsymbol{\mathcal{R}}_j=\boldsymbol{\mathcal{R}}_{j-i}$。

除了旋转矩阵外，在[《Transformer升级之路：4、二维位置的旋转式位置编码》](<https://spaces.ac.cn/archives/8397>)中我们证明了它的一般解是$\boldsymbol{\mathcal{R}}_i = \boldsymbol{O}^i$，其中$\boldsymbol{O}$是任意正交矩阵，上标是矩阵的幂运算。不过后来我们在[《Transformer升级之路：6、旋转位置编码的完备性分析》](<https://spaces.ac.cn/archives/9403>)也证明了其实一般的正交矩阵解本质上也同构于旋转矩阵解。

## 新的用法

如果将RoPE加在$\boldsymbol{v}_j$上，即$\boldsymbol{v}_j\to\boldsymbol{\mathcal{R}}_j\boldsymbol{v}_j$，那又如何呢？显然Attention的结果是
\begin{equation}\boldsymbol{o}_i = \sum_j a_{i,j} \boldsymbol{\mathcal{R}}_j\boldsymbol{v}_j\label{eq:v-rope-abs}\end{equation}
这将会导致Attention显式依赖于绝对位置$j$。如果我们只想要一种位置编码，那么也许问题不大，但如果我们是想要一种相对位置编码，那么它就不能满足我们的目的。

然而，有一个简单的技巧可以解决这个缺陷！我们可以给$\boldsymbol{o}_i$再加一次逆向的RoPE：
\begin{equation}\boldsymbol{o}_i = \boldsymbol{\mathcal{R}}_i^{\top}\left(\sum_j a_{i,j} \boldsymbol{\mathcal{R}}_j\boldsymbol{v}_j\right)=\sum_j a_{i,j} \boldsymbol{\mathcal{R}}_i^{\top}\boldsymbol{\mathcal{R}}_j\boldsymbol{v}_j=\sum_j a_{i,j} \boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{v}_j\label{eq:vo-rope}\end{equation}
这样它再次变成了一个相对位置编码！而形式上同样也是两次绝对位置编码，跟已有的RoPE异曲同工，所以我们称之为“第二类旋转位置编码”，也可以更直观地称为“VO\-RoPE”，因为它分别在Value和Output都加了一次RoPE，相应地，标准的RoPE我们可以称之为“QK\-RoPE”。

## 简单实验

在一个1B左右的类LLAMA模型上快速做了一波实验，对比的几个设置为：

> 1、NoPE：完全不加位置编码；
> 
> 
> 
> 2、QK\-RoPE：标准的旋转位置编码；
> 
> 
> 
> 3、VO\-RoPE：本文新提出的第二类旋转位置编码；
> 
> 
> 
> 4、Q/K/V/O\-RoPE：单独在Q、K、V、O之一加旋转位置编码；
> 
> 
> 
> 5、QKV\-RoPE：Q、K、V都加上旋转位置编码；
> 
> 
> 
> 6、QKVO\-RoPE：Q、K、V、O都加上旋转位置编码。

注意，第4、5点都算是绝对位置编码。大致结论是：
$$\text{QK-RoPE}\approx \text{QKVO-RoPE} > \text{K-RoPE}\approx \text{VO-RoPE} > \text{QKV-RoPE} > \text{NoPE} > \text{Q/V/O-RoPE}$$

具体损失函数差异是：
\begin{array}{c|c}
\hline
& \text{Loss} \\
\hline
\text{QK-RoPE} &  2.712 \\
\text{QKVO-RoPE} &  2.719 \\
\text{K-RoPE} &  2.769 \\
\text{VO-RoPE} &  2.770 \\
\text{QKV-RoPE} &  2.783 \\
\text{NoPE} &  2.795 \\
\text{O-RoPE} &  2.841 \\
\text{Q-RoPE} &  2.851 \\
\text{V-RoPE} &  2.856 \\
\hline
\end{array}

## 一些思考

从上述结果可以看出，VO\-RoPE优于NoPE，但不如QK\-RoPE，而且VO\-RoPE和QK\-RoPE叠加并不没有增益。这样看来，VO\-RoPE似乎没有提出的必要了？

在笔者看来，将RoPE的用法补充完整，回答“RoPE可以加在Value上吗”这个问题，然后实验清楚“没有什么收益”这件事，本身就很有价值。而且，从长远来看它不见得就一直没有收益，只是在我们当前主流言语模型设置下它可能体现不出什么作用。当时笔者提出RoPE时，动机也单纯是好玩而已，并没有期望它是有竞争力的位置编码（后来的事则是幸运了）。

就当前来看，VO\-RoPE也有一个潜在应用场景，它跟[《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》](<https://spaces.ac.cn/archives/10091>)介绍的MLA有关。我们知道，MLA在推理阶段约等于一个K、V共享的MQA：
\begin{equation}\boldsymbol{o}_i = \sum_{j=1}^i a_{i,j}\boldsymbol{c}_j,\qquad a_{i,j} = \frac{e^{s_{i,j}}}{\sum\limits_{j=1}^i e^{s_{i,j}}},\qquad s_{i,j} = \exp(\boldsymbol{q}_i^{\top}\boldsymbol{c}_j)\end{equation}
这个特性使得它的KV Cache只有一个$\boldsymbol{c}$。然而，这个重要特性与QK\-RoPE并不兼容，因为一旦给Attention矩阵里边的$\boldsymbol{c}_j$加上RoPE，那么就有两种结果：

> 1、Value这边的$\boldsymbol{c}_j$不加RoPE，那么K、V就不完全共享了，这就导致了要不KV Cache翻倍（RoPE前后都要Cache），要不K实时注入RoPE（带来了延迟）；
> 
> 
> 
> 2、如果Value这边的$\boldsymbol{c}_j$加RoPE，倒是可以达到K、V共享的效果，但此时就不是相对位置编码了。

MLA为了解决这个问题，采用了“大部分NoPE\+小部分RoPE”拼接的做法。但是，从本文的第二类旋转位置编码我们知道，只需要再给Output加一次O\-RoPE就行了：
\begin{equation}\boldsymbol{o}_i = \boldsymbol{\mathcal{R}}_i^{\top}\sum_{j=1}^i a_{i,j}(\boldsymbol{\mathcal{R}}_j\boldsymbol{c}_j),\qquad a_{i,j} = \frac{e^{s_{i,j}}}{\sum\limits_{j=1}^i e^{s_{i,j}}},\qquad s_{i,j} = (\boldsymbol{\mathcal{R}}_i\boldsymbol{q}_i)^{\top} (\boldsymbol{\mathcal{R}}_j\boldsymbol{c}_j)\end{equation}
不过，这个思路还没完全走通，还无法直接用在MLA的训练形式上，只是先写出来给大家参考。

## 相关工作

事实上，VO\-RoPE还巧妙地提供了一个从Attention到复线性RNN（如[LRU](<https://spaces.ac.cn/archives/9554>)、[RetNet](<https://papers.cool/arxiv/2307.08621>)）的中间形式。我们从式$\eqref{eq:vo-rope}$出发，考虑Causal场景，然后取一个特殊例子$a_{i,j}=\gamma^{i-j}$，其中$0 < \gamma < 1$，那么得到
\begin{equation}\boldsymbol{o}_i = \sum_{j=1}^i \gamma^{i-j} \boldsymbol{\mathcal{R}}_{j-i}\boldsymbol{v}_j\end{equation}
我们知道旋转矩阵$\boldsymbol{\mathcal{R}}_{j-i}$用复数形式写其实就是$e^{\mathbb{I}\theta (j - i)}$的对角阵，其中$\mathbb{I}$是虚数单位（即$\mathbb{I}^2=-1$)，为了区别$i,j$的$i$这里写成了$\mathbb{I}$。这样一来，上式相当于
\begin{equation}\boldsymbol{o}_i = \sum_{j=1}^i \gamma^{i-j} e^{\mathbb{I}\theta (j - i)} \boldsymbol{v}_j = \sum_{j=1}^i (\gamma e^{-\mathbb{I}\theta})^{i-j} \boldsymbol{v}_j\end{equation}
这其实就是最简单的带有复数Decay的线性RNN。从[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](<https://spaces.ac.cn/archives/9554>)的推导来看，这种RNN在理论上比纯实数Decay的RNN更完备。

所以说，给RoPE补上VO\-RoPE形式，相当于从实线性RNN到复线性RNN的一般推广，理论上能让它能力更完整，尽管这种更完整在语言模型任务上不见得有什么帮助，正如引入复数的LRU并没有比纯实数的RWKV有优势。但，理论上的完备也许隐含着某些场景的特殊价值，谁知道呢～

> <strong>番外</strong>：在推特分享本文后，有一些读者反馈他们曾经尝试过VO\-RoPE，包括
> 
> 
> 
> 1、[@gharik](<https://x.com/gharik/status/1913379569870213200>) 表示他在之前曾尝试过VO\-RoPE，并取得了一些正面的结果，他当时命名为“RoPER”，更多细节可以参考[这里](<https://research.labml.ai/RoPER.html>)和[这里](<https://nn.labml.ai/transformers/rope/value_pe/index.html>)；
> 
> 
> 
> 2、[@vinam\_arora](<https://x.com/vinam_arora/status/1913714691408343457>) 指出他在“脑部解码任务”尝试过QKVO\-RoPE，并且结果也是正面的，论文为[《A Unified, Scalable Framework for Neural Population Decoding》](<https://papers.cool/arxiv/2310.16046>)。

## 文章小结

本文围绕着“RoPE可以加在V上吗”进行展开，讨论了RoPE的第二种用法。
