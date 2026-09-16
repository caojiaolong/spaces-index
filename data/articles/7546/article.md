# 线性Attention的探索：Attention必须有个Softmax吗？

> 作者：苏剑林 · 科学空间 · 2020-07-04
>
> 原文：<https://spaces.ac.cn/archives/7546>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，尽管基于Attention机制的Transformer类模型有着良好的并行性能，但它的空间和时间复杂度都是$\mathcal{O}(n^2)$级别的，$n$是序列长度，所以当$n$比较大时Transformer模型的计算量难以承受。近来，也有不少工作致力于降低Transformer模型的计算量，比如模型剪枝、量化、蒸馏等精简技术，又或者修改Attention结构，使得其复杂度能降低到$\mathcal{O}(n\log n)$甚至$\mathcal{O}(n)$。

前几天笔者读到了论文[《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》](<https://papers.cool/arxiv/2006.16236>)，了解到了线性化Attention（Linear Attention）这个探索点，继而阅读了一些相关文献，有一些不错的收获，最后将自己对线性化Attention的理解汇总在此文中。

## Attention

当前最流行的Attention机制当属[Scaled\-Dot Attention](<https://papers.cool/arxiv/1706.03762>)，形式为
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\boldsymbol{Q}\boldsymbol{K}^{\top}\right)\boldsymbol{V}\label{eq:std-att}\end{equation}
这里的$\boldsymbol{Q}\in\mathbb{R}^{n\times d_k}, \boldsymbol{K}\in\mathbb{R}^{m\times d_k}, \boldsymbol{V}\in\mathbb{R}^{m\times d_v}$，简单起见我们就没显式地写出Attention的缩放因子了。本文我们主要关心Self Attention场景，所以为了介绍上的方便统一设$\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}\in\mathbb{R}^{n\times d}$，一般场景下都有$n > d$甚至$n\gg d$（BERT base里边$d=64$）。相关解读可以参考笔者的[《Attention is All You Need》浅读（简介\+代码）](<https://spaces.ac.cn/archives/4765>)，以及它的一些改进工作也可以参考[《突破瓶颈，打造更强大的Transformer》](<https://spaces.ac.cn/archives/7325>)、[《Google新作Synthesizer：我们还不够了解自注意力》](<https://spaces.ac.cn/archives/7430>)，这里就不多深入介绍了。

### 摘掉Softmax

读者也许想不到，制约Attention性能的关键因素，其实是定义里边的Softmax！事实上，简单地推导一下就可以得到这个结论。$\boldsymbol{Q}\boldsymbol{K}^{\top}$这一步我们得到一个$n\times n$的矩阵，就是这一步决定了Attention的复杂度是$\mathcal{O}(n^2)$；如果没有Softmax，那么就是三个矩阵连乘$\boldsymbol{Q}\boldsymbol{K}^{\top}\boldsymbol{V}$，而矩阵乘法是满足结合率的，所以我们可以先算$\boldsymbol{K}^{\top}\boldsymbol{V}$，得到一个$d\times d$的矩阵，然后再用$\boldsymbol{Q}$左乘它，由于$d \ll n$，所以这样算大致的复杂度只是$\mathcal{O}(n)$（就是$\boldsymbol{Q}$左乘那一步占主导）。

也就是说，去掉Softmax的Attention的复杂度可以降到最理想的线性级别$\mathcal{O}(n)$！这显然就是我们的终极追求：Linear Attention，复杂度为线性级别的Attention。所以，本文的主题就是探究摘掉Softmax后的线形Attention。

### 一般的定义

问题是，直接去掉Softmax还能算是Attention吗？它还能有标准的Attention的效果吗？为了回答这个问题，我们先将Scaled\-Dot Attention的定义$\eqref{eq:std-att}$等价地改写为（本文的向量都是列向量）
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n e^{\boldsymbol{q}_i^{\top}\boldsymbol{k}_j}\boldsymbol{v}_j}{\sum\limits_{j=1}^n e^{\boldsymbol{q}_i^{\top}\boldsymbol{k}_j}}\label{eq:std-att-2}\end{equation}
所以，Scaled\-Dot Attention其实就是以$e^{\boldsymbol{q}_i^{\top}\boldsymbol{k}_j}$为权重对$\boldsymbol{v}_j$做加权平均。所以我们可以提出一个Attention的一般化定义
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)}\label{eq:gen-att}\end{equation}
也就是把$e^{\boldsymbol{q}_i^{\top}\boldsymbol{k}_j}$换成$\boldsymbol{q}_i, \boldsymbol{k}_j$的一般函数$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)$，为了保留Attention相似的分布特性，我们要求$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\geq 0$恒成立。也就是说，我们如果要定义新式的Attention，那么要保留式$\eqref{eq:gen-att}$的形式，并且满足$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\geq 0$。

这种一般形式的Attention在CV中也被称为Non\-Local网络，出自论文[《Non\-local Neural Networks》](<https://spaces.ac.cn/archives/1711.07971>)。

## 几个例子

如果直接去掉Softmax，那么就是$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = \boldsymbol{q}_i^{\top}\boldsymbol{k}_j$，问题是内积无法保证非负性，所以这还不是一个合理的选择。下面我们简单介绍几种可取的方案。

值得指出的是，下面介绍的这几种Linear Attention，前两种来自CV领域，第三种是笔者自己构思的，所以都还没有在NLP任务上做过什么实验，各位做模型改进的NLPer们就有实验方向了（^\_^）～～顺便说一下，CV领域有不少对Attention的改进工作（除了下面介绍的外，还有[EMANet](<https://papers.cool/arxiv/1907.13426>)等），很多内容都值得做NLP的我们参考阅读。

### 核函数形式

一个自然的想法是：如果$\boldsymbol{q}_i,\boldsymbol{k}_j$的每个元素都是非负的，那么内积自然也就是非负的。为了完成这点，我们可以给$\boldsymbol{q}_i,\boldsymbol{k}_j$各自加个激活函数$\phi,\varphi$，即
\begin{equation}\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\label{eq:gen-att-2}\end{equation}
其中$\phi(\cdot),\varphi(\cdot)$是值域非负的激活函数。本文开头提到的论文[《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》](<https://papers.cool/arxiv/2006.16236>)选择的是$\phi(x)=\varphi(x)=\text{elu}(x)+1$。

非要讲故事的话，式$\eqref{eq:gen-att-2}$可以联想到“核方法（kernal method）”，尤其是$\phi=\varphi$时$\phi$就相当于一个核函数，而$\langle \phi(\boldsymbol{q}_i), \phi(\boldsymbol{k}_j)\rangle$就是通过核函数所定义的内积。这方面的思考可以参考论文[《Transformer dissection: An unified understanding for transformer’s attention via the lens of kernel》](<https://papers.cool/arxiv/1908.11775>)，此处不做过多延伸。

### 妙用Softmax

另一篇更早的文章[《Efficient Attention: Attention with Linear Complexities》](<https://papers.cool/arxiv/1812.01243>)则给出了一个更有意思的选择。它留意到在$\boldsymbol{Q}\boldsymbol{K}^{\top}$中，$\boldsymbol{Q}, \boldsymbol{K}, \in\mathbb{R}^{n\times d}$，如果“$\boldsymbol{Q}$在$d$那一维是归一化的、并且$\boldsymbol{K}$在$n$那一维是归一化的”，那么$\boldsymbol{Q}\boldsymbol{K}^{\top}$就是自动满足归一化了，所以它给出的选择是：
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax_2\left(\boldsymbol{Q}\right)softmax_1(\boldsymbol{K})^{\top}\boldsymbol{V}\end{equation}
其中$softmax_1$、$softmax_2$分别指在第一个（$n$）、第二个维度（$d$）进行Softmax运算。也就是说，这时候我们是各自给$\boldsymbol{Q},\boldsymbol{K}$加Softmax，而不是$\boldsymbol{Q}\boldsymbol{K}^{\top}$算完之后才加Softmax。

如果直接取$\phi(\boldsymbol{q}_i)=softmax(\boldsymbol{q}_i),\varphi(\boldsymbol{k}_j)=softmax(\boldsymbol{k}_j)$，那么很显然这个形式也是式$\eqref{eq:gen-att-2}$的一个特例。另外这个设计在CV中出现过不止一次，比如[A<sup>2</sup>\-Nets](<https://papers.nips.cc/paper/7318-a2-nets-double-attention-networks.pdf>)也包含了同样的做法。

### 自己的构思

在这里，笔者给出自己的一种构思。这个构思的出发点不再是式$\eqref{eq:gen-att-2}$，而是源于我们对原始定义$\eqref{eq:std-att-2}$的近似。由泰勒展开我们有
\begin{equation}e^{\boldsymbol{q}_i^{\top}\boldsymbol{k}_j} \approx 1 + \boldsymbol{q}_i^{\top}\boldsymbol{k}_j\end{equation}
如果$\boldsymbol{q}_i^{\top}\boldsymbol{k}_j\geq -1$，那么就可以保证右端的非负性，而从可以让$\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)=1 + \boldsymbol{q}_i^{\top}\boldsymbol{k}_j$。到这里读者可能已经想到了，想要保证$\boldsymbol{q}_i^{\top}\boldsymbol{k}_j\geq -1$，只需要分别对$\boldsymbol{q}_i,\boldsymbol{k}_j$做$l_2$归一化。所以，笔者最终提出的方案就是：
\begin{equation}\text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j) = 1 + \left( \frac{\boldsymbol{q}_i}{\Vert \boldsymbol{q}_i\Vert}\right)^{\top}\left(\frac{\boldsymbol{k}_j}{\Vert \boldsymbol{k}_j\Vert}\right)\end{equation}
这不同于形式$\eqref{eq:gen-att-2}$，但理论上它更加接近原始的Scaled\-Dot Attention。

## 相关工作

通过修改Attention的形式来降低它的计算复杂度，相关的工作有很多，这里简要列举一些。

### 稀疏Attention

我们之前介绍过OpenAI的[Sparse Attention](<https://spaces.ac.cn/archives/6853#Sparse%20Self%20Attention>)，通过“只保留小区域内的数值、强制让大部分注意力为零”的方式，来减少Attention的计算量。经过特殊设计之后，Attention矩阵的大部分元素都是0，因此理论上它也能节省显存占用量和计算量。后续类似工作还有[《Explicit Sparse Transformer: Concentrated Attention Through Explicit Selection》](<https://papers.cool/arxiv/1912.11637>)、[《Longformer: The Long\-Document Transformer》](<https://papers.cool/arxiv/2004.05150>)等。

但是很明显，这种思路有两个不足之处：

> 1、如何选择要保留的注意力区域，这是人工主观决定的，带有很大的不智能性；
> 
> 
> 
> 2、它需要从编程上进行特定的设计优化，才能得到一个高效的实现，所以它不容易推广。

### Reformer

[Reformer](<https://papers.cool/arxiv/2001.04451>)也是有代表性的改进工作，它将Attention的复杂度降到了$\mathcal{O}(n\log n)$。某种意义上来说，Reformer也是稀疏Attention的一种，只不过它的稀疏Pattern不是事先指定的，而是通过LSH（Locality Sensitive Hashing）技术（近似地）快速地找到最大的若干个Attention值，然后只去计算那若干个值。此外，Reformer通过构造可逆形式的FFN（Feedforward Network）替换掉原来的FFN，然后重新设计反向传播过程，从而降低了显存占用量。

所以，相比前述稀疏Attention，Reformer解决了它的第一个缺点，但是依然有第二个缺点：实现起来复杂度高。要实现LSH形式的Attention比标准的Attention复杂多了，对可逆网络重写反向传播过程对普通读者来说更是遥不可及～

### Linformer

跟本文所介绍的Linear Attention很相似的一个工作是Facebook最近放出来的[Linformer](<https://papers.cool/arxiv/2006.04768>)，它依然保留原始的Scaled\-Dot Attention形式，但在进行Attention之前，用两个$m\times n$的矩阵$\boldsymbol{E},\boldsymbol{F}$分别对$\boldsymbol{K},\boldsymbol{V}$进行投影，即变为
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\boldsymbol{Q}(\boldsymbol{E}\boldsymbol{K})^{\top}\right)\boldsymbol{F}\boldsymbol{V}\end{equation}
这样一来，$\boldsymbol{Q}(\boldsymbol{E}\boldsymbol{K})^{\top}$就只是一个$n\times m$的矩阵，而作者声称对于哪怕对于很大的序列长度$n$，$m$也可以保持为一个适中的常数，从而这种Attention也是线性的。跟Linformer类似的思路还出现在更早一些的CV论文[《Asymmetric Non\-local Neural Networks for Semantic Segmentation》](<https://papers.cool/arxiv/1907.13426>)中。

但是，笔者认为“对于超长序列$m$可以保持不变”这个结论是值得质疑的，对于长序列原论文只做了MLM任务，而很明显MLM并不那么需要长程依赖，所以这个实验没什么说服力。因此，Linformer是不是真的Linear，还有待商榷。

### 自回归生成

Linformer的另一个缺点是$\boldsymbol{E}\boldsymbol{K},\boldsymbol{F}\boldsymbol{V}$这两个运算直接把整个序列的信息给“糅合”起来了，所以它没法简单地把将来信息给Mask掉（Causal Masking），从而无法做语言模型、Seq2Seq等自回归生成任务，这也是刚才说的原作者只做了MLM任务的原因。相比之下，本文介绍的几种Linear Attention都能做到这一点。以式$\eqref{eq:gen-att}$和式$\eqref{eq:gen-att-2}$为例，如果要Mask掉未来信息，那么只需要把求和$\sum\limits_{j=1}^n$改为$\sum\limits_{j=1}^i$：
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^i \left(\phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\right)\boldsymbol{v}_j}{\sum\limits_{j=1}^i \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}=\frac{ \phi(\boldsymbol{q}_i)^{\top} \sum\limits_{j=1}^i\varphi(\boldsymbol{k}_j)\boldsymbol{v}_j^{\top}}{ \phi(\boldsymbol{q}_i)^{\top} \sum\limits_{j=1}^i\varphi(\boldsymbol{k}_j)}\end{equation}
实现上式有两种方式：第一方式是设$\boldsymbol{S}_i=\sum\limits_{j=1}^i\varphi(\boldsymbol{k}_j)\boldsymbol{v}_j^{\top}$以及$\boldsymbol{z}_i=\sum\limits_{j=1}^i\varphi(\boldsymbol{k}_j)$，我们有
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i =\frac{ \phi(\boldsymbol{q}_i)^{\top} \boldsymbol{S}_i}{ \phi(\boldsymbol{q}_i)^{\top} \boldsymbol{z}_i},\quad \begin{aligned}&\boldsymbol{S}_i=\boldsymbol{S}_{i-1}+\varphi(\boldsymbol{k}_i)\boldsymbol{v}_i^{\top}\\
&\boldsymbol{z}_i=\boldsymbol{z}_{i-1}+\varphi(\boldsymbol{k}_i)
\end{aligned}\end{equation}
这说明这种Attention可以作为一个RNN模型用递归的方式实现，它的空间复杂度最低，但是要串性计算，适合预测解码时使用；第二种是直接将$\varphi(\boldsymbol{K}),\boldsymbol{V}\in\mathbb{R}^{n\times d}$做外积，得到一个$n\times d\times d$的矩阵，然后对$n$那一维执行$\text{cumsum}$运算，这样就一次性得到$\boldsymbol{S}_1,\boldsymbol{S}_2,\dots,\boldsymbol{S}_n$了，它的速度最快，但空间占用最大，适合训练时使用，不过很多时候都有$d^2\gg n$，一般情况下训练时都很难承受这个空间复杂度，因此多数还是用RNN形式。

### 下采样技术

从结果上来看，Linformer的$\boldsymbol{E}\boldsymbol{K}, \boldsymbol{F}\boldsymbol{V}$就是将序列变短（下采样）了，而将序列变短的一个最朴素的方法就是Pooling了，所以笔者之前也尝试过把Pooling技术引入到Transformer中去。近来也有类似的工作发出来，比如IBM的[《PoWER\-BERT: Accelerating BERT Inference via Progressive Word\-vector Elimination》](<https://papers.cool/arxiv/2001.08950>)和Google的[《Funnel\-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing》](<https://papers.cool/arxiv/2006.03236>)。除了Pooling之外，其实还有其他的下采样技术，比如可以通过stride \> 1的一维卷积来实现，基于这个思路，或许我们可以把FFN里边的Position\-Wise全连接换成stride \> 1的一维卷积？总之这方面应该也能玩出很多花样来，不过跟Linformer一样，这样糅合之后做自回归生成就很难了。

## 文章小结

本文介绍了一些从结构上对Attention进行修改从而降低其计算复杂度的工作，其中最主要的idea是去掉标准Attention中的Softmax，就可以使得Attention的复杂度退化为理想的$\mathcal{O}(n)$级别（Linear Attention）。相比于其他类似的改进结构的工作，这种修改能在把复杂度降到$\mathcal{O}(n)$的同时，依然保留所有的“token\-token“的注意力，同时还能保留用于做自回归生成的可能性。
