# 相对位置编码Transformer的一个理论缺陷与对策

> 作者：苏剑林 · 科学空间 · 2022-06-07
>
> 原文：<https://spaces.ac.cn/archives/9105>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

位置编码是Transformer中很重要的一环，在[《让研究人员绞尽脑汁的Transformer位置编码》](<https://spaces.ac.cn/archives/8130>)中我们就总结了一些常见的位置编码设计。大体上，我们将Transformer的位置编码分为“绝对位置编码”和“相对位置编码”两类，其中“相对位置编码”在众多NLP/CV的实验表现相对来说更加好些。

然而，我们可以发现，目前相对位置编码几乎都是在Softmax之前的Attention矩阵上进行操作的，这种施加方式实际上都存在一个理论上的缺陷，使得Transformer无法成为“万能拟合器”。本文就来分析这个问题，并探讨一些解决方案。

## 简单探针

顾名思义，位置编码就是用来给模型补充上位置信息的。那么，如何判断一个模型有没有足够的识别位置的能力呢？笔者之前曾构思过一个简单的探针实验：

> 对于一个有识别位置能力的模型，应该有能力准确实现如下映射
> \begin{equation}\begin{array}{lc}
\text{输入：} & [0, 0, \cdots, 0, 0] \\
& \downarrow\\
\text{输出：} & [1, 2, \cdots, n-1, n]
\end{array}\end{equation}

也就是说，输入$n$个0，能有序地输出位置编号$1\sim n$。这个探针实验的思想很简单，即模型如果有能力做到这一点，说明识别位置是模型自身具备的能力，跟外部输入无关，这正是我们想要的。不难发现，绝对位置由于是直接施加在输入上的，所以它很容易能够完成探针测试。

## 无法胜任

然而，当笔者带着这个简单的探针实验去思考带有相对位置编码的Transformer模型时，却发现它们几乎都不能完成上述任务。

具体来说，除了[《Self\-Attention with Relative Position Representations》](<https://papers.cool/arxiv/1803.02155>)所提出的设计外，其余所有相对位置编码（包括笔者所提的[RoPE](<https://spaces.ac.cn/archives/8265>)）都只修改了Softmax前的Attention矩阵，那么带有相对位置信息的Attention矩阵依然是一个概率矩阵（即每一行求和等于1）。

另一方面，对于Transformer模型来说，Token之间的交互的唯一来源是Self Attention的$\boldsymbol{A}\boldsymbol{V}$这一步，或者写成$\boldsymbol{o}_i = \sum\limits_j a_{i,j}\boldsymbol{v}_j$。相同的输入意味着每个$\boldsymbol{v}_j$都是相同的，所以
\begin{equation}\boldsymbol{o}_i = \sum_j a_{i,j}\boldsymbol{v}_j = \sum_j a_{i,j}\boldsymbol{v} = \left(\sum_j a_{i,j}\right)\boldsymbol{v} = \boldsymbol{v}\end{equation}
这意味着每个$\boldsymbol{o}_i$也是相同的。换句话说，模型的每个位置自始至终都输出相同的结果，所以模型根本不可能输出各不相同的$[1, 2, \cdots, n-1, n]$。

类似的发现也出现在最近的论文[《Your Transformer May Not be as Powerful as You Expect》](<https://papers.cool/arxiv/2205.13401>)中，作者构建了略有不同的例子来演示相对位置编码Transformer的拟合能力缺陷问题，两者异曲同工、不谋而合了。此外，本文开头说的是“万能拟合”，那解决了这个反例是不是就能做到“万能拟合”了呢？该论文也有相应的理论分析来肯定这一事实，这里就不详述了。

## 初步方案

稍加思考就可以发现，其实问题主要出在Attention矩阵的每一行求和等于1，要解决这个问题，想办法打破这个约束就行了。为此，[《Your Transformer May Not be as Powerful as You Expect》](<https://papers.cool/arxiv/2205.13401>)在其发现之上进一步提出了如下设计
\begin{equation}\boldsymbol{O} = (\boldsymbol{A}\odot \boldsymbol{C})\boldsymbol{V}\quad \text{或者等价地}\quad\boldsymbol{o}_i = \sum_j a_{i,j}c_{i,j}\boldsymbol{v}_j\end{equation}
其中$\boldsymbol{C}$是一个可训练的参数矩阵，$\odot$是逐位相乘（[Hadamard积](<https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29>)）。为了使得整个模型依然只包含相对位置信息（因为本文就是讨论相对位置编码Transfomrer的缺陷），我们要约束$\boldsymbol{C}$为[Toeplitz矩阵](<https://en.wikipedia.org/wiki/Toeplitz_matrix>)，即$c_{i,j}=g(i-j)$。

有了$\boldsymbol{C}$的加入，$\boldsymbol{A}\odot \boldsymbol{C}$作为一个整体，每一行的和显然不一定为1，从而打破了这个限制，因此是可以解决问题的（更多的实验结果请自行看原论文）。但这样一来，引入了新的参数矩阵不说，由于$\boldsymbol{C}$本身是有限大小的，所以它就不能很好地支持变长输入（或者矩阵$\boldsymbol{C}$相应地要做一些截断，即$c_{i,j}=g(\text{clip}(i-j, p_{\min}, p_{\max}))$的形式），总的来说显得不够简洁优雅。

## 去掉分母

再次回到问题所在：Attention矩阵的每一行求和等于1。是什么操作导致了这一现象呢？答案很显然，是Softmax：
\begin{equation}a_{i,j} = \frac{e^{b_{i,j}}}{\sum\limits_j e^{b_{i,j}}}\end{equation}
这里的$\boldsymbol{B}=(b_{i,j})$是Softmax前的矩阵。很明显，就是“除以$\sum\limits_j e^{b_{i,j}}$”这一步导致了$\sum\limits_j a_{i,j}=1$，那么一个很直接的想法就是：

> 如果我不想$\sum\limits_j a_{i,j}=1$，那么干脆别除以$\sum\limits_j e^{b_{i,j}}$就行了？

事实上确实可以！实验结果显示，不除以该分母的Transformer确实能成功地完成前述探针测试。此时就不得不感概一下[GAU](<https://spaces.ac.cn/archives/8934>)的“先见之明”了，它提出的新式Attention直接是$\text{relu}^2$激活然后简单除以$n$来归一化，避免了$\sum\limits_j a_{i,j}=1$，从而增强了模型的理论能力（当然也许作者根本没想那么多，是笔者想象的成分居多）。

## 新归一化

然而，我们在[《听说Attention与Softmax更配哦～》](<https://spaces.ac.cn/archives/9019>)发现像GAU里的不进行概率归一化的Attention设计可能存在外推能力欠佳的问题。也就是说，进行概率归一化导致了前面说的理论缺陷，简单地除以$n$来归一化则外推能力可能欠佳，有没有同时能兼顾两者的方案呢？

让我们再发散一下脑洞。从范数的角度来看，$\sum\limits_j e^{b_{i,j}}$实际上是向量$e^{b_{i,:}}$的$l_1$范数，所以Softmax实际上就是向量的$e^{b_{i,:}}$的$l_1$归一化操作，那么要避免$\sum\limits_j a_{i,j}=1$，又有保留归一化，换成其他的归一化操作是否可以呢？比如$l_2$归一化：
\begin{equation}a_{i,j} = \frac{e^{b_{i,j}}}{\sqrt{\sum\limits_j e^{2b_{i,j}}}}\end{equation}

经过笔者测试，这种$l_2$归一化的Attention，确实能成功完成探针实验。那么，这个改动对我们更关心的NLP预训练场景有没有帮助呢？笔者也做了相应的对比实验，结果是分两部分：

> 1、对于标准的Attention \+ FFN组合，应用$l_2$归一化Attention之前要缩小一下Attention的$\boldsymbol{W}_V,\boldsymbol{W}_O$的初始方差，实验结果则是略差于常规的$l_1$归一化Attention；
> 
> 
> 
> 2、对于全GAU的架构，可以直接应用$l_2$归一化Attention，不需要改动初始化，实验结果则是略优于常规的$l_1$归一化Attention。

两者的差别大概是源于它们本身的初始化方式不同，在标准的Attention \+ FFN组合中，初始Attention矩阵接近一个均匀矩阵（每个数都相同），而在[《门控注意力单元（GAU）还需要Warmup吗？》](<https://spaces.ac.cn/archives/8990>)我们则分析过，GAU的初始Attention矩阵更接近一个单位阵（的若干倍）。

## 峰回路转

再次纵观前文，我们发现是因为“每个$\boldsymbol{v}_j$都是相同的”，所以“$\sum\limits_j a_{i,j}=1$的模型无法完成探针实验”。但如果每个$\boldsymbol{v}_j$不全相同呢？

我们知道，从BERT开始，主流的Transformer模型都是像“&#91;CLS&#93; SENT &#91;SEP&#93;”设计输入的，也就是在输入前后会附加一些标记性的Token，如果我们将这些标记Token当作模型的一部分而不是输入（也就是说输入“&#91;CLS&#93; 0 0 ⋯ 0 0 &#91;SEP&#93;”而不是全0），那么是否有可能完成探针呢？

笔者也对此做了实验，发现对输入补充上标记行Token后，不需要对相对位置编码Transformer的其他部分做修改，确实也能够完成探针实验。这结果就有点啼笑皆非了，原来BERT的作者们也很有“先见之明”啊，所添加的特殊Token &#91;CLS&#93;、&#91;SEP&#93;还有辅助定位的作用，我们分析那么久的理论缺陷，居然就这样被两个特殊Token解决了。这不禁让人想起[《How Much Position Information Do Convolutional Neural Networks Encode?》](<https://papers.cool/arxiv/2001.08248>)所提到的“CNN是通过padding来识别绝对位置的”这一结论，两者有一定的相通之处。

当然，这也不意味着我们前面的思考全无意义。比如对GAU模型来说，Attention换用$l_2$归一化确确实实有加快收敛、轻微提升效果的作用。此外，既然可以接受$l_2$归一化，那么$e^{b_{i,j}}$是不是还可以换成一般的激活函数（比如去掉非负性约束）呢？笔者也简单做了“$\text{swish}(b_{i,j})$ \+ $l_2$归一化”的实验，发现有一定的可行性。从这个角度来看，$l_2$归一化下的Attention实际上有更多的拓展空间。

## 曲终人散

本文分析了相对位置编码Transformer的一个隐含缺陷，并探讨了相应的对策，从中引申出关于Attention矩阵的非负性、归一化方式的思考。
