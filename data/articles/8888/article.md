# GPLinker：基于GlobalPointer的实体关系联合抽取

> 作者：苏剑林 · 科学空间 · 2022-01-30
>
> 原文：<https://spaces.ac.cn/archives/8888>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在将近三年前的百度“2019语言与智能技术竞赛”（下称LIC2019）中，笔者提出了一个新的关系抽取模型（参考[《基于DGCNN和概率图的轻量级信息抽取模型》](<https://spaces.ac.cn/archives/6671>)），后被进一步发表和命名为“[CasRel](<https://papers.cool/arxiv/1909.03227>)”，算是当时关系抽取的SOTA。然而，CasRel提出时笔者其实也是首次接触该领域，所以现在看来CasRel仍有诸多不完善之处，笔者后面也有想过要进一步完善它，但也没想到特别好的设计。

后来，笔者提出了[GlobalPointer](<https://spaces.ac.cn/archives/8373>)以及近日的[Efficient GlobalPointer](<https://spaces.ac.cn/archives/8877>)，感觉有足够的“材料”来构建新的关系抽取模型了。于是笔者从概率图思想出发，参考了CasRel之后的一些SOTA设计，最终得到了一版类似TPLinker的模型。

## 基础思路

关系抽取乍看之下是三元组$(s,p,o)$（即subject, predicate, object)的抽取，但落到具体实现上，它实际是“五元组”$(s_h,s_t,p,o_h,o_t)$的抽取，其中$s_h,s_t$分别是$s$的首、尾位置，而$o_h,o_t$则分别是$o$的首、尾位置。

从概率图的角度来看，我们可以这样构建模型：

> 1、设计一个五元组的打分函数$S(s_h,s_t,p,o_h,o_t)$；
> 
> 
> 
> 2、训练时让标注的五元组$S(s_h,s_t,p,o_h,o_t) > 0$，其余五元组则$S(s_h,s_t,p,o_h,o_t) < 0$；
> 
> 
> 
> 3、预测时枚举所有可能的五元组，输出$S(s_h,s_t,p,o_h,o_t) > 0$的部分。

然而，直接枚举所有的五元组数目太多，假设句子长度为$l$，$p$的总数为$n$，即便加上$s_h\leq s_t$和$o_h\leq o_t$的约束，所有五元组的数目也有
\begin{equation}n\times \frac{l(l+1)}{2}\times \frac{l(l+1)}{2}=\frac{1}{4}nl^2(l+1)^2\end{equation}
这是长度的四次方级别的计算量，实际情况下难以实现，所以必须做一些简化。

## 简化分解

以我们目前的算力来看，一般最多也就能接受长度平方级别的计算量，所以我们每次顶多能识别“一对”首或尾，为此，我们可以用以下的分解：
\begin{equation}S(s_h,s_t,p,o_h,o_t) = S(s_h,s_t) + S(o_h,o_t) + S(s_h,o_h| p) + S(s_t, o_t| p)\label{eq:factor}\end{equation}
要注意的是，该等式属于模型假设，是基于我们对任务的理解以及算力的限制所设计出来的，而不是理论推导出来的。其中，每一项都具直观的意义，比如$S(s_h,s_t)$、$S(o_h,o_t)$分别是subject、object的首尾打分，通过$S(s_h,s_t) > 0$和$S(o_h,o_t) > 0$来析出所有的subject和object。至于后两项，则是predicate的匹配，$S(s_h,o_h|p)$这一项代表以subject和object的首特征作为它们自身的表征来进行一次匹配，如果我们能确保subject内和object内是没有嵌套实体的，那么理论上$S(s_h,o_h|p) > 0$就足够析出所有的predicate了，但考虑到存在嵌套实体的可能，所以我们还要对实体的尾再进行一次匹配，即$S(s_t, o_t|p)$这一项。

此时，训练和预测过程变为：

> 1、训练时让标注的五元组$S(s_h,s_t) > 0$、$S(o_h,o_t) > 0$、$S(s_h,o_h| p) > 0$、$S(s_t, o_t| p) > 0$，其余五元组则$S(s_h,s_t) < 0$、$S(o_h,o_t) < 0$、$S(s_h,o_h| p) < 0$、$S(s_t, o_t| p) < 0$；
> 
> 
> 
> 2、预测时枚举所有可能的五元组，逐次输出$S(s_h,s_t) > 0$、$S(o_h,o_t) > 0$、$S(s_h,o_h| p) > 0$、$S(s_t, o_t| p) > 0$的部分，然后取它们的交集作为最终的输出（即同时满足4个条件）。

在实现上，由于$S(s_h,s_t)$、$S(o_h,o_t)$是用来识别subject、object对应的实体的，它相当于有两种实体类型的NER任务，所以我们可以用一个GlobalPointer来完成；至于$S(s_h,o_h| p)$，它是用来识别predicate为$p$的$(s_h,o_h)$对，跟NER不同的是，它这里不需要$s_h \leq o_h$的约束，这里我们同样用GlobalPointer来完成，但为了识别出$s_h > o_h$的部分，要去掉GlobalPointer默认的下三角mask；最后$S(s_t, o_t|p)$跟$S(s_h,o_h| p)$同理，不再赘述。

这里再回顾一遍：我们知道，作为NER模块，GlobalPointer可以统一识别嵌套和非嵌套的实体，而这是它基于token\-pair的识别来做到的。所以，我们应该进一步将GlobalPointer理解为一个token\-pair的识别模型，而不是局限在NER范围内理解它。认识到这一点之后，我们就能明白上述$S(s_h,s_t)$、$S(o_h,o_t)$、$S(s_h,o_h| p)$、$S(s_t, o_t|p)$其实都可以用GlobalPointer来实现了，而要不要加下三角mask，则自行根据具体任务背景设置就好。

## 损失函数

现在我们已经把打分函数都设计好了，那么为了训练模型，就差损失函数了。这里继续使用GlobalPointer默认使用的、在[《将“Softmax\+交叉熵”推广到多标签分类问题》](<https://spaces.ac.cn/archives/7359>)中提出的多标签交叉熵，它的一般形式为：
\begin{equation}\log \left(1 + \sum\limits_{i\in \mathcal{P}} e^{-S_i}\right) + \log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right)\label{eq:loss-1}\end{equation}
其中$\mathcal{P},\mathcal{N}$分别是正、负类别的集合。在之前的文章中，我们都是用“multi hot”向量来标记正、负类别的，即如果总类别数为$K$，那么我们用一个$K$维向量来表示，其中正类的位置为1，负类的位置为0。然而，在$S(s_h,o_h| p)$和$S(s_t, o_t|p)$的场景，我们各需要一个$n\times l\times l$的矩阵来标记，两个加在一起并算上batch\_size总维度就是$2bnl^2$，以$b=64,n=50,l=128$为例，那么$2bnl^2\approx 1\text{亿}$。这也就意味着，如果我们还坚持用“multi hot”的形式表示标签的话，每一步训练我们都要创建一个1亿参数量的矩阵，然后还要传到GPU中，这样不管是创建还是传输成本都很大。

所以，为了提高训练速度，我们需要实现一个“稀疏版”的多标签交叉熵，即每次都只传输正类所对应的的下标就好，由于正类远远少于负类，这样标签矩阵的尺寸就大大减少了。而“稀疏版”多标签交叉熵，意味着我们要在只知道$\mathcal{P}$和$\mathcal{A}=\mathcal{P}\cup\mathcal{N}$的前提下去实现式$\eqref{eq:loss-1}$。为此，我们使用的实现方式是：
\begin{equation}\begin{aligned}
&\,\log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right) = \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i} - \sum\limits_{i\in \mathcal{P}} e^{S_i}\right) \\
=&\, \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right) + \log \left(1 - \left(\sum\limits_{i\in \mathcal{P}} e^{S_i}\right)\Bigg/\left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right)\right)
\end{aligned}\end{equation}
如果即$a = \log \left(1 + \sum\limits_{i\in \mathcal{A}} e^{S_i}\right),b=\log \left(\sum\limits_{i\in \mathcal{P}} e^{S_i}\right)$，那么可以写为
\begin{equation}\log \left(1 + \sum\limits_{i\in \mathcal{N}} e^{S_i}\right) = a + \log\left(1 - e^{b - a}\right)\end{equation}
这样就通过$\mathcal{P}$和$\mathcal{A}$算出了负类对应的损失，而正类部分的损失保持不变就好。

最后，一般情况下的多标签分类任务正类个数是不定的，这时候我们可以将类的下标从1开始，将0作为填充标签使得每个样本的标签矩阵大小一致，最后在loss的实现上对0类进行mask处理即可。相应的实现已经内置在bert4keras中，详情可以参考“[sparse\_multilabel\_categorical\_crossentropy](<https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272>)”。

## 实验结果

为了方便称呼，我们暂且将上述模型称为GPLinker（GlobalPointer\-based Linking），一个基于bert4keras的参考实现如下：

> <strong>脚本链接：[task\_relation\_extraction\_gplinker\.py](<https://github.com/bojone/bert4keras/tree/master/examples/task_relation_extraction_gplinker.py>)</strong>

在LIC2019上的实验结果如下（CasRel的代码为[task\_relation\_extraction\.py](<https://github.com/bojone/bert4keras/tree/master/examples/task_relation_extraction.py>)）：
\begin{array}{c|c}
\hline
\text{模型} & \text{F1} \\
\hline
\text{CasRel} & 0.8220 \\
\text{GPLinker (Standard)} & 0.8272\\
\text{GPLinker (Efficient)} & 0.8268\\
\hline
\end{array}

预训练模型是BERT base，Standard和Efficient的区别是分别使用了[标准版GlobalPointer](<https://spaces.ac.cn/archives/8373>)和[Efficient GlobalPointer](<https://spaces.ac.cn/archives/8877>)。该实验结果说明了两件事情，一是GPLinker确实比CasRel更加有效，二是Efficient GlobalPointer的设计确实能在更少参数的情况下媲美标准版GlobalPointer的效果。要知道在LIC2019这个任务下，如果使用标准版GlobalPointer，那么GPLinker的参数量接近1千万，而用Efficient GlobalPointer的话只有30万左右。

此外，在3090上，相比于“multi hot”版的多标签交叉熵，使用稀疏版多标签交叉熵的模型在训练速度上能提高1\.5倍而不会损失精度，跟CasRel相比，使用了稀疏版多标签交叉熵的GPLinker在训练速度上只慢15%，但是解码速度快将近一倍，算得上又快又好了。

## 相关工作

而对于了解这两年关系抽取SOTA模型进展的同学来说，理解上述模型后，会发现它跟[TPLinker](<https://papers.cool/arxiv/2010.13415>)是非常相似的。确实如此，模型在设计之初确实充分借鉴了TPLinker，最后的结果也同样跟TPLinker很相似。

大体上来说，TPLinker与GPLinker的区别如下：

> 1、TPLinker的token\-pair分类特征是首尾特征后拼接做Dense变换得到的，其思想来源于Additive Attention；GPLinker则是用GlobalPointer实现，其思想来源于Scaled Dot\-Product Attention。平均来说，后者拥有更少的显存占用和更快的计算速度。
> 
> 
> 
> 2、GPLinker分开识别subject和object的实体，而TPLinker将subject和object混合起来统一识别。笔者也在GPLinker中尝试了混合识别，发现最终效果跟分开识别没有明显区别。
> 
> 
> 
> 3、在$S(s_h,o_h|p)$和$S(s_t,o_t|p)$，TPLinker将其转化为了$l(l+1)/2$个3分类问题，这会有明显的类别不平衡问题；而GPLinker用到了笔者提出的多标签交叉熵，则不会存在不平衡问题，更容易训练。事实上后来TPLinker也意识到了这个问题，并提出了[TPLinker\-plus](<https://github.com/131250208/TPlinker-joint-extraction/tree/master/tplinker_plus>)，其中也用到了该多标签交叉熵。

当然，在笔者看来，本文的最主要贡献，并不是提出GPLinker的这些改动，而是对关系联合抽取模型进行一次“自上而下”的理解：从开始的五元组打分$S(s_h,s_t,p,o_h,o_t)$出发，分析其难处，然后简化分解式$\eqref{eq:factor}$来“逐个击破”。希望这个自上而下的理解过程，能给读者在为更复杂的任务设计模型时提供一定的思路。

## 文章小结

本文分享了一个基于GlobalPointer的实体关系联合抽取模型——“GPLinker”，并提供了一个“自上而下”的推导理解给大家参考。
