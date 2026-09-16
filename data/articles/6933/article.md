# 从语言模型到Seq2Seq：Transformer如戏，全靠Mask

> 作者：苏剑林 · 科学空间 · 2019-09-18
>
> 原文：<https://spaces.ac.cn/archives/6933>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

相信近一年来（尤其是近半年来），大家都能很频繁地看到各种Transformer相关工作（比如Bert、GPT、XLNet等等）的报导，连同各种基础评测任务的评测指标不断被刷新。同时，也有很多相关的博客、专栏等对这些模型做科普和解读。

俗话说，“外行看热闹，内行看门道”，我们不仅要在“是什么”这个层面去理解这些工作，我们还需要思考“为什么”。这个“为什么”不仅仅是“为什么要这样做”，还包括“为什么可以这样做”。比如，在谈到XLNet的乱序语言模型时，我们或许已经从诸多介绍中明白了乱序语言模型的好处，那不妨更进一步思考一下：

> 为什么Transformer可以实现乱序语言模型？是怎么实现的？RNN可以实现吗？

本文从对Attention矩阵进行Mask的角度，来分析为什么众多Transformer模型可以玩得如此“出彩”的基本原因，正如标题所述“Transformer如戏，全靠Mask”，这是各种花式Transformer模型的重要“门道”之一。

读完本文，你或许可以了解到：

> 1、Attention矩阵的Mask方式与各种预训练方案的关系；
> 
> 
> 
> 2、直接利用预训练的Bert模型来做Seq2Seq任务。

## 背景

自[《Attention is All You Need》](<https://spaces.ac.cn/archives/4765>)以后，基于纯Attention的Transformer类模型逐渐变得流行起来，而Bert的出现则将这股潮流推向了一个新的高度。而后，各种基于大规模预训练的Transformer模型的工作不断出现，有基于现成的模型做应用的，有试图更好地去解释和可视化这些模型的，还有改进架构、改进预训练方式等以得到更好结果的。总的来说，这些以预训练为基础的工作层出不穷，有种琳琅满目的感觉。甚至一定程度上来说，如果你还没有微调过Bert，那已经算是落后于主流的NLP技术了。

### 花式预训练

众所周知，传统的模型预训练手段就是语言模型，比如[ELMo](<https://papers.cool/arxiv/1802.05365>)模型就是以BiLSTM为基础架构、用两个方向的语言模型分别预训练两个方向的LSTM的，后面的OpenAI的GPT、[GPT\-2](<https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>)也是坚定不移地坚持着用祖传的（标准的、单向的）语言模型来预训练。

然而，还有更多花样的预训练玩法。比如[Bert](<https://papers.cool/arxiv/1810.04805>)就用了称之为“掩码语言模型（Masked Language Model）”的方式来预训练，不过这只是普通语言模型的一种变体；还有[XLNet](<https://papers.cool/arxiv/1906.08237>)则提出了更彻底的“Permutation Language Modeling”，我们可以称之为“乱序语言模型”；还有[UNILM](<https://papers.cool/arxiv/1905.03197>)模型，直接用单个Bert的架构做Seq2Seq，你可以将它作为一种预训练手段，又或者干脆就用它来做Seq2Seq任务\.\.\.

如此花样百出，让我们不禁疑问：<u>为什么刚好在Transformer流行的时代，才出现这种各种大型预训练模型“百花齐放，百家争鸣”的现象？</u>

### Transformer专属

事实上，除了单向语言模型及其简单变体掩码语言模型之外，UNILM的Seq2Seq预训练、XLNet的乱序语言模型预训练，基本可以说是专为Transformer架构定制的。说白了，如果是RNN架构，根本就不能用乱序语言模型的方式来预训练，至于Seq2Seq的预训练方式，则必须同时引入两个模型（encoder和decoder），而无法像Transformer架构一样，可以一个模型搞定。

这其中的奥妙主要在Attention矩阵之上。Attention实际上相当于将输入两两地算相似度，这构成了一个$n^2$大小的相似度矩阵（即Attention矩阵，$n$是句子长度，本文的Attention均指Self Attention），这意味着它的空间占用量是$\mathcal{O}(n^2)$量级，相比之下，RNN模型、CNN模型只不过是$\mathcal{O}(n)$，所以实际上Attention通常更耗显存。然而，有弊也有利，更大的空间占用也意味着拥有了更多的可能性，我们可以通过往这个$\mathcal{O}(n^2)$级别的Attention矩阵加入各种先验约束，使得它可以做更灵活的任务。说白了，也就只有纯Attention的模型，才有那么大的“容量”去承载那么多的“花样”。

而加入先验约束的方式，就是对Attention矩阵进行不同形式的Mask，这便是本文要关注的焦点。

## 分析

在[《〈Attention is All You Need〉浅读（简介\+代码）》](<https://spaces.ac.cn/archives/4765>)一文中笔者已经对Attention做了基本介绍，这里仅做简单回顾。Attention的数学形式为：
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}\end{equation}

这里的$\boldsymbol{Q}\in \mathbb{R}^{l_q\times d_q},\boldsymbol{K}\in\mathbb{R}^{l_k\times d_q},\boldsymbol{V}\in\mathbb{R}^{l_k\times d_v}$，分别代表query、key、value的向量序列，其中我们可以认为key和value是一一对应的，而$\boldsymbol{Q}\boldsymbol{K}^{\top}$则是将query、key的向量两两做内积，然后用$softmax$归一化，就得到一个$l_q\times l_k$的Attention矩阵，它描述的就是query和key之间任意两个元素的关联强度，后面我们要讲的故事，都是在这个Attention矩阵上下功夫。最后再与$\boldsymbol{V}$相乘，相当于按照这个关联强度将$\boldsymbol{V}$的各个向量加权求和，最终输出一个$l_q\times d_v$的向量序列。

目前最常用的Attention方式当数Self Attention，即$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$都是同一个向量序列经过线性变换而来的，而Transformer则是Self Attention跟Position\-Wise全连接层（相当于kernel size为1的一维卷积）的组合。所以，Transformer就是基于Attention的向量序列到向量序列的变换。

在本节中，我们将会比较详细地分析Attention矩阵的Mask方式，这分别对应单向语言模型、乱序语言模型、Seq2Seq的实现原理。

### 单向语言模型

语言模型可以说是一个无条件的文本生成模型，如果读者还不了解文本生成模型，可以自行查阅相关资料并配合[《玩转Keras之seq2seq自动生成标题》](<https://spaces.ac.cn/archives/5861>)一文来理解。单向语言模型相当于把训练语料通过下述条件概率分布的方式“记住”了：
\begin{equation}p(x_1,x_2,x_3,\dots,x_n)=p(x_1) p(x_2|x_1) p(x_3|x_1,x_2) \dots p(x_n|x_1,\dots,x_{n-1})\end{equation}

我们一般说的“语言模型”，就是指单向的（更狭义的只是指正向的）语言模型。语言模型的关键点是要防止看到“未来信息”。如上式，预测$x_1$的时候，是没有任何外部输入的；而预测$x_2$的时候，只能输入$x_1$，预测$x_3$的时候，只能输入$x_1,x_2$；依此类推。

[![单向语言模型图示。每预测一个token，只依赖于前面的token。](<https://spaces.ac.cn/usr/uploads/2019/09/2487583134.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/2487583134.png>)

单向语言模型图示。每预测一个token，只依赖于前面的token。

RNN模型是天然适合做语言模型的，因为它本身就是递归的运算；如果用CNN来做的话，则需要对卷积核进行Mask，即需要将卷积核对应右边的部分置零。如果是Transformer呢？那需要一个下三角矩阵形式的Attention矩阵：

[![单向（正向）语言模型的Mask方式](<https://spaces.ac.cn/usr/uploads/2019/09/1936593842.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1936593842.png>)

单向（正向）语言模型的Mask方式

如图所示，Attention矩阵的每一行事实上代表着输出，而每一列代表着输入，而Attention矩阵就表示输出和输入的关联。假定白色方格都代表0，那么第1行表示“北”只能跟起始标记\<s\>相关了，而第2行就表示“京”只能跟起始标记\<s\>和“北”相关了，依此类推。所以，只需要在Transformer的Attention矩阵中引入下三角形形式的Mask，并将输入输出错开一位训练，就可以实现单向语言模型了。（至于Mask的实现方式，可以参考[《“让Keras更酷一些！”：层中层与mask》的Mask一节](<https://spaces.ac.cn/archives/6810#Mask>)。）

### 乱序语言模型

乱序语言模型是XLNet提出来的概念，它主要用于XLNet的预训练上。说到XLNet，我觉得它的乱序语言模型这种预训练方式是很有意思的，但是我并不喜欢它将基本架构换成了Transformer\-XL。我觉得谁有资源可以试试“Bert\+乱序语言语言模型预训练”的组合，或许会有意外的发现。

乱序语言模型跟语言模型一样，都是做条件概率分解，但是乱序语言模型的分解顺序是随机的：
\begin{equation}\begin{aligned}p(x_1,x_2,x_3,\dots,x_n)=&p(x_1) p(x_2|x_1) p(x_3|x_1,x_2) \dots p(x_n|x_1,x_2,\dots,x_{n-1})\\
=&p(x_3) p(x_1|x_3) p(x_2|x_3,x_1) \dots p(x_n|x_3,x_1,\dots,x_{n-1})\\
=&\dots\\
=&p(x_{n-1})p(x_1|x_{n-1})p(x_n|x_{n-1}, x_1)\dots p(x_2|x_{n-1}, x_1,\dots,x_3)\end{aligned}\end{equation}

总之，$x_1,x_2,\dots,x_n$任意一种“出场顺序”都有可能。原则上来说，每一种顺序都对应着一个模型，所以原则上就有$n!$个语言模型。而基于Transformer的模型，则可以将这所有顺序都做到一个模型中去！

那怎么做到这一点呢？还是以“北京欢迎你”的生成为例，假设随机的一种生成顺序为“\<s\> → 迎 → 京 → 你 → 欢 → 北 → \<e\>”，那么我们只需要用下图中第二个子图的方式去Mask掉Attention矩阵，就可以达到目的了：

[![正向语言模型的Mask](<https://spaces.ac.cn/usr/uploads/2019/09/1319012901.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1319012901.png>)

正向语言模型的Mask

[![乱序语言模型的Mask](<https://spaces.ac.cn/usr/uploads/2019/09/3858909307.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/3858909307.png>)

乱序语言模型的Mask

[![倒序语言模型的Mask](<https://spaces.ac.cn/usr/uploads/2019/09/1916971758.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1916971758.png>)

倒序语言模型的Mask

跟前面的单向语言模型类似，第4行只有一个蓝色格，表示“迎”只能跟起始标记\<s\>相关，而第2行有两个蓝色格，表示“京”只能跟起始标记\<s\>和“迎”相关，依此类推。直观来看，这就像是<u>把单向语言模型的下三角形式的Mask“打乱”了</u>。

也就是说，实现某种特定顺序的语言模型，就相当于将原来的下三角形式的Mask以某种方式打乱。正因为Attention提供了这样的一个$n\times n$的Attention矩阵，我们才有足够多的自由度去以不同的方式去Mask这个矩阵，从而实现多样化的效果。

说到这里，读者可能会有一个实现上的疑问：打乱后的Mask似乎没看出什么规律呀，难道每次都要随机生成一个这样的似乎没有什么明显概率的Mask矩阵？事实上有一种更简单的、数学上等效的训练方案。这个训练方案源于纯Attention的模型本质上是一个无序的模型，它里边的词序实际上是通过Position Embedding加上去的。也就是说，我们输入的不仅只有token本身，还包括token所在的位置id；再换言之，你觉得你是输入了序列“&#91;北, 京, 欢, 迎, 你&#93;”，实际上你输入的是集合“\{(北, 1), (京, 2), (欢, 3), (迎, 4), (你, 5)\}”。

[![重新排序，使得正向语言模型就可以实现乱序语言模型](<https://spaces.ac.cn/usr/uploads/2019/09/2887156520.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/2887156520.png>)

重新排序，使得正向语言模型就可以实现乱序语言模型

既然只是一个集合，跟顺序无关，那么我们完全可以换一种顺序输入，比如刚才的“\<s\> → 迎 → 京 → 你 → 欢 → 北 → \<e\>”，我们可以按“(迎, 4), (京, 2), (你, 5), (欢, 3), (北, 1)”的顺序输入，也就是说将token打乱为“迎,京,你,欢,北”输入到Transformer中，但是第1个token的position就不是1了，而是4；依此类推。这样换过来之后，Mask矩阵可以恢复为下三角矩阵，所以只需要在输入层面打乱即可，这样操作起来就更简单了。

### Seq2Seq

现在到我们的“重头戏”了：<strong>将Bert等Transformer架构跟Seq2Seq结合起来</strong>。为什么说重头戏呢？因为原则上来说，任何NLP问题都可以转化为Seq2Seq来做，它是一个真正意义上的万能模型。所以如果能够做到Seq2Seq，理论上就可以实现任意任务了。

将Bert与Seq2Seq结合的比较知名的工作有两个：[MASS](<https://papers.cool/arxiv/1905.02450>)和[UNILM](<https://papers.cool/arxiv/1905.03197>)，两者都是微软的工作，两者还都在同一个月发的～其中MASS还是普通的Seq2Seq架构，分别用Bert类似的Transformer模型来做encoder和decoder，它的主要贡献就是提供了一种Seq2Seq思想的预训练方案；真正有意思的是UNILM，它提供了一种很优雅的方式，能够<u>让我们直接用单个Bert模型就可以做Seq2Seq任务，而不用区分encoder和decoder</u>。而实现这一点几乎不费吹灰之力——只需要一个特别的Mask。

（插曲：事实的顺序是笔者前两周自己独立地想到了用单个Bert模型做Seq2Seq的思路，然后去找资料发现这个思路已经被做了，正是UNILM。）

UNILM直接将Seq2Seq当成句子补全来做。假如输入是“你想吃啥”，目标句子是“白切鸡”，那UNILM将这两个句子拼成一个：&#91;CLS&#93; 你 想 吃 啥 &#91;SEP&#93; 白 切 鸡 &#91;SEP&#93;。经过这样转化之后，最简单的方案就是训练一个语言模型，然后输入“&#91;CLS&#93; 你 想 吃 啥 &#91;SEP&#93;”来逐字预测“白 切 鸡”，直到出现“&#91;SEP&#93;”为止，即如下面的左图：

[![用单向语言模型的方式做Seq2Seq](<https://spaces.ac.cn/usr/uploads/2019/09/1798189033.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1798189033.png>)

用单向语言模型的方式做Seq2Seq

[![设计更适合的Mask做Seq2Seq](<https://spaces.ac.cn/usr/uploads/2019/09/1625339461.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1625339461.png>)

设计更适合的Mask做Seq2Seq

不过左图只是最朴素的方案，它把“你想吃啥”也加入了预测范围了（导致它这部分的Attention是单向的，即对应部分的Mask矩阵是下三角），事实上这是不必要的，属于额外的约束。真正要预测的只是“白切鸡”这部分，所以我们可以把“你想吃啥”这部分的Mask去掉，得到上面的右图的Mask。

这样一来，输入部分的Attention是双向的，输出部分的Attention是单向，满足Seq2Seq的要求，而且没有额外约束。这便是UNILM里边提供的用单个Bert模型就可以完成Seq2Seq任务的思路，只要添加上述形状的Mask，而不需要修改模型架构，并且还可以直接沿用Bert的Masked Language Model预训练权重，收敛更快。这符合“一Bert在手，天下我有”的万用模型的初衷，个人认为这是非常优雅的方案。

[![UNILM做Seq2Seq模型图示。输入部分内部可做双向Attention，输出部分只做单向Attention。](<https://spaces.ac.cn/usr/uploads/2019/09/1879768703.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1879768703.png>)

UNILM做Seq2Seq模型图示。输入部分内部可做双向Attention，输出部分只做单向Attention。

## 实验

事实上，上述的这些Mask方案，基本上都已经被集成在笔者写的[bert4keras](<https://spaces.ac.cn/archives/6915>)，读者可以直接用bert4keras加载bert的预训练权重，并且调用上述Mask方案来做相应的任务。下面，我们给出一个利用UNILM的思路做一个快速收敛的Seq2Seq模型的例子。

### 代码开源

这次代码的测试任务依然是之前的标题生成，代码调整自[《玩转Keras之seq2seq自动生成标题》](<https://spaces.ac.cn/archives/5861>)里边的代码，并且得益于[bert4keras](<https://github.com/bojone/bert4keras>)的封装，模型部分的代码实现非常简单清爽。这一次直接使用了[THUCNews](<http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews>)的原始数据集，读者可以自行下载数据集和源码测试复现。

详细请看：[<strong>task\_seq2seq\_autotitle\.py</strong>](<https://github.com/bojone/bert4keras/blob/master/examples/task_seq2seq_autotitle.py>)

这个效果能有多好呢？经过实验，在标题生成的任务上，从第一个epoch（1000个iteration）开始，就已经能生成基本可读的标题了。相应地，以前用LSTM做的时候，大概需要多几十倍的iteration才有同样的效果。

[![第一个epoch（1000步）就可以得到基本可读的生成结果](<https://spaces.ac.cn/usr/uploads/2019/09/1361605953.png>)](<https://spaces.ac.cn/usr/uploads/2019/09/1361605953.png>)

第一个epoch（1000步）就可以得到基本可读的生成结果

### 简单说明

下面对代码的关键部分做简要说明。

首先，输入格式还是以<code>token_id</code>和<code>segment_id</code>输入，比如：



<pre><code class="language-python">tokens = ['[ClS]', u'你', u'想', u'吃', u'啥', '[SEP]', u'白', u'切', u'鸡', '[SEP]']
token_ids = [token_dict[t] for t in tokens]
segment_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]</code></pre>



<code>segment_ids</code>用来区分输入句子和目标句子，0对应的为输入句子，1对应的为目标句子，只需要自带的<code>tokenizer.encode</code>就可以生成这种<code>token_id</code>和<code>segment_id</code>了。

至于搭建模型，就只有寥寥几行：



<pre><code class="language-python">model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens
)

model.summary()

y_in = model.input[0][:, 1:] # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1] # 预测tokens，预测与目标错开一位

# 交叉熵作为loss，并mask掉输入部分的预测
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)</code></pre>



注意<code>build_transformer_model</code>中只要设置<code>application='unilm'</code>，就会自动加载Bert的MLM部分，并且传入对应的Mask，剩下就只需要把loss写好就行了。另外还有一个<code>keep_tokens</code>，这个是用来精简Embedding层用的，对于中文Bert来说，总的tokens大概有2万个，这意味着最后预测生成的token时是一个2万分类问题。但事实上有接近一半的tokens都不会分出来（理论上都不会），因此这2万分类浪费了一些计算量。于是这里提供了一个选项，我们可以自行维护一个token表，然后传入对应的id，只保留这部分token，这样就可以降低计算量了（精简后一般只是原来的一半，甚至更少）。

剩下的就是通过beam search来解码等步骤了，这与一般的Seq2Seq无异，不再赘述，大家看[《玩转Keras之seq2seq自动生成标题》](<https://spaces.ac.cn/archives/5861>)和代码即可。

## 总结

<strong>本文相对系统地总结了Transformer中Attention矩阵的Mask技巧，并且给出了用UNILM方案来做Seq2Seq的实现。对于同语言的Seq2Seq的文本生成任务来说，采用UNILM的思路加载Bert的MLM预训练权重，能够有效、快速地实现并提升生成效果，值得一试。</strong>
