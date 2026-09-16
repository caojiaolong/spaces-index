# “噪声对比估计”杂谈：曲径通幽之妙

> 作者：苏剑林 · 科学空间 · 2018-06-13
>
> 原文：<https://spaces.ac.cn/archives/5617>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

说到噪声对比估计，或者“负采样”，大家可能立马就想到了Word2Vec。事实上，它的含义远不止于此，噪音对比估计（NCE, Noise Contrastive Estimation）是一个迂回但却异常精美的技巧，它使得我们在没法直接完成归一化因子（也叫配分函数）的计算时，就能够去估算出概率分布的参数。本文就让我们来欣赏一下NCE的曲径通幽般的美妙。

注：由于出发点不同，本文所介绍的“噪声对比估计”实际上更偏向于所谓的“负采样”技巧，但两者本质上是一样的，在此不作区分。

## 问题起源

问题的根源是难分难舍的指数概率分布～

### 指数族分布

在很多问题中都会出现指数族分布，即对于某个变量$\boldsymbol{x}$的概率$p(\boldsymbol{x})$，我们将其写成
$$p(\boldsymbol{x}) = \frac{e^{G(\boldsymbol{x})}}{Z}\tag{1}$$
其中$G(\boldsymbol{x})$是$\boldsymbol{x}$的某个“能量”函数，而$Z=\sum_{\boldsymbol{x}} e^{G(\boldsymbol{x})}$则是归一化常数，也叫配分函数。这种分布也称为“玻尔兹曼分布”。

在机器学习中，指数族分布的主要来源有两个。<u>第一个来源是softmax</u>：我们做分类预测时，通常最后都会将全连接层的结果用softmax激活，这就是一个离散的、有限个点的玻尔兹曼分布了；<u>第二个则是来源于最大熵原理</u>：当我们引入某个特征并且已经能估算出特征的期望时，最大熵模型告诉我们其分布应该是特征的指数形式。（参考[《“熵”不起：从熵、最大熵原理到最大熵模型（二）》](<https://spaces.ac.cn/archives/3552>)。）

### 难算的配分函数

总的来说，指数族分布是非常实用的一类分布，不论是机器学习、数学还是物理领域，都能够碰见它。然而，它却有一个比较大的问题：不容易算，准确来说是配分函数不容易算。

具体来说，不好算的原因可能有两个。一个是计算量太大，比如语言模型（包括Word2Vec）的场景，因为要通过上下文来预测当前词的分布情况，这就需要对几十万甚至几百万项（取决于词表大小）进行求和来算归一化因子，这种情况下不是不能算，而是计算量大到难以承受了；另一种情况是根本算不出来～比如假设$p(x)=\frac{e^{-ax^2-bx^4}}{Z}$那么就有
$$Z = \int e^{-ax^2-bx^4} dx\tag{2}$$
这积分根本就没法简单地算出来呀，更不用说更加复杂的函数了。现在我们也许能从这个角度感受到为什么高斯分布那么常用了，因为，因为，因为，换个分布就没法算下去了\.\.\.

在机器学习中，如果只是分类、预测，那么归一化因子算不算出来都无所谓，因为我们只要相对比较取出最大的那个。但是在预测之前，我们还面临着训练的问题，也就是参数估计，具体来说，$G(\boldsymbol{x})$其实是含有一些未知参数$\boldsymbol{\theta}$的，准确来说要写成$G(\boldsymbol{x};\boldsymbol{\theta})$，那么概率分布就是
$$p(\boldsymbol{x})=\frac{e^{G(\boldsymbol{x};\boldsymbol{\theta})}}{Z(\boldsymbol{\theta})}\tag{3}$$
我们要从$\boldsymbol{x}$的样本中推算出$\boldsymbol{\theta}$来，通常我们会用最大似然，但是不算出$Z(\boldsymbol{\theta})$来我们就没法算似然函数，也就没法做下去了。

## NCE登场

非常幸运的是，NCE诞生了，它成功地绕开了这个困难。对于配分函数算不出来的情形，它提供了一种算下去的可能性；对于配分函数计算量太大的情形，它还提供了一种降低计算量的方案。

### 变成二分类问题

NCE的思想很简单，它希望我们将真实的样本和一批“噪声样本”进行对比，从中发现真实样本的规律出来。

具体来说，能量还是原来的能量$G(\boldsymbol{x};\boldsymbol{\theta})$，但这时候我们不直接算概率$p(\boldsymbol{x})$了，因为归一化因子很难算。我们去算
$$p(1|\boldsymbol{x})=\sigma\Big(G(\boldsymbol{x};\boldsymbol{\theta})-\gamma\Big)=\frac{1}{1+e^{-G(\boldsymbol{x};\boldsymbol{\theta})+\gamma}}\tag{4}$$
这里的$\boldsymbol{\theta}$还是原来的待优化参数，而$\gamma$则是新引入的要优化的参数。

然后，NCE的损失函数变为
$$\mathop{\text{argmin}}_{\boldsymbol{\theta},\gamma} - \mathbb{E}_{\boldsymbol{x}\sim \tilde{p}(\boldsymbol{x})}\log  p(1|\boldsymbol{x})- \mathbb{E}_{\boldsymbol{x}\sim U(\boldsymbol{x})}\log  p(0|\boldsymbol{x})\tag{5}$$
其中$\tilde{p}(\boldsymbol{x})$是真实样本，$U(\boldsymbol{x})$是某个“均匀”分布或者其他的、确定的、方便采样的分布。

说白了，<strong>NCE的做法就是将它转化为二分类问题，将真实样本判为1，从另一个分布采样的样本判为0</strong>。

### 等价于原来分布

现在的问题是，从$(5)$式估算出来的$\boldsymbol{\theta}$，跟直接从$(3)$式的最大似然估计（理论上是可行的）出来的结果是不是一样的。

<strong>答案是基本一样的。</strong>我们将$(5)$式中的loss改写为
$$-\int \tilde{p}(\boldsymbol{x})\log  p(1|\boldsymbol{x}) d\boldsymbol{x}- \int U(\boldsymbol{x})\log  p(0|\boldsymbol{x})d\boldsymbol{x}\tag{6}$$
因为$\tilde{p}(\boldsymbol{x})$和$U(\boldsymbol{x})$都跟参数$\boldsymbol{\theta},\gamma$没关，因此将loss改为下面的形式，不会影响优化结果
$$\begin{aligned}&\int \big(\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})\big) \left(\tilde{p}(1|\boldsymbol{x}) \log  \frac{\tilde{p}(1|\boldsymbol{x})}{p(1|\boldsymbol{x})} + \tilde{p}(0|\boldsymbol{x})\log  \frac{\tilde{p}(0|\boldsymbol{x})}{p(0|\boldsymbol{x})}\right)d\boldsymbol{x}\\
=&\int \big(\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})\big) KL\Big(\tilde{p}(y|\boldsymbol{x})\Big\Vert p(y|\boldsymbol{x})\Big) d\boldsymbol{x}\end{aligned}\tag{7}$$
其中
$$\tilde{p}(1|\boldsymbol{x})=\frac{\tilde{p}(\boldsymbol{x})}{\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})}\tag{8}$$
$(7)$式是KL散度的积分，而KL散度非负，那么当“假设的分布形式是满足的、并且充分优化”时，$(7)$式应该为0，从而我们有$\tilde{p}(y|\boldsymbol{x})= p(y|\boldsymbol{x})$，也就是
$$\frac{\tilde{p}(\boldsymbol{x})}{\tilde{p}(\boldsymbol{x})+U(\boldsymbol{x})}=\tilde{p}(1|\boldsymbol{x})=p(1|\boldsymbol{x})=\sigma\Big(G(\boldsymbol{x};\boldsymbol{\theta})-\gamma\Big)\tag{9}$$
从中可以解得
$$\begin{aligned}\tilde{p}(\boldsymbol{x})=&\frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})}U(\boldsymbol{x})\\
=&\exp\Big\{G(\boldsymbol{x};\boldsymbol{\theta})-\gamma\Big\}U(\boldsymbol{x})\\
=&\exp\Big\{G(\boldsymbol{x};\boldsymbol{\theta})-\big(\gamma-\log U(\boldsymbol{x})\big)\Big\}\end{aligned}\tag{10}$$
如果$U(\boldsymbol{x})$取均匀分布，那么$U(\boldsymbol{x})$就只是一个常数，所以最终的效果表明$\gamma - \log U(\boldsymbol{x})$起到了$\log Z$的作用，而分布还是原来的分布$(3)$，$\boldsymbol{\theta}$还是原来的$\boldsymbol{\theta}$。

这就表明了NCE就是一种间接优化$(3)$式的巧妙方案：看似迂回，实则结果等价，并且$(5)$式的计算量也大大减少，因为计算量就只取决于采样的数目了。

## 一些插曲

一些跟NCE相关的话题，就都放在这里了。

### NCE与负采样简述

NCE的系统提出是在2010年的论文[《Noise\-contrastive estimation: A new estimation principle for unnormalized statistical models》](<http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>)中，后面训练大规模的神经语言模型基本上都采用NCE或者类似的loss了。论文的标题其实就表明了NCE的要点：<u>它是“非归一化模型”的一个“参数估计原理”，专门应对归一化因子难算的场景。</u>

但事实上，“负采样”的思想其实早就被使用了，比如就在2008年的ICML上，Ronan Collobert和Jason Weston在发表的[《A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning》](<https://ronan.collobert.com/pub/matos/2008_nlp_icml.pdf>)中已经用到了负采样的方法来训练词向量。要知道，那时候距离Word2Vec发布还有四五年！关于词向量和语言模型的故事，请参考licstar的[《词向量和语言模型》](<http://licstar.net/archives/328>)。

基于同样的为了降低计算量的需求，后来Google的Word2Vec也用上了负采样技巧，在很多任务下，它还比基于Huffman Softmax的效果要好，尤其是那个“词类比(word analogy)”实验。这里边的奥妙，我们马上就来分析。

### Word2Vec

现在我们落实到Word2Vec来分析一些事情。以Skip Gram模型为例，Word2Vec的目标是
$$p(w_j|w_i)=\frac{e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle}}{Z_i}\tag{11}$$
其中$\boldsymbol{u}_i, \boldsymbol{v}_j$都是待优化参数，代表着中心词和上下文的两套不同的词向量空间。显然地，这里的问题就是归一化因子计算量大，其中应对方案有Huffman Softmax和负采样。这里我们不关心Huffman Softmax，只需要知道它就是原来标准Softmax的一种近似就行了。我们来看负采样的，Word2Vec将优化目标变为了：
$$\mathop{\text{argmin}}_{\boldsymbol{u},\boldsymbol{v}} - \mathbb{E}_{w_j\sim \tilde{p}(w_j|w_i)}\log \sigma\Big(\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\Big) - \mathbb{E}_{w_j\sim \tilde{p}(w_j)}\log \Big[1-\sigma\Big(\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\Big)\Big]\tag{12}$$
这个式子看着有点眼花，总之它就是表达了“语料出现的Skip Gram视为正样本，随机采样的词作为负样本”的意思。

首先最明显的是，$(12)$式相比$(4),(5)$式，少引入了$\gamma$这个训练参数，或者就是说默认了$\gamma=0$，这允许吗？据说确实有人做过对比实验，结果显示训练出来的$\gamma$确实在0上下浮动，因此这个默认操作基本上是合理的。

其次，对于负样本，Word2Vec可不是“均匀地采样每一个词”，而是按照每个词本身的总词频来采样的。这样一来，$(10)$式就变成了
$$\tilde{p}(w_j|w_i)=\frac{p(1|w_i, w_j)}{p(0|w_i, w_j)}p(w_j)=e^{\langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle}\tilde{p}(w_j)\tag{13}$$
也就是说，最终的拟合效果是
$$\log \frac{\tilde{p}(w_j|w_i)}{\tilde{p}(w_j)} = \langle \boldsymbol{u}_i, \boldsymbol{v}_j\rangle\tag{14}$$
大家可以看到，左边就是两个词的互信息！<strong>本来我们的拟合目标是两个词的内积等于条件概率$\tilde{p}(w_j|w_i)$（的对数），现在经过负采样的Word2Vec，两个词的内积就是两个词的互信息。</strong>

现在大概就可以解释为什么Word2Vec的负采样会比Huffman Softmax效果要好些了。Huffman Softmax只是对Softmax做了近似，它本质上还是在拟合$\tilde{p}(w_j|w_i)$，而负采样技巧则是在拟合互信息$\log\frac{\tilde{p}(w_j|w_i)}{\tilde{p}(w_j)}$。我们之后，Word2Vec是靠词的共现来反应词义的，互信息比条件概率$\tilde{p}(w_j|w_i)$更能反映词与词之间“真正的”共现关系。换言之，$\tilde{p}(w_j|w_i)$反映的可能是“我认识周杰伦，周杰伦却不认识我”的关系，而互信息反映的是“你认识我，我也认识你”的关系，后者更能体现出语义关系。

我之前构造的另一个词向量模型[《更别致的词向量模型(三)：描述相关的模型》](<https://spaces.ac.cn/archives/4671>)中也表明了，基于互信息出发构造的模型，能理论上解释“词类比(word analogy)”等很多实验结果，这也间接证实了，基于互信息的“Skip Gram \+ 负采样”组合，是Word2Vec的一个绝佳组合。所以，根本原因不是Huffman Softmax和负采样本身谁更优的问题，而是它们的优化目标就已经不同。

## 列车已到终点站

本文的目的是介绍NCE这种精致的参数估算技巧，指出它可以在难以为完成归一化时来估算概率分布中的参数，原则上这是一种通用的方法，而且很可能，在某些场景下它是唯一可能的方案。

最后我们以Word2Vec为具体例子进行简单的分析，谈及了使用NCE时的一些细节问题，并且顺带解释了负采样为什么好的这个问题～

相关链接：[《词嵌入系列博客Part2：比较语言建模中近似softmax的几种方法》](<https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650720050&idx=2&sn=9fedc937d3128462c478ef7911e77687&chksm=871b034cb06c8a5a8db8a10f708c81025fc62084d871ac5d184bab5098cb64e939c1c23a7369&mpshare=1&scene=1&srcid=0613xBLYGgZUw99YG99QMP6p#rd>)
