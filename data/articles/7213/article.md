# CRF用过了，不妨再了解下更快的MEMM？

> 作者：苏剑林 · 科学空间 · 2020-02-24
>
> 原文：<https://spaces.ac.cn/archives/7213>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

HMM、MEMM、CRF被称为是三大经典概率图模型，在深度学习之前的机器学习时代，它们被广泛用于各种序列标注相关的任务中。一个有趣的现象是，到了深度学习时代，HMM和MEMM似乎都“没落”了，舞台上就只留下CRF。相信做NLP的读者朋友们就算没亲自做过也会听说过BiLSTM\+CRF做中文分词、命名实体识别等任务，却几乎没有听说过BiLSTM\+HMM、BiLSTM\+MEMM的，这是为什么呢？

今天就让我们来学习一番MEMM，并且通过与CRF的对比，来让我们更深刻地理解概率图模型的思想与设计。

## 模型推导

MEMM全称Maximum Entropy Markov Model，中文名可译为“最大熵马尔可夫模型”。不得不说，这个名字可能会吓退80%的初学者：最大熵还没搞懂，马尔可夫也不认识，这两个合起来怕不是天书？而事实上，不管是MEMM还是CRF，它们的模型都远比它们的名字来得简单，它们的概念和设计都非常朴素自然，并不难理解。

### 回顾CRF

作为对比，我们还是来回顾一下CRF。说是“回顾”，是因为笔者之前已经撰文介绍过CRF了，如果对CRF还不是很了解的读者，可以先去阅读旧作[《简明条件随机场CRF介绍（附带纯Keras实现）》](<https://spaces.ac.cn/archives/5542>)。简单起见，本文介绍的CRF和MEMM都是最简单的“线性链”版本。

本文都是以序列标注为例，即输入序列$\boldsymbol{x}=(x_1,x_2,\dots,x_n)$，希望输出同样长度的标签序列$\boldsymbol{y}=(y_1,y_2,\dots,y_n)$，那么建模的就是概率分布
\begin{equation}P(\boldsymbol{y}|\boldsymbol{x})=P(y_1,y_2,\dots,y_n|\boldsymbol{x})\label{eq:target}\end{equation}
CRF把$\boldsymbol{y}$看成一个整体，算一个总得分，计算公式如下
\begin{equation}\begin{aligned}f(y_1,y_2,\dots,y_n;\boldsymbol{x})=&\,f(y_1;\boldsymbol{x})+g(y_1,y_2)+\dots+g(y_{n-1},y_n)+f(y_n;\boldsymbol{x})\\
=&\,f(y_1;\boldsymbol{x}) + \sum_{k=2}^n \big(g(y_{k-1},y_k)+f(y_k;\boldsymbol{x})\big)\end{aligned}\end{equation}
这个打分函数的特点就是显式地考虑了相邻标签的关联，其实$g(y_{k-1},y_k)$被称为转移矩阵。现在得分算出来了，概率就是得分的softmax，所以最终概率分布的形式设为
\begin{equation}P(\boldsymbol{y}|\boldsymbol{x})=\frac{e^{f(y_1,y_2,\dots,y_n;\boldsymbol{x})}}{\sum\limits_{y_1,y_2,\dots,y_n}e^{f(y_1,y_2,\dots,y_n;\boldsymbol{x})}}\label{eq:crf-p}\end{equation}

如果仅局限于概念的话，那么CRF的介绍到此就结束了。总的来说，就是将目标序列当成一个整体，先给目标设计一个打分函数，然后对打分函数进行整体的softmax，这个建模理念跟普通的分类问题是一致的。CRF的困难之处在于代码实现，因为上式的分母项包含了所有路径的求和，这并不是一件容易的事情，但在概念理解上，笔者相信并没有什么特别困难之处。

### 更朴素的MEMM

现在我们来介绍MEMM，它可以看成一个极度简化的seq2seq模型。对于目标$\eqref{eq:target}$，它考虑分解
\begin{equation}P(y_1,y_2,\dots,y_n|\boldsymbol{x})=P(y_1|\boldsymbol{x})P(y_2|\boldsymbol{x},y_1)P(y_3|\boldsymbol{x},y_1,y_2)\dots P(y_n|\boldsymbol{x},y_1,y_2,\dots,y_{n-1})\end{equation}
然后假设标签的依赖只发生在相邻位置，所以
\begin{equation}P(y_1,y_2,\dots,y_n|\boldsymbol{x})=P(y_1|\boldsymbol{x})P(y_2|\boldsymbol{x},y_1)P(y_3|\boldsymbol{x},y_2)\dots P(y_n|\boldsymbol{x},y_{n-1})\label{eq:p-f}\end{equation}
接着仿照线性链CRF的设计，我们可以设
\begin{equation}P(y_1|\boldsymbol{x})=\frac{e^{f(y_1;\boldsymbol{x})}}{\sum\limits_{y_1}e^{f(y_k;\boldsymbol{x})}},\quad P(y_k|\boldsymbol{x},y_{k-1})=\frac{e^{g(y_{k-1},y_k)+f(y_k;\boldsymbol{x})}}{\sum\limits_{y_k}e^{g(y_{k-1},y_k)+f(y_k;\boldsymbol{x})}}\label{eq:memm}\end{equation}
至此，这就得到了MEMM了。由于MEMM已经将整体的概率分布分解为逐步的分布之积了，所以算loss只需要把每一步的交叉熵求和。

### 两者的关系

将式$\eqref{eq:memm}$代回式$\eqref{eq:p-f}$，我们可以得到
\begin{equation}P(\boldsymbol{y}|\boldsymbol{x})=\frac{e^{f(y_1;\boldsymbol{x})+g(y_1,y_2)+\dots+g(y_{n-1},y_n)+f(y_n;\boldsymbol{x})}}{\left(\sum\limits_{y_1}e^{f(y_1;\boldsymbol{x})}\right)\left(\sum\limits_{y_2}e^{g(y_1,y_2)+f(y_2;\boldsymbol{x})}\right)\dots\left(\sum\limits_{y_n}e^{g(y_{n-1},y_n)+f(y_n;\boldsymbol{x})}\right)}\label{eq:memm-p}\end{equation}

对比式$\eqref{eq:memm-p}$和式$\eqref{eq:crf-p}$，我们可以发现，MEMM跟CRF的区别仅在于分母（也就是归一化因子）的计算方式不同，CRF的式$\eqref{eq:crf-p}$我们称之为是全局归一化的，而MEMM的式$\eqref{eq:memm-p}$我们称之为是局部归一化的。

## 模型分析

这一节我们来分析MEMM的优劣、改进与效果。

### MEMM的优劣

MEMM的一个明显的特点是实现简单、速度快，因为它只需要每一步单独执行softmax，所以MEMM是完全可以并行的，速度跟直接逐步Softmax基本一样。而对于CRF，式$\eqref{eq:crf-p}$的分母并不是那么容易算的，它最终转化为一个递归计算，可以在$\mathcal{O}(n)$的时间内算出来（具体细节还请参考[《简明条件随机场CRF介绍（附带纯Keras实现）》](<https://spaces.ac.cn/archives/5542>)），递归就意味着是串行的，因此<u>当我们模型的主体部分是高度可并行的架构（比如纯CNN或纯Attention架构）时，CRF会严重拖慢模型的训练速度</u>。后面我们有比较MEMM和CRF的训练速度（当然，仅仅是训练慢了，预测阶段MEMM和CRF的速度都一样）。

至于劣势，自然也是有的。前面我们提到过，MEMM可以看成是一个极度简化的seq2seq模型。既然是这样，那么普通seq2seq模型有的弊端它都有。seq2seq中有一个明显的问题是exposure bias，对应到MEMM中，它被称之为label bias，大概意思是：在训练MEMM的时候，对于当前步的预测，都是假设前一步的真实标签已知，这样一来，如果某个标签$A$后面只能接标签$B$，那么模型只需要通过优化转移矩阵就可以实现这一点，而不需要优化输入$\boldsymbol{x}$对$B$的影响（即没有优化好$f(B;\boldsymbol{x})$）；然而，在预测阶段，真实标签是不知道的，我们可能无法以较高的置信度预测前一步的标签$A$，而由于在训练阶段，当前步的$f(B;\boldsymbol{x})$也没有得到强化，所以当前步$B$也无法准确预测，这就有可能导致错误的预测结果。

### 双向MEMM

label bias可能不好理解，但我们可以从另外一个视角看MEMM的不足：事实上，相比CRF，MEMM明显的一个不够漂亮的地方就是它的不对称性——它是从左往右进行概率分解的。笔者的实验表明，如果能解决这个不对称性，能够稍微提高MEMM的效果。笔者的思路是：将MEMM也从右往左做一遍，这时候对应的概率分布是
\begin{equation}P(\boldsymbol{y}|\boldsymbol{x})=\frac{e^{f(y_1;\boldsymbol{x})+g(y_1,y_2)+\dots+g(y_{n-1},y_n)+f(y_n;\boldsymbol{x})}}{\left(\sum\limits_{y_n}e^{f(y_n;\boldsymbol{x})}\right)\left(\sum\limits_{y_{n-1}}e^{g(y_n,y_{n-1})+f(y_{n-1};\boldsymbol{x})}\right)\dots\left(\sum\limits_{y_1}e^{g(y_2,y_1)+f(y_1;\boldsymbol{x})}\right)}\end{equation}
然后也算一个交叉熵，跟从左往右的式$\eqref{eq:memm-p}$的交叉熵平均，作为最终的loss。这样一来，模型同时考虑了从左往右和从右往左两个方向，并且也不用增加参数，弥补了不对称性的缺陷。作为区分，笔者类比Bi\-LSTM的命名，将其称为Bi\-MEMM。

> 注：Bi\-MEMM这个名词并不是在此处首次出现，据笔者所查，首次提出Bi\-MEMM这个概念的，是论文[《Bidirectional Inference with the Easiest\-First Strategy for Tagging Sequence Data》](<https://www.aclweb.org/anthology/H05-1059/>)，里边的Bi\-MEMM是指一种MEMM的双向解码策略，跟本博客的Bi\-MEMM含义并不相同。

### 实验结果演示

为了验证和比较MEMM的效果，笔者将CRF和MEMM同时写进了[bert4keras](<https://github.com/bojone/bert4keras>)中，并且写了中文分词（[task\_sequence\_labeling\_cws\_crf\.py](<https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_cws_crf.py>)）和中文命名实体识别（[task\_sequence\_labeling\_ner\_crf\.py](<https://github.com/bojone/bert4keras/blob/master/examples/task_sequence_labeling_ner_crf.py>)）两个脚本。在这两个脚本中，从CRF切换到MEMM非常简单，只需将<code>ConditionalRandomField</code>替换为<code>MaximumEntropyMarkovModel</code>。

详细的实验数据就不贴出来了，反正就是一些数字罢了，下面直接给出一些相对比较的结果：

> 1、相同的实验参数下，Bi\-MEMM总是优于MEMM，MEMM总是优于Softmax；
> 
> 
> 
> 2、相同的实验参数下，CRF基本上不差于Bi\-MEMM；
> 
> 
> 
> 3、当编码模型能力比较强时，CRF与Bi\-MEMM效果持平；当编码模型较弱时，CRF优于Bi\-MEMM，幅度约为0\.5%左右；
> 
> 
> 
> 4、用12层bert base模型作为编码模型时，Bi\-MEMM比CRF快25%；用2层bert base模型作为编码模型时，Bi\-MEMM比CRF快1\.5倍。

（注：由于笔者发现Bi\-MEMM效果总比MEMM略好，并且两者的训练时间基本无异，所以bert4keras里边的MaximumEntropyMarkovModel默认就是Bi\-MEMM。）

## 思考与拓展

根据上面的结论，在深度学习时代，MEMM的“没落”似乎就可以理解了——MEMM除了训练速度快点之外，相比CRF似乎也就没什么好处了，两者的预测速度是一样的，而很多时候我们主要关心预测速度和效果，训练速度稍微慢点也无妨。这两个模型的比较结果是有代表性的，可以说这正是所有全局归一化和局部归一化模型的差异：全局归一化模型效果通常好些，但实现通常相对困难一些；局部归一化模型效果通常不超过全局归一化模型，但胜在易于实现，并与易于拓展。

如何易于拓展？这里举两个方向的例子。

第一个例子，假如标签数很大的时候，比如用序列标注的方式做文本纠错或者文本生成时（相关例子可以参考论文[《Fast Structured Decoding for Sequence Models》](<https://papers.cool/arxiv/1910.11555>)），标签的数目就是词表里边的词汇数$|V|$，就算用了subword方法，词汇数少说也有一两万，这时候转移矩阵的参数量就达到数亿（$|V|^2$），难以训练。读者可能会想到低秩分解，不错，低秩分解可以将转移矩阵的参数量控制为$2d|V|$，其中$d$是分解的中间层维度。不幸的是，对于CRF来说，低秩分解不能改变归一化因子计算量大的事实，因为CRF的归一化因子依然需要恢复为$|V|\times|V|$的转移矩阵才能计算下去，所以对于标签数目巨大的场景，CRF无法直接使用。但幸运的是，对于MEMM来说，低秩分解可以有效低降低训练时的计算量，从而依然能够有效的使用。bert4keras里边所带的<code>MaximumEntropyMarkovModel</code>已经把低秩序分解集成进去了，有兴趣的读者可以查看源码了解细节。

第二个例子，上述介绍的CRF和MEMM都只考虑相邻标签之间的关联，如果我要考虑更复杂的邻接关联呢？比如同时考虑$y_k$跟$y_{k-1},y_{k-2}$的关联？这时候CRF的全局归一化思路就很难操作了，归根结底还是归一化因子难算；但如果是MEMM的局部归一化思路就容易进行。事实上，笔者之前设计的[信息抽取分层标注思路](<https://spaces.ac.cn/archives/%E5%9F%BA%E4%BA%8EDGCNN%E5%92%8C%E6%A6%82%E7%8E%87%E5%9B%BE%E7%9A%84%E8%BD%BB%E9%87%8F%E7%BA%A7%E4%BF%A1%E6%81%AF%E6%8A%BD%E5%8F%96%E6%A8%A1%E5%9E%8B>)，也可以说是一种跟MEMM类似的局部归一化的概率图模型，它考虑的关联就更复杂化了。

## 文章小结

本文介绍并简单推广了跟CRF一样同为概率图经典案例的MEMM，它与CRF的区别主要是归一化方式上的不一样，接着笔者从实验上对两者做了简单的比较，得出MEMM训练更快但效果不优于CRF的结论。尽管如此，笔者认为MEMM仍有可取之处，所以最后构思了MEMM的一些拓展。
