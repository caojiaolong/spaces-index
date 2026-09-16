# Naive Bayes is all you need ?

> 作者：苏剑林 · 科学空间 · 2023-06-08
>
> 原文：<https://spaces.ac.cn/archives/9648>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

很抱歉，起了这么个具有标题党特征的题目。在写完[《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度》](<https://spaces.ac.cn/archives/9617>)之后，笔者就觉得朴素贝叶斯（Naive Bayes）跟Attention机制有很多相同的特征，后来再推导了一下发现，<u>Attention机制其实可以看成是一种广义的、参数化的朴素贝叶斯</u>。既然如此，“[Attention is All You Need](<https://spaces.ac.cn/archives/4765>)”不也就意味着“Naive Bayes is all you need”了？这就是本文标题的缘由。

接下来笔者将介绍自己的思考过程，分析如何从朴素贝叶斯角度来理解Attention机制。

## 朴素贝叶斯

本文主要考虑语言模型，它要建模的是$p(x_t|x_1,\cdots,x_{t-1})$。根据贝叶斯公式，我们有
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) = \frac{p(x_1,\cdots,x_{t-1}|x_t)p(x_t)}{p(x_1,\cdots,x_{t-1})}\propto p(x_1,\cdots,x_{t-1}|x_t)p(x_t)\end{equation}
根据独立假设$p(x_1,\cdots,x_{t-1}|x_t) = \prod\limits_{j=1}^{t-1} p(x_j|x_t)$，我们有
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) \propto \prod_{j=1}^{t-1} p(x_j|x_t)p(x_t)\end{equation}
再次根据贝叶斯公式$p(x_j|x_t)=\frac{p(x_t|x_j)p(x_j)}{p(x_t)}\propto\frac{p(x_t|x_j)}{p(x_t)}$，得到
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) \propto \frac{1}{[p(x_t)]^{t-2}}\prod_{j=1}^{t-1} p(x_t|x_j)\end{equation}
两边取对数得到
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \sum_{j=1}^{t-1}\log p(x_t|x_j) - (t - 2) \log p(x_t) + \text{常数}\end{equation}

## 一般化结果

相同的推导我们在[《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度》](<https://spaces.ac.cn/archives/9617>)也进行过，跟该文章一样，我们将上式一般化为：
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = (1 + \beta)\mathcal{P}[\log p(x_t|x_j)] - \beta \log p(x_t) + \text{常数}\end{equation}
这里的$\beta$作为超参数来调，$\mathcal{P}$是某种Pooling方式。接下来我们主要看$\beta=0$、以加权平均为Pooling的例子，即
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \sum_j a_{t,j} \log p(x_t|x_j) + \text{常数}\label{eq:nb-core}\end{equation}
这里的$a_{t,j}$是$x_{t-1}$与$x_j$的函数。

可能有读者想问，这个一般化的式子还能算是朴素贝叶斯吗？笔者认为它可以作为广义的朴素贝叶斯来看待，因为朴素贝叶斯可以视为各个$\log p(x_t|x_j)$的等权平均，这里则是换成了更一般化的加权平均。不过，将$a_{t,j}$选取为$x_{t-1}$与$x_j$的函数，突出了$x_{t-1}$的地位，改善了朴素贝叶斯的无序性这一弊端。所以更准确来说，式$\eqref{eq:nb-core}$是2\-gram语言模型与朴素贝叶斯的结合。

## 注意力初现

接下来，将$\log p(x_t|x_j)$进一步参数化，我们就可以得见Attention的形式了。不难发现，$p(x_t|x_j)$实质上就是以前Word2Vec的Skip Gram模型，它的常规建模方式是“Embedding \+ 内积 \+ Softmax”，即
\begin{equation}p(x_t|x_j) = \frac{e^{v(x_j)\cdot w(x_t)}}{Z(x_j)},\quad Z(x_j) = \sum_{x_t\in Vocab}e^{v(x_j)\cdot w(x_t)}\end{equation}
所以我们简单地认为
\begin{equation}\log p(x_t|x_j) = v(x_j)\cdot w(x_t) + \text{常数}\end{equation}
代入到式$\eqref{eq:nb-core}$，得到
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \left(\sum_j a_{t,j} v(x_j)\right)\cdot w(x_t) + \text{常数}\label{eq:nb-core-2}\end{equation}
括号中的式子，我们将它单独拿出来，当作通常用特征融合运算，它其实就是常规的Attention。所以说，单层的Attention做语言模型，实则就是广义的朴素贝叶斯。

当然，这里我们还没有将$a_{t,j}$确定下来。上一节我们说$a_{t,j}$是$x_{t-1}$与$x_j$的函数，然后同时还要归一化（加权平均），所以比较简单的方式就是像Skip Gram一样“Embedding \+ 内积 \+ Softmax”：
\begin{equation}a_{t,j} = \frac{e^{q(x_{t-1})\cdot k(x_j)}}{Z_t},\quad Z_t = \sum_{j=1}^{t-1} e^{q(x_{t-1})\cdot k(x_j)}\end{equation}
代入到式$\eqref{eq:nb-core-2}$，就是目前最常用的Dot\-Product Attention了。当然，这种方式不是唯一的，还有加性Attention等，选择Dot\-Product的最主要原因是它可以在比较省显存的前提下实现并行。

## 层叠与残差

不管怎么参数化，单层的朴素贝叶斯能力总是有限的，所以需要进一步提高模型的复杂度。从神经网络的角度来看，提高模型复杂度的主要方式是增加深度，也就是层与层之间的堆叠。那么，从概率分布的角度如何理解这种堆叠呢？答案是隐变量模型。

所谓隐变量模型，就是引入隐变量$z_1,z_2,\cdots,z_{t-1}$，使得
\begin{equation}p(x_t|x_1,\cdots,x_{t-1}) = \int p(x_t|z_1,\cdots,z_{t-1})p(z_1,\cdots,z_{t-1}|x_1,\cdots,x_{t-1})dz_1 \cdots dz_{t-1}\end{equation}
说白了，就是通过简单分布的叠加来拟合更复杂的分布，跟GMM（高斯混合模型）的思想是一致的。基于前面的讨论，$p(x_t|z_1,\cdots,z_{t-1})$我们同样用朴素贝叶斯建模，即从特征层面就是单层Attention。而对于$p(z_1,\cdots,z_{t-1}|x_1,\cdots,x_{t-1})$，我们按照自回归模型的特点，分解为
\begin{equation}p(z_1,\cdots,z_{t-1}|x_1,\cdots,x_{t-1}) = \prod_{k=1}^{t-1} p(z_k|x_1,\cdots,x_k)\end{equation}
这样每个$p(z_k|x_1,\cdots,x_k)$形式上就跟$p(x_t|z_1,\cdots,z_{t-1})$一样了，于是同样可以用朴素贝叶斯建模。简单起见，$z_k$我们定义为连续型变量，$p(z_k|x_1,\cdots,x_k)$则定义为[狄拉克分布](<https://spaces.ac.cn/archives/1870>)，于是积分可以直接算出来，结果就是两层Attention的堆叠了。

最后，Transfromer中还有一个关键成分是残差，实际上它就是将式$\eqref{eq:nb-core}$一般化为
\begin{equation}\log p(x_t|x_1,\cdots,x_{t-1}) = \log p(x_t|x_{t-1}) + \sum_j a_{t,j} \log p(x_t|x_j) + \text{常数}\end{equation}
可以理解为一种突出了2\-gram的地位的Pooling方式，算是一种先验。最后，还剩下的FeedForward层、LayerNorm层等，这些层不涉及token之间的交互，可以理解为是更复杂地参数化的朴素贝叶斯。

当然，这样笼统的解释看上去有些勉强，但笔者原本的想法，也不是精准地解释Transformer或Attention，而是期望是能从朴素贝叶斯角度来够获得一些关于长度外推的新思路。但很遗憾，目前笔者还没有得到预期的结果。然而，尽管看上去像是盲目的自恋，但笔者依然认为上述朴素贝叶斯和隐变量模型的视角还有进一步挖掘的潜力，比如看上去我们可以从朴素贝叶斯角度解释基于Attention的语言模型的In\-Context Learning为啥会有效。

## 文章总概述

本文阐述了朴素贝叶斯与Attention机制之间的关联，显示了Attention可被视为一种广义的朴素贝叶斯。从这个视角，我们还可以进一步地理解Attention中的层叠与残差等内容。
