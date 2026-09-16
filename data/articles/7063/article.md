# JoSE：球面上的词向量和句向量

> 作者：苏剑林 · 科学空间 · 2019-11-11
>
> 原文：<https://spaces.ac.cn/archives/7063>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这篇文章介绍一个发表在NeurIPS 2019的做词向量和句向量的模型JoSE（Joint Spherical Embedding），论文名字是[《Spherical Text Embedding》](<https://papers.cool/arxiv/1911.01196>)。JoSE模型思想上和方法上传承自Doc2Vec，评测结果更加漂亮，但写作有点故弄玄虚之感。不过笔者决定写这篇文章，是因为觉得里边的某些分析过程有点意思，可能会对一般的优化问题都有些参考价值。

## 优化目标

在思想上，这篇文章基本上跟[Doc2Vec](<https://papers.cool/arxiv/1405.4053>)是一致的：为了训练句向量，把句子用一个id表示，然后把它也当作一个词，跟句内所有的词都共现，最后训练一个Skip Gram模型，训练的方式都是基于负采样的。跟Doc2Vec不一样的是，JoSE将全体向量的模长都归一化了（也就是只考虑单位球面上的向量），然后训练目标没有用交叉熵，而是用hinge loss：
\begin{equation}\max(0, m - \cos(\boldsymbol{u}, \boldsymbol{v}) - \cos(\boldsymbol{u}, \boldsymbol{d}) + \cos(\boldsymbol{u}', \boldsymbol{v}) + \cos(\boldsymbol{u}', \boldsymbol{d})\label{eq:loss}\end{equation}
其中$\boldsymbol{u}$是“中心词”的词向量，$\boldsymbol{v}$是“上下文词”的词向量，它们分别来自两套词向量空间，$\boldsymbol{d}$则是当前句的句向量，而$\boldsymbol{u}'$负采样得到的“中心词”词向量，最后的$m > 0$是一个常数。以前做相似度模型的读者应该能很轻松读懂这个优化目标的含义，它就是希望句子内的“词\-词\-句”打分$\cos(\boldsymbol{u}, \boldsymbol{v}) + \cos(\boldsymbol{u},\boldsymbol{d})$要高于“词\-随机词\-句”打分$\cos(\boldsymbol{u}', \boldsymbol{v}) + \cos(\boldsymbol{u}', \boldsymbol{d})$，但不需要太高，只要高出$m$就行了。

假定$\boldsymbol{u},\boldsymbol{v},\boldsymbol{d}$都已经归一化的情况下，那么目标$\eqref{eq:loss}$就是（每个向量被假设为列向）：
\begin{equation}\max(0, m - \boldsymbol{v}^{\top}\boldsymbol{u} - \boldsymbol{d}^{\top}\boldsymbol{u} + \boldsymbol{v}^{\top} \boldsymbol{u}' + \boldsymbol{d}^{\top} \boldsymbol{u}')\label{eq:loss2}\end{equation}

## 梯度下降

目标$\eqref{eq:loss}$或$\eqref{eq:loss2}$其实并没有什么新鲜之处，跟大多数词向量的目标类似，都是用内积衡量词的相关性，只不过这里的向量归一化过，所以内积就是$\cos$，至于hinge loss和交叉熵孰优孰劣，我倒觉得不会有什么太大差别。

事实上，笔者觉得文章比较有意思的是它后面对梯度的几何分析，在这里笔者用自己的话重复一下求解过程。设$\boldsymbol{x}$是全体$\boldsymbol{u},\boldsymbol{v},\boldsymbol{d}$向量中的其中一个，然后假设现在固定所有的其他向量，只优化$\boldsymbol{x}$，设总的loss为$f(\boldsymbol{x})$，那这个优化过程有两种描述方式：
\begin{equation}\mathop{\text{argmin}}_{\boldsymbol{x},\,\Vert\boldsymbol{x}\Vert=1} f(\boldsymbol{x})\quad\text{或}\quad \mathop{\text{argmin}}_{\boldsymbol{\theta}} f\left(\frac{\boldsymbol{\theta}}{\Vert \boldsymbol{\theta}\Vert}\right)
\end{equation}
也就是说，我们可以将这个问题理解为带有约束$\Vert \boldsymbol{x}\Vert=1$的$f(\boldsymbol{x})$最小化问题，也可以通过设$\boldsymbol{x}=\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert$将它转化为无约束的$f(\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert)$最小化问题。由于带约束的优化问题我们不熟悉，所以只好按照后一种方式来理解。

跟复杂模型不同的是，词向量算是一个比较简单的模型，所以我们最好手动求出它的梯度形式，然后编写对应函数进行梯度下降来优化，而不借助于一些自动求导工具。对于$f(\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert)$，我们不难求得
\begin{equation}\nabla_{\boldsymbol{\theta}}\,f\left(\frac{\boldsymbol{\theta}}{\Vert \boldsymbol{\theta}\Vert}\right) = \frac{1}{\Vert\boldsymbol{\theta}\Vert}\left(\boldsymbol{I} - \boldsymbol{x}\boldsymbol{x}^{\top}\right)\nabla_{\boldsymbol{x}}\,f\left(\boldsymbol{x}\right)
\end{equation}
其中已经对部分变量进行了$\boldsymbol{x}=\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert$的替换，根据上述结果，梯度下降的迭代公式为
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_{t} -  \eta_t\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)
\end{equation}
其中$\eta_t$是当前时刻的学习率，而因子$1/\Vert\boldsymbol{\theta}\Vert$由于只是个标量，所以被整合到学习率中了。然后我们也可以写出：
\begin{equation}\begin{aligned}\boldsymbol{x}_{t+1} = \frac{\boldsymbol{\theta}_{t+1}}{\Vert \boldsymbol{\theta}_{t+1}\Vert} =& \frac{\boldsymbol{\theta}_{t} -  \eta_t\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \boldsymbol{\theta}_{t} -  \eta_t\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}\\
=& \frac{\boldsymbol{x}_{t} -  \eta_t/\Vert\boldsymbol{\theta}\Vert\times\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \boldsymbol{x}_{t} - \eta_t/\Vert\boldsymbol{\theta}\Vert\times\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}
\end{aligned}\end{equation}
再次将$1/\Vert\boldsymbol{\theta}\Vert$整合到学习率中，我们可以得到只有$\boldsymbol{x}_t$的更新公式：
\begin{equation}\boldsymbol{x}_{t+1} = \frac{\boldsymbol{x}_{t} -  \eta_t\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \boldsymbol{x}_{t} - \eta_t\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}\end{equation}

## 更新量修正

至此，上述内容都是很常规的推导，而接下来就是我说的比较有意思的地方了。首先有
\begin{equation}\begin{aligned}\boldsymbol{g}=&\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\\
=&\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right) - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\\
=&\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right) - \boldsymbol{x}_t\Vert \nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\Vert \cos\left(\boldsymbol{x}_t,\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right)
\end{aligned}
\end{equation}
可以看到，$\boldsymbol{x}_t\boldsymbol{x}_t^{\top}\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$实际上就是向量$\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$在$\boldsymbol{x}_t$方向上的投影分量，而整个$\boldsymbol{g}$其实就是一个与$\boldsymbol{x}_t$垂直的向量，如下图示：

[![梯度的几何图示](<https://spaces.ac.cn/usr/uploads/2019/11/2555037745.png>)](<https://spaces.ac.cn/usr/uploads/2019/11/2555037745.png>)

梯度的几何图示

在上图中，红色向量代表$\boldsymbol{x}_t$，蓝色向量代表$\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$，如果没有$\Vert \boldsymbol{x}\Vert=1$这个约束的话，那更新量应该直接由$\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$决定，但是因为有了约束，所以更新量由$\boldsymbol{g}=\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$决定。然而，有下面两种不同的$\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$，都可能导致同一个$\boldsymbol{g}$：

[![第一种情况，∇xf(x)跟x的方向很靠近](<https://spaces.ac.cn/usr/uploads/2019/11/1303082149.png>)](<https://spaces.ac.cn/usr/uploads/2019/11/1303082149.png>)

第一种情况，∇xf(x)跟x的方向很靠近

[![第二种情况，∇xf(x)跟x的方向几乎相反](<https://spaces.ac.cn/usr/uploads/2019/11/1752952142.png>)](<https://spaces.ac.cn/usr/uploads/2019/11/1752952142.png>)

第二种情况，∇xf(x)跟x的方向几乎相反

第一种情况的$\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$的方向跟$\boldsymbol{x}_t$很靠近，第二种情况则相反，但它们的$\boldsymbol{g}$是一致的。前面说了，如果没有约束的话，$\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$才是梯度，换言之$-\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$就是合理的更新方向；现在有了约束，$-\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$虽然不能指出最合理的梯度方向，但直觉来看，它应该还是跟更新量有关的。

在第一种情况下，$-\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$跟$\boldsymbol{x}_t$方向差得比较远，意味着这种情况下更新量应该大一些；而第二种情况下，$-\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$跟$\boldsymbol{x}_t$方向比较一致，而我们只关心$\boldsymbol{x}_{t+1}$的方向，不关心它的模长，所以按理说这种情况下更新量应该小一些。

所以，哪怕这两种情况下$\boldsymbol{g}$都一样，我们还是需要有所区分，一个很自然的想法是：既然<strong>$-\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)$和$\boldsymbol{x}_t$的方向的一致性会对更新量的大小有所影响</strong>，所以不妨用
\begin{equation}1-\cos(-\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right),\boldsymbol{x}_t)=1+\frac{\boldsymbol{x}_t^{\top}\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}
\end{equation}
来调节更新量，这个调节因子正好满足“方向越一致，调节因子越小”的特性。这个自然的想法就导致了最终的更新公式
\begin{equation}\boldsymbol{x}_{t+1} = \frac{\boldsymbol{x}_{t} -  \eta_t\left(1+\frac{\boldsymbol{x}_t^{\top}\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}\right)\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \boldsymbol{x}_{t} - \eta_t\left(1+\frac{\boldsymbol{x}_t^{\top}\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)}{\left\Vert \nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}\right)\left(\boldsymbol{I} - \boldsymbol{x}_t\boldsymbol{x}_t^{\top}\right)\nabla_{\boldsymbol{x}_t}\,f\left(\boldsymbol{x}_t\right)\right\Vert}\end{equation}

## 故弄玄虚

有意思的地方讲完了，下面讲一下没有意思的地方了。对NLP有稍微深入一点了解的读者（看过Word2Vec的数学原理，推导过常规模型的梯度）应该会觉得，上面前两节内容并没有什么很深奥的内容，第三节的几何解释和学习率调节有点新颖，但也是有迹可循的内容。不过要是去看原论文的话，那感觉可能就完全不一样了，作者用“概率分布”、“黎曼流形上的优化”等语言，把上述本该比较容易理解的内容，描述得让人云里雾里，深有故弄玄虚之感。

首先，我最不理解的一点是，作者在一开始就做了一个不合理的假设（将词向量连续化），然后花了不少篇幅来论证$p(v|u)\sim e^{\cos(\boldsymbol{v},\boldsymbol{u})}$和$p(u|d)\sim e^{\cos(\boldsymbol{u},\boldsymbol{d})}$对应着Von Mises–Fisher分布。然后呢？就没有然后了，后面的所有内容跟这个Von Mises–Fisher分布可以说没有半点关系，所以不理解作者写这部分内容的目的是什么。

接着，在优化那部分，作者说带约束$\Vert\boldsymbol{x}\Vert=1$的$f(\boldsymbol{x})$最小化问题不能用梯度下降，所以只能用“黎曼梯度下降”，然后就开始“炫技”了：先说说黎曼流形，然后给出一般的指数映射，再然后给出黎曼梯度，一波高端操作下来，最后却只保留了一个大家都能懂的方案：$\boldsymbol{x}=\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert$。这时我就很“服气”了，虽然作者的逻辑和推导都没有毛病，但是一波操作下来最后却给看众一个$\boldsymbol{x}=\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert$的朴素结果，那为什么不一开始就直接讨论$f(\boldsymbol{x}=\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert)$的优化呢？非得要去黎曼流形上面把普通读者绕晕？

此外，我说的比较有意思的部分，就是更新量的几何解释以及得到的调节因子，作者也说得挺迷糊的。总之，笔者认为，论文的理论推导部分，很多地方都充斥着很多不必要的专业术语，无端加深了普通看众的理解难度。

最后强调一下，笔者从来不反对“一题多解”，也不反对将简单的内容深化、抽象化，因为“深化”、“抽象化”确实也可能获得更全面的认识，或者能显示各个分支之间的联系。但是这种“深化”、“抽象化”应该要建立在一个大多数人都能理解的简单解的基础上进行的，而不是为了“深化”、“抽象化”而特意舍去了大多数人能理解的简单解。

## 实验结果

吐槽归吐槽，在实验部分，JoSE做得还是很不错的。首先给出了JoSE的高效的C语言实现：

<strong>Github</strong>：[https://github\.com/yumeng5/Spherical\-Text\-Embedding](<https://github.com/yumeng5/Spherical-Text-Embedding>)

我试用了一下，训练确实很快速，训练好的词／句向量结果可以用gensim的KeyedVectors加载。另外我还看了一下源代码，很简练清晰，也方便做二次修改。

至于实验结果，论文给出的词／句向量评测上面，JoSE也是比较领先的：

[![词相似度评测](<https://spaces.ac.cn/usr/uploads/2019/11/1500294122.png>)](<https://spaces.ac.cn/usr/uploads/2019/11/1500294122.png>)

词相似度评测

[![文本聚类评测](<https://spaces.ac.cn/usr/uploads/2019/11/214489308.png>)](<https://spaces.ac.cn/usr/uploads/2019/11/214489308.png>)

文本聚类评测

## 文章总结

本文分享了一个发表在NeurIPS 2019的文本向量模型JoSE，着重讲了一下笔者觉得有启发性的部分，并用自己的方法给出了推导过程。JoSE可以认为是Doc2Vec的自然变种，在细微之处做了调整，并且在优化方法上提出了作者自己的见解，除却一些疑似故弄玄虚的地方之外，还不失为一个可圈可点的工作。
