# DiVeQ：一种非常简洁的VQ训练方案

> 作者：苏剑林 · 科学空间 · 2025-10-08
>
> 原文：<https://spaces.ac.cn/archives/11328>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

对于坚持离散化路线的研究人员来说，VQ（Vector Quantization）是视觉理解和生成的关键部分，担任着视觉中的“Tokenizer”的角色。它提出在2017年的论文[《Neural Discrete Representation Learning》](<https://papers.cool/arxiv/1711.00937>)，笔者在2019年的博客[《VQ\-VAE的简明介绍：量子化自编码器》](<https://spaces.ac.cn/archives/6760>)也介绍过它。

然而，这么多年过去了，我们可以发现VQ的训练技术几乎没有变化，都是STE（Straight\-Through Estimator）加额外的Aux Loss。STE倒是没啥问题，它可以说是给离散化运算设计梯度的标准方式了，但Aux Loss的存在总让人有种不够端到端的感觉，同时还引入了额外的超参要调。

幸运的是，这个局面可能要结束了，上周的论文[《DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick》](<https://papers.cool/arxiv/2509.26469>)提出了一个新的STE技巧，它最大亮点是不需要Aux Loss，这让它显得特别简洁漂亮！

## 离散编码

老规矩，我们还是先回顾一下VQ现有的训练方案。首先要指出的是，[VQ（Vector Quantizatio）](<https://en.wikipedia.org/wiki/Vector_quantization>)本身实际上是一个很古老的概念，可以追溯到上世纪80年代，原意是将向量聚类并用对应的聚类中心代替，从而实现数据压缩。

但我们这里说的VQ，主要是指论文[《Neural Discrete Representation Learning》](<https://papers.cool/arxiv/1711.00937>)所提的VQ\-VAE中的VQ。当然，VQ的定义本身没啥变化，都是向量到聚类中心的映射，VQ\-VAE一文的核心是提供了一套对latent变量进行VQ然后解码重构的端到端训练方案，其难点是VQ这一步是离散化操作，没有现成的梯度，需要为它设计梯度。

用公式来说，普通的AE（AutoEncoder）是：
\begin{equation}z = encoder(x),\quad \hat{x}=decoder(z),\quad \mathcal{L}=\Vert x - \hat{x}\Vert^2 \end{equation}
其中$x$是原始输入，$z$是编码向量，$\hat{x}$是重构结果。VQ\-VAE想要做的是，基于VQ的思想，将$z$变成编码表$E=\{e_1,e_2,\cdots,e_K\}$中的一个：
\begin{equation}q = \newcommand{argmin}{\mathop{\text{argmin}}}\argmin_{e\in E} \Vert z - e\Vert\end{equation}
然后$decoder$以$q$为输入来进行重构。由于$q$是编码表的下标一一对应，所以$q$实际上是$x$的一个整数编码。当然，为了保证重构效果，实际使用时肯定不止编码成单个向量，而是多个向量，VQ之后就变成了多个整数。所以，VQ\-VAE想要做的事情，就是将输入编码成一个整数序列，这跟文本的Tokenizer异曲同工。

## 梯度设计

现在我们要训练的模块包括$encoder$、$decoder$和编码表$E$，由于VQ操作带有$\argmin$运算，梯度到$q$这里就断了，无法传到$encoder$中去。

VQ\-VAE使用了一个名为STE的技巧，它说送入$decoder$的还是VQ后的$q$，但反向传播求梯度的时候，我们用的是VQ前的$z$，这样就可以把梯度传回给$encoder$，它可以用stop\_gradient算子（$\newcommand{sg}{\mathop{\text{sg}}}\sg$）实现：
\begin{equation}z = encoder(x),\quad q = \argmin_{e\in E} \Vert z - e\Vert,\quad z_q = z + \sg[q - z],\quad \hat{x} = decoder(z_q)\end{equation}
简单来说，STE实现的效果是$z_q=q$但$\nabla z_q = \nabla z$，这样$encoder$就有梯度了，但$q$没梯度了，没法优化编码表了。为了解决这个问题，VQ\-VAE补了两项Aux Loss：
\begin{equation}\mathcal{L} = \Vert x - \hat{x}\Vert^2 + \beta\Vert q -  \sg[z]\Vert^2 + \gamma\Vert z -  \sg[q]\Vert^2 \end{equation}
这两项Loss分别代表$q$向$z$靠近和$z$向$q$靠近，跟VQ的原始思想一致。STE跟这两项Aux Loss组合起来，就构成了标准的VQ\-VAE。此外还有一个简单的变体，就是直接设$\beta=0$，但用$z$的滑动平均来更新编码表，这也等价于指定用SGD来更新$q$的Aux Loss。

这里顺便指出一点，VQ\-VAE虽然被原论文冠以“[VAE](<https://spaces.ac.cn/tag/vae/>)”之名，但它实际上就是个AE，所以叫“VQ\-AE”原则上更贴切，只不过这个名字已经被叫开了，我们只好沿用它。后来的[VQGAN](<https://papers.cool/arxiv/2012.09841>)，则是在VQ\-VAE基础上叠加GAN Loss等技巧，提高了重构的清晰度。

## 替代作品

对于笔者来说，这两个额外的Aux Loss是让人很难受的地方。估计业界跟笔者同样感受的人也不少，所以时不时也有一些相关的改进工作出来。

其中，最为“釜底抽薪”的做法是换用VQ以外的离散化方案，比如[《简单得令人尴尬的FSQ：“四舍五入”超越了VQ\-VAE》](<https://spaces.ac.cn/archives/9826>)介绍的FSQ，它就不需要Aux Loss。如果说VQ是对高维向量进行聚类，那么FSQ就是对低维向量做“四舍五入”，从而实现离散化。然而，笔者在[这篇文章](<https://spaces.ac.cn/archives/9844>)评价过，FSQ并不能在任意场景都取代VQ，所以改进VQ本身依然是有价值的。

在提出DiVeQ之前，其实原作者还提出过一个名为“[NSVQ](<https://ieeexplore.ieee.org/abstract/document/9696322>)”的方案，向“废除”Aux Loss迈出了一小步，它将$z_q$改为
\begin{equation}z_q = z + \Vert q - z\Vert \times \frac{\varepsilon}{\Vert \varepsilon\Vert},\qquad \varepsilon\sim\mathcal{N}(0, I)\label{eq:nsvq}\end{equation}
这里$\varepsilon$是跟$z,q$同大小的向量，其分量服从标准正态分布。替换为这个新的$z_q$之后，由于$\Vert q - z\Vert$的可导性，$q$也是有梯度的，所以原则上不需要Aux Loss就能训练编码表了。NSVQ的几何意义很直观，就是在“以$z$为中心、$\Vert q-z\Vert$为半径的圆”上均匀采样，缺点是此时送入$decoder$的并不是$q$，而推理阶段我们关心的是$q$的重构效果，所以NSVQ的训练和推理具有不一致性。

## 主角登场

从NSVQ出发，如果想保持前向为$q$，但同时保留$\Vert q - z\Vert$带来的梯度，那么很容易提出改进版：
\begin{equation}z_q = z + \Vert q - z\Vert \times \sg\left[\frac{q - z}{\Vert q - z\Vert}\right]\label{eq:diveq0}\end{equation}
它在前向的时候严格有$z_q = q$，但反向时保留了$z$和$\Vert q - z\Vert$的梯度。这便是DiVeQ论文附录中的“DiVeQ\-detach”，而正文的DiVeQ，则算是式$\eqref{eq:diveq0}$和$\eqref{eq:nsvq}$的某种插值
\begin{equation}z_q = z + \Vert q - z\Vert \times \sg\left[\frac{q - z + \varepsilon}{\Vert q - z + \varepsilon\Vert}\right],\qquad \varepsilon\sim\mathcal{N}(0, \sigma^2 I)\label{eq:diveq}\end{equation}
很明显，当$\sigma=0$时，结果是“DiVeQ\-detach”，当$\sigma\to\infty$时，结果是“NSVQ”。原论文附录对$\sigma$做了搜索，大致结论是$\sigma^2 = 10^{-3}$是一个普遍较优的选择。

论文的实验结果显示，尽管式$\eqref{eq:diveq}$引入了随机性，同时也有一定程度的训练推理不一致性，但相比式$\eqref{eq:diveq0}$它的效果更好。不过，就笔者的审美而言，效果不能以牺牲优雅为代价，所以式$\eqref{eq:diveq0}$的“DiVeQ\-detach”才是笔者心中的理想方案，下面分析中的DiVeQ都是指代“DiVeQ\-detach”。

## 理论分析

遗憾的是，原论文并没有多少理论分析，所以这一节笔者尝试对DiVeQ的有效性、它跟VQ原本训练方案的相关性等做一个基本的分析。首先，考虑式$\eqref{eq:diveq0}$的一般形式
\begin{equation}z_q = z + r(q, z) \times \sg\left[\frac{q - z}{r(q, z)}\right]\end{equation}
其中$r(q,z)$是任意关于$q,z$的可导的标量函数，它可以视为$q,z$的任意的距离函数。我们将损失函数记为$\mathcal{L}(z_q)$，那么它的微分是
\begin{equation}d\mathcal{L} = \langle\nabla_{z_q} \mathcal{L},d z_q\rangle = \left\langle\nabla_{z_q} \mathcal{L},dz + dr \times\frac{q-z}{r}\right\rangle = \langle\nabla_{z_q} \mathcal{L},d z\rangle + \langle\nabla_{z_q} \mathcal{L}, q-z\rangle d(\ln r)\end{equation}
其中$\langle\nabla_{z_q} \mathcal{L},d z\rangle$是原本VQ就有的，DiVeQ则多出了额外的$\langle\nabla_{z_q} \mathcal{L}, q-z\rangle d(\ln r)$，或者说，相当于引入了Aux Loss $\sg[\langle\nabla_{z_q} \mathcal{L}, q-z\rangle] \ln r$，如果$r$代表$q,z$的某种距离函数，那么它就是在拉近$q,z$的距离，跟VQ引入的Aux Loss异曲同工，这就成功从理论上解释了DiVeQ。

但别高兴太早，这个解释成立的前提是系数$\langle\nabla_{z_q} \mathcal{L}, q-z\rangle > 0$，否则反而在拉大距离了。为了证明这一点，我们考虑损失函数$\mathcal{L}(z)$在$z_q$处的一阶近似
\begin{equation}\mathcal{L}(z) \approx \mathcal{L}(z_q) + \langle\nabla_{z_q} \mathcal{L}, z - z_q\rangle = \mathcal{L}(z_q) + \langle\nabla_{z_q} \mathcal{L}, z - q\rangle\end{equation}
即$\langle\nabla_{z_q} \mathcal{L}, q-z\rangle\approx \mathcal{L}(z_q) - \mathcal{L}(z)$。注意$z,z_q$分别是VQ前、后的特征，VQ是一个损失信息的过程，所以用$z$来做目标任务（比如重构）会比$z_q$容易，因此只要开始收敛，可以预期$z$的损失函数更低，即$\mathcal{L}(z_q) - \mathcal{L}(z) > 0$，这样我们就证明了$\langle\nabla_{z_q} \mathcal{L}, q-z\rangle > 0$大概率是成立的。

## 改进方向

严格来讲，$\langle\nabla_{z_q} \mathcal{L}, q-z\rangle > 0$只能说是DiVeQ有效的必要条件，要充分论证它的有效性还需要证明这个系数“恰到好处”。由于$r(q,z)$的任意性，我们只能具体函数具体分析。我们考虑$r(q,z)=\Vert q-z\Vert^{\alpha}$，那么它等效于引入如下Aux Loss
\begin{equation}\sg[\langle\nabla_{z_q} \mathcal{L}, q-z\rangle] \ln \Vert q-z\Vert^{\alpha}\approx \sg[\mathcal{L}(z_q) - \mathcal{L}(z)]\times \alpha\ln \Vert q-z\Vert\end{equation}
系数$\mathcal{L}(z_q) - \mathcal{L}(z)$跟主损失$\mathcal{L}(z_q)$是齐次的，这意味着它能够较好地自适应主损失的Scale，还能根据VQ前后的效果差距来调节Aux Loss权重。至于$\alpha$取多少好，笔者认为要看实验，个人尝试调了一下，发现$\alpha=1$确实普遍表现较好。有兴趣的读者也可以自行尝试调整$\alpha$，甚至尝试换用别的$r(q, z)$。

需要指出的是，DiVeQ只是提供了一种新的免Aux Loss的VQ训练方案，原则上它不解决VQ的其他问题，比如编码表利用率低、编码表坍缩等，原本在“STE \+ Aux Loss”场景下性质有效的增强技巧，都可以考虑叠加到DiVeQ上。原论文就将DiVeQ跟[SFVQ](<https://papers.cool/arxiv/2410.20573>)结合起来，提出SF\-DiVeQ，用来缓解编码表坍缩等问题。

不过，个人感觉SFVQ有点繁琐，所以这里就不打算展开介绍了，而且作者选择叠加SFVQ，更多是因为SFVQ本就是他之前的作品，属于一脉相承下来了。笔者更喜欢的是[《VQ的又一技巧：给编码表加一个线性变换》](<https://spaces.ac.cn/archives/10519>)中介绍的线性变换技巧，即在编码表后面再加一个线性变换，实测这样也能明显增强DiVeQ的效果。

## 文章小结

本文介绍了VQ（Vector Quantization）的一种新的训练方案，它只需通过STE实现，不需要加额外的Aux Loss，从而显得特别简洁优雅。
