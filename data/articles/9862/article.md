# 我在Performer中发现了Transformer\-VQ的踪迹

> 作者：苏剑林 · 科学空间 · 2023-11-29
>
> 原文：<https://spaces.ac.cn/archives/9862>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前些天我们在[《VQ一下Key，Transformer的复杂度就变成线性了》](<https://spaces.ac.cn/archives/9844>)介绍了“Transformer\-VQ”，这是通过将Key序列做VQ（Vector Quantize）变换来实现Attention复杂度线性化的方案。诚然，Transformer\-VQ提供了标准Attention到线性Attentino的一个非常漂亮的过渡，给人一种“大道至简”的美感，但熟悉VQ的读者应该能感觉到，当编码表大小或者模型参数量进一步增加时，VQ很可能会成为效果提升的瓶颈，因为它通过STE（Straight\-Through Estimator）估计的梯度大概率是次优的（[FSQ](<https://spaces.ac.cn/archives/9826>)的实验结果也算是提供了一些佐证）。此外，Transformer\-VQ为了使训练效率也线性化所做的梯度截断，也可能成为将来的效果瓶颈之一。

为此，笔者花了一些时间思考可以替代掉VQ的线性化思路。从Transformer\-VQ的$\exp\left(QC^{\top}\right)$形式中，笔者联想到了[Performer](<https://spaces.ac.cn/archives/7921>)，继而“顺藤摸瓜”地发现原来Performer可以视为Soft版的Transformer\-VQ。进一步地，笔者尝试类比Performer的推导方法来重新导出Transformer\-VQ，为其后的优化提供一些参考结果。

## 前情回顾

首先，让我们花一些时间回顾一下Transformer\-VQ。设$Q,K\in\mathbb{R}^{n\times d_k},V\in\mathbb{R}^{n\times d_v}$，Transformer\-VQ的关键，是对$K$做了如下VQ近似：
\begin{equation}K\approx\hat{K}\triangleq\Delta C\end{equation}
这里的$\Delta\in\{0,1\}^{n\times c},C\in\mathbb{R}^{c\times d_k}$都是矩阵，其中$C$是可训练的参数，$\Delta$则定义为：
\begin{equation}\Delta_{i,j} = \left\{\begin{aligned}& 1, \quad j=\mathop{\text{argmin}}_{k=1,2,\cdots,c} \Vert K_i - C_k\Vert \\
& 0, \quad\text{其他}\end{aligned}\right.\end{equation}
说白了，VQ就是用与$K_i$最相近的那个$C_j$来近似$K_i$。在这个近似之下，我们有（简单起见，以Encoder为例）
\begin{equation}\exp\left(Q\hat{K}{}^{\top}\right)V = \exp\left(QC^{\top}\Delta^{\top}\right)V = \exp\left(QC^{\top}\right)\Delta^{\top}V = \exp\left(QC^{\top}\right)(\Delta^{\top}V)\label{eq:transformer-vq}\end{equation}
了解线性Attention的读者很容易认出来，最后一个式子的运算就是线性复杂度的，它就是本文的主角之一Transformer\-VQ（的分子，还有分母同理）。

没有很复杂的推导，线性Attention就出来了，这就给我们一种感觉，仿佛我们是在对Key做近似的“不经意间”就将Attention的复杂度降为了线性，美感十足。因此，再次回到了我们已经提过多次的评价——Transformer\-VQ提供了标准Attention到线性Attentino的一个非常漂亮的过渡。

## 似曾相识

Transformer\-VQ的$\exp\left(QC^{\top}\right)$让笔者联想到了之前的文章[《Transformer升级之路：3、从Performer到线性Attention》](<https://spaces.ac.cn/archives/8338>)。在那篇文章中，笔者对Performer的结果做了一些简化，然后断言线性Attention的$Q,K$的最佳激活函数是$\exp$，而Transformer\-VQ同样出现了$\exp$，所以它们之间也许有着某种相关性。

为了挖掘这种联系，让我们请出Performer，它基于一个漂亮的近似：
\begin{equation}
e^{\boldsymbol{q}\cdot \boldsymbol{k}}=\mathbb{E}_{\boldsymbol{\omega}\sim \mathcal{N}(\boldsymbol{\omega};0,\boldsymbol{1}_d)}\left[e^{\boldsymbol{\omega}\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2} \,e^{\boldsymbol{\omega}\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2}\right]\approx\underbrace{\frac{1}{\sqrt{m}}\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{q}}}
\cdot  \underbrace{\frac{1}{\sqrt{m}}\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{k}}}
\label{eq:performer}\end{equation}
由于最后还要对所有$\boldsymbol{k}$的注意力归一化，所以去掉上式中的$\frac{1}{\sqrt{m}}$、$-\Vert \boldsymbol{q}\Vert^2/2$都不会影响最终结果，同时，如果假设$\boldsymbol{\omega}_1,\boldsymbol{\omega}_2,\cdots,\boldsymbol{\omega}_m$的模长都相等（参考[JL引理](<https://spaces.ac.cn/archives/8679>)），那么$\boldsymbol{k}$的指数都减去$\Vert\boldsymbol{\omega}_i\Vert^2/2$也不会影响结果。于是，Performer等价于用以下的格式做$\tilde{\boldsymbol{q}},\tilde{\boldsymbol{k}}$：
\begin{equation}\underbrace{\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}} \end{pmatrix}}_{\tilde{\boldsymbol{q}}}
\cdot  \underbrace{\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2-\Vert \boldsymbol{\omega}_1\Vert^2 / 2} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2-\Vert \boldsymbol{\omega}_2\Vert^2 / 2}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2-\Vert \boldsymbol{\omega}_m\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{k}}} = \underbrace{\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}} \end{pmatrix}}_{\tilde{\boldsymbol{q}}}
\cdot  \underbrace{\begin{pmatrix}e^{-\Vert \boldsymbol{k}-\boldsymbol{\omega}_1\Vert^2 / 2} \\
e^{-\Vert \boldsymbol{k} - \boldsymbol{\omega}_2\Vert^2 / 2}\\
\vdots\\
e^{-\Vert \boldsymbol{k} - \boldsymbol{\omega}_m\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{k}}} \propto \underbrace{\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}} \end{pmatrix}}_{\tilde{\boldsymbol{q}}}
\cdot  \underbrace{softmax\begin{pmatrix}e^{-\Vert \boldsymbol{k}-\boldsymbol{\omega}_1\Vert^2 / 2} \\
e^{-\Vert \boldsymbol{k} - \boldsymbol{\omega}_2\Vert^2 / 2}\\
\vdots\\
e^{-\Vert \boldsymbol{k} - \boldsymbol{\omega}_m\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{k}}} \end{equation}
对比最后一个式子和$\eqref{eq:transformer-vq}$，就会发现它们有诸多相似之处：$\boldsymbol{\omega}_1,\boldsymbol{\omega}_2,\cdots,\boldsymbol{\omega}_m$不就相当于编码表$C$？$\tilde{\boldsymbol{q}}$不就相当于$\exp\left(QC^{\top}\right)$？至于最后的$\tilde{\boldsymbol{k}}$，它以$-\Vert \boldsymbol{k} - \boldsymbol{\omega}_i\Vert^2 / 2$为logits做softmax，突出的不就是与$\boldsymbol{k}$最相近的那个$\boldsymbol{\omega}_i$？而softmax的极限就是one hot，所以这不正好对应着Transformer\-VQ的$\Delta$矩阵？因此，这不能说一模一样，但也有六七分相似了。

## 依样葫芦

当然，上述结果更多的是一种形象的类比而不是等价性，因为Performer本质上基于完全不同的近似思路，比如它里边的$\boldsymbol{\omega}_1,\boldsymbol{\omega}_2,\cdots,\boldsymbol{\omega}_m$是随机采样并固定下来的，这意味它们作为中心向量的近似程度其实是很差的。但这种类似引发了一个思考：能否模仿Performer的思路来重新推导一遍Transformer\-VQ呢？即像式$\eqref{eq:performer}$一样，先构造一个精确相等的结果，然后再转化为采样近似来得到线性版本。

经过几天的思考，笔者发现了一种可以构造出期望推导的方案。首先，我们借助[狄拉克函数](<https://spaces.ac.cn/archives/1870>)写出
\begin{equation}e^{\boldsymbol{q}\cdot \boldsymbol{k}} = \int e^{\boldsymbol{q}\cdot \boldsymbol{\omega}}\delta(\boldsymbol{\omega} - \boldsymbol{k})d\boldsymbol{\omega}\end{equation}
这是纯粹有狄拉克函数的定义给出的恒等式，还没涉及到任何精巧的运算或者近似。然而，当我们将它代入Attention（的分子）时，出现了一些有意思的结果：
\begin{equation}\sum_j e^{\boldsymbol{q}\cdot \boldsymbol{k}_j} \boldsymbol{v}_j = \sum_j \boldsymbol{v}_j\int e^{\boldsymbol{q}\cdot \boldsymbol{\omega}}\delta(\boldsymbol{\omega} - \boldsymbol{k}_j)d\boldsymbol{\omega} = \int e^{\boldsymbol{q}\cdot \boldsymbol{\omega}} \left[\sum_j \delta(\boldsymbol{\omega} - \boldsymbol{k}_j) \boldsymbol{v}_j\right]d\boldsymbol{\omega}\label{eq:inf-vq}\end{equation}
最后一个等号，不就正好是线性Attention的形式？！当然，由于需要对$\boldsymbol{\omega}$积分，所以上式跟[《Transformer升级之路：5、作为无限维的线性Attention》](<https://spaces.ac.cn/archives/8601>)一样，都是“无限维”的线性Attention，暂时只有形式上的价值。

通常来说，我们会将$\delta(\boldsymbol{\omega} - \boldsymbol{k}_j)$理解为正态分布$\mathcal{N}(\boldsymbol{\omega};\boldsymbol{k}_j,\sigma^2\boldsymbol{I})$在$\sigma\to 0$的极限，这也意味着$\delta(\boldsymbol{\omega} - \boldsymbol{k}_j)$具有条件分布$p(\boldsymbol{\omega}|\boldsymbol{k}_j)$的意义。不过，从生成模型的角度来看，狄拉克函数就是单点分布，说白了就是把训练集背下来，所以它没有抽象和泛化能力。为了缓解这一点，我们将$p(\boldsymbol{\omega}|\boldsymbol{k}_j)$用[GMM](<https://en.wikipedia.org/wiki/Mixture_model>)（Gaussian Mixture Model，高斯混合模型）来近似：
\begin{equation}p(\boldsymbol{\omega}|\boldsymbol{k}_j) \approx \sum_{y=1}^m \mathcal{N}(\boldsymbol{\omega};\boldsymbol{c}_y,\sigma^2\boldsymbol{I}) \,p(y|\boldsymbol{k}_j)  \end{equation}
代入式$\eqref{eq:inf-vq}$，然后取$\sigma\to 0$的极限，我们就得到
\begin{equation}\sum_j e^{\boldsymbol{q}\cdot \boldsymbol{k}_j} \boldsymbol{v}_j \approx \sum_{y=1}^m e^{\boldsymbol{q}\cdot \boldsymbol{c}_y} \left[\sum_j p(y|\boldsymbol{k}_j) \boldsymbol{v}_j\right]\end{equation}
这就得到一个有限维的线性Attention。如果将$p(y|\boldsymbol{k}_j)$对齐Transformer\-VQ的one hot分布$\Delta$的定义，那么得到的结果就是Transformer\-VQ的式$\eqref{eq:transformer-vq}$。

## 文章小结

本文介绍了笔者的一个发现：早期的线性Attention工作“Peformer”可以视为一个“Soft”版的Transformer\-VQ。然后，在这个观察上进一步得到了Transformer\-VQ的一个新推导：利用狄拉克函数将标准Attention转化为无限维线性Attention，然后加上GMM近似就可以得到Transformer\-VQ。
