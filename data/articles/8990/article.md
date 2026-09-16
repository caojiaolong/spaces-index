# 门控注意力单元（GAU）还需要Warmup吗？

> 作者：苏剑林 · 科学空间 · 2022-03-11
>
> 原文：<https://spaces.ac.cn/archives/8990>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《训练1000层的Transformer究竟有什么困难？》](<https://spaces.ac.cn/archives/8978>)发布之后，很快就有读者问到如果将其用到[《FLASH：可能是近来最有意思的高效Transformer设计》](<https://spaces.ac.cn/archives/8934>)中的“门控注意力单元（GAU）”，那结果是怎样的？跟标准Transformer的结果有何不同？本文就来讨论这个问题。

## 先说结论

事实上，GAU是非常容易训练的模型，哪怕我们不加调整地直接使用“Post Norm \+ Xavier初始化”，也能轻松训练个几十层的GAU，并且还不用Warmup。所以关于标准Transformer的很多训练技巧，到了GAU这里可能就无用武之地了\.\.\.

为什么GAU能做到这些？很简单，因为在默认设置之下，理论上$\text{GAU}(\boldsymbol{x}_l)$相比$\boldsymbol{x}_l$几乎小了两个数量级，所以
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \text{GAU}(\boldsymbol{x}_l))\approx \boldsymbol{x}_l\end{equation}
因此，GAU配合残差，在标准的初始化之下就已经很接近一个恒等函数，有这种性质的模型是非常容易训练的，通常都不需要Warmup。如果要对应上[《训练1000层的Transformer究竟有什么困难？》](<https://spaces.ac.cn/archives/8978>)的结论，这两个数量级相当于$\lambda=1,\alpha=100$，意味着它自动地包含了上百层的模型DeepNorm操作，因此理论上我们可以直接训练上百层的GAU模型而不需要特别的调整技巧。

## 模型假设

其实我们只需要对GAU的输入和输出做一个量级分析就行了。标准的GAU运算如下：
\begin{equation}\begin{aligned}
&\boldsymbol{O}=(\boldsymbol{U}\odot\boldsymbol{A}\boldsymbol{V})\boldsymbol{W}_o,\quad \boldsymbol{A}=\frac{1}{ns}\text{relu}^2\left(\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}\right)\\
&\boldsymbol{U}=\phi(\boldsymbol{X}\boldsymbol{W}_u),\quad\boldsymbol{V}=\phi(\boldsymbol{X}\boldsymbol{W}_v),\quad\boldsymbol{Z}=\phi(\boldsymbol{X}\boldsymbol{W}_z)
\end{aligned}\end{equation}
其中$\boldsymbol{X}\in\mathbb{R}^{n\times d}$、$\boldsymbol{W}_u,\boldsymbol{W}_v\in\mathbb{R}^{d\times e}$、$\boldsymbol{W}_z\in\mathbb{R}^{d\times s}$、$\boldsymbol{W}_o\in\mathbb{R}^{e\times d}$，$\mathcal{Q},\mathcal{K}$是简单的仿射变换，$\phi$是激活函数，默认是Swish。如果还有不清楚的地方，可以参考[《FLASH：可能是近来最有意思的高效Transformer设计》](<https://spaces.ac.cn/archives/8934>)。

我们假设$\boldsymbol{X}$的各个分量独立地服从标准正态分布$\mathcal{N}(0,1)$，然后$\boldsymbol{W}_u,\boldsymbol{W}_v,\boldsymbol{W}_z$的初始化分布是$\mathcal{N}(0,1/d)$而$\boldsymbol{W}_o$的初始化分布则是$\mathcal{N}(0,1/e)$独立重复采样出来的，这种初始化分布被称为[LeCun初始化](<http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>)，它的特点是能让输出的均值为0，并且保持输入输出的二阶矩一致，相关内容可以参考笔者之前的文章[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620>)。

## 基本积分

在这些假设之下，我们来逐一估计每步运算之后的分布。结合假设，由于LeCun初始化能保持二阶矩不变，所以$\boldsymbol{X}\boldsymbol{W}$也可以近似认为是标准正态分布的，于是我们可以用下面的式子估计加了激活函数$\phi$之后的均值和二阶矩：
\begin{equation}\begin{aligned}
\mu\triangleq\mathbb{E}[\phi(\varepsilon)] =&\, \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}\varepsilon^2\right)\phi(\varepsilon)d\varepsilon = 0.2066\cdots \\
\nu^2\triangleq\mathbb{E}[\phi(\varepsilon)^2] =&\, \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}\varepsilon^2\right)\phi(\varepsilon)^2d\varepsilon = 0.3557\cdots
\end{aligned}\end{equation}
换言之，$\boldsymbol{U},\boldsymbol{V},\boldsymbol{Z}$的分量均值和二阶矩分别是$\mu$和$\nu^2$，事实上后面只用到了二阶矩$\nu^2$，简单估计时，取$\nu=0.6$就行了。

## 自注意力

在初始阶段，我们有$\mathcal{Q}(\boldsymbol{Z})=\mathcal{K}(\boldsymbol{Z})=\boldsymbol{Z}$，所以初始阶段有$\boldsymbol{A}=\frac{1}{ns}\text{relu}^2\left(\boldsymbol{Z}\boldsymbol{Z}^{\top}\right)$，即（下面$i\neq j$）
\begin{equation}\begin{aligned}
&\boldsymbol{A}_{i,i} = \frac{1}{ns}\text{relu}^2\big(\left\langle\boldsymbol{Z}_i, \boldsymbol{Z}_i\right\rangle\big) \approx \frac{1}{ns}\text{relu}^2\big(s\mathbb{E}[\phi(\varepsilon)^2]\big) = \frac{sv^4}{n} \\
&\boldsymbol{A}_{i,j} = \frac{1}{ns}\text{relu}^2\big(\left\langle\boldsymbol{Z}_i, \boldsymbol{Z}_j\right\rangle\big) \approx \frac{1}{ns}\text{relu}^2\big(s\mathbb{E}[\phi(\varepsilon)]^2\big) = \frac{s\mu^4}{n}
\end{aligned}\end{equation}
注意到$\boldsymbol{A}_{i,i} / \boldsymbol{A}_{i,j} \approx \nu^4 / \mu^4 \approx 69 \gg 1$，也就是对角线元素远远大于非对角线元素，因此初始阶段的$\boldsymbol{A}$其实很接近单位阵的$\frac{sv^4}{n}$倍，即$\boldsymbol{A}\approx \frac{sv^4}{n}\boldsymbol{I}$，于是
\begin{equation}\boldsymbol{O}=(\boldsymbol{U}\odot\boldsymbol{A}\boldsymbol{V})\boldsymbol{W}_o\approx \frac{sv^4}{n}(\boldsymbol{U}\odot\boldsymbol{V})\boldsymbol{W}_o\end{equation}

## 剩余部分

对于$\boldsymbol{U}\odot\boldsymbol{V}$，它近似于两个独立同分布的变量$\varepsilon_i,\varepsilon_j$算出来的$\phi(\varepsilon_i)\phi(\varepsilon_j)$，所以
\begin{equation}\mathbb{E}[(\boldsymbol{U}\odot\boldsymbol{V})^2] \approx \mathbb{E}[\phi(\varepsilon_i)^2\phi(\varepsilon_j)^2] = \mathbb{E}[\phi(\varepsilon_i)^2]\mathbb{E}[\phi(\varepsilon_j)^2] = \nu^4\end{equation}
于是有（$\boldsymbol{W}_o$不改变二阶矩）
\begin{equation}\mathbb{E}[\boldsymbol{O}^2] \approx \mathbb{E}\left[\left(\frac{sv^4}{n}\boldsymbol{U}\odot\boldsymbol{V}\right)^2\right] = \mathbb{E}[\phi(\varepsilon_i)^2\phi(\varepsilon_j)^2] = \frac{s^2\nu^{12}}{n^2}\end{equation}
因此$\boldsymbol{O}$的量级是
\begin{equation}\boldsymbol{O} = \mathcal{O}\left(\sqrt{\frac{s^2\nu^{12}}{n^2}}\right) = \mathcal{O}\left(\frac{s\nu^{6}}{n}\right) \end{equation}
以常规的预训练设置$s=128,n=512$为例，$s\nu^6/n\approx 0.01$，因此在初始阶段经过$\text{GAU}(\boldsymbol{x}_l)$后出来的结果大致是$0.01\boldsymbol{x}_l$这个级别的，小两个数量级。当然，这是理论结果，实际上由于随机误差原因可能会更大或更小，不过就算更大了也不用担心，因为GAU还有下面的“疯狂尺度”性质。

## 疯狂尺度

在GAU论文的附录参考代码中，作者所用的初始化方法还不是LeCun初始化，而是0\.02标准差的正态分布。对于BERT base来说$d=786$，LeCun初始化给出的标准差是$1/\sqrt{d}\approx 0.036$，也就是说附录所用的初始化标准差大约只有LeCun初始化的一半。

当我们将GAU中所有的$\boldsymbol{W}$都换成$\lambda \boldsymbol{W}$时，我们将有
\begin{equation}\begin{aligned}
&\tilde{\boldsymbol{U}}=\phi(\boldsymbol{X}\lambda\boldsymbol{W}_u) \approx  \lambda\phi(\boldsymbol{X}\boldsymbol{W}_u)=\lambda \boldsymbol{U}\\
&\tilde{\boldsymbol{V}}=\phi(\boldsymbol{X}\lambda\boldsymbol{W}_v) \approx  \lambda\phi(\boldsymbol{X}\boldsymbol{W}_v)=\lambda \boldsymbol{V}\\
&\tilde{\boldsymbol{Z}}=\phi(\boldsymbol{X}\lambda\boldsymbol{W}_z) \approx  \lambda\phi(\boldsymbol{X}\boldsymbol{W}_z)=\lambda \boldsymbol{Z}\\
&\tilde{\boldsymbol{A}}=\frac{1}{ns}\text{relu}^2\left(\lambda^2\mathcal{Q}(\boldsymbol{Z})\mathcal{K}(\boldsymbol{Z})^{\top}\right) = \lambda^4 \boldsymbol{A}\\
&\tilde{\boldsymbol{O}}=(\tilde{\boldsymbol{U}}\odot\tilde{\boldsymbol{A}}\tilde{\boldsymbol{V}})\lambda\boldsymbol{W}_o \approx \lambda^7 \boldsymbol{O}
\end{aligned}\end{equation}
也就是说，如果所有初始化都缩小到原来的$\lambda$倍，那么GAU的输出将会缩小到原来的$\lambda^7$倍！这是关于GAU的一个相当疯狂的Scale，按照$\lambda=1/2$算，$\lambda^7$同样是0\.01级别，再次缩小了两个数量级！所以，如果按照原论文的初始化选择，我们理论上可以直接训练上万层的GAU模型！

## 本文小结

本文主要简单分析了一下GAU在初始阶段的数量级，得出标准初始化下的GAU其实已经接近恒等函数，因此具有相当容易训练的特点，基本上训练上百层的GAU模型也用不着额外的调整。
