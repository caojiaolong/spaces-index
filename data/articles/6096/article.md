# 再谈非方阵的行列式

> 作者：苏剑林 · 科学空间 · 2018-10-16
>
> 原文：<https://spaces.ac.cn/archives/6096>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

几年前，笔者曾经以自己对矩阵的粗浅理解写了一个“理解矩阵”系列，其中有一篇[《为什么只有方阵有行列式？》](<https://spaces.ac.cn/archives/2757>)讨论了非方阵的行列式问题，里边给出了“非方针的行列式不好看”和“方阵的行列式就够了”的观点。本文来再次思考这个问题。

首先回顾方阵的行列式，其实行列式最重要的价值在于它的几何意义：

> n维方阵的行列式的绝对值，等于它的各个行（或列）向量所张成的n维立体的超体积。

这个几何意义是行列式的一切重要性的源头，相关的讨论可以参考[《行列式的点滴》](<https://spaces.ac.cn/archives/1770>)，它也是我们讨论非方阵行列式的基础。

## 分析

对于方阵$\boldsymbol{A}_{n\times n}$来说，可以将它看成$n$个行向量的组合，也可以看成$n$个列向量的组合，不管是哪一种，行列式的绝对值都等于这$n$个向量所张成的$n$维立体的超体积。换句话说，对于方阵来说，行、列向量的区分不改变行列式。

对于非方阵$\boldsymbol{B}_{n \times k}$就不一样了，不失一般性，假设$n > k$。我们可以将它看成<u>$n$个$k$维行向量</u>的组合，也可以看成<u>$k$个$n$维列向量</u>的组合。非方针的行列式，应该也具有同样含义，即它们所张成的立体的超体积。

我们来看第一种情况，如果看成$n$个$k$维行向量，那么就得视为这$n$个向量张成的$n$维体的超体积了，但是要注意$n > k$，因此这$n$个向量必然线性相关，因此它们根本就张不成一个$n$维体，也许是一个$n-1$维体甚至更低，这样一来，它的$n$维体的超体积自然为0。

但是第二种情况就没有那么平凡了。如果看成$k$个$n$维列向量，那么这$k$个向量虽然是$n$维的，但它们张成的是一个$k$维体，这$k$维体的超体积未必为0。我们就以这个非平凡的体积作为非方阵行列式的定义好了。

## 定义

对于第二种情况，有一个非常巧妙的、可以借助方阵行列式的定义：
\begin{equation}|\det \boldsymbol{B}| = \sqrt{\det (\boldsymbol{B}^{\top}\boldsymbol{B})}\label{eq:dingyi}\end{equation}
当然，这样只定义了行列式的绝对值，不过已经够用了。大多数情况下，我们都只用到了行列式的绝对值。

可以发现，这个定义兼容了方阵行列式的结果，并且后面我们会进一步证明，这个定义果能保留行列式的几何意义。

我们现在来算两个例子。第一个例子，考虑$n\times 1$的矩阵
\begin{equation}\boldsymbol{X} = \begin{pmatrix}x_1 \\ x_2 \\ \vdots \\ x_n\end{pmatrix}\end{equation}
根据定义$\eqref{eq:dingyi}$，算得
\begin{equation}|\det \boldsymbol{X}| = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}\end{equation}
根据我们的定义，它应该表示$1$个$n$维列向量的“1维体积”，而类比之下，所谓“1维体积”就是长度，而上式正好是向量的模长公式。也就是说，在$n\times 1$情形，定义$\eqref{eq:dingyi}$跟我们的设想是兼容的。

第二个例子是$n\times 2$矩阵
\begin{equation}\boldsymbol{Z} = \begin{pmatrix}x_1 & y_1 \\ x_2 & y_2 \\ \vdots & \vdots \\ x_n & y_n\end{pmatrix}=(\boldsymbol{x}, \boldsymbol{y})\end{equation}
根据定义$\eqref{eq:dingyi}$算，可以得到最终的结果是
\begin{equation}|\det \boldsymbol{Z}| = \sqrt{\boldsymbol{x}^{\top}\boldsymbol{x}\boldsymbol{y}^{\top}\boldsymbol{y} - (\boldsymbol{x}^{\top}\boldsymbol{y})^2}\end{equation}
不难发现，这个结果正好是$\boldsymbol{x},\boldsymbol{y}$所张成的平行四边形的面积的平方，因为根据定义计算的平行四边形的面积应该是
\begin{equation}\begin{aligned}|\boldsymbol{x}|\cdot|\boldsymbol{y}|\cdot\sin\theta =& |\boldsymbol{x}|\cdot|\boldsymbol{y}|\cdot\sqrt{1-\cos^2\theta}\\
=&|\boldsymbol{x}|\cdot|\boldsymbol{y}|\cdot\sqrt{1-\left(\frac{\boldsymbol{x}^{\top}\boldsymbol{y}}{|\boldsymbol{x}|\cdot|\boldsymbol{y}|}\right)^2}
\end{aligned}\end{equation}
也就是说，对于$n\times 2$矩阵，定义$\eqref{eq:dingyi}$跟我们的期望也是一样的。

## 证明

现在来考虑一般的证明，对于$\boldsymbol{B}_{n \times k}$矩阵
\begin{equation}\boldsymbol{B}_{n \times k} = \begin{pmatrix}b_{11} & \dots & b_{1k}\\ b_{21} & \dots & b_{2k}\\
\vdots & \ddots & \vdots\\
b_{n1} & \dots & b_{nk}\end{pmatrix} = (\boldsymbol{b}_1,\dots,\boldsymbol{b}_k)\end{equation}
并且$n > k$。首先，由我们熟悉的Gram–Schmidt正交化过程我们可以知道，存在$n\times k$的正交矩阵$\boldsymbol{U}_{n\times k}$（$k$个两两正交的$n$维单位列向量）和$k\times k$的下三角矩阵$\boldsymbol{C}_{k\times k}$，使得
\begin{equation}\boldsymbol{B}_{n \times k}=\boldsymbol{U}_{n\times k}\boldsymbol{C}_{k\times k}\end{equation}
这在数学上称为“QR分解”。我们知道正交变换不会改变任何几何性质，所以$\boldsymbol{B}_{n \times k}$的行列式应该等于$\boldsymbol{C}_{k\times k}$的行列式（的绝对值），即$|\det \boldsymbol{C}_{k\times k}|$。

这样一来，就有
\begin{equation}\begin{aligned}|\det \boldsymbol{B}_{n\times k}| =& |\det \boldsymbol{C}_{k\times k}|\\
=& \sqrt{\det\left(\boldsymbol{C}_{k\times k}^{\top}\boldsymbol{C}_{k\times k}\right)}\\
=& \sqrt{\det\left[\left(\boldsymbol{U}_{n\times n}^{\top}\boldsymbol{B}_{n \times k}\right)^{\top}\left(\boldsymbol{U}_{n\times n}^{\top}\boldsymbol{B}_{n \times k}\right)\right]}\\
=& \sqrt{\det\left(\boldsymbol{B}_{n \times k}^{\top}\boldsymbol{B}_{n \times k}\right)}
\end{aligned}\end{equation}
所以，对于$n\times k$且$n > k$的矩阵$\boldsymbol{B}$来说，一个非平凡的、合理的矩阵定义就是$\sqrt{\det (\boldsymbol{B}^{\top}\boldsymbol{B})}$。显然，如果$n < k$，那么定义是$\sqrt{\det (\boldsymbol{B}\boldsymbol{B}^{\top})}$

## 结语

本来从几何意义出发讨论了非方阵的行列式问题，最终表明式$\eqref{eq:dingyi}$可以作为一个比较合理的非方阵行列式的定义。尽管理论上$\eqref{eq:dingyi}$只能定义行列式的绝对值，但是大多数情况下都已经足够了。

至于非方阵的行列式的应用，我们知道做积分变换的时候我们有个雅可比行列式来保证变换的非平凡的，类似地，或许也可以通过非方阵的行列式来保证升维、降维变换的非平凡性。当然，这是一个构思，目前还在思考这类问题，欢迎有兴趣的读者讨论。
