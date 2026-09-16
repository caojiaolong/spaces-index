# 【理解黎曼几何】7\. 高斯\-博内公式

> 作者：苏剑林 · 科学空间 · 2016-10-21
>
> 原文：<https://spaces.ac.cn/archives/4033>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

令人兴奋的是，我们导出黎曼曲率的途径，还能够让我们一瞥高斯\-博内公式（ Gauss–Bonnet formula）的风采，真正体验一番研究内蕴几何的味道。

高斯\-博内公式是大范围微分几何学的一个经典的公式，它建立了空间的局部性质和整体性质之间的联系。而我们从一条几何的路径出发，结合一些矩阵变换和数学分析的内容，逐步导出了测地线、协变导数、曲率张量，现在可以还可以得到经典的高斯\-博内公式，可见我们在这条路上已经走得足够远了。虽然过程不尽善尽美，然而并没有脱离这个系列的核心：几何直观。本文的目的，正是分享黎曼几何的一种直观思路，既然是思路，以思想交流为主，不以严格证明为目的。因此，对于大家来说，这个系列权当黎曼几何的补充材料吧。

### 形式改写

首先，我们可以将式$(48)$重写为更有几何意义的形式。从
$$\Delta A^{\mu} =-R^{\mu}_{\alpha\beta\gamma} A^{\alpha} dx^{\beta}\delta x^{\gamma}=-g^{\mu\nu}R_{\nu\alpha\beta\gamma} A^{\alpha} dx^{\beta}\delta x^{\gamma} \tag{54} $$
出发，交换$\beta, \gamma$位置，得到
$$\Delta A^{\mu} =-g^{\mu\nu}R_{\nu\alpha\gamma\beta} A^{\alpha} dx^{\gamma}\delta x^{\beta} \tag{55} $$
利用$R_{\nu\alpha\beta\gamma}=-R_{\nu\alpha\gamma\beta}$，两式相加，得到
$$\begin{aligned}\Delta A^{\mu} =&-\frac{1}{2}g^{\mu\nu}R_{\nu\alpha\beta\gamma} A^{\alpha} (dx^{\beta}\delta x^{\gamma}-dx^{\gamma}\delta x^{\beta})\\
=&-\frac{1}{2}g^{\mu\nu}R_{\nu\alpha\beta\gamma} A^{\alpha} \det\begin{pmatrix} dx^{\beta} & \delta x^{\beta}\\ dx^{\gamma} & \delta x^{\gamma}\end{pmatrix}
\end{aligned} \tag{56} $$
这个式子的一般几何意义需要用外微分、曲面积分等内容诠释，我们不作讨论。但我们可以在二维空间这个特例中（即考虑三维欧式空间的二维曲面）考虑，这样几何意义变得清晰起来。在$n=2$时，每个求和指标实际上只有两项求和。即
$$\Delta A^{\mu} = -\frac{1}{2}\sum_{\nu=1}^2 \sum_{\alpha=1}^2\sum_{\beta=1}^2\sum_{\gamma=1}^2 g^{\mu\nu}R_{\nu\alpha\beta\gamma} A^{\alpha} \det\begin{pmatrix} dx^{\beta} & \delta x^{\beta}\\ dx^{\gamma} & \delta x^{\gamma}\end{pmatrix} \tag{57} $$
可以先算$\beta,\gamma$的求和，因为行列式的存在，实际上只有$\beta \neq \gamma$时有意义，然后再利用$R_{\nu\alpha\beta\gamma}=-R_{\nu\alpha\gamma\beta}$，得到
$$\Delta A^{\mu} = -\sum_{\nu=1}^2 \sum_{\alpha=1}^2 g^{\mu\nu}R_{\nu\alpha 12} A^{\alpha} \det\begin{pmatrix} dx^{1} & \delta x^{1}\\ dx^{2} & \delta x^{2}\end{pmatrix} \tag{58} $$
接着考虑$\nu,\alpha$的求和，类似地也只有$\nu\neq\alpha$的时候才有意义，而且也有反对称$R_{\nu\alpha\beta\gamma}=-R_{\alpha\nu\beta\gamma}$，因此，可以得到
$$\Delta A^{\mu} = - (g^{\mu 1} A^{2}-g^{\mu 2} A^{1}) R_{12 12}\det\begin{pmatrix} dx^{1} & \delta x^{1}\\ dx^{2} & \delta x^{2}\end{pmatrix} \tag{59} $$
改写成
$$\Delta A^{\mu} = - \sqrt{g}(g^{\mu 1} A^{2}-g^{\mu 2} A^{1}) \frac{R_{12 12}}{g}\sqrt{g}\det\begin{pmatrix} dx^{1} & \delta x^{1}\\ dx^{2} & \delta x^{2}\end{pmatrix} \tag{60} $$
注意到，在二维空间中，$\sqrt{g}\det\begin{pmatrix} dx^{1} & \delta x^{1}\\ dx^{2} & \delta x^{2}\end{pmatrix}$是有明确的几何意义的，它就是向量$(dx^1, dx^2)$和向量$(\delta x^1, \delta x^2)$张成的四边形的面积（参考式$(15)$的结果），我们简单记为$\Delta S$。而$\frac{R_{12 12}}{g}$正好是微分几何中定义的高斯曲率$K$，因此可以写成
$$\Delta A^{\mu} = - \sqrt{g}(g^{\mu 1} A^{2}-g^{\mu 2} A^{1}) K\Delta S \tag{61} $$

### 角差变化

现在，我们可以分析，向量$A^{\mu}$变到了$A^{\mu}+\Delta A^{\mu}$之后，这两个向量的夹角是多少。假设$A^{\mu}$是单位向量，那么先要计算内积
$$\begin{aligned}&\frac{g_{\mu\nu}A^{\mu}(A^{\nu}+\Delta A^{\nu})}{\sqrt{g_{\mu\nu}A^{\mu}A^{\nu}}\sqrt{g_{\mu\nu}(A^{\mu}+\Delta A^{\mu})(A^{\nu}+\Delta A^{\nu})}}\\
=&\frac{1+g_{\mu\nu}A^{\mu}\Delta A^{\nu}}{\sqrt{1+2g_{\mu\nu}A^{\mu}\Delta A^{\nu}+g_{\mu\nu}\Delta A^{\mu}\Delta A^{\nu}}} \end{aligned}\tag{62} $$
因为$\cos \Delta \theta = 1-\frac{\Delta\theta^2}{2}+\dots$，因此我们要算到二阶项，即$\Delta A^{\mu} \Delta A^{\nu}$这一项，近似到二阶，结果为
$$1-\frac{1}{2}\left[g_{\mu\nu}\Delta A^{\mu}\Delta A^{\nu}-(g_{\mu\nu}A^{\mu}\Delta A^{\nu})^2 \right] \tag{63} $$
所以
$$\begin{aligned}\Delta \theta =& \sqrt{g_{\mu\nu}\Delta A^{\mu}\Delta A^{\nu}-(g_{\mu\nu}A^{\mu}\Delta A^{\nu})^2}\\
=&\sqrt{(g_{\mu\nu} A^{\mu} A^{\nu})(g_{\mu\nu}\Delta A^{\mu}\Delta A^{\nu})-(g_{\mu\nu}A^{\mu}\Delta A^{\nu})^2} \end{aligned}\tag{64} $$
这实际上就是$A^{\mu}$与$\Delta A^{\mu}$张成的平行四边形的面积，在$n=2$时，也就是$\sqrt{g} \det\begin{pmatrix}A_1 & \Delta A_1\\ A_2 & \Delta A_2\end{pmatrix}=\sqrt{g} (A^{1}\Delta A^{2}-A^{2}\Delta A^{1})$，这时候代入$\Delta A^{1}, \Delta A^{2}$的表达式，就得到
$$\Delta \theta = g\left[g^{22}(A^1)^2-g^{12}A^2 A^1 - g^{21}A^1 A^2+g^{11}(A^2)^2\right]K\Delta S \tag{65} $$
对于二阶矩阵，有一个求逆公式
$$\begin{aligned}\begin{pmatrix} g_{11} & g_{12} \\ g_{21} & g_{22}\end{pmatrix}^{-1}=&\frac{1}{g_{11}g_{22}-g_{12}g_{21}}\begin{pmatrix} g_{22} & -g_{12} \\ -g_{21} & g_{11} \end{pmatrix}\\
=&\frac{1}{g}\begin{pmatrix} g_{22} & -g_{12} \\ -g_{21} & g_{11} \end{pmatrix} \end{aligned}\tag{66} $$
因此
$$g^{11}=\frac{g_{22}}{g},\,g^{12}=-\frac{g_{12}}{g},\,g^{21}=-\frac{g_{21}}{g},\,g^{22}=\frac{g_{11}}{g} \tag{67} $$
代入式$(65)$得到
$$\Delta \theta = [g_{11} (A^1)^2 + g_{12} A^1 A^2 + g_{21} A^2 A^1 + g_{22} (A^2)^2 ] K\Delta S = K\Delta S \tag{68} $$
最后一个等号是因为开始就假定了$A^{\mu}$是单位向量。这样，一个向量沿着小闭合曲线回来之后产生的角差，等于高斯曲率$K$和面元$\Delta S$的乘积。因此，我们可以推得，如果向量沿着大范围的闭合曲线$\mathbb{C}$平行移动回来，那么变化量则表示为面积分
$$\Delta \theta = \int_{\mathbb{C}} K d S \tag{69} $$
这就是微分几何中的高斯\-博内公式的主要内容，即角差等于高斯曲率的面积分，诸如球面三角形的内角和等内容都与它有关。它是整体微分几何的开山之作之一。

### 一点评述

值得一提的是，我们上述的讨论完全是内蕴的，也就是没有引入三维空间的曲面的概念，这是很吸引人的。大数学家陈省身自己说过，他一生最好的工作就是高维高斯\-博内特公式的内蕴证明（在他之前的证明是外蕴的）。可以看到，纯粹内蕴的工作是研究黎曼几何的追求。当然，我们这里最多是一个形象的引导，说不上一个完整的证明。但对于本系列来说，这个程度足够了。

注意到，读者可能困惑的一点是：作为体积，应该是非负的，但是如果写成矩阵行列式的形式，则有正有负，似乎存在矛盾。这在不引入外微分的前提之下，确实比较难澄清。在初等的分析范畴内，唯一的解决办法是，如果出现负体积，就直接加绝对值了。
