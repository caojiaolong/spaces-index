# 纠缠的时空（二）：洛仑兹变换的矩阵(续)

> 作者：苏剑林 · 科学空间 · 2013-02-27
>
> 原文：<https://spaces.ac.cn/archives/1923>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章中，我们以矩阵的方式推导出了洛仑兹变换。矩阵表述不仅仅具有形式上的美，还具有很重要的实用价值，比如可以很方便地寻找各种不变量。当洛仑兹变换用矩阵的方式表达出来后，很多线性代数中已知的理论都可以用在上边。在这篇小小的续集中，我们将尝试阐述这个思想。

本文中，继续设光速$c=1$。

我们已经得到了洛仑兹变换的矩阵形式：
\begin{equation}\left[\begin{array}{c} x\\t \end{array}\right]=\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & v\\ v & 1 \end{array}\right]\left[\begin{array}{c}x'\\t' \end{array}\right]\end{equation}

时间与空间虽然在相对论中已经融合为一个整体，但是时间的的确确具有它自己的独特性。为了体现这种独特性，我们沿用理论物理中一贯的做法，将时间乘上虚数单位$i$，那么洛仑兹变换矩阵变成了：
\begin{equation}\left[\begin{array}{c} x\\it \end{array}\right]=\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & -iv\\ iv & 1 \end{array}\right]\left[\begin{array}{c}x'\\it' \end{array}\right] \end{equation}

这种形式有它自己的优越性。我们将变换矩阵$\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & -iv\\ iv & 1 \end{array}\right]$记为$\boldsymbol{A}$，将$\left[\begin{array}{c} x\\it \end{array}\right]$和$\left[\begin{array}{c} x'\\it' \end{array}\right]$分别记为向量$\boldsymbol{x}$和$\boldsymbol{x'}$。即
\begin{equation}\boldsymbol{x}=\boldsymbol{A}\boldsymbol{x'}\end{equation}

### 光速不变原理

矩阵$\boldsymbol{A}$满足：$\boldsymbol{A}^T \boldsymbol{A}=I$，这就给我们带来了极大的便利，因为我们可以立马发现一个不变的量：
\begin{equation}\boldsymbol{x}^T \boldsymbol{x}=\boldsymbol{x'}^T \boldsymbol{A}^T \boldsymbol{A} \boldsymbol{x'}=\boldsymbol{x'}^T \boldsymbol{x'}\end{equation}

翻译过来就是$x^2-t^2=x'^2-t'^2$。这是光速不变原理的体现。

当然，我们也可以以$x^2-t^2=x'^2-t'^2$为出发点来推导洛仑兹变换的矩阵，这将得到矩阵的$\left[\begin{array}{c c}a & c\\ b& d \end{array}\right]$的两个关系式。再结合相对性原理以及相对运动就可以得到完整的矩阵的形式。读者不妨亲自尝试一下？

### 速度合成

如果两个洛仑兹变换的合成会出现什么结果呢？假设（同向运动）物体A相对于B的速度为$v$，B相对于C的速度为$w$，那么A相对于C的速度是多少呢？我们看看对应的两个洛仑兹变换矩阵$\boldsymbol{A}_v$和$\boldsymbol{A}_w$的合成，即
\begin{equation}\begin{aligned}\boldsymbol{x}=&\,\boldsymbol{A}_v \boldsymbol{A}_w \boldsymbol{x'} \\
=&\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & -iv\\ iv & 1 \end{array}\right]\frac{1}{\sqrt{1-w^2}}\left[\begin{array}{c c}1 & -iw\\ iw & 1 \end{array}\right] \\ =&\frac{1}{\sqrt{(1-v^2)(1-w^2)}}\left[\begin{array}{c c}1+wv & -i(w+v)\\ i(w+v) & 1+wv \end{array}\right] \\
=&\frac{1+wv}{\sqrt{(1-v^2)(1-w^2)}}\left[\begin{array}{c c}1 & -i(\frac{w+v}{1+wv})\\ i(\frac{w+v}{1+wv}) & 1 \end{array}\right]
\end{aligned}\end{equation}

将最后一个矩阵与$\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & -iv\\ iv & 1 \end{array}\right]$对比，不难速度的合成为：
\begin{equation}\frac{w+v}{1+wv}\end{equation}

这就是相对论的速度合成公式。（当然，如果你对$\frac{1+wv}{\sqrt{(1-v^2)(1-w^2)}}$不放心，可以再检验一下。）对于这种处理来说，这些结果是相当简洁显然的，足以充分体现矩阵研究的巨大魅力。
