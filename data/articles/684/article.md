# 从牛顿力学角度研究宇宙学

> 作者：苏剑林 · 科学空间 · 2010-06-17
>
> 原文：<https://spaces.ac.cn/archives/684>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![Universe_expansion](<https://spaces.ac.cn/usr/uploads/2010/06/2879022515.png>)](<https://spaces.ac.cn/attachment/685/>)

Universe\_expansion

不少天文爱好者对宇宙学这方面的内容“听而生畏”，觉得没有爱因斯坦的广义相对论等复杂理论基础是不可理解的。的确，这种观点没有错，当前的宇宙学对宇宙的精确描述，的确是建立在广义相对论和量子力学等理论的基础之上的。BoJone也只是在书上略略浏览，根本谈不上有什么了解。但是，对于一般的天文爱好者来说，只要对牛顿力学和微积分有一定的了解，就可以对我们的宇宙有一个大概的描述，也能够得出很多令人惊喜的结论。相信进行了这项工作之后，很多爱好者都会改观：原来宇宙学也并不是那么难\.\.\.并且能够得出这样的一个结论：广义相对论虽然对牛顿引力理论进行了彻底的改革，但是从数学的角度来讲，它仅仅对牛顿力学进行了修正。

维基上对“宇宙学”的解释是

> 宇宙学（或宇宙论）译自英文之“Cosmology”。宇宙学是对宇宙整体的研究，并且延伸探讨至人类在宇宙中的地位。虽然宇宙学这个词是最近才有的，人们对宇宙的研究已经有很长的一段历史，牵涉到科学、哲学、神秘学以及宗教。

[![WMAP_the_CMB_anisotropy](<https://spaces.ac.cn/usr/uploads/2010/06/1550004969.jpg>)](<https://spaces.ac.cn/attachment/686/>)

WMAP\_the\_CMB\_anisotropy

简单来讲，宇宙学就是专门研究宇宙的物理起源及其演化、从最大的尺度去研究宇宙的本质的学科。我们平常所听到的“哈勃定律”、“大爆炸模型”、“宇宙膨胀”等内容，都属于宇宙学的范畴。如果我们设宇宙是一个各向同性的球体，半径为$R$（$R$是时间的函数），总质量为$M$（虽然质量可以转变为能量，但是对于宇宙这样的大质量，在很长时间内$M$仍然可以视为常数），只是考虑物质之间的引力作用，那么对于宇宙边缘的一个质量为m的物体，则有
\begin{equation}m\frac{d^2 R}{dt^2}=-\frac{GMm}{R^2}\end{equation}
则牛顿宇宙学的基本方程是
\begin{equation}\frac{d^2 R}{dt^2}=-\frac{GM}{R^2}\label{eq:1}\end{equation}
这种方程我们之前已经讨论过，只要把$\frac{d^2 R}{dt^2}$改写成$\dot{R}\frac{d\dot{R}}{dR}$，代入$\eqref{eq:1}$
\begin{equation}\dot{R}\frac{d\dot{R}}{dR}=-\frac{GM}{R^2}\end{equation}
积分得到
\begin{equation}\dot{R}^2=(\frac{dR}{dt})^2=\frac{2GM}{R}+k\label{eq:2}\end{equation}
其中$\dot{R}=V$即为宇宙的膨胀速度，k是积分常数，它的数值决定了宇宙演化方向。这个方程中，我们忽略了压力的影响，因此这个宇宙模型也称为“零压宇宙”。它分为三种情况：

> 当$k = 0$时，$V$恒大于0，宇宙一直膨胀下去，但是速度会趋于0，这个就是我们所说的平直宇宙的情况。
> 
> 
> 
> 当$k > 0$时，$V$恒大于0，宇宙一直膨胀下去，而且速度趋于$k$，这样的宇宙是一个开放的三维双曲面，简称开放宇宙。
> 
> 
> 
> 当$k < 0$是，$V$一开始是正数，但逐渐变小直到变为负数（负数代表宇宙收缩），这样的宇宙是一个闭合的三维球面，简称闭合宇宙。

$\eqref{eq:2}$也是宇宙能量的体现，我们可以把$\eqref{eq:2}$写成
\begin{equation}\frac{1}{2}mV^2-\frac{GMm}{R}=\frac{k}{2}\label{eq:3}\end{equation}
$\eqref{eq:3}$的左端正是引力势能和动能之和，k的正负揭示了宇宙是否受到束缚。我们设$V=HR$，代入$\eqref{eq:3}$，整理得到
\begin{equation}H^2=\frac{2GM}{R^3}+\frac{k}{R^2}=\frac{8}{3}\pi G \rho+\frac{k}{R^2}\label{eq:4}\end{equation}
$\rho$是宇宙平均密度，$H$就是我们所说的哈勃常数，但是它和$\rho$一样，都是时间t的函数。说它是常数，只不过是对于当前的宇宙来说是一个常数，当前的$H$记为$H_0$。空间是整体膨胀的，$V=HR$是宇宙边缘的膨胀速度，不难推出距离宇宙中心$D$处的膨胀速度（星系退行速度）为$V=HD$（你可以想象一下，你拉着一根一端固定的长度为R橡皮筋，速度为$HR$，那么距离定点$D$处的速度为$HD$），这就是“哈勃定律”。宇宙的各向同性告诉我们，宇宙的每一点都是宇宙的中心，所以距离地球D处的星系退行速度速度也是$V=HD$。

[![CMB_Timeline](<https://spaces.ac.cn/usr/uploads/2010/06/3592175358.jpg>)](<https://spaces.ac.cn/attachment/687/>)

CMB\_Timeline

当$k = 0$时候，我们得到了一个介乎“开放”与“闭合”宇宙之间的宇宙模型，这也称为“临界宇宙”，从$\eqref{eq:4}$我们不难求出，当前宇宙临界密度为
\begin{equation}\rho_c=\frac{3H_0^2}{8\pi G}\end{equation}
并且我们可以把这个微分方程解出来，这是最简单的形式
\begin{equation}\begin{aligned}\frac{dR}{dt}=&\sqrt{\frac{2GM}{R}}\\
dt=&\sqrt{\frac{1}{2GM}}R^{0.5}dR\end{aligned}\end{equation}
积分得到
\begin{equation}t=\frac{2}{3}\sqrt{\frac{1}{2GM}}R^{3/2}+C\end{equation}
$C$是积分常数。当$t=0$时，$R=0$，所以推出$C=0$。此时可以把上式写成
\begin{equation}\begin{aligned}R=&(\frac{9GM}{2})^{3/4}t^{3/2}\\
\dot{R}=&\frac{3}{2}(\frac{9GM}{2})^{3/4}t^{1/2}
\end{aligned}\end{equation}
因为$HR=\dot{R}$，所以$\frac{2}{3}H^{-1}=t$。换句话说，在各宇宙模型中，宇宙年龄只有哈勃常数的倒数的三分之二。

对于$k\neq 0$的情况，我们要完成积分$t=\int (\frac{2GM}{R}+k)^{-0.5}dR$，这也是一个常见的积分（可以参考[积分表](<https://spaces.ac.cn/sci/integral/10.html>)），对于$k$的正与负有着不同的结果，列举如下：

> 对于$k < 0$，有
> \begin{equation}t=\frac{\sqrt{\frac{2GM}{R}+k}}{k}R+\frac{2GM\times \arctan\left(\frac{\sqrt{\frac{2GM}{R}+k}}{\sqrt{-k}}\right) }{k\sqrt{-k}}+C\end{equation}
> 初始条件$t=0,R=0$，推出$C=-\frac{GM\pi}{k\sqrt{-k}}$
> 
> 
> 
> 对于$k > 0$，有
> \begin{equation}t=\frac{\sqrt{\frac{2GM}{R}+k}}{k}R-\frac{GM\cdot \ln\left(\frac{\sqrt{\frac{2GM}{R}+k}+\sqrt{k}}{\sqrt{\frac{2GM}{R}+k}-\sqrt{k}}\right)}{k\sqrt{k}}+C\end{equation}
> 同样根据初始条件$t=0,R=0$，推出$C=0$

已经已经到尾声了，我们可以把我们用牛顿力学的结果与广义相对论的结果进行一个对比。用广义相对论处理的结果为
\begin{equation}\frac{\ddot{R}}{R} =-\frac{4 \pi G}{3}(\rho+\frac{3p}{c^2})+\frac{\Lambda c^2}{3}\end{equation}

其中$p$是压力项，$\Lambda$称为宇宙学常数。可以看出，广义相对论比牛顿力学只是多出了两项：压力、宇宙学常数。而宇宙学常数属于人为加上去的一项，牛顿力学也可以这样做。所以，真正不同的是多出了压力项。这是广义相对论带来的实质性改进，这一改进对宇宙演化是至关重要的：压力$p$的出现使我们可以有物态方程，从而描述宇宙物质的真实形态。

用牛顿力学得到的结果，与广义相对论的结果基本相同，这归功于宇宙学原理。宇宙学原理告诉我们，宇宙各个局部的运动状态都是一样的，因此我们可以在一个足够小的局部范围内来研究宇宙膨胀运动。在小范围内，星体的相对运动速度小于光速$c$，所以可以应用牛顿理论。而对于大尺度的问题，如高红移天体、距离、光度等，牛顿力学就不适用了。
