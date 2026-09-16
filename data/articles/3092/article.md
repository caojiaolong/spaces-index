# 力的无穷分解与格林函数法

> 作者：苏剑林 · 科学空间 · 2014-11-24
>
> 原文：<https://spaces.ac.cn/archives/3092>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我小时候一直有个疑问：

> 直升机上的螺旋桨能不能用来挡雨？

一般的螺旋桨是若干个“条状”物通过旋转对称而形成的，也就是说，它并非一个面，按常理来说，它是没办法用来挡雨的。但是，如果在高速旋转的情况下，甚至假设旋转速度可以任意大，那么我们任意时刻都没有办法穿过它了，这种情况下，它似乎与一个实在的面无异？

<strong>力的无穷分解</strong>

[![力的离散化](<https://spaces.ac.cn/usr/uploads/2014/12/1650916168.png>)](<https://spaces.ac.cn/usr/uploads/2014/12/1650916168.png>)

力的离散化

当然，以上只是笔者小时候的一个“异想天开”的念头，读者不必较真。不过，这个疑问跟本文有什么联系呢？我们在研究振动问题之时，通常会遇到在变力的作用下的受迫振动问题，已知变力是时间的函数，比如$f(t)$，然而，虽然知道$f(t)$的具体形式，但是由于$f$的非线性性，加上外力之后的运动，不一定容易求解。然而，如果可以将一个变化的力分段为无数个无穷小时间内的恒力（冲力），那么我们就可以分段讨论我们要研究的运动，而通常来说，恒力的问题会比变力容易。将一个变力离散化，然后再取极限，那么是不是跟原来在变力下的运动是一样的呢？这跟文章开头的疑问有着类似的思想——离线的极限，跟连续本身，是不是等价的？

而让人惊喜的是，在通常的物理系统中，将力分段为无数个小区间内的恒力的做法，能够导致正确的答案，而且，这恰好是线性常微分方程的格林函数法。下面我们来分析这一做法。

<strong>一阶常微分方程</strong>

作为第一个简单的例子，我们来求解
$$\frac{dx(t)}{dt}+x(t)=f(t)$$
这当然是简单的常微分方程，我们有常数变易法可用。现在让我们暂时忘记常数变易法，重新来理解这个方程。当$f(t)\equiv 0$时，方程是容易求解的；而当$f(t)$是一个非零常数时，求解也不难，只要原来的解基础上加上一个常数就行了。但是当$f(t)$是一个一般的函数时，解就复杂起来了。假设我们已经知道$t$时刻的状态$x(t)$，那么我们试着来推导出$t+\varepsilon$时刻的状态$x(t+\varepsilon)$。其中，$\varepsilon$是一个很小的时间间隔，它是如此之小，以至于$f(t)$在这段时间内几乎没有改变，那么$[t,t+\varepsilon]$这段时间内，将外力看成恒定的$f(t)$。这时我们有
$$\frac{d x(t+s)}{ds}+x(t+s)=f(t),\quad 0\leq s\leq \varepsilon$$
该方程的通解为
$$x(t+s)=C e^{-t-s}+f(t)$$
并且在$s=0$时有$x(t)=C e^{-t}+f(t)$，那么$C=[x(t)-f(t)]e^t$，于是我们得到
$$x(t+s)=[x(t)-f(t)]e^{-s}+f(t)$$
从而
$$\begin{aligned}x(t+\varepsilon)=&[x(t)-f(t)]e^{-\varepsilon}+f(t)\\
=&x(t)e^{-\varepsilon}+(1-e^{-\varepsilon})f(t)\\
=&x(t)(1-\varepsilon)+\varepsilon f(t)\end{aligned}$$
当然，这一步是很平凡的，我们可以直接从原来的微分方程得到这一结果，因为我们只在意一阶无穷小，从而
$$\frac{x(t+\varepsilon)-x(t)}{\varepsilon}\approx\frac{dx(t)}{dt}=-x(t)+f(t)$$
简单整理就可以得到$x(t+\varepsilon)=x(t)(1-\varepsilon)+f(t)\varepsilon$，这跟原来的式子等价。但是，通过以上论述，可以赋予我们的做法以直观的物理意义，所以我们愿意通过稍微长一点的步骤来做这个事情。

留意这个解的特点，我们从$t$时刻的状态$x(t)$，得到了$t+\varepsilon$时刻的状态，如此迭代$n$次，我们就能够得到$t+n\varepsilon$时刻的状态了：
$$\begin{aligned}x(t+n\varepsilon)=&x(t+(n-1)\varepsilon)e^{-\varepsilon}+f(t+(n-1)\varepsilon)\varepsilon\\
=&[x(t+(n-2)\varepsilon)e^{-\varepsilon}+f(t+(n-2)\varepsilon)\varepsilon] e^{-\varepsilon}+f(t+(n-1)\varepsilon)\varepsilon\\
=&x(t+(n-2)\varepsilon)e^{-2\varepsilon}+ [e^{-\varepsilon}f(t+(n-2)\varepsilon)+f(t+(n-1)\varepsilon)]\varepsilon\\
=&\dots\\
=&x(t)e^{-n\varepsilon}+\varepsilon\sum_{i=1}^n e^{-(i-1)\varepsilon}f(t+n\varepsilon-i\varepsilon)\end{aligned}$$
固定$n\varepsilon=\tau$，让$n\to\infty$，则
$$\begin{aligned}x(t+\tau)=&x(t)e^{-\tau}+\int_0^\tau e^{-s}f(t+\tau-s)ds\\
=&x(t)e^{-\tau}-\int_t^{t+\tau} e^{-t-\tau+s}f(s)ds\end{aligned}$$
于是我们得到了$t+\tau$时刻的（精确的）解，将任意一个已知的初始状态代入，我们就能得到任意时刻的状态。其中
$$G(t,s)=e^{-t+s}$$
被称为该系统的<strong>格林函数</strong>，借助它我们可以由$x(s)$得到$x(t)$。

<strong>二阶常微分方程</strong>

一阶常微分方程的情况太过平凡了，以至于我们可能会认为这没有价值。下面我们通过同样的思想，给出
$$\frac{d^2 x(t)}{dt}+x(t)=f(t)$$
的格林函数解。与一阶不同的是，由于阶的增加，推导过程变成了矩阵和向量的运算。该方程等价于方程组
$$\frac{d}{dt}\begin{pmatrix}
x(t)\\
x'(t)\end{pmatrix}=\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\begin{pmatrix}
x(t)\\
x'(t)\end{pmatrix}+\begin{pmatrix}
0\\
f(t)\end{pmatrix}$$
也就是
$$\begin{pmatrix}
x(t+\varepsilon)\\x'(t+\varepsilon)\end{pmatrix}=\left[I+\varepsilon\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\right]\begin{pmatrix}
x(t)\\
x'(t)\end{pmatrix}+\varepsilon\begin{pmatrix}
0\\
f(t)\end{pmatrix}$$
这是无穷小间隔的递推公式。要注意我们直接从原方程得到了这一条等式。当然，我们也可以模仿前面的一阶方程的情况，通过更物理的方法得到这一等式，两者的结果基本上是一样的（顶多相差$\varepsilon^2$的项）。但不管我们怎样得到这一条等式，我们都知道它的物理意义就是一阶方程说的那样——将变力分解为无穷多个小时间内的恒力。

将上式迭代$n$次，得到
$$\begin{aligned}\begin{pmatrix}
x(t+n\varepsilon)\\x'(t+n\varepsilon)\end{pmatrix}=&\left[I+\varepsilon\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\right]^n\begin{pmatrix}
x(t)\\
x'(t)\end{pmatrix}\\&+\varepsilon\sum_{i=1}^{n}\left[I+\varepsilon\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\right]^{n-i}\begin{pmatrix}
0\\
f(t+n\varepsilon-i\varepsilon)\end{pmatrix}\end{aligned}$$
同样令$\tau=n\varepsilon$，固定$\tau$，令$n\to\infty$，得到
$$\begin{aligned}\begin{pmatrix}
x(t+\tau)\\x'(t+\tau)\end{pmatrix}=&\exp \left[\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\tau\right]\begin{pmatrix}
x(t)\\
x'(t)\end{pmatrix}\\&+\int_0^{\tau}ds\exp\left[\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}(\tau-s)\right]\begin{pmatrix}
0\\
f(t+\tau-s)\end{pmatrix}\end{aligned}$$
注意到
$$\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}^4=I$$
也就是说，<u>这个矩阵具有类似虚数单位$i$的周期性</u>，也就是说，它在二阶矩阵中的作用，就相当于$i$在复数中的作用，那么根据<strong>欧拉公式</strong>，我们就可以直接写出：
$$\exp \left[\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\tau\right]=I \cos \tau+\begin{pmatrix}
0&1\\
-1&0
\end{pmatrix}\sin\tau$$
代入得
$$\begin{aligned}\begin{pmatrix}
x(t+\tau)\\x'(t+\tau)\end{pmatrix}=&\begin{pmatrix}
x(t)\\
x'(t)\end{pmatrix}\cos\tau+\begin{pmatrix}
x'(t)\\
-x(t)\end{pmatrix}\sin\tau\\&+\int_0^{\tau}ds\begin{pmatrix}
0\\
f(t+\tau-s)\end{pmatrix}\cos\left(\tau-s\right)\\&+\int_0^{\tau}ds\begin{pmatrix}
f(t+\tau-s)\\0
\end{pmatrix}\sin\left(\tau-s\right)\end{aligned}$$
也就是
$$\begin{aligned}x(t+\tau)=&x(t)\cos\tau+x'(t)\sin\tau+\int_0^{\tau}f(t+\tau-s)\sin(\tau-s)ds\\
=&x(t)\cos\tau+x'(t)\sin\tau-\int_t^{t+\tau}f(s)\sin(s-t)ds\end{aligned}$$
以及
$$x'(t+\tau)=x'(t)\cos\tau-x(t)\sin\tau-\int_t^{t+\tau}f(s)\cos(s-t)ds$$

<strong>费恩曼物理讲义</strong>

可能会让读者意外的是，上述思想，来自于费曼的《费恩曼物理讲义》！它们在第一卷的§25\-2节的后半部分。

不知道其他教程有没有阐述这一思想，反正笔者是最近从费曼的了解到的，反正就有眼前一亮、醍醐灌顶的感觉！如此清晰的物理意义，怎么可以不为数学、物理爱好者所知道呢！之前我读《常微分方程》之类的教程时，遇到格林函数法，我是直接跳过的，因为常微分方程的格林函数，似乎还没有偏微分方程的格林函数那么意义明显，我勉强能记住它，却不知道它的意义。现在，经过费曼的“指导”，我就清晰地理解了常微分方程的格林函数！注意，这样清晰的理解，是在一本物理讲义中学到的！

不能不说，费曼本人以及费曼的著作，不管怎么读都不为过！
