# 斯特灵(stirling)公式与渐近级数

> 作者：苏剑林 · 科学空间 · 2016-04-15
>
> 原文：<https://spaces.ac.cn/archives/3731>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

斯特灵近似，或者称斯特灵公式，最开始是作为阶乘的近似提出
$$n!\sim \sqrt{2\pi n}\left(\frac{n}{e}\right)^n$$
符号$\sim$意味着
$$\lim_{n\to\infty}\frac{\sqrt{2\pi n}\left(\frac{n}{e}\right)^n}{n!}=1$$
将斯特灵公式进一步提高精度，就得到所谓的斯特灵级数
$$n!=\sqrt{2\pi n}\left(\frac{n}{e}\right)^n\left(1+\frac{1}{12n}+\frac{1}{288n^2}\dots\right)$$
很遗憾，这个是渐近级数。

相关资料有：
[https://zh\.wikipedia\.org/zh\-cn/斯特灵公式](<https://zh.wikipedia.org/zh-cn/%E6%96%AF%E7%89%B9%E7%81%B5%E5%85%AC%E5%BC%8F>)

[https://en\.wikipedia\.org/wiki/Stirling%27s\_approximation](<https://en.wikipedia.org/wiki/Stirling%27s_approximation>)

本文将会谈到斯特灵公式及其渐近级数的一个改进的推导，并解释渐近级数为什么渐近。

### 斯特灵公式

斯特灵级数有几种推导方法，维基上给出了基于欧拉\-麦克劳林求和公式的推导。然而，它有一些不足的地方。因为它是通过求级数和$n!=\sum_{k=1}^n \ln k$的方法来求的，它不能直接给出常数值$\sqrt{2\pi}$，而且这种思路的结果不能直接推广到伽马函数对应的级数（$\Gamma(x+1)$作为阶乘的推广，具有同样的近似公式，但是基于级数求和的思路不能得出这一点。）。

英文维基还给出了利用拉普拉斯方法的一种推导。我在此基础上做了进一步的推算，能够直接得到伽马函数的渐近级数，在这里跟大家分享一下。这种方法的思路是考虑
$$\Gamma(x+1)=\int_0^{\infty}e^{-t}t^x dt$$
对于任意正整数$n$，都有$\Gamma(n+1)=n!$，因此我们只需要致力于上述积分的估计就行了。设$t=xs$，代进入得到
$$\Gamma(x+1)=x^{x+1}\int_0^{\infty}e^{-x(s-\ln s)} ds$$
到这里，维基上就直接说利用拉普拉斯方法了。但是，一般的读者并不熟悉拉普拉斯方法，而且通过拉普拉斯方法只能得到一个有限的近似，没有办法得到级数解。笔者的改进正是在这里，设$s=e^u$，代入得到
$$\Gamma(x+1)=x^{x+1}\int_{-\infty}^{\infty}e^{-x(e^u-u)+u} du$$
进一步改写成
$$\Gamma(x+1)=x\left(\frac{x}{e}\right)^{x}\int_{-\infty}^{\infty}e^{-xu^2/2-x(e^u-1-u-u^2/2)+u} du$$
事实上，当$x$充分大的时候，积分的主要部分由$\int_{-\infty}^{\infty}e^{-xu^2/2}du=\sqrt{2\pi/x}$贡献，从而直接得到斯特灵公式
$$\Gamma(x+1)\approx \sqrt{2\pi x}\left(\frac{x}{e}\right)^{x}$$

### 斯特灵级数

将被积函数的因子$e^{-x(e^u-1-u-u^2/2)+u}$展开为$u$的级数
$$e^{-x(e^u-1-u-u^2/2)+u}=\sum_{k=0}^{\infty}a_k u^k$$
然后乘上$e^{-xu^2/2}$再进行积分，就可以得到斯特灵级数。为了辩明各项的阶，并且方便计算机推导，需要做一些处理。处理方法参考[《一个非线性差分方程的隐函数解》](<https://spaces.ac.cn/archives/3696/>)，主要还是人工引入参数。

首先，可以很容易得到
$$\int_{-\infty}^{\infty}e^{-xu^2/2}u^kdu=c_k u^{-(k+1)/2}$$
其中$c_k$与$x$无关。可以看成，级数应该以$x^{-1/2}$为阶的，而一个$u$则贡献了一个$x^{-1/2}$，故引入参数$u\to qu$，$x\to x/q^2$（事实上是$x^{-1/2}\to qx^{-1/2}$，因为已经说了，以$x^{-1/2}$为阶），即考虑
$$\int_{-\infty}^{\infty}e^{-xu^2/2-x/q^2\cdot(e^{qu}-1-qu-q^2 u^2/2)+q u} du$$
当$q=1$时正是我们需要的情况。将被积函数展开为$q$的级数，得到
$$e^{-u^2 x}+q e^{-u^2 x} \left(u-\frac{u^3 x}{6}\right)+\frac{1}{72} q^2 u^2 e^{-\frac{1}{2} u^2 x} \left(u^4 x^2-15 u^2 x+36\right)\dots$$
代入$q=1$，然后逐项积分，得到
$$\sqrt{\frac{2 \pi }{x}}\left(1+\frac{1}{12x}+\frac{1}{288x^2}\dots\right)$$
这就得到了斯特灵级数
$$\Gamma(x+1)\approx \sqrt{2\pi x}\left(\frac{x}{e}\right)^{x}\left(1+\frac{1}{12x}+\frac{1}{288x^2}\dots\right)$$

Mathematica代码为

> Integrate&#91;
> Normal&#91;Series&#91;
> Exp&#91;\-x\\cdot u^2/2 \- x/q^2\\cdot (Exp&#91;q\\cdot u&#93; \- 1 \- q\\cdot u \- q^2\\cdot u^2/2) \+
> q\\cdot u&#93;, \{q, 0, 5\}&#93;&#93; /\. q \-\> 1, \{u, \-Infinity, Infinity\},
> Assumptions \-\> x \> 0&#93;/Sqrt&#91;2\\cdot Pi&#93;/Sqrt&#91;x&#93; // Expand

为了加快计算速度，可以在积分之前做一次Expand：

> Integrate&#91;
> Normal&#91;Series&#91;
> Exp&#91;\-x\\cdot u^2/2 \- x/q^2\\cdot (Exp&#91;q\\cdot u&#93; \- 1 \- q\\cdot u \- q^2\\cdot u^2/2) \+
> q\\cdot u&#93;, \{q, 0, 16\}&#93;&#93; /\. q \-\> 1 // Expand, \{u, \-Infinity,
> Infinity\}, Assumptions \-\> x \> 0&#93;/Sqrt&#91;2\\cdot Pi&#93;/Sqrt&#91;x&#93; // Expand

如果需要进一步加速计算，可以自己预先写入积分的计算结果，而不是用内置的积分函数。但是就我们的学习探究而言，此举就没有必要了。

### 渐近级数为何渐近？

如果取定一个$x$，代入到级数中，算无穷多项，结果必然是无穷大！因为这个是渐近级数，收敛半径是0，取前几项可能效果还不错，但是如果真的计算无穷多项的和，结果就会出现发散情况。

为什么会出现渐近级数呢？我们先从一个我们以前讨论过的简单的例子，考虑积分
$$I(\varepsilon)=\int_{-\infty}^{\infty}e^{-x^2-\varepsilon x^4}dx$$
如果将它展开计算：
$$\begin{aligned}&\int_{-\infty}^{\infty}e^{-x^2}\left(1-\varepsilon x^4+\frac{1}{2}\varepsilon^2 x^8\dots\right)dx\\
=&\sqrt{\pi}\left(1-\frac{3}{4}\varepsilon+\frac{105}{32}\varepsilon^2\dots\right)\end{aligned}$$
这也是渐近级数。为什么呢？<strong>这有点像短板效应——复合函数的泰勒级数的收敛半径取决于收敛半径最小的那个函数。我们展开$e^{-\varepsilon x^4}$为$x$的级数，虽然在理论上，这个级数的收敛半径为无穷，但这只是在极限的形式下成立。而对于取有限项的级数，这个展开式并非在整个实数范围内有效，当$x$取得越大的时候，$\varepsilon$的取值范围就越小。而我们的积分是在正负无穷之间进行的，用到了无穷的$x$，这时候$\varepsilon$的取值只能为0。因此级数的渐近的。</strong>

对比之下，如果考虑
$$\int_{-M}^{M}e^{-x^2-\varepsilon x^4}dx$$
的级数展开式，不管$M$多大，得到的总是一个收敛半径为无穷的级数。这就是有限与无限的差别。

那为什么渐近级数又具有一定的精确度？那是因为，对积分$\int_{-\infty}^{\infty}e^{-x^2-\varepsilon x^4}dx$的主要贡献是在$x=0$附近的值，无穷远处的贡献被因子$e^{-x^2}$大大削弱了，因此，在前几项的时候，级数还是有不错的精确度。然而，虽然被削弱，但毕竟还是存在的，当无穷多个弱项叠加起来时，却成为了无穷大～

用同样的理由，可以解释斯特灵级数的渐近性——事实上它只在$x\to\infty$处收敛。
