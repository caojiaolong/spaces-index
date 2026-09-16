# 高斯型积分的微扰展开（一）

> 作者：苏剑林 · 科学空间 · 2015-02-14
>
> 原文：<https://spaces.ac.cn/archives/3217>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前段时间在研究费曼的路径积分理论，看到路径积分的微扰方法，也就是通过小参数展开的方式逐步逼近传播子。这样的技巧具有非常清晰的物理意义，有兴趣了解路径积分以及量子力学的读者，请去阅读费曼的《量子力学与路径积分》。然而从数学角度看来，这种逼近的技巧实际上非常粗糙，收敛范围和速度难以得到保证。事实上，数学上发展了各种各样的摄动技巧，来应对不同情况的微扰。下面我们研究积分
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx\tag{1}$$
或者更一般地
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon V(x)} dx\tag{2}$$
路径积分的级数展开比它稍微复杂一些，但是仍然是类似的形式。

### 简单直接的展开

为了计算上述积分，最简单直接的方法，就是把$e^{-\varepsilon V(x)}$作级数展开
$$e^{-\varepsilon V(x)}=\sum_{n=0}^{\infty} (-1)^n \frac{1}{n!}\varepsilon^n V^n (x)$$
从而
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon V(x)} dx=\sum_{n=0}^{\infty} (-1)^n\frac{1}{n!}\varepsilon^n \int_{-\infty}^{+\infty} e^{-ax^2} V^n (x)dx$$
如果$e^{-ax^2}V^n (x)$的定积分容易计算，那么我们就可以得到原积分的一个逼近，比如对于$V(x)=x^4$的情况，我们有
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx=\sqrt{\frac{\pi}{a}}\left(1-\frac{3\varepsilon}{4a^2}+\frac{105\varepsilon^2}{32a^4}-\frac{3465\varepsilon^3}{128a^6}+\dots\right)\tag{3}$$
该级数在$\varepsilon=a^2$的时候便完全失效了，而且就算$\varepsilon < a^2$，收敛速度也是很慢的。事实上，这是一个渐近级数，严格来讲，它的收敛区域就是$\varepsilon=0$的时候！当然，我们可以取它的前几项，来对比较大的（相对于0来说）$\varepsilon$进行近似计算。渐近级数的“渐近”就是说，取的项越少，有效计算的半径越大，能计算，但精度不一定高；取的项越多，自然越精确，但是有效半径越小。取无穷多项时，收敛半径就只有0了。

### 在指数中展开：一个封闭的解

在研究微分方程摄动展开的时候，为了得到周期解，我们不仅仅会把解关于小参数展开，而且还会把周期作关于小参数的展开。由此，笔者受到启发：能不能将原来的系数作参数展开呢？具体来讲，就是我们引入$A$
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx=\int_{-\infty}^{+\infty} e^{-Ax^2-[(a-A)x^2+\varepsilon x^4]} dx$$
现在把方括号中的$(a-A)x^2+\varepsilon x^2$当作$V(x)$作一阶展开，得到
$$\int_{-\infty}^{+\infty} e^{-Ax^2} \left\{1-[(a-A)x^2+\varepsilon x^4]\right\}dx$$
计算
$$\int_{-\infty}^{+\infty} e^{-Ax^2}[(a-A)x^2+\varepsilon x^4]dx=\sqrt{\frac{\pi}{A}}\frac{2 A (a-A)+3 \varepsilon}{4 A^2}$$
接下来是关键的一步，选取适当的$A$，让这一项为$0$，也就是
$$2 A (a-A)+3 \varepsilon=0\,\to\, A=\frac{1}{2} \left(a+\sqrt{a^2+6 \varepsilon}\right)$$
所以
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx\approx\int_{-\infty}^{+\infty} e^{-Ax^2} =\sqrt{\frac{\pi}{A}}=\sqrt{\frac{2\pi}{a+\sqrt{a^2+6 \varepsilon}}}\tag{4}$$
这是一个非常好的近似。有多好？好到难以置信！考虑最极端的情况：$a=0$，上式给出
$$\int_{-\infty}^{+\infty} e^{-\varepsilon x^4} dx\approx\sqrt{\frac{2\pi}{\sqrt{6 \varepsilon}}}$$
上式不仅给出了积分的一个近似值，而且就连尺度都是正确的，也就是说$\int_{-\infty}^{+\infty} e^{-\varepsilon x^4} dx$的精确值，就是一个常数乘上$\frac{1}{\varepsilon^{1/4}}$的形式！对于不同的$a,\varepsilon$组合，算出来的值比$(3)$式的一阶近似要好（有时甚至比它的高阶近似都要好）。另外一个极端情况是$\varepsilon\to+\infty$，此时积分的精确值为0，而$(4)$居然也可以给出这一结果，而$(3)$则再次完全失效了。

下表是精确值、取$(3)$式的一阶近似$\sqrt{\frac{\pi}{a}}\left(1-\frac{3\varepsilon}{4a^2}\right)$、$(4)$式的计算结果。从表中就可以看出，公式$(4)$的在所有场合都优于公式$(3)$的一阶近似。
$$\begin{array}{c|cccc}
\hline
\text{情形} & \varepsilon=1,a=10 & \varepsilon=0.1,a=1 & \varepsilon=1,a=1 & \varepsilon=1,a=0 \\
\hline
\text{精确值} & 0.556466 &1.67409 &1.36843 &1.8128 \\
\sqrt{\frac{\pi}{a}}\left(1-\frac{3\varepsilon}{4a^2}\right) & 0.556295 &1.63952 & 0.443113 & \text{负无穷} \\
\sqrt{\frac{2\pi}{a+\sqrt{a^2+6 \varepsilon}}} & 0.556402 &1.66558 & 1.31279 & 1.60159 \\
\hline
\end{array}$$
虽然$(4)$式的稳定性和精确度都较优，但它本质上仍然是一阶的近似式，后面我们会继续谈及它的高阶近似。
