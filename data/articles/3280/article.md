# 高斯型积分的微扰展开（三）

> 作者：苏剑林 · 科学空间 · 2015-04-26
>
> 原文：<https://spaces.ac.cn/archives/3280>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### 换一个小参数

比较[《高斯型积分的微扰展开（一）》](<https://spaces.ac.cn/archives/3217/>)和[《高斯型积分的微扰展开（二）》](<https://spaces.ac.cn/archives/3241/>)两篇文章，我们可以得出关于积分
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx\tag{1}$$
的两个结论：第一，我们发现类似$(4)$式的近似结果具有良好的性质，对任意的$\varepsilon$都能得到一个相对靠谱的近似；第二，我们发现在指数中逐阶展开，得到的级数效果会比直接展开为幂级数的效果要好。那么，两者能不能结合起来呢？

我们将$(4)$式改写成
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx\approx\sqrt{\frac{2\pi}{a+\sqrt{a^2+6 \varepsilon}}}=\sqrt{\frac{\pi}{a+\frac{1}{2}\left(\sqrt{a^2+6 \varepsilon}-a\right)}}\tag{6}$$
其中
$$\lambda=\frac{1}{2}\left(\sqrt{a^2+6 \varepsilon}-a\right)\tag{7}$$
实际上也是一个微小量（当$\varepsilon$很小时），因此，何不以$\lambda$为小参数在指数中展开？事实上，这就相当于让
$$\varepsilon=\frac{2}{3}\left(\lambda^2+a\lambda\right)\tag{8}$$
代入$(1)$，即有
$$\int_{-\infty}^{+\infty} e^{-ax^2-\frac{2}{3}\left(\lambda^2+a\lambda\right) x^4} dx\tag{9}$$
在指数中展开，即
$$\int_{-\infty}^{+\infty} e^{-ax^2-\frac{2}{3}\left(\lambda^2+a\lambda\right) x^4} dx=\int_{-\infty}^{+\infty} e^{-\left(a+a_1 \lambda+a_2 \lambda^2+\dots \right)x^2} dx\tag{10}$$
展开为$\lambda$的幂级数，然后对比各项系数即得
$$\left\{\begin{aligned}&a_1 - 1=0\\
&12 a a_2-9 a_1^2+23=0\\
&-24 a^2 a_3+36 a a_1 a_2-15 a_1^3+245=0\\
&\dots\end{aligned}\right.$$
解得
$$\left\{\begin{aligned}&a_1 = 1\\
&a_2=-\frac{7}{6 a}\\
&a_3=\frac{47}{6 a^2}\\
&\dots\end{aligned}\right.$$
所以我们有
$$\begin{aligned}&\int_{-\infty}^{+\infty} e^{-ax^2-\frac{2}{3}\left(\lambda^2+a\lambda\right) x^4} dx\\
=&\int_{-\infty}^{+\infty} e^{-\left(a+\lambda-\frac{7}{6 a}\lambda^2+\frac{47}{6 a^2}\lambda^3+\dots \right)x^2} dx\\
=&\sqrt{\frac{\pi}{a+\lambda-\frac{7}{6 a}\lambda^2+\frac{47}{6 a^2}\lambda^3+\dots}}\\
=&\sqrt{\frac{\pi}{a+\frac{1}{2}\left(\sqrt{a^2+6 \varepsilon}-a\right)-\frac{7}{24 a}\left(\sqrt{a^2+6 \varepsilon}-a\right)^2+\dots}}\end{aligned}\tag{11}$$
这比$(5)$式的结果又要好很多。当然，它没有改变$a\to 0$时，最终结果发散的情况，同时，它仍然还是一个渐近级数，只不过无穷大的阶次有所改变罢了。

### 更好的参数

回顾上面的过程，我们只不过是对参数做了一个二次的变换：
$$\varepsilon=\frac{2}{3}\left(\lambda^2+a\lambda\right)\tag{8}$$
然而，这两个系数是不是最优的呢？让我们来考虑一般的二次变换
$$\varepsilon=b_2 \lambda^2+b_1\lambda\tag{12}$$
代入$(1)$式，重复上述过程，我们就得到
$$\left\{\begin{aligned}&2 a a_1-3 b_1=0\\
&16 a^3 a_2-12 a^2 \left(a_1^2+2 b_2\right)+105 b_1^2=0\\
&64 a^5 a_3-96 a^4 a_1 a_2+40 a^3 a_1^3+840 a^2 b_1 b_2-3465 b_1^3=0\\
&\dots\end{aligned}\right.$$
解得
$$\left\{\begin{aligned}&a_1 = \frac{3 b_1}{2 a}\\
&a_2=\frac{3 \left(4 a^2 b_2-13 b_1^2\right)}{8 a^3}\\
&a_3=-\frac{3 \left(52 a^2 b_1 b_2-219 b_1^3\right)}{16 a^5}\\
&\dots\end{aligned}\right.$$
取$b_1=2a,b_2=13$，可以得到$a_1=3,a_2=0$，也就是说，如果作变换
$$\varepsilon=13 \lambda^2+2\lambda\tag{13}$$
或
$$\lambda=\frac{\sqrt{a^2+13\varepsilon}-a}{13}$$
那么可以得到
$$\int_{-\infty}^{+\infty} e^{-ax^2-\varepsilon x^4} dx\approx\int_{-\infty}^{+\infty} e^{-(a+3\lambda)x^2}dx=\sqrt{\frac{13\pi}{10a+3\sqrt{a^2+13\varepsilon}}}\tag{14}$$
这是比$(4)$式更好的结果，它精确到二阶，而且具有$(4)$式所具有的好性质。
$$\begin{array}{c|cccc}
\hline
\text{情形} & \varepsilon=1,a=10 & \varepsilon=0.1,a=1 & \varepsilon=1,a=1 & \varepsilon=1,a=0 \\
\hline
\text{精确值} & 0.556466 &1.67409 &1.36843 &1.8128 \\
(4) & 0.556402 &1.66558 & 1.31279 & 1.60159 \\
(14) & 0.556468 & 1.6754 & 1.38715 & 1.94312 \\
\hline \end{array}$$

<strong>其实，整个过程就是一个创意之旅——把你所能够想到的技巧，都应用在展开式上面中去就对了，其上限是你的想象力。</strong>
