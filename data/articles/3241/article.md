# 高斯型积分的微扰展开（二）

> 作者：苏剑林 · 科学空间 · 2015-03-07
>
> 原文：<https://spaces.ac.cn/archives/3241>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### 为什么第二篇姗姗来迟？

其实要写这系列之前，我已经构思好了接下来几篇的内容，本来想要自信地介绍自己想到的一些积分展开的技巧；而且摄动法我本身就比较熟悉，所以正常来说不会这么迟才有第二篇。然而，在我写完第一篇，准备写第二篇的期间，我看到了知乎上的这篇回复：
[http://www\.zhihu\.com/question/24735673](<http://www.zhihu.com/question/24735673>)

这篇文章大大地拓展了我对级数的认识。里边谈及到了积分的展开是一个渐近级数。这让我犹豫了，怀疑这系列有没有价值，因为渐近级数意味着不管怎样的展开技巧，得到的级数收敛半径都是0。

后来再想想，就算是渐近级数，也有改进的空间，有加速收敛的方法，所以我想我这几篇文章，应该还有一点点意义吧，还可以顺便介绍一下渐近级数和奇点的相关理论。嗯，就这么办吧。

### 在指数中逐阶展开

前一篇文章已经提到的技巧就是对常系数进行变易，通过调整变量使得一阶项为0，该方法主要问题就是不能逐阶地求出高阶近似。然而，它的思想是很不错的，即引入可调整的变量，让展开的各阶项为0，而不要纯粹地做幂级数展开。下面，我们先来试着把展开式放到指数中去，得到一个可以逐级求近似的方法，作为我们以后要进一步发展的方法的原始模型。

总的来说，我们要求积分
$$\int_{-\infty}^{+\infty} e^{-a x^2-\varepsilon x^4} dx$$

上一篇文章中，我们希望用积分$\int_{-\infty}^{+\infty} e^{-A x^2} dx$来逼近它，逼近的思路是通过调整$A$来使得一阶项为0，但这样子我们只能做到这里的，因为只有一个可调整变量，一般来讲只能让一项为0。为了实现逐阶展开，即引入多个可调整的量，我们设
$$A=a+a_1 \varepsilon + a_2\varepsilon^2 + a_3 \varepsilon^3 + \dots$$
然后考虑
$$\int_{-\infty}^{+\infty} \left[e^{-a x^2-\varepsilon x^4}-e^{-(a+a_1 \varepsilon + a_2\varepsilon^2 + a_3 \varepsilon^3 + \dots)x^2}\right] dx$$
将括号内的被积函数按$\varepsilon$的阶展开，就有
$$\begin{aligned}x^2 e^{-a x^2} \left(a_1-x^2\right)\varepsilon &+ \frac{1}{2} x^2 e^{-a x^2} \left(-a_1^2 x^2+2 a_2+x^6\right)\varepsilon^2 \\&- \frac{1}{6} x^2 e^{-a x^2} \left(-a_1^3 x^4+6 a_1 a_2 x^2-6 a_3+x^{10}\right)\varepsilon+\dots\end{aligned}$$
逐项积分，就有
$$\begin{aligned}\sqrt{\frac{\pi}{a} }&\left[\frac{1}{4}(2 a a_1 -3)\left(\frac{\varepsilon}{a^2} \right)+\frac{1}{32}\left(4 a^2 \left(4 a a_2-3 a_1^2\right)+105\right)\left(\frac{\varepsilon}{a^2} \right)^2\right.\\
&\left.+\frac{1}{128}\left(3465-8 a^3 \left(8 a^2 a_3-12 a a_1 a_2+5 a_1^3\right)\right)\left(\frac{\varepsilon}{a^2} \right)^3+\dots\right]\end{aligned}$$
让每项为0，得到方程组
$$\left\{\begin{aligned}&2 a a_1 -3=0\\
&4 a^2 \left(4 a a_2-3 a_1^2\right)+105=0\\
&3465-8 a^3 \left(8 a^2 a_3-12 a a_1 a_2+5 a_1^3\right)=0\\
&\dots\end{aligned}\right.$$
解得
$$a_1=\frac{3}{2 a},\,a_2=-\frac{39}{8 a^3},\,a_3=\frac{657}{16 a^5},\,\dots$$
所以我们有级数
$$\begin{aligned}&\int_{-\infty}^{+\infty} e^{-a x^2-\varepsilon x^4} dx \\
=&\int_{-\infty}^{+\infty} e^{-\left(a+\frac{3}{2}\frac{\varepsilon}{a}-\frac{39}{8}\frac{\varepsilon^2}{a^3}+\frac{657}{16}\frac{\varepsilon^3}{a^5}\dots\right)x^2} dx\\=&\sqrt{\frac{\pi}{a+\frac{3}{2}\frac{\varepsilon}{a}-\frac{39}{8}\frac{\varepsilon^2}{a^3}+\frac{657}{16}\frac{\varepsilon^3}{a^5}\dots}}\end{aligned}\tag{5}$$
这是关于原积分的有一个渐近表达式。$(5)$式效果比$(3)$式好一些；相比于$(4)$式，当$\varepsilon/a^2$很小的时候，截断$(5)$的前几项进行计算，能够得到比$(4)$式更多位有效数字。当然，它在$a\to 0$或者$\varepsilon\to \infty$时则失效了——不过就算是失效，它也只是得到一个有限的0，而不是其它不确定的结果。这也是这种展开方式的好处之一。

### 渐近级数的意义

已经说过，取无穷多项时，渐近级数的收敛半径实际是0，那么渐近级数还有什么意义呢？事实上，首先，对于近似计算来说，可以截断渐近级数的前几项，从而对于某个范围的自变量得到一个比较不错的近似值。其次，对于某些渐近级数，我们可以通过再求和的技巧，来还原它的原始级数，从而得到精确解——量子场论中有大量的例子！

所以，我们需要渐近级数，同时，还可以考虑改进渐近级数的方式（即降低发散速度）。有各种各样的技巧可以做这样的事情，只要你有足够多的创意。在下一篇文章中，我们会再展示一次。
