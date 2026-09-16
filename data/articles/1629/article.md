# 费曼积分法——积分符号内取微分(3)

> 作者：苏剑林 · 科学空间 · 2012-06-23
>
> 原文：<https://spaces.ac.cn/archives/1629>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

由于自行车之旅的原因，这篇文章被搁置了一个星期，其实应该在一个星期前就把它写好的。这篇文章继续讲讲费曼积分法的一些例子。读者或许可以从这些不同类型的例子中，发现它应用的基本方向和方法，从而提升对它的认识。

<strong>例子2：</strong>

$$\int_0^{\infty} \frac{\sin x}{x}dx$$

这也是一种比较常见的类型，它的形式为$\int \frac{f(x)}{x}dx$，对于这种形式，我们的第一感觉就是将其改写成参数形式$\int \frac{f(ax)}{x}dx$，这样的目的很简单，就是把分母给消去了，与$\int \frac{x}{f(x)}dx$的求积思想是一致的。但是深入一点研究就会发现，纵使这样能够消去分母，使得第一次积分变得简单，但是到了第二次积分的时候，我们发现，它又会变回$\int \frac{f(x)}{x}dx$的积分，使我们不能继续进行下去，因此这个取参数的方法大多数情况下都是不行的。

有一个神奇的方法，让我们将其变换成：

$$\begin{aligned}G(a)=\int_0^{\infty} e^{-ax}\frac{\sin x}{x}dx \\ f(x,a)=e^{-ax}\frac{\sin x}{x}\end{aligned}$$

这样我们在原来的基础上增加了一块$e^{-ax}$，其作用也是让我们把分母给消去了，因为

$$\frac{\partial f(x,a)}{\partial a}=-e^{-ax} \sin x$$

所以
$$\begin{aligned}G'(a)=\int_0^{\infty} -e^{-ax} \sin x dx \\ =\frac{1}{a^2+1} e^{-ax}(a \sin x+\cos x)|_0^{\infty} \\ =-\frac{1}{a^2+1}\end{aligned}$$
（[指数函数《积分表》](<https://spaces.ac.cn/sci/integral/13.html>)）

最后得到
$$G(a)=-\int \frac{1}{a^2+1} da=-arctan a +C$$

当$a\to \infty$时，$f(x,a)=0$，$G(a)=0$，得到$C=\frac{\pi}{2}$。最终的结果是

$$\int_0^{\infty} e^{-ax}\frac{\sin x}{x}dx=-arctan a+\frac{\pi}{2}$$

所以
$$\int_0^{\infty} \frac{\sin x}{x}dx=G(0)=\frac{\pi}{2}$$

<strong>例子3：</strong>

在费曼所看的《高等微积分》中，有一个很典型的例子，它的求解过程结合了微分方程的知识。

已知$\int_0^{\infty} e^{-x^2}dx=\frac{\sqrt{\pi}}{2}$，求积分：$u=\int_0^{\infty} e^{-x^2-\frac{a^2}{x^2}}dx$

这个积分已经给出了参数a，我们不妨直接对该参数求导，看看会发生什么？

$$\begin{aligned}\frac{du}{da}=\int_0^{\infty} \frac{\partial (e^{-x^2-\frac{a^2}{x^2}})}{\partial a}dx \\ =2\int_0^{\infty} e^{-x^2-\frac{a^2}{x^2}}d(\frac{a}{x})\end{aligned}$$

令$t=\frac{a}{x}$，则变成了
$$\int_0^{\infty} e^{-x^2-\frac{a^2}{x^2}}d(\frac{a}{x})=-\int_0^{\infty} e^{-t^2-\frac{a^2}{t^2}}dt$$

这个积分的形式和所求积分的形式完全一样，因为变量的符号（x,t）只是一个记号，积分本身和变量的符号无关，因此我们有把握地说
$$\int_0^{\infty} e^{-t^2-\frac{a^2}{t^2}}dt=u$$

总的来说：$\frac{du}{da}=-2u$

解得：$u=C \times e^{-2a}$，利用给出的已知条件，可以确定常数$C=\frac{\sqrt{\pi}}{2}$，所以

$$\int_0^{\infty} e^{-x^2-\frac{a^2}{x^2}}dx=\frac{\sqrt{\pi}}{2} \times e^{-2a}$$

<strong>小结</strong>

费曼积分法的变换形式多种多样，捉摸不定，需要在实际应用中敢想、敢试，而且时常需要“灵光一现”，才能妙笔生花！也许正是它的这种灵活性，因此通常有意想不到的惊喜等着我们，或者正是这个原因，费曼对它情有独钟。
