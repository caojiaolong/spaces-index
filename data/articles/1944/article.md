# 费曼积分法(6)：教科书上的两道练习题

> 作者：苏剑林 · 科学空间 · 2013-03-24
>
> 原文：<https://spaces.ac.cn/archives/1944>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们的《数学分析》教程上有两道比较有趣的定积分，经测试可以用费曼积分法的思路解决。

$$\begin{aligned}\int_0^1 \frac{\ln(1+x)}{1+x^2}dx \\ \int_0^{\pi} \frac{x \sin x}{1+\cos^2 x}dx\end{aligned}$$

<strong>No\.1</strong>

首先考虑第一道，其实这道题的思路是比较明显的，因为对数函数求导之后就变成了多项式分式的形式，这种形式的积分是有统一的处理方法的，于是乎：

$$\begin{aligned}F(a)=\int_0^1 \frac{\ln(1+ax)}{1+x^2}dx \\ F'(a)=\int_0^1 \frac{x}{(1+ax)(1+x^2)}dx=\int_0^1 \frac{1}{1+a^2}(\frac{-a}{1+ax}+\frac{x+a}{1+x^2})dx \\ =-\frac{\ln(1+a)}{1+a^2}+\frac{\ln 2}{2}(\frac{1}{1+a^2})+\frac{\pi}{4} \frac{a}{1+a^2}\end{aligned}$$

下面是对a积分，并且易知$F(0)=0$，因此
$$F(a)=\int_0^a -\frac{\ln(1+t)}{1+t^2}dt + \frac{\ln 2}{2} arctan(a)+\frac{\pi}{8}\ln(1+a)$$

所以
$$\begin{aligned}F(1)=\int_0^1 -\frac{\ln(1+t)}{1+t^2}dt + \frac{\pi}{4}\ln 2 \\ =-F(1)+ \frac{\pi}{4}\ln 2\end{aligned}$$

所以$F(1)=\frac{\pi}{8}\ln 2$

<strong>No\.2</strong>

第二道积分中，作者（作者就是我们的数学分析老师）给出了一个巧妙的解法，该解法高度利用了三角函数的对称性，值得学习和模仿。原证明将在最后给出。在此先探讨用费曼积分法求解的过程。

我们有：
$$\int_0^{\pi} \frac{x \sin x}{1+\cos^2 x}dx=-\int_0^{\pi} \frac{x}{1+\cos^2 x}d(\cos x)$$

<u>那单独的一项x在求导之后也变成关于$\cos x$的函数就好了</u>，这就相当于普通的多项式积分了。那好办，让$x=\arccos(a \cos x)$，即

$$F(a)=-\int_0^{\pi} \frac{\arccos(a \cos x)}{1+\cos^2 x}d(\cos x)$$

<strong><u>不对不对</u></strong>，x是遍历$[0,\pi]$的，但是函数\\arccos(x)的值域是$[-\frac{\pi}{2},\frac{\pi}{2}]$，显然是不对应的，那么只得做一点修改：
$$F(a)=-\int_0^{\frac{\pi}{2}} \frac{\arccos(a \cos x)}{1+\cos^2 x}d(\cos x)-\int_{\frac{\pi}{2}}^{\pi} \frac{\arccos(a \cos x)+\frac{\pi}{2}}{1+\cos^2 x}d(\cos x)$$

即
$$F(a)=-\int_0^{\pi} \frac{\arccos(a \cos x)}{1+\cos^2 x}d(\cos x)- \frac{\pi}{2}\int_{\frac{\pi}{2}}^{\pi}\frac{1}{1+\cos^2 x}d(\cos x)$$

其中等式右端的第二部分是容易积分出来的，即$-\frac{\pi^2}{4}$，关键是左端积分：
$$f(a)=-\int_0^{\pi} \frac{\arccos(a \cos x)}{1+\cos^2 x}d(\cos x)$$

求导后变成了
$$\begin{aligned}f'(a)=-\int_0^{\pi} \frac{\cos x}{\sqrt{1-a^2 \cos^2 x}(1+\cos^2 x)}d(\cos x) \\ =-\frac{1}{2} \int_0^{\pi} \frac{1}{\sqrt{1-a^2 \cos^2 x}(1+\cos^2 x)}d(\cos^2 x)\end{aligned}$$
设$t=\cos^2 x$，则变成了：
$$f'(a)=-\frac{1}{2}\int_1^1 \frac{1}{\sqrt{1-a^2 t}(1+t)}dt$$
<strong>由于积分上下限相等，显然积分值为0！！！</strong>所以$f(a)$是一个与a无关的常数，并且根据
$$f(o)=-\frac{\pi}{2}\int_0^{\pi} \frac{1}{1+\cos^2 x}d(\cos x)=\frac{\pi^2}{2}$$

得出$f(a)=\frac{\pi^2}{2}$。

所以$F(1)=F(a)=f(a)-\frac{\pi^2}{4}=\frac{\pi^2}{4}$

妙吧？对称性发挥作用的地方，费曼积分法也能够占有一席之地！

<strong>附：书本上的解法</strong>
$$I=\int_0^{\pi} \frac{x \sin x}{1+\cos^2 x}dx$$
令$x=\pi-t$，则
$$\begin{aligned}I=\int_{\pi}^0 \frac{(\pi-t) \sin (\pi-t) }{1+\cos^2 (\pi-t)}d(\pi-t) \\ =\int_0^{\pi} \frac{\pi \sin t }{1+\cos^2 t}dt-\int_0^{\pi} \frac{t \sin t }{1+\cos^2 t}dt \\ =\frac{\pi^2}{2}-I\end{aligned}$$

所以$I=\frac{\pi^2}{4}$\.
