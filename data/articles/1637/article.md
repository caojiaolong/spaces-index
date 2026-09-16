# 费曼积分法——积分符号内取微分(4)

> 作者：苏剑林 · 科学空间 · 2012-06-26
>
> 原文：<https://spaces.ac.cn/archives/1637>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

趁着早上有空，就赶紧把这篇文章写好吧。下午高考成绩要公布了，公布后也许又会有一段时间忙碌了。这应该是“费曼积分法”系列最后一篇文章了。它主要讲的还是费曼积分法的一个实例。不同的是，这是BoJone首次独立地用费曼积分法解决了一个问题。之前提到的一些例子，都是书本提供并结合了提示，BoJone才把它们算出来的。所以这个问题有着点点纪念意义。

数学研发论坛上wayne曾求证这样的命题：

> $\int_0^{\infty}\frac{f(x,2m-1)-\sin x}{x^{2m+1}}dx$其中，f(x,2m\-1)表示sinx的2m\-1阶泰勒展开
> 如m=1时，
> $$\int_0^{\infty}\frac{x-\sin x}{x^3}dx$$
> m=2时
> $$\int_0^{\infty}\frac{x-\frac{x^3}{6}-\sin x}{x^5}dx$$
> 借助软件我发现结果是：
> $\frac{\pi(-1)^{m-1}}{2(2m)!}$

有人利用拉普拉斯变换给出了证明，而我对拉普拉斯变换只是有概念，不懂具体方法。我利用费曼积分法，将问题一般化。假设有函数y=f(x)，求积分

$$F(t)=\int_a^b \frac{f(0)+f'(0)(tx)+f''(0)\frac{(tx)^2}{2}+...+f^{(n)}(0)\frac{(tx)^n}{n!}-f(tx)}{x^{n+1}} dx$$

我们有
$$\begin{aligned}\frac{d F(t)}{dt}=\int_a^b \frac{f'(0)+f''(0)(tx)+...+f^{(n)}(0)\frac{(tx)^{n-1}}{(n-1)!}-f'(tx)}{x^n} dx\end{aligned}$$

连续n次微分得到
$$\begin{aligned}\frac{d^n F(t)}{dt^n}=\int_a^b \frac{f^{(n)}(0)-f^{(n)}(tx)}{x} dx\end{aligned}$$

对于某些情况，可以n\+1次微分得到
$$\begin{aligned}\frac{d^{n+1} F(t)}{dt^{n+1}}=\int_a^b -f^{(n+1)}(tx) dx=-f^{(n)} (tx)|_a^b\end{aligned}$$

然后对变量t积分n次或n\+1次即可，这个过程会得出多个积分常数。我们知道当t=0时原积分值为0，可以确定每一个积分常数都为0。

对于wayne的题目，f(x)=\\sin(x)，n=2m，$f^{(2m)}(x)=(-1)^m sin x$

$$\begin{aligned}\frac{d^{2m} F(t)}{dt^{2m}}=\int_0^{\infty} \frac{-(-1)^m \sin (tx)}{x} dx\end{aligned}$$

（这里不能够再微分了，在微分会得到$-(-1)^m sin (tx)|_0^{\infty}$，这没有意义）

根据$\int_0^{\infty} \frac{sinx}{x}dx=\frac{\pi}{2}$

我们有
$$\frac{d^{2m} F(t)}{dt^{2m}}=-(-1)^m\frac{\pi}{2}=(-1)^{m-1} \frac{\pi}{2}$$

2m次积分后：$F(t)=\frac{(-1)^{m-1} \pi \cdot t^{2m}}{2(2m)!}$

取t=1即可。

于是，一个有趣的定积分被费曼积分法漂亮地解决了！

<strong>附：拉普拉斯变换的证明</strong>

[![拉普拉斯变换证明](<https://spaces.ac.cn/usr/uploads/2012/06/3082520371.png>)](<https://spaces.ac.cn/attachment/1638/>)

拉普拉斯变换证明

<strong>结语</strong>

<u>在一般的微积分教材基本已经不讲“积分符号内取微分”这个工具了，应对那些复杂一点的积分，基本会用到复变函数的工具（即之前提到的“围道积分”），或者输入计算机，直接了事。但是，对于这个相当有趣的技巧，BoJone还是想把它介绍给读者，让大家知道有这么一个东西的存在，也有这么一个东西的历史。毕竟，“小飞侠”费曼用它解决了不少难题，难道我们就不想一试？</u>
