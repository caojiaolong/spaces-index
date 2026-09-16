# 实数集到无理数集的双射

> 作者：苏剑林 · 科学空间 · 2014-09-22
>
> 原文：<https://spaces.ac.cn/archives/2953>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

集合论的结果告诉我们，全体实数的集合$\mathbb{R}$跟全体无理数的集合$\mathbb{R} \backslash \mathbb{Q}$是等势的，那么，如何构造出它们俩之间的一个双射出来呢？这是一个颇考读者想象力的问题。当然，如果把答案给出来，又似乎显得没有那么神秘。下面给出笔者构造的一个例子，读者可以从中看到这种映射是怎么构造的。

为了构造这样的双射，一个很自然的想法是，让全体有理数和部分无理数在它们自身内相互映射，剩下的无理数则恒等映射。构造这样的一个双射首先得找出一个函数，它的值只会是无理数。要找到这样的函数并不难，比如我们知道：

> 1、方程$x^4 + 1 = y^2$没有除$x=0,y=\pm 1$外的有理点，否则将与费马大定理$n=4$时的结果矛盾。
> 
> 
> 
> 2、无理数的平方根依然是无理数。

根据这些信息，足以构造一个正实数$\mathbb{R}^+$到正无理数$\mathbb{R}^+ \backslash \mathbb{Q}^+$的双射，然后稍微修改一下，就可以得到$\mathbb{R}$到$\mathbb{R} \backslash \mathbb{Q}$的双射。

一个$\mathbb{R}^+$到$\mathbb{R}^+ \backslash \mathbb{Q}^+$的双射如下
$$f(x)=\left\{\begin{aligned}&\sqrt{x^4+1},\, x\in \mathbb{Q}^+\\
&\sqrt{x},\, x\in \mathbb{A}\\
&x,\, x\in \mathbb{R}^+ -\mathbb{Q}^+ -\mathbb{A}\end{aligned}\right.$$

其中集合$\mathbb{A}$是下面的$a_i\,(i \geq 1)$的集合（初始值遍历正有理数）：
$$a_{n+1}=\sqrt{a_n}\ (n\geq 1),\quad a_1=\sqrt{a_0^4+1},\quad a_0\in \mathbb{Q}^+$$
也就是说，从任意一个正有理数$q$出发，从以上表达式构造出来的所有无理数的集合为$A_{q}$，然后让$q$遍历所有正有理数，取所有$A_{q}$之并，就是$\mathbb{A}$，虽然看起来这样的集合的元素个数比有理数集合多得多，但实际上它跟有理数集是等势的。

$f(x)$是一个满射，这几乎是显然的（无理数就这么两类了）。下面只需证明它是一个单射。假如它不是单射，那么“意外”只能出现在集合$\mathbb{A}$里边了，也就是存在正有理数$x,y$以及正整数$m,n$，使得
$$(x^4+1)^{\left(\frac{1}{2}\right)^m}=(y^4+1)^{\left(\frac{1}{2}\right)^n}$$
也就是
$$(x^4+1)^{(2^n)}=(y^4+1)^{(2^m)}$$
不妨设$n\geq m$，那么有
$$(x^4+1)^{(2^{n-m})}=y^4+1$$
如果$n-m$不等于0，那么$a=y,b=(x^4+1)^{(2^{n-m-1})}$就是$a^4+1=b^2$的一组有理数解了，矛盾，因此$n=m$，这样必有$x=y$。因此，$f(x)$是单射。

稍微修改一下，得到实数到无理数的映射
$$f(x)=\left\{\begin{aligned}&\sqrt{x^4+1},\, x\in \mathbb{Q}^+\\
&\sqrt{x},\, x\in \mathbb{A}\\
&-\sqrt{(x-1)^4+1},\, x\in \mathbb{Q}^-\,\,\text{or}\,\, x=0\\
&-\sqrt{-x},\, x\in \mathbb{B}\\
&x,\, x\in \mathbb{R} -\mathbb{Q} -\mathbb{A}-\mathbb{B}\end{aligned}\right.$$
其中集合$\mathbb{A}$是下面的$a_i\,(i \geq 1)$的集合：
$$a_{n+1}=\sqrt{a_n}\ (n\geq 1),\quad a_1=\sqrt{a_0^4+1},\quad a_0\in \mathbb{Q}^+$$
集合$\mathbb{B}$是下面的$b_i\,(i \geq 1)$的集合：
$$b_{n+1}=-\sqrt{-b_n}\ (n\geq 1),\quad b_1=-\sqrt{(b_0-1)^4+1},\quad b_0\in \mathbb{Q}^-\,\text{or}\, b_0=0 $$
