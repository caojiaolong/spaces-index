# 结果恒为整数的多项式

> 作者：苏剑林 · 科学空间 · 2014-12-04
>
> 原文：<https://spaces.ac.cn/archives/3103>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨晚上初等数论的时候，有这么一道题

> 求证
> $$\frac{1}{3}x^3+\frac{1}{5}x^5+\frac{7}{15}x$$
> 恒为整数，其中$x$是一个整数。

更一般地，可以得到
$$\sum_{p\in\mathbb{P}}\frac{1}{p}x^p + \left(1-\sum_{p\in\mathbb{P}}\frac{1}{p}\right)x$$
恒为整数，其中$\mathbb{P}$是有限个素数的集合，还有更多整数值函数问题。要证明这些函数的值恒为整数，可以通过同余分析，证明分子总能被分母整除。但是，更妙的、同时往往会更简单的方法是，将结果赋予必然为整数的意义——可以是计算上的，也可以是操作上的。

比如证明${n\choose k}=\frac{n !}{(n-k)! k!}$恒为整数（假设目前的情况是只知道等号右边的表达式，不知道它的意义），只要表明它等于从$n$个东西中选择$k$个的组合数就行了，因为组合数必然是整数了。这是赋予函数操作上的意义。另外一个例子是证明$\frac{n(n+1)}{2}$恒为整数，既可以简单证明$n,n+1$之一必为偶数，从而恒为整数；又可以证明$\frac{n(n+1)}{2}=1+2+\dots+n$，右端是一个整数列的前$n$项和，所以其和自然为整数。也就是说，证明多项式值恒为整数的方法之一，就是将它表示为一个整数列的前$n$项和。

现在的问题是，如何求出这个整数列呢？作差分即可！比如$f(x)=\frac{1}{3}x^3+\frac{1}{5}x^5+\frac{7}{15}x$，$f(0)=0$，而
$$f(x)-f(x-1)=x^4-2 x^3+3 x^2-2 x+1$$
等号右端是一个整系数多项式，其值自然是整数，而因为
$$f(x)=\sum_{t=1}^{x}\left(t^4-2 t^3+3 t^2-2 t+1\right)$$
所以$f(x)$恒为整数。当然，该问题使用同余分析也不算复杂，当时，对于下面的题目，如果使用同余分析，就会相当复杂了。

记
$$f(x)=\sum_{p\in\mathbb{P}}\frac{1}{p}x^p + \left(1-\sum_{p\in\mathbb{P}}\frac{1}{p}\right)x$$
只需要注意到
$$\begin{aligned}f(x)-f(x-1)=&\sum_{p\in\mathbb{P}}\frac{1}{p}x^p + \left(1-\sum_{p\in\mathbb{P}}\frac{1}{p}\right)x\\
&-\sum_{p\in\mathbb{P}}\frac{1}{p}(x-1)^p - \left(1-\sum_{p\in\mathbb{P}}\frac{1}{p}\right)(x-1)\\
=&\sum_{p\in\mathbb{P}}\frac{1}{p}\left[ x^p -(x-1)^p \right] + \left(1-\sum_{p\in\mathbb{P}}\frac{1}{p}\right)\\
=&\sum_{p\in\mathbb{P}}\frac{1}{p}\left[\sum_{k=1}^{p-1} {p\choose k} (-1)^{p-k+1} x^k \right] +\sum_{p\in\mathbb{P}}\frac{1}{p}+ \left(1-\sum_{p\in\mathbb{P}}\frac{1}{p}\right)\\
=&\sum_{p\in\mathbb{P}}\left[\sum_{k=1}^{p-1} \frac{1}{p}{p\choose k} (-1)^{p-k+1} x^k \right] +1
\end{aligned}$$
注意，$p$是素数，因此$p\left|{p\choose k}\right.$，所以最后的式子是一个整系数多项式，因此$f(n)$是一个整数列的前$n$项之和，从而$f(x)$恒为整数。
