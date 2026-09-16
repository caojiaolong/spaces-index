# 柯西命题：盯着它到显然成立为止！

> 作者：苏剑林 · 科学空间 · 2015-04-19
>
> 原文：<https://spaces.ac.cn/archives/3272>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

数学分析中数列极限部分，有一个很基本的“柯西命题”：

> 如果$\lim_{n\to\infty} x_n=a$，则
> $$\lim_{n\to\infty}\frac{x_1+x_2+\dots+x_n}{n}=a$$

本文所要谈的便是这个命题，当然还包括类似的一些题目。

### 柯西命题的证明

柯西命题的证明并不难，只需要根据极限收敛的定义，由于$\lim_{n\to\infty} x_n=a$，所以任意给定$\varepsilon > 0$，存在足够大的$N$，使得对于任意的$n > N$，都有
$$\left|x_n - a\right| < \varepsilon/2\quad(\forall n > N)$$

那么，对于足够大的$n$，就有
$$\begin{aligned}&\left|\frac{x_1+x_2+\dots+x_n}{n}-a\right|\\
=&\left|\frac{(x_1-a)+(x_2-a)+\dots+(x_n-a)}{n}\right|\\
\leq &\left|\frac{(x_1-a)+(x_2-a)+\dots+(x_N-a)}{n}\right|\\
&\quad+\left|\frac{(x_{N+1}-a)}{n}\right|+\left|\frac{(x_{N+2}-a)}{n}\right|+\dots+\left|\frac{(x_{n}-a)}{n}\right|\\
< & \left|\frac{(x_1-a)+(x_2-a)+\dots+(x_N-a)}{n}\right|+\frac{n-N}{n}\frac{\varepsilon}{2}\\
< & \left|\frac{(x_1-a)+(x_2-a)+\dots+(x_N-a)}{n}\right|+\frac{\varepsilon}{2}\\
\end{aligned}$$
取足够大的$M > N >0$，只要$n > M$，就有
$$\left|\frac{(x_1-a)+(x_2-a)+\dots+(x_N-a)}{n}\right| < \frac{\varepsilon}{2}$$
从而
$$\left|\frac{x_1+x_2+\dots+x_n}{n}-a\right| < \varepsilon$$
由$\varepsilon$的任意性知
$$\lim_{n\to\infty}\frac{x_1+x_2+\dots+x_n}{n}=a$$

其实柯西命题的直观意义也很明显，它只不过告诉我们，如果一个数列是收敛的，那么它是越来越平缓的，取平均的话，就算一开始起伏很大，也可以用足够大的$n$来抹平它。

### 一个变形

下面请读者来做一下这道题目。

> 已知$\lim_{n\to\infty} x_n=a$，求证
> $$\lim_{n\to\infty}\frac{x_1+2 x_2+\dots+n x_n}{n^2}=\frac{1}{2}a$$

请读者别那么着急动笔，盯着它，死盯着它，直到题目害怕你了而把自己把答案交出来！！

========华丽的分割线========

事实上，有了柯西命题，这道题目根本就不用证，它就是显然成立的！为什么？我们干嘛要去考虑数列$\{x_n\}$呢，我们为什么不考虑下面的数列呢？
$$\{y_n\}={x_1, x_2, x_2, x_3, x_3, x_3, x_4, x_4, x_4, x_4, x_5,\dots}$$
显然也有
$$\lim_{n\to\infty} y_n =a$$
而且
$$\frac{1}{n^2}\sum_{k=1}^n k a_k=\frac{1}{n^2}\sum_{k=1}^{\frac{1}{2}n(n+1)} y_n=\frac{\frac{1}{2}n(n+1)}{n^2}\frac{\sum_{k=1}^{\frac{1}{2}n(n+1)}y_n}{\frac{1}{2}n(n+1)}$$
由柯西命题立马知道最右边求和项的极限为$a$，而$\frac{\frac{1}{2}n(n+1)}{n^2}$的极限显然为$\frac{1}{2}$，所以
$$\lim_{n\to\infty}\frac{1}{n^2}\sum_{k=1}^n k a_k=\lim_{n\to\infty}\frac{1}{n^2}\sum_{k=1}^{\frac{1}{2}n(n+1)} y_n=\frac{1}{2}a$$
所以说，凡事换个角度看看，就可能看到意想不到的东西，有时候只要一直盯着它，就觉得显然成立了。我想，能够从不同角度认识同一个问题，才算是把问题看透吧。
