# 微积分学习（一）：极限

> 作者：苏剑林 · 科学空间 · 2009-08-16
>
> 原文：<https://spaces.ac.cn/archives/75>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>本文不是微积分教程，而是发表自己学习中的一些看法，以及与同好们讨论相关问题。</strong>

拿起任何一本“微积分”教程，都可以看见那专业而严格的数学语言，因此很多人望而生畏。的确，由于牛顿和莱布尼茨创立的微积分是不严格的，因此引发了第二次数学危机。经过法国数学家柯西和德国数学家魏尔斯特拉斯的努力，使得微积分有了前所未有的严密化，克服了第二次数学危机。加之后来的第三次数学危机，数学就更加严密了。

但是对于初学者，严密化的微积分令人十分费解。因此，我们不妨按照微积分的创立顺序，即“不严密——严密”的顺序来学习。这样不仅能够让我们更高效率地学习，而且增加学习数学的兴趣。

一般来讲，微积分的教程的顺序为：集合、映射，函数、极限、导数、微分、积分、\.\.\.一直深入下去的。其中“集合、映射，函数”这三个部分只是为了严密地给出定义，真正的求解是从极限开始的（当然前面的基础也不可忽略）。

极限的意义为：当函数$f(x)$中，x以无限接近的程度去逼近预先给出的$x_0$（或者是$\infty$，<strong>这里是“无限接近”，并不等于，是克服数学危机的主要步骤</strong>），求得$f(x)$的值。若$f(x)$趋向一个确定的值，则称极限存在；反之则极限不存在。

注：当$x_0=\infty$或者$x_0=0$的时候，有正负之分。即$x_0=+\infty,x_0=-\infty$或者$x_0=+0,x_0=-0$，极限存在的条件，除了要趋向一个稳定的值外，对于正负的极限也必须相等，比如$\lim_{x \to +\infty} f(x)=\lim_{x \to -\infty} f(x)$，这样才能够称为极限存在。若$\lim_{x \to +\infty} f(x) \neq \lim_{x \to -\infty} f(x)$，则极限不存在。

<strong>微积分的运算过程是：先运算、化简，然后取值代入。</strong>

下面是一些关于求极限的练习题目，我进行了解答。这只是我认为值得一写的题目，并不全面。解答的过程有可能不严密，但是有效，这是我的风格^\_^：

<strong>(1) $\lim_{n \to \infty} (\sqrt{n^4+n+1}-n^2)(n+3)$</strong>

<strong>解答：</strong>
这首先要了解一条公式：$\sqrt{a^2+b}=a+\frac{b}{2a+\frac{b}{2a+\frac{b}{2a+...}}}$，我是根据一种方程迭代法推导出来的。应用到求极限很方便。如下：
$$\lim_{n \to \infty} (\sqrt{n^4+n+1}-n^2)(n+3) \\ =\lim_{n \to \infty} (n^2+\frac{n+1}{2n^2+\frac{n+1}{2n^2+...}}-n^2)(n+3) \\ =\lim_{n \to \infty} \frac{(n+1)(n+3)}{2n^2+\frac{n+1}{2n^2+...}}$$

因为$\lim_{n \to \infty} \frac{n+1}{2n^2}  \to 0$，所以可以忽略后面的，只考虑
$$\lim_{n \to \infty} \frac{(n+1)(n+3)}{2n^2} \\ =\lim_{n \to \infty} 1/2 (1+1/n)(1+3/n)=1/2$$

<strong>(2) $\lim_{x \to \infty} \frac{\sqrt[3]{x} sin x}{x+1}$</strong>

<strong>解答：</strong>
$$\frac{\sqrt[3]{x} \sin x}{x+1} < \frac{\sqrt[3]{x} \sin x}{x} = x^{-2/3}\sin x$$
当$x \to \infty$，有$x^{-2/3}  \to 0,|sin x| \leq 1$，所以
$$\lim_{x \to \infty} \frac{\sqrt[3]{x} \sin x}{x+1} \to 0$$

<strong>(3) $\lim_{n \to 0} \frac{e^n-1}{n}=1$</strong>

<strong>解答：</strong>
$e^n=1+n+\frac{n^2}{2!}+\frac{n^3}{3!}+\frac{n^4}{4!}+...$，所以有
$$\lim_{n \to 0} \frac{e^n-1}{n} \\ =\lim_{n \to 0} \frac{n+\frac{n^2}{2!}+\frac{n^3}{3!}+\frac{n^4}{4!}+...}{n} \\ =\lim_{n \to 0} 1+\frac{n}{2!}+\frac{n^2}{3!}+\frac{n^3}{4!}+...=1$$

<strong>(4) $\lim_{x \to 0} \frac{\sqrt{1+x sin x}-1}{e^{x^2}-1}$</strong>

<strong>解答：</strong>
这是我认为比较难的题目，要根据结论(3)
$$\lim_{x \to 0} \frac{\sqrt{1+x \sin x}-1}{e^{x^2}-1} \\ = \lim_{x \to 0} \frac{(\frac{x \sin x}{2+\frac{x \sin x}{2+...}})}{e^{x^2}-1}$$
因为$\lim_{x \to 0} x sin x \to 0$，所以把后面的忽略，只考虑：
$$\lim_{x \to 0} \frac{x \sin x}{(e^{x^2}-1)\cdot 2} \\ =\lim_{x \to 0} \frac{{\sin x}/{x}}{{e^{x^2}-1}/{x^2}\cdot 2}$$
已知：$\lim_{x \to 0} \frac{sin x}{x} \to 1,\lim_{x \to 0} \frac{e^{x^2}-1}{x^2} \to 1$，
所以
$$\lim_{x \to 0} \frac{\sqrt{1+x \sin x}-1}{e^{x^2}-1} \to 1/2$$

<strong>陆续补充中，将会与大家探讨更多的内容。</strong>
