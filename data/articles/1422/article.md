# 三角函数幂的积分

> 作者：苏剑林 · 科学空间 · 2011-07-22
>
> 原文：<https://spaces.ac.cn/archives/1422>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨天在研究一个最优化问题时，遇到了一个这样的积分：
$$\int \frac{1}{\cos^3 \theta} d\theta$$

然后就顺便研究了一下这种类型的函数的积分。一般来讲，这类积分可以写成$\int cos^n \theta d\theta$或$\int sin^n \theta d\theta$，其中n是一个整数。

首先我们来解决n=1的情况，我们很容易就有$\int cos\theta d\theta=sin\theta +C$或$\int sin\theta d\theta=-cos\theta +C$，这是一个基本的结果。

如果n是大于1的正整数，那么可以用递推的方法来搞定：
$$\begin{aligned}\int \cos^n \theta d\theta=\int \cos^{n-1} \theta d(\sin\theta)=\cos^{n-1} \theta \sin\theta-\int \sin\theta d(\cos^{n-1}\theta) \\ =\cos^{n-1} \theta \sin\theta+(n-1)\int \sin^2\theta \cos^{n-2}\theta d\theta \\ =\cos^{n-1} \theta \sin\theta+(n-1)\int (1-\cos^2\theta) \cos^{n-2}\theta d\theta \\ =\cos^{n-1} \theta \sin\theta+(n-1)\int \cos^{n-2}\theta d\theta-(n-1)\int \cos^n\theta d\theta\end{aligned}$$
（这里运用了分部积分法）

即$n\int cos^n \theta d\theta=cos^{n-1} \theta sin\theta+(n-1)\int  cos^{n-2}\theta d\theta$
$$\int \cos^n \theta d\theta=\frac{1}{n}[\cos^{n-1} \theta \sin\theta+(n-1)\int \cos^{n-2}\theta d\theta]$$
这样就完成了递推。关于正弦也有类似公式，这里直接写出：
$$\int \sin^n \theta d\theta=\frac{1}{n}[-\sin^{n-1} \theta \cos\theta+(n-1)\int \sin^{n-2}\theta d\theta]$$

不过，我昨晚想到将其变为多项式积分来算：
$$\int \cos^n \theta d\theta=\int \cos^{n-1} \theta d(\sin\theta)$$
如果令$sin\theta=x$，则积分成了
$$\int (1-x^2)^{\frac{n-1}{2}} dx$$

要是n是奇数（正或负都行）的话，这个积分是相当好办的。要是它是偶数或其它有理数的话，就需要用到超几何函数（[Hypergeometric Function](<http://mathworld.wolfram.com/HypergeometricFunction.html>)）来表达。这里只对n是奇数进行讨论。

当n是正数时，直接多项式展开进行积分，不再讨论。当n是负数时，相当于讨论积分
$$\int \frac{1}{(1-x^2)^k} dx$$
其中k是正整数。当k=1时，有
$$\int \frac{1}{1-x^2} dx=\frac{1}{2} ln|\frac{1+x}{1-x}|$$

当k \> 1时，我们即$\int \frac{1}{(1-x^2)^k} dx=I_k$，有
$$\begin{aligned}I_k=\int \frac{1-x^2+x^2}{(1-x^2)^k} dx=I_{k-1}+\int \frac{x^2}{(1-x^2)^k} dx \\ =I_{k-1}-\frac{1}{2}\int \frac{x}{(1-x^2)^k} d(1-x^2) \\ =I_{k-1}+\frac{1}{2(k-1)}\int x d[(1-x^2)^{1-k}] \\ =I_{k-1}+\frac{1}{2(k-1)}[x (1-x^2)^{1-k}-\int (1-x^2)^{1-k}dx] \\ =(\frac{2k-3}{2k-2})I_{k-1}+\frac{x}{2(k-1)(1-x^2)^{k-1}}\end{aligned}$$
递归完毕。

这里直接写出我可能用到的结果：
$$\int \frac{1}{\cos^3 \theta} d\theta=\frac{1}{4}(\frac{2\sin \theta}{\cos^2 \theta}+ln|\frac{1+\sin \theta}{1-\sin \theta}|)$$
$\int \frac{1}{sin^3 \theta} d\theta=-\frac{1}{4}(\frac{2cos \theta}{sin^2 \theta}+ln|\frac{1+cos \theta}{1-cos \theta}|)$
