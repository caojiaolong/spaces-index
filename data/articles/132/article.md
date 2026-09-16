# 一道从小学到高中都可能考到的题目

> 作者：苏剑林 · 科学空间 · 2009-09-20
>
> 原文：<https://spaces.ac.cn/archives/132>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这是一道很多时候都会考到的题目：
<strong>比较$n^{n+1}$与$(n+1)^n$的大小（其中n非负）。</strong>

在小学我们会使用直接计算；
在初中我们会从一些例子找规律；
在高中我们就会直接去证明了。

这道题目的答案是：当n\>e时，有$n^{n+1}>(n+1)^n$。

我给出的证明有两个：

<strong>证明一：</strong>

要证$n^{n+1}>(n+1)^n$，等价于证明$n>\frac{(n+1)^n}{n^n}=(1+1/n)^n$。

而在研究e的时候可以知道，对于非负的n，$(1+1/n)^n$是单调递增的；而$\lim_{x->+\infty}(1+1/n)^n=e$，所以$(1+1/n)^n < e$。因此当n\>e时，$n>\frac{(n+1)^n}{n^n}=(1+1/n)^n$是成立的。证毕。

<strong>证明二：</strong>

这个证明相对通用一些，可以用来比较$n^{n+m}$与$(n+m)^n$等情况。现在只讨论m=1的情况。

函数$(\frac{ln x}{x})'=\frac{1-ln x}{x^2}$，当x\>e时，$\frac{1-ln x}{x^2}<0$，即此时函数$\frac{ln x}{x}$单调递减。若有e
$$\frac{ln a}{a}>\frac{ln b}{b}<\Rightarrow \frac{ln a}{ln b}>\frac{a}{b}$$

而当n\>e时，
$$n^{n+1}=(n+1)^{(n+1)\cdot \frac{ln n}{ln(n+1)}}>(n+1)^{(n+1)\cdot \frac{n}{n+1}}=(n+1)^n$$

证毕。
