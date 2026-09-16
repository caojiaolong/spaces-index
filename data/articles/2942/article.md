# 生成函数法与整数的分拆

> 作者：苏剑林 · 科学空间 · 2014-09-16
>
> 原文：<https://spaces.ac.cn/archives/2942>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们在高中甚至初中，都有可能遇到这样的题目：

> 设$x,y,z$是非负整数，问$x+y+z=2014$有多少组不同的解？（不同顺序视为不同的解）

难度稍高点，可以改为

> 设$x,y,z$是非负整数，$0\leq x\leq y\leq z$，问$x+y+z=2014$有多少组不同的解？

这些问题都属于整数的分拆问题（广为流传的哥德巴赫猜想也是一个整数分拆问题）。有很多不同的思路可以求解这两道题，然而，个人认为这些方法中最引人入胜的（可能也是最有力的）首推“生成函数法”。

关于生成函数，本节就不多作介绍了，如果缺乏相关基础的朋友，请先阅读相关资料了解该方法。不少数论的、离散数学的、计算机科学的书籍中，都介绍了生成函数法（也叫母函数法）。本质上讲，母函数法能有诸多应用，是因为$x^a\times x^b=x^{a+b}$这一性质的成立。

<strong>有序的分拆</strong>

我们首先来解决第一个问题，也就是把整数$n$分拆为$k$个整数和的拆分数，不同的顺序也视为不同的分拆方式。记$n$的有序分拆数记为$\alpha(n,k)$，不难推出
$$\begin{aligned}\sum_{n=0}^{\infty}\alpha(n,k)x^n&=\left(1+x+x^2+x^3+x^4+\dots\right)^k\\
&=\left(\frac{1}{1-x}\right)^k\end{aligned}$$
生成函数的有效性和便捷性还在于诸如$\left(\frac{1}{1-x}\right)^k$这一类函数的泰勒展开式是很容易求得的：
$$\begin{aligned}\left(\frac{1}{1-x}\right)^k&=\frac{1}{(k-1)!}\left(\frac{1}{1-x}\right)^{(k-1)}\\
&=\frac{1}{(k-1)!}\left(1+x+x^2+x^3+x^4+\dots\right)^{(k-1)}\\
&=\frac{1}{(k-1)!}\sum_{n=k-1}^{\infty}\frac{n !}{(n-k+1)!}x^{n-k+1}\\
&=\sum_{n=0}^{\infty}\binom{n+k-1}{k-1}x^{n}
\end{aligned}$$
也就是说，将$n$进行有序分拆为$k$个正整数之和的分拆数$\alpha(n,k)=\binom{n+k-1}{k-1}$。这跟从排列组合出发的解法是一样的。生成函数法的优点在于，方法本身是很美妙的，而计算未必简单，但都是可以按部就班计算下去的。事实上，用其他方法能够解决的问题，如果改用生成函数法，复杂程度并不会增加。

在本题中，对于$n=2014$，给出答案为$\binom{2016}{2}=2031120$。

<strong>无序的分拆</strong>

第二个问题属于无序地分拆，也就是把整数$n$分拆为$k$个整数和的拆分数，不同的顺序也视为同一种分拆方式。如果不用生成函数法，则可以通过对称性的分析逐步求得答案，也可以通过小的数字逐步枚举归纳，但都是步骤较多，而且难以得到一般的公式。用生成函数法则显得更为直接。记$n$的无序分拆数为$\beta(n,k)$，关于该分拆的生成函数稍微复杂一些：
$$\begin{aligned}\sum_{n=0}^{\infty}\beta(n,k)x^n&=\prod_{r=1}^{k}\left(\frac{1}{1-x^r}\right)\end{aligned}$$
我们把推导放到最后，首先来看这样的生成函数给我们带来什么样的结果。这里就$k=3$为例：
$$\begin{aligned}
&\left(\frac{1}{1-x}\right)\left(\frac{1}{1-x^2}\right)\left(\frac{1}{1-x^3}\right)\\
=&\frac{1+x}{(1-x^2)^2 (1-x^3)}\\
=&\frac{(1+x)(1+x^2+x^4)^2}{(1-x^6)^2 (1-x^3)}\\
=&\frac{(1+x)(1+x^2+x^4)^2 (1+x^3)}{(1-x^6)^2 (1-x^6)}\\
=&\frac{(1+x)(1+x^2+x^4)^2 (1+x^3)}{(1-x^6)^3}
\end{aligned}$$
化简的要诀在于将分母变成$(1-x^p)^q$的形式，因为这样形式的分式的泰勒展式是容易求出来的。下面，我们算得：
$$\left\{\begin{aligned}&(1+x)(1+x^2+x^4)^2 (1+x^3)=1 + x + 2 x^2 + 3 x^3 + 4 x^4 + 5 x^5 \\
&\qquad\qquad\qquad\qquad\qquad\qquad+ 4 x^6 + 5 x^7 + 4 x^8 + 3 x^9 + 2 x^{10} + x^{11} + x^{12}\\
&\frac{1}{(1-x^6)^3}=\sum_{n=0}^{\infty}\binom{n+2}{2}x^{6n}
\end{aligned}\right.$$
相乘得
$$\begin{aligned}
&\sum_{n=0}^{\infty}\left[\binom{n+2}{2}+\binom{n}{2}+4\binom{n+1}{2}\right]x^{6n}\\
+&\sum_{n=0}^{\infty}\left[\binom{n+2}{2}+5\binom{n+1}{2}\right]x^{6n+1}\\
+&\sum_{n=0}^{\infty}\left[2\binom{n+2}{2}+4\binom{n+1}{2}\right]x^{6n+2}\\
+&\sum_{n=0}^{\infty}\left[3\binom{n+2}{2}+3\binom{n+1}{2}\right]x^{6n+3}\\
+&\sum_{n=0}^{\infty}\left[4\binom{n+2}{2}+2\binom{n+1}{2}\right]x^{6n+4}\\
+&\sum_{n=0}^{\infty}\left[5\binom{n+2}{2}+\binom{n+1}{2}\right]x^{6n+5}\\
\end{aligned}$$
以上的计算表明，答案跟被分拆数模6的结果有关，比如$\beta(6n,3) = \binom{n+2}{2} + \binom{n}{2} + 4\binom{n+1}{2}$而$\beta(6n+1,3)=\binom{n+2}{2}+5\binom{n+1}{2}$，等等。倘若不通过生成函数法，则难以求得一般表达式出来。

在本题中，$2014=6\times 335+4$，所以给出答案$4\binom{337}{2}+2\binom{336}{2}=339024$。

<strong>无序的分拆：推导</strong>

为什么无序分拆的生成函数会是上述的样子？我们先考虑$n=x+y,x\leq y$的分拆数，显然，根据定义，它可以对应于生成函数：
$$\begin{aligned}&1\times(1+x+x^2+x^3+\dots)+x\times(x+x^2+x^3+x^4+\dots)\\
&+x^2\times(x^2+x^3+x^4+x^5\dots)+x^3\times(x^3+x^4+x^5+x^6\dots)+\dots\\
=&\frac{1}{1-x}+\frac{x^2}{1-x}+\frac{x^4}{1-x}+\frac{x^6}{1-x}+\dots\\
=&\frac{1}{(1-x)(1-x^2)}\end{aligned}$$
更一般地，可以归纳地证明，分拆为$k$个正整数之和时，其生成函数是
$$\prod_{r=1}^{k}\left(\frac{1}{1-x^r}\right)$$
有兴趣的朋友可以亲自动手完成这一过程，相信会收获不少。

<strong>2014年11月20日更新：</strong>

关于无序的分拆，更方便的一个推导是这样的。比如说$n=x+y+z, x\leq y\leq z$，那么设$y=x+a,z=y+b=x+a+b$，那么$n=x+y+z, x\leq y\leq z$的自然数解，就相当于$n=3x+2a+b$的自然数解，显然$3x,2a,b$的生成函数分别是$\frac{1}{1-x^3}$、$\frac{1}{1-x^2}$和$\frac{1}{1-x}$，相乘起来即可。
