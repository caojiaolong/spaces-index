# 特殊的通项公式：二次非线性递推

> 作者：苏剑林 · 科学空间 · 2014-11-12
>
> 原文：<https://spaces.ac.cn/archives/3064>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## 特殊的通项公式

对数学或编程感兴趣的读者，相信都已经很熟悉斐波那契数列了

> 0, 1, 1, 2, 3, 5, 8, 13, \.\.\.

它是由
$$a_{n+2}=a_{n+1}+a_n,\quad a_0=0,a_1=1$$
递推所得。读者或许已经见过它的通项公式
$$a_{n}=\frac{\sqrt{5}}{5} \cdot \left[\left(\frac{1 + \sqrt{5}}{2}\right)^{n} - \left(\frac{1 - \sqrt{5}}{2}\right)^{n}\right]$$
这里假设我们没有如此高的智商可以求出这个复杂的表达式出来，但是我们通过研究数列发现，这个数列越来越大时，相邻两项趋于一个常数，这个常数也就是（假设我们只发现了后面的数值，并没有前面的根式）
$$\beta=\frac{1 + \sqrt{5}}{2}=1.61803398\dots$$
因此，我们会想到用等比数列$\alpha \beta^n$近似表示这个数列，并且通过数值计算，发现$\alpha$确实趋于一个常数，它就是
$$\alpha=\frac{1}{\sqrt{5}}=0.4472135\dots$$
后来我们就发现，斐波那契数列的通项可以表示为
$$a_n=\left[\alpha \beta^n\right]$$
其中$\left[x\right]$表示对$x$进行四舍五入，这结果当然可以严格地证明。但是我们只留意这里出现的一些新奇的东西，它表明，<strong>有些数列的通项可以表示为某个无理数列的取整</strong>，这是一种比较奇特的通项公式。下面我们考虑非线性递推
$$a_{n+1}=a_n^2+c$$
其中$a_0$和$c$都为正数，我们将会发现，这将导致一个类似的取整通项。

## 二次非线性递推

可以看到，如果$c=0$，那么通项是很容易的，我们有$a_n=a_0^{2^n}$，这是一个快速增长的数列（$a_0 > 1$时），如果$c$不为0，那么这将是一个递增得非常快的数列（指数的指数，像费马数一般），比如$a_0=1,c=1$时，我们有

> 1, 2, 5, 26, 677, 458330, 210066388901, \.\.\.

在$n$比较大的时候，$a_n^2+c$与$a_n^2$的差别是微不足道的。于是还是可以认为，存在某个常数$k$，使得$a_n\sim k^{2^n}$，本文的研究思路正是基于此，而计算主要来自数学研发论坛的[kastin大神](<http://bbs.emath.ac.cn/forum.php?mod=redirect&goto=findpost&ptid=5794&pid=55470&fromuid=70>)。

设$x_n=\ln a_n$，那么
$$x_{n+1}=2x_n+\ln\left(1+\frac{1}{a_n^2}\right)$$
迭代得
$$\begin{aligned}x_n=&2^n x_0 +\sum_{i=0}^{n-1} 2^{n-i}\ln\left(1+\frac{1}{a_i^2}\right)\\
=&2^n x_0+\sum_{i=0}^{\infty}2^{n-1-i}\ln\left(1+\frac{1}{a_i^2}\right) -\sum_{i=n}^{\infty}2^{n-1-i}\ln\left(1+\frac{1}{a_i^2}\right)\\
=&2^n\left(x_0+\sum_{i=0}^{\infty}\frac{1}{2^{i+1}}\ln\left(1+\frac{1}{a_i^2}\right)\right) -2^{n-1}\sum_{i=n}^{\infty}\frac{1}{2^{i}}\ln\left(1+\frac{1}{a_i^2}\right)\end{aligned}$$
从$a_n$的增长速度可以看出，第一个无穷级数必然收敛，于是我们就可以（目前还只能通过数值方法）求出它的（近似）极限值。然后我们估计第二个级数的影响。于是
$$\begin{aligned}a_n=&\exp(x_n)\\
=&\exp(-r_n)k^{2^n}\end{aligned}$$
其中
$$\begin{aligned}&k=\exp\left[x_0+\sum_{i=0}^{\infty}\frac{1}{2^{i+1}}\ln\left(1+\frac{1}{a_i^2}\right)\right]\\
&r_n=2^{n-1}\sum_{i=n}^{\infty}\frac{1}{2^{i}}\ln\left(1+\frac{1}{a_i^2}\right) < \ln\left(1+\frac{1}{a_n^2}\right)\end{aligned}$$
那么
$$a_n > \exp\left(-\ln\left(1+\frac{1}{a_n^2}\right)\right)k^{2^n}$$
即
$$a_n+\frac{1}{a_n} > k^{2^n}$$
同理，有
$$r_n > -\ln\left(1+\frac{1}{a_n^2}\right)$$
所以
$$a_n < \exp\left(\ln\left(1+\frac{1}{a_n^2}\right)\right)k^{2^n}$$
即
$$k^{2^n}> a_n \left/\left(1+\frac{1}{a_n^2}\right)\right. > a_n - \frac{1}{a_n}$$
从而
$$\left|a_n-k^{2^n}\right| < \frac{1}{a_n} < \frac{1}{2}\quad (n\geq 1)$$
也就是说$a_n$是最接近$k^{2^n}$的整数，于是得到通项公式
$$a_n=\left[k^{2^n}\right]$$
实际上，可以进一步得到，该取整符号总是下取整的（只有四舍，没有五入）。这是Aho和Sloane在1973年得到的结果。

## 编程计算

下面编程计算$k$，Python代码



<pre><code class="language-python">from math import log,exp
n=6 # 计算6步已经很精确了。
x=1
l=[log(1+1/x**2)/2]
for i in range(n):
    x=x**2+1
    l.append(log(1+1/x**2)/2**(i+2))

k=exp(sum(l))
print(k)</code></pre>



算得

> k=1\.5028368010497564\.\.\.

<strong>要注意，原则上来说，精确计算$k$需要知道所有的$a_n$，不然计算出来的近似的$k$，只能用于计算前几项，后面的误差就大了，于是这项工作看来是本末倒置了，从这里可以看出，如果要计算$a_n$的具体值，最好的方法还是老老实实递归吧。然而，对于一个数列，我们要做的事情，不仅仅是要计算每一项，还要分析它的性态等。写出这个形式的解，就有助于让我们看出它是如何增长的——基本上按照$k$为底的指数的指数增长！</strong>

## 方法评述

可以看出，我们求二次非线性递推通项的主要思路就是发现数列是递增的，然后就可以认为在很大的时候，常数的影响很小，因此近似于没有常数的情况，从而可以退化为没常数的情况，并且估计通项公式。类似地，该方法可以推广到三次甚至更高次的情况，思路和计算都是类似的，主要是在$n$很大的时候，通项几乎只取决于最高次项。事实上，上例中的常数$k$，可以更简洁表述为
$$k=\lim_{n\to\infty}\left(a_n\right)^{2^{-n}}$$

同时对比斐波那契数列，我们可以写出斐波那契数列数列的取整形式的通项公式，它与非取整形式的通项公式相比，就多出了另外一项（负数幂），通过另外一项来“填补”取整那部分。这是不是意味着：二次非线性递推，也可能存在另外一项，来补充取整的那部分，从而去掉取整符号？让人惊喜的是，对于某些二次非线性递推，确实是这样子的！不过这我们将在以后再谈到了。

然而，还有一些其他的非线性递推问题，比如离散阻滞增长模型，其递推是
$$a_{n+1}-a_n=\alpha a_n-\beta a_n^2$$
其中$\alpha,\beta$都是正数，且一般来讲$\alpha \gg \beta$，这样的情况解的性态是很良好的（那条S型曲线），但是我们并没有很好地写出它的显式解来——哪怕只是高于一阶的近似。因此，如果有兴趣的话，可以往这个方向研究下去。

## 参考：

[http://mathworld\.wolfram\.com/QuadraticMap\.html](<http://mathworld.wolfram.com/QuadraticMap.html>)
