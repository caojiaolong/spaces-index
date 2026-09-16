# 三角函数幂的定积分

> 作者：苏剑林 · 科学空间 · 2013-12-05
>
> 原文：<https://spaces.ac.cn/archives/2192>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近的我的主要学习是在研究路径积分，在推导路径积分的一种新的变换方法（或者是一个新的视角吧），但是有道坎还是迈不过去，因此blog中也一直更新寥寥。说到积分与微分，这两个本是互逆的东西，但是在复数的统一之下，它们两个去可以相互转化。比如说，薛定谔方程是量子力学的微分形式，而路径积分实际上可以说是量子力学的积分形式，这让我有些想法，是不是任何微分形式的数学都存在一个积分形式的版本呢？如果是，是微分版本优还是积分版本优？

在数学分析中，我们会感觉到求导会比求积分容易很多，求导有现成的公式等等。但是微分有个最大的缺点，它是多分量的，比如，势函数是一个标量，但是微分（求梯度）之后就变成了三分量的矢量（即作用力），多分量事实上是不好处理了，为了处理这类问题，又引入了大量的算符。积分的特点在于它的标量性，也许计算很复杂，但是思想确实容易把握的，我更喜欢积分形式的理论（比如作用量原理、路径积分等。）

说到数学分析中常见而又著名的定积分，不得不提到以下三角函数积分了。
$$\int_0^{\pi/2} \sin^{2n} \theta d\theta$$
不难证明，它也等于
$$\int_0^{\pi/2} \cos^{2n} \theta d\theta$$

有很多方法可以求出这个定积分。教科书上给出的方法是数学归纳法递推的，而我觉得比较简单而又容易记忆的（也就是说如果忘记了也可以很快重新推导一次）方法是利用三角函数的指数形式。只需利用$\cos\theta=\frac{e^{i\theta}+e^{-i\theta}}{2}$，将积分写成
$$\begin{aligned}
\int_0^{\pi/2} \cos^{2n} \theta d\theta &=\int_0^{\pi/2} \left(\frac{e^{i\theta}+e^{-i\theta}}{2}\right)^{2n} d\theta\\
&=\frac{1}{2^{2n}}\int_0^{\pi/2} \left(e^{i\theta}+e^{-i\theta}\right)^{2n} d\theta
\end{aligned}$$
考虑$\left(e^{i\theta}+e^{-i\theta}\right)^{2n}$，它必然是
$$\frac{(2n)!}{(n!)^2}+\sum_{k=-n,k\neq 0}^{n} a_k e^{2ik\theta}$$
的形式（可以写出系数$a_k$的具体形式来，但是这不重要，只需要知道它是实数就好）。并且
$$\int_0^{\pi/2} e^{2ik\theta}d\theta=\left. -i\frac{1}{2k}e^{2ik\theta} \right|_0^{\pi/2}=\frac{[(-1)^{k+1}+1]i}{2k}$$
事实上，也不需要算出它的积分值，只需要确定它是一个纯虚数就行。那么
$$\begin{aligned}\int_0^{\pi/2}\cos^{2n} \theta d\theta &=\frac{1}{2^{2n}}\int_0^{\pi/2} \left[ \frac{(2n)!}{(n!)^2} +\sum_{jk=-n,k\neq 0}^{n} a_k e^{2ik\theta}\right]d\theta\\
&=\frac{1}{2^{2n}}\frac{(2n)!}{(n!)^2}\frac{\pi}{2}+i\sum\limits_{k=-n,k\neq 0}^{n}\frac{[(-1)^{k+1}+1]a_k}{2k}
\end{aligned}$$
不管$a_k$的形式是什么，它求和之后与$i$相乘要得到一个纯虚数。但是左边是一个实数，应该虚数项必须为0。因此
$$\begin{aligned}\int_0^{\pi/2}\cos^{2n} \theta d\theta &=\frac{1}{2^{2n}}\frac{(2n)!}{(n!)^2}\frac{\pi}{2}\\
&=\frac{(2n)!}{(2n!!)^2}\frac{\pi}{2}\end{aligned}$$

不过，要是想用同样的技巧求$\int_0^{\pi/2}\sin^{2n+1}\theta d\theta$和$\int_0^{\pi/2}\cos^{2n+1}\theta d\theta$的话，就不显得简单了。事实上，对于奇数次幂，似乎不存在比较简单而直接的方法。
