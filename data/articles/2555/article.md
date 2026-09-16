# 傅里叶变换：只需要异想天开？

> 作者：苏剑林 · 科学空间 · 2014-04-25
>
> 原文：<https://spaces.ac.cn/archives/2555>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在对数学或物理进行事后分析，往往会发现一些奇怪的现象，也有可能得到一些更为深刻有趣的结果。比如本文所要谈及的傅里叶变换，可以由一种“异想天开”的思路得来。

## 洛朗展式

我们知道，在原点处形态良好的函数，可以展开为泰勒级数
$$f(x)=\sum_{n=0}^{\infty}a_n x^n$$
我们发现，上面的幂都是正的，为什么不能包含$x$的负数次幂呢？比如$\frac{\sin z}{z^2}$展开为
$$\frac{1}{z}-\frac{z}{6}+\frac{z^3}{120}\dots$$
显然也是一件合理的事情。于是，结合复变函数，我们得到解析函数的洛朗展式
$$f(z)=\sum_{n=-\infty}^{+\infty}a_n z^n$$
这是函数的双边展开。其中
$$a_n=\frac{1}{2\pi i}\int_{\gamma} \frac{f(z)}{z^{n+1}}dz$$
$\gamma$是积分曲线$|z|=\rho,\rho>0$。这个公式基于下面一个很显然的事实（$\alpha\in\mathbb{Z}$）
$$\int_{\gamma} z^{\alpha}dz=\left\{ {\begin{array}{\cdot {20}{c}}
{2\pi i,\alpha=-1;}\\
{0,\alpha\neq -1.}
\end{array}} \right.$$

## 半整数幂级数

可是，还有一个问题，为什么不能包含分数次幂呢？诸如$\sqrt{z}$的函数如何展开呢？这说明，洛朗展式还是可以拓展的。为了举例说明，我们不妨把“半整数幂”也加进去，考虑：
$$f(z)=\sum_{n=-\infty}^{+\infty}a_n z^{n/2}$$

如何求各项系数？我们将它转化为熟悉的洛朗展式即可。设$z^{1/2}=\xi$，则
$$f(\xi^2)=\sum_{n=-\infty}^{+\infty}a_n \xi^n$$
洛朗展式我们知道怎么求系数了，即
$$\begin{aligned}
a_n=\frac{1}{2\pi i}\int_{\gamma}\frac{f(\xi^2)}{\xi^{n+1}}d\xi &=\frac{1}{2\pi i}\int_{2\gamma}\frac{f(z)}{z^{(n+1)/2}}dz^{1/2}\\
&=\frac{1}{4\pi i}\int_{2\gamma}\frac{f(z)}{z^{n/2+1}}dz
\end{aligned}$$
$\gamma$是以原点为圆心的一个圆（逆时针绕原点一周），$n\gamma$则表示逆时针绕原点$n$周。新级数是洛朗展式的推广之一。

## 一劳永逸

可是，疑问又来了，为什么不能有1/3整数幂呢？为什么不能有无理数次幂呢？这类“异想天开”可以是无穷尽的。于是乎，为了避免被进一步提问，我们干脆把所有实数幂都考虑进去，这也就是本节的“一劳永逸”的意思。函数
$$f(z)=\int_{-\infty}^{+\infty}a(x)z^x dx$$
这包含了上面的两种形式。（离散的$a_n$对应带有狄拉克函数$\delta(x)$的$a(x)$。）。下面我们来推导$a(x)$的表达式。我们把求和离散化，从原点出发，以$\Delta x$为步长进行求和
$$f(z)=\lim_{\Delta x \to 0} \sum_{n=-\infty}^{+\infty}a(n\Delta x)\Delta x z^{n\Delta x}$$
设$z^{\Delta x}=u$，代入
$$f(\xi^{1/\Delta x})=\sum_{n=-\infty}^{+\infty}a(n\Delta x)\Delta x \xi^{n}$$
同样用洛朗展式的方法求系数，得
$$\begin{aligned}a(n\Delta x)\Delta x=\frac{1}{2\pi i}\int_{\gamma}\frac{f(\xi^{1/\Delta x})}{\xi^{n+1}}d\xi &=\frac{1}{2\pi i}\int_{\left(\frac{1}{\Delta x}\right)\gamma}\frac{f(z)}{z^{(n+1)\Delta x}}dz^{\Delta x}\\
&=\frac{1}{2\pi i}\int_{\left(\frac{1}{\Delta x}\right)\gamma}\frac{\Delta x f(z)}{z^{n\Delta x+1}}dz
\end{aligned}$$
此即
$$\begin{aligned}a(x)=\frac{1}{2\pi i}\int_{\left(\frac{1}{\Delta x}\right)\gamma}\frac{f(z)}{z^{x+1}}dz
\end{aligned}$$
积分路径为绕原点的圆无穷多圈。由于变换之间可逆，这成为了两个函数之间的相互变换。

## 傅里叶变换

现在我们换个熟悉的记号，设$z=e^{-i\omega}$，记$a(x)$换成记号$ f(x)$，对应的$f(z)$则换成$F(\omega)$，有
$$F(\omega)=\int_{-\infty}^{+\infty}f(x)e^{-i\omega x} dx$$
以及
$$f(x)=-\frac{1}{2\pi i}\int_{-\infty}^{+\infty} \frac{F(\omega)}{e^{-i\omega(x+1)}}de^{-i\omega}=\frac{1}{2\pi }\int_{-\infty}^{+\infty} F(\omega) e^{i\omega x}d\omega$$
我们取的积分曲线$\gamma$是逆时针的，但是$e^{-i\omega x}$是顺时针的，所以要加个负号。这就得到了傅里叶变换及其逆变换。

## 稍微总结

读者最大的疑问应该是为什么积分限是负无穷到正无穷，而不是0到正无穷？这个确实不容易解释清楚，严格推导傅里叶逆变换还是需要《数学物理方法》上面的步骤。当然，勉强来说，从本文的角度，还是可以给积分限一个解释的（解释而不是证明）。

<strong>如果积分上限选取适当，那么积分下限时可以随意选取的，比如积分下限取0，那么积分上限就必须取$2\pi N,N\in \mathbb{Z}$且$N\to \infty$。但是，趋于无穷时，加上个限制（$N\in\mathbb{Z}$而不是$N\in \mathbb{R}$）总是不舒服的，由于正无穷具有不确定性，我们干脆在下限也来个不确定性（负无穷），这样子，由于没有确定起点，也就没有必要给终点加上个限制了。于是积分上下限就可以写成正负无穷，至于区域无穷的部分的瑕疵，在积分的时候自然地被抹平了。（PS：<u>这段话相当含糊，但是为了使文章不至于陷入繁琐的技术细节中，只能说到这个份上。事实上，基于数学分析中严格的瑕积分理论，可以给上面这段话一个证明。这主要用到诸如$\int_0^{\infty} f(x)\cos(\omega x)dx$之类的积分技巧。希望深究的朋友，不妨尝试一下？</u>）</strong>

事实上，上面的推导基本上都是不严格的，本文的目的是想通过这个思路给大家一种对傅里叶变换的“比较自然”的认识，并且提供大家将两种看似不相关的东西（傅里叶变换和洛朗展式）联系起来的思路。通过复变函数，可以将很多数学分支巧妙地联系起来。这似乎是设计者的一个法则：和谐统一。如果真是这样子的话，我们将有更深层次的理由去探索数学、探索科学，那就是——为了美！
