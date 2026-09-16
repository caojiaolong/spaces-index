# 封闭曲线所围成的面积：一个新技巧

> 作者：苏剑林 · 科学空间 · 2015-08-30
>
> 原文：<https://spaces.ac.cn/archives/3441>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>本文主要做了一个尝试，尝试不通过Green公式而实现将封闭曲线的面积与线积分相互转换。这种转换的思路，因为仅仅利用了二重积分的积分变换，较为容易理解，而且易于推广。至于这种技巧是否真正具有实际价值，还请读者评论。</strong>

假设平面上一条简单封闭曲线由以下参数方程给出：
$$\begin{equation}\left\{\begin{aligned}x = f(t)\\y = g(t)\end{aligned}\right.\end{equation}$$
其中参数$t$位于某个区间$[a,b]$上，即$f(a)=f(b),g(a)=g(b)$。现在的问题是，求该封闭曲线围成的区域的面积。

### Green公式

通常的解决思路是使用Green公式：
$$\begin{equation}\iint\limits_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)dxdy = \oint\limits_{\partial D} Pdx+Qdy \end{equation}$$
它告诉我们面积分和线积分可以相互转化。当$\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}=1$时，左边就是求区域内的面积。满足这样的$P,Q$有很多，如$Q=x,P=0$，那么
$$\begin{equation}\iint\limits_D dxdy = \oint\limits_{\partial D} xdy=\int_a^b f(t)g'(t)dt\label{eq:xdy}\end{equation}$$
由此可见，换用不同的$P,Q$，可以构造出各种各样的计算面积的公式，其中，在很多情形会相对方便一些的公式是
$$\begin{equation}\iint\limits_D dxdy = \frac{1}{2}\oint\limits_{\partial D} -ydx+xdy=\frac{1}{2}\int_a^b [f(t)g'(t)-g(t)f'(t)]dt\label{eq:xdy-ydx}\end{equation}$$
因为它对于过原点的直线的积分自动为0。

### 新技巧：行列式变换

上述利用Green公式的确方便实用，令人惊叹。然而对我来说，会有一些美中不足之处：

> 1、基础是Green公式，然而格林公式在一般人眼中会觉得是比较高端的内容，对于笔者也是如此；
> 
> 
> 
> 2、如何推广到其他坐标系（如极坐标）？甚至是其他的二位曲面坐标系（如球面上的几何）？格林公式似乎不能很好地做到这一点，当然，可以考虑微分几何中的Stokes公式，但这又是更加深入的内容了。

事实上，笔者一直在构思不利用Green公式的、更自然的做法。算是“功夫不负有心人”，笔者找到了让自己还算满意的、比利用Green公式更加自然的思路，在此与大家分享。

何为更加“自然”？笔者的思路是直接通过积分变换来给出结果的。求面积，本质上就是一个二重积分，二重积分不能直接求出来，就考虑变换坐标系，这思路够显然了吧？好，进入正题，不失一般性，假设原点位于这条封闭曲线内，并且这条曲线能够<strong>等比例地、径直地、不重叠地</strong>“收缩”到原点，用数学表示出来，就是说该区域可以用以下参数方程来表示：
$$\begin{equation}\left\{\begin{aligned}x = sf(t)\\y = sg(t)\end{aligned}\right. ,\,t\in[a,b],s\in[0,1]\label{eq:x-y-t}\end{equation}$$
这样我们就可以用$s,t$作为新坐标求积分，雅可比行列式是（有可能需要取绝对值，这里是特殊情况）：
$$\begin{equation}J=\begin{vmatrix}\frac{\partial[sf(t)]}{\partial s} & \frac{\partial[sf(t)]}{\partial t}\\
\frac{\partial[sg(t)]}{\partial s} & \frac{\partial[sg(t)]}{\partial t}\end{vmatrix}=\begin{vmatrix}f(t) & sf'(t)\\
g(t) & sg'(t)\end{vmatrix}=s[f(t)g'(t)-g(t)f'(t)]\end{equation}$$
所以
$$\begin{equation}\begin{aligned}\iint\limits_D dxdy = &\int_a^b \int_0^1 s[f(t)g'(t)-g(t)f'(t)]dsdt \\
= &\frac{1}{2}\int_a^b [f(t)g'(t)-g(t)f'(t)]dt\end{aligned}\end{equation}$$
再次得到了公式$\eqref{eq:xdy-ydx}$！

上述是怎样的一个过程呢？我们发现，我们是从中假设了封闭曲线能够以某种方式不重叠地“收缩”到原点，从而变换坐标系。换不同的收缩方式，就可以得到不同的面积公式，如
$$\begin{equation}\left\{\begin{aligned}x = sf(t)\\y = g(t)\end{aligned}\right. ,\,t\in[a,b],s\in[0,1]\end{equation}$$
表示曲线能够不重叠地压缩到$y$轴上，这样我们通过积分变换得到雅可比行列式为$J=f(t)g'(t)$，从而
$$\begin{equation}\begin{aligned}\iint\limits_D dxdy = &\int_a^b \int_0^1 f(t)g'(t)dsdt \\
= &\int_a^b f(t)g'(t)dt
\end{aligned}\end{equation}$$
我们再次得到了公式$\eqref{eq:xdy}$！

<strong>换言之，每一条公式也对应着一种收缩的方式。当然，从Green公式可以得出，这些公式都是等价的。但是我们不借助格林公式的话，那么这些公式并非都是等价的，而是各有各的用途，因为，并非所有的封闭曲线都可以用同一种收缩方式不重叠地收缩到原点。至于更一般化的曲线，如原点不在曲线内，可以通过分段或者平移的方式处理，最终的结果仍然不变——换言之，图形的面积跟图形位于哪里无关。</strong>

### 极坐标

从上面的讨论中可以看出，新技巧得到的结果，好像都没有用Green公式得来的多，干嘛还要煞费苦心去钻研这样一套方法呢？其中原因之一是，上述坐标变换的方法，可以比较容易推广到其他坐标系中去，而直接通过Green公式则不是特别方便。

我们来考虑极坐标，假设通过极坐标的参数方程
$$\begin{equation}\left\{\begin{aligned}r = f(t)\\\theta = g(t)\end{aligned}\right.\end{equation}$$
描述了一条简单封闭曲线，求曲线内的面积。

自然，可以考虑曲线如何收缩到原点。还是不失一般性，假设原点位于封闭曲线内，并且还是假设这条曲线能够<strong>等比例地、径直地、不重叠地</strong>“收缩”到原点，这时候用数学公式来描述，就是下面参数方程描述了该区域：
$$\begin{equation}\left\{\begin{aligned}&r = sf(t)\\ &\theta = g(t)\end{aligned}\right. ,\,t\in[a,b],s\in[0,1]\label{eq:r-s-t}\end{equation}$$
<strong>对比$\eqref{eq:x-y-t}$式，就可以发现普通的直角坐标跟极坐标的差别。</strong>

$\eqref{eq:r-s-t}$式的雅可比行列式是$J=f(t)g'(t)$，从而区域面积是：
$$\begin{equation}\begin{aligned}\iint\limits_{D}rdrd\theta=&\int_a^b\int_0^1 sf(t)\cdot f(t)g'(t)dsdt\\
=&\frac{1}{2}\int_a^b f^2 (t)g'(t)dt\\
=&\frac{1}{2}\oint\limits_{\partial D}r^2 d\theta\end{aligned}\label{eq:r2ds}\end{equation}$$
$\eqref{eq:r2ds}$式的最后，就是将极坐标下的封闭曲线内的面积转化为极坐标下的线积分，跟公式$\eqref{eq:xdy}$或$\eqref{eq:xdy-ydx}$是一样的。对于不能以此方式收缩的曲线，可以对它进行分段处理，最终的结果仍然是$\eqref{eq:r2ds}$式。

<strong>看起来，$\eqref{eq:r2ds}$式的结果是平凡的，因为直接对$\iint\limits_{D}rdrd\theta$的$dr$先积分，也得到差不多的结果。但是要注意，最后的$\frac{1}{2}\oint\limits_{\partial D}r^2 d\theta$是一个线积分，对任意曲线都奏效，而直接对$\iint\limits_{D}rdrd\theta$的$dr$先积分的可行性跟曲线的形状有关。</strong>

### 球面坐标

最后，我们以球面坐标下的情形结束本文。球面是一个二维曲面的例子，在球面上，可以通过$\varphi,\theta$两个参数来确定一条球面上封闭曲线：
$$\begin{equation}\left\{\begin{aligned}\varphi = f(t)\\ \theta = g(t)\end{aligned}\right.\end{equation}$$
这里的$\varphi,\theta$，对应三维的球坐标变换：
$$\begin{equation}\left\{\begin{aligned}&x = r\sin\varphi\cos\theta\\
&y = r\sin\varphi\sin\theta\\
&z = r\cos\varphi
\end{aligned}\right.\end{equation}$$
仿照$\eqref{eq:r-s-t}$式，可以得到如下收缩方式（在球面坐标中，$\varphi$的作用跟极坐标的$r$是类似的）：
$$\begin{equation}\left\{\begin{aligned}&\varphi = sf(t)\\ &\theta = g(t)\end{aligned}\right. ,\,t\in[a,b],s\in[0,1]\label{eq:t-v-t}\end{equation}$$
从而
$$\begin{equation}\begin{aligned}\iint\limits_{D}\sin \varphi d\varphi d\theta=&\int_a^b\int_0^1 \sin[sf(t)]\cdot f(t)g'(t)dsdt\\
=&-\int_a^b \left.\cos[sf(t)]\right|_0^1 \cdot g'(t)dt\\
=&\int_a^b [1 - \cos f(t)] \cdot g'(t)dt\\
=&\oint\limits_{\partial D} (1 - \cos \varphi) d\theta\end{aligned}\label{eq:sinds}\end{equation}$$
