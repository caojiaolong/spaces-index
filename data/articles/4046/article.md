# 【理解黎曼几何】8\. 处处皆几何 (力学几何化)

> 作者：苏剑林 · 科学空间 · 2016-11-02
>
> 原文：<https://spaces.ac.cn/archives/4046>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

黎曼几何在广义相对论中的体现和应用，虽然不能说家喻户晓，但想必大部分读者都有所听闻。一谈到黎曼几何在物理学中的应用，估计大家的第一反应就是广义相对论。常见的观点是，广义相对论的发现大大推动了黎曼几何的发展。诚然，这是事实，然而，大多数人不知道的事，哪怕经典的牛顿力学中，也有黎曼几何的身影。

本文要谈及的内容，就是如何将力学几何化，从而使用黎曼几何的概念来描述它们。整个过程事实上是提供了一种框架，它可以将不少其他领域的理论纳入到黎曼几何体系中。

黎曼几何的出发点就是黎曼度量，通过黎曼度量可以通过变分得到测地线。从这个意义上来看，黎曼度量提供了一个变分原理。那反过来，一个变分原理，能不能提供一个黎曼度量呢？众所周知，不少学科的基础原理都可以归结为一个极值原理，而有了极值原理就不难导出变分原理（泛函极值），如物理中就有最小作用量原理、最小势能原理，概率论中有最大熵原理，等等。如果有一个将变分原理导出黎曼度量的方法，那么就可以用几何的方式来描述它。幸运的是，对于二次型的变分原理，是可以做到的。

### 从作用量原理到黎曼几何

我们来考虑经典力学的最小作用量原理，为了更清晰地说明要义，我们以二维系统为例。一个二维保守系统的运动轨迹，是如下作用量的极值曲线：
$$S = \int \left\{\frac{1}{2}\left[\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2\right]-U(x,y)\right\}dt\tag{70} $$
这里已经假定了$m=1$，导出的运动方程是
$$\frac{d^2 x}{dt^2}=-\frac{\partial U}{\partial x},\quad \frac{d^2 y}{dt^2}=-\frac{\partial U}{\partial y}\tag{71} $$
由于是保守系统，那么满足能量守恒
$$\frac{1}{2}\left[\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2\right] + U=E\tag{72} $$
我们可以用它来消去式$(70)$中的$dt$参数，利用式$(72)$，得到
$$U=E-\frac{1}{2}\left[\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2\right]\tag{73} $$
代入$S$，得到：
$$S = \int \left[\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2-E\right]dt\tag{74} $$
从变分的角度看，$Edt$这一项是全微分，不会带来任何实际效应，因此等效的作用量为
$$S = \int \left[\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2\right]dt=\int \frac{dx^2+dy^2}{dt}\tag{75} $$
再次利用式$(72)$，可以得到
$$dt^2 = \frac{dx^2+dy^2}{2(E-U)}\tag{76} $$
消去$dt$，得到
$$S = \int \sqrt{2(E-U)(dx^2+dy^2)}\tag{77} $$
变分的结果跟常数因子无关，因此最后可以作用量等效于
$$ S = \int \sqrt{(E-U)(dx^2+dy^2)}\tag{78} $$
这个作用量变分得到的结果，就是得到运动曲线的形状（相轨迹）了，而它本身具有黎曼度量的形式，即
$$ds^2 = (E-U)(dx^2+dy^2)\tag{79} $$
结果是一个等温参数。

### 从黎曼几何到运动方程

为了反过来证明这个度量的测地线确实是运动曲线的形状，我们变分式$(78)$，得到
$$\begin{aligned}\delta S =& \int \sqrt{(E-U)(dx^2+dy^2)}\\
=&\int\delta \sqrt{(E-U)(dx^2+dy^2)}\\
=&\int \frac{\delta[(E-U)(dx^2+dy^2)]}{2\sqrt{(E-U)(dx^2+dy^2)}}
\end{aligned}\tag{80} $$
习惯上我们会用自然参数$ds=\sqrt{(E-U)(dx^2+dy^2)}$为参数，但如果使用自然参数，则没法回到经典力学中。因此，这里使用式$(76)$的时间参数，那么
$$\begin{aligned}\delta S =&\int \frac{\delta[(E-U)(dx^2+dy^2)]}{2\sqrt{2}(E-U)dt}\\
=&\int \frac{-\left(\frac{\partial U}{\partial x}\delta x+\frac{\partial U}{\partial y}\delta y\right)(dx^2+dy^2)+2(E-U)(dx d\delta x+dy d\delta y)]}{2\sqrt{2}(E-U)dt}\\
=&\frac{1}{\sqrt{2}}\int \left[-\left(\frac{\partial U}{\partial x}\delta x+\frac{\partial U}{\partial y}\delta y\right)\frac{dx^2+dy^2}{2(E-U)dt}+\left(\frac{dx}{dt} d\delta x+\frac{dy}{dt} d\delta y\right)\right]
\end{aligned}\tag{81} $$
再次利用式$(76)$，然后利用分步积分法，得到
$$\begin{aligned}\delta S \sim& \int \left[-\left(\frac{\partial U}{\partial x}\delta x+\frac{\partial U}{\partial y}\delta y\right)dt+\left(\frac{dx}{dt} d\delta x+\frac{dy}{dt} d\delta y\right)\right]\\
=&\int \left[-\left(\frac{\partial U}{\partial x}\delta x+\frac{\partial U}{\partial y}\delta y\right)dt-\left(\frac{d^2 x}{dt^2} \delta x+\frac{d^2 y}{dt^2} \delta y\right)dt\right]\\
=&-\int \left[\left(\frac{d^2 x}{dt^2}+\frac{\partial U}{\partial x}\right)\delta x dt+\left(\frac{d^2 y}{dt^2}+\frac{\partial U}{\partial y}\right)\delta y dt\right]
\end{aligned}\tag{82} $$
因此$\frac{d^2 x}{dt^2}+\frac{\partial U}{\partial x}=0,\frac{d^2 y}{dt^2}+\frac{\partial U}{\partial y}=0$，重新得到了运动方程$(71)$，这也表明两者确实可以相互转换。

### 一般结果

以上结果可以一般化，即如下作用量的保守系统
$$S = \int \left[\frac{1}{2}g_{\mu\nu} \frac{dx^{\mu}}{dt}\frac{dx^{\nu}}{dt}-U(\boldsymbol{x})\right]dt\tag{83} $$
的能量为$E$的运动曲线的形状（相轨线），等价于黎曼度量
$$ds^2=[E-U(\boldsymbol{x})]g_{\mu\nu}dx^{\mu}dx^{\nu}\tag{84} $$
下的测地线，其推导过程是类似的。这样我们就实现了将力学的问题几何化，或者说将二次型的变分问题几何化了。可能让人意外的是，以上结果在1837年就由雅可比完成了。

上面结果告诉我们，广义相对论不再是黎曼几何在物理中的代名词，即便没有广义相对论，物理中也有黎曼几何。将力学几何化，有助于我们将力学、场论等与几何联系起来。黎曼几何其实就是一套几何的研究框架，只要能够对应地转换，就可以直接应用黎曼几何的很多结论，并且可能导出更丰富、更全面的内容。

### 求解测地线

上述结果不仅仅具有理论价值，有时候还具有实用价值，比如帮助我们求测地线方程。让我们继续考虑等温参数$ds^2 = f(x,y)(dx^2+dy^2)$，在自然参数$ds=\sqrt{f(x,y)(dx^2+dy^2)}$下，它的测地线方程是：
$$\begin{aligned}\frac{d^2 x}{ds^2} =& -\frac{1}{2f}\frac{\partial f}{\partial x}\left(\frac{dx}{ds}\right)^2+\frac{1}{2f}\frac{\partial f}{\partial x}\left(\frac{dy}{ds}\right)^2-\frac{1}{f}\frac{\partial f}{\partial y}\frac{dx}{ds}\frac{dy}{ds}\\
\frac{d^2 y}{ds^2} =& -\frac{1}{2f}\frac{\partial f}{\partial y}\left(\frac{dy}{ds}\right)^2+\frac{1}{2f}\frac{\partial f}{\partial y}\left(\frac{dx}{ds}\right)^2-\frac{1}{f}\frac{\partial f}{\partial x}\frac{dx}{ds}\frac{dy}{ds}\end{aligned}\tag{84}$$
除非一些非常特殊的情况，否则求解这个方程不是容易的事情，哪怕是对于$f(x,y)=\frac{1}{2}(x^2+y^2)$这样特殊情形。然而，根据我们前面所探索的结果，我们知道可以使用时间参数
$$dt=\sqrt{\frac{dx^2+dy^2}{2f(x,y)}}\tag{85}$$
就可以将系统等价于势能$U=-f(x,y)$的保守系统在$E=0$时的相轨迹，也就是测地线方程为
$$\frac{d^2 x}{dt^2}=\frac{\partial f}{\partial x},\quad \frac{d^2 y}{dt^2}=\frac{\partial f}{\partial y}\tag{86}$$
这就大大简化了测地线方程的形式。这时，我们所举的例子$f(x,y)=\frac{1}{2}(x^2+y^2)$时的测地线，只不过是两个已经分离好变量的线性微分方程罢了，完全可解。
