# 【外微分浅谈】5\. 几何意义

> 作者：苏剑林 · 科学空间 · 2016-11-06
>
> 原文：<https://spaces.ac.cn/archives/4062>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

对于前面所述的外微分，包括后面还略微涉及到的微分形式的积分，都是纯粹代数定义的内容，本身不具有任何的几何意义。但是，我们可以将某些公式或者定义，与一些几何内容对应起来，使我们更深刻地理解它，并且更灵活运用它。但是，它仅仅是一种对应，而且取决于我们的诠释。比如，我们说外微分公式
$$\int_{\partial D} Pdx+Qdy = \int_{D} \left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dx\land dy \tag{32} $$
对应于格林公式
$$\int_{\partial D} Pdx+Qdy = \int_{D} \left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right)dxdy \tag{33} $$
。这是没问题的，但它们并不等价，它们仅仅是形式上刚好一样。因为格林公式是描述闭合曲线的积分跟面积分的联系，而外微分的公式是一种纯粹的代数运算。因为你完全可以将$dx\land dy$对应于$-dxdy$而不是$dxdy$，这样就得到另外一种几何的对应。

更深刻的问题是：为什么恰好有这个对应？也就是说，为什么经过一些调整和诠释后，就能够得到与积分公式的对应？首先要明确的是外积与普通的数的乘积，除了反对称性之外，是没有任何区别的，因此不少性质得以保留；其次，还应该要回到反对称本身来考虑，矩阵的行列式代表着矩阵所对应的向量组张成的$n$维立体的体积，然而行列式是反对称的，这就意味着反对称运算跟体积、积分等有着先天的联系。当然，更细致的认识，笔者也还没做到。

此外，我们说寻求微分形式的几何意义，通常只是针对不超过3维的空间来讨论的，更高维的几何图像我们很难想象出来，尤其是高维的曲面积分，一般只是类比，但类比是否成立，有时还需要进一步商榷。因此，这种情况下，倒不如干脆点，说微分形式描述的东西就是几何，而不再去寻找所谓的几何意义了。也就是说，反过来，将微分形式和外微分作为公理式的第一性原理来定义几何。

甚至，你可以只将外微分当作是一种记忆各种微分、积分公式的有效途径，比如现在我要大家默写三维空间中的斯托克斯公式，大家估计会乱，因为不一定记得是哪个减哪个。但是在外微分框架下，可以很快地将它推导一遍。好比式$(11)$，如果非要寻求几何解释，那就是开普勒第二定律：单位时间内扫过的面积相等；然而没有几何解释，你依旧可以把方程解下去。

下面将围绕着几何诠释进行展开。

### 外积：张成并投影

我们考虑两个微分1形式的外积，比如
$$\begin{aligned}\alpha_{\mu}dx^{\mu} \land \beta_{\nu}dx^{\nu} =&\alpha_{\mu}\beta_{\nu} dx^{\mu}\land dx^{\nu}\\
=&\sum_{\mu < \nu} (\alpha_{\mu}\beta_{\nu}-\beta_{\mu}\alpha_{\nu}) dx^{\mu}\land dx^{\nu} \\
=&\frac{1}{2}(\alpha_{\mu}\beta_{\nu}-\beta_{\mu}\alpha_{\nu}) dx^{\mu}\land dx^{\nu}\end{aligned} \tag{34} $$
省略求和符号时，表示$\mu,\nu$各自分别无约束地遍历求和。留意到
$$\alpha_{\mu}\beta_{\nu}-\beta_{\mu}\alpha_{\nu}=\det\begin{pmatrix}\alpha_{\mu}&\alpha_{\nu}\\\beta_{\mu}&\beta_{\nu}\end{pmatrix} \tag{35} $$
那么，如果将$dx^{\mu}$看成基，那么对于选定的一对$\mu,\nu$，$\alpha_{\mu}\beta_{\nu}-\beta_{\mu}\alpha_{\nu}$正好对应于向量$\alpha_{\mu}$和向量$\beta_{\nu}$所张成的平行四边形在$dx^{\mu},dx^{\nu}$平面上的投影的有向面积。

对于一般的微分$p$形式和微分$q$形式，它们的外积可以类似地构造，只是高维的更难想象罢了。比如一个微分1形式跟一个微分2形式作外积，可以想象着一个普通向量跟一个“面积向量”（实际上是一个张量）张成了一个立方体，外积结果的每一项，就是该立方体在相应的三维子空间上的投影的体积，等等。特别地，如果在$n$维空间中，有$n$个微分1形式作外积，结果将是
$$\alpha_{\mu_1}^{1} dx^{\mu_1}\land \dots \land \alpha_{\mu_n}^{n} dx^{\mu_n}=\det(\alpha_{\mu}^{\nu}) dx^1 \land \dots \land dx^n \tag{36} $$
即刚好产生了一个矩阵的行列式，这是很神奇的，正好是反对称性的体现。反对称性也存在也行列式中，即交换行列式的两行或者两列，那么行列式反号。设$f$是任意函数，我们有$df=\frac{\partial f}{\partial x^{\mu}} dx^{\mu}$，那么
$$df^1 \land \dots \land df^n = \det\left(\frac{\partial f^{\mu}}{\partial x^{\nu}}\right) dx^1 \land \dots \land dx^n \tag{37} $$
从变换的角度来看，$\det\left(\frac{\partial f^{\mu}}{\partial x^{\nu}}\right)$就是积分变换的雅可比行列式。这使得我们想冲动地把$\land$忽略掉，把$dx^1 \land \dots \land dx^n$直接看成积分元$dx^1\dots dx^n$。事实上正是这样做的！我们定义$dx^{\mu_1}\land \dots \land dx^{\mu_k}=\pm dx^{\mu_1}\dots dx^{\mu_k}$，至于是正还是负，取决于我们想将它诠释为什么具体的几何内容。这样我们就可以用外微分来表示积分理论了。

### 微分算子：绕圈子

更值得深刻认识的是式$(25)$，即从一个$p$形式到$p+1$形式，究竟发生了什么，或者说，对应什么几何内容。

我们还是从微分1形式出发，考虑$\omega_{\nu} dx^{\nu}$，在算符$d$作用下，有
$$\begin{aligned}d(\omega_{\nu} dx^{\nu}) =& \frac{\partial \omega_{\nu}}{\partial x^{\mu}} dx^{\mu} \land dx^{\nu}\\
=&\sum_{\mu < \nu} \left(\frac{\partial \omega_{\nu}}{\partial x^{\mu}}-\frac{\partial \omega_{\mu}}{\partial x^{\nu}}\right) dx^{\mu} \land dx^{\nu}\\
=&\frac{1}{2} \left(\frac{\partial \omega_{\nu}}{\partial x^{\mu}}-\frac{\partial \omega_{\mu}}{\partial x^{\nu}}\right) dx^{\mu} \land dx^{\nu}\end{aligned} \tag{38} $$
这样的形状的东西有什么几何对应呢？我们可以把$\omega_{\nu} dx^{\nu}$看成是量$\Omega$从$x$到$x+dx$的增量，即
$$\Omega (x+dx) = \Omega (x) + \omega_{\nu}(x) dx^{\nu} \tag{39} $$
那么，如果再从$x+dx$到$x+dx+\delta x$呢？自然有
$$\begin{aligned}\Omega_1 (x+dx+\delta x) =& \Omega (x+dx) + \omega_{\nu}(x+dx) \delta x^{\nu}\\
=&\Omega (x) + \omega_{\nu}(x) dx^{\nu} + \omega_{\nu}(x) \delta x^{\nu} + \frac{\partial \omega_{\nu}}{\partial x^{\mu}} dx^{\mu} \delta x^{\nu} \end{aligned} \tag{40} $$
这是走了$x\to x+dx\to x+dx+\delta x$这条路径的，交换$dx$和$\delta x$，即走$x\to x+\delta x\to x+\delta x+ dx$这条路径，得到
$$\Omega_2 (x+dx+\delta x) =\Omega (x) + \omega_{\nu}(x) \delta x^{\nu} + \omega_{\nu}(x) d x^{\nu} + \frac{\partial \omega_{\nu}}{\partial x^{\mu}} \delta x^{\mu} d x^{\nu} \tag{41} $$
两者之差
$$\label{bihecha}\begin{aligned}\left(\frac{\partial \omega_{\nu}}{\partial x^{\mu}}-\frac{\partial \omega_{\mu}}{\partial x^{\nu}}\right)dx^{\mu} \delta x^{\nu}=&\frac{1}{2}\left(\frac{\partial \omega_{\nu}}{\partial x^{\mu}}-\frac{\partial \omega_{\mu}}{\partial x^{\nu}}\right)(dx^{\mu} \delta x^{\nu}-dx^{\nu} \delta x^{\mu})\\
=&\frac{1}{2}\left(\frac{\partial \omega_{\nu}}{\partial x^{\mu}}-\frac{\partial \omega_{\mu}}{\partial x^{\nu}}\right) \det\begin{pmatrix}dx^{\mu} & \delta x^{\mu}\\ dx^{\nu} & \delta x^{\nu}\end{pmatrix}\end{aligned} \tag{42} $$
就是绕着闭合路径$x\to x+dx\to x+dx+\delta x\to x+\delta x\to x$溜达了一圈之后，所产生的变化量。

如果令
$$dx^{\mu} \land dx^{\nu} = \det\begin{pmatrix}dx^{\mu} & \delta x^{\mu}\\ dx^{\nu} & \delta x^{\nu}\end{pmatrix} \tag{43} $$
那么式$(42)$正好是$d(\omega_{\nu} dx^{\nu})$。而$\det\begin{pmatrix}dx^{\mu} & \delta x^{\mu}\\ dx^{\nu} & \delta x^{\nu}\end{pmatrix}$就是$dx$和$\delta x$这两个向量，张成的平行四边形在$x^{\mu},x^{\nu}$平面的投影的面积，它也是反对称的。从这个角度看，就可以将$dx^{\mu} \land dx^{\nu}$解释为有向面积元，而$d(\omega_{\nu} dx^{\nu})$的含义就是一个量在绕了一个小圈子回来之后的变化量！

### 微积分基本定理

通过绕圈子的途径，我们解释了从微分1形式到2形式的含义。但遗憾的是，从一般的$p$形式到$p+1$形式，并不那么容易想象，而且，事实上对于超过3维的空间中的积分的几何图像，我们也很难想象出来。所以，我们这里使用了一条“本末倒置”的路径。如果$\omega$是一个微分$p$形式，$D$是一给定区域，那么
$$\int_{\partial D} \omega = \int_{D} d\omega \tag{44} $$
也就是说，$\omega$在边界上的积分，等于$d\omega$在区域内的积分，这就是微分形式中的“斯托克斯公式”（Stokes公式），也可以说是外微分中的微积分基本定理。

这个公式为人称颂的地方就是统一了牛顿\-莱布尼兹公式、格林公式、高斯定理、斯托克斯公式，并将其一般化。大家可能困惑“什么是微分形式的积分？”，事实上微分形式的积分并没有什么特殊的地方，因为诸如$dx^{\mu}\land dx^{\nu}$的式子除了反对称之外，跟普通的微分元$dx^{\mu}dx^{\nu}$没有什么区别，而积分的定义（比如简单地采用黎曼积分定义）与对称还是反对称无关。

这样，我们就能够想象，从一般的$p$形式到$p+1$形式，或者说从$\omega$到$d\omega$，事实上跟1形式到2形式一样，做了类似“绕圈子”的事情，即把$\omega$理解为边界上的运动变化，而$d\omega$则是遍历一个小区域回来之后，所产生的变化，那么很自然在封闭区域$D$就有$\int_{\partial D} \omega = \int_{D} d\omega$了。

当然，前面已经说了，这是一条“本末倒置”的途径。这个积分定理其实是“结果”而非“原因”，它需要冗长的证明。而我们这里不加证明地引用了它，反过来用来解释$d\omega$的含义，这只是为了给大家一条尽快并且尽可能清晰的思路。
