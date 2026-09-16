# 一阶偏微分方程的特征线法

> 作者：苏剑林 · 科学空间 · 2017-12-07
>
> 原文：<https://spaces.ac.cn/archives/4718>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

本文以尽可能清晰、简明的方式来介绍了一阶偏微分方程的特征线法。个人认为这是偏微分方程理论中较为简单但事实上又容易让人含糊的一部分内容，因此尝试以自己的文字来做一番介绍。当然，更准确来说其实是笔者自己的备忘。

## 拟线性情形

### 一般步骤

考虑偏微分方程
$$\begin{equation}\boldsymbol{\alpha}(\boldsymbol{x},u) \cdot \frac{\partial}{\partial \boldsymbol{x}} u = \beta(\boldsymbol{x},u)\end{equation}$$
其中$\boldsymbol{\alpha}$是一个$n$维向量函数，$\beta$是一个标量函数，$\cdot$是向量的点积，$u\equiv u(\boldsymbol{x})$是$n$元函数，$\boldsymbol{x}$是它的自变量。

特征线法的思路是，设想$\boldsymbol{x}$是某个参数$s$的函数，这时候$\boldsymbol{x}(s)$实际上就是某条高维曲线的参数方程，也就是所谓的特征线，这样$u$也成为了参数$s$的函数。并且我们有
$$\begin{equation}\frac{du}{ds}=\frac{\partial u}{\partial \boldsymbol{x}}\cdot\frac{d\boldsymbol{x}}{ds}\end{equation}$$
对比原来的偏微分方程$(1)$，我们发现可以让
$$\begin{equation}\frac{d\boldsymbol{x}}{ds}=\boldsymbol{\alpha}(\boldsymbol{x},u)\end{equation}$$
那么就有
$$\begin{equation}\frac{du}{ds}=\beta(\boldsymbol{x},u)\end{equation}$$
联合$(3)$和$(4)$，我们就得到了一个$n+1$个方程组成的常微分方程组了。
$$\begin{equation}\left\{\begin{aligned}&\frac{d\boldsymbol{x}}{ds}=\boldsymbol{\alpha}(\boldsymbol{x},u)\\
&\frac{du}{ds}=\beta(\boldsymbol{x},u)\end{aligned}\right.\end{equation}$$
由于$s$只是一个额外引入的变量，所以原则上我们可以解得与$s$无关的结果
$$\begin{equation}\boldsymbol{c}=\boldsymbol{f}(\boldsymbol{x},u)\end{equation}$$
其中$\boldsymbol{c}$是一个$n$维向量，是该常微分方程组的积分常数，$\boldsymbol{f}$是$n$维向量函数。剩下的就是根据初值条件来确定各个积分常数之间的关系了。当然，如果要求出一个通解表达式，那就是
$$\begin{equation}G(\boldsymbol{f}(\boldsymbol{x},u))=0\end{equation}$$
其中$G$是任意的$n$元函数，原则上我们可以从中求解得到$u$关于$\boldsymbol{x}$的表达式。

### 简单例子

上面的步骤比较理论化，在实际处理问题中，可以更加灵活些。下面我们来求解
$$\begin{equation}\frac{\partial u}{\partial t} + x \frac{\partial u}{\partial x} = u^2,\quad u(x,0)=f(x)\end{equation}$$
我们得到特征线方程
$$\begin{equation}dt = \frac{dx}{x}=\frac{du}{u^2}\end{equation}$$
求得
$$\begin{equation}x=C_1 e^t, u = \frac{1}{C_2-t}\end{equation}$$
当$t=0$时，有$x=C_1,u=\frac{1}{C_2}=f(C_1)$，从而解得$C_2 = \frac{1}{f(C_1)}$，因为我们有$C_1 = xe^{-t}, C_2 = u^{-1} + t$，所以代入得到
$$\begin{equation}u^{-1} + t = \frac{1}{f(xe^{-t})}\end{equation}$$
即
$$\begin{equation}u = \frac{f(xe^{-t})}{1-t\times f(xe^{-t})} \end{equation}$$

### 一点讨论

特征线究竟是什么含义呢？对于初学者来说，上述过程可能像变魔术一样，先求解出常数，然后消去常数，好像把握不到主干。这就是笔者一开始学习特征线法的困惑。

事实上，我们可以这样认为，<u>特征线已经是偏微分方程的解了，只不过它是在解上的一条线，而完整来说解应该是一个高维的曲面。显然，点动成线、线动成面，想办法让这些线动起来，那么就可以得到这个曲面的方程了，也就是说，各个积分常数$\boldsymbol{c}$要“动起来”</u>。

当然，它们不能毫无约束地乱动，随意动的话，可能就覆盖了整个空间了，该怎么动是取决于初值条件的，因此我们根据初值条件来确定积分常数之间的约束关系。确定完后，我们已经得到了这个曲面的参数方程，从求解的角度，我们没有必要消去各个常数，但很多时候我们更喜欢显式解，所以我们想办法消去常数。

整个过程的逻辑大概就是这样了。

## 一般情形

一般的教材对特征线法的介绍仅限于拟线性偏微分方程，事实上，对于一般的一阶偏微分方程
$$\begin{equation}F\left(\boldsymbol{x}, u, \frac{\partial u}{\partial\boldsymbol{x}}\right)=0\end{equation}$$
特征线法也是适用的，其中$F$是任意多元函数。

这部分工作主要参考自英文维基百科：
[https://en\.wikipedia\.org/wiki/Method\_of\_characteristics](<https://en.wikipedia.org/wiki/Method_of_characteristics>)

### 推导

为此，我们先记
$$\begin{equation}\boldsymbol{p} = \frac{\partial u}{\partial\boldsymbol{x}}\end{equation}$$
然后对$F\left(\boldsymbol{x}, u, \boldsymbol{p}\right)=0$两边求导，得到
$$\begin{equation}\begin{aligned}0 =& \frac{\partial F}{\partial\boldsymbol{x}}\cdot\frac{d\boldsymbol{x}}{ds}+\frac{\partial F}{\partial u}\frac{\partial u}{\partial \boldsymbol{x}}\cdot\frac{d\boldsymbol{x}}{ds}+\frac{\partial F}{\partial\boldsymbol{p}}\cdot\frac{d\boldsymbol{p}}{ds}\\
&=\left(\frac{\partial F}{\partial\boldsymbol{x}}+\frac{\partial F}{\partial u}\boldsymbol{p}\right)\cdot\frac{d\boldsymbol{x}}{ds}+\frac{\partial F}{\partial\boldsymbol{p}}\cdot\frac{d\boldsymbol{p}}{ds}
\end{aligned}\end{equation}$$
可以发现，上面是两组向量的点积，它们加起来为0，那么一个有意思的解决方法就是让
$$\begin{equation}\frac{d\boldsymbol{x}}{ds}=\frac{\partial F}{\partial\boldsymbol{p}},\quad \frac{d\boldsymbol{p}}{ds}=-\frac{\partial F}{\partial\boldsymbol{x}}-\frac{\partial F}{\partial u}\boldsymbol{p}\end{equation}$$
并且我们还有
$$\begin{equation}\frac{du}{ds}=\frac{\partial u}{\partial\boldsymbol{x}}\cdot\frac{d\boldsymbol{x}}{ds}=\boldsymbol{p}\cdot\frac{\partial F}{\partial\boldsymbol{p}}\end{equation}$$
联合起来我们就得到常微分方程组
$$\begin{equation}\left\{\begin{aligned}&\frac{d\boldsymbol{x}}{ds}=\frac{\partial F}{\partial\boldsymbol{p}}\\
&\frac{d\boldsymbol{p}}{ds}=-\frac{\partial F}{\partial\boldsymbol{x}}-\frac{\partial F}{\partial u}\boldsymbol{p}\\
&\frac{du}{ds}=\boldsymbol{p}\cdot\frac{\partial F}{\partial\boldsymbol{p}}\\
&F\left(\boldsymbol{x}, u, \boldsymbol{p}\right)=0\end{aligned}\right.\end{equation}$$
接下来的步骤就跟拟线性情形的没什么差别了，只不过多引入了$n$个变量$\boldsymbol{p}$。解出这个方程来，然后得到$2n$个与$s$积分常数的积分常数，根据初值条件来找出常数之间的关系。不同的是，因为多了$n$个变量$\boldsymbol{p}$，所以要对初值条件也考虑偏导数，这导致了求解过程更加复杂。请看下面的例子。

### 又一个例子

$$\begin{equation}\frac{\partial u}{\partial t} = \left(\frac{\partial u}{\partial x}\right)^2,\quad u(x,0)=f(x)\end{equation}$$
也就是$0=F(\boldsymbol{p})=p_t - p_x^2$，于是根据$(18)$式我们得到特征线方程
$$\begin{equation}\left\{\begin{aligned}&\frac{dt}{ds}=1,\frac{dx}{ds}=-2p_x\\
&\frac{dp_t}{ds}=0,\frac{dp_x}{ds}=0\\
&\frac{du}{ds}=p_t - 2p_x^2 = -p_x^2\end{aligned}\right.\end{equation}$$
直接取$s=t$，然后可以得到$p_t=C_1,p_x=C_2$都是常数，从而$x=-2C_2 t + C_3, u = -C_2^2 t + C_4$。接着根据初值条件，当$t=0$时，有$x=C_3,u=C_4$，这表明$C_4=f(C_3)$，代入后有
$$\begin{equation}u = -C_2^2 t + f(x + 2C_2 t)\end{equation}$$
然后对初值条件的$x$变量求偏导，那么当$t=0$时有
$$\begin{equation}C_2 = p_x = f'(x)=f'(C_3)\end{equation}$$
注意$f$是事先给定的初值函数，因此上式实际上是一个代数方程，所以我们有
$$\begin{equation}\begin{aligned}u =& -t\times f'(C_3)^2 + f\Big(x + 2t\times f'(C_3)\Big)\\
=& -t\times f'(C_3)^2 + f(C_3)
\end{aligned}\end{equation}$$
并且
$$\begin{equation}x=-2C_2 t + C_3=-2t\times f'(C_3) + C_3\end{equation}$$
从中反解出$C_3$并代入前一式就可以得到完整解。比如当$f(x)=x^2$时，解得
$$\begin{equation}C_3=\frac{x}{1-4t}\end{equation}$$
代入得到
$$\begin{equation}u=\frac{x^2}{1-4t}\end{equation}$$

### 自上而下的过程

从形式上来看，一般情形的特征线法跟拟线性情形的特征线法差别很大。那么一般情形的特征线法能不能退化到拟线性情形？

事实上，将$F=\boldsymbol{\alpha}\cdot\boldsymbol{p}-\beta$代入$(18)$式，就可以得到
$$\begin{equation}\left\{\begin{aligned}&\frac{d\boldsymbol{x}}{ds}=\boldsymbol{\alpha}\\
&\frac{du}{ds}=\boldsymbol{p}\cdot\boldsymbol{\alpha}\end{aligned}\right.\end{equation}$$
关于$\boldsymbol{p}$的方程就不必写出来了，因为我们有$\boldsymbol{p}\cdot\boldsymbol{\alpha}=\beta$，代入上式就可以得到封闭的方程组。这就退化为$(5)$式了。

另外，一种特别简单的情形是$F$仅仅是$\boldsymbol{p}$的函数，这时候$\frac{\partial F}{\partial\boldsymbol{x}}$和$\frac{\partial F}{\partial u}$都是0，那么特征线方程中$\boldsymbol{p}$是常数，从而$\boldsymbol{x}$和$u$都是关于$s$的线性函数，整个方程都是完全可解的。之后，就变成了一个纯粹的代数方程问题了。

## 方程组的情形

上述特征线技巧能否推广到一阶偏微分方程组？一般是不可行的，因为解决了一阶偏微分方程组就相当于解决了任意阶的偏微分方程了，显然我们没有看到这样的工作（要是可能的话，早就被人做了）。

不过，要是方程组中偏导算子的部分是一致的话，那么特征线法还是可用的。具体来说，考虑偏微分方程
$$\begin{equation}\left(\boldsymbol{\alpha}(\boldsymbol{x},\boldsymbol{u}) \cdot \frac{\partial}{\partial \boldsymbol{x}}\right) \boldsymbol{u} = \boldsymbol{\beta}(\boldsymbol{x},\boldsymbol{u})\end{equation}$$
这时候$\boldsymbol{u}$也是一个向量，但是左边的偏导数算子是共享的，右边的$\boldsymbol{\beta}$的各个分量可以不一样。这时候也可以用特征线法得到
$$\begin{equation}\left\{\begin{aligned}&\frac{d\boldsymbol{x}}{ds}=\boldsymbol{\alpha}(\boldsymbol{x},u)\\
&\frac{d\boldsymbol{u}}{ds}=\boldsymbol{\beta}(\boldsymbol{x},u)\end{aligned}\right.\end{equation}$$
当然，这只是原来方法的平凡推广。
