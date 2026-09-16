# 差分方程的摄动法

> 作者：苏剑林 · 科学空间 · 2016-08-04
>
> 原文：<https://spaces.ac.cn/archives/3889>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近一直在考虑一些自然语言处理问题和一些非线性分析问题，无暇总结发文，在此表示抱歉。本文要说的是对于一阶非线性差分方程（当然高阶也可以类似地做）的一种摄动格式，理论上来说，本方法可以得到任意一阶非线性差分方程的显式渐近解。

### 非线性差分方程

对于一般的一阶非线性差分方程
$$\begin{equation}\label{chafenfangcheng}x_{n+1}-x_n = f(x_n)\end{equation}$$
通常来说，差分方程很少有解析解，因此要通过渐近分析等手段来分析非线性差分方程的性质。很多时候，我们首先会考虑将差分替换为求导，得到微分方程
$$\begin{equation}\label{weifenfangcheng}\frac{dx}{dn}=f(x)\end{equation}$$
作为差分方程$\eqref{chafenfangcheng}$的近似。其中的原因，除了微分方程有比较简单的显式解之外，另一重要原因是微分方程$\eqref{weifenfangcheng}$近似保留了差分方程$\eqref{chafenfangcheng}$的一些比较重要的性质，如渐近性。例如，考虑离散的阻滞增长模型：
$$\begin{equation}\label{zuzhizengzhang}x_{n+1}=(1+\alpha)x_n -\beta x_n^2\end{equation}$$
对应的微分方程为（差分替换为求导）：
$$\begin{equation}\frac{dx}{dn}=\alpha x -\beta x^2\end{equation}$$
此方程解得
$$\begin{equation}x_n = \frac{\alpha}{\beta+c e^{-\alpha n}}\end{equation}$$
其中$c$是任意常数。上述结果已经大概给出了原差分方程$\eqref{zuzhizengzhang}$的解的变化趋势，并且成功给出了最终的渐近极限$x_n \to \frac{\alpha}{\beta}$。下图是当$\alpha=\beta=1$且$c=1$（即$x_0=\frac{1}{2}$）时，微分方程的解与差分方程的解的值比较。

[![差分方程的摄动法1](<https://spaces.ac.cn/usr/uploads/2016/08/3693064820.png>)](<https://spaces.ac.cn/usr/uploads/2016/08/3693064820.png>)

差分方程的摄动法1

现在的问题是，既然微分方程的解可以作为一个形态良好的近似解了，那么是否可以在微分方程的解的基础上，进一步加入修正项提高精度？

### 一般格式

我们引入参数$q$，考虑更一般的差分方程
$$\begin{equation}\label{yibanchafenfangcheng}x_{n+q}-x_n = q f(x_n)\end{equation}$$
当$q=1$时正是我们考虑的差分方程$\eqref{chafenfangcheng}$，而当$q\to 0$时则是微分方程
$$\begin{equation}\frac{dx}{dn}=f(x)\end{equation}$$
要注意，我们引入了$q$，因此$x_n$的具体形式跟$q$有关，换言之，$x_n$实际上也是$q$的函数，我们显式地写为$x_{n,q}$，即
$$\begin{equation}\label{yibanchafenfangcheng2}x_{n+q,q}-x_{n,q}=qf(x_{n,q})\end{equation}$$
而我们最终要关心的是$x_n\equiv x_{n,1}$。为了逐步求解该方程，我们设想能够将$x_{n,q}$展开为$q$的幂级数
$$\begin{equation}x_{n,q}=\sum_{k=0}^{\infty}a^{(k)}_n q^n\end{equation}$$
其中
$$\begin{equation}a_n^{(k)} = \frac{1}{k!}\left.\frac{\partial^k x_{n,q}}{\partial q^k}\right|_{q=0},\quad k=0,1,2,\dots\end{equation}$$
现在的问题是各$a_n^{(k)}$的求法。

### 逐步展开

在$\eqref{yibanchafenfangcheng2}$式两边对$q$求导（对$q$的全导数），得到
$$\begin{equation}\label{qiudao}\frac{\partial x_{n+q,q}}{\partial n}+\frac{\partial x_{n+q,q}}{\partial q}-\frac{\partial x_{n,q}}{\partial q}=f(x_{n,q})+q\frac{\partial f}{\partial x_{n,q}}\frac{\partial x_{n,q}}{\partial q}\end{equation}$$
要说明的是，当我们写出
$$\begin{equation}\frac{\partial x_{n+q,q}}{\partial q}\end{equation}$$
时，默认只对第二个变量$q$求导，而不对第一个变量$n+q$中的$q$求导。现在取$q=0$，则
$$\begin{equation}\label{qiudao-q-0}\frac{\partial x_{n,0}}{\partial n}=f(x_{n,0})\end{equation}$$
这正好是作为首次近似的微分方程$\eqref{weifenfangcheng}$。

接着，对$\eqref{qiudao}$再次求$q$的导数
$$\begin{equation}\label{qiudao2}\begin{aligned}&\frac{\partial^2 x_{n+q,q}}{\partial n^2}+2\frac{\partial^2 x_{n+q,q}}{\partial n \partial q}+\frac{\partial^2 x_{n+q,q}}{\partial q^2}-\frac{\partial^2 x_{n,q}}{\partial q^2}\\
=&2\frac{\partial f}{\partial x_{n,q}}\frac{\partial x_{n,q}}{\partial q}+q\frac{\partial^2 f}{\partial x_{n,q}^2}\left(\frac{\partial x_{n,q}}{\partial q}\right)^2+q\frac{\partial f}{\partial x_{n,q}}\frac{\partial^2 x_{n,q}}{\partial q^2}\end{aligned}\end{equation}$$
让$q=0$，得到
$$\begin{equation}\label{qiudao2-q-0}\frac{\partial^2 x_{n,0}}{\partial n^2}+2\frac{\partial}{\partial n}\left(\frac{\partial x_{n,0}}{\partial q}\right)=2\frac{\partial f}{\partial x_{n,0}}\frac{\partial x_{n,0}}{\partial q}\end{equation}$$
其中$\frac{\partial^2 x_{n,0}}{\partial n^2}$项可以代入上一步得到的$x_{n,0}$进行计算，因此这是关于$\frac{\partial x_{n,0}}{\partial q}$的一阶线性微分方程，也是可解的。这个过程可以一直做下去，逐步提高精度。后面的形式太烦，就不写出来了，可以用mathematica来做。

### 离散阻滞增长模型

还是具体的例子会比较容易说清楚过程。仍然考虑阻滞增长模型$\eqref{zuzhizengzhang}$，引入参数$q$后，方程写为
$$\begin{equation}x_{n+q,q}-x_{n,q}=q\alpha x_{n,q}-q\beta x_{n,q}^2\end{equation}$$
这里$f(x)=\alpha x - \beta x^2$，对应的$\eqref{qiudao-q-0}$式为
$$\begin{equation}\frac{\partial x_{n,0}}{\partial n}=\alpha x_{n,0}-\beta x_{n,0}^2\end{equation}$$
解已经求出，是
$$\begin{equation}x_{n,0}=\frac{\alpha}{\beta+c e^{-\alpha n}}\end{equation}$$
这是零阶近似。相应地，对应的$\eqref{qiudao2-q-0}$式为
$$\begin{equation}\frac{\partial^2 x_{n,0}}{\partial n^2}+2\frac{\partial}{\partial n}\left(\frac{\partial x_{n,0}}{\partial q}\right)=2(\alpha-2\beta x_{n,0})\frac{\partial x_{n,0}}{\partial q}\end{equation}$$
解是（我是用mathematica求的，经过人工整理，这里已经让$\frac{\partial x_{0,0}}{\partial q}=0$）
$$\begin{equation}\frac{\partial x_{n,0}}{\partial q}=\left(\frac{\alpha}{\beta+ce^{-\alpha n}}\right)^2\left(\frac{1}{2}\alpha n-\ln \frac{\beta+c}{\beta+ce^{-\alpha n}}\right)ce^{-\alpha n}\end{equation}$$
因此有
$$\begin{equation}x_{n,q}\approx \frac{\alpha}{\beta+c e^{-\alpha n}}+q\left(\frac{\alpha}{\beta+ce^{-\alpha n}}\right)^2\left(\frac{1}{2}\alpha n-\ln \frac{\beta+c}{\beta+ce^{-\alpha n}}\right)ce^{-\alpha n}\end{equation}$$
以及
$$\begin{equation}x_n\equiv x_{n,1}\approx \frac{\alpha}{\beta+c e^{-\alpha n}}+\left(\frac{\alpha}{\beta+ce^{-\alpha n}}\right)^2\left(\frac{1}{2}\alpha n-\ln \frac{\beta+c}{\beta+ce^{-\alpha n}}\right)ce^{-\alpha n}\end{equation}$$
这是它的一阶近似。下图是当$\alpha=\beta=1$且$c=1$（即$x_0=\frac{1}{2}$）时，上述零阶和一阶近似解与精确值的比较。可以看到，相比于零阶近似，一阶近似几乎与精确值重合了。

[![差分方程的摄动法2](<https://spaces.ac.cn/usr/uploads/2016/08/39010436.png>)](<https://spaces.ac.cn/usr/uploads/2016/08/39010436.png>)

差分方程的摄动法2

从这个例子也可以看出摄动法的特点，虽然可以继续做下去，且每一步都是可积的，但是积分可能有用过于复杂而难以使用。

### 一点说明

理论上来说，这种摄动格式可以求任意一阶差分方程的近似解。当然，在真正应用中是否真的如此实用，还得具体问题具体分析。

首先，如果作为零阶项的微分方程（非线性的）的解过于复杂，那这种方法就很难进行下去了；其次，虽然每一步都只是产生了一个理论可解的一阶非齐次线性微分方程，但这个微分方程的解也不是一个简单的事情，往往多做一步耗费的时间和精力都是加倍的。

最后，要注意，引入参数的格式并不仅仅是$\eqref{yibanchafenfangcheng2}$那样的，还可以根据具体问题做简化，比如文章[《一个非线性差分方程的隐函数解》](<https://spaces.ac.cn/archives/3696/>)中的例子
$$\begin{equation}x_{n+1}=x_n+3+\frac{3}{x_n}+\frac{1}{x_n^2},\,x_1=1\end{equation}$$
引入参数的方式可以是
$$\begin{equation}x_{n+q,q}=x_{n,q}+3+\frac{3q}{x_{n,q}}+\frac{q^2}{x_{n,q}^2},\,x_1=1\end{equation}$$
这样会得到与文章[《一个非线性差分方程的隐函数解》](<https://spaces.ac.cn/archives/3696/>)中的一样的渐近解。这体现了摄动法的本质，就是通过参数$q$来辨明每一项的阶，从而能够逐级进行近似。而以什么为阶，跟以什么为参数一样，很大程度上都是人为的。
