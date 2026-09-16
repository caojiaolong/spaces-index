# 测试函数法推导连续性方程和Fokker\-Planck方程

> 作者：苏剑林 · 科学空间 · 2023-02-11
>
> 原文：<https://spaces.ac.cn/archives/9461>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《生成扩散模型漫谈（六）：一般框架之ODE篇》](<https://spaces.ac.cn/archives/9228>)中，我们推导了SDE的Fokker\-Planck方程；而在[《生成扩散模型漫谈（十二）：“硬刚”扩散ODE》](<https://spaces.ac.cn/archives/9280>)中，我们单独推导了ODE的连续性方程。它们都是描述随机变量沿着SDE/ODE演化的分布变化方程，连续性方程是Fokker\-Planck方程的特例。在推导Fokker\-Planck方程时，我们将泰勒展开硬套到了狄拉克函数上，虽然结果是对的，但未免有点不伦不类；在推导连续性方程时，我们结合了雅可比行列式和泰勒展开，方法本身比较常规，但没法用来推广到Fokker\-Planck方程。

这篇文章我们介绍“测试函数法”，它是推导连续性方程和Fokker\-Planck方程的标准方法之一，其分析过程比较正规，并且适用场景也比较广。

## 分部积分

正式推导之前，这里先介绍后面推导会用到的关键结果——分部积分法的高维推广。

一般教程对分部积分的介绍仅限于一维情形，即
\begin{equation}\int_a^b uv'dx = uv|_a^b - \int_a^b vu'dx\end{equation}
这里$u,v$是$x$的函数，$'$表示求函数关于$x$的导数。下面我们需要它的一个高维版本，为此，我们先回顾一维分部积分的推导过程，它依赖于求导的乘法法则：
\begin{equation}(uv)' = uv' + vu'\end{equation}
然后两边对$x$积分并移项，就得到分部积分公式。对于高维情形，我们考虑类似的公式：
\begin{equation}\nabla\cdot(u\boldsymbol{v}) = \boldsymbol{v}\cdot\nabla u + u\nabla \cdot\boldsymbol{v}\end{equation}
其中$u$是$\boldsymbol{x}$的标量函数，$\boldsymbol{v}$是$\boldsymbol{x}$的向量函数（维度跟$\boldsymbol{v}$一致），$\nabla$表示求函数关于$\boldsymbol{x}$的梯度。现在我们对两端在区域$\Omega$积分：
\begin{equation}\int_{\Omega}\nabla\cdot(u\boldsymbol{v})d\boldsymbol{x} = \int_{\Omega}\boldsymbol{v}\cdot\nabla u d\boldsymbol{x} + \int_{\Omega}u\nabla \cdot\boldsymbol{v} d\boldsymbol{x}\end{equation}
根据[高斯散度定理](<https://en.wikipedia.org/wiki/Divergence_theorem>)，左侧等于$\int_{\partial\Omega}u\boldsymbol{v}\cdot\hat{\boldsymbol{n}}dS$，$\partial\Omega$是$\Omega$的边界，$\hat{\boldsymbol{n}}$是边界的外向单位法向量，$dS$是面积微元。所以，移项后有
\begin{equation}\int_{\Omega}\boldsymbol{v}\cdot\nabla u d\boldsymbol{x} = \int_{\partial\Omega}u\boldsymbol{v}\cdot\hat{\boldsymbol{n}}dS - \int_{\Omega}u\nabla \cdot\boldsymbol{v} d\boldsymbol{x}\label{eq:int-by-parts}\end{equation}
这就是我们要推导的高维空间分部积分公式。特别地，对于概率密度函数$p$，那么由于非负性和积分为1的限制，无穷远处必然有$p\to 0$和$\nabla p\to \boldsymbol{0}$，所以如果$\Omega$选为全空间（没有特别注明积分区域的，默认为全空间），那么分别将$u=p$和$\boldsymbol{v}=\nabla p$代入上式，得到
\begin{align}\int\boldsymbol{v}\cdot\nabla p d\boldsymbol{x} =&\, - \int p\nabla \cdot\boldsymbol{v} d\boldsymbol{x}\label{eq:int-by-parts-p} \\
\int u\nabla \cdot\nabla p d\boldsymbol{x} = &\,-\int\nabla p\cdot\nabla u d\boldsymbol{x}\label{eq:int-by-parts-gp}\end{align}
如果要进一步严格化上述结论，可以假设$p$具有紧的支撑集。不过这纯粹是数学上的严格化，事实上对于一般理解来说，直接默认在无穷远处成立$p\to 0$和$\nabla p\to \boldsymbol{0}$就够了。

## ODE演化

测试函数法的原理，是如果对于任意函数$\phi(\boldsymbol{x})$，都成立
\begin{equation}\int f(\boldsymbol{x})\phi(\boldsymbol{x})d\boldsymbol{x} = \int g(\boldsymbol{x})\phi(\boldsymbol{x})d\boldsymbol{x}\end{equation}
那么就成立$f(\boldsymbol{x})=g(\boldsymbol{x})$，其中$\phi(\boldsymbol{x})$就叫做测试函数。更严谨的定义需要声明$\phi(\boldsymbol{x})$的选取空间，以及等号的具体含义（如严格相等/几乎处处相等/依概率相等之类），这里我们就不引入这些细节了。

对于ODE
\begin{equation}\frac{d\boldsymbol{x}_t}{dt}=\boldsymbol{f}_t(\boldsymbol{x}_t)\label{eq:ode}\end{equation}
我们将它离散化为
\begin{equation}\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t\label{eq:ode-diff}\end{equation}
那么就有
\begin{equation}\phi(\boldsymbol{x}_{t+\Delta t}) = \phi(\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t)\Delta t)\approx \phi(\boldsymbol{x}_t)  + \Delta t\,\,\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)\end{equation}
两边求期望，得到：
\begin{equation}\int p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})\phi(\boldsymbol{x}_{t+\Delta t}) d\boldsymbol{x}_{t+\Delta t}\approx \int p_t(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t  + \Delta t\int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\end{equation}
由于积分的结果不依赖于被积自变量的记号，所以左边将$\boldsymbol{x}_{t+\Delta t}$换成$\boldsymbol{x}_t$也是等价的：
\begin{equation}\int p_{t+\Delta t}(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t\approx \int p_t(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t  + \Delta t\int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\label{eq:change-var}\end{equation}
将右边第一项移到左边，然后取$\Delta t\to 0$的极限，得到：
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\label{eq:dt-0}\end{equation}
右端利用分部积分公式$\eqref{eq:int-by-parts-p}$得到
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = -\int \Big[\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)\Big]\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\end{equation}
根据测试函数法的相等原理，就有
\begin{equation}\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} = -\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)\end{equation}
这称为“[连续性方程](<https://en.wikipedia.org/wiki/Continuity_equation>)”。

## SDE演化

对于SDE
\begin{equation}d\boldsymbol{x}_t = \boldsymbol{f}_t(\boldsymbol{x}_t) dt + g_t d\boldsymbol{w}\label{eq:sde}\end{equation}
我们离散化为
\begin{equation}\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:sde-diff}\end{equation}
那么
\begin{equation}\begin{aligned}
\phi(\boldsymbol{x}_{t+\Delta t}) =&\, \phi(\boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}) \\
\approx&\, \phi(\boldsymbol{x}_t) + \left(\boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}\right)\cdot \nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) + \frac{1}{2} \left(g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}\cdot \nabla_{\boldsymbol{x}_t}\right)^2\phi(\boldsymbol{x}_t)
\end{aligned}\end{equation}
两边求期望，注意右边要同时对$\boldsymbol{x}_t$和$\boldsymbol{\varepsilon}$求期望，其中$\boldsymbol{\varepsilon}$的期望可以事先求出，结果是
\begin{equation}\phi(\boldsymbol{x}_t) + \Delta t\,\,\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot \nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) + \frac{1}{2} \Delta t\,g_t^2\nabla_{\boldsymbol{x}_t}\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)
\end{equation}
于是
\begin{equation}\begin{aligned}
&\,\int p_{t+\Delta t}(\boldsymbol{x}_{t+\Delta t})\phi(\boldsymbol{x}_{t+\Delta t}) d\boldsymbol{x}_{t+\Delta t}\\
\approx&\, \int p_t(\boldsymbol{x}_t)\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t  + \Delta t\int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \int\frac{1}{2} \Delta t\,g_t^2 p_t(\boldsymbol{x}_t)\nabla_{\boldsymbol{x}_t}\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t
\end{aligned}\end{equation}
跟式$\eqref{eq:change-var}$、式$\eqref{eq:dt-0}$类似，取$\Delta\to 0$的极限，得到
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t + \int\frac{1}{2} \,g_t^2 p_t(\boldsymbol{x}_t)\nabla_{\boldsymbol{x}_t}\cdot\nabla_{\boldsymbol{x}_t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t\end{equation}
对右边第一项应用式$\eqref{eq:int-by-parts-p}$、对右边第二项先应用式$\eqref{eq:int-by-parts-gp}$再应用式$\eqref{eq:int-by-parts-p}$，得到
\begin{equation}\int \frac{\partial p_t(\boldsymbol{x}_t)}{\partial t}\phi(\boldsymbol{x}_t) d\boldsymbol{x}_t = \int \left[-\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)+\frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\right]\phi(\boldsymbol{x}_t)d\boldsymbol{x}_t\end{equation}
根据测试函数法的相等原理得
\begin{equation}\frac{\partial p_t(\boldsymbol{x}_t)}{\partial t} = -\nabla_{\boldsymbol{x}_t}\cdot\big(p_t(\boldsymbol{x}_t)\boldsymbol{f}_t(\boldsymbol{x}_t)\big)+\frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\end{equation}
这就是“[Fokker\-Planck方程](<https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation>)”。

## 文章小结

本文介绍了用于推导某些概率方程的测试函数法，主要内容包括分部积分法的高维推广，以及ODE的连续性方程和SDE的Fokker\-Planck方程的推导过程。
