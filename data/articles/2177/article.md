# 力学系统及其对偶性（三）

> 作者：苏剑林 · 科学空间 · 2013-11-15
>
> 原文：<https://spaces.ac.cn/archives/2177>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章中，我已经初步地从最小作用量原理的角度来观察对偶定律的表现。虽然那是一种便捷有效的方法，但是还是给我们流下了一些遗憾。上一节是从几何形式的作用量原理出发的，而没有在一般形式的作用量框架下讨论。因为如果在$S=\int Ldt=\int (T-U)dt$的形式下讨论坐标变换问题会出现困难，困难源于我们进行了变换$d\tau=|z|^2 dt$，这导致了时间和空间的耦合，变分不能简单地进行。但是，这并非无法解决的问题。我们还是可以在基本的作用量原理之下讨论变换问题。下面将对此问题进行讨论。

<strong>变分中的变量代换</strong>

考虑一个一般的保守系统的作用量：
$$S=\int_{t_1}^{t_2} L(q,\frac{dq}{dt})dt$$

其变分为
$$\begin{aligned} \delta S&=\int_{t_1}^{t_2} (\delta L) dt\\
&=\int_{t_1}^{t_2} \left[\frac{\partial L}{\partial q} \delta q+\frac{\partial L}{\partial(\frac{dq}{dt})}\delta(\frac{dq}{dt}) \right]dt\\
&=\int_{t_1}^{t_2} \left[\frac{\partial L}{\partial q} \delta q+\frac{\partial L}{\partial(\frac{dq}{dt})}(\frac{d\delta q}{dt}) \right]dt \end{aligned}$$

最后一个等号利用了$\delta(\frac{dq}{dt})=\frac{d\delta q}{dt}$，这是至关重要的，因为只有最右边的变分形式才是有用的，下面的过程是进行分部积分，以得到欧拉\-拉格朗日方程，如果不能转化为最右边的形式，就无法进行分部积分了。至于导出欧拉\-拉格朗日方程的过程，在此不再重复。我们要考虑在变换$dt=f(q,\frac{dq}{dt})d\tau$之下作用量的形式。此时
$$S=\int_{t_1}^{t_2} Ldt=\int_{\tau_1}^{\tau_2} (Lf)d\tau$$

假如$Lf$已经写成了以$q,\frac{dq}{d\tau}$为变量的形式，那么是否可以直接将$Lf$代入欧拉\-拉格朗日方程，从而得到变换后的运动方程呢？显然没有这么简单，答案是否定的，我们来验算它。假设可以直接将$Lf$代入欧拉\-拉格朗日方程，那么相应地有
$$\delta S=\int_{\tau_1}^{\tau_2} \delta(Lf)d\tau=\int_{\tau_1}^{\tau_2} \left[\frac{\partial (Lf)}{\partial q} \delta q+\frac{\partial (Lf)}{\partial(\frac{dq}{d\tau})}(\frac{d\delta q}{d\tau})\right]dt$$

同时可以计算
$$\delta S=\int_{\tau_1}^{\tau_2} \delta(Lf)d\tau=\int_{\tau_1}^{\tau_2} (f\delta L +L \delta f)d\tau$$

让我们把精力集中在$\delta L$上，我们知道：
$$\delta L=\frac{\partial L}{\partial q} \delta q+\frac{\partial L}{\partial(\frac{dq}{dt})}\delta(\frac{dq}{dt})$$

基于我们的假设，我们有$\delta(\frac{dq}{d\tau})=\frac{d\delta q}{d\tau}$，但是$\delta(\frac{dq}{dt})$和$\frac{d\delta q}{dt}$的相等性是未知的。事实上它们不相等，因为
$$\begin{aligned} \delta(\frac{dq}{dt})&=\delta(\frac{dq}{d\tau}\frac{1}{f})\\
&=\frac{1}{f}\delta(\frac{dq}{d\tau})-\frac{\delta f}{f^2}\frac{dq}{d\tau}\\
&=\frac{1}{f}(\frac{d\delta q}{d\tau})-\frac{\delta f}{f^2}\frac{dq}{d\tau}\end{aligned}$$

所以其实
$$\begin{aligned} \delta S &=\int_{\tau_1}^{\tau_2} \left[ \left(\frac{\partial L}{\partial q} \delta q+\frac{\partial L}{\partial(\frac{dq}{dt})}(\frac{d\delta q}{dt}) \right)f d\tau- \left(\frac{\partial L}{\partial (\frac{dq}{dt})}\frac{dq}{dt}-L \right)(\delta f)d\tau\right]\\
&=\int_{\tau_1}^{\tau_2} \left[ \left(\frac{\partial L}{\partial q} \delta q+\frac{\partial L}{\partial(\frac{dq}{dt})}(\frac{d\delta q}{dt}) \right)f d\tau-H(\delta f)d\tau\right]\end{aligned}$$

其中$H$为原系统的哈密顿函数。如果假设成立，则必须要求$H=0$ ，即系统的能量为0，这并非总是成立的，因此假设是不成立的。不过，这不意味着就给我们“判了死刑”。由于原系统是一个保守系统，能量是守恒的，即存在积分$H=E$。我们知道，$L\mapsto L+E$之下原系统的运动方程保持不变，但可以使得$H\mapsto H-E=0$。因此这里为我们找到了一个解，即拉格朗日量$L$的在变量$q,t$下的运动方程等价于拉格朗日量$\tilde{L}=f(L+E)$在变量$q,\tau$的运动方程，其中$dt=f d\tau$。

<strong>广义对偶</strong>

以上内容参考了文献The Kustaanheimo\-Stiefel transformation in geometric algebra而写。从以上过程可以看出，当变分中的变量代换是纯粹的时间\-时间、空间\-空间变换时，结论是很简单的，只需要将新拉格朗日函数代入欧拉\-拉格朗日方程即可；但是当变分出现时间\-空间耦合时必须小心对待，因为这意味着出现更多的复杂性。在此，仅称变换前后的两个保守系统互为广义对偶系统。

如果$f$只显含q，可以得到更显式的解。新系统的哈密顿函数为
$$\begin{aligned}\tilde{H} &=\frac{\partial f(L+E)}{\partial (\frac{dq}{d\tau})}\frac{dq}{d\tau}-f(L+E)\\
&=\frac{\partial f(L+E)}{\partial (f\frac{dq}{dt})}\frac{fdq}{dt}-f(L+E)\\
&=f(H-E) \end{aligned}$$

可见，当我们进行巧妙的限制和变换时，一切可以恢复到简单的形式。这便是数学物理的魅力了，在复杂中隐藏中简洁，在简洁中也包含着各种复杂性。顺便提一下，路径积分可以看作最小作用量原理的量子版本，最小作用量存在着对偶变换，可以猜测路径积分会存在类似的东西。事实上，的确存在中一类变换，可以对路径积分进行简化，然而，经典版本的最小作用量变分已经如此复杂了，路径积分的变换显然只会更加复杂。有机会的话，我会好好写写路径积分的内容的。
