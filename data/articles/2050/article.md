# 势能最小问题的探讨

> 作者：苏剑林 · 科学空间 · 2013-08-19
>
> 原文：<https://spaces.ac.cn/archives/2050>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

本文我们来探讨下列积分的极值曲线：
$$S=\int f(x,y)\sqrt{dx^2+dy^2}=\int f(x,y)ds$$

这本质上也是一个短程线问题。但是它形式比较简答，物理含义也更加明显。比如，如果$f(x,y)$是势函数的话，那么这就是一个求势能最小的二维问题；如果$f(x,y)$是摩擦力函数，那么这就是寻找摩擦力最小的路径问题。不管是哪一种，该问题都有相当的实用价值。下面将其变分：

$$\begin{aligned} \delta S =&\int \delta[f(x,y)\sqrt{dx^2+dy^2}] \\ =&\int [ds\delta f(x,y)+f(x,y)\frac{\delta (dx^2+dy^2)}{2ds}]\\ =&\int ds(\frac{\partial f}{\partial x}\delta x+\frac{\partial}{\partial y}\delta y)+f \frac{dx d(\delta x)+dy d(\delta y)}{ds} \\=&\int ds(\frac{\partial f}{\partial x}\delta x+\frac{\partial}{\partial y}\delta y)+f \frac{dx}{ds} d(\delta x)+\frac{dy}{ds} d(\delta y) \end{aligned}$$

对右边的$d(\delta x)$和$d(\delta y)$项使用分部积分法，即
$$\delta S=f \frac{dx}{ds} (\delta x)+\frac{dy}{ds} (\delta y)+\int ds(\frac{\partial f}{\partial x}\delta x+\frac{\partial}{\partial y}\delta y)-d(f \frac{dx}{ds})(\delta x)-\frac{dy}{ds} (\delta y) $$

<strong>按照边界条件，我们是从过两定点的所有曲线中挑选出适当的一条</strong>，因此$f \frac{dx}{ds} (\delta x)+\frac{dy}{ds} (\delta y)=0$，因为在边界处有$\delta x=0$即$\delta y=0$。让我们再把上式整理一下：
$$\delta S=\int [ds(\frac{\partial f}{\partial x})-d(f\frac{dx}{ds})]\delta x+[ds(\frac{\partial f}{\partial y})-d(f\frac{dy}{ds})]\delta y $$

由于对于任意的$\delta x$和$\delta y$上式都成立，因此有：
$$\begin{eqnarray*} ds(\frac{\partial f}{\partial x})-d(f\frac{dx}{ds})=0 \\ ds(\frac{\partial f}{\partial y})-d(f\frac{dy}{ds})=0 \end{eqnarray*}$$

或者
$$\begin{eqnarray*} \frac{d}{ds}(f\frac{dx}{ds})=\frac{\partial f}{\partial x} \\ \frac{d}{ds}(f\frac{dy}{ds})=\frac{\partial f}{\partial y} \end{eqnarray*}$$

这是看起来比较简洁的方程，其中$ds^2=dx^2+dy^2$，但是这个关系也可以不预先给定，这可以作为上述方程组的积分之一。因为我们有：
$$\begin{eqnarray*} (f\frac{dx}{ds})\frac{d}{ds}(f\frac{dx}{ds})=f\frac{\partial f}{\partial x}\frac{dx}{ds} \\ (f\frac{dy}{ds})\frac{d}{ds}(f\frac{dy}{ds})=f\frac{\partial f}{\partial y}\frac{dy}{ds} \end{eqnarray*}$$

两式相加得到：
$$\begin{eqnarray*} (f\frac{dx}{ds})\frac{d}{ds}(f\frac{dx}{ds})+(f\frac{dy}{ds})\frac{d}{ds}(f\frac{dy}{ds})=f\frac{\partial f}{\partial x}\frac{dx}{ds}+f\frac{\partial f}{\partial y}\frac{dy}{ds} \end{eqnarray*}$$

于是
$$\frac{d}{ds}[f^2 (\frac{dx}{ds})^2+f^2 (\frac{dy}{ds})^2]=\frac{d}{ds}(f^2)$$

即
$$f^2 [(\frac{dx}{ds})^2+(\frac{dy}{ds})^2 -1]=C$$

而$ds^2=dx^2+dy^2$对应于$C=0$，也就是说，在求解过程中我们完全可以把s看成与x,y无关的参数。

然而，简单的方程不一定意味着简单的解。一个很清楚的事实是：大多数曲线的方程的弧长都没有简单的表达式。也就是说，即使是很简单的曲线（比如抛物线$y=x^2$），它的弧长（周长）跟x,y本身的关系也是很复杂的，原因在于$s=\int \sqrt{1+(y')^2}dx$这里出现了根号，因此使用s作为参数时，我们很难会得到简洁的解析解。因此，求解这部分工作大多数都是计算机完成的，对于计算机来说，这些方程并不困难。事实上也正是如此，能够解析求解的只有非常有限的情况。
