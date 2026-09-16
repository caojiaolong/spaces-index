# 不确定性原理的矩阵形式

> 作者：苏剑林 · 科学空间 · 2014-01-05
>
> 原文：<https://spaces.ac.cn/archives/2242>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

作为量子理论的一个重要定理，不确定性原理总是伴随着物理意义出现的，但是从数学的角度来讲，把不确定性原理的数学形式抽象出来，有助于我们发现更多领域的“不确定性原理”。

本文中，我们将谈及不确定性原理的n维矩阵形式。首先需要解释给大家的是，不确定性原理其实是关于“两个厄密算符与一个单位向量之间的一条不等式”。在量子力学中，厄密算符对应着无穷维的厄密矩阵；而所谓厄密矩阵，就是一个矩阵同时取共轭和转置之后，等于它自身。但是本文讨论一个更简单的情况，那就是n维实矩阵，n维实矩阵中的厄密矩阵就是我们所说的实对称矩阵了。

设$\boldsymbol{x}$是一个$n$维单位向量，即$|\boldsymbol{x}|=1$，而$\boldsymbol{A}$和$\boldsymbol{B}$是n阶实对称矩阵。在量子力学中，$\boldsymbol{x}$就是波函数，但是在这里，它只不过是一个单位实向量；并记$\boldsymbol{I}$是$n$阶单位阵。

考虑
$$\bar{A}=\boldsymbol{x}^{T}\boldsymbol{A}\boldsymbol{x},\bar{B}=\boldsymbol{x}^{T}\boldsymbol{B}\boldsymbol{x}$$
从这些记号可以看出，这些量对应着可观测量的期望值。当然，如果不懂量子力学，可以只看上面的矩阵形式。

然后我们再考虑
$$\begin{aligned} &\left(\Delta A\right)^2=\boldsymbol{x}^{T}\left( \boldsymbol{A}-\bar{A}\boldsymbol{I} \right)^2 \boldsymbol{x}\\
&\left(\Delta B\right)^2=\boldsymbol{x}^{T}\left( \boldsymbol{B}-\bar{B}\boldsymbol{I} \right)^2 \boldsymbol{x}
\end{aligned}$$

由于$\boldsymbol{A}$和$\boldsymbol{B}$是$n$阶实对称矩阵，那么自然有
$$\begin{aligned} &\left(\Delta A\right)^2=\left| \left( \boldsymbol{A}-\bar{A}\boldsymbol{I} \right)\boldsymbol{x}\right|^2\\
&\left(\Delta B\right)^2=\left| \left( \boldsymbol{B}-\bar{B}\boldsymbol{I} \right)\boldsymbol{x}\right|^2
\end{aligned}$$

利用施瓦兹\-柯西不等式，我们有
$$\begin{aligned}
\left(\Delta A\right)^2\left(\Delta B\right)^2 &=\left| \left( \boldsymbol{A}-\bar{A}\boldsymbol{I} \right)\boldsymbol{x}\right|^2\left| \left( \boldsymbol{B}-\bar{B}\boldsymbol{I} \right)\boldsymbol{x}\right|^2\\
&\geq \left| \boldsymbol{x}^{T} \left( \boldsymbol{A}-\bar{A}\boldsymbol{I} \right)\left( \boldsymbol{B}-\bar{B}\boldsymbol{I} \right) \boldsymbol{x} \right|^2\\
&=\left| \boldsymbol{x}^{T} \left( \boldsymbol{A}\boldsymbol{B}\right)\boldsymbol{x}-\left(\boldsymbol{x}^{T}\boldsymbol{A}\boldsymbol{x} \right)\left(\boldsymbol{x}^{T}\boldsymbol{B}\boldsymbol{x}\right)\right|^2
\end{aligned}$$

同样可以证明
$$\begin{aligned}
\left(\Delta A\right)^2\left(\Delta B\right)^2 \geq \left|\left( \boldsymbol{x}^{T}\boldsymbol{A}\boldsymbol{x}\right)\left(\boldsymbol{x}^{T}\boldsymbol{B}\boldsymbol{x}\right)-\boldsymbol{x}^{T} \left( \boldsymbol{B}\boldsymbol{A}\right)\boldsymbol{x} \right|^2
\end{aligned}$$

所以
$$\begin{aligned}
&\left(\Delta A\right)^2\left(\Delta B\right)^2 \\
\geq &\frac{1}{2}\left| \boldsymbol{x}^{T} \left( \boldsymbol{A}\boldsymbol{B}\right)\boldsymbol{x}-\left(\boldsymbol{x}^{T}\boldsymbol{A}\boldsymbol{x} \right)\left(\boldsymbol{x}^{T}\boldsymbol{B}\boldsymbol{x}\right)\right|^2\\
&+\frac{1}{2}\left|\left( \boldsymbol{x}^{T}\boldsymbol{A}\boldsymbol{x}\right)\left(\boldsymbol{x}^{T}\boldsymbol{B}\boldsymbol{x}\right)-\boldsymbol{x}^{T} \left( \boldsymbol{B}\boldsymbol{A}\right)\boldsymbol{x} \right|^2\\
\geq &\frac{1}{4}\left| \boldsymbol{x}^{T} \left( \boldsymbol{A}\boldsymbol{B}\right)\boldsymbol{x}-\boldsymbol{x}^{T} \left( \boldsymbol{B}\boldsymbol{A}\right)\boldsymbol{x} \right|^2\\
=&\frac{1}{4}\left| \boldsymbol{x}^{T} \left( [\boldsymbol{A},\boldsymbol{B}]\right)\boldsymbol{x}\right|^2
\end{aligned}$$

其中
$$[\boldsymbol{A},\boldsymbol{B}]=\boldsymbol{A}\boldsymbol{B}-\boldsymbol{B}\boldsymbol{A}$$
称为换位子、对易式等。以后我们会经常见到这个式子。这里的对易式定义与量子力学中的定义有一点不同。

最后我们就得出了关于两个实对称矩阵和一个单位向量的不等式了。
$$\begin{aligned}
\left(\Delta A\right)\left(\Delta B\right) \geq \frac{1}{2}\left| \boldsymbol{x}^{T} \left( [\boldsymbol{A},\boldsymbol{B}]\right)\boldsymbol{x}\right|
\end{aligned}$$

只要两个向量是不对易的，那么该不等式就告诉了我们一个“不确定性原理”。量子力学做的，是使用了推广到复数域以及无穷维的类似上式的不等式，并赋予其统计意义。
