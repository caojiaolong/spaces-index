# 路径积分系列：5\.例子和综述

> 作者：苏剑林 · 科学空间 · 2016-06-09
>
> 原文：<https://spaces.ac.cn/archives/3766>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

路径积分方法为解决某些随机问题带来了新视角\.

### 一个例子：股票价格模型

考虑有风险资产(如股票)，在$t$时刻其价格为$S_t$，考虑的时间区间为$[0,T]$，0表示初始时间，$T$表示为到期日\. $S_t$看作是随时间变化的连续时间变量，并服从下列随机微分方程:
$$dS_t^0=rS_t^0 dt;\quad dS_t=S_t(\mu dt+\sigma dW_t).\tag{70}$$
其中，$\mu$和$\sigma$是两个常量，$W_t$是一个标准布朗运动\.

关于$S_t$的方程是一个随机微分方程，一般解决思路是通过随机微积分\. 随机微积分有别于一般的微积分的地方在于，随机微积分在做一阶展开的时候，不能忽略$dS_t^2$项，因为$dW_t^2=dt$\. 比如，设$S_t=e^{x_t}$，则$x_t=\ln S_t$
$$\begin{aligned}dx_t=&\ln(S_t+dS_t)-\ln S_t=\frac{dS_t}{S_t}-\frac{dS_t^2}{2S_t^2}\\
=&\frac{S_t(\mu dt+\sigma dW_t)}{S_t}-\frac{[S_t(\mu dt+\sigma dW_t)]^2}{2S_t^2}\\
=&\mu dt+\sigma dW_t-\frac{1}{2}\sigma^2 dW_t^2\quad(\text{其余项均低于}dt\text{阶})\\
=&\left(\mu-\frac{1}{2}\sigma^2\right) dt+\sigma dW_t\end{aligned}
,\tag{71}$$
这就将它转化为了$(48)$的形式\. 根据前面的研究，它可以等价于一个不对称的随机游走模型，这样我们可以进行数值模拟；或者根据$(12)$写出等价的偏微分方程
$$\sigma\frac{\partial P}{\partial t}=\frac{\sigma^2}{2}\frac{\partial^2 P}{\partial x^2}+\left(\mu-\frac{1}{2}\sigma^2\right)\frac{\partial P}{\partial x},\tag{72}$$
或者等价于路径积分
$$\int_{x_a}^{x_b}\exp\left\{-\frac{1}{2\sigma}\int_{t_a}^{t_b}\left[\dot{x}-\left(\mu-\frac{1}{2}\sigma^2\right)\right]^2 dt\right\}\mathscr{D}x(t).\tag{73}$$

该问题的路径积分是二次型的，因而可以精确求解\. 答案是
$$\begin{aligned}&\exp\left[-\frac{1}{2\sigma}\frac{(x_b-x_a)^2}{t_b-t_a}-\frac{1}{2\sigma}\left(\mu-\frac{1}{2}\sigma^2\right)^2(t_b-t_a)\right.\\
&\qquad\qquad\qquad\left.+\frac{1}{\sigma}\left(\mu-\frac{1}{2}\sigma^2\right)(x_b-x_a)\right]\\
=&\exp\left\{-\frac{t_b-t_a}{2\sigma}\left[\frac{x_b-x_a}{t_b-t_a}-\left(\mu-\frac{1}{2}\sigma^2\right)\right]^2\right\}
\end{aligned},\tag{74}$$
可见，它只与相对的值$T=t_b-t_a$和$\Delta x = x_b-x_a$有关：
$$\exp\left\{-\frac{T}{2\sigma}\left[\frac{\Delta x}{T}-\left(\mu-\frac{1}{2}\sigma^2\right)\right]^2\right\},\tag{75}$$
已经强调过，这是在相差一个归一化因子的情况下的结果，通过归一化，得到的完整结果是
$$P(\Delta x)=\frac{1}{\sqrt{2\pi \sigma T}}\exp\left\{-\frac{T}{2\sigma}\left[\frac{\Delta x}{T}-\left(\mu-\frac{1}{2}\sigma^2\right)\right]^2\right\}.\tag{76}$$
注意，这是$x_t$的分布，我们要分析的是$S_t$的分布，通过变量代换$\Delta x=\ln S_b - \ln S_a = \ln(S_b/S_a)$得到
$$P(S_b)=\frac{1}{S_b\sqrt{2\pi \sigma T}}\exp\left\{-\frac{T}{2\sigma}\left[\frac{\ln (S_b/S_a)}{T}-\left(\mu-\frac{1}{2}\sigma^2\right)\right]^2\right\}.\tag{77}$$
这是一个对数正态分布\. 它告诉我们如果股票现在的价值是$S_a$，那么经过$T$时间后，价值为$S_b$的可能性为$P(S_b)$\.

很多金融问题可以用随机微分方程描述，而随机微分方程可以转化为对应的偏微分方程或者路径积分，路径积分源于量子力学，因此近年来两者结合产生了一个新兴的领域——量子金融，或者叫量子经济学\. 但它实际上就是通过路径积分方法，把量子力学领域的结论，搬到金融领域，从数学的角度看，并没有什么实质的新的内容，然而从实用的角度，它节省了研究时间和研究成本，是非常有意义的\. 关于这方面的著作有《Quantum Finance》&#91;10&#93;\.

### 论文综述

本文通过一些说明和例子，展示了路径积分方法一大类随机问题中的应用\.

然而，本文的论述并不完备，关于路径积分方法在这方面的应用，还有很多值得研究的方向：

> 1、对于高阶的非线性的随机场微分方程，找出相应的路径积分，并且为其找到类似不对称随机游走的模型；
> 2、研究随机偏微分方程所对应的路径积分；
> 3、为随机偏微分方程寻找类似不对称随机游走的模型\.

兹认为以上的这些工作都将会相当有意义\.
