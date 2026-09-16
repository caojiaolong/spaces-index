# 线性微分方程组：已知特解求通解

> 作者：苏剑林 · 科学空间 · 2014-06-18
>
> 原文：<https://spaces.ac.cn/archives/2644>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

> 含有$n$个一阶常微分方程的一阶常微分方程组
> $$\dot{\boldsymbol{x}}=\boldsymbol{A}\boldsymbol{x}$$
> 其中$\boldsymbol{x}=(x_1(t),\dots,x_n(t))^{T}$为待求函数，而$\boldsymbol{A}=(a_{ij}(t))_{n\times n}$为已知的函数矩阵。现在已知该方程组的$n-1$个线性无关的特解$\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_{n-1}$（解的列向量），求方程的通解。

这是我的一位同学在6月5号问我的一道题目，我当时看了一下，感觉可以通过李对称的方法很容易把解构造出来，当晚就简单分析了一下，发现根据李对称的思想，由上面已知的信息确实足以把通解构造出来。但是我尝试了好几天，尝试了几何、代数等思想，都没有很好地构造出相应的正则变量出来，从而也没有写出它的显式解，于是就搁置下来了。今天再分析这道题目时，竟在无意之间构造出了让我比较满意的解来\~

<strong>可行性分析</strong>

<strong>（没有了解李对称的读者可以直接跳过这部分）</strong>

我们来分析该方程组所拥有的对称，首先，这是一道线性方程组，一个最基本的无穷小生成元是
$$X=\boldsymbol{x}\frac{\partial}{\partial \boldsymbol{x}}$$
由于已经知道$n-1$个特解，根据叠加原理，该方程组还有对称
$$X_i=\boldsymbol{x}_i\frac{\partial}{\partial \boldsymbol{x}}$$
这样我们已经拥有了这个常微分方程组的$n$个对称，并且检验发现这$n$个无穷小生成元正好构成了$n$维可解李代数，从而从这些条件出发，求解这个方程组的通解在理论上是可行的。但是适当的变换却不好找。

<strong>求解过程</strong>

什么样的变换才适合呢？笔者今天发现下面的过程有助于化简原问题。设$\boldsymbol{\xi}$是任意一个与$\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1}$线性无关的列向量（它可以是常向量，也可以是$t$的函数），考虑变换
$$\boldsymbol{x}=[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]\boldsymbol{u}$$
代入原方程
$$\begin{aligned}
\dot{\boldsymbol{x}}& =[\dot{\boldsymbol{x}}_1,\dots,\dot{\boldsymbol{x}}_{n-1},\dot{\boldsymbol{\xi}}]\boldsymbol{u}+[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]\dot{\boldsymbol{u}} \\
&=\boldsymbol{A}[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]\boldsymbol{u}\\
&=[\boldsymbol{A}\boldsymbol{x}_1,\dots,\boldsymbol{A}\boldsymbol{x}_{n-1},\boldsymbol{A}\boldsymbol{\xi}]\boldsymbol{u}\\
&=[\dot{\boldsymbol{x}}_1,\dots,\dot{\boldsymbol{x}}_{n-1},\boldsymbol{A}\boldsymbol{\xi}]\boldsymbol{u}\end{aligned}$$
所以
$$\begin{aligned}\dot{\boldsymbol{u}} &=[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]^{-1}[\boldsymbol{0},\dots,\boldsymbol{0},\boldsymbol{A}\boldsymbol{\xi}-\dot{\boldsymbol{\xi}}]\boldsymbol{u}\\
&=[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]^{-1}\left(\boldsymbol{A}\boldsymbol{\xi}-\dot{\boldsymbol{\xi}}\right)u_n
\end{aligned}$$
注意到右边只含有$u_n$和$t$，若记
$$[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]^{-1}\left(\boldsymbol{A}\boldsymbol{\xi}-\dot{\boldsymbol{\xi}}\right)=\boldsymbol{\eta}=(\eta_1,\eta_2,\dots,\eta_n)^T$$
那么
$$\begin{aligned}
\dot{u}_n&=\eta_n u_n\\
\dot{u}_{n-1}&=\eta_{n-1} u_n\\
\dot{u}_{n-2}&=\eta_{n-2} u_n\\
&\dots\\
\dot{u}_1&=\eta_1 u_n
\end{aligned}$$
积分之，即得原方程的通解。这分两种情况，第一种是易知$u_n\equiv 0$是上述方程组的一个解，继而解得$u_i=c_i,\quad i=1,2,\dots,n-1$，这将得到原来的$n-1$个特解。第二种情况是直接积分得
$$u_n=\exp\left(\int \eta_n(t)dt\right)$$
从而可以逐步积分
$$u_i=\int \eta_i(t)\exp\left(\int \eta_n(t)dt\right)dt,\quad i=1,2,\dots,n-1$$
这将导致与前$n-1$个特解线性无关的新解。（为什么一定线性无关？或者这样理解吧，我每求一个$u_i$，都需要积分一次，每次积分都带有一个独立常数，于是各$u_i$共带有$n$个独立常数！因此我这样求出来的是通解！！通解自然包含部分特解以及新的线性无关的解。）

<strong>结果分析</strong>

由于$\boldsymbol{\xi}$向量的选取是任意的，我们可以根据易知特解的情况选择使得求解尽可能简单的$\boldsymbol{\xi}$。当然，在大多数情况下，我们可以选取
$$\boldsymbol{\xi}=(\underbrace {0, \cdots ,0}_{n - 1\text{个}},1)^T$$
此时，我们发现
$$\boldsymbol{A}\boldsymbol{\xi}=(a_{1n},a_{2n},\dots,a_{nn})^T$$
也就是只有最后一列保留了下来，于是从$\dot{\boldsymbol{u}} =[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]^{-1}[\boldsymbol{0},\dots,\boldsymbol{0},\boldsymbol{A}\boldsymbol{\xi}-\dot{\boldsymbol{\xi}}]\boldsymbol{u}$可以看出，在已知$n-1$个特解的前提下，矩阵$\boldsymbol{A}$的大部分信息是多余的。同时，这也表明$\boldsymbol{A}$中每一列所含的信息都是相等的。

<strong>简单例子</strong>

假设已知微分方程
$$\ddot{x}+a(t)\dot{x}+b(t)x=0$$
的一个非零特解$x=\phi(t)$，求该方程的通解。我们将上述方程改写成一阶方程组
$$\frac{d}{{dt}}\left( {\begin{array}{\cdot {20}{c}}
x_1\\
x_2
\end{array}} \right) = \left[ {\begin{array}{\cdot {20}{c}}
0&1\\
{ - b}&{ - a}
\end{array}} \right]\left( {\begin{array}{\cdot {20}{c}}
x_1\\
x_2
\end{array}} \right)$$
特解是$[x_1,x_2]=[\phi(t),\phi'(t)]$，作变换
$$\left( {\begin{array}{\cdot {20}{c}}
x_1\\
x_2
\end{array}} \right) = \left[ {\begin{array}{\cdot {20}{c}}
\phi &0\\
\phi' &1
\end{array}} \right]\left( {\begin{array}{\cdot {20}{c}}
u_1\\
u_2
\end{array}} \right)$$
因为$\dot{\boldsymbol{u}} =[\boldsymbol{x}_1,\dots,\boldsymbol{x}_{n-1},\boldsymbol{\xi}]^{-1}[\boldsymbol{0},\dots,\boldsymbol{0},\boldsymbol{A}\boldsymbol{\xi}-\dot{\boldsymbol{\xi}}]\boldsymbol{u}$，所以
$$\frac{d}{dt}\left( {\begin{array}{\cdot {20}{c}}
u_1\\
u_2
\end{array}} \right)=\left[ {\begin{array}{\cdot {20}{c}}
\phi &0\\
\phi' &1
\end{array}} \right]^{-1}\left[ {\begin{array}{\cdot {20}{c}}
0 &1\\
0 &-a
\end{array}} \right]\left( {\begin{array}{\cdot {20}{c}}
u_1\\
u_2
\end{array}} \right)$$
即
$$\frac{d}{dt}\left( {\begin{array}{\cdot {20}{c}}
u_1\\
u_2
\end{array}} \right)=\left[ {\begin{array}{\cdot {20}{c}}
0 &1/\phi\\
0 &-a-\phi'/\phi
\end{array}} \right]\left( {\begin{array}{\cdot {20}{c}}
u_1\\
u_2
\end{array}} \right)$$
解得
$$\begin{aligned}
u_2&=\frac{\exp\left(-\int a dt\right)}{\phi}\\
u_1&=\int\frac{u_2}{\phi} dt=\int\frac{\exp\left(-\int a dt\right)}{\phi^2}dt\\
x=x_1&=\phi u_1=\phi\int\frac{\exp\left(-\int a dt\right)}{\phi^2}dt
\end{aligned}$$
