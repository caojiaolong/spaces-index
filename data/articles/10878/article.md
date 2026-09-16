# SVD的导数

> 作者：苏剑林 · 科学空间 · 2025-04-26
>
> 原文：<https://spaces.ac.cn/archives/10878>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

SVD（Singular Value Decomposition，奇异值分解）是常见的矩阵分解算法，相信很多读者都已经对它有所了解，此前我们在[《低秩近似之路（二）：SVD》](<https://spaces.ac.cn/archives/10407>)也专门介绍过它。然而，读者是否想到，SVD竟然还可以求导呢？笔者刚了解到这一结论时也颇感意外，因为直觉上“分解”往往都是不可导的。但事实是，SVD在一般情况下确实可导，这意味着理论上我们可以将SVD嵌入到模型中，并用基于梯度的优化器来端到端训练。

问题来了，既然SVD可导，那么它的导函数长什么样呢？接下来，我们将参考文献[《Differentiating the Singular Value Decomposition》](<https://j-towns.github.io/papers/svd-derivative.pdf>)，逐步推导SVD的求导公式。

## 推导基础

假设$\boldsymbol{W}$是满秩的$n\times n$矩阵，且全体奇异值两两不等，这是比较容易讨论的情形，后面我们也会讨论哪些条件可以放宽一点。接着，我们设$\boldsymbol{W}$的SVD为：
\begin{equation}\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\end{equation}
所谓SVD求导，实际上就是设法分别求出$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}$关于$\boldsymbol{W}$的梯度或微分。为此，我们先对上式两边求微分
\begin{equation}d\boldsymbol{W} = (d\boldsymbol{U})\boldsymbol{\Sigma}\boldsymbol{V}^{\top} + \boldsymbol{U}(d\boldsymbol{\Sigma})\boldsymbol{V}^{\top} + \boldsymbol{U}\boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\end{equation}
左乘$\boldsymbol{U}^{\top}$、右乘$\boldsymbol{V}$，并利用$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}, \boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}$得到
\begin{equation}\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V} = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}\label{eq:core}\end{equation}
这便是后面推导的基础。注意对恒等式$\boldsymbol{U}^{\top}\boldsymbol{U} = \boldsymbol{I}, \boldsymbol{V}^{\top}\boldsymbol{V} = \boldsymbol{I}$两端求微分，我们还可以得到
\begin{equation}(d\boldsymbol{U})^{\top}\boldsymbol{U} + \boldsymbol{U}^{\top}(d\boldsymbol{U}) = \boldsymbol{0},\quad (d\boldsymbol{V})^{\top}\boldsymbol{V} + \boldsymbol{V}^{\top}(d\boldsymbol{V}) = \boldsymbol{0}\end{equation}
这表明$\boldsymbol{U}^{\top}(d\boldsymbol{U})$和$(d\boldsymbol{V})^{\top}\boldsymbol{V}$都是反对称矩阵。

## 奇异值

反对称矩阵的特点是对角线元素都是零，而$\boldsymbol{\Sigma}$是对角阵，非对角线元素都是零，这启发我们可能需要分别处理处理对角线和非对角线元素。

首先，我们定义矩阵$\boldsymbol{I}$和$\bar{\boldsymbol{I}}$：$\boldsymbol{I}$就是单位阵，即对角线元素全是1，非对角线元素全是0；$\bar{\boldsymbol{I}}$则是单位阵的互补阵，即对角线元素全是0，而非对角线元素全是1。利用$\boldsymbol{I}$、$\bar{\boldsymbol{I}}$和Hadamard积$\otimes$，我们可以分别提取出式$\eqref{eq:core}$的对角线和非对角线部分：
\begin{align}
\boldsymbol{I}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) =&\, \boldsymbol{I}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}) = d\boldsymbol{\Sigma} \label{eq:value} \\[8pt]
\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) =&\, \bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}) = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}\label{eq:vector}
\end{align}
现在我们先来看式$\eqref{eq:value}$，它可以等价地写成
\begin{equation}d\sigma_i = \boldsymbol{u}_i^{\top}(d\boldsymbol{W})\boldsymbol{v}_i\label{eq:d-sigma}\end{equation}
这就是第$i$个奇异值$\sigma_i$的微分，其中$\boldsymbol{u}_i,\boldsymbol{v}_i$分别是$\boldsymbol{U},\boldsymbol{V}$的第$i$列。[《从谱范数梯度到新式权重衰减的思考》](<https://spaces.ac.cn/archives/10648>)中讨论的谱范数梯度，实际上就只是这里$i=1$时的特例。

## 奇异向量

然后我们再来看式$\eqref{eq:vector}$：
\begin{equation}\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}) = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V}\label{eq:vector-1}\end{equation}
转置一下
\begin{equation}\begin{aligned}
\bar{\boldsymbol{I}}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}) =&\, \boldsymbol{\Sigma}(d\boldsymbol{U})^{\top}\boldsymbol{U} + \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} \\[6pt]
=&\, -\boldsymbol{\Sigma}\boldsymbol{U}^{\top}(d\boldsymbol{U}) - (d\boldsymbol{V})^{\top}\boldsymbol{V}\boldsymbol{\Sigma}
\end{aligned}\label{eq:vector-2}\end{equation}
第二个等号利用了“$\boldsymbol{U}^{\top}(d\boldsymbol{U})$和$(d\boldsymbol{V})^{\top}\boldsymbol{V}$都是反对称矩阵”这一事实。式$\eqref{eq:vector-1}$和式$\eqref{eq:vector-2}$就是关于$d\boldsymbol{U},d\boldsymbol{V}$的线性方程组，我们要从中解出$d\boldsymbol{U},d\boldsymbol{V}$。

求解思路就是普通的消元法。首先，由$\eqref{eq:vector-1}\times\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\times\eqref{eq:vector-2}$得到
\begin{equation}\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}) = \boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma}^2 - \boldsymbol{\Sigma}^2\boldsymbol{U}^{\top}(d\boldsymbol{U})\end{equation}
这里利用了对角阵$\boldsymbol{\Sigma}$满足$\boldsymbol{\Sigma}(\bar{\boldsymbol{I}}\otimes \boldsymbol{M}) = \bar{\boldsymbol{I}}\otimes (\boldsymbol{\Sigma}\boldsymbol{M})$以及$(\bar{\boldsymbol{I}}\otimes \boldsymbol{M})\boldsymbol{\Sigma} = \bar{\boldsymbol{I}}\otimes (\boldsymbol{M}\boldsymbol{\Sigma})$的事实。我们知道，左（右）乘一个对角阵，等于矩阵的每一行（列）都乘以对角阵上相应的元素，所以如果我们定义矩阵$\boldsymbol{E}$，其中$\boldsymbol{E}_{i,j} = \sigma_j^2 - \sigma_i^2$，那么$\boldsymbol{U}^{\top}(d\boldsymbol{U})\boldsymbol{\Sigma}^2 - \boldsymbol{\Sigma}^2\boldsymbol{U}^{\top}(d\boldsymbol{U}) = \boldsymbol{E}\otimes (\boldsymbol{U}^{\top}(d\boldsymbol{U}))$，于是上式可以写成
\begin{equation}\bar{\boldsymbol{I}}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}) = \boldsymbol{E}\otimes (\boldsymbol{U}^{\top}(d\boldsymbol{U}))\label{eq:dU-0}\end{equation}
继而可以解得
\begin{equation}d\boldsymbol{U} = \boldsymbol{U}(\boldsymbol{F}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}))\label{eq:dU}\end{equation}
类似地，由$\boldsymbol{\Sigma}\times \eqref{eq:vector-1} + \eqref{eq:vector-2}\times \boldsymbol{\Sigma}$解得：
\begin{equation}d\boldsymbol{V} = \boldsymbol{V}(\boldsymbol{F}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}))\label{eq:dV}\end{equation}
式$\eqref{eq:dU},\eqref{eq:dV}$便是特征向量的微分。其中
\begin{equation}\boldsymbol{F}_{i,j} = \left\{\begin{aligned}
&\, 1/(\sigma_j^2 - \sigma_i^2), &\, i\neq j \\
&\, 0, &\, i = j
\end{aligned}\right.\end{equation}

## 梯度（一）

微分有了，怎么把梯度求出来呢？这还真有点麻烦。倒不是技术上的麻烦，而是表示上的麻烦，比如$\boldsymbol{W},\boldsymbol{U}$是一个$n\times n$矩阵，那么$\boldsymbol{U}$关于$\boldsymbol{W}$的完整梯度就是一个$n\times n\times n\times n$的4阶张量，而高阶张量是大多数人包括笔者都不熟悉的内容。

为了绕开高阶张量的麻烦，我们有两个方案。首先，从编程角度看，我们根本没必要求出梯度的形式，而是根据微分的结果写出等价的前向形式，然后交给框架的自动求导就行。比如，从式$\eqref{eq:d-sigma}$我们可以断定$\sigma_i$的梯度等于$\newcommand{\sg}[1]{\color{skyblue}{#1}} \sg{\boldsymbol{u}_i}^{\top} \boldsymbol{W} \sg{\boldsymbol{v}_i}$的梯度，即
\begin{equation}\nabla_{\boldsymbol{W}} \sigma_i = \nabla_{\boldsymbol{W}} (\sg{\boldsymbol{u}_i}^{\top} \boldsymbol{W} \sg{\boldsymbol{v}_i})\end{equation}
这里将符号颜色改为$\sg{\blacksquare}$色代表stop\_gradient算子，避免公式过于臃肿。刚好$\sigma_i$又等于$\boldsymbol{u}_i^{\top}\boldsymbol{W}\boldsymbol{v}_i$，所以我们只需要把代码中出现$\sigma_i$的地方，都替换成$\sg{\boldsymbol{u}_i}^{\top} \boldsymbol{W} \sg{\boldsymbol{v}_i}$，那么就可以自动获得正确的梯度。一般地，我们有
\begin{equation}\nabla_{\boldsymbol{W}} \boldsymbol{\Sigma} = \nabla_{\boldsymbol{W}} (\boldsymbol{I}\otimes(\sg{\boldsymbol{U}}^{\top} \boldsymbol{W} \sg{\boldsymbol{V}}))\end{equation}
即所有$\boldsymbol{\Sigma}$换成$\boldsymbol{I}\otimes(\sg{\boldsymbol{U}}^{\top} \boldsymbol{W} \sg{\boldsymbol{V}})$即可。

同理，从式$\eqref{eq:dU}$我们知道
\begin{equation}\nabla_{\boldsymbol{W}}\boldsymbol{U} = \nabla_{\boldsymbol{W}}(\sg{\boldsymbol{U}}(\sg{\boldsymbol{F}}\otimes(\sg{\boldsymbol{U}}^{\top}\boldsymbol{W}\sg{\boldsymbol{V}\boldsymbol{\Sigma}} + \sg{\boldsymbol{\Sigma}\boldsymbol{V}}^{\top}\boldsymbol{W}^{\top}\sg{\boldsymbol{U}})))\end{equation}
可以验证$\boldsymbol{U}(\boldsymbol{F}\otimes(\boldsymbol{U}^{\top}\boldsymbol{W}\boldsymbol{V}\boldsymbol{\Sigma} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}\boldsymbol{W}^{\top}\boldsymbol{U}))$刚好是零矩阵，所以我们只需要将代码中所有出现$\boldsymbol{U}$的地方，都替换成
\begin{equation}\boldsymbol{U} \quad \to \quad \sg{\boldsymbol{U}} + \sg{\boldsymbol{U}}(\sg{\boldsymbol{F}}\otimes(\sg{\boldsymbol{U}}^{\top}\boldsymbol{W}\sg{\boldsymbol{V}\boldsymbol{\Sigma}} + \sg{\boldsymbol{\Sigma}\boldsymbol{V}}^{\top}\boldsymbol{W}^{\top}\sg{\boldsymbol{U}}))\end{equation}
那就能保持正确的前向结果，同时获得正确的梯度。基于同样的原理，$\boldsymbol{V}$的替换格式是
\begin{equation}\boldsymbol{V} \quad \to \quad \sg{\boldsymbol{V}} + \sg{\boldsymbol{V}}(\sg{\boldsymbol{F}}\otimes(\sg{\boldsymbol{V}}^{\top}\boldsymbol{W}^{\top}\sg{\boldsymbol{U}\boldsymbol{\Sigma}} + \sg{\boldsymbol{\Sigma}\boldsymbol{U}}^{\top}\boldsymbol{W}\sg{\boldsymbol{V}}))\end{equation}

## 梯度（二）

第二个方案是直接求出损失函数关于$\boldsymbol{W}$的梯度。具体来说，假设损失函数是$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}$的函数，记为$\mathcal{L}(\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V})$，我们直接求$\nabla_{\boldsymbol{W}}\mathcal{L}$，它是一个矩阵，可以用$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V},\nabla_{\boldsymbol{U}}\mathcal{L},\nabla_{\boldsymbol{\Sigma}}\mathcal{L},\nabla_{\boldsymbol{V}}\mathcal{L}$表示出来，这些量也都只是矩阵，所以不用涉及到高阶张量。

在上一节中，我们已经求出了具有相同梯度的$\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}$的等效函数，除开被stop\_gradient的部分外，这些等效函数关于$\boldsymbol{W}$都是线性的，所以此时问题本质上已经变成了线性复合函数的梯度。我们之前在[《低秩近似之路（一）：伪逆》](<https://spaces.ac.cn/archives/10366#%E7%9F%A9%E9%98%B5%E6%B1%82%E5%AF%BC>)的“矩阵求导”一节便已经讨论过相关方法。具体来说，我们有
\begin{align}
\boldsymbol{X} = \boldsymbol{A}\boldsymbol{B}\boldsymbol{C} &\,\quad\Rightarrow\quad \nabla_{\boldsymbol{B}}f(\boldsymbol{X}) = \boldsymbol{A}^{\top}(\nabla_{\boldsymbol{X}}f(\boldsymbol{X}))\boldsymbol{C}^{\top} \\[8pt]
\boldsymbol{X} = \boldsymbol{A}\boldsymbol{B}^{\top}\boldsymbol{C} &\,\quad\Rightarrow\quad \nabla_{\boldsymbol{B}}f(\boldsymbol{X}) = \boldsymbol{C}(\nabla_{\boldsymbol{X}}f(\boldsymbol{X}))^{\top}\boldsymbol{A} \\[8pt]
\boldsymbol{X} = \boldsymbol{A}\otimes\boldsymbol{B} &\,\quad\Rightarrow\quad \nabla_{\boldsymbol{B}}f(\boldsymbol{X}) = \boldsymbol{A}\otimes \nabla_{\boldsymbol{X}}f(\boldsymbol{X})
\end{align}
利用这些基本公式，以及复合函数求导的链式法则，我们可以写出
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{W}}\mathcal{L} \quad = \qquad &\,\boldsymbol{U}(\boldsymbol{F}\otimes(\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{U}}\mathcal{L}) - (\nabla_{\boldsymbol{U}}\mathcal{L})^{\top}\boldsymbol{U}))\boldsymbol{\Sigma}\boldsymbol{V}^{\top} \\[6pt]
+ &\,\boldsymbol{U}(\boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L}))\boldsymbol{V}^{\top} \\[6pt]
+ &\,\boldsymbol{U}\boldsymbol{\Sigma}(\boldsymbol{F}\otimes(\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L}) - (\nabla_{\boldsymbol{V}}\mathcal{L})^{\top}\boldsymbol{V})))\boldsymbol{V}^{\top}
\end{aligned}\end{equation}
整个过程就是反复地利用基本公式和链式法则，以及$\boldsymbol{F}^{\top} = -\boldsymbol{F}$，原则上没有难度，就是需要谨慎地集中注意力，建议读者亲自动手完成这个推导过程，这是一道相当实用的矩阵求导练习题。最后引入两个记号
\begin{equation}\newcommand{\sym}[1]{\color{red}{[}#1\color{red}{]_{sym}}} \newcommand{\skew}[1]{\color{red}{[}#1\color{red}{]_{skew}}}  \sym{\boldsymbol{X}} = \frac{1}{2}(\boldsymbol{X} + \boldsymbol{X}^{\top}),\qquad \skew{\boldsymbol{X}} = \frac{1}{2}(\boldsymbol{X} - \boldsymbol{X}^{\top})\end{equation}
我们可以将梯度结果简写成
\begin{equation}\nabla_{\boldsymbol{W}}\mathcal{L} = \boldsymbol{U}\Big(2(\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{U}}\mathcal{L})})\boldsymbol{\Sigma} + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L}) + 2\boldsymbol{\Sigma}(\boldsymbol{F}\otimes\skew{\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L})})\Big)\boldsymbol{V}^{\top}
\label{eq:w-grad-l}\end{equation}

## 梯度（三）

现在可以牛刀小试一下，求$\boldsymbol{O}=\mathop{\text{msign}}(\boldsymbol{W})=\boldsymbol{U}\boldsymbol{V}^{\top}$的梯度，其中$\mathop{\text{msign}}$的概念我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)介绍Muon时已经讨论过。

根据$\mathop{\text{msign}}$的定义，我们有
\begin{equation}\nabla_{\boldsymbol{U}}\mathcal{L} = (\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V},\qquad \nabla_{\boldsymbol{V}}\mathcal{L} = (\nabla_{\boldsymbol{O}}\mathcal{L})^{\top} \boldsymbol{U}\end{equation}
代入式$\eqref{eq:w-grad-l}$得到
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{W}}\mathcal{L} =&\, 2\boldsymbol{U}\Big((\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}})\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(\boldsymbol{F}\otimes\skew{\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})^{\top}\boldsymbol{U}})\Big)\boldsymbol{V}^{\top} \\[6pt]
=&\, 2\boldsymbol{U}\Big((\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}})\boldsymbol{\Sigma} - \boldsymbol{\Sigma}(\boldsymbol{F}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}})\Big)\boldsymbol{V}^{\top} \\[6pt]
=&\, 2\boldsymbol{U}\big(\boldsymbol{G}\otimes\skew{\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}}\big)\boldsymbol{V}^{\top}
\end{aligned}\end{equation}
其中$\boldsymbol{G}_{i,j} = 1/(\sigma_i + \sigma_j)$。

## 梯度（四）

最后让我们来考虑一个常用的特殊例子，当$\boldsymbol{W}$还是正定对称矩阵时，它的SVD具有$\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$的形式，即$\boldsymbol{U}=\boldsymbol{V}$。我们重复前面的推导，先对两边求微分
\begin{equation}d\boldsymbol{W} = (d\boldsymbol{V})\boldsymbol{\Sigma}\boldsymbol{V}^{\top} + \boldsymbol{V}(d\boldsymbol{\Sigma})\boldsymbol{V}^{\top} + \boldsymbol{V}\boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\end{equation}
然后左乘$\boldsymbol{V}^{\top}$、右乘$\boldsymbol{V}$：
\begin{equation}\begin{aligned}
\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V} =&\, \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} + \boldsymbol{\Sigma}(d\boldsymbol{V})^{\top}\boldsymbol{V} \\[6pt]
=&\, \boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} + d\boldsymbol{\Sigma} - \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{V})
\end{aligned}\end{equation}
由此可见
\begin{gather}
d\boldsymbol{\Sigma} = \boldsymbol{I}\otimes(\boldsymbol{V}(d\boldsymbol{W})\boldsymbol{V}^{\top}) \\[8pt]
\boldsymbol{V}^{\top}(d\boldsymbol{V})\boldsymbol{\Sigma} - \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{V}) = \bar{\boldsymbol{I}}\otimes(\boldsymbol{V}(d\boldsymbol{W})\boldsymbol{V}^{\top})
\end{gather}
由第二式可以进一步解得
\begin{equation}d\boldsymbol{V} = \boldsymbol{V}(\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(d\boldsymbol{W})\boldsymbol{V}))\end{equation}
其中
\begin{equation}\boldsymbol{K}_{i,j} = \left\{\begin{aligned}
&\, 1/(\sigma_i - \sigma_j), &\, i\neq j \\
&\, 0, &\, i = j
\end{aligned}\right.\end{equation}
根据这个结果，我们有
\begin{equation}\nabla_{\boldsymbol{W}}\mathcal{L} = \boldsymbol{V}(\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L})) + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L}))\boldsymbol{V}^{\top} \end{equation}
注意陷阱！上式其实是错误的，它是链式法则的结果，但链式法则只适用于无约束求导，而这里带有约束$\boldsymbol{W}=\boldsymbol{W}^{\top}$，正确的梯度应该还要包含对称化：
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{W}}\mathcal{L} =&\, \boldsymbol{V}\Big(\sym{\boldsymbol{K}^{\top}\otimes(\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L}))} + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L})\Big)\boldsymbol{V}^{\top} \\[6pt]
=&\, \boldsymbol{V}\Big(\boldsymbol{K}^{\top}\otimes\skew{\boldsymbol{V}^{\top}(\nabla_{\boldsymbol{V}}\mathcal{L})} + \boldsymbol{I}\otimes(\nabla_{\boldsymbol{\Sigma}}\mathcal{L})\Big)\boldsymbol{V}^{\top}\end{aligned}\end{equation}
另一个陷阱是直接将$\boldsymbol{U}=\boldsymbol{V}$代入式$\eqref{eq:w-grad-l}$来推导，这将会导致$\boldsymbol{K}$项翻倍，原因是式$\eqref{eq:w-grad-l}$区分了$\nabla_{\boldsymbol{U}}\mathcal{L}$和$\nabla_{\boldsymbol{V}}\mathcal{L}$，而在正定对称假设下，$\boldsymbol{U},\boldsymbol{V}$是相同的，对$\boldsymbol{V}$求梯度实际上无形中求了原本的$\nabla_{\boldsymbol{U}}\mathcal{L},\nabla_{\boldsymbol{V}}\mathcal{L}$之和，所以会导致重复计算。相关文献还可以参考[《Matrix Backpropagation for Deep Networks With Structured Layers》](<https://openaccess.thecvf.com/content_iccv_2015/html/Ionescu_Matrix_Backpropagation_for_ICCV_2015_paper.html>)。

## 数值问题

可能有些读者疑问：我只能保证初始化矩阵的奇异值两两不等，怎么保证经过训练后的矩阵仍然满足这个条件呢？答案是梯度自然会帮我们保证。从式$\eqref{eq:dU}$和式$\eqref{eq:dV}$可以看出，它的梯度包含了$\boldsymbol{F}$，而由$\boldsymbol{F}_{i,j} = \frac{1}{\sigma_j^2 - \sigma_i^2}$可知，一旦两个奇异值接近，那么梯度将会非常大，所以优化器会自动把它们推开。

不过，这个特性也给实际训练带来了数值不稳定性，主要体现在$\sigma_i,\sigma_j$接近时的梯度爆炸。对此，论文[《Backpropagation\-Friendly Eigendecomposition》](<https://papers.cool/arxiv/1906.09023>)提出用“幂迭代（Power Iteration）”替代精确的SVD。后来，论文[《Robust Differentiable SVD》](<https://papers.cool/arxiv/2104.03821>)证明了它理论上等于对$\boldsymbol{F}_{i,j} = -\frac{1}{（\sigma_i + \sigma_j)(\sigma_i - \sigma_j）}$中的$\frac{1}{\sigma_i - \sigma_j}$做泰勒近似（假设$\sigma_j < \sigma_i$）：
\begin{equation}\frac{1}{\sigma_i - \sigma_j} = \frac{1}{\sigma_i}\frac{1}{1-(\sigma_j/\sigma_i)}\approx \frac{1}{\sigma_i}\left(1 + \left(\frac{\sigma_j}{\sigma_i}\right) + \left(\frac{\sigma_j}{\sigma_i}\right)^2 + \cdots + \left(\frac{\sigma_j}{\sigma_i}\right)^N \right)\end{equation}
使用泰勒近似后，至少对于$\sigma_j \to \sigma_i$的情况不会出现梯度爆炸了。再后来，作者在[《Why Approximate Matrix Square Root Outperforms Accurate SVD in Global Covariance Pooling?》](<https://papers.cool/arxiv/2105.02498>)将其推广到更一般的Padé近似。这一系列工作笔者并不是太熟悉，因此就不过多展开了。

只是笔者有一个疑问，如果只是单纯为了避免数值爆炸，似乎没必要上这些工具，直接给$\sigma_j/\sigma_i$加个截断不就好了？比如
\begin{equation}\frac{1}{1-(\sigma_j/\sigma_i)}\approx \frac{1}{1-\min(\sigma_j/\sigma_i, 0.99)}\end{equation}
这样不就简单避免了它趋于无穷？还是笔者有什么认识不到位的地方？如果读者了解相关背景，敬请在评论区指正，谢谢。

## 一般结果

到目前为止，上面出现的所有矩阵都是$n\times n$矩阵，因为这是在文章开头的假设“$\boldsymbol{W}$是满秩的$n\times n$矩阵，且全体奇异值两两不等”下进行的推导。这一节我们来讨论该条件可以放宽到什么程度。

简单来说，SVD可导的条件是“全体非零奇异值两两不等”，换言之，方阵可以去掉，满秩也可以去掉，但是非零奇异值仍须互不相同。因为一旦有相等的奇异值，那么SVD就不唯一，这从根本上破坏了可导性。当然，如果只要部分导数，那我们还可以放宽条件，比如只想要谱范数即$\sigma_1$的导数，那么只需要$\sigma_1 > \sigma_2$。

那么，放宽条件后，微分结果怎样变化呢？我们不妨一般地设$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，秩为$r$，SVD为
\begin{equation}\boldsymbol{W} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top},\quad\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m},\boldsymbol{V}\in\mathbb{R}^{m\times m}\end{equation}
之所以允许零奇异值，是因为此时我们只关心$d\boldsymbol{U}_{[:, :r]}$和$d\boldsymbol{V}_{[:, :r]}$，在非零奇异值两两不等的假设下，它们是可以唯一确定的。我们从式$\eqref{eq:dU-0}$出发，直到式$\eqref{eq:dU-0}$的推导都是通用的，从式$\eqref{eq:dU-0}$也可以看出为什么要拒绝相同奇异值，因为$\sigma_i = \sigma_j$时$\sigma_j - \sigma_i$就无法求逆了。在式$\eqref{eq:dU-0}$中只保留跟$d\boldsymbol{U}_{[:, :r]}$相关的部分，我们得到
\begin{equation}\bar{\boldsymbol{I}}_{[:, :r]}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma}_{[:, :r]} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}) = \boldsymbol{E}_{[:, :r]}\otimes (\boldsymbol{U}^{\top}(d\boldsymbol{U}_{[:, :r]}))\end{equation}
这里$\boldsymbol{E}$的定义依然是$\boldsymbol{E}_{i,j} = \sigma_j^2 - \sigma_i^2$，但$\boldsymbol{E}_{[:, :r]}$正好排除了所有0，所以可以顺利求逆，结果是
\begin{equation}d\boldsymbol{U}_{[:, :r]} = \boldsymbol{U}(\boldsymbol{F}_{[:, :r]}\otimes(\boldsymbol{U}^{\top}(d\boldsymbol{W})\boldsymbol{V}\boldsymbol{\Sigma}_{[:, :r]} + \boldsymbol{\Sigma}\boldsymbol{V}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}))\end{equation}
进一步地，做如下分块
\begin{equation}\boldsymbol{U} = \begin{pmatrix}\boldsymbol{U}_{[:,:r]} & \boldsymbol{U}_{[:,r:]}\end{pmatrix},\quad \boldsymbol{F}_{[:, :r]} = \begin{pmatrix} \boldsymbol{F}_{[:r, :r]} \\ \boldsymbol{F}_{[r:, :r]}\end{pmatrix}\end{equation}
再加上$\boldsymbol{V}\boldsymbol{\Sigma}_{[:, :r]} = \boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}$，可以得到
\begin{equation}\begin{aligned}
d\boldsymbol{U}_{[:, :r]} =&\, \boldsymbol{U}_{[:, :r]}(\boldsymbol{F}_{[:r, :r]}\otimes(\boldsymbol{U}_{[:, :r]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]} + \boldsymbol{\Sigma}_{[:r, :r]}\boldsymbol{V}_{[:, :r]}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]})) \\[6pt]
&\,\qquad + \boldsymbol{U}_{[:, r:]}(\boldsymbol{F}_{[r:, :r]}\otimes(\boldsymbol{U}_{[:, r:]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}))
\end{aligned}\label{eq:dU-r}\end{equation}
根据假设，当$i > r$时$\sigma_i=0$，所以$\boldsymbol{F}_{[r:, :r]}$的每一行都是$(\sigma_1^{-2},\sigma_2^{-2},\cdots,\sigma_r^{-2})$，于是$\boldsymbol{F}_{[r:, :r]}\otimes$等于右乘$\boldsymbol{\Sigma}_{[:r, :r]}^{-2}$，因此
\begin{equation}\boldsymbol{F}_{[r:, :r]}\otimes(\boldsymbol{U}_{[:, r:]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}) = \boldsymbol{U}_{[:, r:]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}^{-1}\end{equation}
最后利用$\boldsymbol{U}\boldsymbol{U}^{\top} = \boldsymbol{I}$和$\boldsymbol{U} = (\boldsymbol{U}_{[:,:r]},\boldsymbol{U}_{[:,r:]})$，得到$\boldsymbol{U}_{[:, r:]}\boldsymbol{U}_{[:, r:]}^{\top} = \boldsymbol{I} - \boldsymbol{U}_{[:, :r]}\boldsymbol{U}_{[:, :r]}^{\top}$，代入式$\eqref{eq:dU-r}$得
\begin{equation}\begin{aligned}
d\boldsymbol{U}_{[:, :r]} =&\, \boldsymbol{U}_{[:, :r]}(\boldsymbol{F}_{[:r, :r]}\otimes(\boldsymbol{U}_{[:, :r]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]} + \boldsymbol{\Sigma}_{[:r, :r]}\boldsymbol{V}_{[:, :r]}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]})) \\[6pt]
&\,\qquad \color{orange}{+ (\boldsymbol{I} - \boldsymbol{U}_{[:, :r]}\boldsymbol{U}_{[:, :r]}^{\top})(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}^{-1}}
\end{aligned}\end{equation}
这就将$d\boldsymbol{U}_{[:, r:]}$表示成了$\boldsymbol{U}_{[:, :r]},\boldsymbol{\Sigma}_{[:r, :r]},\boldsymbol{V}_{[:, :r]}$的函数，在非零奇异值两两不等的假设下，三者是唯一确定的，所以$d\boldsymbol{U}_{[:, :r]}$是唯一确定的，相比式$\eqref{eq:dU}$多出了橙色这一项。类似地
\begin{equation}\begin{aligned}
d\boldsymbol{V}_{[:, :r]} =&\, \boldsymbol{V}_{[:, :r]}(\boldsymbol{F}_{[:r, :r]}\otimes(\boldsymbol{V}_{[:, :r]}^{\top}(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]} + \boldsymbol{\Sigma}_{[:r, :r]}\boldsymbol{U}_{[:, :r]}^{\top}(d\boldsymbol{W})\boldsymbol{V}_{[:, :r]})) \\[6pt]
&\,\qquad \color{orange}{+ (\boldsymbol{I} - \boldsymbol{V}_{[:, :r]}\boldsymbol{V}_{[:, :r]}^{\top})(d\boldsymbol{W})^{\top}\boldsymbol{U}_{[:, :r]}\boldsymbol{\Sigma}_{[:r, :r]}^{-1}}
\end{aligned}\end{equation}

## 文章小结

本文较为详细地推导了SVD的求导公式。
