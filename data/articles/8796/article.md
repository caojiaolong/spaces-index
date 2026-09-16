# 输入梯度惩罚与参数梯度惩罚的一个不等式

> 作者：苏剑林 · 科学空间 · 2021-12-11
>
> 原文：<https://spaces.ac.cn/archives/8796>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在本博客中，已经多次讨论过梯度惩罚相关内容了。从形式上来看，梯度惩罚项分为两种，一种是关于输入的梯度惩罚$\Vert\nabla_{\boldsymbol{x}} f(\boldsymbol{x};\boldsymbol{\theta})\Vert^2$，在[《对抗训练浅谈：意义、方法和思考（附Keras实现）》](<https://spaces.ac.cn/archives/7234>)、[《泛化性乱弹：从随机噪声、梯度惩罚到虚拟对抗训练》](<https://spaces.ac.cn/archives/7466>)等文章中我们讨论过，另一种则是关于参数的梯度惩罚$\Vert\nabla_{\boldsymbol{\theta}} f(\boldsymbol{x};\boldsymbol{\theta})\Vert^2$，在[《从动力学角度看优化算法（五）：为什么学习率不宜过小？》](<https://spaces.ac.cn/archives/7787>)、[《我们真的需要把训练集的损失降低到零吗？》](<https://spaces.ac.cn/archives/7643>)等文章我们讨论过。

在相关文章中，两种梯度惩罚都声称有着提高模型泛化性能的能力，那么两者有没有什么联系呢？笔者从Google最近的一篇论文[《The Geometric Occam's Razor Implicit in Deep Learning》](<https://papers.cool/arxiv/2111.15090>)学习到了两者的一个不等式，算是部分地回答了这个问题，并且感觉以后可能用得上，在此做个笔记。

## 最终结果

假设有一个$l$层的MLP模型，记为
\begin{equation}\boldsymbol{h}^{(t+1)} = g^{(t)}(\boldsymbol{W}^{(t)}\boldsymbol{h}^{(t)}+\boldsymbol{b}^{(t)})\end{equation}
其中$g^{(t)}$是当前层的激活函数，$t\in\{1,2,\cdots,l\}$，并记$\boldsymbol{h}^{(1)}$为$\boldsymbol{x}$，即模型的原始输入，为了方便后面的推导，我们记$\boldsymbol{z}^{(t+1)}=\boldsymbol{W}^{(t)}\boldsymbol{h}^{(t)}+\boldsymbol{b}^{(t)}$；参数全体为$\boldsymbol{\theta}=\{\boldsymbol{W}^{(1)},\boldsymbol{b}^{(1)},\boldsymbol{W}^{(2)},\boldsymbol{b}^{(2)},\cdots,\boldsymbol{W}^{(l)},\boldsymbol{b}^{(l)}\}$。设$f$是$\boldsymbol{h}^{(l+1)}$的任意标量函数，那么成立不等式
\begin{equation}\Vert\nabla_{\boldsymbol{x}} f\Vert^2\left(\frac{1 + \Vert \boldsymbol{h}^{(1)}\Vert^2}{\Vert\boldsymbol{W}^{(1)}\Vert^2 \Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(1)}\Vert^2}+\cdots+\frac{1 + \Vert \boldsymbol{h}^{(l)}\Vert^2}{\Vert\boldsymbol{W}^{(l)}\Vert^2 \Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(l)}\Vert^2}\right)\leq \Vert\nabla_{\boldsymbol{\theta}} f\Vert^2\label{eq:f}\end{equation}
其中上式中$\Vert\nabla_{\boldsymbol{x}} f\Vert$、$\Vert\nabla_{\boldsymbol{\theta}} f\Vert^2$和$\Vert \boldsymbol{h}^{(i)}\Vert$用的是普通的$l_2$范数，也就是每个元素的平方和再开平方，而$\Vert\boldsymbol{W}^{(1)}\Vert$和$\Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(1)}\Vert$用的则是矩阵的“谱范数”（参考[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)）。该不等式显示，参数的梯度惩罚一定程度上包含了输入的梯度惩罚。

## 推导过程

显然，为了不等式$\eqref{eq:f}$，我们只需要对每一个参数证明：
\begin{align}\Vert\nabla_{\boldsymbol{x}} f\Vert^2\left(\frac{\Vert \boldsymbol{h}^{(t)}\Vert^2}{\Vert\boldsymbol{W}^{(t)}\Vert^2 \Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)}\Vert^2}\right)\leq&\, \Vert\nabla_{\boldsymbol{W}^{(t)}} f\Vert^2 \label{eq:w}\\
\Vert\nabla_{\boldsymbol{x}} f\Vert^2\left(\frac{1}{\Vert\boldsymbol{W}^{(t)}\Vert^2 \Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)}\Vert^2}\right)\leq&\, \Vert\nabla_{\boldsymbol{b}^{(t)}} f\Vert^2 \label{eq:b}
\end{align}
然后遍历所有$t$，将每一式左右两端相加即可。这两个不等式的证明本质上是一个矩阵求导问题，但多数读者可能跟笔者一样，都不熟悉矩阵求导，这时候最佳的办法就是写出分量形式，然后就变成标量的求导问题。

具体来说，$\boldsymbol{z}^{(t+1)}=\boldsymbol{W}^{(t)}\boldsymbol{h}^{(t)}+\boldsymbol{b}^{(t)}$写成分量形式：
\begin{equation}z^{(t+1)}_i = \sum_j w^{(t)}_{i,j} h_j^{(t)} + b^{(t)}_i\end{equation}
然后由链式法则：
\begin{equation}\frac{\partial f}{\partial x_i} = \sum_{j,k} \frac{\partial f}{\partial z^{(t+1)}_j} \frac{\partial z^{(t+1)}_j}{\partial h^{(t)}_k} \frac{\partial h^{(t)}_k}{\partial x_i} = \sum_{j,k} \frac{\partial f}{\partial z^{(t+1)}_j} w^{(t)}_{j,k} \frac{\partial h^{(t)}_k}{\partial x_i}\label{eq:l}\end{equation}
然后
\begin{equation}\frac{\partial z^{(t+1)}_j}{\partial w^{(t)}_{m,n}} = \delta_{j,m}h^{(t)}_n\end{equation}
这里$\delta_{j,m}$是克罗内克符号。现在我们可以写出
\begin{equation}w^{(t)}_{j,k} = \sum_m \delta_{j,m}w^{(t)}_{m,k} = \sum_m \frac{\partial z^{(t+1)}_j}{\partial w^{(t)}_{m,n}} (h^{(t)}_n)^{-1} w^{(t)}_{m,k}\end{equation}
代入$\eqref{eq:l}$得到
\begin{equation}\frac{\partial f}{\partial x_i} = \sum_{j,k,m} \frac{\partial f}{\partial z^{(t+1)}_j} \frac{\partial z^{(t+1)}_j}{\partial w^{(t)}_{m,n}} (h^{(t)}_n)^{-1} w^{(t)}_{m,k} \frac{\partial h^{(t)}_k}{\partial x_i}=\sum_{k,m} \frac{\partial f}{\partial w^{(t)}_{m,n}} (h^{(t)}_n)^{-1} w^{(t)}_{m,k} \frac{\partial h^{(t)}_k}{\partial x_i}\end{equation}
两边乘以$h^{(t)}_n$得
\begin{equation}h^{(t)}_n\frac{\partial f}{\partial x_i} = \sum_{k,m} \frac{\partial f}{\partial w^{(t)}_{m,n}} w^{(t)}_{m,k} \frac{\partial h^{(t)}_k}{\partial x_i}\end{equation}
约定原始向量为列向量，求梯度后矩阵的形状反转，那么上述可以写成矩阵形式：
\begin{equation}\boldsymbol{h}^{(t)}(\nabla_{\boldsymbol{x}} f) = (\nabla_{\boldsymbol{W}^{(t)}} f )\boldsymbol{W}^{(t)}(\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)})\end{equation}
两边左乘$(\boldsymbol{h}^{(t)})^{\top}$得
\begin{equation}\Vert\boldsymbol{h}^{(t)}\Vert^2(\nabla_{\boldsymbol{x}} f) = (\boldsymbol{h}^{(t)})^{\top}(\nabla_{\boldsymbol{W}^{(t)}} f )\boldsymbol{W}^{(t)}(\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)})\end{equation}
两边取范数得
\begin{equation}\Vert\boldsymbol{h}^{(t)}\Vert^2 \Vert\nabla_{\boldsymbol{x}} f\Vert = \Vert (\boldsymbol{h}^{(t)})^{\top}(\nabla_{\boldsymbol{W}^{(t)}} f )\boldsymbol{W}^{(t)}(\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)})\Vert \leq \Vert\boldsymbol{h}^{(t)}\Vert \Vert\nabla_{\boldsymbol{W}^{(t)}} f \Vert \Vert \boldsymbol{W}^{(t)}\Vert \Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)}\Vert\end{equation}
等于第二个不等号来说，矩阵的范数用$l_2$范数或者谱范数都是成立的。于是选择所需要的范数后，整理可得式$\eqref{eq:w}$；至于式$\eqref{eq:b}$的证明类似，这里不再重复。

## 简单评析

可能有读者会想问具体该如何理解式$\eqref{eq:f}$？事实上，笔者主要觉得式$\eqref{eq:f}$本身有点意思，以后说不准在某个场景用得上，所以本文主要是对此做个“笔记”，但对它并没有很好的解读结果。

至于原论文的逻辑顺序是这样的：在[《从动力学角度看优化算法（五）：为什么学习率不宜过小？》](<https://spaces.ac.cn/archives/7787>)中我们介绍了[《Implicit Gradient Regularization》](<https://spaces.ac.cn/archives/7787>)（跟本篇论文同一作者），里边指出SGD隐式地包含了对参数的梯度惩罚项，而式$\eqref{eq:f}$则说明对参数的梯度惩罚隐式地包含了对输入的梯度惩罚，而对输入的梯度惩罚又跟Dirichlet能量有关，Dirichlet能量则可以作为模型复杂度的表征。所以总的一串推理下来，结论就是：SGD本身会倾向于选择复杂度比较小的模型。

不过，原论文在解读式$\eqref{eq:f}$时，犯了一个小错误。它说初始阶段的$\Vert \boldsymbol{W}^{(t)}\Vert$会很接近于0，所以式$\eqref{eq:f}$中括号的项会很大，因此如果要降低式$\eqref{eq:f}$右边的参数梯度惩罚，那么必须要使得式$\eqref{eq:f}$左边的输入梯度惩罚足够小。然而从[《从几何视角来理解模型参数的初始化策略》](<https://spaces.ac.cn/archives/7180>)我们知道，常用的初始化方法其实接近于正交初始化，而正交矩阵的谱范数其实为1，如果考虑激活函数，那么初始化的谱范数其实还大于1，所以初始化阶段$\Vert \boldsymbol{W}^{(t)}\Vert$会很接近于0是不成立的。

事实上，对于一个没有训练崩的网络，模型的参数和每一层的输入输出基本上都会保持一种稳定的状态，所以其实整个训练过程中$\Vert \boldsymbol{h}^{(t)}\Vert$、$\Vert\boldsymbol{W}^{(t)}\Vert$、$\Vert\nabla_{\boldsymbol{x}}\boldsymbol{h}^{(t)}\Vert$其实波动都不大，因此右端对参数的梯度惩罚近似等价于左端对输入的乘法惩罚。这是笔者的理解，不需要“$\Vert \boldsymbol{W}^{(t)}\Vert$会很接近于0”的假设。

## 文章小结

本文主要介绍了两种梯度惩罚项之间的一个不等式，并给出了自己的证明以及一个简单的评析。
