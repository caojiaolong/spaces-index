# Transformer升级之路：6、旋转位置编码的完备性分析

> 作者：苏剑林 · 科学空间 · 2022-12-28
>
> 原文：<https://spaces.ac.cn/archives/9403>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在去年的文章[《Transformer升级之路：2、博采众长的旋转式位置编码》](<https://spaces.ac.cn/archives/8265>)中，笔者提出了旋转位置编码（RoPE），当时的出发点只是觉得用绝对位置来实现相对位置是一件“很好玩的事情”，并没料到其实际效果还相当不错，并为大家所接受，不得不说这真是一个意外之喜。后来，在[《Transformer升级之路：4、二维位置的旋转式位置编码》](<https://spaces.ac.cn/archives/8397>)中，笔者讨论了二维形式的RoPE，并研究了用矩阵指数表示的RoPE的一般解。

既然有了一般解，那么自然就会引出一个问题：我们常用的RoPE，只是一个以二维旋转矩阵为基本单元的分块对角矩阵，如果换成一般解，理论上效果会不会更好呢？本文就来回答这个问题。

## 指数通解

在[《Transformer升级之路：4、二维位置的旋转式位置编码》](<https://spaces.ac.cn/archives/8397>)中，我们将RoPE抽象地定义为任意满足下式的方阵
\begin{equation}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}\label{eq:re}\end{equation}
然后，我们探讨了如下矩阵指数形式的解
\begin{equation}\boldsymbol{\mathcal{R}}_n=\exp n\boldsymbol{B}\end{equation}
这里的矩阵指数，不是像Softmax那样的激活函数式的element\-wise运算，而是按照泰勒级数定义的“[Matrix Exponential](<https://en.wikipedia.org/wiki/Matrix_exponential>)”。根据“[Baker–Campbell–Hausdorff公式](<https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula>)”，我们有
\begin{equation}\begin{aligned}
\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=&\,\big(\exp m\boldsymbol{B}\big)^{\top}\big(\exp n\boldsymbol{B}\big) = \big(\exp m\boldsymbol{B}^{\top}\big)\big(\exp n\boldsymbol{B}\big) \\
=&\,\exp\left(m\boldsymbol{B}^{\top} + n\boldsymbol{B} + \frac{1}{2}mn\left[\boldsymbol{B}^{\top}, \boldsymbol{B}\right]+\cdots\right)
\end{aligned}\end{equation}
这里$\left[\boldsymbol{A}, \boldsymbol{B}\right]=\boldsymbol{A}\boldsymbol{B}-\boldsymbol{B}\boldsymbol{A}$，$\cdots$省略的都是$m,n$的三次或三次以上的项。按照式$\eqref{eq:re}$，那么上式指数部分应该等于$(n-m)\boldsymbol{B}$，这就推出
\begin{equation}\boldsymbol{B}^{\top} = - \boldsymbol{B}\end{equation}
即要求$\boldsymbol{B}$是反对称矩阵。

## 正交通解

进一步地，我们有$(\exp \boldsymbol{B})^{\top}(\exp \boldsymbol{B})=\exp(\boldsymbol{B}-\boldsymbol{B}) = \boldsymbol{I}$和$\exp n\boldsymbol{B} = \left(\exp\boldsymbol{B}\right)^n$，前者说明$\exp\boldsymbol{B}$是正交矩阵，后者则启示我们这是不是可以推广到任意正交矩阵？不难验证，答案是肯定的，我们有结论：

> 对于任意正交矩阵$\boldsymbol{O}$，$\boldsymbol{\mathcal{R}}_n=\boldsymbol{O}^n$是满足式$\eqref{eq:re}$的解。

值得指出的是，在实数域内，并非所有正交矩阵能写成$\exp\boldsymbol{B}$的形式，所以$\boldsymbol{O}^n$实际上比矩阵指数形式更宽泛。从[《恒等式 det(exp(A)) = exp(Tr(A)) 赏析》](<https://spaces.ac.cn/archives/6377>)可知$\det(\exp(\boldsymbol{A})) = \exp(\text{Tr}(\boldsymbol{A})) > 0$，所以能写成矩阵指数形式的正交矩阵行列式必然大于0（即等于1），事实上这个结果反过来也成立，即行列式等于1的正交矩阵，必然可以写成$\exp\boldsymbol{B}$的形式，其中$\boldsymbol{B}$是反对称矩阵。（参考[《Why can any orthogonal matrix be written as O=e^A》](<https://math.stackexchange.com/questions/2467531/why-can-any-orthogonal-matrix-be-written-as-o-ea>)）。

对于$\det(\boldsymbol{O}) = -1$的正交矩阵，我们有$\boldsymbol{O}=\boldsymbol{O}_+ \boldsymbol{I}_-$，其中$\boldsymbol{I}_-$是对角线有一个\-1、剩下都是1的对角阵，$\boldsymbol{O}_+$是$\det(\boldsymbol{O}_+) = 1$的正交矩阵，它可以写成$\exp\boldsymbol{B}$的形式，此时$\boldsymbol{O}^n = (\boldsymbol{O}_+ \boldsymbol{I}_-)^n = \boldsymbol{I}_-^n\exp n\boldsymbol{B}$。这也就是说，即便对于$\det(\boldsymbol{O}) = -1$的$\boldsymbol{O}^n$，也只是$\exp n\boldsymbol{B}$的简单变换，所以接下来我们主要研究$\exp n\boldsymbol{B}$形式的解。

## 完备分析

众所周知，我们平时所用的RoPE位置编码，是如下形式的分块对角矩阵：
\begin{equation}\scriptsize{\left(\begin{array}{cc:cc:cc:cc}
\cos n\theta_0 & -\sin n\theta_0 & 0 & 0 & \cdots & \cdots & 0 & 0 \\
\sin n\theta_0 & \cos n\theta_0 & 0 & 0 & \cdots & \cdots & 0 & 0 \\
\hdashline
0 & 0 & \cos n\theta_1 & -\sin n\theta_1 & \cdots & \cdots & 0 & 0 \\
0 & 0 & \sin n\theta_1 & \cos n\theta_1 & \cdots & \cdots & 0 & 0 \\
\hdashline
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\
\vdots & \vdots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots \\
\hdashline
0 & 0 & 0 & 0 & \cdots & \cdots & \cos n\theta_{d/2-1} & -\sin n\theta_{d/2-1} \\
0 & 0 & 0 & 0 & \cdots & \cdots & \sin n\theta_{d/2-1} & \cos n\theta_{d/2-1} \\
\end{array}\right)}\end{equation}
它可以简写成
\begin{equation}\begin{pmatrix}
\boldsymbol{R}_{n\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{R}_{n\theta_1} & \cdots & \boldsymbol{0} \\
\vdots & \vdots & \ddots & \vdots \\
\boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{R}_{n\theta_{d/2-1}} \\
\end{pmatrix} = \exp n\begin{pmatrix}
\boldsymbol{J}_{\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{J}_{\theta_1} & \cdots & \boldsymbol{0} \\
\vdots & \vdots & \ddots & \vdots \\
\boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{J}_{\theta_{d/2-1}} \\
\end{pmatrix}\end{equation}
其中
\begin{equation}\boldsymbol{R}_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix},\quad \boldsymbol{J}_{\theta} = \begin{pmatrix} 0 & -\theta \\ \theta & 0\end{pmatrix}\end{equation}
这种选择可以说是最简单的一种，其本质原因可以说是为了降低计算量。那么，所谓完备性问题，就是要回答：如上的分块对角矩阵的特例，相比全参数$\exp n\boldsymbol{B}$，是否有能力上的缺失？换句话说，如果不考虑计算量，将$\boldsymbol{B}$替换为一般的反对称矩阵，效果是否可能会有提升？

回答这个问题不困难，事实上，对于任意偶数阶反对称矩阵，它都可以对角化为分块对角矩阵
\begin{equation}\boldsymbol{\Lambda} = \begin{pmatrix}
\boldsymbol{J}_{\theta_0} & \boldsymbol{0} & \cdots & \boldsymbol{0} \\
\boldsymbol{0} & \boldsymbol{J}_{\theta_1} & \cdots & \boldsymbol{0} \\
\vdots & \vdots & \ddots & \vdots \\
\boldsymbol{0} & \boldsymbol{0} & \cdots & \boldsymbol{J}_{\theta_{d/2-1}} \\
\end{pmatrix}\end{equation}
该结论可以参考[Skew\-symmetric matrix](<https://en.wikipedia.org/wiki/Skew-symmetric_matrix#Spectral_theory>)。也就是说，存在可逆矩阵$\boldsymbol{P}$，使得$\boldsymbol{B}=\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，于是
\begin{equation}\exp n\boldsymbol{B} = \exp \left(n\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}\right) = \boldsymbol{P}(\exp n\boldsymbol{\Lambda})\boldsymbol{P}^{-1}\end{equation}
也就是说，任意的$\exp n\boldsymbol{B}$与分块对角的$\exp n\boldsymbol{\Lambda}$，仅仅相差一个相似变换，而我们在Self Attention中应用RoPE时，是
\begin{equation}\boldsymbol{q}^{\top}\big(\exp (n-m)\boldsymbol{B}\big)\boldsymbol{k} = \big(\boldsymbol{P}^{\top}\boldsymbol{q}\big)^{\top}\big(\exp (n-m)\boldsymbol{\Lambda}\big)\big(\boldsymbol{P}^{-1}\boldsymbol{k}\big)\end{equation}
由于$\boldsymbol{q},\boldsymbol{k}$一般都是输入$\boldsymbol{x}$经过某个可学习的线性变换而来，$\boldsymbol{P}^{\top},\boldsymbol{P}^{-1}$原则上都可以吸收到线性变换的训练参数中，因此直接设为$\boldsymbol{q}^{\top}\big(\exp (n-m)\boldsymbol{\Lambda}\big)\boldsymbol{k}$理论上不会损失一般性。

所以，对于Self Attention来说，问题的答案是否定的。不过，如果是[线性Attention](<https://spaces.ac.cn/archives/%E7%BA%BF%E6%80%A7Attention%E7%9A%84%E6%8E%A2%E7%B4%A2%EF%BC%9AAttention%E5%BF%85%E9%A1%BB%E6%9C%89%E4%B8%AASoftmax%E5%90%97%EF%BC%9F>)，答案会有少许区别，因为线性Attention的$\boldsymbol{q},\boldsymbol{k}$加了个激活函数：
\begin{equation}\phi(\boldsymbol{q})^{\top}\big(\exp (n-m)\boldsymbol{B}\big)\varphi(\boldsymbol{k}) = \big(\boldsymbol{P}^{\top}\phi(\boldsymbol{q})\big)^{\top}\big(\exp (n-m)\boldsymbol{\Lambda}\big)\big(\boldsymbol{P}^{-1}\varphi(\boldsymbol{k})\big)\end{equation}
这就导致了$\boldsymbol{P}^{\top},\boldsymbol{P}^{-1}$不一定能吸收到线性变换的训练参数中，因此对线性Attention补上两个参数矩阵，是有可能带来提升的。

## 文章小结

本文简单分析了RoPE的完备性问题，表明对于Self Attention来说，目前的分块对角型RoPE不会损失一般性。
