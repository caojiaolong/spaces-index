# 排序不等式及其推广

> 作者：苏剑林 · 科学空间 · 2026-09-21
>
> 原文：<https://spaces.ac.cn/archives/11910>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

排序不等式是一个经典的不等式，本身不难理解，很多读者可能在中学阶段就已经了解过它，尤其是准备过数学竞赛的同学。但排序不等式也远不止表面上那么简单：它在矩阵世界中有一个深刻的对应物——冯·诺伊曼迹不等式；它还可以沿着“多序列”、“多矩阵”的方向继续推广。

本文就沿着这条线索，将这些内容串联起来介绍一下。

## 基本形式

设有两个从大到小排好序的实数列$a_1\geq a_2\geq \cdots\geq a_n$和$b_1\geq b_2\geq \cdots \geq b_n$，注意这里不需要非负约束，$\tau(1),\tau(2),\cdots,\tau(n)$表示$1,2,\cdots,n$的任意排列，那么成立排序不等式
\begin{equation}\sum_{i=1}^n a_i b_{n+1-i} \leq \sum_{i=1}^n a_i b_{\tau(i)}\leq \sum_{i=1}^n a_i b_i\end{equation}
即“倒序积 ≤ 乱序积 ≤ 同序积”。排序不等式的标准证明方式是局部调整法：设$a_i \geq a_j$和$b_i\geq b_j$，那么
\begin{equation}(a_i b_i + a_j b_j) - (a_i b_j + a_j b_i) = (a_i - a_j)(b_i - b_j) \geq 0\end{equation}
也就是说，如果序列中有一对不同序的位置，那么将某一边交换后将它变得同序，相应的内积至少不会变差，于是最大值必然在完全同序时取到。对$a_1,a_2,\cdots,a_n$和$-b_1,-b_2,\cdots,-b_n$重复相同的流程，便可得到最小值端。

注意排序不等式本身不要求序列非负，但后面我们推广它时，会加上非负约束。值得指出的是，加非负约束并不改变当前的排序不等式的覆盖面，因为任意有限的实数列，都可以通过加一个足够大的常数来变成非负列，并且不改变排序，而两端展开后，就恢复无非负约束的排序不等式。

## 迹不等式

排序不等式在矩阵中有一个非常重要的对应物：设$\boldsymbol{A},\boldsymbol{B}$是两个$n\times n$的实对称矩阵，$\lambda_i(\cdot)$表示从大到小排序的第$i$个特征值，那么成立不等式
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\sum_{i=1}^n \lambda_i(\boldsymbol{A}) \lambda_{n+1-i}(\boldsymbol{B}) \leq \tr(\boldsymbol{A}\boldsymbol{B})\leq \sum_{i=1}^n \lambda_i(\boldsymbol{A}) \lambda_i(\boldsymbol{B})\label{eq:tr-symm}\end{equation}
很显然，$\boldsymbol{A},\boldsymbol{B}$取对角矩阵时，等价于上一节的排序不等式，所以这是它的矩阵推广，称为“冯·诺伊曼迹不等式（Von Neumann's Trace Inequality）”，又称“Fan迹不等式（Ky Fan's Trace Inequality）”，本文统一简称“迹不等式”。

当$\boldsymbol{A},\boldsymbol{B}$是一般的$n\times m$实矩阵时，则成立不等式
\begin{equation}|\tr(\boldsymbol{A}^{\top}\boldsymbol{B})|\leq \sum_{i=1}^{\min(n,m)} \sigma_i(\boldsymbol{A}) \sigma_i(\boldsymbol{B})\label{eq:tr-gen}\end{equation}
$\sigma_i(\cdot)$表示从大到小排序的第$i$个奇异值。迹不等式在数学、物理中都有很广泛的应用，比如推导[Muon](<https://spaces.ac.cn/tag/muon/>)优化器，如果我们已知该不等式，可以显著降低证明难度。一般版的不等式，可以通过Jordan–Wielandt对称化技巧构造
\begin{equation}\tilde{\boldsymbol{A}} = \begin{pmatrix}\boldsymbol{0} & \boldsymbol{A} \\ \boldsymbol{A}^{\top} & \boldsymbol{0}\end{pmatrix},\qquad \tilde{\boldsymbol{B}} = \begin{pmatrix}\boldsymbol{0} & \boldsymbol{B} \\ \boldsymbol{B}^{\top} & \boldsymbol{0}\end{pmatrix}\end{equation}
来转化为对称版，所以我们只需要关注对称版的证明。

## 正交优化

这一节我们证明对称版$\eqref{eq:tr-symm}$。不失一般性，可设$\boldsymbol{A}$是对角阵，若否，可将它特征值分解为$\boldsymbol{U}_A \boldsymbol{\Lambda}_A \boldsymbol{U}_A^{\top}$，然后利用$\tr(\boldsymbol{A}\boldsymbol{B}) = \tr(\boldsymbol{U}_A \boldsymbol{\Lambda}_A \boldsymbol{U}_A^{\top}\boldsymbol{B}\boldsymbol{U}_A \boldsymbol{U}_A^{\top}) = \tr(\boldsymbol{\Lambda}_A \boldsymbol{U}_A^{\top}\boldsymbol{B}\boldsymbol{U}_A)$，其中$\boldsymbol{\Lambda}_A$是对角阵，$\boldsymbol{U}_A^{\top}\boldsymbol{B}\boldsymbol{U}_A$的特征值跟$\boldsymbol{B}$一致。

冯·诺伊曼迹不等式的证明方式有很多，这里我们考虑一个比较别致、更接近冯·诺伊曼原始思想的证明。引入$n\times n$的正交矩阵$\boldsymbol{O}$，显然我们有
\begin{equation}\min_{\boldsymbol{O}} \tr(\boldsymbol{A}\boldsymbol{O}^{\top}\boldsymbol{B}\boldsymbol{O}) \leq \tr(\boldsymbol{A}\boldsymbol{B}) \leq \max_{\boldsymbol{O}} \tr(\boldsymbol{A}\boldsymbol{O}^{\top}\boldsymbol{B}\boldsymbol{O})\end{equation}
接下来我们要证明，$\tr(\boldsymbol{A}\boldsymbol{O}^{\top}\boldsymbol{B}\boldsymbol{O})$的最大值和最小值，正是我们期望的上下界。先考虑最大值，假设最大值为$\tr(\boldsymbol{A}\boldsymbol{O}_*^{\top}\boldsymbol{B}\boldsymbol{O}_*)$，注意对于任意反对称矩阵$\boldsymbol{S}$，矩阵指数$e^{t\boldsymbol{S}}$都是正交的，考虑扰动$\boldsymbol{O}_*\to \boldsymbol{O}_* e^{t\boldsymbol{S}}$，根据$\boldsymbol{O}_*$最优的假设，$t=0$处导数应当为$0$。直接计算得
\begin{equation}\frac{d}{dt}\tr(\boldsymbol{A}(\boldsymbol{O}_* e^{t\boldsymbol{S}})^{\top}\boldsymbol{B}(\boldsymbol{O}_* e^{t\boldsymbol{S}})) = \tr([\boldsymbol{A},(\boldsymbol{O}_* e^{t\boldsymbol{S}})^{\top}\boldsymbol{B}(\boldsymbol{O}_* e^{t\boldsymbol{S}})]\boldsymbol{S})\end{equation}
其中$[\boldsymbol{X},\boldsymbol{Y}]\triangleq\boldsymbol{X}\boldsymbol{Y}-\boldsymbol{Y}\boldsymbol{X}$通常称为对易子，化简过程中还利用了$\boldsymbol{S}$的反对称性。代入$t=0$，令它为零，并由$\boldsymbol{S}$的任意性知$[\boldsymbol{A},\boldsymbol{O}_*^{\top}\boldsymbol{B}\boldsymbol{O}_*]=\boldsymbol{0}$，即$\boldsymbol{A}$与$\boldsymbol{O}_*^{\top}\boldsymbol{B}\boldsymbol{O}_*$可交换。相同的论述过程对最小值也成立。

注意$\boldsymbol{A}$是对角阵，进一步假设对角线元素两两不同（若否，总可以将它表示成满足条件的矩阵序列的极限），那么与$\boldsymbol{A}$可交换的矩阵也只有对角阵，所以$\boldsymbol{O}_*$必然是$\boldsymbol{B}$的特征矩阵，或者说$\boldsymbol{O}_*^{\top}\boldsymbol{B}\boldsymbol{O}_*$必然是$\boldsymbol{B}$的对角化。于是问题变成了两个矩阵的特征值向量内积的上下界，由排序不等式即可得出。

## 多个序列

一个很自然的问题是：排序不等式可以推广到多个序列吗？我们以三序列为例，设有非负的序列$a_1\geq \cdots\geq a_n\geq 0$、$b_1\geq \cdots \geq b_n\geq 0$和$c_1\geq \cdots \geq c_n\geq 0$，$\tau(1),\cdots,\tau(n)$和$\pi(1),\cdots,\pi(n)$表示$1,\cdots,n$的任意两个排列，那么最大值依然是在同序时取到
\begin{equation}\sum_{i=1}^n a_i b_{\tau(i)} c_{\pi(i)}\leq \sum_{i=1}^n a_i b_i c_i\end{equation}
证明并不复杂，一行就可以写出来：
\begin{equation}\sum_{i=1}^n a_i b_{\tau(i)} c_{\pi(i)} = \sum_{k=1}^n(a_k - a_{k+1})\sum_{i=1}^k b_{\tau(i)} c_{\pi(i)} \leq \sum_{k=1}^n(a_k - a_{k+1})\sum_{i=1}^k b_i c_i = \sum_{i=1}^n a_i b_i c_i\end{equation}
第一个等号是利用[分部求和法](<https://en.wikipedia.org/wiki/Summation_by_parts>)做恒等变换（设$a_{n+1}=0$）；第二个等号是利用分部求和法重新变换回来；关键是不等号$\leq$，它是在$a_k \geq a_{k+1}$条件下，对$\sum_{i=1}^k b_{\tau(i)} c_{\pi(i)}$用排序不等式，注意到$b_i,c_i$已经从大到小排序过，所以$\sum_{i=1}^k b_i c_i$必然最大，$\sum_{i=1}^k b_{\tau(i)} c_{\pi(i)}$不管怎么重新配对都不可能超过它，从而有这个不等号。

这个证明和结果可以递归地推广到任意多个非负序列。注意非负是必要的，因为“$\sum_{i=1}^k b_{\tau(i)} c_{\pi(i)}$不管怎么重新配对都不可能超过$\sum_{i=1}^k b_i c_i$”只对非负序列保证成立。

## 自动均衡

然而，最小值侧并无法像最大值侧那样简单推广。

理由也很直观，“同序”是满足传递性的，两个单调递减的非负序列，其Hadamard积依然是单调递减的，这样我们就可以多次复用二元的排序不等式来得到最大值。然而，“倒序”本就是针对两个序列来说的，多个序列我们很难定义什么是“倒序”，一个递增和一个递减的非负序列的Hadamard积，也没有固定的单调性。

不过，我们依然可以从均值不等式来获得一些关于最小值的信号：
\begin{equation}\sum_{i=1}^n a_i b_{\tau(i)} c_{\pi(i)}\geq n\left(\prod_{i=1}^n a_i b_{\tau(i)} c_{\pi(i)}\right)^{1/n} = n\left(\prod_{i=1}^n a_i b_i c_i\right)^{1/n}\end{equation}
等号成立的条件是全体$a_i b_{\tau(i)} c_{\pi(i)}$相等。当然，由于我们只改变了$b,c$的序，所以很难实现全体$a_i b_{\tau(i)} c_{\pi(i)}$这一条件，但这可以告诉我们一个定性的方向：最小值对应的排序，会尽量使得所有$a_i b_{\tau(i)} c_{\pi(i)}$均衡。

有经验的同学可能敏锐地意识到，这种离散情形的均衡配对问题求解通常是比较困难的。事实也确实如此，它本质上可以抽象为一个[三维轴向指派问题](<https://www.sciencedirect.com/science/article/pii/0166218X9500031L>)，已知它是NP\-Hard的。也就是说，多序列排序不等式的最小值侧不存在一般的解析解，这是问题的本质困难。

## 多个矩阵

类似地，迹不等式可以推广到多个矩阵吗？考虑三个$n\times n$矩阵$\boldsymbol{A},\boldsymbol{B},\boldsymbol{C}$，我们有
\begin{equation}|\tr(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C})|\leq \sum_{i=1}^n \sigma_i(\boldsymbol{A}) \sigma_i(\boldsymbol{B}) \sigma_i(\boldsymbol{C})\label{eq:tr-gen-ext}\end{equation}
证明也是类似的
\begin{equation}\begin{aligned}
|\tr(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C})|\leq&\, \sum_{i=1}^n \sigma_i(\boldsymbol{A}) \sigma_i(\boldsymbol{B}\boldsymbol{C}) \\
=&\, \sum_{k=1}^n (\sigma_k(\boldsymbol{A}) - \sigma_{k+1}(\boldsymbol{A}))\sum_{i=1}^k \sigma_i(\boldsymbol{B}\boldsymbol{C}) \\
\leq&\, \sum_{k=1}^n (\sigma_k(\boldsymbol{A}) - \sigma_{k+1}(\boldsymbol{A}))\sum_{i=1}^k \sigma_i(\boldsymbol{B})\sigma_i(\boldsymbol{C}) \\
=&\, \sum_{i=1}^n \sigma_i(\boldsymbol{A})\sigma_i(\boldsymbol{B})\sigma_i(\boldsymbol{C})
\end{aligned}\end{equation}
即先用二元的迹不等式放缩一次，然后还是分部求和、放缩、分部求和，其中第二个$\leq$用到了如下不等式
\begin{equation}\sum_{i=1}^k \sigma_i(\boldsymbol{B}\boldsymbol{C}) \leq \sum_{i=1}^k \sigma_i(\boldsymbol{B})\sigma_i(\boldsymbol{C}),\qquad k=1,2,\cdots,n\end{equation}
这也可以看成是式$\eqref{eq:tr-gen}$的一个推广，证明思路就是先将左端转化为迹，然后再用式$\eqref{eq:tr-gen}$。具体来说，设$\boldsymbol{B}\boldsymbol{C}$的SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，那么
\begin{equation}\sum_{i=1}^k \sigma_i(\boldsymbol{B}\boldsymbol{C}) = \tr(\boldsymbol{U}_{[:,:k]}^{\top}\boldsymbol{B}\boldsymbol{C}\boldsymbol{V}_{[:,:k]})\leq \sum_{i=1}^k \sigma_i(\boldsymbol{U}_{[:,:k]}^{\top}\boldsymbol{B})\sigma_i(\boldsymbol{C}\boldsymbol{V}_{[:,:k]})\leq \sum_{i=1}^k \sigma_i(\boldsymbol{B})\sigma_i(\boldsymbol{C})\end{equation}
最后一个不等号用了$\sigma_i(\boldsymbol{X}\boldsymbol{Y})\leq \sigma_i(\boldsymbol{X})\sigma_1(\boldsymbol{Y})$以及$\boldsymbol{U}_{[:,:k]},\boldsymbol{V}_{[:,:k]}$的正交性。同样的归纳可以继续下去，得到任意多个矩阵的版本。

## 另一推广

式$\eqref{eq:tr-gen-ext}$推广了式$\eqref{eq:tr-gen}$，那式$\eqref{eq:tr-symm}$如何推广呢？设有$n\times n$的半正定矩阵$\boldsymbol{A},\boldsymbol{B},\boldsymbol{C}$，考虑$\tr(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C})$，最大值侧是平凡的，即式$\eqref{eq:tr-gen-ext}$，最小值侧是否可以类似地写成$\sum_{i=1}^n \lambda_i(\boldsymbol{A}) \lambda_{\tau(i)}(\boldsymbol{B}) \lambda_{\pi(i)}(\boldsymbol{C})$的形式呢？

答案是否定的。半正定矩阵的特征值一定非负，所以$\sum_{i=1}^n \lambda_i(\boldsymbol{A}) \lambda_{\tau(i)}(\boldsymbol{B}) \lambda_{\pi(i)}(\boldsymbol{C})$也一定非负，但是三个半正定矩阵之积的迹却可以是负的！因此这种形式的最小值必不成立。问题出在哪呢？注意到对于半正定矩阵$\boldsymbol{A},\boldsymbol{B}$，成立
\begin{equation}\tr(\boldsymbol{A}\boldsymbol{B}) = \tr((\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2})^{\top}(\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2})) = \Vert\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\Vert_F^2 \geq 0\end{equation}
这就保证了非负性。由此可以猜测，三矩阵的合理推广也许是$\Vert\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}\Vert_F^2$，而非更简单的$\tr(\boldsymbol{A}\boldsymbol{B}\boldsymbol{C})$。但很遗憾，这个式子虽然看起来更加“正确”，但可以举例证明它的最小值并不一定等于某个$\sum_{i=1}^n \lambda_i(\boldsymbol{A}) \lambda_{\tau(i)}(\boldsymbol{B}) \lambda_{\pi(i)}(\boldsymbol{C})$，所以它也无法作为式$\eqref{eq:tr-symm}$的最小值侧的推广。

不过，$\Vert\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}\Vert_F^2$确实也有一些与多序列排序不等式最小值端类似的性质，比如成立
\begin{equation}\begin{aligned}
\Vert\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}\Vert_F^2 =&\, \sum_{i=1}^n \sigma_i^2(\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}) \\
\geq&\, n\left(\prod_{i=1}^n \sigma_i(\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2})\right)^{2/n} \\
=&\, n [\det(\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2})]^{2/n} \\[4pt]
=&\, n [\det(\boldsymbol{A})\det(\boldsymbol{B})\det(\boldsymbol{C})]^{1/n} \\
\end{aligned}\end{equation}
等号在$\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}$全体奇异值相等时成立。这也就意味着，$\Vert\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}\Vert_F^2$取最小值时，$\boldsymbol{A}^{1/2}\boldsymbol{B}^{1/2}\boldsymbol{C}^{1/2}$的全体奇异值会趋于均匀分布，“自动均衡”再次出现！

## 文章小结

本文从排序不等式出发，沿着“矩阵化”和“多序列化”两条线索，梳理了它的主要推广。
