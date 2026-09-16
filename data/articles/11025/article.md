# msign的导数

> 作者：苏剑林 · 科学空间 · 2025-06-13
>
> 原文：<https://spaces.ac.cn/archives/11025>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这篇文章我们来推导$\newcommand{msign}{\mathop{\text{msign}}}\msign$算子的求导公式。如果读者想要像[《Test\-Time Training Done Right》](<https://papers.cool/arxiv/2505.23884>)一样，将[TTT](<https://papers.cool/arxiv/2407.04620>)和[Muon](<https://spaces.ac.cn/archives/10592>)结合起来，那么本文可能会对你有帮助。

## 两种定义

本文依然假设大家已经对$\msign$有所了解，如果还没有，可以先移步阅读[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)和[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)。现设有矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，那么
\begin{equation}\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}^{\top} = \text{SVD}(\boldsymbol{M}) \quad\Rightarrow\quad \msign(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}\end{equation}
其中$\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m},\boldsymbol{V}\in\mathbb{R}^{m\times m}$，$r$是$\boldsymbol{M}$的秩。简单来说，$\msign$就是把矩阵的所有非零奇异值都变成1后所得的新矩阵。基于SVD，我们还可以证明
\begin{equation}\msign(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}\end{equation}
这里的$^{-1/2}$的矩阵的$1/2$次幂的逆，由于$\boldsymbol{M}\boldsymbol{M}^{\top}$和$\boldsymbol{M}^{\top}\boldsymbol{M}$是（半）正定对称的，所以$1/2$次幂总是可求，但逆未必，不可逆的时候我们可以用“[伪逆](<https://spaces.ac.cn/archives/10366>)”。$\msign$这个名字，源于上式与实数符号函数$\newcommand{sign}{\mathop{\text{sign}}}\sign(x) = x/\sqrt{x^2}$的相似性。然而，我们之前也提到过，符号函数还有[另一个矩阵版](<https://en.wikipedia.org/wiki/Matrix_sign_function>)，这里称为$\newcommand{mcsgn}{\mathop{\text{mcsgn}}}\newcommand{csgn}{\mathop{\text{csgn}}}\mcsgn$：
\begin{equation}\mcsgn(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}\end{equation}
即$\msign$的$\boldsymbol{M}^{\top}\boldsymbol{M}$换成了$\boldsymbol{M}^2$。由于只有方阵才能算平方，所以这种定义只适用于方阵。在一篇文章内引入两种相似但不同的定义是容易引起混淆的，但很不幸，后面的计算中两种定义都需要用到，所以不得不同时出现。

$\mcsgn$具备相似不变性，如果$\boldsymbol{M}=\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，那么$\mcsgn(\boldsymbol{M})=\boldsymbol{P}\mcsgn(\boldsymbol{\Lambda})\boldsymbol{P}^{-1}$。进一步地，如果$\boldsymbol{\Lambda}$是对角阵（在复数域内几乎总是可以做到），那么有
\begin{equation}\mcsgn(\boldsymbol{M}) = \boldsymbol{P}\csgn(\boldsymbol{\Lambda})\boldsymbol{P}^{-1}\end{equation}
$\csgn(\boldsymbol{\Lambda})$表示对角线的元素都取$\csgn$，其中$\csgn(z) = z/\sqrt{z^2}$是符号函数的复数版，如果$z$的实部非零那么它等于$\sign(\mathop{\text{Re}}[z])$。这样看来，$\msign$和$\mcsgn$的区别在于，前者是在奇异值分解基础上对奇异值取符号函数，后者是在特征值分解基础上对特征值取符号函数。当$\boldsymbol{M}$是对称矩阵时，它们是相等的。

## 同一计算

目前而言，$\msign$的数值计算主要靠如下格式的“Newton\-Schulz迭代”：
\begin{equation}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\Vert\boldsymbol{M}\Vert_F},\qquad \boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c_{t+1}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2\end{equation}
至于系数的选择，这我们在[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)和[《msign算子的Newton\-Schulz迭代（下）》](<https://spaces.ac.cn/archives/10996>)已经详细探讨过了，其中出自下篇的比较新的结果是：
$$\begin{array}{c|ccc}
\hline
t & a\times 1.01 & b\times 1.01^3 & c\times 1.01^5 \\
\hline
\quad 1\quad & 8.28721 & -23.5959 & 17.3004 \\
2 & 4.10706 & -2.94785 & 0.544843 \\
3 & 3.94869 & -2.9089 & 0.551819 \\
4 & 3.31842 & -2.48849 & 0.510049 \\
5 & 2.30065 & -1.6689 & 0.418807 \\
6 & 1.8913  & 1.268 & 0.376804 \\
7 & 1.875 & -1.25 & 0.375 \\
8 & 1.875 & -1.25 & 0.375 \\
\hline
\end{array}$$
这个结果的好处是可以任意截断和叠加，比如只保留前5行它就是最佳的5步迭代，保留前6行就是最佳6步迭代，并且近似程度是有保证地优于5步迭代，依此类推。

至于$\mcsgn$，它只是把$\msign$的$\boldsymbol{M}^{\top}\boldsymbol{M}$换成了$\boldsymbol{M}^2$，所以理论上也可以用Newton\-Schulz迭代，但由于特征值可以是复数，因此一般的收敛会困难得多。不过，如果我们可以实现确认矩阵$\boldsymbol{M}$的特征值都是实数（比如本文后面要应用$\mcsgn$的分块三角矩阵），那么就可以复用$\msign$的迭代和系数：
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\sqrt{\tr(\boldsymbol{M}^2)}},\qquad \boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t^3 + c_{t+1}\boldsymbol{X}_t^5\end{equation}

## 推导过程

下面正式进入主题——求$\boldsymbol{O}=\msign(\boldsymbol{M})$的导数。如果读者只是将Muon当成一个普通优化器用，那么本文多半跟你无关了。当我们需要参考TTT，用Muon优化器来构建RNN模型时，才需要$\msign$的导数，此时$\msign$在模型表现为前向传播，而要对整个模型反向传播，自然就涉及到了$\msign$的导数。

由于$\msign$是通过Newton\-Schulz迭代计算的，实际上它可以直接反向传播，所以$\msign$的数值求导本身不是问题，但基于迭代反传意味着有很多中间状态要存，显存往往要爆炸，所以希望能得到导数的解析解来简化。另一方面，在[《SVD的导数》](<https://spaces.ac.cn/archives/10878>)中我们其实也求过$\msign$的导数，但那是基于SVD的表达式，而SVD并不是GPU高效的算法。

所以，我们的目的是寻求一个不依赖于SVD的、能够高效计算的结果。我们从恒等式
\begin{equation}\boldsymbol{M} = \boldsymbol{O}\boldsymbol{M}^{\top}\boldsymbol{O}\end{equation}
出发（由$\msign$的定义即可证明），两边微分得到
\begin{equation}d\boldsymbol{M} = (d\boldsymbol{O})\boldsymbol{M}^{\top}\boldsymbol{O} + \boldsymbol{O}(d\boldsymbol{M}^{\top})\boldsymbol{O} + \boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})\label{eq:dm-do}\end{equation}
这个结果的难度在于无法简单地分离出$d\boldsymbol{M}=f(d\boldsymbol{O})$或$d\boldsymbol{O}=f(d\boldsymbol{M})$的形式，因此不大好看出$\nabla_{\boldsymbol{O}}\mathcal{L}$与$\nabla_{\boldsymbol{W}}\mathcal{L}$的关系（$\mathcal{L}$是损失函数）。这种情况下最好的办法是回归到矩阵求导的根本思路——“迹技巧”：

> <strong>迹技巧（trace trick）</strong> 如果我们能找到跟$\boldsymbol{M}$同形状的矩阵$\boldsymbol{G}$满足
> \begin{equation}d\mathcal{L}=\langle \boldsymbol{G}, d\boldsymbol{M}\rangle_F = \tr(\boldsymbol{G}^{\top} (d\boldsymbol{M}))\end{equation}
> 那么$\boldsymbol{G} = \nabla_{\boldsymbol{M}}\mathcal{L}$。

迹技巧的要义是化矩阵/向量为标量，然后化标量为迹，继而可以利用迹的恒等式：
\begin{equation}\tr(\boldsymbol{A}\boldsymbol{B}) = \tr(\boldsymbol{B}\boldsymbol{A}) = \tr(\boldsymbol{A}^{\top}\boldsymbol{B}^{\top}) = \tr(\boldsymbol{B}^{\top}\boldsymbol{A}^{\top})\end{equation}
现在设$\boldsymbol{X}$是任意跟$\boldsymbol{M}$同形状矩阵，给式$\eqref{eq:dm-do}$两边乘$\boldsymbol{X}^{\top}$，然后求迹
\begin{equation}\begin{aligned}
\tr(\boldsymbol{X}^{\top}(d\boldsymbol{M})) =&\, \tr(\boldsymbol{X}^{\top}(d\boldsymbol{O})\boldsymbol{M}^{\top}\boldsymbol{O}) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}(d\boldsymbol{M}^{\top})\boldsymbol{O}) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})) \\[7pt]
=&\, \tr(\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top}(d\boldsymbol{O})) + \tr(\boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O}(d\boldsymbol{M}^{\top})) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})) \\[7pt]
=&\, \tr(\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top}(d\boldsymbol{O})) + \tr(\boldsymbol{O}^{\top}\boldsymbol{X}\boldsymbol{O}^{\top}(d\boldsymbol{M})) + \tr(\boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}(d\boldsymbol{O})) \\[7pt]
\end{aligned}\end{equation}
由此可得
\begin{equation}\tr((\boldsymbol{X}^{\top} - \boldsymbol{O}^{\top}\boldsymbol{X}\boldsymbol{O}^{\top})(d\boldsymbol{M})) = \tr((\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top} + \boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top})(d\boldsymbol{O}))\end{equation}
如果我们让$\boldsymbol{M}^{\top}\boldsymbol{O}\boldsymbol{X}^{\top} + \boldsymbol{X}^{\top}\boldsymbol{O}\boldsymbol{M}^{\top}=(\nabla_{\boldsymbol{O}}\mathcal{L})^{\top}$，那么上式便具有$d\mathcal{L}$的含义，那么根据迹技巧就有$\boldsymbol{X}^{\top} - \boldsymbol{O}^{\top}\boldsymbol{X}\boldsymbol{O}^{\top}=(\nabla_{\boldsymbol{M}}\mathcal{L})^{\top}$，这表明$\nabla_{\boldsymbol{M}}\mathcal{L}$和$\nabla_{\boldsymbol{O}}\mathcal{L}$的关系，由下述方程组描述
\begin{gather}\boldsymbol{X} - \boldsymbol{O}\boldsymbol{X}^{\top}\boldsymbol{O} = \nabla_{\boldsymbol{M}}\mathcal{L} \label{eq:g-m}\\[7pt]
\boldsymbol{X}\boldsymbol{O}^{\top}\boldsymbol{M} + \boldsymbol{M}\boldsymbol{O}^{\top}\boldsymbol{X} = \nabla_{\boldsymbol{O}}\mathcal{L}\label{eq:g-o}\end{gather}

## 理论形式

所以，现在问题变成了，从式$\eqref{eq:g-o}$中解出$\boldsymbol{X}$，然后代入式$\eqref{eq:g-m}$得到$\nabla_{\boldsymbol{M}}\mathcal{L}$，即将$\nabla_{\boldsymbol{M}}\mathcal{L}$表示为$\nabla_{\boldsymbol{O}}\mathcal{L}$的函数，避免直接求$\nabla_{\boldsymbol{M}}\boldsymbol{O}$。很明显，唯一的难度就是方程$\eqref{eq:g-o}$的求解。

这一节我们先基于SVD求一个不那么实用的理论解，它可以帮助我们了解方程$\eqref{eq:g-o}$的性质，并且跟之前的结果对齐。设$\boldsymbol{X}=\boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top}$，然后我们还有$\boldsymbol{O}^{\top}\boldsymbol{M} = (\boldsymbol{M}^{\top}\boldsymbol{M})^{1/2} = \boldsymbol{V}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})^{1/2}\boldsymbol{V}^{\top}$和$\boldsymbol{M}\boldsymbol{O}^{\top}=(\boldsymbol{M}\boldsymbol{M}^{\top})^{1/2} = \boldsymbol{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})^{1/2}\boldsymbol{U}^{\top}$，将这些等式代入方程$\eqref{eq:g-o}$得到
\begin{equation}\boldsymbol{U}\boldsymbol{Y}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})^{1/2}\boldsymbol{V}^{\top} + \boldsymbol{U}(\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})^{1/2}\boldsymbol{Y}\boldsymbol{V}^{\top} = \nabla_{\boldsymbol{O}}\mathcal{L}\end{equation}
即
\begin{equation}\boldsymbol{Y}(\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma})^{1/2} + (\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top})^{1/2}\boldsymbol{Y} = \boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}\label{eq:g-o-2}\end{equation}
上式左端如果写成分量形式是$\boldsymbol{Y}_{i,j}\sigma_j + \sigma_i \boldsymbol{Y}_{i,j} = (\sigma_i + \sigma_j)\boldsymbol{Y}_{i,j}$，其中$\sigma_1,\sigma_2,\cdots,\sigma_r$是$\boldsymbol{M}$的非零奇异值，而$0=\sigma_{r+1}=\sigma_{r+2}=\cdots$。很明显，如果当$\boldsymbol{M}$是满秩方阵时，可以解得
\begin{equation}\boldsymbol{Y} = (\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}) \oslash \boldsymbol{S}\end{equation}
其中$\boldsymbol{S}_{i,j} = \sigma_i+\sigma_j$，$\oslash$是Hadamard除（逐位相除）。这时候我们将$\boldsymbol{X}=\boldsymbol{U}\boldsymbol{Y}\boldsymbol{V}^{\top}$代入式$\eqref{eq:g-m}$，就得到跟[《SVD的导数》](<https://spaces.ac.cn/archives/10878>)里边一致的结果。这个殊途同归也增强了我们的信心，看来至少到目前为止我们的推导都还是正确的。

若$\boldsymbol{M}$不满秩或不是方阵呢？此时如果右端的$\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}$“不配合”，那么方程$\eqref{eq:g-o-2}$无解。但方程$\eqref{eq:g-o-2}$是从实际问题得到的，所以它肯定有解，那么右端“必须配合”！怎样才是配合呢？如果$\boldsymbol{M}$的秩为$r$，那么矩阵$\boldsymbol{S}$只有$\boldsymbol{S}_{[:r,:r]}$是非零的，为了使得方程$\eqref{eq:g-o-2}$有解，$(\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V})_{[:r,:r]}$以外的部分只能是零。在这个条件下，我们可以写出
\begin{equation}\boldsymbol{Y} = \lim_{\epsilon\to 0}\,\, (\boldsymbol{U}^{\top}(\nabla_{\boldsymbol{O}}\mathcal{L})\boldsymbol{V}) \oslash (\boldsymbol{S} + \epsilon) \end{equation}
这相当于说，我们可以给奇异值加些扰动，转化为全体奇异值非零的情况，计算完成后再让扰动趋于零，从而得到正确的结果。

## 高效求解

上一节的SVD解往往只有理论价值，为了在GPU中高效计算，我们还需要寻求其他形式的解。引入记号$\boldsymbol{M}\boldsymbol{O}^{\top}=\boldsymbol{A},\boldsymbol{O}^{\top}\boldsymbol{M}=\boldsymbol{B},\nabla_{\boldsymbol{O}}\mathcal{L}=\boldsymbol{C}$，那么式$\eqref{eq:g-o}$实际上是一个[Sylvester方程](<https://en.wikipedia.org/wiki/Sylvester_equation>)：
\begin{equation}\boldsymbol{A}\boldsymbol{X}+\boldsymbol{X}\boldsymbol{B} = \boldsymbol{C}\end{equation}
求解Sylvester方程的方法有很多，其中最精妙、对GPU最高效的，可能是基于$\mcsgn$（不是$\msign$）的求解方案（这里参考了[《Fast Differentiable Matrix Square Root》](<https://papers.cool/arxiv/2201.08663>)）。首先，从上述方程出发，我们可以验证下式成立
\begin{equation}\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix} = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{A} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1}
\end{equation}
两边取$\mcsgn$，根据$\mcsgn$的性质，我们有
\begin{equation}\mcsgn\left(\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\right) = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \mcsgn(\boldsymbol{A}) & \boldsymbol{0} \\ \boldsymbol{0} & -\mcsgn(\boldsymbol{B})\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1}
\end{equation}
注意$\boldsymbol{A}=\boldsymbol{M}\boldsymbol{O}^{\top}=(\boldsymbol{M}\boldsymbol{M}^{\top})^{1/2}, \boldsymbol{B}=\boldsymbol{O}^{\top}\boldsymbol{M}=(\boldsymbol{M}^{\top}\boldsymbol{M})^{1/2}$，假设$\boldsymbol{M}$是满秩方阵，那么$\boldsymbol{A},\boldsymbol{B}$都是正定对称的，正定对称矩阵的$\mcsgn$都是方阵，所以
\begin{equation}\mcsgn\left(\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}\right) = \begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1} = \begin{bmatrix} \boldsymbol{I} & -2\boldsymbol{X} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}
\end{equation}
最后的化简用到了等式$\begin{bmatrix} \boldsymbol{I} & \boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}^{-1}=\begin{bmatrix} \boldsymbol{I} & -\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}$。从该结果可以看出，我们只需要对分块矩阵$\begin{bmatrix} \boldsymbol{A} & -\boldsymbol{C} \\ \boldsymbol{0} & -\boldsymbol{B}\end{bmatrix}$算$\mcsgn$，然后就可以从结果的右上角读出$\boldsymbol{X}$。$\mcsgn$可以通过Newton\-Schulz迭代高效地计算，因此该方案是GPU友好的。

当$\boldsymbol{M}$不满秩或非方阵时，$\boldsymbol{A},\boldsymbol{B}$只是半正定的，这时候它们就$\mcsgn$就不是$\boldsymbol{I}$。不过，上一节的经验告诉我们，由于$\nabla_{\boldsymbol{O}}\mathcal{L}$“必须配合”，所以只需要给$\boldsymbol{\Sigma}$加点扰动，让它变成正定的情况即可解。这里给$\boldsymbol{\Sigma}$加扰动，相当于给$\boldsymbol{A},\boldsymbol{B}$加$\epsilon \boldsymbol{I}$，所以
\begin{equation}\boldsymbol{X} = -\frac{1}{2} \left(\lim_{\epsilon\to 0}\,\, \mcsgn\left(\begin{bmatrix} \boldsymbol{A} + \epsilon \boldsymbol{I} & -\boldsymbol{C} \\ \boldsymbol{0} & -\boldsymbol{B} - \epsilon \boldsymbol{I}\end{bmatrix}\right)\right)_{[:n,n:]}
\end{equation}
实际计算时，就只能选择一个比较小的$\epsilon > 0$来近似计算了，可以考虑$\epsilon=10^{-3}$，它在我们之前寻找Newton\-Schulz迭代的下界范围内。

## 文章小结

本文讨论了$\msign$算子的导数计算，如果你关心“TTT \+ Muon”的组合，那么本文也许对你有帮助。
