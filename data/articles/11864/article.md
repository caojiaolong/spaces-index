# 流形上的最速下降：7\. Stiefel的解析解

> 作者：苏剑林 · 科学空间 · 2026-08-17
>
> 原文：<https://spaces.ac.cn/archives/11864>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

对于正交流形，我们此前的结论是：方阵情形下，我们可以完整地写出对应最速下降的解析解（参考[《流形上的最速下降：2\. Muon \+ 正交》](<https://spaces.ac.cn/archives/11215>)）；但对于非方阵的Stiefel流形，其最速下降问题需要求解一个非线性方程组，其解析解无法直接写出（参考[《流形上的最速下降：3\. Muon \+ Stiefel》](<https://spaces.ac.cn/archives/11221>)）。

但近日，[《Muon on the Stiefel Manifold Admits an Exact Closed\-Form Update》](<https://papers.cool/arxiv/2608.06218>)推翻了这个断言，它指出Stiefel流形上同样可以写出显式解，无须求解方程组。本文将用自己的思路，复述整个推导过程。

## 问题回顾

问题背景我们就不过多复述了，直接给出要求解的问题：设$\boldsymbol{W}\in\mathbb{R}^{n\times m}(n \geq m)$是当前参数，$\boldsymbol{G}$是目标函数在$\boldsymbol{W}$处的梯度，我们要找的是正交约束下步长为$\eta$的最速下降方向$\boldsymbol{\Phi}\in\mathbb{R}^{n\times m}$，即
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 \leq 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}_m,\,\,(\boldsymbol{W} - \eta \boldsymbol{\Phi})^{\top}(\boldsymbol{W} - \eta \boldsymbol{\Phi})=\boldsymbol{I}_m\end{equation}
按照“一阶近似够用”原则，将最后一个约束展开并略去$\eta^2$项，简化成
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 \leq 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}_m,\,\,\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}\end{equation}
其中$\Vert\cdot\Vert_2$是谱范数，$\boldsymbol{I}_m$是$m\times m$单位阵。这里$\Vert\boldsymbol{\Phi}\Vert_2 \leq 1$改$\Vert\boldsymbol{\Phi}\Vert_2 = 1$结果是完全等价的（线性目标函数的最优点总能在边界上取到），但这里我们保留$\leq$，因为$\Vert\boldsymbol{\Phi}\Vert_2 \leq 1$是一个凸集，论证时的腾挪空间更大一些。此前我们的求解结果是
\begin{equation}\boldsymbol{\Phi} = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})\label{eq:Phi-ori}\end{equation}
其中$\boldsymbol{X}\in\mathbb{R}^{m\times m}$是对称矩阵，并且满足如下矩阵方程：
\begin{equation}\boldsymbol{W}^{\top}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})+\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})^{\top}\boldsymbol{W} = \boldsymbol{0}\label{eq:X-ori}\end{equation}
$n=m$时的解我们已在[《流形上的最速下降：2\. Muon \+ 正交》](<https://spaces.ac.cn/archives/11215>)求出，但对于$n > m$的标准Stiefel流形，该方程的求解并不是平凡的，即便是数值求解也有些麻烦，“Stiefel情形没有解析解”的断言便由此而来。

## 弱化版本

破局的关键，是通过某种显式的表示去掉等式约束$\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}$。为此，我们先引入原问题的一个弱化版本：设$\boldsymbol{\Phi} = \boldsymbol{S}\boldsymbol{W}$，$\boldsymbol{S}\in\mathbb{R}^{n\times n}$是任意反对称矩阵，并且满足$\Vert\boldsymbol{S}\Vert_2\leq 1$，那么我们考虑求解
\begin{equation}\max_{\boldsymbol{S}} \tr(\boldsymbol{G}^{\top}\boldsymbol{S}\boldsymbol{W}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{S}\Vert_2 \leq 1\end{equation}
之所以说这是原问题的弱化，是因为这样设出来的$\boldsymbol{\Phi}$显然满足原命题两个约束条件：
\begin{align}1) &\quad \Vert\boldsymbol{\Phi}\Vert_2 = \Vert\boldsymbol{S}\boldsymbol{W}\Vert_2\leq \Vert\boldsymbol{S}\Vert_2 \Vert\boldsymbol{W}\Vert_2\leq 1 \\[5pt]
2) &\quad \boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{W}^{\top}\boldsymbol{S}\boldsymbol{W}+\boldsymbol{W}^{\top}\boldsymbol{S}^{\top}\boldsymbol{W} = \boldsymbol{W}^{\top}(\boldsymbol{S} + \boldsymbol{S}^{\top})\boldsymbol{W} = \boldsymbol{0}
\end{align}
所以新问题是在原问题可行域的子集上求解的，求出来的最大值不超过原问题的最大值。进一步地，利用迹的恒等式和$\boldsymbol{S}$的反对称性，可以将目标函数变换成
\begin{equation}\tr(\boldsymbol{G}^{\top}\boldsymbol{S}\boldsymbol{W}) = \tr(\boldsymbol{W}\boldsymbol{G}^{\top}\boldsymbol{S}) = -\tr(\boldsymbol{W}\boldsymbol{G}^{\top}\boldsymbol{S}^{\top}) = -\tr(\boldsymbol{G}\boldsymbol{W}^{\top}\boldsymbol{S}) = \tr([\boldsymbol{W}\boldsymbol{G}^{\top}]_{\text{skew}}\boldsymbol{S})\end{equation}
其中$[\boldsymbol{X}]_{\text{skew}} = (\boldsymbol{X} - \boldsymbol{X}^{\top})/2$。所以新问题等价于
\begin{equation}\max_{\boldsymbol{S}}\tr([\boldsymbol{W}\boldsymbol{G}^{\top}]_{\text{skew}}\boldsymbol{S}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{S}\Vert_2 \leq 1,\,\,\boldsymbol{S} + \boldsymbol{S}^{\top}=\boldsymbol{0}\end{equation}
这已经是标准Muon问题的形状了，由Muon的结果可以直接得出
\begin{equation}\boldsymbol{S} = \msign([\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}})\qquad\Rightarrow\qquad \boldsymbol{\Phi} = \msign([\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}})\boldsymbol{W}\label{eq:sol}\end{equation}
实际上，这正好是方阵情形的解的形式，也是[《流形上的最速下降：6\. Muon \+ 双旋转》](<https://spaces.ac.cn/archives/11777>)中的左侧旋转。这里值得解释一下的是，为什么要经过一通恒等变换后才应用Muon的$\msign$？这主要是利用“反对称矩阵的$\msign$依然是反对称矩阵”这一结论，来保证最终结果满足$\boldsymbol{S}$的反对称性要求。

## 完全等价

接下来，我们将要证明的是，上一节我们求解的弱化版问题，跟原始问题的最优解实际上是一致的！

证明思路也很直接，弱化问题之所以“弱”，是因为目前参数化$\boldsymbol{\Phi} = \boldsymbol{S}\boldsymbol{W}$只是原问题可行域的一个子集，它可能会漏掉一些探索区域，从而无法达到原问题的最优解。但如果我们能证明，原问题任意一个可行的$\boldsymbol{\Phi}$，都能找到一个满足条件的$\boldsymbol{S}$使得$\boldsymbol{\Phi} = \boldsymbol{S}\boldsymbol{W}$，那就说明两者的探索空间完全一致，所以最优解也一致了。

接下来的证明分三步进行。（注：事后来看，给适当的提示，K3也能独立完成推导和证明，参考[这里](<https://www.kimi.com/share/1a011285-7f82-87e0-8000-0000da55cf07>)和[这里](<https://www.kimi.com/share/1a0110e7-c702-8915-8000-00000fff70c4>)。）

### 解方程

首先要做的是，给定$\boldsymbol{\Phi},\boldsymbol{W}$，解方程
\begin{equation}\boldsymbol{\Phi} = \boldsymbol{S}\boldsymbol{W},\qquad \boldsymbol{S}+\boldsymbol{S}^{\top} = \boldsymbol{0}\end{equation}
不失一般性，只考虑$n > m$，那么我们可以将$\boldsymbol{W}$补全成$n\times n$的正交矩阵$\boldsymbol{U} = [\boldsymbol{W}, \boldsymbol{W}_{\perp}]$。由于$\boldsymbol{S}$是反对称的，因此$\boldsymbol{U}^{\top}\boldsymbol{S}\boldsymbol{U}$必然也是，所以我们可以将$\boldsymbol{S}$写成
\begin{equation}\boldsymbol{S} = \boldsymbol{U}\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}\end{bmatrix}\boldsymbol{U}^{\top} = [\boldsymbol{W}, \boldsymbol{W}_{\perp}]\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}\end{bmatrix}\begin{bmatrix}\boldsymbol{W}^{\top} \\ \boldsymbol{W}_{\perp}^{\top}\end{bmatrix}\label{eq:S-block}\end{equation}
其中$\boldsymbol{A}\in\mathbb{R}^{m\times m},\boldsymbol{C}\in\mathbb{R}^{(n-m)\times (n-m)}$都是反对称矩阵，$\boldsymbol{B}\in\mathbb{R}^{(n-m)\times m}$。接着两端右乘$\boldsymbol{W}$得
\begin{equation}\boldsymbol{\Phi} = \boldsymbol{S}\boldsymbol{W} = [\boldsymbol{W}, \boldsymbol{W}_{\perp}]\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}\end{bmatrix}\begin{bmatrix}\boldsymbol{I}_m \\ \boldsymbol{0}\end{bmatrix} = [\boldsymbol{W}, \boldsymbol{W}_{\perp}]\begin{bmatrix}\boldsymbol{A} \\ \boldsymbol{B}\end{bmatrix}\end{equation}
两端左乘$\boldsymbol{U}^{\top}$，可以直接读出$\boldsymbol{A} = \boldsymbol{W}^{\top}\boldsymbol{\Phi},\boldsymbol{B} = \boldsymbol{W}_{\perp}^{\top}\boldsymbol{\Phi}$，即$\boldsymbol{A},\boldsymbol{B}$都可以唯一解出，剩下$\boldsymbol{C}$这一自由度。注意$\boldsymbol{A}$是反对称矩阵，这就提出了相容性条件
\begin{equation}\boldsymbol{0} = \boldsymbol{A} + \boldsymbol{A}^{\top} = \boldsymbol{W}^{\top}\boldsymbol{\Phi} + \boldsymbol{\Phi}^{\top}\boldsymbol{W}\end{equation}
而这正是原问题的条件之一。所以到目前为止，全部结果都是自洽的。

### 谱范数

现在剩下的问题是，我们能否找到反对称矩阵$\boldsymbol{C}$，使得$\Vert \boldsymbol{S}\Vert_2 \leq 1$。答案是肯定的，但证明需要用到一个“[Parrott引理](<https://www.sciencedirect.com/science/article/pii/0022123678900605>)”（原论文引用的是[Davis–Kahan–Weinberger](<https://epubs.siam.org/doi/10.1137/0719029>)，但调研发现Parrott引理更早）：

> 设有分块矩阵$\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ 
\boldsymbol{B} & \boldsymbol{D}\end{bmatrix}$，固定$\boldsymbol{A},\boldsymbol{B},\boldsymbol{C}$，我们总能找到矩阵$\boldsymbol{D}$，使得
> \begin{equation}\left\Vert\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ 
\boldsymbol{B} & \boldsymbol{D}\end{bmatrix}\right\Vert_2 = \max\left\{\left\Vert\begin{bmatrix}\boldsymbol{A} \\ 
\boldsymbol{B}\end{bmatrix}\right\Vert_2, \left\Vert\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C}\end{bmatrix}\right\Vert_2\right\}\end{equation}

回到我们的问题上。要使$\Vert \boldsymbol{S}\Vert_2 \leq 1$，只需$\left\Vert\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}\end{bmatrix}\right\Vert_2\leq 1$，对于这个分块矩阵，我们有
\begin{equation}\left\Vert\begin{bmatrix}\boldsymbol{A} \\
\boldsymbol{B}\end{bmatrix}\right\Vert_2 = \left\Vert\begin{bmatrix}\boldsymbol{W}^{\top}\boldsymbol{\Phi} \\
\boldsymbol{W}_{\perp}^{\top}\boldsymbol{\Phi}\end{bmatrix}\right\Vert_2 = \Vert\boldsymbol{U}^{\top}\boldsymbol{\Phi}\Vert_2 = \Vert\boldsymbol{\Phi}\Vert_2 \leq 1\end{equation}
至于$[\boldsymbol{A},-\boldsymbol{B}^{\top}] = [-\boldsymbol{A}^{\top},-\boldsymbol{B}^{\top}]$是$\begin{bmatrix}\boldsymbol{A} \\
\boldsymbol{B}\end{bmatrix}$的负转置，谱范数不变（同样不超过1）。因此，根据Parrott引理，我们能找到矩阵$\boldsymbol{C}$，使得$\Vert \boldsymbol{S}\Vert_2 = \left\Vert\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}\end{bmatrix}\right\Vert_2 \leq 1$。

### 反对称

不过事情还没完。我们对$\boldsymbol{C}$还有一个反对称的要求，Parrott引理并不保证这一点。

幸运的是，这个修补并不困难。设Parrott引理给出的矩阵是$\boldsymbol{C}_0$，我们直接取$\boldsymbol{C} = [\boldsymbol{C}_0]_{\text{skew}}$，由于剩下部分本就满足反对称要求，因此有
\begin{equation}\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & [\boldsymbol{C}_0]_{\text{skew}}\end{bmatrix} = \frac{1}{2}\left(\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}_0\end{bmatrix} - \begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}_0\end{bmatrix}^{\top}\right) \end{equation}
这里的关键之处是，反对称化不会增加谱范数：由三角不等式可得
\begin{equation}\left\Vert\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & [\boldsymbol{C}_0]_{\text{skew}}\end{bmatrix}\right\Vert_2 \leq \frac{1}{2}\left\Vert\begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}_0\end{bmatrix}\right\Vert_2 + \frac{1}{2}\left\Vert \begin{bmatrix}\boldsymbol{A} & -\boldsymbol{B}^{\top} \\
\boldsymbol{B} & \boldsymbol{C}_0\end{bmatrix}^{\top}\right\Vert_2 \leq \frac{1}{2} + \frac{1}{2} = 1 \end{equation}
所以$[\boldsymbol{C}_0]_{\text{skew}}$就是我们最终要寻找的反对称矩阵。

至此，我们证明了最初希望证明的结果：对于原问题的每个$\boldsymbol{\Phi}$，都能找到一个$n\times n$的、反对称的、谱范数不超过1的矩阵$\boldsymbol{S}$，使得$\boldsymbol{\Phi}=\boldsymbol{S}\boldsymbol{W}$，所以弱化版的探索空间跟原问题的探索空间一致。换言之，式$\eqref{eq:sol}$就是原问题的精确解！

## 高效计算

从理论上看，问题已经圆满解决；但从实践上看，还有一个效率问题值得讨论。Muon需要对一个$n\times m$的矩阵做$\msign$，而Stiefel流形上的精确解$\eqref{eq:sol}$，则需要对一个$n\times n$的矩阵做$\msign$，如果直接计算的话，在$n \gg m$时它显然比Muon要昂贵得多，因此有必要探讨一下节省计算量的方案。

其实也不难想，首先将核心运算改写成低秩形式
\begin{equation}\msign([\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}}) = \msign(\boldsymbol{G}\boldsymbol{W}^{\top} - \boldsymbol{W}\boldsymbol{G}^{\top}) = \msign\bigg(\begin{bmatrix}\boldsymbol{G} & \boldsymbol{W}\end{bmatrix} \underbrace{\begin{bmatrix}\boldsymbol{0} & \boldsymbol{I}_m \\ -\boldsymbol{I}_m & \boldsymbol{0}\end{bmatrix}}_{\boldsymbol{J}} \begin{bmatrix}\boldsymbol{G}^{\top} \\ \boldsymbol{W}^{\top}\end{bmatrix}\bigg)\end{equation}
可见$[\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}}$的秩至多是$2m$。假设$n \geq 2m$，先将$[\boldsymbol{G},\boldsymbol{W}]$做QR分解（用$\msign$做Polar分解也行）得$\boldsymbol{Q}\boldsymbol{R}$，其中$\boldsymbol{Q}\in\mathbb{R}^{n\times 2m}$满足$\boldsymbol{Q}^{\top}\boldsymbol{Q}=\boldsymbol{I}_{2m}$，$\boldsymbol{R}\in\mathbb{R}^{2m\times 2m}$是一个上三角阵，那么待计算部分变成$\msign(\boldsymbol{Q}\boldsymbol{R}\boldsymbol{J}\boldsymbol{R}^{\top} \boldsymbol{Q}^{\top})$。注意$\msign$运算对于正交矩阵具有协变性，即
\begin{equation}\msign(\boldsymbol{Q}\boldsymbol{R}\boldsymbol{J}\boldsymbol{R}^{\top} \boldsymbol{Q}^{\top}) = \boldsymbol{Q}\msign(\boldsymbol{R}\boldsymbol{J}\boldsymbol{R}^{\top} )\boldsymbol{Q}^{\top}\end{equation}
所以实际上我们只需对$2m\times 2m$大小的矩阵$\boldsymbol{R}\boldsymbol{J}\boldsymbol{R}^{\top}$做一次$\msign$，这对于$n\gg 2m$的场景来说，加速是明显的。但如果$n$与$m$没有差一个数量级以上，还是按照原始公式计算更为便捷，毕竟QR分解也是需要成本的。

此外，如何切换到动量场景也值得讨论一下。我们有两种可以考虑的方案：第一种是$\boldsymbol{M} = \mathop{\text{EMA}}(\boldsymbol{G})$，这时候只需要把$\boldsymbol{G}$换成$\boldsymbol{M}$，其余不变；另一种是$\boldsymbol{M} = \mathop{\text{EMA}}(\boldsymbol{G}\boldsymbol{W}^{\top})$，这时候$[\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}}$换成$[\boldsymbol{M}]_{\text{skew}}$，这种形式能保留更多信息，代价是占用空间更大，且在长期滑动平均之后，即便$n\gg 2m$，$\boldsymbol{M}$通常也不会低秩，因此无法加速。

当然，其实主流的模型架构下，$n\gg 2m$的场景并不多（只有Per\-Head Muon时，每个Head的矩阵勉强算得上），所以这个加速技巧的实践意义有限，我们稍作了解即可。

## 开放问题

这里笔者提出一个开放问题：如果去掉正交性约束，只考虑切流形的最速下降，其最优解又是怎样的呢？即考虑
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 \leq 1,\,\,\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}\end{equation}
这个问题可以看作“Stiefel版Muon”在一般矩阵上的推广，它不依赖于$\boldsymbol{W}$本身的正交性，因此有望适用于更一般的场景。笔者目前只能得到的结果是，式$\eqref{eq:Phi-ori}$和式$\eqref{eq:X-ori}$依然适用，即最优解有如下形式：
\begin{equation}\boldsymbol{\Phi} = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})\end{equation}
其中$\boldsymbol{X}\in\mathbb{R}^{m\times m}$是满足如下矩阵方程的对称矩阵：
\begin{equation}\boldsymbol{W}^{\top}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})+\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})^{\top}\boldsymbol{W} = \boldsymbol{0}\end{equation}
至于$\boldsymbol{W}^{\top}\boldsymbol{W}\neq\boldsymbol{I}_m$时怎么求解，暂时不得而知（假设$n=m$似乎也无法简化）。另外，[弱化问题](<https://spaces.ac.cn/archives/11864#%E5%BC%B1%E5%8C%96%E7%89%88%E6%9C%AC>)的构造对它也是成立的，只需稍微修改一下
\begin{equation}\boldsymbol{\Phi}_{\text{weak}} = \msign([\boldsymbol{G}\boldsymbol{W}^{\top}]_{\text{skew}})\boldsymbol{W}/\Vert\boldsymbol{W}\Vert_2\end{equation}
但由于前面的等价性证明强依赖于$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}_m$，所以看上去这个弱化解真的只是弱化解，而不是最优解。欢迎大家提供更多的进展～

## 文章小结

在这篇文章中，我们成功为“Muon \+ Stiefel”的流形最速下降找到解析解，为该问题划上一个圆满的句号。自此，不管是否方阵，正交约束下的Muon优化器都可解析计算，求解本身将不再是困难。
