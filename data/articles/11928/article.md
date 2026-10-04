# 通过微扰分析求解等式约束优化问题

> 作者：苏剑林 · 科学空间 · 2026-10-03
>
> 原文：<https://spaces.ac.cn/archives/11928>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

对于无约束的光滑函数来说，求极值的通用方法自然是求导（梯度）并让导数等于零了。那带等式约束的极值问题呢？很多读者应该也不陌生，标准方法是拉格朗日乘子法。然而，拉格朗日乘子总让人觉得是一种纯粹的代数技巧，没有求无约束极值时那种微扰分析的直观感觉。

这篇文章我们就来尝试一下：如果将无约束极值的微扰思想贯彻到约束优化中，会得到什么结果？

## 驻点条件

首先来重温一下，为什么梯度等于零就是无约束极值的必要条件（驻点条件）。考虑目标函数$f(\boldsymbol{x})$，其中$\boldsymbol{x}\in\mathbb{R}^d$，我们想要求$f(\boldsymbol{x})$的最小值点，我们在$\boldsymbol{x}$附近做一阶近似
\begin{equation}f(\boldsymbol{x} + \delta\boldsymbol{x}) = f(\boldsymbol{x}) + \delta\boldsymbol{x}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})\end{equation}
如果$f(\boldsymbol{x})$已经是最小值，那么对于任意微扰$\delta\boldsymbol{x}\neq \boldsymbol{0}$，必然恒成立$f(\boldsymbol{x} + \delta\boldsymbol{x})\geq f(\boldsymbol{x})$，这意味着$\delta\boldsymbol{x}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})\geq 0$，但$\delta\boldsymbol{x}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})$是关于$\delta\boldsymbol{x}$的线性函数，它要想始终非负，那么只有
\begin{equation}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})=\boldsymbol{0}\end{equation}
这一选择。同样的讨论对最大值也成立，所以这是极大值、极小值共同的驻点条件。而当驻点条件成立时，近似展开的下一主导项是
\begin{equation}f(\boldsymbol{x} + \delta\boldsymbol{x}) = f(\boldsymbol{x}) +  \frac{1}{2}\delta\boldsymbol{x}^{\top}\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x}) \,\delta\boldsymbol{x}\end{equation}
根据同样的原理，要想$f(\boldsymbol{x} + \delta\boldsymbol{x})\geq f(\boldsymbol{x})$恒成立，那么$\delta\boldsymbol{x}^{\top}\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x}) \,\delta\boldsymbol{x}\geq 0$恒成立，即Hessian矩阵$\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x})$半正定，这是最小值点的必要条件；反之，最大值点的必要条件则是$\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x})$半负定。

当然这个讨论并不是严格的数学证明，只能算是一个启发式推导。

## 等式约束

现在再来看，当引入等式约束后，微扰过程将会发生什么变化。考虑
\begin{equation}\min_{\boldsymbol{x}} f(\boldsymbol{x})\qquad\text{s.t.}\qquad \boldsymbol{g}(\boldsymbol{x})=\boldsymbol{0}\end{equation}
其中$\boldsymbol{g}(\boldsymbol{x})\in\mathbb{R}^k$，代表有$k$个等式约束。任取一个满足$\boldsymbol{g}(\boldsymbol{x})=\boldsymbol{0}$的$\boldsymbol{x}$，同样考虑微扰的一阶近似得
\begin{equation}f(\boldsymbol{x} + \delta\boldsymbol{x}) = f(\boldsymbol{x}) + \delta\boldsymbol{x}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})\end{equation}
类似地，$f(\boldsymbol{x})$是最小值的必要条件依然是$\delta\boldsymbol{x}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})\geq 0$恒成立，但区别在于，此时不需要对任意的微扰$\delta\boldsymbol{x}$都恒成立，而只需要对所有满足$\boldsymbol{g}(\boldsymbol{x} + \delta\boldsymbol{x})=\boldsymbol{0}$的$\delta\boldsymbol{x}$恒成立。对$\boldsymbol{g}(\boldsymbol{x} + \delta\boldsymbol{x})=\boldsymbol{0}$做一阶近似，得
\begin{equation}\boldsymbol{J}\delta\boldsymbol{x} = \boldsymbol{0},\qquad \boldsymbol{J} = \nabla_{\boldsymbol{x}} \boldsymbol{g}(\boldsymbol{x})\in\mathbb{R}^{k\times d}\end{equation}
注意到$\boldsymbol{J}\delta\boldsymbol{x} = \boldsymbol{0}$是关于$\delta\boldsymbol{x}$的线性方程组，可以用“[伪逆](<https://spaces.ac.cn/archives/10366>)”写出它的通解：
\begin{equation}\delta\boldsymbol{x} = (\boldsymbol{I} - \boldsymbol{J}^{\dagger}\boldsymbol{J})\delta\boldsymbol{z}\end{equation}
其中$\boldsymbol{J}^{\dagger}$是$\boldsymbol{J}$的伪逆，$\delta\boldsymbol{z}$表示任意无约束的微扰向量，当$\boldsymbol{J}\boldsymbol{J}^{\top}$可逆时$\boldsymbol{J}^{\dagger}=\boldsymbol{J}^{\top}(\boldsymbol{J}\boldsymbol{J}^{\top})^{-1}$，由此可见$\boldsymbol{J}^{\dagger}\boldsymbol{J}$是对称矩阵。将上式代入$\delta\boldsymbol{x}^{\top} \nabla_{\boldsymbol{x}} f(\boldsymbol{x})\geq 0$并由$\delta\boldsymbol{z}$的任意性得驻点条件
\begin{equation}(\boldsymbol{I} - \boldsymbol{J}^{\dagger} \boldsymbol{J}) \nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \boldsymbol{0} \label{eq:stable-g}\end{equation}
其中$(\boldsymbol{I} - \boldsymbol{J}^{\dagger} \boldsymbol{J}) \nabla_{\boldsymbol{x}} f(\boldsymbol{x})$称为$f(\boldsymbol{x})$在流形$\boldsymbol{g}(\boldsymbol{x})=\boldsymbol{0}$上的投影梯度，它是欧氏梯度$\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$在切空间上的正交投影，在黎曼几何的语言中，它又叫黎曼梯度。

## 拉格朗日

现在我们将它与拉格朗日乘子法做个对比。拉格朗日乘子法通过引入待定系数$\boldsymbol{c}\in\mathbb{R}^k$，将目标函数改为
\begin{equation}F(\boldsymbol{x}, \boldsymbol{c}) = f(\boldsymbol{x}) - \boldsymbol{c}^{\top}\boldsymbol{g}(\boldsymbol{x})\end{equation}
然后改为求无约束目标$F(\boldsymbol{x}, \boldsymbol{c})$的驻点，那么直接令两边梯度等于零得
\begin{equation}\nabla_{\boldsymbol{x}} F(\boldsymbol{x}, \boldsymbol{c}) = \nabla_{\boldsymbol{x}} f(\boldsymbol{x}) - \boldsymbol{J}^{\top}\boldsymbol{c} = \boldsymbol{0},\qquad \nabla_{\boldsymbol{c}} F(\boldsymbol{x}, \boldsymbol{c}) = -\boldsymbol{g}(\boldsymbol{x}) = \boldsymbol{0}\end{equation}
注意伪逆成立$\boldsymbol{J}^{\dagger} \boldsymbol{J}=\boldsymbol{J}^{\top}(\boldsymbol{J}^{\top})^{\dagger}$，所以式$\eqref{eq:stable-g}$又可以写成$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) - \boldsymbol{J}^{\top}(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \boldsymbol{0}$，如果记$(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$为$\boldsymbol{c}$，那么就跟上式一模一样了，即从式$\eqref{eq:stable-g}$可以推出跟拉格朗日乘子法一样的条件。

那么反过来，从拉格朗日乘子法能否推出式$\eqref{eq:stable-g}$呢？自然也可以。假设$k < d$且$\boldsymbol{J}$的秩为$k$，那么将$\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) - \boldsymbol{J}^{\top}\boldsymbol{c} = \boldsymbol{0}$看作关于$\boldsymbol{c}$的方程，它是一个超定方程，如果有解只能是$\boldsymbol{c}=(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$，代回方程后便得到式$\eqref{eq:stable-g}$。

这表明两者给出的条件在数学上是完全等价的。当然，拉格朗日乘子法的写法看起来会更干净一些，但笔者认为微扰思想上与无约束情形更为一致，所以更偏爱这种理解。

此外，拉格朗日乘子法的简单只是形式上的，要想真正理解它，并不比微扰视角容易。读者完全可以将从微扰思想到投影梯度的过程，看作是拉格朗日乘子法的一种推导。

## 两个例子

我们通过以下两个例子来加深理解。

### 球面约束

第一个例子，考虑简单的球面极值
\begin{equation}\min_{\boldsymbol{x}} f(\boldsymbol{x})\qquad\text{s.t.}\qquad \Vert\boldsymbol{x}\Vert_2=1\end{equation}
容易求得$\nabla_{\boldsymbol{x}}\Vert\boldsymbol{x}\Vert_2 = \boldsymbol{x}/\Vert\boldsymbol{x}\Vert_2 = \boldsymbol{x}$，所以$\boldsymbol{J} = \boldsymbol{x}^{\top}$，由于$\boldsymbol{x}$是单位向量，所以$\boldsymbol{J}^{\dagger} = \boldsymbol{x}$，因此驻点条件为
\begin{equation}\boldsymbol{0} = (\boldsymbol{I} - \boldsymbol{x}\boldsymbol{x}^{\top})\nabla_{\boldsymbol{x}} f(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} f(\boldsymbol{x}) - \boldsymbol{x}(\boldsymbol{x}^{\top}\nabla_{\boldsymbol{x}} f(\boldsymbol{x}))\end{equation}
最后的式子，正是梯度在球面切空间上的投影。

### 正交约束

另一个例子是Stiefel流形上的优化问题
\begin{equation}\min_{\boldsymbol{X}} f(\boldsymbol{X})\qquad\text{s.t.}\qquad \boldsymbol{X}^{\top}\boldsymbol{X} = \boldsymbol{I}_m,\quad\boldsymbol{X}\in\mathbb{R}^{n\times m}\,(n\geq m)\end{equation}
如果只是想求驻点条件，用拉格朗日乘子法做会快不少，但这里还是完整地把投影梯度求出来。这里的困难在于$\boldsymbol{X}$本身就是一个矩阵，那么$\boldsymbol{J}$将会是一个高阶矩阵，思考的心智负担较大。

对此，我们采用线性算子视角，对$\boldsymbol{X}^{\top}\boldsymbol{X} - \boldsymbol{I}_m$做一阶近似得$\delta\boldsymbol{X}^{\top}\boldsymbol{X} + \boldsymbol{X}^{\top}\delta\boldsymbol{X}$，按照前面的推导，它应该等于$\boldsymbol{J}\delta\boldsymbol{x}$，我们直接定义线性算子
\begin{equation}\boldsymbol{J}[\boldsymbol{V}] = \boldsymbol{V}^{\top}\boldsymbol{X} + \boldsymbol{X}^{\top}\boldsymbol{V}\end{equation}
下面我们需要推导$\boldsymbol{J}^{\top}[\boldsymbol{V}]$，它按照$\langle\boldsymbol{U}, \boldsymbol{J}^{\top}[\boldsymbol{V}]\rangle_F=\langle\boldsymbol{V}, \boldsymbol{J}[\boldsymbol{U}]\rangle_F$来理解（类比$\boldsymbol{u}^{\top}\boldsymbol{J}^{\top}\boldsymbol{v}=\boldsymbol{v}^{\top}\boldsymbol{J}\boldsymbol{u}$），$\boldsymbol{J}[\boldsymbol{U}]$和$\langle\cdot,\cdot\rangle_F$我们都是已知的，所以有
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\langle\boldsymbol{U}, \boldsymbol{J}^{\top}[\boldsymbol{V}]\rangle_F = \tr(\boldsymbol{V}^{\top}(\boldsymbol{U}^{\top}\boldsymbol{X} + \boldsymbol{X}^{\top}\boldsymbol{U})) = \tr(\boldsymbol{U}^{\top}\boldsymbol{X}(\boldsymbol{V} + \boldsymbol{V}^{\top})) = \langle\boldsymbol{U}, \boldsymbol{X}(\boldsymbol{V} + \boldsymbol{V}^{\top})\rangle_F \end{equation}
由此可读出$\boldsymbol{J}^{\top}[\boldsymbol{V}] = \boldsymbol{X}(\boldsymbol{V} + \boldsymbol{V}^{\top})$，根据$\boldsymbol{J}^{\dagger}\boldsymbol{J}=\boldsymbol{J}^{\top}(\boldsymbol{J}\boldsymbol{J}^{\top})^{-1}\boldsymbol{J}$，我们还要求$\boldsymbol{J}\boldsymbol{J}^{\top}$和$(\boldsymbol{J}\boldsymbol{J}^{\top})^{-1}$：
\begin{equation}\boldsymbol{J}\boldsymbol{J}^{\top}[\boldsymbol{V}] = \boldsymbol{J}[\boldsymbol{X}(\boldsymbol{V} + \boldsymbol{V}^{\top})] = (\boldsymbol{V} + \boldsymbol{V}^{\top})\boldsymbol{X}^{\top}\boldsymbol{X} + \boldsymbol{X}^{\top}\boldsymbol{X}(\boldsymbol{V} + \boldsymbol{V}^{\top}) = 2(\boldsymbol{V} + \boldsymbol{V}^{\top})\end{equation}
如果$\boldsymbol{V}$是对称矩阵，那么$\boldsymbol{J}\boldsymbol{J}^{\top}[\boldsymbol{V}]=4\boldsymbol{V}$，所以$(\boldsymbol{J}\boldsymbol{J}^{\top})^{-1}[\boldsymbol{V}]=\boldsymbol{V}/4$，而$\boldsymbol{J}[\boldsymbol{V}]$刚好是对称的，所以我们只需要这个特殊情况。综合起来有
\begin{equation}\boldsymbol{J}^{\dagger}\boldsymbol{J}[\boldsymbol{V}] = \boldsymbol{J}^{\top}(\boldsymbol{J}\boldsymbol{J}^{\top})^{-1}\boldsymbol{J}[\boldsymbol{V}] = \boldsymbol{J}^{\top}[(\boldsymbol{V}^{\top}\boldsymbol{X} + \boldsymbol{X}^{\top}\boldsymbol{V})/4] = \boldsymbol{X}(\boldsymbol{V}^{\top}\boldsymbol{X} + \boldsymbol{X}^{\top}\boldsymbol{V})/2\end{equation}
记$\boldsymbol{G} = \nabla_{\boldsymbol{X}}f(\boldsymbol{X})$，那么它在Stiefel流形上的投影梯度为
\begin{equation}\boldsymbol{G} - \boldsymbol{J}^{\dagger}\boldsymbol{J}[\boldsymbol{G}] =  \boldsymbol{G} - \boldsymbol{X}[\boldsymbol{X}^{\top}\boldsymbol{G}]_{\text{sym}}\end{equation}
其中$[\boldsymbol{Z}]_{\text{sym}} = (\boldsymbol{Z} + \boldsymbol{Z}^{\top})/2$为对称化算子。令它等于零，然后两边左乘$\boldsymbol{X}^{\top}$得$\boldsymbol{X}^{\top}\boldsymbol{G} = [\boldsymbol{X}^{\top}\boldsymbol{G}]_{\text{sym}}$，这也就是说，驻点条件为$\boldsymbol{X}^{\top}\boldsymbol{G}$本身就是对称矩阵！

## 二阶条件

那等式约束优化之下，有没有类似$\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x})$正/负定的最小/大值判断方法呢？自然也是有的，而且有一个很巧妙的推导。跟拉格朗日乘子法一样，我们定义
\begin{equation}F(\boldsymbol{x}, \boldsymbol{c}) = f(\boldsymbol{x}) - \boldsymbol{c}^{\top}\boldsymbol{g}(\boldsymbol{x})\end{equation}
对于任意驻点$\boldsymbol{x}$，自动有$\boldsymbol{g}(\boldsymbol{x})=\boldsymbol{0}$，所以对于任意$\boldsymbol{c}\in\mathbb{R}^k$都有$F(\boldsymbol{x}, \boldsymbol{c})\equiv f(\boldsymbol{x})$，于是展开到二阶有
\begin{equation}f(\boldsymbol{x} + \delta \boldsymbol{x}) = F(\boldsymbol{x} + \delta\boldsymbol{x}, \boldsymbol{c}) = \underbrace{F(\boldsymbol{x}, \boldsymbol{c})}_{\equiv f(\boldsymbol{x})} + \delta \boldsymbol{x}^{\top}(\nabla_{\boldsymbol{x}}f(\boldsymbol{x}) - \boldsymbol{J}^{\top} \boldsymbol{c}) + \frac{1}{2}\delta \boldsymbol{x}^{\top}\boldsymbol{H}(\boldsymbol{c})\delta \boldsymbol{x}\end{equation}
这里$\boldsymbol{H}[\boldsymbol{c}] \triangleq \nabla_{\boldsymbol{x}}^2 (f(\boldsymbol{x}) - \boldsymbol{c}^{\top}\boldsymbol{g}(\boldsymbol{x}))$。由于$\boldsymbol{c}$的任意性，我们可以直接取$\boldsymbol{c}=(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})$，使得一阶项为零，于是
\begin{equation}f(\boldsymbol{x} + \delta \boldsymbol{x}) = f(\boldsymbol{x}) + \frac{1}{2}\delta \boldsymbol{x}^{\top}\boldsymbol{H}[(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})]\delta \boldsymbol{x}\end{equation}
现在只剩二阶项，所以$\delta \boldsymbol{x}$的近似精度只需一阶，即$\delta\boldsymbol{x} = (\boldsymbol{I} - \boldsymbol{J}^{\dagger}\boldsymbol{J})\delta\boldsymbol{z}$足矣，代入得
\begin{equation}f(\boldsymbol{x} + \delta \boldsymbol{x}) = f(\boldsymbol{x}) + \frac{1}{2}\delta \boldsymbol{z}^{\top}(\boldsymbol{I} - \boldsymbol{J}^{\dagger}\boldsymbol{J})\boldsymbol{H}[(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})](\boldsymbol{I} - \boldsymbol{J}^{\dagger}\boldsymbol{J})\delta\boldsymbol{z}\end{equation}
由$\delta \boldsymbol{z}$的任意性知，$f(\boldsymbol{x})$为最小值的必要条件是$(\boldsymbol{I} - \boldsymbol{J}^{\dagger}\boldsymbol{J})\boldsymbol{H}[(\boldsymbol{J}^{\top})^{\dagger}\nabla_{\boldsymbol{x}} f(\boldsymbol{x})](\boldsymbol{I} - \boldsymbol{J}^{\dagger}\boldsymbol{J})$半正定。同理，最大值则为该矩阵半负定。

当然，不管是无约束还是有约束优化，通过Hessian矩阵的正负定来判断最大/最小，通常只具有理论价值，实践中通常都是筛出所有符合驻点条件的位置，然后逐一具体情况具体分析。

## 文章小结

本文将无约束极值的微扰分析思想用于等式约束优化中，推导了切空间上的投影梯度，而极值的驻点条件便是让相应的投影梯度为零，最后我们还证明了它与拉格朗日乘子法在结论上的等价性。
