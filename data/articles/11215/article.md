# 流形上的最速下降：2\. Muon \+ 正交

> 作者：苏剑林 · 科学空间 · 2025-08-06
>
> 原文：<https://spaces.ac.cn/archives/11215>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

本文继续我们的约束优化系列。在上文[《流形上的最速下降：1\. SGD \+ 超球面》](<https://spaces.ac.cn/archives/11196>)中，我们重温了优化器的“最小作用量”原理，提出不同优化器的核心差异在于给更新量施加的不同约束，如果这个约束是欧几里得范数，那么对应的最速下降便是SGD。进一步地，我们还讨论了同时给参数增加模长约束后的结果，这构成了超球面流形上的最速下降。

不过，上文只能算是“热身”，因为它处理的是相对简单的向量参数优化。本文正式进入更具挑战性的部分——优化参数从向量变成矩阵，并且增量约束改为谱范数，由此衍生出Muon优化器；接着，我们再给参数添加正交约束，这将得到正交流形下的Muon优化器。

## 命题描述

设待优化参数具有矩阵形式$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，不失一般性，设$n\geq m$。根据上一篇文章的“最小作用量”原理，我们得出最速下降的增量$\Delta\boldsymbol{W}$应该满足
\begin{equation}\min_{\Delta \boldsymbol{W}} \mathcal{L}(\boldsymbol{W} +\Delta\boldsymbol{W}) \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{W})\leq \eta\end{equation}
如果$\rho$取$F$范数（Frobenius Norm），那么将得到跟上一节一样的结果，因为$F$范数就是将矩阵当向量来算向量的L2范数，所以结果也是将矩阵当向量处理的SGD。为了得到更揭示和贴合矩阵本质的结果，这里我们选择的范数是谱范数（Spectral Norm），又称“$2$范数”，记号也是$\Vert\cdot\Vert_2$。

至于为什么要选谱范数，大家可以看[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)、[《Muon续集：为什么我们选择尝试Muon？》](<https://spaces.ac.cn/archives/10739>)、[《高阶MuP：更简明但更高明的谱条件缩放》](<https://spaces.ac.cn/archives/10795>)，这里不重复介绍。简单来说，谱范数是揭示线性层变化的最紧凑的范数，所以它更适合用作矩阵的“稳”的度量。

沿着之前的步骤，对$\mathcal{L}(\boldsymbol{W} +\Delta\boldsymbol{W})$做一阶近似得$\mathcal{L}(\boldsymbol{W}) + \langle \boldsymbol{G}, \Delta\boldsymbol{W}\rangle_F$，其中$\boldsymbol{G}=\nabla_{\boldsymbol{W}}\mathcal{L}(\boldsymbol{W})$，这里$\langle\cdot,\cdot\rangle_F$就是将两个矩阵展平为向量后算内积，它又等于$\newcommand{tr}{\mathop{\text{tr}}}\tr(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})$。然后再设$\Delta\boldsymbol{W} = -\eta \boldsymbol{\Phi}$，那么原命题就可以简化成
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1\label{eq:muon-obj}\end{equation}
到目前为止的这些变换步骤都是通用的，如果忘记了细节，请大家自行翻翻上一篇文章。

## 基本结果

目标$\eqref{eq:muon-obj}$的求解过程，我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)的“矩阵范数”一节已经给出了，不过为了介绍的完整性，这里还是复述一遍。设$\boldsymbol{G}$的SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \sum\limits_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$，$r$是$\boldsymbol{G}$的秩，我们有
\begin{equation}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})=\tr\left(\sum_{i=1}^r \sigma_i \boldsymbol{v}_i \boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\right) = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i\end{equation}
根据定义，当$\Vert\boldsymbol{\Phi}\Vert_2=1$时$\Vert\boldsymbol{\Phi}\boldsymbol{v}_i\Vert_2\leq \Vert\boldsymbol{v}_i\Vert_2=1$，于是$\boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i\leq 1$，因此
\begin{equation}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})\leq \sum_{i=1}^r \sigma_i = \Vert \boldsymbol{G}\Vert_*\end{equation}
这里的$\Vert\cdot\Vert_*$称为矩阵的[核范数（Nuclear Norm）](<https://en.wikipedia.org/wiki/Schatten_norm>)，等号在所有$\boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i$都等于1时取到，此时
\begin{equation}\newcommand{msign}{\mathop{\text{msign}}}\boldsymbol{\Phi} = \sum_{i=1}^r \boldsymbol{u}_i \boldsymbol{v}_i^{\top} = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top} = \msign(\boldsymbol{G})\end{equation}
注意，如果$r < m$，那么把$\boldsymbol{u}_{r+1} \boldsymbol{v}_{r+1}^{\top},\boldsymbol{u}_{r+2} \boldsymbol{v}_{r+2}^{\top},\cdots$叠加上去，同样能使得等号成立，也就是说解并不唯一，但此时大于$r$的项是不能唯一确定的，所以上式可以说是一个确定性的、最Minimal的解。此外，有兴趣的读者还可以尝试用“牛刀”——[冯·诺伊曼迹恒等式](<https://en.wikipedia.org/wiki/Trace_inequality#Von_Neumann's_trace_inequality_and_related_results>)——来求[Schatten\-$p$范数](<https://en.wikipedia.org/wiki/Schatten_norm>)下的通解，谱范数对应于$p\to\infty$的特例。

## 正交流形

至此，我们证明了，对于矩阵参数来说，在谱范数约束下的下降最快的方向，也不是梯度的反方向$-\boldsymbol{G}$，而是要多加一个$\msign$算子，即$-\msign(\boldsymbol{G})$，这正是训练[Kimi K2](<https://spaces.ac.cn/archives/11126>)所用的Muon优化器，它是当前极具竞争力的优化器之一，这反过来表明谱范数作为矩阵的稳定性约束是非常恰当的。

当然，到目前为止的结果都还只是旧的，现在我们开始折腾点新东西——给参数$\boldsymbol{W}$加上正交约束$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$（来源：[《Orthogonal manifold》](<https://docs.modula.systems/algorithms/manifold/orthogonal/>)），这又分为两种情况：一是$n=m$，这时候$\boldsymbol{W}$是正儿八经的正交矩阵，满足$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{W}\boldsymbol{W}^{\top}=\boldsymbol{I}$；二是$n > m$，这时候无法满足$\boldsymbol{W}\boldsymbol{W}^{\top}=\boldsymbol{I}$，通常称为半正交矩阵，对应的空间称为Stiefel流形。

具体来说，现在我们要解决的问题是：
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I},\,\,(\boldsymbol{W} - \eta \boldsymbol{\Phi})^{\top}(\boldsymbol{W} - \eta \boldsymbol{\Phi})=\boldsymbol{I}\end{equation}
依旧贯彻“一阶近似够用”的宗旨，最后一个条件可以简化成$\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}$，也就是$\boldsymbol{W}^{\top}\boldsymbol{\Phi}$是反对称矩阵：
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I},\,\,\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}\label{eq:muon-obj-orth}\end{equation}

什么时候会用到正交约束呢？事实上场景并不少，比如分类问题，如果已知各个类别之间没什么相关性，那么就可以考虑给类别矩阵条件正交约束，只不过很多时候我们都是通过给模型加上正则项$\Vert\boldsymbol{W}^{\top}\boldsymbol{W}-\boldsymbol{I}\Vert_F^2$来实现近似正交。再比如LoRA场景下，$\boldsymbol{A}\boldsymbol{B}$形式的参数化其实是有冗余的，可以通过正交约束降低冗余（[参考](<https://papers.cool/arxiv/2508.17901>)），等等。

## 求解过程

为了求解目标$\eqref{eq:muon-obj-orth}$，跟上一篇文章类似，我们引入待定系数矩阵$\boldsymbol{\Lambda}\in\mathbb{R}^{m\times m}$，得到
\begin{equation}\begin{aligned}
\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) =&\, \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) + \tr(\boldsymbol{\Lambda}^{\top}(\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W})) \\
=&\, \tr((\boldsymbol{G} + \boldsymbol{W}(\boldsymbol{\Lambda} + \boldsymbol{\Lambda}^{\top}))^{\top}\boldsymbol{\Phi}) \\
\leq &\,\Vert\boldsymbol{G} + \boldsymbol{W}(\boldsymbol{\Lambda} + \boldsymbol{\Lambda}^{\top})\Vert_*
\end{aligned}\end{equation}
第二个等号用到的迹的恒等式$\begin{equation}\tr(\boldsymbol{A}\boldsymbol{B}) = \tr(\boldsymbol{B}\boldsymbol{A}) = \tr(\boldsymbol{A}^{\top}\boldsymbol{B}^{\top}) = \tr(\boldsymbol{B}^{\top}\boldsymbol{A}^{\top})\end{equation}$。根据上一节Muon的结果，取等号的条件是
\begin{equation}\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \boldsymbol{W}(\boldsymbol{\Lambda} + \boldsymbol{\Lambda}^{\top}))\end{equation}
剩下的问题是求实对称矩阵$\boldsymbol{X} = \boldsymbol{\Lambda} + \boldsymbol{\Lambda}^{\top}$，使得$\boldsymbol{W}^{\top}\boldsymbol{\Phi}$是反对称矩阵。这个对于$n=m$来说很好解，因为此时$\boldsymbol{W}^{\top}$可以吸收到$\msign$里边：
\begin{equation}\boldsymbol{W}^{\top}\boldsymbol{\Phi} = \boldsymbol{W}^{\top}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}) = \msign(\boldsymbol{W}^{\top}(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})) = \msign(\boldsymbol{W}^{\top}\boldsymbol{G} +\boldsymbol{X})\end{equation}
注意$\msign$还有一个特性，那就是保持反对称性，即如果方阵$\boldsymbol{M}$是反对称矩阵，那么$\msign(\boldsymbol{M})$也是（请证明它）。于是为了使$\boldsymbol{W}^{\top}\boldsymbol{\Phi}$反对称，只需使$\boldsymbol{W}^{\top}\boldsymbol{G} +\boldsymbol{X}$反对称，注意$\boldsymbol{X}$是对称的，所以这相当于将$\boldsymbol{W}^{\top}\boldsymbol{G}$分解为对称矩阵与反对称矩阵之和，这是有现成答案的：
\begin{equation}\boldsymbol{W}^{\top}\boldsymbol{G} = \underbrace{\frac{1}{2}(\boldsymbol{W}^{\top}\boldsymbol{G} + \boldsymbol{G}^{\top}\boldsymbol{W})}_{[\boldsymbol{W}^{\top}\boldsymbol{G}]
_{\text{sym}}} + \underbrace{\frac{1}{2}(\boldsymbol{W}^{\top}\boldsymbol{G} - \boldsymbol{G}^{\top}\boldsymbol{W})}_{[\boldsymbol{W}^{\top}\boldsymbol{G}]
_{\text{skew}}} \end{equation}
其中$[\boldsymbol{M}]_{\text{sym}} = (\boldsymbol{M}+\boldsymbol{M}^{\top})/2, [\boldsymbol{M}]_{\text{skew}} = (\boldsymbol{M}-\boldsymbol{M}^{\top})/2$。基于上述恒等式我们可以直接得出$\boldsymbol{X} = -[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}$。至于$n > m$时的求解会比较复杂，我们留到下一篇文章再详细讨论，本文先尽量完全解决$n=m$的情形。

## 回缩操作

综上所述，在$n=m$时，我们求得的最终结果是
\begin{equation}\begin{aligned}
\boldsymbol{\Phi} =&\, \msign(\boldsymbol{G} - \boldsymbol{W}[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}) \\
=&\, \boldsymbol{W}\boldsymbol{W}^{\top}\msign(\boldsymbol{G} - \boldsymbol{W}[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}) \\
=&\, \boldsymbol{W}\msign(\boldsymbol{W}^{\top}\boldsymbol{G} - [\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}) \\
=&\, \boldsymbol{W}\msign([\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}) \\
\end{aligned}\end{equation}
因此新的变量为
\begin{equation}\boldsymbol{W} - \eta \boldsymbol{\Phi} = \boldsymbol{W}(\boldsymbol{I} - \eta\,\underbrace{\msign([\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}})}_{\text{记为}\boldsymbol{O}})\label{eq:updated-W}\end{equation}
它并不是一个正交矩阵，但能准确到$\mathcal{O}(\eta^2)$，这跟我们的“一阶近似够用”原则相符。为了观察这一点，只需要验证
\begin{equation}\begin{aligned}
(\boldsymbol{I} - \eta\boldsymbol{O})^{\top}\boldsymbol{W}^{\top}\boldsymbol{W}(\boldsymbol{I} - \eta\boldsymbol{O}) =&\,(\boldsymbol{I} - \eta\boldsymbol{O})^{\top}(\boldsymbol{I} - \eta\boldsymbol{O}) \\
=&\,\boldsymbol{I} - \eta(\boldsymbol{O}^{\top} + \boldsymbol{O}) + \eta^2\boldsymbol{O}^{\top}\boldsymbol{O} \\
=&\,\boldsymbol{I} + \eta^2\boldsymbol{O}^{\top}\boldsymbol{O} \\
\end{aligned}\label{eq:orth-check}\end{equation}
若$[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}$满秩，那么$\boldsymbol{O}$是正交矩阵，即$\boldsymbol{O}^{\top}\boldsymbol{O}=\boldsymbol{I}$，此时只要多除以个$\sqrt{1+\eta^2}$，就可以让$\eqref{eq:updated-W}$满足正交性。然而，当$[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}$不满秩时，没有简单变换让它满足正交性，这时通用提法是寻找离它最近的正交矩阵，而这正是$\msign$所做的（参考[这里](<https://spaces.ac.cn/archives/10592#%E7%AC%A6%E5%8F%B7%E5%87%BD%E6%95%B0>)）！因此，完整的更新规则是
\begin{equation}\boldsymbol{W} \quad \leftarrow\quad \msign(\boldsymbol{W} - \eta \boldsymbol{\Phi}) = \msign(\boldsymbol{W}(\boldsymbol{I} - \eta\boldsymbol{O})) = \boldsymbol{W}\msign(\boldsymbol{I} - \eta\boldsymbol{O})\end{equation}
但这样要算两次$\msign$，我们尽量简化一下。根据定义和式$\eqref{eq:orth-check}$得
\begin{equation}\msign(\boldsymbol{I} - \eta\boldsymbol{O}) = (\boldsymbol{I} - \eta\boldsymbol{O})(\boldsymbol{I} + \eta^2\boldsymbol{O}^{\top}\boldsymbol{O})^{-1/2}\end{equation}
注意不管是否满秩都有$(\boldsymbol{O}^{\top}\boldsymbol{O})^2 = \boldsymbol{O}^{\top}\boldsymbol{O}$，设$(1+\eta^2 x)^{-1/2}=1 + a_1 x + a_2 x^2 + a_2 x^3 + \cdots $，那么
\begin{equation}\begin{aligned}
(\boldsymbol{I} + \eta^2\boldsymbol{O}^{\top}\boldsymbol{O})^{-1/2} =&\, \boldsymbol{I} + a_1 (\boldsymbol{O}^{\top}\boldsymbol{O}) + a_2 (\boldsymbol{O}^{\top}\boldsymbol{O})^2 + a_3 (\boldsymbol{O}^{\top}\boldsymbol{O})^3 + \cdots \\
=&\, \boldsymbol{I} + a_1 (\boldsymbol{O}^{\top}\boldsymbol{O}) + a_2 (\boldsymbol{O}^{\top}\boldsymbol{O}) + a_3 (\boldsymbol{O}^{\top}\boldsymbol{O}) + \cdots \\
=&\, \boldsymbol{I} - \boldsymbol{O}^{\top}\boldsymbol{O} + \underbrace{(1 + a_1 + a_2  + a_3 + \cdots)}_{(1+\eta^2 x)^{-1/2}\text{代入}x=1}\boldsymbol{O}^{\top}\boldsymbol{O} \\
\end{aligned}\end{equation}
这就去掉了一次$\msign$运算，化简后的完整结果是
\begin{equation}\boldsymbol{W} \quad \leftarrow\quad \boldsymbol{W}(\boldsymbol{I} - \eta\boldsymbol{O})\left(\boldsymbol{I} - \boldsymbol{O}^{\top}\boldsymbol{O} + \frac{\boldsymbol{O}^{\top}\boldsymbol{O}}{\sqrt{1+\eta^2}}\right)\end{equation}

## 文章小结

在这篇文章中，我们重温了给矩阵参数的更新量加上谱范数约束就得到Muon优化器的结论，然后探讨了给参数加上正交约束后的Muon优化器形式。如果你希望参数在更新时始终保持为正交矩阵，那么本文也许会有一定的参考价值。
