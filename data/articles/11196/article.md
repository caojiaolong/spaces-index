# 流形上的最速下降：1\.  SGD \+ 超球面

> 作者：苏剑林 · 科学空间 · 2025-08-01
>
> 原文：<https://spaces.ac.cn/archives/11196>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

类似“梯度的反方向是下降最快的方向”的描述，经常用于介绍梯度下降（SGD）的原理。然而，这句话是有条件的，比如“方向”在数学上是单位向量，它依赖于“范数（模长）”的定义，不同范数的结论也不同，[Muon](<https://spaces.ac.cn/archives/10592>)实际上就是给矩阵参数换了个谱范数，从而得到了新的下降方向。又比如，当我们从无约束优化转移到约束优化时，下降最快的方向也未必是梯度的反方向。

为此，在这篇文章中，我们将新开一个系列，以“约束”为主线，重新审视“最速下降”这一命题，探查不同条件下的“下降最快的方向”指向何方。

## 优化原理

作为第一篇文章，我们先从SGD出发，理解“梯度的反方向是下降最快的方向”这句话背后的数学意义，然后应用于超球面上的优化。不过在此之前，笔者还想带大家重温一下[《Muon续集：为什么我们选择尝试Muon？》](<https://spaces.ac.cn/archives/10739>)所提的关于优化器的“<strong>最小作用量原理（Least Action Principle）</strong>”。

这个原理尝试回答“什么才是好的优化器”。首先，我们肯定是希望模型收敛速度越快越好，但由于神经网络本身的复杂性，如果步子迈得太大，那么反而容易训崩。所以，一个好的优化器应该是又<strong>稳</strong>又<strong>快</strong>，最好是不用大改模型，但却可以明显降低损失，写成数学形式是
\begin{equation}\min_{\Delta \boldsymbol{w}} \mathcal{L}(\boldsymbol{w} +\Delta\boldsymbol{w}) \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{w})\leq \eta\end{equation}
其实$\mathcal{L}$是损失函数，$\boldsymbol{w}\in\mathbb{R}^n$是参数向量，$\Delta \boldsymbol{w}$是更新量，$\rho(\Delta\boldsymbol{w})$是更新量$\Delta\boldsymbol{w}$大小的某种度量。上述目标很直观，就是在“步子”不超过$\eta$（<strong>稳</strong>）的前提下，寻找让损失函数下降最多（<strong>快</strong>）的更新量，这便是“最小作用量原理”的数学含义，也是“最速下降”的数学含义。

## 目标转化

假设$\eta$足够小，那么$\Delta\boldsymbol{w}$也足够小，以至于一阶近似足够准确，那么我们就可以将$\mathcal{L}(\boldsymbol{w} +\Delta\boldsymbol{w})$替换为$\mathcal{L}(\boldsymbol{w}) + \langle\boldsymbol{g},\Delta\boldsymbol{w}\rangle$，其中$\boldsymbol{g} = \nabla_{\boldsymbol{w}}\mathcal{L}(\boldsymbol{w})$，得到等效目标
\begin{equation}\min_{\Delta \boldsymbol{w}} \langle\boldsymbol{g},\Delta\boldsymbol{w}\rangle \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{w})\leq \eta\end{equation}
这就将优化目标简化成$\Delta \boldsymbol{w}$的线性函数，降低了求解难度。进一步地，我们设$\Delta \boldsymbol{w} = -\kappa \boldsymbol{\varphi}$，其中$\rho(\boldsymbol{\varphi})=1$，那么上述目标等价于
\begin{equation}\max_{\kappa,\boldsymbol{\varphi}} \kappa\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \rho(\boldsymbol{\varphi}) = 1, \,\,\kappa\in[0, \eta]\end{equation}
假设我们至少能找到一个满足条件的$\boldsymbol{\varphi}$使得$\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle\geq 0$，那么有$\max\limits_{\kappa\in[0,\eta]} \kappa\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \eta\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle$，也就是$\kappa$的优化可以事先求出来，结果是$\kappa=\eta$，最终等效于只剩下$\boldsymbol{\varphi}$的优化
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \rho(\boldsymbol{\varphi}) = 1\label{eq:core}\end{equation}
这里的$\boldsymbol{\varphi}$满足某种“模长”$\rho(\boldsymbol{\varphi})$等于1的条件，所以它代表了某种“方向向量”的定义，最大化它与梯度$\boldsymbol{g}$的内积，就意味着寻找让损失下降最快的方向（即$\boldsymbol{\varphi}$的反方向）。

## 梯度下降

从式$\eqref{eq:core}$可以看出，对于“下降最快的方向”，唯一不确定的是度量$\rho$，这是优化器里边很本质的一个先验（Inductive Bias），不同的度量将会得到不同的最速下降方向。比较简单的就是L2范数或者说欧几里得范数$\rho(\boldsymbol{\varphi})=\Vert \boldsymbol{\varphi} \Vert_2$，也就是我们通常意义下的模长，这时候我们有柯西不等式：
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g}\Vert_2 \Vert\boldsymbol{\varphi}\Vert_2 = \Vert\boldsymbol{g}\Vert_2\end{equation}
等号成立的条件是$\boldsymbol{\varphi} \propto\boldsymbol{g}$，加上模长为1的条件，我们得到$\boldsymbol{\varphi}=\boldsymbol{g}/\Vert\boldsymbol{g}\Vert_2$，这正是梯度的方向。所以说，“梯度的反方向是下降最快的方向”前提是所选取的度量是欧几里得范数。更一般地，我们考虑$p$范数
\begin{equation}\rho(\boldsymbol{\varphi}) = \Vert\boldsymbol{\varphi}\Vert_p = \sqrt[\uproot{10}p]{\sum_{i=1}^n |\varphi_i|^p}\end{equation}
柯西不等式可以推广成[Hölder不等式](<https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality>)：
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g}\Vert_q \Vert\boldsymbol{\varphi}\Vert_p = \Vert\boldsymbol{g}\Vert_q,\qquad 1/p + 1/q=1\end{equation}
等号成立的条件$\boldsymbol{\varphi}^{[p]} \propto\boldsymbol{g}^{[q]}$，所以解得
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}}\boldsymbol{\varphi} = \frac{\boldsymbol{g}^{[q/p]}}{\Vert\boldsymbol{g}^{[q/p]}\Vert_p},\qquad \boldsymbol{g}^{[\alpha]} \triangleq \big[\sign(g_1) |g_1|^{\alpha},\sign(g_2) |g_2|^{\alpha},\cdots,\sign(g_n) |g_n|^{\alpha}\big]\end{equation}
以它为方向向量的优化器叫做pbSGD，出自[《pbSGD: Powered Stochastic Gradient Descent Methods for Accelerated Non\-Convex Optimization》](<https://www.ijcai.org/proceedings/2020/451>)。它有两个特例，一是$p=q=2$是退化为SGD，二是$p\to\infty$时$q\to 1$，此时$|g_i|^{q/p}\to 1$，更新方向为梯度的符号函数，即SignSGD。

## 超球面上

前面的讨论中，我们只是对参数的增量$\Delta\boldsymbol{w}$施加了约束，接下来我们希望的是给参数$\boldsymbol{w}$也添加约束。具体来说，我们假设参数$\boldsymbol{w}$位于单位球面上，我们希望更新后的参数$\boldsymbol{w}+\Delta\boldsymbol{w}$依然位于单位球面上（参考[《Hypersphere》](<https://docs.modula.systems/algorithms/manifold/hypersphere/>)）。从目标$\eqref{eq:core}$出发，我们可以将新目标写成
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \Vert\boldsymbol{\varphi}\Vert_2 = 1,\,\,\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2 = 1,\,\,\Vert\boldsymbol{w}\Vert_2=1\end{equation}
我们依然贯彻“$\eta$足够小，一阶近似够用”的原则，得到
\begin{equation}1 = \Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2^2 = \Vert\boldsymbol{w}\Vert_2^2 - 2\eta\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle + \eta^2 \Vert\boldsymbol{\varphi}\Vert_2^2\approx 1 - 2\eta\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle\end{equation}
所以这相当于将约束转化为线性形式$\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0$。为了求解新的目标，我们引入待定系数$\lambda$，写出
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle + \lambda\langle\boldsymbol{w},\boldsymbol{\varphi}\rangle =\langle \boldsymbol{g} + \lambda\boldsymbol{w},\boldsymbol{\varphi}\rangle\leq \Vert\boldsymbol{g} + \lambda\boldsymbol{w}\Vert_2 \Vert\boldsymbol{\varphi}\Vert_2 = \Vert\boldsymbol{g} + \lambda\boldsymbol{w}\Vert_2\end{equation}
等号成立的条件是$\boldsymbol{\varphi}\propto \boldsymbol{g} + \lambda\boldsymbol{w}$，再加上$\Vert\boldsymbol{\varphi}\Vert_2=1,\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0,\Vert\boldsymbol{w}\Vert_2=1$的条件，可以解得
\begin{equation}\boldsymbol{\varphi} = \frac{\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}}{\Vert\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}\Vert_2}\end{equation}
注意现在有$\Vert\boldsymbol{w}\Vert_2=1,\Vert\boldsymbol{\varphi}\Vert_2=1$，并且$\boldsymbol{w}$和$\boldsymbol{\varphi}$是正交的，那么$\boldsymbol{w} - \eta\boldsymbol{\varphi}$的模长是并不是精确地等于1，而是$\sqrt{1 + \eta^2}=1 + \eta^2/2 + \cdots$，精确到$\mathcal{O}(\eta^2)$，这跟我们前面的假设“$\eta$的一阶项够用”吻合。如果想更新后的参数模长严格等于1，那么可以在更新规则上多加一步缩回操作：
\begin{equation}\boldsymbol{w}\quad\leftarrow\quad \frac{\boldsymbol{w} - \eta\boldsymbol{\varphi}}{\sqrt{1 + \eta^2}}\end{equation}

## 几何意义

刚才我们通过“一阶近似够用”原则，将非线性约束$\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_2 = 1$简化为线性约束$\langle \boldsymbol{w}, \boldsymbol{\varphi}\rangle=0$，后者的几何意义是“与$\boldsymbol{w}$垂直”，这还有个更专业的说法，叫做$\Vert\boldsymbol{w}\Vert_2=1$的“切空间”，而$\boldsymbol{g} - \langle \boldsymbol{g}, \boldsymbol{w}\rangle\boldsymbol{w}$这一运算，正对应于把梯度$\boldsymbol{g}$投影到切空间的投影运算。

所以很幸运，球面上的SGD有非常清晰的几何意义，如下图所示：

[![球面上的最速下降-几何意义](<https://spaces.ac.cn/usr/uploads/2025/07/1221229703.png>)](<https://spaces.ac.cn/usr/uploads/2025/07/1221229703.png>)

球面上的最速下降\-几何意义

相信很多读者都喜欢这种几何视角，它确实让人赏心悦目。但这是还是要提醒大家一下，应当优先认真理解代数求解过程，因为清晰的几何意义很多时候都只是一种奢望，属于可遇而不可求的，大多数情况下复杂的代数过程才是本质。

## 一般结果

接下来是不是有读者想要将它推广到一般的$p$范数？让我们一起来尝试下，看看会遇到什么困难。这时候问题是：
\begin{equation}\max_{\boldsymbol{\varphi}} \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle \qquad \text{s.t.}\qquad \Vert\boldsymbol{\varphi}\Vert_p = 1,\,\,\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_p = 1,\,\,\Vert\boldsymbol{w}\Vert_p=1\end{equation}
一阶近似将$\Vert\boldsymbol{w}-\eta\boldsymbol{\varphi}\Vert_p = 1$转换成$\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = 0$，然后引入待定系数$\lambda$：
\begin{equation}\langle\boldsymbol{g},\boldsymbol{\varphi}\rangle = \langle\boldsymbol{g},\boldsymbol{\varphi}\rangle + \lambda\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = \langle \boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle \leq \Vert\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]}\Vert_q \Vert\boldsymbol{\varphi}\Vert_p = \Vert\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]}\Vert_q
\end{equation}
取等号的条件是
\begin{equation}\boldsymbol{\varphi} = \frac{(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})^{[q/p]}}{\Vert(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})^{[q/p]}\Vert_p}\end{equation}
到目前为止，都没有实质困难。然而，接下来我们需要寻找$\lambda$，使得$\langle\boldsymbol{w}^{[p-1]},\boldsymbol{\varphi}\rangle = 0$，当$p \neq 2$时这是一个复杂的非线性方程，并没有很好的求解办法（当然，一旦求解出来，我们就肯定能得到最优解，这是Hölder不等式保证的）。所以，一般$p$的求解我们只能止步于此，等遇到$p\neq 2$的实例时我们再具体探寻数值求解方法。

不过除了$p=2$，我们还可以尝试求解$p\to\infty$，此时$\boldsymbol{\varphi}=\sign(\boldsymbol{g} + \lambda\boldsymbol{w}^{[p-1]})$，条件$\Vert\boldsymbol{w}\Vert_p=1$给出$|w_1|,|w_2|,\cdots,|w_n|$的最大值等于1。如果进一步假设最大值只有一个，那么$\boldsymbol{w}^{[p-1]}$是一个one hot向量，绝对值最大值的位置为$\pm 1$，其余是零，这时候就可以解出$\lambda$，结果是把最大值位置的梯度裁剪成零。

## 文章小结

这篇文章新开一个系列，主要围绕“等式约束”来讨论优化问题，试图给一些常见的约束条件寻找“下降最快的方向”。作为第一篇文章，本文讨论了“超球面”约束下的SGD变体。
