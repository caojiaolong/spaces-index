# 为什么官方版Muon比MuP版多出一个max(1, ⋅)？

> 作者：苏剑林 · 科学空间 · 2026-06-03
>
> 原文：<https://spaces.ac.cn/archives/11772>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《Muon优化器指南：快速上手与关键细节》](<https://spaces.ac.cn/archives/11416>)中，我们罗列了Muon的几个版本，它们的区别是学习率的矩阵形状相关的缩放因子不同，其中“官方版（KellerJordan版）”只比“MuP版”多出了一个$\max(1,\cdot)$的截断操作。本文就来专门讨论一下，这个截断操作是怎么来的呢？

## 几个版本

Muon的更新规则，可以统一地写成
\begin{equation}\begin{aligned}\newcommand{msign}{\mathop{\text{msign}}}
\boldsymbol{M}_t =&\, \beta \boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t (\alpha \msign(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1})
\end{aligned}\end{equation}
几个不同的版本区别在于$\alpha$，分别是：
$$\alpha = \left\{
\begin{aligned} &1 & \color{skyblue}{(\text{朴素版})} \\[5pt]
& \sqrt{\max(1, d_{out}/d_{in})} & \color{skyblue}{(\text{KellerJordan版})} \\[5pt]
& \sqrt{d_{out}/d_{in}} & \color{skyblue}{(\text{MuP版})} \\[5pt]
& 0.2\times\sqrt{\max(d_{out},d_{in})} & \color{skyblue}{(\text{Moonlight版})}
\end{aligned}\right.$$

其中矩阵$\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$，代表线性层$\boldsymbol{y}=\boldsymbol{x}\boldsymbol{W}$的训练参数，其中输入$\boldsymbol{x}\in\mathbb{R}^{d_{in}}$是行向量。

本文主要关心“KellerJordan版”和“MuP版”，前者在后者的基础上多加了个$\max(1,)$。按照我们在[《高阶MuP：更简明但更高明的谱条件缩放》](<https://spaces.ac.cn/archives/10795>)、[《MuP之上：2\. 线性层与最速下降》](<https://spaces.ac.cn/archives/11605>)等文章的分析，在MuP相关的谱条件约束下，最速下降应该就是MuP版Muon才对，多出来的$\max(1,\cdot)$该如何解释呢？

## 特征增量

简单起见，下面的讨论都省掉下标$t$。不失一般性，我们假设动量$\boldsymbol{M}$是满秩的，那么$\boldsymbol{\Phi} = \msign(\boldsymbol{M})$的奇异值全为1，那么当$d_{in} \leq d_{out}$时，$\boldsymbol{\Phi} \boldsymbol{\Phi}^{\top} = \boldsymbol{I}_{d_{in}}$，当$d_{in} > d_{out}$时，$\boldsymbol{\Phi}^{\top} \boldsymbol{\Phi} = \boldsymbol{I}_{d_{out}}$。

记$\Delta \boldsymbol{W} = \eta\alpha \boldsymbol{\Phi}$，我们要做的事情，就是寻找$\alpha$与$d_{in},d_{out}$的关系。从[《为什么我们偏爱各向同性？基于最速下降的理解》](<https://spaces.ac.cn/archives/11549>)中我们知道，参数实际上只是模型的副产物，特征层面的变化可能会更加本质。将$\Delta \boldsymbol{W}$转换到特征层面是$\Delta \boldsymbol{y} = \boldsymbol{x} \Delta\boldsymbol{W} = \eta\alpha \boldsymbol{x}\boldsymbol{\Phi}$，那么$\Vert\Delta \boldsymbol{y}\Vert_{RMS} = \eta\alpha \Vert\boldsymbol{x}\boldsymbol{\Phi}\Vert_{RMS}$。

接下来要分情况讨论。首先是当$d_{in} \leq d_{out}$时，$\boldsymbol{\Phi}$可以写成$\boldsymbol{U}[\boldsymbol{I}_{d_{in}}, \boldsymbol{0}_{d_{in}\times (d_{out}-d_{in})}]\boldsymbol{V}^{\top}$的形式，其中$\boldsymbol{U}\in\mathbb{R}^{d_{in}\times d_{in}}, \boldsymbol{V}\in\mathbb{R}^{d_{out}\times d_{out}}$都是正交矩阵，那么
\begin{equation}\begin{aligned}
\Vert\Delta \boldsymbol{y}\Vert_{RMS} =&\, \eta\alpha\big\Vert\boldsymbol{x}\boldsymbol{U}[\boldsymbol{I}_{d_{in}}, \boldsymbol{0}_{d_{in}\times (d_{out}-d_{in})}]\boldsymbol{V}^{\top}\big\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\big\Vert\boldsymbol{x}\boldsymbol{U}[\boldsymbol{I}_{d_{in}}, \boldsymbol{0}_{d_{in}\times (d_{out}-d_{in})}]\big\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\big\Vert[\boldsymbol{x}\boldsymbol{U}, \boldsymbol{0}_{d_{out}-d_{in}}]\big\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\boldsymbol{U}\Vert_{RMS} \\[4pt]
=&\, \eta\alpha\sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\Vert_{RMS} \\
\end{aligned}\end{equation}
注意全程都是等号，所以我们只需要设$\alpha = \sqrt{d_{out}/d_{in}}$，就可以让“每一个”$\Delta \boldsymbol{y}$的RMS都是$\eta\Vert\boldsymbol{x}\Vert_{RMS}$，即所有Token的相对更新幅度一致。

## 各向同性

很遗憾，对于第二种情况$d_{in} > d_{out}$，上述“完全一致”的目标并无法实现。具体来说，此时$\boldsymbol{\Phi}$的SVD要写成$\boldsymbol{U}\begin{bmatrix}\boldsymbol{I}_{d_{out}} \\ \boldsymbol{0}_{(d_{in}-d_{out})\times d_{out}}\end{bmatrix}\boldsymbol{V}^{\top}$，于是
\begin{equation}\begin{aligned}
\Vert\Delta \boldsymbol{y}\Vert_{RMS} =&\, \eta\alpha\left\Vert\boldsymbol{x}\boldsymbol{U}\begin{bmatrix}\boldsymbol{I}_{d_{out}} \\ \boldsymbol{0}_{(d_{in}-d_{out})\times d_{out}}\end{bmatrix}\boldsymbol{V}^{\top}\right\Vert_{RMS} \\[5pt]
=&\, \eta\alpha\left\Vert\boldsymbol{x}\boldsymbol{U}\begin{bmatrix}\boldsymbol{I}_{d_{out}} \\ \boldsymbol{0}_{(d_{in}-d_{out})\times d_{out}}\end{bmatrix}\right\Vert_{RMS} \\[5pt]
=&\, \eta\alpha\big\Vert(\boldsymbol{x}\boldsymbol{U})_{[:d_{out}]}\big\Vert_{RMS}
\end{aligned}\end{equation}
$\boldsymbol{x}\boldsymbol{U}$是一个$d_{in}$维向量，$d_{in} > d_{out}$，所以$(\boldsymbol{x}\boldsymbol{U})_{[:d_{out}]}$只是截取了$\boldsymbol{x}\boldsymbol{U}$的前$d_{out}$个维度去算RMS，这时候它的RMS是不确定的，最大可以去到$\sqrt{d_{in}/d_{out}}\Vert\boldsymbol{x}\Vert_{RMS}$（Worst Case），最小可以是0。

我们知道正交矩阵不改变RMS，所以$\Vert\boldsymbol{x}\boldsymbol{U}\Vert_{RMS}=\Vert\boldsymbol{x}\Vert_{RMS}$，当$\boldsymbol{x}$的分布足够各向同性时，我们可以认为这代表着$\boldsymbol{x}\boldsymbol{U}$每个分量的平均尺度都是$\Vert\boldsymbol{x}\Vert_{RMS}$，于是取前$d_{out}$个分量去RMS，平均来说也约等于$\Vert\boldsymbol{x}\Vert_{RMS}$，即$\Vert\Delta \boldsymbol{y}\Vert_{RMS}\approx \eta\alpha\Vert\boldsymbol{x}\Vert_{RMS}$，因此只需要取$\alpha = 1$，就可以实现跟上一节类似的效果。

## 各向异性

将前两节的结果综合起来，我们就得到
\begin{equation}\alpha = \sqrt{\max\left(1, \frac{d_{out}}{d_{in}}\right)}\end{equation}
这正是KellerJordan版Muon出现的$\max(1,\cdot)$。

然而，上一节的结论依赖于输入$\boldsymbol{x}$足够各向同性的假设，这在训练早期或许近似成立，但随着训练的推进，特征的分布会逐渐各向异性，集中在让$\Vert\Delta \boldsymbol{y}\Vert_{RMS}$最大化的“Worst Case”上面，此时平均近似$\Vert\Delta \boldsymbol{y}\Vert_{RMS}\approx \eta\alpha\Vert\boldsymbol{x}\Vert_{RMS}$就不够准确了，反而是最大值$\eta\alpha\sqrt{d_{in}/d_{out}}\Vert\boldsymbol{x}\Vert_{RMS}$会更加准确。

这种情况下，让$\Vert\Delta \boldsymbol{y}\Vert_{RMS}\approx \eta\Vert\boldsymbol{x}\Vert_{RMS}$的$\alpha$是$\sqrt{d_{out}/d_{in}}$，跟$d_{in} \leq d_{out}$时的结论一致，重新得到MuP版结果。也就是说，对于训练的中后期，MuP版Muon更加科学。对于这种不一致性，我们有两种策略：一是始终用MuP版Muon，这会稍微降低一点初期的收敛速度，但毕竟中后期才是训练的“重中之重”；二是将缩放因子改为
\begin{equation}\alpha = \sqrt{\max\left(\tau_t, \frac{d_{out}}{d_{in}}\right)}\end{equation}
其中$\tau_t$从1到0单调衰减，这样实现了从KellerJordan版到MuP版的渐变，代价是多了一个Schedule要调。

## 文章小结

本文主要从“特征增量”均匀性的角度解释了KellerJordan版中$\max(1,\cdot)$的来源。
