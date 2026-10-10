# 如何让GDN2更稳定一些？

> 作者：苏剑林 · 科学空间 · 2026-10-09
>
> 原文：<https://spaces.ac.cn/archives/11945>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们知道，DeltaNet的核心是$\boldsymbol{I}-\eta\boldsymbol{k}\boldsymbol{k}^{\top}$形式的更新矩阵（参考[《线性注意力简史：从模仿、创新到反哺》](<https://spaces.ac.cn/archives/11033>)），在$\eta \leq 2$并对$\boldsymbol{k}$施加L2 Normalize时，它的连乘积具备良好的数值稳定性（参考[《为什么DeltaNet要加L2 Normalize？》](<https://spaces.ac.cn/archives/11486>)和[《DeltaNet的核心逆矩阵的元素总是在&#91;\-1, 1&#93;内》](<https://spaces.ac.cn/archives/11563>)），这是它能稳定训练的关键，后续工作[GDN](<https://papers.cool/arxiv/2412.06464>)、[KDA](<https://papers.cool/arxiv/2510.26692>)也沿用了这一设置。

后来，[GDN2](<https://papers.cool/arxiv/2605.22791>)试图将更新矩阵一般化为$\boldsymbol{I}-\eta\boldsymbol{k}\boldsymbol{j}^{\top}$，类似的选择也出现在[RWKV7](<https://papers.cool/arxiv/2503.14456>)中。然而，这种一般化可能会带来一些新的不稳定问题，本文对此做一个简单分析。

## 问题描述

GDN2的一般形式为
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1}\boldsymbol{D}_t(\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{j}_t^{\top}) + \eta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\end{equation}
其中$\boldsymbol{k}_t,\boldsymbol{j}_t\in\mathbb{R}^d$，$\eta_t\geq 0$，$\boldsymbol{D}_t$是对角矩阵，对角线元素通常限制在$[0,1]$内，代表遗忘矩阵。KDA对应于$\boldsymbol{j}_t=\boldsymbol{k}_t$这一特例，而GDN进一步对应于$\boldsymbol{D}_t=\gamma_t \boldsymbol{I}$这一特例。

当我们将递归展开时，$\boldsymbol{S}_t$会出现$\boldsymbol{D}_t(\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{j}_t^{\top})$的连乘积，所以为了保证数值稳定性，我们应当希望
\begin{equation}\Vert\boldsymbol{D}_t(\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{j}_t^{\top})\Vert_2 \leq 1\end{equation}
其中$\Vert\cdot\Vert_2$是谱范数。由于$\boldsymbol{D}_t$是对角矩阵且对角元已经在$[0,1]$内，所以它满足$\Vert\boldsymbol{D}_t\Vert_2\leq 1$，因此上式的一个充分条件是
\begin{equation}\Vert\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{j}_t^{\top}\Vert_2 \leq 1\end{equation}
很不幸的是，接下来我们将会看到，当$\boldsymbol{k}_t,\boldsymbol{j}_t$不共线时（即GDN2希望拓展的区间），上式是不可能成立的。

## 谱之范数

简单起见，下面的讨论省略下标$t$，并且假设$\boldsymbol{k},\boldsymbol{j}$都已经归一化（否则总可以将$\Vert\boldsymbol{k}\Vert,\Vert\boldsymbol{j}\Vert$吸收到$\eta$中），我们有
\begin{equation}\Vert\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}\Vert_2 = \frac{\eta + \sqrt{\eta^2 - 4\eta\,\boldsymbol{k}^{\top}\boldsymbol{j} + 4}}{2}\label{eq:spec}\end{equation}
这个结论大概是这篇文章最繁琐的地方，所以我们将它的证明放到最后。由于$\boldsymbol{k},\boldsymbol{j}$已经归一化，所以有$\boldsymbol{k}^{\top}\boldsymbol{j}\in[-1,1]$，而当$\boldsymbol{k}^{\top}\boldsymbol{j} < 1$且$\eta > 0$时，我们有
\begin{equation}\frac{\eta + \sqrt{\eta^2 - 4\eta\,\boldsymbol{k}^{\top}\boldsymbol{j} + 4}}{2} > \frac{\eta + \sqrt{\eta^2 - 4\eta + 4}}{2} = \frac{\eta + |\eta - 2|}{2} = \max(\eta, 2) - 1 \geq 1\end{equation}
也就是说只要$\boldsymbol{k}\neq\boldsymbol{j}$且$\eta > 0$，那不管怎么调，$\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}$的谱范数都严格大于1！更准确地有
\begin{equation}\Vert\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}\Vert_2 \leq 1 \quad\Longleftrightarrow\quad \boldsymbol{k}=\boldsymbol{j}\ \text{且}\ \eta\leq 2\quad(\text{或}\ \eta=0)\end{equation}
但如果$\boldsymbol{k}=\boldsymbol{j}$，那么就退化为KDA了，失去了GDN2的推广意义。而在$\boldsymbol{k}\neq\boldsymbol{j}$时，$\Vert\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}\Vert_2$一定大于1，这意味着GDN2的稳定性没有保障。该如何破局呢？

## 保证策略

庆幸的是，我们还有遗忘门矩阵$\boldsymbol{D}$，由于它被限制在$[0,1]$内，它们乘积$\boldsymbol{D}(\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top})$的谱范数大概率会小于1，于是实际训练并不一定失败。然而，这仅仅是概率上的，并非理论上的保证。

最理想的办法是直接求出$\boldsymbol{D}(\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top})$的谱范数，然后对它做谱裁剪之类的操作，然而对于一般的$\boldsymbol{D}$，该矩阵的谱范数没有通用的闭式解。

好在$\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}$的谱范数有解析解，我们可以退而求其次，利用它的谱范数，对$\boldsymbol{D}$做裁剪操作：
\begin{equation}\boldsymbol{D}(\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top})\quad\to\quad \min(\boldsymbol{D},1/\rho)(\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top})\end{equation}
其中$\rho = \Vert\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}\Vert_2$。这样一来，我们只会修改$\boldsymbol{D}$中大于$1/\rho$的对角元，由于实践中$\eta$通常不大，所以平均而言$1/\rho$不会明显小于1，那么改动面不会很广，即以尽可能小和简洁的修改，保证了GDN2的稳定性。

值得指出的是，$\rho$和$\min(\boldsymbol{D},1/\rho)$都不过是Token\-wise的运算，可以独立在RNN的算子之外，并且$\rho$也有闭式解，那么经过融合后，基本不会影响效率，因此算得上是一个心智负担极小的改动了。

## 证明过程

最后补上式$\eqref{eq:spec}$的证明。不失一般性，假设$\boldsymbol{k},\boldsymbol{j}$不共线，将它们正交化为$\boldsymbol{u}_1=\boldsymbol{j}$、$\boldsymbol{u}_2=\frac{\boldsymbol{k} - c\boldsymbol{j}}{\sqrt{1-c^2}}$，其中$c = \boldsymbol{k}^{\top}\boldsymbol{j}$。记$\boldsymbol{U}_1 = [\boldsymbol{u}_1, \boldsymbol{u}_2]$，将它扩充成完整正交基$\boldsymbol{U} = [\boldsymbol{U}_1, \boldsymbol{U}_2]$，记$\boldsymbol{A}=\boldsymbol{I} - \eta \boldsymbol{k}\boldsymbol{j}^{\top}$。

对于与$\boldsymbol{k},\boldsymbol{j}$都正交的向量$\boldsymbol{x}$，有$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{x}$和$\boldsymbol{x}^{\top}\boldsymbol{A}=\boldsymbol{x}^{\top}$，那么$\boldsymbol{A}\boldsymbol{U}_2=\boldsymbol{U}_2$和$\boldsymbol{U}_2^{\top}\boldsymbol{A}=\boldsymbol{U}_2^{\top}$，于是
\begin{equation}\boldsymbol{U}^{\top}\boldsymbol{A}\boldsymbol{U} = \begin{pmatrix}\boldsymbol{U}_1^{\top} \\ \boldsymbol{U}_2^{\top}\end{pmatrix}\boldsymbol{A}\begin{pmatrix}\boldsymbol{U}_1 & \boldsymbol{U}_2\end{pmatrix} = \begin{pmatrix}\boldsymbol{U}_1^{\top}\boldsymbol{A}\boldsymbol{U}_1 & \boldsymbol{U}_1^{\top}\boldsymbol{A}\boldsymbol{U}_2 \\ \boldsymbol{U}_2^{\top}\boldsymbol{A}\boldsymbol{U}_1 & \boldsymbol{U}_2^{\top}\boldsymbol{A}\boldsymbol{U}_2\end{pmatrix} = \begin{pmatrix}\boldsymbol{U}_1^{\top}\boldsymbol{A}\boldsymbol{U}_1 & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{I}_{d-2}\end{pmatrix}\end{equation}
由于正交变换不改变奇异值，所以$\boldsymbol{A}$的非平凡奇异值只能来自矩阵$\boldsymbol{U}_1^{\top}\boldsymbol{A}\boldsymbol{U}_1$，直接算出来得
\begin{equation}\boldsymbol{B}\triangleq\boldsymbol{U}_1^{\top}\boldsymbol{A}\boldsymbol{U}_1 = \begin{pmatrix} 1-\eta c & 0 \\ -\eta \sqrt{1-c^2} & 1 \end{pmatrix}\end{equation}
$2\times 2$矩阵的奇异值，有比较巧妙的直接计算方法。设两个奇异值为$\sigma_1\geq \sigma_2$，利用奇异值与F范数和行列式的关系可得
\begin{gather}\underbrace{\sigma_1^2 + \sigma_2^2 = \Vert\boldsymbol{B}\Vert_F^2 = \eta^2 - 2\eta c + 2,\qquad \sigma_1 \sigma_2 = |\det(\boldsymbol{B})| = |1 - \eta c|} \\
\Downarrow \notag \\
\sigma_1 \pm \sigma_2 = \sqrt{\sigma_1^2 + \sigma_2^2 \pm 2\sigma_1\sigma_2} = \sqrt{\eta^2 - 2\eta c + 2 \pm 2|1 - \eta c|}\end{gather}
我们主要关心$\sigma_1$，它等于$\pm$两项的平均，此时可以去掉绝对值（反正正负都要算），最后化简得
\begin{equation}\sigma_1 = \frac{\eta + \sqrt{\eta^2 - 4\eta c + 4}}{2}\end{equation}
前面已经证明了它不小于1，所以$\sigma_1$就是谱范数。证毕。

## 文章小结

本文分析了GDN2将更新矩阵从$\boldsymbol{I}-\eta\boldsymbol{k}\boldsymbol{k}^{\top}$推广到$\boldsymbol{I}-\eta\boldsymbol{k}\boldsymbol{j}^{\top}$后可能出现的稳定性问题，并讨论了相应的应对策略。
