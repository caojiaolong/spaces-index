# 低秩近似之路（三）：CR

> 作者：苏剑林 · 科学空间 · 2024-10-11
>
> 原文：<https://spaces.ac.cn/archives/10427>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《低秩近似之路（二）：SVD》](<https://spaces.ac.cn/archives/10407>)中，我们证明了SVD可以给出任意矩阵的最优低秩近似。那里的最优近似是无约束的，也就是说SVD给出的结果只管误差上的最小，不在乎矩阵的具体结构，而在很多应用场景中，出于可解释性或者非线性处理等需求，我们往往希望得到具有某些特殊结构的近似分解。

因此，从这篇文章开始，我们将探究一些具有特定结构的低秩近似，而本文将聚焦于其中的CR近似（Column\-Row Approximation），它提供了加速矩阵乘法运算的一种简单方案。

## 问题背景

矩阵的最优$r$秩近似的一般提法是
\begin{equation}\mathop{\text{argmin}}_{\text{rank}(\tilde{\boldsymbol{M}})\leq r}\Vert \tilde{\boldsymbol{M}} - \boldsymbol{M}\Vert_F^2\label{eq:loss-m2}\end{equation}
其中$\boldsymbol{M},\tilde{\boldsymbol{M}}\in\mathbb{R}^{n\times m},r < \min(n,m)$。在前两篇文章中，我们已经讨论了两种情况：

> 1、如果$\tilde{\boldsymbol{M}}$不再有其他约束，那么$\tilde{\boldsymbol{M}}$的最优解就是$\boldsymbol{U}_{[:,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$，其中$\boldsymbol{M}=\boldsymbol{U}\boldsymbol{\Sigma} \boldsymbol{V}^{\top}$是$\boldsymbol{M}$的奇异值分解（SVD）；
> 
> 
> 
> 2、如果约定$\tilde{\boldsymbol{M}}=\boldsymbol{A}\boldsymbol{B}$（$\boldsymbol{A}\in\mathbb{R}^{n\times r},\boldsymbol{B}\in\mathbb{R}^{r\times m}$），且$\boldsymbol{A}$（或$\boldsymbol{B}$）已经给定，那么$\tilde{\boldsymbol{M}}$的最优解是$\boldsymbol{A} \boldsymbol{A}^{\dagger} \boldsymbol{M}$（或$\boldsymbol{M} \boldsymbol{B}^{\dagger} \boldsymbol{B}$），这里的${}^\dagger$是“[伪逆](<https://spaces.ac.cn/archives/10366>)”。

这两个结果都有很广泛的应用，但它们都没有显式地引入$\tilde{\boldsymbol{M}}$与$\boldsymbol{M}$结构上的关联，这就导致了很难直观地看到$\tilde{\boldsymbol{M}}$与$\boldsymbol{M}$的关联，换言之$\tilde{\boldsymbol{M}}$的可解释性不强。

此外，如果目标中包含非线性运算如$\phi(\boldsymbol{X}\boldsymbol{W})$，通常也不允许我们使用任意实投影矩阵来降维，而是要求使用“选择矩阵（Selective Matrix）”，比如$\phi(\boldsymbol{X}\boldsymbol{W})\boldsymbol{S} = \phi(\boldsymbol{X}\boldsymbol{W}\boldsymbol{S})$对于任意矩阵$\boldsymbol{S}$不是恒成立的，但对于选择矩阵$\boldsymbol{S}$是恒成立的。

所以，接下来我们关注选择矩阵约束下的低秩近似。具体来说，我们有$\boldsymbol{X}\in\mathbb{R}^{n\times l},\boldsymbol{Y}\in\mathbb{R}^{l\times m}$，然后选定$\boldsymbol{M}=\boldsymbol{X}\boldsymbol{Y}$，我们的任务是从$\boldsymbol{X}$中选出$r$列、从$\boldsymbol{Y}$中选出相应的$r$行来构建$\tilde{\boldsymbol{M}}$，即
\begin{equation}\mathop{\text{argmin}}_S\Vert \underbrace{\boldsymbol{X}_{[:,S]}}_{\boldsymbol{C}}\underbrace{\boldsymbol{Y}_{[S,:]}}_{\boldsymbol{R}} - \boldsymbol{X}\boldsymbol{Y}\Vert_F^2\quad\text{s.t.}\quad S\subset \{0,1,\cdots,l-1\}, |S|=r\end{equation}
这里的$S$可以理解为slice，即按照Python的切片规则来理解，我们称$\boldsymbol{X}_{[:,S]}\boldsymbol{Y}_{[S,:]}$为$\boldsymbol{X}\boldsymbol{Y}$的“CR近似”。注意这种切片结果也可以用选择矩阵来等价描述，假设$\boldsymbol{X}_{[:,S]}$的第$1,2,\cdots,r$列分别为$\boldsymbol{X}$的第$s_1,s_2,\cdots,s_r$列，那么可以定义选择矩阵$\boldsymbol{S}\in\{0,1\}^{l\times r}$：
\begin{equation}S_{i,j}=\left\{\begin{aligned}&1, &i = s_j \\ &0, &i\neq s_j\end{aligned}\right.\end{equation}
即$\boldsymbol{S}$的第$j$列的第$s_j$个元素为1，其余都为0，这样一来就有$\boldsymbol{X}_{[:,S]}=\boldsymbol{X}\boldsymbol{S}$以及$\boldsymbol{Y}_{[S,:]}=\boldsymbol{S}^{\top} \boldsymbol{Y}$。

## 初步近似

如果我们将$\boldsymbol{X},\boldsymbol{Y}$分别表示成
\begin{equation}\boldsymbol{X} = (\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_l),\quad \boldsymbol{Y}=\begin{pmatrix}\boldsymbol{y}_1^{\top} \\ \boldsymbol{y}_2^{\top} \\ \vdots \\ \boldsymbol{y}_l^{\top}\end{pmatrix}\end{equation}
其中$\boldsymbol{x}_i\in\mathbb{R}^{n\times 1},\boldsymbol{y}_i\in\mathbb{R}^{m\times 1}$都是列向量，那么$\boldsymbol{X}\boldsymbol{Y}$可以写成
\begin{equation}\boldsymbol{X}\boldsymbol{Y} = \sum_{i=1}^l \boldsymbol{x}_i\boldsymbol{y}_i^{\top}\end{equation}
而找$\boldsymbol{X}\boldsymbol{Y}$的最优CR近似则可以等价地写成
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\{0,1\}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{x}_i\boldsymbol{y}_i^{\top} - \sum_{i=1}^l\boldsymbol{x}_i\boldsymbol{y}_i^{\top}\right\Vert_F^2\quad\text{s.t.}\quad \sum_{i=1}^l \lambda_i = r\label{eq:xy-l-k}\end{equation}
我们知道，矩阵的$F$范数实际上就是将矩阵展平成向量来算模长，所以这个优化问题本质上就相当于给定$l$个向量$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l\in\mathbb{R}^d$，求
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\{0,1\}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{v}_i - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^l \lambda_i = r\label{eq:v-l-k}\end{equation}
其中$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i \boldsymbol{y}_i^{\top})$，$d=nm$。记$\gamma_i = 1 - \lambda_i$，那么可以进一步简化成
\begin{equation}\mathop{\text{argmin}}_{\gamma_1,\gamma_2,\cdots,\gamma_l\in\{0,1\}}\left\Vert\sum_{i=1}^l \gamma_i \boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^l \gamma_i = l-r\label{eq:v-l-k-0}\end{equation}
如果笔者没有理解错，这个优化问题的精确求解是NP\-Hard的，所以一般情况下只能寻求近似算法。一个可精确求解的简单例子是$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l$两两垂直，此时
\begin{equation}\left\Vert\sum_{i=1}^l \gamma_i \boldsymbol{v}_i\right\Vert^2 = \sum_{i=1}^l \gamma_i^2 \Vert\boldsymbol{v}_i\Vert^2\end{equation}
所以它的最小值就是最小的$l-r$个$\Vert\boldsymbol{v}_i\Vert^2$之和，即让模长最小的$l-r$个$\boldsymbol{v}_i$的$\gamma_i$等于1，剩下的$\gamma_i$则等于0。当两两正交的条件不严格成立时，我们依然可以将选择模长最小的$l-r$个$\boldsymbol{v}_i$作为一个近似解。回到原始的CR近似问题上，我们有$\Vert\boldsymbol{x}_i\boldsymbol{y}_i^{\top}\Vert_F = \Vert\boldsymbol{x}_i\Vert \Vert \boldsymbol{y}_i\Vert$，所以$\boldsymbol{X}\boldsymbol{Y}$的最优CR近似的一个baseline，就是保留$\boldsymbol{X}$的列向量与$\boldsymbol{Y}$对应的行向量模长乘积最大的$r$个列/行向量。

## 采样视角

有一些场景允许我们将式$\eqref{eq:xy-l-k}$放宽为
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\mathbb{R}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{x}_i\boldsymbol{y}_i^{\top} - \sum_{i=1}^l\boldsymbol{x}_i\boldsymbol{y}_i^{\top}\right\Vert_F^2\quad\text{s.t.}\quad \sum_{i=1}^l \#[\lambda_i\neq 0] = r\end{equation}
其中$\#[\lambda_i\neq 0]$表示$\lambda_i\neq 0$时输出1，否则输出0。这个宽松版本其实就是将CR近似的形式从$\boldsymbol{C}\boldsymbol{R}$拓展成$\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$，其中$\boldsymbol{\Lambda}\in\mathbb{R}^{r\times r}$是对角阵，即允许我们调整对角阵$\boldsymbol{\Lambda}\in\mathbb{R}^{r\times r}$来达到更高的精度。相应地，式$\eqref{eq:v-l-k}$变为
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_l\in\mathbb{R}}\left\Vert\sum_{i=1}^l \lambda_i \boldsymbol{v}_i - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\quad\text{s.t.}\quad \sum_{i=1}^l \#[\lambda_i\neq 0] = r\end{equation}

这样放宽之后，我们可以从采样视角来看待这个问题。首先我们引入任意$l$元分布$\boldsymbol{p}=(p_1,p_2,\cdots,p_l)$，然后我们就可以写出
\begin{equation}\sum_{i=1}^l\boldsymbol{v}_i = \sum_{i=1}^l p_i\times\frac{\boldsymbol{v}_i}{p_i} = \mathbb{E}_{i\sim \boldsymbol{p}} \left[\frac{\boldsymbol{v}_i}{p_i}\right] \end{equation}
也就是说，$\boldsymbol{v}_i/p_i$的数学期望正好是我们要逼近的目标，所以我们可以通过从$\boldsymbol{p}$分布<u><strong>独立重复采样</strong></u>来构建近似：
\begin{equation}\sum_{i=1}^l\boldsymbol{v}_i = \mathbb{E}_{i\sim \boldsymbol{p}} \left[\frac{\boldsymbol{v}_i}{p_i}\right] \approx \frac{1}{r}\sum_{j=1}^r \frac{\boldsymbol{v}_{s_j}}{p_{s_j}},\quad s_1,s_2,\cdots,s_r\sim \boldsymbol{p}\end{equation}
这意味着当$i$是$s_1,s_2,\cdots,s_r$之一时有$\lambda_i = (r p_i)^{-1}$，否则$\lambda_i=0$。可能有读者疑问为什么要独立重复采样，而不是更符合逼近需求的不放回采样呢？无他，纯粹是因为独立重复采样使得后面的分析更简单。

到目前为止，我们的理论结果跟分布$\boldsymbol{p}$的选择无关，也就是说对于任意$\boldsymbol{p}$都是成立的，这给我们提供了选择最优$\boldsymbol{p}$的可能性。那如何衡量$\boldsymbol{p}$的优劣呢？很显然我们希望每次采样估计的误差越小越好，因此可以用采样估计的误差
\begin{equation}\mathbb{E}_{i\sim \boldsymbol{p}} \left[\left\Vert\frac{\boldsymbol{v}_i}{p_i} - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\right] = \left(\sum_{i=1}^l \frac{\Vert\boldsymbol{v}_i\Vert^2}{p_i}\right) - \left\Vert\sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2 \end{equation}
来比较不同的$\boldsymbol{p}$之间的优劣。接着利用均值不等式有
\begin{equation}\sum_{i=1}^l \frac{\Vert\boldsymbol{v}_i\Vert^2}{p_i} = \left(\sum_{i=1}^l \frac{\Vert\boldsymbol{v}_i\Vert^2}{p_i} + p_i Z^2\right) - Z^2\geq \left(\sum_{i=1}^l 2\Vert\boldsymbol{v}_i\Vert Z\right) - Z^2\end{equation}
等号在$\Vert\boldsymbol{v}_i\Vert^2 / p_i = p_i Z^2$时取到，由此可得最优的$\boldsymbol{p}$为
\begin{equation}p_i^* = \frac{\Vert\boldsymbol{v}_i\Vert}{Z},\quad Z = \sum\limits_{i=1}^l \Vert\boldsymbol{v}_i\Vert\end{equation}
对应的误差为
\begin{equation}\mathbb{E}_{i\sim \boldsymbol{p}} \left[\left\Vert\frac{\boldsymbol{v}_i}{p_i} - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\right] = \left(\sum_{i=1}^l \Vert\boldsymbol{v}_i\Vert\right)^2 - \left\Vert\sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2 \end{equation}
最优的$p_i$正好正比于$\Vert\boldsymbol{v}_i\Vert$，所以概率最大的$r$个$\boldsymbol{v}_i$也正是模长最大的$r$个$\boldsymbol{v}_i$，这就跟上一节的近似联系起来了。该结果出自2006年的论文[《Fast Monte Carlo Algorithms for Matrices I: Approximating Matrix Multiplication》](<https://www.stat.berkeley.edu/~mmahoney/pubs/matrix1_SICOMP.pdf>)，初衷是加速矩阵乘法，它表明只要按照$p_i\propto \Vert \boldsymbol{x}_i\boldsymbol{y}_i^{\top}\Vert_F = \Vert \boldsymbol{x}_i\Vert \Vert\boldsymbol{y}_i\Vert$来采样$\boldsymbol{X},\boldsymbol{Y}$对应的列/行，并乘以$(r p_i)^{-1/2}$，就可以得到$\boldsymbol{X}\boldsymbol{Y}$的一个CR近似，从而将乘法复杂度从$\mathcal{O}(lmn)$降低到$\mathcal{O}(rmn)$。

## 延伸讨论

不管是按模长排序还是按$p_i\propto \Vert\boldsymbol{v}_i\Vert$随机采样，它们都允许我们在线性复杂度【即$\mathcal{O}(l)$】内构建一个CR近似，这对于实时计算来说当然是很理想的，但由于排序或采样都只依赖于$\Vert\boldsymbol{v}_i\Vert$，所以精度只能说一般。如果我们可以接受更高的复杂度，那么如何提高CR近似的精度呢？

我们可以尝试将排序的单位改为$k$元组。简单起见，假设$k \leq l-r$是$l-r$的一个因数，$l$个向量$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l$选$k$个的组合数为$C_l^k$，对于每个组合$\{s_1,s_2,\cdots,s_k\}$我们都可以算出向量和的模长$\Vert \boldsymbol{v}_{s_1} + \boldsymbol{v}_{s_2} + \cdots + \boldsymbol{v}_{s_k}\Vert$。有了这些数据，我们就可以贪婪地构建$\eqref{eq:v-l-k-0}$的近似解：

> 初始化$\Omega = \{1,2,\cdots,l\},\Theta=\{\}$
> 
> 
> 
> 对于$t=1,2,\cdots,(l-r)/k$，执行：
>      $\Theta = \Theta\,\cup\,\mathop{\text{argmin}}\limits_{\{s_1,s_2,\cdots,s_k\}\subset \Omega}\Vert \boldsymbol{v}_{s_1} + \boldsymbol{v}_{s_2} + \cdots + \boldsymbol{v}_{s_k}\Vert$；
>      $\Omega = \Omega\,\backslash\,\Theta$；
> 
> 
> 
> 返回$\Theta$。

说白了，就是每次都从剩下的向量中挑选和模长最小的$k$个向量，重复挑选$(l-r)/k$次即得到$l-r$个向量，它是按照单个向量模长来排序的自然推广，其复杂度为$\mathcal{O}(C_l^k)$，当$k > 1$且$l$比较大时可能难以承受，这也侧面体现了原问题精确求解的复杂性。

另一个值得思考的问题是如果允许CR近似放宽为$\boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R}$，那么$\boldsymbol{\Lambda}$的最优解是什么呢？如果不限定$\boldsymbol{\Lambda}$的结构，那么答案可以由伪逆给出
\begin{equation}\boldsymbol{\Lambda}^* = \mathop{\text{argmin}}_{\boldsymbol{\Lambda}}\Vert \boldsymbol{C}\boldsymbol{\Lambda}\boldsymbol{R} - \boldsymbol{X}\boldsymbol{Y}\Vert_F^2 = \boldsymbol{C}^{\dagger}\boldsymbol{X}\boldsymbol{Y}\boldsymbol{R}^{\dagger}\end{equation}
如果$\boldsymbol{\Lambda}$必须是对角阵呢？那可以先将问题重述为给定$\{\boldsymbol{u}_1,\boldsymbol{u}_2,\cdots,\boldsymbol{u}_r\}\subset\{\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l\}$，求
\begin{equation}\mathop{\text{argmin}}_{\lambda_1,\lambda_2,\cdots,\lambda_r}\left\Vert\sum_{i=1}^r \lambda_i \boldsymbol{u}_i - \sum_{i=1}^l\boldsymbol{v}_i\right\Vert^2\end{equation}
我们记$\boldsymbol{U} = (\boldsymbol{u}_1,\boldsymbol{u}_2,\cdots,\boldsymbol{u}_r), \boldsymbol{V} = (\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_l), \boldsymbol{\lambda}=(\lambda_1,\lambda_2,\cdots,\lambda_r)^{\top}$，那么优化目标可以写成
\begin{equation}\mathop{\text{argmin}}_{\boldsymbol{\lambda}}\left\Vert\boldsymbol{U}\boldsymbol{\lambda} - \boldsymbol{V}\boldsymbol{1}_{l\times 1}\right\Vert^2\end{equation}
这同样可以通过伪逆写出最优解
\begin{equation}\boldsymbol{\lambda}^* =  \boldsymbol{U}^{\dagger}\boldsymbol{V}\boldsymbol{1}_{l\times 1} = (\boldsymbol{U}^{\top}\boldsymbol{U})^{-1}\boldsymbol{U}^{\top}\boldsymbol{V}\boldsymbol{1}_{l\times 1} \end{equation}
最后一个等号假设了$\boldsymbol{U}^{\top}\boldsymbol{U}$可逆，这通常能满足，如果不满足的话$(\boldsymbol{U}^{\top}\boldsymbol{U})^{-1}$改$(\boldsymbol{U}^{\top}\boldsymbol{U})^{\dagger}$就行。

现在的问题是直接套用上式的话对原始问题来说计算量太大，因为$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i \boldsymbol{y}_i^{\top})$，即$\boldsymbol{v}_i$是$mn$维向量，所以$\boldsymbol{V}$大小为$mn\times l$、$\boldsymbol{U}$大小为$mn\times r$，这在$m,n$较大时比较难受。利用$\boldsymbol{v}_i = \text{vec}(\boldsymbol{x}_i \boldsymbol{y}_i^{\top})$能帮我们进一步化简上式。不妨设$\boldsymbol{u}_i = \text{vec}(\boldsymbol{c}_i \boldsymbol{r}_i^{\top})$，那么
\begin{equation}\begin{aligned}(\boldsymbol{U}^{\top}\boldsymbol{V})_{i,j} =&\, \langle \boldsymbol{c}_i \boldsymbol{r}_i^{\top}, \boldsymbol{x}_j \boldsymbol{y}_j^{\top}\rangle_F = \text{Tr}(\boldsymbol{r}_i \boldsymbol{c}_i^{\top}\boldsymbol{x}_j \boldsymbol{y}_j^{\top}) = (\boldsymbol{c}_i^{\top}\boldsymbol{x}_j)(\boldsymbol{r}_i^{\top} \boldsymbol{y}_j) \\[5pt]
=&\, [(\boldsymbol{C}^{\top}\boldsymbol{X})\otimes (\boldsymbol{R}\boldsymbol{Y}^{\top})]_{i,j}
\end{aligned}\end{equation}
即$\boldsymbol{U}^{\top}\boldsymbol{V}=(\boldsymbol{C}^{\top}\boldsymbol{X})\otimes (\boldsymbol{R}\boldsymbol{Y}^{\top}),\boldsymbol{U}^{\top}\boldsymbol{U}=(\boldsymbol{C}^{\top}\boldsymbol{C})\otimes (\boldsymbol{R}\boldsymbol{R}^{\top})$，这里的$\otimes$是[Hadamard积](<https://en.wikipedia.org/wiki/Hadamard_product_%28matrices%29>)，这样恒等变换之后$\boldsymbol{U}^{\top}\boldsymbol{V}$和$\boldsymbol{U}^{\top}\boldsymbol{U}$的计算量就降低了。

## 文章小结

本文介绍了矩阵乘法的CR近似，这是一种具有特定行列结构的低秩近似，相比由SVD给出的最优低秩近似，CR近似具有更直观的物理意义以及更好的可解释性。
