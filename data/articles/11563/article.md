# DeltaNet的核心逆矩阵的元素总是在&#91;\-1, 1&#93;内

> 作者：苏剑林 · 科学空间 · 2026-01-26
>
> 原文：<https://spaces.ac.cn/archives/11563>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从[《线性注意力简史：从模仿、创新到反哺》](<https://spaces.ac.cn/archives/11033>)中我们可以看到，DeltaNet的并行形式涉及到了形如$(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}$的逆矩阵。近日读者 [@Arch123](<https://spaces.ac.cn/archives/11486/comment-page-1#comment-29143>) 提出，通过实验可观察到该逆矩阵的元素总是在$[-1, 1]$内，问是否可以从数学上证实或证伪它。

在这篇文章中，我们将通过两种不同的方式证明这个结论是严格成立的。

## 问题描述

首先，我们准确地重述一下问题。设有矩阵$\boldsymbol{K}=[\boldsymbol{k}_1,\boldsymbol{k}_2,\cdots,\boldsymbol{k}_n]^{\top}\in\mathbb{R}^{n\times d}$，其中每个$\boldsymbol{k}_i\in\mathbb{R}^{d\times 1}$是模长不超过1的列向量，$\boldsymbol{M}\in\mathbb{R}^{n\times n}$是一个下三角的掩码矩阵，定义为
\begin{equation}M_{i,j} = \left\{\begin{aligned} &1, &i \geq j \\ &0, &i < j\end{aligned}\right.\end{equation}
$\boldsymbol{I}$是单位阵，$\boldsymbol{M}^- = \boldsymbol{M} - \boldsymbol{I}$。我们要证明的是：
\begin{equation}(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}\quad\in\quad [-1, 1]^{n\times n}\end{equation}
记$\boldsymbol{X}=\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-$，它的逆矩阵记为$\boldsymbol{Y}$，将$\boldsymbol{X}$显式写出来是
\begin{equation}\boldsymbol{X}=\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^- = \left(\begin{array}{cccccc}
1 & 0 & 0 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 & 0 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3 & 1 & \cdots & 0 & 0 \\[5pt]
\vdots & \vdots & \vdots & \ddots &  \vdots & \vdots \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_2^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_3^{\top}\boldsymbol{k}_{n-1} & \cdots & 1 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_n & \boldsymbol{k}_2^{\top}\boldsymbol{k}_n & \boldsymbol{k}_3^{\top}\boldsymbol{k}_n & \cdots & \boldsymbol{k}_{n-1}^{\top}\boldsymbol{k}_n & 1
\end{array}\right)\end{equation}
由于每个$\boldsymbol{k}_i$的模长都不超过1，显然$\boldsymbol{X}$的每个元素都在$[-1,1]$内，我们要证明的是$\boldsymbol{Y}=\boldsymbol{X}^{-1}$的每个元素也都在$[-1,1]$内。

## 数学归纳

既然这个逆矩阵出现在DeltaNet中，那么它肯定是有明显的模型背景的，结合这些背景也许能帮助我们更快地证明它，这部分我们留到下一节。这一节我们还是将它当作一个纯粹的数学问题去思考。

### 对角性质

在[《“对角\+低秩”三角阵的高效求逆方法》](<https://spaces.ac.cn/archives/11072>)我们已经体会过，下三角矩阵有一些优良性质，首先它对矩阵乘法封闭，其次它满足“逆矩阵的对角线等于对角线的逆”，这还可以推广到块下三角矩阵，即“逆矩阵的对角块等于对角块的逆”。

由此，我们有$\boldsymbol{Y}_{[:n-1,:n-1]}=(\boldsymbol{X}_{[:n-1,:n-1]})^{-1}$以及$\boldsymbol{Y}_{[1:,1:]}=(\boldsymbol{X}_{[1:,1:]})^{-1}$，其中切片按Numpy理解。这一性质启发我们可以对$n$使用数学归纳法。假设结论对任意满足条件的$(n-1)\times d$的$\boldsymbol{K}$矩阵成立，那么对于矩阵$\boldsymbol{X}$，我们只需要考虑如下两种分块方式：
\begin{equation}\boldsymbol{X} = \left(\begin{array}{ccccc:c}
1 & 0 & 0 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 & 0 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3 & 1 & \cdots & 0 & 0 \\[5pt]
\vdots & \vdots & \vdots & \ddots &  \vdots & \vdots \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_2^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_3^{\top}\boldsymbol{k}_{n-1} & \cdots & 1 & 0 \\[5pt] \hdashline
\boldsymbol{k}_1^{\top}\boldsymbol{k}_n & \boldsymbol{k}_2^{\top}\boldsymbol{k}_n & \boldsymbol{k}_3^{\top}\boldsymbol{k}_n & \cdots & \boldsymbol{k}_{n-1}^{\top}\boldsymbol{k}_n & 1
\end{array}\right) = \left(\begin{array}{c:cccc}
1 & 0 & 0 & \cdots & 0 & 0 \\[5pt] \hdashline
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 & 0 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3 & 1 & \cdots & 0 & 0 \\[5pt]
\vdots & \vdots & \vdots & \ddots &  \vdots & \vdots \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_2^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_3^{\top}\boldsymbol{k}_{n-1} & \cdots & 1 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_n & \boldsymbol{k}_2^{\top}\boldsymbol{k}_n & \boldsymbol{k}_3^{\top}\boldsymbol{k}_n & \cdots & \boldsymbol{k}_{n-1}^{\top}\boldsymbol{k}_n & 1
\end{array}\right)\end{equation}
然后根据$\boldsymbol{Y}_{[:n-1,:n-1]}=(\boldsymbol{X}_{[:n-1,:n-1]})^{-1}$和$\boldsymbol{Y}_{[1:,1:]}=(\boldsymbol{X}_{[1:,1:]})^{-1}$，就可以证明$\boldsymbol{Y}$除左下角的$Y_{n,1}$外的所有元素都在$[-1,1]$内，所以只需要补充证明$Y_{n,1}\in [-1,1]$。

### 伴随矩阵

为证$Y_{n,1}\in [-1,1]$，我们考虑用[伴随矩阵](<https://en.wikipedia.org/wiki/Adjugate_matrix>)将$\boldsymbol{X}^{-1}$显式表示出来。由于$\boldsymbol{X}$是一个下三角矩阵并且对角线都是1，所以它的行列式为1，于是根据伴随矩阵的求逆公式，我们有
\begin{equation}Y_{n,1} = (-1)^{n+1}\det\left(\begin{array}{ccccc}
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 & 0 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3 & 1 & \cdots & 0 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_4 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_4 & \boldsymbol{k}_3^{\top}\boldsymbol{k}_4 & \cdots & 0 & 0 \\[5pt]
\vdots & \vdots & \vdots & \ddots &  \vdots & \vdots \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_2^{\top}\boldsymbol{k}_{n-1} & \boldsymbol{k}_3^{\top}\boldsymbol{k}_{n-1} & \cdots & \boldsymbol{k}_{n-2}^{\top}\boldsymbol{k}_{n-1} & 1 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_n & \boldsymbol{k}_2^{\top}\boldsymbol{k}_n & \boldsymbol{k}_3^{\top}\boldsymbol{k}_n & \cdots & \boldsymbol{k}_{n-2}^{\top}\boldsymbol{k}_n & \boldsymbol{k}_{n-1}^{\top}\boldsymbol{k}_n
\end{array}\right)\end{equation}
这是在下三角矩阵的基础上将次对角线元素都变成了1，所以行列式的计算更为复杂一些，但也不算特别复杂，因为最后一列只有两个非零元素，因此按照最后一列来展开行列式计算，它就转化成了两个$n-1$阶的行列式相减，这一特点依然将我们向数学归纳法引导。

### 连乘形式

我们可以尝试手算前几个结果：
\begin{gather}Y_{2,1} = -\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 ,\qquad Y_{3,1} = \det\left(\begin{array}{cc}
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3
\end{array}\right) = \boldsymbol{k}_1^{\top}\boldsymbol{k}_2\boldsymbol{k}_2^{\top}\boldsymbol{k}_3 - \boldsymbol{k}_1^{\top}\boldsymbol{k}_3 = -\boldsymbol{k}_1^{\top}(\boldsymbol{I}-\boldsymbol{k}_2\boldsymbol{k}_2^{\top})\boldsymbol{k}_3 \\[8pt]
Y_{4,1} = -\det\left(\begin{array}{ccc}
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 & 0 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3 & 1 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_4 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_4 & \boldsymbol{k}_3^{\top}\boldsymbol{k}_4
\end{array}\right) = \det\left(\begin{array}{cc}
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_4 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_4
\end{array}\right) -\det\left(\begin{array}{cc}
\boldsymbol{k}_1^{\top}\boldsymbol{k}_2 & 1 \\[5pt]
\boldsymbol{k}_1^{\top}\boldsymbol{k}_3 & \boldsymbol{k}_2^{\top}\boldsymbol{k}_3
\end{array}\right) \boldsymbol{k}_3^{\top}\boldsymbol{k}_4 = -\boldsymbol{k}_1^{\top}(\boldsymbol{I}-\boldsymbol{k}_2\boldsymbol{k}_2^{\top})(\boldsymbol{I}-\boldsymbol{k}_3\boldsymbol{k}_3^{\top})\boldsymbol{k}_4\end{gather}
由此可以猜测
\begin{equation}Y_{n,1} = -\boldsymbol{k}_1^{\top}(\boldsymbol{I}-\boldsymbol{k}_2\boldsymbol{k}_2^{\top})(\boldsymbol{I}-\boldsymbol{k}_3\boldsymbol{k}_3^{\top})\cdots (\boldsymbol{I}-\boldsymbol{k}_{n-1}\boldsymbol{k}_{n-1}^{\top})\boldsymbol{k}_n \end{equation}
请大家自行通过数学归纳法证明这个猜测。假设大家已经完成了这个猜测的证明，那么接着根据条件$\Vert\boldsymbol{k}_i\Vert\leq 1$，可得$\Vert(\boldsymbol{I}-\boldsymbol{k}_i\boldsymbol{k}_i^{\top})\boldsymbol{x}\Vert^2 = \Vert\boldsymbol{x}\Vert^2 + (\boldsymbol{k}_i^{\top} \boldsymbol{x})^2 (\Vert\boldsymbol{k}_i\Vert^2 - 2) \leq \Vert\boldsymbol{x}\Vert^2$，因此
\begin{equation}|Y_{n,1}| \leq \Vert\boldsymbol{k}_1\Vert\times \Vert(\boldsymbol{I}-\boldsymbol{k}_2\boldsymbol{k}_2^{\top})(\boldsymbol{I}-\boldsymbol{k}_3\boldsymbol{k}_3^{\top})\cdots (\boldsymbol{I}-\boldsymbol{k}_{n-1}\boldsymbol{k}_{n-1}^{\top})\boldsymbol{k}_n\Vert \leq \Vert\boldsymbol{k}_1\Vert\times \Vert\boldsymbol{k}_n\Vert \leq 1\end{equation}
这就完成了$Y_{n,1}\in [-1,1]$的证明，继而由数学归纳法知$\boldsymbol{Y}\in[-1, 1]^{n\times n}$恒成立。

## 双重计算

上一节的直接证明显得比较“暴力”，后来FLA群的 [@zhxlin](<https://x.com/zhxlin>) 同学提醒，我们可以回到该逆矩阵的原始背景——DeltaNet，用两种不同的方式对DeltaNet进行计算，通过对照结果得出逆矩阵的显式表达式，从而完成一个相对比较简洁的证明。

### 逆阵形式

我们知道，DeltaNet的递归形式是
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1}(\boldsymbol{I} - \boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \boldsymbol{v}_t\boldsymbol{k}_t^{\top} = \boldsymbol{S}_{t-1} + (\boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top}\end{equation}
其中$\boldsymbol{S}_0 = \boldsymbol{0}$。先看第二个等号，定义$\boldsymbol{u}_t=\boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t$，那么
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{u}_t\boldsymbol{k}_t^{\top} = \sum_{i=1}^t \boldsymbol{u}_i\boldsymbol{k}_i^{\top}\end{equation}
所以
\begin{equation}\boldsymbol{u}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t = \boldsymbol{v}_t - \left(\sum_{i=1}^{t-1} \boldsymbol{u}_i\boldsymbol{k}_i^{\top}\right) \boldsymbol{k}_t = \boldsymbol{v}_t - \sum_{i=1}^{t-1} \boldsymbol{u}_i(\boldsymbol{k}_i^{\top} \boldsymbol{k}_t)\end{equation}
定义$\boldsymbol{U}=[\boldsymbol{u}_1,\boldsymbol{u}_2,\cdots,\boldsymbol{u}_n]^{\top}, \boldsymbol{V}=[\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_n]^{\top}$，上式等价于$\boldsymbol{U} = \boldsymbol{V} - (\boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)\boldsymbol{U}$，或$\boldsymbol{U}=(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}\boldsymbol{V}$，这便是逆矩阵$(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}$的来源，它是DeltaNet及其后续工作如GDN、KDA等并行计算的关键。

### 递推展开

再看第一个等号，直接逐次展开得
\begin{equation}\boldsymbol{S}_t = \boldsymbol{v}_1\boldsymbol{k}_1^{\top} \boldsymbol{H}_{2\to t} + \cdots + \boldsymbol{v}_{t-1}\boldsymbol{k}_{t-1}^{\top} \boldsymbol{H}_{t\to t} + \boldsymbol{v}_t\boldsymbol{k}_t^{\top} = \sum_{i=1}^t \boldsymbol{v}_i\boldsymbol{k}_i^{\top} \boldsymbol{H}_{i+1\to t} \end{equation}
这里$\boldsymbol{H}_{r\to t} \triangleq (\boldsymbol{I} - \boldsymbol{k}_r\boldsymbol{k}_r^{\top})(\boldsymbol{I} - \boldsymbol{k}_{r+1}\boldsymbol{k}_{r+1}^{\top})\cdots(\boldsymbol{I} - \boldsymbol{k}_t\boldsymbol{k}_t^{\top})$，并约定$\boldsymbol{H}_{t+1\to t}=\boldsymbol{I}$。

将上式代入$\boldsymbol{u}_t$的定义得
\begin{equation}\boldsymbol{u}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t = \boldsymbol{v}_t -  \sum_{i=1}^{t-1} \boldsymbol{v}_i\boldsymbol{k}_i^{\top} \boldsymbol{H}_{i+1\to t-1}\boldsymbol{k}_t\end{equation}

### 对比结果

仍记$\boldsymbol{Y}=(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}$，那么$\boldsymbol{U}=\boldsymbol{Y}\boldsymbol{V}$给出$\boldsymbol{u}_t = \sum_{i=1}^n Y_{t, i} \boldsymbol{v}_i$，对照上式可以读出
\begin{equation}Y_{t, i} = \left\{\begin{aligned}& 0, & i > t \\
& 1, & i = t \\
& -\boldsymbol{k}_i^{\top} \boldsymbol{H}_{i+1\to t-1}\boldsymbol{k}_t, & i < t \\
\end{aligned}\right.\end{equation}
因此只需要证明$|\boldsymbol{k}_i^{\top} \boldsymbol{H}_{i+1\to t-1}\boldsymbol{k}_t|\leq 1$，这由$\Vert\boldsymbol{k}_i\Vert\leq 1$以及第一个证明证得的$\Vert(\boldsymbol{I}-\boldsymbol{k}_i\boldsymbol{k}_i^{\top})\boldsymbol{x}\Vert \leq \Vert\boldsymbol{x}\Vert$立马可得
\begin{equation}|\boldsymbol{k}_i^{\top} \boldsymbol{H}_{i+1\to t-1}\boldsymbol{k}_t| \leq \Vert\boldsymbol{k}_i\Vert\times \Vert\boldsymbol{H}_{i+1\to t-1}\boldsymbol{k}_t\Vert \leq \Vert\boldsymbol{k}_i\Vert\times \Vert\boldsymbol{k}_t\Vert \leq 1\end{equation}
因此待证结论成立。

## 文章小结

本文给出了DeltaNet中的核心逆矩阵的有界性的两个证明。
