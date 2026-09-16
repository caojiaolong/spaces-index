# 矩阵符号函数mcsgn能计算什么？

> 作者：苏剑林 · 科学空间 · 2025-06-23
>
> 原文：<https://spaces.ac.cn/archives/11056>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《msign的导数》](<https://spaces.ac.cn/archives/11025>)一文中，我们正式引入了两种矩阵符号函数$\newcommand{msign}{\mathop{\text{msign}}}\msign$和$\newcommand{mcsgn}{\mathop{\text{mcsgn}}}\mcsgn$，其中$\msign$是Muon的核心运算，而$\mcsgn$则是用来解[Sylvester方程](<https://en.wikipedia.org/wiki/Sylvester_equation>)。那么$\mcsgn$除了用来解Sylvester方程外，还能干些什么呢？本文就来整理一下这个问题的答案。

## 两种符号

设矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，我们有两种矩阵符号函数
\begin{gather}\msign(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2} \\[6pt]
\mcsgn(\boldsymbol{M}) = (\boldsymbol{M}^2)^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^2)^{-1/2}
\end{gather}
第一种适用于任意形状的矩阵，第二种只适用于方阵，$^{-1/2}$是矩阵$1/2$次幂的逆，如果不可逆则按“[伪逆](<https://spaces.ac.cn/archives/10366>)”来算。一般情况下$\msign$和$\mcsgn$会得到不同的结果，但当$\boldsymbol{M}$是对称矩阵时它们则相等。

它们的区别是，如果$\boldsymbol{M}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，其中$\boldsymbol{U},\boldsymbol{V}$是正交矩阵，那么$\msign(\boldsymbol{M}) = \boldsymbol{U}\msign(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}$；如果$\boldsymbol{M}=\boldsymbol{P}\boldsymbol{\Lambda}\boldsymbol{P}^{-1}$，其中$\boldsymbol{P}$是可逆矩阵，那么$\mcsgn(\boldsymbol{M})=\boldsymbol{P}\mcsgn(\boldsymbol{\Lambda})\boldsymbol{P}^{-1}$。说白了，一个具有正交不变性，一个具有相似不变性，一个会把所有非零奇异值变成1，一个会把所有非零特征值变成$\pm 1$。

关于$\msign$的计算，可以看[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)和[《msign算子的Newton\-Schulz迭代（下）》](<https://spaces.ac.cn/archives/10996>)，它是GPU高效的。至于$\mcsgn$，由于特征值允许复数，一般情况会很复杂，但当$\boldsymbol{M}$的特征值全是实数时（实际上用到$\mcsgn$的场景几乎都是这样），可以复用$\msign$的迭代：
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\sqrt{\tr(\boldsymbol{M}^2)}},\qquad \boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t^3 + c_{t+1}\boldsymbol{X}_t^5\end{equation}

更多的性质我们就不展开了，接下来我们主要看$\mcsgn$的应用。

## 分块恒等

历史上，$\mcsgn$就是为了解方程而引入的，只不过不单单是Sylvester方程，还包含更一般的[代数Riccati方程](<https://en.wikipedia.org/wiki/Algebraic_Riccati_equation>)，原论文可见[《Solving the algebraic Riccati equation with the matrix sign function》](<https://www.sciencedirect.com/science/article/pii/0024379587902229>)。

考虑分块矩阵$\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}$，我们有$\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} = \begin{bmatrix}\boldsymbol{0} & \boldsymbol{I} \\ -\boldsymbol{I} & \boldsymbol{X}\end{bmatrix}$，可以验算
\begin{equation}\begin{bmatrix}\boldsymbol{0} & \boldsymbol{I} \\ -\boldsymbol{I} & \boldsymbol{X}\end{bmatrix}\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{D} & \boldsymbol{B}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}=\begin{bmatrix}\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X} & -\boldsymbol{D} \\ \boldsymbol{X}\boldsymbol{D}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} - \boldsymbol{C} & \boldsymbol{A} - \boldsymbol{X}\boldsymbol{D}\end{bmatrix}\end{equation}
如果
\begin{equation}\boldsymbol{X}\boldsymbol{D}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} - \boldsymbol{C} = \boldsymbol{0}\label{eq:riccati}\end{equation}
那么
\begin{equation}\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{D} & \boldsymbol{B}\end{bmatrix}=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X} & -\boldsymbol{D} \\ \boldsymbol{0} & \boldsymbol{A} - \boldsymbol{X}\boldsymbol{D}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1}\end{equation}
式$\eqref{eq:riccati}$即为代数Riccati方程。两边取$\mcsgn$，我们有恒等式
\begin{equation}\begin{aligned}
\mcsgn\left(\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{D} & \boldsymbol{B}\end{bmatrix}\right)=&\,\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\mcsgn\left(\begin{bmatrix}\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X} & -\boldsymbol{D} \\ \boldsymbol{0} & \boldsymbol{A} - \boldsymbol{X}\boldsymbol{D}\end{bmatrix}\right)\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} \\[6pt]
=&\,\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\mcsgn(\boldsymbol{B} + \boldsymbol{D}\boldsymbol{X}) & \boldsymbol{Y} \\ \boldsymbol{0} & \mcsgn(\boldsymbol{A} - \boldsymbol{X}\boldsymbol{D})\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1}
\end{aligned}\end{equation}
第二个等号是利用了（分块）三角矩阵的特性。三角矩阵的特征值就是它的对角线元素，所以三角矩阵取$\mcsgn$，结果也是一个三角矩阵，其对角线元素等于原矩阵对角线元素的$\mathop{\text{csgn}}$，这个特性对分块三角矩阵也成立，所以结果具有第二个等号的形式，$\boldsymbol{Y}$是一个待定矩阵。

## 几个结果

下面就是根据具体情况来进一步化简，得到一些可能会用到的结果。

### 第一例子

假设$\boldsymbol{D}=\boldsymbol{0}$，$\boldsymbol{B}$是正定的，$\boldsymbol{A}$是负定的，分块对角阵运算是封闭的，于是$\boldsymbol{Y}=\boldsymbol{0}$，那么
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{0} & \boldsymbol{B}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{0} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} = \begin{bmatrix}-\boldsymbol{I} & 2\boldsymbol{X} \\ \boldsymbol{0} & \boldsymbol{I}\end{bmatrix}\end{equation}
这意味着直接从$\mcsgn\left(\begin{bmatrix}\boldsymbol{A} & \boldsymbol{C} \\ \boldsymbol{0} & \boldsymbol{B}\end{bmatrix}\right)$就可以读出Sylvester方程$\boldsymbol{X}\boldsymbol{B} - \boldsymbol{A}\boldsymbol{X} = \boldsymbol{C}$的解。

### 第二例子

假设$\boldsymbol{A},\boldsymbol{B}=\boldsymbol{0}$，$\boldsymbol{D}=\boldsymbol{I}$，$\boldsymbol{C}$是正定矩阵，那么Riccati方程简化为$\boldsymbol{X}^2 = \boldsymbol{C}$，即$\boldsymbol{X}=\boldsymbol{C}^{1/2}$，那么$\mcsgn(\boldsymbol{C}^{1/2})=\boldsymbol{I}$，所以
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{Y} \\ \boldsymbol{0} & - \boldsymbol{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1} = \begin{bmatrix}-\boldsymbol{X}\boldsymbol{Y}-\boldsymbol{I} & 2\boldsymbol{X} + \boldsymbol{X}\boldsymbol{Y}\boldsymbol{X} \\ -\boldsymbol{Y} & \boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}\end{bmatrix}\end{equation}
注意到$\mcsgn$是奇函数，反对角阵的奇函数必然也是反对角阵，因此$\boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}=\boldsymbol{0}$，从中解得$\boldsymbol{Y} = -\boldsymbol{X}^{-1} = -\boldsymbol{C}^{-1/2}$，代入上式得到
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C}^{1/2} \\ \boldsymbol{C}^{-1/2} & \boldsymbol{0}\end{bmatrix}\end{equation}
这表明$\mcsgn$还可以用来算矩阵的平方根和逆平方根。

### 第三例子

假设$\boldsymbol{A},\boldsymbol{B}=\boldsymbol{0}$，$\boldsymbol{D}=\boldsymbol{C}^{\top}$，那么Riccati方程简化为$\boldsymbol{X}\boldsymbol{C}^{\top}\boldsymbol{X} = \boldsymbol{C}$，容易验证$\boldsymbol{X}=\msign(\boldsymbol{C})$正是它的解。我们只演示最理想的情况，$\boldsymbol{C}$是满秩方阵，那么$\boldsymbol{C}^{\top}\boldsymbol{X}$和$\boldsymbol{X}\boldsymbol{C}^{\top}$都是正定的，于是有
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\ \boldsymbol{C}^{\top} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}\begin{bmatrix}\boldsymbol{I} & \boldsymbol{Y} \\ \boldsymbol{0} & -\boldsymbol{I}\end{bmatrix}\begin{bmatrix}\boldsymbol{X} & -\boldsymbol{I} \\ \boldsymbol{I} & \boldsymbol{0}\end{bmatrix}^{-1}=\begin{bmatrix}-\boldsymbol{X}\boldsymbol{Y}-\boldsymbol{I} & 2\boldsymbol{X} + \boldsymbol{X}\boldsymbol{Y}\boldsymbol{X} \\ -\boldsymbol{Y} & \boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}\end{bmatrix}\end{equation}
跟上一节同理$\boldsymbol{Y}\boldsymbol{X} + \boldsymbol{I}=0$，所以
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\ \boldsymbol{C}^{\top} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{0} & \msign(\boldsymbol{C}) \\ \msign(\boldsymbol{C}^{\top}) & \boldsymbol{0}\end{bmatrix}\end{equation}
即$\mcsgn$也可以用来算$\msign$。其实可以直接证明这个等式对于任意矩阵$\boldsymbol{C}$都是成立的，但如果从这里的解Riccati方程角度来证，则会有些繁琐的细节，读者可以自行补充一下。

### 第四例子

第二例子和第三例子可以推广为一个更一般的结论：
\begin{equation}\mcsgn\left(\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C} \\ \boldsymbol{D} & \boldsymbol{0}\end{bmatrix}\right)=\begin{bmatrix}\boldsymbol{0} & \boldsymbol{C}(\boldsymbol{D}\boldsymbol{C})^{-1/2} \\ \boldsymbol{D}(\boldsymbol{C}\boldsymbol{D})^{-1/2} & \boldsymbol{0}\end{bmatrix}\end{equation}
对任意形状适合的$\boldsymbol{C},\boldsymbol{D}$都恒成立。矩阵请读者自行完成证明。

## 文章小结

本文主要从解代数Riccati方程的角度，整理了几个$\mcsgn$相关的恒等式。
