# 基于流式幂迭代的Muon实现：4\. 原理

> 作者：苏剑林 · 科学空间 · 2026-04-13
>
> 原文：<https://spaces.ac.cn/archives/11710>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

经过[《基于流式幂迭代的Muon实现：1\. 初识》](<https://spaces.ac.cn/archives/11654>)、[《基于流式幂迭代的Muon实现：2\. 加速》](<https://spaces.ac.cn/archives/11673>)和[《基于流式幂迭代的Muon实现：3\. 雕琢》](<https://spaces.ac.cn/archives/11697>)三篇文章，想必大家已经对流式幂迭代（Streaming Power Iteration）的思想、实现、加速等细节有所了解，总的来说，这称得上是一种颇有竞争力的Muon实现方式，并且得益于它直接近似计算SVD，所以它还具备更好的拓展性。

受限于篇幅，当时我们对相关运算的数学原理描述得相对简略，因此在这篇文章中，我们补充部分关于幂迭代和QR分解的数学推导，以建立更完整的理论图景。不过，这里的推导依然是侧重解释性而非严格性，主要是为了帮大家（包括笔者）理清思路，还请专业读者海涵。

## 共轴等价

在开始推导之前，我们需要先引入“共轴等价”的概念。对于矩阵$\boldsymbol{A},\boldsymbol{B}\in\mathbb{R}^{n\times m}$，如果存在一个符号矩阵$\boldsymbol{S}$满足$\boldsymbol{A} = \boldsymbol{B}\boldsymbol{S}$，那么称$\boldsymbol{A}$与$\boldsymbol{B}$“共轴等价（Coaxial Equivalent）”，它们互为对方的“共轴矩阵”。这里的“[符号矩阵（Signature matrix）](<https://en.wikipedia.org/wiki/Signature_matrix>)”是指为对角线为$\pm 1$的对角矩阵，即$\newcommand{diag}{\mathop{\text{diag}}}\diag(\pm 1, \pm 1, \cdots, \pm 1)$。

需要说明的是，“共轴（Coaxial）”这个词是笔者自行提出用来描述这种等价关系的，因为在坐标系视角下，满足条件的矩阵$\boldsymbol{A},\boldsymbol{B}$实际上描述了同一个坐标系，只不过某些轴的正方向选择有所不同，有些文献似乎称它们为“符号等价（Sign Equivalent）”，但笔者感觉不如“共轴”来得直观。

之所以引入共轴的概念，是因为很多矩阵分解都是在共轴等价的意义下才唯一的（所以笔者也疑惑，这么常用的一种等价关系居然没有标准命名）。比如QR分解，设矩阵$\boldsymbol{A}\in\mathbb{R}^{n\times m}(n\geq m)$且满秩，$\boldsymbol{Q}_1\boldsymbol{R}_1$和$\boldsymbol{Q}_2\boldsymbol{R}_2$是它的两种QR分解，那么$\boldsymbol{Q}_1$与$\boldsymbol{Q}_2$共轴、$\boldsymbol{R}_1$与$\boldsymbol{R}_2$共轴。

这种唯一性不难理解，因为在Gram\-Schmidt正交化的过程中，我们可以随意反转正交化后的向量而不改变正交性。一般的教程通常是限定$\boldsymbol{R}$的对角线元素为正来表达QR分解的唯一性，这也是一种途径，但有时候会带来不必要的麻烦。此外，[SVD](<https://spaces.ac.cn/archives/10407>)的唯一性也是在共轴等价下成立的。

## 幂之迭代

在[《基于流式幂迭代的Muon实现：1\. 初识》](<https://spaces.ac.cn/archives/11654>)中，我们是通过逐特征向量求解的方式来引入幂迭代的，然后直接切换到并行版本，认为它具有相同的收敛结果。这里则直接从并行版本出发，证明它的收敛性。

还是设矩阵$\boldsymbol{A}\in\mathbb{R}^{n\times m}(n\geq m)$，并且设它满秩（秩为$m$），考虑幂迭代
\begin{equation}\newcommand{QR}{\mathop{\text{QR}}}\boldsymbol{V}_t = \QR(\boldsymbol{A}^{\top}\boldsymbol{A} \boldsymbol{V}_{t-1}),\qquad \boldsymbol{V}_0 = \boldsymbol{I}\end{equation}
设$\boldsymbol{A}^{\top}\boldsymbol{A} \boldsymbol{V}_{t-1}$的QR分解是$\boldsymbol{Q}_t \boldsymbol{R}_t$，那么$\QR(\boldsymbol{A}^{\top}\boldsymbol{A} \boldsymbol{V}_{t-1}) = \boldsymbol{A}^{\top}\boldsymbol{A} \boldsymbol{V}_{t-1}\boldsymbol{R}_t^{-1}$，那么逐步迭代得
\begin{equation}\boldsymbol{V}_t = (\boldsymbol{A}^{\top}\boldsymbol{A})^t \boldsymbol{R}_1^{-1}\boldsymbol{R}_2^{-1} \cdots \boldsymbol{R}_t^{-1}\end{equation}
接着设$\boldsymbol{A}$的SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，那么$\boldsymbol{A}^{\top}\boldsymbol{A}=\boldsymbol{V}\boldsymbol{\Sigma}^2\boldsymbol{V}^{\top}$，代入到上式得
\begin{equation}\boldsymbol{V}_t = \boldsymbol{V}\boldsymbol{\Sigma}^{2t}\boldsymbol{V}^{\top} \boldsymbol{R}_1^{-1}\boldsymbol{R}_2^{-1} \cdots \boldsymbol{R}_t^{-1} = \boldsymbol{V}\underbrace{(\boldsymbol{\Sigma}^{2t}\boldsymbol{V}^{\top}\boldsymbol{\Sigma}^{-2t})}_{\boldsymbol{B}}\,\underbrace{(\boldsymbol{\Sigma}^{2t} \boldsymbol{R}_1^{-1}\boldsymbol{R}_2^{-1} \cdots \boldsymbol{R}_t^{-1})}_{\boldsymbol{C}}\end{equation}
结果分$\boldsymbol{V},\boldsymbol{B},\boldsymbol{C}$三部分，我们要证明的是在共轴意义下$\boldsymbol{V}_t\to \boldsymbol{V}$。已知$\boldsymbol{V}_t,\boldsymbol{V}$都是正交矩阵，然后由上三角矩阵的封闭性可知$\boldsymbol{C}$是一个上三角矩阵，如果我们能够证明$\boldsymbol{B}$会趋于一个上三角阵，那么由QR分解的唯一性便可得到$\boldsymbol{V}_t\to \boldsymbol{V}$。关于$\boldsymbol{B}=\boldsymbol{\Sigma}^{2t}\boldsymbol{V}^{\top}\boldsymbol{\Sigma}^{-2t}$，用分量形式写出来是
\begin{equation}B_{i,j} = V_{j,i} (\sigma_i / \sigma_j)^{2t}\end{equation}
假设$\sigma_1 > \sigma_2 > \cdots > \sigma_m$，那么当$i > j$时$(\sigma_i / \sigma_j)^{2t}\to 0$，即$\boldsymbol{B}$的下三角部分趋于0，换言之$\boldsymbol{B}$趋于上三角阵，条件得证。由此还可知，幂迭代的收敛速度取决于相邻奇异值之比，$\sigma_i / \sigma_{i+1}$越大，收敛越快，一旦存在相等的奇异值，幂迭代就会失效，但实践中可以假设两个奇异值严格相等的概率为0，从而回避这个问题。

## 本质思考

让我们仔细思考一下整个证明过程，理解一下它本质上在做什么。

首先，$\boldsymbol{C} = \boldsymbol{\Sigma}^{2t} \boldsymbol{R}_1^{-1}\boldsymbol{R}_2^{-1} \cdots \boldsymbol{R}_t^{-1}$这一块看起来复杂，但它的唯一作用仅仅是“一个上三角矩阵”，即它等于什么根本不重要，只要保持上三角矩阵的形式即可。真正发挥核心作用的是$\boldsymbol{B}=\boldsymbol{\Sigma}^{2t}\boldsymbol{V}^{\top}\boldsymbol{\Sigma}^{-2t}$会趋于一个上三角矩阵，这使得前面的$\boldsymbol{V}$能够被QR分解提炼出来。

真正达到这个效果的，其实是$(\boldsymbol{A}^{\top}\boldsymbol{A})^t$这一步！换句话说，理论上有$\boldsymbol{V}_t = \QR((\boldsymbol{A}^{\top}\boldsymbol{A})^t)$，即前面那么多次$\QR$理论上都是多余的，只要在最后做一次$\QR$或者说正交化就行了。当然，这种等价性只有理论价值，直接计算$(\boldsymbol{A}^{\top}\boldsymbol{A})^t$会数值爆炸/坍缩，导致结果不可用。所以，定期执行$\QR$（而不是最后才执行一次），本质作用就是保持数值稳定性。

我们也可以从“[预处理器（Preconditioner）](<https://en.wikipedia.org/wiki/Preconditioner>)”的角度去理解多次QR的操作。预处理器指对输入的一些预处理操作，这些操作理论上不改变输出，但通常对数值计算有益。对于$\QR$来说，右乘任意满秩的上三角阵$\boldsymbol{R}$并不会改变结果，即$\QR(\boldsymbol{A}) = \QR(\boldsymbol{A}\boldsymbol{R})$，所以任意右乘的满秩上三角阵都可以称为$\QR$的预处理器。

我们知道，$\QR$本身也可以写成右乘上三角阵的形式，所以$\QR$本身就是$\QR$的一种预处理器，因此我们可以往$(\boldsymbol{A}^{\top}\boldsymbol{A})^t$里边插入一些$\QR$，变成
\begin{equation}(\boldsymbol{A}^{\top}\boldsymbol{A})^t\qquad\to\qquad \boldsymbol{A}^{\top}\boldsymbol{A}\QR(\boldsymbol{A}^{\top}\boldsymbol{A}\QR(\boldsymbol{A}^{\top}\boldsymbol{A}\cdots\QR(\boldsymbol{A}^{\top}\boldsymbol{A})))\end{equation}
最后两端都$\QR$，结果不会改变，但由于每次$\QR$出来一个正交矩阵，右乘一个正交矩阵几乎没有数值爆炸的风险，所以$\QR$是它自身良好的预处理器。

## 相关变体

从预处理器的角度，我们还可以理解很多幂迭代变体。比如，我们在开篇[《基于流式幂迭代的Muon实现：1\. 初识》](<https://spaces.ac.cn/archives/11654>)就用过的$\newcommand{ColNorm}{\mathop{\text{ColNorm}}}\ColNorm$版：
\begin{equation}\boldsymbol{V}_t = \QR(\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{V}_{t-1}) \qquad\to\qquad \boldsymbol{V}_t = \QR(\boldsymbol{A}^{\top}\ColNorm(\boldsymbol{A}\boldsymbol{V}_{t-1}))\end{equation}
左右两个$\boldsymbol{V}_t$是相等的，因为对一个矩阵做$\ColNorm$等价于右乘一个对角阵：
\begin{equation}\underbrace{\left[\frac{\boldsymbol{a}_1}{\Vert\boldsymbol{a}_1\Vert},\frac{\boldsymbol{a}_2}{\Vert\boldsymbol{a}_2\Vert},\cdots,\frac{\boldsymbol{a}_m}{\Vert\boldsymbol{a}_m\Vert}\right]}_{\ColNorm([\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m])} =  [\boldsymbol{a}_1,\boldsymbol{a}_2,\cdots,\boldsymbol{a}_m]\begin{bmatrix}\Vert\boldsymbol{a}_1\Vert^{-1} & 0 & \cdots & 0 \\
0 & \Vert\boldsymbol{a}_2\Vert^{-1} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \Vert\boldsymbol{a}_m\Vert^{-1}\end{bmatrix}\end{equation}
对角阵是上三角阵的一个特例，所以$\ColNorm$也是$\QR$的一个预处理器，它不改变结果。基于同样的理由，我们也可以把$\ColNorm$往外挪一步，这同样不改变$\QR$的结果
\begin{equation}\boldsymbol{V}_t = \QR(\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{V}_{t-1}) \qquad\to\qquad \boldsymbol{V}_t = \QR(\ColNorm(\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{V}_{t-1}))\end{equation}
此外，[《基于流式幂迭代的Muon实现：2\. 加速》](<https://spaces.ac.cn/archives/11673>)和[《基于流式幂迭代的Muon实现：3\. 雕琢》](<https://spaces.ac.cn/archives/11697>)的主角是双重$\QR$的幂迭代：
\begin{equation}\boldsymbol{V}_t = \QR(\boldsymbol{A}^{\top}\boldsymbol{A}\boldsymbol{V}_{t-1}) \qquad\to\qquad \boldsymbol{V}_t = \QR(\boldsymbol{A}^{\top}\QR(\boldsymbol{A}\boldsymbol{V}_{t-1}))\end{equation}
基于这两节的讨论，容易证明给$\boldsymbol{A}\boldsymbol{V}_{t-1}$多加一步$\QR$理论上也并不改变$\boldsymbol{V}_t$的结果。

## 有限误差

我们多次用到了“理论上”之类的字眼，是因为很多结果都是在无限精度的、精确的$\QR$基础上才能得到的，实际计算中的$\QR$是有限精度的，而且为了效率我们通常还会用不那么准确的“SCQR（Shifted Cholesky QR）”， 所以分析误差的累积效应就非常有必要。

对于SCQR，大家这几篇读下来应该不再陌生，它将矩阵$\boldsymbol{A}$的QR分解分为两步：1、对$\boldsymbol{A}^{\top}\boldsymbol{A}+\lambda \boldsymbol{I}$做Cholesky分解得到$\boldsymbol{R}$，2、通过$\boldsymbol{Q}=\boldsymbol{A}\boldsymbol{R}^{-1}$得到正交矩阵$\boldsymbol{Q}$。如果$\lambda=0$，那么SCQR跟精确QR等价，然而$\lambda=0$在实际计算时几乎总会是因为条件数过大而失败，所以设置适当的$\lambda > 0$，这就带来了误差。

幸运的是，这种误差并不会累积！不管是$\boldsymbol{A}^{\top}\boldsymbol{A}$还是$\boldsymbol{A}^{\top}\boldsymbol{A}+\lambda \boldsymbol{I}$，它们做Cholesky分解的结果都是一个上三角矩阵，然后$\boldsymbol{A}$会右乘这个上三角阵的逆，我们知道右乘上三角阵不改变$\QR$的结果。因此，如果不考虑矩阵乘法的误差，那么可以认为SCQR对$\QR$本身是“无损”的。

换言之，只要最后一步换用精确的$\QR$，那么不管前面做多少次不精准的SCQR，结果都还是精确的，反过来说，就是总的误差等价于单步SCQR（最后一步）的误差，而不会累积起来。依赖于这一原理的，还有我们在[《基于流式幂迭代的Muon实现：2\. 加速》](<https://spaces.ac.cn/archives/11673>)提到的“SCQR2”加速技巧。

相反，如果我们所提的一些变换不能写成“右乘上三角阵”的形式，那么就会有损地破坏了原始矩阵的信息，长期迭代下来就会累积误差，直至完全失效。上篇[《基于流式幂迭代的Muon实现：3\. 雕琢》](<https://spaces.ac.cn/archives/11697>)我们提到的 @Ji\_Ha\_Kim 同学的化简方案，就是一个经典反例。

## 乔列分解

最后，我们再来简单复习一下“[Cholesky分解](<https://en.wikipedia.org/wiki/Cholesky_decomposition>)”，它的目的是将给定正定对称矩阵$\boldsymbol{B}$表示成$\boldsymbol{L}\boldsymbol{L}^{\top}$的形式，其中$\boldsymbol{L}$是一个下三角阵。然后记$\boldsymbol{R}=\boldsymbol{L}^{\top}$，我们就可以得到上三角阵的形式$\boldsymbol{R}^{\top}\boldsymbol{R}$。

Cholesky分解很快，是因为它有可以直接递归计算的解析解。具体来说，从如下等式
\begin{equation}\begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1,m} \\
a_{2,1} & a_{2,2} & \cdots & a_{2,m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m,1} & a_{m,2} & \cdots & a_{m,m}
\end{bmatrix}=\begin{bmatrix}
l_{1,1} &        &        &        \\
l_{2,1} & l_{2,2} &        &        \\
\vdots & \vdots & \ddots &        \\
l_{m,1} & l_{m,2} & \cdots & l_{m,m}
\end{bmatrix}\begin{bmatrix}
l_{1,1} & l_{2,1} & \cdots & l_{m,1} \\
& l_{2,2} & \cdots & l_{m,2} \\
&        & \ddots & \vdots \\
&        &        & l_{m,m}
\end{bmatrix}\end{equation}
逐项对应相等，可解得递归公式
\begin{equation}l_{j,j}=\pm\sqrt {a_{j,j}-\sum_{k=1}^{j-1}l_{j,k}^2},\qquad l_{i,j}=\frac{1}{l_{j,j}}\left(a_{i,j}-\sum_{k=1}^{j-1}l_{i,k}l_{j,k}\right)\quad \text{对于 }i > j\end{equation}

Cholesky QR是Cholesky分解的经典应用之一，它跟施密特正交化类似，但允许我们跳过正交化的流程，直接得到三角矩阵$\boldsymbol{R}$。它们也面临着相同的数值困难：施密特正交化是逐向量正交化的，一旦遇到线性相关的向量或者零向量，那么正交化便无法进行下去，这在奇异值上体现为[有效秩](<https://spaces.ac.cn/archives/10847>)低或者[条件数](<https://en.wikipedia.org/wiki/Condition_number>)大，而在这种情况下Cholesky分解同样容易失败。

Cholesky分解的另一经典引用是正定对称矩阵求逆。比如解方程$\boldsymbol{B}\boldsymbol{X}=\boldsymbol{C}$，其中$\boldsymbol{B}$是正定对称的，那么我们可以先将它分解为$\boldsymbol{L}\boldsymbol{L}^{\top}$，然后变成$\boldsymbol{L}(\boldsymbol{L}^{\top}\boldsymbol{X})=\boldsymbol{C}$，这只需要解两次三角系数方程，每一步都比较高效。其中，[@Ji\_Ha\_Kim](<https://x.com/Ji_Ha_Kim/status/2039043040233275804>) 所提的分式迭代求$\newcommand{msign}{\mathop{\text{msign}}}\msign$便是基于这个思路来求的逆矩阵。

## 文章小结

这篇文章主要补充讨论了一下流式幂迭代的部分数学推导，包括幂迭代的收敛性、QR分解的预处理器、Cholesky分解的简单介绍等，希望能帮助大家从原理上更好地理解这个方法。
