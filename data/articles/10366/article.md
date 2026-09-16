# 低秩近似之路（一）：伪逆

> 作者：苏剑林 · 科学空间 · 2024-09-15
>
> 原文：<https://spaces.ac.cn/archives/10366>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

可能很多读者跟笔者一样，对矩阵的低秩近似有种熟悉而又陌生的感觉。熟悉是因为，低秩近似的概念和意义都不难理解，加之目前诸如LoRA等基于低秩近似的微调技术遍地开花，让低秩近似的概念在耳濡目染间就已经深入人心；然而，低秩近似所覆盖的内容非常广，在低秩近似相关的论文中时常能看到一些不熟悉但又让我们叹为观止的新技巧，这就导致了一种似懂非懂的陌生感。

因此，在这个系列文章中，笔者将试图系统梳理一下矩阵低秩近似相关的理论内容，以补全对低秩近似的了解。而在第一篇文章中，我们主要介绍低秩近似系列中相对简单的一个概念——伪逆。

## 优化视角

伪逆（Pseudo Inverse），也称“广义逆（Generalized Inverse）”，顾名思义就是“广义的逆矩阵”，它实际上是“逆矩阵”的概念对于不可逆矩阵的推广。

我们知道，对于矩阵方程$\boldsymbol{A}\boldsymbol{B}=\boldsymbol{M}$，如果$\boldsymbol{A}$是方阵且可逆，那么直接得到$\boldsymbol{B}=\boldsymbol{A}^{-1}\boldsymbol{M}$，可如果$\boldsymbol{A}$不可逆或者干脆不是方阵呢？这种情况下我们很可能找不到$\boldsymbol{B}$满足$\boldsymbol{A}\boldsymbol{B}=\boldsymbol{M}$，这时候我们如果还想继续下去的话，通常是转为优化问题
\begin{equation}\mathop{\text{argmin}}_\boldsymbol{B} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{M}\Vert_F^2\label{eq:loss-ab-m}\end{equation}
其中$\boldsymbol{A}\in\mathbb{R}^{n\times r}, \boldsymbol{B}\in\mathbb{R}^{r\times m}, \boldsymbol{M}\in\mathbb{R}^{n\times m}$，注意数域是$\mathbb{R}$，表明本系列文章关注的都是实矩阵，而$\Vert\cdot\Vert_F$是矩阵的$F$范数（Frobenius Norm），用来衡量矩阵$\boldsymbol{A}\boldsymbol{B} - \boldsymbol{M}$与全零矩阵的距离，其定义为
\begin{equation}\Vert \boldsymbol{M}\Vert_F = \sqrt{\sum_{i=1}^n\sum_{j=1}^m M_{i,j}^2}\end{equation}
说白了，就是从求精确的逆矩阵改为最小化$\boldsymbol{A}\boldsymbol{B}$与$\boldsymbol{M}$的平方误差。本系列的主题是低秩近似，所以下面假设$r \ll \min(n,m)$，其机器学习意义就是通过低维的、有损的输入矩阵$\boldsymbol{A}$和线性变换$\boldsymbol{B}$来重建完整的$\boldsymbol{M}$矩阵。

当$m=n$且$\boldsymbol{M}$取单位阵$\boldsymbol{I}_n$时，我们就得到一个只依赖于$\boldsymbol{A}$的结果，记为
\begin{equation}\boldsymbol{A}^{\dagger} = \mathop{\text{argmin}}_\boldsymbol{B} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{I}_n\Vert_F^2\label{eq:loss-ab-m-b}\end{equation}
它的作用类似于$\boldsymbol{A}$的逆矩阵，所以称为$\boldsymbol{A}$的“(右)伪逆”。类似地，如果给定的是$\boldsymbol{B}$矩阵，那么我们也可以将优化参数改为$\boldsymbol{A}$，得到$\boldsymbol{B}$的“(左)伪逆”：
\begin{equation}\boldsymbol{B}^{\dagger} = \mathop{\text{argmin}}_\boldsymbol{A} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{I}_n\Vert_F^2\end{equation}

## 范数相关

在进一步推导之前，我们先补充一下$F$范数的相关介绍。向量的范数想必很多读者已经熟知，比较经典的就是“$p$\-范数”：对于$\boldsymbol{x}=(x_1,x_2,\cdots,x_m)$，其$p$\-范数定义为
\begin{equation}\Vert \boldsymbol{x}\Vert_p = \sqrt[\uproot{10}p]{\sum_{i=1}^m |x_i|^p}\end{equation}
而$p$\-范数中，最常见的就是$p=2$的情形，它就是我们经常说的向量模长，也叫“欧几里得范数”，如果省略范数的下标只写$\Vert \boldsymbol{x}\Vert$，那么基本上就是默认$p=2$。

矩阵的范数稍微复杂一些，它至少有两种不同但都常用的范数，其中一种就是上一节已经提到的$F$范数，它是直接将矩阵展平为向量来计算的范数：
\begin{equation}\Vert \boldsymbol{M}\Vert_F =  \Vert \text{vec}(\boldsymbol{M})\Vert_2 = \sqrt{\sum_{i=1}^n\sum_{j=1}^m M_{i,j}^2}\end{equation}
其他矩阵范数我们遇到时再作介绍。由于矩阵范数的多样性，所以$\Vert \boldsymbol{M}\Vert_F$的下标${}_F$通常不能省略，以免引起混淆。$F$范数是将矩阵当成向量然后照搬向量范数的定义而来的，由此启发我们可以尝试把更多的向量运算搬到矩阵中去，比如内积：
\begin{equation}\langle \boldsymbol{P}, \boldsymbol{Q} \rangle_F =  \langle \text{vec}(\boldsymbol{P}), \text{vec}(\boldsymbol{Q}) \rangle = \sum_{i=1}^n\sum_{j=1}^m P_{i,j} Q_{i,j}\end{equation}
这称为矩阵$\boldsymbol{P},\boldsymbol{Q}$的$F$内积（Frobenius Inner Product），其中$\boldsymbol{P},\boldsymbol{Q}\in\mathbb{R}^{n\times m}$，它可以用向量的迹运算来表示
\begin{equation}\langle \boldsymbol{P}, \boldsymbol{Q} \rangle_F =  \text{Tr}(\boldsymbol{P}^{\top} \boldsymbol{Q})\end{equation}
这可以直接由矩阵乘法和迹的定义来证明（请读者尝试一下）。当$\boldsymbol{P},\boldsymbol{Q}$是由多个矩阵连乘而来时，转换为等价的迹运算通常能帮助我们进行化简。比如，利用它我们可以证明正交变换不改变$F$范数：假设$\boldsymbol{U}$是一个正交矩阵，利用$\Vert \boldsymbol{M}\Vert_F^2 = \langle \boldsymbol{M}, \boldsymbol{M} \rangle_F$以及$F$内积与迹的关系，得到
\begin{equation}\Vert \boldsymbol{U}\boldsymbol{M}\Vert_F^2 = \text{Tr}((\boldsymbol{U}\boldsymbol{M})^{\top} \boldsymbol{U}\boldsymbol{M})=  \text{Tr}(\boldsymbol{M}^{\top} \boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{M})=\text{Tr}(\boldsymbol{M}^{\top} \boldsymbol{M}) = \Vert \boldsymbol{M}\Vert_F^2\end{equation}

## 矩阵求导

言归正传，对于一个优化目标，最理想的结果自然是能够通过求导来求出解析解，而$\eqref{eq:loss-ab-m}$正好能实现这一点！改结论可以简单“目测”出来：$\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M}$是关于$\boldsymbol{B}$的线性函数，所以$\Vert \boldsymbol{A}\boldsymbol{B}-\boldsymbol{M}\Vert_F^2$是关于$\boldsymbol{A}$或$\boldsymbol{B}$的二次函数，二次函数的最小值有解析解的。

要求$\mathcal{L}=\Vert \boldsymbol{A}\boldsymbol{B}-\boldsymbol{M}\Vert_F^2$关于$\boldsymbol{B}$的导数，首先要求$\mathcal{L}$关于$\boldsymbol{E}=\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M}$的导数，然后求$\boldsymbol{E}$关于$\boldsymbol{B}$的导数，最后通过链式法则组合起来，即
\begin{equation}\frac{\partial \mathcal{L}}{\partial B_{i,j}} = \sum_{k,l}\frac{\partial \mathcal{L}}{\partial E_{k,l}}\frac{\partial E_{k,l}}{\partial B_{i,j}} \end{equation}
根据定义$\mathcal{L}=\Vert \boldsymbol{E}\Vert_F^2 = \sum_{i,j} E_{i,j}^2$，很明显在求和的众多平方中只有当$(i,j)=(k,l)$时，关于$E_{k,l}$的导数才不为零，所以$\mathcal{L}$关于$E_{k,l}$的导数就是$E_{k,l}^2$关于$E_{k,l}$的导数，即$\frac{\partial \mathcal{L}}{\partial E_{k,l}}=2E_{k,l}$；接着，根据矩阵乘法的定义有
\begin{equation}E_{k,l} = \left(\sum_{\alpha} A_{k,\alpha}B_{\alpha,l}\right) - M_{k,l}\end{equation}
类似地，只有当$(\alpha,l)=(i,j)$时，上式对$B_{i,j}$的导数才会产生非零的结果$A_{k,i}$，所以我们可以写出$\frac{\partial E_{k,l}}{\partial B_{i,j}}=A_{k,i}\delta_{l,j}$，这里的$\delta_{l,j}$是Kronecker符号，用来声明$l=j$的条件。将结果组合起来，我们就得到
\begin{equation}\frac{\partial \mathcal{L}}{\partial B_{i,j}} = 2\sum_{k,l}E_{k,l}A_{k,i}\delta_{l,j} = 2\sum_k E_{k,j}A_{k,i}\end{equation}
如果我们约定，标量对矩阵的梯度形状跟矩阵本身一致，那么可以写出
\begin{equation}\frac{\partial \mathcal{L}}{\partial \boldsymbol{B}} = 2\boldsymbol{A}^{\top}(\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M})\end{equation}
虽然推导过程破费周折，但好在结果还是很直观的：直觉上$\frac{\partial \mathcal{L}}{\partial \boldsymbol{B}}$就是$2(\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M})$与$\boldsymbol{A}$的乘积（类比标量求导），而我们已经约定$\frac{\partial \mathcal{L}}{\partial \boldsymbol{B}}$形状跟$\boldsymbol{B}$形状一致（即$r\times m$），所以就要想办法通过$2(\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M})\in\mathbb{R}^{n\times m}$和$\boldsymbol{A}\in\mathbb{R}^{n\times r}$相乘来凑出一个$r\times m$的结果来，结果就只有上式右端一种方式。根据同样原理，我们可以快速写出
\begin{equation}\frac{\partial \mathcal{L}}{\partial \boldsymbol{A}} = 2(\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M})\boldsymbol{B}^{\top}\end{equation}

## 基本结果

现在我们已经分别求出了$\frac{\partial \mathcal{L}}{\partial \boldsymbol{B}}$和$\frac{\partial \mathcal{L}}{\partial \boldsymbol{A}}$，让它们等于零便可以解出相应的最优解
\begin{align}
2\boldsymbol{A}^{\top}(\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M})=0\quad\Rightarrow\quad (\boldsymbol{A}^{\top} \boldsymbol{A})^{-1}\boldsymbol{A}^{\top}\boldsymbol{M} = \mathop{\text{argmin}}_\boldsymbol{B} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{M}\Vert_F^2 \\
2(\boldsymbol{A}\boldsymbol{B}-\boldsymbol{M})\boldsymbol{B}^{\top}=0\quad\Rightarrow\quad \boldsymbol{M}\boldsymbol{B}^{\top}(\boldsymbol{B} \boldsymbol{B}^{\top})^{-1} = \mathop{\text{argmin}}_\boldsymbol{A} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{M}\Vert_F^2
\end{align}
代入$\boldsymbol{M}=\boldsymbol{I}_n$，就得到
\begin{align}\boldsymbol{A}^{\dagger} = (\boldsymbol{A}^{\top} \boldsymbol{A})^{-1}\boldsymbol{A}^{\top} \label{eq:p-inv-a}\\
\boldsymbol{B}^{\dagger} = \boldsymbol{B}^{\top}(\boldsymbol{B} \boldsymbol{B}^{\top})^{-1}\label{eq:p-inv-b}\end{align}
如果$\boldsymbol{A}$或$\boldsymbol{B}$是可逆方阵，那么容易证明伪逆就等于逆矩阵，即$\boldsymbol{A}^{\dagger}=\boldsymbol{A}^{-1},\boldsymbol{B}^{\dagger}=\boldsymbol{B}^{-1}$。此外，根据上式我们还可以验证：

> 1、$(\boldsymbol{A}^{\dagger})^{\dagger}=\boldsymbol{A},(\boldsymbol{B}^{\dagger})^{\dagger}=\boldsymbol{B}$，即<u>伪逆的伪逆</u>等于自身，这意味着伪逆在作为近似逆矩阵的同时，保全了自身的信息；
> 
> 
> 
> 2、$\boldsymbol{A}\boldsymbol{A}^{\dagger}\boldsymbol{A}=\boldsymbol{A},\boldsymbol{B}^{\dagger}\boldsymbol{B}\boldsymbol{B}^{\dagger}=\boldsymbol{B}^{\dagger}$，即$\boldsymbol{A}\boldsymbol{A}^{\dagger},\boldsymbol{B}^{\dagger}\boldsymbol{B}$虽然没法成为单位阵$\boldsymbol{I}$，但对$\boldsymbol{A},\boldsymbol{B}^{\dagger}$来说它们起到了单位阵的作用。

顺便说一下，矩阵的伪逆实际上是一个很宽泛的概念，它有很多种不同的形式，这里我们介绍的实际上是最常见的“[Moore–Penrose逆](<https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse>)”，除此之外还有“Drazin逆”、“Bott–Duffin逆”等，但这些笔者也不了解，所以就不作展开，读者可以自行参考维基百科的“[广义逆](<https://en.wikipedia.org/wiki/Generalized_inverse>)”条目。

## 一般形式

不过，事情还没完。式$\eqref{eq:p-inv-a},\eqref{eq:p-inv-b}$成立还有个关键前提是相应的$\boldsymbol{A}^{\top} \boldsymbol{A}$或$\boldsymbol{B} \boldsymbol{B}^{\top}$可逆，如果不可逆呢？

我们以$\boldsymbol{A}^{\dagger}$为例，假设$\boldsymbol{A}^{\top} \boldsymbol{A}$不可逆，那么意味着$\boldsymbol{A}$的秩不足$r$，我们只能从中找到$s < r$个列向量构成的极大线性无关组构成矩阵$\boldsymbol{A}_s\in\mathbb{R}^{n\times s}$，然后$\boldsymbol{A}$可以表示成$\boldsymbol{A}_s \boldsymbol{P}$，其中$\boldsymbol{P}\in\mathbb{R}^{s\times r}$是$\boldsymbol{A}_s$到$\boldsymbol{A}$的矩阵。此时
\begin{equation}\mathop{\text{argmin}}_\boldsymbol{B} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{I}_n\Vert_F^2 = \mathop{\text{argmin}}_\boldsymbol{B} \Vert \boldsymbol{A}_s \boldsymbol{P}\boldsymbol{B} - \boldsymbol{I}_n\Vert_F^2\end{equation}
如果$\boldsymbol{B}$的最优解仍然记为$\boldsymbol{A}^{\dagger}$，那么我们只能确定
\begin{equation}\boldsymbol{P}\boldsymbol{A}^{\dagger} = \boldsymbol{A}_s^{\dagger} = (\boldsymbol{A}_s^{\top} \boldsymbol{A}_s)^{-1}\boldsymbol{A}_s^{\top}\end{equation}
由于已经假设了$\boldsymbol{A}_s$是极大线性无关组，所以$\boldsymbol{A}_s^{\top} \boldsymbol{A}_s$必然可逆，因此上式是良好定义的。然而，从$\boldsymbol{A}^{\dagger}$到$\boldsymbol{P}\boldsymbol{A}^{\dagger}$是一个降维到过程，这意味着存在多个$\boldsymbol{A}^{\dagger}$使得$\boldsymbol{P}\boldsymbol{A}^{\dagger} = \boldsymbol{A}_s^{\dagger}$，即此时目标$\eqref{eq:loss-ab-m-b}$的最优解并不唯一，换言之当$\boldsymbol{A}^{\top} \boldsymbol{A}$不可逆时我们无法只凭目标$\eqref{eq:loss-ab-m-b}$来确定唯一的伪逆$\boldsymbol{A}^{\dagger}$。

一个可能的思路是补充$(\boldsymbol{A}^{\dagger})^{\dagger}=\boldsymbol{A}$或$\boldsymbol{A}^{\dagger}\boldsymbol{A}\boldsymbol{A}^{\dagger}=\boldsymbol{A}^{\dagger}$，这样结合$\boldsymbol{P}\boldsymbol{A}^{\dagger} = \boldsymbol{A}_s^{\dagger}$就可以唯一地确定$\boldsymbol{A}^{\dagger}$。然而，这样打补丁的味道太浓了，实际上我们可以用一个精妙的技巧更优雅和统一地处理这个问题。问题出在当$\boldsymbol{A}^{\top} \boldsymbol{A}$不可逆时，目标函数$\eqref{eq:loss-ab-m-b}$的最优解不唯一，我们可以加上一个正则项让它变得唯一，求出结果后再让正则项的权重趋于零：
\begin{equation}\boldsymbol{A}^{\dagger} = \lim_{\epsilon\to 0}\,\mathop{\text{argmin}}_\boldsymbol{B} \Vert \boldsymbol{A}\boldsymbol{B} - \boldsymbol{I}_n\Vert_F^2 + \epsilon\Vert \boldsymbol{B}\Vert_F^2\end{equation}
这里$\epsilon > 0$，$\epsilon\to 0$是指正向趋于零。由上式可以解得
\begin{equation}\boldsymbol{A}^{\dagger} = \lim_{\epsilon\to 0}\,(\boldsymbol{A}^{\top} \boldsymbol{A} + \epsilon \boldsymbol{I}_r)^{-1}\boldsymbol{A}^{\top}\label{eq:a-pinv-lim}\end{equation}
当$\epsilon > 0$时，$\boldsymbol{A}^{\top} \boldsymbol{A} + \epsilon \boldsymbol{I}_r$必然是可逆的（请读者证明一下），因此上式是良好定义的。由于$\epsilon\to 0$时，正则项可以忽略不计，所以上述极限必然是存在的。注意，这里说的是整体极限的存在性，当$\boldsymbol{A}^{\top} \boldsymbol{A}$不可逆时，极限$\lim\limits_{\epsilon\to 0}\,(\boldsymbol{A}^{\top} \boldsymbol{A} + \epsilon \boldsymbol{I}_r)^{-1}$是不存在的（结果会出现无穷大），只有乘上$\boldsymbol{A}^{\top}$后整体再取极限才是正常的。

式$\eqref{eq:a-pinv-lim}$作为伪逆的一般推广有什么优点呢？首先，我们已经有了$\boldsymbol{A}^{\top} \boldsymbol{A}$可逆时的$\boldsymbol{A}^{\dagger}$表达式$\eqref{eq:p-inv-a}$，式$\eqref{eq:a-pinv-lim}$作为它的推广，具有直观且形式一致的理论优雅性；其次，形式上的一致也使得$\boldsymbol{A}^{\top} \boldsymbol{A}$可逆时$\boldsymbol{A}^{\dagger}$的性质如$(\boldsymbol{A}^{\dagger})^{\dagger}$能够得以保留，从而让我们在讨论$\boldsymbol{A}^{\dagger}$时几乎可以完全不考虑$\boldsymbol{A}^{\top} \boldsymbol{A}$的可逆性。

## 数值计算

当然，目前的式$\eqref{eq:a-pinv-lim}$只是一个形式化的定义，如果直接利用它来数值计算的话，就必须取一个足够小的$\epsilon$然后把$(\boldsymbol{A}^{\top} \boldsymbol{A} + \epsilon \boldsymbol{I}_r)^{-1}$算出来，这样必然会面临严重的数值不稳定性。为了得到一个稳定的计算方式，我们利用实对称矩阵总可以正交对角化这一特点（[谱定理](<https://en.wikipedia.org/wiki/Spectral_theorem>)），对$\boldsymbol{A}^{\top} \boldsymbol{A}$作如下分解：
\begin{equation}\boldsymbol{A}^{\top} \boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Lambda} \boldsymbol{U}^{\top}\end{equation}
其中$\boldsymbol{U}$是正交矩阵，$\boldsymbol{\Lambda}=\text{diag}(\lambda_1,\lambda_2,\cdots,\lambda_r)$是特征值组成的对角矩阵，由于$\boldsymbol{A}^{\top} \boldsymbol{A}$的半正定性，它的特征值总是非负的。利用这个分解，我们有
\begin{equation}\begin{aligned}
(\boldsymbol{A}^{\top} \boldsymbol{A} + \epsilon \boldsymbol{I}_r)^{-1}\boldsymbol{A}^{\top} =&\, (\boldsymbol{U}\boldsymbol{\Lambda} \boldsymbol{U}^{\top} + \epsilon \boldsymbol{I}_r)^{-1} \boldsymbol{A}^{\top} \\
=&\, [\boldsymbol{U}(\boldsymbol{\Lambda} + \epsilon \boldsymbol{I}_r) \boldsymbol{U}^{\top}]^{-1}\boldsymbol{A}^{\top} \\
=&\, \boldsymbol{U}(\boldsymbol{\Lambda} + \epsilon \boldsymbol{I}_r)^{-1} \boldsymbol{U}^{\top}\boldsymbol{A}^{\top}
\end{aligned}\end{equation}
对于$(\boldsymbol{\Lambda} + \epsilon \boldsymbol{I}_r)^{-1}$我们有
\begin{equation}(\boldsymbol{\Lambda} + \epsilon \boldsymbol{I}_r)^{-1} = \text{diag}\Big((\lambda_1 + \epsilon)^{-1},(\lambda_2 + \epsilon)^{-1},\cdots,(\lambda_r + \epsilon)^{-1}\Big)\end{equation}
如果$\lambda_i > 0$，那么$\lim\limits_{\epsilon\to 0}\,(\lambda_i + \epsilon)^{-1}=\lambda_i^{-1}$，这是有限的结果，不妨碍计算，问题出现在$\lambda_i = 0$时$\lim\limits_{\epsilon\to 0}\,(\lambda_i + \epsilon)^{-1}=\lim\limits_{\epsilon\to 0}\,\epsilon^{-1}\to\infty$上。然而，我们知道$\epsilon\to 0$时正则项的影响就会消失，所以断定极限$\eqref{eq:a-pinv-lim}$必然不会出现无穷大值，因此如果存在$\lambda_i=0$，那么右端所乘的$\boldsymbol{U}^{\top}\boldsymbol{A}^{\top}$必然有办法抵消$\lim\limits_{\epsilon\to 0}\, \epsilon^{-1}$带来的无穷大。而能抵消这种无穷大的，唯有“乘以0”，即$\lim\limits_{\epsilon\to 0}\, \epsilon^{-1}\times 0 = 0$。

换句话说，如果$\lambda_i=0$，那么$\boldsymbol{U}^{\top}\boldsymbol{A}^{\top}$给$(\lambda_i+\epsilon)^{-1}$所乘的因子必然是$0$。既然如此，由于“0乘任何数都得0”，所以其实$\lambda_i=0$时$(\lambda_i+\epsilon)^{-1}$的取值反而不重要，我们可以简单地让它等于0。这样一来，我们就得到了一种通用的计算$\boldsymbol{A}^{\dagger}$的简单方法：
\begin{equation}\boldsymbol{A}^{\dagger} = \boldsymbol{U}\boldsymbol{\Lambda}^{\dagger}\boldsymbol{U}^{\top}\boldsymbol{A}^{\top}, \quad \boldsymbol{A}^{\top} \boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Lambda} \boldsymbol{U}^{\top}\end{equation}
其中$\boldsymbol{\Lambda}^{\dagger}$表示对角线上的元素如果等于零则不变，非零则取倒数。

可能有读者疑问，既然“0乘任何数都得0”，那么为什么等于零的$\lambda_i$要不变呢？随意取一个别的值可以吗？其实这里随便取一个值也不会影响结果的，但由于我们用了$\boldsymbol{\Lambda}^{\dagger}$这个记号，那么就要保持它跟式$\eqref{eq:a-pinv-lim}$的一致性，即它跟将对角阵$\boldsymbol{\Lambda}$代入式$\eqref{eq:a-pinv-lim}$的直接计算结果要一致
\begin{equation}\boldsymbol{\Lambda}^{\dagger} = \lim_{\epsilon\to 0}\,(\boldsymbol{\Lambda}^{\top} \boldsymbol{\Lambda} + \epsilon \boldsymbol{I}_r)^{-1}\boldsymbol{\Lambda}^{\top} = \text{diag}(\kappa_1,\kappa_2,\cdots,\kappa_n),\quad \kappa_i = \left\{\begin{aligned}\lambda_i^{-1}, \,\,\lambda_i\neq 0 \\ 0, \,\,\lambda_i=0 \end{aligned} \right. \end{equation}

## 文章小结

在这篇文章中，我们从低秩近似的角度介绍了伪逆，这是逆矩阵概念对于非方阵或不可逆方阵的扩展，使我们可以更有效地分析和求解一般的矩阵方程。
