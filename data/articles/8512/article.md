# 两个多元正态分布的KL散度、巴氏距离和W距离

> 作者：苏剑林 · 科学空间 · 2021-07-08
>
> 原文：<https://spaces.ac.cn/archives/8512>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

正态分布是最常见的连续型概率分布之一。它是给定均值和协方差后的最大熵分布（参考[《“熵”不起：从熵、最大熵原理到最大熵模型（二）》](<https://spaces.ac.cn/archives/3552>)），也可以看作任意连续型分布的二阶近似，它的地位就相当于一般函数的线性近似。从这个角度来看，正态分布算得上是最简单的连续型分布了。也正因为简单，所以对于很多估计量来说，它都能写出解析解来。

本文主要来计算两个多元正态分布的几种度量，包括KL散度、巴氏距离和W距离，它们都有显式解析解。

## 正态分布

这里简单回顾一下正态分布的一些基础知识。注意，仅仅是回顾，这还不足以作为正态分布的入门教程。

### 概率密度

正态分布，也即高斯分布，是定义在$\mathbb{R}^n$上的连续型概率分布，其概率密度函数为
\begin{equation}p(\boldsymbol{x})=\frac{1}{\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})}}\exp\left\{-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right\}\end{equation}
这里的$\boldsymbol{x},\boldsymbol{\mu}\in\mathbb{R}^n$，$\boldsymbol{\mu}$即均值向量（本文的向量默认情况下都为列向量），而$\boldsymbol{\Sigma}\in\mathbb{R}^{n\times n}$即为协方差矩阵，它要求是正定对称的。可以看到，正态分布由$\boldsymbol{\mu}$和$\boldsymbol{\Sigma}$唯一确定，因此不难想象它的统计量都是$\boldsymbol{\mu}$和$\boldsymbol{\Sigma}$的函数。当$\boldsymbol{\mu}=\boldsymbol{0}, \boldsymbol{\Sigma}=\boldsymbol{I}$时，对应的分布称为“标准正态分布”。

### 基本性质

通常来说，基本的统计量就是均值和方差了，它们对应着正态分布的两个参数：
\begin{equation}\begin{aligned}
\mathbb{E}_{\boldsymbol{x}}\left[\boldsymbol{x}\right]=&\int p(\boldsymbol{x}) \boldsymbol{x} dx=\boldsymbol{\mu}\\
\mathbb{E}_{\boldsymbol{x}}\left[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top}\right]=&\int p(\boldsymbol{x}) (\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top} dx=\boldsymbol{\Sigma}\\
\end{aligned}\end{equation}
由此也可以推出二阶矩的结果：
\begin{equation}
\mathbb{E}_{\boldsymbol{x}}\left[\boldsymbol{x}\boldsymbol{x}^{\top}\right]=\boldsymbol{\mu}\boldsymbol{\mu}^{\top} + \mathbb{E}_{\boldsymbol{x}}\left[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top}\right]=\boldsymbol{\mu}\boldsymbol{\mu}^{\top} + \boldsymbol{\Sigma}\end{equation}
还有一个常用的统计量是它的熵：
\begin{equation}\mathcal{H} = \mathbb{E}_{\boldsymbol{x}}\left[-\log p(\boldsymbol{x})\right]=\frac{n}{2}(1 + \log 2\pi) + \frac{1}{2}\log \det(\boldsymbol{\Sigma})
\end{equation}
其计算过程可以参考后面KL散度的推导。

### 高斯积分

概率密度函数意味着$\int p(\boldsymbol{x}) d\boldsymbol{x} = 1$，这就可以推出：
\begin{equation}\begin{aligned}
\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})} =& \int\exp\left\{-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right\}d\boldsymbol{x} \\
=& \int\exp\left\{-\frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{x}+\boldsymbol{\mu}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{x}-\frac{1}{2}\boldsymbol{\mu}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}\right\}d\boldsymbol{x}
\end{aligned}\end{equation}
设$\boldsymbol{\omega} = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}$，那么得到高斯积分
\begin{equation}
\int\exp\left\{-\frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{x}+\boldsymbol{\omega}^{\top}\boldsymbol{x}\right\}d\boldsymbol{x} = \sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})}\exp\left\{\frac{1}{2}\boldsymbol{\omega}^{\top}\boldsymbol{\Sigma}\boldsymbol{\omega}\right\}\label{eq:g-int}
\end{equation}
利用它我们可以算出正态分布的特征函数
\begin{equation}\mathbb{E}_{\boldsymbol{x}}\left[\exp\left(\boldsymbol{\omega}^{\top}\boldsymbol{x}\right)\right]=\exp\left(\boldsymbol{\omega}^{\top}\boldsymbol{\mu}+\frac{1}{2}\boldsymbol{\omega}^{\top}\boldsymbol{\Sigma}\boldsymbol{\omega}\right)\\
\end{equation}
特征函数可以用来算正态分布的各阶矩。

## 线性代数

这里补充一些线性代数基础，它们在后面的推导中会频繁用到。同样地，这仅仅是“回顾”，并不能作为线性代数教程。

### 内积范数

首先，我们来定义内积和范数。对于向量$\boldsymbol{x}=(x_1,\cdots,x_n)$和$\boldsymbol{y}=(y_1,\cdots,y_n)$，内积按照
\begin{equation}\langle\boldsymbol{x},\boldsymbol{y}\rangle = \sum_{i=1}^n x_i y_i\end{equation}
来定义，而模长定义为$\Vert \boldsymbol{x}\Vert = \sqrt{\langle\boldsymbol{x},\boldsymbol{x}\rangle}$。对于$m\times n$的矩阵$\boldsymbol{A}=(a_{i,j}),\boldsymbol{B}=(b_{i,j})$，我们按照类似的方式定义：
\begin{equation}\langle\boldsymbol{A},\boldsymbol{B}\rangle_F = \sum_{i=1}^m\sum_{j=1}^n a_{i,j} b_{i,j}\end{equation}
这称为Frobenius内积，对应的$\Vert \boldsymbol{A}\Vert_F = \sqrt{\langle\boldsymbol{A},\boldsymbol{A}\rangle_F}$称为Frobenius范数。不难看到，Frobenius内积和范数，事实上就是把矩阵展平为向量后，当作常规的向量来运算。

关于Frobenius内积，最关键的性质之一是成立恒等式
\begin{equation}\langle\boldsymbol{A},\boldsymbol{B}\rangle_F = \text{Tr}\left(\boldsymbol{A}^{\top}\boldsymbol{B}\right) = \text{Tr}\left(\boldsymbol{B}\boldsymbol{A}^{\top}\right) = \text{Tr}\left(\boldsymbol{A}\boldsymbol{B}^{\top}\right) = \text{Tr}\left(\boldsymbol{B}^{\top}\boldsymbol{A}\right) \end{equation}
也就是说，矩阵的Frobenius内积可以转化为矩阵乘法的迹，并且交换相乘顺序不改变结果（不改变迹的结果，但是矩阵乘法的整体结果会改变）。

### 正定对称

接着，来看正定对称矩阵的一些性质。$\boldsymbol{\Sigma}$是一个正定对称矩阵，对称说的是$\boldsymbol{\Sigma}^{\top}=\boldsymbol{\Sigma}$，正定说的是对于任意非零向量$\boldsymbol{\xi}\in\mathbb{R}^n$，都有$\boldsymbol{\xi}^{\top}\boldsymbol{\Sigma}\boldsymbol{\xi} > 0$。可以证明，如果$\boldsymbol{\Sigma}_1,\boldsymbol{\Sigma}_2$都是正定对称矩阵，那么$\boldsymbol{\Sigma}_1^{-1},\boldsymbol{\Sigma}_2^{-1},\boldsymbol{\Sigma}_1+\boldsymbol{\Sigma}_2$也都是正定对称矩阵。如果$\boldsymbol{C} = \boldsymbol{B}^{\top}\boldsymbol{A}\boldsymbol{B}$，$\boldsymbol{B}$是可逆阵，那么$\boldsymbol{C}$是正定对称的当且仅当$\boldsymbol{A}$是正定对称的。

此外还有半正定的概念，指对于任意非零向量$\boldsymbol{\xi}\in\mathbb{R}^n$，都有$\boldsymbol{\xi}^{\top}\boldsymbol{\Sigma}\boldsymbol{\xi} \geq 0$，也就是说可能存在非零向量$\boldsymbol{\xi}$使得$\boldsymbol{\xi}^{\top}\boldsymbol{\Sigma}\boldsymbol{\xi} = 0$。不过考虑到正定矩阵在半正定矩阵中稠密，所以我们不严格区分正定和半正定了，统一按照正定矩阵来处理。

正定对称矩阵有一个重要的性质，那就是它的SVD分解跟特征值分解一致，即具有下述形式的分解
\begin{equation}\boldsymbol{\Sigma} = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}\end{equation}
其中$\boldsymbol{U}$是正交矩阵，而$\boldsymbol{\Lambda}$是对角阵，并且对角线上的元素都是正的。该结果的一个直接推论是：正定对称矩阵都可以“开平方”，其平方根为$\boldsymbol{\Sigma}^{1/2} = \boldsymbol{U}\boldsymbol{\Lambda}^{1/2}\boldsymbol{U}^{\top}$，其中$\boldsymbol{\Lambda}^{1/2}$是指将对角线上的元素都开平方，可以检验平方根矩阵也是正定对称的。反过来，可以开平方的对称矩阵，一定也是正定对称矩阵。

### 矩阵求导

最后，在求Wasserstein距离的时候，还需要用到一些矩阵求导公式，如果不了解的读者，可以直接参考维基百科的“[Matrix Calculus](<https://en.wikipedia.org/wiki/Matrix_calculus>)”。当然，其实也不难，主要用到了
\begin{equation}\frac{\partial\,\text{Tr}\left(\boldsymbol{X}\boldsymbol{A}\right)}{\partial \boldsymbol{X}} = \boldsymbol{A}\end{equation}
剩下的可以结合迹的运算公式来派生出来，比如
\begin{equation}\frac{\partial\,\text{Tr}\left(\boldsymbol{A}\boldsymbol{X}\boldsymbol{B}\right)}{\partial \boldsymbol{X}} = \frac{\partial\,\text{Tr}\left(\boldsymbol{X}\boldsymbol{B}\boldsymbol{A}\right)}{\partial \boldsymbol{X}} = \boldsymbol{B}\boldsymbol{A}\end{equation}

## KL散度

作为第一个尝试，我们来算两个高斯分布的[KL散度（Kullback\-​Leibler divergence）](<https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>)。KL散度算是最常用的分布度量之一了，因为它积分之前需要取对数，这对于指数簇分布来说通常能得到相对简单的结果。此外它还跟“熵”有着比较紧密的联系。

### 计算结果

两个概率分布的KL散度定义为
\begin{equation}KL(p(\boldsymbol{x})\Vert q(\boldsymbol{x}))=\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right]=\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\log p(\boldsymbol{x})\right]+\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[-\log q(\boldsymbol{x})\right]\end{equation}
对于两个正态分布来说，计算结果是
\begin{equation}
KL(p(\boldsymbol{x})\Vert q(\boldsymbol{x}))=\frac{1}{2}\left[(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)-\log \det(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p) + \text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p\right) - n\right]
\end{equation}
特别地，当$q$是标准正态分布时，结果简化为
\begin{equation}
KL(p(\boldsymbol{x})\Vert q(\boldsymbol{x}))=\frac{1}{2}\left[\Vert\boldsymbol{\mu}_p\Vert^2-\log \det(\boldsymbol{\Sigma}_p) + \text{Tr}(\boldsymbol{\Sigma}_p) - n\right]\end{equation}

### 推导过程

从KL散度的定义知道，我们主要把$\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[-\log q(\boldsymbol{x})\right]$算出来就行了：
\begin{equation}\begin{aligned}
\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[-\log q(\boldsymbol{x})\right] =&\, \mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\frac{n}{2}\log 2\pi + \frac{1}{2}\log \det(\Sigma_q) + \frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right]\\
=&\,\frac{n}{2}\log 2\pi + \frac{1}{2}\log \det(\boldsymbol{\Sigma}_q) + \frac{1}{2}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right]
\end{aligned}\end{equation}
现在，关于迹的恒等式就可以派上用场了：
\begin{equation}\begin{aligned}
\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right]=&\,\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\text{Tr}\left((\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right)\right]\\
=&\,\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\right)\right]\\
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[(\boldsymbol{x}-\boldsymbol{\mu}_q)(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\right]\right)\\
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\left[\boldsymbol{x}\boldsymbol{x}^{\top}-\boldsymbol{\mu}_q\boldsymbol{x}^{\top} - \boldsymbol{x}\boldsymbol{\mu}_q^{\top} +  \boldsymbol{\mu}_q\boldsymbol{\mu}_q^{\top}\right]\right)\\
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\left(\boldsymbol{\Sigma}_p + \boldsymbol{\mu}_p\boldsymbol{\mu}_p^{\top}-\boldsymbol{\mu}_q\boldsymbol{\mu}_p^{\top} - \boldsymbol{\mu}_p\boldsymbol{\mu}_q^{\top} +  \boldsymbol{\mu}_q\boldsymbol{\mu}_q^{\top}\right)\right)\\
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\right)\\
=&\,\text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p\right) + (\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)\\
\end{aligned}\end{equation}
注意$\boldsymbol{\mu}_q=\boldsymbol{\mu}_p,\boldsymbol{\Sigma}_q=\boldsymbol{\Sigma}_p$时，上式就等于$n$，此时就对应正态分布的熵。所以最终得到
\begin{equation}\begin{aligned}
KL(p(\boldsymbol{x})\Vert q(\boldsymbol{x}))=&\,\frac{1}{2}\left[n\log 2\pi + \log \det(\boldsymbol{\Sigma}_q) + \text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p\right) + (\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)\right] \\
&\,\qquad- \frac{1}{2}\left[n\log 2\pi + \log \det(\boldsymbol{\Sigma}_p) + n\right]\\
=&\,\frac{1}{2}\left[(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)-\log \det(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p) + \text{Tr}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Sigma}_p\right) - n\right]
\end{aligned}\end{equation}

## 巴氏距离

然后，我们来看看[巴氏距离（Bhattacharyya distance）](<https://en.wikipedia.org/wiki/Bhattacharyya_distance>)，它定义为
\begin{equation}BD(p(\boldsymbol{x}), q(\boldsymbol{x})) = -\log \int \sqrt{p(\boldsymbol{x}) q(\boldsymbol{x})} d\boldsymbol{x}\end{equation}
与之相关的还有一个叫做“[Hellinger距离](<https://en.wikipedia.org/wiki/Hellinger_distance>)”的概念，它的平方定义为$\frac{1}{2}\int \left(\sqrt{p(\boldsymbol{x})} - \sqrt{q(\boldsymbol{x})}\right)^2 d\boldsymbol{x}$，展开后就能发现跟巴氏距离本质是等价的。

### 计算结果

对于两个正态分布来说，它们的巴氏距离为
\begin{equation}
BD(p(\boldsymbol{x}), q(\boldsymbol{x})) = \frac{1}{2}\log \frac{\det(\boldsymbol{\Sigma})}{\sqrt{\det(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)}} + \frac{1}{8}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)
\end{equation}
这里$\boldsymbol{\Sigma}=\frac{1}{2}(\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q)$。可以看到结果是对称的，这是因为巴氏距离的定义本身就是对称的。

当两者之一为标准正态分布时，结果并没有明显简化，所以这里就不单独写出来了。

### 推导过程

按照定义，两个正态分布的巴氏距离，是下述积分的负对数：
\begin{equation}\begin{aligned}
&\qquad\int \sqrt{p(\boldsymbol{x}) q(\boldsymbol{x})} d\boldsymbol{x}=\frac{1}{\sqrt[4]{(2\pi)^{2n}\det(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)}}\times \\
&\int \exp\left\{-\frac{1}{4}(\boldsymbol{x}-\boldsymbol{\mu}_p)^{\top}\boldsymbol{\Sigma}_p^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_p)-\frac{1}{4}(\boldsymbol{x}-\boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_q)\right\}d\boldsymbol{x}
\end{aligned}\end{equation}
记$\boldsymbol{y}=\boldsymbol{x}-\boldsymbol{\mu}_p, \boldsymbol{\Delta}=\boldsymbol{\mu}_p - \boldsymbol{\mu}_q$，积分部分可以换元为
\begin{equation}\begin{aligned}
&\int \exp\left\{-\frac{1}{4}\boldsymbol{y}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{y}-\frac{1}{4}(\boldsymbol{y}+\boldsymbol{\Delta})^{\top}\boldsymbol{\Sigma}_q^{-1}(\boldsymbol{y}+\boldsymbol{\Delta})\right\}d\boldsymbol{y}\\
=&\int \exp\left\{-\frac{1}{4}\boldsymbol{y}^{\top}\left(\boldsymbol{\Sigma}_p^{-1}+\boldsymbol{\Sigma}_q^{-1}\right)\boldsymbol{y}-\frac{1}{2}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}_q^{-1}\boldsymbol{y} - \frac{1}{4}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Delta}\right\}d\boldsymbol{y}\\
=&\int \exp\left\{-\frac{1}{2}\boldsymbol{y}^{\top}\left(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\Sigma}\boldsymbol{\Sigma}_q^{-1}\right)\boldsymbol{y}-\frac{1}{2}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}_q^{-1}\boldsymbol{y} - \frac{1}{4}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Delta}\right\}d\boldsymbol{y}\end{aligned}\end{equation}
这里$\boldsymbol{\Sigma}=\frac{1}{2}(\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q)$。按照前面介绍的高斯积分公式$\eqref{eq:g-int}$，积分结果是
\begin{equation}\begin{aligned}
&\,\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\Sigma}\boldsymbol{\Sigma}_q^{-1})^{-1}}\exp\left\{\frac{1}{8}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Delta}\right)^{\top}\left(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\Sigma}\boldsymbol{\Sigma}_q^{-1}\right)^{-1}\left(\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Delta}\right)-\frac{1}{4}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}_q^{-1}\boldsymbol{\Delta}\right\}\\
=&\,\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}_p)}\exp\left\{\frac{1}{8}\boldsymbol{\Delta}^{\top}\left(\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q^{-1} - 2\boldsymbol{\Sigma}_q^{-1}\right)\boldsymbol{\Delta}\right\}\\
=&\,\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}_p)}\exp\left\{\frac{1}{8}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q^{-1} - 2\boldsymbol{\Sigma}\boldsymbol{\Sigma}_q^{-1}\right)\boldsymbol{\Delta}\right\}\\
=&\,\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}_p)}\exp\left\{-\frac{1}{8}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Delta}\right\}
\end{aligned}\end{equation}
所以最终
\begin{equation}\begin{aligned}
BD(p(\boldsymbol{x}), q(\boldsymbol{x})) =&\, -\log \left[\frac{\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}_p)}}{\sqrt[4]{(2\pi)^{2n}\det(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)}}\exp\left\{-\frac{1}{8}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Delta}\right\}\right] \\
=&\, -\log \left[\frac{\sqrt[4]{\det(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)}}{\sqrt{\det\left(\boldsymbol{\Sigma}\right)}}\exp\left\{-\frac{1}{8}\boldsymbol{\Delta}^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Delta}\right\}\right]\\
=&\,\frac{1}{2}\log \frac{\det(\boldsymbol{\Sigma})}{\sqrt{\det(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)}} + \frac{1}{8}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^{\top}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)
\end{aligned}
\end{equation}

## W距离

如果读者还想看了解更多关于概率散度的内容，可以参考书籍[《Statistical Inference Based on Divergence Measures》](<https://www.taylorfrancis.com/books/mono/10.1201/9781420034813/statistical-inference-based-divergence-measures-leandro-pardo>)。现在我们转向另一类概率度量——基于最优传输的W距离（Wasserstein距离）。

沿用[《从Wasserstein距离、对偶理论到WGAN》](<https://spaces.ac.cn/archives/6280>)中的记号，W距离的定义如下：
\begin{equation}\begin{aligned}
\mathcal{W}_{\rho}[p,q]=&\,\left(\inf_{\gamma\in \Pi[p,q]} \iint \gamma(\boldsymbol{x},\boldsymbol{y}) \Vert\boldsymbol{x} - \boldsymbol{y}\Vert^{\rho} d\boldsymbol{x}d\boldsymbol{y}\right)^{1/\rho}\\
=&\,\left(\inf_{\gamma\in \Pi[p,q]} \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})} \left[\Vert\boldsymbol{x} - \boldsymbol{y}\Vert^{\rho}\right]\right)^{1/\rho}
\end{aligned}\end{equation}
根据笔者所找到的资料，对于两个正态分布的W距离，目前仅$\rho=2$时求出了解析解，因此下面都是在求$\mathcal{W}_2[p,q]$，并且简单起见，记
\begin{equation}\mathcal{W}_2^2[p,q] = \left(\mathcal{W}_2[p,q]\right)^2\end{equation}

### 计算结果

有意思的是，关于两个正态分布的W距离的结果，流传着两个不同的版本，这两个版本都有一定的认知度，但却没有看到有明确说两者等价的资料。两个版本出自不同的论文，还被冠以了不同的名字。

#### 版本1

首先是流传相对较广的版本，很多时候搜索“正态分布的Wasserstein距离”都是给出这个结果：
\begin{equation}\mathcal{W}_2^2[p,q]=\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \text{Tr}(\boldsymbol{\Sigma}_p) + \text{Tr}(\boldsymbol{\Sigma}_q) - 2\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2})\label{eq:w-v1}\end{equation}关于这个结果，有的读者可能困惑于“怎么关于$p,q$不是对称的”，事实上，它关于$p,q$是对称的，因为
\begin{equation}\begin{aligned}\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2})=&\,\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{\Sigma}_q^{-1/2}\boldsymbol{\Sigma}_q^{1/2}\boldsymbol{\Sigma}_p^{1/2})\\
=&\,\text{Tr}(\boldsymbol{\Sigma}_q^{1/2}\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{\Sigma}_q^{-1/2})
\end{aligned}\end{equation}
然后我们可以直接验证$(\boldsymbol{\Sigma}_q^{1/2}\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{\Sigma}_q^{-1/2})^2=\boldsymbol{\Sigma}_q^{1/2}\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q^{1/2}$，所以有$\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2})=\text{Tr}((\boldsymbol{\Sigma}_q^{1/2}\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q^{1/2})^{1/2})$。

#### 版本2

第二个看起来稍微简单一些的版本，结果是：
\begin{equation}\mathcal{W}_2^2[p,q]=\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \text{Tr}(\boldsymbol{\Sigma}_p) + \text{Tr}(\boldsymbol{\Sigma}_q) - 2\text{Tr}((\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)^{1/2})\label{eq:w-v2}\end{equation}
这个版本通常被称为“[Fréchet距离](<https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance>)”，一般情况下通过关键词“正态分布的Fréchet距离”才能搜到这个结果。GAN中经常使用的评价指标FID（[Frechet Inception Distance](<https://papers.cool/arxiv/1706.08500>)），就是基于这个公式进行计算的。可以模仿前面证明它关于$p,q$的对称性，当然也可以从下面的等价性讨论中直接得出。

#### 等价性

按道理，第2个版本简洁一些，应该以第二个版本为标准传播才对。所以为什么现在依然流传着两个不同的版本，笔者也相当困惑。从理论上来看，证明两个版本的等价性也不难，根据迹的恒等式我们有：
\begin{equation}\begin{aligned}\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2})=&\,\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{\Sigma}_p^{1/2})\\
=&\,\text{Tr}(\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}\boldsymbol{\Sigma}_p^{-1/2})
\end{aligned}\end{equation}
然后直接验证$(\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}\boldsymbol{\Sigma}_p^{-1/2})^2=\boldsymbol{\Sigma}_p \boldsymbol{\Sigma}_q$即可。

#### 特殊时

特别地，如果$\boldsymbol{\Sigma}_p,\boldsymbol{\Sigma}_q$的乘法可以交换，那么将会简化为非常直观的形式：
\begin{equation}\mathcal{W}_2^2[p,q]=\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \Vert \boldsymbol{\Sigma}_p^{1/2} - \boldsymbol{\Sigma}_q^{1/2}\Vert_F^2\label{eq:w-jiaohuan}\end{equation}
为什么说它非常直观呢？因为正态分布的参数为$\boldsymbol{\mu},\boldsymbol{\Sigma}$，所以比较正态分布的差异其实就是比较$\boldsymbol{\mu},\boldsymbol{\Sigma}$的差异，按照机器学习的习惯，一个很容易相当想到的指标是平方误差
\begin{equation}\mathcal{W}_2^2[p,q]=\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \Vert \boldsymbol{\Sigma}_p - \boldsymbol{\Sigma}_q\Vert_F^2\end{equation}
但从物理角度来看，这个指标是不妥的，因为如果将$\boldsymbol{\mu}$看成是长度量纲，那么$\boldsymbol{\Sigma}$就具有长度平方的量纲，所以$\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2$和$\Vert \boldsymbol{\Sigma}_p - \boldsymbol{\Sigma}_q\Vert_F^2$是具有不同量纲的两个量，不能相加。而为了使得量纲一致，直观的想法就是把$\boldsymbol{\Sigma}$“开平方”后再算平方误差，这就得到了式$\eqref{eq:w-jiaohuan}$。

特别地，当$q$为标准正态分布时，结果简化为
\begin{equation}\mathcal{W}_2^2[p,q]=\Vert \boldsymbol{\mu}_p\Vert^2 + \Vert \boldsymbol{\Sigma}_p^{1/2} - \boldsymbol{I}\Vert_F^2\end{equation}

### 推导过程1

现在介绍第一个证明，主要参考了论文[《A class of Wasserstein metrics for probability distributions》](<https://projecteuclid.org/journals/michigan-mathematical-journal/volume-31/issue-2/A-class-of-Wasserstein-metrics-for-probability-distributions/10.1307/mmj/1029003026.full>)。另外[《The distance between two random vectors with given dispersion matrices》](<https://www.sciencedirect.com/science/article/pii/0024379582901124>)也提供了一个类似的证明，也可以参考。

下面的推导过程则是经过笔者简化的，相对原论文的证明来说简单一些，但依然不可避免地会涉及到较多的线性代数知识，我们将分几个部分介绍。

#### 去均值

不失一般性，我们可以只考虑均值为0的分布$p,q$。因为如果$p,q$的均值不为0，那么设对应的均值为0的分布为$\tilde{p},\tilde{q}$，此时有
\begin{equation}\begin{aligned}
&\,\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})}\left[\Vert \boldsymbol{x} - \boldsymbol{y}\Vert^2\right] \\
=&\, \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\tilde{\gamma}(\boldsymbol{x},\boldsymbol{y})}\left[\Vert (\boldsymbol{x} + \boldsymbol{\mu}_p) - (\boldsymbol{y} + \boldsymbol{\mu}_q)\Vert^2  \right]\\
=&\,\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\tilde{\gamma}(\boldsymbol{x},\boldsymbol{y})}\left[\Vert \boldsymbol{x} - \boldsymbol{y}\Vert^2 + \Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + 2\langle\boldsymbol{x} - \boldsymbol{y}, \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\rangle\right]\\
=&\,\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\tilde{\gamma}(\boldsymbol{x},\boldsymbol{y})}\left[\Vert \boldsymbol{x} - \boldsymbol{y}\Vert^2  \right]
\end{aligned}\end{equation}
该结果意味着
\begin{equation}\mathcal{W}_2^2[p,q]=\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \mathcal{W}_2^2[\tilde{p},\tilde{q}]\end{equation}
所以，只需要算出均值都为零时的Wasserstein距离，然后加上$\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2$就得到了一般情况的结果。

#### 纯代数

现在我们假设$p,q$的均值均为0，然后计算
\begin{equation}\begin{aligned}
\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})}\left[\Vert \boldsymbol{x} - \boldsymbol{y}\Vert^2\right] =&\, \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})}\left[\boldsymbol{x}^{\top} \boldsymbol{x} + \boldsymbol{y}^{\top} \boldsymbol{y} - 2\boldsymbol{y}^{\top} \boldsymbol{x}\right]\\
=&\, \mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})}\left[\text{Tr}\left(\boldsymbol{x} \boldsymbol{x}^{\top} + \boldsymbol{y} \boldsymbol{y}^{\top} - 2\boldsymbol{x}\boldsymbol{y}^{\top} \right)\right]\\
=&\, \text{Tr}\left(\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})}\left[\boldsymbol{x} \boldsymbol{x}^{\top} + \boldsymbol{y} \boldsymbol{y}^{\top} - 2\boldsymbol{x}\boldsymbol{y}^{\top}\right]\right)\\
=&\, \text{Tr}(\boldsymbol{\Sigma}_p) + \text{Tr}(\boldsymbol{\Sigma}_q) - 2\text{Tr}(\boldsymbol{C})
\end{aligned}\end{equation}
其中
\begin{equation}\boldsymbol{\Sigma}_{\gamma}= \begin{pmatrix} \boldsymbol{\Sigma}_p & \boldsymbol{C}\\ \boldsymbol{C}^{\top} & \boldsymbol{\Sigma}_q\end{pmatrix}=\mathbb{E}_{(\boldsymbol{x},\boldsymbol{y})\sim\gamma(\boldsymbol{x},\boldsymbol{y})}\left[\begin{pmatrix}\boldsymbol{x} \\ \boldsymbol{y}\end{pmatrix}\begin{pmatrix}\boldsymbol{x}^{\top} & \boldsymbol{y}^{\top}\end{pmatrix}\right]\end{equation}
构成联合分布$\gamma$的协方差矩阵。我们知道协方差矩阵是正定对阵矩阵，所以从代数的角度看，问题变成了：

> 已知$\boldsymbol{\Sigma}_{\gamma}= \begin{pmatrix} \boldsymbol{\Sigma}_p & \boldsymbol{C}\\ \boldsymbol{C}^{\top} & \boldsymbol{\Sigma}_q\end{pmatrix}$为正定对称矩阵，求$\text{Tr}(\boldsymbol{C})$的最大值。

#### 舒尔补

为此，我们需要利用下述关于“舒尔补”的恒等式：
\begin{equation}
\begin{pmatrix} \boldsymbol{\Sigma}_p & \boldsymbol{C}\\ \boldsymbol{C}^{\top} & \boldsymbol{\Sigma}_q\end{pmatrix} = \begin{pmatrix} \boldsymbol{I} & \boldsymbol{0}\\ \boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1} & \boldsymbol{I}\end{pmatrix} \begin{pmatrix} \boldsymbol{\Sigma}_p & \boldsymbol{0}\\ \boldsymbol{0} & \boldsymbol{\Sigma}_q - \boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{C}\end{pmatrix} \begin{pmatrix} \boldsymbol{I} & \boldsymbol{\Sigma}_p^{-1}\boldsymbol{C} \\ \boldsymbol{0} & \boldsymbol{I}\end{pmatrix}
\end{equation}
其中对称矩阵$\boldsymbol{S} = \boldsymbol{\Sigma}_q - \boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{C}$称为“[舒尔补（Schur Complement）](<https://en.wikipedia.org/wiki/Schur_complement>)”，该分解具有$\boldsymbol{B}^{\top}\boldsymbol{A}\boldsymbol{B}$的形式，要想它是正定的，那么$\boldsymbol{A}$要是正定的，而$\boldsymbol{\Sigma}_p$已经是正定的，所以$\boldsymbol{S}$需要是正定的。

#### 分参数

尝试分离参数，即从$\boldsymbol{S} = \boldsymbol{\Sigma}_q - \boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{C}$中把$\boldsymbol{C}$解出来。首先移项得到$\boldsymbol{\Sigma}_q - \boldsymbol{S} = \boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{C}$，由于$\boldsymbol{\Sigma}_p$是正定对称的，所以$\boldsymbol{\Sigma}_p^{-1}$也是，从而$\boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{C}$也是正定对称的，那么它具有正定对称的平方根，即存在正定对称矩阵$\boldsymbol{R}$，使得
\begin{equation}\boldsymbol{C}^{\top}\boldsymbol{\Sigma}_p^{-1}\boldsymbol{C} = \boldsymbol{R}^2\quad\Leftrightarrow\quad \left(\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{C}\boldsymbol{R}^{-1}\right)^{\top}\left(\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{C}\boldsymbol{R}^{-1}\right)=\boldsymbol{I}\end{equation}
这说明$\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{C}\boldsymbol{R}^{-1}$是正交矩阵，记为$\boldsymbol{O}$，那么$\boldsymbol{C} = \boldsymbol{\Sigma}_p^{1/2}\boldsymbol{O}\boldsymbol{R}$。

#### 乘子法

此时，变量分别是$\boldsymbol{O}$和$\boldsymbol{R}$，求$\text{Tr}(\boldsymbol{C})=\text{Tr}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{O}\boldsymbol{R})$的最大值。我们先固定$\boldsymbol{R}$，求取最大值时的$\boldsymbol{O}$，此时相当于在$\boldsymbol{O}^{\top}\boldsymbol{O}=\boldsymbol{I}$的约束下，求$\text{Tr}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{O}\boldsymbol{R})$的最大值，我们用“[拉格朗日乘子法](<https://en.wikipedia.org/wiki/Lagrange_multiplier>)”：引入新参数矩阵$\boldsymbol{W}$，转化为下述无约束极值问题
\begin{equation}F = \text{Tr}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{O}\boldsymbol{R}) - \frac{1}{2}\text{Tr}(\boldsymbol{W}(\boldsymbol{O}^{\top}\boldsymbol{O} - \boldsymbol{I}))\end{equation}
求导：
\begin{equation}\begin{aligned}
&\frac{\partial F}{\partial \boldsymbol{O}} = \boldsymbol{0} \quad \Rightarrow\quad \boldsymbol{R}\boldsymbol{\Sigma}_p^{1/2} = \boldsymbol{W}\boldsymbol{O}^{\top}\\
&\frac{\partial F}{\partial \boldsymbol{W}} = \boldsymbol{0} \quad \Rightarrow\quad \boldsymbol{O}^{\top}\boldsymbol{O} = \boldsymbol{I}\\
\end{aligned}\end{equation}
首先留意到$\boldsymbol{O}^{\top}\boldsymbol{O} - \boldsymbol{I}$是对称的，因此对应的参数矩阵$\boldsymbol{W}$也是对称的，于是我们有：
\begin{equation}\left(\boldsymbol{O}\boldsymbol{W}\boldsymbol{O}^{\top}\right)^2=\left(\boldsymbol{W}\boldsymbol{O}^{\top}\right)^{\top}\left(\boldsymbol{W}\boldsymbol{O}^{\top}\right)=\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{R}^2\boldsymbol{\Sigma}_p^{1/2}\end{equation}
即$\boldsymbol{O}\boldsymbol{W}\boldsymbol{O}^{\top}=(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{R}^2\boldsymbol{\Sigma}_p^{1/2})^{1/2}$，所以此时
\begin{equation}\text{Tr}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{O}\boldsymbol{R})=\text{Tr}(\boldsymbol{O}\boldsymbol{R}\boldsymbol{\Sigma}_p^{1/2})=\text{Tr}(\boldsymbol{O}\boldsymbol{W}\boldsymbol{O}^{\top})=\text{Tr}((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{R}^2\boldsymbol{\Sigma}_p^{1/2})^{1/2})\end{equation}

#### 不等式

最后需要把$\boldsymbol{R}$确定下来。回顾$\boldsymbol{R}$的定义，我们有$\boldsymbol{R}^2=\boldsymbol{\Sigma}_q - \boldsymbol{S}$，其中$\boldsymbol{S}$是正定矩阵。直觉上$\boldsymbol{S}=\boldsymbol{0}$时取得最大值，事实上也确实如此，这算是“[Weyl不等式](<https://en.wikipedia.org/wiki/Weyl%27s_inequality>)”的一个直接推论。

根据Weyl不等式，如果矩阵$\boldsymbol{A},\boldsymbol{B}$都是正定对称矩阵，设$\boldsymbol{A},\boldsymbol{B},\boldsymbol{A}+\boldsymbol{B}$的特征值从小到大排列分别为$0\leq\lambda_1^{(A)} \leq \dots \leq \lambda_n^{(A)}$、$\lambda_1^{(B)} \leq \dots \leq \lambda_n^{(B)}$和$0\leq\lambda_1^{(A+B)} \leq \dots \leq \lambda_n^{(A+B)}$，那么对于任意$1\leq i \leq n$，都有$\lambda_i^{(A)}\leq \lambda_i^{(A+B)}$和$\lambda_i^{(B)}\leq \lambda_i^{(A+B)}$，也就是说：

> 正定对称矩阵的和的特征值，一一对应地大于它们各自的特征值。

有了这个结论，那就简单了，设$(\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_q - \boldsymbol{S})\boldsymbol{\Sigma}_p^{1/2})^{1/2}$的特征值为$0 \leq \lambda_1 \leq \dots \leq \lambda_n$，那么它的迹就是$\lambda_1 + \dots + \lambda_n$，对应地，$\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_q - \boldsymbol{S})\boldsymbol{\Sigma}_p^{1/2}$的特征值为$0 \leq \lambda_1^2 \leq \dots \leq \lambda_n^2$，注意$\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_q - \boldsymbol{S})\boldsymbol{\Sigma}_p^{1/2}$是正定对称矩阵（对称是显然的，而因为它能开平方，所以正定），$\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{S}\boldsymbol{\Sigma}_p^{1/2}$也是正定对称的（因为$\boldsymbol{S}$是正定对称的），所以它们的特征值，都不超过它们的和——也就是$\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2}$的特征值，所以说，$(\boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_q - \boldsymbol{S})\boldsymbol{\Sigma}_p^{1/2})^{1/2}$每个特征值的最大值（也就是迹的最大值），在$\boldsymbol{S}=\boldsymbol{0}$处取到。

至于Weyl不等式的证明，主要利用到了[Rayleigh quotient](<https://en.wikipedia.org/wiki/Rayleigh_quotient>)和[Courant–Fischer定理](<https://en.wikipedia.org/wiki/Min-max_theorem>)，有兴趣了解的读者自行查阅这两部分资料后，再查阅Wely不等式的证明就好。事实上，熟悉这两部分内容后，Weyl不等式基本上就“水到渠成”了。

### 推导过程2

这里继续介绍另一个更为简单的证明，原始证明可以在[《The Fréchet distance between multivariate normal distributions》](<https://core.ac.uk/download/pdf/82269844.pdf>)找到。相对于第一个证明而言，该证明更简单直接，尤其是不需要太多的纯线性代数知识。下面的推导过程依旧是经过笔者进一步简化的，比原始论文更好理解一些。

在这个推导过程中，“去均值”、“纯代数”两个步骤跟“推导过程1”是一样的，不再重复。所以，此时问题已经被转化为

> 已知$\boldsymbol{\Sigma}_{\gamma}= \begin{pmatrix} \boldsymbol{\Sigma}_p & \boldsymbol{C}\\ \boldsymbol{C}^{\top} & \boldsymbol{\Sigma}_q\end{pmatrix}$为正定对称矩阵，求$\text{Tr}(\boldsymbol{C})$的最大值。

#### 分块阵

由于$\boldsymbol{\Sigma}_{\gamma}$是正定对称矩阵，所以它必然可以表达成$\boldsymbol{D}\boldsymbol{D}^{\top}$的形式，我们将$\boldsymbol{D}$表达为分块矩阵$\begin{pmatrix}\boldsymbol{A} \\ \boldsymbol{B}\end{pmatrix}$，其中$\boldsymbol{A},\boldsymbol{B}\in \mathbb{R}^{n\times 2n}$，此时
\begin{equation}\begin{pmatrix} \boldsymbol{\Sigma}_p & \boldsymbol{C}\\ \boldsymbol{C}^{\top} & \boldsymbol{\Sigma}_q\end{pmatrix} = \begin{pmatrix}\boldsymbol{A} \\ \boldsymbol{B}\end{pmatrix} \begin{pmatrix}\boldsymbol{A}^{\top} & \boldsymbol{B}^{\top}\end{pmatrix} = \begin{pmatrix} \boldsymbol{A}\boldsymbol{A}^{\top} & \boldsymbol{A}\boldsymbol{B}^{\top}\\ \boldsymbol{B}\boldsymbol{A}^{\top} & \boldsymbol{B}\boldsymbol{B}^{\top}\end{pmatrix}\end{equation}
对应地有$\boldsymbol{\Sigma}_p=\boldsymbol{A}\boldsymbol{A}^{\top}, \boldsymbol{\Sigma}_q=\boldsymbol{B}\boldsymbol{B}^{\top},\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B}^{\top}$。

#### 乘子法

在上述参数化之下，问题转化为：

> 已知$\boldsymbol{A}\boldsymbol{A}^{\top}=\boldsymbol{\Sigma}_p, \boldsymbol{B}\boldsymbol{B}^{\top}=\boldsymbol{\Sigma}_q$，求$\text{Tr}(\boldsymbol{A}\boldsymbol{B}^{\top})$的最大值。

这是一个带约束的最大值问题，我们用“[拉格朗日乘子法](<https://en.wikipedia.org/wiki/Lagrange_multiplier>)”：引入新参数矩阵$\boldsymbol{W}_p, \boldsymbol{W}_q$，转化为下述无约束极值问题
\begin{equation}F = \text{Tr}(\boldsymbol{A}\boldsymbol{B}^{\top}) - \frac{1}{2}\text{Tr}(\boldsymbol{W}_p(\boldsymbol{A}\boldsymbol{A}^{\top} - \boldsymbol{\Sigma}_p)) - \frac{1}{2}\text{Tr}(\boldsymbol{W}_q(\boldsymbol{B}\boldsymbol{B}^{\top} - \boldsymbol{\Sigma}_q))\end{equation}
求导：
\begin{equation}\begin{aligned}
&\frac{\partial F}{\partial \boldsymbol{A}} = \boldsymbol{0} \quad \Rightarrow\quad \boldsymbol{B}^{\top} = \boldsymbol{A}^{\top}\boldsymbol{W}_p\\
&\frac{\partial F}{\partial \boldsymbol{B}} = \boldsymbol{0} \quad \Rightarrow\quad \boldsymbol{A}^{\top} = \boldsymbol{B}^{\top}\boldsymbol{W}_q\\
&\frac{\partial F}{\partial \boldsymbol{W}_p} = \boldsymbol{0} \quad \Rightarrow\quad \boldsymbol{A}\boldsymbol{A}^{\top} = \boldsymbol{\Sigma}_p\\
&\frac{\partial F}{\partial \boldsymbol{W}_q} = \boldsymbol{0} \quad \Rightarrow\quad \boldsymbol{B}\boldsymbol{B}^{\top} = \boldsymbol{\Sigma}_q\\
\end{aligned}\end{equation}
注意到$\boldsymbol{A}\boldsymbol{A}^{\top} - \boldsymbol{\Sigma}_p$和$\boldsymbol{B}\boldsymbol{B}^{\top} - \boldsymbol{\Sigma}_q$都是对称的，所以对应的参数矩阵$\boldsymbol{W}_p, \boldsymbol{W}_q$也是对称的，此时
\begin{equation}\boldsymbol{\Sigma}_q = \boldsymbol{B}\boldsymbol{B}^{\top} = \left(\boldsymbol{A}^{\top}\boldsymbol{W}_p\right)^{\top}\left(\boldsymbol{A}^{\top}\boldsymbol{W}_p\right)=\boldsymbol{W}_p\boldsymbol{A}\boldsymbol{A}^{\top}\boldsymbol{W}_p=\boldsymbol{W}_p\boldsymbol{\Sigma}_p\boldsymbol{W}_p\\
\end{equation}
令$\boldsymbol{W}_p=\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{R}\boldsymbol{\Sigma}_p^{-1/2}$，代入上式得$\boldsymbol{\Sigma}_q=\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{R}^2\boldsymbol{\Sigma}_p^{-1/2}$，即
\begin{equation}\boldsymbol{R} = (\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}
\end{equation}
而
\begin{equation}\begin{aligned}
\text{Tr}(\boldsymbol{A}\boldsymbol{B}^{\top}) =&\, \text{Tr}(\boldsymbol{A}\boldsymbol{A}^{\top}\boldsymbol{W}_p)=\text{Tr}(\boldsymbol{\Sigma}_p\boldsymbol{W}_p)\\
=&\,\text{Tr}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{R}\boldsymbol{\Sigma}_p^{-1/2}) = \text{Tr}(\boldsymbol{R}\boldsymbol{\Sigma}_p^{-1/2}\boldsymbol{\Sigma}_p^{1/2})\\
=&\,\text{Tr}(\boldsymbol{R})
\end{aligned}\end{equation}

## 文章小结

本文详细计算了两个多元正态分布的KL散度、巴氏距离和W距离，给出了它们的显式解析解，这些结果在某些场景下可以作为隐变量的正则项使用，来规范隐变量的分布。此外，本文还可以作为比较有挑战性的线性代数练习题，供大家参考练习。
