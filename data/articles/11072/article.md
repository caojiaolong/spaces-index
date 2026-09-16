# “对角\+低秩”三角阵的高效求逆方法

> 作者：苏剑林 · 科学空间 · 2025-07-01
>
> 原文：<https://spaces.ac.cn/archives/11072>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从文章[《线性注意力简史：从模仿、创新到反哺》](<https://spaces.ac.cn/archives/11033>)我们可以发现，DeltaNet及其后的线性Attention模型，基本上都关联到了逆矩阵$(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-)^{-1}$。本文就专门来探讨一下这类具有“对角\+低秩”特点的三角矩阵的逆矩阵计算。

## 基本结果

我们将问题一般地定义如下：

> 给定矩阵$\boldsymbol{Q},\boldsymbol{K}\in\mathbb{R}^{n\times d}$和对角矩阵$\boldsymbol{\Lambda}\in\mathbb{R}^{n\times n}$，满足$n\gg d$，定义
> \begin{equation}\boldsymbol{T} = \boldsymbol{\Lambda} + \boldsymbol{Q}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-\end{equation}
> 其中$\boldsymbol{M}^-=\boldsymbol{M} - \boldsymbol{I}$，矩阵$\boldsymbol{M}$定义为
> \begin{equation}M_{i,j} = \left\{\begin{aligned} &1, &i \geq j \\ &0, &i < j\end{aligned}\right.\end{equation}
> 现在要求逆矩阵$\boldsymbol{T}^{-1}$，并且证明其复杂度是$\mathcal{O}(n^2)$。

首先，如果没有$\odot\boldsymbol{M}^-$的下三角阵约束，那么它可以直接由“[Woodbury恒等式](<https://en.wikipedia.org/wiki/Woodbury_matrix_identity>)”解决：
\begin{equation}(\boldsymbol{\Lambda} + \boldsymbol{Q}\boldsymbol{K}^{\top})^{-1} = \boldsymbol{\Lambda}^{-1} - \boldsymbol{\Lambda}^{-1} \boldsymbol{Q}(\boldsymbol{I} + \boldsymbol{K}^{\top}\boldsymbol{\Lambda}^{-1}\boldsymbol{Q})^{-1}\boldsymbol{K}^{\top}\boldsymbol{\Lambda}^{-1}\end{equation}
容易验证右端的计算复杂度是$\mathcal{O}(n^2)$的。然而，加上$\odot\boldsymbol{M}^-$后，$\boldsymbol{T}$本身就不再具有“对角\+低秩”的结构，因此不能直接由该恒等式解决了。针对下三角矩阵这一特点，一个基本的思路是递归，因为我们有分块矩阵恒等式
\begin{equation}\begin{bmatrix}\boldsymbol{A} & \boldsymbol{0} \\ \boldsymbol{C} & \boldsymbol{B}\end{bmatrix}^{-1} = \begin{bmatrix}\boldsymbol{A}^{-1} & \boldsymbol{0} \\ -\boldsymbol{B}^{-1}\boldsymbol{C}\boldsymbol{A}^{-1} & \boldsymbol{B}^{-1}\end{bmatrix}\end{equation}
这允许我们将$\boldsymbol{T}^{-1}$转化递归形式（约定：没有括号的情况下，切片的优先级最高）
\begin{equation}\boldsymbol{T}_{[:l+1,:l+1]}^{-1} = \begin{bmatrix}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{0} \\ -\boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{T}_{[l:l+1,:l]}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\end{bmatrix}\end{equation}
其中主要计算是$\boldsymbol{T}_{[l:l+1,:l]}\boldsymbol{T}_{[:l,:l]}^{-1}$，它是一个$1\times l$和$l\times l$矩阵相乘，复杂度是$\mathcal{O}(\mathcal{l^2})$，即每一步迭代的复杂度是平方增长的，所以总复杂度是$\mathcal{O}(n^3)$。

## 低秩结构

当然，这是因为我们还没用上$\boldsymbol{T}$（$\odot\boldsymbol{M}^-$前）的低秩结构，现在我们把它利用起来，那么将会得到$\boldsymbol{T}_{[l:l+1,:l]} = \boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}$，代入上式得：
\begin{equation}\boldsymbol{T}_{[:l+1,:l+1]}^{-1} = \begin{bmatrix}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{0} \\ -\boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\end{bmatrix}\end{equation}
注意$\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1}\in\mathbb{R}^{d\times l}$，如果我们能以它为递归变量，那么每一步迭代的复杂度就只是$\mathcal{O}(l)$，总复杂度就能成功降到$\mathcal{O}(n^2)$。根据这个思路，我们有
\begin{equation}\begin{aligned}
\boldsymbol{K}_{[:l+1]}^{\top}\boldsymbol{T}_{[:l+1,:l+1]}^{-1} =&\, \begin{bmatrix}\boldsymbol{K}_{[:l]}^{\top} & \boldsymbol{K}_{[l:l+1]}^{\top}\end{bmatrix}\begin{bmatrix}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{0} \\ -\boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\end{bmatrix} \\[6pt]
=&\, \begin{bmatrix}\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{0}\end{bmatrix} +  \boldsymbol{K}_{[l:l+1]}^{\top}\underbrace{\begin{bmatrix}-\boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\end{bmatrix}}_{\text{实际上就是 }(\boldsymbol{T}^{-1})_{[l:l+1,:l+1]}}\end{aligned}\end{equation}
可以看到这个递归过程也没有涉及到$\mathcal{O}(l^2)$的运算，因此思路是可行的，只需要引入一个新变量来缓存$\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1}$。如果我们将$l+1$换成$l+c$，那么就可以得到chunk格式的递归。

测试代码如下：



<pre><code class="language-python">import numpy as np

n, d, c = 1000, 100, 200
Q = np.random.randn(n, d) / d**0.5
K = np.random.randn(n, d) / d**0.5
T = np.tril(Q @ K.T, -1) + np.eye(n)

Y, Z = np.zeros((n, n)), np.zeros((d, n))
for l in range(0, n, c):
    Y[l:l + c, l:l + c] = np.linalg.inv(T[l:l + c, l:l + c])
    Y[l:l + c, :l] = - Y[l:l + c, l:l + c] @ Q[l:l + c] @ Z[:, :l]
    Z[:, :l + c] += K[l:l + c].T @ Y[l:l + c, :l + c]

np.allclose(Y @ T, np.eye(n))</code></pre>



## 乘法计算

基于同样的思路，我们还可以证明：

> 对于任意矩阵$\boldsymbol{V}\in\mathbb{R}^{n\times d}$，计算$\boldsymbol{T}^{-1}\boldsymbol{V}$只需要$\mathcal{O}(n)$的复杂度。

证明只需要把前述过程稍微改动一下。首先有
\begin{equation}\begin{aligned}
(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l+1]} =&\, \boldsymbol{T}_{[:l+1,:l+1]}^{-1}\boldsymbol{V}_{[:l+1]} \\[6pt]
=&\, \begin{bmatrix}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{0} \\ -\boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1} & \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\end{bmatrix}\begin{bmatrix}\boldsymbol{V}_{[:l]} \\ \boldsymbol{V}_{[l:l+1]}\end{bmatrix} \\[6pt]
=&\, \begin{bmatrix}\boldsymbol{T}_{[:l,:l]}^{-1}\boldsymbol{V}_{[:l]} \\ -\boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}\boldsymbol{T}_{[:l,:l]}^{-1}\boldsymbol{V}_{[:l]} + \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}\boldsymbol{V}_{[l:l+1]}\end{bmatrix} \\[6pt]
=&\, \begin{bmatrix}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l]} \\ \boldsymbol{T}_{[l:l+1,l:l+1]}^{-1}(\boldsymbol{V}_{[l:l+1]} - \boldsymbol{Q}_{[l:l+1]}\boldsymbol{K}_{[:l]}^{\top}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l]})\end{bmatrix}
\end{aligned}\end{equation}
然后
\begin{equation}\begin{aligned}
\boldsymbol{K}_{[:l+1]}^{\top}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l+1]} =&\, \begin{bmatrix}\boldsymbol{K}_{[:l]}^{\top} & \boldsymbol{K}_{[l:l+1]}^{\top}\end{bmatrix}\begin{bmatrix}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l]} \\ (\boldsymbol{T}^{-1}\boldsymbol{V})_{[l:l+1]} \end{bmatrix} \\[8pt]
=&\,\boldsymbol{K}_{[:l]}^{\top}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l]} + \boldsymbol{K}_{[l:l+1]}^{\top}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[l:l+1]}
\end{aligned}\end{equation}
因此，只需要缓存$\boldsymbol{K}_{[:l]}^{\top}(\boldsymbol{T}^{-1}\boldsymbol{V})_{[:l]}\in\mathbb{R}^{d\times d}$，就可以使得每步的计算复杂度与$l$无关，因此总复杂度是$\mathcal{O}(n)$。同样，只需要将$l+1$换成$l+c$就可以得到chunk格式。

测试代码如下：



<pre><code class="language-python">import numpy as np

n, d, c = 1000, 100, 200
Q = np.random.randn(n, d) / d**0.5
K = np.random.randn(n, d) / d**0.5
V = np.random.randn(n, d) / d**0.5
T = np.tril(Q @ K.T, -1) + np.eye(n)

Y, Z = np.zeros((n, d)), np.zeros((d, d))
for l in range(0, n, c):
    X = np.linalg.inv(T[l:l + c, l:l + c])
    Y[l:l + c] = X @ (V[l:l + c] - Q[l:l + c] @ Z)
    Z += K[l:l + c].T @ Y[l:l + c]

np.allclose(T @ Y, V)</code></pre>



## 文章小结

本文讨论了“对角\+低秩”特点的三角矩阵求逆问题，这类矩阵普遍出现在新式线性Attention模型中。
