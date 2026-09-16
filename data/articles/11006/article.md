# 通过msign来计算奇异值裁剪mclip（上）

> 作者：苏剑林 · 科学空间 · 2025-06-07
>
> 原文：<https://spaces.ac.cn/archives/11006>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前面我们用了两篇文章[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)和[《msign算子的Newton\-Schulz迭代（下）》](<https://spaces.ac.cn/archives/10996>)讨论了矩阵的$\newcommand{msign}{\mathop{\text{msign}}}\newcommand{sign}{\mathop{\text{sign}}}\newcommand{clip}{\mathop{\text{clip}}}\newcommand{mclip}{\mathop{\text{mclip}}}\msign$算子的数值计算，这篇文章我们来关注“奇异值裁剪（Singular Value Clipping）”运算，它最近在 [@\_arohan\_](<https://x.com/_arohan_/status/1929945590366122037>) 的推特上引起了热议，我们此前在[《高阶MuP：更简明但更高明的谱条件缩放》](<https://spaces.ac.cn/archives/10795>)也提到过，接下来我们简称为$\mclip$。

## 基本概念

对于标量$x$，$\clip$运算定义为
\begin{equation}\clip(x) = \max(\min(x, 1), -1) = \left\{\begin{aligned}1, &\quad x\geq 1 \\
x, &\quad x\in(-1, 1)\\
-1, &\quad x\leq -1
\end{aligned}\right.\end{equation}
即大于$1$或者小于$-1$就被截断，否则不变。我们将矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$的$\mclip$定义为
\begin{equation}\mclip(\boldsymbol{M}) = \boldsymbol{U}\clip(\boldsymbol{\Sigma})\boldsymbol{V}^{\top} \end{equation}
其中$\boldsymbol{M}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$是矩阵$\boldsymbol{M}$的SVD，$\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{V}\in\mathbb{R}^{m\times m}$是正交矩阵，$\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m}$是奇异值对角阵，对角矩阵加$\clip$表示对它的对角线元素分别进行$\clip$。留意到矩阵$\boldsymbol{\Sigma}$是对角阵并且总是非负的，所以我们也有
\begin{equation}\mclip(\boldsymbol{M}) = \boldsymbol{U}\min(\boldsymbol{\Sigma}, 1)\boldsymbol{V}^{\top} \end{equation}

SVD自然是计算$\mclip$的标准方式，但SVD的效率并不高。而有$\msign$的经验在前，我们不难想到可以像$\msign$一样给$\mclip$找一个Newton\-Schulz迭代来计算它。这个思路自然没有什么问题，但在$\msign$的基础上，我们还可以有更聪明的办法。

## 巨人肩膀

这个聪明的想法来自[@leloykun](<https://x.com/leloykun>)，他在博客[《Numerically Stable Spectral Clipping Via Newton\-Schulz Iteration》](<https://leloykun.github.io/ponder/spectral-clipping/>)提出可以站在$\msign$的肩膀上，用$\msign$来表示$\mclip$，这样就不用另外寻找Newton\-Schulz迭代了。他在博客中也提了一个巧妙的解法，但个人感觉不够直观并且效率也不算高，下面给出笔者的思路。

笔者的出发点是标量恒等式（Kimi帮忙找的）
$$\min(x, 1) = \frac{1}{2} [x + 1 - (x-1)\sign(x-1)] $$
简单起见，先假设$\boldsymbol{M}$是满秩方阵，那么
\begin{equation}\begin{aligned}
2\mclip(\boldsymbol{M}) =&\, \boldsymbol{U} [2\min(\boldsymbol{\Sigma},1)] \boldsymbol{V}^{\top} \\[6pt]
=&\, \boldsymbol{U} [\boldsymbol{\Sigma} + \boldsymbol{I} - (\boldsymbol{\Sigma} - \boldsymbol{I})\sign(\boldsymbol{\Sigma} - \boldsymbol{I})] \boldsymbol{V}^{\top} \\[6pt]
=&\, \boldsymbol{U} [\boldsymbol{\Sigma} + \boldsymbol{I} - (\boldsymbol{\Sigma} - \boldsymbol{I})\msign(\boldsymbol{\Sigma} - \boldsymbol{I})] \boldsymbol{V}^{\top} \\[6pt]
=&\, \boldsymbol{M} + \boldsymbol{U}\boldsymbol{V}^{\top} - \boldsymbol{U}(\boldsymbol{\Sigma} - \boldsymbol{I})\msign(\boldsymbol{\Sigma} - \boldsymbol{I}) \boldsymbol{V}^{\top}
\end{aligned}\label{eq:2-mclip-M}\end{equation}
注意
\begin{equation}\begin{aligned}
&\,\boldsymbol{U}(\boldsymbol{\Sigma} - \boldsymbol{I})\msign(\boldsymbol{\Sigma} - \boldsymbol{I}) \boldsymbol{V}^{\top} \\[6pt]
=&\, \boldsymbol{U}(\boldsymbol{\Sigma} - \boldsymbol{I}) \boldsymbol{U}^{\top} \boldsymbol{U}\msign(\boldsymbol{\Sigma} - \boldsymbol{I}) \boldsymbol{V}^{\top} \\[6pt]
=&\, (\boldsymbol{U}\boldsymbol{\Sigma} \boldsymbol{U}^{\top} - \boldsymbol{I}) \msign(\boldsymbol{M} - \boldsymbol{U}\boldsymbol{V}^{\top}) \\[6pt]
=&\, (\boldsymbol{U}\boldsymbol{\Sigma} \boldsymbol{V}^{\top} (\boldsymbol{U}\boldsymbol{V}^{\top})^{\top} - \boldsymbol{I}) \msign(\boldsymbol{M} - \boldsymbol{U}\boldsymbol{V}^{\top}) \\[6pt]
=&\, (\boldsymbol{M} (\boldsymbol{U}\boldsymbol{V}^{\top})^{\top} - \boldsymbol{I}) \msign(\boldsymbol{M} - \boldsymbol{U}\boldsymbol{V}^{\top}) \\[6pt]
\end{aligned}\end{equation}
其中第二个等号用到了对于任意正交矩阵$\boldsymbol{P},\boldsymbol{Q}$成立$\boldsymbol{P}\msign(\boldsymbol{R})\boldsymbol{Q} = \msign(\boldsymbol{P}\boldsymbol{R}\boldsymbol{Q})$。将上式代回式$\eqref{eq:2-mclip-M}$得到
\begin{equation}2\mclip(\boldsymbol{M}) = \boldsymbol{M} + \boldsymbol{U}\boldsymbol{V}^{\top} +  (\boldsymbol{I} - \boldsymbol{M}(\boldsymbol{U}\boldsymbol{V}^{\top})^{\top}) \msign(\boldsymbol{M} - \boldsymbol{U}\boldsymbol{V}^{\top})\label{eq:mclip-M-core}\end{equation}
若$\boldsymbol{M}$是一般的$r$秩矩阵，则$\boldsymbol{U}\boldsymbol{V}^{\top}$改为$\boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$，我们可以直接将$\boldsymbol{M} = \boldsymbol{U}_{[:,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$代入上式验证等号成立。

（注：这一节感谢 [@YouJiacheng](<https://x.com/YouJiacheng>) 的交流讨论。）

## 参考实现

我们知道$\boldsymbol{U}\boldsymbol{V}^{\top}=\msign(\boldsymbol{M})$，所以用式$\eqref{eq:mclip-M-core}$计算$\mclip$只需要算两次$\msign$：
\begin{equation}2\mclip(\boldsymbol{M}) = \boldsymbol{M} + \msign(\boldsymbol{M}) +  (\boldsymbol{I} - \boldsymbol{M}\msign(\boldsymbol{M})^{\top}) \msign(\boldsymbol{M} - \msign(\boldsymbol{M}))\end{equation}

计算量大致是$\msign$的2倍；相比之下[《Numerically Stable Spectral Clipping Via Newton\-Schulz Iteration》](<https://leloykun.github.io/ponder/spectral-clipping/>)需要对一个约4倍大小的矩阵算一次$\msign$，计算量大致是$\msign$的8倍。

在$\msign$的基础上，实现式$\eqref{eq:mclip-M-core}$最短只需要两行代码，参考如下：



<pre><code class="language-python">import numpy as np

def msign(m):
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    return u @ vh

def mclip(m):
    ms2 = msign(m - (ms := msign(m)))
    return (m + ms + ms2 - m @ ms.mT @ ms2) / 2

m = np.random.randn(10, 20)
u, s, vh = np.linalg.svd(m, full_matrices=False)

result1 = u @ np.diag(s.clip(0, 1)) @ vh
result2 = mclip(m)
np.abs(result1 - result2).mean()</code></pre>



这里直接用SVD来计算$\msign$，以便快速验证式$\eqref{eq:mclip-M-core}$的正确性，实际计算中，读者可以自行将$\msign$函数换成相应的Newton\-Schulz迭代。

## 其他函数

我们还可以用同样思路去计算其他函数的矩阵版，比如阶跃函数。我们定义标量的阶跃函数$\newcommand{mstep}{\mathop{\text{mstep}}}\newcommand{step}{\mathop{\text{step}}}$
\begin{equation}\step(x) = \frac{1}{2}[\sign(x - 1) + 1]\end{equation}
这表示大于1就变成1，小于1就变成0。于是我们可以定义
\begin{equation}\mstep(\boldsymbol{M}) = \boldsymbol{U}\step(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}\end{equation}
也就是只保留大于1的奇异值并截断为1，小于1的奇异值则直接置零。基于同样的步骤，我们可以得到
\begin{equation}\mstep(\boldsymbol{M}) = \frac{1}{2}[\msign(\boldsymbol{M}) + \msign(\boldsymbol{M} - \msign(\boldsymbol{M}))]\end{equation}
我们甚至可以表示偶函数，比如定义
\begin{equation}\mathop{\text{msquare}}(\boldsymbol{M}) = \boldsymbol{U} \boldsymbol{\Sigma}^2\boldsymbol{V}^{\top} = \boldsymbol{U}\boldsymbol{V}^{\top}(\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{U}^{\top})(\boldsymbol{U} \boldsymbol{\Sigma}\boldsymbol{V}^{\top}) = \msign(\boldsymbol{M})\boldsymbol{M}^{\top}\boldsymbol{M}\end{equation}
这跟直接由$\boldsymbol{M}^2$定义的矩阵平方不同，后者只对方阵有效，是在特征值分解下对特征值平方，而上式则是在奇异值分解下对奇异值平方。一般地，我们有
\begin{equation}\boldsymbol{U} \boldsymbol{\Sigma}^{2n}\boldsymbol{V}^{\top} = \msign(\boldsymbol{M})(\boldsymbol{M}^{\top}\boldsymbol{M})^n,\quad \boldsymbol{U} \boldsymbol{\Sigma}^{2n+1}\boldsymbol{V}^{\top} = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^n\end{equation}
这表明对于任意多项式$f(x)$（而不单单是奇多项式），$\boldsymbol{U}f(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}$都可以由$\boldsymbol{M}$和$\msign(\boldsymbol{M})$以及矩阵的有限步加乘得到。

## 文章小结

本文介绍了利用矩阵及其$\msign$，来对矩阵的奇异值进行一般运算的思路，包括奇异值裁剪、阶跃函数、任意次多项式（而不单单是奇多项式）等。
