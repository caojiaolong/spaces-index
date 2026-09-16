# 通过msign来计算奇异值裁剪mclip（下）

> 作者：苏剑林 · 科学空间 · 2025-06-23
>
> 原文：<https://spaces.ac.cn/archives/11059>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前面我们在[《通过msign来计算奇异值裁剪mclip（上）》](<https://spaces.ac.cn/archives/11006>)讨论了奇异值裁剪$\newcommand{mclip}{\mathop{\text{mclip}}}\mclip$的数值计算，核心思路来自 [@leloykun](<https://x.com/leloykun>) 的文章[《Numerically Stable Spectral Clipping Via Newton\-Schulz Iteration》](<https://leloykun.github.io/ponder/spectral-clipping/>)（现已重新修订和改名），通过寻找基于$\newcommand{msign}{\mathop{\text{msign}}}\msign$的表达式来避免另外寻找Newton\-Schulz迭代，在文章中笔者提出了一个计算量更低的嵌套$\msign$方案。

不过前两天，@leloykun 在[推特](<https://x.com/leloykun/status/1936199820479205420>)上指出笔者的方案实际计算中存在误差偏大的问题。本文来具体分析一下这个问题，并给出一个更高效、误差更低的新方案。

## 基本概念

按照惯例，先整理一下基本概念。首先是标量$x$的$\newcommand{clip}{\mathop{\text{clip}}}\clip$算子，这次我们一般地定义
\begin{equation}\clip\nolimits_{[\alpha,\beta]}(x) = \max(\min(x, \beta), \alpha) = \left\{\begin{aligned}\beta, &\quad \geq \beta \\
x, &\quad x\in(\alpha, \beta)\\
\alpha, &\quad x\leq \alpha
\end{aligned}\right.\end{equation}
当没有特别注明区间时，区间默认是$[-1,1]$，即$\clip(x) = \clip_{[-1,1]}(x)$。设矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$的SVD为$\boldsymbol{M}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，$\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{V}\in\mathbb{R}^{m\times m}$是正交矩阵，$\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m}$是奇异值对角阵，那么定义
\begin{equation}\mclip\nolimits_{[\alpha,\beta]}(\boldsymbol{M}) = \boldsymbol{U}\clip\nolimits_{[\alpha,\beta]}(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}\end{equation}
对角矩阵加$\clip$表示对它的对角线元素分别进行$\clip$，说白了，$\mclip_{[\alpha,\beta]}$就是把$\boldsymbol{M}$的奇异值裁剪到$[\alpha,\beta]$内。

由于奇异值是非负的，所以当$\alpha < 0$时有$\mclip_{[\alpha,\beta]}(\boldsymbol{M})=\mclip_{[0,\beta]}(\boldsymbol{M})$，但后面我们将会看到，由于实际计算的误差，考虑负数的$\alpha$会有一些神奇的抵消误差效果。

## 理论通解

这一节的目标是用$\msign$表示出$\mclip$，出发点是恒等式
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}}\mclip\nolimits_{[\alpha,\beta]} (x) =  \frac{\alpha + \beta + (\alpha - x)\sign(\alpha - x) - (\beta - x)\sign(\beta - x)}{2}\end{equation}
找到恒等式的关键是将$\clip$表示为绝对值与自身的线性运算，然后通过$|x|=x\sign(x)$过渡到$\sign$运算，这里就不展开了。

简单起见，先设$\boldsymbol{M}$是满秩方阵，基于该恒等式，我们有
\begin{equation}2\mclip\nolimits_{[\alpha,\beta]}(\boldsymbol{M}) = \boldsymbol{U}\Big((\alpha + \beta)\boldsymbol{I} + (\alpha \boldsymbol{I} - \boldsymbol{\Sigma})\sign(\alpha \boldsymbol{I} - \boldsymbol{\Sigma}) - (\beta \boldsymbol{I} - \boldsymbol{\Sigma})\sign(\beta \boldsymbol{I} - \boldsymbol{\Sigma})\Big)\boldsymbol{V}^{\top}\end{equation}
展开右式，分别包含几种项（$\gamma\in\{\alpha,\beta\}$）：
\begin{array}{c|c}
\hline
\text{原始} & \text{化简} \\
\hline
\boldsymbol{U}\boldsymbol{V}^{\top}  & \msign(\boldsymbol{M}) \\
\hline
\boldsymbol{U}\sign(\gamma \boldsymbol{I} - \boldsymbol{\Sigma})\boldsymbol{V}^{\top} &
\begin{aligned}&\, \msign(\gamma \boldsymbol{U}\boldsymbol{V}^{\top} - \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}) \\
=&\, \msign(\gamma \msign(\boldsymbol{M}) - \boldsymbol{M})
\end{aligned} \\
\hline
\boldsymbol{U}\boldsymbol{\Sigma}\sign(\gamma \boldsymbol{I} - \boldsymbol{\Sigma})\boldsymbol{V}^{\top} & \begin{aligned}&\, \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\boldsymbol{V}\boldsymbol{U}^{\top}\boldsymbol{U}\sign(\gamma \boldsymbol{I} - \boldsymbol{\Sigma})\boldsymbol{V}^{\top} \\
=&\, \boldsymbol{M}\msign(\boldsymbol{M})^{\top}\msign(\gamma \msign(\boldsymbol{M}) - \boldsymbol{M})
\end{aligned} \\
\hline
\end{array}
代入整理可得
\begin{equation}\mclip\nolimits_{[\alpha,\beta]}(\boldsymbol{M}) = \frac{1}{2}\left\{\begin{aligned}&\,(\alpha + \beta)\msign(\boldsymbol{M}) \\
+ &\, (\alpha \boldsymbol{I} - \boldsymbol{M}\msign(\boldsymbol{M})^{\top})\msign(\alpha \msign(\boldsymbol{M}) - \boldsymbol{M})\\
- &\, (\beta \boldsymbol{I} - \boldsymbol{M}\msign(\boldsymbol{M})^{\top})\msign(\beta \msign(\boldsymbol{M}) - \boldsymbol{M})
\end{aligned}\right\}\label{eq:general}\end{equation}
对于非方形、非满秩矩阵，可以代入$\msign(\boldsymbol{M})=\boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}$到上式检验成立，所以上式是$\mclip$的理论通解。

## 初始形式

式$\eqref{eq:general}$看起来至少需要计算三次$\msign$，并且后面两次$\msign$的输入带有第一次$\msign$的结果，所以形式上是$\msign$的嵌套。当我们取$\alpha=0,\beta=1$时，$\msign$的次数可以降低到两次：
\begin{equation}\mclip(\boldsymbol{M}) = \frac{1}{2}\Big[\boldsymbol{M} + \msign(\boldsymbol{M}) +  (\boldsymbol{I} - \boldsymbol{M}\msign(\boldsymbol{M})^{\top}) \msign(\boldsymbol{M} - \msign(\boldsymbol{M}))\Big]\label{eq:mclip-1}\end{equation}
这就是笔者在上一篇文章[《通过msign来计算奇异值裁剪mclip（上）》](<https://spaces.ac.cn/archives/11006>)给出的结果，只需要两次$\msign$。

然而，实测显示该式在$\boldsymbol{M}$的奇异值较大且$\msign$的计算精度较低时，会产生较大的误差，远大于 @leloykun 所给出的方案。但 @leloykun 的方案需要对一个大约4倍大小的矩阵$\begin{bmatrix}\boldsymbol{I} & \boldsymbol{M} \\ \boldsymbol{M}^{\top} & \boldsymbol{I}\end{bmatrix}$算$\msign$，代价不菲，所以还是想看看这里的方案还有什么提升空间。

## 去掉嵌套

直觉上，误差的来源是嵌套$\msign$导致的累积误差，所以尝试想办法去掉嵌套，幸运的是，利用一个简单的技巧还真的能去掉嵌套！

首先可以证明
\begin{equation}\begin{aligned}
&\,(\boldsymbol{I} - \boldsymbol{M}\msign(\boldsymbol{M})^{\top}) \msign(\boldsymbol{M} - \msign(\boldsymbol{M})) \\[6pt]
=&\, (\msign(\boldsymbol{M}) - \boldsymbol{M}) \msign(\msign(\boldsymbol{M})^{\top}\boldsymbol{M} - \boldsymbol{I})
\end{aligned}\end{equation}
然后我们有
\begin{equation}\msign(\boldsymbol{M})^{\top}\boldsymbol{M} - \boldsymbol{I} = \boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} - \boldsymbol{I} = \boldsymbol{V}(\boldsymbol{\Sigma}-\boldsymbol{I})\boldsymbol{V}^{\top}\end{equation}
根据上式，我们断言
\begin{equation}\msign(\msign(\boldsymbol{M})^{\top}\boldsymbol{M} - \boldsymbol{I}) = \msign(\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I}) = \msign(\boldsymbol{V}(\boldsymbol{\Sigma}^2-\boldsymbol{I})\boldsymbol{V}^{\top})\end{equation}
这利用了一个很简单的性质：$\forall x \geq 0, \sign(x-1) = \sign(x^2-1)$。利用该结果，可以得到
\begin{equation}\mclip(\boldsymbol{M}) = \frac{1}{2}\Big[\boldsymbol{M} + \msign(\boldsymbol{M}) +  (\msign(\boldsymbol{M}) - \boldsymbol{M}) \msign(\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I})\Big]\label{eq:mclip-2}\end{equation}
还是两次$\msign$，但它们之间已经不再有嵌套关系，意味着理论上已经没有嵌套$\msign$带来的累积误差，实测显示式$\eqref{eq:mclip-2}$的误差确实能比式$\eqref{eq:mclip-1}$小一半左右，但极端情况下还是不如 @leloykun 的方案，这说明嵌套并不是主要误差来源。

## 相互抵消

还有什么改进空间呢？@leloykun 的方案要求是奇函数，所以它实际上考虑的是$\mclip_{[-1,1]}$而不是$\mclip_{[0,1]}$。有没有可能是这个选择导致了它某两部分误差相互抵销，从而得到更好的计算精度呢？

为了验证这一点，我们在式$\eqref{eq:general}$代入$\alpha=-1,\beta=1$，得到
\begin{equation}\mclip(\boldsymbol{M}) = \frac{1}{2}\left\{\begin{aligned}
&\,(\boldsymbol{I} + \boldsymbol{M}\msign(\boldsymbol{M})^{\top})\msign(\msign(\boldsymbol{M}) + \boldsymbol{M}) \\
- &\,(\boldsymbol{I} - \boldsymbol{M}\msign(\boldsymbol{M})^{\top})\msign(\msign(\boldsymbol{M}) - \boldsymbol{M})
\end{aligned}\right\}\end{equation}
基于上一节一样的去嵌套技巧，我们得到
\begin{equation}\mclip(\boldsymbol{M}) = \frac{1}{2}\left\{\begin{aligned}
&\,(\msign(\boldsymbol{M}) + \boldsymbol{M})\msign(\boldsymbol{M}^{\top}\boldsymbol{M} + \boldsymbol{I}) \\
+ &\,(\msign(\boldsymbol{M}) - \boldsymbol{M})\msign(\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I})
\end{aligned}\right\}\label{eq:mclip-3}\end{equation}
注意，$\boldsymbol{M}^{\top}\boldsymbol{M} + \boldsymbol{I}$一定是正定对称矩阵，所以理论上$\msign(\boldsymbol{M}^{\top}\boldsymbol{M} + \boldsymbol{I})=\boldsymbol{I}$，这样我们就恢复了式$\eqref{eq:mclip-2}$。但实际计算中，$\msign(\boldsymbol{M}^{\top}\boldsymbol{M} + \boldsymbol{I})$与$\boldsymbol{I}$之间的误差可能会抵消$\msign(\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I})$带来的误差，所以我们通过实验决定是否保留它。

不出所料，式$\eqref{eq:mclip-3}$的数值误差比 @leloykun 的方案还要小！这就肯定了我们的猜测，设置$\alpha=-1$和$\beta=1$让$\mclip$变成奇函数，有助于抵消误差。

## 原因浅思

为什么这么巧能抵消误差呢？我们可以简单做个定量分析。大误差的出现前提有两个，一是$\boldsymbol{M}$有非常大的奇异值，二是$\msign$的迭代步数并不多，导致$\msign$本身的精度不高。

我们观察式$\eqref{eq:mclip-3}$，它可以拆分为4项求和，其实$\msign(\boldsymbol{M})\msign(\boldsymbol{M}^{\top}\boldsymbol{M} \pm \boldsymbol{I})$这两项有界的，即便$\msign$精度不高也基本无法发散，所以主要误差来自
\begin{equation}\boldsymbol{M}\msign(\boldsymbol{M}^{\top}\boldsymbol{M} + \boldsymbol{I}) - \boldsymbol{M}\msign(\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I})\label{eq:error-1}\end{equation}
它正比于$\boldsymbol{M}$，最有可能把误差放大。相应地，式$\eqref{eq:mclip-2}$的主要误差项则是
\begin{equation}\boldsymbol{M} - \boldsymbol{M}\msign(\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I})\label{eq:error-2}\end{equation}
我们考虑远大于1的奇异值，如果$\msign$是精确的，那么$\msign$的结果就是1，上面两个式子的结果中对应大奇异值部分将会都是我们期望的0。

然而，如果是迭代步数不多的$\msign$，它可能变成$0.6$或者$1.4$这样的值，式$\eqref{eq:error-2}$相应的部分就会出现$\sim\pm 0.4 \boldsymbol{M}$这样的巨大误差；但如果是式$\eqref{eq:error-1}$，当奇异值很大时，$\boldsymbol{M}^{\top}\boldsymbol{M} - \boldsymbol{I}$和$\boldsymbol{M}^{\top}\boldsymbol{M} + \boldsymbol{I}$的相对差异并不大，因此$\msign(\boldsymbol{M}^{\top}\boldsymbol{M} \pm \boldsymbol{I})$的差异很小，所以式$\eqref{eq:error-1}$依然能抵消大部份误差。

但要记住，这始终有个前提，就是$\boldsymbol{M}$有明显大于1的奇异值，以及迭代步数不多。如果不满足这两个条件，那么式$\eqref{eq:mclip-2}$本来的误差就不大，式$\eqref{eq:mclip-3}$反而会因为多算了一次$\msign$而增加误差。因此，哪个公式的实际表现最优，还需要具体情况具体分析。

## 对比代码

构造一个奇异值有大于1也有小于1，且最大奇异值接近1000的奇异值，然后在bfloat16精度下测试各个算法，参考代码如下（大致运行结果已在注释写出）：



<pre><code class="language-python">import numpy as np
import jax.numpy as jnp
import jax.lax as lax

def msign(x, steps=4, eps=1e-20):
    """The coefficients come from https://kexue.fm/archives/10996
    """
    abc = [
        (8.287212018145622, -23.59588651909882, 17.300387312530923),
        (4.107059111542197, -2.9478499167379084, 0.54484310829266),
        (3.9486908534822938, -2.908902115962947, 0.5518191394370131),
        (3.3184196573706055, -2.488488024314878, 0.5100489401237208),
        (2.3006520199548186, -1.6689039845747518, 0.4188073119525678),
        (1.8913014077874002, -1.2679958271945908, 0.37680408948524996),
        (1.875, -1.25, 0.375)
    ]
    y = x.mT if x.shape[-2] &gt; x.shape[-1] else x
    y = y * lax.rsqrt((y**2).sum(axis=[-2, -1], keepdims=True) + eps)
    for a, b, c in abc[:steps] + max(steps - 7, 0) * abc[-1:]:
        a, b, c = a / 1.01, b / 1.01**3, c / 1.01**5
        y = a * y + (b * (u := y @ y.mT) + c * u @ u) @ y
    return y.mT if x.shape[-2] &gt; x.shape[-1] else y

def mclip1(m):
    """1st version (2 nested msign)
    """
    ms2 = msign(m - (ms1 := msign(m)))
    return (m + ms1 + ms2 - m @ ms1.mT @ ms2) / 2

def mclip2(m):
    """2nd version (2 non-nested msign)
    """
    ms1 = msign(m)
    ms2 = msign(m.mT @ m - jnp.eye(m.shape[-1]))
    return (m + ms1 + (ms1 - m) @ ms2) / 2

def mclip3(m):
    """3rd version (3 non-nested msign)
    """
    ms1 = msign(m)
    ms2 = msign(m.mT @ m + jnp.eye(m.shape[-1]))
    ms3 = msign(m.mT @ m - jnp.eye(m.shape[-1]))
    return ((ms1 + m) @ ms2  + (ms1 - m) @ ms3) / 2

def spectral_clip(W):
    """@leloykun verision: https://leloykun.github.io/ponder/spectral-clipping/
    """
    m, n = W.shape
    H = jnp.block([[jnp.eye(m), W], [W.T, jnp.eye(n)]])
    OH = msign(H)
    P, Q = OH[:m, :m], OH[:m, m:]
    return Q + P @ W

m = np.random.randn(4096, 1024)
u, s, vh = jnp.linalg.svd(m, full_matrices=False)
s = np.concatenate([np.linspace(1, 1000, 128), np.linspace(0, 1, 896)])
s = np.sort(s)[::-1]
m = u @ jnp.diag(s) @ vh  # matrix with large singular values

result0 = u @ np.diag(s.clip(0, 1)) @ vh  # exact result via SVD
result1 = mclip1(m.astype('bfloat16'))
result2 = mclip2(m.astype('bfloat16'))
result3 = mclip3(m.astype('bfloat16'))
result4 = spectral_clip(m.astype('bfloat16'))

# spectral norm of the resulting matrix, closer to 1 is better.
jnp.linalg.svd(result0.astype('float32'))[1][0]  # = 1
jnp.linalg.svd(result1.astype('float32'))[1][0]  # ≈ 700
jnp.linalg.svd(result2.astype('float32'))[1][0]  # ≈ 250
jnp.linalg.svd(result3.astype('float32'))[1][0]  # ≈ 1.5
jnp.linalg.svd(result4.astype('float32'))[1][0]  # ≈ 13

# mean absolute error of singular values, closer to 0 is better.
jnp.abs(jnp.linalg.svd(result1.astype('float32'))[1] - s.clip(0, 1)).mean()  # ≈ 20
jnp.abs(jnp.linalg.svd(result2.astype('float32'))[1] - s.clip(0, 1)).mean()  # ≈ 10
jnp.abs(jnp.linalg.svd(result3.astype('float32'))[1] - s.clip(0, 1)).mean()  # ≈ 0.5
jnp.abs(jnp.linalg.svd(result4.astype('float32'))[1] - s.clip(0, 1)).mean()  # ≈ 0.7

# mean absolute error of total matrix, closer to 0 is better.
jnp.abs(result0 - result1).mean()  # ≈ 1
jnp.abs(result0 - result2).mean()  # ≈ 0.5
jnp.abs(result0 - result3).mean()  # ≈ 0.01
jnp.abs(result0 - result4).mean()  # ≈ 0.02</code></pre>



## 文章小结

本文继续完善了上一篇文章用$\msign$来计算$\mclip$的方案，通过去掉$\msign$的嵌套以及引入额外的修正项，成功降低了计算误差。
