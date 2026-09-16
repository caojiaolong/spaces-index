# 矩阵r次方根和逆r次方根的高效计算

> 作者：苏剑林 · 科学空间 · 2025-07-21
>
> 原文：<https://spaces.ac.cn/archives/11175>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章[《矩阵平方根和逆平方根的高效计算》](<https://spaces.ac.cn/archives/11158>)中，笔者从$\newcommand{mcsgn}{\mathop{\text{mcsgn}}}\mcsgn$算子出发，提出了一种很漂亮的矩阵平方根和逆平方根的计算方法。比较神奇的是，该方案经过化简之后，最终公式已经看不到最初$\mcsgn$形式的样子。这不禁引发了更深层的思考：该方案更本质的工作原理是什么？是否有推广到任意$r$次方根的可能性？

沿着这个角度进行分析后，笔者惊喜地发现，我们可以从一个更简单的角度去理解之前的迭代算法，并且在新角度下可以很轻松推广到任意$r$次方根和逆$r$次方根的计算。接下来我们将分享这一过程。

## 前情回顾

设$\boldsymbol{G}\in\mathbb{R}^{m\times n}$是任意矩阵，$\boldsymbol{P}\in\mathbb{R}^{n\times n}$是任意特征值都在$[0,1]$内的矩阵，上一篇文章给出：
\begin{gather}
\boldsymbol{G}_0 = \boldsymbol{G}, \quad \boldsymbol{P}_0 = \boldsymbol{P} \notag\\[6pt]
\boldsymbol{G}_{t+1} = \boldsymbol{G}_t(a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2) \label{eq:r2-rsqrt}\\[6pt]
\boldsymbol{P}_{t+1} = (a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2)^2\boldsymbol{P}_t \label{eq:r3-rsqrt}\\[6pt]
\lim_{t\to\infty} \boldsymbol{G}_t = \boldsymbol{G}\boldsymbol{P}^{-1/2}\notag
\end{gather}
代入$\boldsymbol{G}=\boldsymbol{P}$就可以求得$\boldsymbol{P}^{1/2}$，代入$\boldsymbol{G}=\boldsymbol{I}$就可以求得$\boldsymbol{P}^{-1/2}$。仔细观察我们就会发现，上述迭代实际上是如下极限的体现：
\begin{equation} \prod_{t=0}^{\infty}(a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2) = \boldsymbol{P}^{-1/2}\label{eq:prod-rsqrt}\end{equation}
有趣的是，直接证明这个极限并不复杂，直接对式$\eqref{eq:r3-rsqrt}$两边开方，然后代入上式可得
\begin{equation} \prod_{t=0}^{\infty}(a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2) = \prod_{t=0}^{\infty} \boldsymbol{P}_{t+1}^{1/2}\boldsymbol{P}_t^{-1/2} = \lim_{t\to\infty} \boldsymbol{P}_t^{1/2}\boldsymbol{P}_0^{-1/2} = \lim_{t\to\infty} \boldsymbol{P}_t^{1/2}\boldsymbol{P}^{-1/2}\end{equation}
由此可见，只要序列$\{\boldsymbol{P}_t\}$始终保持可逆，并且最终极限为$\boldsymbol{I}$，那么极限$\eqref{eq:prod-rsqrt}$自动成立。至于迭代$\eqref{eq:r3-rsqrt}$如何让$\{\boldsymbol{P}_t\}$保持这两个条件，我们等会再一起讨论。

## 一般形式

我们不妨一般地考虑迭代
\begin{gather}
\boldsymbol{G}_0 = \boldsymbol{G}, \quad \boldsymbol{P}_0 = \boldsymbol{P} \notag\\[6pt]
\boldsymbol{G}_{t+1} = \boldsymbol{G}_t(a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2)^s\\[6pt]
\boldsymbol{P}_{t+1} = (a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2)^r\boldsymbol{P}_t
\end{gather}
类似地，如果序列$\{\boldsymbol{P}_t\}$始终保持可逆，并且最终极限为$\boldsymbol{I}$，那么可以证明成立
\begin{equation}\lim_{t\to\infty} \boldsymbol{G}_t = \boldsymbol{G}\boldsymbol{P}^{-s/r}\end{equation}
于是我们就得到了一种求矩阵的任意$-s/r$次幂的通用迭代形式。在这个结果之上，我们只需选择$\boldsymbol{G}=\boldsymbol{P}, s=r-1$，就可以得到$\boldsymbol{P}^{1/r}$了，所以只需要集中精力解决$0\sim 1$次幂的逆就可以了。

这样一来，问题变成了如何选择适当$\{a_t,b_t,c_t\}$，让序列$\{\boldsymbol{P}_t\}$能够尽可能快地收敛到$\boldsymbol{I}$，收敛速度越快，意味着我们可以用越少的迭代步数达到指定精度。

## 迭代系数

根据假设，$\boldsymbol{P}_0 = \boldsymbol{P}$是一个特征值都在$[0,1]$内的矩阵，而目标矩阵$\boldsymbol{I}$则是特征值全为1的矩阵，所以序列$\{\boldsymbol{P}_t\}$实际上就是特征值从任意$[0,1]$内的数变成$1$的过程，这实际上就是$\mcsgn$所做的事情！

我们设$\boldsymbol{X}_t = \boldsymbol{P}_t^{1/r}$，那么$\boldsymbol{X}_0 = \boldsymbol{P}^{1/r}$同样是一个特征值都在$[0,1]$内的矩阵，且迭代方程变为
\begin{equation}\boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t^{r+1} + c_{t+1}\boldsymbol{X}_t^{2r+1}\end{equation}
现在问题则变成了如何让$\boldsymbol{X}_0$尽可能快地变成$\boldsymbol{I}$，这跟我们在[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)和[《msign算子的Newton\-Schulz迭代（下）》](<https://spaces.ac.cn/archives/10996>)所讨论的问题实质是一样的。其中，“[下篇](<https://spaces.ac.cn/archives/10996>)”给出了$r=2$的理论最优解，但它的求解过程和结论都可以推广到任意$r$的。

具体来说，我们先将问题转化为标量的迭代：
\begin{equation}x_{t+1} = f_t(x_t) = a_{t+1}x_t + b_{t+1}x_t^{r+1} + c_{t+1}x_t^{2r+1}\end{equation}
然后证明贪心解就是最优解，而求贪心解则变成了解方程
\begin{equation}\begin{gathered}
f_t(l_t) = 1 - \mathcal{E}, \quad f_t(u_t) = 1 + \mathcal{E} \\
f_t(x_1) = 1 + \mathcal{E}, \quad f_t(x_2) = 1 - \mathcal{E} \\
f_t'(x_1) = 0, \quad f_t'(x_2) = 0
\end{gathered}\end{equation}
简单起见，将$f_t$参数化为
\begin{equation}f_t'(x) = k(x^r-x_1^r)(x^r-x_2^r)\end{equation}
那么就可以像“[下篇](<https://spaces.ac.cn/archives/10996>)一样，用Mathematica求解了。

## 初始分析

不过正式求解之前，我们还要分析一下初始化。在上一篇文章[《矩阵平方根和逆平方根的高效计算》](<https://spaces.ac.cn/archives/11158>)中我们提到，在$\boldsymbol{P}$的特征值均非负的假设下，我们可以通过除以$\newcommand{tr}{\mathop{\text{tr}}}\tr(\boldsymbol{P})$将特征值都压缩到$[0,1]$内。不过这个压缩比例往往过大了，本文我们改为
\begin{equation}\boldsymbol{P}_0 = \frac{\boldsymbol{P}}{\sqrt{\tr(\boldsymbol{P}^2)}}\end{equation}
我们知道，$\tr(\boldsymbol{P}^2)$等于全体特征值的平方和，$\tr(\boldsymbol{P})^2$则等于全体特征值的和平方，在特征值非负时恒成立$\tr(\boldsymbol{P}^2)\leq\tr(\boldsymbol{P})^2$，所以上式提供了一个更紧凑的初始值。特别地，算$\tr(\boldsymbol{P}^2)$并不需要把$\boldsymbol{P}^2$显式计算出来，因为我们有恒等式
\begin{equation}\tr(\boldsymbol{P}^2) = \langle \boldsymbol{P}, \boldsymbol{P}^{\top}\rangle_F\end{equation}

接着，我们还要分析需要处理多小的特征值，这跟[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)的初始奇异值分析是一样的。除以$\sqrt{\tr(\boldsymbol{P}^2)}$后，$\boldsymbol{P}_0$的特征值组成一个单位向量，如果特征值全部相等，那么每个特征值是$1/\sqrt{n}$。由鸽笼原理可知一般情况下必然存在小于$1/\sqrt{n}$的特征值，保守起见我们兼容到$0.01/\sqrt{n}$。

考虑到足够大的LLM，$n$已经到了$100^2$这个级别，所以我们需要兼容到$0.0001$。注意这只是$\boldsymbol{P}_0$的特征值，而$\boldsymbol{X}_0 = \boldsymbol{P}_0^{1/r}$，所以$\boldsymbol{X}_0$我们只需要兼容到$0.0001^{1/r}$，这比$\mcsgn$、$\newcommand{msign}{\mathop{\text{msign}}}\msign$的情况要理想一些，因为$\mcsgn$、$\msign$的输入是$\boldsymbol{X}_0$，我们需要兼容$\boldsymbol{X}_0$的小特征值，但这里的输入是$\boldsymbol{P}_0$，我们只需要从$\boldsymbol{P}_0$出发考虑。

## 计算结果

综合上述考虑，我们最终的求解代码如下



<pre><code class="language-python">r = 4;
df[x_] = k*(x^r - x1^r) (x^r - x2^r);
f[x_] = Integrate[df[x], {x, 0, x}];
sol[l_, u_] := 
 NSolve[{f[l] == 1 - e, f[x1] == 1 + e, f[x2] == 1 - e, f[u] == 1 + e,
    l &lt; x1 &lt; x2 &lt; u, e &gt; 0, k &gt; 0}, {k, x1, x2, e}]
ff[x_, l_, u_] = f[x]*2/(f[l] + f[u]) // Expand;
lt = 0.0001^(1/r); ut = 1; lambda = 0.1;
While[1 - lt &gt; 0.0001,
 fff[x_] = ff[x, lt, ut] /. sol[Max[lt, lambda*ut], ut][[1]];
 Print[fff[x]];
 lt = fff[lt]; ut = 2 - lt]
f[x] /. Solve[f[1] == 1, k][[1]] /. {x1 -&gt; 1, x2 -&gt; 1}
</code></pre>



$r=1\sim 5$的计算结果如下：
\begin{array}{c|ccc}
\hline
r & t & a & b & c \\
\hline
& \quad 1\quad & 14.2975 & -31.2203 & 18.9214 \\
&       2      & 7.12258 & -7.78207 & 2.35989 \\
\quad 1\quad &       3      & 6.9396  & -7.61544 & 2.3195  \\
&       4      & 5.98456 & -6.77016 & 2.12571 \\
&       5      & 3.79109 & -4.18664 & 1.39555 \\
&  \geq 6      & 3       & -3       & 1       \\
\hline
&      1 & 7.42487 & -18.3958 & 12.8967  \\
&      2 & 3.48773 & -2.33004 & 0.440469 \\
2 &      3 & 2.77661 & -2.07064 & 0.463023 \\
&      4 & 1.99131 & -1.37394 & 0.387593 \\
& \geq 5 & 15/8    & -5/4     & 3/8      \\
\hline
&      1 & 5.05052 & -13.5427  & 10.2579  \\
&      2 & 2.31728 & -1.06581  & 0.144441 \\
3 &      3 & 1.79293 & -0.913562 & 0.186699 \\
&      4 & 1.56683 & -0.786609 & 0.220008 \\
& \geq 5 & 14/9    & -7/9      & 2/9      \\
\hline
&      1 & 3.85003 & -10.8539  & 8.61893   \\
4 &      2 & 1.80992 & -0.587778 & 0.0647852 \\
&      3 & 1.50394 & -0.594516 & 0.121161  \\
& \geq 4 & 45/32   & -9/16     & 5/32      \\
\hline
&      1 & 3.11194 & -8.28217  & 6.67716   \\
5 &      2 & 1.5752  & -0.393327 & 0.0380364 \\
&      3 & 1.3736  & -0.44661  & 0.0911259 \\
& \geq 4 & 33/25   & -11/25    & 3/25      \\
\hline
\end{array}

其中最后一步的收敛值，由$x_1=x_2=1$和$f(1)=1$得出。

## 测试一下

一个简单的测试代码如下：



<pre><code class="language-python">import numpy as np
import jax.numpy as jnp

coefs = [
    None,
    [
        (14.2975, -31.2203, 18.9214),
        (7.12258, -7.78207, 2.35989),
        (6.9396, -7.61544, 2.3195),
        (5.98456, -6.77016, 2.12571),
        (3.79109, -4.18664, 1.39555),
        (3, -3, 1),
    ],
    [
        (7.42487, -18.3958, 12.8967),
        (3.48773, -2.33004, 0.440469),
        (2.77661, -2.07064, 0.463023),
        (1.99131, -1.37394, 0.387593),
        (15 / 8, -5 / 4, 3 / 8),
    ],
    [
        (5.05052, -13.5427, 10.2579),
        (2.31728, -1.06581, 0.144441),
        (1.79293, -0.913562, 0.186699),
        (1.56683, -0.786609, 0.220008),
        (14 / 9, -7 / 9, 2 / 9),
    ],
    [
        (3.85003, -10.8539, 8.61893),
        (1.80992, -0.587778, 0.0647852),
        (1.50394, -0.594516, 0.121161),
        (45 / 32, -9 / 16, 5 / 32),
    ],
    [
        (3.11194, -8.28217, 6.67716),
        (1.5752, -0.393327, 0.0380364),
        (1.3736, -0.44661, 0.0911259),
        (33 / 25, -11 / 25, 3 / 25),
    ],
]

def abc(r=1, steps=None, scale=1):
    w, steps = coefs[r], steps or len(coefs[r])
    for a, b, c in w[:steps] + w[-1:] * max(steps - len(w), 0):
        yield a / scale, b / scale**(r + 1), c / scale**(2 * r + 1)

def matmul_invroot(G, P, r, s=1, steps=None, eps=1e-5):
    """return G @ P^(-s/r)
    """
    I = jnp.eye(P.shape[0], dtype=P.dtype)
    P = P / (t := (P * P.mT).sum()**0.5) + eps * I
    for a, b, c in abc(r, steps, 1.001):
        W = a * I + b * P + c * P @ P
        W1, W2 = jnp.linalg.matrix_power(W, s), jnp.linalg.matrix_power(W, r)
        G, P = G @ W1, P @ W2
    return G * t**(-s / r)

def matmul_invroot_by_eigh(G, P, r, s=1):
    """return G @ P^(-s/r)
    """
    S, Q = jnp.linalg.eigh(P)
    return G @ Q @ jnp.diag(S**(-s / r)) @ jnp.linalg.inv(Q)

d = 1000
s, r = 1, 4
G = np.random.randn(2 * d, d) / d**0.5
P = (x := np.random.randn(d, d) / d**0.5) @ x.T + 0.001 * np.eye(d)

X1 = matmul_invroot_by_eigh(G, P, r, s)
X2 = matmul_invroot(G, P, r, s, eps=0)
jnp.abs(X1 - X2).mean()  # ~= 1e-3

X2 = matmul_invroot(jnp.array(G, dtype='bfloat16'), jnp.array(P, dtype='bfloat16'), r, s, eps=0)
jnp.abs(X1 - X2).mean()  # ~= 2e-3</code></pre>



这里有几点注意事项。首先输入$\boldsymbol{P}$的最小特征值不能太小，否则迭代过程极其容易爆炸，哪怕我们只想要求正数次幂如$\boldsymbol{P}^{1/2}$也是如此。这个其实不难理解，因为$\sqrt{x}$在$x=0$处也是比较病态的，一旦由于误差问题“不小心”到了负半轴，那直接就不存在（实数解）了，这时候迭代的表现就无法预估。

多小才不算“太小”呢？大概是$\boldsymbol{P}/\sqrt{\tr(\boldsymbol{P}^2)}$的最小特征值明显不能小于我们考虑的最小特征值，即$0.0001$。如果没法保证这一点，那么建议直接设置
\begin{equation}\boldsymbol{P}_0 = \frac{\boldsymbol{P}}{\sqrt{\tr(\boldsymbol{P}^2)}} + \epsilon \cdot\boldsymbol{I} \end{equation}
其中$\epsilon\sim 0.0001$。这样会损失一点精度，但能够明显增加数值稳定性。

此外，迭代步数在大多数情况下不需要超过推荐值<code>len(coefs[r])</code>，尤其是低精度计算场景，因为迭代步数越多，越容易由于累积误差而爆炸。实际上只要特征值在考虑范围内，推荐步数已经足够达到理想精度了，除非我们用fp32甚至更高精度迭代，那么可以考虑设置$\epsilon=0$、<code>scale=1</code>以及使用更多的迭代步数。

## 文章小结

本文将上一篇文章的结果，推广到任意的$r$次方根和逆$r$次方根的计算，得到了一种矩阵的任意$-1/r$次方的通用迭代格式。
