# 流形上的最速下降：3\. Muon \+ Stiefel

> 作者：苏剑林 · 科学空间 · 2025-08-08
>
> 原文：<https://spaces.ac.cn/archives/11221>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上回说到，当我们把优化对象从向量参数转移到矩阵参数，并选用更适合矩阵的谱范数约束后，Muon优化器便自然而然地出现了。进一步地，我们考虑了给参数加上正交约束后的最速下降方向，这其中又分方阵和非方阵两部分讨论，其中方阵的求解我们在上一篇文章已经完成，但非方阵部分依然悬而未决。

本文的目标，则是把非方阵部分的求解补上，使得正交约束下的优化得以完全解决。

## 任务信息

先简单回顾一下上文[《流形上的最速下降：2\. Muon \+ 正交》](<https://spaces.ac.cn/archives/11215>)的结果。我们要求解的目标是
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I},\,\,(\boldsymbol{W} - \eta \boldsymbol{\Phi})^{\top}(\boldsymbol{W} - \eta \boldsymbol{\Phi})=\boldsymbol{I}\end{equation}
其中$\boldsymbol{W},\boldsymbol{\Phi}\in\mathbb{R}^{n\times m}(n \geq m)$，$\Vert\cdot\Vert_2$是谱范数。基于“一阶近似够用”的原则，可以简化成
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I},\,\,\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}\label{eq:ori-obj}\end{equation}
其中满足$\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}$的全体$\boldsymbol{\Phi}$也称为$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$的“切空间”。在上一篇文章中我们已经求出了通解的形式
\begin{equation}\boldsymbol{\Phi} = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})\end{equation}
其中$\boldsymbol{X}\in\mathbb{R}^{m\times m}$是一个待定的对称矩阵。

剩下的难题就是给出对称矩阵$\boldsymbol{X}$的计算方法，使得$\boldsymbol{W}^{\top}\boldsymbol{\Phi}$是一个反对称矩阵。一旦完成求解，那么对应的$\boldsymbol{\Phi}$自然是最优解。对于$n=m$，我们已经求得闭式解$\boldsymbol{X}=-[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}$；真正困难的是$n > m$的情形，此时亦称为“Stiefel流形”，它正是[《Orthogonal manifold》](<https://docs.modula.systems/algorithms/manifold/orthogonal/#open-problem-extending-to-the-stiefel-manifold>)所留下的Open problem。

## 方程变换

说白了，我们现在的任务是求解方程组：
\begin{equation}\boldsymbol{W}^{\top}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})+\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})^{\top}\boldsymbol{W} = \boldsymbol{0}\label{eq:start}\end{equation}
当$n=m$时，$\boldsymbol{W}^{\top}$可以直接吸收到$\msign$里边。所以求解得以简化，然而$n > m$时并不能做这样的吸收，这也是求解的困难所在。笔者倾向于$n > m$时没有简单的显式解，所以我们来寻求数值算法。

根据定义$\msign(\boldsymbol{M})=\boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$，可以写出
\begin{equation}\boldsymbol{W}^{\top}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}) = \boldsymbol{W}^{\top}(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})\boldsymbol{Q}^{-1} = (\boldsymbol{W}^{\top}\boldsymbol{G} + \boldsymbol{X})\boldsymbol{Q}^{-1}\end{equation}
其中$\boldsymbol{Q} = ((\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})^{\top}(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}))^{1/2}$。在这个新记号下，方程组变为
\begin{equation}(\boldsymbol{W}^{\top}\boldsymbol{G} + \boldsymbol{X})\boldsymbol{Q}^{-1} + \boldsymbol{Q}^{-1}(\boldsymbol{G}^{\top}\boldsymbol{W} + \boldsymbol{X}) = \boldsymbol{0}\end{equation}
同时左乘和右乘$\boldsymbol{Q}$，得到
\begin{equation}\boldsymbol{Q}(\boldsymbol{W}^{\top}\boldsymbol{G} + \boldsymbol{X}) + (\boldsymbol{G}^{\top}\boldsymbol{W} + \boldsymbol{X})\boldsymbol{Q} = \boldsymbol{0}\label{eq:r-x}\end{equation}
其中$\boldsymbol{Q}$又成立
\begin{equation}\boldsymbol{Q} = (\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})^{\top}\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})\label{eq:r-q}\end{equation}

## 迭代求解

现在笔者的想法是，从某个初值的$\boldsymbol{X}$出发，代入式$\eqref{eq:r-q}$得到$\boldsymbol{Q}$，然后将$\boldsymbol{Q}$代入方程组$\eqref{eq:r-x}$求解出新的$\boldsymbol{X}$，反复迭代，直到收敛。在已知$\msign$的情况下，式$\eqref{eq:r-q}$是可以显式计算的，所以唯一的难度是解方程组$\eqref{eq:r-x}$。

我们可以整理一下式$\eqref{eq:r-x}$：
\begin{equation}\boldsymbol{Q}\boldsymbol{X} + \boldsymbol{X}\boldsymbol{Q} =  -2[\boldsymbol{Q}\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}\label{eq:r-xx}\end{equation}
在给定$\boldsymbol{Q}$的前提下，这其实是关于$\boldsymbol{X}$是线性方程组，名为“[连续型Lyapunov方程](<https://en.wikipedia.org/wiki/Lyapunov_equation>)”，也可以看成是“[Sylvester方程](<https://en.wikipedia.org/wiki/Sylvester_equation>)”的一个特例。如果我们只用CPU进行计算，Scipy其实已经自带了该方程的求解函数<code>scipy.linalg.solve_continuous_lyapunov</code>，直接调用即可。

至于初值的选择，我们可以考虑方阵时的解$-[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}$，这样显然是方阵到非方阵的一个自然过渡。我们也可以从方程$\eqref{eq:r-xx}$的另一个等价形式，来观察初值$-[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}$的合理性：
\begin{equation}\boldsymbol{Q}(\boldsymbol{X} + [\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}) + (\boldsymbol{X} + [\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}})\boldsymbol{Q}  =[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}\boldsymbol{Q} -\boldsymbol{Q}[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}\end{equation}
所以$-[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}$的精确程度，取决于$[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{skew}}$与$\boldsymbol{Q}$乘法的可交换程度，它们越接近交换矩阵，那么$-[\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}$就越准确。不过后面的实测结果显示，我们的迭代算法对初值并不是特别敏感，即便以全零矩阵为初值问题也不大。

## 自己动手

刚才我们说到Scipy自带了求解Lyapunov方程函数，因此可以直接调用而无需关心求解过程。但这也仅限于CPU的Scipy，笔者查了一下，Torch和Jax都没有同类函数，所以要用GPU计算的话，只能“自力更生”。

自己编程求解方程$\eqref{eq:r-xx}$的做法有两个。一是按照[《矩阵符号函数mcsgn能计算什么？》](<https://spaces.ac.cn/archives/11056>)的思路，用$\newcommand{mcsgn}{\mathop{\text{mcsgn}}}\mcsgn$（不是$\msign$）来求解：
\begin{equation}\boldsymbol{X} = \mcsgn\left(\begin{bmatrix}-\boldsymbol{Q} & -[\boldsymbol{Q}\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}} \\ \boldsymbol{0} & \boldsymbol{Q}\end{bmatrix}\right)_{[:m,m:]}\end{equation}
二是基于SVD求解，这个方法我们在[《msign的导数》](<https://spaces.ac.cn/archives/11025>)里计算$\msign$的梯度时已经用过，这里结合方程$\eqref{eq:r-xx}$再介绍一遍。根据$\boldsymbol{Q}$的定义知它是正定对称的，那么可以特征值分解为$\boldsymbol{V}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，其中$\boldsymbol{V}$是正交矩阵而$\boldsymbol{\Sigma}=\mathop{\text{diag}}(\sigma_1,\cdots,\sigma_m)$是对角矩阵，代入到式$\eqref{eq:r-xx}$，可整理得
\begin{equation}\boldsymbol{\Sigma}(\boldsymbol{V}^{\top}\boldsymbol{X}\boldsymbol{V}) + (\boldsymbol{V}^{\top}\boldsymbol{X}\boldsymbol{V})\boldsymbol{\Sigma} =  -2\boldsymbol{V}^{\top}[\boldsymbol{Q}\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}\boldsymbol{V}\end{equation}
左端可以表示成$(\boldsymbol{V}^{\top}\boldsymbol{X}\boldsymbol{V})\otimes \boldsymbol{S}$，其中$\otimes$是Hadamard积，$\boldsymbol{S}_{i,j} = \sigma_i + \sigma_j$。由此，可以解得
\begin{equation}\boldsymbol{X} =  -2\boldsymbol{V}((\boldsymbol{V}^{\top}[\boldsymbol{Q}\boldsymbol{W}^{\top}\boldsymbol{G}]_{\text{sym}}\boldsymbol{V})\oslash \boldsymbol{S})\boldsymbol{V}^{\top}\end{equation}
其中$\oslash$是Hadamard商。这里比较有趣的地方是，对$\boldsymbol{Q}$做特征值分解，基本等价于对$\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}$做SVD，而对$\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}$做SVD也可以用来求$\msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})$，所以只需一遍SVD就可以把$\msign$和方程$\eqref{eq:r-xx}$的解都算出来。

两个思路各有特点。思路一需要先对$m\times m$的矩阵算$\msign$，再对$2m\times 2m$的矩阵算$\mcsgn$，虽然它们都可以用Newton\-Schulz迭代来高效计算，但也代价不菲。此外，这里我们还要选择能收敛且精度高的系数（推荐[《msign算子的Newton\-Schulz迭代（下）》](<https://spaces.ac.cn/archives/10996>)的结果），要不然$\mcsgn$和$\msign$的计算都不收敛，更别说$\boldsymbol{X}$了。

思路二需要用到SVD。虽然SVD的复杂度较高，而且往往要强制使用FP32精度，但在这里的问题上，每一步迭代只需要一次SVD就可以同时求$\msign$和$\boldsymbol{X}$，总体效率也不会太差。如果我们需要正交约束的矩阵参数并不多，那么SVD可能是最简便的选择。

## 相关结果

本文之前，[@leloy](<https://x.com/leloykun>) 在他的博客文章[《Heuristic Solutions for Steepest Descent on the Stiefel Manifold》](<https://leloykun.github.io/ponder/steepest-descent-stiefel/>)也提出了原始目标$\eqref{eq:ori-obj}$的两种启发式求解方法。这里的“启发式”，指的是在大多数情况下，它能得到一个还不错的解，但无法保证是最优解，这里我们也一起学习下。

第一种方法可以说是纯几何法。首先我们定义投影运算：
\begin{equation}\newcommand{proj}{\mathop{\mathcal{P}}}\proj\nolimits_{\boldsymbol{W}}(\boldsymbol{M}) = \boldsymbol{M} - \boldsymbol{W}[\boldsymbol{W}^{\top}\boldsymbol{M}]_{\text{sym}}\end{equation}
可以验证$\boldsymbol{W}^{\top}\proj\nolimits_{\boldsymbol{W}}(\boldsymbol{M})$一定是反对称矩阵，也就是说$\proj\nolimits_{\boldsymbol{W}}(\boldsymbol{M})$一定在切空间中，所以我们将它视为任意矩阵$\boldsymbol{M}$投影到$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$的切空间中的投影运算。

我们从梯度$\boldsymbol{G}$出发，$\proj\nolimits_{\boldsymbol{W}}(\boldsymbol{M})$肯定是在切空间了，但我们知道Muon的更新量一定是个正交矩阵（满秩时），而$\proj\nolimits_{\boldsymbol{W}}(\boldsymbol{M})$不一定正交，所以我们可以通过$\msign$来寻找与之最邻近的正交矩阵，即$\msign(\proj\nolimits_{\boldsymbol{W}}(\boldsymbol{M}))$。然而$\msign$之后又不一定在切空间了，我们又可以将它投影到切空间去，然后又寻找最近正交矩阵，反复迭代：
\begin{equation}\boldsymbol{\Phi} = (\msign\circ\proj\nolimits_{\boldsymbol{W}}\circ\cdots\circ\msign\circ\proj\nolimits_{\boldsymbol{W}})(\boldsymbol{M})\end{equation}
这便是 @leloy 的第一种思路，交替投影到切空间和正交空间直到收敛，可以说相当直观。而且在比较随机的情况下，它跟最优解也非常接近，甚至能精确到小数点后4位，以至于笔者一开始认为它就是精确解。不过后来经过搜索，发现了它跟最优解偏差足够大的case，才确认了这只是巧合，并非最优解。

第二种方法可以称为线搜索。具体来说，当$n > m$时，我们可以考虑将$\boldsymbol{W}$补成标准的$n\times n$的正交矩阵$[\boldsymbol{W},\overline{\boldsymbol{W}}]$，然后将所求$\boldsymbol{\Phi}$分解为$\boldsymbol{W}^{\top}\boldsymbol{\Phi}$和$\overline{\boldsymbol{W}}{}^{\top}\boldsymbol{\Phi}$两部份。接着 @leloy 做了一个贪心近似，先求$\boldsymbol{W}^{\top}\boldsymbol{\Phi}$的最优解，然后再求$\overline{\boldsymbol{W}}{}^{\top}\boldsymbol{\Phi}$的最优解，两者之间再引入一个线搜索来提高准确度。

这样一套操作下来，确实能得到一个近似程度还不错的解，并且它一定在切空间内且满足正交性。求解过程需要计算谱范数、$\msign$和[Cholesky分解](<https://en.wikipedia.org/wiki/Cholesky_decomposition>)，细节大家自行看作者的文章了。此外，当$m=2$时，理论上它是能搜出最优解的，这是因为$2\times 2$的反对称矩阵只有一个自由参数，而线搜索刚好是一个自由度。

## 测试一下

下面在Numpy中实测上面几个方法，其中主要目的是验证方法本身的正确性，所以我们直接用奇异值分解和特征值分解来实现$\msign$和$\mcsgn$。



<pre><code class="language-python">import numpy as np
import scipy as sp

def mcsgn(x):
    """特征值分解精确计算mcsgn
    """
    s, v = np.linalg.eig(x)
    return v @ np.diag(np.sign(s)) @ np.linalg.inv(v)

def msign(g):
    """奇异值分解精确计算msign
    """
    u, s, vh = np.linalg.svd(g, full_matrices=False)
    return u @ np.diag(np.sign(s)) @ vh

def sym(x):
    """对称化
    """
    return (x + x.T) * 0.5

def skew(x):
    """反对称化
    """
    return (x - x.T) * 0.5

def proj(g, w):
    """投影到正交的切空间
    """
    return g - w @ sym(w.T @ g)

def jianlin_by_mcsgn(g, w, steps=20):
    """通过mcsgn来构建本文的迭代
    """
    n, m = g.shape
    x = -sym(w.T @ g)
    for i in range(1, steps + 1):
        phi = msign(z := g + w @ x)
        print('step:', i, ', inner product:', (phi * g).sum(), ', tangent error:', np.abs(sym(w.T @ phi)).mean())
        if i == steps:
            return phi
        q = z.T @ phi
        x = mcsgn(np.block([[-q, -sym(q @ w.T @ g)], [np.zeros_like(q), q]]))[:m, m:]
        # x = -2 * sp.linalg.solve_continuous_lyapunov(q, sym(q @ w.T @ g))

def jianlin_by_svd(g, w, steps=20):
    """通过svd来构建本文的迭代
    """
    x = -sym(w.T @ g)
    for i in range(1, steps + 1):
        u, s, vh = np.linalg.svd(z := g + w @ x, full_matrices=False)
        phi = (u * np.sign(s)) @ vh
        print('step:', i, ', inner product:', (phi * g).sum(), ', tangent error:', np.abs(sym(w.T @ phi)).mean())
        if i == steps:
            return phi
        x = -2 * vh.T @ (vh @ sym(z.T @ phi @ w.T @ g) @ vh.T / (s + s[:, None])) @ vh

def leloy_v1(g, w, steps=20):
    """交替投影到切空间和正交空间
    """
    phi = g
    for i in range(1, steps + 1):
        phi = msign(proj(phi, w))
        print('step:', i, ', inner product:', (phi * g).sum(), ', tangent error:', np.abs(sym(w.T @ phi)).mean())
    return phi

def leloy_v2(g, w, steps=20):
    """分部贪心求解 + 线搜索（形式经过笔者的简化）
    """
    n, m = g.shape
    taus = np.linspace(0, 1, steps + 2)[1:-1]
    p_max, tau_opt, phi_opt = 0, 0, None
    for tau in taus:
        b = (b := skew(w.T @ g)) * tau / max(np.linalg.norm(b, ord=2), 1e-8)
        r = np.linalg.cholesky(np.eye(m) - b.T @ b)
        c = msign((np.eye(n) - w @ w.T) @ g @ r) @ r
        phi = w @ b + c
        print('tau:', tau, ', inner product:', p := (phi * g).sum())
        if p &gt; p_max:
            p_max, tau_opt, phi_opt = p, tau, phi
    print('best inner product:', p_max, ', tau:', tau_opt)
    return phi_opt

w = np.array([[ 0.69453734, -0.26590866, -0.44721806,  0.2753041 ],
              [-0.11738148, -0.5588003 , -0.17580748,  0.3218624 ],
              [-0.4515288 , -0.23489913, -0.26683152, -0.25739142],
              [ 0.02392521,  0.02664689,  0.48423648,  0.6193399 ],
              [ 0.45194831, -0.25206333,  0.27654836, -0.60242337],
              [ 0.21197332, -0.09174792,  0.24521762, -0.08484317],
              [-0.15496767, -0.26446804, -0.34942415, -0.01877318],
              [-0.16181251, -0.6474956 ,  0.45243263, -0.01776086]])

g = np.array([[-17.85745   , -10.758921  ,  -2.9583392 ,   6.245008  ],
              [-28.883093  ,  19.772121  ,   8.086545  , -21.564013  ],
              [ -1.6274693 , -14.96859   ,   3.4465332 ,   3.1070817 ],
              [ -7.8890743 ,   1.5304767 ,  -8.949573  ,   9.579629  ],
              [  2.246596  ,  14.46572   ,  12.8451    ,  -2.7370298 ],
              [ -0.9496974 ,   6.9879804 ,   2.849277  ,   1.1148484 ],
              [ -8.115278  , -18.054405  ,  -0.19287404,   7.0389237 ],
              [-15.062008  , -15.02901   ,   2.9083247 ,  21.706533  ]])

phi1 = jianlin_by_mcsgn(g, w, steps=100)
phi2 = jianlin_by_svd(g, w, steps=100)
phi3 = leloy_v1(g, w, steps=100)
phi4 = leloy_v2(g, w, steps=100)
assert np.allclose(phi1, phi2)

w = np.linalg.qr(np.random.randn(100, 50))[0]
g = np.random.randn(100, 50)

phi1 = jianlin_by_mcsgn(g, w, steps=10)
phi2 = jianlin_by_svd(g, w, steps=10)
phi3 = leloy_v1(g, w, steps=10)
phi4 = leloy_v2(g, w, steps=10)
assert np.allclose(phi1, phi2)</code></pre>



对于代码中给出的第一组$\boldsymbol{W},\boldsymbol{G}$，笔者方法求得的最优$\tr(\boldsymbol{G}^{\top} \boldsymbol{\Phi})$大致是$90$，并且$\mcsgn$和SVD的结果是完全一样的；而 @leloy 的第一种方法求得的结果是大致是$70$，第二种方法求得的结果大致是$80$，都跟最优解有一定差距。

不过，第一组$\boldsymbol{W},\boldsymbol{G}$只是为了显示出三个方法差距特意搜出来的极端例子，如果我们更换相对随机的数值，那么其实本文的解法和 @leloy 的第一种解法会很接近，并且迭代步数也可以少很多（5～10步），此时 @leloy 的第二种解法跟最优解差距更大。读者可以自行构建一些例子测试。

## 拓展思考

关于原始问题$\eqref{eq:ori-obj}$的求解，这里就暂告一段落了。接下来补充讨论几个可能有疑惑的细节问题。

首先，为了方便描述，笔者前面给出的迭代求解过程有一个隐含假设，那就是$\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}$自始至终都是满秩的（秩为$m$），要不然矩阵$\boldsymbol{S}$就会有零分量，$\oslash\boldsymbol{S}$就不大好操作。但这个困难不是本质的，因为方程$\eqref{eq:start}$必然会有解，所以遇到分母为零时，分子必然也为零，于是我们只需要简单将$\boldsymbol{S}$的零分量换成一个小正数，就能得到正确的结果。

从数值计算的角度看，我们也很少有机会能遇到绝对等于零的奇异值，所以不需要太担心这个问题，默认$\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}$满秩就好。在这个默认假设之下，回缩操作会变得很简单，因为
\begin{equation}(\boldsymbol{W} - \eta\boldsymbol{\Phi})^{\top}(\boldsymbol{W} - \eta\boldsymbol{\Phi}) = \boldsymbol{W}^{\top} \boldsymbol{W} - \eta(\boldsymbol{W}^{\top} \boldsymbol{\Phi} + \boldsymbol{\Phi}^{\top}\boldsymbol{W}) + \eta^2 \boldsymbol{\Phi}^{\top}\boldsymbol{\Phi}\end{equation}
根据Stiefel流形的定义，右端第一项是$\boldsymbol{I}$，根据切空间的条件，第二项是$\boldsymbol{0}$，最后是满秩时$\msign$出来的也是一个Stiefel流形的矩阵，所以第三项是$\eta^2 \boldsymbol{I}$，总的结果是$(1+\eta^2)\boldsymbol{I}$，只需要除以$\sqrt{1+\eta^2}$就可以实现回缩：
\begin{equation}\boldsymbol{W}\quad\leftarrow\quad\frac{\boldsymbol{W} - \eta\boldsymbol{\Phi}}{\sqrt{1+\eta^2}}\end{equation}

看到这里，不知道笔者有没有发现，这里其实有一个更深刻的问题：不管是相对简单的正交流形，还是相对复杂的Stiefel流形，我们应该使用何种精度计算？要知道“正交”是一个精确的定量约束，$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$包含$m(m+1)/2$个等式约束，可以预见在低精度下用上式进行迭代，久而久之肯定会严重偏离正交的，更不用说求解$\boldsymbol{\Phi}$过程中的误差了。

因此，笔者认为，除非我们定期给参数施加正交化操作（即$\boldsymbol{W}\leftarrow\msign(\boldsymbol{W})$）来将它拉回到正交流形上，否则求解过程的计算精度起码要FP32起步。考虑到通常要加正交约束的参数并不会很多，所以一般来说这也不算太大的代价。

## 文章小结

这篇文章将上一篇文章的“Muon \+ 正交流形”推广到了更一般的“Muon \+ Stiefel流形”，主要发现是一个求解对应更新量的迭代算法。
