# 矩阵函数近似中的暴力美学

> 作者：苏剑林 · 科学空间 · 2026-06-25
>
> 原文：<https://spaces.ac.cn/archives/11787>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

此前，我们在[《矩阵平方根和逆平方根的高效计算》](<https://spaces.ac.cn/archives/11158>)和[《矩阵r次方根和逆r次方根的高效计算》](<https://spaces.ac.cn/archives/11175>)探讨过矩阵幂的高效计算，它们将原本用于计算$\newcommand{msign}{\mathop{\text{msign}}}\msign$的Newton\-Schulz迭代推广到了矩阵幂中。而近日发布到arXiv上的[《Muon<sup>p</sup>: Muon with Fractional Spectral Powers》](<https://papers.cool/arxiv/2606.13867>)则提供了另外一个构造迭代来计算矩阵幂的思路，也给了笔者一些启发。

但不管是笔者还是Muon<sup>p</sup>的思路，总的来说都是“能用，但不够通用”——比如它们只能用来计算矩阵的有理次方，并且复杂度随着最简分数的分子分母增大而增大，这显然不够科学。为了克服这些缺陷，本文提出一套理论上可以拟合任意矩阵函数的通用近似框架。

## 两种函数

说到矩阵函数，实际上它有两种略微不同的含义，这里简单介绍一下。

第一种是对特征值操作的矩阵函数（EIG型）。设矩阵$\boldsymbol{M}$的特征值分解为$\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{-1}$，那么定义$f[\boldsymbol{M}]\triangleq\boldsymbol{Q}f(\boldsymbol{\Lambda})\boldsymbol{Q}^{-1}$，其中$f(\boldsymbol{\Lambda})$是指将$f$运算逐元素应用到对角线上。这实际上就是我们通常说的矩阵函数，比如[矩阵指数](<https://en.wikipedia.org/wiki/Matrix_exponential>)、[矩阵对数](<https://en.wikipedia.org/wiki/Logarithm_of_a_matrix>)都属于此类，它们通常用幂级数来定义。

第二种是对奇异值操作的矩阵函数（SVD型）。设矩阵$\boldsymbol{M}$的奇异值分解为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，那么定义$f\{\boldsymbol{M}\} \triangleq \boldsymbol{U}f(\boldsymbol{\Sigma})\boldsymbol{V}^{\top}$。Muon的$\msign$运算，我们在[《通过msign来计算奇异值裁剪mclip（上）》](<https://spaces.ac.cn/archives/11006>)和[《通过msign来计算奇异值裁剪mclip（下）》](<https://spaces.ac.cn/archives/11059>)讨论的$\newcommand{mclip}{\mathop{\text{mclip}}}\mclip$运算，都属于此类。

EIG型矩阵函数只对方阵成立，并且只有在复数域才能保证特征值分解的存在性，所以$f$的定义通常还需要延拓到复数（除非将输入矩阵限定在实对称矩阵内）；SVD型矩阵函数对任意形状的矩阵都可以定义，任意实矩阵都有实数范围内的奇异值分解，且奇异值总是非负，这些特性使得SVD型矩阵函数的定义和分析相对简化一些。

这两类矩阵函数各有各的应用场景，它们的计算难度也没有本质不同，但在计算细节上会有所区别。对于幂来说，特征值的任意正整数次幂都可以直接计算，但奇异值我们只能高效计算奇数次幂：
\begin{equation}\begin{aligned}[]
[\boldsymbol{M}]^n =&\, \boldsymbol{Q}\boldsymbol{\Lambda}^n\boldsymbol{Q}^{-1} = (\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{-1})^n = \boldsymbol{M}^n \\[4pt]
\{\boldsymbol{M}\}^{2n+1} =&\, \boldsymbol{U}\boldsymbol{\Sigma}^{2n+1}\boldsymbol{V}^{\top} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}(\boldsymbol{V}\boldsymbol{\Sigma}^2\boldsymbol{V}^{\top})^n = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^n
\end{aligned}\end{equation}
即$[\boldsymbol{M}]^n$跟我们用矩阵乘法定义的$\boldsymbol{M}^n$是一致的，而$\{\boldsymbol{M}\}^{2n+1}$需要借助$(\boldsymbol{M}^{\top}\boldsymbol{M})^n$来间接计算。所以当我们想要通过多项式来逼近矩阵函数时，EIG型函数可以选择任意次的多项式，SVD型函数则只能选择奇次多项式。

## 现有方法

这一节我们以SVD型分数幂$\{\boldsymbol{M}\}^{1/3}$为例，介绍现有的两种近似计算方案。简单起见，假设$\boldsymbol{M}$的奇异值都在$[0, 1]$内。

第一种思路是利用恒等式$\{\boldsymbol{M}\}^{1/3} = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/3}$，然后代入[《矩阵r次方根和逆r次方根的高效计算》](<https://spaces.ac.cn/archives/11175>)的结果，得到
\begin{gather}
\boldsymbol{G}_0 = \boldsymbol{M}, \quad \boldsymbol{P}_0 = \boldsymbol{M}^{\top}\boldsymbol{M} \notag\\[6pt]
\boldsymbol{G}_{t+1} = \boldsymbol{G}_t(a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2)\\[6pt]
\boldsymbol{P}_{t+1} = (a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2)^3\boldsymbol{P}_t
\end{gather}
最终将有$\lim_{t\to\infty} \boldsymbol{G}_t = \{\boldsymbol{M}\}^{1/3}$。注意$\boldsymbol{P}_t$的迭代是独立的，这个方案的原理是选择适当的$a_t,b_t,c_t$让$\boldsymbol{P}_t\to \boldsymbol{I}$，然后单独分离出来的$a_{t+1}\boldsymbol{I} + b_{t+1}\boldsymbol{P}_t + c_{t+1}\boldsymbol{P}_t^2$的连乘就能趋于$\boldsymbol{P}_0^{-1/3}$。它的主要问题是要先显式计算$\boldsymbol{M}^{\top}\boldsymbol{M}$，导致条件数平方，计算精度也随之下降。

第二种思路来自[《Muon<sup>p</sup>: Muon with Fractional Spectral Powers》](<https://papers.cool/arxiv/2606.13867>)，它将$m^{1/3}$的计算视为求方程$x^3 - m = 0$的根，那么我们只需要设计一个奇次多项式迭代来求该方程的根，就可以写出对应的矩阵迭代。论文考虑的是不动点迭代
\begin{equation}x_{t+1} = x_t + c (m - x_t^3) \qquad\Leftrightarrow\qquad \boldsymbol{X}_{t+1} = \boldsymbol{X}_t + c (\boldsymbol{M} - \boldsymbol{X}_t\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)\end{equation}
通过简单分析可以证明，保证任意$m\in[0, 1]$都收敛的条件是$c\leq 2/3$，如果取固定值可以考虑$c=2/3$。这个思路给笔者最大的启发是将输入$\boldsymbol{M}$加入到每一步迭代过程中，而不单单依赖于上一步结果$\boldsymbol{X}_t$，但它仍留有很多的未解决的问题。比如当迭代格式有多参数时，如何逐一确定这些参数，是否可以每一步选择不同的参数，提高收敛速度，这些问题的答案都不得而知。

除此之外，上述两种思路有一个共同的问题，就是计算量强依赖于所求幂次的最简分数形式，比如，如果我们想要求$0.33$次幂，最简分数是$33/100$，那么这两种思路都需要我们设法求某个矩阵的$100$次方，这是很可观的计算量，但实际上$0.33$跟$1/3$相差无几，不应该有这么大区别才对。

## 暴力美学

所以，我们需要一个更通用的框架，能够为指定矩阵函数推导一个可用的近似，并且对于相近的矩阵函数能够自动给出相近的结果，最好理论上还能支持幂函数外的任意函数。这便是本文的目标。

不失一般性，考虑SVD型矩阵函数$f\{\boldsymbol{M}\}$，它将奇异值$m$变成$f(m)$，我们需要构造一个奇次多项式迭代来逼近$f(m)$。假设每一步同时依赖于当前值$x_t$和输入$m$（一般地，我们还可以考虑将$x_1,\cdots,x_{t-1}$都加入到迭代中），每一步迭代的阶次不超过3，那么可以构造出一般的迭代格式
\begin{equation}x_{t+1} = c_{t+1,1} x_t + c_{t+1,2} m + c_{t+1,3} x_t^3 + c_{t+1,4} m^3 + c_{t+1,5} x_t^2 m + c_{t+1,6} x_t m^2\end{equation}
其中$\boldsymbol{c}_{t+1} = (c_{t+1,1},\cdots,c_{t+1,6})$是待求参数，每步有6个参数，允许每一步都取不同值，以提高近似程度。这里的思路其实很直接，就是把我们能想到的项都加进来，不用强求解释性，由接下来的拟合结果来决定它们是否有用。这样做虽然没有从理论上求得解析解的优雅感，但却有着一种通用计算的暴力美感。

接下来该怎么求$\boldsymbol{c}_{t+1}$呢？朴素的做法是像之前的文章[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)、[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)一样，先固定一个迭代步数$T$，这样就能将整个迭代视为一个模型，接着再选择一个回归目标，就可以用梯度优化器去端到端训练。

然后，尽管这样做理论上有机会得到更接近全局最优的解，但实践中往往会面临很多的困难——多步迭代后模型的非线性、非凸性，以及初始化的随机性，这都会明显影响效果。

## 贪心策略

为了稳定效果，笔者提出一种逐层求解的贪心策略——每步都假设$\boldsymbol{c}_1,\boldsymbol{c}_2,\cdots,\boldsymbol{c}_t$都已知的前提下，去求$\boldsymbol{c}_{t+1}$，优化目标为
\begin{equation}\newcommand{argmin}{\mathop{\text{argmin}}}\boldsymbol{c}_{t+1}^* = \argmin_{\boldsymbol{c}_{t+1}} d(x_{t+1}, f)\qquad\text{s.t.}\qquad \Vert \boldsymbol{c}_{t+1}\Vert_{\infty} \leq B\end{equation}
这里$\Vert\boldsymbol{c}_{t+1}\Vert_{\infty} \leq B$约束了$\boldsymbol{c}_{t+1}$的最大绝对值都不能超过$B$，避免极端放大后缩小导致的精度丢失，我们也可以考虑给每一个$c_{t+1,i}$都选择不同的$B_i$。$d(x_{t+1}, f)$是成本函数，我们有两种选择
\begin{align}L_2:&\,\qquad \int_a^b (x_{t+1} - f(m))^2 dm \\
L_{\infty}:&\,\qquad \max_{m\in [a, b]} |x_{t+1} - f(m)| \\
\end{align}
其中$[a,b]$是我们关心的奇异值区间。很明显，$L_2$关心的是平均误差，$L_{\infty}$关心的是最大误差，后面我们将会看到，在这两个选择下我们都可以精确求解。

贪心策略虽然没法保证最优，但也有它的独特优势。首先，[Polar Express](<https://spaces.ac.cn/archives/10996>)证明了，如果$f(m)=1$（即计算$\msign$），那么贪心解即最优解，这表明在某些情况下贪心解也能很接近最优解；其次，贪心解每一步是精确的、递进的，确保每一步都在更接近目标，我们可以预计算足够多的$\boldsymbol{c}^*$，然后按需截断前面若干步，都是一个很好的近似。

最后，由于贪心解的每一步都在近似目标，所以它的波动还是可控的，不会出现某一步剧烈放大然后下一步再缩小的现象，这对于数值计算尤其是低精度计算尤为重要。我们还可以继续添加约束来实现我们期望的性质，比如$\Vert \boldsymbol{c}_{t+1}\Vert_{\infty} \leq B$就是很基本的一个约束，如果有必要，还可以添加更多。

## 逐一求解

之所以选择这两个成本函数，主要原因还是它们对应的优化问题都可精确求解，接下来我们演示它们的求解过程。

首先是$L_2$，它需要算一个积分，我们用离散近似后，它就变成了一个有限样本的回归问题，注意到$x_{t+1}$关于$\boldsymbol{c}_{t+1}$是线性的，所以这只不过是一个线性回归问题，精确可解！即便在此基础上再增加一些线性约束，那也只不过是一个凸二次规划，依然精确可解并且求解工具很成熟。

然后是$L_{\infty}$，它需要对整个区间的误差取$\max$，这一点我们也是离散化近似。接下来，我们可以通过一个巧妙的变换，将它转化成一个线性规划问题：引入新变量$z$，我们有
\begin{equation}\min_{\boldsymbol{c}_{t+1}} \max_{m\in [a, b]} |x_{t+1} - f(m)| = \min_{\boldsymbol{c}_{t+1}, z} \big\{z \,\big| -z \leq x_{t+1} - f(m) \leq z, \forall m\in[a, b]\big\}\end{equation}
如果将$[a,b]$离散化成$N$个点，那么这些样本将转化成关于$7$个未知数的$2N$个线性不等式约束，在这些约束下求$z$的最小值。线性规划的求解比二次规划更简单也更成熟，自然也没有什么困难。

能将$L_{\infty}$情形转化为线性规划，也是贪心策略的优势之一。如果我们试图用梯度下降类的优化器去联合优化各个$\boldsymbol{c}$，那$L_{\infty}$的$\max$将会是优化的核心障碍，因为梯度难以通过$\max$有效传播，加上多步迭代后的函数已经极度非线性，使得求全局解的努力难以奏效。而通过“贪心解\+线性规划”的转化，我们可以稳定地获得一个有效的解。

## 参考实现

如果只局限在Numpy和Scipy内，对于带边界约束的线性回归，我们可以用<code>scipy.optimize.lsq_linear</code>来求解，至于一般的线性规划，我们可以用<code>scipy.optimize.linprog</code>来求解。但考虑到代码的通用性和简洁性，以及随时加自定义约束的可能性，我们建议基于[CVXPY](<https://www.cvxpy.org/>)等专门的凸优化工具来求解。

下面以$f(m) = m^{1/3}$为例，给出基于CVXPY的一个参考实现：



<pre><code class="language-python">import numpy as np
import cvxpy as cp

N = 10000  # 离散化点数
B = 10  # 参数边界
m = np.linspace(0, 1.01, N)  # 我们关心的奇异值范围是0～1，但参数估计时考虑大一点，以保证稳定性

x, f = m, m**(1 / 3)
coefs = []
for t in range(10):
    A = np.array([x, m, x**3, m**3, x**2 * m, x * m**2]).T
    c = cp.Variable(6)
    objective = cp.Minimize(cp.sum_squares(A @ c - f))  # L2
    # objective = cp.Minimize(cp.max(cp.abs(A @ c - f)))  # L∞
    constraints = [cp.abs(c) &lt;= B]
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    coefs.append(c.value.round(3))
    x = A @ c.value.round(3)
    print(f'iter {t + 1}, max error:', np.abs(x - f).max())
    print(f'iter {t + 1}, mse error:', np.square(x - f).mean())

coefs = np.array(coefs)</code></pre>



## 效果比较

下面我们比较$L_2$贪心解、$L_{\infty}$贪心解与Muon<sup>p</sup>的固定步长迭代$x_{t+1} = x_t + \frac{2}{3}(m - x_t^3)$在立方根$f(m)=m^{1/3}$上的表现。所有方法都从$x_0=m$出发，离散区间为$[0, 1.01]$，参数边界$B=10$，$T$为总迭代步数。

> <strong>$L_2$贪心解迭代10步</strong>：最大误差约$8.5\times 10^{-2}$，均方误差约$5.4\times 10^{-5}$；
> 
> 
> 
> <strong>$L_{\infty}$贪心解迭代10步</strong>：最大误差约$4.6\times 10^{-2}$，均方误差约$8.2\times 10^{-4}$；
> 
> 
> 
> <strong>Muon<sup>p</sup>迭代10步</strong>：最大误差约$1.4\times 10^{-1}$，均方误差约$6.3\times 10^{-4}$；
> 
> 
> 
> <strong>Muon<sup>p</sup>迭代20步</strong>：最大误差约$1.0\times 10^{-1}$，均方误差约$1.3\times 10^{-4}$。

对比图如下：

[![立方根近似方法对比](<https://spaces.ac.cn/usr/uploads/2026/06/3330219580.svg>)](<https://spaces.ac.cn/usr/uploads/2026/06/3330219580.svg>)

立方根近似方法对比

从数值上看，$L_{\infty}$贪心解在最大误差上优势明显，$L_2$次之，固定步长的Muon<sup>p</sup>方案在$T=20$时，最大误差还不如$T=10$的$L_2$贪心解；从图上可以看到，Muon<sup>p</sup>方案的劣势区间主要是0附近，这是因为$c=2/3$虽然兼顾了$[0,1]$整个区间，但对于0附近来说太小了，这是静态步长的主要缺陷。

通过将代码中的$1/3$换成$1/5$，我们就可以得到5次方根的近似，且迭代阶次不变，但如果用Muon<sup>p</sup>方案或者笔者之前的$r$次方根算法，则至少需要将迭代阶次提升到5，这便是本文框架通用性的体现。

## 一些结果

下面给出求$\{\boldsymbol{M}\}^0$（$\msign$）、$\{\boldsymbol{M}\}^{1/2}$、$\{\boldsymbol{M}\}^{1/3}$、$\{\boldsymbol{M}\}^{1/4}$的$L_{\infty}$贪心解的迭代系数。所有系数均保留三位小数，参数边界$B=10$。其中$m^{1/2}, m^{1/3}, m^{1/4}$的拟合区间为$[0, 1.01]$；$m^0$（即常数1）的拟合区间取$[0.001, 1.01]$。

每步迭代格式为
\begin{equation}\begin{aligned}
\boldsymbol{X}_{t+1} =&\, c_{t+1, 1} \boldsymbol{X}_t + c_{t+1, 2} \boldsymbol{M} + c_{t+1, 3} \boldsymbol{X}_t \boldsymbol{X}_t^{\top} \boldsymbol{X}_t \\[4pt]
&\, + c_{t+1, 4} \boldsymbol{M} \boldsymbol{M}^{\top} \boldsymbol{M} + c_{t+1, 5} \boldsymbol{X}_t \boldsymbol{X}_t^{\top}\boldsymbol{M} + c_{t+1, 6} \boldsymbol{X}_t \boldsymbol{M}^{\top}\boldsymbol{M}
\end{aligned}\end{equation}
其中$\boldsymbol{X}_0 = \boldsymbol{M}$，假设$\boldsymbol{M}$的奇异值都已经归一化到$[0,1]$内。注意$\boldsymbol{M}^{\top} \boldsymbol{M}$和$\boldsymbol{M} \boldsymbol{M}^{\top} \boldsymbol{M}$是第一步必须计算的，可以存下来，$c_{t+1, 3} \boldsymbol{X}_t \boldsymbol{X}_t^{\top} \boldsymbol{X}_t$与$c_{t+1, 5} \boldsymbol{X}_t \boldsymbol{X}_t^{\top}\boldsymbol{M}$可以合并成$ \boldsymbol{X}_t \boldsymbol{X}_t^{\top} (c_{t+1, 3}\boldsymbol{X}_t + c_{t+1, 5}\boldsymbol{M})$，所以这个迭代虽然有六项，但实际相比Muon<sup>p</sup>的迭代只增加一步矩阵乘法$\boldsymbol{X}_t \boldsymbol{M}^{\top}\boldsymbol{M}$。

各系数如下：
$$\begin{array}{c|c|cccccc}
\hline
& t & c_{t,1} & c_{t,2} & c_{t,3} & c_{t,4} & c_{t,5} & c_{t,6} \\
\hline
& 1 & 2.564 & 2.564 & -1.256 & -1.256 & -1.256 & -1.256 \\
& 2 & 2.668 & 1.243 & -1.987 & 0.715 & 7.158 & -10.000 \\
& 3 & 2.670 & -0.793 & -0.791 & 3.138 & 1.727 & -4.170 \\
& 4 & 2.487 & 0.018 & -0.635 & 0.033 & 0.019 & -0.060 \\
\{\boldsymbol{M}\}^0 & 5 & 2.350 & 0.017 & -0.616 & -0.013 & -0.003 & 0.002 \\
& 6 & 2.095 & 0.001 & -0.582 & 0.001 & 0.000 & -0.002 \\
& 7 & 1.761 & -0.000 & -0.537 & 0.001 & 0.001 & -0.002 \\
& 8 & 1.547 & -0.001 & -0.508 & 0.001 & 0.001 & -0.001 \\
& 9 & 1.503 & -0.000 & -0.501 & -0.000 & 0.000 & 0.000 \\
& 10 & 1.499 & -0.000 & -0.499 & 0.000 & 0.000 & -0.000 \\
\hline
& 1 & 0.931 & 0.931 & -0.245 & -0.245 & -0.245 & -0.245 \\
& 2 & 1.366 & 0.368 & -5.209 & 0.216 & 10.000 & -5.723 \\
& 3 & 1.495 & -0.070 & -5.241 & 0.379 & 10.000 & -5.557 \\
& 4 & 1.261 & 0.402 & -5.486 & -0.066 & 10.000 & -5.143 \\
\{\boldsymbol{M}\}^{1/2} & 5 & 1.145 & 0.370 & -4.675 & 1.648 & 10.000 & -7.530 \\
& 6 & 1.112 & 0.429 & -4.937 & 1.000 & 10.000 & -6.635 \\
& 7 & 1.071 & 0.535 & -5.013 & 1.129 & 10.000 & -6.762 \\
& 8 & 1.060 & 0.427 & -4.697 & 1.472 & 10.000 & -7.288 \\
& 9 & 1.035 & 0.651 & -5.296 & 0.695 & 10.000 & -6.123 \\
& 10 & 1.027 & 0.506 & -4.613 & 1.888 & 10.000 & -7.839 \\
\hline
& 1 & 1.199 & 1.199 & -0.405 & -0.405 & -0.405 & -0.405 \\
& 2 & 1.168 & 1.717 & -4.360 & 0.767 & 10.000 & -8.393 \\
& 3 & 1.897 & -0.485 & -4.022 & 2.809 & 10.000 & -9.167 \\
& 4 & 1.702 & -0.140 & -4.187 & 2.457 & 10.000 & -8.854 \\
\{\boldsymbol{M}\}^{1/3} & 5 & 1.525 & 0.087 & -4.142 & 2.415 & 10.000 & -8.921 \\
& 6 & 1.392 & 0.406 & -4.142 & 2.750 & 10.000 & -9.500 \\
& 7 & 1.292 & 0.384 & -3.870 & 3.122 & 10.000 & -10.000 \\
& 8 & 1.245 & 0.809 & -4.187 & 3.013 & 10.000 & -10.000 \\
& 9 & 1.173 & 0.517 & -3.433 & 3.409 & 9.255 & -10.000 \\
& 10 & 1.180 & 1.091 & -4.355 & 2.894 & 10.000 & -9.933 \\
\hline
& 1 & 1.385 & 1.385 & -0.517 & -0.517 & -0.517 & -0.517 \\
& 2 & 1.398 & 1.910 & -3.967 & 1.031 & 10.000 & -9.554 \\
& 3 & 2.155 & -0.986 & -3.272 & 3.804 & 9.311 & -10.000 \\
& 4 & 1.957 & -0.663 & -3.329 & 3.625 & 9.393 & -10.000 \\
\{\boldsymbol{M}\}^{1/4} & 5 & 1.789 & -0.099 & -3.540 & 3.340 & 9.381 & -10.000 \\
& 6 & 1.640 & 0.020 & -3.559 & 3.267 & 9.537 & -10.000 \\
& 7 & 1.517 & 0.324 & -3.590 & 3.190 & 9.429 & -10.000 \\
& 8 & 1.405 & 0.343 & -3.395 & 3.280 & 9.260 & -10.000 \\
& 9 & 1.345 & 0.651 & -3.514 & 3.224 & 9.145 & -10.000 \\
& 10 & 1.258 & 0.748 & -3.030 & 3.640 & 8.201 & -10.000 \\
\hline
\end{array}$$

## 文章小结

本文提出了一种基于贪心策略的通用矩阵函数近似框架——不追求严格的可解释性，而是直接构造简单的多项式迭代，基于贪心策略每一步都去回归目标，然后选择适当的成本函数，转化成二次规划或线性规划问题，精确、稳定地求解迭代参数，最终获得一个有效的计算方案。
