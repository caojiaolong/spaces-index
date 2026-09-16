# 基于流式幂迭代的Muon实现：2\. 加速

> 作者：苏剑林 · 科学空间 · 2026-03-26
>
> 原文：<https://spaces.ac.cn/archives/11673>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在第一篇文章[《基于流式幂迭代的Muon实现：1\. 初识》](<https://spaces.ac.cn/archives/11654>)中，笔者将流式幂迭代（Streaming Power Iteration）单独抽象出来，作为一种新的Muon实现方式。由于新方案是直接对SVD进行近似计算，所以相比基于Newton\-Schulz迭代的标准实现，它具有更丰富的拓展空间，值得继续深入研究。

从计算上看，新方案的主要变化是Newton\-Schulz迭代换成了$\newcommand{QR}{\mathop{\text{QR}}}\QR$分解，这带来了一些降速。上篇我们已经讨论了一些基本的加速手段，但尚未比肩标准实现。这篇文章我们继续研究$\QR$的加速，以求尽可能缩小差距。

## 流式迭代

我们将沿用第一篇文章的所有概念和记号，有相关疑惑的读者请先往前翻看一下。首先，Muon的更新公式是
\begin{equation}\newcommand{msign}{\mathop{\text{msign}}}\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t [\msign(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}] \\
\end{aligned}\end{equation}
其中$\msign$的标准实现是[Newton\-Schulz迭代](<https://spaces.ac.cn/archives/10996>)，这也是Muon优化器中最昂贵的计算。相比之下，流式幂迭代方案的更新公式是
\begin{equation}\newcommand{ColNorm}{\mathop{\text{ColNorm}}}\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{V}_t =&\, \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}) \\[5pt]
\boldsymbol{U}_t =&\, \ColNorm(\boldsymbol{M}_t\boldsymbol{V}_t) \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t (\boldsymbol{U}_t\boldsymbol{V}_t^{\top} + \lambda \boldsymbol{W}_{t-1}) \\
\end{aligned}\end{equation}
如果反复执行$\boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1})$，那么就是标准的幂迭代，结果将收敛到$\boldsymbol{M}_t$的右奇异矩阵，从而实现$\boldsymbol{M}_t$的SVD，继而计算$\msign$。然而每步都执行完整的幂迭代成本过大，我们改为缓存上一步的结果$\boldsymbol{V}_{t-1}$，然后每步只迭代一次$\QR$作为近似，这就是“流式”的含义。

现在最昂贵的运算变为$\QR$分解，最朴素的实现自然是调用自带的QR函数，其背后的原理是Householder变换，稳定性很好，但速度较慢。

## 首次提速

为了加速，上篇中我们引入了Cholesky QR，它将矩阵$\boldsymbol{A}$的QR分解分为两步：1\. 对$\boldsymbol{A}^{\top}\boldsymbol{A}$做Cholesky分解得到上三角阵$\boldsymbol{R}$；2\. 解方程$\boldsymbol{Q}\boldsymbol{R}=\boldsymbol{A}$得到正交矩阵$\boldsymbol{Q}$。这两步理论上都非常高效，但实际计算如果条件数过大会失败。为此，我们又引入了[Shift技巧](<https://papers.cool/arxiv/1809.11085>)，它给$\boldsymbol{A}^{\top}\boldsymbol{A}$加上正则项$\lambda \boldsymbol{I}$（$\lambda=\epsilon \Vert\boldsymbol{A}^{\top}\boldsymbol{A}\Vert_F$）来降低条件数。

两者叠加起来，我们简称为“SCQR（Shifted Cholesky QR）”，一个基于Jax的参考实现如下：



<pre><code class="language-python">import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax import lax

def shift(A, eps=1e-9):
    return A + eps * jnp.linalg.matrix_norm(A, keepdims=True) * jnp.eye(A.shape[-1])

def scqr(A, eps=1e-9):
    """先按Shifted Cholesky QR算，失败则回退到默认QR
    """
    R = jnp.linalg.cholesky(shift(A.mT @ A, eps), upper=True)
    Q = solve_triangular(R.mT, A.mT, lower=True).mT
    return lax.cond(jnp.isfinite(Q).all(), lambda: Q, lambda: jnp.linalg.qr(A)[0])</code></pre>



注意，$\lambda$越小，SCQR越容易失败，而$\lambda$越大，结果越偏离正交，效果会越差，所以$\lambda$必须“恰到好处”，这就导致现在这个方案依然有较大几率回退到标准QR。此外，上篇还提到给幂迭代加上$\ColNorm$（即变为$\boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\ColNorm(\boldsymbol{M}_t\boldsymbol{V}_{t-1}))$）能稳定训练效果，SCQR下作用更为明显。

## 精度全开

以上基本就是第一篇文章的全部内容，只能说它确实把全流程跑通了，验证了可行性，并且相比直接调用框架自带的QR分解函数，SCQR也提供了一些加速，但速度还是明显慢于Newton\-Schulz迭代实现的$\msign$，所以还是得想办法提速。

这一节介绍第一个加速技巧——<strong>打开“完全体”的FP32精度矩阵乘法</strong>。首先要指出的是，流式幂迭代新增的几个步骤，都是在FP32精度下进行计算的。然而，从A100引入TF32格式开始，有些框架（比如笔者跑小实验的Jax，或者某些版本的Torch）的FP32数组在做矩阵乘法时，默认配置下会转成TF32格式来加速，需要手动开启才能实现真正的FP32精度相乘。

可能有读者疑问，提高乘法精度不应该降速吗，怎么反而提速了？这一点确实很反直觉，但其实也不难理解，降低矩阵精度往往会增大它的条件数，从而增加SCQR失败的概率，提高到标准QR的几率，从而增加耗时；相反，提高精度则可以增加SCQR的成功率，而QR恰恰是最耗时的地方，所以总耗时反而变短了。

资料显示，Jax一直以来都是默认按TF32来做FP32乘法的，所以它要通过<code>jax.config.update('jax_default_matmul_precision', 'highest')</code>来手动打开；Torch则复杂一些，1\.7到1\.11版本默认是TF32乘法，但从1\.12开始则默认FP32乘法。考虑到Torch现在已经2\.11版，估计对大部分用户来说都不用手动打开了。

## 双正交化

笔者想到的第二个加速技巧是<strong>多加一步对左特征矩阵的正交化</strong>，即将幂迭代这一步改为
\begin{equation}\boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\QR(\boldsymbol{M}_t\boldsymbol{V}_{t-1}))\end{equation}
这一步同样很反直觉，增加一次$\QR$，最后速度反而变快了，理由跟上一节是类似的，都是通过降低待分解矩阵的条件数，从而增加SCQR的成功率。由于SCQR本身非常快，所以执行两次也没增加多少时间，反而由于大大减少回退到标准QR的次数而明显提速。

理解这个加速技巧分两步，一是多加这步$\QR$理论上不改变幂迭代，二是多加这步$\QR$确实降低了条件数。第一个问题很好理解，若$\boldsymbol{A}=\boldsymbol{Q}\boldsymbol{R}$，那么$\boldsymbol{Q}=\boldsymbol{A}\boldsymbol{R}^{-1}$，其中$\boldsymbol{R}^{-1}$也是一个上三角阵，即QR分解可以写成右乘上三角阵的形式，那么
\begin{equation}\boldsymbol{M}_t^{\top}\QR(\boldsymbol{M}_t\boldsymbol{V}_{t-1}) = \boldsymbol{M}_t^{\top}(\boldsymbol{M}_t\boldsymbol{V}_{t-1}\times \text{某个上三角阵}) = \boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}\times \text{某个上三角阵} \end{equation}
由QR分解的唯一性，右乘上三角阵不改变$\QR$的结果，所以理论上跟$\QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1})$等价。

至于条件数，它等于矩阵的最大奇异值与最小奇异值之比，如果单次$\QR$，那么Cholesky分解矩阵是$\boldsymbol{V}_{t-1}^{\top}(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t)^2\boldsymbol{V}_{t-1}$，注意正交变换不改变奇异值，也就不改变条件数，所以此时待分解矩阵条件数达到了$\boldsymbol{M}_t$条件数的4次方！如果两次$\QR$，那么待Cholesky分解矩阵将是$\boldsymbol{Q}_t^{\top}(\boldsymbol{M}_t\boldsymbol{M}_t^{\top})\boldsymbol{Q}_t$，其中$\boldsymbol{Q}_t$是第一次$\QR$的正交矩阵，此时条件数只是$\boldsymbol{M}_t$的平方，明显降低。

## 平移不变

第三个加速技巧是笔者跟 [@YouJiacheng](<https://x.com/YouJiacheng>) 讨论过程中得出的，它<strong>利用特征矩阵的平移不变性</strong>。我们知道，幂迭代$\boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1})$也可以理解为求矩阵正定矩阵$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t$的特征矩阵，而正定矩阵有个性质——给它加上单位阵的若干倍，特征矩阵不变。

换句话说，$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t$跟$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t + \lambda \boldsymbol{I}$具有相同的特征矩阵，所以我们可以将幂迭代改为
\begin{equation}\boldsymbol{V}_t = \QR((\boldsymbol{M}_t^{\top}\boldsymbol{M}_t + \lambda \boldsymbol{I})\boldsymbol{V}_{t-1}) = \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1} + \lambda \boldsymbol{V}_{t-1})\end{equation}
而不改变幂迭代的收敛结果。那给$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t$加上$\lambda \boldsymbol{I}$有什么好处呢？答案同样是为了降低条件数，即$(\sigma_{\max} + \lambda)/(\sigma_{\min} + \lambda) < \sigma_{\max}/\sigma_{\min}$，所以也能提高Cholesky QR的成功率。注意这里我们说的是Cholesky QR而不是SCQR，因为在外边设置适当的$\lambda$就可以保证条件数，不需要再Shift了，这样出来的结果就一定正交，也是一个比较好的性质。

但别高兴得太早。$\lambda$越大，Cholesky QR自然是越容易成功，但是同时也会降低幂迭代的收敛速度！这是因为幂迭代的收敛速度取决于相邻奇异值之比，$\sigma_{i+1}/\sigma_i$越小收敛越快（奇异值从大到小排序），而$(\sigma_{i+1} + \lambda)/(\sigma_i + \lambda) > \sigma_{i+1}/\sigma_i$，所以$\lambda$越大，幂迭代收敛越慢，最终效果也会变差。

所以，我们必须小心调整$\lambda$的值，来平衡Cholesky QR的成功率和幂迭代的收敛速度，笔者测试发现，取$\lambda = \epsilon\Vert\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\Vert_F$且$\epsilon=10^{-4}$能够取得比较不错的结果。另外一个做法是使用较大的$\lambda$保证Cholesky QR成功率，然后迭代两步来提高幂迭代的收敛速度，即
\begin{equation}\boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\tilde{\boldsymbol{V}}_t + \lambda \tilde{\boldsymbol{V}}_t),\qquad \tilde{\boldsymbol{V}}_t = \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1} + \lambda \boldsymbol{V}_{t-1})\end{equation}
这样可以同时兼顾Cholesky QR和幂迭代，代价自然是每步需要两次$\QR$了。

## 多步修正

第四个加速技巧称为“SCQR2”，这是一种<strong>针对SCQR的通用的多步修正技巧</strong>。我们再来回顾一下SCQR的两个步骤（给定待分解矩阵$\boldsymbol{A}$）：
\begin{align}1)\quad&\,  \boldsymbol{R}^{\top}\boldsymbol{R}= \boldsymbol{A}^{\top}\boldsymbol{A} + \lambda \boldsymbol{I} &\,(\text{对}\boldsymbol{A}^{\top}\boldsymbol{A}+\lambda\boldsymbol{I}\text{做Cholesky分解}) \\[5pt]
2)\quad&\,   \boldsymbol{Q} = \boldsymbol{A}\boldsymbol{R}^{-1}&\,(\text{解三角型线性方程}\boldsymbol{Q}\boldsymbol{R}=\boldsymbol{A})\end{align}
SCQR的问题是，$\lambda$越大，Cholesky分解越容易成功，但是$\boldsymbol{Q} = \boldsymbol{A}\boldsymbol{R}^{-1}$就越不正交。SCQR2的想法是，先用较大的$\lambda$做一次SCQR，此时结果虽然不正交，但相比原来的$\boldsymbol{A}$会更加接近正交，说明已经降低了条件数，这时可以用较小的$\lambda$对结果再做一次SCQR，修正正交性，大致实现如下：



<pre><code class="language-python">def shift(A, eps=1e-9):
    return A + eps * jnp.linalg.matrix_norm(A, keepdims=True) * jnp.eye(A.shape[-1])

def scqr(A, eps=1e-9):
    """Shifted Cholesky QR
    """
    R = jnp.linalg.cholesky(shift(A.mT @ A, eps), upper=True)
    return solve_triangular(R.mT, A.mT, lower=True).mT

def scqr2(A, eps1=1e-4, eps2=1e-8):
    """SCQR两次，失败则回退到默认QR
    """
    Q = scqr(scqr(A, eps1), eps2)
    return lax.cond(jnp.isfinite(Q).all(), lambda: Q, lambda: jnp.linalg.qr(A)[0])</code></pre>



原理上，我们需要理解为什么二次修正是可行的。设第一次SCQR得到的是$\boldsymbol{Q}_1 = \boldsymbol{A}\boldsymbol{R}_1^{-1}$，它虽然偏离正交，但它具备“$\boldsymbol{A}\times \text{上三角阵}$”的形式，前面我们说了，右乘上三角阵不改变QR结果，所以这允许我们在第一次SCQR的基础上再执行一次SCQR。当然，原则上我们也可以做更多步的修正。

## 方法汇总

前面我们已经讨论了四个加速技巧，这里对它们的特性做一个简单的总结。

第一个技巧是提高FP32的矩阵乘法精度，这是通用的，Jax需要手动开启，而新版的Torch已经默认开启；第二、三、四个技巧都是孤立的，它们之间无法相互叠加。直觉上技巧二的上限更高，因为技巧三、四都是以$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}$为输入的，也就是说条件数已经放大了，然后试图把它补救回来，而方法二是修改输入为$\boldsymbol{M}_t^{\top}\QR(\boldsymbol{M}_t\boldsymbol{V}_{t-1})$，从源头上降低条件数。

有意思的是，技巧二、三、四似乎都指向了两次$\QR$。除了技巧三在精心调节$\lambda$时可以只用一次$\QR$外，剩下的都至少需要两次$\QR$，看起来这确实是最稳妥的选择了。速度上，技巧三如果能调到只用一次$\QR$，那么它是最快的，否则它跟技巧二一样快；技巧四则有些不稳定，如果对$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}$做SCQR2，那么速度很快，但效果不行，改为$\boldsymbol{M}_t^{\top}\ColNorm(\boldsymbol{M}_t\boldsymbol{V}_{t-1})$能保证效果，但速度会降下来。

笔者推荐技巧一、二的组合，它在效果和效率方面都比较有保证。单独测试下来，它的速度大概是$\msign$的Newton\-Schulz迭代的一半左右。可能有读者会想“费那么大劲才一半的速度？”，其实这已经很理想了，毕竟我们都是在FP32下计算，并且还要两次$\QR$。另一方面，$\msign$这一步的计算时间端到端占比也就1%左右，翻个倍也就是再多出1%的时间，尚可接受。

此外，Newton\-Schulz迭代的效率取决于迭代步数，如果用[Polar Express](<https://spaces.ac.cn/archives/10996>)的系数进一步增加步数来提高精度，那么它跟我们这里的速度差距也会进一步缩小。总之，流式幂迭代确实更慢了，但它也得到了更丰富、更准确的结果（SVD），能做更多的事情。

## 文章小结

这篇文章介绍了进一步流式幂迭代的技巧，其本质是设法降低矩阵的条件数，从而提高Cholesky QR的成功率。
