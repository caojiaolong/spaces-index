# 基于流式幂迭代的Muon实现：1\. 初识

> 作者：苏剑林 · 科学空间 · 2026-03-12
>
> 原文：<https://spaces.ac.cn/archives/11654>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[Muon](<https://spaces.ac.cn/archives/10592>)的核心运算是$\newcommand{msign}{\mathop{\text{msign}}}\msign$，当前标准实现是[Newton\-Schulz迭代](<https://spaces.ac.cn/archives/10996>)。不得不说，这确实是一个非常高效且GPU友好的算法，Muon能流行起来，起码有一大半是这个算法的功劳。然而，这个算法也给人一种“只此一家，别无分号”的感觉，因为它似乎就局限在算$\msign$了，一旦我们想要魔改Muon（比如$\msign$换成[这里](<https://spaces.ac.cn/archives/11059>)的$\newcommand{mclip}{\mathop{\text{mclip}}}\mclip$），那么相应的计算就会变得麻烦起来。

本文提出一种新的实现思路——通过流式幂迭代（Streaming Power Iteration）来近似计算SVD。这并不是完全新的思路，而是已经出现之前的一些优化器工作中，但这里我们将它单独提炼出来，作为一个独立的算法使用。

## 内容回顾

Muon的细节我们就不展开了，大家自行翻看之前的文章如[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)、[《Muon续集：为什么我们选择尝试Muon？》](<https://spaces.ac.cn/archives/10739>)、[《Muon优化器指南：快速上手与关键细节》](<https://spaces.ac.cn/archives/11416>)即可，这里直接给出它的公式：
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t [\msign(\boldsymbol{M}_t) + \lambda \boldsymbol{W}_{t-1}] \\
\end{aligned}\end{equation}
其中$\msign$为
\begin{equation}\msign(\boldsymbol{M})=\boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}=\boldsymbol{U}_{[:, :r]}\boldsymbol{V}_{[:, :r]}^{\top}\end{equation}
这里$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，不失一般性，约定$n\geq m$，并且简单起见，大多数情况下我们假设$r=m$（即满秩），只有在某些非常必要的情况下，才讨论不满秩的问题。

由于SVD比较昂贵，所以多数情况下我们都是采用Newton\-Schulz迭代去计算$\msign$，这我们在[《msign算子的Newton\-Schulz迭代（上）》](<https://spaces.ac.cn/archives/10922>)和[《msign算子的Newton\-Schulz迭代（下）》](<https://spaces.ac.cn/archives/10996>)已经有过详细讨论。总的来说，Newton\-Schulz迭代非常巧妙，是Muon成功的主要功臣，但它的可拓展性比较弱。

为了拓展Newton\-Schulz迭代的应用场景，笔者之前也做了一些工作，比如[《通过msign来计算奇异值裁剪mclip（上）》](<https://spaces.ac.cn/archives/11006>)、[《通过msign来计算奇异值裁剪mclip（下）》](<https://spaces.ac.cn/archives/11059>)、[《矩阵平方根和逆平方根的高效计算》](<https://spaces.ac.cn/archives/11158>)、[《矩阵r次方根和逆r次方根的高效计算》](<https://spaces.ac.cn/archives/11175>)等，但能做的事情总体还是比较有限。

很明显，一劳永逸的方法就是直接把SVD求出来，这也是接下来要聚焦的思路。

## 幂之迭代

在[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)、[《从谱范数梯度到新式权重衰减的思考》](<https://spaces.ac.cn/archives/10648>)等文章中，我们已经初步接触过[幂迭代（Power Iteration）](<https://en.wikipedia.org/wiki/Power_iteration>)，我们用它来求$\boldsymbol{M}^{\top}\boldsymbol{M}$的主特征向量，或者说$\boldsymbol{M}$的右主奇异向量，迭代格式如下：
\begin{equation}\boldsymbol{v}_1^{(t)} = \frac{\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{v}_1^{(t-1)}}{\Vert\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{v}_1^{(t-1)}\Vert_2}\end{equation}
假设我们已经求得了主特征向量$\boldsymbol{v}_1$，我们可以给幂迭代加上正交化来求次特征向量：
\begin{equation}\boldsymbol{v}_2^{(t)} = \frac{\tilde{\boldsymbol{v}}_2^{(t)} - \langle\tilde{\boldsymbol{v}}_2^{(t)},\boldsymbol{v}_1\rangle\boldsymbol{v}_1}{\Vert\tilde{\boldsymbol{v}}_2^{(t)} - \langle\tilde{\boldsymbol{v}}_2^{(t)},\boldsymbol{v}_1\rangle\boldsymbol{v}_1\Vert_2},\qquad \tilde{\boldsymbol{v}}_2^{(t)} = \boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{v}_2^{(t-1)}\end{equation}
由于保证了跟$\boldsymbol{v}_1$的正交性，这将收敛到次特征向量$\boldsymbol{v}_2$。类似地，假设已知$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_{k-1}$，我们可以配合Gram\-Schmidt正交化求第$k+1$个特征向量：
\begin{equation}\boldsymbol{v}_k^{(t)} = \frac{\tilde{\boldsymbol{v}}_k^{(t)} - \langle\tilde{\boldsymbol{v}}_k^{(t)},\boldsymbol{v}_1\rangle\boldsymbol{v}_1 - \cdots - \langle\tilde{\boldsymbol{v}}_k^{(t)},\boldsymbol{v}_{k-1}\rangle\boldsymbol{v}_{k-1}}{\Vert\tilde{\boldsymbol{v}}_k^{(t)} - \langle\tilde{\boldsymbol{v}}_k^{(t)},\boldsymbol{v}_1\rangle\boldsymbol{v}_1 - \cdots - \langle\tilde{\boldsymbol{v}}_k^{(t)},\boldsymbol{v}_{k-1}\rangle\boldsymbol{v}_{k-1}\Vert_2},\qquad \tilde{\boldsymbol{v}}_k^{(t)} = \boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{v}_k^{(t-1)}\label{eq:vk-pi}\end{equation}
实际上，我们不必要等到$\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_{k-1}$都算完后才算$\boldsymbol{v}_k$，全体$\boldsymbol{V}=[\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_m]$可以并行迭代。具体来说，从已有近似$\boldsymbol{V}_{t-1}$出发，批量算$\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{V}_{t-1}$，然后按列向量重新正交化（可以用QR分解），将会得到一个更好的近似，我们将其记为$\boldsymbol{V}_t$，反复迭代，最终将会收敛到我们的目标$\boldsymbol{V}$：
\begin{equation}\newcommand{QR}{\mathop{\text{QR}}}\boldsymbol{V}_t = \QR(\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{V}_{t-1})\end{equation}
这里$\QR$是指QR分解后的正交矩阵。有了$\boldsymbol{V}$之后，显然有$\newcommand{ColNorm}{\mathop{\text{ColNorm}}}\boldsymbol{U} = \ColNorm(\boldsymbol{M}\boldsymbol{V})$，其中$\ColNorm$是指每一列做L2 Normalize（axis=0），以及$\newcommand{diag}{\mathop{\text{diag}}}\boldsymbol{\Sigma}=\diag(\boldsymbol{U}^{\top}\boldsymbol{M}\boldsymbol{V})$，这样我们就得到SVD的一个基于幂迭代和QR分解的近似计算方案。当然，当$n > m$时它只能得到不完整的分解，其中$\boldsymbol{U}\in\mathbb{R}^{n\times m}$以及$\boldsymbol{\Sigma},\boldsymbol{V}\in\mathbb{R}^{m\times m}$，但已经够用了。

## 流式更新

然而，用幂迭代去计算SVD的实际效率极低，远远低于直接调用框架自带的SVD函数，所以这并不切实际。但是，考虑到训练本身就是一个长期迭代的过程，我们可以假设每一步的$\boldsymbol{V}$变化并不大，于是可以把上一步的$\boldsymbol{V}$存起来作为当前步的初始化，然后每一步只幂迭代一次，即
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{V}_t =&\, \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}) \\[5pt]
\boldsymbol{U}_t =&\, \ColNorm(\boldsymbol{M}_t\boldsymbol{V}_t) \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t (\boldsymbol{U}_t\boldsymbol{V}_t^{\top} + \lambda \boldsymbol{W}_{t-1}) \\
\end{aligned}\label{eq:muon-qr}\end{equation}
其中$\boldsymbol{V}_0=\boldsymbol{I}$。实测显示，通过这种流式幂迭代实现的Muon，在LM Loss上确实能跑出跟Newton\-Schulz版近乎重合的收敛曲线，这表明它确实是一个可行方案。这大体上是动量机制和小学习率的缘故，使得“每一步的$\boldsymbol{V}$变化并不大”的假设能够近似成立，从而允许将幂迭代成本“摊平”到每一步上。

得益于直接近似计算SVD，我们还可以给奇异值做一些操作，补充到优化器中去，例如
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{V}_t =&\, \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}) \\[5pt]
\boldsymbol{U}_t =&\, \ColNorm(\boldsymbol{M}_t\boldsymbol{V}_t) \\[5pt]
\boldsymbol{\Sigma}_t =&\, \diag(\boldsymbol{U}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_t) \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t (\boldsymbol{U}_t f(\boldsymbol{\Sigma}_t)\boldsymbol{V}_t^{\top} + \lambda \boldsymbol{W}_{t-1}) \\
\end{aligned}\end{equation}
这样我们要实现$\mclip$或者基于一般[Schatten范数](<https://en.wikipedia.org/wiki/Schatten_norm>)的Muon变体就轻松多了。总之，有了$\boldsymbol{U}_t,\boldsymbol{\Sigma}_t,\boldsymbol{V}_t$的显式结果（哪怕只是近似的）后，我们可以轻松尝试一些小修改，拓展性、可玩性明显增强。

## 加速分解

现在压力来到QR分解这边。式$\eqref{eq:muon-qr}$中最耗时的步骤是QR分解，标准实现是Householder QR，尽管它已经比SVD快上不少，但它仍然比Newton\-Schulz迭代计算的$\msign$要慢（多项式迭代且允许BF16乘法，简直是作弊级的存在）。因此，为了增强这个新方案的竞争力，我们还需要给QR分解提提速。

对于给定矩阵$\boldsymbol{A}\in\mathbb{R}^{n\times m}$（$n\geq m$），QR分解是想找到正交矩阵$\boldsymbol{Q}\in\mathbb{R}^{n\times m}$和上三角矩阵$\mathbb{R}\in\mathbb{R}^{m\times m}$，使得$\boldsymbol{A}=\boldsymbol{Q}\boldsymbol{R}$（这里的正交矩阵只需满足$\boldsymbol{Q}^{\top}\boldsymbol{Q}=\boldsymbol{I}$，更准确的称呼是Stiefel矩阵）。留意到$\boldsymbol{A}^{\top}\boldsymbol{A}=\boldsymbol{R}^{\top}\boldsymbol{R}$，也就是说只需将$\boldsymbol{A}^{\top}\boldsymbol{A}$分解成一个下三角矩阵及其转置的乘积，就可以得到$\boldsymbol{R}$了，而这正是[Cholesky分解](<https://en.wikipedia.org/wiki/Cholesky_decomposition>)要做的事情！

Cholesky分解是非常高效的，所以第一步我们可以用它来得到$\boldsymbol{R}$，然后就可以解方程$\boldsymbol{Q}\boldsymbol{R}=\boldsymbol{A}$得到$\boldsymbol{Q}$，方程又可以写成$\boldsymbol{R}^{\top}\boldsymbol{Q}^{\top}=\boldsymbol{A}^{\top}$，可以用 [solve\_triangular](<https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.solve_triangular.html>) 解决，这也是非常高效的。这两步组合下来，就构成了名为“Cholesky QR”的QR分解算法。如果不考虑数值上的稳定性，它可能是速度最快的QR分解方法。

很遗憾，相比标准的QR分解，Cholesky QR非常不稳定，它对$\boldsymbol{A}^{\top}\boldsymbol{A}$的[条件数](<https://en.wikipedia.org/wiki/Condition_number>)极其敏感。为此，[《Shifted CholeskyQR for computing the QR factorization of ill\-conditioned matrices》](<https://papers.cool/arxiv/1809.11085>)（简称“SCQR”）提出给$\boldsymbol{A}^{\top}\boldsymbol{A}$加一个正则项$\lambda \boldsymbol{I}$（$\lambda=\epsilon \Vert\boldsymbol{A}^{\top}\boldsymbol{A}\Vert_F$）来缓解这个问题。不过这也是一把双刃剑，$\epsilon$越大，SCQR越稳定，但所得的$\boldsymbol{Q}$会越不正交，最终效果会越差。

此外，即便引入了$\epsilon$，也无法保证SCQR一定能成功，所以我们还需要多加一步检测，若失败则回退到标准QR。

## 参考实现

一个基于Jax的简单参考实现如下：



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



简单的测试表明，如果能顺利执行，那么SCQR的效率跟Newton\-Schulz版$\msign$是相当的。然而，为了保证近似程度，$\epsilon$的取值不能太小，否则效果会明显下降，实测通常要取到$\epsilon=10^{-9}$时，效果才比较有保证，此时SCQR还是有比较大的概率回退到标准QR的，以至于最终速度跟Newton\-Schulz迭代还是有一定差距。

除了直接改进QR分解算法外，还有另外一些加速技巧，比如只保留前$k$个特征向量，那么$\boldsymbol{V}$就只需要初始化$m\times k$大小而不是$m\times m$，这样也能降低一些计算。关于QR分解的进一步加速，就留给大家继续探索了，这里就不继续展开。

## 其他细节

另外，还有一些细节需要特别留意一下，它们跟训练的稳定性和最终的效果都密切相关。

首先，按照$\boldsymbol{M}_t\in\mathbb{R}^{n\times m}$的约定，我们需要保证$n\geq m$，否则要转置一下。如果$n < m$，那么矩阵$\boldsymbol{M}_t^{\top}\boldsymbol{M}_t \boldsymbol{V}_{t-1}$必然是不满秩，对一个不满秩的矩阵去做QR分解是不适定的，尤其是SCQR更容易产生各种病态现象，最终降低效果。所以，确保$n\geq m$既能保证数值稳定、提升效果，还能加速，一举多得。

其次，实测发现给$\QR$这步多加个$\ColNorm$，对效果的提升也很有帮助
\begin{equation}\boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\boldsymbol{M}_t\boldsymbol{V}_{t-1}) \qquad\to\qquad \boldsymbol{V}_t = \QR(\boldsymbol{M}_t^{\top}\ColNorm(\boldsymbol{M}_t\boldsymbol{V}_{t-1}))\end{equation}
这只不过相当于把式$\eqref{eq:vk-pi}$中的$\tilde{\boldsymbol{v}}_k^{(t)} = \boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{v}_k^{(t-1)}$改为$\tilde{\boldsymbol{v}}_k^{(t)} = \boldsymbol{M}^{\top}(\boldsymbol{M}\boldsymbol{v}_k^{(t-1)} / \Vert\boldsymbol{M}\boldsymbol{v}_k^{(t-1)}\Vert_2)$，不改变幂迭代本身的收敛性。但实验显示，多这一步$\ColNorm$对训练效果有明显帮助，SCQR下更为显著，能明显缩小它与标准QR的效果差距。

可以证明，这一操作理论上不改变幂迭代，只会对QR的数值计算产生影响。根据实验观察，它实际上会使得SCQR回退到标准QR的概率更高（但不多，所以不会明显变慢），然后对标准QR也略有帮助，所以看上去还是改良了待QR分解的矩阵性态，获得了更好的QR质量。

## 相关工作

本文开头就说了，流式幂迭代这一思路，实际上已经多次出现在部分优化器工作中，如[《4\-bit Shampoo for Memory\-Efficient Network Training》](<https://papers.cool/arxiv/2405.18144>)、[《SOAP: Improving and Stabilizing Shampoo using Adam》](<https://papers.cool/arxiv/2409.11321>)、[《COSMOS: A Hybrid Adaptive Optimizer for Memory\-Efficient Training of LLMs》](<https://papers.cool/arxiv/2502.17410>)、[《Dion: Distributed Orthonormalized Updates》](<https://papers.cool/arxiv/2504.05295>)等。

事实上，将一个需要多步迭代才能收敛的算法，结合模型训练本身就需要长期更新的特点，改为每一步训练只迭代一次的流式版本来摊平成本，并不是一个太难想的思路，此前我们在[《流形上的最速下降：5\. 对偶梯度下降》](<https://spaces.ac.cn/archives/11388>)也已经尝试过。所以，已有这么多相关工作并不是一件让人意外的事情。

本文主要参考的文献是上月出的[《ARO: A New Lens On Matrix Optimization For Large Models》](<https://papers.cool/arxiv/2602.09006>)，这篇论文实际上已经包含了本文的大部分内容，并且还做了推广。推广的思路也非常值得思考，注意到现在Muon的更新量可以写成
\begin{equation}\ColNorm(\boldsymbol{M}_t\boldsymbol{V}_t)\boldsymbol{V}_t^{\top}\end{equation}
其中$\ColNorm(\boldsymbol{M}_t)$可以认为是一个基础优化器，它只对动量做简单的列归一化，没有太多竞争力，于是我们给它重新找一组正交基$\boldsymbol{V}_t$，在新基底下应用基础优化器后再恢复过来。现在我们知道，这其实就是Muon，相比直接$\ColNorm$确实强了不少。那么，接下来的想法很自然是：$\ColNorm$能否换成其他的基础优化器呢？对此，ARO提出了一般的优化器框架（旋转最速下降）
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta\boldsymbol{M}_{t-1} + \boldsymbol{G}_t \\[5pt]
\boldsymbol{R}_t =&\, \QR(\boldsymbol{M}_t^{\top}f(\boldsymbol{M}_t\boldsymbol{R}_{t-1})) \\[5pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta_t (f(\boldsymbol{M}_t\boldsymbol{R}_t)\boldsymbol{R}_t^{\top} + \lambda \boldsymbol{W}_{t-1}) \\
\end{aligned}\end{equation}
其中$f$代表任何矩阵函数，原本的记号$\boldsymbol{V}$换成了$\boldsymbol{R}$（Rotation）。关于旋转最速下降的更多内容，我们后面有机会再讨论。

## 文章小结

这篇文章主要介绍了通过流式幂迭代（Streaming Power Iteration）来计算SVD、继而实现Muon的思路，它每一步只需做一次QR分解，相比标准的Newton\-Schulz迭代实现，这一思路具有更灵活的拓展性。
