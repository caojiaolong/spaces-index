# msign算子的Newton\-Schulz迭代（上）

> 作者：苏剑林 · 科学空间 · 2025-05-11
>
> 原文：<https://spaces.ac.cn/archives/10922>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<div data-spaces-format="html-v1">
<p data-spaces-tag="p" data-spaces-attrs="{}">在之前的<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/10592&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/10592">《Muon优化器赏析：从向量到矩阵的本质跨越》</a>、<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/10739&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/10739">《Muon续集：为什么我们选择尝试Muon？》</a>等文章中，我们介绍了一个极具潜力、有望替代Adam的新兴优化器——“Muon”。随着相关研究的不断深入，Muon优化器受到的关注度也在日益增加。</p><p data-spaces-tag="p" data-spaces-attrs="{}">了解过Muon的读者都知道，Muon的核心运算是$\newcommand{msign}{\mathop{\text{msign}}}\msign$算子，为其寻找更高效的计算方法是学术社区的一个持续目标。本文将总结一下它的最新进展。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;写在前面&quot;}">写在前面</h2><p data-spaces-tag="p" data-spaces-attrs="{}">$\msign$的定义跟SVD密切相关。假设矩阵$\boldsymbol{M}\in\mathbb{R}^{n\times m}$，那么<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{U},\boldsymbol{\Sigma},\boldsymbol{V}^{\top} = \text{SVD}(\boldsymbol{M}) \quad\Rightarrow\quad \msign(\boldsymbol{M}) = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top}\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
其中$\boldsymbol{U}\in\mathbb{R}^{n\times n},\boldsymbol{\Sigma}\in\mathbb{R}^{n\times m},\boldsymbol{V}\in\mathbb{R}^{m\times m}$，$r$是$\boldsymbol{M}$的秩。简单来说，$\msign$就是把矩阵的所有非零奇异值都变成1后所得的新矩阵。基于SVD，我们还可以证明<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\text{msign}(\boldsymbol{M}) = (\boldsymbol{M}\boldsymbol{M}^{\top})^{-1/2}\boldsymbol{M}= \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
这里的$^{-1/2}$的矩阵的$-1/2$次幂。这个形式跟标量的$\mathop{\text{sign}}(x) = x / \sqrt{x^2}$很相似，所以笔者用了$\msign$这个名字。但要注意的是这跟维基的“<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://en.wikipedia.org/wiki/Matrix_sign_function&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://en.wikipedia.org/wiki/Matrix_sign_function">Matrix Sign</a>”不完全相同，维基的概念只适用于方阵，但当$\boldsymbol{M}$是对称矩阵时，两者是一致的。</p><p data-spaces-tag="p" data-spaces-attrs="{}">当$m=n=r$时，$\text{msign}(\boldsymbol{M})$还有一个意义是“最优正交近似”：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\text{msign}(\boldsymbol{M}) = \mathop{\text{argmin}}_{\boldsymbol{O}^{\top}\boldsymbol{O} = \boldsymbol{I}}\Vert \boldsymbol{M} - \boldsymbol{O}\Vert_F^2\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
证明过程可参考<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/10592&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/10592">《Muon优化器赏析：从向量到矩阵的本质跨越》</a>。因为这个特性，$\msign$也被称为“对称正交化”，这个名字最早出自<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://www.sciencedirect.com/science/article/abs/pii/S0065327608603391&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://www.sciencedirect.com/science/article/abs/pii/S0065327608603391">《On the Nonorthogonality Problem》</a>（参考维基百科的“<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://en.wikipedia.org/wiki/Orthogonalization&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://en.wikipedia.org/wiki/Orthogonalization">Orthogonalization</a>”条目）。</p><p data-spaces-tag="p" data-spaces-attrs="{}">最后，在<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/10795&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/10795">《高阶MuP：更简明但更高明的谱条件缩放》</a>中，$\msign$还被笔者视为“奇异值裁剪”的极限版本。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;迭代计算&quot;}">迭代计算</h2><p data-spaces-tag="p" data-spaces-attrs="{}">$\msign$由SVD定义，自然也可以直接用SVD来精确计算，然而精确的SVD计算复杂度比较高，所以实践中往往都是用“Newton-Schulz迭代”近似计算。</p><p data-spaces-tag="p" data-spaces-attrs="{}">Newton-Schulz迭代是求矩阵函数的常用迭代算法，在$\msign$这里它的迭代格式是<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\Vert\boldsymbol{M}\Vert_F},\qquad \boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2+\cdots\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
其中$\Vert\boldsymbol{M}\Vert_F$是$\boldsymbol{M}$的$F$范数，即所有元素的平方和的平方根，$(a,b,c,\cdots)$是待定系数，实际计算中我们需要截断有限项，常见的是2项或者3项，即如下二选一：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{gather}\boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) \\[8pt]<br data-spaces-tag="br" data-spaces-attrs="{}">
\boldsymbol{X}_{t+1} = a\boldsymbol{X}_t + b\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2\label{eq:power-5}\end{gather}<br data-spaces-tag="br" data-spaces-attrs="{}">
最后返回$T$步迭代后的$\boldsymbol{X}_T$作为$\msign(\boldsymbol{M})$的近似。这样一来，系数$(a,b,c)$和迭代步数$T$就构成了Newton-Schulz迭代的全部超参数，Muon作者<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://github.com/KellerJordan/Muon&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://github.com/KellerJordan/Muon">KellerJordan</a>给出的参考选择是<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}(a,b,c)=(3.4445, -4.7750, 2.0315),\qquad T = 5\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
接下来我们的主题就是理解它，然后尝试改进它。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;参考实现&quot;}">参考实现</h2><p data-spaces-tag="p" data-spaces-attrs="{}">这里给出一个极简的参考实现：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">def msign(x, steps=5, eps=1e-20):
    a, b, c, y = 3.4445, -4.7750, 2.0315, x.astype('bfloat16')
    y = y.mT if x.shape[-2] &gt; x.shape[-1] else y
    y /= ((y**2).sum(axis=(-2, -1), keepdims=True) + eps)**0.5
    for _ in range(steps):
        y = a * y + (b * (y2 := y @ y.mT) + c * y2 @ y2) @ y
    return y.mT if x.shape[-2] &gt; x.shape[-1] else y</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">这个实现已经包含了batch运行能力（只对最后两个dims做$\msign$），可以在Jax跑通；如果将<code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">x.astype('bfloat16')</code>改为<code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">x.to(torch.bfloat16)</code>可以在Torch跑通；直接将<code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">x.astype('bfloat16')</code>改为<code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">x</code>也可以在Numpy跑通。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;原理分析&quot;}">原理分析</h2><p data-spaces-tag="p" data-spaces-attrs="{}">为了理解Newton-Schulz迭代的原理，我们的逐一分析它的步骤。首先是$\boldsymbol{X}_0 = \boldsymbol{M}/\Vert\boldsymbol{M}\Vert_F$，我们代入$\boldsymbol{M}$的SVD：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\Vert\boldsymbol{M}\Vert_F} = \boldsymbol{U}_{[:,:r]}\left(\frac{\boldsymbol{\Sigma}_{[:r,:r]}}{\Vert\boldsymbol{M}\Vert_F}\right)\boldsymbol{V}_{[:,:r]}^{\top} = \boldsymbol{U}_{[:,:r]}\underbrace{\left(\frac{\boldsymbol{\Sigma}_{[:r,:r]}}{\Vert\boldsymbol{\Sigma}_{[:r,:r]}\Vert_F}\right)}_{\boldsymbol{S}_0}\boldsymbol{V}_{[:,:r]}^{\top}\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
最后一个等号，是因为$F$范数的平方，既等于全体分量的平方和，又等于全体奇异值的平方和。最后的结果表明$\boldsymbol{S}_0$是一个分量均在$[0,1]$内的对角阵，换言之$\boldsymbol{X}_0=\boldsymbol{U}_{[:,:r]}\boldsymbol{S}_0\boldsymbol{V}_{[:,:r]}^{\top}$的全体奇异值都不超过1，这就是第一步$\boldsymbol{X}_0 = \boldsymbol{M}/\Vert\boldsymbol{M}\Vert_F$的目的。</p><p data-spaces-tag="p" data-spaces-attrs="{}">接着，我们代入$\boldsymbol{U}_{[:,:r]}\boldsymbol{S}_t\boldsymbol{V}_{[:,:r]}^{\top}$到式$\eqref{eq:power-5}$，将会得到<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{X}_{t+1} = \boldsymbol{U}_{[:,:r]}\left(a\boldsymbol{S}_t + b\boldsymbol{S}_t^3 + c\boldsymbol{S}_t^5\right)\boldsymbol{V}_{[:,:r]}^{\top}\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
也就是说，迭代不改变左右的$\boldsymbol{U}_{[:,:r]}$和$\boldsymbol{V}_{[:,:r]}^{\top}$，本质上式对角阵的迭代<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{S}_{t+1} = a\boldsymbol{S}_t + b\boldsymbol{S}_t^3 + c\boldsymbol{S}_t^5\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
然后对角阵的幂又等价于对角线元素各自取幂，所以这本质又等价于标量$x_t$的迭代<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}x_{t+1} = a x_t + b x_t^3 + c x_t^5\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
由于$\boldsymbol{X}_0 = \boldsymbol{M}/\Vert\boldsymbol{M}\Vert_F$已经将奇异值都压缩在$(0,1]$内，所以我们希望从任意$x_0\in(0,1]$出发，经过$T$步迭代后，$x_T$能够尽可能接近于1，那么迭代$\eqref{eq:power-5}$就可以足够近似$\msign$。这样一来，我们将矩阵的迭代分析简化为标量的迭代分析，大大降低了分析难度。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;优化求解&quot;}">优化求解</h2><p data-spaces-tag="p" data-spaces-attrs="{}">其实$a,b,c$的求解，我们在<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/10592&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/10592">《Muon优化器赏析：从向量到矩阵的本质跨越》</a>首次介绍Muon时也简单讨论过，基本思路是将$a,b,c$视为优化参数，用$x_T$与$1$的差来构建Loss，然后用SGD来优化。</p><p data-spaces-tag="p" data-spaces-attrs="{}">本文的思路大致相同，但稍作调整。很明显，优化结果将会依赖于奇异值分布，之前笔者的思路是用随机矩阵SVD来模拟真实奇异值分布，但SVD费时费力，并且结果还会依赖于矩阵的shape，现在看来没太大必要，我们改为在$[0,1]$内均匀取点，然后选择$|x_T-1|$最大的$k$个点来构建Loss，这样转化为一个$\min\text{-}\max$问题，尽可能减轻奇异值分布的影响：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">import jax
import jax.numpy as jnp
from tqdm import tqdm

def loss(w, x, k=50):
    for a, b, c in [w] * iters:
        x = a * x + b * x**3 + c * x**5
    return jnp.abs(x - 1).sort()[-k:].mean()

@jax.jit
def grad(w, x, tol=0.1):
    G = lambda w, x: (g := jax.grad(loss)(w, x)) / jnp.fmax(jnp.linalg.norm(g), 1)
    return 0.6 * G(w, x) + 0.2 * (G(w + tol / 2, x) + G(w - tol / 2, x))

iters = 5
x = jnp.linspace(0, 1, 10001)[1:]
w = jnp.array([1.5, -0.5, 0])
m, v = jnp.zeros_like(w), jnp.zeros_like(w)
lr = 1e-3
pbar = tqdm(range(20000), ncols=0, desc='Adam')

for i in pbar:
    l, g = loss(w, x), grad(w, x)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g**2
    w = w - lr * m / jnp.sqrt(v + 1e-20)
    pbar.set_description(f'Loss: {l:.6f}, LR: {lr:.6f}')
    if i in [10000]:
        lr *= 0.1</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">此外，优化器也从SGD改为Adam，这比较容易控制参数的更新幅度，同时为了增强解的噪声抵御能力，我们给$a,b,c$加上一定扰动，然后将扰动后的梯度也混合进来。上述脚本的优化结果是：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}(a,b,c)=(3.3748, -4.6969, 2.1433)\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
可以看到跟KellerJordan解相差不远。进一步通过图像比较一下两者差异：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/812396616.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/812396616.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 1]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/812396616.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/812396616.png" alt="[0, 1]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 1]的近似效果</p></div><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/519231176.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/519231176.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 0.01]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/519231176.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/519231176.png" alt="[0, 0.01]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 0.01]的近似效果</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">可以看到，全局来看，我们这里求出的解平均误差小一点，KellerJordan解的好处则是在$[0, 0.01]$区间内的斜率大一点，这意味着它对更小的奇异值会更有利。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;初值分布&quot;}">初值分布</h2><p data-spaces-tag="p" data-spaces-attrs="{}">在进一步讨论之前，我们需要明确一个问题：我们究竟要关心多小的奇异值？这要回到$\boldsymbol{S}_0$的分布上。由于$\boldsymbol{S}_0$经过$F$范数归一化，所以$\mathop{\text{diag}}(\boldsymbol{S}_0)$实际上是一个$r$维单位向量。如果全体奇异值都相等，那么可以推出每个奇异值都是$1/\sqrt{r}$。</p><p data-spaces-tag="p" data-spaces-attrs="{}">于是，根据鸽笼原理，我们得到非均匀的情况下必然存在少于$1/\sqrt{r}$的奇异值。保险起见，我们可以考虑一个倍数，比如10倍，这意味着我们至少要兼顾到$0.1/\sqrt{r}$大小的奇异值。实际情况中，一个矩阵严格低秩（即奇异值严格等于0）的概率是很小的，所以我们一般都是假设矩阵满秩，即$r = \min(n,m)$，因此至少要兼顾$0.1/\sqrt{\min(n,m)}$的奇异值。</p><p data-spaces-tag="p" data-spaces-attrs="{}">考虑到现在最大的LLM，hidden_size已经来到了$8192\sim 100^2$这个级别，所以根据这个数值估计，一个通用的Muon优化器，它的$\msign$算法至少要兼顾到$0.001$大小的奇异值，即能够将$0.001$映射到接近于1的值，这样看来不管是KellerJordan解还是我们新求出的解，都差点意思。</p><p data-spaces-tag="p" data-spaces-attrs="{}">注：关于初值分布的讨论，我们还可以参考<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://papers.cool/arxiv/2505.04005&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://papers.cool/arxiv/2505.04005">《Iterative Orthogonalization Scaling Laws》</a>。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;解开约束&quot;}">解开约束</h2><p data-spaces-tag="p" data-spaces-attrs="{}">这时候，推特上<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://twitter.com/YouJiacheng/status/1893704552689303901&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://twitter.com/YouJiacheng/status/1893704552689303901">@YouJiacheng</a>（Muon主要推动者之一）提出了一个非常机智的想法：每一步迭代我们可以使用不同的系数！也就是将迭代改为<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{X}_{t+1} = a_{t+1}\boldsymbol{X}_t + b_{t+1}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t) + c_{t+1}\boldsymbol{X}_t(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)^2\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
这样改动的好处是当选定$T$后，总计算量完全不会有任何变化，但从拟合的角度看，原本只有$3$个训练参数，现在变成了$3T$个，拟合能力将会大大加强。他本人给出的参考结果是一个6步迭代：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{array}{c|ccc}<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
t &amp; a &amp; b &amp; c \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\quad 1\quad &amp; 3955/1024 &amp; -8306/1024 &amp; 5008/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
2 &amp; 3735/1024 &amp; -6681/1024 &amp; 3463/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
3 &amp; 3799/1024 &amp; -6499/1024 &amp; 3211/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
4 &amp; 4019/1024 &amp; -6385/1024 &amp; 2906/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
5 &amp; 2677/1024 &amp; -3029/1024 &amp; 1162/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
6 &amp; 2172/1024 &amp; -1833/1024 &amp;  682/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{array}<br data-spaces-tag="br" data-spaces-attrs="{}">
我们可以画出来对比一下：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/350533843.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/350533843.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 1]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/350533843.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/350533843.png" alt="[0, 1]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 1]的近似效果</p></div><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/3892110485.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/3892110485.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 0.01]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/3892110485.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/3892110485.png" alt="[0, 0.01]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 0.01]的近似效果</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">公平起见，KellerJordan解和Ours解也都改为了$T=6$。可见，不管是从光滑程度还是整体近似程度来看，YouJiacheng解都有非常明显的提升，这充分体现了去掉参数共享后所释放的“完全体”威力。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;自己试试&quot;}">自己试试</h2><p data-spaces-tag="p" data-spaces-attrs="{}">YouJiacheng解是怎么求出来的呢？作者在<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae">这里</a>分享了他的代码，思路也是用Adam来求解，但包含了很多不同的Loss，理解起来有点麻烦。事实上，用我们前面的脚本，配合他提供的初始化，可以得到同样好的结果：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{array}{c|ccc}<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
t &amp; a &amp; b &amp; c \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\quad 1\quad &amp; 4140/1024 &amp; -7553/1024 &amp; 3571/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
2 &amp; 3892/1024 &amp; -6637/1024 &amp; 2973/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
3 &amp; 3668/1024 &amp; -6456/1024 &amp; 3021/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
4 &amp; 3248/1024 &amp; -6211/1024 &amp; 3292/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
5 &amp; 2792/1024 &amp; -5759/1024 &amp; 3796/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
6 &amp; 3176/1024 &amp; -5507/1024 &amp; 4048/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{array}</p><p data-spaces-tag="p" data-spaces-attrs="{}">参考代码：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">import jax
import jax.numpy as jnp
from tqdm import tqdm

def loss(w, x, k=50):
    for a, b, c in w:
        x = a * x + b * x**3 + c * x**5
    return jnp.abs(x - 1).sort()[-k:].mean()

@jax.jit
def grad(w, x, tol=0.1):
    G = lambda w, x: (g := jax.grad(loss)(w, x)) / jnp.fmax(jnp.linalg.norm(g), 1)
    return 0.6 * G(w, x) + 0.2 * (G(w + tol / 2, x) + G(w - tol / 2, x))

iters = 6
x = jnp.linspace(0, 1, 10001)[1:]
w = jnp.array([[3.5, -6.04444444444, 2.84444444444]] * iters)
m, v = jnp.zeros_like(w), jnp.zeros_like(w)
lr = 1e-3
pbar = tqdm(range(20000), ncols=0, desc='Adam')

for i in pbar:
    l, g = loss(w, x), grad(w, x)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g**2
    w = w - lr * m / jnp.sqrt(v + 1e-20)
    pbar.set_description(f'Loss: {l:.6f}, LR: {lr:.6f}')
    if i in [10000]:
        lr *= 0.1</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">对比如下（标记为“Ours-X”）：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/518737221.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/518737221.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 1]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/518737221.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/518737221.png" alt="[0, 1]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 1]的近似效果</p></div><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/3430771328.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/3430771328.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 0.01]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/3430771328.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/3430771328.png" alt="[0, 0.01]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 0.01]的近似效果</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">由图可见，相比YouJiacheng解，我们的结果振荡更多，但换来了$[0,0.001]$处更大的斜率。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;其他解集&quot;}">其他解集</h2><p data-spaces-tag="p" data-spaces-attrs="{}">如果读者想要振荡更少的解，那么只需要调大$k$值，比如$k=200$的结果是：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{array}{c|ccc}<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
t &amp; a &amp; b &amp; c \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\quad 1\quad &amp; 4059/1024 &amp; -7178/1024 &amp; 3279/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
2 &amp; 3809/1024 &amp; -6501/1024 &amp; 2925/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
3 &amp; 3488/1024 &amp; -6308/1024 &amp; 3063/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
4 &amp; 2924/1024 &amp; -5982/1024 &amp; 3514/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
5 &amp; 2439/1024 &amp; -5439/1024 &amp; 4261/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
6 &amp; 3148/1024 &amp; -5464/1024 &amp; 4095/1024 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{array}</p><p data-spaces-tag="p" data-spaces-attrs="{}">这时候就跟YouJiacheng解相差无几了（Ours-X2）：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/1050151153.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/1050151153.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 1]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/1050151153.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/1050151153.png" alt="[0, 1]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 1]的近似效果</p></div><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/617077151.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/617077151.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 0.01]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/617077151.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/617077151.png" alt="[0, 0.01]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 0.01]的近似效果</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">另外再给出一个5步的解，方便大家跟原始解对比：<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{array}{c|ccc}<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
t &amp; a &amp; b &amp; c \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\quad 1\quad &amp; 4.6182 &amp; -12.9582 &amp; 9.3299 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
2 &amp; 3.8496 &amp; -7.9585 &amp; 4.3052 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
3 &amp; 3.5204 &amp; -7.2918 &amp; 4.0606 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
4 &amp; 3.2067 &amp; -6.8243 &amp; 4.2802 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
5 &amp; 3.2978 &amp; -5.7848 &amp; 3.8917 \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\hline<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{array}<br data-spaces-tag="br" data-spaces-attrs="{}">
效果图（Ours-X3）：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/2624824428.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/2624824428.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 1]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/2624824428.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/2624824428.png" alt="[0, 1]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 1]的近似效果</p></div><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 300px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2025/05/1368764254.png&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2025/05/1368764254.png" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;[0, 0.01]的近似效果&quot;, &quot;src&quot;: &quot;/usr/uploads/2025/05/1368764254.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2025/05/1368764254.png" alt="[0, 0.01]的近似效果"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">[0, 0.01]的近似效果</p></div></div><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;改良初值&quot;}">改良初值</h2><p data-spaces-tag="p" data-spaces-attrs="{}">至此，我们关于$a,b,c$的求解告一段落。总的来说，每一步使用不同的$a,b,c$，确实能大幅提高Newton-Schulz迭代的收敛性质，并且不增加任何计算成本，算得上免费午餐了。</p><p data-spaces-tag="p" data-spaces-attrs="{}">除了优化Newton-Schulz迭代的系数外，还有其他思路可以改进迭代的收敛性质吗？还真有。<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://github.com/KellerJordan/modded-nanogpt/discussions/23#discussioncomment-11293594&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://github.com/KellerJordan/modded-nanogpt/discussions/23#discussioncomment-11293594">@johanwind</a>、<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://x.com/YouJiacheng/status/1866505635329978803&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://x.com/YouJiacheng/status/1866505635329978803">@YouJiacheng</a>、<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://x.com/ZhangRuichong/status/1866496714733211809&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://x.com/ZhangRuichong/status/1866496714733211809">@ZhangRuichong</a>等人发现，我们可以利用Newton-Schulz迭代的特点，近乎免费地提高初值的质量，从而提高收敛速度。<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://x.com/leloykun&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://x.com/leloykun">@leloykun</a>则在<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://github.com/KellerJordan/Muon/pull/14/files&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://github.com/KellerJordan/Muon/pull/14/files">这里</a>给出了一个参考实现。</p><p data-spaces-tag="p" data-spaces-attrs="{}">具体来说，目前改进Newton-Schulz迭代的主要努力都可以总结为“在保证收敛的前提下，尽可能地提高接近于零的奇异值的收敛速度”。如果我们能事先把这些接近于零的奇异值放大一点，那么在不改变迭代算法的前提下也能提高收敛速度。目前为了将奇异值压缩到$[0,1]$内，我们使用的是$F$范数归一化$\boldsymbol{M}/\Vert\boldsymbol{M}\Vert_F$，它将奇异值压缩成<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\sigma_i \quad\to\quad \frac{\sigma_i}{\Vert\boldsymbol{M}\Vert_F} = \frac{\sigma_i}{\sqrt{\sum\limits_{j=1}^r \sigma_i^2}} \in [0, 1]\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
这样做确实能达到目标，但也有压缩过度的问题，最紧凑的压缩方式应该是$\sigma_i\to \sigma_i/\sigma_1$，即谱归一化。问题是谱范数不如$F$范数容易计算，所以我们不得已选择了$F$范数。但是，我们有<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\sigma_1 \quad\leq\quad \underbrace{\sqrt[\uproot{10}8]{\sum_{j=1}^r \sigma_i^8}}_{\sqrt[4]{\Vert(\boldsymbol{M}^{\top}\boldsymbol{M})^2\Vert_F}}\quad\leq\quad \underbrace{\sqrt[\uproot{10}4]{\sum_{j=1}^r \sigma_i^4}}_{\sqrt{\Vert\boldsymbol{M}^{\top}\boldsymbol{M}\Vert_F}} \quad\leq\quad \underbrace{\sqrt{\sum_{j=1}^r \sigma_i^2}}_{\Vert\boldsymbol{M}\Vert_F} \end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
这意味着用$\sqrt[4]{\Vert(\boldsymbol{M}^{\top}\boldsymbol{M})^2\Vert_F}$或$\sqrt{\Vert\boldsymbol{M}^{\top}\boldsymbol{M}\Vert_F}$作为归一化因子，理论上都比$\Vert\boldsymbol{M}\Vert_F$更好。非常巧妙的是，在Newton-Schulz迭代下，它们的计算是近乎免费的！为理解这一点，我们写出第一步迭代<br data-spaces-tag="br" data-spaces-attrs="{}">
\begin{equation}\boldsymbol{X}_0 = \frac{\boldsymbol{M}}{\Vert\boldsymbol{M}\Vert_F},\qquad \boldsymbol{X}_1 = a\boldsymbol{X}_0 + b\boldsymbol{X}_0(\boldsymbol{X}_0^{\top}\boldsymbol{X}_0) + c\boldsymbol{X}_0(\boldsymbol{X}_0^{\top}\boldsymbol{X}_0)^2\end{equation}<br data-spaces-tag="br" data-spaces-attrs="{}">
可以看到$\boldsymbol{X}_0^{\top}\boldsymbol{X}_0$和$(\boldsymbol{X}_0^{\top}\boldsymbol{X}_0)^2$是必然要算出来的，所以我们直接拿它们算$F$范数，然后重新归一化就行，参考代码：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">def msign(x, steps=5, eps=1e-20):
    a, b, c, y = 3.4445, -4.7750, 2.0315, x.astype('bfloat16')
    y = y.mT if x.shape[0] &gt; x.shape[1] else y
    y /= ((y**2).sum(axis=[-2, -1], keepdims=True) + eps)**0.5
    for i in range(steps):
        y4 = (y2 := y @ y.mT) @ y2
        if i == 0:
            n = ((y4**2).sum(axis=[-2, -1], keepdims=True) + eps)**0.125
            y, y2, y4 = y / n, y2 / n**2, y4 / n**4
        y = a * y + (b * y2 + c * y4) @ y
    return y.mT if x.shape[0] &gt; x.shape[1] else y</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">实测结果，对于一个$100\times 100$的随机高斯矩阵，改进后的最小奇异值，大多数都在改进前的2倍以上，平均奇异值也更接近于1。不过，Muon作者也表示它可能会带来额外的不稳定性，因此还没有采纳到官方代码中。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;文章小结&quot;}">文章小结</h2><p data-spaces-tag="p" data-spaces-attrs="{}">本文介绍了通过Newton-Schulz迭代来计算$\msign$的优化思路，所得结果相比Muon的官方解，能够明显提高迭代的收敛速度和效果。</p><p data-spaces-tag="p" data-spaces-attrs="{}">最后需要指出的是，对于Muon来说，小规模的实验结果显示，$\msign$的计算精度跟模型的最终效果似乎没有必然联系，小模型下提高$\msign$的精度似乎只能在前期加速一点收敛速度，但最终结果并无变化。目前尚不清楚这个结论在更大规模的模型下是否成立。</p>
</div>
