# 如何更科学地估计矩阵的谱范数？

> 作者：苏剑林 · 科学空间 · 2026-05-04
>
> 原文：<https://spaces.ac.cn/archives/11736>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

谱范数（Spectral Norm）是矩阵分析中的核心概念，在深度学习领域中也扮演着重要角色——从WGAN时代所需的Lipschitz约束，到如今LLM的训练稳定性，再到方兴未艾的Muon优化器，都跟矩阵参数的谱范数密切相关。因此，如何高效、准确地估计谱范数，也愈发显得重要，值得我们深入研究。

众所周知，幂迭代（Power Iteration）是估计谱范数的标准方案，但它仍有很大改进空间。这篇文章将简要地整理谱范数的一些估计思路，包括改进幂迭代的收敛速度，以及如何估计谱范数的严格上界，等等。

## 谱范数

谱范数的定义是
\begin{equation}\Vert\boldsymbol{W}\Vert_2 = \max_{\Vert\boldsymbol{x}\Vert_2=1} \Vert \boldsymbol{W}\boldsymbol{x}\Vert_2\end{equation}
其中$\boldsymbol{W}\in\mathbb{R}^{n\times m}, \boldsymbol{x}\in\mathbb{R}^m$，不失一般性，设$n\geq m$。从定义上看来，谱范数代表着线性层的“膨胀率”——输入向量经过线性层后，输出向量模长至多膨胀$\Vert\boldsymbol{W}\Vert_2$倍。谱范数又等于矩阵的最大奇异值（证明参考[《低秩近似之路（二）：SVD》](<https://spaces.ac.cn/archives/10407>)），这些基本内容我们就不过多展开了。

笔者第一次接触谱范数，是在GAN流行的年代，[WGAN](<https://spaces.ac.cn/archives/6280>)“横空出世”，强调了判别器满足Lipschitz约束的必要性。而一个线性层$\boldsymbol{W}\boldsymbol{x}$的Lipschitz常数，正是矩阵$\boldsymbol{W}$的谱范数，所以引申出了谱归一化、谱正则化等技巧，有兴趣的同学可以参考[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)。

近年来，逐渐流行起来的[Muon](<https://spaces.ac.cn/tag/muon/>)优化器，也跟谱范数关系匪浅，它通常被视为谱范数下的最速下降；同时，在[《MuP之上：4\. 坚守参数的稳定性》](<https://spaces.ac.cn/archives/11729>)中，我们还提出要约束参数的谱范数，以保证训练的稳定性。这些内容都对谱范数的准确计算提出了需求。

## 幂迭代

通过SVD来计算谱范数是最直接的方法，但显然过于昂贵。我们通常是用幂迭代
\begin{equation}\boldsymbol{v}^{(t)} = \frac{\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{v}^{(t-1)}}{\Vert\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{v}^{(t-1)}\Vert_2}\end{equation}
它以$(\sigma_2/\sigma_1)^{2t}$的速度收敛至右主奇异向量$\boldsymbol{v}_1$，$T$步迭代之后，我们有
\begin{equation}\sigma_1 \approx \Vert\boldsymbol{W}\boldsymbol{v}^{(t)}\Vert_2\end{equation}
证明可见[《从谱范数梯度到新式权重衰减的思考》](<https://spaces.ac.cn/archives/10648>)。实践中，如果我们只关心谱范数而不关心奇异向量，那么收敛速度往往会比$(\sigma_2/\sigma_1)^{2t}$更乐观。此外，幂迭代的结果容易受到初值$\boldsymbol{v}^{(0)}$的影响而产生波动，我们可以考虑并行计算$k$路，然后求最大值。

如果同时计算$k$路，并且每步迭代将L2 Normalize换成对这$k$个向量的正交化，即
\begin{equation}\newcommand{QR}{\mathop{\text{QR}}}\boldsymbol{V}^{(t)} = \QR(\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{V}^{(t-1)})\end{equation}
那么结果将收敛到前$k$个右奇异向量，继而可以求出前$k$个奇异值，原理可以参考[《基于流式幂迭代的Muon实现：4\. 原理》](<https://spaces.ac.cn/archives/11710>)。不过这个延伸本文也不展开了，将注意力集中在谱范数——也就是最大奇异值——的估算上。

## 求梯度

首先提供一段基于Jax的幂迭代参考实现：



<pre><code class="language-python">import jax, jax.lax as lax
import jax.numpy as jnp

@jax.jit(static_argnums=(1,))
def l2_normalize(x, axis=-2):
    return x / jnp.linalg.vector_norm(x, axis=axis, keepdims=True)

@jax.jit(static_argnums=(1, 2, 3))
def spec_norm_v1(w, T=10, k=1, key=42):
    v_shape = w.shape[:-2] + w.shape[-1:] + (k,)
    v_init = jax.random.normal(jax.random.PRNGKey(key), v_shape)
    v_step = lambda i, v: l2_normalize(w.mT @ (w @ v))
    v = lax.fori_loop(0, T, v_step, v_init)
    return jnp.linalg.vector_norm(w @ v, axis=-2).max(axis=-1)</code></pre>



很明显，幂迭代的复杂度是$\mathcal{O}(mnTk)$，单看复杂度来说已经是比较理想的了。如果我们只是监控用途，那么上面这种写法已经足够了，但如果我们是谱归一化、谱正则化的用途，那么就需要谱范数的梯度，按照上面这种写法，它需要沿着幂迭代的循环逐步反向传播，计算复杂度比较高。

事实上，在[《从谱范数梯度到新式权重衰减的思考》](<https://spaces.ac.cn/archives/10648>)我们就已经推过谱范数的梯度$\nabla_{\boldsymbol{W}}\sigma_1 = \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$，所以我们完全可以直接自定义它的梯度为$\nabla_{\boldsymbol{W}}\sigma_1 = \boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$，以降低反向传播的复杂度。除此之外，我们也可以利用$\sigma_1 = \boldsymbol{u}_1^{\top}\boldsymbol{W}\boldsymbol{v}_1$，在前向的时候直接输出
\begin{equation}\sigma_1 = \color{skyblue}{[}\boldsymbol{u}_1^{\top}\color{skyblue}{]_{sg}}\boldsymbol{W}\color{skyblue}{[}\boldsymbol{v}_1\color{skyblue}{]_{sg}}\end{equation}
其中$\color{skyblue}{[\cdot]_{sg}}$是stop gradient算子，这样在自动求导的时候，就只会对$\boldsymbol{W}$求，结果正是$\boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$，避免反向传播到$\boldsymbol{u}_1,\boldsymbol{v}_1$的幂迭代内部。

## 不浪费

通过$T$步幂迭代，我们求出了向量序列$\boldsymbol{v}^{(1)},\boldsymbol{v}^{(2)},\cdots,\boldsymbol{v}^{(T)}$，但最终我们只用到了$\boldsymbol{v}^{(T)}$来估计$\sigma_1$，这看起来颇为“浪费”，于是有人想着把它们都利用起来，形成对谱范数更好的估计。

如无意外，$\boldsymbol{v}^{(1)},\boldsymbol{v}^{(2)},\cdots,\boldsymbol{v}^{(T)}$互不相同，但会越来越接近$\boldsymbol{v}_1$，我们可以将它们想象成“$\boldsymbol{v}_1$加上某种噪声”，如果放在一起进行“降噪”，确实有可能取得更准确的结果。具体来说，我们考虑通过$\boldsymbol{v}^{(1)},\boldsymbol{v}^{(2)},\cdots,\boldsymbol{v}^{(T)}$的线性组合，来给$\boldsymbol{v}_1$构建一个更好的近似，而由这组向量张成的子空间，我们称为“Krylov子空间”。

简单起见，我们先执行一次正交化，得到等价的标准正交基$\boldsymbol{Q} = [\boldsymbol{q}_1,\boldsymbol{q}_2,\cdots,\boldsymbol{q}_T]\in\mathbb{R}^{m\times T}$；接着，我们要寻找系数$\boldsymbol{x}=[x_1,x_2,\cdots,x_T]^{\top}\in\mathbb{R}^T$，使得向量$\sum_{i=1}^T x_i \boldsymbol{q}_i = \boldsymbol{Q}\boldsymbol{x}$尽可能满足
\begin{equation}\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{Q}\boldsymbol{x}\approx \sigma_1^2\boldsymbol{Q}\boldsymbol{x}\end{equation}
两边乘$\boldsymbol{Q}^{\top}$得$\boldsymbol{Q}^{\top}\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{Q}\boldsymbol{x}\approx \sigma_1^2\boldsymbol{x}$，这表明$\sigma_1^2,\boldsymbol{x}$分别是$(\boldsymbol{W}\boldsymbol{Q})^{\top}\boldsymbol{W}\boldsymbol{Q}$的特征值和特征向量，这意味着我们只需对它做特征值分解，获取它的最大特征值然后开方，就得到更好的估计，由于这只是一个$T\times T$矩阵，当$T$比较小时，对它做特征值分解并不是什么高成本的事，因此直接调用现成函数即可。

这一思路相当于现代SVD求解器的“Lanczos算法”的简化版，实际SVD中会有更多更精细的处理。此外，该加速思路跟随机SVD（Randomized SVD）也有相通之处，随机SVD通常是用高斯随机矩阵来投影，而这里用Krylov子空间中的标准正交基来投影。

## 加速技

融入Krylov子空间加速的幂迭代参考代码如下：



<pre><code class="language-python">@jax.jit(static_argnums=(1, 2))
def spec_norm_v2(w, T=10, key=42):
    v_shape = w.shape[:-2] + w.shape[-1:] + (1,)
    v_init = jax.random.normal(jax.random.PRNGKey(key), v_shape)
    v_step = lambda v, _: (l2_normalize(w.mT @ (w @ v)),) * 2
    v = lax.scan(v_step, v_init, length=T)[1].swapaxes(0, -1)[0]
    v = jnp.linalg.qr(v)[0]
    return jnp.linalg.eigvalsh((u := w @ v).mT @ u)[..., -1]**0.5</code></pre>



实测显示，子空间加速的效果非常明显，$T=5$时它的精度已经超过了原版幂迭代$T=10$的精度，并且速度更快。在$T=10$时，该函数大概2/3耗时源于幂迭代，1/3耗时源于QR分解，至于特征值分解的耗时占比非常少。

看过“[流式幂迭代](<https://spaces.ac.cn/search/%E6%B5%81%E5%BC%8F%E5%B9%82%E8%BF%AD%E4%BB%A3/>)”系列的读者可能会想着用里边介绍的Cholesky QR来加速QR分解，但很遗憾，由于幂迭代的结果会越来越接近共线，待QR矩阵的条件数会持续恶化，这正好是Cholesky QR “无能为力”的区间，失败率极高，所以基本没法用它来加速。

一个更有效的加速手段是：只用幂迭代最后几步的结果，比如最后三步的$\boldsymbol{v}^{(T-2)},\boldsymbol{v}^{(T-1)},\boldsymbol{v}^{(T)}$，来构建Krylov子空间加速。这样已经能获得大部分效果收益，并且减少了QR分解和特征值分解的计算量，从而实现加速。

## 估上界

幂迭代及其加速版，严格来讲只能获得谱范数的下界，当然这很多时候也够用了。但如果在某些场景下，我们必须获取上界，比如想要通过谱归一化来保证某个矩阵的范数严格小于1，那就需要另寻他法了。

此时一个基本方案是计算[Schatten范数](<https://en.wikipedia.org/wiki/Schatten_norm>)，它正是谱范数的上界。设矩阵$\boldsymbol{W}$的SVD为$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$，奇异值为$\sigma_1\geq\sigma_2\geq\cdots\geq\sigma_m$，那么它的Schatten\-$p$范数定义为
\begin{equation}\Vert\boldsymbol{W}\Vert_{S,p} = \sqrt[\uproot{10}p]{\sum_{i=1}^m \sigma_i^p} \quad\geq\quad \sigma_1\end{equation}
$p$越大越接近准确值。当$p$是偶数时，Schatten\-$p$范数的计算相对可行，这主要利用了如下恒等式
\begin{gather}\newcommand{tr}{\mathop{\text{tr}}}\Vert\boldsymbol{W}\Vert_{S,2k} = \sqrt[\uproot{10}2k]{\sum_{i=1}^m \sigma_i^{2k}} = \sqrt[\uproot{3}2k]{\tr(\boldsymbol{V}\boldsymbol{\Sigma}^{2k}\boldsymbol{V}^{\top})} = \sqrt[\uproot{3}2k]{\tr((\boldsymbol{W}^{\top}\boldsymbol{W})^k)} \\
\Vert\boldsymbol{W}\Vert_{S,4k} = \sqrt[\uproot{10}4k]{\sum_{i=1}^m \sigma_i^{4k}} = \sqrt[\uproot{3}2k]{\Vert\boldsymbol{V}\boldsymbol{\Sigma}^{2k}\boldsymbol{V}^{\top}\Vert_{S,2}} = \sqrt[\uproot{3}2k]{\Vert(\boldsymbol{W}^{\top}\boldsymbol{W})^k\Vert_{S,2}}\label{eq:S-4k}\end{gather}
不难看出，Schatten\-$2$范数实际上就是F范数，它可以通过逐元素平方和再开方求得。所以上式两个恒等式表明，只要我们求得$(\boldsymbol{W}^{\top}\boldsymbol{W})^k$，那么就可以用较低的成本求得$\Vert\boldsymbol{W}\Vert_{S,2k}$和$\Vert\boldsymbol{W}\Vert_{S,4k}$。

第一步$\boldsymbol{W}^{\top}\boldsymbol{W}$，复杂度是$\mathcal{O}(nm^2)$，后面反复自乘，每一步复杂度是$\mathcal{O}(m^3)$，那么$\mathcal{O}(nm^2 + Tm^3)$的复杂度就可以求得$(\boldsymbol{W}^{\top}\boldsymbol{W})^{2^T}$。理论上，这个复杂度远超于幂迭代，甚至当$n,m$比较大时，第一步$\boldsymbol{W}^{\top}\boldsymbol{W}$就远超幂迭代了。因此，如果只想估计谱范数而不要求上界，还是应该优先考虑幂迭代。

不过，矩阵乘法通常可以充分并行，所以当$n,m$不大或$n\gg m$时，Schatten\-$p$范数不至于出现计算瓶颈，这时候我们可以考虑它。又或者，当我们非得要严格上界时，这似乎也是最直接的途径了。

## 保数值

为了计算$\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}$，朴素的实现是从$\boldsymbol{M} = \boldsymbol{W}^{\top}\boldsymbol{W}$出发，$\boldsymbol{M}\leftarrow \boldsymbol{M}^2$重复执行$T$次得到$(\boldsymbol{W}^{\top}\boldsymbol{W})^{2^T}$，然后代入公式$\eqref{eq:S-4k}$计算，但由于是超指数运算，所以很快就会爆炸至NaN或者坍缩至零。

一般情况下，$\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}$本身是不会数值爆炸的，问题出在显式计算$(\boldsymbol{W}^{\top}\boldsymbol{W})^{2^T}$上。解决办法是每一步乘方之后都重新归一化【$\boldsymbol{M}\leftarrow \boldsymbol{M}^2/\tr(\boldsymbol{M}^2)$】，这样能杜绝数值爆炸，也保证了缩放会足够紧凑，不至于坍缩至零；同时，我们将每一步的归一化因子在对数域累积起来，供最后计算$\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}$所用。

计算流程总结如下：
$$\begin{array}{|l|}
\hline
\text{计算}\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}\text{作为谱范数的上界} \\[4pt]
\hline
\begin{array}{ll}
1: & \text{Initialize }\log S = \log\tr(\boldsymbol{W}^{\top}\boldsymbol{W}), \boldsymbol{M}=\frac{\boldsymbol{W}^{\top}\boldsymbol{W}}{ \tr(\boldsymbol{W}^{\top}\boldsymbol{W})}\\
2: & \textbf{For }t=1,2,\cdots,T\textbf{ do } \\
3: & \qquad \log S\leftarrow 2\log S + \log \tr(\boldsymbol{M}^2) \\
4: & \qquad \boldsymbol{M} \leftarrow \frac{\boldsymbol{M}^2}{ \tr(\boldsymbol{M}^2)} \\
5: & \text{Output } \exp\left(\frac{\log S + \log\Vert\boldsymbol{M}\Vert_F}{2^{T+1}}\right)
\end{array} \\
\hline
\end{array}$$

一个简单的参考实现：



<pre><code class="language-python">@jax.jit
def tr(w):
    return w.trace(axis1=-1, axis2=-2)[..., None, None]

@jax.jit(static_argnums=(1,))
def spec_norm_v3(w, T=5):
    m = (m := w.mT @ w) / (s := tr(m))
    ms_step = lambda i, ms: ((m := ms[0] @ ms[0]) / (s := tr(m)), 2 * ms[1] + jnp.log(s))
    m, logs = lax.fori_loop(0, T, ms_step, (m, jnp.log(s)))
    logf = 0.5 * jnp.log((m**2).sum(axis=[-1, -2], keepdims=True))
    return jnp.exp((logs + logf) / 2**(T + 1))</code></pre>



## 多阶矩

为了计算$\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}$，我们要以某种方式计算出$\boldsymbol{W}^{\top}\boldsymbol{W},\cdots,(\boldsymbol{W}^{\top}\boldsymbol{W})^{2^{T-1}},(\boldsymbol{W}^{\top}\boldsymbol{W})^{2^T}$，这意味着我们可以同时获得$\Vert\boldsymbol{W}\Vert_{S,2},\cdots,\Vert\boldsymbol{W}\Vert_{S,2^{T+1}},\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}$，但最后只用上了$\Vert\boldsymbol{W}\Vert_{S,2^{T+2}}$，同样显得有些“浪费”了。

有没有办法跟幂迭代的Krylov子空间方法一样，将这些结果都利用起来，提高估计精度呢？还真有！[《Fast Tight Spectral\-Norm Bounds》](<https://jiha-kim.github.io/posts/fast-spectral-norm-bounds-from-scaled-gram-moments/>)提供了一个通过非线性规划来改良估计结果的思路。我们先来看一个简单情况，假设我们求得奇异值的2阶矩和4阶矩
\begin{equation}\sum_{i=1}^m \sigma_i^2 = S_2, \qquad \sum_{i=1}^m \sigma_i^4 = S_4\end{equation}
那么按照上两节的结果，$S_4^{1/4}$比$S_2^{1/2}$更接近谱范数，所以我们返回$S_4^{1/4}$作为谱范数的上界，$S_2$弃之不用。但$S_2$真的没用吗？试想一下，如果$m=2$，那么刚好两个方程、两个未知数，理论上我们可以把$\sigma_1,\sigma_2$解出来！$m > 2$时虽然做不到精确求解，但依然可以缩小$\sigma_1$的范围。具体来说，我们有
\begin{equation}\frac{S_2 - \sigma_1^2}{m-1} = \frac{1}{m-1}\sum_{i=2}^m \sigma_i^2 \leq \sqrt{\frac{1}{m-1}\sum_{i=2}^m \sigma_i^4} = \sqrt{\frac{S_4 - \sigma_1^4}{m-1}}\end{equation}
即$(S_2 - \sigma_1^2)^2\leq (m-1)(S_4 - \sigma_1^4)$，这本质上是一元二次不等式，容易解得
\begin{equation}\sigma_1 \leq \sqrt{\frac{S_2+\sqrt{(m-1)(mS_4-S_2^2)}}{m}}\label{eq:s2-s4}\end{equation}
这就得到了一个用$S_2$和$S_4$表示出来的$\sigma_1$的上界，它是比$S_4^{1/4}$更好的估计。[《Fast Tight Spectral\-Norm Bounds》](<https://jiha-kim.github.io/posts/fast-spectral-norm-bounds-from-scaled-gram-moments/>)还将它推广到了同时用上$S_2,S_4,S_6,S_8$的形式，更紧凑但也更繁琐一些，有兴趣的读者自行到原文学习即可。用更多阶矩去改进估计理论上是可行的，但实践中往往要求解复杂的非线性方程组，实用价值不大。

## 有局限

理论上，结果$\eqref{eq:s2-s4}$还可以推广到用任意$S_{2k}$和$S_{4k}$来获得更准确上界：
\begin{equation}\sigma_1 \leq \sqrt[\uproot{10}2k]{\frac{S_{2k}+\sqrt{(m-1)(mS_{4k}-S_{2k}^2)}}{m}}\end{equation}
然而，当$k$比较大时，这个结果的实践意义非常有限，因为它需要显式计算出$S_{2k}$和$S_{4k}$，这在$k$比较大时会爆炸或坍缩，这并不现实。一个可以考虑的改进是将$S_{4k}^{1/4k}$分离出来：
\begin{equation}\sqrt[\uproot{10}2k]{\frac{S_{2k}+\sqrt{(m-1)(mS_{4k}-S_{2k}^2)}}{m}} = S_{4k}^{1/4k}\cdot\sqrt[\uproot{10}2k]{\frac{S_{2k}/S_{4k}^{1/2}+\sqrt{(m-1)(m-S_{2k}^2/S_{4k})}}{m}}\end{equation}
然后设$S_{2k}/S_{4k}^{1/2}=e^{\epsilon}$，在对数域进行近似展开
\begin{equation}\sqrt[\uproot{10}2k]{\frac{S_{2k}/S_{4k}^{1/2}+\sqrt{(m-1)(m-S_{2k}^2/S_{4k})}}{m}} \approx 1 - \frac{\epsilon^2}{4k(m-1)}\end{equation}
然而，我们的目的是求谱范数上界，如何在近似展开的同时保证上界性质，看上去比较复杂。另一方面，如果我们已经算到了比较大的$k$了，其实$S_{4k}^{1/4k}$准确度本身已经比较高，再利用规划思想去提高精度的意义也不大了。

## 全文终

本文简要整理了谱范数的几种估计思路。在实际应用中，如果只需要近似监控谱范数，幂迭代及其子空间加速版通常已经足够；如果必须保证严格上界，则可以考虑计算Schatten范数。两种路线各自延伸出的“不浪费”思想——Krylov子空间利用迭代历史，非线性规划利用多阶矩信息——也为我们提供了很好的算法设计启发。
