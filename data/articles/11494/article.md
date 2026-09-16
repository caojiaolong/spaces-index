# 让炼丹更科学一些（四）：新恒等式，新学习率

> 作者：苏剑林 · 科学空间 · 2025-12-26
>
> 原文：<https://spaces.ac.cn/archives/11494>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上篇文章[《让炼丹更科学一些（三）：SGD的终点损失收敛》](<https://spaces.ac.cn/archives/11480>)中我们成功将收敛结论从平均损失转化成终点损失，得到了$\mathcal{O}(\sqrt{\ln T/T})$的收敛速度。然而，仔细思考之下我们会发现这个结果其实不大符合直觉：按照经验，终点损失应该更接近最优值才对，平均损失的收敛速度都能做到$\mathcal{O}(1/\sqrt{T})$，怎么终点收敛速度反而更慢呢？

这个问题的最新进展是[《Optimal Linear Decay Learning Rate Schedules and Further Refinements》](<https://papers.cool/arxiv/2310.07831>)，论文先推广了之前证明的关键恒等式，然后指出学习率调度对终点收敛的重要性，由此将终点损失的收敛加速至$\mathcal{O}(1/\sqrt{T})$。

## 新恒等式

原论文的结果很丰富，我们将分多篇文章介绍，这篇文章主要顺着上一篇的思路先做个初步介绍。为了将平均损失的收敛结论转换成终点损失，上一篇文章引入的关键恒等式是
\begin{equation}q_T = \frac{1}{T}\sum_{t=1}^T q_t + \sum_{k=1}^{T-1} \frac{1}{k(k+1)}\sum_{t=T-k}^T (q_t - q_{T-k})\end{equation}
这篇文章我们将它推广成加权平均版：定义$w_{k:T}\triangleq\sum_{t=k}^T w_t$，那么有
\begin{equation}q_T = \frac{1}{w_{1:T}}\sum_{t=1}^T w_t q_t + \sum_{k=1}^{T-1}\left(\frac{1}{w_{k+1:T}} - \frac{1}{w_{k:T}}\right)\sum_{t=k}^T w_t (q_t - q_k)\label{eq:qt-g}\end{equation}
证明思路基本一致，设$\lambda_k = \sum_{t=T-k+1}^T w_t, S_k = \frac{1}{\lambda_k}\sum_{t=T-k+1}^T w_t q_t$，那么
\begin{equation}\begin{aligned}
\lambda_k S_k =&\, \lambda_{k+1} S_{k+1} - w_{T-k} q_{T-k} \\[5pt]
=&\, \lambda_k S_{k+1} + w_{T-k} (S_{k+1} - q_{T-k}) \\
=&\, \lambda_k S_{k+1} + \frac{w_{T-k}}{\lambda_{k+1}}\sum_{t=T-k}^T w_t (q_t - q_{T-k})
\end{aligned}\end{equation}
两边除以$\lambda_k$，然后对$k=1\sim T-1$求和得
\begin{equation}S_1 = S_T + \sum_{k=1}^{T-1}\frac{w_{T-k}}{\lambda_k\lambda_{k+1}}\sum_{t=T-k}^T w_t (q_t - q_{T-k})\end{equation}
留意到$\frac{w_{T-k}}{\lambda_k\lambda_{k+1}} = \frac{1}{\lambda_k} - \frac{1}{\lambda_{k+1}}$，然后代入$S_1, S_T$的定义，得
\begin{equation}q_T = \frac{1}{\sum_{t=1}^T w_t}\sum_{t=1}^T w_t q_t + \sum_{k=1}^{T-1}\left(\frac{1}{\sum_{t=T-k+1}^T w_t} - \frac{1}{\sum_{t=T-k}^T w_t}\right)\sum_{t=T-k}^T w_t (q_t - q_{T-k})\end{equation}
最后用$k$换元$T-k$，即得式$\eqref{eq:qt-g}$。

## 一般结论

接着，我们还是从第二篇文章[《让炼丹更科学一些（二）：将结论推广到无界域》](<https://spaces.ac.cn/archives/11469>)的核心不等式出发
\begin{equation}\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\varphi})]\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2}{2} + \frac{G^2}{2}\sum_{t=1}^T \eta_t^2\label{leq:avg-2-mid3}\end{equation}
照着[《让炼丹更科学一些（三）：SGD的终点损失收敛》](<https://spaces.ac.cn/archives/11480>)的“[准备工作](<https://spaces.ac.cn/archives/11480#%E5%87%86%E5%A4%87%E5%B7%A5%E4%BD%9C>)”一节的思路，我们将起点改为$k$，然后代入$\boldsymbol{\varphi}=\boldsymbol{\theta}_k$，但不需要假设$\eta_t$的单调性来两边除以$\eta_T$，而是直接得到
\begin{equation}\sum_{t=k}^T \eta_t\mathbb{E}[L(\boldsymbol{\theta}_t) -  L(\boldsymbol{\theta}_k)] \leq \frac{G^2}{2} \sum_{t=k}^T \eta_t^2\label{leq:avg-2-mid4}\end{equation}
代入$w_t = \eta_t, q_t = \mathbb{E}[L(\boldsymbol{\theta}_t)] - L(\boldsymbol{\theta}^*)$到恒等式$\eqref{eq:qt-g}$，我们得到
\begin{equation}\begin{aligned}
\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] =&\, \frac{1}{\eta_{1:T}}\underbrace{\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]}_{\eqref{leq:avg-2-mid3}} + \sum_{k=1}^{T-1}\left(\frac{1}{\eta_{k+1:T}} - \frac{1}{\eta_{k:T}}\right)\underbrace{\sum_{t=k}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) -  L(\boldsymbol{\theta}_k)]}_{\eqref{leq:avg-2-mid4}} \\
\leq&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\eta_{1:T}} + \frac{G^2}{2\eta_{1:T}}\sum_{t=1}^T \eta_t^2 + \frac{G^2}{2}\sum_{k=1}^{T-1}\left(\frac{1}{\eta_{k+1:T}} - \frac{1}{\eta_{k:T}}\right)\sum_{t=k}^T \eta_t^2 \\
\end{aligned}\label{leq:avg-2-mid5}\end{equation}
对第二项利用$\sum_{k=1}^{T-1}\sum_{t=k}^T = \sum_{t=1}^{T} \sum_{k=1}^{\min(t, T-1)}$：
\begin{equation}\begin{aligned}
\sum_{k=1}^{T-1}\left(\frac{1}{\eta_{k+1:T}} - \frac{1}{\eta_{k:T}}\right)\sum_{t=k}^T \eta_t^2 =&\, \sum_{t=1}^T\eta_t^2\sum_{k=1}^{\min(t, T-1)}\left(\frac{1}{\eta_{k+1:T}} - \frac{1}{\eta_{k:T}}\right) = \sum_{t=1}^T\eta_t^2\left(\frac{1}{\eta_{\min(t+1, T):T}} - \frac{1}{\eta_{1:T}}\right)
\end{aligned}\end{equation}
代入到式$\eqref{leq:avg-2-mid5}$得
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\eta_{1:T}} + \frac{G^2}{2}\sum_{t=1}^T\frac{\eta_t^2}{\eta_{\min(t+1, T):T}}\label{leq:last-2}\end{equation}
这就是终点损失收敛的加强版结果，它不依赖于学习率的单调递减性，也不依赖于两端除以$\eta_T$的操作，从而给学习率调度提供了更灵活的空间。上一篇文章只相当于把$\eta_{1:T},\eta_{\min(t+1, T):T}$都简单替换成了$T\eta_T,\max(1, T-t)\eta_T$，结论明显更为粗糙。这个结果是新的，但它跟原论文附录F中的定理10本质上是等价的。

## 加速收敛

这一节我们将会看到，在适当的设置下，式$\eqref{leq:last-2}$能实现$\mathcal{O}(1/\sqrt{T})$的收敛速度。这里“适当的设置”主要是学习率调度策略，不同于以往的常数学习率或$\alpha/\sqrt{t}$、$\alpha/t$这类无终点学习率，这一次我们选择“线性衰减（Linear Decay）”：
\begin{equation}\eta_t = \alpha\left(1 - \frac{t}{T+1}\right)\label{eq:linear-decay}\end{equation}
这个学习率函数值得单开一行来强调一下，因为它是学习率策略的最佳实践之一，比如[《Straight to Zero: Why Linearly Decaying the Learning Rate to Zero Works Best for LLMs》](<https://papers.cool/arxiv/2502.15938>)就声称它比Cosine Decay还优，这体现了我们的讨论已经越来越接近实用场景。

直接逐一计算得：
\begin{gather}\eta_{1:T} = \sum_{\tau=1}^T \alpha\left(1 - \frac{\tau}{T+1}\right) = \frac{\alpha T}{2}  \\[6pt]
\eta_{t+1:T} = \sum_{\tau=t+1}^T \alpha\left(1 - \frac{\tau}{T + 1}\right) = \frac{\alpha(T-t)(T+1-t)}{2(T+1)} \\[6pt]
\frac{\eta_t^2}{\eta_{t+1:T}} = \frac{2\alpha (T+1-t)}{(T-t)(T+1)}\leq  \frac{4\alpha}{T+1} \\[6pt]
\sum_{t=1}^T\frac{\eta_t^2}{\eta_{\min(t+1, T):T}} = \eta_T + \sum_{t=1}^{T-1}\frac{\eta_t^2}{\eta_{t+1:T}} \leq \frac{\alpha}{T+1} + \sum_{t=1}^{T-1} \frac{4\alpha}{T+1} \leq 4\alpha\end{gather}
将这些结果代入到式$\eqref{leq:last-2}$即得
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{\alpha T} + 2 G^2 \alpha\end{equation}
取$\alpha=\frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert}{G\sqrt{2T}}$，就可以让右端取到最小值，实现$\mathcal{O}(1/\sqrt{T})$的终点收敛速度。要指出的是，如果不引入更强的假设，$\mathcal{O}(1/\sqrt{T})$已经没法改进，这是由信息论保证的（参考[《Information\-theoretic lower bounds on the oracle complexity of stochastic convex optimization》](<https://papers.cool/arxiv/1009.0571>)），所以这已经是理论最优的收敛速度。

## 变分之法

可能有读者疑问，怎么看出线性衰减学习率能取得最优的收敛速度呢？或者更一般地问，如何求$\eta_1,\eta_2,\cdots,\eta_T\geq 0$，使得不等式$\eqref{leq:last-2}$右端取最小值？这是一个多元函数最小化问题，标准解法是求偏导，但精确解很复杂，不好观察性态。

这里我们考虑将它连续化，然后用[变分法](<https://spaces.ac.cn/search/%E5%8F%98%E5%88%86%E6%B3%95/>)给出一个快速的启发式推导（当然快速的前提是读者熟悉变分法）。记$S_t = \eta_{\min(t+1, T):T}$，那么对于$t < T - 1$都有$\eta_t = S_{t-1} - S_t\approx -\dot{S}_t$，我们统一用$-\dot{S}_t$近似$\eta_t$、用积分近似求和，那么原始问题可以近似为最小化如下的积分函数
\begin{equation}\frac{c}{S_0} + \int_0^T \frac{(\dot{S}_t)^2}{S_t} dt \label{eq:func-min}\end{equation}
根据定义$S_T=0$，考虑先把$S_0$固定，那么这就是一个边界固定的变分问题，代入[欧拉\-拉格朗日方程](<https://en.wikipedia.org/wiki/Euler%E2%80%93Lagrange_equation>)得
\begin{equation}\frac{d}{dt}\frac{2\dot{S}_t}{S_t} = -\frac{(\dot{S}_t)^2}{S_t^2}\end{equation}
容易解得$S_t = S_0(1-t/T)^2$，代入到$\eqref{eq:func-min}$得$c/S_0 + 4S_0/T$，最小值在$S_0=\sqrt{c T/4}$取到，于是最优解是$S_t = \sqrt{c T/4}(1-t/T)^2$，求导得$\eta_t = -\dot{S}_t = \sqrt{c/T}(1-t/T)$，这就得到了线性衰减，连同前面的常数$\propto 1/\sqrt{T}$都得到了。

现在这个结果还是近似，我们要代回原始的离散化格式做严格证明，此时$\eta_t\propto 1 - t/T$会遇到分母为零的麻烦，所以要稍微变动一下，最终得到式$\eqref{eq:linear-decay}$。

## 延伸思考

在上面的推导和结论中，有几个值得我们特别注意的关键点，某种意义上，这是随机优化收敛理论的里程碑式的进展。

首先，如果学习率设为常数，那么结论$\eqref{leq:last-2}$跟上一篇文章是一致的，我们已经证明过，它至多能达到$\mathcal{O}(\sqrt{\ln T/T})$的收敛速度，并不是最优的，而式$\eqref{eq:linear-decay}$的线性衰减学习率则可以做到$\mathcal{O}(1/\sqrt{T})$。这一方面显示了学习率衰减对于终点收敛的必要性，另一方面也从理论上支撑了线性衰减策略。

不难证明，在前三篇文章中，最佳的收敛速度都在常数学习率下取到，但这个常数是跟训练总步数$T$相关的，比如$\alpha/\sqrt{T}$，很多工作认为这是一个不足之处，它们会更偏爱$\alpha/\sqrt{t}$、$\alpha/t$之类的学习率策略，因为不需要预知训练步数$T$，属于“随时停止、随时续训、想训练多少步就训练多少步”的无终点学习率策略。

然而，这类策略的实践表现通常并不佳。本文的新结论显示，切换到终点损失后，出现了一些新特性，收敛最快的既不是与$T$有关的常数学习率，也不是$T$无关的动态学习率，而应该是“二者兼之”，比如线性衰减。此外，实践常用的还有Cosine型衰减，它们的共同关键词是：终点更优、与$T$有关、动态变化。

换言之，不存在“一劳永逸”的学习率策略，根据训练步数精调学习率策略，才能获得最佳的终点收敛结果。这其实跟现在的Scaling Law实践很吻合，比如[Step Law](<https://papers.cool/arxiv/2503.04715>)发现最优学习率和最优Batch Size都应该根据数据量精细调整，注意到数据量和Batch Size都给定后训练步数$T$也就确定了，所以它们也可以说是训练步数$T$的函数。

后面的某篇文章中，我们也会简要讨论这系列结论与如今的Scaling Law之间的联系，敬请大家期待。

## 文章小结

在这篇文章中，我们推广了上篇的核心恒等式，然后得到了理论最佳的终点损失收敛速度。有意思的是，取得这一成绩的学习率策略并不是常数学习率，也不是传统的逆步数、逆步数平方根学习率，而是更贴近我们日常实践的线性衰减。接下来，我们还会继续探讨该结论背后的深刻意义。
