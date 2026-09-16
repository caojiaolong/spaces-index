# 让炼丹更科学一些（七）：步长调度与权重平均

> 作者：苏剑林 · 科学空间 · 2026-07-06
>
> 原文：<https://spaces.ac.cn/archives/11804>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

精通炼丹的同学都知道，步长调度，或者说学习率调度（LR Schedule），对模型最终效果来说至关重要。我们在前几篇文章中已经推导过，即便只考虑SGD，终点收敛的最优学习率函数也具有Warmup\-Decay这一实践常用的形态。不过，近来也有一些工作，比如[Schedule\-Free](<https://papers.cool/arxiv/2405.15682>)，尝试通过模型权重的某种平均来替代学习率调度，也取得了一些进展。

如何从理论上分析学习率调度与权重平均的联系呢？它们在多大程度上可以相互替换呢？这篇文章我们来尝试回答一下这些问题。

## 权重平均

对于权重平均，事实上我们在[《让炼丹更科学一些（三）：SGD的终点损失收敛》](<https://spaces.ac.cn/archives/11480>)就已经给出过一个基本结果
\begin{equation}\mathbb{E}[L(\bar{\boldsymbol{\theta}}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T \frac{\eta_t^2}{\eta_T}\end{equation}
其中$\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=1}^T \boldsymbol{\theta}_t$。这个结果意味着只要我们选择常数学习率$\eta\propto 1/\sqrt{T}$，右端就可以达到$\mathcal{O}(1/\sqrt{T})$的理想收敛速度，左端则意味着$\bar{\boldsymbol{\theta}}_T$收敛于$\boldsymbol{\theta}^*$，于是我们设计如下优化器
\begin{equation}\begin{aligned}
\boldsymbol{\theta}_{t+1} =&\, \boldsymbol{\theta}_t - \eta\, \boldsymbol{g}(\boldsymbol{x}_t, \boldsymbol{\theta}_t) \\[4pt]
\boldsymbol{\mu}_{t+1} =&\, (1 - c_{t+1}) \boldsymbol{\mu}_t + c_{t+1} \boldsymbol{\theta}_{t+1}
\end{aligned}\label{eq:pr}\end{equation}
其中$c_t = 1/t$。也就是说，新增一步完全不干扰训练轨迹的迭代，把训练轨迹等权平均起来，最终用来做预测的模型权重并不是终点值$\boldsymbol{\theta}_T$，而是平均值$\boldsymbol{\mu}_T$。然而，该做法实践表现并不佳，这是理论与实践不符的经典例子之一。

不过，如果我们将$c_t$改为某个与$t$无关的适当的常数，比如$10^{-3}$，那么实践中往往有着不俗的表现——这正是笔者之前打比赛常用的指数滑动平均（EMA）技巧。相比$c_t=1/t$，这种滑动平均的做法更倾向于平均最近的优化轨迹，而非整个历史轨迹，它的表现优异也表明我们不需要关注太久远的历史结果。

如何从理论上解释这种滑动平均技巧，或者更一般地，解释任意权重平均的可行性，便是接下来的核心任务。

## 再次推广

在文章[《让炼丹更科学一些（四）：新恒等式，新学习率》](<https://spaces.ac.cn/archives/11494>)中，为了将平均损失的结果推广到终点损失，我们引入了恒等式
\begin{equation}q_T = \frac{1}{w_{1:T}}\sum_{t=1}^T w_t q_t + \sum_{k=1}^{T-1}\left(\frac{1}{w_{k+1:T}} - \frac{1}{w_{k:T}}\right)\sum_{t=k+1}^T w_t (q_t - q_k)\end{equation}
其中$w_{k:T}\triangleq\sum_{t=k}^T w_t$。这一节我们再次将它推广为
\begin{equation}\frac{1}{v_{1:T}}\sum_{t=1}^T v_t q_t = \frac{1}{w_{1:T}}\sum_{t=1}^T w_t q_t + \frac{1}{v_{1:T}}\sum_{k=1}^{T-1}\left(\frac{v_{k+1:T}}{w_{k+1:T}} - \frac{v_{k:T}}{w_{k:T}}\right)\sum_{t=k+1}^T w_t (q_t - q_k)\label{eq:qt-gg}\end{equation}
直观来看，它提供了将一种加权求和转换成另一种加权求和的思路，当$v_T=1$且剩下的$v_t$等于0时，它便退化为终点形式。证明实际上也是之前证明的推广，记$S_k = \frac{1}{w_{k:T}}\sum_{t=k}^{T}w_t q_t$，可以写出
\begin{equation}\begin{aligned}
v_{k:T} S_k - v_{k+1:T} S_{k+1} =&\, v_{k:T} \left(\frac{w_{k+1:T}S_{k+1} + w_k q_k}{w_{k:T}}\right) - v_{k+1:T} S_{k+1} \\
=&\, v_k q_k + \left(\frac{v_{k:T}}{w_{k:T}} w_k - v_k\right) q_k + \left(\frac{v_{k:T}}{w_{k:T}}w_{k+1:T} - v_{k+1:T}\right) S_{k+1} \\
=&\, v_k q_k + \left(\frac{v_{k:T}}{w_{k:T}}w_{k+1:T} - v_{k+1:T}\right) (S_{k+1} - q_k) \\
=&\, v_k q_k + \left(\frac{v_{k:T}}{w_{k:T}} - \frac{v_{k+1:T}}{w_{k+1:T}}\right) (w_{k+1:T} S_{k+1} - w_{k+1:T} q_k) \\
=&\, v_k q_k + \left(\frac{v_{k:T}}{w_{k:T}} - \frac{v_{k+1:T}}{w_{k+1:T}}\right) \sum_{t=k+1}^T w_t (q_t - q_k)
\end{aligned}\end{equation}
第三个等号是因为直接将第二个等号的两个括号相加等于0，所以这两个括号互为相反数。现在左右两端对$k=1\sim T-1$求和，左端等于$v_{1:T} S_1 - v_T q_T$，将$v_T q_T$挪到右端，两端除以$v_{1:T}$并稍加整理，即得待证恒等式$\eqref{eq:qt-gg}$。

## 放缩变换

接着我们沿着上一篇[《让炼丹更科学一些（六）：自上而下的精妙构造》](<https://spaces.ac.cn/archives/11540>)的思路，去推导更一般的收敛结论。设$q_t=\mathbb{E}[L(\boldsymbol{\theta}_t)-L(\boldsymbol{\theta}^*)]$，代入到式$\eqref{eq:qt-gg}$得
\begin{equation}\frac{1}{v_{1:T}}\sum_{t=1}^T v_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] = \frac{1}{w_{1:T}}\sum_{t=1}^T w_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] + \frac{1}{v_{1:T}}\sum_{k=1}^{T-1}\left(\frac{v_{k+1:T}}{w_{k+1:T}} - \frac{v_{k:T}}{w_{k:T}}\right)\sum_{t=k+1}^T w_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}_k)]\end{equation}
为了便于后续推导，我们加上假设：
\begin{equation}\frac{v_{k+1:T}}{w_{k+1:T}}-\frac{v_{k:T}}{w_{k:T}}\geq 0 \qquad\iff\qquad\frac{v_k}{v_{k+1:T}}\leq\frac{w_k}{w_{k+1:T}},\end{equation}
那么就可以利用$L$的凸性，对$\mathbb{E}[L(\boldsymbol{\theta}_t)-L(\boldsymbol{\theta}^*)]$和$\mathbb{E}[L(\boldsymbol{\theta}_t)-L(\boldsymbol{\theta}_k)]$进行放缩，然后像上一篇文章的“[恒等变换](<https://spaces.ac.cn/archives/11540#%E6%81%92%E7%AD%89%E5%8F%98%E6%8D%A2>)”一节那样仔细整理，可得
\begin{equation}\frac{1}{v_{1:T}}\sum_{t=1}^T v_t\,\mathbb{E}[L(\boldsymbol{\theta}_t)-L(\boldsymbol{\theta}^*)]\leq \frac{1}{w_{1:T}}\sum_{t=1}^T w_t\,\mathbb{E}[\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\cdot(\boldsymbol{\psi}_t-\boldsymbol{\theta}^*)] \label{leq:mid}\end{equation}
其中
\begin{equation}\boldsymbol{\psi}_t=\frac{w_{1:T}}{v_{1:T}}\left[\frac{v_{t:T}}{w_{t:T}}\boldsymbol{\theta}_t-\sum_{k=1}^{t-1}\left(\frac{v_{k+1:T}}{w_{k+1:T}}-\frac{v_{k:T}}{w_{k:T}}\right)\boldsymbol{\theta}_k\right]\end{equation}
并且直接验证可得
\begin{equation}\boldsymbol{\psi}_{t+1}-\boldsymbol{\psi}_t=\frac{v_{t+1:T}}{v_{1:T}}\frac{w_{1:T}}{w_{t+1:T}}(\boldsymbol{\theta}_{t+1}-\boldsymbol{\theta}_t)\end{equation}
因此，如果我们让$\boldsymbol{\psi}_t$按照$\boldsymbol{\psi}_{t+1}=\boldsymbol{\psi}_t-w_t\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)$更新，那么对应的$\boldsymbol{\theta}_t$的更新规则是
\begin{equation}\boldsymbol{\theta}_{t+1}=\boldsymbol{\theta}_t-\underbrace{\frac{v_{1:T}}{v_{t+1:T}}\frac{w_t w_{t+1:T}}{w_{1:T}}}_{\eta_t}\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\end{equation}

## 一般结论

接下来的操作就比较朴素了，对式$\eqref{leq:mid}$右端应用[《让炼丹更科学一些（五）：基于梯度精调学习率》](<https://spaces.ac.cn/archives/11530>)等文章的结论得
\begin{equation}\frac{1}{w_{1:T}}\sum_{t=1}^T w_t\,\mathbb{E}[\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\cdot(\boldsymbol{\psi}_t-\boldsymbol{\theta}^*)] \leq \frac{1}{2 w_{1:T}}\left(R^2 + \sum_{t=1}^T w_t^2 G_t^2\right) \end{equation}
其中$R = \Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert, G_t^2 = \mathbb{E}[\Vert\boldsymbol{g}(\boldsymbol{x}_t, \boldsymbol{\theta}_t)\Vert^2]$，约定$\boldsymbol{\psi}_1=\boldsymbol{\theta}_1$。至于左端，我们应用凸性得
\begin{equation}\mathbb{E}\left[L\left(\frac{1}{v_{1:T}}\sum_{t=1}^T v_t \boldsymbol{\theta}_t \right) - L(\boldsymbol{\theta}^*)\right] \leq \frac{1}{v_{1:T}}\sum_{t=1}^T v_t\,\mathbb{E}[L(\boldsymbol{\theta}_t)-L(\boldsymbol{\theta}^*)]\end{equation}
综合起来是
\begin{equation}\mathbb{E}\left[L\left(\frac{1}{v_{1:T}}\sum_{t=1}^T v_t \boldsymbol{\theta}_t \right) - L(\boldsymbol{\theta}^*)\right] \leq \frac{1}{2 w_{1:T}}\left(R^2 + \sum_{t=1}^T w_t^2 G_t^2\right)\label{eq:avg-final}\end{equation}
右端的最小值在
\begin{equation}w_t = \frac{R G_t^{-2}}{\sqrt{Q_T}},\qquad Q_T=\sum_{k=1}^T G_k^{-2}\end{equation}
取到，所以意味着最优学习率具有形式
\begin{equation}\eta_t = \frac{v_{1:T}}{v_{t+1:T}}\frac{w_t w_{t+1:T}}{w_{1:T}} = \frac{v_{1:T}}{v_{t+1:T}}\frac{R G_t^{-2}}{\sqrt{Q_T}} \left(1 - \frac{Q_t}{Q_T}\right) \label{eq:opt-lr}\end{equation}

## 结果解读

上一篇文章[《让炼丹更科学一些（六）：自上而下的精妙构造》](<https://spaces.ac.cn/archives/11540>)给出的终点收敛结果是
\begin{equation}\mathbb{E}\left[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)\right] \leq \frac{1}{2 w_{1:T}}\left(R^2 + \sum_{t=1}^T w_t^2 G_t^2\right)\end{equation}
右端跟式$\eqref{eq:avg-final}$是一致的，但学习率变为$\eta_t = \frac{w_t w_{t+1:T}}{w_{1:T}}$。由此可见，平均权重如果想要跟终点权重效果相近，学习率需要多乘以一个$\frac{v_{1:T}}{v_{t+1:T}}$，或者反过来说，在相同的学习率下，权重平均的效果相当于学习率乘以$\frac{v_{t+1:T}}{v_{1:T}}$后的终点效果。很明显$\frac{v_{t+1:T}}{v_{1:T}}$小于等于1且单调递减，乘上这一项就起到了类似于LR Decay的作用。

让我们考虑一些具体例子。首先是最简单的$v_t\equiv 1$，对应开头的式$\eqref{eq:pr}$，此时$\frac{v_{t+1:T}}{v_{1:T}} = 1 - t/T$，正好是线性衰减，这意味着“常数学习率 \+ 等权平均”理论上跟线性衰减的学习率有着相近的效果。然而，实测效果往往是，直接将学习率做线性衰减的终点效果不错，但平均权重也就是式$\eqref{eq:pr}$的效果欠佳。

回顾整个推导过程，可以猜测问题大概出在“凸函数”的假设上。我们这个系列文章，其实都依赖于“目标函数是凸的”这一假设，但实际情况往往是非凸的，凸优化的结论无法精确复现。折中方案是假设局部近似为凸，那么可以在有限区间内对等权重平均，又或者直接考虑它的平滑版本——指数滑动平均（EMA），此时$v_t = \gamma^{-t}$（其中$0 < \gamma < 1$），那么
\begin{equation}\frac{v_{t+1:T}}{v_{1:T}} = \frac{1 - \gamma^{T-t}}{1 - \gamma^T}\end{equation}
它维持了一个$\mathcal{O}((1-\gamma)^{-1})$的平均区间。由于指数函数的无记忆性，如果我们在常数学习率训练的同时运行EMA，那么相当于每一步同时得到了经过$\mathcal{O}((1-\gamma)^{-1})$步线性衰减后的权重，可以提前窥见线性衰减后的效果。

## 相关工作

对于式$\eqref{eq:pr}$效果欠佳的问题，论文[《The Road Less Scheduled》](<https://papers.cool/arxiv/2405.15682>)提出了“Schedule\-Free”，它将求梯度的点改为了$\boldsymbol{\theta}_t$与$\boldsymbol{\mu}_t$的插值：
\begin{equation}\begin{aligned}
\hat{\boldsymbol{\theta}}_t =&\, (1-\beta)\boldsymbol{\theta}_t + \beta \boldsymbol{\mu}_t \\[4pt]
\boldsymbol{\theta}_{t+1} =&\, \boldsymbol{\theta}_t - \eta\, \boldsymbol{g}(\boldsymbol{x}_t, \hat{\boldsymbol{\theta}}_t) \\[4pt]
\boldsymbol{\mu}_{t+1} =&\, (1 - c_{t+1}) \boldsymbol{\mu}_t + c_{t+1} \boldsymbol{\theta}_{t+1}
\end{aligned}\end{equation}
直观来想，就是原本的平均操作完全独立于训练过程，模型从未感知过$\boldsymbol{\mu}_t$究竟好不好，现在我们将求梯度的点换成$\boldsymbol{\theta}_t$与$\boldsymbol{\mu}_t$的插值，模型能感知到$\boldsymbol{\mu}_t$的好坏，从而作出调整。这个改动理论上是有保证的，同时确实能显著改进平均权重$\boldsymbol{\mu}_T$的效果。

不过，随着训练规模的扩大，Schedule\-Free逐渐表现出了颓势，作者近来又提出了[ScheduleFree\+](<https://papers.cool/arxiv/2605.19095>)，给原本的Schedule\-Free打上了不少“补丁”，使它重新成为了SOTA。不过，Plus版额外添加的操作比较多，整套算法显得复杂起来了，理解难度也增加不少，所以这里就不详细展开了，后面我们再回来讨论它。

值得一提的是，Schedule\-Free及其Plus版的作者，正是前几篇文章我们介绍过的经典论文[《Optimal Linear Decay Learning Rate Schedules and Further Refinements》](<https://papers.cool/arxiv/2310.07831>)的作者，这几篇工作都成功将经典凸优化理论用到了实践训练之上，而不是停留在玩具模型的“纸上谈兵”之上，值得我们致敬！

## 延伸思考

真正值得思考的是，Schedule\-Free的设想真的能成立吗？所谓Schedule\-Free，它有两个目标，一是希望以常数学习率从头训到尾，二是希望各种超参数不依赖于总训练步数$T$。如果满足这两个条件，那么训练过程就可以随时停止，也可以随时续训，并且随便一步停下来，它都是截止到当前步数的最优解，确实非常理想。

然而，这两个目标本身是值得商榷的。首先，对于实际训练来说，常数学习率还是动态的Cosine Decay、Linear Decay等，并无本质区别，反而都是调参对比选最优，所以追求常数学习率实践意义不大；而且即便是Schedule\-Free及其Plus版，都没摆脱Warmup的依赖，所以实际上它们都还某种程度上依赖着人为定制的LR Schedule。

其次，希望各种超参数不依赖于总训练步数$T$，确实是一个好目标，但只从理论上看它都很难成立。还是以式$\eqref{eq:pr}$为例，即便在满足所有假设下，式$\eqref{eq:pr}$能表现良好，但它的最优学习率$\eta$依然是依赖于$T$；同样，Schedule\-Free无法摆脱这一事实，它是通过引入新的假设，来获得$\eta$不依赖于$T$的结果。

我们也可以从式$\eqref{eq:opt-lr}$来观察，对于等权平均有$\frac{v_{1:T}}{v_{t+1:T}}=1/(1-t/T)$，进一步假设$G_t$是常数$G$，那么$1 - Q_t/Q_T=1-t/T$，刚好抵消，因此最优学习率是$\frac{R G^{-2}}{\sqrt{Q_T}}$，$Q_T$依赖于$T$。另一方面，$G_t$通常是先大后小，所以根据式$\eqref{eq:opt-lr}$反比于$G_t^2$的特点，早期学习率应该更小，这反而可以解释Schedule\-Free仍无法彻底摆脱Schedule、仍需要Warmup的现象。

为此，ScheduleFree\+以及一些同期工作，已经逐渐从“Schedule\-Free”往“LR\-Free”考虑，但总的来说，这一块仍处于“方兴未艾”的阶段，有待更本质的探索。

## 文章小结

本文引入了新的恒等式，继续推广上一篇文章的结果，这使得我们可以从理论上定量地揭示权重平均与学习率衰减的联系。此外，我们还讨论了等权平均效果不如滑动平均、Schedule\-Free仍无法彻底摆脱Schedule等问题。
