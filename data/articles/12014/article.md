# Muon背后的优化理论：1\. 全量、随机与动量

> 作者：苏剑林 · 科学空间 · 2026-11-23
>
> 原文：<https://spaces.ac.cn/archives/12014>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

关于优化器，我们可以找到很多经典的收敛结论，比如“[炼丹](<https://spaces.ac.cn/tag/%E7%82%BC%E4%B8%B9/>)”系列文章介绍的基于凸假设和遗憾界的一系列结果。除此之外，还有很多从不同角度切入、基于不同假设的分析，这些假设有强有弱，所得结论适用范围也有所不同，如何平衡理论推导所需要的简洁性与实际训练时的复杂性，是这个领域最艰难的问题之一。

本系列文章围绕Muon优化器介绍另一条分析路线，主要内容是来自论文[《Understanding Gradient Orthogonalization for Deep Learning via Non\-Euclidean Trust\-Region Optimization》](<https://papers.cool/arxiv/2503.12645>)的一系列分析方法和结论，并补充自己的一些思考。

## 背景介绍

这篇文章主要关注的是LMO（Linear Minimization Oracle）类优化器，它的特点是：先选定一种范数，再在单位球内寻找与梯度内积最大的方向，并沿该方向的反方向更新参数。SignSGD、Muon等优化器都可以纳入这个框架，区别在于参数类型和所选范数的不同。

LMO优化器的提法出自[《Training Deep Learning Models with Norm\-Constrained LMOs》](<https://papers.cool/arxiv/2502.07529>)，但在这之前，它已经被广泛提及，比如我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)首次介绍Muon时，就已经用过类似提法。更多的中文资料，还可以参考我们的“[流形上的最速下降](<https://spaces.ac.cn/archives/11196>)”系列博文。

接下来，我们主要围绕着论文[《Understanding Gradient Orthogonalization for Deep Learning via Non\-Euclidean Trust\-Region Optimization》](<https://papers.cool/arxiv/2503.12645>)进行讨论。这是一篇为LMO类优化器提供统一收敛分析的工作，它不仅讨论了Momentum、Weight Decay等经典收敛分析中鲜有涉及的对象，还提供了梯度范数和损失函数两种结论形式，可谓应有尽有。

值得指出的是，这篇论文还是后续论文[《Deriving Hyperparameter Scaling Laws via Modern Optimization Theory》](<https://papers.cool/arxiv/2603.15958>)和[《On the Role of Batch Size in Stochastic Conditional Gradient Methods》](<https://papers.cool/arxiv/2603.21191>)从理论上推导Scaling Law的基础工作，由此可见它的实践意义，并非纸上谈兵的纯理论工作。

我们会分多篇文章逐一介绍这些内容，这篇文章先集中在比较基础的部分结论。

## 范数对偶

为了便于讨论，我们需要先准备一些概念。设$\Vert\cdot\Vert$表示某种范数，LMO类优化器的更新方向，是通过最大化内积来定义的对偶梯度：
\begin{equation}\newcommand{argmax}{\mathop{\text{argmax}}}\boldsymbol{g}^* \triangleq \argmax_{\Vert\boldsymbol{\phi}\Vert\leq 1} \langle \boldsymbol{g}, \boldsymbol{\phi}\rangle\end{equation}
其中$\boldsymbol{g} = \nabla_{\boldsymbol{w}}\ell(\boldsymbol{w})$是梯度，$\ell$是损失函数，$\boldsymbol{w}$代表向量参数（矩阵参数则记为大写的$\boldsymbol{W}$，其梯度则记为$\boldsymbol{G}$）。从定义可知必然有$\Vert\boldsymbol{g}^*\Vert\leq 1$。同时，可以定义对偶范数
\begin{equation}\Vert\boldsymbol{g}\Vert_* \triangleq \langle \boldsymbol{g}, \boldsymbol{g}^*\rangle = \max_{\Vert\boldsymbol{\phi}\Vert\leq 1} \langle \boldsymbol{g}, \boldsymbol{\phi}\rangle\end{equation}
这里要注意的是，$\Vert\cdot\Vert_*$也是一个范数，即它也满足三角不等式等范数性质，但从上述定义不容易看出，请读者自行完成证明。由这个定义，还可以直接诱导出广义的柯西不等式
\begin{equation}\langle\boldsymbol{g}, \boldsymbol{\phi}\rangle \leq \Vert\boldsymbol{g}\Vert_* \, \Vert\boldsymbol{\phi}\Vert\end{equation}
所有LMO类优化器的更新规则，都可以写成
\begin{equation}\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta_t \boldsymbol{g}_t^*,\qquad \boldsymbol{g}_t = \nabla_{\boldsymbol{w}_t}\ell(\boldsymbol{w}_t)\end{equation}
三个经典例子是（NSGD的“N”是“Normalized”）
$$\newcommand{sign}{\mathop{\text{sign}}}\newcommand{msign}{\mathop{\text{msign}}}
\begin{array}{c|cccc}
\hline
& \text{参数类型} & \text{原始范数} & \text{对偶范数} & \text{对偶梯度} \\
\hline
\text{NSGD} & \text{向量} & l_2 & l_2 & \boldsymbol{g}/\Vert\boldsymbol{g}\Vert_2\\
\text{SignSGD} & \text{向量} & l_{\infty} & l_1 & \sign(\boldsymbol{g}) \\
\text{Muon} & \text{矩阵} & \text{谱范数} & \text{核范数} & \msign(\boldsymbol{G}) \\
\hline
\end{array}$$
至于什么参数该选用什么范数的问题，可以参考我们之前的“[MuP之上](<https://spaces.ac.cn/archives/11605>)”系列文章，本文不作展开。

## 光滑假设

接着，我们还需要给梯度引入Lipschitz连续性假设，即假设任意两点的梯度满足不等式
\begin{equation}\Vert\nabla_{\boldsymbol{w}_1}\ell(\boldsymbol{w}_1) - \nabla_{\boldsymbol{w}_2}\ell(\boldsymbol{w}_2)\Vert_* \leq L \Vert\boldsymbol{w}_1 - \boldsymbol{w}_2\Vert\label{eq:grad-L-smooth}\end{equation}
其中$L$是某个给定的常数。这其实也是优化分析中的经典假设，又叫（关于损失的）$L$\-光滑性，因为它的主要作用是推出损失函数的不等式
\begin{equation}\ell(\boldsymbol{w} + \Delta\boldsymbol{w}) \leq \ell(\boldsymbol{w}) + \langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w}), \Delta \boldsymbol{w}\rangle + \frac{L}{2}\Vert\Delta\boldsymbol{w}\Vert^2\label{eq:loss-L-bound}\end{equation}
证明如下：
\begin{equation}\begin{aligned}
\ell(\boldsymbol{w} + \Delta\boldsymbol{w}) - \ell(\boldsymbol{w}) =&\, \int_0^1 d\,\ell(\boldsymbol{w} + t\Delta\boldsymbol{w}) \\
=&\, \int_0^1\langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w} + t\Delta\boldsymbol{w}), \Delta\boldsymbol{w}\rangle dt \\
=&\, \langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w}), \Delta\boldsymbol{w}\rangle + \int_0^1\langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w} + t\Delta\boldsymbol{w}) - \nabla_{\boldsymbol{w}}\ell(\boldsymbol{w}), \Delta\boldsymbol{w}\rangle dt \\
\leq&\, \langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w}), \Delta\boldsymbol{w}\rangle + \int_0^1\Vert\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w} + t\Delta\boldsymbol{w}) - \nabla_{\boldsymbol{w}}\ell(\boldsymbol{w})\Vert_* \Vert\Delta\boldsymbol{w}\Vert dt \\
\leq&\, \langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w}), \Delta\boldsymbol{w}\rangle + \int_0^1 t L\Vert\Delta\boldsymbol{w}\Vert^2 dt \\
=&\, \langle\nabla_{\boldsymbol{w}}\ell(\boldsymbol{w}), \Delta\boldsymbol{w}\rangle + \frac{L}{2} \Vert\Delta\boldsymbol{w}\Vert^2
\end{aligned}\end{equation}
从近似的视角看，式$\eqref{eq:loss-L-bound}$实际上就是将损失函数的泰勒展开式的二阶项，替以简单的$\frac{L}{2} \Vert\Delta\boldsymbol{w}\Vert^2$。

那么，假设$\eqref{eq:grad-L-smooth}$多大程度上是成立的呢？很遗憾，全局看的话，可以说基本都不成立。比如，简单的$\ell(w)=w^4$在$\mathbb{R}$上就已经不满足这个假设；又比如，训练过程中时常出现的Grad Spike，也暗示它对神经网络也普遍不成立。我们只能无力地期望，对于能平稳训练和收敛的模型，其训练轨迹附近能近似满足，从而相关结论是有参考价值的。

更宽松的假设也是有探索的，比如我们之前简单介绍过的[(L<sub>0</sub>, L<sub>1</sub>)\-smooth](<https://spaces.ac.cn/archives/7469>)，这些我们将来有机会再作推广。

## 全量下降

做好上述准备工作后，我们就可以进入结论的推导了。作为最简单的情况，我们先考虑常数学习率的非随机下降（全量下降）：
\begin{equation}\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \boldsymbol{g}_t^*,\qquad \boldsymbol{g}_t = \nabla_{\boldsymbol{w}_t}\ell(\boldsymbol{w}_t)\end{equation}
约定迭代从$t=1$开始，即$\boldsymbol{w}_1$是初始化权重。根据式$\eqref{eq:loss-L-bound}$，得到
\begin{equation}\ell(\boldsymbol{w}_{t+1}) - \ell(\boldsymbol{w}_t)\leq \underbrace{\langle\boldsymbol{g}_t, -\eta\boldsymbol{g}_t^*\rangle}_{=-\eta \Vert\boldsymbol{g}_t\Vert_*} + \frac{L}{2}\underbrace{\Vert\eta\boldsymbol{g}_t^*\Vert^2}_{\leq \eta^2} \leq -\eta \Vert\boldsymbol{g}_t\Vert_* + \frac{L \eta^2}{2}\end{equation}
两端对$t=1,2,\cdots,T$求和得
\begin{equation}\ell(\boldsymbol{w}_{T+1}) - \ell(\boldsymbol{w}_1)\leq  - \eta \sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert_* + \frac{L\eta^2 T}{2}\label{eq:gd-loss}\end{equation}
继而整理得
\begin{equation}\frac{1}{T}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert_* \leq \frac{\ell(\boldsymbol{w}_1) - \ell(\boldsymbol{w}_{T+1})}{T\eta} + \frac{L\eta}{2}\label{eq:gd-grad}\end{equation}
读者可能会有疑问：式$\eqref{eq:gd-loss}$看上去就已经是损失函数的收敛结论了，为什么还要变换成式$\eqref{eq:gd-grad}$？式$\eqref{eq:gd-loss}$在某些情况下确实也能用，但它没有出现最小值点$\ell_{\min}$，反而有一个初始值$\ell(\boldsymbol{w}_1)$，这意味着它不能告诉我们当前点距离理想的终点还有多远。

而式$\eqref{eq:gd-grad}$的左端是梯度范数，实际训练通常只能找到一个驻点（梯度等于零），所以左端与零的差距，一定程度上代表了模型与终点的距离，这是我们想要的信号。特别地，整个推导并没有用到凸性之类的假设，所以这个结论适用于非凸函数，或者说，非凸函数通常只能得到左端是梯度范数的收敛结论。

最后，我们还有$\min\limits_{1\leq t \leq T} \Vert\boldsymbol{g}_t\Vert_* \leq \frac{1}{T}\sum\limits_{t=1}^T\Vert\boldsymbol{g}_t\Vert_*$和$\ell(\boldsymbol{w}_1) - \ell(\boldsymbol{w}_{T+1})\leq \ell(\boldsymbol{w}_1) - \ell_{\min}\triangleq \Delta_0$，所以$\eqref{eq:gd-grad}$还可以放缩成
\begin{equation}\min_{1\leq t \leq T} \Vert\boldsymbol{g}_t\Vert_*\leq \frac{\Delta_0}{T\eta} + \frac{L\eta}{2}\end{equation}
这告诉我们训练过程至少有1步，对应的梯度范数能不超过右端的值，即训练过程至少有一个Checkpoint是能直接满足条件的，而不用诉诸权重平均等技巧。

## 随机梯度

实际训练模型时，我们通常只能通过采样小批量数据来估计梯度，这就导致了随机下降：
\begin{equation}\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta \tilde{\boldsymbol{g}}_t^*,\qquad \tilde{\boldsymbol{g}}_t = \nabla_{\boldsymbol{w}_t}\ell(\boldsymbol{w}_t, \boldsymbol{x}_t)\end{equation}
其中$\boldsymbol{x}_t$是独立同分布采样出来的数据，Batch Size为$B$，我们有$\ell(\boldsymbol{w}_t)=\mathbb{E}_{\boldsymbol{x}_t}[\ell(\boldsymbol{w}_t, \boldsymbol{x}_t)]$。此时代入式$\eqref{eq:loss-L-bound}$得
\begin{equation}\begin{aligned}
\ell(\boldsymbol{w}_{t+1}) - \ell(\boldsymbol{w}_t) \leq&\, \langle\boldsymbol{g}_t, -\eta\tilde{\boldsymbol{g}}_t^*\rangle + \frac{L}{2}\Vert\eta\tilde{\boldsymbol{g}}_t^*\Vert^2 \\
=&\, \underbrace{\langle\tilde{\boldsymbol{g}}_t, -\eta\tilde{\boldsymbol{g}}_t^*\rangle}_{=-\eta\Vert\tilde{\boldsymbol{g}}_t\Vert_*} + \underbrace{\langle\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t, -\eta\tilde{\boldsymbol{g}}_t^*\rangle}_{\leq \Vert\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t\Vert_* \underbrace{\Vert\eta\tilde{\boldsymbol{g}}_t^*\Vert}_{\leq\eta}} + \frac{L}{2}\underbrace{\Vert\eta\tilde{\boldsymbol{g}}_t^*\Vert^2}_{\leq\eta^2} \\
\leq&\, -\eta\Vert\tilde{\boldsymbol{g}}_t\Vert_* + \eta\Vert\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t\Vert_* + \frac{L\eta^2}{2} \\
=&\, -\eta\Vert\boldsymbol{g}_t\Vert_* + \underbrace{\eta\Vert\boldsymbol{g}_t\Vert_* - \eta\Vert\tilde{\boldsymbol{g}}_t\Vert_*}_{\leq \eta\Vert\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t\Vert_*} + \eta\Vert\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t\Vert_* + \frac{L\eta^2}{2} \\
\leq &\, -\eta\Vert\boldsymbol{g}_t\Vert_* + 2\eta\Vert\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t\Vert_* + \frac{L\eta^2}{2} \\
\end{aligned}\label{eq:l-g-s-detail}\end{equation}
还是对$t=1,2,\cdots,T$求和，整理得
\begin{equation}\frac{1}{T}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert_* \leq \frac{\ell(\boldsymbol{w}_1) - \ell(\boldsymbol{w}_{T+1})}{T\eta} + \frac{2}{T}\sum_{t=1}^T\Vert\boldsymbol{g}_t - \tilde{\boldsymbol{g}}_t\Vert_* + \frac{L\eta}{2}\end{equation}
因为带了随机性，我们需要在两边取期望（对全体$\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_T$），以得到一个确定性的结论，此时右端会出现$\mathbb{E}[\Vert\boldsymbol{g}_t-\tilde{\boldsymbol{g}}_t\Vert_*]$，注意根据定义$\mathbb{E}_{\boldsymbol{x}_t}[\tilde{\boldsymbol{g}}_t] = \boldsymbol{g}_t$，所以其含义正是随机向量与其均值的平均波动。如果是$l_2$范数，那么根据Jensen不等式有
\begin{equation}\mathbb{E}_{\boldsymbol{x}_t}[\Vert\boldsymbol{g}_t-\tilde{\boldsymbol{g}}_t\Vert_2] \leq \sqrt{\mathbb{E}_{\boldsymbol{x}_t}[\Vert\boldsymbol{g}_t-\tilde{\boldsymbol{g}}_t\Vert_2^2]} \leq \sqrt{\frac{\sigma^2}{B}}\end{equation}
这里$\mathbb{E}_{\boldsymbol{x}_t}[\Vert\boldsymbol{g}_t-\tilde{\boldsymbol{g}}_t\Vert_2^2]$正是方差的定义，所以它反比于$B$，注意这里我们要考虑每一步的方差，它们不一定相同，我们将它写成$\leq\sigma^2/B$，是假设它们有共同上界。另一边，$\Vert\cdot\Vert_*$不一定是$\Vert\cdot\Vert_2$，这时候如果要得到严格的上界结论，只能借助于范数的等价性，认为存在常数$\rho > 0$，使得$\Vert\cdot\Vert_*\leq \rho\Vert\cdot\Vert_2$恒成立，那么$\mathbb{E}[\Vert\boldsymbol{g}_t-\tilde{\boldsymbol{g}}_t\Vert_*]\leq \rho\sigma/\sqrt{B}$，于是
\begin{equation}\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert_*\right] \leq \frac{\Delta_0}{T\eta} + \frac{2\rho\sigma}{\sqrt{B}} + \frac{L\eta}{2}\label{eq:g-s}\end{equation}
范数等价性的问题是，为了保证不等关系，$\rho$通常要取得比较大，导致上界比较宽松。如果想要近似关系的话，我们可以假设存在一个适当的$\rho$，使得两者平均来说是近似的，这样右端的形式不变，但作为一个约等关系出现。

## 引入动量

接下来把动量考虑进来（$0\leq\beta < 1$）
\begin{equation}\begin{aligned}
\tilde{\boldsymbol{g}}_t =&\, \nabla_{\boldsymbol{w}_t}\ell(\boldsymbol{w}_t, \boldsymbol{x}_t) \\
\boldsymbol{m}_{t+1} =&\, \beta \boldsymbol{m}_t + (1-\beta) \tilde{\boldsymbol{g}}_t \\
\boldsymbol{w}_{t+1} =&\, \boldsymbol{w}_t - \eta \boldsymbol{m}_{t+1}^*
\end{aligned}\end{equation}
将推导$\eqref{eq:l-g-s-detail}$的$\tilde{\boldsymbol{g}}_t$换成$\boldsymbol{m}_{t+1}$，将得到
\begin{equation}\ell(\boldsymbol{w}_{t+1}) - \ell(\boldsymbol{w}_t) \leq -\eta\Vert\boldsymbol{g}_t\Vert_* + 2\eta\Vert\boldsymbol{g}_t - \boldsymbol{m}_{t+1}\Vert_* + \frac{L\eta^2}{2}\end{equation}
整理一下，两端取期望，得到
\begin{equation}\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert_*\right] \leq \frac{\Delta_0}{T\eta} + \frac{2}{T}\sum_{t=1}^T\mathbb{E}[\Vert\boldsymbol{g}_t - \boldsymbol{m}_{t+1}\Vert_*] + \frac{L\eta}{2}\label{eq:g-m-base}\end{equation}
所以接下来的任务是计算$\mathbb{E}[\Vert\boldsymbol{g}_t - \boldsymbol{m}_{t+1}\Vert_*]$。

计算之前，我们不妨先定性地把“动量究竟起到什么作用”想清楚。一个基本作用是降噪，即通过将相邻的梯度平均起来模拟大Batch Size的效果，这也是接下来主要讨论的角度。不过需要指出的是，降噪并不是动量的唯一作用，即便是没有随机性的全量下降，打开动量往往也能有加速作用，这一块我们将来再作分析。

在降噪视角下，完美的优化器是$\boldsymbol{g}_t^*$（全量梯度的对偶），$\boldsymbol{m}_{t+1}$只不过是为了逼近$\boldsymbol{g}_t$（从而让$\boldsymbol{m}_{t+1}^*$在更新效果上逼近$\boldsymbol{g}_t^*$）。这里的关键是要同时考虑“噪声”和“漂移”两个效应：$\tilde{\boldsymbol{g}}_t$可以看作$\boldsymbol{g}_t$加噪而来，$\tilde{\boldsymbol{g}}_{t-1}$也可以看作$\boldsymbol{g}_{t-1}$加噪而来，两者平均确实能降噪，但$\boldsymbol{g}_t$是不等于$\boldsymbol{g}_{t-1}$的，所以这就会引入偏差。

为了观察这两个效应，我们将动量的更新规则等价地改写成
\begin{equation}\boldsymbol{m}_{t+1} - \boldsymbol{g}_t = \beta (\boldsymbol{m}_t - \boldsymbol{g}_{t-1}) + (1-\beta) \underbrace{(\tilde{\boldsymbol{g}}_t - \boldsymbol{g}_t)}_{\text{噪声}} + \beta \underbrace{(\boldsymbol{g}_{t-1} - \boldsymbol{g}_t)}_{\text{漂移}}\end{equation}
根据约定，迭代从$t=1$开始，这里的$\boldsymbol{m}_1,\boldsymbol{g}_0$都是可以任意选择的，取$\boldsymbol{m}_1=\boldsymbol{g}_0$，递归展开得
\begin{equation}\boldsymbol{m}_{t+1} - \boldsymbol{g}_t = (1-\beta) \sum_{k=1}^t \beta^{t-k}(\tilde{\boldsymbol{g}}_k - \boldsymbol{g}_k) + \beta \sum_{k=1}^t \beta^{t-k}(\boldsymbol{g}_{k-1} - \boldsymbol{g}_k)\end{equation}
两端取范数$\Vert\cdot\Vert_*$，由三角不等式可知
\begin{equation}\Vert\boldsymbol{m}_{t+1} - \boldsymbol{g}_t\Vert_* \leq (1-\beta) \left\Vert\sum_{k=1}^t \beta^{t-k}(\tilde{\boldsymbol{g}}_k - \boldsymbol{g}_k)\right\Vert_* + \beta \sum_{k=1}^t \beta^{t-k}\Vert\boldsymbol{g}_{k-1} - \boldsymbol{g}_k\Vert_*\label{eq:m-g-dist}\end{equation}

## 逐一计算

先看叠加起来的噪声项，对于任意$k > j$，我们有
\begin{equation}\begin{aligned}
\mathbb{E}[\langle \tilde{\boldsymbol{g}}_k - \boldsymbol{g}_k, \tilde{\boldsymbol{g}}_j - \boldsymbol{g}_j\rangle] =&\, \mathbb{E}_{\boldsymbol{x}_1,\cdots,\boldsymbol{x}_{k-1}}[\mathbb{E}_{\boldsymbol{x}_k}[\langle \tilde{\boldsymbol{g}}_k - \boldsymbol{g}_k, \tilde{\boldsymbol{g}}_j - \boldsymbol{g}_j\rangle]] \\
=&\, \mathbb{E}_{\boldsymbol{x}_1,\cdots,\boldsymbol{x}_{k-1}}[\langle \mathbb{E}_{\boldsymbol{x}_k}[\tilde{\boldsymbol{g}}_k - \boldsymbol{g}_k], \tilde{\boldsymbol{g}}_j - \boldsymbol{g}_j\rangle] \\
=&\, 0
\end{aligned}\end{equation}
即叠加噪声方差的交叉项自动为零。还是假设每一项的方差不超过$\sigma^2/B$，那么在以$\beta^{t-k}$为权的叠加之下，方差不超过
\begin{equation}\frac{\sigma^2}{B}\sum_{k=1}^t (\beta^{t-k})^2 = \frac{\sigma^2}{B}\frac{1-\beta^{2t}}{1-\beta^2}\leq \frac{\sigma^2}{B}\frac{1}{1-\beta^2}\end{equation}
那么对式$\eqref{eq:m-g-dist}$两边取期望，代入上述放缩，并结合范数等价性得到
\begin{equation}\mathbb{E}[\Vert\boldsymbol{m}_{t+1} - \boldsymbol{g}_t\Vert_*] \leq \frac{\rho\sigma}{\sqrt{B}}\underbrace{\frac{1-\beta}{\sqrt{1-\beta^2}}}_{\leq \sqrt{1-\beta}} +  \beta L\eta \underbrace{\sum_{k=2}^t \beta^{t-k}}_{\leq \frac{1}{1-\beta}} + \beta^t G_0\end{equation}

对于漂移项，我们由梯度的假设$\eqref{eq:grad-L-smooth}$得
\begin{equation}\Vert\boldsymbol{g}_{k-1} - \boldsymbol{g}_k\Vert_* = \Vert\nabla_{\boldsymbol{w}_{k-1}}\ell(\boldsymbol{w}_{k-1}) - \nabla_{\boldsymbol{w}_k}\ell(\boldsymbol{w}_k)\Vert_* \leq L \Vert \boldsymbol{w}_{k-1} - \boldsymbol{w}_k\Vert = L\Vert \eta \boldsymbol{m}_k^*\Vert \leq L\eta\end{equation}
但注意，这个只对$k\geq 2$成立。我们还没确定$\boldsymbol{g}_0$（即$\boldsymbol{m}_1$）取什么，所以不能认为它对于$k=1$也成立。原论文的选择是$\boldsymbol{m}_1=\tilde{\boldsymbol{g}}_1$，对应的首项为$\Vert\tilde{\boldsymbol{g}}_1 - \boldsymbol{g}_1\Vert_*$，正好是噪声范数的形式，可以按噪声项处理；更常见的选择是$\boldsymbol{m}_1=\boldsymbol{0}$，对应的首项是$G_0=\Vert\boldsymbol{g}_1\Vert_*$，它是一个常数，本文按后者处理。

将这些放缩代回$\eqref{eq:g-m-base}$得
\begin{equation}\begin{aligned}
\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert_*\right] \leq&\, \frac{\Delta_0}{T\eta} + \frac{2\rho\sigma\sqrt{1-\beta}}{\sqrt{B}} + \frac{2\beta L\eta}{1-\beta} + \frac{2G_0}{T}\sum_{t=1}^T \beta^t + \frac{L\eta}{2} \\
\leq&\, \frac{\Delta_0}{T\eta} + \frac{2\rho\sigma\sqrt{1-\beta}}{\sqrt{B}} + \frac{2\beta L\eta}{1-\beta} + \frac{2\beta G_0}{(1-\beta)T} + \frac{L\eta}{2} \\
\end{aligned}\end{equation}
从上式我们可以清晰地看到，降噪有两种方式——增大$B$或者增大$\beta$——增大$\beta$对实现来说比较“便宜”，但它在降噪的同时也会引入偏差（旧梯度带来的滞后性），最优的$\beta$应该在噪声和偏差方面取得平衡。特别地，当$\beta=0$时，结论自动退化为无动量的式$\eqref{eq:g-s}$，说明推导是自洽的。

## 文章小结

本文沿着一条统一的路线，围绕着以Muon为代表的LMO类优化器进行推导，给出了它们在全量、随机和动量三种情形下的梯度范数收敛上界。结果表明，随机梯度会引入随Batch Size增大而减小的噪声，动量虽然也能降噪，却也会因旧梯度的滞后带来额外偏差，如何在噪声和偏差之间找到最优超参数，这些我们将来都会讨论。
