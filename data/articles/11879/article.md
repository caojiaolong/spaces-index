# 让炼丹更科学一些（八）：多阶段训练的学习率

> 作者：苏剑林 · 科学空间 · 2026-08-31
>
> 原文：<https://spaces.ac.cn/archives/11879>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上篇文章[《让炼丹更科学一些（七）：步长调度与权重平均》](<https://spaces.ac.cn/archives/11804>)我们简单介绍了无调度（Schedule\-Free）学习率的工作，它试图通过某种权重平均来替代学习率调度，实现常数学习率就可以训出最优模型的效果。然而，上文我们也提到，如果不引入额外假设，这个最优的常数学习率也是依赖于训练步数的，所以也做不到真正的无调度。

这篇文章我们从多阶段训练角度，来重新思考这个问题，主要思想是将调度目标折中为“每个阶段结束时都接近最优”，使其在实践上更为简化与可行。

## 经典结论

我们先从一个经典的收敛结论出发，来理解为什么最优学习率依赖于训练步数。这个收敛结论我们首次介绍自[《让炼丹更科学一些（二）：将结论推广到无界域》](<https://spaces.ac.cn/archives/11469>)：
\begin{equation}\frac{\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]}{\sum_{t=1}^T \eta_t}\leq \frac{R^2}{2\sum_{t=1}^T \eta_t} + \frac{G^2}{2}\frac{\sum_{t=1}^T \eta_t^2}{\sum_{t=1}^T \eta_t}\triangleq f_T(\eta)\end{equation}
记号的具体含义我们就不重复介绍了。这个收敛结论是有代表性的，因为后续的[《让炼丹更科学一些（六）：自上而下的精妙构造》](<https://spaces.ac.cn/archives/11540>)和上文[《让炼丹更科学一些（七）：步长调度与权重平均》](<https://spaces.ac.cn/archives/11804>)都以它为基础。

现在我们要做的是，给定$R,G,T > 0$，求$\eta_1,\cdots,\eta_T$使得$f_T(\eta)$最小，这通过简单的不等式即可解决。首先易证$\sum_{t=1}^T \eta_t^2\geq (\sum_{t=1}^T \eta_t)^2/T$，于是
\begin{equation}f_T(\eta)\geq \frac{R^2}{2\sum_{t=1}^T \eta_t} + \frac{G^2}{2T}\sum_{t=1}^T \eta_t \geq \frac{RG}{\sqrt{T}}\triangleq f^*_T\end{equation}
两部分等号同时成立的条件是$\eta_1=\cdots=\eta_T=\frac{R}{G\sqrt{T}}$，依赖于$T$。如果换用真正跟$T$无关的学习率调度，比如$\eta_t \propto 1/\sqrt{t}$，那么至多能实现$f_T(\eta)=\mathcal{O}(\log T / \sqrt{T})$，并且还可以证明无论怎么改进，不依赖$T$的学习率都无法实现比$\mathcal{\Theta}(\sqrt{\log T / T})$更好的$f$。

## 转换思路

现在我们回来思考一下Schedule\-Free的动机。理想情况下，Schedule\-Free希望能通过某种不需要预知总训练步数$T$的方式，持续训练下去，使得我们不管在哪一步停止训练，都能得到截止当前步近乎最优的模型权重。

然而，这个完美目标看上去近乎不可能。因为即便在Schedule\-Free的凸优化假设下，我们可以将LR Decay与权重平均等价起来，实现常数学习率训练，但正如上一节所分析的，这个常数学习率的最优值也是依赖于$T$的，做不到随时停止都获得截止当前的最优解，除非我们进一步引入额外的假设。

既然如此，我们不妨再反思一下，对Schedule\-Free我们真正想要的是什么。首先常数学习率不是刚需的，假如某个时变学习率能训出好的效果，那么我们也能接受；其次，“任意时刻停止即最优”这当然是个好性质，但其实也不是强需求，我们主要是希望在多阶段训练时，每个阶段的终点能尽量逼近最优。

换句话说，当我们的训练被分成了多个阶段时，我们希望每个阶段结束时，都能得到一个尽可能接近最优的模型。这个阶段数不会太多，所以简单起见，先考虑两阶段训练，第一阶段训练$T_1$步，第二阶段训练$T_2 - T_1$步，共训练$T_2$步。

这里又可以接着分为“意外”和“计划”两种情况：“意外”是指本无计划续训，所以第一阶段会尽可能调到最优，决定续训后，第二阶段则是在现有基础上设法调到最优，这样就得到一个贪心解，缺点是第二阶段不论怎么调，结果都是相对次优的；至于“计划”情形，则是提前做了多阶段训练的规划，理论上我们可以更好地平衡两个阶段的效果。

## 贪心决策

我们先来看贪心版。第一阶段取最优，那自然是常数学习率$\eta_{(1)}^* = \frac{R}{G\sqrt{T_1}}$，此时第一阶段的$f$达到理论最优$f^*_{T_1}$。第二阶段我们需要在此基础上求$\eta_{T_1+1},\cdots,\eta_{T_2}$，最小化
\begin{equation}f_{T_2}(\eta) = \frac{R^2 + G^2(T_1 (\eta_{(1)}^*)^2 + \sum_{t=T_1+1}^{T_2} \eta_t^2)}{2(T_1 \eta_{(1)}^* + \sum_{t=T_1+1}^{T_2} \eta_t)}\end{equation}
基于同样的不等式$\sum_{t=T_1+1}^{T_2} \eta_t^2 \geq (\sum_{t=T_1+1}^{T_2} \eta_t)^2/(T_2-T_1)$，可知最小值依然在常数学习率取到，所以简化成
\begin{equation}f_{T_2}(\eta) = \frac{R^2 + G^2(T_1 (\eta_{(1)}^*)^2 + (T_2 - T_1) (\eta_{(2)})^2)}{2(T_1 \eta_{(1)}^* + (T_2 - T_1) \eta_{(2)})}\end{equation}
这可以通过配凑不等式或者直接求导完成，最小值点是
\begin{equation}\eta_{(2)}^* = \frac{R/G}{\sqrt{T_2^{\#}}}, \qquad f(\eta_{(2)}^*) = \frac{RG}{\sqrt{T_2^{\#}}}\end{equation}
这里
\begin{equation}T_2^{\#} = \frac{1}{2}T_2 + \frac{1}{2}\sqrt{T_1(2T_2 - T_1)}\end{equation}

## 等效步数

这里我们引入了记号$T_2^{\#}$，它的含义是“等效步数”。

假如没有第一阶段的限制，那么我们可以从一开始就取常数学习率$\frac{R}{G\sqrt{T_2}}$，训练结束的最小值达到$\frac{RG}{\sqrt{T_2}}$，但现在我们只能做到$\frac{RG}{\sqrt{T_2^{\#}}}$，第二阶段的学习率也是$\frac{R}{G\sqrt{T_2^{\#}}}$的形式，这就相当于说，有效步数从$T_2$变成了$T_2^{\#}$，不难验证
\begin{equation}T_2^{\#} \leq T_2\end{equation}
也就是说，两阶段贪心训练，在效果上相当于亏了一些步数。以$T_2=2T_1$为例，$T_2^{\#}\approx 1.866T_1$，相比$2T_1$，大致亏$6.7\%$的步数。注意如果$T_2\to\infty$，那么$T_2^{\#}/T_2\to 1/2$，也就是说如果无限续训练下去，亏的比例将会越来越多，最高可亏一半。

为什么要抽象出这个概念呢？因为“步数”这个概念是通用的，它使得结论更容易迁移。上述结论是基于SGD推导的，但实际训练通常是Adam、Muon等非SGD优化器，学习率等超参数的设置也并非直接用上述公式，而是通常基于搜出来的Scaling Law确定。

所以，SGD给出的各种结论并无法直接用，但如果我们能通过“等效步数”这一概念把相对变化量提炼出来，那实践所用Scaling Law中的步数输入，也可以类似地换成等效步数，从而得到一个启发式的修正结果，这是将SGD结论通向实践的关键思路。

## 计划情形

接下来看“计划”情形。此时$T_1,T_2$都是已知的，两个阶段的学习率可以联合调优。这里的核心问题在于，我们有两个交付目标——第一阶段末的模型和最终模型——需要设法平衡这两阶段效果。为此，我们引入minimax优化目标：
\begin{equation}\min_{\eta\geq 0} \max_{T\in\{T_1,T_2\}} \frac{f_T(\eta)}{f_T^*}\end{equation}
它希望每个阶段的实际效果与其理想值之比都尽可能小。这个优化目标的好处是不引入额外的超参数，同时易于推广到多阶段训练。引入新变量$m$，我们可以将它转为非线性规划问题
\begin{equation}\min\Big\{ m \,\,\Big|\,\, f_{T_1}(\eta)/f_{T_1}^*\leq m ,\,\, f_{T_2}(\eta)/f_{T_2}^*\leq m ,\,\, m\geq 1,\,\, \eta\geq 0\Big\}\end{equation}
把问题无量纲化：令$x_t = G\eta_t/R$，则对任意$T$都可以配平方得
\begin{equation}\frac{f_T(\eta)}{f_T^*} = \frac{\sqrt{T}(1 + \sum_{t=1}^T x_t^2)}{2\sum_{t=1}^T x_t}\leq m \iff \sum_{t=1}^T\left(x_t - \frac{m}{\sqrt{T}}\right)^2 \leq m^2 - 1\end{equation}
即$x_t$的可行集构成以$\frac{m}{\sqrt{T}}\boldsymbol{1}$为中心、$\sqrt{m^2-1}$为半径的一个超球，由此可见原问题等价于一个以$m$为目标的二次规划问题。

## 求解过程

对于两阶段情形，这个问题可以解析求解。注意第一个约束只涉及$x_1,\cdots,x_{T_1}$，第二个约束涉及全部$T_2$个变量。对于$t > T_1$的$x_t$，它们只出现在第二个约束中，所以直接取$x_t = m/\sqrt{T_2}$使其贡献为零即可。于是问题归结为：前$T_1$个$x_t$要同时落在两个球内
\begin{equation}\sum_{t=1}^{T_1}\left(x_t - \frac{m}{\sqrt{T_1}}\right)^2 \leq m^2 - 1, \qquad \sum_{t=1}^{T_1}\left(x_t - \frac{m}{\sqrt{T_2}}\right)^2 \leq m^2 - 1\end{equation}
两球半径都是$\sqrt{m^2-1}$，球心都在全一向量方向上，球心距的平方为
\begin{equation}T_1\left(\frac{m}{\sqrt{T_1}} - \frac{m}{\sqrt{T_2}}\right)^2 = m^2 (1 - \sqrt{\tau} )^2,\qquad \tau = T_1/T_2\end{equation}
两球有交集当且仅当球心距不超过两倍半径，即$m^2 (1 - \sqrt{\tau} )^2 \leq 4(m^2 - 1)$，由此解出最优的$m$：
\begin{equation}m^* = \frac{2}{\sqrt{4 - (1 - \sqrt{\tau})^2}}\end{equation}
对应的解可以取两球心连线的中点，即
\begin{equation}x_t = \left\{\begin{aligned} &\frac{m^*}{2}\left(\frac{1}{\sqrt{T_1}} + \frac{1}{\sqrt{T_2}}\right), & t \leq T_1 \\ &\frac{m^*}{\sqrt{T_2}}, & t > T_1 \end{aligned}\right.\end{equation}

## 结果分析

还原$\eta_t = Rx_t/G$得
\begin{equation}\eta_{(1)}^* = \frac{m^* R}{2G}\left(\frac{1}{\sqrt{T_1}} + \frac{1}{\sqrt{T_2}}\right),\qquad \eta_{(2)}^* = \frac{m^* R}{G\sqrt{T_2}}\end{equation}
有意思的是，$\eta_{(1)}^*$正是“只训$T_1$步”和“从头训$T_2$步”的两个单阶段最优常数学习率的平均（再乘上$m^*$）。从等效步数的角度看，$\eta_{(1)}^*$相当于预设了一个更大的训练步数，$\eta_{(2)}^*$则相当于预设了一个更小的训练步数
\begin{equation}T_1^{\#} = \left(\frac{R}{G\eta_{(1)}^*}\right)^2 = T_1\left(\frac{3-\sqrt{\tau}}{1 + \sqrt{\tau}}\right),\qquad T_2^{\#} = \left(\frac{R}{G\eta_{(2)}^*}\right)^2 = T_2\left(1-\frac{(1-\sqrt{\tau})^2}{4}\right)\end{equation}
也就是说，第一阶段只需要训练$T_1$步，在设置学习率时，假设它要训练$T_1^{\#}$步；两个阶段共训练$T_2$步，但在第二阶段设置学习率时，假设它要训练$T_2^{\#}$步。至于效果，此时$f_{T_1}/f_{T_1}^* = f_{T_2}/f_{T_2}^* = m^*$，即两个阶段的相对损失完全相等——这正是minimax目标的均衡特性：不偏向任何一个交付点。

还是以$T_2=2T_1$为例，此时$\tau=1/2$。代入可得$T_2^{\#} \approx 0.979 T_2$，所以在效果上，两个阶段相比于最优解都相当于亏损了约$2.1\%$的步数，从两个阶段的平均效果来看，要比贪心解优。学习率设置上，第一阶段的学习率应按照$T_1^{\#} \approx 1.343 T_1$的等效步数来设，第二阶段的学习率则按照$T_2^{\#} \approx 0.979 T_2$的等效步数来设。

另一个极限是$\tau\to 0$时，$T_2^{\#}\to 0.75 T_2$，也就是说当$T_2$阶段的步数足够多时，minimax解在终点至多亏$25\%$的步数，比贪心解的$50\%$要更好一些。

## 多段推广

不管是“意外”情形的贪心解，还是“计划”情形的minimax解，理论上都很容易推广到多阶段训练。贪心解自不必说，我们主要展开一下minimax解。设有$K$个交付点$T_1 < T_2 < \cdots < T_K$，minimax目标变为
\begin{equation}\min_{\eta\geq 0}\ \max_{T\in\{T_1,\cdots,T_K\}} \frac{f_T(\eta)}{f_T^*}\end{equation}
同样的无量纲化给出$K$个球约束
\begin{equation}\sum_{t=1}^{T_k}\left(x_t - \frac{m}{\sqrt{T_k}}\right)^2 \leq m^2 - 1,\qquad k = 1,2,\cdots,K\end{equation}
虽然变量理论上有$T_K$个，但依然可以证明最优解总可以取成“分段常数”形式，所以问题缩减为只有$K+1$个变量小规模规划。当$K\geq 3$时一般没有简洁的闭式解，但数值求解并不困难。

## 文章小结

本文从多阶段训练的角度重新审视了“无调度学习率”，通过将调度目标改为“每个阶段结束时都接近最优”，使得实践上更简化且更可行，进一步地，我们提炼出“等效步数”的概念，使得结论可能迁移到实践所用的优化器和Scaling Law上。
