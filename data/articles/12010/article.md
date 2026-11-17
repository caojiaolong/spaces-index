# 最佳学习率调度的变分估计

> 作者：苏剑林 · 科学空间 · 2026-11-17
>
> 原文：<https://spaces.ac.cn/archives/12010>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

近日，笔者刷到一篇论文[《Your LR Schedule May Not Be Principled: Effective Learning Rate Scheduling in Weight\-Normalized Training》](<https://papers.cool/arxiv/2610.40362>)，里边分享了一种估计最优学习率的新思路：先将模型的训练动态简化成一个一维微分方程，然后在这个微分方程约束之下，基于变分思想去最大化某个目标，从而得到最佳调度。

这个过程最吸引人的地方，不是得到了一个具体公式，而是提供了一个把“训练直觉”翻译成“调度形状”的一般框架。接下来，笔者将分享自己阅读过后的理解，并做一些延伸讨论。

## 更新规则

原论文考虑的是Weight Norm（下面简称“WN”）类优化器，所以我们也从WN出发，其更新规则为
\begin{equation}\boldsymbol{\theta}_{t+1} = \gamma\times\frac{\boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t}{\Vert\boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t\Vert}\end{equation}
这里的$\Vert\cdot\Vert$是L2范数，$\gamma$是设定的模长，$\boldsymbol{\theta}_t$是参数，$\boldsymbol{u}_t$是更新量。注意在WN类优化器中，我们通常会把更新量的径向分量去掉，所以$\boldsymbol{\theta}_t\cdot\boldsymbol{u}_t=0$，于是
\begin{equation}\Vert\boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t\Vert=\sqrt{\gamma^2 + \eta_t^2\Vert\boldsymbol{u}_t\Vert^2}\end{equation}
进一步地，在自适应学习率优化器（比如Adam或Muon）中，更新量模长$\Vert\boldsymbol{u}_t\Vert$通常近似为常数，即便去掉径向分量也不会改变这一点，所以我们直接设$\Vert\boldsymbol{u}_t\Vert=\mu$，得到
\begin{equation}\boldsymbol{\theta}_{t+1} = \gamma\times\frac{\boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t}{\sqrt{\gamma^2 + \mu^2\eta_t^2}} = (1 + \gamma^{-2}\mu^2\eta_t^2)^{-1/2}(\boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t)\end{equation}
按$\eta_t$展开到二阶得
\begin{equation}\boldsymbol{\theta}_{t+1} \approx  (1 - b\eta_t^2)\boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t\label{eq:wn-wd-approx}\end{equation}
其中$b=\mu ^2/2\gamma^2$。由此可见，WN并不是没有Weight Decay（下面简称“WD”），而是它的Normalize操作隐式地起到了WD效果，而接下来的分析仍然要回到显式WD的近似形式上。

## 变分问题

假设理想终点为$\boldsymbol{\theta}^*$，它也满足$\Vert\boldsymbol{\theta}^*\Vert=\gamma$，那么我们可以用$x_t\triangleq\boldsymbol{\theta}_t\cdot \boldsymbol{\theta}^*$来度量当前状态与理想终点的距离，它越大越好。在式$\eqref{eq:wn-wd-approx}$两边同时点乘$\boldsymbol{\theta}^*$，我们得到
\begin{equation}x_{t+1} \approx  (1 - b\eta_t^2)x_t + \eta_t \boldsymbol{u}_t\cdot \boldsymbol{\theta}^*\end{equation}
将$x_t$视为关于$t$的连续函数，用导数$\dot{x}_t$近似差分，把它当成精确成立的约束，得到微分方程
\begin{equation}\dot{x}_t =  \eta_t \boldsymbol{u}_t\cdot \boldsymbol{\theta}^* - b\eta_t^2 x_t\end{equation}
接下来做一个简化：假设$\boldsymbol{u}_t\cdot \boldsymbol{\theta}^*$为常数$a > 0$，这意味着训练过程自始至终以相同的速度奔向终点，这假设可能会被质疑过强，但我们先做着，后面再考虑把它换成一般的$a_t$。此时微分方程简化成
\begin{equation}\dot{x}_t =  a\eta_t - b\eta_t^2 x_t\label{eq:wn-wd-ode}\end{equation}
给定初始值$x_0$和训练步数$T$，我们的目标应该是让$\boldsymbol{\theta}_T$尽可能接近理想终点$\boldsymbol{\theta}^*$，那就可以最大化$x_T$，这里唯一可变的就是$\eta_t$，所以问题就是寻找一个函数$\eta_t$，使得$x_T$最大化，用公式写出来是
\begin{equation}\max_{\eta\geq 0} x_T \qquad\text{s.t.}\qquad \dot{x}_t =  a\eta_t - b\eta_t^2 x_t\end{equation}
这实际上属于泛函中的“变分法”范畴，也就是说，我们将最佳学习率调度抽象成了一个变分问题。不过碰巧的是，这个变分问题刚好有一个非常初等的求解过程，以至于我们不需要动用复杂的变分工具。

## 初等过程

在式$\eqref{eq:wn-wd-ode}$两边乘$2x_t$，左边是$x_t^2$的导数，右端配平方，整理得
\begin{equation}\frac{d}{dt}x_t^2 =  \frac{a^2}{2b} - 2b\left(\eta_t x_t - \frac{a}{2b}\right)^2 \leq \frac{a^2}{2b}\end{equation}
两端对$t$从$0\sim T$积分得
\begin{equation}x_T^2 - x_0^2 \leq \frac{a^2 T}{2b}\end{equation}
即$x_T$的最大值是$\sqrt{x_0^2 + a^2 T/2b}$，取等号的条件是
\begin{equation}\eta_t x_t = \frac{a}{2b}\qquad\Rightarrow\qquad \eta_t = \frac{a}{2b x_t}\end{equation}
此时微分方程简化成
\begin{equation}\frac{d}{dt}x_t^2 =  \frac{a^2}{2b}\end{equation}
显然$x_t=\sqrt{x_0^2 + a^2 t/2b}$，于是$\eta_t$的最优解是
\begin{equation}\eta_t^* = \frac{a}{2b \sqrt{x_0^2 + a^2 t/2b}} = \frac{\eta_0}{\sqrt{1 + 2b\eta_0^2 t}},\qquad \eta_0 = \frac{a}{2b x_0}\end{equation}
即（带时间平移的）逆平方根衰减。

## 一般结果

有意思的是，我们求$x_T$的最大值，预期结果应该依赖于$T$，但结果居然不是，这意味着$\eta_t^*$对任意终点都是最优的，是真正的“Schedule\-Free”。这有点过于理想了，实际上不大可能，原论文接下来是将它改造成依赖于终点的形式来使用，这部分我们不展开讨论。

我们知道，当前的结果依赖于假设学习速率$\boldsymbol{u}_t\cdot \boldsymbol{\theta}^*$为常数$a > 0$。不过，即便将静态的$a$换成动态的$a_t$，上一节的推导依然可行，最终答案是
\begin{equation}x_t^* = \sqrt{x_0^2 + \frac{1}{2b}\int_0^t a_s^2 ds},\qquad \eta_t^* = \frac{a_t}{2b x_t^*}\end{equation}
那么，常数速率究竟在多大程度上是科学或者不科学的呢？直觉上想，这显然是不成立的，按照经验，训练越接近后期，学习速率应该越慢，所以这好歹是个单调递减的函数才对。但另一方面，对于足够大的模型，在我们能提供的有限训练数据内，模型可能都还处于欠训练阶段，此时这个假设或许能成立。

由于WN限定了$\Vert\boldsymbol{\theta}_t\Vert=\Vert\boldsymbol{\theta}^*\Vert=\gamma$，所以$x_t\leq \gamma^2$。根据“越接近最优点，学习速率越慢”的直观假设，我们也可以考虑将静态$a$换成$a\cdot(\gamma^2-x_t)$之类的函数，这样它就自动随着$x_t$的变大而缩小了。这时候的微分方程是$\dot{x}_t =  a\eta_t(\gamma^2-x_t) - b\eta_t^2 x_t$，相应的变分问题也是可解的，但$\eta_t^*$没有闭式解，所以就不特别讨论了。

总之，变分转化提供了一个灵活的分析框架，让我们能得到一些有意思的结果，同时也存在一些让人质疑的地方，大家可以继续沿着这个方向思考下去。

## 似曾相识

笔者首次看到这个分析过程时，隐约有种“似曾相识”的熟悉感，总觉得在哪里看到过类似的。后来终于想起来了，我们曾在[《基于Amos优化器思想推导出来的一些“炼丹策略”》](<https://spaces.ac.cn/archives/9344>)采用过类似切入点，不过当时只做了一些对齐分析，没有提出变分求法。这一节我们将两篇文章结合起来，重做一波分析。

简单起见，先考虑没有WN和WD的更新规则
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t\end{equation}
由于没有WN，$\Vert\boldsymbol{\theta}_t\Vert$就没有约束了，所以这次不能考虑内积了。还是设理想终点为$\boldsymbol{\theta}^*$，我们考虑它们的L2范数：记$\boldsymbol{\varepsilon}_t = \boldsymbol{\theta}_t - \boldsymbol{\theta}^*$，那么
\begin{equation}\Vert\boldsymbol{\varepsilon}_{t+1}\Vert^2 = \Vert\boldsymbol{\varepsilon}_t + \eta_t \boldsymbol{u}_t\Vert^2 = \Vert\boldsymbol{\varepsilon}_t\Vert^2 + 2 \eta_t\boldsymbol{u}_t\cdot \boldsymbol{\varepsilon}_t + \eta_t^2 \Vert\boldsymbol{u}_t\Vert^2\end{equation}
跟前面一样，$\Vert\boldsymbol{u}_t\Vert$假设为常数$\mu$，至于中间项，我们有
\begin{equation}\boldsymbol{u}_t\cdot \boldsymbol{\varepsilon}_t = \Vert\boldsymbol{u}_t\Vert\cdot\Vert\boldsymbol{\varepsilon}_t\Vert\cdot\cos(\boldsymbol{u}_t, \boldsymbol{\varepsilon}_t)\end{equation}
这里$\cos(\boldsymbol{u}_t, \boldsymbol{\varepsilon}_t)$也相当于学习速率，我们设为$-\alpha \Vert\boldsymbol{\varepsilon}_t\Vert$，其中$\alpha > 0$，这个选择有两重含义：负号是假设训练过程总是往目标方向走，正比于$\Vert\boldsymbol{\varepsilon}_t\Vert$是假设越接近目标学习速率越低。代入这个假设，并记$x_t = \Vert\boldsymbol{\varepsilon}_t\Vert^2$，我们得到
\begin{equation}x_{t+1} = x_t  - 2 \alpha \mu \eta_t x_t + \mu^2 \eta_t^2\end{equation}
同样用导数代替差分，得到微分方程$\dot{x}_t = \mu^2 \eta_t^2 - 2 \alpha \mu \eta_t x_t$。给定$x_0 > 0$，此时我们要最小化$x_T$，得到变分问题
\begin{equation}\min_{\eta\geq 0} x_T \qquad \text{s.t.}\qquad \dot{x}_t = \mu^2 \eta_t^2 - 2 \alpha \mu \eta_t x_t\label{eq:wd0-q}\end{equation}

## 相同解法

这个新的变分问题，解法跟“[初等过程](<https://spaces.ac.cn/archives/12010#%E5%88%9D%E7%AD%89%E8%BF%87%E7%A8%8B>)”一节基本一样。首先在微分方程两端乘以$x_t^{-2}$，整理得
\begin{equation}-\frac{d}{dt}x_t^{-1} = x_t^{-2}\dot{x}_t =  \mu^2 \eta_t^2 x_t^{-2} - 2 \alpha \mu \eta_t x_t^{-1} = (\mu \eta_t x_t^{-1} - \alpha)^2 - \alpha^2 \geq - \alpha^2\end{equation}
两端对$t$从$0\sim T$积分得
\begin{equation}x_0^{-1} - x_T^{-1} \geq - \alpha^2 T \qquad\Rightarrow\qquad x_T \geq \frac{1}{x_0^{-1} + \alpha^2 T}\label{eq:wd0-x-opt}\end{equation}
等号成立的条件是$\mu \eta_t x_t^{-1} = \alpha$，即$\eta_t = \mu^{-1}\alpha x_t$，此时有
\begin{equation}\eta_t^* = \frac{\mu^{-1}\alpha}{x_0^{-1} + \alpha^2 t} = \frac{\eta_0}{1 + \alpha^2 x_0 t},\qquad \eta_0 = \frac{\alpha x_0}{\mu}\end{equation}
即（带时间平移的）倒数衰减，实践中我们也可以将它改造成固定起点和终点以及训练步数的形式。

一个很自然的问题是，它的实践效果跟常用的Cosine Decay、Linear Decay相比如何？笔者觉得不一定有多大区别，因为LR Decay的重点是“Decay”，固定起点、终点和训练步数后，这些不同的Decay曲线，其实就提供了一个从1到0的不同形状而已，个人感觉可能很难会有本质区别。

## 权重衰减

同样的过程还可以推广到带WD的一般更新规则
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta_t \boldsymbol{u}_t - \lambda_t\eta_t\boldsymbol{\theta}_t \end{equation}
其中$\lambda_t$的经典选择是固定为某个常数$\lambda > 0$（另一个新兴选择是$\lambda_t = \lambda\eta_t$，即[AdamC](<https://papers.cool/arxiv/2506.02285>)，但计算结果比较复杂，就不展开了）。接下来我们要做个假设：WD只是辅助的，更新量$\eta_t \boldsymbol{u}_t$才是“正主”，WD不能喧宾夺主，所以假设$\mathcal{O}(\lambda_t\eta_t\boldsymbol{\theta}_t\cdot\boldsymbol{\varepsilon}_t\Vert)=\mathcal{O}(\eta_t^2\Vert\boldsymbol{u}_t\Vert^2)$，这样展开到二阶只多出一项
\begin{equation}\Vert\boldsymbol{\varepsilon}_{t+1}\Vert^2 = \Vert\boldsymbol{\varepsilon}_t + \eta_t \boldsymbol{u}_t - \lambda_t\eta_t\boldsymbol{\theta}_t\Vert^2 \approx \Vert\boldsymbol{\varepsilon}_t\Vert^2 + 2 \eta_t\boldsymbol{u}_t\cdot \boldsymbol{\varepsilon}_t + \eta_t^2 \Vert\boldsymbol{u}_t\Vert^2 - 2\lambda_t\eta_t\boldsymbol{\theta}_t\cdot\boldsymbol{\varepsilon}_t\end{equation}
$\boldsymbol{\theta}_t\cdot\boldsymbol{\varepsilon}_t$是新引入的项，我们有
\begin{equation}\boldsymbol{\theta}_t\cdot\boldsymbol{\varepsilon}_t = (\boldsymbol{\varepsilon}_t + \boldsymbol{\theta}^*)\cdot\boldsymbol{\varepsilon}_t = \Vert\boldsymbol{\varepsilon}_t\Vert^2 + \boldsymbol{\theta}^*\cdot\boldsymbol{\varepsilon}_t\end{equation}
假设在适当的设置下，$\boldsymbol{\theta}^*$和$\boldsymbol{\varepsilon}_t$是两个相对“随机”的量，那么$\boldsymbol{\theta}^*\cdot\boldsymbol{\varepsilon}_t\approx 0$，即$\boldsymbol{\theta}_t\cdot\boldsymbol{\varepsilon}_t\approx \Vert\boldsymbol{\varepsilon}_t\Vert^2$。其他项我们都已经有处理经验了，基于相同的假设和记号，得微分方程$\dot{x}_t = \mu^2 \eta_t^2 - 2 (\alpha \mu + \lambda_t) \eta_t x_t$，于是有变分问题
\begin{equation}\min_{\eta\geq 0} x_T \qquad \text{s.t.}\qquad \dot{x}_t = \mu^2 \eta_t^2 - 2 (\alpha \mu + \lambda_t) \eta_t x_t\end{equation}
当$\lambda_t = \lambda$时，问题跟$\eqref{eq:wd0-q}$是完全一样的，只不过将$\alpha$换成了$\alpha+\lambda\mu^{-1}$，所以最优解是
\begin{equation}\eta_t^* = \frac{\mu^{-1}(\alpha+\lambda\mu^{-1})}{x_0^{-1} + (\alpha+\lambda\mu^{-1})^2 t}\end{equation}
即依然是（带时间平移的）倒数衰减。能不能将$\lambda_t$像$\eta_t$那样用变分法优化呢？不能。现在多出来的一项是$-2\lambda_t\eta_t x_t$，它是负的，这意味着我们假设WD对收敛有帮助，这只有在精心选择的一个小范围内才能做如此假设，而不能随意设置，因此不能用变分法最优化。

## 无独有偶

事实上，当意识到这种提法本质上是一个变分问题后，笔者还想起来早年在知乎看到过的[《科学炼丹：用变分法，优化 lr decay 方案》](<https://zhuanlan.zhihu.com/p/497459075>)，它通过一些经验假设，同样将最优学习率抽象为一个变分问题，这个过程也有一些值得学习的地方。

设$\ell$是损失函数，它越小越好。在无限训练下，我们假设学习率越小越好，注意这跟Scaling Law的结论不矛盾，Scaling Law研究的是有限预算下的学习率选择，这里考虑的是以固定学习率近乎无限地训练下去的效果，记为$\ell_{\infty}(\eta)$，即以学习率$\eta$训练无限步的最终损失。进一步地，假设一定范围内，成立
\begin{equation}\ell_{\infty}(\eta) = \ell_{\infty}(0) + \alpha\eta,\qquad \alpha > 0\end{equation}
即学习率越大，极限损失越大（效果越差），假设它们是线性关系。接着，设当前损失为$\ell_t$，学习率为$\eta_t$，我们假设以学习率$\eta_t$训练时，学习速率正比于学习率$\eta_t$，以及正比$\ell_t$与其极限损失$\ell_{\infty}(\eta_t)$之差（符合越接近终点，学习速率越慢的经验），于是
\begin{equation}\ell_{t+1} = \ell_t - \beta \cdot\eta_t\cdot (\ell_t - \ell_{\infty}(\eta_t)) = \ell_t - \beta \cdot\eta_t\cdot (\ell_t - \ell_{\infty}(0) - \alpha\eta_t)\end{equation}
记$x_t = \ell_t - \ell_{\infty}(0)$，还是用导数替代差分，最终我们可以抽象出变分问题
\begin{equation}\min_{\eta\geq 0} x_T \qquad \text{s.t.}\qquad \dot{x}_t = \alpha \beta \eta_t^2 - 2 \beta \eta_t x_t\end{equation}
问题形式跟$\eqref{eq:wd0-q}$同出一辙，最优解自然也是倒数衰减。在这条路线中，我们直接对损失函数做一些经验假设来得到相同的变分问题，跳过了对优化器的假设，显得更加直观。

但另一边，相同形式也意味着最优$x_T$跟$\eqref{eq:wd0-x-opt}$一样具有反比于$T$的形式，即损失以$\mathcal{O}(T^{-1})$的速率下降，这似乎过于理想了——[之前](<https://spaces.ac.cn/archives/11494#%E5%8A%A0%E9%80%9F%E6%94%B6%E6%95%9B>)我们提过，在没有更强的假设下，损失至多以$\mathcal{O}(T^{-1/2})$的速率下降。如果设法修改这些假设，使所得结论更吻合这些经典结果，也许在实践中更加有效，这也是一个值得思考的问题。

## 文章小结

这篇文章我们讨论了将最佳学习率调度抽象为一个变分问题，继而求最优解的做法。
