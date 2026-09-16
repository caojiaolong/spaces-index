# 让炼丹更科学一些（三）：SGD的终点损失收敛

> 作者：苏剑林 · 科学空间 · 2025-12-16
>
> 原文：<https://spaces.ac.cn/archives/11480>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

目前我们已经有两篇文章讨论SGD的收敛性质，不过它们都只是损失值的收敛结果，所以它们只保证我们能找到最优的损失值，但不能保证找到最优值的所在位置$\boldsymbol{\theta}^*$，这是目前的结论跟实践之间的一个显著gap。直觉上，训练结束时的权重$\boldsymbol{\theta}_T$应该更接近理论最优的$\boldsymbol{\theta}^*$，我们也想知道理论上是否支撑这一点。

所以，这篇文章我们就将平均损失的收敛结果转化为终点损失的收敛结果，初步从理论上了解$\boldsymbol{\theta}_T$与$\boldsymbol{\theta}^*$差多远。

## 找出位置

我们从文章[《让炼丹更科学一些（二）：将结论推广到无界域》](<https://spaces.ac.cn/archives/11469>)出发，它的核心结果是不等式
\begin{equation}\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\varphi})]\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2}{2} + \frac{G^2}{2}\sum_{t=1}^T \eta_t^2\label{leq:avg-2-mid3}\end{equation}
然后假设$\eta_t$的单调递减性，用$\eta_T$替换左端的$\eta_t$，代入$\boldsymbol{\varphi}=\boldsymbol{\theta}^*$就可以得到上篇的结论之一：
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T \frac{\eta_t^2}{\eta_T}\label{leq:avg-2}\end{equation}
开头说了，这只是损失值的收敛结果，我们更想找到收敛的位置。为此，一个比较简单的思路是利用$L$的凸性和Jensen不等式，得到
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t)] = \mathbb{E}\left[\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{\theta}_t)\right] \geq \mathbb{E}\left[L\left(\frac{1}{T}\sum_{t=1}^T \boldsymbol{\theta}_t\right)\right]\end{equation}
定义$\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=1}^T \boldsymbol{\theta}_t$，那么我们就有
\begin{equation}\mathbb{E}[L(\bar{\boldsymbol{\theta}}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T \frac{\eta_t^2}{\eta_T}\end{equation}
即训练轨迹$\boldsymbol{\theta}_1,\boldsymbol{\theta}_2,\cdots,\boldsymbol{\theta}_T$的质心$\bar{\boldsymbol{\theta}}_T$对应的损失值$L(\bar{\boldsymbol{\theta}}_T)$平均而言是收敛于$L(\boldsymbol{\theta}^*)$，即意味着$\bar{\boldsymbol{\theta}}_T$平均而言是收敛于$\boldsymbol{\theta}^*$的（因为严格凸函数的最小值点唯一）。这一定程度上解释了对训练轨迹进行滑动平均来获得更好权重的操作，这也包括[WSM（Warmup\-Stable and Merge）](<https://papers.cool/arxiv/2507.17634>)的Merge操作的合理性等。

## 准备工作

计算$\bar{\boldsymbol{\theta}}_T$提供了一种寻找$\boldsymbol{\theta}^*$的方式，但它没有完全回答本文开头的问题——我们更想知道$\boldsymbol{\theta}_T$收敛于$\boldsymbol{\theta}^*$的结论。接下来我们沿着[《Last Iterate of SGD Converges (Even in Unbounded Domains)》](<https://parameterfree.com/2020/08/07/last-iterate-of-sgd-converges-even-in-unbounded-domains/>)的思路，将平均损失收敛转化成终点损失收敛。

正式证明之前，需要一些准备工作，其中之一是推广式$\eqref{leq:avg-2-mid3}$。从它的证明过程我们知道，求和的下限理论上可以是任意的，即可以将起点$\boldsymbol{\theta}_1$换成任意的$\boldsymbol{\theta}_{T-k}$，不等式依然成立，不过此时$\boldsymbol{\theta}_{T-k}$可能也跟$\boldsymbol{x}_t$相关了，所以右端也要补上$\mathbb{E}$，得到
\begin{equation}\sum_{t=T-k}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\varphi})]\leq \frac{\mathbb{E}[\Vert\boldsymbol{\theta}_{T-k} - \boldsymbol{\varphi}\Vert^2]}{2} + \frac{G^2}{2}\sum_{t=T-k}^T \eta_t^2\label{leq:last-mid1}\end{equation}
这里$\boldsymbol{\varphi}$是任意数据无关的向量，但这个“数据无关”是相对的，回顾证明过程可以得出，当我们的起点选为$T-k$时，它可以至多跟$\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_{T-k-1}$相关。特别地，$\boldsymbol{\theta}_{T-k}$满足这个条件，代入$\boldsymbol{\varphi}=\boldsymbol{\theta}_{T-k}$得到
\begin{equation}\sum_{t=T-k}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}_{T-k})]\leq \frac{G^2}{2}\sum_{t=T-k}^T \eta_t^2\label{leq:last-mid2}\end{equation}
这是后面要用的重要中间结论。

注意这里有个小陷阱：当我们代入$\boldsymbol{\varphi}=\boldsymbol{\theta}_{T-k}$时，并无法保证$\mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}_{T-k})]$的非负性，只有$\mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]$才能保证非负，所以像前面一样“假设$\eta_t$的单调递减性，用$\eta_T$替换左端的$\eta_t$”的做法对上式不再成立。笔者一开始就是想当然地犯了这个错误，所以这里专门提一下。

## 关键等式

为了将平均损失的结论转化成终点形式，我们还需要准备一个非常关键的恒等式：
\begin{equation}q_T = \frac{1}{T}\sum_{t=1}^T q_t + \sum_{k=1}^{T-1} \frac{1}{k(k+1)}\sum_{t=T-k}^T (q_t - q_{T-k})\label{eq:qt}\end{equation}

这个恒等式巧妙地建立了终点值和平均值之间的联系，笔者曾花了几天时间想要得到一个直观理解，但没有成功，所以只能按部就班地介绍它的证明了。证明的思路是考虑$q_t$从尾到头的累积平均，即定义$S_k = \frac{1}{k}\sum_{t=T-k+1}^T q_t$，那么可以写出
\begin{equation}\begin{aligned}
k S_k =&\, (k + 1) S_{k+1} - q_{T-k} \\[5pt]
=&\, k S_{k+1} + (S_{k+1} - q_{T-k}) \\
=&\, k S_{k+1} + \frac{1}{k+1}\sum_{t=T-k}^T (q_t - q_{T-k})
\end{aligned}\end{equation}
两边除以$k$，然后对$k=1\sim T-1$求和得
\begin{equation}S_1 = S_T + \sum_{k=1}^{T-1}\frac{1}{k(k+1)}\sum_{t=T-k}^T (q_t - q_{T-k})\end{equation}
再代入$S_1,S_T$的原始定义，即得式$\eqref{eq:qt}$。整个推导的核心是通过“累积平均”这一运算，作为终点值$q_T$到平均值$\frac{1}{T}\sum_{t=1}^T q_t$的自然过渡。最后，我们将$q_t$换成$\eta_t q_t$，并假设$\eta_t$的非增性，以及$q_t$的非负性，可得
\begin{equation}\eta_T q_T = \frac{1}{T}\sum_{t=1}^T \eta_t q_t + \sum_{k=1}^{T-1} \frac{1}{k(k+1)}\sum_{t=T-k}^T\underbrace{(\eta_t q_t - \eta_{T-k} q_{T-k})}_{\leq \eta_t(q_t - q_{T-k})}\label{leq:etat-qt}\end{equation}
这便是原博客出现的不等式，也是下一节直接用到的核心不等式。

## 完成证明

现在我们就可以一鼓作气完成证明了。代入$q_t = \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]$，代入到不等式$\eqref{leq:etat-qt}$得：
\begin{equation}\eta_T\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq \underbrace{\frac{1}{T}\sum_{t=1}^T \eta_t\mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]}_{\eqref{leq:avg-2-mid3}} + \sum_{k=1}^{T-1} \frac{1}{k(k+1)}\underbrace{\sum_{t=T-k}^T \eta_t\big(\mathbb{E}[L(\boldsymbol{\theta}_t) -L(\boldsymbol{\theta}_{T-k})]\big)}_{\eqref{leq:last-mid2}}\end{equation}
分别代入不等式$\eqref{leq:avg-2-mid3}$（取$\boldsymbol{\varphi}=\boldsymbol{\theta}^*$）和$\eqref{leq:last-mid2}$，整理得
\begin{equation}\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T \frac{\eta_t^2}{\eta_T} + \frac{G^2}{2}\sum_{k=1}^{T-1} \frac{1}{k(k+1)}\sum_{t=T-k}^T \frac{\eta_t^2}{\eta_T}\label{leq:last-mid3}\end{equation}
对于最后一项，我们有
\begin{equation}\sum_{k=1}^{T-1}\frac{1}{k(k+1)}\sum_{t=T-k}^{T}\frac{\eta_t^2}{\eta_T}
=\sum_{t=1}^{T}\frac{\eta_t^2}{\eta_T}\sum_{k=\max(1,\,T-t)}^{T-1} \frac{1}{k(k+1)}
=\sum_{t=1}^{T}\frac{\eta_t^2}{\eta_T}\left(\frac{1}{\max(1,\,T-t)}-\frac{1}{T}\right)\label{eq:last-mid4}\end{equation}
代入到式$\eqref{leq:last-mid3}$得
\begin{equation}\begin{aligned}
\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2}{2}\sum_{t=1}^{T}\frac{\eta_t^2/\eta_T}{\max(1,\,T-t)} \\
=&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2\eta_T}{2} + \frac{G^2}{2}\sum_{t=1}^{T-1}\frac{\eta_t^2/\eta_T}{T-t}
\end{aligned}\label{leq:last-1}\end{equation}
这就得到了我们的最终结论。由于我们通过恒等变换$\eqref{eq:last-mid4}$交换了求和顺序，提前进行了化简，所以该结果相比原博客[《Last Iterate of SGD Converges (Even in Unbounded Domains)》](<https://parameterfree.com/2020/08/07/last-iterate-of-sgd-converges-even-in-unbounded-domains/>)会更简洁和通用一些。

## 两个例子

不难看出，结论$\eqref{leq:last-1}$右端的形状跟$\eqref{leq:avg-2}$大同小异，这暗示着终点损失跟平均损失应该有着相近的收敛速度。我们依旧从静态学习率和动态学习率两个例子来观察最终结论的表现。首先是静态学习率$\eta_t = \eta$，此时
\begin{equation}\begin{aligned}
\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta} + \frac{G^2\eta}{2} + \frac{G^2\eta}{2}\sum_{t=1}^{T-1}\frac{1}{T-t} \\
\leq&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta} + \frac{G^2\eta}{2} (2 + \ln T)
\end{aligned}\end{equation}
取$\eta = \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert/G}{\sqrt{T(2+\ln T)}}$可让最右端取到最小值，其收敛速度为$\mathcal{O}(\sqrt{\ln T/T})$，这比平均损失的收敛速度稍慢，上一篇我们证明了，在常数学习率下，平均损失的收敛速度可以做到$\mathcal{O}(1/\sqrt{T})$。当然，这都是极限情形下的区别，实践中$\sqrt{\ln T}$的差异可能毫无区别。

然后考虑动态学习率$\eta_t = \frac{\alpha}{\sqrt{t}}$，代入式$\eqref{leq:last-1}$得
\begin{equation}\begin{aligned}
\mathbb{E}[L(\boldsymbol{\theta}_T) - L(\boldsymbol{\theta}^*)] \leq&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\alpha\sqrt{T}} + \frac{G^2\alpha}{2\sqrt{T}} + \frac{G^2\alpha\sqrt{T}}{2}\sum_{t=1}^{T-1}\frac{1}{t(T-t)} \\
=&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\alpha\sqrt{T}} + \frac{G^2\alpha}{2\sqrt{T}} + \frac{G^2\alpha}{2\sqrt{T}}\sum_{t=1}^{T-1}\left(\frac{1}{t} + \frac{1}{T-t}\right) \\
=&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\alpha\sqrt{T}} + \frac{G^2\alpha}{2\sqrt{T}} + \frac{G^2\alpha}{\sqrt{T}}\sum_{t=1}^{T-1}\frac{1}{t} \\
\leq&\, \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\alpha\sqrt{T}} + \frac{G^2\alpha}{2\sqrt{T}} + \frac{G^2\alpha}{\sqrt{T}}(1 + \ln T) \\
\sim&\, \mathcal{O}\left(\frac{\ln T}{\sqrt{T}}\right)
\end{aligned}\end{equation}
跟上一篇文章中无界域的平均损失收敛一样，在动态学习率$\eta_t = \frac{\alpha}{\sqrt{t}}$下收敛速度都是$\mathcal{O}(\ln T / \sqrt{T})$，只不过这里的常数会大一些。

## 文章小结

在这篇文章中，我们将SGD的收敛结论从平均损失推广到了终点损失，即考虑训练结束时损失值与理论最优值的接近程度，这种设置更贴合我们的训练实践。
