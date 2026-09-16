# 让炼丹更科学一些（一）：SGD的平均损失收敛

> 作者：苏剑林 · 科学空间 · 2023-12-19
>
> 原文：<https://spaces.ac.cn/archives/9902>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

很多时候我们将深度学习模型的训练过程戏称为“炼丹”，因为整个过程跟古代的炼丹术一样，看上去有一定的科学依据，但整体却给人一种“玄之又玄”的感觉。尽管本站之前也关注过一些[优化器](<https://spaces.ac.cn/tag/%E4%BC%98%E5%8C%96%E5%99%A8/>)相关的工作，甚至也写过[《从动力学角度看优化算法》](<https://spaces.ac.cn/search/%E4%BB%8E%E5%8A%A8%E5%8A%9B%E5%AD%A6%E8%A7%92%E5%BA%A6%E7%9C%8B%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/>)系列，但都是比较表面的介绍，并没有涉及到更深入的理论。为了让以后的炼丹更科学一些，笔者决定去补习一些优化相关的理论结果，争取让炼丹之路多点理论支撑。

在本文中，我们将学习随机梯度下降（SGD）的一个非常基础的收敛结论。虽然现在看来，该结论显得很粗糙且不实用，但它是优化器收敛性证明的一次非常重要的尝试，特别是它考虑了我们实际使用的是<u>随机</u>梯度下降（SGD）而不是<u>全量</u>梯度下降（GD）这一特性，使得结论更加具有参考意义。

## 问题设置

设损失函数是$L(\boldsymbol{x},\boldsymbol{\theta})$，其实$\boldsymbol{x}$是训练集，而$\boldsymbol{\theta}\in\mathbb{R}^N$是训练参数。受限于算力，我们通常只能执行随机梯度下降（SGD），即每步只能采样一个训练子集来计算损失函数并更新参数，假设采样是独立同分布的，第$t$步采样到的子集为$\boldsymbol{x}_t$，那么我们可以合理地认为实际优化的最终目标是
\begin{equation}L(\boldsymbol{\theta}) = \lim_{T\to\infty}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta})\label{eq:loss}\end{equation}
实际情况下，我们也只能训练有限步，所以我们假设$T$是一个足够大的正整常数。我们的目标是寻找$L(\boldsymbol{\theta})$的最小值点，即希望找到$\boldsymbol{\theta}^*$：
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}\in\mathbb{R}^N} L(\boldsymbol{\theta})\label{eq:argmin}\end{equation}
现在，我们考虑如下SGD迭代：
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\label{eq:sgd}\end{equation}
其中$\eta_t > 0$是学习率，其中$\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta})\triangleq \nabla_{\boldsymbol{\theta}}L(\boldsymbol{x}_t,\boldsymbol{\theta})$是$L(\boldsymbol{x}_t,\boldsymbol{\theta})$关于$\boldsymbol{\theta}$的梯度。我们的任务就是分析如此迭代下去，$\boldsymbol{\theta}_t$是否能够收敛到目标点$\boldsymbol{\theta}^*$。

## 结论初探

首先，我们给出最终要证明的不等式：在适当的假设之下，有
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{R^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T\eta_t\label{neq:core}\end{equation}

其中$R,G$是跟优化过程无关的常数。后面我们会逐一介绍“适当的假设”具体是什么，在此之前我们先来观察一下不等式$\eqref{neq:core}$所表达的具体含义：

> 1、左端第一项，是优化过程中每一步的损失函数的平均结果；
> 
> 
> 
> 2、左端第二项，根据式$\eqref{eq:loss}$，当$T$足够大时可以认为它就是$L(\boldsymbol{\theta}^*)$；
> 
> 
> 
> 3、左端合并起来，就是优化过程中损失函数的平均与损失函数的理论最小值之差；
> 
> 
> 
> 4、右端是一个只与学习率策略$\{\eta_t\}$有关的式子。

综合1、2、3、4点，不等式$\eqref{neq:core}$就是说：在适当的假设之下，SGD的平均损失与我们要寻找的理想目标的差距，可以被一个只与学习率策略有关的式子控制，如果我们可以选择适当的学习率让该式趋于零，那么意味着SGD的平均损失一定能收敛到理论最优点。（当然，从理论上来说，该结论只能保证找到损失函数的最小值$L(\boldsymbol{\theta}^*)$，但无法保证找到具体的最小值点$\boldsymbol{\theta}^*$。）

说白了，这就是关于SGD在什么情况下会收敛的一个理论结果。对了，将不等式$\eqref{neq:core}$左端乘以$T$后，将得到一个有特别名字的概念，叫做“遗憾”（Regret，有些教程也直接翻译为“悔”）。

## 两个例子

例如，假设学习率是常数$\eta$，那么我们有不等式$\eqref{neq:core}$右端有
\begin{equation}\frac{R^2}{2T\eta} + \frac{G^2}{2T}\sum_{t=1}^T\eta = \frac{R^2}{2T\eta} + \frac{G^2\eta}{2}\geq \frac{RG}{\sqrt{T}}\end{equation}
等号成立时$\eta=\frac{R}{G\sqrt{T}}$，也就是说学习率取常数$\frac{R}{G\sqrt{T}}$，那么就有
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{RG}{\sqrt{T}}\label{neq:case-1}\end{equation}
当$T\to\infty$时，右端趋于零，这意味着当训练步数$T$足够大时，将学习率设为常数$\frac{R}{G\sqrt{T}}$，就可以让SGD迭代的平均与理论最优点的差距任意小。

另一个例子是考虑衰减策略$\eta_t = \frac{\alpha}{\sqrt{t}}$，利用
\begin{equation}\sum_{t=1}^T \frac{1}{\sqrt{t}} = 1+\sum_{t=2}^T \frac{1}{\sqrt{t}}\leq 1+\sum_{t=2}^T \frac{2}{\sqrt{t-1} + \sqrt{t}}=1+\sum_{t=2}^T 2(\sqrt{t}-\sqrt{t-1})=2\sqrt{T}-1 < 2\sqrt{T}\end{equation}
代入式$\eqref{neq:core}$得到
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*) < \frac{R^2}{2\alpha\sqrt{T}} + \frac{G^2\alpha}{\sqrt{T}}\label{neq:case-2}\end{equation}
式$\eqref{neq:case-2}$和式$\eqref{neq:case-1}$关于$T$都是$\mathcal{O}\left(\frac{1}{\sqrt{T}}\right)$的，因此理论上它们都能收敛。跟式$\eqref{neq:case-1}$相比，式$\eqref{neq:case-2}$的常数更大，这意味着$\eta_t\equiv\frac{R}{G\sqrt{T}}$的收敛速度很可能比$\eta_t = \frac{\alpha}{\sqrt{t}}$快。然而，在实际中我们更愿意用后者，因为前者的需要提前确定训练总步数$T$，训练完就结束了并且精度也固定了，后者并没有这些限制，甚至$\alpha$也不需要调，直接$\eta_t = \frac{1}{\sqrt{t}}$就可以持续训练下去，并且理论上平均损失与理论最小值的差距会越来越小。

可即便如此，$\eta_t = \frac{1}{\sqrt{t}}$这种学习率策略，不管在量级或者变化规律上都依然与我们平时训练所用的相距甚远，因此不难猜测里边必然加了不少很强的假设。事不宜迟，我们马上来展开证明过程，逐一展开其中的假设。

## 证明过程

证明的开始，我们假设对于任意$\boldsymbol{x}$，$L(\boldsymbol{x},\boldsymbol{\theta})$都是关于$\boldsymbol{\theta}$的凸函数。这是一个非常强且通常非常不符合训练事实的假设，但没办法，理论分析通常都只能做一些很强的假设，然后将这些假设之下的结论启发性地用到实际场景。

凸函数有很多不同的定义方式，这里直接采用如下定义：
\begin{equation}L(\boldsymbol{x}_t,\boldsymbol{\theta}_2) - L(\boldsymbol{x}_t,\boldsymbol{\theta}_1) \geq (\boldsymbol{\theta}_2-\boldsymbol{\theta}_1)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_1),\quad \forall \boldsymbol{\theta}_1,\boldsymbol{\theta}_2\label{eq:convex}\end{equation}
其中$\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_1)\triangleq \nabla_{\boldsymbol{\theta}}L(\boldsymbol{x}_t,\boldsymbol{\theta})$是$L(\boldsymbol{x}_t,\boldsymbol{\theta})$关于$\boldsymbol{\theta}$的梯度，$\cdot$是向量内积，上述定义的几何意义就是凸函数的图像总在其切线（面）的上方。

证明的要点，是考虑$\boldsymbol{\theta}_{t+1}$与$\boldsymbol{\theta}^*$的距离：
\begin{equation}\begin{aligned}
\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2=&\, \Vert\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)- \boldsymbol{\theta}^*\Vert^2 \\
=&\, \Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2 - 2\eta_t (\boldsymbol{\theta}_t- \boldsymbol{\theta}^*)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) + \eta_t^2\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2
\end{aligned}\end{equation}
将它改写成
\begin{equation}(\boldsymbol{\theta}_t- \boldsymbol{\theta}^*)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) = \frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} + \frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\end{equation}
根据式$\eqref{eq:convex}$，我们有$L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq (\boldsymbol{\theta}_t- \boldsymbol{\theta}^*)\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)$，代入上式有
\begin{equation}L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} + \frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\end{equation}
两端对$t=1,2,\cdots,T$求和：
\begin{equation}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \sum_{t=1}^TL(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \sum_{t=1}^T\left(\frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t}\right) + \sum_{t=1}^T\frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\label{neq:base}\end{equation}
再次引入一个新的假设——$\eta_t$是关于$t$的单调递减函数（即$\eta_t\geq \eta_{t+1}$），并且记$R = \max_t \Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert$，那么就有
\begin{equation}\begin{aligned}
&\,\sum_{t=1}^T\left(\frac{\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2}{2\eta_t} - \frac{\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_t}\right)\\
=&\,\frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\eta_1} - \frac{\Vert\boldsymbol{\theta}_{T+1} - \boldsymbol{\theta}^*\Vert^2}{2\eta_T} + \sum_{t=2}^T\left(\frac{1}{2\eta_t} - \frac{1}{2\eta_{t-1}}\right)\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert^2\\
\leq&\,\frac{R^2}{2\eta_1} + \sum_{t=2}^T\left(\frac{1}{2\eta_t} - \frac{1}{2\eta_{t-1}}\right)R^2\\
=&\, \frac{R^2}{2\eta_T}
\end{aligned}\end{equation}
最后我们记$G = \max_t \Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert$，然后将上式代入式$\eqref{neq:base}$得到
\begin{equation}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{R^2}{2\eta_T} + \sum_{t=1}^T\frac{1}{2}\eta_t\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2\leq \frac{R^2}{2\eta_T} + \frac{G^2}{2}\sum_{t=1}^T\eta_t\end{equation}
最后两端除以$T$即得不等式$\eqref{neq:core}$。

注意现在的常数$R,G$是优化相关的，即先要确定学习率策略$\{\eta_t\}$，然后完成优化过程才能得到$R,G$。要想$R,G$成为优化无关的常数，我们需要假设对于任意的$\{\eta_t\}$，$\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert$和$\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert$都分别不超过某个常数$R,G$，这样才能使得不等式$\eqref{neq:core}$右端只与学习率策略有关。

## 域内投影

然而，最后的假设“对于任意的$\{\eta_t\}$，$\Vert\boldsymbol{\theta}_t - \boldsymbol{\theta}^*\Vert$和$\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert$都分别不超过某个常数$R,G$”总有种“本末倒置”的感觉：看上去需要先完成优化才能确定$R,G$，而我们的目的却是要通过所证明的结果来改进优化。为了去掉这种奇怪的感觉，我们干脆将这两个假设改为：

> 1、式$\eqref{eq:argmin}$中的$\boldsymbol{\theta}\in\mathbb{R}^N$改为$\boldsymbol{\theta}\in\boldsymbol{\Theta}$，其中$\boldsymbol{\Theta}\subseteq \mathbb{R}^N$是一个有界凸集；
> 
> 
> 
> 1\.1）<strong>有界</strong>：$R=\max\limits_{\boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}}\Vert\boldsymbol{\theta}_1-\boldsymbol{\theta}_2\Vert < \infty$；
> 
> 
> 
> 1\.2）<strong>凸集</strong>：$\forall \boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}$以及$\forall\lambda\in[0,1]$，都有$\lambda \boldsymbol{\theta}_1 + (1-\lambda)\boldsymbol{\theta}_2 \in \boldsymbol{\Theta}$。
> 
> 
> 
> 2、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，都有$\Vert\boldsymbol{g}(\boldsymbol{x},\boldsymbol{\theta})\Vert\leq G$

第2点可能更容易接受，无非是给损失函数$L(\boldsymbol{x},\boldsymbol{\theta})$再加了个假设而已，就好比“债多不愁”，凸函数这么强的假设都加了，再多加点也无妨。但第1点假设似乎不那么容易理解：凸集是因为凸函数本身只能定义在凸集上，这个也能接受，但有界如何保证呢？即如何保证迭代$\eqref{eq:sgd}$的输出一定有界？

答案是“多加一步投影”。我们定义投影运算：
\begin{equation}\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}) = \mathop{\text{argmin}}_{\boldsymbol{\theta}\in\boldsymbol{\Theta}}\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert\end{equation}
即在$\boldsymbol{\Theta}$中找到与$\boldsymbol{\varphi}$最相近的向量，于是我们可以将式$\eqref{eq:sgd}$改为
\begin{equation}\boldsymbol{\theta}_{t+1} = \Pi_{\boldsymbol{\Theta}}\big(\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\big)\in \boldsymbol{\Theta}\label{eq:sgd-p}\end{equation}
这就能保证迭代结果一定在集合$\boldsymbol{\Theta}$中。

然而，这样修改之后，上一节的证明和前面的结论（主要是不等式$\eqref{neq:core}$）还成立吗？很幸运，还成立，我们只需要证明对于式$\eqref{eq:sgd-p}$所定义的投影SGD有
\begin{equation}\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert \leq \Vert\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \boldsymbol{\theta}^*\Vert\end{equation}
那么“[证明过程](<https://spaces.ac.cn/archives/9902#%E8%AF%81%E6%98%8E%E8%BF%87%E7%A8%8B>)”一节的推导依然可以进行下去，只不过部分等号变成$\leq$而已。为此，我们只需要证明$\forall \boldsymbol{\varphi}\in\mathbb{R}^N, \boldsymbol{\theta}\in \boldsymbol{\Theta}$，都有
\begin{equation}\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}) - \boldsymbol{\theta}\Vert \leq \Vert \boldsymbol{\varphi} - \boldsymbol{\theta}\Vert\end{equation}

> <strong>证明</strong>：证明的关键是将凸集和$\Pi_{\boldsymbol{\Theta}}$的定义结合起来。首先，根据凸集的定义，我们知道$\forall \lambda\in(0,1)$都有$\lambda\boldsymbol{\theta} + (1-\lambda)\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\in \boldsymbol{\Theta}$，于是根据$\Pi_{\boldsymbol{\Theta}}$的定义，恒成立
> \begin{equation}\Vert\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert\leq \Vert\boldsymbol{\varphi} - \lambda\boldsymbol{\theta} - (1-\lambda)\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert = \Vert(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})) + \lambda(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\Vert\end{equation}
> 两端平方然后相减，得到
> \begin{equation}\lambda^2\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + 2\lambda(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\cdot(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}))\geq 0\end{equation}
> 注意我们刚限制了$\lambda\in(0,1)$，所以两端可以除以$\lambda$：
> \begin{equation}\lambda\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + 2(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\cdot(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}))\geq 0\end{equation}
> 这是个恒成立的式子，那么对于$\lambda\to 0^+$依然是恒成立的，于是
> \begin{equation}(\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta})\cdot(\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}))\geq 0\end{equation}
> 两端乘2后再同时加上$\Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + \Vert\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert^2$，左端正好是$\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert^2$，所以有
> \begin{equation}\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert^2\geq \Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2 + \Vert\boldsymbol{\varphi} - \Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})\Vert^2 \geq \Vert\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi})-\boldsymbol{\theta}\Vert^2\end{equation}

## 假设分析

至此，我们已经完成了证明过程。以上结果出自[《Online Convex Programming and Generalized Infinitesimal Gradient Ascent》](<https://www.cs.cmu.edu/~maz/publications/techconvex.pdf>)，是2003年的论文。特别地，优化相关的文献非常多，笔者作为一个初学者，在后面的文献溯源方面大概率会有错漏，请了解的读者不吝指正。

现在我们可以来“盘点”一下完整的证明过程用到的所有假设：

> 1、$\boldsymbol{\Theta}$是一个有界凸集，$R=\max\limits_{\boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}}\Vert\boldsymbol{\theta}_1-\boldsymbol{\theta}_2\Vert < \infty$；
> 
> 
> 
> 2、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，$L(\boldsymbol{x},\boldsymbol{\theta})$都是关于$\boldsymbol{\theta}$的凸函数；
> 
> 
> 
> 3、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，都有$\Vert\nabla_{\boldsymbol{\theta}}L(\boldsymbol{x},\boldsymbol{\theta})\Vert\leq G < \infty$；
> 
> 
> 
> 4、学习率$\eta_t$是关于$t$的单调递减函数（即$\eta_t\geq \eta_{t+1}$）；

在这些假设之下，投影SGD即式$\eqref{eq:sgd-p}$成立不等式$\eqref{neq:core}$。

其中，第1、4点假设都无可厚非，甚至可以说非常合理，比如第1点，对于实际计算来说，一个充分大的球体跟$\mathbb{R}^N$并没有实质区别了，而第4点递减的学习率更是符合已有认知；最强且最不符合事实的是第2点凸函数假设，但没办法，多看几篇优化相关的文献就释然了，因为几乎所有优化理论都是基于凸函数假设进行的，我们只能寄望于优化进入一定区域后损失函数能部分符合凸函数的性质；第3点本质上也是很强的假设，但实际运算中如果初始化做得好，并且学习率也设置得适当，基本上能将梯度模长控制在一定范围内，因此也通常能接受。

## 文章小结

在这篇文章中，我们重温了一篇凸优化的旧论文，介绍了SGD的一个非常基础的收敛性证明：在适当（实际上非常强）的假设下，SGD的收敛性可以得到保证。尽管这些假设在实际应用中可能并不总是成立，例如凸函数假设和梯度模长的限制，但这些理论结果仍能为我们提供关于SGD收敛性的重要见解。
