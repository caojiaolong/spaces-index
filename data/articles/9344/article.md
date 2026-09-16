# 基于Amos优化器思想推导出来的一些“炼丹策略”

> 作者：苏剑林 · 科学空间 · 2022-11-22
>
> 原文：<https://spaces.ac.cn/archives/9344>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

如果将训练模型比喻为“炼丹”，那么“炼丹炉”显然就是优化器了。据传AdamW优化器是当前训练神经网络最快的方案，这一点笔者也没有一一对比过，具体情况如何不得而知，不过目前做预训练时多数都用AdamW或其变种LAMB倒是真的。然而，正如有了炼丹炉也未必能炼出好丹，即便我们确定了选择AdamW优化器，依然有很多问题还没有确定的答案，比如：

> 1、学习率如何适应不同初始化和参数化？
> 
> 
> 
> 2、权重衰减率该怎么调？
> 
> 
> 
> 3、学习率应该用什么变化策略？
> 
> 
> 
> 4、能不能降低优化器的显存占用？

尽管在实际应用时，我们大多数情况下都可以直接套用前人已经调好的参数和策略，但缺乏比较系统的调参指引，始终会让我们在“炼丹”之时感觉没有底气。在这篇文章中，我们基于Google最近提出的Amos优化器的思路，给出一些参考结果。

## 基础回顾

Amos优化器出自Google最近的论文[《Amos: An Adam\-style Optimizer with Adaptive Weight Decay towards Model\-Oriented Scale》](<https://papers.cool/arxiv/2210.11693>)，它对上述几个问题都推导了比较完整的推导，并通过实验证实了它的有效性。然而，原论文的推导实在是不好读，各种记号和估计都过于随意，给人很“凌乱”感觉。不过好在Amos的思想还不算复杂，我们可以借用一下。

在开始推导之前，我们不妨先回顾一下对于上述几个问题，现有的解决方案是怎样的。

首先，第一个问题，大家可能不大理解“初始化”和“参数化”分别是什么含义，其实这就是模型权重的两种设置方式，常见的就是一个$n\times n$的矩阵，一般用“均值为0、方差为$1/n$”的方式初始化，详细介绍可以参考笔者之前[《从几何视角来理解模型参数的初始化策略》](<https://spaces.ac.cn/archives/7180>)、[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620>)。从“方差为$1/n$”我们就可以看到，不同参数有着不同的尺度（或者说数量级），如果我们用同一个学习率更新所有参数，那么就会导致每个参数的更新幅度不一样。这个问题笔者觉得比较优雅的解决方案就是LAMB优化器，它每次更新的模长直接取决于参数本身的模长，学习率只是用来描述相对更新量的大小。

至于权重衰减率问题，至少在预训练领域，笔者观察到的是都是沿用最早的选择0\.01，没有发现去调整该参数的工作。而对于学习率变化策略，大家都知道应该要将学习率慢慢降到零，但具体应该选用什么什么下降策略，暂时也没有太多的理论指导，多数结果也只是实验总结出来的。最后，关于节省显存问题，比较经典的工作就是AdaFactor优化器，笔者之前在[《AdaFactor优化器浅析（附开源实现）》](<https://spaces.ac.cn/archives/7302>)也有过介绍。降低优化器显存占用的主要就两个思路，一是去掉动量，二是对二阶矩做低秩分解，Amos本质上也是沿用了这两个思路。

## 问题设置

本文主要关心开头的前三个问题，希望能够推导出一些“即插即用”的结果。首先，我们将优化器的更新规则简写成：
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha_t \boldsymbol{u}_t\end{equation}
其实$\boldsymbol{\theta}_t, \boldsymbol{\theta}_{t+1}$分别代表$t,t+1$时刻的参数值，$\boldsymbol{u}_t$代表$t$时刻的更新向量（依赖于任务和数据），而标量$\alpha_t > 0$（向量的每个元素都大于0）代表$t$时刻的学习率。

自AdamW起，主流优化器都倾向于把权重衰减（Weight Decay）项从$\boldsymbol{u}_t$中独立出来，即
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - (\alpha_t \boldsymbol{u}_t + \rho_t\boldsymbol{\theta}_t)\end{equation}
其中$\rho_t > 0$是权重衰减率。本文的主要任务，就是希望能解决$\alpha_t$和$\rho_t$该怎么设置的问题。

## 权重衰减

我们知道，权重衰减也好，L2正则也好，它本身是跟训练目标无关的，它只是一个辅助项，目的是提高模型的泛化能力。既然是辅助，那么一个基本的要求就是它不应该“喧宾夺主”，为此，我们不妨加入一个限制：
\begin{equation}\mathcal{O}(\alpha_t^2) = \mathcal{O}(\rho_t)\end{equation}
也就是说，在整个更新过程中，权重衰减带来的更新量始终要比目标相关的更新量高一阶，由于$\alpha_t,\rho_t$基本上都是小于1的，所以更高阶意味着更小。

设优化的参数终点是$\boldsymbol{\theta}^*$，我们记$\boldsymbol{\varepsilon}_t = \boldsymbol{\theta}_t - \boldsymbol{\theta}^*$，根据更新规则可以得到
\begin{equation}\begin{aligned}
\Vert\boldsymbol{\varepsilon}_{t+1}\Vert^2 =&\, \Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^*\Vert^2 \\
=&\, \Vert\boldsymbol{\theta}_t - (\alpha_t \boldsymbol{u}_t + \rho_t\boldsymbol{\theta}_t) - \boldsymbol{\theta}^*\Vert^2 \\
\approx&\, \Vert\boldsymbol{\varepsilon}_t\Vert^2 - 2 \alpha_t \boldsymbol{u}_t \cdot \boldsymbol{\varepsilon}_t + \left(\alpha_t^2 \Vert\boldsymbol{u}_t\Vert^2 - 2 \rho_t \boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t\right)
\end{aligned}\label{eq:base-approx}\end{equation}
最后的近似只保留了不超过$\mathcal{O}(\alpha_t^2)$的项。

很明显，$\Vert\boldsymbol{\varepsilon}_t\Vert$是当前结果与终点的距离，它自然是越小越好，因此我们自然也希望每一步的更新都能缩小这个距离，即$\Vert\boldsymbol{\varepsilon}_{t+1}\Vert < \Vert\boldsymbol{\varepsilon}_t\Vert$。而我们看式$\eqref{eq:base-approx}$，$- 2 \alpha_t \boldsymbol{u}_t \cdot \boldsymbol{\varepsilon}_t$可正可负，如果它为负就有助于实现$\Vert\boldsymbol{\varepsilon}_{t+1}\Vert < \Vert\boldsymbol{\varepsilon}_t\Vert$，但是$\alpha_t^2 \Vert\boldsymbol{u}_t\Vert^2$必然是正的，它是不利于实现$\Vert\boldsymbol{\varepsilon}_{t+1}\Vert < \Vert\boldsymbol{\varepsilon}_t\Vert$，不过在引入权重衰减后，多出了一项$- 2 \rho_t \boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t$，如果这一项能抵消掉$\alpha_t^2 \Vert\boldsymbol{u}_t\Vert^2$的负面作用，那么权重衰减的引入就不仅能增强泛化能力，还有利于模型收敛了。

## 可行分析

所以，接下来的事情，我们就是要考察
\begin{equation}\alpha_t^2 \Vert\boldsymbol{u}_t\Vert^2 = 2 \rho_t \boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t\label{eq:base-cond}\end{equation}
的可行性。所谓可行性，就是$\boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t$能否大于0，只有它大于0，左右两端才有可能相等。利用$\boldsymbol{\varepsilon}_t$的定义我们得到$\boldsymbol{\theta}_t = \boldsymbol{\varepsilon}_t + \boldsymbol{\theta}^*$，于是
\begin{equation}\boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t = (\boldsymbol{\varepsilon}_t + \boldsymbol{\theta}^*) \cdot \boldsymbol{\varepsilon}_t = \Vert \boldsymbol{\varepsilon}_t\Vert^2 + \boldsymbol{\theta}^* \cdot \boldsymbol{\varepsilon}_t\end{equation}
注意$\boldsymbol{\theta}^*$是我们的目标，是一个固定的点，而$\boldsymbol{\varepsilon}_t$是当前时刻与目标的差异向量，两者一般来说没什么必然的相关性，于是我们可以近似认为它们是高维空间中两个随机向量。根据[《n维空间下两个随机向量的夹角分布》](<https://spaces.ac.cn/archives/7076>)，我们知道高维空间中两个随机向量几乎都是垂直的，于是$\boldsymbol{\theta}^* \cdot \boldsymbol{\varepsilon}_t\approx 0$，即$\boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t \approx \Vert \boldsymbol{\varepsilon}_t\Vert^2$。当然，如果不放心，还可以引入一个参数$q$：
\begin{equation}\boldsymbol{\theta}_t \cdot \boldsymbol{\varepsilon}_t \approx q\Vert \boldsymbol{\varepsilon}_t\Vert^2\end{equation}
此时式$\eqref{eq:base-cond}$就变成了
\begin{equation}\alpha_t^2 \Vert\boldsymbol{u}_t\Vert^2 \approx 2 \rho_t q\Vert \boldsymbol{\varepsilon}_t\Vert^2\label{eq:base-cond-approx}\end{equation}
两端都大于0，因此式$\eqref{eq:base-cond}$是有可能成立的。

## 渐近估计

如果式$\eqref{eq:base-cond}$成立，那么式$\eqref{eq:base-approx}$就简化为了\begin{equation}\Vert\boldsymbol{\varepsilon}_{t+1}\Vert^2 \approx \Vert\boldsymbol{\varepsilon}_t\Vert^2 - 2 \alpha_t \boldsymbol{u}_t \cdot \boldsymbol{\varepsilon}_t = \Vert\boldsymbol{\varepsilon}_t\Vert^2 - 2 \alpha_t \Vert\boldsymbol{u}_t\Vert \Vert\boldsymbol{\varepsilon}_t\Vert \cos(\boldsymbol{u}_t, \boldsymbol{\varepsilon}_t)\end{equation}
我们说了$\boldsymbol{u}_t$代表的是任务相关的更新量，平均来说它必然是有利于任务的（否则原来的优化器就是有缺陷的了），所以平均来说应该有$\cos(\boldsymbol{u}_t, \boldsymbol{\varepsilon}_t) > 0$。这里我们进一步假设，存在一个$p > 0$，使得$\cos(\boldsymbol{u}_t, \boldsymbol{\varepsilon}_t)\sim p$，于是我们有
\begin{equation}\Vert\boldsymbol{\varepsilon}_{t+1}\Vert^2 \approx  \Vert\boldsymbol{\varepsilon}_t\Vert^2 - 2 \alpha_t p\Vert\boldsymbol{u}_t\Vert \Vert\boldsymbol{\varepsilon}_t\Vert\end{equation}
根据近似$\eqref{eq:base-cond-approx}$我们有$\alpha_t \Vert\boldsymbol{u}_t \Vert \Vert \boldsymbol{\varepsilon}_t\Vert \approx \sqrt{2 \rho_t q}\Vert \boldsymbol{\varepsilon}_t\Vert^2$，代入上式得到
\begin{equation}\Vert\boldsymbol{\varepsilon}_{t+1}\Vert^2 \approx  \Vert\boldsymbol{\varepsilon}_t\Vert^2(1 - 2 p\sqrt{2 \rho_t q})\approx \Vert\boldsymbol{\varepsilon}_t\Vert^2\exp(- 2 p\sqrt{2 \rho_t q})\end{equation}
一步一步往前递推，可以得到
\begin{equation}\Vert\boldsymbol{\varepsilon}_t\Vert^2 \approx\Vert\boldsymbol{\varepsilon}_0\Vert^2\exp\left(- 2 \sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right)\label{eq:varepsilon-t}\end{equation}
可以看出右端的指数必然是单调递减的，它是一个衰减函数。现在我们再看近似$\eqref{eq:base-cond-approx}$，它有两个参数$\alpha_t$和$\rho_t$要调，但只有一个（近似）等式。为了使$\alpha_t$和$\rho_t$能够同等程度地衰减，我们设$2\rho_t q \approx \lambda^2 \Vert\boldsymbol{\varepsilon}_t\Vert^2$，于是解得
\begin{equation}\begin{aligned}\alpha_t \approx \frac{\lambda\Vert\boldsymbol{\varepsilon}_t\Vert^2}{\Vert\boldsymbol{u}_t\Vert} \approx&\, \frac{\lambda\Vert\boldsymbol{\varepsilon}_0\Vert^2}{\Vert\boldsymbol{u}_t\Vert} \exp\left(- 2 \sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right) \\
\rho_t \approx \frac{\lambda^2\Vert\boldsymbol{\varepsilon}_t\Vert^2}{2q} \approx&\, \frac{\lambda^2\Vert\boldsymbol{\varepsilon}_0\Vert^2}{2q} \exp\left(- 2 \sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right)
\end{aligned}\label{eq:alpha-rho}\end{equation}
这就是本文推出的$\alpha_t,\rho_t$的变化规律。当然，变化规律是有了，可是还有四个参数$\lambda,\Vert\boldsymbol{\varepsilon}_0\Vert,p,q$要确定，其中$q$相对来说比较简单，直接设$q=1$问题也不大，但即便这样还有三个参数要确定。

## 尺度预判

根据定义，$\Vert\boldsymbol{\varepsilon}_0\Vert = \Vert\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\Vert$，也就是初始化参数与目标参数的距离，可以理解为参数的变化尺度，它有几种不同的情况。

第一种，参数是矩阵乘法核，比如全连接层、卷积层的kernel矩阵，它们的初始化一般是“均值为0、方差为$\sigma^2$”（$\sigma$取决于shape）的随机初始化，这样如果$\boldsymbol{\theta}\in\mathbb{R}^k$，那么我们就可以估算出$\Vert\boldsymbol{\theta}_0\Vert^2\approx k\sigma^2$。另外，这类参数有一个特点，就是在合理的初始化下，训练完成后参数的均值方差也不会有太大变化，至少量级是一致的，因此也可以认为$\Vert\boldsymbol{\theta}^*\Vert^2\approx k\sigma^2$，而因为初始化是随机的，所以$\boldsymbol{\theta}_0 \cdot \boldsymbol{\theta}^*\approx 0$，因此
\begin{equation}\Vert\boldsymbol{\varepsilon}_0\Vert^2 = \Vert\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\Vert^2 = \Vert\boldsymbol{\theta}_0\Vert^2 + \Vert\boldsymbol{\theta}^*\Vert^2 - 2\boldsymbol{\theta}_0 \cdot \boldsymbol{\theta}^* \approx 2k\sigma^2\end{equation}

第二种，参数是加性偏置项，比如全连接层、卷积层的bias向量，以及Normalization层的$\boldsymbol{\beta}$向量，这些参数一般是“全零初始化”，所以$\Vert\boldsymbol{\varepsilon}_0\Vert^2 = \Vert\boldsymbol{\theta}^*\Vert^2$，如果我们根据经验预测训练好的模型偏置项都在$\pm\sigma$附近，那么也可以估计出$\Vert\boldsymbol{\theta}^*\Vert^2\approx k\sigma^2$，Amos原论文取了$\sigma=0.5$。最后还有Normalization层的$\boldsymbol{\gamma}$向量，它一般是“全1初始化”，训练完成后也是在1附近，不妨假设误差为$\pm\sigma$，那么也可以估算出$\Vert\boldsymbol{\theta}^*\Vert^2\approx k\sigma^2$。这里的$k$都是指向量维度。

可以看出，$\Vert\boldsymbol{\varepsilon}_0\Vert^2$的结果都有一个共性，那就是都可以写成$k\sigma^2$，其中$\sigma$是我们对参数变化尺度的一个预判。乘性矩阵的$\sigma$可以直接取初始化的标准差，加性偏置或者$\boldsymbol{\gamma}$向量可以直接简单地取$\sigma=0.5$，或者有其他特殊参数的再做特殊处理。

## 分离尺度

现在我们来看完整的更新量，根据式$\eqref{eq:alpha-rho}$，有
\begin{equation}\alpha_t \boldsymbol{u}_t \approx \lambda\Vert\boldsymbol{\varepsilon}_0\Vert^2 \times \frac{\boldsymbol{u}_t}{\Vert\boldsymbol{u}_t\Vert} \times \exp\left(- 2 \sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right)\end{equation}
其中$\frac{\boldsymbol{u}_t}{\Vert\boldsymbol{u}_t\Vert}$是一个单位向量，控制更新方向，$\exp$部分是一个衰减项，我们可以先不管它，所以更新量的模长由$\lambda\Vert\boldsymbol{\varepsilon}_0\Vert^2$控制。

回到文章开头的第一个问题“学习率如何适应不同初始化和参数化？”，很明显，直观想法应该就是变化尺度大的参数每一步的更新量应该更大，或者直接简单地正比于变化尺度，而变化尺度我们刚才估计了，可以用$\Vert\boldsymbol{\varepsilon}_0\Vert$来描述，所以我们认为应该有$\lambda\Vert\boldsymbol{\varepsilon}_0\Vert^2=\alpha_0 \Vert\boldsymbol{\varepsilon}_0\Vert$，其中$\alpha_0$是全局的初始学习率。反过来解得$\lambda=\alpha_0/\Vert\boldsymbol{\varepsilon}_0\Vert$，代入式$\eqref{eq:alpha-rho}$得到
\begin{equation}\alpha_t \approx \frac{\alpha_0\Vert\boldsymbol{\varepsilon}_0\Vert}{\Vert\boldsymbol{u}_t\Vert} \exp\left(- 2 \sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right),\quad \rho_t \approx \frac{\alpha_0^2}{2q} \exp\left(- 2 \sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right)\label{eq:alpha-rho-2}\end{equation}
其中$\alpha_0$代表了每一步的相对更新幅度（全局学习率），这一步没啥推导空间了，一般取$10^{-3}$左右就行，如果任务简单也可以取到$10^{-2}$；$\Vert\boldsymbol{\varepsilon}_0\Vert$在上一节已经做了估计，大概是$\sqrt{k}\sigma$，$\sigma$代表参数平均变化尺度，不同参数不一样，我们正是通过它把参数尺度显式地分离了出来，从而达到了自适应参数尺度的效果（更新量正比$\sigma$）。特别地，如果将上式的$\Vert\boldsymbol{\varepsilon}_0\Vert$换成$\Vert\boldsymbol{\theta}_t\Vert$，那么就是LAMB优化器。从这里也可以看出，如果$\boldsymbol{\theta}$的初始化均值不是0（像$\boldsymbol{\gamma}$向量），用$\Vert\boldsymbol{\theta}_t\Vert$替代$\Vert\boldsymbol{\varepsilon}_0\Vert$是会有问题的，所以LAMB的做法是直接不对这些参数的更新量进行变换（即保留原来的更新规则）。

## 解析近似

其实目前的结果已经适合编程实现了，只是参数$p$不好调罢了。为了进一步看出参数$p$是怎么影响衰减函数的，我们可以进一步求出$\rho_t$的解析近似！

在式$\eqref{eq:alpha-rho-2}$的$\rho_t$两边乘以$2q$，然后两边开平方，得到
\begin{equation}\sqrt{2q\rho_t} \approx \alpha_0 \exp\left(-\sum_{i=1}^{t-1}p\sqrt{2 \rho_i q}\right)\end{equation}
将指数的求和$\sum\limits_{i=1}^{t-1}p\sqrt{2 \rho_i q}$记为$S_t$，那么上式就对应差分方程
\begin{equation}\frac{S_t - S_{t-1}}{p} \approx \alpha_0 \exp\left(- S_{t-1}\right) \quad \Rightarrow \quad S_{t+1} - S_t \approx \alpha_0 p\exp\left(- S_t\right)\end{equation}
此时衰减函数就是$\exp\left(-2S_t\right)$。为了求渐近近似，我们用导数代替差分（参考[《差分方程的摄动法》](<https://spaces.ac.cn/archives/3889>)），得到
\begin{equation}\frac{dS_t}{dt} \approx \alpha_0 p \exp\left(- S_t\right)\end{equation}
这是个简单的微分方程，可以解得（结合$S_0=0$）
\begin{equation}\exp\left(-2S_t\right) \approx \frac{1}{(\alpha_0 p t + 1)^2}\end{equation}
这就是衰减函数的显式解，表明超参数应该按照步数的平方反比衰减，代入式$\eqref{eq:alpha-rho-2}$后的完整结果是
\begin{equation}\alpha_t \approx \frac{\alpha_0\Vert\boldsymbol{\varepsilon}_0\Vert}{\Vert\boldsymbol{u}_t\Vert} \frac{1}{(\alpha_0 p t + 1)^2},\quad \rho_t \approx \frac{\alpha_0^2}{2q} \frac{1}{(\alpha_0 p t + 1)^2}\label{eq:alpha-rho-3}\end{equation}
这个显式解不但能让编程实现更方便，还使得$p$的含义更为清晰。比如我们希望学习率在$T$步后就降低为原来的一半，那么就有$(\alpha_0 p T + 1)^2=2$，从中解得
\begin{equation}\alpha_0 p = \frac{\sqrt{2}-1}{T}\end{equation}
至于$T$应该是多少，这依赖于任务难度和数据量，也没有太大推导空间了。

## 动态收敛

上述讨论的假设是存在常数$p > 0$，使得$\cos(\boldsymbol{u}_t, \boldsymbol{\varepsilon}_t)\sim p$，这可以理解为模型按照固定的速度收敛，这在实际中很难成立，更常见的是越接近训练的后期，收敛速度相对来说越慢。为此，我们可以进一步假设$p$是步数$t$的函数$p_t$，这样一来，前面的推导大体上还是成立，只不过相应的常数$p$要换成带下标的$p_i$：
\begin{equation}\sqrt{2\rho_t q} \approx \alpha_0 \exp\left(- \sum_{i=1}^{t-1}p_i\sqrt{2 \rho_i q}\right)\end{equation}
重复上一节的推导，我们得到
\begin{equation}\frac{S_t - S_{t-1}}{p_t} \approx \alpha_0 \exp\left(- S_{t-1}\right) \quad \Rightarrow \quad S_{t+1} - S_t \approx \alpha_0 p_t\exp\left(- S_t\right)\end{equation}
近似的微分方程就是
\begin{equation}\frac{dS_t}{dt} \approx \alpha_0 p_t \exp\left(- S_t\right)\end{equation}
积分的结果是
\begin{equation}\exp\left(-S_t\right) \approx \frac{1}{\alpha_0 \int_0^t p_{\tau} d\tau + 1}\end{equation}
但现在多了一个$p_t$需要确定。为了降低调参成本，我们不妨假设收敛的下降速度跟$\Vert\boldsymbol{\varepsilon}_t\Vert$的下降速度一致，而根据式$\eqref{eq:varepsilon-t}$，$\Vert\boldsymbol{\varepsilon}_t\Vert$的衰减函数就是$\exp\left(-S_t\right)$，所以我们设$p_t = p_0\exp\left(-S_t\right)$，代入上式得到
\begin{equation}\exp\left(-S_t\right) \approx \frac{1}{\alpha_0 p_0 \int_0^t \exp\left(-S_{\tau}\right) d\tau + 1}\end{equation}
这本质就是一个简单的微分方程，容易解得
\begin{equation}\exp\left(-2S_t\right) \approx \frac{1}{2\alpha_0 p_0 t + 1}\end{equation}
代入式$\eqref{eq:alpha-rho-2}$之后，得到
\begin{equation}\alpha_t \approx \frac{\alpha_0\Vert\boldsymbol{\varepsilon}_0\Vert}{\Vert\boldsymbol{u}_t\Vert} \frac{1}{2\alpha_0 p_0 t + 1},\quad \rho_t \approx \frac{\alpha_0^2}{2q} \frac{1}{2\alpha_0 p_0 t + 1}\label{eq:alpha-rho-4}\end{equation}
单看衰减策略，这正好是“逆时间衰减（Inverse Time Decay）”，也是学习率的常见衰减策略之一。理论上来说，这个结果在假设上比前面的式$\eqref{eq:alpha-rho-3}$更为合理。

## 文章小结

本文借鉴了Amos优化器的思路，推导了一些关于学习率和权重衰减率的结果$\eqref{eq:alpha-rho-3}$、$\eqref{eq:alpha-rho-4}$，这些结果可以即插即用地应用到现有优化器中，能一定程度上简化调参难度。
