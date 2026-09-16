# MoE环游记：2、不患寡而患不均

> 作者：苏剑林 · 科学空间 · 2025-02-21
>
> 原文：<https://spaces.ac.cn/archives/10735>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章[《MoE环游记：1、从几何意义出发》](<https://spaces.ac.cn/archives/10699>)中，我们介绍了MoE的一个几何诠释，旨在通过Dense模型的最佳逼近出发来推导和理解MoE。同时在文末我们也说了，给出MoE的计算公式仅仅是开始，训练一个实际有效的MoE模型还有很多细节补，比如本文要讨论的负载均衡（Load Balance）问题。

负载均衡，即“不患寡而患不均”，说白了就是让每个Expert都在干活，并且都在干尽可能一样多的活，避免某些Expert浪费算力。负载均衡既是充分利用训练算力的需求，也是尽可能发挥MoE大参数量潜力的需求。

## 需求分析

我们知道，MoE的基本形式是
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\end{equation}
对于传统MoE，$\boldsymbol{\rho}$是一个概率分布（Router），$\boldsymbol{e}_i=\boldsymbol{v}_i$，$\boldsymbol{v}_i$是一个小型FFN（Expert）的输出；而对于我们上一篇推导的几何MoE，$\boldsymbol{\rho}$没有归一化的要求，它预测的是Expert的模长，而$\boldsymbol{e}_i=\boldsymbol{v}_i/\Vert\boldsymbol{v}_i\Vert$预测的是Expert的方向。

不管哪种格式的MoE，实际表现都差不多，只是理解视角的不同。但要注意，虽然MoE的公式给人的感觉是“每遇到一个Token，就去找相应的Expert来计算”，但实际训练时其实是反过来的：先给每个Expert分配好相应的算力，然后将Token分配（Route）到所属的Expert中并行计算，这也就为什么负责打分的$\boldsymbol{\rho}$被称为Router。

这样一来，如果Expert的分配不均衡，就可能出现如下局面：某些Expert（Dead Expert）几乎一直闲置，浪费算力；某些Expert要处理的Token太多，根本忙不过来，只能Token Drop（即放弃处理部分Token）。从理论上来说，出现Dead Expert意味着MoE没有达到预期的参数量，即花了大参数量的显存，结果只训出来小参数量的效果。

所以，不管是从训练还是性能角度看，我们都希望保证Expert的负载均衡。

## 辅助损失

促进负载均衡的常规思路是添加与之相关的损失函数，我们通常称之为“Aux Loss（Auxiliary Loss）”，目前主流用的Aux Loss最早可以追溯到2020年的[《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》](<https://papers.cool/arxiv/2006.16668>)。

介绍Aux Loss之前，我们需要先引入一些新概念。首先，我们已经提到对于一般的MoE来说，$\boldsymbol{\rho}$未必是概率分布，我们将归一化的$\boldsymbol{\rho}$记为$\boldsymbol{p}=[p_1,p_2,\cdots,p_n]$，以及它Top\-$k$版为$\boldsymbol{f}=[f_1,f_2,\cdots,f_n]$，其中
\begin{equation}p_i = \frac{\rho_i}{\sum_{i=1}^n \rho_i},\qquad f_i = \left\{\begin{aligned}1/k, \quad i\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho} \\
0, \quad i\not\in \mathop{\text{argtop}}\nolimits_k \boldsymbol{\rho}\end{aligned}\right.\end{equation}
接着我们定义$\boldsymbol{P}=\mathbb{E}[\boldsymbol{p}],\boldsymbol{F}=\mathbb{E}[\boldsymbol{f}]$，这里的$\mathbb{E}$是指对所有样本的所有Token做平均。不难看出，$\boldsymbol{F}$就是Expert当前的负载分布，而$\boldsymbol{P}$则相当于$\boldsymbol{F}$的一个光滑近似。

有了这些记号，我们就可以写出Aux Loss为：
\begin{equation}\mathcal{L}_{\text{aux}} = \boldsymbol{F}\cdot \boldsymbol{P} = \sum_{i=1}^n F_i P_i\label{eq:aux-loss}\end{equation}
一般文献定义Aux Loss会多乘一个$n$，即它们的Aux Loss等于这里的$n \mathcal{L}_{\text{aux}}$。此外，有些大型MoE可能会按设备来算Aux Loss，以达到设备内的均衡，减少设备间的通信，这些就各自发挥了。但也有较新的实验显示，强行局部均衡极有可能影响模型最终效果。

## 直通估计

不知道大家有没有发现一个奇怪的现象：不管是最早出处、后续文献还是科普文章，总之笔者阅读过的资料中，对Aux Loss的引用都是不加证明的，似乎大家都公认上述Aux Loss能促进均衡是一件显然成立的事情。可真有这么显然易得吗？

反正笔者是没看出来，所以接下来笔者给出式$\eqref{eq:aux-loss}$的一种推导思路，由此思路我们还可以自定义其他形式的Aux Loss。首先，定义均匀分布$\boldsymbol{Q}=(1/n,1/n,\cdots,1/n)$，刚才我们说了$\boldsymbol{F}$就是当前负载分布，因此负载均衡等价于$\boldsymbol{F}=\boldsymbol{Q}$，那么下式就是一个比较直观的Aux Loss：
\begin{equation}\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (F_i - 1/n)^2\label{eq:aux-loss-2}\end{equation}
问题是$\boldsymbol{F}$是由$\mathop{\text{argtop}}_k$出来的，这意味着上式并不是一个能直接用的可导目标。怎么解决这个问题呢？答案是[STE（Straight\-Through Estimator）](<https://spaces.ac.cn/archives/6760#%E8%87%AA%E8%A1%8C%E8%AE%BE%E8%AE%A1%E6%A2%AF%E5%BA%A6>)技巧，分别设计前向传播和反向传播的函数。具体来说，$\boldsymbol{F}$不可导，$\boldsymbol{P}$作为它的光滑近似是可导的，那么我们在反向传播的时候将$\boldsymbol{F}$替换成$\boldsymbol{P}$就行了，即
\begin{equation}\mathcal{L}_{\text{aux}} = \frac{1}{2}\Vert \boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}] - \boldsymbol{Q}\Vert^2 = \frac{1}{2}\sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n)^2\label{eq:aux-loss-3}\end{equation}
其中$\text{sg}[]$是stop gradient算子，特点是保持前向输出不变，但强制梯度为零。这样改动之后，$\mathcal{L}_{\text{aux}}$就是一个切实可行的Aux Loss了，我们可以试求一下它的梯度：
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{aux}} =&\, \frac{1}{2}\nabla_{\boldsymbol{\theta}}\sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n)^2 \\
=&\, \sum_{i=1}^n (P_i + \text{sg}[F_i - P_i] - 1/n) \nabla_{\boldsymbol{\theta}}(P_i + \text{sg}[F_i - P_i] - 1/n)\\
=&\, \sum_{i=1}^n (F_i - 1/n) \nabla_{\boldsymbol{\theta}}P_i = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n (F_i - 1/n) P_i\\
=&\, \nabla_{\boldsymbol{\theta}}\left(\sum_{i=1}^n F_i P_i\right)
\end{aligned}\end{equation}
这里$\boldsymbol{\theta}$是模型参数。最后的结果表明式$\eqref{eq:aux-loss-3}$的梯度等于式$\eqref{eq:aux-loss}$梯度，这意味着用式$\eqref{eq:aux-loss}$作为Aux Loss跟式$\eqref{eq:aux-loss-3}$在梯度上是等价的，所以就出现了式$\eqref{eq:aux-loss}$的Aux Loss。

然而，式$\eqref{eq:aux-loss}$只有等效梯度的意义，但没有Loss的意义，不算一个真正的Loss，比如当$\boldsymbol{F} = \boldsymbol{P}$时我们可以算出式$\eqref{eq:aux-loss}$等于$1/n$，但实际上我们可以构造出一个不等于$\boldsymbol{P}$的$\boldsymbol{F}$让它小于$1/n$，所以式$\eqref{eq:aux-loss}$并不是像正常的Loss一样越小越好，最小值也不是$\boldsymbol{F} = \boldsymbol{P}$时取到。

## 一般形式

上述推导实际上提供了构建Aux Loss的一般思路：<strong>首先基于$\boldsymbol{F}$构建符合要求的损失，然后在实现时将$\boldsymbol{F}$替换成$\boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}]$。</strong>比如，我们知道最大熵也可以将分布推向均衡，因此也可以用熵的相反数来构建Aux Loss：
\begin{equation}\mathcal{L}_{\text{aux}} = \sum_{i=1}^n (P_i + \text{sg}[F_i - P_i])\log(P_i + \text{sg}[F_i - P_i])\end{equation}
上式就可以直接用作代码实现，当然如果我们追求简化，也可以类似地求梯度，结果将是
\begin{equation}\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\text{aux}} = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n(P_i + \text{sg}[F_i - P_i]) \log(P_i + \text{sg}[F_i - P_i]) = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n P_i \log F_i\end{equation}
两次简化梯度的过程中，我们都用到了如下恒等式
\begin{equation}\sum_{i=1}^n \nabla_{\boldsymbol{\theta}}P_i = \nabla_{\boldsymbol{\theta}}\sum_{i=1}^n P_i = \nabla_{\boldsymbol{\theta}}1 = \boldsymbol{0}\end{equation}
这依赖于$\boldsymbol{P}$是一个概率分布，以及目标分布$\boldsymbol{Q}$是均匀分布的事实。而如果我们不追求简化后的等价结果，而是直接用$\boldsymbol{F}\to \boldsymbol{P} + \text{sg}[\boldsymbol{F}-\boldsymbol{P}]$形式的Aux Loss，那么可以不受这两个约束。

比如，$\boldsymbol{P}$作为$\boldsymbol{F}$光滑近似这一点，我们只用到了“$P_i$大$F_i$通常也大”的性质，所以用非归一化的$\mathbb{E}[\boldsymbol{\rho}]$作为$\boldsymbol{P}$通常也没问题，这一点在一些特殊场景（例如有正有负的$\boldsymbol{\rho}$）可能会比较关键，因为此时无法归一化为概率分布。又比如目标$\Vert\boldsymbol{F} - \boldsymbol{Q}\Vert^2$，显然能将$\boldsymbol{F}$推向任意我们想要的、不一定是均匀的目标分布$\boldsymbol{Q}$。

## 文章小结

本文介绍了MoE的负载均衡问题，并给出了一种构建Aux Loss的一般思路。除了Aux Loss外，促进负载均衡还有一些其他方案，我们下回再谈。
