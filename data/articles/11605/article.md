# MuP之上：2\. 线性层与最速下降

> 作者：苏剑林 · 科学空间 · 2026-02-15
>
> 原文：<https://spaces.ac.cn/archives/11605>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章[《MuP之上：1\. 好模型的三个特征》](<https://spaces.ac.cn/archives/11340>)中，我们提出了前向稳定性、依赖稳定性、更新稳定性这三个核心指标，并给出了相应的数学定义。同时，我们提出以它们是否满足$\Theta(1)$来刻画一个模型的好坏，这将作为我们后续分析和计算的理论基石。接下来，我们将会把它们跟最速下降思想结合，给每个参数定制“稳中求快”的更新规则。

\begin{align}
&\text{前向稳定性:}\quad\max_{\boldsymbol{x}} \Vert \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\Vert_{RMS} = \Theta(1) \label{eq:c1} \\[5pt]
&\text{依赖稳定性:}\quad\max_{\boldsymbol{x}_1,\boldsymbol{x}_2} \frac{\Vert \boldsymbol{f}(\boldsymbol{x}_1;\boldsymbol{\omega}) - \boldsymbol{f}(\boldsymbol{x}_2;\boldsymbol{\omega})\Vert_{RMS}}{\Vert\boldsymbol{x}_1 - \boldsymbol{x}_2\Vert_{RMS}} = \Theta(1) \label{eq:c2} \\[5pt]
&\text{更新稳定性:}\quad\max_{\boldsymbol{x}} \Vert \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega} + \Delta\boldsymbol{\omega}) - \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})\Vert_{RMS} = \Theta(1) \label{eq:c3}
\end{align}

我们以线性层作为第一个例子，其结果对部分读者来说应该不陌生，它就是去年逐渐兴起的Muon优化器。当然，我们的目的并不是重新发现Muon，而是展示从第一性原理出发来设计模型和优化器的过程，为我们后续处理其他参数提供统一的方法论。

## 线性变换

对于线性层，输入是向量$\boldsymbol{x}\in\mathbb{R}^{d_{in}}$, 参数是矩阵$\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$，模型则是$\boldsymbol{f}(\boldsymbol{x};\boldsymbol{W})=\boldsymbol{x}\boldsymbol{W}$。注意，三个指标的定义中我们都没有限定$\boldsymbol{x}$有界，所以对于朴素的线性层，三个指标都不一定存在，比如$\max\limits_{\boldsymbol{x}}\Vert\boldsymbol{x}\boldsymbol{W}\Vert_{RMS}$一般都是无穷大。对此，我们只需给模型补充一些让结果有界的运算，比如：
\begin{align} \newcommand{Norm}{\mathop{\text{Norm}}}
&\text{In Norm:}\quad \Norm(\boldsymbol{x})\boldsymbol{W} \\[5pt]
&\text{Out Norm:}\quad \Norm(\boldsymbol{x}\boldsymbol{W})
\end{align}
其中$\Norm(\boldsymbol{x}) = \boldsymbol{x} / \Vert\boldsymbol{x}\Vert_{RMS}$，这里省掉了RMS Norm带有的gamma参数，假设它的影响是次要的。我们知道，残差有Pre Norm和Post Norm两种常见用法，Pre Norm显然对应于In Norm，但这里要指出的是Post Norm实际上也是In Norm：
\begin{align} \newcommand{Norm}{\mathop{\text{Norm}}}
&\text{Pre Norm:}\quad \boldsymbol{x}_{t+1} = \boldsymbol{x}_t + \boldsymbol{F}_t(\Norm(\boldsymbol{x}_t)) \\[5pt]
&\text{Post Norm:} \quad \boldsymbol{x}_{t+1} = \Norm(\underbrace{\boldsymbol{x}_t + \boldsymbol{F}_t(\boldsymbol{x}_t)}_{\text{记为}\boldsymbol{y}_{t+1}}) \quad \Rightarrow\quad \boldsymbol{y}_{t+1} = \Norm(\boldsymbol{y}_t) + \boldsymbol{F}_t(\Norm(\boldsymbol{y}_t))
\end{align}
所以Post Norm相比Pre Norm只不过是$\boldsymbol{x}_t + \boldsymbol{F}_t(\Norm(\boldsymbol{x}_t))$换成了$\Norm(\boldsymbol{x}_t) + \boldsymbol{F}_t(\Norm(\boldsymbol{x}_t))$，对于$\boldsymbol{F}_t$来说，它们都是In Norm，本文也是以In Norm为例。

相比Out Norm，In Norm还有一个好处是提速空间更大，因为$(\boldsymbol{x} / \Vert\boldsymbol{x}\Vert_{RMS})\boldsymbol{W}=\boldsymbol{x}\boldsymbol{W} / \Vert\boldsymbol{x}\Vert_{RMS}$，理论上$\boldsymbol{x}\boldsymbol{W}$和$\Vert\boldsymbol{x}\Vert_{RMS}$可以并行计算，最后再相除，减少延迟，这一思想体现在[《FlashNorm: fast normalization for LLMs》](<https://papers.cool/arxiv/2407.09577>)、[《Block\-level AI Operator Fusion》](<https://papers.cool/arxiv/2505.07829>)、[《Superoptimizing RMSNorm and Linear》](<https://mirage-project.readthedocs.io/en/latest/tutorials/rms-norm-linear.html>)等工作中。

## 初始方差

根据上一节的讨论，约定我们只考虑带有In Norm的线性层，那么由谱范数定义可以计算三个指标：
\begin{align}
&\text{前向稳定性:}\quad\max_{\Vert\boldsymbol{x}\Vert_{RMS}=1} \Vert \boldsymbol{x}\boldsymbol{W}\Vert_{RMS} = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{W}\Vert_2 \\[5pt]
&\text{依赖稳定性:}\quad\max_{\Vert\boldsymbol{x}_1\Vert_{RMS}=\Vert\boldsymbol{x}_2\Vert_{RMS}=1} \frac{\Vert \boldsymbol{x}_1\boldsymbol{W} - \boldsymbol{x}_2\boldsymbol{W}\Vert_{RMS}}{\Vert \boldsymbol{x}_1 - \boldsymbol{x}_2\Vert_{RMS}} = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{W}\Vert_2 \\[5pt]
&\text{更新稳定性:}\quad\max_{\Vert\boldsymbol{x}\Vert_{RMS}=1} \Vert \boldsymbol{x}(\boldsymbol{W} + \Delta\boldsymbol{W}) - \boldsymbol{x}\boldsymbol{W}\Vert_{RMS} = \sqrt{\frac{d_{in}}{d_{out}}}\Vert\Delta\boldsymbol{W}\Vert_2
\end{align}
其中对一个矩阵取$\Vert\cdot\Vert_2$表示该矩阵的谱范数，可以看到，三个指标都是谱范数的某个变体，或者更准确地说，笔者所提的这三个指标，本就是从谱范数出发所做的推广。

前两个指标是关于$\boldsymbol{W}$的函数，对线性层来说它们刚好是一样的，如果我们希望它们是$\Theta(1)$，那么有$\Vert\boldsymbol{W}\Vert_2 = \Theta(\sqrt{d_{out}/d_{in}})$，这至少对$\boldsymbol{W}$的初始化提出了要求。根据[《随机矩阵的谱范数的快速估计》](<https://spaces.ac.cn/archives/11335>)，一个$d_{in}\times d_{out}$大小的标准正态矩阵，它的谱范数大概是$\sqrt{d_{in}} + \sqrt{d_{out}}$，所以要想初始化满足$\Vert\boldsymbol{W}\Vert_2 = \Theta(\sqrt{d_{out}/d_{in}})$，初始方差$\sigma^2$应当满足
\begin{equation}\sigma = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\frac{1}{\sqrt{d_{in}} + \sqrt{d_{out}}}\right)\end{equation}

此外，我们也可以考虑在优化过程中一直约束$\Vert\boldsymbol{W}\Vert_2$，这启发了一些工作，比如[《流形上的最速下降：4\. Muon \+ 谱球面》](<https://spaces.ac.cn/archives/11241>)和[《Controlled LLM Training on Spectral Sphere》](<https://papers.cool/arxiv/2601.08393>)，这一块我们后面再谈。

## 最速下降

接下来我们主要看“更新稳定性”指标$\sqrt{d_{in}/d_{out}}\Vert\Delta\boldsymbol{W}\Vert_2$，这是参数增量$\Delta\boldsymbol{W}$的谱范数变体。众所周知，更新量由优化器决定，所以这一块提供的是优化器的指导。按照“稳中求快”原则，现在“稳”已经有了，那什么时候最快呢？

这便是最速下降要回答的问题，此前我们在[《Muon续集：为什么我们选择尝试Muon？》](<https://spaces.ac.cn/archives/10739>)、[《流形上的最速下降：1\. SGD \+ 超球面》](<https://spaces.ac.cn/archives/11196>)、[《流形上的最速下降：2\. Muon \+ 正交》](<https://spaces.ac.cn/archives/11215>)等文章也有过相关讨论，但为了该系列文章的完整性，我们还是不厌其烦地重复一遍。最速下降是指在某个约束下让损失下降最快的更新量，形式化定义为
\begin{equation}\min_{\Delta \boldsymbol{W}} \mathcal{L}(\boldsymbol{W} +\Delta\boldsymbol{W}) \qquad \text{s.t.}\qquad \rho(\Delta\boldsymbol{W})\leq \eta\end{equation}
其中$\mathcal{L}$是损失函数，$\rho(\Delta\boldsymbol{W})$是增量$\Delta\boldsymbol{W}$的稳定性指标，这里我们已经有了，即$\sqrt{d_{in}/d_{out}}\Vert\Delta\boldsymbol{W}\Vert_2$。但直接求解该问题仍然过于复杂了，我们需要将$\mathcal{L}(\boldsymbol{W} +\Delta\boldsymbol{W})$换成一阶近似$\mathcal{L}(\boldsymbol{W}
) + \langle \boldsymbol{G}, \Delta\boldsymbol{W}\rangle_F$，才能使得求解变得可行。此时，待求问题等价于
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\min_{\Delta \boldsymbol{W}} \tr(\boldsymbol{G}^{\top}\Delta\boldsymbol{W}) \qquad \text{s.t.}\qquad \Vert\Delta\boldsymbol{W}\Vert_2\leq\eta\sqrt{\frac{d_{out}}{d_{in}}}\end{equation}
其中$\boldsymbol{G}=\nabla_{\boldsymbol{W}}\mathcal{L}(\boldsymbol{W})$是损失函数的梯度，并且我们利用了恒等式$\langle \boldsymbol{G}, \Delta\boldsymbol{W}\rangle_F=\tr(\boldsymbol{G}^{\top}\Delta\boldsymbol{W})$。

## 求解过程

进一步地，我们设$\Delta\boldsymbol{W}=-\kappa \boldsymbol{\Phi}$，将优化目标改写成
\begin{equation}\max_{\kappa,\boldsymbol{\Phi}}\kappa\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad 0\leq \kappa \leq \eta\sqrt{\frac{d_{out}}{d_{in}}}, \quad\Vert\boldsymbol{\Phi}\Vert_2=1\end{equation}
很明显，$\kappa$的优化可以单独完成，最大值在$\kappa = \eta\sqrt{d_{out}/d_{in}}$取到，所以我们只需要求解
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2=1\end{equation}
接下来，设$\boldsymbol{G}$可以SVD成$\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top} = \sum\limits_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top}$，$r$是$\boldsymbol{G}$的秩，我们有
\begin{equation}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})=\tr\left(\sum_{i=1}^r \sigma_i \boldsymbol{v}_i \boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\right) = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i\end{equation}
根据定义，当$\Vert\boldsymbol{\Phi}\Vert_2=1$时$\Vert\boldsymbol{\Phi}\boldsymbol{v}_i\Vert_2\leq \Vert\boldsymbol{v}_i\Vert_2=1$，于是$\boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i\leq 1$，因此
\begin{equation}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})\leq \sum_{i=1}^r \sigma_i = \Vert \boldsymbol{G}\Vert_*\end{equation}
其中$\Vert\cdot\Vert_*$称为矩阵的[核范数（Nuclear Norm）](<https://en.wikipedia.org/wiki/Schatten_norm>)，等号在所有$\boldsymbol{u}_i^{\top}\boldsymbol{\Phi}\boldsymbol{v}_i$都等于1时取到，此时
\begin{equation}\newcommand{msign}{\mathop{\text{msign}}}\boldsymbol{\Phi} = \sum_{i=1}^r \boldsymbol{u}_i \boldsymbol{v}_i^{\top} = \boldsymbol{U}_{[:,:r]}\boldsymbol{V}_{[:,:r]}^{\top} = \msign(\boldsymbol{G})\end{equation}

## 结果汇总

简单总结一下，到目前为止，我们从三个稳定性指标触发，至少得到了两个结论，一是参数$\boldsymbol{W}$的初始化方差$\sigma^2$应当满足
\begin{equation}\sigma = \Theta\left(\sqrt{\frac{d_{out}}{d_{in}}}\frac{1}{\sqrt{d_{in}} + \sqrt{d_{out}}}\right)\end{equation}
二是它的增量$\Delta\boldsymbol{W}$应当取如下形式
\begin{equation}\Delta\boldsymbol{W} = -\eta\sqrt{\frac{d_{out}}{d_{in}}}\msign(\boldsymbol{G})\end{equation}
这正好是MuP版Muon（几个版本的区别可以参考[《Muon优化器指南：快速上手与关键细节》](<https://spaces.ac.cn/archives/11416>)，标准的Muon会将$\boldsymbol{G}$换成它的动量，这可以视为更平滑的梯度估计）。此外，对于$\boldsymbol{W}$的约束，我们也还有一些工作可做，这留到之后的文章再探讨。

由于此前我们已有多篇博客对MuP和Muon做了充分的介绍，所以目前这两个结果都不是新的。所以，本文只是作为第一个案例，演示指标$\eqref{eq:c1},\eqref{eq:c2},\eqref{eq:c3}$的合理性，它们将为任意层的参数及其增量提供统一的稳定性指标公式，进而将Muon的结论一般化。

## 遗留问题

在推广之前，我们还有一个问题需要回答：前面的推导都基于In Norm设计，那么需要给每一个线性层都加上In Norm吗？如果没有In Norm还可以用Muon吗？要回答这个问题，我们先借用上一篇文章的一段话：

> 这里的$\boldsymbol{f}(\boldsymbol{x};\boldsymbol{\omega})$可以是一个层、若干层组成的块甚至是整个模型，理论上，颗粒度越粗所得约束就越宽松或者说越准确，但$\max$的求解也越困难，所以这取决于我们计算$\max$的能力。

简单来说，就是稳定性指标的计算越准确越好，但允许近似。所以没有In Norm的情况下，Muon多大程度上可用，取决于“$\Vert\boldsymbol{x}\Vert_{RMS}=\text{某个常数}$”多大程度上成立。比如FFN层$\boldsymbol{y}=\phi(\boldsymbol{x}\boldsymbol{W}_{up})\boldsymbol{W}_{down}$，如果我们假设激活函数$\phi$的Lipschitz系数为1，那么仍成立
\begin{equation}\Vert\boldsymbol{y}\Vert_{RMS} \leq \Vert\boldsymbol{x}\Vert_{RMS} \times\sqrt{\frac{d_{in}}{d_{mid}}}\Vert\boldsymbol{W}_{up}\Vert_2\times \sqrt{\frac{d_{mid}}{d_{out}}}\Vert\boldsymbol{W}_{down}\Vert_2\end{equation}
其中$\boldsymbol{W}_{up}\in\mathbb{R}^{d_{in}\times d_{mid}},\boldsymbol{W}_{down}\in\mathbb{R}^{d_{mid}\times d_{out}}$。这样一来，即便我们只给$\boldsymbol{x}$加RMS Norm，对第二个参数$\boldsymbol{W}_{down}$来说，同样的稳定性指标也是近似成立的，因此Muon也是可用的。

类似地，即便完全不加RMS Norm，但如果我们依然认为“$\Vert\boldsymbol{x}\Vert_{RMS}=\text{某个常数}$”能够在某种程度上成立，那么对于其后的线性层，我们仍然可以尝试Muon优化器。

## 文章小结

本文以上一篇文章的三个稳定性指标为出发点，针对线性层展示了“复现”MuP和Muon相关结论的过程。接下来，我们会将这套方法论，用于为线性层以外的参数“定制”初始化和优化器等内容。
