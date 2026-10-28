# 从参数对称性到有效学习率

> 作者：苏剑林 · 科学空间 · 2026-10-27
>
> 原文：<https://spaces.ac.cn/archives/11985>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

近来，越来越多的论文或者博客，开始基于参数的对称性来分析和指导模型的优化，比如[Hyperball](<https://papers.cool/arxiv/2606.16899>)、[HyperTransfer](<https://papers.cool/arxiv/2609.07017>)、[ELR](<https://hy.tencent.com/research/elr>)等，它们总体都是围绕着$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$这一对称性假设来展开探索。如果我们让优化器的设计也“顺应”相应的对称性，往往能让训练过程更为稳健，也能降低超参数调优的心智负担。

然而，对称性假设本身也有一些争议的地方，如何为这条路找到更可靠的基础，让它可以更合理地走得更远？这篇文章分享一下笔者的思考。

## 结果回顾

这一节我们先来整理一下开头提到的三篇文章的逻辑线。

首先，它们都有一个核心假设：模型恒成立$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$，其中$\boldsymbol{w}$是模型的某个参数，$c$是任意正数，$\ell$是损失函数。显然，这个假设意味着模型的效果跟参数$\boldsymbol{w}$模长无关，所以[Hyperball](<https://papers.cool/arxiv/2606.16899>)主张每步更新后都将$\boldsymbol{w}$的模长重新归一化为某个数值$\rho$：
\begin{equation}\boldsymbol{w}_{t+1}=\boldsymbol{w}_t - \eta_t \boldsymbol{u}_t\qquad\to\qquad \boldsymbol{w}_{t+1}=\frac{\boldsymbol{w}_t - \eta_t \boldsymbol{u}_t}{\Vert\boldsymbol{w}_t - \eta_t \boldsymbol{u}_t\Vert_2}\times \rho\end{equation}
另外两篇则提出，我们可以不改更新规则，但既然模型效果与模长无关，那么在训练过程中，$\boldsymbol{w}$的方向向量$\tilde{\boldsymbol{w}} = \boldsymbol{w}/\Vert\boldsymbol{w}\Vert_2$的变化率更为本质，所以它们主张观测和对齐AUS（Angular Update Size）：
\begin{equation}\text{AUS}\triangleq \Vert\tilde{\boldsymbol{w}}_{t+1} - \tilde{\boldsymbol{w}}_t\Vert_2\approx \frac{\eta_t \Vert\boldsymbol{u}_t - \tilde{\boldsymbol{w}}_t \tilde{\boldsymbol{w}}_t^{\top}\boldsymbol{u}_t\Vert_2}{\Vert\boldsymbol{w}_t\Vert_2}\end{equation}
进一步地，由于$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$恒成立，两边对$c$求导然后取$c=1$得$\boldsymbol{w}^{\top}\nabla_{\boldsymbol{w}}\ell = 0$，即$\boldsymbol{w}$的梯度始终与$\boldsymbol{w}$垂直。由于更新量$\boldsymbol{u}_t$基于梯度构建，所以干脆近似地认为这个垂直性质能延伸到更新量上，于是$\tilde{\boldsymbol{w}}_t^{\top}\boldsymbol{u}_t\approx 0$，从而有
\begin{equation}\text{AUS} \approx \frac{\eta_t \Vert\boldsymbol{u}_t\Vert_2}{\Vert\boldsymbol{w}_t\Vert_2}\end{equation}
相比$\Vert\eta_t \boldsymbol{u}_t\Vert_2 = \eta_t \Vert\boldsymbol{u}_t\Vert_2$，上式多除了个$\Vert\boldsymbol{w}_t\Vert_2$，含义是相对更新大小，我们认为它才是更本质的学习率，称为“有效学习率（Effective Learning Rate，ELR）”，不同优化器通过对齐它，可以得到相近的表现，甚至学习率相关的Scaling Law，也应该以它为准。

## 假设分析

然而，这一套下来最根本的问题是：在当前的主流架构下，“$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$恒成立”是一个近乎完全不成立的假设！

注意到，我们现在主流的模型架构都是In Norm形式，不管你搭配的是常规Residual还是MHC、AttnRes，它的每一个子层都是$f(\mathop{\text{Norm}}(\boldsymbol{X}))$，其中
\begin{align}f_{attn}(\boldsymbol{X}) =&\, \varphi(\boldsymbol{X}\boldsymbol{W}_q (\boldsymbol{X}\boldsymbol{W}_k)^{\top}) (\boldsymbol{X} \boldsymbol{W}_v) \boldsymbol{W}_o \\[5pt]
f_{glu}(\boldsymbol{X}) =&\, (\boldsymbol{X}\boldsymbol{W}_{up}\odot \psi(\boldsymbol{X}\boldsymbol{W}_{gate})) \boldsymbol{W}_{down}
\end{align}
$\mathop{\text{Norm}}$只能消除输入尺度的影响，但没法消除这一层参数尺度的影响，那么$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$就无从说起。事实上，对于主流Transformer架构而言，几乎没有一个参数能够满足这种缩放不变性。

唯一例外的是带QK\-Norm的模型，它们的$\boldsymbol{W}_q,\boldsymbol{W}_k$就具备这个特性；此外，由于LM Head的输入也带RMS Norm，所以Embedding层以及所有层的输出矩阵同时乘一个数，输出也保持不变。但这些毕竟只是零星的、有条件的不变性，并不普遍成立，因此我们说$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$是难以成立的假设，甚至近似成立都说不上。

另一方面，Hyperball、ELR、HyperTransfer所得的结论，又能在实践中表现出一定程度上的有效性，这相当于说“结果是对的，但过程有问题”。那么该如何修正这个论证过程，来得到类似的结论呢？

## 联合对称

庆幸的是，虽然$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$这种对称性无法成立，但现有模型架构普遍成立一种联合对称性：对于某些参数对$\boldsymbol{w}_1,\boldsymbol{w}_2$，恒成立$\ell(\boldsymbol{w}_1, \boldsymbol{w}_2)=\ell(c\boldsymbol{w}_1, c^{-1}\boldsymbol{w}_2)$。

这些参数对包括：

> 1、Attention层的$(\boldsymbol{W}_q, \boldsymbol{W}_k)$和$(\boldsymbol{W}_v, \boldsymbol{W}_o)$，以及GLU层的$(\boldsymbol{W}_{up}, \boldsymbol{W}_{down})$；
> 
> 
> 
> 2、如果GLU层的门控用ReLU激活，那么$(\boldsymbol{W}_{gate}, \boldsymbol{W}_{down})$、$(\boldsymbol{W}_{gate}, \boldsymbol{W}_{up})$也是这样的一个参数对（如果是Swish，那么可以视为近似成立）；
> 
> 
> 
> 3、将$\boldsymbol{W}_q, \boldsymbol{W}_k, \boldsymbol{W}_v$视为一个大参数，它与当前层RMS Norm的$\boldsymbol{\gamma}$组成的一个参数对；
> 
> 
> 
> 4、同理，将$\boldsymbol{W}_{up}, \boldsymbol{W}_{gate}$视为一个大参数，它与当前层RMS Norm的$\boldsymbol{\gamma}$组成的一个参数对；
> 
> 
> 
> 5、LM Head层的RMS Norm的$\boldsymbol{\gamma}$参数与LM Head的参数矩阵组成这样的一个参数对。

第3、4点尤其值得留意：这里必须把整组输入矩阵一起缩放。只缩放其中一个矩阵，虽然能抵消这对参数的尺度变化，却会让组内其他参数多缩放一个因子，前向结果通常仍会改变。此外，有些联合对称还能细化到逐向量甚至逐通道层面，这说明尺度冗余在现代模型中并不少见，只是它通常依附于参数之间的配合。

## 尝试替代

上一节介绍的普遍存在的对称性，能否替代难以成立的$\ell(\boldsymbol{w})=\ell(c\boldsymbol{w})$，重新支撑起Hyperball、ELR等结论呢？

首先，$\ell(\boldsymbol{w}_1, \boldsymbol{w}_1)=\ell(c\boldsymbol{w}_1, c^{-1}\boldsymbol{w}_2)$意味着我们可以固定其中一个参数的模长，而不损失模型的表达能力。留意到Attention层和GLU层多数的矩阵参数都和它们相应的$\boldsymbol{\gamma}$参数有这种联合对称性，因此我们可以考虑固定矩阵参数的模长，但保留$\boldsymbol{\gamma}$模长的可训性，这才是Hyperball所主张的。

但这里还要注意一个细节：以Attention层为例，$\boldsymbol{\gamma}$的联合对称性，是需要将$\boldsymbol{W}_q, \boldsymbol{W}_k, \boldsymbol{W}_v$视为整体才能得出的，并且也不涉及到$\boldsymbol{W}_o$，但Hyperball的实践是分别固定每个参数的模长，所以它理论上还是有损表达力的，其有效性不能完全由联合对称性解释，它更多表明模型的容错性要比我们想象中强得多。

接着，$\ell(\boldsymbol{w}_1, \boldsymbol{w}_1)=\ell(c\boldsymbol{w}_1, c^{-1}\boldsymbol{w}_2)$两边对$c$求导，然后代入$c=1$，将得到
\begin{equation}0 = \boldsymbol{w}_1^{\top}\nabla_{\boldsymbol{w}_1}\ell - \boldsymbol{w}_2^{\top}\nabla_{\boldsymbol{w}_2}\ell = [\boldsymbol{w}_1, -\boldsymbol{w}_2]^{\top} [\nabla_{\boldsymbol{w}_1}\ell, \nabla_{\boldsymbol{w}_2}\ell]\end{equation}
这表明将$(\boldsymbol{w}_1, -\boldsymbol{w}_2)$视为一个向量，它与梯度$(\nabla_{\boldsymbol{w}_1}\ell, \nabla_{\boldsymbol{w}_2}\ell)$始终保持垂直。当联合对称性足够多时（比如$\boldsymbol{W}_q$既与$\boldsymbol{W}_k$联合对称，也与$\boldsymbol{\gamma}$联合对称），我们假设这个垂直性质可以分解到每一个向量上，即认为
\begin{equation}[\boldsymbol{w}_1, -\boldsymbol{w}_2]^{\top} [\nabla_{\boldsymbol{w}_1}\ell, \nabla_{\boldsymbol{w}_2}\ell] = 0 \qquad\Rightarrow\qquad \boldsymbol{w}_1^{\top} \nabla_{\boldsymbol{w}_1}\ell\approx 0,\quad \boldsymbol{w}_2^{\top} \nabla_{\boldsymbol{w}_2}\ell\approx 0\end{equation}
于是前面关于AUS、ELR的分析还可以复现。当然，这里涉及到从整体垂直到局部垂直、从梯度垂直到更新量垂直，有很长的假设链，并不保证准确，所以[《Hyperball May Not Be a Free Lunch》](<https://papers.cool/arxiv/2607.22444>)在有效学习率的定义中，显式保留参数和更新量的内积项，这也不失为一种更合理的选择。

## 推导反思

回顾整个推导链条，我们主要在试图论证如下几点：模长不重要、AUS更本质、AUS与ELR的等价性。但如果进一步追问，我们会发现不管是模长还是AUS，这些观测对象的选择本身就有非常主观，没有说明这背后更本质的期望是什么。

我们知道，每一种对称性都代表着模型的一个不变性质，即对于前向计算来说，变换前后的模型是两个数学上完全等价的函数。对于训练来说，我们期望两种不同但前向等价的参数化，能获得完全一样的训练轨迹，这样能增强各种超参数的可迁移性，让训练过程中的心智负担更低。

举个例子，假设有模型$\ell(\boldsymbol{w})$，然后引入另外一个模型$\ell(c^{-1}\hat{\boldsymbol{w}})$，其中$\hat{\boldsymbol{w}}=c\boldsymbol{w}$。显然这两个模型是完全等价的，但它们在同一优化器、同一超参数下，训练轨迹往往不等价，比如SGD下我们分别有
\begin{align}\boldsymbol{w}_{t+1} =&\, \boldsymbol{w}_t - \eta_t \nabla_{\boldsymbol{w}_t}\ell \\[5pt]
\hat{\boldsymbol{w}}_{t+1} =&\, \hat{\boldsymbol{w}}_t - \eta_t \nabla_{\hat{\boldsymbol{w}}_t}\ell = \hat{\boldsymbol{w}}_t - \eta_t c^{-1}\nabla_{\boldsymbol{w}_t}\ell \quad\Rightarrow\quad \boldsymbol{w}_{t+1} = \boldsymbol{w}_t - c^{-2}\eta_t \nabla_{\boldsymbol{w}_t}\ell
\end{align}
由此可见，从$\ell(\boldsymbol{w})$变成$\ell(c^{-1}\hat{\boldsymbol{w}})$，虽然前向是等价的，但后者在SGD下相当于学习率缩放到$c^{-2}$倍，因此优化轨迹不等价。这个特性对优化并不友好，因为它不仅要我们保证模型的函数等价性，还要求精细对齐参数化方式，才能复现模型的训练过程和效果。

## 理想更新

什么样的更新量才是理想更新量呢？设更新量一般形式为$\phi(\boldsymbol{w}, \boldsymbol{g})$，其中$\boldsymbol{g}=\nabla_{\boldsymbol{w}}\ell$，从上一节可以看出，参数乘以$c$，梯度就会除以$c$，此时我们期望优化器给出的更新量也会同时乘以$c$，这样才能使得更新轨迹理论上不变，即满足如下协变性
\begin{equation}\phi(c\boldsymbol{w}, c^{-1}\boldsymbol{g}) = c\,\phi(\boldsymbol{w}, \boldsymbol{g})\end{equation}
但这还不够。对优化器我们还有一个更基本的期望，那就是给损失函数随便乘个正常数，优化轨迹也不应该发生变化，因为乘上的正常数理论上不携带任何有效信息。这样一来，上式$c^{-1}$应该可以换成任意常数，于是完整的期望的协变性是：对于任意$c_1,c_2 > 0$，恒成立
\begin{equation}\phi(c_1\boldsymbol{w}, c_2\boldsymbol{g}) = c_1\phi(\boldsymbol{w}, \boldsymbol{g})\label{eq:phi}\end{equation}
两端除以$c_1$，然后代入$c_1=1/\Vert\boldsymbol{w}\Vert, c_2 = 1/\Vert\boldsymbol{g}\Vert$，得
\begin{equation}\phi(\boldsymbol{w}, \boldsymbol{g}) = \Vert\boldsymbol{w}\Vert\, \phi\left(\frac{\boldsymbol{w}}{\Vert\boldsymbol{w}\Vert}, \frac{\boldsymbol{g}}{\Vert\boldsymbol{g}\Vert}\right)\end{equation}
留意到当$\boldsymbol{w},\boldsymbol{g}$本就满足$\Vert\boldsymbol{w}\Vert=1,\Vert\boldsymbol{g}\Vert=1$时，上式自动成立，换句话说，当$\Vert\boldsymbol{w}\Vert=1,\Vert\boldsymbol{g}\Vert=1$时，$\phi$可以是任意二元函数$F(\cdot, \cdot)$，所以它的通解是
\begin{equation}\phi(\boldsymbol{w}, \boldsymbol{g}) = \Vert\boldsymbol{w}\Vert\, F\left(\frac{\boldsymbol{w}}{\Vert\boldsymbol{w}\Vert}, \frac{\boldsymbol{g}}{\Vert\boldsymbol{g}\Vert}\right)\label{eq:phi-F}\end{equation}

## 结果分析

式$\eqref{eq:phi-F}$给出了满足协变性$\eqref{eq:phi}$的解的一般结构：对$\boldsymbol{w}$、$\boldsymbol{g}$分别归一化，然后允许它们俩做任意运算，最后的运算结果再乘上$\Vert\boldsymbol{w}\Vert$。从这个形式不难看出，将SGD变得满足协变性的最简单方案是
\begin{equation}\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta_t \Vert\boldsymbol{w}_t\Vert \frac{\boldsymbol{g}_t}{\Vert\boldsymbol{g}_t\Vert}\label{eq:elr-g}\end{equation}
对于一般优化器，可以考虑将$\boldsymbol{g}_t$换成$\boldsymbol{u}_t$，得
\begin{equation}\boldsymbol{w}_{t+1} = \boldsymbol{w}_t - \eta_t \Vert\boldsymbol{w}_t\Vert \frac{\boldsymbol{u}_t}{\Vert\boldsymbol{u}_t\Vert}\label{eq:elr-u}\end{equation}
由于我们已经显式分离出了$\Vert\boldsymbol{w}_t\Vert$和$\Vert\boldsymbol{u}_t\Vert$，所以这里的$\eta_t$就是前面的有效学习率。至此，我们以一种截然不同的方式导出了有效学习率的概念，它是将优化器重新改写成满足协变性的最简单方式，或者说，对齐有效学习率来实现训练轨迹的可迁移性，本质上就是在实现优化器的协变性。

注意一个细节：模长我们写的是$\Vert\cdot\Vert$，而不是$\Vert\cdot\Vert_2$。这意味着它并非天然绑定欧氏范数或者RMS，实际上代入任意一种范数都能满足所期望的协变性，甚至非范数也行，只需要它是向量到标量的一个正齐次的映射，选择欧氏范数或RMS，更多是因为它计算简单且几何意义直观。

最后，式$\eqref{eq:phi-F}$中$F$的任意性也提醒我们：满足协变性的优化器并不唯一，式$\eqref{eq:elr-u}$只是其中看上去最简单的一个。

## 似曾相识

如果我们重温“让炼丹更科学一些”系列文章，就会发现其实类似式$\eqref{eq:elr-g}$的结论早已出现过。比如在[《让炼丹更科学一些（五）：基于梯度精调学习率》](<https://spaces.ac.cn/archives/11530>)中我们推导过，在最小化收敛上界的设置下，最优学习率的一个经典形式是
\begin{equation}\eta_t = \frac{R}{\sqrt{\sum_{k=1}^t\Vert\boldsymbol{g}(\boldsymbol{x}_k,\boldsymbol{\theta}_k)\Vert^2}}\end{equation}
其中$\boldsymbol{g}(\boldsymbol{x}_k,\boldsymbol{\theta}_k)$是梯度，而$R$是参数与最优点的距离上界。由此可见，即便是经典的凸优化结果，它的最佳学习率也已经指向了
\begin{equation}\eta_t \propto \frac{\text{参数模长 (的变式)}}{\text{梯度模长 (的变式)}}\end{equation}
的形式，而这个比例系数，便相当于有效学习率！所以，有效学习率的概念其实早有踪迹，只不过之前我们只是把参数模长的部分当作超参数来搜索，没有把它单独分离出来，只留下基于二阶矩的自适应学习率部分。

不过，需要指出的是，在相关收敛结论的推导中，为了分析的方便，我们通常将$R$取成常数（某个共同上界或者初始值，取决于不同的分析手段），所以它跟有效学习率的动态定义不完全一致。但原则上我们可以做更精细的假设，得到更符合当前有效学习率的动态结论，这些后面再找机会讨论了。

## 调度形式

多个不同的视角，都指向了有效学习率，这让我们相信对齐有效学习率也许在某种程度上确实更为本质。那么接着很自然就引出一个问题：有效学习率的调度形式（ELR Schedule）该怎么调整呢？

在[《让炼丹更科学一些（六）：自上而下的精妙构造》](<https://spaces.ac.cn/archives/11540>)、[《让炼丹更科学一些（十）：单调性假设的拆与补》](<https://spaces.ac.cn/archives/11885>)等文章中，我们基于一些假设，推导出SGD或AdaGrad等自适应学习率优化器的最佳LR Schedule是Linear Decay，我们便以此为出发点。当然实践中Cosine Decay也很常见，但笔者的实验中，两者表现类似，所以还是以Linear为例。
\begin{equation}\eta_t = \eta_0(1 - t/T)\end{equation}
简单起见，这里假设了终点是0。接下来，我们主要面向Adam推导。在[《为什么Adam的Update RMS是0\.2？》](<https://spaces.ac.cn/archives/11267>)和[《AdamW的Weight RMS的渐近估计（下）》](<https://spaces.ac.cn/archives/11404>)中，我们分别推出
\begin{align}\Vert\boldsymbol{u}_t\Vert_{RMS} \approx&\,  \sqrt{\frac{1 - \beta_1}{1 + \beta_1}} \\
\Vert\boldsymbol{w}_t\Vert_{RMS}^2 \approx&\, e^{-\lambda\eta_0 t}\Vert\boldsymbol{w}_0\Vert_{RMS}^2 + (1 - e^{- \lambda\eta_0 t})\frac{\eta_0}{3\lambda}
\end{align}
其中$\lambda$是Weight Decay。于是一个参考的ELR Schedule形式是
\begin{equation}\frac{\eta_t \Vert\boldsymbol{u}_t\Vert_{RMS}}{\Vert\boldsymbol{w}_t\Vert_{RMS}} \propto \frac{\eta_0 (1 - t/T)}{\sqrt{e^{-\lambda\eta_0 t}\Vert\boldsymbol{w}_0\Vert_{RMS}^2 + (1 - e^{- \lambda\eta_0 t})\eta_0/3\lambda}}\propto \frac{1 - t/T}{\sqrt{1 - c_0 e^{-c_1 t}}}\end{equation}
由于我们一般会取偏小的初始化，即$\Vert\boldsymbol{w}_0\Vert_{RMS}^2 < \eta_0/3\lambda$（对应$c_0 > 0$），所以训练过程中模长趋势是上涨的，此时上述ELR曲线跟[《From LR to ELR》](<https://hy.tencent.com/research/elr>)、[《Training nGPT》](<https://papers.cool/arxiv/2608.01284>)等文章给出的参考ELR Schedule形状类似，都是“起点更高、前期下降更快、后期接近线性”的形式。

至于Muon等优化器，虽然结果跟Adam不完全相同，但我们通常认为它们有相似的渐近规律，所以上述结果一般也可以推广到Muon上。

## 文章小结

本文围绕着参数对称性和有效学习率进行讨论，重新审视了有效学习率常见的论证链，引入了更严谨的联合对称性来修正原有的推导，最后补充了一个新的基于协变性的理解视角。
