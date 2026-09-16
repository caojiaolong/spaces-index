# 直接以FID为Loss：从梯度计算到流式训练

> 作者：苏剑林 · 科学空间 · 2026-05-08
>
> 原文：<https://spaces.ac.cn/archives/11738>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

关注视觉生成模型的读者都知道，FID一直是其关键的评价指标之一，它越小往往意味着生成效果越真实。那么一个自然的问题是：为什么不干脆直接以FID为损失函数来训练生成模型呢？难道是因为FID不可导？非也，FID实际上是可导的，它作为Loss理论上没有问题，但实践中会遇到计算困难。

近日，论文[《Representation Fréchet Loss for Visual Generation》](<https://papers.cool/arxiv/2604.28190>)做了一些克服困难的尝试，成功将FID用于生成模型的微调，并明显改进了单步生成的效果。本文将简要探讨一下其中的数学原理与实现技巧。

## 生成指标

FID，全称是“Fréchet Inception Distance”，我们可以分“Fréchet Distance（FD）”和“Inception（I）”两部分来理解。

假设有两个分布$p$和$q$，它们分别代表真实样本、生成样本，我们将各自的样本$\boldsymbol{x}$通过某个预训练好的编码器$\phi$编码成特征向量$\boldsymbol{z}=\phi(\boldsymbol{x})\in\mathbb{R}^d$，并估计各自的均值向量$\boldsymbol{\mu}_p,\boldsymbol{\mu}_q$和协方差矩阵$\boldsymbol{\Sigma}_p,\boldsymbol{\Sigma}_q$。然后，假设编码结果服从多元正态分布，那么我们就可以用正态分布的差异函数来度量它们的差距。Fréchet Distance（FD）选择的是W距离：
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\begin{aligned}
\mathcal{F}\triangleq\mathcal{W}_2^2[p,q]=&\,\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \tr(\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q - 2(\boldsymbol{\Sigma}_p\boldsymbol{\Sigma}_q)^{1/2})\\[4pt]
=&\,\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2 + \tr(\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q - 2(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2})
\end{aligned}\label{eq:w-p-q}\end{equation}
将各自编码结果的均值向量和协方差矩阵代入上式，所得结果便称为“Fréchet Distance（FD）”。至于上式的推导过程，有兴趣的读者可以参考[《两个多元正态分布的KL散度、巴氏距离和W距离》](<https://spaces.ac.cn/archives/8512>)。

如果将编码器$\phi$选取为[InceptionV3](<https://papers.cool/arxiv/1512.00567>)（I），那么对应的结果就称为“Fréchet Inception Distance”，即FID，这个评测指标首次提出自2017年论文[《GANs Trained by a Two Time\-Scale Update Rule Converge to a Local Nash Equilibrium》](<https://papers.cool/arxiv/1706.08500>)，某种意义上已经属于“上古时代”的产物了。

当然，不管是训练和评测，现在都不一定非要用InceptionV3了，可以用其他更先进的特征模型，比如[SigLIP](<https://papers.cool/arxiv/2303.15343>)，或者多个不同的编码器各算一次Fréchet Distance然后加起来，等等。我们将这些做法统一称为“FD Loss”。

## 相关文献

FID虽然看上去复杂，但它并没有什么不可导的运算，所以将它作为Loss是一个很自然的想法。早在几年前，就已经有这方面的尝试，如[《Image Generation Via Minimizing Fréchet Distance in Discriminator Feature Space》](<https://papers.cool/arxiv/2003.11774>)和[《Backpropagating through Fréchet Inception Distance》](<https://papers.cool/arxiv/2009.14075>)。

然而，早年的尝试并没有跑出很惊艳的结果，本质原因是Batch Size。一般的损失函数是单个样本算单个损失，然后全体样本平均，而FID是全体样本先算出均值、协方差，然后再代入式$\eqref{eq:w-p-q}$做非线性计算。这就导致用小Batch估计出来的FID是有偏的，且无法通过持续训练消除偏差，只能增加Batch Size缓解，导致训练成本比较难受。

“需要跨样本做非线性运算”、“需要大Batch Size”，有没有觉得这些字眼有点熟悉？事实上，视觉中“对比学习”通常也具有这两个特点，由于样本间的非线性运算，我们也不能通过梯度累积来实现增大Batch Size的效果，但这并非完全无法解决，比如[《对比学习可以使用梯度累积吗？》](<https://spaces.ac.cn/archives/8471>)。后面我们会看到，其实FID解决Batch Size问题的方法也是类似的。

此外，从“用预训练好的模型去抽取特征来构建损失函数”这个角度看，还有一个相关工作是[Perceptual Loss](<https://papers.cool/arxiv/1603.08155>)，但这个是作为样本的重构Loss使用的，通常用于VAE等模型的训练，不涉及到跨样本的统计运算，因此没什么计算上的困难。

## 梯度计算

现在让我们一步步推导，看FD作为Loss究竟会遇到什么困难。首先要解决的是梯度的计算问题。$p$代表真实分布，它的$\boldsymbol{\mu}_p,\boldsymbol{\Sigma}_p$是固定的，我们只需要对$\boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q$求梯度。$\boldsymbol{\mu}_q$的梯度比较简单：
\begin{equation}\nabla_{\boldsymbol{\mu}_q}\mathcal{F} = \nabla_{\boldsymbol{\mu}_q}\Vert \boldsymbol{\mu}_p - \boldsymbol{\mu}_q\Vert^2  = 2(\boldsymbol{\mu}_q - \boldsymbol{\mu}_p) \end{equation}
而$\boldsymbol{\Sigma}_q$的梯度则是
\begin{equation}\nabla_{\boldsymbol{\Sigma}_q}\mathcal{F} = \nabla_{\boldsymbol{\Sigma}_q}\tr(\boldsymbol{\Sigma}_p + \boldsymbol{\Sigma}_q - 2(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2}) = \boldsymbol{I} - 2\nabla_{\boldsymbol{\Sigma}_q}\tr((\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{1/2})\end{equation}
这里我们用的是式$\eqref{eq:w-p-q}$的第二行，它看上去更复杂，但它有一个好处：矩阵$\boldsymbol{S} = \boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2}$是正定对称的，这一点可以用来简化计算。设$\boldsymbol{S}$的奇异值分解也就是特征值分解为$\boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top}$，那么$\boldsymbol{S}^{1/2}=\boldsymbol{U}\boldsymbol{\Lambda}^{1/2}\boldsymbol{U}^{\top}$，于是
\begin{align}\tr(\boldsymbol{S}^{1/2})=&\,\tr(\boldsymbol{\Lambda}^{1/2})=\sqrt{\lambda_1}+\sqrt{\lambda_2}+\cdots+\sqrt{\lambda_d} \\[4pt]
\nabla_{\boldsymbol{S}}\tr(\boldsymbol{S}^{1/2}) =&\, \frac{1}{2}\sum_{i=1}^d\frac{\nabla_{\boldsymbol{S}} \lambda_i}{\sqrt{\lambda_i}} =  \frac{1}{2}\sum_{i=1}^d\frac{\boldsymbol{u}_i\boldsymbol{u}_i^{\top}}{\sqrt{\lambda_i}} = \frac{1}{2}\boldsymbol{U}\boldsymbol{\Lambda}^{-1/2}\boldsymbol{U}^{\top} = \frac{1}{2}\boldsymbol{S}^{-1/2}\end{align}
其中特征值的求导，可以参考[《SVD的导数》](<https://spaces.ac.cn/archives/10878>)。最后的结果跟$\sqrt{x}$的导数是$\frac{1}{2\sqrt{x}}$类似，看上去很符合直觉，但它不是平凡的，如果$\boldsymbol{S}$不是正定对称矩阵，那么通常不成立。最后，由链式法则，我们有
\begin{equation}\nabla_{\boldsymbol{\Sigma}_q} \tr(\boldsymbol{S}^{1/2}) = \boldsymbol{\Sigma}_p^{1/2}[\nabla_{\boldsymbol{S}}\tr(\boldsymbol{S}^{1/2})] \boldsymbol{\Sigma}_p^{1/2} = \frac{1}{2}\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{S}^{-1/2}\boldsymbol{\Sigma}_p^{1/2} \end{equation}
合并起来得到
\begin{equation}\nabla_{\boldsymbol{\Sigma}_q}\mathcal{W}_2^2[p,q] = \boldsymbol{I} - \boldsymbol{\Sigma}_p^{1/2}(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{-1/2}\boldsymbol{\Sigma}_p^{1/2}\label{eq:Sigma-grad}\end{equation}
这个形式看似复杂，但$\boldsymbol{\Sigma}_p^{1/2}$可以提前计算好，我们只需对正定对称矩阵$\boldsymbol{S} = \boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2}$求平方根和逆平方根，就可以计算FID及其梯度了，这可以通过<code>eigh</code>函数完成，或者用[《矩阵平方根和逆平方根的高效计算》](<https://spaces.ac.cn/archives/11158>)、[《矩阵r次方根和逆r次方根的高效计算》](<https://spaces.ac.cn/archives/11175>)介绍的Newton\-Schulz迭代方案。

## 超大批次

引入记号
\begin{equation}\begin{gathered}
\boldsymbol{\mu}_p = \mathbb{E}[\boldsymbol{z}_p], \qquad \boldsymbol{V}_p = \mathbb{E}[\boldsymbol{z}_p \boldsymbol{z}_p^{\top}], \qquad \boldsymbol{z}_p = \phi(\boldsymbol{x}_p),\qquad \boldsymbol{x}_p\sim p \\[4pt]
\boldsymbol{\mu}_q = \mathbb{E}[\boldsymbol{z}_q], \qquad \boldsymbol{V}_q = \mathbb{E}[\boldsymbol{z}_q \boldsymbol{z}_q^{\top}], \qquad \boldsymbol{z}_q = \phi(\boldsymbol{x}_q),\qquad \boldsymbol{x}_q\sim q
\end{gathered}\end{equation}
那么
\begin{equation}\boldsymbol{\Sigma}_p = \boldsymbol{V}_p - \boldsymbol{\mu}_p \boldsymbol{\mu}_p^{\top},\qquad\boldsymbol{\Sigma}_q = \boldsymbol{V}_q - \boldsymbol{\mu}_q \boldsymbol{\mu}_q^{\top}\end{equation}
注意$\boldsymbol{z}=\phi(\boldsymbol{x})$通常都有数千维度（InceptionV3是2048），所以为了使得估计准确，通常需要数万样本。真实分布是固定的，它的$\boldsymbol{\mu}_p,\boldsymbol{\Sigma}_p$可以提前算好，问题不大；但生成分布是实时变化的，如果每步都用数万样本算，那么意味着Batch Size要达到数万，这在很多情况下都是相当昂贵的。

另一方面，从梯度公式$\eqref{eq:Sigma-grad}$我们也可以看出大Batch的必要性，如果Batch Size很小，那么估算出来的$\boldsymbol{V}_q$都还不满秩，从而$\boldsymbol{\Sigma}_q$也不满秩，这时候$(\boldsymbol{\Sigma}_p^{1/2}\boldsymbol{\Sigma}_q\boldsymbol{\Sigma}_p^{1/2})^{-1/2}$的求逆便无从说起（会面临$0^{-1/2}$）。因此，FD作为Loss，对训练的Batch Size提出了要求，这应该是实践中最核心的困难。

受限于算力，我们只能设法用小Batch Size模拟出大Batch Size的效果，这跟“对比学习\+梯度累积”的需求类似。

## 等效损失

假设Batch Size为$B$时，对应的$\boldsymbol{\mu}_q,\boldsymbol{V}_q$才是足够准确的，但我们每次只能跑一个小Batch Size $b$，所以一共需要跑$k=B/b$次来模拟大Batch Size的效果，每次产生的结果为$\tilde{\boldsymbol{\mu}}_q^{(1)},\tilde{\boldsymbol{V}}_q^{(1)}$、$\tilde{\boldsymbol{\mu}}_q^{(2)},\tilde{\boldsymbol{V}}_q^{(2)}$、\.\.\.、$\tilde{\boldsymbol{\mu}}_q^{(k)},\tilde{\boldsymbol{V}}_q^{(k)}$，那么有关系
\begin{equation}\boldsymbol{\mu}_q = \frac{1}{k}\sum_{i=1}^k\tilde{\boldsymbol{\mu}}_q^{(i)},\qquad \boldsymbol{V}_q = \frac{1}{k}\sum_{i=1}^k\tilde{\boldsymbol{V}}_q^{(i)}\end{equation}
我们想要寻找一个理想的等效损失，使得总梯度等效于每个小Batch梯度的和，这就达到了无偏估计的效果。为此，对式$\eqref{eq:w-p-q}$两端求微分得
\begin{equation}\begin{aligned}
d\mathcal{F}(\boldsymbol{\mu}_q,\boldsymbol{V}_q) =&\, \langle\nabla_{\boldsymbol{\mu}_q}\mathcal{F}, d\boldsymbol{\mu}_q \rangle + \langle\nabla_{\boldsymbol{V}_q}\mathcal{F}, d\boldsymbol{V}_q \rangle_F \\
=&\, \sum_{i=1}^k \left[\langle\nabla_{\boldsymbol{\mu}_q}\mathcal{F}, d\tilde{\boldsymbol{\mu}}_q^{(i)}/k \rangle + \langle\nabla_{\boldsymbol{V}_q}\mathcal{F}, d\tilde{\boldsymbol{V}}_q^{(i)}/k \rangle_F\right] \\
=&\, d\sum_{i=1}^k \mathcal{F}(\color{skyblue}{[}\boldsymbol{\mu}_q - \tilde{\boldsymbol{\mu}}_q^{(i)}/k\color{skyblue}{]_{sg}} + \tilde{\boldsymbol{\mu}}_q^{(i)}/k,\color{skyblue}{[}\boldsymbol{V}_q - \tilde{\boldsymbol{V}}_q^{(i)}/k\color{skyblue}{]_{sg}} + \tilde{\boldsymbol{V}}_q^{(i)}/k) \\
\end{aligned}\end{equation}
这个等式的意思是，我们可以逐次小批量地前向计算得到$\tilde{\boldsymbol{\mu}}_q^{(i)},\tilde{\boldsymbol{V}}_q^{(i)}$，将它们平均得到足够准确的$\boldsymbol{\mu}_q,\boldsymbol{V}_q$，然后我们再逐个批次按照损失
\begin{equation}\mathcal{F}_i = \mathcal{F}(\color{skyblue}{[}\boldsymbol{\mu}_q - \tilde{\boldsymbol{\mu}}_q^{(i)}/k\color{skyblue}{]_{sg}} + \tilde{\boldsymbol{\mu}}_q^{(i)}/k,\color{skyblue}{[}\boldsymbol{V}_q - \tilde{\boldsymbol{V}}_q^{(i)}/k\color{skyblue}{]_{sg}} + \tilde{\boldsymbol{V}}_q^{(i)}/k)\label{eq:Fi}\end{equation}
来正常求梯度，最后将它们的梯度累加起来，这个梯度就是等效于Batch Size $B$的梯度，其中$\color{skyblue}{[\cdot]_{sg}}$是stop gradient算子。当然我们也可以考虑不累加梯度，而是每一步梯度都执行一次更新，并相应地调小一点学习率，那么效果是类似的。

## 历史来凑

上述的方案虽然理论上可行，但由于要$k$步前向才能算出一个准确的$\boldsymbol{\mu}_q,\boldsymbol{V}_q$，然后才反过来算每一步的梯度，整个流程显得不大“流畅”。这里的瓶颈，就是我们必须要知道全局的$\boldsymbol{\mu}_q,\boldsymbol{V}_q$，才能设法求一个无偏的局部梯度。

一个自然的想法是，$\boldsymbol{\mu}_q,\boldsymbol{V}_q$能否搞点近似呢？考虑到学习率比较小，参数的更新是缓慢的，那么$\boldsymbol{\mu}_q,\boldsymbol{V}_q$的变化应该也是缓慢的，引入当前批数据后，新的$\boldsymbol{\mu}_q,\boldsymbol{V}_q$应当只是在旧的基础上做一些微调，我们考虑用滑动平均（EMA）来近似这个操作
\begin{equation}\boldsymbol{\mu}_q^{(t)} = \beta \boldsymbol{\mu}_q^{(t-1)} + (1-\beta) \tilde{\boldsymbol{\mu}}_q^{(t)},\qquad \boldsymbol{V}_q^{(t)} = \beta \boldsymbol{V}_q^{(t-1)} + (1-\beta) \tilde{\boldsymbol{V}}_q^{(t)}\end{equation}
这大致上维护了一个$\mathcal{O}(1/(1-\beta))$大小的平均窗口，约等于将$\boldsymbol{\mu}_q,\boldsymbol{V}_q$的统计Batch Size扩大到了$\mathcal{O}(1/(1-\beta))$倍。这样一来，每一步我们就可以按照如下损失求梯度来更新
\begin{equation}\mathcal{F}_t = \mathcal{F}(\underbrace{\beta \color{skyblue}{[}\boldsymbol{\mu}_q^{(t-1)}\color{skyblue}{]_{sg}} + (1-\beta) \tilde{\boldsymbol{\mu}}_q^{(t)}}_{\boldsymbol{\mu}_q^{(t)}},\underbrace{\beta \color{skyblue}{[}\boldsymbol{V}_q^{(t-1)}\color{skyblue}{]_{sg}} + (1-\beta) \tilde{\boldsymbol{V}}_q^{(t)}}_{\boldsymbol{V}_q^{(t)}})\end{equation}
额外代价是要缓存$\boldsymbol{\mu}_q,\boldsymbol{V}_q$，成本很小。这种“Batch Size不够，历史来凑”的操作，其实也是“[流式幂迭代](<https://spaces.ac.cn/search/%E6%B5%81%E5%BC%8F%E5%B9%82%E8%BF%AD%E4%BB%A3/>)”的“流式”思想的体现。此外，论文还讨论了队列做法，它维护$k$个历史Batch的队列，按式$\eqref{eq:Fi}$融入当前Batch来计算梯度，并剔除最久远的Batch，这个做法其实比较朴素，占用空间也远比EMA大，实测效果还没有EMA好。

## 实验赏析

论文的实验主要集中在生成模型的后训练，旨在通过FD Loss训练，改良原本的单步生成模型的效果，或者将原本的多步生成模型微调成单步生成模型。当混合使用多个不同的编码器去计算FD Loss时，论文使用了损失归一化技巧来平衡不同量级的损失：
\begin{equation}\mathcal{L} = \sum_i \frac{\mathcal{F}[\phi_i]}{\color{skyblue}{[}\mathcal{F}[\phi_i]\color{skyblue}{]_{sg}} + \epsilon}\end{equation}
这个技巧我们在[《多任务学习漫谈（一）：以损失之名》](<https://spaces.ac.cn/archives/8870>)也讨论过。

论文的核心战绩，就是将单步生成的效果（FID）推向了一个全新的高度，并且超越了所有其他的单步和多步生成模型，这一结果看来已经到了天花板。部分图表如下所示：

[![FD loss 改进单步生成效果](<https://spaces.ac.cn/usr/uploads/2026/05/3014248186.png>)](<https://spaces.ac.cn/usr/uploads/2026/05/3014248186.png>)

FD loss 改进单步生成效果

[![FD loss 的综合比较](<https://spaces.ac.cn/usr/uploads/2026/05/3462357209.png>)](<https://spaces.ac.cn/usr/uploads/2026/05/3462357209.png>)

FD loss 的综合比较

## 文章小结

这篇文章主要从理论上分析了将FID作为生成模型损失函数所面临的困难，以及如何从推导过程中引出对应的克服困难的技巧。
