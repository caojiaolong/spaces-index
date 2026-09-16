# MoE环游记：9、门控归一化之争

> 作者：苏剑林 · 科学空间 · 2026-06-17
>
> 原文：<https://spaces.ac.cn/archives/11782>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

追溯一下MoE历史，我们都能发现，早些年MoE的Router在作为Gate去乘到Expert上时，基本都是用Softmax激活，直到现在它仍是MoE的标准形式之一。不过，为了配合[Loss\-Free](<https://spaces.ac.cn/archives/10757>)负载均衡，DeepSeek将激活函数改成了Sigmoid，并证明了这也是一个颇有竞争力的方案，这引发了大家对Router形式更深入的思考和尝试。

即便在Softmax内讨论，也有着两种略微不同做法：是先Softmax再选Top\-$k$，还是先选Top\-$k$再Softmax呢？后者也可以理解为在选出Top\-$k$后再做一次归一化，即Re\-Norm。那么，Gate的激活函数是否要归一化，要的话是归一化后选Top\-$k$还是Re\-Norm，这便是本文要讨论的主题。

## 问题描述

我们知道，MoE的一般形式是
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i \label{eq:moe-1}\end{equation}
这里的$\boldsymbol{\rho}$实际上扮演着两个角色：当它用于选Top\-$k$个Expert时，角色是Router；当它用于乘到Expert上时，角色是Gate。从MoE的设计来说，$\boldsymbol{\rho}$的核心角色显然是Router，Gate的作用是在训练时为它提供梯度。

我们要讨论的问题，也可以理解为如何更科学地构造$\boldsymbol{\rho}=(\rho_1, \rho_2, \cdots, \rho_n)$，使得Router能获得更好的梯度。在很长的时间内，标准答案一直是Softmax，即
\begin{equation}\rho_i = \frac{e^{s_i}}{\sum_{j=1}^n e^{s_j}} \end{equation}
其中$\boldsymbol{s}=(s_1, s_2, \cdots, s_n)$是由线性层直接投影出来的Logits。然而，这个答案虽然“标准”，但笔者并没有找到什么解释，似乎大家都直接接受了它，并沿用下去，这使得笔者一度对MoE的训练机理非常疑惑。

## 其他选择

开头说了，DeepSeek在Loss\-Free负载均衡尝试了Sigmoid激活，后面还用到了[DeepSeek\-V3](<https://papers.cool/arxiv/2412.19437>)中，它的成功表明非Softmax激活也能起到不错的效果。这启发大家尝试更一般的做法，比如[ReMoE](<https://papers.cool/arxiv/2412.14711>)就使用了ReLU激活，而我们在[《MoE环游记：1、从几何意义出发》](<https://spaces.ac.cn/archives/10699>)的几何视角，则允许用任意非负的激活函数。

此外，在MoE的形式上，还有一个Re\-Norm的选择，即将$\eqref{eq:moe-1}$改为
\begin{equation}\boldsymbol{y} = \frac{\sum\limits_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i}{\sum\limits_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i} \label{eq:moe-2}\end{equation}
也就是对选出的Top\-$k$ $\rho_i$重新执行一次归一化。对于Softmax来说，这相当于用$\boldsymbol{s}$选出Top\-$k$后，将没选中的设为$-\infty$，然后再Softmax。Re\-Norm的好处是让前向计算的数值更稳定，但要注意，用Re\-Norm时$k$至少要大于1，否则$\boldsymbol{\rho}$将完全没有梯度，从而无法训练。

综合当前各方的实践来看，这些MoE变体效果都大差不差，并没有哪一个显著占优。既然实践区分不出胜负，我们就从理论上探讨一下，究竟是哪种形式更为科学。

## 设计原理

我们的目标，是找到一个更贴近本质的第一性原理，并以它来推导出当前MoE的门控机制。

所以，第一个问题自然是这个“原理”是什么。简单起见，先考虑$k=1$。我们知道，MoE最重要的特点是Sparse，先通过一个Router来确定要激活的Expert，然后只计算这些Expert，从而实现增大参数量的同时控制计算量。如果单纯是这个想法，那么朴素的模型应该是
\begin{equation}\newcommand{argmax}{\mathop{\text{argmax}}}\boldsymbol{f}\left(\boldsymbol{e}_{\argmax\boldsymbol{\rho}}\right)\end{equation}
即从Router $\boldsymbol{\rho}$中挑出分数最大者，激活对应的Expert。这个形式对推理来说完全没问题，但在训练时，Router会收不到任何梯度，从而无法得到更新，所以我们须设法给Router设计梯度。怎么给Router设计梯度呢？要回答这个问题，我们首先要想清楚：我们需要怎么样的Router。

由于只能激活1个Expert，所以我们自然是希望这个Expert是效果最好的那个Expert，如果用$\ell$表示损失函数，那么我们的期望可以写成
\begin{equation}\newcommand{argmin}{\mathop{\text{argmin}}}\argmax \boldsymbol{\rho} = \argmin\, [\ell(\boldsymbol{e}_1),\ell(\boldsymbol{e}_2),\cdots,\ell(\boldsymbol{e}_n)]\label{eq:target}\end{equation}
这便是我们要寻找的设计原理。

## 目标转化

不过，目标$\eqref{eq:target}$还不是能直接用来训练的损失函数，它还需进一步转化。为此，我们构造两个分布，第一个是基于损失函数构建的目标分布$\boldsymbol{q}=(q_1,q_2,\cdots,q_n)$，定义为
\begin{equation}q_i = \frac{e^{-\ell(\boldsymbol{e}_i)/\tau}}{\sum_{j=1}^n e^{-\ell(\boldsymbol{e}_j)/\tau}}\end{equation}
这个分布跟Router无关，对Router的学习来说它属于“目标分布”。第二个分布是基于$\boldsymbol{\rho}$构建出的预测分布$\boldsymbol{p}$，这里有很多可能性，比如$\boldsymbol{\rho}$本身就是分布$\boldsymbol{p}$（如果$\boldsymbol{\rho}$已经归一化），也可能$\boldsymbol{p}$是$\boldsymbol{\rho}$的Softmax（此时$\boldsymbol{\rho}$是Logits），或者Softmax以外的归一化手段。总之，$\boldsymbol{p}$是Router的某种概率分布表示，设它的参数为$\boldsymbol{\theta}$。

我们将目标$\eqref{eq:target}$，转化为拉近$\boldsymbol{p}$与$\boldsymbol{q}$的距离，从而为$\boldsymbol{\theta}$提供梯度。为此，我们考虑最小化KL散度
\begin{equation}KL(\boldsymbol{p}\Vert \boldsymbol{q}) = \sum_{i=1}^n p_i \log \frac{p_i}{q_i}\end{equation}
稍加变形得
\begin{equation}KL(\boldsymbol{p}\Vert \boldsymbol{q}) = - \mathcal{H}(\boldsymbol{p}) + \frac{1}{\tau}\sum_{i=1}^n p_i \ell(\boldsymbol{e}_i) - \log \sum_{i=1}^n e^{-\ell(\boldsymbol{e}_i)/\tau}\end{equation}
可以看到，这个目标分三项，第一项是负熵$-\mathcal{H}(\boldsymbol{p})$，最小化它意味着最大熵，这实际上鼓励模型进行充分的尝试，可以认为负载均衡已经承担了类似角色，所以先不管它；第三项跟$\boldsymbol{p}$无关，也就是跟$\boldsymbol{\theta}$无关，所以等效的损失函数是$\mathcal{L} = \sum_{i=1}^n p_i \ell(\boldsymbol{e}_i)$

## 直通估计

对等效损失求梯度得
\begin{equation}\nabla_{\boldsymbol{\theta}}\mathcal{L} = \sum_{i=1}^n \nabla_{\boldsymbol{\theta}} p_i \cdot \ell(\boldsymbol{e}_i) = \sum_{i=1}^n p_i \nabla_{\boldsymbol{\theta}} \log p_i \cdot \ell(\boldsymbol{e}_i) = \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(\boldsymbol{e}_i)]\end{equation}
这里的关键是利用$\nabla_{\boldsymbol{\theta}} p_i = p_i \nabla_{\boldsymbol{\theta}} \log p_i$单独分离出了一项$p_i$，这样才能求和转化为期望，进而通过采样来达到MoE的稀疏计算目标。可能有读者已经认出来了，这正是策略梯度中的REINFORCE！（参考[《从采样看优化：可导优化与不可导优化的统一视角》](<https://spaces.ac.cn/archives/7521>)、[《殊途同归的策略梯度与零阶优化》](<https://spaces.ac.cn/archives/7737>)）

REINFORCE的问题是噪声大，直观来看，这是因为它将$p_i$放到了损失函数$\ell$的外边，如果有可能，我们更希望使用$p_i$在$\ell$里边的“重参数”形式。为了推导这样的一个形式，我们利用REINFORCE对减baseline的不变性得
\begin{equation}\begin{aligned}
\mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(\boldsymbol{e}_i)] =&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot (\ell(\boldsymbol{e}_i) - \ell(\boldsymbol{0}))] \\[4pt]
\approx&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \langle\nabla_{\boldsymbol{e}_i} \ell(\boldsymbol{e}_i), \boldsymbol{e}_i - \boldsymbol{0}\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \langle\nabla_{\boldsymbol{e}_i} \ell(\boldsymbol{e}_i), \log p_i \cdot \boldsymbol{e}_i\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \ell((\log p_i + \color{skyblue}{[}1 - \log p_i \color{skyblue}{]_{\text{sg}}}) \cdot\boldsymbol{e}_i)] \\[4pt]
= &\, \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i\sim \boldsymbol{p}} [\ell((\log p_i + \color{skyblue}{[}1 - \log p_i \color{skyblue}{]_{\text{sg}}}) \cdot\boldsymbol{e}_i)] \\
\end{aligned}\end{equation}
其中约等号$\approx$是在$\boldsymbol{e}_i$做了一阶泰勒近似，$\color{skyblue}{[}\cdot\color{skyblue}{]_{\text{sg}}}$代表Stop Gradient。最终我们得到了一个“前向传播用$1$、反向传播用$\log p_i$”的Straight\-Through Estimator（STE）来为Router提供梯度。

## 最终形式

STE虽然能提供一个可行的训练方案，但由于前向传播和反向传播的不一致性，它往往只能得到次优的效果。这时候一个非常神奇的改进是——将每个Expert改为$p_i\boldsymbol{e}_i$！重复上述推导，我们有
\begin{equation}\begin{aligned}
\mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \ell(p_i\boldsymbol{e}_i)] =&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot (\ell(p_i\boldsymbol{e}_i) - \ell(\boldsymbol{0}))] \\[4pt]
\approx&\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i \cdot \langle\nabla_{p_i\boldsymbol{e}_i} \ell(p_i\boldsymbol{e}_i), p_i\boldsymbol{e}_i - \boldsymbol{0}\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \langle\nabla_{p_i \boldsymbol{e}_i} \ell(p_i \boldsymbol{e}_i),  p_i \boldsymbol{e}_i\rangle] \\[4pt]
= &\, \mathbb{E}_{i\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}} \ell(p_i \boldsymbol{e}_i)] \\[4pt]
= &\, \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i\sim \boldsymbol{p}} [\ell(p_i \boldsymbol{e}_i)] \\[4pt]
\end{aligned}\end{equation}
这个变换过程非常精妙，值得细细回味。通过将Expert从$\boldsymbol{e}_i$改为$p_i\boldsymbol{e}_i$，我们消除了Stop Gradient，实现了前后向的一致性，理论上也提高了模型效果的天花板。

现在，我们可以回答开头的问题了：

> 如果我们需要有一个自上而下的概率推导，那么Router作为Gate时应当要归一化，但不应当Re\-Norm。

## 采样与否

一个值得留意的细节是：$\mathbb{E}_{i\sim \boldsymbol{p}}$意味着我们应该从$\boldsymbol{p}$采样，但实际中我们通常都是直接选Top\-$k$，如何理解这个差异呢？

这实际上是多样性和稳定性的权衡。随机采样鼓励模型进行更充分的尝试，但采样会增加梯度方差，引入额外的不稳定性；直接选Top\-$k$比较稳定，但担心模型陷入次优解，甚至导致模型坍缩。不过好在现在各种负载均衡策略都很成熟，一定程度上已经鼓励了模型全方面探索，所以目前依然是选Top\-$k$为主流。

如果想要采样的同时保持稳定性，可以在Top\-$k$的基础上稍加拓展而不是完全放开采样，比如先选出Top\-$k+c$，然后在$k+c$个Expert中随机挑$k$个，或者在$\boldsymbol{p}$的Logits基础上加轻微噪声，然后再选Top\-$k$，这样在增加随机性的同时，不会离原本的Top\-$k$太远，兼顾了稳定性。

## 相关工作

需要澄清的是，本文的推导其实不是新的，它是笔者从刘力源老师的文章[《Sparse Backpropagation for MoE Training》](<https://papers.cool/arxiv/2310.00811>)中提炼、修改而来的，该文章还有一个上篇[《Bridging Discrete and Backpropagation: Straight\-Through and Beyond》](<https://papers.cool/arxiv/2304.08612>)和下篇[《GRIN: GRadient\-INformed MoE》](<https://papers.cool/arxiv/2409.12136>)。

尽管它们已经是23、24年那会的文章了，但想要加深对MoE Router的理解，仍非常建议阅读这三部曲，它们提供了一个统一的概率框架去为各种离散化操作设计梯度。当然，概率框架也有它的局限性，那就是形式化比较严重，操作起来有点“束手束脚”。

比如当$k = 2$时，如果将前面的结果平行推广过来，应该是
\begin{equation}\mathbb{E}_{i,j\sim \boldsymbol{p}} [\nabla_{\boldsymbol{\theta}}\log p_i p_j \cdot \ell(p_i p_j (\boldsymbol{e}_i + \boldsymbol{e}_j))] \approx \nabla_{\boldsymbol{\theta}} \mathbb{E}_{i,j\sim \boldsymbol{p}} [\ell(p_i p_j (\boldsymbol{e}_i + \boldsymbol{e}_j))]\end{equation}
即以联合分布$p_i p_j$和专家对之和$\boldsymbol{e}_i + \boldsymbol{e}_j$为基本单位，将Top\-2转化为Top\-1。然而，我们始终用的MoE，其实是$\ell(p_i \boldsymbol{e}_i + p_j\boldsymbol{e}_j)$的形式，这就不大好寻找一个精确的概率推导。

这时候，一种更“放宽心”的理解方式，或许是直接当它是MaxPooling的类似物，不去强求概率解释，当然也可以选择按照[《MoE环游记：1、从几何意义出发》](<https://spaces.ac.cn/archives/10699>)的几何意义来理解。总的来说，概率框架只是证明了某种方案的可行性，但原则上没有否定其他方案的可行性

## 文章小结

本文尝试从第一性原理出发，探讨MoE中Router与Gate的设计问题，为门控的归一化提供了一个概率解释。
