# 多任务学习漫谈（一）：以损失之名

> 作者：苏剑林 · 科学空间 · 2022-01-18
>
> 原文：<https://spaces.ac.cn/archives/8870>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

能提升模型性能的方法有很多，多任务学习（Multi\-Task Learning）也是其中一种。简单来说，多任务学习是希望将多个相关的任务共同训练，希望不同任务之间能够相互补充和促进，从而获得单任务上更好的效果（准确率、鲁棒性等）。然而，多任务学习并不是所有任务堆起来就能生效那么简单，如何平衡每个任务的训练，使得各个任务都尽量获得有益的提升，依然是值得研究的课题。

最近，笔者机缘巧合之下，也进行了一些多任务学习的尝试，借机也学习了相关内容，在此挑部分结果与大家交流和讨论。

## 加权求和

从损失函数的层面看，多任务学习就是有多个损失函数$\mathcal{L}_1,\mathcal{L}_2,\cdots,\mathcal{L}_n$，一般情况下它们有大量的共享参数、少量的独立参数，而我们的目标是让每个损失函数都尽可能地小。为此，我们引入权重$\alpha_1,\alpha_2,\cdots,\alpha_n\geq 0$，通过加权求和的方式将它转化为如下损失函数的单任务学习
\begin{equation}\mathcal{L} = \sum_{i=1}^n \alpha_i \mathcal{L}_i\label{eq:w-loss}\end{equation}
在这个视角下，多任务学习的主要难点就是如何确定各个$\alpha_i$了。

### 初始状态

按道理，在没有任务先验和偏见的情况下，最自然的选择就是平等对待每个任务，即$a_i=1/n$。然而，事实上每个任务可能有很大差别，比如不同类别数的分类任务混合、分类与回归任务混合、分类与生成任务混合等等，从物理的角度看，每个损失函数的量纲和量级都不一样，直接相加是没有意义的。

如果我们将每个损失函数看成具有不同量纲的物理量，那么从“无量纲化”的思想出发，我们可以用损失函数的初始值倒数作为权重，即
\begin{equation}\mathcal{L} = \sum_{i=1}^n \frac{\mathcal{L}_i}{\mathcal{L}_i^{(\text{init})}}\label{eq:init}\end{equation}
其中$\mathcal{L}_i^{(\text{init})}$表示任务$i$的初始损失值。该式关于每个$\mathcal{L}_i$是“齐次”的，所以它的一个明显优点是缩放不变性，即如果让任务$i$的损失乘上一个常数，那么结果不会变化。此外，由于每个损失都除以了自身的初始值，较大的损失会缩小，较小的损失会放大，从而使得每个损失能够大致得到平衡。

那么，怎么估计$\mathcal{L}_i^{(\text{init})}$呢？最直接的方法当然是直接拿几个batch的数据来估算一下。除此之外，我们可以基于一些假设得到一个理论值。比如，在主流的初始化之下，我们可以认为初始模型（加激活函数之前）的输出是一个零向量，如果加上softmax则是均匀分布，那么对于一个“$K$分类\+交叉熵”问题，它的初始损失就是$\log K$；对于“回归\+L2损失”问题，则可以用零向量来估计初始损失，即$\mathbb{E}_{y\sim \mathcal{D}}[\Vert y-0\Vert^2] = \mathbb{E}_{y\sim \mathcal{D}}[\Vert y\Vert^2]$，$\mathcal{D}$是训练集的全体标签。

### 先验状态

用初始损失的一个问题是初始状态不一定能很好地反应当前任务的学习难度，更好的方案应该是将“初始状态”改为“先验状态”：
\begin{equation}\mathcal{L} = \sum_{i=1}^n \frac{\mathcal{L}_i}{\mathcal{L}_i^{(\text{prior})}}\label{eq:prior}\end{equation}
比如，如果$K$分类中每个类的频率分别是$[p_1,p_2,\dots,p_K]$（先验分布），那么虽然初始状态的预测分布为均匀分布，但我们可以合理地认为模型可以很容易学会将每个样本的结果都预测为$[p_1,p_2,\dots,p_K]$，此时模型的损失为熵
\begin{equation}\mathcal{L}_i^{(\text{prior})}=\mathcal{H} = -\sum_{i=1}^K p_i\log p_i\end{equation}
某种意义上来说，“先验分布”比“初始分布”更能体现出“初始”的本质，它是“就算模型啥都学不会，也知道按照先验分布来随机出结果”的体现，所以此时的损失值更能代表当前任务的初始难度，因此用$\mathcal{L}_i^{(\text{prior})}$代替$\mathcal{L}_i^{(\text{init})}$应该更加合理；类似地，对于“回归\+L2损失”问题，它的先验结果应该是全体标签的期望$\mu = \mathbb{E}_{y\sim \mathcal{D}}[y]$，所以我们用$\mathcal{L}_i^{(\text{prior})}=\mathbb{E}_{y\sim \mathcal{D}}[\Vert y-\mu\Vert^2]$代替$\mathcal{L}_i^{(\text{init})}=\mathbb{E}_{y\sim \mathcal{D}}[\Vert y\Vert^2]$，有望取得更合理的结果。

## 动态调节

不管是用初始状态的式$\eqref{eq:init}$还是先验状态的式$\eqref{eq:prior}$，它们的任务权重在确定之后就保持不变了，并且它们确定权重的方法不依赖于学习过程。然而，尽管我们可以通过先验分布等信息简单感知一下学习难度，但究竟有多难其实要真正去学习才知道，所以更合理的方案应该是根据训练进程动态地调整权重。

### 实时状态

纵观前文，式$\eqref{eq:init}$和式$\eqref{eq:prior}$的核心思想都是用损失值的倒数来作为任务权重，那么能不能干脆用“实时”的损失值倒数来实现动态调整权重？即
\begin{equation}\mathcal{L} = \sum_{i=1}^n \frac{\mathcal{L}_i}{\mathcal{L}_i^{(\text{sg})}}\label{eq:sg}\end{equation}
这里的$\mathcal{L}_i^{(\text{sg})}$是$\text{stop_gradient}(\mathcal{L}_i)$的简写。在这个方案中，每个任务的损失函数都被调整恒为1，所以不管是量纲还是量级上都是一致的。由于$\text{stop_gradient}$算子的存在，虽然损失恒为1，但梯度并非恒为0：
\begin{equation}\nabla_{\theta}\left(\frac{\mathcal{L}_i}{\mathcal{L}_i^{(\text{sg})}}\right) = \frac{\nabla_{\theta}\mathcal{L}_i}{\mathcal{L}_i^{(\text{sg})}} = \frac{\nabla_{\theta}\mathcal{L}_i}{\mathcal{L}_i}\label{eq:sg-grad}\end{equation}
简单来说就是某个函数被$\text{stop_gradient}$算子包住后，就变成了一个新函数，其值与原来的函数恒等，但是它的导函数强制设为了0，所以最终结果就是以动态权重$1/\mathcal{L}_i$来实时调整了梯度的比例。很多“民间实验”表明，式$\eqref{eq:sg}$确实在多数情况下都可以作为一个相当不错的baseline。

### 等价梯度

我们可以从另一个角度来看该方案。从式$\eqref{eq:sg-grad}$我们可以得到
\begin{equation}\nabla_{\theta}\left(\frac{\mathcal{L}_i}{\mathcal{L}_i^{(\text{sg})}}\right) = \frac{\nabla_{\theta}\mathcal{L}_i}{\mathcal{L}_i} = \nabla_{\theta} \log \mathcal{L}_i\end{equation}
因此从梯度上看，式$\eqref{eq:sg}$与$\mathcal{L} = \sum\limits_{i=1}^n \log \mathcal{L}_i$没有实质区别，而我们进一步有
\begin{equation}\mathcal{L} = \sum_{i=1}^n \log \mathcal{L}_i = n\log \sqrt[n]{\prod_{i=1}^n\mathcal{L}_i}\end{equation}
由于$\log$是单调递增的，所以式$\eqref{eq:sg}$与下式在梯度方向上是一致：
\begin{equation}\mathcal{L} = \sqrt[n]{\prod_{i=1}^n\mathcal{L}_i}\end{equation}

### 广义平均

显然，上式正是$\mathcal{L}_1,\mathcal{L}_2,\cdots,\mathcal{L}_n$的“几何平均”，而如果我们约定$a_i$恒等于$1/n$，那么原始的式$\eqref{eq:w-loss}$就是$\mathcal{L}_1,\mathcal{L}_2,\cdots,\mathcal{L}_n$的“代数平均”。也就是说，我们发现这一系列的推导其实隐藏了从代数平均到几何平均的转变，这启发我们或许可以考虑“广义平均”：
\begin{equation}\mathcal{L}(\gamma) = \sqrt[\gamma]{\frac{1}{n}\sum_{i=1}^n\mathcal{L}_i^{\gamma}}\end{equation}
也就是将每个损失函数算$\gamma$次方后再平均最后再开$\gamma$次方，这里的$\gamma$可以是任意实数，代数平均对应$\gamma=1$，而几何平均对应$\gamma=0$（需要取极限）。可以证明，$\mathcal{L}(\gamma)$是关于$\gamma$的单调递增函数，并且有
\begin{equation}\min(\mathcal{L}_1,\cdots,\mathcal{L}_n)=\lim_{\gamma\to-\infty} \mathcal{L}(\gamma) \leq\cdots\leq  \mathcal{L}(\gamma) \leq\cdots\leq \lim_{\gamma\to+\infty} \mathcal{L}(\gamma)=\max(\mathcal{L}_1,\cdots,\mathcal{L}_n)\end{equation}
这就意味着，当$\gamma$增大时，模型愈发关心损失中的最大值，反之则更关心损失中的最小值。这样一来，虽然依然存在超参数$\gamma$要调整，但是相比于原始的式$\eqref{eq:w-loss}$，超参数的个数已经从$n$个变为只有1个，简化了调参过程。

## 平移不变

重新回顾式$\eqref{eq:init}$、式$\eqref{eq:prior}$和式$\eqref{eq:sg}$，它们都是通过每个任务损失除以自身的某个状态来调节权重，并且获得了缩放不变性。然而，尽管它们都具备了缩放不变性，但却失去了更基本的“平移不变性”，也就是说，如果每个损失都加上一个常数，$\eqref{eq:init}$、式$\eqref{eq:prior}$和式$\eqref{eq:sg}$的梯度方向是有可能改变的，这对于优化来说并不是一个好消息，因为原则上来说常数没有带来任何有意义的信息，优化结果不应该随之改变。

### 理想目标

一方面，我们用损失函数（的某个状态）的倒数作为当前任务的权重，但损失函数的导数不具备平移不变性；另一方面，损失函数可以理解为当前模型与目标状态的距离，而梯度下降本质上是在寻找梯度为0的点，所以梯度的模长其实也能起到类似作用，因此我们可以用梯度的模长来替换掉损失函数，从而将式$\eqref{eq:sg}$变成
\begin{equation}\mathcal{L} = \sum_{i=1}^n \frac{\mathcal{L}_i}{\Vert\nabla_{\theta}\mathcal{L}_i\Vert^{(\text{sg})}}\label{eq:grad}\end{equation}
跟损失函数的一个明显区别是，梯度模长显然具备平移不变性，并且分子分母关于$\mathcal{L}_i$依然是齐次的，所以上式还保留了缩放不变性。因此，这是一个能同时具备平移和缩放不变性的理想目标。

### 梯度归一

对式$\eqref{eq:grad}$求梯度，我们得到
\begin{equation}\nabla_{\theta}\mathcal{L} = \sum_{i=1}^n \frac{\nabla_{\theta}\mathcal{L}_i}{\Vert\nabla_{\theta}\mathcal{L}_i\Vert}\label{eq:grad-norm}\end{equation}
可以看到，式$\eqref{eq:grad}$本质上是将每个任务损失的梯度进行归一化后再把梯度累加起来。它同时也告诉了我们一种实现方案，即可以让每个任务依次训练，每次只训练一个任务，然后将每个任务的梯度归一化后累积起来再更新，这样就免除了在定义损失函数的时候就要算梯度的麻烦了。

关于梯度归一化，笔者能找到相关工作是[《GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks》](<https://papers.cool/arxiv/1711.02257>)，它本质上是式$\eqref{eq:init}$和式$\eqref{eq:grad-norm}$的混合，里边也包含了对梯度模长重新标定的思想，但却要通过额外的优化来确定任务权重，个人认为显得繁琐和冗余了。

## 本文小结

在损失函数的视角下，多任务学习的关键问题是如何调节每个任务的权重来平衡各自的损失，本文从缩放不变和平移不变两个角度介绍了一些参考做法，并补充了“广义平均”的概念，将多个任务的权重调节转化为单个参数的调节问题，可以简化调参难度。
