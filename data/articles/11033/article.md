# 线性注意力简史：从模仿、创新到反哺

> 作者：苏剑林 · 科学空间 · 2025-06-20
>
> 原文：<https://spaces.ac.cn/archives/11033>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在中文圈，本站应该算是比较早关注线性Attention的了，在2020年写首篇相关博客[《线性Attention的探索：Attention必须有个Softmax吗？》](<https://spaces.ac.cn/archives/7546>)时，大家主要讨论的还是BERT相关的Softmax Attention。事后来看，在BERT时代考虑线性Attention并不是太明智，因为当时训练长度比较短，且模型主要还是Encoder，用线性Attention来做基本没有优势。对此，笔者也曾撰文[《线性Transformer应该不是你要等的那个模型》](<https://spaces.ac.cn/archives/8610>)表达这一观点。

直到ChatGPT的出世，倒逼大家都去做Decoder\-only的生成式模型，这跟线性Attention的RNN形式高度契合。同时，追求更长的训练长度也使得Softmax Attention的二次复杂度瓶颈愈发明显。在这样的新背景下，线性Attention越来越体现出竞争力，甚至出现了“反哺”Softmax Attention的迹象。

## 平方复杂度

首先引入一些记号：
\begin{equation}\begin{gathered}
\boldsymbol{q}_i,\boldsymbol{k}_i,\boldsymbol{v}_i,\boldsymbol{o}_i \in \mathbb{R}^{d\times 1} \\[6pt]
\boldsymbol{Q}=[\boldsymbol{q}_1,\boldsymbol{q}_2,\cdots,\boldsymbol{q}_n]^{\top}\in\mathbb{R}^{n\times d} \\[6pt]
\boldsymbol{K}=[\boldsymbol{k}_1,\boldsymbol{k}_2,\cdots,\boldsymbol{k}_n]^{\top}\in\mathbb{R}^{n\times d} \\[6pt]
\boldsymbol{V}=[\boldsymbol{v}_1,\boldsymbol{v}_2,\cdots,\boldsymbol{v}_n]^{\top}\in\mathbb{R}^{n\times d} \\[6pt]
\boldsymbol{O}=[\boldsymbol{o}_1,\boldsymbol{o}_2,\cdots,\boldsymbol{o}_n]^{\top}\in\mathbb{R}^{n\times d} \\[6pt]
\end{gathered}\end{equation}
一个Attention模型，本质上是一个$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}\to \boldsymbol{O}$的映射。本文主要关心Causal场景，这意味着$\boldsymbol{o}_t$至多跟$\boldsymbol{Q}_{[:t]},\boldsymbol{K}_{[:t]},\boldsymbol{V}_{[:t]}$相关。原则上，$\boldsymbol{Q},\boldsymbol{K}$的$d$与$\boldsymbol{V},\boldsymbol{O}$的$d$可以不一致，比如[GAU](<https://spaces.ac.cn/archives/8934>)和[MLA](<https://spaces.ac.cn/archives/10091>)便是如此，但将它们简化成同一个并不改变问题本质。

标准的Softmax Attention，通常是指[《Attention is All You Need》](<https://spaces.ac.cn/archives/4765>)所提的Attention机制：
\begin{equation}\boldsymbol{O} = \mathop{\text{softmax}}(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log \boldsymbol{M})\boldsymbol{V}\end{equation}
这里省略了缩放因子$1/\sqrt{d}$，因为它总可以吸收到$\boldsymbol{Q},\boldsymbol{K}$里边，$\mathop{\text{softmax}}$是对第二个维度进行指数归一化，而$\boldsymbol{M}\in\mathbb{R}^{n\times n}$是一个下三角阵，称为掩码矩阵，定义为
\begin{equation}M_{i,j} = \left\{\begin{aligned} &1, &i \geq j \\ &0, &i < j\end{aligned}\right.\end{equation}
$\log\boldsymbol{M}$是指对$\boldsymbol{M}$的分量逐一取$\log$，其中$\log 0 = -\infty$。Softmax Attention用分量形式写出来则是
\begin{equation}\boldsymbol{o}_t = \frac{\sum_{j=1}^t \exp(\boldsymbol{q}_t^{\top}\boldsymbol{k}_j) \boldsymbol{v}_j}{\sum_{j=1}^t \exp(\boldsymbol{q}_t^{\top}\boldsymbol{k}_j) }\end{equation}
其中分母的作用主要是保持数值稳定性，另外就是如果我们给$\boldsymbol{O}$加上RMSNorm，那么分母也会自动消去，所以Softmax Attention的核心是分子部分，即
\begin{equation}\boldsymbol{O} = \exp(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log \boldsymbol{M})\boldsymbol{V} = (\exp(\boldsymbol{Q}\boldsymbol{K}^{\top})\odot \boldsymbol{M})\boldsymbol{V}\end{equation}
其中$\odot$是Hadamard积，$\exp$是逐分量取指数。不难看出，分母其实就是将$\boldsymbol{V}$换成一个$n\times 1$的全1矩阵，如果有需要，我们再补上即可。Softmax Attention的标准实现需要把$n\times n$的矩阵$\exp(\boldsymbol{Q}\boldsymbol{K}^{\top})$算出来，所以空间和时间复杂度都正比于$n^2$。[Flash Attention](<https://papers.cool/arxiv/2205.14135>)的出现降低了空间需求，但平方的时间复杂度依然无法避免。

## 最初的模样

线性Attention最早的思路主要是模仿和近似Softmax Attention，其中最简单的方案是直接去掉$\exp$，得到
\begin{equation}\boldsymbol{O} = (\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{M})\boldsymbol{V}\label{eq:linear-attn}\end{equation}
简单起见，我们约定矩阵乘法的优先级高于Hadamard积，这样可以省掉一组括号。为什么这个形式是“线性”Attention的呢？为了快速理解这一点，我们不妨先考虑去掉$\odot \boldsymbol{M}$的非Causal版，此时成立$\boldsymbol{O} = (\boldsymbol{Q}\boldsymbol{K}^{\top})\boldsymbol{V} = \boldsymbol{Q}(\boldsymbol{K}^{\top}\boldsymbol{V})$，注意计算$\boldsymbol{K}^{\top}\boldsymbol{V}$的复杂度是$\mathcal{O}(nd^2)$，结果是$d\times d$矩阵，然后跟$\boldsymbol{Q}$相乘复杂度也是$\mathcal{O}(nd^2)$，所以它复杂度是线性依赖于$n$。

至于Causal版$\eqref{eq:linear-attn}$，我们可以从分量形式理解，写出：
\begin{equation}\boldsymbol{o}_t = \sum_{j=1}^t \boldsymbol{v}_j (\boldsymbol{k}_j^{\top} \boldsymbol{q}_t) = \sum_{j=1}^t (\boldsymbol{v}_j \boldsymbol{k}_j^{\top}) \boldsymbol{q}_t = \left(\sum_{j=1}^t \boldsymbol{v}_j \boldsymbol{k}_j^{\top}\right) \boldsymbol{q}_t\end{equation}
如果我们记括号部分为$\boldsymbol{S}_t$，那么有
\begin{equation}\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t, \qquad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\label{eq:linear-attn-rnn}\end{equation}
由此可见，Causal形式的Attention可以写成一个以$\boldsymbol{S}_t$为State的线性RNN，因此每一步的复杂度是常数，总的复杂度正比于序列长度$n$。注意这里出现了“线性RNN”，它是更广义的概念，线性Attention属于线性RNN的一种，线性RNN也单独发展过一段时间，比如之前介绍过的[LRU](<https://spaces.ac.cn/archives/9554>)、[SSM](<https://spaces.ac.cn/tag/ssm/>)等，但最近比较有竞争力的线性架构都具有线性Attention的形式。

早年的线性Attention还有一些非常明显的模仿Softmax Attention的特点，比如会给式$\eqref{eq:linear-attn}$加入分母来归一化，而为了归一化，那么$\boldsymbol{k}_j^{\top} \boldsymbol{q}_t$就必须非负，于是又给$\boldsymbol{Q},\boldsymbol{K}$加上了非负的激活函数，以[Performer](<https://spaces.ac.cn/archives/7921>)、[RFA](<https://papers.cool/arxiv/2103.02143>)为代表的一系列工作，更是以近似$\exp(\boldsymbol{Q}\boldsymbol{K}^{\top})$为出发点来构建模型。

然而，后来的研究如[《The Devil in Linear Transformer》](<https://papers.cool/arxiv/2210.10340>)发现，在序列长度维度归一化并不能完全避免数值不稳定性，倒不如直接事后归一化，如
\begin{equation}\boldsymbol{O} = \mathop{\text{RMSNorm}}((\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{M})\boldsymbol{V})\end{equation}
而既然不用归一化，那么给$\boldsymbol{Q},\boldsymbol{K}$加非负的激活函数来保证$\boldsymbol{k}_j^{\top} \boldsymbol{q}_t$非负就非必须了。那给$\boldsymbol{Q},\boldsymbol{K}$加（不一定非负的）激活函数还有意义吗？笔者的观点是，加激活函数是大家的自由，不排除加某个激活函数能够调出更好的效果，但加激活函数并不改变线性Attention的形式，所以不影响我们的描述，另外就是现有的结果表明，其实不加已经足够好。

## 花式遗忘门

从式$\eqref{eq:linear-attn-rnn}$我们可以看出，目前的线性Attention本质上就是个$\mathop{\text{cumsum}}$，即将所有历史信息都等权地叠加，不难想象当叠加的token足够多时，每个token的信息占比都会变得极小，于是单靠固定大小的$\boldsymbol{S}_t$矩阵甚至无法准确重建任意一个token，直观类比就是每个token的记忆都变得模糊不清。

为了缓解这个问题，[RetNet](<https://papers.cool/arxiv/2307.08621>)给线性Attention引入了遗忘效应：
\begin{equation}\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t, \qquad \boldsymbol{S}_t = \gamma\boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\label{eq:linear-attn-retnet}\end{equation}
其中衰减因子$\gamma\in(0,1)$，在RetNet中被设为常数，也有设为可训练参数的，以及将$\gamma$改为对角矩阵的，等等，[MiniMax\-01](<https://papers.cool/arxiv/2501.08313>)所用的线性Attention也是这种。注意，衰减因子在RetNet前也有，不过它们多以线性RNN的形式出现，如上一节提到的[LRU](<https://spaces.ac.cn/archives/9554>)、[SSM](<https://spaces.ac.cn/tag/ssm/>)等，RetNet应该是首次将它跟线性Attention结合起来。加入衰减因子后，模型会倾向于遗忘掉更为久远的历史信息，从而至少保证最近token的分辨率，说白了就是跟语言模型特性相符的“就近原则（Recency Bias）”的体现，从而往往能工作得更好。

此外，一个值得关注的细节是RetNet还给$\boldsymbol{Q},\boldsymbol{K}$加上了[RoPE](<https://spaces.ac.cn/archives/9403>)，这相当于将衰减因子推广到复数$\gamma e^{\text{i}\theta}$，从[LRU](<https://spaces.ac.cn/archives/9554>)的角度看则是考虑了复数的特征值。尽管给RNN加位置编码的操作看上去似乎有点违和，但有些实验比如最近的[TransXSSM](<https://papers.cool/arxiv/2506.09507>)表明，给线性Attention加RoPE也有一定的正面作用。当然，这可能取决于具体的模型变体和实验设置。

式$\eqref{eq:linear-attn-retnet}$的一个简单推广是将$\gamma$更换为位置$t$的函数$\gamma_t$，这在[SSM](<https://spaces.ac.cn/tag/ssm/>)中已经有所体现。后来，[DFW](<https://papers.cool/arxiv/2210.04243>)、[Mamba](<https://papers.cool/arxiv/2312.00752>)、[Mamba2](<https://papers.cool/arxiv/2405.21060>)等工作，将它推广成跟输入相关，形成了“data\-dependent decay”相关的一系列工作，这跟以往GRU、LSTM等非线性RNN的“遗忘门（forget gate）”其实已经非常相似了，只不过为了保持模型的线性性，去掉了遗忘门对State（如$\boldsymbol{S}_t$）的依赖。

为什么我们偏爱线性RNN呢？因为线性RNN基本都能找到某种方式来并行训练，这使得它相比Softmax Attention更具竞争力——在训练效率和推理效率上都不逊色。其中，并行化的“通解”是转化为[Prefix Sum](<https://en.wikipedia.org/wiki/Prefix_sum>)问题然后Associative Scan，大体思路我们在[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](<https://spaces.ac.cn/archives/9554>)的“并行化”一节也简单介绍过。

然而，“通解”并不是GPU高效的，GPU最高效的是矩阵乘法，所以找到大量使用矩阵乘法的并行算法是最理想的，甚至都不用并行，只要找到充分使用矩阵乘法的Chunk by Chunk递归格式，都能明显提高训练效率。这反过来对模型提出了要求，如只有外积形式的遗忘门才能实现这个目的，典型反例就是Mamba，它是非外积的遗忘门，无法充分发挥GPU的性能，所以才有了后续Mamba2和[GLA](<https://papers.cool/arxiv/2312.06635>)等变化。

## 测试时训练

至此，线性Attention从最初的简单模仿Softmax Attention，到引入静态衰减因子乃至“data\-dependent decay”，已经形成了自身的特色并在不少任务上发挥价值。然而，这些进展多数是靠人工凭经验设计出来的，我们不禁要问：<strong>有没有更上层的原则来指导线性Attention甚至是一般的序列模型（Token\-Mixer）的设计？</strong>

对于这个问题，[TTT（Test Time Training）](<https://papers.cool/arxiv/2407.04620>)给出了自己的答案，它将序列模型的构建视为一个“在线学习（Online Learning）”问题，并提出用优化器来构建（不一定是线性的）RNN的做法。具体来说，它将$\boldsymbol{K},\boldsymbol{V}$视作语料对$(\boldsymbol{k}_1, \boldsymbol{v}_1),(\boldsymbol{k}_2, \boldsymbol{v}_2),\cdots,(\boldsymbol{k}_t, \boldsymbol{v}_t)$，根据这些语料训练得到一个模型$\boldsymbol{v} = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{k})$，最后输出$\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t;\boldsymbol{q}_t)$，其中$\boldsymbol{S}_t$是模型参数，至于模型结构很大程度上是任意的。

这跟RNN有什么关系呢？很简单，优化器如SGD、Adam等，它们本质上就是一个关于模型参数的RNN！其实这个观点并不新鲜，早在2017年Meta Learning盛行那会就已经有研究人员提出并利用了这点，只不过当时的想法是尝试用RNN（LSTM）去模拟一个更好的优化器，详情可以参考[《Optimization as a Model for Few\-Shot Learning》](<https://openreview.net/forum?id=rJY0-Kcll>)。

正所谓“风水轮流转”，时隔多年TTT反过来提出通过优化器来构建RNN。它的流程是这样的：首先，当前模型参数为$\boldsymbol{S}_{t-1}$，优化器（SGD）接收到新数据$(\boldsymbol{k}_t, \boldsymbol{v}_t)$，根据该数据将模型参数更新为$\boldsymbol{S}_t$，最后返回$\boldsymbol{q}_t$的预测结果$\boldsymbol{f}(\boldsymbol{S}_{t-1};\boldsymbol{q}_t)$，依此类推。所以，TTT所实现的RNN可以统一地写成
\begin{equation}\boldsymbol{o}_t = \boldsymbol{f}(\boldsymbol{S}_t; \boldsymbol{q}_t), \qquad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t\nabla_{\boldsymbol{S}_{t-1}}\mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1};\boldsymbol{k}_t), \boldsymbol{v}_t)\label{eq:ttt-rnn}\end{equation}
其中$\mathcal{L}(\boldsymbol{f}(\boldsymbol{S}_{t-1};\boldsymbol{k}_t), \boldsymbol{v}_t)$是当前数据$(\boldsymbol{k}_t, \boldsymbol{v}_t)$在当前参数$\boldsymbol{S}_{t-1}$下的损失函数，$\eta_t$则是学习率参数，参考上一节的“data\-dependent decay”，它也可以做成data\-dependent的。这个形式可以覆盖非常多的RNN模型，比如式$\eqref{eq:linear-attn-rnn}$和$\eqref{eq:linear-attn-retnet}$都是它的一个特例：
$$\begin{array}{c|cc|ccc}
\hline
& \text{RNN} & \boldsymbol{o}_t & \boldsymbol{f}(\boldsymbol{S};\boldsymbol{k}) & \mathcal{L}(\boldsymbol{f}(\boldsymbol{S};\boldsymbol{k}),\boldsymbol{v}) & \eta_t \\
\hline
\eqref{eq:linear-attn-rnn} & \boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} & \boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t & \boldsymbol{S}\boldsymbol{k} & -\boldsymbol{v}^{\top}(\boldsymbol{S}\boldsymbol{k}) & 1 \\
\eqref{eq:linear-attn-retnet} & \boldsymbol{S}_t = \gamma\boldsymbol{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^{\top} & \boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t & \boldsymbol{S}\boldsymbol{k} & -\boldsymbol{v}^{\top}(\boldsymbol{S}\boldsymbol{k}) + \frac{1-\gamma}{2}\Vert\boldsymbol{S}\Vert_F^2 & 1 \\
\hline
\end{array}$$
TTT原文则致力于探索mini\-batch下的非线性RNN，后来的[Titans](<https://papers.cool/arxiv/2501.00663>)则给TTT的SGD加上了动量，再后面[《Test\-Time Training Done Right》](<https://papers.cool/arxiv/2505.23884>)则探索了large\-batch的TTT用法，还探索了“TTT \+ Muon”的组合。注意，TTT只是利用优化器来构建RNN，RNN以外的参数如$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$的可训练参数，还是将整个模型构建起来后用整体的优化器训练的。

一个更值得思考的问题是：为什么TTT可以成为构建RNN的“指导原则”呢？RNN的核心目标，是将历史数据有效地压缩到一个固定大小的State中，而模型参数正好是固定大小的，训练模型某种程度上就相当于把训练数据压缩到模型权重中，TTT正是利用了它跟RNN目标的高度契合性。说直白一点，如果将RNN视为一个压缩任务，TTT将模型$\boldsymbol{f}$视为“解压器”，它的权重则是“压缩包”，而压缩算法则是SGD，压缩率则是损失$\mathcal{L}$。

这样一来，我们就不用花心思构建递归格式了，转而构建模型$\boldsymbol{f}$和损失$\mathcal{L}$，一个RNN强不强、靠不靠谱，我们也只需看对应的$\boldsymbol{f}$和$\mathcal{L}$就可以心中有数。

除此之外，TTT用Online Learning构建RNN，意味着所得RNN必然非常契合ICL（In Context Learning）任务，这也是TTT作为“指导原则”的优势。此前[《Why Can GPT Learn In\-Context? Language Models Implicitly Perform Gradient Descent as Meta\-Optimizers》](<https://papers.cool/arxiv/2212.10559>)甚至反过来，将Softmax Attention去掉Softmax成线性Attention来解释它的ICL能力，用现在的视角看它就是构造了对应的TTT出来。

## 除旧而迎新

例如，最早的线性Attention对应的损失函数是$-\boldsymbol{v}^{\top}(\boldsymbol{S}\boldsymbol{k})$，这一看就是个不大靠谱的目标，因为它是无下界的，这可能会导致$\boldsymbol{S}$趋于无穷。相比之下，RetNet往损失函数加入了L2正则项，避免了这种风险，从优化角度看也缓解了过拟合的风险，从而得到一个更好的RNN。

然而，用内积作为损失函数虽然简洁且有一定道理，但它不是直接鼓励$\boldsymbol{S}\boldsymbol{k}=\boldsymbol{v}$，所以并非一个理想的回归损失。更好的目标函数应该是平方损失，即$\frac{1}{2}\Vert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\Vert^2$，将它代入到TTT的公式$\eqref{eq:ttt-rnn}$得到
\begin{equation}\boldsymbol{o}_t = \boldsymbol{S}_t\boldsymbol{q}_t, \qquad \boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t \underbrace{(\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top}}_{\nabla_{\boldsymbol{S}_{t-1}}\frac{1}{2}\Vert\boldsymbol{S}_{t-1}\boldsymbol{k}_t - \boldsymbol{v}_t\Vert^2}\end{equation}
这便是DeltaNet，这个名字出自[《Parallelizing Linear Transformers with the Delta Rule over Sequence Length》](<https://papers.cool/arxiv/2406.06484>)，更早则是由[《Linear Transformers Are Secretly Fast Weight Programmers》](<https://papers.cool/arxiv/2102.11174>)提出。留意到$\eta_t (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top} = (\boldsymbol{S}_{t-1} (\sqrt{\eta_t}\boldsymbol{k}_t) - (\sqrt{\eta_t}\boldsymbol{v}_t))(\sqrt{\eta_t}\boldsymbol{k}_t)^{\top}$，这意味着$\eta_t$总可以吸收到$\boldsymbol{k}_t,\boldsymbol{v}_t$的定义中去，所以我们接下来的分析都只考虑$\eta_t=1$的情况：
\begin{equation}\begin{aligned}
\boldsymbol{S}_t =&\, \boldsymbol{S}_{t-1} -(\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top} \\
=&\, \boldsymbol{S}_{t-1} -(\boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top} + \boldsymbol{v}_t\boldsymbol{k}_t^{\top} \\
=&\, \boldsymbol{S}_{t-1} (\boldsymbol{I} - \boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \boldsymbol{v}_t\boldsymbol{k}_t^{\top}
\end{aligned}\label{eq:linear-attn-deltanet}\end{equation}
如果有需要，我们再把$\boldsymbol{k}_t,\boldsymbol{v}_t$换成$\sqrt{\eta_t}\boldsymbol{k}_t,\sqrt{\eta_t}\boldsymbol{v}_t$，就可以将$\eta_t$恢复出来。对比线性Attention最早的形式$\eqref{eq:linear-attn-rnn}$，DeltaNet的区别是在加$\boldsymbol{v}_t\boldsymbol{k}_t^{\top}$前多减了个$(\boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top}$，其中$\boldsymbol{S}_{t-1} \boldsymbol{k}_t$可以理解为新输入$\boldsymbol{k}_t$在旧模型$\boldsymbol{S}_{t-1}$下的预测结果。

直观来想，“先减后加”就是先移除模型对$\boldsymbol{k}_t$的旧认知，然后根据$(\boldsymbol{k}_t,\boldsymbol{v}_t)$补充新认知，达到“除旧迎新”的效果。这个规则称为“[Delta Rule](<https://en.wikipedia.org/wiki/Delta_rule>)”，正是DeltaNet一词中“Delta”的来源。Delta Rule并不新鲜，它又称为[Least Mean Square](<https://en.wikipedia.org/wiki/Least_mean_squares_filter>)、Widrow\-Hoff Algorithm等，已经是上个世纪60年代的产物了。事实上，这个领域完全新的东西很少，很多改动都可以追溯到某个“上古时期”的工作，目前的努力主要集中在挖掘其中能Scalable的部分。

另外需要指出的是，按照时间的顺序，是DeltaNet在前，TTT在后，从Online Learning角度理解RNN，其实在TTT之前已经零星地体现在一些工作中，但TTT系统地提出了这个“指导原则”，并且将它用于构建新RNN模型，所以我们把TTT放在前面，使得整个介绍更加流畅自然一些。

有些读者可能疑问：DeltaNet还算线性RNN吗？答案是肯定的。我们所说的线性RNN，是指递归公式对State变量的依赖关系是线性的，但对输入或$\boldsymbol{q},\boldsymbol{k},\boldsymbol{v}$的依赖可以是非线性的（当然不同依赖形式的并行效率会有所不同），从式$\eqref{eq:linear-attn-deltanet}$可以看出，等号右端始终只是出现了$\boldsymbol{S}_{t-1}$的一次方，所以它满足线性的定义。

## 求逆与推广

前面我们说了，线性RNN最理想的（即GPU高效的）并行算法是充分使用矩阵乘法的形式。为了完成这一目标，我们先将DeltaNet写成
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + (\boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t)\boldsymbol{k}_t^{\top}\end{equation}
记$\boldsymbol{u}_t = \boldsymbol{v}_t - \boldsymbol{S}_{t-1} \boldsymbol{k}_t$，那么$\boldsymbol{S}_t = \boldsymbol{S}_{t-1} + \boldsymbol{u}_t\boldsymbol{k}_t^{\top}$，也就是说它只是在最早的线性Attention基础上把$\boldsymbol{V}$换成了$\boldsymbol{U}=[\boldsymbol{u}_1,\boldsymbol{u}_2,\cdots,\boldsymbol{u}_n]^{\top}$，将它迭代$t-1$次，我们有
\begin{equation}\boldsymbol{S}_{t-1} = \sum_{j=1}^{t-1} \boldsymbol{u}_j\boldsymbol{k}_j^{\top}\qquad\Rightarrow\qquad \boldsymbol{u}_t = \boldsymbol{v}_t - \left(\sum_{j=1}^{t-1} \boldsymbol{u}_j\boldsymbol{k}_j^{\top}\right)\boldsymbol{k}_t = \boldsymbol{v}_t - \sum_{j=1}^{t-1} \boldsymbol{u}_j(\boldsymbol{k}_j^{\top}\boldsymbol{k}_t)\end{equation}
最后的等式写成矩阵形式是$\boldsymbol{U} = \boldsymbol{V} - (\boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)\boldsymbol{U}$，其中$\boldsymbol{M}^-=\boldsymbol{M} - \boldsymbol{I}$，这是一个线性方程组，它的解可以直接表示为
\begin{equation}\boldsymbol{U} = (\boldsymbol{I} + \underbrace{\boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-}_{\text{记为}\boldsymbol{B}})^{-1}\boldsymbol{V}\end{equation}
这里出现了$(\boldsymbol{I}+\boldsymbol{B})^{-1}$，一个$n\times n$矩阵的逆，标准复杂度是$\mathcal{O}(n^3)$，比Softmax Attention还高！不过好在我们不需要显式的逆而是只要$\boldsymbol{U}$，这可以转化为解方程组$(\boldsymbol{I}+\boldsymbol{B})\boldsymbol{U}=\boldsymbol{V}$，复杂度降到$\mathcal{O}(n^2)$。进一步地，利用$\boldsymbol{I}+\boldsymbol{B}$是下三角阵以及$\boldsymbol{B}$的低秩结构，可以将复杂度降到线性，写成分块矩阵乘法后就可以充分利用GPU。这些细节只能请大家阅读原论文了，本文先把主要数学原理介绍清楚。

DeltaNet之后，[Gated DeltaNet（GDN）](<https://papers.cool/arxiv/2412.06464>)进一步地将遗忘门引入到DeltaNet之中，这倒是可以预料的变化。Gated DeltaNet的原始引入方式是
\begin{equation}\boldsymbol{S}_t = \alpha_t \boldsymbol{S}_{t-1} (\boldsymbol{I} - \beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top}\label{eq:gdn-orgi}\end{equation}
但个人认为，这个提法其实显式打破了Delta Rule，更好的提法应该是像[Comba](<https://papers.cool/arxiv/2506.02475>)一样，只乘到第一个$\boldsymbol{S}_{t-1}$上：
\begin{equation}\boldsymbol{S}_t = \gamma_t\boldsymbol{S}_{t-1} + \eta_t(\boldsymbol{v}_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t)\boldsymbol{k}_t^{\top}\label{eq:gdn-comba}\end{equation}
它相当于将损失函数取$\frac{1}{2}\Vert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\Vert^2 + \frac{1-\gamma}{\eta}\Vert\boldsymbol{S}\Vert_F^2$。当然，从数学上来说，这两个提法都是等价的：
\begin{equation}\alpha_t\boldsymbol{S}_{t-1} (\boldsymbol{I} - \beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \beta_t\boldsymbol{v}_t\boldsymbol{k}_t^{\top} = \alpha_t \boldsymbol{S}_{t-1} + \alpha_t \beta_t (\boldsymbol{v}_t/\alpha_t - \boldsymbol{S}_{t-1}\boldsymbol{k}_t)\boldsymbol{k}_t^{\top}\end{equation}
即$\gamma_t = \alpha_t, \eta_t = \alpha_t \beta_t$然后把$1/\alpha_t$吸收到$\boldsymbol{v}_t$就可以转化为后者了。所以说，这两个形式在数学上并没有区别，由于多数$\alpha_t$会接近于1，所以能力上估计也没啥区别（Comba说$\eqref{eq:gdn-comba}$会好一点），只不过后者更直观地保留了Delta Rule的样子。

从理论上来说，Gated DeltaNet也可以写成DeltaNet的形式，因为只需要定义$\bar{\alpha}_t = \prod_{j=1}^t \alpha_t$，那么式$\eqref{eq:gdn-orgi}$两边同时除以$\bar{\alpha}_t$，就得到
\begin{equation}\bar{\alpha}_t^{-1}\boldsymbol{S}_t = \bar{\alpha}_{t-1}^{-1}\boldsymbol{S}_{t-1}  (\boldsymbol{I} - \beta_t\boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \beta_t(\bar{\alpha}_t^{-1}\boldsymbol{v}_t)\boldsymbol{k}_t^{\top}\end{equation}
然后结合$\boldsymbol{o}_t = \boldsymbol{S}_t \boldsymbol{q}_t = (\bar{\alpha}_t^{-1}\boldsymbol{S}_t) (\bar{\alpha}_t\boldsymbol{q}_t)$，可以发现只需要分别将$\bar{\alpha}_t\boldsymbol{q}_t,\bar{\alpha}_t^{-1}\boldsymbol{v}_t$设置为新的$\boldsymbol{q}_t,\boldsymbol{v}_t$，那么就能简化成DeltaNet的形式。不过，这个结果只有在某些情况下具有理论推导的价值（比如推导下一节的Attention矩阵），因为实际计算中，不管怎么参数化，对于足够大的$t$，$\bar{\alpha}_t$和$\bar{\alpha}_t^{-1}$之一必有溢出的风险。

DeltaNet之后还有另一个推广[DeltaProduct](<https://papers.cool/arxiv/2502.10297>)，它是将$\boldsymbol{k},\boldsymbol{v}$扩展若干倍后再做DeltaNet或者Gated DeltaNet，试图增强模型的状态追踪能力。不过，就笔者的审美而言，与其像DeltaProduct那样扩展常数倍，还不如像[《时空之章：将Attention视为平方复杂度的RNN》](<https://spaces.ac.cn/archives/10017>)一样尝试平方复杂度的RNN，看有没有机会超越Softmax Attention。

## 反哺进行时

说到超越Softmax Attention，开头提到，如今的线性Attention不仅能与Softmax Attention一较高低，甚至开始“反哺”它。这看似不可思议，但细思之下并不难理解。某种意义上，这些年Softmax Attention一直在退步，从MHA、GQA到MQA都是为了压缩KV Cache而做减法。而线性Attention没有KV Cache问题，所以一直往更好的方向前进。

为了更好看出这一点，我们不妨将前面提到的Attention机制都以矩阵形式写出来：
\begin{array}{c|c}
\hline
& \text{公式} \\[4pt]
\hline
\text{Softmax Attention} & (\exp(\boldsymbol{Q}\boldsymbol{K}^{\top})\odot \boldsymbol{M})\boldsymbol{V} \\[4pt]
\text{最早的线性Attention} & (\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{M})\boldsymbol{V} \\[4pt]
\text{加入遗忘门后} & (\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{\Gamma})\boldsymbol{V} \\[4pt]
\text{DeltaNet} & (\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{M})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}\boldsymbol{V} \\[4pt]
\text{Gated DeltaNet} & \begin{gathered}((\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{M})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{M}^-)^{-1}\odot\boldsymbol{\Gamma})\boldsymbol{V} \\ =(\boldsymbol{Q}\boldsymbol{K}^{\top}\odot \boldsymbol{\Gamma})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot \boldsymbol{\Gamma}^-)^{-1}\boldsymbol{V}\end{gathered} \\[4pt]
\hline
\end{array}
其中
\begin{equation}\Gamma_{i,j} = \left\{\begin{aligned} &\prod_{\tau=j+1}^i \gamma_{\tau}, &i > j \\[6pt] &\qquad 1, &i = j \\[6pt] &\qquad 0, &i < j\end{aligned}\right.\end{equation}
以及$\boldsymbol{\Gamma}^- = \boldsymbol{\Gamma} - \boldsymbol{I}$。这样看来，Softmax Attention的形式还仅停留在最早的线性Attention那会（当然这也证明了它的强大）。那“反哺”怎么实现呢？首先我们需要一种方法把Softmax Attention转化为线性Attention，这个并不难，早在[《Transformer升级之路：5、作为无限维的线性Attention》](<https://spaces.ac.cn/archives/8601>)我们就总结了三种将Softmax Attention转化为<u>无限维</u>线性Attention的方案。

总之，就是存在一个映射$\phi$，将$\boldsymbol{Q},\boldsymbol{K}$从$n\times d$映射到$n\times \infty$，满足$\exp(\boldsymbol{Q}\boldsymbol{K}^{\top}) = \phi(\boldsymbol{Q})\phi(\boldsymbol{K})^{\top}$，这称为“核技巧”。那接下来的事情就简单了，我们只需将上述表格中的线性Attention的$\boldsymbol{Q},\boldsymbol{K}$换成$\phi(\boldsymbol{Q}),\phi(\boldsymbol{K})$，最后再设法恢复$\exp$并归一化，就得到新的Softmax Attention变体了。例如，代入到遗忘门的公式，我们有
\begin{equation}(\phi(\boldsymbol{Q})\phi(\boldsymbol{K})^{\top}\odot \boldsymbol{\Gamma})\boldsymbol{V} = \exp(\boldsymbol{Q}\boldsymbol{K}^{\top} + \log\boldsymbol{\Gamma})\boldsymbol{V}\end{equation}
如果$\gamma_t$取常数，那么其实就是[《Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation》](<https://papers.cool/arxiv/2108.12409>)所提的ALIBI，而如果$\gamma_t$是依赖于输入的，那么就是[《Forgetting Transformer: Softmax Attention with a Forget Gate》](<https://papers.cool/arxiv/2503.02130>)所提的FoX。

一个更有意思的结果是[《Understanding Transformer from the Perspective of Associative Memory》](<https://papers.cool/arxiv/2505.19488>)所提的DeltaFormer，顾名思义它是Softmax Attention的DeltaNet版本。将DeltaNet的$\boldsymbol{Q},\boldsymbol{K}$换成$\phi(\boldsymbol{Q}),\phi(\boldsymbol{K})$，我们有
\begin{equation}\begin{aligned}
&\,(\phi(\boldsymbol{Q})\phi(\boldsymbol{K})^{\top}\odot \boldsymbol{M})(\boldsymbol{I} + \phi(\boldsymbol{K})\phi(\boldsymbol{K})^{\top}\odot \boldsymbol{M}^-)^{-1}\boldsymbol{V} \\[8pt]
=&\,\underbrace{\exp(\boldsymbol{Q} \boldsymbol{K}^{\top} + \log\boldsymbol{M})}_{\text{记为}\boldsymbol{A}}(\boldsymbol{I} + \underbrace{\exp(\boldsymbol{K} \boldsymbol{K}^{\top} + \log\boldsymbol{M}^-)}_{\text{记为}\boldsymbol{B}})^{-1}\boldsymbol{V}
\end{aligned}\end{equation}
如果要归一化，我们将$\exp$换成$\text{softmax}$即可。相比Softmax Attention，DeltaFormer将原本的$\boldsymbol{A}\boldsymbol{V}$改成了$\boldsymbol{A}(\boldsymbol{I}+\boldsymbol{B})^{-1}\boldsymbol{V}$，注意到
\begin{equation}\begin{aligned}
\boldsymbol{A}(\boldsymbol{I}+\boldsymbol{B})^{-1}\boldsymbol{V} =&\, \boldsymbol{A}(\boldsymbol{I}-\boldsymbol{B}+\boldsymbol{B}^2- \boldsymbol{B}^3 + \cdots)\boldsymbol{V} \\
=&\, \boldsymbol{A}(\boldsymbol{V}-\boldsymbol{B}\boldsymbol{V}+\boldsymbol{B}^2\boldsymbol{V}- \boldsymbol{B}^3\boldsymbol{V} + \cdots)
\end{aligned}\end{equation}
所以DeltaFormer相当于先用$\boldsymbol{K},\boldsymbol{K},\boldsymbol{V}$算多次Attention，将结果叠加起来后作为新的$\boldsymbol{V}$，再跟$\boldsymbol{Q},\boldsymbol{K}$算一次Attention，这个特性让它对Multi\-Hop的任务有奇效（比如Code）。此外，DeltaFormer的这个特点还意味着它跟MQA特别搭配，因为$(\boldsymbol{I}+\boldsymbol{B})^{-1}\boldsymbol{V}$这部分只有$\boldsymbol{K},\boldsymbol{V}$参与，而对于MQA来说$\boldsymbol{K},\boldsymbol{V}$只有Single\-Head，计算量相比MHA会明显降低。

不过，在笔者看来，这种固定系数的叠加可能是“没有免费午餐”，比如笔者的实验结果显示，DeltaFormer的语言模型损失并无太大变化，这意味着如果某些任务的损失明显降低，必然有另一些任务的损失上升了。

## 硬核编码术

还有一个值得关注的反哺工作是PaTH Attention，出自[《PaTH Attention: Position Encoding via Accumulating Householder Transformations》](<https://papers.cool/arxiv/2505.16381>)，它从位置编码的角度将DeltaNet反哺到Softmax Attention。

我们在[《Transformer升级之路：6、旋转位置编码的完备性分析》](<https://spaces.ac.cn/archives/9403>)指出，对于任何正交矩阵$\boldsymbol{\Omega}$，$\boldsymbol{R}_m = \boldsymbol{\Omega}^m$都是广义的RoPE。除了旋转矩阵，还有哪些容易构建的正交矩阵呢？PaTH用的是[Householder矩阵](<https://en.wikipedia.org/wiki/Householder_transformation>)：设$\boldsymbol{w}$是任意模长为$\sqrt{2}$的列向量，那么$\boldsymbol{I}-\boldsymbol{w}\boldsymbol{w}^{\top}$是一个正交矩阵，这我们在[《从一个单位向量变换到另一个单位向量的正交矩阵》](<https://spaces.ac.cn/archives/8453>)也推导过，几何意义是镜面反射。

容易看出，这跟DeltaNet中$\boldsymbol{S}_{t-1}$所乘的$\boldsymbol{I}-\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$是一样的，所以PaTH干脆把这部分照搬过来，即放弃$\boldsymbol{\Omega}^m$这个形式，也放弃$\boldsymbol{w}$模长为$\sqrt{2}$的约束，直接用一系列$\boldsymbol{I}-\boldsymbol{w}\boldsymbol{w}^{\top}$连乘来表达位置信息：
\begin{equation}\boldsymbol{q}_i^{\top}\boldsymbol{k}_j \qquad\to\qquad \boldsymbol{q}_i^{\top}\underbrace{(\boldsymbol{I}-\boldsymbol{w}_i\boldsymbol{w}_i^{\top})(\boldsymbol{I}-\boldsymbol{w}_{i-1}\boldsymbol{w}_{i-1}^{\top})\cdots(\boldsymbol{I}-\boldsymbol{w}_{j+1}\boldsymbol{w}_{j+1}^{\top})}_{\text{记为}\boldsymbol{R}_{i,j}}\boldsymbol{k}_j \end{equation}
将$\boldsymbol{R}_{i,j}$写成递归形式是$\boldsymbol{R}_{i,j} = (\boldsymbol{I}-\boldsymbol{w}_i\boldsymbol{w}_i^{\top})\boldsymbol{R}_{i-1,j},\boldsymbol{R}_{j,j} = \boldsymbol{I}$。对比DeltaNet的式$\eqref{eq:linear-attn-deltanet}$，上式相当于$\boldsymbol{v}_t$恒等于零，但初值$\boldsymbol{S}_0$不再是零。使用“[求逆与推广](<https://spaces.ac.cn/archives/11033#%E6%B1%82%E9%80%86%E4%B8%8E%E6%8E%A8%E5%B9%BF>)”一节同样的过程，我们可以得到
\begin{equation}\boldsymbol{R}_{i,j} = \boldsymbol{I} - \boldsymbol{W}_{[j:i]}^{\top}(\boldsymbol{I} + \boldsymbol{W}_{[j:i]}\boldsymbol{W}_{[j:i]}^{\top}\odot\boldsymbol{M}^-)^{-1}\boldsymbol{W}_{[j:i]}\end{equation}
其中$\boldsymbol{W}=[\boldsymbol{w}_1,\boldsymbol{w}_2,\cdots,\boldsymbol{w}_n]^{\top}$，切片按Numpy来理解，如$\boldsymbol{W}_{[j:i]}=[\boldsymbol{w}_{j+1},\boldsymbol{w}_{j+2},\cdots,\boldsymbol{w}_i]^{\top}$，切片优先级高于转置。注意求逆的是下三角阵，三角阵有一个重要特性，逆矩阵的对角线元素等于原矩阵对角线元素的倒数，如果是分块三角阵则对角块也满足这个特性，于是我们可以写出
\begin{equation}(\boldsymbol{I} + \boldsymbol{W}_{[j:i]}\boldsymbol{W}_{[j:i]}^{\top}\odot\boldsymbol{M}^-)^{-1} = (\underbrace{(\boldsymbol{I} + \boldsymbol{W}\boldsymbol{W}^{\top}\odot\boldsymbol{M}^-)^{-1}}_{\text{记为}\boldsymbol{J}})_{[j:i,j:i]}\end{equation}
接下来的变换，写成分量形式可能好理解一些
\begin{equation}\begin{aligned}
A_{i,j} =&\, \boldsymbol{q}_i^{\top} \boldsymbol{R}_{i,j} \boldsymbol{k}_j \\[6pt]
=&\, \boldsymbol{q}_i^{\top}\boldsymbol{k}_j - \boldsymbol{q}_i^{\top}\boldsymbol{W}_{[j:i]}^{\top}\boldsymbol{J}_{[j:i,j:i]}\boldsymbol{W}_{[j:i]}\boldsymbol{k}_j \\
=&\, \boldsymbol{q}_i^{\top}\boldsymbol{k}_j - \sum_{p=1}^d  \sum_{l=j+1}^i \sum_{r=j+1}^i \sum_{s=1}^d Q_{i,p} W_{l,p} J_{l,r} W_{r,s} K_{j,s} \\
=&\, \boldsymbol{q}_i^{\top}\boldsymbol{k}_j - \sum_{p=1}^d \sum_{l=1}^i \sum_{r=j+1}^n \sum_{s=1}^d Q_{i,p} W_{l,p} J_{l,r} W_{r,s} K_{j,s} \\
=&\, \boldsymbol{q}_i^{\top}\boldsymbol{k}_j - \sum_{p=1}^d \sum_{l=1}^n \sum_{r=1}^n \sum_{s=1}^d Q_{i,p} W_{l,p} \chi_{l \leq i} J_{l,r} \chi_{r \geq j+1}W_{r,s} K_{j,s} \\
=&\, \boldsymbol{q}_i^{\top}\boldsymbol{k}_j -  \sum_{l=1}^n \sum_{r=1}^n \underbrace{\left(\chi_{l \leq i}\sum_{p=1}^d Q_{i,p} W_{l,p}\right)}_{\boldsymbol{Q}\boldsymbol{W}^{\top}\odot\boldsymbol{M}} J_{l,r} \underbrace{\left(\chi_{r \geq j+1} \sum_{s=1}^d W_{r,s} K_{j,s}\right)}_{\boldsymbol{W}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-} \\
\end{aligned}\end{equation}
这里有几个关键点：比较巧妙的是第4个等号，它利用了$\boldsymbol{J}$是下三角矩阵这一点，所以$l < r$时$J_{l,r}$自动为零；第5个等号，$\chi$为示性函数，满足下标的条件时为1，否则为0；第6个等号，当我们分别处理$p,s$两部分求和时，结果是$\boldsymbol{Q}\boldsymbol{W}^{\top}$和$\boldsymbol{W}\boldsymbol{K}^{\top}$，而乘$\chi_{l \leq i}$刚好表示保留$\boldsymbol{Q}\boldsymbol{W}^{\top}$的下三角部分（连同对角线），而乘$\chi_{r \geq j+1}$则表示保留$\boldsymbol{W}\boldsymbol{K}^{\top}$的下三角部分（不包括对角线）。

至此，我们可以把整个（Softmax之前的）注意力矩阵写出来：
\begin{equation}\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{K}^{\top}\odot\boldsymbol{M} - (\boldsymbol{Q}\boldsymbol{W}^{\top}\odot\boldsymbol{M})(\boldsymbol{I} + \boldsymbol{W}\boldsymbol{W}^{\top}\odot\boldsymbol{M}^-)^{-1}(\boldsymbol{W}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-) \label{eq:path-attn}\end{equation}
有没有被震惊到？这还没完。直接求逆复杂度是$\mathcal{O}(n^3)$，这肯定无法接受，还要想办法利用$\boldsymbol{W}\boldsymbol{W}^{\top}$的低秩特点将复杂度降低到$\mathcal{O}(n^2)$，然后还要推反向传播，最后写成类似Flash Attention的高效实现，这些细节大家只能看原论文挖掘了，总之全程都非常硬核。

从位置编码的角度看，PaTH是[CoPE（Contextual Position Encoding）](<https://papers.cool/arxiv/2405.18719>)的一种，它的位置并不是编号$1,2,3,\cdots$，而是根据上下文内容自动生成的位置信号。类似地，FoX也可以看成是Contextual版的ALIBI。上下文相关的位置信息是当前线性Attention的主要特征，也可能是反哺Softmax Attention的主要方向。

## 化简乐无穷

我们不妨再深入点探讨一下PaTH，这不仅有助于我们了解PaTH，也能帮助我们更熟悉DeltaNet，两者本身就是高度相关的。这一节我们从PaTH的两个特例入手，它可以帮助我们更好地理解PaTH与DeltaNet的关联。

第一个特例是$\boldsymbol{W}=\boldsymbol{K}$，代入到$\eqref{eq:path-attn}$得到
\begin{equation}\begin{aligned}
\boldsymbol{A} =&\, (\boldsymbol{Q}\boldsymbol{K}^{\top}\odot\boldsymbol{M})(\boldsymbol{I} - (\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-)^{-1}(\boldsymbol{K}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-)) \\[6pt]
=&\, (\boldsymbol{Q}\boldsymbol{K}^{\top}\odot\boldsymbol{M})(\boldsymbol{I} + \boldsymbol{K}\boldsymbol{K}^{\top}\odot\boldsymbol{M}^-)^{-1} \qquad (\text{注}:\boldsymbol{I} - (\boldsymbol{I} + \boldsymbol{A})^{-1} \boldsymbol{A} = (\boldsymbol{I}+\boldsymbol{A})^{-1})
\end{aligned}\end{equation}
有没有觉得有点熟悉？这刚好就是DeltaNet的Attention矩阵！从这个特例看来，PaTH和DeltaFormer的区别就在于，DeltaFormer基于核技巧，给DeltaNet的$\boldsymbol{Q}\boldsymbol{K}^{\top}$和$\boldsymbol{K}\boldsymbol{K}^{\top}$分别加上$\exp$，而PaTH直接给DeltaNet的Attention矩阵加上$\exp$。

第二个特例是重新引入$\Vert\boldsymbol{w}\Vert=\sqrt{2}$这个约束，此时$\boldsymbol{I}-\boldsymbol{w}\boldsymbol{w}^{\top}$是正交矩阵，我们引入
\begin{equation}\begin{aligned}
\boldsymbol{R}_i \triangleq&\, (\boldsymbol{I}-\boldsymbol{w}_i\boldsymbol{w}_i^{\top})(\boldsymbol{I}-\boldsymbol{w}_{i-1}\boldsymbol{w}_{i-1}^{\top})\cdots(\boldsymbol{I}-\boldsymbol{w}_1\boldsymbol{w}_1^{\top}) \\
=&\, \boldsymbol{I} - \boldsymbol{W}_{[:i]}^{\top}(\boldsymbol{I} + \boldsymbol{W}_{[:i]}\boldsymbol{W}_{[:i]}^{\top}\odot\boldsymbol{M}^-)^{-1}\boldsymbol{W}_{[:i]} \\
=&\,\boldsymbol{R}_{i,0}
\end{aligned}\end{equation}
那么$\boldsymbol{R}_{i,j} = \boldsymbol{R}_i \boldsymbol{R}_j^{\top}$。这个等式意味着我们可以像RoPE一样，用绝对位置的方式实现相对位置的PaTH，即只需要给每个$\boldsymbol{q}_i^{\top},\boldsymbol{k}_i^{\top}$都乘上$\boldsymbol{R}_i$，然后套用Softmax Attention的实现就行。那么乘$\boldsymbol{R}_i$是什么运算呢？重复上一节的展开过程，我们有
\begin{equation}\begin{aligned}
(\boldsymbol{q}_i^{\top} \boldsymbol{R}_{i})_s =&\, (\boldsymbol{q}_i^{\top} - \boldsymbol{q}_i^{\top}\boldsymbol{W}_{[:i]}^{\top}\boldsymbol{J}_{[:i,:i]}\boldsymbol{W}_{[:i]})_s \\
=&\, Q_{i,s} - \sum_{p=1}^d  \sum_{l=1}^i \sum_{r=1}^i  Q_{i,p} W_{l,p} J_{l,r} W_{r,s}  \\
=&\, Q_{i,s}  - \sum_{p=1}^d  \sum_{l=1}^i \sum_{r=1}^n  Q_{i,p} W_{l,p} J_{l,r} W_{r,s}  \\
=&\, Q_{i,s}  - \sum_{p=1}^d  \sum_{l=1}^n \sum_{r=1}^n  \chi_{l\leq i} Q_{i,p} W_{l,p} J_{l,r} W_{r,s}  \\
=&\, Q_{i,s}  - \sum_{l=1}^n \underbrace{\chi_{l\leq i} \sum_{p=1}^d   Q_{i,p} W_{l,p}}_{\boldsymbol{Q}\boldsymbol{W}^{\top}\odot\boldsymbol{M}}\, \underbrace{\sum_{r=1}^n J_{l,r} W_{r,s}}_{\boldsymbol{J}\boldsymbol{W}}
\end{aligned}\end{equation}
写成矩阵形式就是
\begin{equation}\boldsymbol{\boldsymbol{Q}} - (\boldsymbol{Q}\boldsymbol{W}^{\top}\odot\boldsymbol{M})(\boldsymbol{I} + \boldsymbol{W}\boldsymbol{W}^{\top}\odot\boldsymbol{M}^-)^{-1}\boldsymbol{W}\end{equation}
是不是又觉得有点熟悉？其实第二部分就是$\text{DeltaNet}(\boldsymbol{Q},\boldsymbol{W},\boldsymbol{W})$！所以这种情况下PaTH实现的效果等价于是
\begin{equation}\mathop{\text{SoftmaxAttention}}(\underbrace{\boldsymbol{Q}-\mathop{\text{DeltaNet}}(\boldsymbol{Q},\boldsymbol{W},\boldsymbol{W})}_{\tilde{\boldsymbol{Q}}},\underbrace{\boldsymbol{K}-\mathop{\text{DeltaNet}}(\boldsymbol{K},\boldsymbol{W},\boldsymbol{W})}_{\tilde{\boldsymbol{K}}},\boldsymbol{V})\end{equation}
也就是用DeltaNet给$\boldsymbol{Q},\boldsymbol{K}$加位置编码。这样看PaTH（在$\Vert\boldsymbol{w}\Vert=\sqrt{2}$这个约束下）就相当于Softmax Attention与DeltaNet的某种层内混合。当然我们也可以考虑放弃前面的推导，即便$\Vert\boldsymbol{w}\Vert\neq\sqrt{2}$时也按照上式来实现，这就类似于通过[Canon Layers](<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330>)的方案，用卷积给$\boldsymbol{Q},\boldsymbol{K}$加位置信息了，只不过这里的卷积不再是短卷积，而是DeltaNet这种长卷积。

## 剑走偏锋法

最后，我们再看最近的一个同样值得关注的线性Attention模型——MesaNet（还有一个大同小异的同期工作[Atlas](<https://papers.cool/arxiv/2505.23735>)）。TTT的Online Learning视角告诉我们，DeltaNet其实就是在用SGD优化目标函数$\frac{1}{2}\Vert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\Vert^2$，而我们仔细观察就会发现，$\boldsymbol{S}\boldsymbol{k}$只是$\boldsymbol{k}$的线性函数，所以这实际上只是一个线性回归问题，线性回归是有解析解的！
\begin{equation}\boldsymbol{S}_t = \boldsymbol{G}_t \boldsymbol{H}_t^{-1},\quad \boldsymbol{G}_t = \sum_{j=1}^t \boldsymbol{v}_j \boldsymbol{k}_j^{\top},\quad \boldsymbol{H}_t = \sum_{j=1}^t \boldsymbol{k}_j \boldsymbol{k}_j^{\top}\end{equation}
MesaNet就是利用这个解析解来构建序列模型的，其想法起源于[《Uncovering mesa\-optimization algorithms in Transformers》](<https://papers.cool/arxiv/2309.05858>)，高效训练则是由[《MesaNet: Sequence Modeling by Locally Optimal Test\-Time Training》](<https://papers.cool/arxiv/2506.05233>)实现。MesaNet在上述公式基础上给$\boldsymbol{G}_t,\boldsymbol{H}_t$加入遗忘门，然后求时加上对角阵$\boldsymbol{\Lambda}_t$避免不可逆，总的模型是
\begin{equation}\boldsymbol{o}_t = \boldsymbol{G}_t (\boldsymbol{H}_t + \boldsymbol{\Lambda}_t)^{-1} \boldsymbol{q}_t,\quad \boldsymbol{G}_t = \gamma_t \boldsymbol{G}_{t-1} +  \boldsymbol{v}_t \boldsymbol{k}_t^{\top},\quad\boldsymbol{H}_t = \gamma_t \boldsymbol{H}_{t-1} + \boldsymbol{k}_t \boldsymbol{k}_t^{\top}\end{equation}
很明显，$\boldsymbol{G}_t,\boldsymbol{H}_t$关于序列长度的复杂度是线性的，所以$\boldsymbol{o}_t$的计算复杂度也是线性的，因此MesaNet仍然属于线性Attention的范畴，并且由于解析解的缘故，基本上可以保证大多数情况下它优于DeltaNet甚至Gated DeltaNet。从信号处理的角度看，MesaNet与DeltaNet是[Recursive Least Square](<https://en.wikipedia.org/wiki/Recursive_least_squares_filter>)和[Least Mean Square](<https://en.wikipedia.org/wiki/Least_mean_squares_filter>)的区别。

看上去都是优点，为啥笔者会将它归入“剑走偏锋”呢？在笔者看来，MesaNet“成也解析解，败也解析解”，解析解使得它通常优于DeltaNet，但也给人一种“到此为止”的感觉，因为只要稍变一下就几乎没有机会求得解析解了。纵观整个数学史，所有依赖于解析解的分支在今天几乎已经都没落了，因为解析解实在太稀罕、太没有代表性了。

从实现上来看，MesaNet需要求逆的矩阵$\boldsymbol{H}_t + \boldsymbol{\Lambda}_t$并不是三角阵，尽管$(\boldsymbol{H}_t + \boldsymbol{\Lambda}_t)^{-1} \boldsymbol{q}_t$仍然可以转化为解方程而不需要显式逆，但非三角阵仍使得它求解复杂度会增加不少。如何尽可能低成本地并行计算全体$(\boldsymbol{H}_t + \boldsymbol{\Lambda}_t)^{-1} \boldsymbol{q}_t$将会是MesaNet长期的难点，目前论文用到的是“[共轭梯度法](<https://en.wikipedia.org/wiki/Conjugate_gradient_method>)”求近似解，能用但并不完美。

再就是从理论能力上看，MesaNet也并非严格优于DeltaNet。这是因为MesaNet的$\boldsymbol{G}_t,\boldsymbol{H}_t$更新规则还是简单的滑动平均，它的求逆也不涉及到Token之间的交互，所以它的能力极限大概不如拥有Delta Rule的DeltaNet。直观理解就是，MesaNet会尽力记住全体$\boldsymbol{k},\boldsymbol{v}$，“全都要”可能会导致比较模糊的记忆，而DeltaNet的原则是“除旧迎新”，因为“除旧”的缘故，它可以实现长期、精准地记忆某些内容。

我们还可以从一个特殊例子来理解这个非最优性，那就是目前为止除MesaNet的所有Attention，都允许K、V共享的选择，“允许”的意思是不一定最优，但至少能训出非平凡的结果，然而MesaNet并不行，因为K、V相同的话，MesaNet的$\boldsymbol{S}_t$就恒为单位阵了。

总的来说，MesaNet是一个让人赏心悦目的模型，但解析解也增加了它的复杂性和限制了它的灵活性，留下了不少亟待探索的空间。如果读者想要了解更多基于线性回归来构建序列模型的内容，还可以阅读[TTR](<https://papers.cool/arxiv/2501.12352>)，它对各种线性回归目标下的序列模型做了详细讨论。

## 方兴未艾路

本文简要梳理了线性Attention的发展脉络，并介绍了部分模型的数学原理。线性Attention从模仿Softmax Attention起步，逐渐发展出自身特色，如今已成为极具竞争力的序列建模方案，甚至反过来为Softmax Attention的发展提供了新思路，这一过程本身充满了趣味性和启发性。
