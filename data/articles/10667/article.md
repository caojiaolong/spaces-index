# 细水长flow之TARFLOW：流模型满血归来？

> 作者：苏剑林 · 科学空间 · 2025-01-17
>
> 原文：<https://spaces.ac.cn/archives/10667>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

不知道还有没有读者对这个系列有印象？这个系列取名“细水长flow”，主要介绍flow模型的相关工作，起因是当年（2018年）OpenAI发布了一个新的流模型[Glow](<https://spaces.ac.cn/archives/5807>)，在以GAN为主流的当时来说着实让人惊艳了一番。但惊艳归惊艳，事实上在相当长的时间内，Glow及后期的一些改进在生成效果方面都是比不上GAN的，更不用说现在主流的扩散模型了。

不过局面可能要改变了，上个月的论文[《Normalizing Flows are Capable Generative Models》](<https://papers.cool/arxiv/2412.06329>)提出了新的流模型TARFLOW，它在几乎在所有的生成任务效果上都逼近了当前SOTA，可谓是流模型的“满血”回归。

## 写在前面

这里的流模型，特指Normalizing Flow，是指模型架构具有可逆特点、以最大似然为训练目标、能实现一步生成的相关工作，当前扩散模型的分支Flow Matching不归入此列。

自从Glow闪耀登场之后，流模型的后续进展可谓“乏善可陈”，简单来说就是让它生成没有明显瑕疵的CelebA人脸都难，更不用说更复杂的ImageNet了，所以“细水长flow”系列也止步于2019年的[《细水长flow之可逆ResNet：极致的暴力美学》](<https://spaces.ac.cn/archives/6482>)。不过，TARFLOW的出现，证明了流模型“尚能一战”，这一次它的生成画风是这样的：

[![TARFLOW的生成效果](<https://spaces.ac.cn/usr/uploads/2025/01/3856218159.jpg>)](<https://spaces.ac.cn/usr/uploads/2025/01/3856218159.jpg>)

TARFLOW的生成效果

相比之下，此前Glow的生成画风是这样的：

[![Glow的生成效果](<https://spaces.ac.cn/usr/uploads/2025/01/3609746299.jpg>)](<https://spaces.ac.cn/usr/uploads/2025/01/3609746299.jpg>)

Glow的生成效果

Glow演示的还只是相对简单的人脸生成，但瑕疵已经很明显了，更不用说更复杂的自然图像生成了，由此可见TARFLOW的进步并不只是一星半点。从数据上看，它的表现也逼近模型模型的最佳表现，超过了GAN的SOTA代表BigGAN：

[![TARFLOW与其他模型的定量对比](<https://spaces.ac.cn/usr/uploads/2025/01/3739397147.png>)](<https://spaces.ac.cn/usr/uploads/2025/01/3739397147.png>)

TARFLOW与其他模型的定量对比

要知道，流模型天然就是一步生成模型，并且不像GAN那样对抗训练，它也是单个损失函数训练到底，某种程度上它的训练比扩散模型还简单。所以，TARFLOW把流模型的效果提升了上来，意味着它同时具备了GAN和扩散模型的优点，同时还有自己独特的优势（可逆、可以用来估计对数似然等）。

## 模型回顾

言归正传，我们来看看TARFLOW用了什么“灵丹妙药”来让流模型重新焕发活力。不过在此之前，我们简要回顾一下流模型的理论基础，更详细的历史溯源，可以参考[《细水长flow之NICE：流模型的基本概念与实现》](<https://spaces.ac.cn/archives/5776>)和[《细水长flow之RealNVP与Glow：流模型的传承与升华》](<https://spaces.ac.cn/archives/5807>)。

从最终目标来看，流模型和GAN都是希望得到一个确定性函数$\boldsymbol{x}=\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$，将随机噪声$\boldsymbol{z}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$映射到目标分布的图片$\boldsymbol{x}$，用概率分布的语言说，就是用如下形式的分布来建模目标分布：
\begin{equation}q_{\boldsymbol{\theta}}(\boldsymbol{x}) = \int \delta(\boldsymbol{x} - \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))q(\boldsymbol{z})d\boldsymbol{z}\label{eq:q-int}\end{equation}
其中$q(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，$\delta()$是[狄拉克δ函数](<https://en.wikipedia.org/wiki/Dirac_delta_function>)。训练概率模型的理想目标是最大似然，即以$-\log q_{\boldsymbol{\theta}}(\boldsymbol{x})$为损失函数。但目前的$q_{\boldsymbol{\theta}}(\boldsymbol{x})$带有积分，只有形式意义，无法用于训练。

此时流模型和GAN就“分道扬镳”了：GAN大致上是用另外一个模型（判别器）去近似$-\log q_{\boldsymbol{\theta}}(\boldsymbol{x})$，这导致了交替训练；流模型则通过设计适当的$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$，让积分$\eqref{eq:q-int}$可以直接算出来。把积分$\eqref{eq:q-int}$算出来需要什么条件呢？设$\boldsymbol{y} = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$，其逆函数为$\boldsymbol{z} = \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{y})$，那么
\begin{equation}d\boldsymbol{z} = \left|\det \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{y}}\right|d\boldsymbol{y} = \left|\det \frac{\partial \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{y})}{\partial \boldsymbol{y}}\right|d\boldsymbol{y}\end{equation}
以及
\begin{equation}\begin{aligned}
q_{\boldsymbol{\theta}}(\boldsymbol{x}) =&\, \int \delta(\boldsymbol{x} - \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))q(\boldsymbol{z})d\boldsymbol{z} \\
=&\, \int \delta(\boldsymbol{x} - \boldsymbol{y})q(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{y}))\left|\det \frac{\partial \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{y})}{\partial \boldsymbol{y}}\right|d\boldsymbol{y} \\
=&\, q(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}))\left|\det \frac{\partial \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})}{\partial \boldsymbol{x}}\right|
\end{aligned}\end{equation}
因此
\begin{equation}-\log q_{\boldsymbol{\theta}}(\boldsymbol{x}) = -\log q(\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})) - \log \left|\det \frac{\partial \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})}{\partial \boldsymbol{x}}\right|\end{equation}
这表明，将积分$\eqref{eq:q-int}$算出来需要两个条件：一、需要知道$\boldsymbol{x} = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$的逆函数$\boldsymbol{z} = \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})$；二、需要计算雅可比矩阵$\frac{\partial \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})}{\partial \boldsymbol{x}}$的行列式。

## 仿射耦合

为此，流模型提出了一个关键设计——“仿射耦合层”：
\begin{equation}\begin{aligned}&\boldsymbol{h}_1 = \boldsymbol{x}_1\\
&\boldsymbol{h}_2 = \exp(\boldsymbol{\gamma}(\boldsymbol{x}_1))\otimes\boldsymbol{x}_2 + \boldsymbol{\beta}(\boldsymbol{x}_1)\end{aligned}\label{eq:couple}\end{equation}
其中$\boldsymbol{x} = [\boldsymbol{x}_1,\boldsymbol{x}_2]$，$\boldsymbol{\gamma}(\boldsymbol{x}_1)$、$\boldsymbol{\beta}(\boldsymbol{x}_1)$是以$\boldsymbol{x}_1$为输入、输出形状跟$\boldsymbol{x}_2$一致的模型，$\otimes$是Hadamard积。上式说的是，将$\boldsymbol{x}$分成两部份，分法随意，也不要求等分，将其中一份原封不动输出，另一边按照指定规则运算输出。注意仿射耦合层是可逆的，其逆为
\begin{equation}\begin{aligned}&\boldsymbol{x}_1 = \boldsymbol{h}_1\\
&\boldsymbol{x}_2 = \exp(-\boldsymbol{\gamma}(\boldsymbol{h}_1))\otimes(\boldsymbol{h}_2 - \boldsymbol{\beta}(\boldsymbol{h}_1))\end{aligned}\end{equation}
这就满足了第一个条件可逆性。另一方面，仿射耦合层的雅可比矩阵是一个下三角阵：
\begin{equation}\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}} = \begin{pmatrix}\frac{\partial \boldsymbol{h}_1}{\partial \boldsymbol{x}_1} & \frac{\partial \boldsymbol{h}_1}{\partial \boldsymbol{x}_2} \\ \frac{\partial \boldsymbol{h}_2}{\partial \boldsymbol{x}_1} & \frac{\partial \boldsymbol{h}_2}{\partial \boldsymbol{x}_2}\end{pmatrix}=\begin{pmatrix}\boldsymbol{I} & \boldsymbol{O} \\
\frac{\partial (\exp(\boldsymbol{\gamma}(\boldsymbol{x}_1))\otimes\boldsymbol{x}_2 + \boldsymbol{\beta}(\boldsymbol{x}_1))}{\partial \boldsymbol{x}_1} & \text{diag}(\exp(\boldsymbol{\gamma}(\boldsymbol{x}_1)))\end{pmatrix}\end{equation}
三角矩阵的行列式，等于对角线元素之积，所以
\begin{equation}\log\left|\det\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}\right| = \sum_i \boldsymbol{\gamma}_i(\boldsymbol{x}_1)\end{equation}
即雅可比矩阵行列式的绝对值对数等于$\boldsymbol{\gamma}(\boldsymbol{x}_1)$的各分量之和，这就满足了第二个条件雅可比矩阵行列式的可计算性。

仿射耦合层首次提出自[RealNVP](<https://papers.cool/arxiv/1605.08803>)，NVP的含义是“Non\-Volume Preserving”，也就是“不保体积”，这个名字对标的是$\boldsymbol{\gamma}(\boldsymbol{x}_1)$恒等于零时的特殊情形，此时称为“加性耦合层”，提出自[NICE](<https://papers.cool/arxiv/1410.8516>)，特点是其雅可比行列式等于1，即加性耦合层是“保体积”的（行列式的几何意义是体积）。

注意，如果直接堆叠多个仿射耦合层，那么$\boldsymbol{x}_1$将一直保持不变，这并不是我们想要的，我们想要做的是将整个$\boldsymbol{x}$映射到标准正态分布。为了解决这个问题，我们每次应用仿射耦合层前，都要输入的分量以某种方式“打乱”，这样就不至于出现始终不变的分量。“打乱”运算对应于置换矩阵变换，行列式绝对值始终为1。

## 核心改进

到目前为止，这些内容都还只是流模型的基础内容，接下来才正式进入到TARFLOW的贡献点。

首先，TARFLOW留意到仿射耦合层$\eqref{eq:couple}$可以推广到多块划分，即将$\boldsymbol{x}$分成更多份$[\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n]$，然后按照类似的规则运算
\begin{equation}\begin{aligned}&\boldsymbol{h}_1 = \boldsymbol{x}_1\\
&\boldsymbol{h}_k = \exp(\boldsymbol{\gamma}_k(\boldsymbol{x}_{< k}))\otimes\boldsymbol{x}_k + \boldsymbol{\beta}_k(\boldsymbol{x}_{< k})\end{aligned}\label{eq:couple-2}\end{equation}
其中$k > 1$，$\boldsymbol{x}_{< k}=[\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_{k-1}]$，其逆运算为
\begin{equation}\begin{aligned}&\boldsymbol{x}_1 = \boldsymbol{h}_1\\
&\boldsymbol{x}_k = \exp(-\boldsymbol{\gamma}_k(\boldsymbol{x}_{< k}))\otimes(\boldsymbol{h}_k - \boldsymbol{\beta}_k(\boldsymbol{x}_{< k}))\end{aligned}\label{eq:couple-2-inv}\end{equation}
类似地，推广版本的仿射耦合层的雅可比行列式绝对值对数为$\boldsymbol{\gamma}_2(\boldsymbol{x}_{< 2}),\cdots,\boldsymbol{\gamma}_n(\boldsymbol{x}_{< n})$所有分量之和，所以流模型要求的两个条件都能满足。这个推广的雏形最早在2016年的[IAF](<https://papers.cool/arxiv/1606.04934>)就提出了，比Glow还早。

可为什么后来的工作鲜往这个方向深入呢？这大体是历史原因。早年CV模型的主要架构是CNN，用CNN的前提是特征满足局部相关性，这就导致了将$\boldsymbol{x}$分块时，往往只考虑在通道（channel）维度划分。因为每一层还有一个必要的打乱运算，一旦选择在长、宽两个空间维度划分，那么随机打乱后特征就失去了局部相关性了，从而没法用CNN。而如果在通道维度划分多份，那么多个通道特征图比较难高效地交互。

然而，到了Transformer时代，情况截然不同了。Transformer的输入本质上是一个无序的向量集合，换言之不依赖局部相关性，因此以Transformer为主架构，我们就可以选择在空间维度划分，这就是Patchify。此外，式$\eqref{eq:couple-2}$中$\boldsymbol{h}_{k} = \cdots(\boldsymbol{x}_{< k})$形式，意味着这是一个Causal模型，这也正好可以用Transformer高效实现。

除了形式上的契合外，在空间维度划分有什么本质好处呢？这就要回到流模型的目标了：$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$将噪声变成图片，逆模型$\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x})$则将图片变成噪声，噪声的特点是随机，说白了就是乱，图片的显著特点是局部相关性，所以图片变噪声的关键之一就是要打乱这种局部相关性，直接在空间维度Patchify配合耦合层自带的打乱操作，无疑是最高效的选择。

所以，式$\eqref{eq:couple-2}$跟Transformer可谓是“一拍即合”、“相得益彰”，这就是TARFLOW前三个字母TAR的含义（Transformer AutoRegressive Flow），也是它的核心改进。

## 加噪去噪

流模型常用的一个训练技巧是加噪，也就是往图片加入微量噪声后再送入模型进行训练。虽然我们将图片视为连续向量，但它实际是以离散的格式存储的，加噪可以进一步平滑这种不连续性，使得图片更接近连续向量。噪声的加入还可以防止模型过度依赖训练数据中的特定细节，从而减少过拟合的风险。

加噪是流模型的基本操作，并不是TARFLOW首先提出的，TARFLOW提出的是去噪。理论上来说，图片加噪后训练出来的流模型，生成结果也是带有噪声的，只不过以往的流模型生成效果也没多好，所以这点噪声也无所谓了。但TARFLOW把流模型的能力提上去后，去噪就“势在必行”了，不然噪声就称为影响效果的主要因素了。

怎么去噪呢？另外训练一个去噪模型？没这个必要。我们在[《从去噪自编码器到生成模型》](<https://spaces.ac.cn/archives/7038>)已经证明了，如果$q_{\boldsymbol{\theta}}(\boldsymbol{x})$是加噪$\mathcal{N}(\boldsymbol{0},\sigma^2 \boldsymbol{I})$训练后的概率密度函数，那么
\begin{equation}\boldsymbol{r}(\boldsymbol{x}) = \boldsymbol{x} + \sigma^2 \nabla_{\boldsymbol{x}} \log q_{\boldsymbol{\theta}}(\boldsymbol{x})\end{equation}
就是去噪模型的理论最优解。所以有了$q_{\boldsymbol{\theta}}(\boldsymbol{x})$后，我们就不用另外训练去噪模型了，直接按照上式计算就可以去噪，这也是流模型的优势之一。而正因为加入了去噪步骤，所以TARFLOW的输入的噪声改为高斯分布，并适当增大了噪声的方差，这也是它性能更好的原因之一。

综上所述，TARFLOW完整的采样流程是
\begin{equation}\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}) ,\quad \boldsymbol{y} =\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}),\quad\boldsymbol{x} = \boldsymbol{y} + \sigma^2 \nabla_{\boldsymbol{y}} \log q_{\boldsymbol{\theta}}(\boldsymbol{y})
\end{equation}

## 延伸思考

至此，TARFLOW相比以往流模型的一些关键变化已经介绍完毕，剩下的一些模型细节，大家自行读原论文就好，如果还有不懂的也可以参考官方开源的代码。

> <strong>Github：[https://github\.com/apple/ml\-tarflow](<https://github.com/apple/ml-tarflow>)</strong>

下面主要谈谈笔者对TARFLOW的一些思考。

首先，需要指出的是，TARFLOW虽然效果上达到了SOTA，但它采样速度实际上不如我们期望，原论文附录提到，在A100上采样32张ImageNet64图片大概需要2分钟。为什么会这么慢呢？我们仔细观察耦合层的逆$\eqref{eq:couple-2-inv}$就会发现，它实际上是一个非线性RNN！非线性RNN只能串行计算，这就是它慢的根本原因。

换句话说，TARFLOW实际上是一个训练快、采样慢的模型，当然如果我们愿意也可以改为训练慢、采样快，总之正向和逆向不可避免会有一侧慢，这是分多块的仿射耦合层的缺点，也是TARFLOW想要进一步推广的主要改进方向。

其次，TARFLOW中的AR一词，容易让人联想到现在主流的自回归式LLM，那么它们俩是否可以整合在一起做多模态生成？说实话很难。因为TARFLOW的AR纯粹是仿射耦合层的要求，而耦合层之前还要打乱，所以它并非一个真正的Causal模型，反而是彻头彻尾的Bi\-Directional模型，所以它并不好跟文本的AR强行整合在一起。

总的来说，如果TARFLOW进一步将采样速度也提上去，那么它将会是一个非常有竞争力的纯视觉生成模型。因为除了训练简单和效果优异，流模型的可逆性还有另外一个优点，就是[《The Reversible Residual Network: Backpropagation Without Storing Activations》](<https://papers.cool/arxiv/1707.04585>)所提的反向传播可以完全不用存激活值，并且重计算的成本比普通模型低得多。

至于它有没有可能成为多模态LLM的统一架构，只能说现在还不大明朗。

## 文艺复兴

最后，谈谈深度学习模型的“文艺复兴”。

近年来，已经有不少工作尝试结合当前最新的认知，来反思和改进一些看起来已经落后的模型，并得到了一些新的结果。除了TARFLOW试图让流模型重新焕发活力外，最近还有[《The GAN is dead; long live the GAN\! A Modern GAN Baseline》](<https://papers.cool/arxiv/2501.05441>)对GAN的各种排列组合去芜存菁，得到了同样有竞争力的结果。

更早一些，还有[《Improved Residual Networks for Image and Video Recognition》](<https://papers.cool/arxiv/2004.04989>)、[《Revisiting ResNets: Improved Training and Scaling Strategies》](<https://papers.cool/arxiv/2103.07579>)等工作让ResNet更上一层楼，甚至还有[《RepVGG: Making VGG\-style ConvNets Great Again》](<https://papers.cool/arxiv/2101.03697>)让VGG经典再现。当然，SSM、线性Attention等工作也不能不提，它们代表着RNN的“文艺复兴”。

期待这种百花齐放的“复兴”潮能更热烈一些，它能让我们获得对模型更全面和准确的认知。
