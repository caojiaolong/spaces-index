# 浅谈Transformer的初始化、参数化与标准化

> 作者：苏剑林 · 科学空间 · 2021-08-17
>
> 原文：<https://spaces.ac.cn/archives/8620>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前几天在训练一个新的Transformer模型的时候，发现怎么训都不收敛了。经过一番debug，发现是在做Self Attention的时候$\boldsymbol{Q}\boldsymbol{K}^{\top}$之后忘记除以$\sqrt{d}$了，于是重新温习了一下为什么除以$\sqrt{d}$如此重要的原因。当然，Google的T5确实是没有除以$\sqrt{d}$的，但它依然能够正常收敛，那是因为它在初始化策略上做了些调整，所以这个事情还跟初始化有关。

藉着这个机会，本文跟大家一起梳理一下模型的初始化、参数化和标准化等内容，相关讨论将主要以Transformer为心中展开。

## 采样分布

初始化自然是随机采样的的，所以这里先介绍一下常用的采样分布。一般情况下，我们都是从指定均值和方差的随机分布中进行采样来初始化。其中常用的随机分布有三个：正态分布（Normal）、均匀分布（Uniform）和截尾正态分布（Truncated Normal）。

显然，正态分布和均匀分布都是很常见的分布。其中正态分布通常记为$\mathcal{N}(\mu,\sigma^2)$，其中$\mu$是均值而$\sigma^2$是方差；区间$[a,b]$上的均匀分布一般记为$U[a,b]$，其均值为$\frac{a+b}{2}$、方差为$\frac{(b-a)^2}{12}$，所以指定均值$\mu$和方差$\sigma^2$的话，那么对应的均匀分布是$U[\mu-\sqrt{3}\sigma,\mu + \sqrt{3}\sigma]$。

一般来说，正态分布的采样结果更多样化一些，但它理论上是无界的，如果采样到绝对值过大的结果可能不利于优化；相反均匀分布是有界的，但采样结果通常更单一。于是就出现了结合两者优点的“[截尾正态分布](<https://en.wikipedia.org/wiki/Truncated_normal_distribution>)”。截尾正态分布既指定均值$\mu$和方差$\sigma^2$，也需要指定区间$[a,b]$，它从$\mathcal{N}(\mu,\sigma^2)$中采样，如果采样结果在$[a,b]$中，那么保留该结果，否则重复采样直到采样结果落到$[a,b]$中。

在tensorflow自带的<code>tf.random.truncated_normal</code>中，写死了$a=\mu-2\sigma,b=\mu+2\sigma$。所以根据公式可以算得，该函数采样结果的实际均值依然为$\mu$，但实际方差是$\gamma\sigma^2$，其中：
\begin{equation}\gamma=\frac{\int_{-2}^2 e^{-x^2/2}x^2 dx}{\int_{-2}^2 e^{-x^2/2} dx}=0.7737413\dots\end{equation}
如果要得到方差为$\sigma^2$的采样结果，那么传入函数的标准差要为$\frac{\sigma}{\sqrt{\gamma}}=1.1368472\dots\sigma$。

## 稳定二阶矩

在之前的文章[《从几何视角来理解模型参数的初始化策略》](<https://spaces.ac.cn/archives/7180>)中笔者从几何角度分析了已有的初始化方法，大致的思想是特定的随机矩阵近似于一个正交矩阵，从而能保证初始阶段模型的稳定性。不过几何视角虽然有着直观的优点，但通常难以一般化拓展，因此接下来我们还是从代数的角度来理解初始化方法。

在一般的教程中，推导初始化方法的思想是尽量让输入输出具有同样的均值和方差，通常会假设输入是均值为0、方差为1的随机向量，然后试图让输出的均值为0、方差为1。不过，笔者认为这其实是没有必要的，而且对于某些非负的激活函数来说，根本就做不到均值为0。事实上，我们只需要一个衡量某个指标是否“消失”或者“爆炸”的指标，0均值、1方差是非必要的，这里我们用二阶（原点）矩来代替，它可以看成是L2模长的变体，跟方差的作用类似，都可以用来衡量指标是否“消失”或者“爆炸”，但它相对来说更普适和简单。

现在，我们考察无激活函数的全连接层（设输入节点数为$m$，输出节点数为$n$）
\begin{equation} y_j = b_j + \sum_i x_i w_{i,j}\end{equation}
简单起见，我们通常用全零初始化偏置项$b_j$，并且将$w_{i,j}$的均值$\mathbb{E}[w_{i,j}]$也设为0，这有助于简化下面的结果，但并不是非如此不可，只是这确实是一种比较简明的选择。我们计算二阶矩：
\begin{equation}\begin{aligned}
\mathbb{E}[y_j^2] =&\, \mathbb{E}\left[\left(\sum_i x_i w_{i,j}\right)^2\right]=\mathbb{E}\left[\left(\sum_{i_1} x_{i_1} w_{i_1,j}\right)\left(\sum_{i_2} x_{i_2} w_{i_2,j}\right)\right]\\
=&\,\mathbb{E}\left[\sum_{i_1, i_2} (x_{i_1}x_{i_2}) (w_{i_1,j} w_{i_2,j})\right] = \sum_{i_1, i_2} \mathbb{E}[x_{i_1}x_{i_2}] \mathbb{E}[w_{i_1,j} w_{i_2,j}]
\end{aligned}\end{equation}
注意$w_{i_1,j},w_{i_2,j}$是独立同分布的，所以当$i_1\neq i_2$时$\mathbb{E}[w_{i_1,j}w_{i_2,j}]=\mathbb{E}[w_{i_1,j}]\mathbb{E}[w_{i_2,j}]=0$，因此只需要考虑$i_1=i_2=i$的情况。假设输入的二阶矩为1，那么
\begin{equation}
\mathbb{E}[y_j^2] = \sum_{i} \mathbb{E}[x_i^2] \mathbb{E}[w_{i,j}^2]= m\mathbb{E}[w_{i,j}^2]\label{eq:m}\end{equation}
所以要使得$\mathbb{E}[y_j^2]$为1，那么$\mathbb{E}[w_{i,j}^2]=1/m$，综合均值为0的假设，我们得到$w_{i,j}$初始化策略为“从均值为0、方差为$1/m$的随机分布中独立重复采样”，这就是Lecun初始化。注意，该过程我们并没有对输入的均值做任何假设，因此它哪怕全是非负的也没问题。

## 激活函数

当然，这仅仅是无激活函数的场景，如果加上激活函数考虑，那么需要具体情形具体分析。比如激活函数是$\text{relu}$的话，我们可以假设大致有一半的$y_j$被置零了，于是二阶矩的估计结果是式$\eqref{eq:m}$的一半：
\begin{equation}
\mathbb{E}[y_j^2] = \frac{m}{2}\mathbb{E}[w_{i,j}^2]\end{equation}
从而使得二阶矩不变的初始化方差为$2/m$，这就是专门针对$\text{relu}$网络的He初始化。

不过，要是激活函数是$\text{elu},\text{gelu}$等，那么分析起来就没那么简单了；而如果激活函数是$\tanh,\text{sigmoid}$的话，那么根本找不到任何初始化能使得二阶矩为1。这种情况下如果还想保持二阶矩不变的话，那么可以考虑的方案是“微调激活函数的定义”。

以$\text{sigmoid}$为例，假设输入的均值为0、方差为1，而我们依旧采用“均值为0、方差为$1/m$”的初始化，那么激活前的输出也是均值为0、方差为1，于是我们可以用标准正态分布估算$\text{sigmoid}$后的二阶矩：
\begin{equation}\int_{-\infty}^{\infty} \frac{e^{-x^2/2}}{\sqrt{2\pi}}\text{sigmoid}(x)^2 dx = 0.2933790\dots\end{equation}
也就是说，在此假设下模型激活后的二阶矩大致为$0.293379$。所以，如果我们想要保持输出的二阶矩大致不变，那么可以将输出结果再除以$\sqrt{0.293379}$，换言之，激活函数由$\text{sigmoid}(x)$改为$\frac{\text{sigmoid}(x)}{\sqrt{0.293379}}$，这便是“微调”后的激活函数。如果你觉得有必要，也可以通过减去一个常数的方式，将输出的均值也变为0。

记得2017年时有一篇“轰动一时”的论文[《Self\-Normalizing Neural Networks》](<https://papers.cool/arxiv/1706.02515>)提出了一个激活函数$\text{selu}$，它其实也是基于同样思路进行“微调”后的$\text{elu}$函数，其形式如下：
\begin{equation}\text{selu}(x)=\lambda\left\{\begin{aligned}
&x,& (x > 0) \\
&\alpha e^{x}-\alpha, &(x\leq 0)
\end{aligned}\right.\end{equation}
其中$\lambda=1.0507\dots,\alpha=1.6732\dots$。它当初“轰动一时”，一是因为它号称不用Batch Normalization等手段就实现了网络的自动标准化，二是因为它附带的几十页数学推导比较“唬人”。而如果从上述视角来看，它就是引入两个参数来微调$\text{elu}$函数，使得标准正态分布为输入时，输出的激活值的均值为0、方差为1，所以它顶多算是一种比较好的初始化罢了，因此也就只能轰动“一时”了。它两个参数，我们同样可以用Mathematica数值求解：



<pre><code class="language-python">f[x_] = Exp[-x^2/2]/Sqrt[2 Pi];
s[x_] = Piecewise[{{\[Lambda]*x, 
     x &gt; 0}, {\[Lambda]*\[Alpha]*(Exp[x] - 1), x &lt;= 0}}];
x1 = Integrate[f[x]*s[x], {x, -Infinity, Infinity}];
x2 = Integrate[f[x]*s[x]^2, {x, -Infinity, Infinity}];
N[Solve[{x1 == 0, x2 == 1}, {\[Lambda], \[Alpha]}], 20]</code></pre>



## 直接标准化

当然，相比这种简单的“微调”，更直接的处理方法是各种Normalization方法，如Batch Normalization、Instance Normalization、Layer Normalization等，这类方法直接计算当前数据的均值方差来将输出结果标准化，而不用事先估计积分，有时候我们也称其为“归一化”。这三种标准化方法大体上都是类似的，除了Batch Normalization多了一步滑动平均预测用的均值方差外，它们只不过是标准化的维度不一样，比如NLP尤其是Transformer模型用得比较多就是Layer Normalization是：
\begin{equation}y_{i,j,k} = \frac{x_{i,j,k} - \mu_{i,j}}{\sqrt{\sigma_{i,j}^2 + \epsilon}}\times\gamma_k + \beta_k,\quad \mu_{i,j} = \frac{1}{d}\sum_{k=1}^d x_{i,j,k},\quad \sigma_{i,j}^2 = \frac{1}{d}\sum_{k=1}^d (x_{i,j,k}-\mu_{i,j})^2\end{equation}
其他就不再重复描述了。关于这类方法起作用的原理，有兴趣的读者可以参考笔者之前的[《BN究竟起了什么作用？一个闭门造车的分析》](<https://spaces.ac.cn/archives/6992>)。

这里笔者发现了一个有意思的现象：Normalization一般都包含了减均值（center）和除以标准差（scale）两个部分，但近来的一些工作逐渐尝试去掉center这一步，甚至有些工作的结果显示去掉center这一步后性能还略有提升。

比如2019年的论文[《Root Mean Square Layer Normalization》](<https://papers.cool/arxiv/1910.07467>)比较了去掉center后的Layer Normalization，文章称之为RMS Norm，形式如下：
\begin{equation}y_{i,j,k} = \frac{x_{i,j,k}}{\sqrt{\sigma_{i,j}^2 + \epsilon}}\times\gamma_k,\quad \sigma_{i,j}^2 = \frac{1}{d}\sum_{k=1}^d x_{i,j,k}^2\end{equation}
可以看出，RMS Norm也就是L2 Normalization的简单变体而已，但这篇论文总的结果显示：RMS Norm比Layer Normalization更快，效果也基本一致。

除了这篇文章外，RMS Norm还被Google用在了T5中，并且在另外的一篇文章[《Do Transformer Modifications Transfer Across Implementations and Applications?》](<https://papers.cool/arxiv/2102.11972>)中做了比较充分的对比实验，显示出RMS Norm的优越性。这样看来，未来RMS Norm很可能将会取代Layer Normalization而成为Transformer的标配。

无独有偶，同样是2019年的论文[《Analyzing and Improving the Image Quality of StyleGAN》](<https://papers.cool/arxiv/1912.04958>)提出了StyleGAN的改进版StyleGAN2，里边发现所用的Instance Normalization会导致部分生成图片出现“水珠”，他们最终去掉了Instance Normalization并换用了一个叫“Weight demodulation”的东西，但他们发现如果保留Instance Normalization单去掉center操作也能改善这个现象。这也为Normalization中的center操作可能会带来负面效果提供了佐证。

一个直观的猜测是，center操作，类似于全连接层的bias项，储存到的是关于预训练任务的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以T5不仅去掉了Layer Normalization的center操作，它把每一层的bias项也都去掉了。

## NTK参数化

回到全连接层的Xavier初始化，它说我们要用“均值为0、方差为$1/m$的随机分布”初始化。不过，除了直接用这种方式的初始化外，我们还可以有另外一种参数化的方式：用“均值为0、方差为1的随机分布”来初始化，但是将输出结果除以$\sqrt{m}$，即模型变为：
\begin{equation} y_j = b_j + \frac{1}{\sqrt{m}}\sum_i x_i w_{i,j}\end{equation}
这在高斯过程中被称为“NTK参数化”，可以参考的论文有[《Neural Tangent Kernel: Convergence and Generalization in Neural Networks》](<https://papers.cool/arxiv/1806.07572>)、[《On the infinite width limit of neural networks with a standard parameterization》](<https://papers.cool/arxiv/2001.07301>)等。不过对于笔者来说，第一次看到这种操作是在PGGAN的论文[《Progressive Growing of GANs for Improved Quality, Stability, and Variation》](<https://papers.cool/arxiv/1710.10196>)中。

很显然，利用NTK参数化，我们可以将所有参数都用标准方差初始化，但依然保持二阶矩不变，甚至前面介绍的“微调激活函数”，也可以看成是NTK参数化的一种。一个很自然的问题是：NTK参数化跟直接用Xavier初始化相比，有什么好处吗？

理论上，是有一点好处的。利用NTK参数化后，所有参数都可以用方差为1的分布初始化，这意味着每个参数的量级大致都是相同的$\mathcal{O}(1)$级别，于是我们可以设置较大的学习率，比如$10^{-2}$，并且如果使用自适应优化器，其更新量大致是$\frac{\text{梯度}}{\sqrt{\text{梯度}\otimes\text{梯度}}}\times\text{学习率}$，那么我们就知道$10^{-2}$的学习率每一步对参数的调整程度大致是$1\%$。总的来说，NTK参数化能让我们更平等地处理每一个参数，并且比较形象地了解到训练的更新幅度，以便我们更好地调整参数。

说到这里，我们就可以讨论本文开头的问题了：为什么Attention中除以$\sqrt{d}$这么重要？对于两个$d$维向量$\boldsymbol{q},\boldsymbol{k}$，假设它们都采样自“均值为0、方差为1”的分布，那么它们的内积的二阶矩是：
\begin{equation}\begin{aligned}
\mathbb{E}[(\boldsymbol{q}\cdot \boldsymbol{k})^2]=&\,\mathbb{E}\left[\left(\sum_{i=1}^d q_i k_i\right)^2\right] = \mathbb{E}\left[\left(\sum_i q_i k_i\right)\left(\sum_j q_j k_j\right)\right]\\
=&\,\mathbb{E}\left[\sum_{i,j} (q_i q_j) (k_i k_j)\right] = \sum_{i,j} \mathbb{E}[q_i q_j] \mathbb{E}[k_i k_j]\\
=&\,\sum_i \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = d
\end{aligned}\end{equation}
也就是内积的二阶矩为$d$，由于均值也为0，所以这也意味着方差也是$d$。Attention是内积后softmax，主要设计的运算是$e^{\boldsymbol{q}\cdot \boldsymbol{k}}$，我们可以大致认为内积之后、softmax之前的数值在$-3\sqrt{d}$到$3\sqrt{d}$这个范围内，由于$d$通常都至少是64，所以$e^{3\sqrt{d}}$比较大而$e^{-3\sqrt{d}}$比较小，因此经过softmax之后，Attention的分布非常接近一个one hot分布了，这带来严重的梯度消失问题，导致训练效果差。

相应地，解决方法就有两个，一是像NTK参数化那样，在内积之后除以$\sqrt{d}$，使$\boldsymbol{q}\cdot \boldsymbol{k}$的方差变为1，对应$e^3,e^{-3}$都不至于过大过小，这样softmax之后也不至于变成one hot而梯度消失了，这也是常规的Transformer如BERT里边的Self Attention的做法；另外就是不除以$\sqrt{d}$，但是初始化$\boldsymbol{q},\boldsymbol{k}$的全连接层的时候，其初始化方差要多除以一个$\sqrt{d}$，这同样能使得使$\boldsymbol{q}\cdot \boldsymbol{k}$的初始方差变为1，T5采用了这样的做法。

## 残差连接

最后，不得不讨论的是残差$x + F(x)$的相关设计。容易证明，如果$x$的方差（二阶矩同理）为$\sigma_1^2$而$F(x)$的方差为$\sigma_2^2$，并且假设两者相互独立，那么$x + F(x)$的方差为$\sigma_1^2 + \sigma_2^2$。也就是说，残差会进一步放大方差，所以我们也要想相应的策略缩小其方差。

一种比较朴素的方案是直接在残差后面加个Normalization操作：
\begin{equation}x_{t+1} = \text{Norm}(x_t + F_t(x_t))\end{equation}
这我们可以称为Post Norm结构，它也是原版Transformer和BERT所使用的设计。然而，这种做法虽然稳定了前向传播的方差，但事实上已经严重削弱了残差的恒等分支，所以反而失去了残差“易于训练”的优点，通常要warmup并设置足够小的学习率才能使它收敛。

怎么理解这一点呢？假设初始状态下$x,F(x)$的方差均为1，那么$x+F(x)$的方差就是2，而Normalization操作负责将方差重新降为1，这就说明初始阶段Post Norm相当于
\begin{equation}x_{t+1} = \frac{x_t + F_t(x_t)}{\sqrt{2}}\end{equation}
递归下去，我们得到
\begin{equation}\begin{aligned}
x_l =&\, \frac{x_{l-1}}{\sqrt{2}} + \frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\
=&\, \frac{x_{l-2}}{2} + \frac{F_{l-2}(x_{l-2})}{2} + \frac{F_{l-1}(x_{l-1})}{\sqrt{2}} \\
=&\, \cdots \\
=&\,\frac{x_0}{2^{l/2}} + \frac{F_0(x_0)}{2^{l/2}} + \frac{F_1(x_1)}{2^{(l-1)/2}} + \frac{F_2(x_2)}{2^{(l-2)/2}} + \cdots + \frac{F_{l-1}(x_{l-1})}{2^{1/2}}
\end{aligned}\end{equation}
看到问题了没？本来残差的意思是给前面的层搞一条“绿色通道”，让梯度可以更直接地回传，但是在Post Norm中，这条“绿色通道”被严重削弱了，越靠近前面的通道反而权重越小，残差“名存实亡”，因此还是不容易训练。相关的分析还可以参考论文[《On Layer Normalization in the Transformer Architecture》](<https://papers.cool/arxiv/2002.04745>)。

一个针对性的改进称为Pre Norm，它的思想是“要用的时候才去标准化”，其形式为
\begin{equation}x_{t+1} = x_t + F_t(\text{Norm}(x_t))\end{equation}
类似地，迭代展开之后我们可以认为初始阶段有
\begin{equation}
x_l = x_0 + F_0(x_0) + F_1(x_1/\sqrt{2}) + F_2(x_2/\sqrt{3}) + \cdots + F_{l-1}(x_{l-1}/\sqrt{l})\end{equation}
这样一来，起码每一条残差通道都是平权的，残差的作用会比Post Norm更加明显，所以它也更好优化。当然，这样最后的$x_l$方差将会很大，所以在接预测层之前$x_l$也还要加个Normalization。

就笔者看来，不管是Post Norm还是Pre Norm都不够完美，因为它们都无法在初始阶段保持一个恒等函数。在笔者看来，最漂亮的方法应该是引入一个初始化为0的标量参数$\alpha_t $，使得
\begin{equation}x_{t+1} = x_t + \alpha_t F_t(x_t)\end{equation}
然后再逐渐更新$\alpha_t$。这样在初始阶段，我们就能确保模型是一个恒等函数，从而也就不会有方差的问题了。这个技巧后来出现在两篇论文中，在[《Batch Normalization Biases Residual Blocks Towards the Identity Function in Deep Networks》](<https://papers.cool/arxiv/2002.10444>)中它被称为SkipInit，而在[《ReZero is All You Need: Fast Convergence at Large Depth》](<https://papers.cool/arxiv/2003.04887>)中它被称为ReZero，两篇论文相隔不到一个月，它们的结果都显示这样处理后基本都可以直接替代掉残差中的Normalization操作。此外，[《Fixup Initialization: Residual Learning Without Normalization》](<https://papers.cool/arxiv/1901.09321>)提出过一个叫Fixup的方法，它是将每个残差分支的最后一层用全零初始化，和SkipInit、ReZero也有一定的相通之处。

对于$\alpha_t$的更新，不管是SkipInit还是ReZero，都将它视为模型参数跟着其他模型参数一起更新，笔者一开始也是这样想的。后来发现，$\alpha_t$的地位跟其他参数是不对等的，不能一概而论，比如通过前面介绍的NTK参数化，其他参数我们可以用很大的学习率，但很显然$\alpha_t$不应该用很大的学习率。此外我们知道，如果能成功训练，那么不管是Post Norm还是Pre Norm的效果也都很好（对应$\alpha_t=1$），所以这种残差模式的选择纯粹是一个初始化问题而不是拟合能力问题。综合这几点，笔者后来干脆让$\alpha_t$以固定的、很小的步长慢慢递增，直到增加到$\alpha_t=1$就固定下来，在笔者的实验结果中，这种更新模式取得了最优的结果。

## 炼丹路漫漫

本文讨论了模型的初始化、参数化与标准化等相关问题，希望能对大家的炼丹调参有一定的参考价值。炼丹之路漫漫无际，除了这些内容外，可调的东西还有很多，比如学习率、优化器、数据扩增等。愿各位读者在炼丹的道路上一帆风顺哈～
