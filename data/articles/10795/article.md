# 高阶MuP：更简明但更高明的谱条件缩放

> 作者：苏剑林 · 科学空间 · 2025-03-24
>
> 原文：<https://spaces.ac.cn/archives/10795>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《初探MuP：超参数的跨模型尺度迁移规律》](<https://spaces.ac.cn/archives/10770>)中，我们基于前向传播、反向传播、损失增量和特征变化的尺度不变性推导了MuP（Maximal Update Parametrization）。可能对于部分读者来说，这一过程还是显得有些繁琐，但实际上它比原始论文已经明显简化。要知道，我们是在单篇文章内相对完整地介绍的MuP，而MuP的论文实际上是作者Tensor Programs系列论文的第5篇！

不过好消息是，作者在后续的研究[《A Spectral Condition for Feature Learning》](<https://papers.cool/arxiv/2310.17813>)中，发现了一种新的理解方式（下称“谱条件”），它比MuP的原始推导和笔者的推导都更加直观和简洁，但却能得到比MuP更丰富的结果，可谓MuP的高阶版本，简明且不失高明的代表作。

## 准备工作

顾名思义，谱条件（Spectral Condition）跟谱范数（Spectral Norm）相关，它的出发点是谱范数的一个基本不等式：
\begin{equation}\Vert\boldsymbol{x}\boldsymbol{W}\Vert_2\leq \Vert\boldsymbol{x}\Vert_2 \Vert\boldsymbol{W}\Vert_2\label{neq:spec-2}\end{equation}
其中$\boldsymbol{x}\in\mathbb{R}^{d_{in}}, \boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$，至于$\Vert\cdot\Vert_2$，我们可以叫它“$2$范数”。对于$\boldsymbol{x},\boldsymbol{x}\boldsymbol{W}$来说，它们都是向量，$2$范数就是向量模长；而$\boldsymbol{W}$是一个矩阵，它的$2$范数也称为谱范数，它等于让$\Vert\boldsymbol{x}\boldsymbol{W}\Vert_2\leq C\Vert\boldsymbol{x}\Vert_2$恒成立的最小常数$C$。换句话说，上述不等式其实是谱范数定义的直接推论，无需额外证明。

关于谱范数，大家还可以看[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)、[《低秩近似之路（二）：SVD》](<https://spaces.ac.cn/archives/10407#%E7%9F%A9%E9%98%B5%E8%8C%83%E6%95%B0>)等博文，这里不再展开介绍。矩阵还有一个更简单的$F$范数，它是向量模长的简单推广：
\begin{equation}\Vert \boldsymbol{W}\Vert_F = \sqrt{\sum_{i=1}^{d_{in}}\sum_{j=1}^{d_{out}}W_{i,j}^2}\end{equation}
从奇异值角度看，谱范数等于矩阵最大的奇异值，而$F$范数等于矩阵全体奇异值的平方和的平方根。类似地，我们还可以定义“核范数（Nuclear Norm）”，它等于全体奇异值的和：
\begin{equation}\Vert \boldsymbol{W}\Vert_* = \sum_{i=1}^{\min(d_{in}, d_{out})} \sigma_i\end{equation}
像谱范数、$F$范数、核范数等可以由奇异值表示出来的矩阵范数，它们都属于“[Schatten\-p范数](<https://en.wikipedia.org/wiki/Schatten_norm>)”的一种。最后，我们来定义RMS（Root Mean Square），它是向量模长的变体：
\begin{equation}\Vert\boldsymbol{x}\Vert_{RMS} = \sqrt{\frac{1}{d_{in}}\sum_{i=1}^{d_{in}} x_i^2} = \frac{1}{\sqrt{d_{in}}}\Vert \boldsymbol{x}\Vert_2 \end{equation}
如果要推广到矩阵，那就是$\Vert\boldsymbol{W}\Vert_{RMS} = \Vert \boldsymbol{W}\Vert_F/\sqrt{d_{in} d_{out}}$。其实从名字就很好理解，向量模长或矩阵$F$范数，我们可以称为“Root Sum Square”，而RMS就是把Sum换成Mean，它主要用来作为向量或矩阵元素的平均尺度指标。现在把RMS代入不等式$\eqref{neq:spec-2}$，可以得到
\begin{equation}\Vert\boldsymbol{x}\boldsymbol{W}\Vert_{RMS}\leq \sqrt{\frac{d_{in}}{d_{out}}}\Vert\boldsymbol{x}\Vert_{RMS} \Vert\boldsymbol{W}\Vert_2\label{neq:spec-rms}\end{equation}

## 期望性质

我们之前推导MuP的思路，是仔细分析<strong>前向传播</strong>、<strong>反向传播</strong>、<strong>损失增量</strong>和<strong>特征变化</strong>的形式，通过调整初始化和学习率来实现它们的尺度不变性。谱条件对其“去芜存菁”后发现，只要<strong>前向传播</strong>和<strong>特征变化</strong>两点足矣。

简单来说，谱条件期望<u>每一层的<strong>输出</strong>和<strong>增量</strong>都具有尺度不变性</u>。怎么理解这句话呢？如果我们将每一层简记为$\boldsymbol{x}_k= f(\boldsymbol{x}_{k-1}; \boldsymbol{W}_k)$为例，这句话可以翻译成“期望每个$\Vert\boldsymbol{x}_k\Vert_{RMS}$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}$都是$\Theta(1)$”的（$\Theta$是“[Big Theta Notation](<https://en.wikipedia.org/wiki/Big_O_notation#Family_of_Bachmann%E2%80%93Landau_notations>)”）：

> 1、$\Vert\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$好理解，它代表着前向传播的稳定性，上一篇文章的推导也包含这个要求；
> 
> 
> 
> 2、$\Delta\boldsymbol{x}_k$表示参数变化引起的$\boldsymbol{x}_k$变化量，所以$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$融合了反向传播和特征变化的要求。

可能有读者疑问：那是不是至少应该还有个“损失增量”的要求？并不需要。事实上，我们可以证明，如果每一层的$\Vert\boldsymbol{x}_k\Vert_{RMS}$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}$都是$\Theta(1)$，那么$\Delta\mathcal{L}$自动就是$\Theta(1)$的。这正是谱条件思想第一个美妙之处，它将原本推导MuP需要的四个条件降低到两个，减少了分析步骤。

证明并不困难，这里的关键是我们假设了<u>每一层</u>都成立$\Vert\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$，那么最后一层自然也成立。假设模型一共有$K$层，单个样本损失函数为$\ell$，那么它是$\boldsymbol{x}_K$的函数即$\ell(\boldsymbol{x}_K)$，简单起见这里省掉了标签输入，因为对下面的分析来说它并非变量。

根据假设，$\Vert\boldsymbol{x}_K\Vert_{RMS}$是$\Theta(1)$的，那么$\ell(\boldsymbol{x}_K)$自然是$\Theta(1)$的；又因为$\Vert\Delta\boldsymbol{x}_K\Vert_{RMS}$是$\Theta(1)$的，所以$\Vert\boldsymbol{x}_K + \Delta\boldsymbol{x}_K\Vert_{RMS}\leq \Vert\boldsymbol{x}_K\Vert_{RMS} + \Vert\Delta\boldsymbol{x}_K\Vert_{RMS}$也是$\Theta(1)$的，从而$\ell(\boldsymbol{x}_K + \Delta\boldsymbol{x}_K)$是$\Theta(1)$的，于是
\begin{equation}\Delta \ell = \ell(\boldsymbol{x}_K + \Delta\boldsymbol{x}_K) - \ell(\boldsymbol{x}_K) = \Theta(1)\end{equation}
所以，单个样本的损失增量$\Delta \ell$是$\Theta(1)$的，而$\Delta\mathcal{L}$是全体$\Delta \ell$的平均，所以它也是$\Theta(1)$的。这样我们就证明了$\Vert\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$和$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$自动包含了$\Delta\mathcal{L}=\Theta(1)$，原理说白了就是$\Delta\mathcal{L}$是最后一层输出及其增量的函数，它们都稳定了，$\Delta\mathcal{L}$自然就稳定了。

## 谱条件

接着，我们看如何成立两个期望性质。由于神经网络以矩阵乘法为主，所以我们先考虑最简单的线性层$\boldsymbol{x}_k = \boldsymbol{x}_{k-1} \boldsymbol{W}_k$，其中$\boldsymbol{W}_k\in\mathbb{R}^{d_{k-1}\times d_k}$。为了成立$\Vert\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$的条件，谱条件没有像传统初始化分析一样去假设独立同分布然后算期望方差等，而是直接应用不等式$\eqref{neq:spec-rms}$：
\begin{equation}\Vert\boldsymbol{x}_k\Vert_{RMS}\leq \sqrt{\frac{d_{k-1}}{d_k}}\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}\, \Vert\boldsymbol{W}_k\Vert_2\end{equation}
注意这个不等式是可能取到等号的，并且某种意义上是最精准的，所以如果输入的$\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}$已经是$\Theta(1)$，那么为了使输出的$\Vert\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$，那么就要
\begin{equation}\sqrt{\frac{d_{k-1}}{d_k}}\Vert\boldsymbol{W}_k\Vert_2 = \Theta(1)\quad\Rightarrow\quad \Vert\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)\label{eq:spec-c1}\end{equation}
这就提出了第一个<strong>谱条件</strong>——对$\boldsymbol{W}_k$的谱范数要求。它无关初始化和分布假设，完全是分析和代数的结果，这是笔者认为谱条件的第二个美妙之处——简化分析过程。当然，这里省略了谱范数的基础内容，补上的话整个篇幅不见得比分布假设下的分析短，但分布假设终究显得局限性大，不如这里的代数框架灵活。

分析完$\Vert\boldsymbol{x}_k\Vert_{RMS}$后，就轮到$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}$了，增量的$\Delta\boldsymbol{x}_k$的来源有两部分，一是参数由$\boldsymbol{W}_k$变为$\boldsymbol{W}_k+\Delta \boldsymbol{W}_k$，二是输入$\boldsymbol{x}_{k-1}$的参数变化导致它从$\boldsymbol{x}_{k-1}$变为$\boldsymbol{x}_{k-1} + \Delta\boldsymbol{x}_{k-1}$，所以
\begin{equation}\begin{aligned}
\Delta\boldsymbol{x}_k =&\, (\boldsymbol{x}_{k-1} + \Delta\boldsymbol{x}_{k-1})(\boldsymbol{W}_k+\Delta \boldsymbol{W}_k) - \boldsymbol{x}_{k-1}\boldsymbol{W}_k \\[5pt]
=&\, \boldsymbol{x}_{k-1} (\Delta \boldsymbol{W}_k)  + (\Delta\boldsymbol{x}_{k-1})\boldsymbol{W}_k + (\Delta\boldsymbol{x}_{k-1})(\Delta \boldsymbol{W}_k)
\end{aligned}\end{equation}
所以
\begin{equation}\begin{aligned}
\Vert\Delta\boldsymbol{x}_k\Vert_{RMS} =&\, \Vert\boldsymbol{x}_{k-1} (\Delta \boldsymbol{W}_k)  + (\Delta\boldsymbol{x}_{k-1})\boldsymbol{W}_k + (\Delta\boldsymbol{x}_{k-1})(\Delta \boldsymbol{W}_k)\Vert_{RMS} \\[5pt]
\leq&\, \Vert\boldsymbol{x}_{k-1} (\Delta \boldsymbol{W}_k)\Vert_{RMS}  + \Vert(\Delta\boldsymbol{x}_{k-1})\boldsymbol{W}_k\Vert_{RMS} + \Vert(\Delta\boldsymbol{x}_{k-1})(\Delta \boldsymbol{W}_k)\Vert_{RMS} \\[5pt]
\leq&\, \sqrt{\frac{d_{k-1}}{d_k}}\left({\begin{gathered}\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}\,\Vert\Delta \boldsymbol{W}_k\Vert_2 + \Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}\,\Vert \boldsymbol{W}_k\Vert_2 \\[5pt]
+ \Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}\,\Vert\Delta \boldsymbol{W}_k\Vert_2\end{gathered}} \right)
\end{aligned}\end{equation}
逐项分析一下
\begin{equation}\underbrace{\Vert\boldsymbol{x}_{k-1}\Vert_{RMS}}_{\Theta(1)}\,\Vert\Delta \boldsymbol{W}_k\Vert_2 + \underbrace{\Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}}_{\Theta(1)}\,\underbrace{\Vert \boldsymbol{W}_k\Vert_2}_{\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)} + \underbrace{\Vert\Delta\boldsymbol{x}_{k-1}\Vert_{RMS}}_{\Theta(1)}\,\Vert\Delta \boldsymbol{W}_k\Vert_2\end{equation}
由此可见，要想$\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)$，那么就需要
\begin{equation}\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)\label{eq:spec-c2}\end{equation}
这就是第二个<strong>谱条件</strong>——对$\Delta\boldsymbol{W}_k$的谱范数要求。

上面的分析没有考虑非线性，事实上只要激活函数是Element\-wise的，并且导数能够被某个常数Bound住（常用的ReLU、Sigmoid、Tanh等激活函数都满足），那么即便考虑非线性激活函数的结果也是一致，这就是上一篇文章分析所说的“激活函数的影响是尺度无关的”。如果读者还不放心，可以自行推导一下。

## 谱归一化

现在我们有了两个谱条件$\eqref{eq:spec-c1}$和$\eqref{eq:spec-c2}$，接下来就要看怎么设计才能让模型自身以及模型优化来满足这两个条件了。

注意，$\boldsymbol{W}_k$和$\Delta \boldsymbol{W}_k$都是矩阵，让一个矩阵满足谱范数条件的标准方法通常是谱归一化（Spectral Normalization，SN），这里也不例外。首先，我们要让初始化的$\boldsymbol{W}_k$满足$\Vert\boldsymbol{W}_k\Vert_2=\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，这可以通过选取任意的初始化矩阵$\boldsymbol{W}_k'$，然后谱归一化实现：
\begin{equation}\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k'}{\Vert\boldsymbol{W}_k'\Vert_2}\end{equation}
这里的$\sigma > 0$是尺度无关的常数；同理，对于任意优化器给出的更新量$\boldsymbol{\Phi}_k$，我们可以通过谱归一化来重新构造$\Delta \boldsymbol{W}_k$：
\begin{equation}\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{\Phi}_k}{\Vert\boldsymbol{\Phi}_k\Vert_2}\end{equation}
其中$\eta > 0$也是尺度无关的常数（学习率），这样每一步都有$\Vert\Delta\boldsymbol{W}_k\Vert_2=\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$。由于初始化和每一步更新的谱范数都满足$\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，所以$\Vert\boldsymbol{W}_k\Vert_2$自始至终也满足$\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$，这就满足了两个谱条件。

这时候可能会有读者疑问，只考虑初始化和增量的稳定性，真的能保证$\boldsymbol{W}_k$的稳定性吗？难道不可能出现$\Vert\boldsymbol{W}_k\Vert_{RMS}\to\infty$吗？答案是有可能。这里的$\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$强调的是与模型尺度（目前主要是宽度）的关系，它不排除由于其余超参设置不当导致训练崩溃的可能性，它要表达的是按这样设置之后，即便出现了崩溃现象，原因也跟尺度变化无关。

## 奇异值裁剪

要实现谱范数条件，除了谱归一化这种标准方法外，我们还可以考虑奇异值裁剪（Singular Value Clipping，下面简称“SVC”）。这部分内容是笔者自行补充的，并未在原论文出现，但它可以解释一些有意思的结果。

从奇异值的角度来，谱归一化是将最大奇异值缩放成1，并同步缩放其余奇异值，奇异值裁剪某个角度来看宽松一些，它只将大于1的奇异值设为1，但不改变原本就小于等于1的奇异值：
\begin{equation}\mathop{\text{SVC}}(\boldsymbol{W}) = \boldsymbol{U}\min(\boldsymbol{\Lambda},1)\boldsymbol{V}^{\top},\qquad \boldsymbol{U},\boldsymbol{\Lambda},\boldsymbol{V}^{\top} = \mathop{\text{SVD}}(\boldsymbol{W})\end{equation}
作为对比，谱归一化是$\mathop{\text{SN}}(\boldsymbol{W})=\boldsymbol{U}(\boldsymbol{\Lambda}/\max(\boldsymbol{\Lambda}))\boldsymbol{V}^{\top}$。用奇异值裁剪替代谱归一化，我们得到
\begin{equation}\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{W}_k'), \qquad \Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{\Phi}_k)\end{equation}
奇异值裁剪的缺点是它只有在至少一个奇异值大于等于1时才能保证裁剪后的谱范数等于1，如果不满足，我们可以考虑乘上一个$\lambda > 0$然后再裁剪，即改为$\mathop{\text{SVC}}(\lambda\boldsymbol{W})$。然而，不同的比例因子会得到不同的结果，但我们也不大好确定适合的比例因子。不过，我们可以考虑一个极限版本
\begin{equation}\lim_{\lambda\to\infty} \mathop{\text{SVC}}(\lambda\boldsymbol{W}) = \mathop{\text{msign}}(\boldsymbol{W})\end{equation}
这里的$\mathop{\text{msign}}$就是Muon里的矩阵版符号函数$\mathop{\text{msign}}$（参考[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)）。用$\mathop{\text{msign}}$替换谱归一化或奇异值裁剪，得到
\begin{equation}\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{msign}}(\boldsymbol{\Phi}_k)\end{equation}
这样我们实际得到了广义的Muon优化器，标准的Muon是对动量$\mathop{\text{msign}}$，而它允许我们对任何现成优化器出来的更新量进行$\mathop{\text{msign}}$。无独有偶，前段时间推特上还真有人做了对Adam更新量$\mathop{\text{msign}}$的实验（对方称之为“Mudamw”，[链接](<https://x.com/kyleliang5/status/1896385441571742103>)），发现效果比Muon还略好，如下图所示：

[![Adam+msgin效果似乎优于Muon（来自推特 @KyleLiang5 ）](<https://spaces.ac.cn/usr/uploads/2025/03/585996134.jpeg>)](<https://spaces.ac.cn/usr/uploads/2025/03/585996134.jpeg>)

Adam\+msgin效果似乎优于Muon（来自推特 @KyleLiang5 ）

我们看到后在小模型上也尝试了一下，发现居然可以复现到相似的结论！所以说不准对现有优化器$\mathop{\text{msign}}$一下，都有机会得到更好的结果。这种操作的可行性在原本的Muon框架下是很难解释的，但这里我们将它理解为对更新量做奇异值裁剪（的极限版本），就会自然得到了这个结果。

## 近似估计

一般认为，谱归一化、奇异值裁剪或$\mathop{\text{msign}}$等跟SVD（奇异值分解）相关的运算都是比较昂贵的，所以我们还是希望能寻找更简单的形式。由于我们的目标只是寻求模型尺度间的缩放规律，所以进一步的简化确实是有可能的。

（注：事实上我们的[Moonlight](<https://papers.cool/arxiv/2502.16982>)工作表明，只要实现得好，即便每一步更新都进行$\mathop{\text{msign}}$，所增加的成本是非常有限的，所以这一节的内容，目前看来更多是为了探索显式的缩放规律而不是节省计算成本）。

首先仍然是初始化，初始化其实是一次性的，所以其实计算量大一点也不是啥问题，所以前面的先随机初始化然后谱归一化/奇异值裁剪/$\mathop{\text{msign}}$的方案可以继续保留。如果还是想要精益求精，那么可以利用一个统计结果：一个从标准正态分布独立重复采样出来的$d_{k-1}\times d_k$矩阵，它的最大奇异值大致是$\sqrt{d_{k-1}} + \sqrt{d_k}$。这样相当于说只要采样标准差改为
\begin{equation}\sigma_k = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}(\sqrt{d_{k-1}} + \sqrt{d_k})^{-1}\right) = \Theta\left(\sqrt{\frac{1}{d_{k-1}}\min\left(1, \frac{d_k}{d_{k-1}}\right)}\right) \label{eq:spec-std}\end{equation}
就可以在初始化阶段满足$\Vert\boldsymbol{W}_k\Vert_2=\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$的需求。至于这个统计结果的证明，可以参考[《High\-Dimensional Probability》](<https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html>)、[《Marchenko\-Pastur law》](<https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution>)等资料，这里就不展开了。

然后我们来考察更新量，它相对麻烦一点，因为任意更新量$\boldsymbol{\Phi}_k$的谱范数并不是那么好估计。这里我们需要利用一个经验结论，那就是参数的梯度矩阵通常都是低秩的。这里的低秩并不一定是数学上严格的低秩，而是指最大的几个（数目跟模型尺度无关）奇异值明显超过其余奇异值，使得低秩近似可用，这也是各种[LoRA](<https://spaces.ac.cn/tag/lora/>)优化的理论基础。

这个经验假设的一个直接推论是谱范数与核范数的近似性，因为谱范数是最大的奇异值，核范数是全体奇异值之和，而上述假设之下核范数约等于最大的若干个奇异值的和，那么两者至少在尺度上是一致的，即$\Theta(\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)=\Theta(\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*)$。接着我们利用$\Delta\mathcal{L}$与$\Delta\boldsymbol{W}_k$的关系：
\begin{equation}\Delta\mathcal{L} \approx \sum_k \langle \Delta\boldsymbol{W}_k, \nabla_{\boldsymbol{W}_k}\mathcal{L}\rangle_F \leq \sum_k \Vert\Delta\boldsymbol{W}_k\Vert_2\,  \Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*\end{equation}
这里的$\langle\cdot,\cdot\rangle_F$是$F$内积，即矩阵展平当向量算内积，至于不等号是因为矩阵范数的一个经典不等式$\langle\boldsymbol{A},\boldsymbol{B}\rangle_F \leq \Vert\boldsymbol{A}\Vert_2\,  \Vert\boldsymbol{B}\Vert_*$，类似于[Holder不等式](<https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality>)，实际上我们在[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592#%E7%9F%A9%E9%98%B5%E8%8C%83%E6%95%B0>)推导Muon时就证明的就是它。基于上式并结合梯度的低秩假设，我们有
\begin{equation}\Delta\mathcal{L} \sim \sum_k \Theta(\Vert\Delta\boldsymbol{W}_k\Vert_2\,  \Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_*) \sim \sum_k \Theta(\Vert\Delta\boldsymbol{W}_k\Vert_2\,  \Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2)\end{equation}
别忘了，我们在前面就已经证明过，满足两个谱条件之下必然有$\Delta\mathcal{L}=\Theta(1)$，现在结合上式我们可以得到当$\Vert\Delta\boldsymbol{W}_k\Vert_2=\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$时有
\begin{equation}\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2 = \Theta\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)\label{eq:grad-norm}\end{equation}
这就是关于梯度数量级的重要估计结果，它直接由两个谱条件推出，避免了显式的梯度计算。这便是谱条件的第三个美妙之处，它使得我们不需要通过链式法则来计算梯度表达式就可以获得相关估计。

## 学习率策略

将估计$\eqref{eq:grad-norm}$用于SGD，即$\Delta \boldsymbol{W}_k = -\eta_k \nabla_{\boldsymbol{W}_k}\mathcal{L}$，根据式$\eqref{eq:grad-norm}$我们有$\Vert\nabla_{\boldsymbol{W}_k}\mathcal{L}\Vert_2=\Theta\left(\sqrt{\frac{d_{k-1}}{d_k}}\right)$，为了达到$\Vert\Delta\boldsymbol{W}_k\Vert_2=\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$的目标，我们需要有
\begin{equation}\eta_k = \Theta\left(\frac{d_k}{d_{k-1}}\right)\label{eq:sgd-eta}\end{equation}

至于Adam，我们仍用SignSGD近似$\newcommand{sign}{\mathop{\text{sign}}}\Delta \boldsymbol{W}_k = -\eta_k \sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})$，由于$\sign$一般来说都是$\pm 1$，所以$\Vert\sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_F = \Theta(\sqrt{d_{k-1} d_k})$，而$\sign$这种Element\-wise的运算，一般不会有什么特别的升秩作用，所以我们认为$\sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})$和$\nabla_{\boldsymbol{W}_k}\mathcal{L}$一样都是低秩的，于是跟核范数类似，$F$范数和谱范数将会是同阶的，即$\Vert\sign(\nabla_{\boldsymbol{W}_k}\mathcal{L})\Vert_2 = \Theta(\sqrt{d_{k-1} d_k})$

因此为了达到$\Vert\Delta\boldsymbol{W}_k\Vert_2=\Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)$的目标，我们需要有
\begin{equation}\eta_k = \Theta\left(\frac{1}{d_{k-1}}\right)\label{eq:adam-eta}\end{equation}

现在我们可以把谱条件的结果跟MuP进行对比。MuP假设我们要建立一个$\mathbb{R}^{d_{in}}\mapsto\mathbb{R}^{d_{out}}$的模型，它把模型分三部分，先用一个$d_{in}\times d$的矩阵把输入投影到$d$维，然后在$d$维空间进行建模，其中的参数都是$d\times d$方阵，最后用一个$d\times d_{out}$矩阵得到$d_{out}$维输出。相应地，MuP的结论也分输入、中间、输出三部分。

初始化方面，MuP的输入方差为$1/d_{in}$、输出方差为$1/d^2$，剩余参数的方差为$1/d$，而谱条件的结果只有一个式子$\eqref{eq:spec-std}$。但我们仔细观察就会发现，式$\eqref{eq:spec-std}$已经包含了MuP的三种情况：设输入、中间、输出矩阵大小是$d_{in}\times d,d\times d,d\times d_{out}$，代入式$\eqref{eq:spec-std}$得到
\begin{equation}\begin{aligned}
\sigma_{in}^2 =&\, \Theta\left(\frac{1}{d_{in}}\min\left(1, \frac{d}{d_{in}}\right)\right) = \Theta\left(\frac{1}{d_{in}}\right) \\
\sigma_k^2 =&\, \Theta\left(\frac{1}{d}\min\left(1, \frac{d}{d}\right)\right) = \Theta\left(\frac{1}{d}\right) \\
\sigma_{out}^2 =&\, \Theta\left(\frac{1}{d}\min\left(1, \frac{d_{out}}{d}\right)\right) = \Theta\left(\frac{1}{d^2}\right)
\end{aligned}
\qquad(d\to\infty) \end{equation}
可能有读者奇怪为什么只考虑$d\to\infty$呢？因为$d_{in},d_{out}$都是任务相关的数字，它们相当于常数，可变的模型尺度只有$d$，而MuP研究的是超参数随模型尺度的渐近规律，所以它都是指$d$足够大时的简化版规律。

学习率方面，对SGD来说，MuP的输入学习率是$d$，输出学习率是$1/d$，剩余参数的学习率是$1$，注意这里的关系都是正比于而不是等于，而谱条件的结果$\eqref{eq:sgd-eta}$同样包含了这三种情况；类似地，对Adam来说，MuP的输入学习率是$1$，输出学习率是$1/d$，剩余参数的学习率是$1/d$，谱条件依然用单个式子$\eqref{eq:adam-eta}$就描述了这三种情况。

所以，谱条件以一种（在笔者看来）更简单的方式，得到了更简明的结果，而这个更简明结果的实际含义则比MuP更丰富，因为它的结果没有对模型架构或者参数形状作过强的假设。因此，笔者称谱条件是MuP的更高阶版本。

## 文章小结

这篇文章介绍了MuP的升级版——谱条件，它从谱范数相关的不等式切入来分析模型稳定训练的条件，以一种更便捷的方式得到了比MuP更丰富的结果。

$$\left\{\begin{aligned}
&\,\text{期望性质:}\left\{\begin{aligned}
&\,\Vert\boldsymbol{x}_k\Vert_{RMS}=\Theta(1) \\[5pt] &\,\Vert\Delta\boldsymbol{x}_k\Vert_{RMS}=\Theta(1)
\end{aligned}\right. \\[10pt]
&\,\text{谱条件:}\left\{\begin{aligned}
&\,\Vert\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right) \\[5pt]
&\,\Vert\Delta\boldsymbol{W}_k\Vert_2 = \Theta\left(\sqrt{\frac{d_k}{d_{k-1}}}\right)
\end{aligned}\right. \\[10pt]
&\,\text{实现方式:}\left\{\begin{aligned}
&\,\text{谱归一化:}\left\{\begin{aligned}
&\,\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{W}_k'}{\Vert\boldsymbol{W}_k'\Vert_2} \\[5pt]
&\,\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\frac{\boldsymbol{\Phi}_k}{\Vert\boldsymbol{\Phi}_k\Vert_2}
\end{aligned}\right. \\[10pt]
&\,\text{奇异值裁剪:}\left\{\begin{aligned}
&\,\boldsymbol{W}_k = \sigma\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{W}_k')\xrightarrow{\text{极限}} \sigma\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{msign}}(\boldsymbol{W}_k')\\[5pt]
&\,\Delta \boldsymbol{W}_k = \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{SVC}}(\boldsymbol{\Phi}_k)\xrightarrow{\text{极限}} \eta\sqrt{\frac{d_k}{d_{k-1}}}\mathop{\text{msign}}(\boldsymbol{\Phi}_k)
\end{aligned}\right. \\[10pt]
&\,\text{近似估计:}\left\{\begin{aligned}
&\,\sigma_k = \Theta\left(\sqrt{\frac{1}{d_{k-1}}\min\left(1, \frac{d_k}{d_{k-1}}\right)}\right) \\[5pt]
&\,\eta_k = \left\{\begin{aligned}
&\,\text{SGD: }\Theta\left(\frac{d_k}{d_{k-1}}\right) \\[5pt]
&\,\text{Adam: }\Theta\left(\frac{1}{d_{k-1}}\right)
\end{aligned}\right.
\end{aligned}\right. \\[10pt]
\end{aligned}\right.
\end{aligned}\right.$$
