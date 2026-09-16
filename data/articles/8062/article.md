# 【搜出来的文本】⋅（一）从文本生成到搜索采样

> 作者：苏剑林 · 科学空间 · 2021-01-07
>
> 原文：<https://spaces.ac.cn/archives/8062>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近，笔者入了一个新坑：基于离散优化的思想做一些文本生成任务。简单来说，就是把我们要生成文本的目标量化地写下来，构建一个分布，然后搜索这个分布的最大值点或者从这个分布中进行采样，这个过程通常<u>不需要标签数据</u>的训练。由于语言是离散的，因此梯度下降之类的连续函数优化方法不可用，并且由于这个分布通常没有容易采样的形式，直接采样也不可行，因此需要一些特别设计的采样算法，比如拒绝采样（Rejection Sampling）、MCMC（Markov Chain Monte Carlo）、MH采样（Metropolis\-Hastings Sampling）、吉布斯采样（Gibbs Sampling），等等。

有些读者可能会觉得有些眼熟，似乎回到了让人头大的学习LDA（Latent Dirichlet Allocation）的那些年？没错，上述采样算法其实也是理解LDA模型的必备基础。本文我们就来回顾这些形形色色的采样算法，它们将会出现在后面要介绍的丰富的文本生成应用中。

## 明确目标

很多时候，我们需要根据一些特定的信息$\boldsymbol{c}$来生成目标文本$\boldsymbol{x}$，用数学的话说就是条件语言模型$p(\boldsymbol{x}|\boldsymbol{c})$，不过我们无法得到足够多的语料对$(\boldsymbol{x},\boldsymbol{c})$去直接监督训练一个条件语言模型，而是只能训练一个无条件的语言模型$p(\boldsymbol{x})$，但我们又可以人为地设计一个指标来定量描述$\boldsymbol{x}$和$\boldsymbol{c}$之间的联系。那么在这种情况下，如何根据无条件的语言模型$p(\boldsymbol{x})$和$\boldsymbol{x},\boldsymbol{c}$之间的联系来做有条件的文本生成，便成为了我们的研究对象。我们可以称之为“受限文本生成（Constrained Text Generation）”

举例来说，用关键词造句，那么$\boldsymbol{c}$就是关键词的集合，我们可以定义示性函数：
\begin{equation}\chi(\boldsymbol{x}, \boldsymbol{c})=\left\{\begin{aligned}&1,\,\,\text{如果}\boldsymbol{x}\text{包含关键词集}\boldsymbol{c} \\
&0,\,\,\text{如果}\boldsymbol{x}\text{不包含关键词集}\boldsymbol{c}\end{aligned}\right.
\end{equation}
继而定义
\begin{equation}\rho(\boldsymbol{x}, \boldsymbol{c}) = p(\boldsymbol{x})\chi(\boldsymbol{x}, \boldsymbol{c})\end{equation}
$p(\boldsymbol{x})$保证了生成句子的流畅性，$\chi(\boldsymbol{x}, \boldsymbol{c})$保证了生成句子包含所要求的关键词，那么问题就可以变成最大化操作$\mathop{\text{argmax}}\limits_{\boldsymbol{x}} \rho(\boldsymbol{x}, \boldsymbol{c})$或采样操作$\boldsymbol{x}\sim \rho(\boldsymbol{x}, \boldsymbol{c})$。当然，这里的$\rho(\boldsymbol{x}, \boldsymbol{c})$还不是概率分布，要完成归一化后才是真正的概率分布：
\begin{equation}\frac{\rho(\boldsymbol{x}, \boldsymbol{c})}{\sum\limits_{\boldsymbol{x}}\rho(\boldsymbol{x}, \boldsymbol{c})} = \frac{p(\boldsymbol{x})\chi(\boldsymbol{x}, \boldsymbol{c})}{\sum\limits_{\boldsymbol{x}}p(\boldsymbol{x})\chi(\boldsymbol{x}, \boldsymbol{c})}\end{equation}
但分母通常是难以显式计算出来的。那也就是说，我们对待采样分布也只了解到它正比于某个函数$\rho(\boldsymbol{x}, \boldsymbol{c})$，而不知道精确的分布表达式。

类似的例子并不少，比如说文本摘要。什么是文本摘要呢？其实就是用更少的文字$\boldsymbol{x}$尽可能表达出跟原文$\boldsymbol{c}$一样的意思，这时候我们可以定义：
\begin{equation}\rho(\boldsymbol{x}, \boldsymbol{c}) = p(\boldsymbol{x})\cdot \text{sim}(\boldsymbol{x}, \boldsymbol{c})\cdot \chi(\boldsymbol{x}, \boldsymbol{c})\end{equation}
这里的$\text{sim}(\boldsymbol{x}, \boldsymbol{c})$是某个文本相似度函数，而$\chi(\boldsymbol{x}, \boldsymbol{c})$是长度的示性函数，即$\boldsymbol{x}$的长度在某个范围（可能依赖于$\boldsymbol{c}$）内，它就为1，否则为0。此时我们同样得到了一个未归一化的概率分布$\rho(\boldsymbol{x}, \boldsymbol{c})$，需要最大化它或者从它里边采样。很明显，这个目标就意味着我们要得到一段跟原文语义尽可能相似的、长度满足一定约束的文字，这不就是摘要的存在意义吗？所以，这套思路的核心出发点就在于：我们要把自己要生成的目标定量地捋清楚，然后再去执行下一步操作。

## 困难分析

所以，抛开前面的背景不说，现在我们面临的问题就是有一个分布$p(\boldsymbol{x})$，我们只知道$p(\boldsymbol{x})\propto \rho(\boldsymbol{x})$，即
\begin{equation}p(\boldsymbol{x}) = \frac{\rho(\boldsymbol{x})}{\sum\limits_{\boldsymbol{x}} \rho(\boldsymbol{x})}\end{equation}
中的分母我们无法显式计算出来。在本系列文章中，$\boldsymbol{x}$代表文本，即一个离散元素的序列，但后面的推论同样也适用于$\boldsymbol{x}$是连续型向量的场景。现在我们要搜索最大位置$\mathop{\text{argmax}}\limits_{\boldsymbol{x}} p(\boldsymbol{x})$或进行采样$\boldsymbol{x}\sim p(\boldsymbol{x})$，后面我们将会看到，搜索最大值其实也可以看成是采样的特例，因此我们主要关心采样方式。

前面说了，之所以需要设计一些特别的算法来完成采样，是因为直接从$p(\boldsymbol{x})$中采样是困难的，而我们需要理解采样的困难所在，才能真正理解后面所设计的采样算法的关键之处。困难在哪？如果$\boldsymbol{x}$的候选值空间不大，哪怕有100万个候选值，我们都可以把每个$p(\boldsymbol{x})$都算出来，然后按照普通的类别采样来进行。然而，一般$\boldsymbol{x}$的候选值空间远远不止100万，假如$\boldsymbol{x}$有10个分量，每个分量有1万个选择（对应于词表大小），那么总的排列就有$10^{40}$种了，不可能事先算好每一种排列的概率然后依概率采样。

那怎么办呢？所谓“不积硅步，无以至千里”，那就只能一步步来了，也就是说，我没法直接实现$10^{40}$选1，那我做10次“$10^4$选1”可以吗？这就对应着所谓的“自回归生成”：
\begin{equation}p(\boldsymbol{x})=p(x_1) p(x_2|x_1) p(x_3|x_1, x_2) \cdots p(x_n|x_1,\cdots,x_{n-1}) = \prod_{t=1}^n p(x_t|\boldsymbol{x}_{< t})\end{equation}
这样我们就可以先从$p(x_1)$采样一个$x_1$，然后从$p(x_2|x_1)$中采样一个$x_2$，依此递归了。但是，自回归生成只是对应于无条件的语言模型或者是有监督训练的Seq2Seq模型，而如果希望像前面举的例子那样，往无条件语言模型的生成过程中加点约束，那么对应出来的模型就不再是自回归的了，也就无法按照这样的递归采样了。

所以，我们就不得不需要后面介绍的各种采样算法了，它也是“一步步来”的思想，但所使用的分布形式更加广泛一些。

## 重要采样

在[《从采样看优化：可导优化与不可导优化的统一视角》](<https://spaces.ac.cn/archives/7521>)、[《如何划分一个跟测试集更接近的验证集？》](<https://spaces.ac.cn/archives/7805>)等文章里，我们介绍过“重要性采样”的概念，即如果我们想估计期望$\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[f(\boldsymbol{x})]$，但是$p(\boldsymbol{x})$又不是易于采样的分布，那么我们可以找一个跟$p(\boldsymbol{x})$相近的、易于采样的分布$q(\boldsymbol{x})$，然后根据下述变换
\begin{equation}
\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[f(\boldsymbol{x})] = \sum_{\boldsymbol{x}} p(\boldsymbol{x}) f(\boldsymbol{x}) = \sum_{\boldsymbol{x}} q(\boldsymbol{x})\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}f(\boldsymbol{x}) = \mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}\left[\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}f(\boldsymbol{x})\right]
\end{equation}
转化为从$q(\boldsymbol{x})$采样来算$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}f(\boldsymbol{x})$的期望了，也就是用$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$对每个样本进行加权，所以它被称为“重要性采样（Importance Sampling）”。如果只知道$p(\boldsymbol{x})\propto \rho(\boldsymbol{x})$，那么重要性采样也是可以进行的，这是因为
\begin{equation}1 = \sum_{\boldsymbol{x}} p(\boldsymbol{x}) = \sum_{\boldsymbol{x}} q(\boldsymbol{x})\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})} = \mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}\left[\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right]\end{equation}
所以
\begin{equation}
\mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}\left[\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}f(\boldsymbol{x})\right] = \mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}\left[\frac{p(\boldsymbol{x}) / q(\boldsymbol{x})}{\mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}[p(\boldsymbol{x}) / q(\boldsymbol{x})]}f(\boldsymbol{x})\right]
\end{equation}
这样一来，我们发现上式只依赖于$p(\boldsymbol{x})$的相对值，不依赖于它的绝对值，所以把$p(\boldsymbol{x})$换成跟它成正比的$\rho(\boldsymbol{x})$也是可以的，最终简化成：
\begin{equation}
\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[f(\boldsymbol{x})] \approx \frac{\sum\limits_{i=1}^N \rho(\boldsymbol{x}_i) / q(\boldsymbol{x}_i) \cdot f(\boldsymbol{x}_i)}{\sum\limits_{i=1}^N
\rho(\boldsymbol{x}_i) / q(\boldsymbol{x}_i)},\quad \boldsymbol{x}_1,\cdots,\boldsymbol{x}_N\sim q(\boldsymbol{x})\end{equation}

## 拒绝采样

上一节的重要性采样实现了将复杂分布期望转化为简单分布期望，但这还不是我们的真正目的，我们要实现的是把样本从分布$p(\boldsymbol{x})$中采样出来，而不是估算它的某个期望。思想依然跟重要性采样一样，引入易于采样的分布$q(\boldsymbol{x})$，然后从中随机地筛掉某些样本，使得剩下的样本服从分布$p(\boldsymbol{x})$。

具体来说，假设有函数$\alpha(\boldsymbol{x})\in [0, 1]$，我们按照如下流程进行采样，即“拒绝采样（Rejection Sampling）”：

> <strong>拒绝采样</strong> 从$q(\boldsymbol{x})$采样一个样本$\boldsymbol{x}$，从$U[0,1]$中采样一个随机数$\varepsilon$，若$\varepsilon \leq \alpha(\boldsymbol{x})$则接受该样本，否则拒绝并重新按照此流程采样。

那么，此时采样出来的$\boldsymbol{x}$真正的概率分布是什么呢？其实也不难，由于样本$\boldsymbol{x}$被保留下来的概率是$\alpha(\boldsymbol{x})$，因此它的相对概率就是$q(\boldsymbol{x})\alpha(\boldsymbol{x})$，我们只需要将它重新归一化
\begin{equation}\frac{q(\boldsymbol{x})\alpha(\boldsymbol{x})}{\sum\limits_{\boldsymbol{x}} q(\boldsymbol{x})\alpha(\boldsymbol{x})}\end{equation}
就得到拒绝采样对应的真正的概率分布了，从这个形式也可以看出，<u>将接受率乘以一个0到1之间的数，理论上拒绝采样对应的分布是不变的</u>。

这个过程启示我们，拒绝采样可以让我们实现从正比于$q(\boldsymbol{x})\alpha(\boldsymbol{x})$的分布中采样，那么根据$p(\boldsymbol{x})=q(\boldsymbol{x})\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$，我们可以让$\alpha(\boldsymbol{x})=\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$作为接受概率，来进行从$q(\boldsymbol{x})$出发的拒绝采样，结果就相当于从$p(\boldsymbol{x})$采样了。当然，还没那么简单，根据概率的归一化性质，除非$q(\boldsymbol{x})$恒等于$p(\boldsymbol{x})$，否则$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$不可能一直都在$[0, 1]$内。但这不要紧，只要$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$有上界，那么我们就可以选择一个足够大的常数$M$，使得$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})\cdot M}\in [0, 1]$，此时以$\alpha(\boldsymbol{x})=\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})\cdot M}$为接受概率即可，刚才我们说了，乘以一个常数不会影响拒绝采样对应的分布。换句话说，也就是这个过程同样不依赖于完全精确的$p(\boldsymbol{x})$，可以将$p(\boldsymbol{x})$换成跟它成正比的$\rho(\boldsymbol{x})$。

关于接受率$\alpha(\boldsymbol{x})$，尽管理论上只要求它$\alpha(\boldsymbol{x})\in[0, 1]$就行了，但实际上还是以$\max\limits_{\boldsymbol{x}}\alpha(\boldsymbol{x}) = 1$为好，这是因为过小的接受率会导致拒绝太多（几乎来一个拒绝一个），采样效率太低，生成一个合理的样本的成本过大了。类似地，尽管理论上对$q(\boldsymbol{x})$的要求只是易于采样并且$\frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}$有上界，但实际上$q(\boldsymbol{x})$与$p(\boldsymbol{x})$仍然是越相近越好，否则依然可能造成接受率过低而导致采样成本大到难以接受。所以，尽管拒绝采样看上去提供了一种几乎能从任意分布$p(\boldsymbol{x})$中进行采样的方案，但实际应用时近似分布$q(\boldsymbol{x})$的设计依然是一个不小的难题。

## 本文小结

从本文开始，我们开了个新坑，试图从离散优化的角度来完成某些文本生成任务（受限文本生成）。它通过确定一个定量的评估目标，然后通过最大化这个目标或者从中采样就可以得到我们想要的输出，而不需要标签数据监督训练新模型。在这个过程中，所要用到的工具是一些主要是采样算法，本文先介绍了其中很基本的重要性采样和拒绝采样，后面将会继续完善该系列文章，敬请大家期待。
