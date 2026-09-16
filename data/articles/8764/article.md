# ChildTuning：试试把Dropout加到梯度上去？

> 作者：苏剑林 · 科学空间 · 2021-11-22
>
> 原文：<https://spaces.ac.cn/archives/8764>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

Dropout是经典的防止过拟合的思路了，想必很多读者已经了解过它。有意思的是，最近Dropout有点“老树发新芽”的感觉，出现了一些有趣的新玩法，比如最近引起过热议的[SimCSE](<https://spaces.ac.cn/archives/8348>)和[R\-Drop](<https://spaces.ac.cn/archives/8496>)，尤其是在文章[《又是Dropout两次！这次它做到了有监督任务的SOTA》](<https://spaces.ac.cn/archives/8496>)中，我们发现简单的R\-Drop甚至能媲美对抗训练，不得不说让人意外。

一般来说，Dropout是被加在每一层的输出中，或者是加在模型参数上，这是Dropout的两个经典用法。不过，最近笔者从论文[《Raise a Child in Large Language Model: Towards Effective and Generalizable Fine\-tuning》](<https://papers.cool/arxiv/2109.05687>)中学到了一种新颖的用法：加到梯度上面。

梯度加上Dropout？相信大部分读者都是没听说过的。那么效果究竟如何呢？让我们来详细看看。

## 方法大意

简单来说，这篇论文主要提出了一种名为“ChildTuning”的思路，来提高预训练模型在finetune时的效果，其中“Child”是“Children Network”的意思，指的是从预训练模型中选择一个子网络进行优化，缓解优化整个模型所带来的过拟合风险。其中，在子网络的选择上，又分为两种方式：ChildTuning\-D和ChildTuning\-F。

### ChildTuning\-D

ChildTuning\-D（Task\-Dependent）是任务相关的选择方式，它需要下游任务的训练数据来参与计算。具体来说，假设训练数据为$(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)$，模型为$p(y|x;\theta)$，其中$\theta$是模型的所有参数，而$\theta_i$则是其中的第$i$个参数，那么我们计算如下形式的Fisher信息作为该参数的重要性：
\begin{equation}F_i = \frac{1}{n}\sum_{j=1}^n \left(\frac{\partial \log p(y_j|x_j;\theta)}{\partial\theta_i}\right)^2\end{equation}
有了重要性指标后，我们就可以对每个参数进行排序，然后选出最重要的top\-$p$部分（比如前20%，即$p=0.2$），然后在模型更新的时候只优化这些参数。由于优化的参数变少了，所以过拟合的风险也就降低了。在实际使用时，ChildTuning\-D在finetune之前就把要优化的参数确定下来，然后就一直保持不变了。

要注意的是，这里的参数选择是以每个分量为单位的，也就是说，可能一个参数矩阵里边，只有一部分被选择中，所以不能说单独挑出哪些参数矩阵不优化，而是要通过构建对应的0/1矩阵$M$来将对应的梯度mask掉，即$g\leftarrow g\otimes M / p$，其中除以$p$是保持整理的更新量级不变。这样没被选中的参数梯度一直是0，所以也就没有更新了。这样一来，虽然理论上更新的参数少了，但它也不能节约计算量，所以作者只是将它定位为提高finetune效果的方法。

### ChildTuning\-F

ChildTuning\-F（Task\-Free）是任务无关的选择方式，其实它可以更形象地称为“梯度Dropout”。对于ChildTuning\-D来说，我们就是根据任务数据来构建了固定的0/1矩阵$M$，然后将梯度修改为$g\otimes M / p$，而ChildTuning\-F既然希望与任务无关，那么它每步更新就随机构建一个0/1矩阵$M$，其中1的比例为$p$，然后将梯度修改为$g\otimes M / p$。可以看到，这本质就是对梯度进行Dropout。

要注意，某个参数当前的梯度为0，也不代表该参数当前的更新量为0，因为我们通常用的都是带有动量的优化器如SGDM和Adam，对于此类优化器，更新量是正比于动量，而动量是历史梯度滑动平均过来的，即$m_t = \beta m_{t-1} + (1-\beta)g_t$，所以如果该参数的历史梯度不为0，那么即便当前梯度为0，动量依然很可能不会为0，所以更新量也不为0。

所以在这里笔者就有个疑问，按照ChildTuning的设计思路，它应该是想要每步只选择一个子网络进行更新，说白了就是每一步只更新$p$比例的参数，但根据上面的分析，对梯度进行Dropout其实达不到这个目的，而要实现这个目的，应该要对每步更新量$\Delta\theta$进行Dropout才对。但笔者反复看了原论文，甚至对照了论文开源的代码，最终确认论文的意思确实是对梯度进行Dropout。

## 实验结果

从原论文给出的实验结果来看，ChildTuning的“战绩”是相当耀眼了，几乎都有提升，而且最高提升达到8%～

[![ChildTuning的“战绩”-1](<https://spaces.ac.cn/usr/uploads/2021/11/602640021.png>)](<https://spaces.ac.cn/usr/uploads/2021/11/602640021.png>)

ChildTuning的“战绩”\-1

[![ChildTuning的“战绩”-2](<https://spaces.ac.cn/usr/uploads/2021/11/3333775018.png>)](<https://spaces.ac.cn/usr/uploads/2021/11/3333775018.png>)

ChildTuning的“战绩”\-2

从表中可以看出，对于ChildTuning\-D来说，几乎所有任务上都取得了提升，而ChildTuning\-F也在不少任务上有效。另外，看论文描述可以知道上面给出的都是large版本模型的结果，而私下跟作者交流的时候，作者表示base版本的效果也有提升，只是限于论文篇幅，没有贴出来。

## 原理思考

ChildTuning\-D基于Fisher信息来对进行参数排序，该思路由来已久，它有效并不让人意外，类似的工作还有[《Training Neural Networks with Fixed Sparse Masks》](<https://papers.cool/arxiv/2111.09839>)等。反倒是任务无关的ChildTuning\-F，也就是梯度Dropout，居然也有这么效果，值得我们细细思考。

无独有偶，对梯度进行Dropout的工作，去年也有一篇，名为[《Regularizing Meta\-Learning via Gradient Dropout》](<https://papers.cool/arxiv/2004.05859>)。这表明，Gradient Dropout应该确实能起到一定效果的。那它究竟为什么有效呢？

### 论文推导

原论文给出一个基于SGD的理解，它指出梯度Dropout能扩大更新过程中的方差，从而有助于模型逃离不那么好的局部最优点。

具体来说，因为我们是用了SGD，所以每步所计算的梯度有一定的随机性，假设它服从均值为$\mu$、方差为$\sigma^2$的高斯分布；对于ChildTuning\-F来说，引入一个随机变量$\varepsilon$，有$p$的概率为1，剩下$1-p$的概率为0。那么我们将有
\begin{equation}\begin{aligned}&\mathbb{E}[g\varepsilon/p]=\mathbb{E}[g]\mathbb{E}[\varepsilon]/p=\mu \\
&\mathbb{E}[(g\varepsilon/p)^2]=\mathbb{E}[g^2]\mathbb{E}[\varepsilon^2]/p^2 = (\mu^2+\sigma^2)/p
\end{aligned}\end{equation}
所以
\begin{equation}\mathbb{V}ar[g\varepsilon/p] = \mathbb{E}[(g\varepsilon/p)^2] - \mathbb{E}[g\varepsilon/p]^2=\sigma^2 + \frac{1-p}{p}(\mu^2+\sigma^2) > \sigma^2\end{equation}
也就是说，梯度Dropout能保持梯度的均值不变，但能扩大方差，而在SGD中，更新量正比于梯度，因此梯度Dropout扩大了更新量的方差，论文认为这有助于模型达到更好的收敛结果。

### 答非所问

这个解释看上去挺合理的，也符合很多人的直觉，因为很多人的潜意识里觉得随机梯度下降比全量梯度下降好的原因就是因为有噪声。然而，只要我们稍微深入思考一下，就能发现上述解释其实是“答非所问”。

原因很简单，上面分析的是SGD，但实际上在NLP中我们用的都是Adam（或者是它的变种），上述结论还能在Adam中保持吗？很遗憾，不能，甚至刚好相反。在Adam中，长期来看，更新量可以近似看成（$\eta$是学习率）
\begin{equation}\Delta\theta = \eta\frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}\end{equation}
于是加了梯度Dropout后，更新量变为
\begin{equation}\eta\frac{\mathbb{E}[g\varepsilon/p]}{\sqrt{\mathbb{E}[(g\varepsilon/p)^2]}}=\eta\sqrt{p}\frac{\mathbb{E}[g]}{\sqrt{\mathbb{E}[g^2]}}\end{equation}
可以看到，长期来看，Adam加上梯度Dropout后，仅仅相当于学习率降低为原来的$\sqrt{p}$倍！而且由于降低了学习率，也即降低了更新量，从而更新量的方差也随之降低。也就是说，如果你用了Adam优化器，那么实际情况跟论文的解释刚好相反，更新量的方差不仅没有增加，反而是降低了。

出现这个现象的根本原因就是，当我们使用了带有滑动平均的优化器时，更新量通常已经不在正比于梯度了，所以梯度如何变化，跟更新量如何变化，并没有必然的关联。这就回到了笔者前面的疑问了：为什么作者不干脆直接对更新量进行Dropout？如果是更新量Dropout，那么前面基于SGD的推导也能搬过来了。

### 个人理解

不过，笔者认为，就算把优化器限制为SGD，或者直接对更新量进行Dropout，原论文的推导也不能完全解释它的有效性。理由也很简单，能够达到“均值不变、方差扩大”的操作太多了，比如直接往梯度里边加点高斯噪声也可以，难道所有的这些操作都能达到同样的效果？个人感觉不大可能。笔者认为，要解释梯度Dropout或者更新量Dropout的有效性，得着眼于Dropout带来的稀疏性。

在这个问题上，笔者联想到了之前写过的文章[《从动力学角度看优化算法（七）：SGD ≈ SVM？》](<https://spaces.ac.cn/archives/8009>)，这篇文章告诉我们，所有SGD出来的模型，其解本质上都类似于SVM模型：
\begin{equation}f_{\theta_T}(x) = \beta(x) + \sum_i \alpha_i (x) K(x, x_i)\end{equation}
其中$x_i$是第$i$个训练样本。它有什么特点呢？$K(x,x_i)$的表现类似一个“相似度函数”，上述形式意味着模型实际上会以某种形式把训练集“背”下来了，然后预测的时候会以$K(x,x_i)$为相似度取检索训练集，然后给出预测结果。当然，这只是一个原理性的解释，我们并不是主动将模型设计成这样的形式，我们只是从这个角度看出，梯度下降实际上也是在背样本，然后以类似于KNN的形式给出预测结果，这就不难理解为什么通常来说“训练样本越多，效果越好”的结论了。

回到ChildTuning\-F上，我们每次采样一个batch，然后对算出来的梯度或更新量进行Dropout，结合上面的“背样本”解释，我们可以直观地想象，这本质上就是“只用一小部分参数来背诵一小部分样本”，而不是每次都要用全体参数来背诵那一小批样本。所以，这跟“不要将鸡蛋放在同一个篮子里”应该是相似的，将样本更均匀分散在每一个参数中，从而降低了过拟合风险。

## 尝试一下

对于ChildTuning\-F来说，如果自己懂得改优化器的话，不管是对梯度Dropout还是对更新量Dropout，都只是一行代码的工作量，因此还是值得尝试一下的。万一真的有用呢？

这里笔者在CLUE的几个任务上做了测试，结果如下表。其中，baseline代码来自[《bert4keras在手，baseline我有：CLUE基准代码》](<https://spaces.ac.cn/archives/8739>)，“grad drop”是对梯度进行Dropout，“incre drop”是对更新量进行Dropout，绿色表示相比baseline有提升，红色则表示下降。时间算力都有限，所有结果都只跑了一次，存在一定的随机波动。

$$\begin{array}{c}
\text{CLUE分类任务对比实验（验证集）} \\
{\begin{array}{c|ccccccc}
\hline
& \text{IFLYTEK} & \text{TNEWS} & \text{AFQMC} & \text{OCNLI} & \text{CMNLI} & \text{WSC} & \text{CSL} \\
\hline
\text{BERT} & 60.06 & 56.80 & 72.41 & 73.93 & 79.56 & 78.62 & 83.93 \\
\text{BERT}_{\text{-grad drop}} & \color{green}{60.56} & \color{green}{56.97} & \color{red}{72.13} & \color{green}{74.88} & \color{green}{80.09} & \color{red}{75.99} & \color{red}{83.83} \\
\text{BERT}_{\text{-incre drop}} & \color{red}{59.99} & \color{red}{56.78} & \color{green}{72.66} & \color{green}{74.51} & \color{red}{79.36} & \color{red}{77.30} & \color{green}{84.20} \\
\hline
\text{RoBERTa} & 60.64 & 58.06 & 74.05 & 76.00 & 81.24 & 87.50 & 84.50\\
\text{RoBERTa}_{\text{-grad drop}} & \color{green}{60.72} & \color{red}{57.91} & \color{red}{74.03} & \color{red}{75.19} & \color{red}{80.52} & \color{red}{84.54} & \color{green}{84.73}\\
\text{RoBERTa}_{\text{-incre drop}} & \color{green}{60.87} & \color{red}{57.99} & \color{red}{74.03} & \color{red}{75.97} & \color{red}{81.02} & \color{red}{84.87} & \color{green}{84.73}\\
\hline
\end{array}}
\end{array}$$

从表格中，我们大致可以看出：

> 1、对梯度Dropout和对更新量进行Dropout，大致上各有优劣；
> 
> 
> 
> 2、在BERT上的效果明显一些，在RoBERTa上的效果几乎没有，这跟论文给出的英文实验结果相似。

这结果挺让人无语的，不能说它没效，但正常来说，谁会用速度一样、效果更差的BERT而不用效果更好的RoBERTa呢？那么，如果RoBERTa不怎么work的话，似乎就没啥尝试的价值了？当然，原论文提升最大的是Electra，这个我没尝试过，有兴趣的读者尝试了把结果告诉我一下哈。

另外，笔者对ChildTuning\-D没有特别的兴趣，加上ChildTuning\-D的实现稍微复杂一点，所以也就没有实验ChildTuning\-D了，实验过的读者也欢迎反馈结果哈。

## 文章总结

本文介绍了往梯度里边加入Dropout来提高finetune效果的做法，并给出了自己的理论分析。总的来说，个人的感觉是：可以尝试，可能有效，但不要期望太高～
