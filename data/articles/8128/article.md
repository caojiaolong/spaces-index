# Seq2Seq重复解码现象的理论分析尝试

> 作者：苏剑林 · 科学空间 · 2021-01-26
>
> 原文：<https://spaces.ac.cn/archives/8128>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

去年笔者写过博文[《如何应对Seq2Seq中的“根本停不下来”问题？》](<https://spaces.ac.cn/archives/7500>)，里边介绍了一篇论文中对Seq2Seq解码不停止现象的处理，并指出那篇论文只是提了一些应对该问题的策略，并没有提供原理上的理解。近日，笔者在Arixv读到了AAAI 2021的一篇名为[《A Theoretical Analysis of the Repetition Problem in Text Generation》](<https://papers.cool/arxiv/2012.14660>)的论文，里边从理论上分析了Seq2Seq重复解码现象。从本质上来看，重复解码和解码不停止其实都是同理的，所以这篇新论文算是填补了前面那篇论文的空白。

经过学习，笔者发现该论文确实有不少可圈可点之处，值得一读。笔者对原论文中的分析过程做了一些精简、修正和推广，将结果记录成此文，供大家参考。此外，抛开问题背景不讲，读者也可以将本文当成一节矩阵分析习题课，供大家复习线性代数哈～

## 基本思路

所谓重复解码，指的是解码结果出现重复的片段，比如解码结果为“A B C D B C D B C D E F”，那么“B C D”就是重复片段了，因此这个解码结果就出现了重复解码现象。简单起见，如果解码过程中子序列$s=[w_1,w_2,\cdots,w_n]$后面接着的子序列是$t=[w_1,w_2,\cdots,w_n,w_1]$，我们就称$[w_1,w_2,\cdots,w_n]$为一个“重复子序列”，而我们现在要做的事情，就是要分析解码过程中出现重复子序列的概率。

可能有读者疑问，为什么$t$的最后要多加一个$w_1$？从后面的过程中我们可以明白到，这个其实只是为了分析上的方便，并没有什么必然性。我们希望得到的是一个有代表性的定量指标来衡量这个重复解码问题，最好还能从中能获得一些改进的思路，至于这个指标的具体细节，我们可以不用太在意。将研究目标量化是非常重要的，只有把目标量化后，我们才能更好地把握改进的方向，也才能去比较不同的方法优劣。不然就算吵得面红耳赤的，也终究无法得到个结论出来。

为了得到这样的一个指标，我们接下来先从简单的二元解码出发，得到一些有代表性的结果，然后看它能否推广到一般的自回归解码器中去。

## 二元解码

一般的自回归模型形式为：
\begin{equation}p(\boldsymbol{y}|\boldsymbol{x}) = \prod_{t=1}^l p(y_t|\boldsymbol{y}_{< t}, \boldsymbol{x})\end{equation}
也就是说，位置$t$的解码不仅依赖于输入$\boldsymbol{x}$，还依赖于$t$之前已经获得的所有解码结果。而简单起见，我们先考虑一种简单的情况，假设每一步解码只依赖于前一时刻的结果，即：
\begin{equation}p(\boldsymbol{y}|\boldsymbol{x}) = \prod_{t=1}^l p(y_t|y_{t-1}, \boldsymbol{x})\end{equation}
这样一来，对于固定的输入$\boldsymbol{x}$，解码器事实上就只是一个$n\times n$的转移矩阵$\boldsymbol{P}=(P_{i,j})$，其中$P_{i,j}$表示从$i$后面接$j$的概率，$n$代表词表大小。这样的解码器叫做二元文法模型、2\-gram模型、马尔可夫模型，等等。我们还需要一个终止标记\<eos\>，遇到\<eos\>就停止解码，所以实际上转移矩阵是$(n+1)\times (n+1)$才对，但是我们考虑重复解码都是在终止之前的，所以只需要考虑除去\<eos\>的$n\times n$部分就行了。

我们要计算的是重复子序列的出现概率，假如以$[i, j, k]$是一个三元重复子序列，那么它的出现概率就是序列$[i, j, k, i, j, k, i]$出现的概率：
\begin{equation}P_{i,j}P_{j,k}P_{k,i}P_{i,j}P_{j,k}P_{k,i}=P_{i,j}^2 P_{j,k}^2 P_{k,i}^2\end{equation}
因此所有的三元重复子序列的概率为
\begin{equation}\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2 = \text{Tr}\,(\boldsymbol{P}\otimes\boldsymbol{P})^3\end{equation}
这里的$\otimes$表示逐位元素对应相乘，而$\text{Tr}$则是矩阵的迹，即对角线元素之和。最后，我们将所有长度的重复子序列概率都加起来：
\begin{equation}R = \sum_{k=1}^{\infty}\text{Tr}\,(\boldsymbol{P}\otimes\boldsymbol{P})^k = \text{Tr}\,\left(\sum_{k=1}^{\infty}(\boldsymbol{P}\otimes\boldsymbol{P})^k\right)\label{eq:r}\end{equation}
这个就是二元解码器出现重复解码的概率。当然目前它还只是一个理论公式，不过它是我们重要的出发点。我们将分别推导它的上下界，以获得更具有启发性的结果。

## 一个下界

直接看式$\eqref{eq:r}$不好看出点啥，我们可以先推导它一个更加直观一点的下界。还是以三元重复子序列为例，利用均值不等式我们可以得到：
\begin{equation}
\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2 = n^3\times\frac{\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2}{n^3}\geq n^3\times\left(\frac{\sum_{i,j,k} P_{i,j} P_{j,k} P_{k,i}}{n^3}\right)^2 = \frac{(\text{Tr}\, \boldsymbol{P}^3)^2}{n^3}
\end{equation}
事实上，我们还可以做得更精细一些。假设矩阵$\boldsymbol{P}$有一些元素为0，那么$P_{i,j}^2 P_{j,k}^2 P_{k,i}^2$中的非零元素的个数就不是$n^3$了，我们假设非零元素个数为$N_3(\boldsymbol{P}) < n^3$，那么我们在利用均值不等式的时候，可以只对非零元素进行，结果是将上述的$n^3$换为$N_3(\boldsymbol{P})$：
\begin{equation}
\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2 \geq \frac{(\text{Tr}\, \boldsymbol{P}^3)^2}{N_3(\boldsymbol{P})}
\end{equation}
$N_3(\boldsymbol{P})$的直接计算比较困难，没有一般通项公式，但我们可以做个简单估算：设$\boldsymbol{P}$的非零元素的比例为$\zeta$，也就是非零元素个数为$\zeta n^2$，那么我们可以认为$P_{i,j}^2 P_{j,k}^2 P_{k,i}^2$的非零元素比例近似为$\zeta^3$，而总的排列数为$n^3$，所以我们可以认为$N_3(\boldsymbol{P})\sim \zeta^3 n^3$，或者一般地$N_k(\boldsymbol{P})\sim \zeta^k n^k$。注意可以举例说明这个估计既不能保证是上界，也不能保证是下界，所以将$N_3(\boldsymbol{P})$替换为$\zeta^3 n^3$后，我们无法保证上述不等号的成立。不过，如果我们愿意相信$\zeta^3 n^3$是一个足够好的近似，我们我们依然可以（怀着忐忑而又坚定的信念）写下
\begin{equation}
\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2 \geq \frac{(\text{Tr}\, \boldsymbol{P}^3)^2}{\zeta^3 n^3}
\end{equation}
以及
\begin{equation}R = \sum_{k=1}^{\infty}\text{Tr}\,(\boldsymbol{P}\otimes\boldsymbol{P})^k \geq \sum_{k=1}^{\infty} \frac{(\text{Tr}\, \boldsymbol{P}^k)^2}{\zeta^k n^k}\label{eq:r-2}\end{equation}
或者我们干脆不关心不等号，而是将最右面的结果视为$R$的一个估计。

## 原文下界

对于希望对着本文读原论文的读者，此时可能会有点懵了，因为不管是式$\eqref{eq:r}$还是式$\eqref{eq:r-2}$，都在原论文中找不到对应。事实上，原文并没有给出精确的式$\eqref{eq:r}$，也没有给出估计式$\eqref{eq:r-2}$，而是给出了另一个估计式，它也可以作为式$\eqref{eq:r}$的下界推导出来。

同样利用均值不等式，我们有
\begin{equation}\begin{aligned}
\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2 =&\, \sum_{i} \sum_{j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2= \sum_{i} n^2\times\frac{\sum_{j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2}{n^2}\\
\geq&\, \sum_{i} n^2\times\left(\frac{\sum_{j,k} P_{i,j} P_{j,k} P_{k,i}}{n^2}\right)^2 = \frac{\text{Tr}\, (\boldsymbol{P}^3\otimes \boldsymbol{P}^3)}{n^2}
\end{aligned}\end{equation}
类似地，可以引入非零元素个数的技巧来提高估计精度，非零率依然是$\zeta^3$，而这次求和的总数是$n^2$，因此非零排列数约为$\zeta^3 n^2$，所以我们（依旧是怀着忐忑而又坚定的信念）写下：
\begin{equation}
\sum_{i,j,k} P_{i,j}^2 P_{j,k}^2 P_{k,i}^2 \geq \frac{\text{Tr}\, (\boldsymbol{P}^3\otimes \boldsymbol{P}^3)}{\zeta^3 n^2}\end{equation}
以及
\begin{equation}R = \sum_{k=1}^{\infty}\text{Tr}\,(\boldsymbol{P}\otimes\boldsymbol{P})^k \geq \sum_{k=1}^{\infty}\frac{\text{Tr}\, (\boldsymbol{P}^k\otimes \boldsymbol{P}^k)}{\zeta^k n^{k-1}}\label{eq:r-3}\end{equation}
这基本就是原论文中的“定义2\.3”了，跟原论文不同的是：

> 1、原论文算得是平均到每个字词的概率，所以需要多除以一个$n$，因此它的分母是$n^k$；
> 
> 
> 
> 2、原论文求迹的是$\boldsymbol{P}^{2k}$而不是$\boldsymbol{P}^k\otimes \boldsymbol{P}^k$，事实上这是原论文的错误，它在推导过程中把$(\boldsymbol{P}^k)_{i,i}^2$当成了$(\boldsymbol{P}^{2k})_{i,i}$，事实上它们是不等的，本文的式$\eqref{eq:r-3}$才是正确的结果。

## 初步结论

其实不管是式$\eqref{eq:r-2}$还是式$\eqref{eq:r-3}$，形式都差不多，我们都可以用它来得出一些结论。此时，可能有些读者会疑惑：我们一般所用的模型的概率分布都是softmax出来的，softmax的结果都不等于0，所以$\zeta$应该是恒等1，因此引入$\zeta$似乎没有没有什么价值？

并非如此。的确，softmax出来的概率分布不会有严格等于0的情况，但是我们的解码算法，通常却会将它们强制置零！在文章[《如何应对Seq2Seq中的“根本停不下来”问题？》](<https://spaces.ac.cn/archives/7500>)中我们就罗列了文本生成常用的解码算法，主要包括随机采样和确定性解码两种，其中随机采样分为直接随机采样、Top\-k随机采样、Top\-p随机采样，而确定性解码则包括Greedy Search、Beam Search两种，在这五种不同的解码算法中，除了最不常用的直接随机采样外，其余四种都是强行只保留若干个最优结果来作为候选值，这样就相当于直接截断了转移矩阵，大大降低了非零概率$\zeta$。

比如最极端的Greedy Search，容易推出它实际上对应着最小的非零概率$\zeta=1/n$，由于$\zeta$是在分母中，所以$\zeta$的缩小意味着重复率$R$的增加，这就告诉我们Greedy Search的重复解码风险是相当高的。尽管目前的结论仅仅是在二元解码模型的假设下得出的，但Greedy Search的重复解码确实是我们经常观察到的现象，所以这结论与解释确实已经有代表性了。

## 一个上界

有了下界，怎么可以没有上界呢？下界能帮助我们解释一些实验现象，而上界则可以给我们提供改进的思路。

为了推导上界，我们利用到如下两个结论：

> 1、矩阵的迹等于它所有特征值之和；
> 
> 
> 
> 2、如果$\lambda_1(\boldsymbol{A})\geq\lambda_2(\boldsymbol{A})\geq\cdots\geq\lambda_n(\boldsymbol{A})$是矩阵$\boldsymbol{A}$的所有特征值，那么$\lambda_1^k(\boldsymbol{A})\geq\lambda_2^k(\boldsymbol{A})\geq\cdots\geq\lambda_n^k(\boldsymbol{A})$是矩阵$\boldsymbol{A}^k$的所有特征值。

所以，我们可以推导：
\begin{equation}\begin{aligned}
R =&\, \sum_{k=1}^{\infty}\text{Tr}\,(\boldsymbol{P}\otimes\boldsymbol{P})^k =  \sum_{k=1}^{\infty}\sum_{i=1}^n\lambda_i\left((\boldsymbol{P}\otimes\boldsymbol{P})^k\right)\\
=&\, \sum_{k=1}^{\infty}\sum_{i=1}^n\lambda_i^k\left(\boldsymbol{P}\otimes\boldsymbol{P}\right) = \sum_{i=1}^n \sum_{k=1}^{\infty}\lambda_i^k\left(\boldsymbol{P}\otimes\boldsymbol{P}\right) \\
=&\, \sum_{i=1}^n \frac{\lambda_i \left(\boldsymbol{P}\otimes\boldsymbol{P}\right)}{1 - \lambda_i \left(\boldsymbol{P}\otimes\boldsymbol{P}\right)}
\end{aligned}\label{eq:r-4}\end{equation}
上述过程用到了级数$\frac{x}{1-x}=\sum_{k=1}^{\infty} x^k$，该级数只有在$|x| < 1$才收敛，而很巧的是，我们可以证明$\boldsymbol{P}\otimes\boldsymbol{P}$的特征根绝对值必然不大于1，且通常都小于1：由于$\boldsymbol{P}$是转移矩阵，因此它的每一行之和都为1，因此$\boldsymbol{P}\otimes\boldsymbol{P}$的每一行之和都小于等于1，设$\lambda$、$\boldsymbol{x}$是它的特征值和特征向量，那么$(\boldsymbol{P}\otimes\boldsymbol{P})\boldsymbol{x}=\lambda \boldsymbol{x}$，不失一般性，设$\boldsymbol{x}$绝对值最大的元素为$x_1$，$\boldsymbol{P}\otimes\boldsymbol{P}$的第一个行向量为$\boldsymbol{q}_1^{\top}$，那么我们有$|\lambda| |x_1| = |\boldsymbol{q}_1^{\top}\boldsymbol{x}| \leq |x_1|$，从而$|\lambda| \leq 1$，并且等号成立的条件还是比较苛刻的，所以通常来说都是$|\lambda| < 1$。

注意函数$\frac{x}{1-x}$在$[-1,1)$区间是单调递增的，所以式$\eqref{eq:r-4}$中占主导的是第一项$\frac{\lambda_1 \left(\boldsymbol{P}\otimes\boldsymbol{P}\right)}{1 - \lambda_1 \left(\boldsymbol{P}\otimes\boldsymbol{P}\right)}$，如果非要给整体弄一个上界的话，那么可以是$\frac{n \lambda_1 \left(\boldsymbol{P}\otimes\boldsymbol{P}\right)}{1 - \lambda_1 \left(\boldsymbol{P}\otimes\boldsymbol{P}\right)}$。

## 再次结论

由此可见，<u>如果想要降低重复率$R$，那么我们需要想办法降低矩阵$\boldsymbol{P}\otimes\boldsymbol{P}$的最大特征值</u>。$\boldsymbol{P}\otimes\boldsymbol{P}$是一个非负矩阵，根据非负矩阵的“Frobenius介值定理”，我们有：
\begin{equation}\min_i \sum_j P_{i,j}^2 \leq \lambda_1 (\boldsymbol{P}\otimes\boldsymbol{P}) \leq \max_i \sum_j P_{i,j}^2\end{equation}
关于Frobenius介值定理，基本上在任何一本矩阵分析的书上都有介绍，它说的是“<u>非负矩阵的最大特征值在它每一行的和的最小值于最大值之间</u>”。现在我们知道，为了降低$\boldsymbol{P}\otimes\boldsymbol{P}$的最大特征值，我们需要想办法降低它的每一行之和，即$\sum_j P_{i,j}^2$，并且由于均值不等式
\begin{equation}\sum_j P_{i,j}^2\geq n\left(\frac{\sum_j P_{i,j}}{n}\right)^2 = \frac{1}{n}\end{equation}
知它的最小值为$1/n$，在$P_{i,1}=P_{i,2}=\cdots=P_{i,n}$时取到，因此最终我们得出结论：<strong>要降低最大特征值，就要使得矩阵$\boldsymbol{P}$每一行尽可能均匀，换言之，要降低$\boldsymbol{P}$每一行的方差。</strong>

怎么降低方差呢？很简单，不能出现过高的概率值即可，比如某一行接近one hot的形式，那么平方之后依然接近one hot的形式，那么求和就接近1，远远大于理论最小值$1/n$。什么情况下会出现过高的概率值呢？也不难理解，就是某个字词后面可以接的字词很少，甚至只有1个候选值的时候，比如“忐”几乎只能接“忑”，那么$P_{i=\text{忐},j=\text{忑}}$就相当高，“矩”后面大概接“阵”、“形”比较多，所以“矩”那一行的方差也不小。那怎么才能不出现这种过高的概率值呢？很简单，将高概率值的合并起来，当作一个新词来看待就行了，比如“忐忑”合并为一个词，那么“忐”那一行就不存在了，也就无所谓方差大了。同理，“矩形”、“矩阵”也应该合并为一个词比较好。

所以，说白了这就告诉我们，对于文本生成任务来说，以词为单位比以字为单位更加靠谱（更不容易出现重复解码）。适当地合并一些相关程度比较高的词作为新词加入到词表中，降低转移矩阵的方差，有助于降低重复解码的风险，原论文还给这个操作起了个很高端的名字，叫做Rebalanced Encoding Algorithm，事实上就是这个意思。我们之前词颗粒度的WoBERT在生成任务上比字颗粒度的BERT做得更好，也算是这个结论的验证了吧（参考[《提速不掉点：基于词颗粒度的中文WoBERT》](<https://spaces.ac.cn/archives/7758>)）。

## 一般解码

那这个证明过程容易推广到一般的自回归模型中吗？很遗憾，并不容易。对于一般的自回归模型来说，它相当于每一步的$\boldsymbol{P}$都是不一样的，因此只要模型的性能足够好，其实基本上不会出现重复解码，事实上经过充分预训练的生成式模型，确实很少出现重复解码了。但是，我们又能观察到，哪怕是一般的自回归解码，偶尔也能观察到重复解码现象，尤其是没有经过预训练的模型，这又该怎么解释呢？

前面的小节是基于二元解码模型的，结论是二元解码模型确实容易出现重复解码，那么我们或许可以反过来想，一般的自回归模型出现重复解码现象，是因为它此时退化为了二元解码模型？对于难度比较高的输入，模型可能无法精细捕捉好每一步的转移概率，从而只能将转移矩阵退化为二元解码，这是有可能的。

那么原论文对这一块又是怎么处理的呢？其实也差不多这样。原论文假设一般的自回归模型的转移矩阵，只是在二元解码的转移矩阵$\boldsymbol{P}$的基础上加了个特定时刻的扰动$\tilde{\boldsymbol{P}}_t=\boldsymbol{P}+\boldsymbol{Q}_t$，然后指出在$\boldsymbol{Q}_t$足够小的时候它跟二元解码的差距也足够小（有点像废话），因此二元解码的结果也能代表一般自回归模型了。所以，对一般的自回归模型来说，我们确实很无力了，只能用这种想法跟它沾点边了～

## 文章小结

本文是对Seq2Seq重复解码现象的一次理论分析尝试，主要的篇幅是针对二元解码模型得出一些定量的结果，并且发现这些结果确实能解释一些现象，并且还能带来一些改进的思路，最后比较“勉强”地将二元解码与一般的自回归模型联系了起来。本文在思路上受启发于论文[《A Theoretical Analysis of the Repetition Problem in Text Generation》](<https://papers.cool/arxiv/2012.14660>)，但推导过程都是自己闭门造车的，公式定义也跟原论文略有不同，但总体而言结论是一致的，还请读者自行辨别，如果谬误，敬请斧正。
