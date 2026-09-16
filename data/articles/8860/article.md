# CoSENT（二）：特征式匹配与交互式匹配有多大差距？

> 作者：苏剑林 · 科学空间 · 2022-01-12
>
> 原文：<https://spaces.ac.cn/archives/8860>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

一般来说，文本匹配有交互式（Interaction\-based）和特征式（Representation\-based）两种实现方案，其中交互式是指将两个文本拼接在一起当成单文本进行分类，而特征式则是指两个句子分别由编码器编码为句向量后再做简单的融合处理（算cos值或者接一个浅层网络）。通常的结论是，交互式由于使得两个文本能够进行充分的比较，所以它准确性通常较好，但明显的缺点是在检索场景的效率较差；而特征式则可以提前计算并缓存好句向量，所以它有着较高的效率，但由于句子间的交互程度较浅，所以通常效果不如交互式。

上一篇文章笔者介绍了CoSENT，它本质上也是一种特征式方案，并且相比以往的特征式方案效果有所提高。于是笔者的好胜心就上来了：CoSENT能比得过交互式吗？特征式相比交互式的差距有多远呢？本文就来做个比较。

## 自动阈值

在文章[《CoSENT（一）：比Sentence\-BERT更有效的句向量方案》](<https://spaces.ac.cn/archives/8847>)中，我们评测CoSENT所用的指标是Spearman系数，它是一个只依赖于预测结果相对顺序的指标，不依赖于阈值，比较适合检索场景的评测。但如果评测指标是accuracy或者F1这些分类指标，则必须确定一个阈值，将预测结果大于这个数的预测结果视为正、小于则为负，然后才能计算指标。在二分类的场景，我们用二分法就可以有效地确定这个阈值。

然而，搜索阈值确实并非二分类所独有，一般的多分类任务其实也存在着同样的需求，所以这里将这个问题展开来谈谈。比如对于$n$分类的预测分布$[p_1,p_2,\dots,p_n]$，我们一般是用概率最大的类别，即$\mathop{\text{argmax}}\,(p_1,p_2,\dots,p_n)$作为预测类别，但如果是类别不均衡的场景，这样做其实未必是最优的。我们可以通过验证集来搜索一个向量$[t_1,t_2,\cdots,t_n]$，然后用
\begin{equation}\mathop{\text{argmax}}\,(p_1 t_1,p_2 t_2,\dots,p_n t_n)\end{equation}
作为预测类别，这里的$[t_1,t_2,\cdots,t_n]$就相当于是多类别场景下的阈值。

那么，怎么搜索出$[t_1,t_2,\cdots,t_n]$呢？搜索目标自然是指标最大，但accuracy或者F1这些都是不可导的，因此梯度下降肯定是不考虑了；而又因为待搜索参数是一个多维向量，所以二分法也不大好用。这里介绍一种名为“Powell法”的求解方案。Powell法的数学细节比较多，这里不打算展开，简单来说，Powell法是一种求解低维无约束优化的算法，它不需要梯度，并且效率也相对较高，这里低维指的是待优化参数通常不超过100个（你总不能指望它去求解神经网络）。最关键是，Powell法在Scipy中有现成实现，在<code>scipy.optimize.minimize</code>中指定<code>method='Powell'</code>即可调用它。

对于上述问题，参考代码如下：



<pre><code class="language-python">import numpy as np
from scipy.optimize import minimize

def loss(t):
    """这里的y_true.shape=[batch_size],
    y_pred.shape=[batch_size, num_classes]
    """
    t = (np.tanh(t) + 1) / 2
    return -np.mean(y_true == (y_pred * t[None]).argmax(1))

options = {'xtol': 1e-10, 'ftol': 1e-10, 'maxiter': 100000}
result = minimize(
    loss, np.zeros_like(y_pred[:1]), method='Powell', options=options
)
thresholds = (np.tanh(result.x) + 1) / 2</code></pre>



## 实验结果

有了自动确定阈值的方法后，我们就可以检验分类性能了。笔者在ATEC、BQ、LCQMC、PAWSX这4个数据集上做了实验，分别对比了CoSENT、Sentence\-BERT和交互式（记为Interact）三种方案的效果。公平起见，每种方法都用Powell法在验证集上确定最优阈值，然后用该阈值报告测试集的效果，哪怕交互式的也是如此。

> <strong>实验代码：[https://github\.com/bojone/CoSENT/tree/main/accuracy](<https://github.com/bojone/CoSENT/tree/main/accuracy>)</strong>

实验结果如下（指标是accuracy）：
\begin{array}{c|cccc|c}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{Avg}\\
\hline
\text{BERT+CoSENT} & \textbf{85.81} & 83.24 & 86.67 & 76.30 & 83.00 \\
\text{Sentence-BERT} & 84.93 & 82.46 & 87.42 & 65.33 & 80.04\\
\text{BERT+Interact} & 85.49 & \textbf{83.88} & \textbf{87.80} & \textbf{81.30} & \textbf{84.62} \\
\hline
\text{RoBERTa+CoSENT} & 85.93 & 83.42 & 87.63 & 76.55 & 83.38 \\
\text{Sentence-RoBERTa} & 85.34 & 82.52 & 88.14 & 68.35 & 81.09 \\
\text{RoBERTa+Interact} & \textbf{86.04} & \textbf{83.62} & \textbf{88.22} & \textbf{83.33} & \textbf{85.30} \\
\hline
\end{array}

实验结果显示，从效果上来看，交互式确实是“王者”地位，但是特征式（CoSENT和Sentence\-BERT/RoBERTa）的效果差距并没有笔者想象中大。客观来说，在ATEC和BQ两个任务上，交互式Interact与特征式CoSENT并无显著差异，而在LCQMC任务上，交互式Interact与特征式Sentence\-BERT/RoBERTa并无显著差异。

唯一拉开明显差距的是PAWSX，在[《无监督语义相似度哪家强？我们做了个比较全面的评测》](<https://spaces.ac.cn/archives/8321>)、[《中文任务还是SOTA吗？我们给SimCSE补充了一些实验》](<https://spaces.ac.cn/archives/8348>)中我们可以发现，几乎所有无监督句向量方法都在PAWSX上失效。为什么呢？因为PAWSX的负样本几乎全是“对抗样本”，就是字面重叠度非常高但语义却不同的负样本。所以，对于这种无监督方法“全线崩溃”的“高难”负样本，哪怕用标注数据去训练，也自然需要更深层的交互才能更好地识别出来。

## 理论极限

有些读者可能会好奇：能否通过理论分析出特征式方案的理论极限呢？可能让人意外的是，这个分析其实不难，而答案是：

> 理论上来说，交互式能做到的效果，特征式“几乎”都能做到。

怎么得到这个结果呢？事实上用本博客以往介绍过的文章就足够了。首先，我们假定样本对的相似度在0～1之间，并且样本对是无序的，即$\text{sim}(x,y)=\text{sim}(y,x)$，那么如果有$n$个样本，那么我们每两个样本之间算相似度（不管实际的相似度是怎么算来的），就得到一个相似度矩阵$S$，它是一个“正定对称矩阵”（或者严格点，是半正定）。按照线性代数的结果，正定对称矩阵的SVD分解必然是$S=U\Lambda U^{\top}$的形式，其中$U$是正交矩阵而$\Lambda$是对角阵，那么我们有$S=U\Lambda U^{\top}=(U\sqrt{\Lambda})(U\sqrt{\Lambda})^{\top}$。这就表明了，正定对称矩阵一定可以分解为$S=BB^{\top}$的形式，这等价于说，每个样本$i$可以表示为一个$n$维向量$v_i$，使得$S_{i,j}=\langle x_i, y_j\rangle$。

至此，所有的结果都是有理论保证并且精确相等的，只不过目前的“$n$维向量”实在是太大了，所以接下来应该往降维角度想。此时，我们去年介绍过的“JL引理”（参考[《让人惊叹的Johnson\-Lindenstrauss引理：理论篇》](<https://spaces.ac.cn/archives/8679>)）就可以登场了，它告诉我们，别管原来是多少维的，$n$个向量都可以降到$\mathcal{O}(\log n)$维，而保持内积近似不变，在[《让人惊叹的Johnson\-Lindenstrauss引理：应用篇》](<https://spaces.ac.cn/archives/8706>)中我们还近似估计了这个量级应该是$8\log n$左右，所以对于BERT base的768维向量来说，理论上通过内积来拟合上百万个样本的两两相似度都不成问题。所以，基于内积的、维度达到几百维的“特征式”方案，理论上是可以相当精确地达到交互式效果的。

那为什么在PAWSX这样的困难数据集上两者有明显差异呢？个人认为这是“神经网络和cos指标的连续性”与“文本匹配天然存在的对抗性”之间的矛盾造成的。

神经网络本身就是一个连续函数，然后编码器负责将句子压缩到一个句向量中，其结果的连续性必然是非常好的，这里的连续性，指的是句子的微小改动，导致句向量的改动也是微小的；同时，cos的连续性也非常好，即如果$\Delta v$比较小，那么$\cos(u,v)$和$\cos(u, v+\Delta v)$的差距也很小。所以，总的来说就是“特征式”的方案连续性会非常好。但问题是，人设计出来的语言天然存在对抗性，即字面上的微小改动能导致标注结果的巨大变化，经典的就是加个“不”字导致所谓的“语义反转”，说白了就是连续性并不好。

于是，在此类任务之下，连续性非常好的“特征式”方案要去拟合对抗性明显的数据集，就会非常困难。当然，前面我们已经分析过理论上是可以拟合的，所以实际上是拟合确实能拟合，但需要训练比较多的epoch来“磨”掉特征式方案原有的连续性，但比较多的epoch也造成了更严重的过拟合。因此，CoSENT的训练loss也能降到接近于0（说明拟合能力没问题），但是验证集的效果没有交互式的好。至于交互式，模型一开始就同时接触到了两个样本，在后面的层中模型可以自行拟合和放大差异，从而在交互式方案中连续性与对抗性的矛盾并没有那么严重，因而效果更好。

## 文章小结

本文从理论和实验两个角度地探讨了特征式匹配与交互式匹配的效果差距，此外还讨论了多分类问题中阈值的自动搜索问题。
