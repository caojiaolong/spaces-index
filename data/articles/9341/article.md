# CoSENT（三）：作为交互式相似度的损失函数

> 作者：苏剑林 · 科学空间 · 2022-11-09
>
> 原文：<https://spaces.ac.cn/archives/9341>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《CoSENT（一）：比Sentence\-BERT更有效的句向量方案》](<https://spaces.ac.cn/archives/8847>)中，笔者提出了名为“CoSENT”的有监督句向量方案，由于它是直接训练cos相似度的，跟评测目标更相关，因此通常能有着比Sentence\-BERT更好的效果以及更快的收敛速度。在[《CoSENT（二）：特征式匹配与交互式匹配有多大差距？》](<https://spaces.ac.cn/archives/8860>)中我们还比较过它跟交互式相似度模型的差异，显示它在某些任务上的效果还能直逼交互式相似度模型。

然而，当时笔者是一心想找一个更接近评测目标的Sentence\-BERT替代品，所以结果都是面向有监督句向量的，即特征式相似度模型。最近笔者突然反应过来，CoSENT其实也能作为交互式相似度模型的损失函数。那么它跟标准选择交叉熵相比孰优孰劣呢？本文来补充这部分实验。

## 基础回顾

CoSENT提出之初，是作为一个有监督句向量的损失函数：
\begin{equation}\log \left(1 + \sum\limits_{\text{sim}(i,j) \gt \text{sim}(k,l)} e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))}\right)\end{equation}
其中$i,j,k,l$是四个训练样本（比如四个句子），$u_i, u_j, u_k, u_l$是它们想要学习的句向量（比如它们经过BERT后的&#91;CLS&#93;向量），$\cos(\cdot,\cdot)$代表两个向量的余弦相似度，$\text{sim}(\cdot,\cdot)$则代表它们的相似度标签。所以这个损失函数的定义也很清晰，就是如果你认为$(i,j)$的相似度应该大于$\text{sim}(k,l)$的相似度，那么就往$\log$里边加入一项$e^{\lambda(\cos(u_k, u_l) - \cos(u_i, u_j))}$。

从这个形式就可以看出，当时CoSETN就是为了有监督训练余弦相似度的特征式模型的，包括“CoSENT”这个名字也是这样来的（Cosine Sentence）。然而，抛开余弦相似度这一层面不谈，CoSENT本质上是一个只依赖于标签相对顺序的损失函数，它跟余弦相似度没有必然联系，我们可以将它一般化为
\begin{equation}\log \left(1 + \sum\limits_{\text{sim}(i,j) \gt \text{sim}(k,l)} e^{\lambda(f(k,l) - f(i,j))}\right)\end{equation}
其中$f(\cdot,\cdot)$是任意标量输出函数（一般不需要加激活函数），代表要学习的相似度模型，包括将两个输入拼接成一个文本输入到BERT中的“交互式相似度”模型！

## 实验比较

训练交互式相似度的常规方式是最后构建一个两节点的输出，然后加上softmax，用交叉熵（下表简称CE）作为损失函数，这也等价于在前面的$f(\cdot,\cdot)$上加sigmoid激活，然后用单节点的二分类交叉熵。不过这种做法也就适合二分类形式的标签，如果连续型的打分（比如STS\-B是1～5分），就不大适合了，此时通常要转化为回归问题。但CoSENT没有这个限制，因为它只需要标签的序信息，这个特点跟常用的评测指标spearman系数是一致的。

两者的对比实验，参考代码如下：

> [<strong>https://github\.com/bojone/CoSENT/blob/main/accuracy/interact\_cosent\.py</strong>](<https://github.com/bojone/CoSENT/blob/main/accuracy/interact_cosent.py>)

实验结果为
$$\begin{array}{c}
\text{评测指标为spearman系数} \\
{\begin{array}{c|ccccc}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{avg}\\
\hline
\text{BERT + CE} & 48.01 & 71.96 & 78.53 & 68.59 & 66.77 \\
\text{BERT + CoSENT} & 48.09 & 72.25 & 78.70 & 69.34 & 67.10 \\
\hline
\text{RoBERTa + CE} & 49.70 & 73.20 & 79.13 & 70.52 & 68.14 \\
\text{RoBERTa + CoSENT} & 49.82 & 73.09 & 78.78 & 70.54 & 68.06 \\
\hline
\end{array}} \\
\\
\text{评测指标为accuracy} \\
{\begin{array}{c|ccccc}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{avg}\\
\hline
\text{BERT + CE} & 85.38 & 83.57 & 88.10 & 81.45 & 84.63 \\
\text{BERT + CoSENT} & 85.55 & 83.73 & 87.92 & 81.85 & 84.76 \\
\hline
\text{RoBERTa + CE} & 85.97 & 84.67 & 88.14 & 82.85 & 85.41 \\
\text{RoBERTa + CoSENT} & 86.06 & 84.23 & 88.14 & 83.03 & 85.37 \\
\hline
\end{array}}
\end{array}$$

可以看到，没有惊喜，CE和CoSENT的效果基本一致。非要挖掘一些细致区别的话，可以看到在BERT中，CoSENT的效果相对好些，在RoBERTa中基本没区别了，以及在PAWSX这个任务上，CoSENT的提升相对明显些，其他任务基本持平。如此，可以“弱弱地”下一个结论：

> 当模型较弱（BERT弱于RoBERTa）或者任务较难（PAWSX相对来说比其他三个任务都难）时，CoSENT<strong><u>或许</u></strong>能取得比CE更好的效果。

注意，是“或许”，笔者也不能保证。实事求是地说，我也不认为两者构成什么显著差异。不过可以猜测，因为两种损失函数的形式有明显的差异，所以哪怕最终指标上差不多，模型内部应该也有一定差异，这时候或许可以考虑模型融合？

## 文章小结

本文主要思考和实验了CoSENT在交互式相似度模型中的可行性，最终结论是“可行但效果没什么提升”。
