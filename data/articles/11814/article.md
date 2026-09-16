# LogSumExp和Softmax的泰勒展开

> 作者：苏剑林 · 科学空间 · 2026-07-13
>
> 原文：<https://spaces.ac.cn/archives/11814>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近看到论文[《The Key to Going Linear: Analysis\-Driven Transformer Linearization》](<https://papers.cool/arxiv/2607.07706>)里边直接对Softmax做近似展开来线性化Attention，有点收获，在此记录一下。

## 一些记号

首先引入如下记号：
\begin{gather}\newcommand{logsumexp}{\mathop{\text{logsumexp}}}\newcommand{softmax}{\mathop{\text{softmax}}}
\boldsymbol{x} = (x_1, x_2, \cdots, x_n) \in \mathbb{R}^n,\qquad\overline{\boldsymbol{x}} = \frac{1}{n}\sum_{i=1}^n x_i \\
\logsumexp(\boldsymbol{x}) = \log\sum_{i=1}^n e^{x_i} = \log n + \log \overline{e^{\boldsymbol{x}}} \\
\softmax(\boldsymbol{x}) = \frac{e^{\boldsymbol{x}}}{\sum_{i=1}^n e^{x_i}} = e^{\boldsymbol{x} - \logsumexp(\boldsymbol{x})} \\
\end{gather}
并约定向量的函数运算都是Hadamard意义下的，例如
\begin{equation}e^{\boldsymbol{x}} = (e^{x_1},e^{x_2},\cdots,e^{x_n}),\qquad\boldsymbol{x}^2 = (x_1^2,x_2^2,\cdots,x_n^2)\end{equation}
特别地，要仔细留意$\overline{e^{\boldsymbol{x}}}$与$e^{\overline{\boldsymbol{x}}}$、$\overline{\boldsymbol{x}^2}$与$\overline{\boldsymbol{x}}^2$的区别：前者是先对每个分量做函数再平均，后者是先平均再做函数，二者一般不相等。

除了上述定义外，$\logsumexp$和$\softmax$还可以通过梯度联系起来：
\begin{equation}\softmax(\boldsymbol{x})=\nabla_{\boldsymbol{x}} \logsumexp(\boldsymbol{x})\label{eq:grad-lse}\end{equation}

## 基本展开

接着我们对$\logsumexp(\boldsymbol{x})$做展开。引入$\boldsymbol{y} = \boldsymbol{x} - \overline{\boldsymbol{x}}$，那么
\begin{equation}\begin{aligned}
\logsumexp(\boldsymbol{x}) =&\, \log n + \overline{\boldsymbol{x}} + \log \overline{e^{\boldsymbol{y}}} \\
=&\, \log n + \overline{\boldsymbol{x}} + \log \left(\,\overline{1 + \boldsymbol{y} + \frac{\boldsymbol{y}^2}{2} + \frac{\boldsymbol{y}^3}{6} +\frac{\boldsymbol{y}^4}{24} + \cdots}\,\right) \\
=&\, \log n + \overline{\boldsymbol{x}} + \log \left(1 + \frac{\overline{\boldsymbol{y}^2}}{2} + \frac{\overline{\boldsymbol{y}^3}}{6} + \frac{\overline{\boldsymbol{y}^4}}{24} + \cdots\right) \\
=&\, \log n + \overline{\boldsymbol{x}} + \frac{\overline{\boldsymbol{y}^2}}{2} + \frac{\overline{\boldsymbol{y}^3}}{6} + \left(\,\frac{\overline{\boldsymbol{y}^4}}{24} - \frac{(\,\overline{\boldsymbol{y}^2}\,)^2}{8}\,\right) + \cdots
\end{aligned}\label{eq:logsumexp-series}\end{equation}
最后一步用到了$\log(1+t) = t - t^2/2 + t^3/3 - \cdots$，如果有需要，还可以继续往下展开。如果读者不谙此道，也可以交给Kimi完成。

这里引入$\overline{\boldsymbol{x}}$的偏置是一个简化形式的小技巧，如果不引入，也能得到等价的结果，只不过会将$\boldsymbol{y}$的各个项显式展开：
\begin{equation}\logsumexp(\boldsymbol{x})= \log n + \overline{\boldsymbol{x}} + \underbrace{\left(\frac{\overline{\boldsymbol{x}^2}}{2}-\frac{\overline{\boldsymbol{x}}^{\,2}}{2}\right)}_{\overline{\boldsymbol{y}^2}/2} + \underbrace{\left(\frac{\overline{\boldsymbol{x}^3}}{6}-\frac{\overline{\boldsymbol{x}}\,\overline{\boldsymbol{x}^2}}{2}+\frac{\overline{\boldsymbol{x}}^{\,3}}{3}\right)}_{\overline{\boldsymbol{y}^3}/6} + \cdots\end{equation}

## 求个梯度

对于$\softmax(\boldsymbol{x})$，我们直接利用恒等式$\eqref{eq:grad-lse}$，对式$\eqref{eq:logsumexp-series}$两边求导即得$\softmax(\boldsymbol{x})$的展开式。为此，我们利用
\begin{equation}\nabla_{\boldsymbol{x}} \overline{\boldsymbol{x}} = \frac{\boldsymbol{1}}{n},\qquad\nabla_{\boldsymbol{x}} \overline{\boldsymbol{y}^m} = \frac{m}{n} \left(\boldsymbol{y}^{m-1} - \overline{\boldsymbol{y}^{m-1}}\right) \end{equation}
得
\begin{equation}\softmax(\boldsymbol{x}) = \frac{1}{n}\left(1 + \boldsymbol{y} + \left(\frac{\boldsymbol{y}^2}{2} - \frac{\overline{\boldsymbol{y}^2}}{2}\right) + \left(\frac{\boldsymbol{y}^3}{6} - \frac{\overline{\boldsymbol{y}^3}}{6} - \frac{\overline{\boldsymbol{y}^2}\, \boldsymbol{y}}{2}\right) + \cdots\right)\end{equation}
注意如果截断有限项，那么右端依然能保持所有分量求和为1（从$\nabla_{\boldsymbol{x}} \overline{\boldsymbol{y}^m}$的梯度形式即可看出），但是无法保证每个分量的非负性，这一点在实际应用中需要留意（比如需要取对数求熵的时候）。

除了求梯度外，我们还可以用$\softmax(\boldsymbol{x}) = e^{\boldsymbol{x} - \logsumexp(\boldsymbol{x})} = e^{\boldsymbol{y} - \logsumexp(\boldsymbol{y})}$来展开，结合式$\eqref{eq:logsumexp-series}$得：
\begin{equation}\begin{aligned}
e^{\boldsymbol{y}} e^{-\logsumexp(\boldsymbol{y})} =&\, \left(1 + \boldsymbol{y} + \frac{\boldsymbol{y}^2}{2} + \frac{\boldsymbol{y}^3}{6} + \cdots\right)\exp\left(-\log n -  \frac{\overline{\boldsymbol{y}^2}}{2} - \frac{\overline{\boldsymbol{y}^3}}{6} - \cdots\right) \\
=&\, \frac{1}{n}\left(1 + \boldsymbol{y} + \left(\frac{\boldsymbol{y}^2}{2} - \frac{\overline{\boldsymbol{y}^2}}{2}\right) + \left(\frac{\boldsymbol{y}^3}{6} - \frac{\overline{\boldsymbol{y}^3}}{6} - \frac{\overline{\boldsymbol{y}^2}\, \boldsymbol{y}}{2}\right) + \cdots\right)
\end{aligned}\end{equation}

## 稀疏注意

那这两个展开式有什么应用呢？首先是$\logsumexp(\boldsymbol{x})$的，它跟[MoBA](<https://papers.cool/arxiv/2502.13189>)这类Block Sparse Attention的块打分有关。我们知道，Full Attention按$e^{\boldsymbol{q}\cdot \boldsymbol{k}}$给每个Token打分，那对于某个块$\mathcal{B}$，它的打分可以合理定义为每个Token的分数之和，等价于
\begin{equation}s_\mathcal{B} = \log \sum_{t\in\mathcal{B}} e^{\boldsymbol{q}\cdot \boldsymbol{k}_t}\end{equation}
它的一阶近似是$\log |\mathcal{B}| + \boldsymbol{q}\cdot\overline{\boldsymbol{k}}$，其中$\overline{\boldsymbol{k}}=\frac{1}{|\mathcal{B}|}\sum_{t\in\mathcal{B}} \boldsymbol{k}_t$，这正好对应着MoBA用AvgPooling作为块内Landmark向量的经验做法。如果感觉一阶近似太粗糙，那么可以考虑高阶修正，根据式$\eqref{eq:logsumexp-series}$，二阶项是
\begin{equation}\frac{1}{2|\mathcal{B}|}\sum_{t\in\mathcal{B}} (\boldsymbol{q}\cdot (\boldsymbol{k}_t - \overline{\boldsymbol{k}}))^2 = \frac{1}{2}\boldsymbol{q}^{\top}\underbrace{\left(\frac{1}{|\mathcal{B}|}\sum_{t\in\mathcal{B}}(\boldsymbol{k}_t - \overline{\boldsymbol{k}})(\boldsymbol{k}_t - \overline{\boldsymbol{k}})^{\top}\right)}_{\boldsymbol{\Sigma}}\boldsymbol{q}\end{equation}
即二阶近似是$\log |\mathcal{B}| + \boldsymbol{q}\cdot\overline{\boldsymbol{k}} + \boldsymbol{q}^{\top}\boldsymbol{\Sigma}\boldsymbol{q}/2$，其中$\boldsymbol{\Sigma}$正好是块内$\boldsymbol{k}_t$的协方差矩阵。进一步地，我们可以考虑对角近似来降低计算成本。这一高阶修正的思路，跟[SPLA](<https://papers.cool/arxiv/2601.22379>)本质上是相同的。后续相关工作还有[HiLS](<https://papers.cool/arxiv/2607.02980>)，它另外学了一个非等权平均的中心向量来取代$\overline{\boldsymbol{k}}$。

## 线性注意

至于$\softmax(\boldsymbol{x})$展开式的应用，自然是线性化Attention了。在[《Transformer升级之路：5、作为无限维的线性Attention》](<https://spaces.ac.cn/archives/8601>)中，我们总结了三种线性化的思路，它们都是对$e^{\boldsymbol{q}\cdot\boldsymbol{k}}$做近似。然而，近似$e^{\boldsymbol{q}\cdot\boldsymbol{k}}$后还要归一化，倒不如直接对归一化后的$\softmax(\boldsymbol{x})$做近似。

记$\boldsymbol{x}=(\boldsymbol{q}\cdot \boldsymbol{k}_1,\cdots,\boldsymbol{q}\cdot \boldsymbol{k}_n)$，那么有一阶近似：
\begin{equation}\softmax(\boldsymbol{x})_i \approx \frac{1}{n} + \frac{1}{n}\boldsymbol{q}\cdot (\boldsymbol{k}_i - \overline{\boldsymbol{k}})\end{equation}
这正是开头论文所讨论的方案，它本身也很直观：平均向量$\overline{\boldsymbol{k}}$为注意力的相对大小提供了一个基准，相似度比$\overline{\boldsymbol{k}}$大的Token，要加大注意力，否则减少。为了提高精度，[Based](<https://papers.cool/arxiv/2402.18668>)将$e^{\boldsymbol{q}\cdot\boldsymbol{k}}$展开到了二阶（此时刚好保证非负，参考[这里](<https://spaces.ac.cn/archives/7919>)），但从我们的视角看，考虑$\softmax(\boldsymbol{x})$的二阶近似更科学，二阶项是
\begin{equation}\frac{1}{2n}\left[(\boldsymbol{q}\cdot(\boldsymbol{k}_i-\overline{\boldsymbol{k}}))^2 - \frac{1}{n}\sum_{j=1}^n(\boldsymbol{q}\cdot(\boldsymbol{k}_j-\overline{\boldsymbol{k}}))^2\right]\end{equation}
其中$(\boldsymbol{q}\cdot(\boldsymbol{k}_i-\overline{\boldsymbol{k}}))^2$可以通过引入外积来变成内积：
\begin{equation}(\boldsymbol{q}\cdot(\boldsymbol{k}_i-\overline{\boldsymbol{k}}))^2 = \boldsymbol{q}^{\top}(\boldsymbol{k}_i-\overline{\boldsymbol{k}})(\boldsymbol{k}_i-\overline{\boldsymbol{k}})^{\top}\boldsymbol{q} = \left\langle \boldsymbol{q}\boldsymbol{q}^{\top},\, (\boldsymbol{k}_i-\overline{\boldsymbol{k}})(\boldsymbol{k}_i-\overline{\boldsymbol{k}})^{\top} \right\rangle_F\end{equation}
所以跟Based类似，将Softmax Attention的Softmax截断到二阶近似，结果也是线性注意力。

## 文章小结

本文分别推导了LogSumExp和Softmax的泰勒展开式，并讨论了它们的两个潜在应用。
