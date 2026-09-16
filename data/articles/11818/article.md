# 基于排序不等式的相似度指标

> 作者：苏剑林 · 科学空间 · 2026-07-16
>
> 原文：<https://spaces.ac.cn/archives/11818>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨天看到 [@Ji\_Ha\_Kim](<https://x.com/Ji_Ha_Kim>) 同学分享了一篇有趣的论文[《Beyond Cosine Similarity》](<https://papers.cool/arxiv/2602.05266>)，里边提出基于排序不等式来构建新的更宽松的相似度指标，看了后感觉颇有意思，特来简单记录一下。

## 余弦相似度

对于两个向量$\boldsymbol{x},\boldsymbol{y}\in\mathbb{R}^d$，常用的归一化相似度指标是余弦相似度：
\begin{equation}\cos(\boldsymbol{x},\boldsymbol{y}) = \frac{\boldsymbol{x}\cdot \boldsymbol{y}}{\Vert\boldsymbol{x}\Vert\, \Vert\boldsymbol{y}\Vert} \in [-1, 1]\end{equation}
抛开几何意义，从纯代数的角度看，余弦相似度是基于柯西不等式
\begin{equation}-\Vert\boldsymbol{x}\Vert\, \Vert\boldsymbol{y}\Vert \leq \boldsymbol{x}\cdot \boldsymbol{y}\leq \Vert\boldsymbol{x}\Vert\, \Vert\boldsymbol{y}\Vert\end{equation}
构建的归一化内积，当$\cos(\boldsymbol{x},\boldsymbol{y})=\pm 1$时，意味着存在$k > 0$使得$\boldsymbol{y} = \pm k\boldsymbol{x}$，这表明余弦相似度对模长不敏感，它度量的是两个向量方向的相似程度，或者说线性相关程度。特别地，由均值不等式可得$\Vert\boldsymbol{x}\Vert\, \Vert\boldsymbol{y}\Vert \leq \frac{1}{2}(\Vert\boldsymbol{x}\Vert^2 +  \Vert\boldsymbol{y}\Vert^2)$，代入到柯西不等式进一步放缩得
\begin{equation}-\frac{1}{2}(\Vert\boldsymbol{x}\Vert^2 +  \Vert\boldsymbol{y}\Vert^2) \leq \boldsymbol{x}\cdot \boldsymbol{y} \leq \frac{1}{2}(\Vert\boldsymbol{x}\Vert^2 +  \Vert\boldsymbol{y}\Vert^2)\end{equation}
第一个等号成立的条件是$\boldsymbol{x}=-\boldsymbol{y}$，第二个等号成立的条件是$\boldsymbol{x}=\boldsymbol{y}$，由此我们可以定义新的归一化内积
\begin{equation}\newcommand{dcos}{\mathop{\text{dcos}}}\dcos(\boldsymbol{x},\boldsymbol{y}) = \frac{\boldsymbol{x}\cdot \boldsymbol{y}}{\frac{1}{2}(\Vert\boldsymbol{x}\Vert^2 +  \Vert\boldsymbol{y}\Vert^2)}\end{equation}
这时候有$\dcos(\boldsymbol{x},\boldsymbol{y}) = \pm 1 \Leftrightarrow \boldsymbol{x}=\pm\boldsymbol{y}$，即$\dcos$是度量两个向量完全一致程度的指标。

## 排序不等式

对于内积$\boldsymbol{x}\cdot \boldsymbol{y}$，还成立排序不等式
\begin{equation}\boldsymbol{x}^{\downarrow}\cdot \boldsymbol{y}^{\uparrow}=\boldsymbol{x}^{\uparrow}\cdot \boldsymbol{y}^{\downarrow}\leq \boldsymbol{x}\cdot \boldsymbol{y} \leq \boldsymbol{x}^{\uparrow}\cdot \boldsymbol{y}^{\uparrow} = \boldsymbol{x}^{\downarrow}\cdot \boldsymbol{y}^{\downarrow}\end{equation}
这里$\boldsymbol{x}^{\downarrow},\boldsymbol{x}^{\uparrow}$是指将$\boldsymbol{x}$的各个分量按降/升序排列得到的新向量（$\boldsymbol{y}$同理），简单来说就是“倒序内积 ≤ 乱序内积 ≤ 同序内积”。第一个等号成立的条件，是$\boldsymbol{x},\boldsymbol{y}$完全反序，即对任意$i,j$都恒成立$(x_i - x_j)(y_i - y_j)\leq 0$；第二个等号成立的条件，则是$\boldsymbol{x},\boldsymbol{y}$完全同序了，即$(x_i - x_j)(y_i - y_j)\geq 0$。

显然，重新排序不改变模长，即$\Vert\boldsymbol{x}^{\downarrow}\Vert=\Vert\boldsymbol{x}^{\uparrow}\Vert=\Vert\boldsymbol{x}\Vert$，所以进一步结合柯西不等式得
\begin{equation}-\Vert\boldsymbol{x}\Vert\, \Vert\boldsymbol{y}\Vert \leq \boldsymbol{x}^{\uparrow}\cdot \boldsymbol{y}^{\downarrow}\leq \boldsymbol{x}\cdot \boldsymbol{y} \leq \boldsymbol{x}^{\uparrow}\cdot \boldsymbol{y}^{\uparrow} \leq \Vert\boldsymbol{x}\Vert\, \Vert\boldsymbol{y}\Vert\end{equation}
这说明对于$\boldsymbol{x}\cdot \boldsymbol{y}$来说，排序不等式比柯西不等式更加紧凑。排序不等式的标准证明方式是局部调整法，从恒等式
\begin{equation}(x_i - x_j)(y_i - y_j) = (x_i y_i + x_j y_j) - (x_i y_j + x_j y_i)\end{equation}
出发，右端第一项是$(x_i, x_j)$与$(y_i, y_j)$的内积，第二项是交换$x_i,x_j$或$y_i,y_j$的内积，如果$x_i > x_j$且$y_i > y_j$（同序），那么左端大于0，意味着交换从同序变成反序后，内积会变小。这样一来，最大就是在完全同序情况下取到，同理最小就是在完全反序下取到。

## 排序相似度

如果已知$l(\boldsymbol{x},\boldsymbol{y})\leq \boldsymbol{x}\cdot \boldsymbol{y} \leq u(\boldsymbol{x},\boldsymbol{y})$，并且两边等号都是可达的，那么可以一般地构建相似度指标
\begin{equation}2\cdot\frac{\boldsymbol{x}\cdot \boldsymbol{y} - l(\boldsymbol{x},\boldsymbol{y})}{u(\boldsymbol{x},\boldsymbol{y}) - l(\boldsymbol{x},\boldsymbol{y})} - 1 = \frac{2\cdot\boldsymbol{x}\cdot \boldsymbol{y} - l(\boldsymbol{x},\boldsymbol{y}) - u(\boldsymbol{x},\boldsymbol{y})}{u(\boldsymbol{x},\boldsymbol{y}) - l(\boldsymbol{x},\boldsymbol{y})} \in [-1, 1]\end{equation}
代入排序不等式，得
\begin{equation}\newcommand{rcos}{\mathop{\text{rcos}}}\rcos(\boldsymbol{x},\boldsymbol{y}) = \frac{2\cdot\boldsymbol{x}\cdot \boldsymbol{y} - \boldsymbol{x}^{\uparrow}\cdot\boldsymbol{y}^{\downarrow} - \boldsymbol{x}^{\uparrow}\cdot\boldsymbol{y}^{\uparrow}}{\boldsymbol{x}^{\uparrow}\cdot\boldsymbol{y}^{\uparrow} - \boldsymbol{x}^{\uparrow}\cdot\boldsymbol{y}^{\downarrow}}\end{equation}
那么$\rcos(\boldsymbol{x},\boldsymbol{y})=1$意味着$\boldsymbol{x},\boldsymbol{y}$完全同序，$\rcos(\boldsymbol{x},\boldsymbol{y})=-1$意味着$\boldsymbol{x},\boldsymbol{y}$完全反序，如果说余弦相似度描述了线性相关程度，那么$\rcos$一定程度上描述了一些的非线性相关性。$\rcos$在平移和正缩放下保持不变，即
\begin{equation}\rcos(a \boldsymbol{x} + b\boldsymbol{1},c\boldsymbol{y} + d\boldsymbol{1})=\rcos(\boldsymbol{x},\boldsymbol{y})\end{equation}
但对一般的逐分量严格单调变换（如逐分量立方）并非如此，即$\rcos$保留并利用了数值的大小，而不只依赖排序。

## 联系与区别

需要指出的是，虽然本文是受[《Beyond Cosine Similarity》](<https://papers.cool/arxiv/2602.05266>)启发而来，但这里构建的排序相似度跟原论文略有不同，原论文的构建方式是（$\newcommand{recos}{\mathop{\text{recos}}}\recos$也是原论文的记号）
\begin{equation}\recos(\boldsymbol{x},\boldsymbol{y}) = \frac{\boldsymbol{x}\cdot\boldsymbol{y}}{|\boldsymbol{x}^{\uparrow}\cdot\boldsymbol{y}^{\updownarrow}|},\qquad \boldsymbol{y}^{\updownarrow} = \begin{cases}\boldsymbol{y}^{\uparrow}, & \boldsymbol{x}\cdot\boldsymbol{y} > 0\\ \boldsymbol{y}^{\downarrow}, & \boldsymbol{x}\cdot\boldsymbol{y} < 0\end{cases}\end{equation}
也就是按内积的正负选择同序或反序的界。这样定义虽然也能将结果限制在$[-1,1]$内，但此时内积的正负无法用来区分同序和反序，个人认为不如本文的定义合理。

举个例子，$\boldsymbol{x}=(1,2,3), \boldsymbol{y}=(6,5,4)$，这两个向量是完全反序的，我们认为排序相似度指标应该至少给出负数的结果，但很显然$\recos(\boldsymbol{x},\boldsymbol{y})$给出的结果是正的，而$\rcos(\boldsymbol{x},\boldsymbol{y})$则能给出负数的结果。本文的定义直接用排序不等式的双边界做归一化，让$\pm 1$分别锚定完全同序与完全反序，窃以为语义上更加合理。

当然，实际应用中究竟是$\rcos$还是$\recos$更合理，这取决于具体场景，这里声称的更加合理，仅仅是针对“排序相似度”这一名词的含义而言的。

## 秩相关系数

如果我们将$\boldsymbol{x},\boldsymbol{y}$视为一个个二维数据点$(x_1,y_1),(x_2,y_2),\cdots$，那么余弦相似度跟它们的[线性相关系数（Pearson correlation coefficient）](<https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>)有关：
\begin{equation}\mathop{\text{pearson}}(\boldsymbol{x}, \boldsymbol{y}) = \cos(\boldsymbol{x}-\bar{\boldsymbol{x}}, \boldsymbol{y}-\bar{\boldsymbol{y}})\end{equation}
其中$\bar{\boldsymbol{x}},\bar{\boldsymbol{y}}$是$\boldsymbol{x}, \boldsymbol{y}$各分量的均值。在线性相关系数基础上，衍生出了[秩相关系数（Spearman's rank correlation coefficient）](<https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>)：
\begin{equation}\mathop{\text{spearman}}(\boldsymbol{x}, \boldsymbol{y}) = \mathop{\text{pearson}}(\boldsymbol{r}_x, \boldsymbol{r}_y)\end{equation}
$\boldsymbol{r}_x,\boldsymbol{r}_y$分别是$\boldsymbol{x}, \boldsymbol{y}$的次序向量，即如果$x_i$是$\boldsymbol{x}$从小到大第$k$个分量，那么$(\boldsymbol{r}_x)_i=k$。由定义可知，秩相关系数完全忽略了数据本身的大小，只关心数据的顺序，此前我们在[评测语义相似度](<https://spaces.ac.cn/archives/8321>)时，就用到了这个指标。有趣的是，秩相关系数也可以视为$\rcos$的一个特例
\begin{equation}\mathop{\text{spearman}}(\boldsymbol{x}, \boldsymbol{y}) = \rcos(\boldsymbol{r}_x, \boldsymbol{r}_y)\end{equation}
这个等式请读者自行验证，就不展开了。由此，$\rcos(\boldsymbol{x}, \boldsymbol{y})$可以看作是一个保留了原始数值信息的广义排序相似度。

## 一言以蔽之

本文介绍了从“相似度 = 归一化内积”的视角出发、基于内积的双边不等式来构建相似度指标的思路，着重讨论了基于排序不等式构建的排序相似度指标。
