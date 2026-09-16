# 中位数（Median）简介

> 作者：苏剑林 · 科学空间 · 2026-03-31
>
> 原文：<https://spaces.ac.cn/archives/11693>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近重新学习了一下中位数的概念，趁新鲜记录一下要点。

做异常值剔除或者裁剪时，我们经常需要一个“基准”，比如对于一堆非负数据，我们可能认为大于基准的50倍就是异常值。那这个基准如何选取呢？一个常用的指标是平均值，然而平均值容易被异常值“带偏”，因此以它为基准可能会偏向异常值，从而漏掉一些结果，这时我们可以考虑选取中位数为基准。

## 基本性质

对于一维数据点$x_1,x_2,\cdots,x_n$，它们的平均值（Mean）定义为
\begin{equation}\newcommand{mean}{\mathop{\text{mean}}}\mean(x_1,x_2,\cdots,x_n) = \frac{1}{n}\sum_{i=1}^n x_i\end{equation}
由于全体数据都直接参与平均计算，所以一旦有几个点特别大，那么平均值也会随之变大，从而干扰异常值的判断。

中位数（Median）的想法是找出数据中的“中立者”作为基准，具体来说，它希望找到一个分隔点，使得大于等于和小于等于该分隔点的数据量一致，所以它也叫做“50%分位数（Quantile\-50%）”。如果$n$是奇数，那么它等于这批数中第$(n+1)/2$大的数，如果$n$是偶数，那么第$n/2$大和第$n/2+1$大之间的任意数字，都可以视为中位数，一般情况下我们取它们俩的平均。

从中位数的定义就可以看出它的抗干扰能力：只要不改变每个数跟中位数的大小关系，那么中位数就不会改变。进一步地，我们可以引入“崩溃点（Breakdown Point）”来描述这种能力，它是“使得结果趋于无穷的最少异常值比例”。对于平均值来说答案是约等于0，因为只需1个点趋于无穷它就趋于无穷；但对于中位数来说是50%，因为要大于一半的数趋于无穷，中位数才趋于无穷。

中位数的缺点是计算起来比均值更为复杂，因为它至少需要对数据做某种程度上的排序，这尤其是对于分布式计算场景并不友好。庆幸的是，大多数场景也不需要非常精确的中位数，所以也有一些挪腾空间，比如局部算中位数后，再全局算一次平均值或者中位数，这样可以明显减少通信。

## 优化视角

有趣的是，我们可以在优化视角下，将平均值和中位数统一起来：
\begin{gather}\newcommand{argmin}{\mathop{\text{argmin}}}\newcommand{median}{\mathop{\text{median}}}
\mean(x_1,x_2,\cdots,x_n) = \argmin_{\mu} \sum_{i=1}^n (x_i - \mu)^2 \\
\median(x_1,x_2,\cdots,x_n) = \argmin_{\mu} \sum_{i=1}^n |x_i - \mu| \\
\end{gather}
$\mean$的证明比较简单，就留给读者了，这里简单介绍一下$\median$的证明。不失一般性，假设$x_i$已经排好序，即$x_1\leq x_2\leq\cdots\leq x_n$，设$f(\mu) = \sum_{i=1}^n |x_i - \mu|$，直接求导得
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}} f'(\mu) = \sum_{i=1}^n \sign(\mu - x_i) = \#\{x_i < \mu\} - \#\{x_i > \mu\}\\
\end{equation}
记号$\#$表示满足条件的数目。为了找最小值点，我们希望导数尽可能接近于0，即希望大于$\mu$和小于$\mu$的$x_i$数目尽可能相等，这已经符合中位数的思想。更具体的讨论要分情况：
\begin{equation}f'(\mu) = \left\{\begin{aligned}&\left.\begin{aligned}
&\underbrace{\#\{x_i < \mu\}}_{\leq k} - \underbrace{\#\{x_i > \mu\}}_{\geq k+1} < 0, &\mu < x_{k+1}  \\
&\underbrace{\#\{x_i < \mu\}}_{\geq k+1} - \underbrace{\#\{x_i > \mu\}}_{\leq k} > 0, &\mu > x_{k+1}  \\\end{aligned}\right\} & n = 2k+1 \\
&\left.\begin{aligned}
&\underbrace{\#\{x_i < \mu\}}_{\leq k-1} - \underbrace{\#\{x_i > \mu\}}_{\geq k+1} < 0, &\mu < x_{k\phantom{+1}}  \\
&\underbrace{\#\{x_i < \mu\}}_{\geq k+1} - \underbrace{\#\{x_i > \mu\}}_{\leq k-1} > 0, &\mu > x_{k+1}  \\
\end{aligned}\right\} & n = 2k
\end{aligned}\right.\end{equation}
即当$n$是奇数$2k+1$时，$\mu < x_{k+1}$和$\mu > x_{k+1}$都会让$f(\mu)$增大，所以$\mu^*=x_{k+1}$；当$n$是偶数$2k$时，$\mu < x_k$和$\mu > x_{k+1}$都会让$f(\mu)$增大，所以$\mu^*\in [x_k, x_{k+1}]$，可以检验当$\mu^*$位于这个区间时，$f(\mu^*)$始终不变，所以这个区间的数都是$\mu^*$。综上，$\mu^*$跟中位数的定义完全吻合。

## 高维空间

从优化视角出发，我们也能理解为什么中位数比均值更能抵御异常值的干扰：假如有一个特别大的$x_i$，那么均值带来的损失是$(x_i - \mu)^2$，而中位数则是$|x_i - \mu|$，一般情况下$(x_i - \mu)^2 \gg |x_i - \mu|$，所以均值会更偏向于异常值，这样才能尽可能降低损失。

此外，优化视角还有一个好处，就是它容易推广到高维空间。我们知道，平均值的概念很容易推广到高维，对于一批向量$\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n$，平均向量就是$\frac{1}{n}\sum_{i=1}^n \boldsymbol{x}_i$，然而中位数的概念却不易直接推广，因为它依赖于排序，但对于向量数据来说很难定义出一个良好的序。

然而，在优化视角下，这种推广是很自然的：
\begin{gather}\mean(\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n) = \argmin_{\boldsymbol{\mu}} \sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2^2 \\
\median(\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n) = \argmin_{\boldsymbol{\mu}} \sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2 \\
\end{gather}
其中$\Vert\Vert_2$是欧氏距离，容易证明这样定义出来的平均向量正好是$\frac{1}{n}\sum_{i=1}^n \boldsymbol{x}_i$，与经验定义相符。至于中位向量，我们也称“几何中位数（Geometric Median）”，从目标函数可以看出，如果$n=3$，那么其实就是经典的“[费马点](<https://en.wikipedia.org/wiki/Fermat_point>)”，因此很多时候我们也直接称中位向量为费马点。

很遗憾，中位向量没有解析解，我们通常是基于如下Weiszfeld迭代来计算：
\begin{equation}\boldsymbol{\mu}_{t+1} = \frac{\sum_{i=1}^n \boldsymbol{x}_i / \Vert\boldsymbol{x}_i - \boldsymbol{\mu}_t\Vert_2}{\sum_{i=1}^n 1 / \Vert\boldsymbol{x}_i - \boldsymbol{\mu}_t\Vert_2}\end{equation}

## 继续推广

很明显，我们还可以考虑更一般的推广
\begin{equation}\newcommand{average}{\mathop{\text{average}}}\average(\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n;\alpha) = \argmin_{\boldsymbol{\mu}} \sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2^{\alpha}\end{equation}
记$f(\boldsymbol{\mu}) = \sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2^{\alpha}$，那么
\begin{equation}\nabla_{\boldsymbol{\mu}} f(\boldsymbol{\mu}) = \alpha\sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2^{\alpha - 2}(\boldsymbol{\mu} - \boldsymbol{x}_i)\end{equation}
令$\nabla_{\boldsymbol{\mu}} f(\boldsymbol{\mu})=\boldsymbol{0}$，那么需要求解的方程可以写成
\begin{equation}\boldsymbol{\mu} = \frac{\sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2^{\alpha - 2}\boldsymbol{x}_i}{\sum_{i=1}^n\Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_2^{\alpha - 2}}\end{equation}
将$\boldsymbol{\mu}_t$代入右端记为$\boldsymbol{\mu}_{t+1}$，即得不动点迭代法。代入$\alpha=2$即为平均向量，代入$\alpha=1$即得Weiszfeld迭代。当然严格来讲还要证明收敛性和唯一性，个中细节比较复杂，这里就省略了。

此外，高维空间的中位向量还有一种比较少用的形式，就是将欧氏范数换成L1范数：
\begin{equation}\argmin_{\boldsymbol{\mu}} \sum_{i=1}^n \Vert\boldsymbol{x}_i - \boldsymbol{\mu}\Vert_1\end{equation}
这叫做“坐标中位数（Coordinate\-wise Median）”，因为它本质上就是逐分量取一维的中位数，计算起来比较简单，但由于没有明显的几何意义，使用场景相对少一些。

## 文章小结

本文简单总结了中位数的概念、性质以及它在高维空间中的推广。
