# 概率视角下的线性模型：逻辑回归有解析解吗？

> 作者：苏剑林 · 科学空间 · 2021-07-22
>
> 原文：<https://spaces.ac.cn/archives/8578>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们知道，线性回归是比较简单的问题，它存在解析解，而它的变体逻辑回归（Logistic Regression）却没有解析解，这不能不说是一个遗憾。因为逻辑回归虽然也叫“回归”，但它实际上是用于分类问题的，而对于很多读者来说分类比回归更加常见。准确来说，我们说逻辑回归没有解析解，说的是“最大似然估计下逻辑回归没有解析解”。那么，这是否意味着，如果我们不用最大似然估计，是否能找到一个可用的解析解呢？

[![逻辑回归示意图](<https://spaces.ac.cn/usr/uploads/2021/07/1189736421.png>)](<https://spaces.ac.cn/usr/uploads/2021/07/1189736421.png>)

逻辑回归示意图

本文将会从非最大似然的角度，推导逻辑回归的一个解析解，简单的实验表明它效果不逊色于梯度下降求出来的最大似然解。此外，这个解析解还易于推广到单层Softmax多分类模型。

## 线性回归

我们先来回顾一下线性回归。假设训练数据为$\{(\boldsymbol{x}_i,\boldsymbol{y}_i)\}_{i=1}^N$，其中$\boldsymbol{x}\in\mathbb{R}^n,\boldsymbol{y}\in\mathbb{R}^m$，为了便于跟代码实现对齐，这里默认情况下向量都是<u>行向量</u>。线性回归假设$\boldsymbol{x},\boldsymbol{y}$满足线性关系$\boldsymbol{y}=\boldsymbol{x}\boldsymbol{W}+\boldsymbol{b}$，这里$\boldsymbol{W}\in\mathbb{R}^{n\times m},\boldsymbol{b}\in\mathbb{R}^m$，然后通过最小化下述均方误差来估计它们
\begin{equation}\frac{1}{N}\sum_{i=1}^N \Vert\boldsymbol{y}_i-\boldsymbol{x}_i\boldsymbol{W}-\boldsymbol{b}\Vert^2\label{eq:loss}\end{equation}
这个目标直接展开求导就行了，只是一个二次函数的最小值问题，所以有解析解。

### 概率视角

从概率分布的角度来看，均方误差意味着我们假设了$p(\boldsymbol{y}|\boldsymbol{x})$是均值为$\boldsymbol{\mu}_{y|x}=\boldsymbol{x}\boldsymbol{W}+\boldsymbol{b}$的正态分布。现在，我们来做一个更强的假设：

> 假设联合分布$p(\boldsymbol{x},\boldsymbol{y})$是正态分布。

在这个假设之下，我们可以直接写出对应的条件分布：
\begin{equation}\begin{aligned}
p(\boldsymbol{y}|\boldsymbol{x}) =&\, \mathcal{N}(\boldsymbol{y};\boldsymbol{\mu}_{y|x},\boldsymbol{\Sigma}_{y|x})\\
\boldsymbol{\mu}_{y|x} =&\, \boldsymbol{\mu}_y + (\boldsymbol{x}-\boldsymbol{\mu}_x)\boldsymbol{\Sigma}_{xx}^{-1}\boldsymbol{\Sigma}_{xy}\\
\boldsymbol{\Sigma}_{y|x} =&\, \boldsymbol{\Sigma}_{yy} - \boldsymbol{\Sigma}_{yx}\boldsymbol{\Sigma}_{xx}^{-1}\boldsymbol{\Sigma}_{xy}
\end{aligned}\end{equation}
这里的$\boldsymbol{\mu}_x,\boldsymbol{\mu}_y$是$\boldsymbol{x},\boldsymbol{y}$的均值向量，$\begin{pmatrix}\boldsymbol{\Sigma}_{xx} & \boldsymbol{\Sigma}_{xy} \\ \boldsymbol{\Sigma}_{yx} & \boldsymbol{\Sigma}_{yy}\end{pmatrix}$是$\boldsymbol{x},\boldsymbol{y}$的协方差矩阵。正态分布的条件分布形式可以直接在[维基百科](<https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions>)找到，其证明可以参考[StackExchange](<https://stats.stackexchange.com/questions/30588/deriving-the-conditional-distributions-of-a-multivariate-normal-distribution>)或相关概率统计书籍。

现在对照$\boldsymbol{\mu}_{y|x}=\boldsymbol{x}\boldsymbol{W}+\boldsymbol{b}$，我们可以得到
\begin{equation}\boldsymbol{W} = \boldsymbol{\Sigma}_{xx}^{-1}\boldsymbol{\Sigma}_{xy}, \quad \boldsymbol{b} = \boldsymbol{\mu}_y - \boldsymbol{\mu}_x\boldsymbol{\Sigma}_{xx}^{-1}\boldsymbol{\Sigma}_{xy}\end{equation}
这其实就是线性回归的解析解。

### 思考分析

让我们捋一捋上述过程。首先，默认情况下，线性回归只是做了条件分布$p(\boldsymbol{y}|\boldsymbol{x})$的假设，通过最小二乘法我们就可以获得线性回归的解析解，这是常规的线性回归的介绍过程；然后，在上面中，我们做了一个更加强的假设“联合分布$p(\boldsymbol{x},\boldsymbol{y})$是正态分布”，但是最终依然获得了一样的解析解。

为什么更加强的假设却获得了一样的结果呢？事实上，从损失函数$\eqref{eq:loss}$就可以看到，它关于$\boldsymbol{y}$是二次的，关于$\boldsymbol{x}$同样是二次的，这意味着它顶多用到关于$\boldsymbol{x},\boldsymbol{y}$的二阶矩信息，因此作出联合分布是正态分布的假设，并不会改变最终结果，因为正态分布已经保留了所有不超过二阶的矩信息（均值和协方差）。

进一步地，我们可以想象到，任何线性模型（线性回归、逻辑回归、单层神经网络等）主要用到的数据统计量，应该也都是不超过二阶的矩。因此，在处理线性模型时，我们可以根据具体情况适当地作出正态分布的假设，理论上来说，这可以获得一个等价结果，或者一个足够好的近似结果。

## 逻辑回归

利用上述思路，我们就可以给出逻辑回归的一个解析解，该结果笔者首先在[《Easy Logistic Regression with an Analytical Solution》](<https://towardsdatascience.com/easy-logistic-regression-with-an-analytical-solution-eb4589c2cd2d>)看到，看后颇有启发，特与大家分享。

假设训练数据为$\{(\boldsymbol{x}_i,y_i)\}_{i=1}^N$，其中$\boldsymbol{x}\in\mathbb{R}^n,y\in\{0,1\}$，这意味着它是一个二分类数据集，建立概率模型
\begin{equation}p(y|\boldsymbol{x}) = \left\{\begin{aligned}\sigma\left(\boldsymbol{x}\boldsymbol{w}^{\top}+b\right),\quad (y = 1)\\ 1 - \sigma\left(\boldsymbol{x}\boldsymbol{w}^{\top}+b\right),\quad (y = 0)\end{aligned}\right.\end{equation}
这里$\sigma(t)=1/(1+e^{-t})$。$\boldsymbol{w},b$的常规估计方式是最大似然，也就是最小化下述loss：
\begin{equation}-\frac{1}{N}\sum_{i=1}^N \ln p(y_i|\boldsymbol{x}_i)\end{equation}
我们没法计算出它的解析解。然而，如果不走最大似然这条路，另外设计求解路线，是有可能得到解析解的。

### 别出心裁

首先，不难验证对于逻辑回归模型，我们有：
\begin{equation}\frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})} = \exp\left(\boldsymbol{x}\boldsymbol{w}^{\top}+b\right) \quad\Leftrightarrow\quad \ln \frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})} = \boldsymbol{x}\boldsymbol{w}^{\top}+b\label{eq:log}\end{equation}
这也就是说，逻辑回归相当于以$\ln \frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})}$为输出的线性回归模型。然而$\ln \frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})}$的直接估计并不容易，我们利用贝叶斯公式：
\begin{equation}p(y|\boldsymbol{x}) = \frac{p(\boldsymbol{x}|y)p(y)}{p(\boldsymbol{x})} \quad\Leftrightarrow\quad \frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})} = \frac{p(\boldsymbol{x}|1)p_1}{p(\boldsymbol{x}|0)p_0}\label{eq:bys}\end{equation}
这里的$p_1,p_0$分别是正负两个类别的概率，这个容易估计。$p(\boldsymbol{x}|1),p(\boldsymbol{x}|0)$自然就是正负样本所满足的分布了，现在我们对它们作正态分布假设：

> 假设$p(\boldsymbol{x}|1),p(\boldsymbol{x}|0)$是具有同样协方差矩阵的正态分布。

这里“同样协方差矩阵”读者可能会觉得莫名其妙，这一点我们后面再讨论。在这个假设之下，记
\begin{equation}p(\boldsymbol{x}|1) = \mathcal{N}(\boldsymbol{x};\boldsymbol{\mu}_1,\boldsymbol{\Sigma}),\quad p(\boldsymbol{x}|0) = \mathcal{N}(\boldsymbol{x};\boldsymbol{\mu}_0,\boldsymbol{\Sigma})\end{equation}
其中$\boldsymbol{\mu}_1,\boldsymbol{\mu}_0$分别就是正负样本的均值向量，$\boldsymbol{\Sigma}$可以用全体样本的协方差矩阵来估计。回顾正态分布的概率密度表达式：
\begin{equation}\frac{1}{\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})}}\exp\left\{-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})^{\top}\right\}\end{equation}
代入式$\eqref{eq:bys}$后展开化简，我们发现<u>二次项刚好抵消</u>，于是
\begin{equation}\ln\frac{p(1|\boldsymbol{x})}{p(0|\boldsymbol{x})} = \ln\frac{p(\boldsymbol{x}|1)p_1}{p(\boldsymbol{x}|0)p_0} = \boldsymbol{x}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^{\top} + \frac{1}{2}\left(\boldsymbol{\mu}_0\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0^{\top} - \boldsymbol{\mu}_1\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1^{\top}\right) + \ln\frac{p_1}{p_0}\label{eq:rate}\end{equation}
对比式$\eqref{eq:log}$后我们得到：
\begin{equation}\begin{aligned}
\boldsymbol{w} =&\, (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)\boldsymbol{\Sigma}^{-1}\\
b =&\, \frac{1}{2}\left(\boldsymbol{\mu}_0\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_0^{\top} - \boldsymbol{\mu}_1\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_1^{\top}\right) + \ln\frac{p_1}{p_0}
\end{aligned}\label{eq:sol}\end{equation}
这就是逻辑回归的一个解析解。当然，这也不算特别新的内容，它的思路跟“[线性判别分析（Linear Discriminant Analysis）](<https://en.wikipedia.org/wiki/Linear_discriminant_analysis>)”其实是很一致的。

### 思考分析

目前，对于这个解，读者最有疑虑的可能是“同样协方差矩阵”这个假设是否过强了。从技巧上来看，这个假设是为了让$\ln\frac{p(\boldsymbol{x}|1)}{p(\boldsymbol{x}|0)}$的二次项正好抵消，从而只剩下一次项而直接得到逻辑回归的解析解；那么从理论上来看，这个假设有没有什么必然性呢？事实上，我们可以认为，逻辑回归本身就（近似地）隐含了“同样协方差矩阵”这一假设。

怎么理解这句话呢？首先，对于逻辑回归模型来说，式$\eqref{eq:log}$是自然成立的，而贝叶斯公式也是恒成立的，所以结论就是$\ln\frac{p(\boldsymbol{x}|1)}{p(\boldsymbol{x}|0)}$必然只有一次项和常数项。而线性回归的例子已经告诉我们，对线性模型的数据分布做正态假设一般是不损失什么信息的，所以我们假设$p(\boldsymbol{x}|1),p(\boldsymbol{x}|0)$为正态分布（一定程度上）也是合理的。而假设它们为正态分布后，如果要使得结果没有二次项，那么协方差矩阵必然要一致。

也就是说，当你决定<u>使用逻辑回归模型</u>和<u>接受正态假设</u>的那一刻起，就做出了“正负样本数据具有同样协方差矩阵”这个假设了～

### 多分类器

上述关于逻辑回归的解析解，还可以很方便地推广到“全连接\+Softmax”的多分类场景中，该场景假设类别$i$的概率为
\begin{equation}p(i|\boldsymbol{x}) = \frac{\exp\left(\boldsymbol{x}\boldsymbol{w}_i^{\top}+b_i\right)}{\sum\limits_{i=1}^k \exp\left(\boldsymbol{x}\boldsymbol{w}_i^{\top}+b_i\right)}\end{equation}
基于同样的推理和假设，我们可以得到类似式$\eqref{eq:log}$的结果：
\begin{equation}\ln \frac{p(j|\boldsymbol{x})}{p(i|\boldsymbol{x})} = \boldsymbol{x}(\boldsymbol{w}_j - \boldsymbol{w}_i)^{\top}+(b_j - b_i)\end{equation}
以及类似式$\eqref{eq:rate}$的结果：
\begin{equation}\ln\frac{p(j|\boldsymbol{x})}{p(i|\boldsymbol{x})} = \boldsymbol{x}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_j - \boldsymbol{\mu}_i)^{\top} + \frac{1}{2}\left(\boldsymbol{\mu}_i\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i^{\top} - \boldsymbol{\mu}_j\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_j^{\top}\right) + \ln\frac{p_j}{p_i}\end{equation}
对比之下，我们可以发现一组解是：
\begin{equation}\begin{aligned}
\boldsymbol{w}_i =&\, \boldsymbol{\mu}_i\boldsymbol{\Sigma}^{-1}\\
b_i =&\, \ln p_i - \frac{1}{2}\boldsymbol{\mu}_i\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i^{\top}
\end{aligned}\end{equation}

### 参数估计

理论部分的最后，我们来讨论怎么估计模型的参数。可以看到$\boldsymbol{w}_i,b_i$是$p_i$、$\boldsymbol{\mu}_i$和$\boldsymbol{\Sigma}$的函数，所以本质上是这三者的估计。

前面已经提到，$p_i$比较简单，直接用每个类的频率代替就行；$\boldsymbol{\mu}_i$也不难，就是每个类的均值向量。所以难度在于估计$\boldsymbol{\Sigma}$。根据我们前面的假设，全体数据的分布为
\begin{equation}\tilde{p}(\boldsymbol{x}) = \sum_{i=1}^k p_i \mathcal{N}(\boldsymbol{x};\boldsymbol{\mu}_i,\boldsymbol{\Sigma})\end{equation}
两端乘以$\boldsymbol{x}^{\top}\boldsymbol{x}$后积分，得到
\begin{equation}\tilde{\boldsymbol{\Sigma}}+\tilde{\boldsymbol{\mu}}^{\top} \tilde{\boldsymbol{\mu}} = \sum_{i=1}^k p_i \left(\boldsymbol{\Sigma}+\boldsymbol{\mu}_i^{\top} \boldsymbol{\mu}_i\right) = \boldsymbol{\Sigma} + \sum_{i=1}^k p_i\, \boldsymbol{\mu}_i^{\top} \boldsymbol{\mu}_i\end{equation}
其中$\tilde{\boldsymbol{\mu}},\tilde{\boldsymbol{\Sigma}}$是全体数据的均值向量和协方差矩阵。因此我们有估计
\begin{equation}\boldsymbol{\Sigma} = \tilde{\boldsymbol{\Sigma}}+\tilde{\boldsymbol{\mu}}^{\top} \tilde{\boldsymbol{\mu}} - \sum_{i=1}^k p_i\, \boldsymbol{\mu}_i^{\top} \boldsymbol{\mu}_i\end{equation}

特别地，建议在估计之前，我们先对原始数据做一个白化（参考[《你可能不需要BERT\-flow：一个线性变换媲美BERT\-flow》](<https://spaces.ac.cn/archives/8069#%E6%A0%87%E5%87%86%E5%8C%96%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5>)），使全体数据的均值为0、协方差为单位阵，然后再进行估计。此时
\begin{equation}\boldsymbol{\Sigma} = \boldsymbol{I} - \sum_{i=1}^k p_i\, \boldsymbol{\mu}_i^{\top} \boldsymbol{\mu}_i\end{equation}
理论上来说，先白化再用估计，跟直接估计的结果是完全一样的。然而，<u>对于高维数据来说，白化后在数值计算上会稳定很多</u>，所以实际使用时，都推荐先白化后再估计的方式。

## 实验评估

那么，上面推导的解析解可用性如何呢？能不能比得上梯度下降求出来的解呢？这里做了几个文本方面的实验，都是用[RoFormer\-Sim\-FT](<https://spaces.ac.cn/archives/8541>)来抽取固定的句向量特征，然后后面接一个全连接层分类，比较用梯度下降求出来的解和上述解析解的效果差异。实验代码开源如下：

> <strong>Github：[https://github\.com/bojone/analytical\-classification](<https://github.com/bojone/analytical-classification>)</strong>

### 全量样本

评测包括四个分类任务：情感分类（[SENTIMENT](<https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip>)）、长文本分类（[IFLYTEK](<https://www.cluebenchmarks.com/introduce.html>)）、短新闻分类（[TNEWS](<https://www.cluebenchmarks.com/introduce.html>)）、电商主题分类（[SHOPPING](<https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/online_shopping_10_cats/intro.ipynb>)），大致情况如下：
\begin{array}{c|cccc}
\hline
& \text{总类别数} & \text{训练样本数} & \text{验证样本数} & \text{测试样本数} \\
\hline
\text{SENTIMENT} & 2 & 16883 & 2111 & 2111 \\
\text{IFLYTEK} & 119 & 12133 & 2599 & \text{-}\\
\text{TNEWS} & 15 & 53360 & 10000 & \text{-}\\
\text{SHOPPING} & 10 & 47079 & 15694 & \text{-}\\
\hline
\end{array}

实验的评测指标都是准确率，全量训练样本下效果如下：
\begin{array}{c|ccc}
\hline
& \text{训练准确率} & \text{验证准确率} & \text{测试准确率} \\
\hline
\text{SENTIMENT-梯度下降} & 92.26\% & 91.14\% & 91.14\% \\
\text{SENTIMENT-解析解} & 91.79\% & 90.81\% & 91.57\% \\
\hline
\text{IFLYTEK-梯度下降} & 93.43\% & 51.14\% & \text{-} \\
\text{IFLYTEK-解析解} & 71.70\% & 56.44\% & \text{-} \\
\hline
\text{TNEWS-梯度下降} & 59.62\% & 53.35\% & \text{-} \\
\text{TNEWS-解析解} & 56.12\% & 54.20\% & \text{-} \\
\hline
\text{SHOPPING-梯度下降} & 91.63\% & 86.98\% & \text{-} \\
\text{SHOPPING-解析解} & 87.89\% & 86.38\% & \text{-} \\
\hline
\end{array}

### 小量样本

从上述表格可以看出，就训练集的效果而言，解析解通常是不如梯度下降的，但是它在验证集和测试集的效果都接近甚至超越梯度下降的表现，总的来说，解析解的训练集和验证集效果差距更小，这意味着解析解可能是一个泛化能力比较好的解，它可能更适合于数据量比较小、训练集和验证集分布不一致等场景。

为了验证这个猜测，我们将每个数据集的训练集都只保留1000条数据，然后继续进行实验：
\begin{array}{c|ccc}
\hline
& \text{训练准确率} & \text{验证准确率} & \text{测试准确率} \\
\hline
\text{SENTIMENT-1K-梯度下降} & 99.90\% & 66.08\% & 66.79\% \\
\text{SENTIMENT-1K-解析解} & 100.00\% & 72.67\% & 73.24\% \\
\hline
\text{IFLYTEK-1K-梯度下降} & 99.47\% & 15.43\% & \text{-} \\
\text{IFLYTEK-1K-解析解} & 99.47\% & 15.70\% & \text{-} \\
\hline
\text{TNEWS-1K-梯度下降} & 100.00\% & 22.47\% & \text{-} \\
\text{TNEWS-1K-解析解} & 100.00\% & 26.74\% & \text{-} \\
\hline
\text{SHOPPING-1K-梯度下降} & 100.00\% & 49.82\% & \text{-} \\
\text{SHOPPING-1K-解析解} & 100.00\% & 65.49\% & \text{-} \\
\hline
\end{array}

可以看到，训练数据变少的情况下，训练集的差距也变小了，但是解析解在验证集上的效果全面超过了梯度下降，这进一步显示出解析解在小样本场景下良好的泛化性能。

### 综合评述

其实上述结论也不难理解。大家都是线性模型的情况下，解析解相比于梯度下降多了一条“每个类的样本服从具有同样协方差矩阵的正态分布”的假设。当数据很多的情况下，我们对每个类的分布估计越发准确，此时该假设偏离程度越严重，从而没有自适应训练的梯度下降好；相反，当数据很少的情况下，每个类的分布本身就难以估计，此时该假设反而是一个有用的先验信息，有助于模型“由点到面”地泛化，而梯度下降反而由于缺乏先验而泛化能力不足。

换句话说，数据少的情况下，梯度下降背完几个样本就完事了，没有“触类旁通”，而解析解相当于通过额外的假设“造”了更多的样本出来让模型背，从而学到的东西更多了；数据多的情况下，梯度下降背得多，从而“熟能生巧”，而解析解还在按照自己的假设来“造”样本，这时候造出来的样本还不如真实的样本，从而效果可能有所下降。

那么，解析解有什么提升空间吗？比较直接的思路是，我们想办法对$\ln \frac{p(j|\boldsymbol{x})}{p(i|\boldsymbol{x})}$做更精细的估计，然后转化为线性回归问题来估计参数。至于怎么更好地估计$\ln \frac{p(j|\boldsymbol{x})}{p(i|\boldsymbol{x})}$，思路也不少，比如设为协方差不一致的正态分布或者干脆用核密度估计等，这就留给大家自由发挥了。

## 本文小结

本文介绍了逻辑回归的一个解析解，并且将它推广到了单层Softmax分类的场景。实验显示该解析解相比梯度下降有更好的泛化能力，尤其是在小样本场景通常还有更好的效果。
