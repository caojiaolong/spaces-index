# 当BERT\-whitening引入超参数：总有一款适合你

> 作者：苏剑林 · 科学空间 · 2022-05-18
>
> 原文：<https://spaces.ac.cn/archives/9079>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《你可能不需要BERT\-flow：一个线性变换媲美BERT\-flow》](<https://spaces.ac.cn/archives/8069>)中，笔者提出了BERT\-whitening，验证了一个线性变换就能媲美当时的SOTA方法BERT\-flow。此外，BERT\-whitening还可以对句向量进行降维，带来更低的内存占用和更快的检索速度。然而，在[《无监督语义相似度哪家强？我们做了个比较全面的评测》](<https://spaces.ac.cn/archives/8321>)中我们也发现，whitening操作并非总能带来提升，有些模型本身就很贴合任务（如经过有监督训练的SimBERT），那么额外的whitening操作往往会降低效果。

为了弥补这个不足，本文提出往BERT\-whitening中引入了两个超参数，通过调节这两个超参数，我们几乎可以总是获得“降维不掉点”的结果。换句话说，即便是原来加上whitening后效果会下降的任务，如今也有机会在降维的同时获得相近甚至更好的效果了。

## 方法概要

目前BERT\-whitening的流程是：
\begin{equation}\begin{aligned}
\tilde{\boldsymbol{x}}_i =&\, (\boldsymbol{x}_i - \boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-1/2} \\
\boldsymbol{\mu} =&\, \frac{1}{N}\sum\limits_{i=1}^N \boldsymbol{x}_i \\
\boldsymbol{\Sigma} =&\, \frac{1}{N}\sum\limits_{i=1}^N (\boldsymbol{x}_i - \boldsymbol{\mu})^{\top}(\boldsymbol{x}_i - \boldsymbol{\mu}) = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top} \,\,(\text{SVD分解})
\end{aligned}\end{equation}
其中$\boldsymbol{x}_i$是给定的句向量（如无说明，向量默认为行向量），$\tilde{\boldsymbol{x}}_i$是变换后的向量，SVD分解的结果中，$\boldsymbol{U}$是正交矩阵，$\boldsymbol{\Lambda}$是对角矩阵，并且对角线的元素非负且从大到小排列。可以看到，目前的流程是完全固定的，即没有任何可调的超参数。

为了增加一定的调节空间，我们可以往里边引入两个超参数$\beta,\gamma$（标量），使其变为
\begin{equation}\begin{aligned}
\tilde{\boldsymbol{x}}_i =&\, (\boldsymbol{x}_i - {\color{red}\beta}\boldsymbol{\mu})\boldsymbol{U}\boldsymbol{\Lambda}^{-{\color{red}\gamma}/2} \\
\boldsymbol{\mu} =&\, \frac{1}{N}\sum\limits_{i=1}^N \boldsymbol{x}_i \\
\boldsymbol{\Sigma} =&\, \frac{1}{N}\sum\limits_{i=1}^N (\boldsymbol{x}_i - {\color{red}\beta}\boldsymbol{\mu})^{\top}(\boldsymbol{x}_i - {\color{red}\beta}\boldsymbol{\mu}) = \boldsymbol{U}\boldsymbol{\Lambda}\boldsymbol{U}^{\top} \,\,(\text{SVD分解})
\end{aligned}\end{equation}

## 思路分析

可以看到，当$\beta=\gamma=1$时，就是原来的BERT\-whitening；而当$\beta=\gamma=0$时，净变换就是
\begin{equation}\tilde{\boldsymbol{x}}_i =\boldsymbol{x}_i \boldsymbol{U}\end{equation}
由于$\boldsymbol{U}$是正交矩阵，所以不改变内积结果，即$\tilde{\boldsymbol{x}}_i\tilde{\boldsymbol{x}}_i^{\top} = \boldsymbol{x}_i \boldsymbol{U} (\boldsymbol{x}_i \boldsymbol{U})^{\top} = \boldsymbol{x}_i\boldsymbol{x}_i^{\top}$，所以当我们用余弦相似度作为相似度量时，它不会改变原有结果。换句话说，引入这组超参数后，它提供了“不逊色于变换前的效果”的可能性，那么当我们精调这组参数时，就有可能取得比变换前更好的效果。这也是这两个超参数的设计思路。

此外，在这样的改动之下，原来的降维能力还是得以保留的。我们可以将变换拆开为两部分看：
\begin{equation}\tilde{\boldsymbol{x}}_i = \color{red}{\underbrace{(\boldsymbol{x}_i - \beta\boldsymbol{\mu})\boldsymbol{U}}_{\text{part 1}}}\color{skyblue}{\underbrace{\boldsymbol{\Lambda}^{-\gamma/2}}_{\text{part 2}}}\end{equation}
第一部分主要是正交变换$\boldsymbol{U}$，$\boldsymbol{U}$是$\boldsymbol{\Sigma}$矩阵SVD分解之后的结果，它能将向量$\boldsymbol{x}_i - \beta\boldsymbol{\mu}$变换成每个分量尽量独立的新向量，并且新向量的每个分量与0的平均波动正好是由$\boldsymbol{\Lambda}^{1/2}$的对角线元素来衡量，如果对应的波动很接近于0，那么我们就可以认为它实际就是0，舍去这个分量也不会影响余弦值的计算结果，这就是降维的原理。而由于SVD分解的结果已经提前将$\boldsymbol{\Lambda}$从大到小排好了顺序，因此我们可以直接通过保留前$k$维的操作$\tilde{\boldsymbol{x}}_i\text{[:}k\text{]}$就可以实现降到$k$维了。

至于第二部分$\boldsymbol{\Lambda}^{-\gamma/2}$，我们可以理解为当前任务对各向同性的依赖程度，如果$\gamma=1$，那么相当于每个分量都是各平权的，这可以作为一个无监督的先验结果，但未必对所有任务都是最优的，所以我们可以通过调节$\gamma$来更好地适应当前任务。

## 实验结果

文章[《无监督语义相似度哪家强？我们做了个比较全面的评测》](<https://spaces.ac.cn/archives/8321>)已经显示，在ATEC、BQ、LCQMC三个任务上，SimBERT加上默认的whitening操作（即$\beta=\gamma=1$）都会导致效果下降，而如果我们取$\beta=\gamma=0$，那么结果就不一样了（随便演示了两个组合，其他组合结果相似）：
$$\small{\begin{array}{c}
\text{BERT-P4效果表} \\
{\begin{array}{l|ccccc}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{STS-B} \\
\hline
\beta=\gamma=1 & 24.51 / \color{green}{27.00} / \color{green}{27.91}  & 38.81 / \color{red}{32.29} / \color{red}{37.67}  & 64.75 / \color{green}{64.75} / \color{green}{65.65}  & 15.12 / \color{green}{17.80} / \color{green}{15.34}  & 61.66 / \color{green}{69.45} / \color{green}{69.37}
\\
\beta=\gamma=0 & 24.51 / 24.51 / \color{green}{24.59}  & 38.81 / 38.81 / \color{green}{38.99}  & 64.75 / 64.75 / \color{red}{63.45}  & 15.12 / 15.12 / \color{red}{14.59}  & 61.66 / 61.66 / \color{green}{62.30}  \\
\hline
\end{array}} \\
\\
\text{SimBERT-P1效果表} \\
{\begin{array}{l|ccccc}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{STS-B} \\
\hline
\beta=\gamma=1 & 38.50 / \color{red}{23.64} / \color{red}{30.79}  & 48.54 / \color{red}{31.78} / \color{red}{40.01}  & 76.23 / \color{red}{75.05} / \color{red}{74.50}  & 15.10 / \color{green}{18.49} / \color{green}{15.64}  & 74.14 / \color{red}{73.37} / \color{green}{75.29}  \\
\beta=\gamma=0 & 38.50 / 38.50 / \color{green}{38.81}  & 48.54 / 48.54 / \color{green}{48.66}  & 76.23 / 76.23 / \color{red}{76.22}  & 15.10 / 15.10 / \color{red}{14.88}  & 74.14 / 74.14 / \color{green}{74.46}  \\
\hline
\end{array}}
\end{array}}$$

跟之前的文章一样，表格中的每个元素是$a / b / c$的形式，代表该任务在该模型下“不加whitening”的得分为$a$、“加whitening”的得分为$b$、“加whitening并降到256维”的得分为$c$；如果$b > a$，那么$b$显示为绿色，小于则显示为红色；如果$c > a$，那么$c$显示为绿色，小于则显示为红色。前面说了，如果不降维的话，$\beta=\gamma=0$的净变换就是$\boldsymbol{U}$，不改变余弦值结果，因此$\beta=\gamma=0$时的$a,b$都是相等的。

在这个表格中，我们主要看$a/b/c$中的第三个结果$c$，它是将向量从768维降低到256维的结果，可以看到当$\beta=\gamma=0$时，不管是无监督的BERT还是有监督的SimBERT，该结果基本都很接近原始向量的结果（即$a$），部分结果甚至还有提升。这就意味着，$\beta=\gamma=0,k=256$这个组合几乎可以算是“免费的午餐”，几乎无损效果，并且实现了降维。

笔者也试过精调$\beta,\gamma$，在一些任务上确实能取得比上述两个组合更好的效果，但精调需要标签数据，争议性可能会比较大，这里就不演示了。如果原来的句向量模型本就是有监督训练得到的，用BERT\-whitening仅仅是奔着降维去的，那么就可以用验证集来精调一下$\beta,\gamma$和$k$了，这种场景下就是无争议的了。

## 文章小结

本文通过引入两个超参数的方式来赋予BERT\-whitening一定的调参空间，使其具备“不逊色于变换前的效果”的可能性，并且保留了降维的能力。换言之，即便是之前已经训练好的句向量模型，我们也可以用新的BERT\-whitening将它降维，并且保持效果基本不变，有时候甚至还更优～
