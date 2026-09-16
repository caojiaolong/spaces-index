# Efficient GlobalPointer：少点参数，多点效果

> 作者：苏剑林 · 科学空间 · 2022-01-25
>
> 原文：<https://spaces.ac.cn/archives/8877>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《GlobalPointer：用统一的方式处理嵌套和非嵌套NER》](<https://spaces.ac.cn/archives/8373>)中，我们提出了名为“GlobalPointer”的token\-pair识别模块，当它用于NER时，能统一处理嵌套和非嵌套任务，并在非嵌套场景有着比CRF更快的速度和不逊色于CRF的效果。换言之，就目前的实验结果来看，至少在NER场景，我们可以放心地将CRF替换为GlobalPointer，而不用担心效果和速度上的损失。

在这篇文章中，我们提出GlobalPointer的一个改进版——Efficient GlobalPointer，它主要针对原GlobalPointer参数利用率不高的问题进行改进，明显降低了GlobalPointer的参数量。更有趣的是，多个任务的实验结果显示，参数量更少的Efficient GlobalPointer反而还取得更好的效果。

## 大量的参数

这里简单回顾一下GlobalPointer，详细介绍则请读者阅读[《GlobalPointer：用统一的方式处理嵌套和非嵌套NER》](<https://spaces.ac.cn/archives/8373>)。简单来说，GlobalPointer是基于内积的token\-pair识别模块，它可以用于NER场景，因为对于NER来说我们只需要把每一类实体的“(首, 尾)”这样的token\-pair识别出来就行了。

设长度为$n$的输入$t$经过编码后得到向量序列$[\boldsymbol{h}_1,\boldsymbol{h}_2,\cdots,\boldsymbol{h}_n]$，原始GlobalPointer通过变换$\boldsymbol{q}_{i,\alpha}=\boldsymbol{W}_{q,\alpha}\boldsymbol{h}_i$和$\boldsymbol{k}_{i,\alpha}=\boldsymbol{W}_{k,\alpha}\boldsymbol{h}_i$我们得到序列向量序列$[\boldsymbol{q}_{1,\alpha},\boldsymbol{q}_{2,\alpha},\cdots,\boldsymbol{q}_{n,\alpha}]$和$[\boldsymbol{k}_{1,\alpha},\boldsymbol{k}_{2,\alpha},\cdots,\boldsymbol{k}_{n,\alpha}]$，然后定义
\begin{equation}s_{\alpha}(i,j) = \boldsymbol{q}_{i,\alpha}^{\top}\boldsymbol{k}_{j,\alpha}\end{equation}
作为从$i$到$j$的连续片段是一个类型为$\alpha$的实体的打分。这里我们暂时省略了偏置项，如果觉得有必要，自行加上就好。

这样一来，有多少种类型的实体，就有多少个$\boldsymbol{W}_{q,\alpha}$和$\boldsymbol{W}_{k,\alpha}$。不妨设$\boldsymbol{W}_{q,\alpha},\boldsymbol{W}_{k,\alpha}\in\mathbb{R}^{d\times D}$，那么每新增一种实体类型，我们就要新增$2Dd$个参数；而如果用CRF\+BIO标注的话，每新增一种实体类型，我们只需要增加$2D$的参数（转移矩阵参数较少，忽略不计）。对于BERT base来说，常见的选择是$D=768,d=64$，可见GlobalPointer的参数量远远大于CRF。

## 识别与分类

事实上，不难想象对于任意类型$\alpha$，其打分矩阵$s_{\alpha}(i,j)$必然有很多相似之处，因为对于大多数token\-pair而言，它们代表的都是“非实体”，这些非实体的正确打分都是负的。这也就意味着，我们没必要为每种实体类型都设计独立的$s_{\alpha}(i,j)$，它们应当包含更多的共性。

怎么突出$s_{\alpha}(i,j)$的共性呢？以NER为例，我们知道NER实际上可以分解为“抽取”和“分类”两个步骤，“抽取”就是抽取出为实体的片段，“分类”则是确定每个实体的类型。这样一来，“抽取”这一步相当于只有一种实体类型的NER，我们可以用一个打分矩阵就可以完成，即$(\boldsymbol{W}_q\boldsymbol{h}_i)^{\top}(\boldsymbol{W}_k\boldsymbol{h}_j)$，而“分类”这一步，我们则可以用“特征拼接\+Dense层”来完成，即$\boldsymbol{w}_{\alpha}^{\top}[\boldsymbol{h}_i;\boldsymbol{h}_j]$。于是我们可以将两项组合起来，作为新的打分函数：
\begin{equation}s_{\alpha}(i,j) = (\boldsymbol{W}_q\boldsymbol{h}_i)^{\top}(\boldsymbol{W}_k\boldsymbol{h}_j) + \boldsymbol{w}_{\alpha}^{\top}[\boldsymbol{h}_i;\boldsymbol{h}_j]\label{eq:EGP-1}\end{equation}
这样一来，“抽取”这部分的参数对所有实体类型都是共享的，因此每新增一种实体类型，我们只需要新增对应的$\boldsymbol{w}_{\alpha}\in\mathbb{R}^{2D}$就行了，即新增一种实体类型增加的参数量也只是$2D$。进一步地，我们记$\boldsymbol{q}_i=\boldsymbol{W}_q\boldsymbol{h}_i, \boldsymbol{k}_i=\boldsymbol{W}_k\boldsymbol{h}_i$，然后为了进一步地减少参数量，我们可以用$[\boldsymbol{q}_i;\boldsymbol{k}_i]$来代替$\boldsymbol{h}_i$，此时
\begin{equation}s_{\alpha}(i,j) = \boldsymbol{q}_i^{\top}\boldsymbol{k}_j + \boldsymbol{w}_{\alpha}^{\top}[\boldsymbol{q}_i;\boldsymbol{k}_i;\boldsymbol{q}_j;\boldsymbol{k}_j]\label{eq:EGP}\end{equation}
此时$\boldsymbol{w}_{\alpha}\in\mathbb{R}^{4d}$，因此每新增一种实体类型所增加的参数量为$4d$，由于通常$d \ll D$，所以式$\eqref{eq:EGP}$的参数量往往少于式$\eqref{eq:EGP-1}$，它就是Efficient GlobalPointer最终所用的打分函数。

## 惊喜的实验

Efficient GlobalPointer已经内置在<code>bert4keras&gt;=0.10.9</code>中，读者只需要更改一行代码，就可以切换Efficient GlobalPointer了。



<pre><code class="language-python"># from bert4keras.layers import GlobalPointer
from bert4keras.layers import EfficientGlobalPointer as GlobalPointer</code></pre>



下面我们来对比一下GlobalPointer和Efficient GlobalPointer的结果：
\begin{array}{c}
\text{人民日报NER实验结果} \\
{\begin{array}{c|cc}
\hline
& \text{验证集F1} & \text{测试集F1}\\
\hline
\text{CRF} & 96.39\% & 95.46\% \\
\text{GlobalPointer} & \textbf{96.25%} & \textbf{95.51%} \\
\text{Efficient GlobalPointer} & 96.10\% & 95.36\%\\
\hline
\end{array}} \\ \\
\text{CLUENER实验结果} \\
{\begin{array}{c|cc}
\hline
& \text{验证集F1} & \text{测试集F1} \\
\hline
\text{CRF} & 79.51\% & 78.70\% \\
\text{GlobalPointer} & 80.03\% & 79.44\%\\
\text{Efficient GlobalPointer} & \textbf{80.66%} & \textbf{80.04%} \\
\hline
\end{array}} \\ \\
\text{CMeEE实验结果} \\
{\begin{array}{c|cc}
\hline
& \text{验证集F1} & \text{测试集F1} \\
\hline
\text{CRF} & 63.81\% & 64.39\% \\
\text{GlobalPointer} & 64.84\% & 65.98\%\\
\text{Efficient GlobalPointer} & \textbf{65.16%} & \textbf{66.54%} \\
\hline
\end{array}}
\end{array}

可以看到，Efficient GlobalPointer的实验结果还是很不错的，除了在人民日报任务上有轻微下降外，其他两个任务都获得了一定提升，并且整体而言提升的幅度大于下降的幅度，所以Efficient GlobalPointer不单单是节省了参数量，还提升了效果。而在速度上，Efficient GlobalPointer与原始的GlobalPointer几乎没有差别。

## 分析与评述

考虑到人民日报NER只有3种实体类型，CLUENER和CMeEE分别有10种和9种实体类型，从分数来看也是人民日报比其他两种要高，这说明CLUENER和CMeEE的难度更大。另一方面，在CLUENER和CMeEE上Efficient GlobalPointer都取得了提升，所以我们可以初步推断：实体类别越多、任务越难时，Efficient GlobalPointer越有效。

这也不难理解，原版GlobalPointer参数过大，那么平均起来每个参数更新越稀疏，相对来说也越容易过拟合；而Efficient GlobalPointer共享了“抽取”这一部分参数，仅通过“分类”参数区分不同的实体类型，那么实体抽取这一步的学习就会比较充分，而实体分类这一步由于参数比较少，学起来也比较容易。反过来，Efficient GlobalPointer的实验效果好也间接证明了式$\eqref{eq:EGP}$的分解是合理的。

当然，不排除在训练数据足够多的时候，原版GlobalPointer会取得更好的效果。但即便如此，在类别数目较多时，原版GlobalPointer可能会占用较多显存以至于难以使用，还是以base版$D=768,d=64$为例，如果类别数有100个，那么原版GlobalPointer的参数量为$2\times 768\times 64\times 100$，接近千万，不得不说确实是不够友好了。

## 最后的总结

本文指出了原版GlobalPointer的参数利用率不高问题，并提出了相应的改进版Efficient GlobalPointer。实验结果显示，Efficient GlobalPointer在降低参数量的同时，基本不会损失性能，甚至还可能获得提升。
