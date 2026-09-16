# 不成功的尝试：将多标签交叉熵推广到“n个m分类”上去

> 作者：苏剑林 · 科学空间 · 2022-07-15
>
> 原文：<https://spaces.ac.cn/archives/9158>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

可能有读者留意到，这次更新相对来说隔得比较久了。事实上，在上周末时就开始准备这篇文章了，然而笔者低估了这个问题的难度，几乎推导了整整一周，仍然还没得到一个完善的结果出来。目前发出来的，仍然只是一个失败的结果，希望有经验的读者可以指点指点。

在文章[《将“Softmax\+交叉熵”推广到多标签分类问题》](<https://spaces.ac.cn/archives/7359>)中，我们提出了一个多标签分类损失函数，它能自动调节正负类的不平衡问题，后来在[《多标签“Softmax\+交叉熵”的软标签版本》](<https://spaces.ac.cn/archives/9064>)中我们还进一步得到了它的“软标签”版本。本质上来说，多标签分类就是“$n$个2分类”问题，那么相应的，“$n$个$m$分类”的损失函数又该是怎样的呢？

这就是本文所要探讨的问题。

## 类比尝试

在软标签推广的文章[《多标签“Softmax\+交叉熵”的软标签版本》](<https://spaces.ac.cn/archives/9064>)中，我们是通过直接将“$n$个2分类”的sigmoid交叉熵损失，在$\log$内做一阶截断来得到最终结果的。同样的过程确实也可以推广到“$n$个$m$分类”的softmax交叉熵损失，这是笔者的第一次尝试。

记$\text{softmax}(s_{i,j}) = \frac{e^{s_{i,j}}}{\sum\limits_j e^{s_{i,j}}}$，$s_{i,j}$为预测结果，而$t_{i,j}$则为标签，那么
\begin{equation}\begin{aligned}-\sum_i\sum_j t_{i,j}\log \text{softmax}(s_{i,j}) =&\,\sum_i\sum_j t_{i,j}\log \left(1 + \sum_{k\neq j} e^{s_{i,k} - s_{i,j}}\right)\\
=&\,\sum_j \log \prod_i\left(1 + \sum_{k\neq j} e^{s_{i,k} - s_{i,j}}\right)^{t_{i,j}}\\
=&\,\sum_j \log \left(1 + \sum_i t_{i,j}\sum_{k\neq j} e^{s_{i,k} - s_{i,j}}+\cdots\right)\\
\end{aligned}\end{equation}
对$i$的求和默认是$1\sim n$，对$j$的求和默认是$1\sim m$。截断$\cdots$的高阶项，得到
\begin{equation}l = \sum_j \log \left(1 + \sum_{i,k\neq j} t_{i,j}e^{- s_{i,j} + s_{i,k}}\right)\label{eq:loss-1}\end{equation}
这就是笔者开始得到的loss，它是之前的结果到“$n$个$m$分类”的自然推广。事实上，如果$t_{i,j}$是硬标签，那么该loss基本上没什么问题。但笔者希望它像[《多标签“Softmax\+交叉熵”的软标签版本》](<https://spaces.ac.cn/archives/9064>)一样，对于软标签也能得到推导出相应的解析解。为此，笔者对它进行求导：
\begin{equation}\frac{\partial l}{\partial s_{i,j}} = \frac{- t_{i,j}e^{- s_{i,j}}\sum\limits_{k\neq j} e^{s_{i,k}}}{1 + \sum\limits_{i,k\neq j} t_{i,j}e^{- s_{i,j} + s_{i,k}}} + \sum_{h\neq j} \frac{t_{i,h}e^{- s_{i,h}}e^{s_{i,j}}}{1 + \sum\limits_{i,k\neq h} t_{i,h}e^{- s_{i,h} + s_{i,k}}}\end{equation}
所谓解析解，就是通过方程$\frac{\partial l}{\partial s_{i,j}}=0$来解出。然而笔者尝试了好几天，都求不出方程的解，估计并没有简单的显式解，因此，第一次尝试失败。

## 结果倒推

尝试了几天实在没办法后，笔者又反过来想：既然直接类比出来的结果无法求解，那么我干脆从结果倒推好了，即先把解确定，然后再反推方程应该是怎样的。于是，笔者开始了第二次尝试。

首先，观察发现原来的多标签损失，或者前面得到的损失$\eqref{eq:loss-1}$，都具有如下的形式：
\begin{equation}l = \sum_j \log \left(1 + \sum_i t_{i,j}e^{- f(s_{i,j})}\right)\label{eq:loss-2}\end{equation}
我们就以这个形式为出发点，求导
\begin{equation}\frac{\partial l}{\partial s_{i,k}} = \sum_j \frac{- t_{i,j}e^{- f(s_{i,j})}\frac{\partial f(s_{i,j})}{\partial s_{i,k}}}{1 + \sum\limits_i t_{i,j}e^{- f(s_{i,j})}}\end{equation}
我们希望$t_{i,j}=\text{softmax}(f(s_{i,j}))=e^{f(s_{i,j})}/Z_i$就是$\frac{\partial l}{\partial s_{i,k}}=0$的解析解，其中$Z_i=\sum\limits_j e^{f(s_{i,j})}$。那么代入得到
\begin{equation}0=\frac{\partial l}{\partial s_{i,k}} = \sum_j \frac{- (1/Z_i)\frac{\partial f(s_{i,j})}{\partial s_{i,k}}}{1 + \sum\limits_i 1/Z_i} = \frac{- (1/Z_i)\frac{\partial \left(\sum\limits_j f(s_{i,j})\right)}{\partial s_{i,k}}}{1 + \sum\limits_i 1/Z_i}\end{equation}
所以要让上式自然成立，我们发现只需要让$\sum\limits_j f(s_{i,j})$等于一个跟$i,j$都无关的常数。简单起见，我们让
\begin{equation}f(s_{i,j})=s_{i,j}-
\bar{s}_i,\qquad \bar{s}_i=\frac{1}{m}\sum_j s_{i,j}\end{equation}
这样自然地有$\sum\limits_j f(s_{i,j})=0$，对应的优化目标就是
\begin{equation}l = \sum_j \log \left(1 + \sum_i t_{i,j}e^{- s_{i,j} + \bar{s}_i}\right)\label{eq:loss-3}\end{equation}
$\bar{s}_i$不影响归一化结果，所以它的理论最优解是$t_{i,j}=\text{softmax}(s_{i,j})$。

然而，看上去很美好，然而它实际上的效果会比较糟糕，$t_{i,j}=\text{softmax}(s_{i,j})$确实是理论最优解，但实际上标签越接近硬标签，它的效果会越差。因为我们知道对于损失$\eqref{eq:loss-3}$来说，只要$s_{i,j} \gg \bar{s}_i$，损失就会很接近于0，而要达到$s_{i,j} \gg \bar{s}_i$，$s_{i,j}$不一定是$s_{i,1},s_{i,2},\cdots,s_{i,m}$中的最大者，这就无法实现分类目标了。

## 思考分析

现在我们得到了两个结果，式$\eqref{eq:loss-1}$是原来多标签交叉熵的类比推广，它在硬标签的情况下效果还是不错的，但是由于求不出软标签情况下的解析解，因此软标签的情况无法做理论评估；式$\eqref{eq:loss-3}$是从结果理论倒推出来的，理论上它的解析解就是简单的softmax，但由于实际优化算法的限制，硬标签的表现通常很差，甚至无法保证目标logits是最大值。特别地，当$m=2$时，式$\eqref{eq:loss-1}$和式$\eqref{eq:loss-3}$都能退化为多标签交叉熵。

我们知道，多标签交叉熵能够自动调节正负样本不平衡的问题，同样地，虽然我们目前还没能得到一个完美的推广，但理论上推广到“$n$个$m$分类”后依然能够自动调节$m$个类的不平衡问题。那么平衡的机制是怎样的呢？其实不难理解，不管是类比推广的式$\eqref{eq:loss-1}$，还是一般的假设式$\eqref{eq:loss-2}$，对$i$的求和都放在了$\log$里边，原本每个类的损失占比大体上是正比于“<u>该类的样本数</u>”的，改为放在了$\log$里边求和后，每个类的损失占就大致等于“<u>该类的样本数的对数</u>”，从而缩小了每个类的损失差距，自动缓解了不平衡问题。

遗憾的是，本文还没有得出关于“$n$个$m$分类”的完美推广——它应该包含两个特性：1、通过$\log$的方法自动调节类别不平衡现象；2、能够求出软标签情况下的解析解。对于硬标签来说，直接用式$\eqref{eq:loss-1}$应该是足够了；而对于软标签来说，笔者实在是没辙了，欢迎有兴趣的读者一起思考交流。

## 文章小结

本文尝试将之前的多标签交叉熵推广到“$n$个$m$分类”上去，遗憾的是，这一次的推广并不算成功，暂且将结果分享在此，希望有兴趣的读者能一起参与改进。
