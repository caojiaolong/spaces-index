# 函数光滑化杂谈：不可导函数的可导逼近

> 作者：苏剑林 · 科学空间 · 2019-05-20
>
> 原文：<https://spaces.ac.cn/archives/6620>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

一般来说，神经网络处理的东西都是连续的浮点数，标准的输出也是连续型的数字。但实际问题中，我们很多时候都需要一个离散的结果，比如分类问题中我们希望输出正确的类别，“类别”是离散的，“类别的概率”才是连续的；又比如我们很多任务的评测指标实际上都是离散的，比如分类问题的正确率和F1、机器翻译中的BLEU，等等。

还是以分类问题为例，常见的评测指标是正确率，而常见的损失函数是交叉熵。交叉熵的降低与正确率的提升确实会有一定的关联，但它们不是绝对的单调相关关系。换句话说，交叉熵下降了，正确率不一定上升。显然，如果能用正确率的相反数做损失函数，那是最理想的，但正确率是不可导的（涉及到$\text{argmax}$等操作），所以没法直接用。

这时候一般有两种解决方案；一是动用强化学习，将正确率设为奖励函数，这是“用牛刀杀鸡”的方案；另外一种是试图给正确率找一个光滑可导的近似公式。本文就来探讨一下常见的不可导函数的光滑近似，有时候我们称之为“光滑化”，有时候我们也称之为“软化”。

## max

后面谈到的大部分内容，基础点就是$\max$操作的光滑近似，我们有：
\begin{equation}\max(x_1,x_2,\dots,x_n) = \lim_{K\to +\infty}\frac{1}{K}\log\left(\sum_{i=1}^n e^{K x_i}\right)\end{equation}
所以选定常数$K$，我们就有近似：
\begin{equation}\max(x_1,x_2,\dots,x_n) \approx \frac{1}{K}\log\left(\sum_{i=1}^n e^{K x_i}\right)\end{equation}
在模型中，很多时候可以设$K=1$，这等价于把$K$融合到模型自身之中，所以最简单地有：
\begin{equation}\begin{aligned}\max(x_1,x_2,\dots,x_n) \approx&\, \log\left(\sum_{i=1}^n e^{x_i}\right) \\
\triangleq&\, \text{logsumexp}(x_1, x_2, \dots, x_n)\end{aligned}\label{eq:max-approx}\end{equation}

这里出现了$\text{logsumexp}$，这是一个很常见的算子，在这里它是$\max$函数的光滑近似。没错，$\max$函数的光滑近似其实是$\text{logsumexp}$，而不是字面上很像的$\text{softmax}$。相关推导还可以参考我之前写的[《寻求一个光滑的最大值函数》](<https://spaces.ac.cn/archives/3290>)。

## softmax

刚才说$\text{softmax}$不是$\max$的光滑近似，那它是谁的光滑近似呢？它事实上是$\text{onehot}(\text{argmax}(\boldsymbol{x}))$的光滑近似，即先求出最大值所在的位置，然后生成一个等长的向量，最大值那一位置1，其它位置都置0，比如：
\begin{equation}[2, 1, 4, 5, 3]\quad \to \quad [0, 0, 0, 1, 0]\end{equation}
我们可以简单地给出一个从$\text{logsumexp}$到$\text{softmax}$的推导过程。考虑向量$\boldsymbol{x}=[x_1, x_2, \dots, x_n]$，然后考虑
\begin{equation}\boldsymbol{x}'=[x_1, x_2, \dots, x_n] - \max(x_1, x_2, \dots, x_n)\end{equation}
即每一位都减去整体的最大值，这样的新向量与原向量最大值所在的位置是一样的，即$\text{onehot}(\text{argmax}(\boldsymbol{x}))=\text{onehot}(\text{argmax}(\boldsymbol{x}'))$。不失一般性，考虑$x_1,x_2,\dots,x_n$两两不相等的情况，那么新向量的最大值显然为0，并且除去最大值外，其余各位都是负数。这样一来，我们可以考虑
\begin{equation}e^{\boldsymbol{x}'}=[e^{x_1 - \max(x_1, x_2, \dots, x_n)}, e^{x_2 - \max(x_1, x_2, \dots, x_n)}, \dots, e^{x_n - \max(x_1, x_2, \dots, x_n)}]\end{equation}
作为$\text{onehot}(\text{argmax}(\boldsymbol{x}'))$的近似，因为最大值为0，所以对应的位置是$e^0=1$，而其余为负，取指数后会比较接近于0。

最后将近似$\eqref{eq:max-approx}$代入上式，化简，就可以得到
\begin{equation}\begin{aligned}\text{onehot}(\text{argmax}(\boldsymbol{x}))=&\,\text{onehot}(\text{argmax}(\boldsymbol{x}'))\\
\approx&\, \left(\frac{e^{x_1}}{\sum\limits_{i=1}^n e^{x_i}}, \frac{e^{x_2}}{\sum\limits_{i=1}^n e^{x_i}}, \dots, \frac{e^{x_n}}{\sum\limits_{i=1}^n e^{x_i}}\right)\\
\triangleq&\,\text{softmax}(x_1, x_2, \dots, x_n)
\end{aligned}\end{equation}

## argmax

$\text{argmax}$是指直接给出向量最大值所在的下标（一个整数），比如
\begin{equation}[2, 1, 4, 5, 3]\quad \to \quad 4\end{equation}
这里我们遵循平时的使用习惯，下标是从1开始的，所以返回结果是4；但在编程语言中一般是从0开始的，所以在编程语言中返回结果一般是3。

如果是$\text{argmax}$的光滑近似，自然是希望输出一个接近4的浮点数。为了构造这样的近似，我们先观察到$\text{argmax}$实际上等于
\begin{equation}\text{sum}\Big(\underbrace{[1, 2, 3, 4, 5]}_{\text{序向量[1, 2, ..., n]}}\,\, \otimes\,\,\underbrace{[0, 0, 0, 1, 0]}_{\text{onehot}(\text{argmax}(\boldsymbol{x}))}\Big)\end{equation}
即数组$[1, 2, \dots, n]$与$\text{onehot}(\text{argmax}(\boldsymbol{x}))$的内积。那构造$\text{argmax}$的软化版就简单了，将$\text{onehot}(\text{argmax}(\boldsymbol{x}))$换成$\text{softmax}(\boldsymbol{x})$就行了，即
\begin{equation}\text{argmax} (\boldsymbol{x}) \approx \sum_{i=1}^n i\times \text{softmax}(\boldsymbol{x})_i\end{equation}

## 正确率

上述讨论的若干个近似，基本都是在one hot向量的基础上推导出正确形式，然后用softmax近似one hot，从而得到光滑近似。用这种思想，还可以导出很多算子的光滑近似，比如正确率。

简单起见，引入记号$\boldsymbol{1}_k$，它表示第$k$位为1的one hot向量。假设在分类问题中，目标类别是$i$，预测类别是$j$，那么可以考虑one hot向量$\boldsymbol{1}_i$和$\boldsymbol{1}_j$，然后考虑内积
\begin{equation}\langle \boldsymbol{1}_i, \boldsymbol{1}_j\rangle = \left\{\begin{aligned}&1,\,\,(i=j)\\ &0,\,\,(i\neq j)\end{aligned}\right.\end{equation}
也就是说两个类别一样时，内积刚好为1，而两个类别不一样时，内积刚好为0，所以目标类别和预测类别对应的one hot向量的内积，刚好定义了一个“预测正确”的计数函数，而有了计数函数，就可以计算正确率：
\begin{equation}\text{正确率}=\frac{1}{|\mathcal{B}|}\sum_{\boldsymbol{x}\in\mathcal{B}}\langle \boldsymbol{1}_i(\boldsymbol{x}), \boldsymbol{1}_j(\boldsymbol{x})\rangle\end{equation}
其中$\mathcal{B}$表示当前batch，上式正是计算一个batch内正确率的函数。而在神经网络中，为了保证可导性，最后的输出只能是一个概率分布（softmax后的结果），所以正确率的光滑近似，就是将预测类别的one hot向量，换成概率分布：
\begin{equation}\text{正确率}\approx \frac{1}{|\mathcal{B}|}\sum_{\boldsymbol{x}\in\mathcal{B}}\langle \boldsymbol{1}_i(\boldsymbol{x}), p(\boldsymbol{x})\rangle\end{equation}
类似地，可以导出recall、f1等指标的光滑近似。以二分类为例，假设$p(\boldsymbol{x})$是正类的概率，$t(\boldsymbol{x})$是样本$\boldsymbol{x}$的标签（0或1），则正类的f1的光滑近似是：
\begin{equation}\text{正类F1}\approx\frac{2 \sum\limits_{\boldsymbol{x}\in\mathcal{B}}t(\boldsymbol{x}) p(\boldsymbol{x})}{\sum\limits_{\boldsymbol{x}\in\mathcal{B}}\big[t(\boldsymbol{x}) + p(\boldsymbol{x})\big]}\end{equation}

这样导出来的正确率近似公式是可导的，可以直接将它的相反数作为loss。但事实上在采样估计过程中，它是f1的有偏估计（分母也有对batch的求和），有时候会影响优化轨迹甚至导致发散，所以一般情况下最好不要直接用，而是先用普通的交叉熵训练到差不多了，然后再用f1的相反数作为loss来微调。

## softkmax

$\text{softmax}$是“最大值那一位置1、其余置0”的光滑近似，那如果是“最大的$k$个值的位置都置1、其余置0”的光滑近似呢？我们可以称之为$\text{soft-}k\text{-max}$。

我没有构造出$\text{soft-}k\text{-max}$的简单形式，但可以利用递归地构造出来：

> 输入为$\boldsymbol{x}$，初始化$\boldsymbol{p}^{(0)}$为全0向量；
> 执行$\boldsymbol{x} = \boldsymbol{x} - \min(\boldsymbol{x})$（保证所有元素非负）
> 
> 
> 
> 对于$i=1,2,\dots,k$，执行：
>      $\boldsymbol{y} = (1 - \boldsymbol{p}^{(i-1)})\otimes\boldsymbol{x}$;
>      $\boldsymbol{p}^{(i)} = \boldsymbol{p}^{(i-1)} + \text{softmax}(\boldsymbol{y})$
> 
> 
> 
> 返回$\boldsymbol{p}^{(k)}$。

至于原理，将$\text{softmax}(\boldsymbol{y})$换成$\text{onehot}(\text{argmax}(\boldsymbol{y}))$然后递归下去就明白了，其实就是先$\max$，然后把将$\max$那一位减掉，这样次最大值就变成了最大值，然后重新$\text{softmax}$，递归下去即可。

## 总结

函数光滑化是一个比较有意思的数学内容，在机器学习中也经常出现。一方面，它是处理将某些操作可导化的技巧，使得模型可以直接使用反向传播求解，而不需要“动用”强化学习；另一方面，在某些情况下它也能增强模型的解释性，因为往往对应的不可导函数解释性是比较好的，将光滑化版本代入训练完成后，也许可能恢复为不可导版本来解释模型的输出结果。

当然，作为纯粹的数学之美来欣赏，也是相当赏心悦目的～
