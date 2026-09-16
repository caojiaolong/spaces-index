# 从动力学角度看优化算法（七）：SGD ≈ SVM？

> 作者：苏剑林 · 科学空间 · 2020-12-21
>
> 原文：<https://spaces.ac.cn/archives/8009>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，在深度学习之前，机器学习是SVM（Support Vector Machine，支持向量机）的天下，曾经的它可谓红遍机器学习的大江南北，迷倒万千研究人员，直至今日，“手撕SVM”仍然是大厂流行的面试题之一。然而，时过境迁，当深度学习流行起来之后，第一个革的就是SVM的命，现在只有在某些特别追求效率的场景以及大厂的面试题里边，才能看到SVM的踪迹了。

峰回路转的是，最近Arxiv上的一篇论文[《Every Model Learned by Gradient Descent Is Approximately a Kernel Machine》](<https://papers.cool/arxiv/2012.00152>)做了一个非常“霸气”的宣言：

> 任何由梯度下降算法学出来的模型，都是可以近似看成是一个SVM！

这结论真不可谓不“霸气”，因为它已经不只是针对深度学习了，而是只要你用梯度下降优化的，都不过是一个SVM（的近似）。笔者看了一下原论文的分析，感觉确实挺有意思也挺合理的，有助于加深我们对很多模型的理解，遂跟大家分享一下。

## SVM基础

一般的SVM可以表示为如下形式：
\begin{equation}y = g\left(\beta + \sum_i \alpha_i K(x, x_i)\right)\label{eq:svm}\end{equation}
其中$\{(x_i, y_i)\}$是训练数据对，$\alpha_i, \beta$是可学习参数，标准核机器的输出是一个标量，所以这里考虑的$y,\alpha_i, \beta$都是标量。$K(x, x_i)$则称为“核函数”，它衡量了输入$x$与训练样本$x_i$之间的某种相似度。SVM是更广义的“核机器（Kernel Machine）”模型的一种（可能是最出名的一种），属于“核方法”范畴。

直观理解，其实SVM就是一个检索模型，它检索了输入与所有训练样本的相似度$K(x, x_i)$，然后加权求和。所以，严格上来说，SVM的参数量除了各个$\alpha_i$和$\beta$外，还包括训练集的输入$x_i$，说白了，它就是把整个训练集都给记下来了。相比之下，深度学习模型也有很多参数，但这些参数都是直接由梯度下降求出来的，并不是直接把训练集存起来，而由于这个特点，所以深度学习模型通常认为是能自动学习到更智能的特征。

## 解析推导

SVM理论不是本文的重点，我们知道它的形式如$\eqref{eq:svm}$即可。在这一节中，我们将会推导梯度下降的一个解析解，并且发现这个解跟式$\eqref{eq:svm}$具有非常相似的形式，因而我们说梯度下降出来的模型都可以近似看成一个SVM模型。

假设我们的模型是$y=f_{\theta}(x)$，$\theta$是可训练参数，单个样本的损失函数是$l(y_i, f_{\theta}(x_i))$，那么训练所用的损失函数为
\begin{equation}L(\theta) = \sum_i l(y_i, f_{\theta}(x_i))\end{equation}
为了使得后面的推导更简洁，这里使用了求和的形式，一般情况下是求平均才对，但这不影响最终的结果。在“[从动力学角度看优化算法](<https://spaces.ac.cn/search/%E4%BB%8E%E5%8A%A8%E5%8A%9B%E5%AD%A6%E8%A7%92%E5%BA%A6%E7%9C%8B%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/>)”系列文章中，我们坚持的观点是梯度下降求解参数$\theta$，相当于在求解动力系统
\begin{equation}\frac{d\theta}{dt} = -\frac{\partial L(\theta)}{\partial \theta}=-\sum_i \frac{\partial l(y_i, f_{\theta}(x_i))}{\partial \theta}=-\sum_i \frac{\partial l(y_i, f_{\theta}(x_i))}{\partial f_{\theta}(x_i)}\frac{\partial f_{\theta}(x_i)}{\partial \theta}\end{equation}
这同样也是本文的重要出发点。现在，我们考虑$f_{\theta}(x)$的变化情况：
\begin{equation}\begin{aligned}
\frac{df_{\theta}(x)}{dt} &= \sum_j \frac{\partial f_{\theta}(x)}{\partial \theta_j}\frac{d\theta_j}{dt}\\
&=-\sum_j \frac{\partial f_{\theta}(x)}{\partial \theta_j}\sum_i \frac{\partial l(y_i, f_{\theta}(x_i))}{\partial f_{\theta}(x_i)}\frac{\partial f_{\theta}(x_i)}{\partial \theta_j}\\
&=-\sum_i \frac{\partial l(y_i, f_{\theta}(x_i))}{\partial f_{\theta}(x_i)} \sum_j \frac{\partial f_{\theta}(x)}{\partial \theta_j} \frac{\partial f_{\theta}(x_i)}{\partial \theta_j}
\end{aligned}\end{equation}
可以看到，对$j$求和这一步，事实上就是梯度的内积$\langle\nabla_{\theta} f_{\theta}(x), \nabla_{\theta} f_{\theta}(x_i)\rangle$，在神经网络中它还有一个非常酷的名字，叫做“[Neural Tangent Kernel](<https://papers.cool/arxiv/1806.07572>)”，我们将其记为
\begin{equation}K_{\theta}(x, x_i) = \langle\nabla_{\theta} f_{\theta}(x), \nabla_{\theta} f_{\theta}(x_i)\rangle = \sum_j \frac{\partial f_{\theta}(x)}{\partial \theta_j} \frac{\partial f_{\theta}(x_i)}{\partial \theta_j}\end{equation}
并且记$\alpha_{\theta,i}=-\frac{\partial l(y_i, f_{\theta}(x_i))}{\partial f_{\theta}(x_i)}$，那么
\begin{equation}\frac{df_{\theta}(x)}{dt} = \sum_i \alpha_{\theta,i} K_{\theta}(x, x_i)\end{equation}
可见，模型$f_{\theta}(x)$每个时刻的变化量都是一个SVM，假如我们已经知道优化过程中$\theta$的变化轨迹为$\theta(t),t\in[0, T]$，那么最终的模型就是
\begin{equation}f_{\theta_T}(x) = f_{\theta_0}(x) + \sum_i \int_0^T \alpha_{\theta(t),i} K_{\theta(t)}(x, x_i) dt\label{eq:sgdf}\end{equation}

## 结果分析

经过一番推导，我们的得到了式$\eqref{eq:sgdf}$，它是当学习率趋于0的梯度下降的理论解。从推导过程可以看到，这个结果只依赖于梯度下降本身，跟模型具体结构没关系。对于式$\eqref{eq:sgdf}$，我们可以从下面的角度理解它。

首先，我们将记$\beta(x) = f_{\theta_0}(x)$，它其实就是初始化模型，尽管它理论上是依赖于$x$的，但很多时候它会表现得接近一个常数（比如多分类模型时，初始化模型的输出通常接近一个均匀分布），因此我们可以将当它是一个常数项。然后，我们可以记
\begin{equation}\alpha_i (x) = \frac{\int_0^T \alpha_{\theta(t),i} K_{\theta(t)}(x, x_i) dt}{\int_0^T K_{\theta(t)}(x, x_i) dt}, \quad K(x, x_i) = \int_0^T K_{\theta(t)}(x, x_i) dt
\end{equation}
那么
\begin{equation}f_{\theta_T}(x) = \beta(x) + \sum_i \alpha_i (x) K(x, x_i)\end{equation}
这在形式上就跟SVM很像了，区别就在于SVM的$\alpha_i,\beta$应该是独立于$x$的，而这里则依赖于$x$。$\beta(x)$我们已经分析过了，而$\alpha_i(x)$由于它是数学期望的形式，被期望的对象不依赖于$x$，而是权重依赖于$x$，那么它可能对$x$的依赖也相对弱些，因此跟$\beta(x)$一样，我们也许可以近似地忽略它对$x$的依赖。不过，在笔者看来，依不依赖$x$并不算是关键，最重要的是最终的结果呈现出了$\sum\limits_i \alpha_i(x) K(x, x_i)$的形式，那就意味着它在一定程度上也是学习到了一个检索训练集的过程，这才是它真正跟SVM的相似之处。

上述讨论的是输出标量的模型，如果输出是一个$d$维向量，那么最终形式也是相同的，只不过此时$\beta(x),\alpha_i(x)$也是一个$d$维向量，而$K(x, x_i)$是一个$d\times d$的矩阵，这种情况下哪怕$\beta(x),\alpha_i(x)$与$x$无关，也不是我们通常意义下的（多分类）SVM模型。但它依然具有$\sum\limits_i \alpha_i(x) K(x, x_i)$的形式，因此某种意义上来说它仍然是检索训练集的操作。

此外，上述结论针对的是（全量）梯度下降，而对于随机梯度下降（SGD）来说，我们不再是用全量数据来算损失函数，对此我们在第一篇[《从动力学角度看优化算法（一）：从SGD到动量加速》](<https://spaces.ac.cn/archives/5655>)也做过讨论，可以认为SGD是在梯度下降的基础上引入了噪声，也就是收敛路径$\theta(t)$带有随机噪声，其余结果基本不变，因此上述结论对SGD也是成立的。

## 拓展思考

那么，这个结果能给我们带来什么思想冲击呢？原论文在“Discussion”那一节花了相当长的篇幅讨论这个事情，这里我们也来琢磨一下这个事情。

从深度学习的视角来看，这个结果揭示了深层神经网络模型与传统的核方法之间的联系，借助核方法的可解释性来增强神经网络的可解释性。比如，通过梯度内积作为相似度度量，我们或许可以从训练集中检索出与输入相近的训练样本，以解释输出的决策过程。更进一步地，如果该方向能够得到更为精确的量化，那么它有可能大大改进增量学习的方法，即对于新来的标注样本，我们可能只需要想办法往模型中添加$a_i(x) K(x, x_i)$的项，而不需要重新训练模型。

反过来看，该结果也许能促进核机器、核方法的发展。传统的核函数依赖于人为定义，而上述梯度内积形式的核函数给我们带来了新的构建核函数的思路，增强核方法对复杂函数的建模能力。同时，由于梯度下降与核机器的相似性，我们最终或许可以通过梯度下降来训练核机器，从而克服核机器在大规模数据下的训练难题，等等。

还有一些别的脑洞可以发散一下，比如我们知道对于凸优化问题有唯一解，并且理论上梯度下降总可以找到这个解，而前面又说梯度下降相当于一个SVM。所以，这是不是意味着所有凸优化问题的解都相当于一个SVM？这个脑洞够不够大？

总之，揭示梯度下降与核机器之间的联系，有助于两者的进一步借鉴与融合，并且有可能发展出一些新的研究思路。
