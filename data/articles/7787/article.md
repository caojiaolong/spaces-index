# 从动力学角度看优化算法（五）：为什么学习率不宜过小？

> 作者：苏剑林 · 科学空间 · 2020-10-10
>
> 原文：<https://spaces.ac.cn/archives/7787>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

本文的主题是“为什么我们需要有限的学习率”，所谓“有限”，指的是不大也不小，适中即可，太大容易导致算法发散，这不难理解，但为什么太小也不好呢？一个容易理解的答案是，学习率过小需要迭代的步数过多，这是一种没有必要的浪费，因此从“节能”和“加速”的角度来看，我们不用过小的学习率。但如果不考虑算力和时间，那么过小的学习率是否可取呢？Google最近发布在Arxiv上的论文[《Implicit Gradient Regularization》](<https://papers.cool/arxiv/2009.11162>)试图回答了这个问题，它指出有限的学习率隐式地给优化过程带来了梯度惩罚项，而这个梯度惩罚项对于提高泛化性能是有帮助的，因此哪怕不考虑算力和时间等因素，也不应该用过小的学习率。

对于梯度惩罚，本博客已有过多次讨论，在文章[《对抗训练浅谈：意义、方法和思考（附Keras实现）》](<https://spaces.ac.cn/archives/7234>)和[《泛化性乱弹：从随机噪声、梯度惩罚到虚拟对抗训练》](<https://spaces.ac.cn/archives/7466>)中，我们就分析了对抗训练一定程度上等价于对输入的梯度惩罚，而文章[《我们真的需要把训练集的损失降低到零吗？》](<https://spaces.ac.cn/archives/7643>)介绍的Flooding技巧则相当于对参数的梯度惩罚。总的来说，不管是对输入还是对参数的梯度惩罚，都对提高泛化能力有一定帮助。

## 降得最快的方向

该论文跟[这个系列的文章](<https://spaces.ac.cn/search/%E4%BB%8E%E5%8A%A8%E5%8A%9B%E5%AD%A6%E8%A7%92%E5%BA%A6%E7%9C%8B%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/>)一样，将优化过程看成是求解微分方程。回顾之前的博文[《从动力学角度看优化算法（三）：一个更整体的视角》](<https://spaces.ac.cn/archives/6261>)，设损失函数为$L(\boldsymbol{\theta})$，我们将$\boldsymbol{\theta}$看成是看成是沿着某种时间参数$t$变化的轨迹$\boldsymbol{\theta}(t)$，现在我们考虑它的变化率：
\begin{equation}\frac{d}{dt}L(\boldsymbol{\theta}(t))=\left\langle\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta}(t)),\, \dot{\boldsymbol{\theta}}(t)\right\rangle\end{equation}
我们希望$L(\boldsymbol{\theta}(t))$随着时间的变化是递减的（Loss越小越好），所以希望上式小于0，当模长$\Vert\dot{\boldsymbol{\theta}}(t)\Vert$固定时，上式右端的最小值在梯度的反方向$-\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta}(t))$取到，所以我们说梯度的负方向下降得最快方向。简单期间，我们可以直接令
\begin{equation}\dot{\boldsymbol{\theta}}(t) = -\nabla_{\boldsymbol{\theta}}L(\boldsymbol{\theta}(t))\triangleq - \boldsymbol{g}(\boldsymbol{\theta}(t))\label{eq:odes}\end{equation}
那么求解参数$\boldsymbol{\theta}$就转化为求解上述常微分方程组，这也是“从动力学角度看优化算法”这个系列的基本出发点。

## 藏在学习率中的正则

然而，实际的问题是，我们没法真正去求解微分方程组$\eqref{eq:odes}$，我们只能用数值迭代，比如采用最简单的欧拉法，得到
\begin{equation}\boldsymbol{\theta}_{t+\gamma} = \boldsymbol{\theta}_{t} - \gamma \boldsymbol{g}(\boldsymbol{\theta}_t)\label{eq:gd}\end{equation}
这其实就是最朴素的梯度下降法，其中$\gamma$也就是我们常说的学习率。上式本质上就是一个差分方程。

可以想象，从$t=0$出发，得到的点$\boldsymbol{\theta}_{\gamma},\boldsymbol{\theta}_{2\gamma},\boldsymbol{\theta}_{3\gamma},\cdots$与方程组$\eqref{eq:odes}$的精确解$\boldsymbol{\theta}(\gamma),\boldsymbol{\theta}(2\gamma),\boldsymbol{\theta}(3\gamma),\cdots$会有一定的出入。如何衡量出入到什么程度呢？不妨这样想象，$\boldsymbol{\theta}_{\gamma},\boldsymbol{\theta}_{2\gamma},\boldsymbol{\theta}_{3\gamma},\cdots$其实也是某个类似$\eqref{eq:odes}$的微分方程组的精确解，只不过对应的$\boldsymbol{g}(\boldsymbol{\theta}(t))$换成了某个新的$\tilde{\boldsymbol{g}}(\boldsymbol{\theta}_t)$，我们比较$\tilde{\boldsymbol{g}}(\boldsymbol{\theta}_t)$与$\boldsymbol{g}(\boldsymbol{\theta}(t))$的差异就好了。

经推导，如果仅保留到$\gamma$的一阶项，那么有
\begin{equation}\tilde{\boldsymbol{g}}(\boldsymbol{\theta}_t) = \boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{\gamma}{4}\nabla_{\boldsymbol{\theta}}\Vert \boldsymbol{g}(\boldsymbol{\theta}_t)\Vert^2 = \nabla_{\boldsymbol{\theta}}\left(L(\boldsymbol{\theta}_t) + \frac{1}{4}\gamma\Vert \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_t)\Vert^2\right)\end{equation}
推导过程我们放在下一节。可以看到，其实就相当于往损失函数里边加入了梯度惩罚形式的正则项$\frac{1}{4}\gamma\Vert \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta})\Vert^2$，而梯度惩罚项有助于模型到达更加平缓的区域，有利于提高泛化性能。这也就是说，离散化的迭代过程隐式地带来了梯度惩罚项，反而是对模型的泛化是有帮助的，而如果$\gamma\to 0$，这个隐式的惩罚则会变弱甚至消失。

因此，结论就是学习率不宜过小，较大的学习率不仅有加速收敛的好处，还有提高模型泛化能力的好处。当然，可能有些读者会想，我直接把梯度惩罚加入到loss中，是不是就可以用足够小的学习率了？理论上确实是的，原论文将梯度惩罚加入到loss中的做法，称为“显式梯度惩罚”。

## 差分方程到微分方程

对于差分方程到微分方程的转换，我们可以用普通的“摄动法”来求解，本博客也有过简单介绍（可以查看标签“[摄动](<https://spaces.ac.cn/tag/%E6%91%84%E5%8A%A8/>)”）。不过更漂亮的解法是直接利用算符的级数运算来做，参考之前的文章[《算符的艺术：差分、微分与伯努利数》](<https://spaces.ac.cn/archives/3018>)。

我们用泰勒级数展开$\boldsymbol{\theta}_{t+\gamma}$：
\begin{equation}\boldsymbol{\theta}_{t+\gamma}=\boldsymbol{\theta}_{t}+\gamma \dot{\boldsymbol{\theta}}_{t} + \frac{1}{2}\gamma^2\ddot{\boldsymbol{\theta}}_{t} + \frac{1}{6}\gamma^3\dddot{\boldsymbol{\theta}}_{t} + \cdots\end{equation}
如果将对$t$求导的运算记为$D$，那么上式实际上是
\begin{equation}\boldsymbol{\theta}_{t+\gamma} = \left(1+\gamma D + \frac{1}{2}\gamma^2 D^2 + \frac{1}{6}\gamma^3 D^3 + \cdots\right)\boldsymbol{\theta}_{t} = e^{\gamma D}\boldsymbol{\theta}_{t}\end{equation}
所以差分方程$\eqref{eq:gd}$可以写为
\begin{equation}\left(e^{\gamma D} - 1\right)\boldsymbol{\theta}_{t} = - \gamma \boldsymbol{g}(\boldsymbol{\theta}_t)\end{equation}
跟常规的代数运算一样，我们有
\begin{equation}\begin{aligned}
D\boldsymbol{\theta}_{t} =& - \gamma \left(\frac{D}{e^{\gamma D} - 1}\right)\boldsymbol{g}(\boldsymbol{\theta}_t)\\
=& - \left(1 - \frac{1}{2}\gamma D + \frac{1}{12}\gamma^2 D^2 - \frac{1}{720}\gamma^4 D^4 + \cdots\right)\boldsymbol{g}(\boldsymbol{\theta}_t)
\end{aligned}\end{equation}
等号左端就是$\dot{\boldsymbol{\theta}}_{t}$，因此等号右端就是$-\tilde{\boldsymbol{g}}(\boldsymbol{\theta}_t)$的表达式了，保留到一阶项为
\begin{equation}
- \left(1 - \frac{1}{2}\gamma D\right)\boldsymbol{g}(\boldsymbol{\theta}_t) =  - \boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{1}{2}\gamma \frac{d}{dt}\boldsymbol{g}(\boldsymbol{\theta}_t)
= - \boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{1}{2}\gamma \nabla_{\boldsymbol{\theta}}\boldsymbol{g}(\boldsymbol{\theta}_t)\dot{\boldsymbol{\theta}}_t
\end{equation}
也就是
\begin{equation}\begin{aligned}
\dot{\boldsymbol{\theta}}_{t} =& - \boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{1}{2}\gamma \nabla_{\boldsymbol{\theta}}\boldsymbol{g}(\boldsymbol{\theta}_t)\dot{\boldsymbol{\theta}}_t\\
=&- \boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{1}{2}\gamma \nabla_{\boldsymbol{\theta}}\boldsymbol{g}(\boldsymbol{\theta}_t)\left[- \boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{1}{2}\gamma \nabla_{\boldsymbol{\theta}}\boldsymbol{g}(\boldsymbol{\theta}_t)\dot{\boldsymbol{\theta}}_t\right]\\
=&- \boldsymbol{g}(\boldsymbol{\theta}_t) - \frac{1}{2}\gamma \nabla_{\boldsymbol{\theta}}\boldsymbol{g}(\boldsymbol{\theta}_t)\boldsymbol{g}(\boldsymbol{\theta}_t)\quad\text{(略去二阶项)}\\
=&- \boldsymbol{g}(\boldsymbol{\theta}_t) - \frac{1}{4}\gamma \nabla_{\boldsymbol{\theta}}\Vert\boldsymbol{g}(\boldsymbol{\theta}_t)\Vert^2
\end{aligned}\end{equation}
所以一阶的$\tilde{\boldsymbol{g}}(\boldsymbol{\theta}_t)=\boldsymbol{g}(\boldsymbol{\theta}_t) + \frac{1}{4}\gamma \nabla_{\boldsymbol{\theta}}\Vert\boldsymbol{g}(\boldsymbol{\theta}_t)\Vert^2$，推导完毕。

## 例行公事的小总结

深度学习的发展和普及离不开基于梯度下降的优化器的成功应用，而梯度下降为何能如此成功，依然还没得到深刻的解释。众多研究人员在“炼丹”过程中，多多少少也能总结出一些不知道为什么有效的“奇技淫巧”出来，诸如batch\_size该取多大、学习率该怎么调，估计每个人也有自己的经验。

对于“学习率不能过小”这个现象，大家应该都有所体会，很多时候可能已经默认作为一个“常识”使用，而懒得思考背后的原理了。Google的这篇论文则为理解这个现象提供了一个可能的解释：适当而不是过小的学习率能为优化过程带来隐式的梯度惩罚项，有助于收敛到更平稳的区域。笔者认为其分析过程还是值得参考学习的。
