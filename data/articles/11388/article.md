# 流形上的最速下降：5\. 对偶梯度下降

> 作者：苏剑林 · 科学空间 · 2025-11-03
>
> 原文：<https://spaces.ac.cn/archives/11388>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前四篇文章我们求解了几个具体的给参数加等式约束的最速下降问题，其中第三、四篇的问题没法找到解析解，所以笔者提出了相应的不动点迭代法。其中的其中，第三篇文章[《流形上的最速下降：3\. Muon \+ Stiefel》](<https://spaces.ac.cn/archives/11221>)所研究的“Stiefel流形上的Muon”，问题提出自Jeremy Bernstein的[《Orthogonal manifold》](<https://docs.modula.systems/algorithms/manifold/orthogonal/>)一文。

对于这个问题，Jeremy Bernstein最后也给出了一个自己的解法，笔者称之为“对偶梯度下降（Dual Gradient Descent）”，也颇为值得学习一番。

## 基本概念

Jeremy Bernstein的解法，最后发表在Thinking Machines Lab的博客[《Modular Manifolds》](<https://thinkingmachines.ai/blog/modular-manifolds/>)中，是该实验室的第二篇博客，文章中将它称为“对偶上升（Dual Ascent）”，但笔者这里还是结合前四篇的内容，将其称为“对偶梯度下降”。

事实上，对偶梯度下降可以说是拉格朗日乘数法的自然结果，但是拉格朗日乘数法的严格讨论其实是很麻烦的，比如要引入[Minimax theorem](<https://en.wikipedia.org/wiki/Minimax_theorem>)，所以在这个系列中，我们为了避免这些麻烦，采用了“待定系数”这样的推导方式，这就使得对偶梯度下降并不是那么自然。但不要紧，我们还是可以沿着我们的思路把它推出来的，只不过可能多费点篇幅。

首先回顾一下各种记号。$\boldsymbol{W}\in\mathbb{R}^{n\times m}$是一个矩阵参数，不失一般性设$n\geq m$，$\boldsymbol{G}\in\mathbb{R}^{n\times m}$是它的梯度。$\Vert\boldsymbol{G}\Vert_2$是矩阵$\boldsymbol{G}$的谱范数，等于最大奇异值；$\Vert\boldsymbol{G}\Vert_*$是矩阵$\boldsymbol{G}$的核范数，等于全体奇异值之和。特别地，根据[《SVD的导数》](<https://spaces.ac.cn/archives/10878>)一文的结论，我们有
\begin{equation}\nabla_{\boldsymbol{G}}\Vert\boldsymbol{G}\Vert_* = \sum_i \nabla_{\boldsymbol{G}} \sigma_i = \sum_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} = \boldsymbol{U}\boldsymbol{V}^{\top} = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{G}) \label{eq:nuclear-grad}\end{equation}
其中$\boldsymbol{G}=\sum_i \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^{\top} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}$是$\boldsymbol{G}$的SVD。也就是说，核范数的梯度正好是$\msign$算子，这是接下来推导的重要基础。

## 问题描述

我们还是沿着之前的推导思路介绍对偶梯度下降，所以这一节先把问题和已有的结果复述一下。

在[《流形上的最速下降：3\. Muon \+ Stiefel》](<https://spaces.ac.cn/archives/11221>)中，我们要解决的问题是
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I},\,\,\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0} \label{eq:muon-stiefel}\end{equation}
解是$\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})$，其中$\boldsymbol{X}\in\mathbb{R}^{m\times m}$是待定的对称矩阵，使得$\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} = \boldsymbol{0}$。

在[《流形上的最速下降：4\. Muon \+ 谱球面》](<https://spaces.ac.cn/archives/11241>)中，我们要求解的问题是
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0 \label{eq:muon-spectral}\end{equation}
答案是$\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \lambda\boldsymbol{\Theta})$，其中$\lambda$是待定系数，使得$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$。

可以看到，我们的最终任务都变成了寻找待定系数，使它满足额外引入的等式约束，这本质上就是非线性方程（组）的求解。而对偶梯度下降，就是将方程的求解转化为某个目标函数的最小化，从而用梯度下降求解。

## 对偶目标

转化的关键就是核范数的梯度等式$\eqref{eq:nuclear-grad}$。简单起见，我们先看“Muon\+谱球面”的问题$\eqref{eq:muon-spectral}$，待定系数只是一个标量，比较好观察。不难验证
\begin{equation}\nabla_{\lambda} \Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_* = \tr(\boldsymbol{\Theta}^{\top}\msign(\boldsymbol{G} + \lambda\boldsymbol{\Theta})) = \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\end{equation}
这意味着解方程$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$等价于寻找让$\Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_*$的梯度等于0的点，这可能是它的（局部）最小/最大值点。由于$\Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_*$显然没有最大值，所以我们转化为寻找它的最小值点：
\begin{equation}\lambda^* = \newcommand{argmin}{\mathop{\text{argmin}}}\argmin_{\lambda} \Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_*\label{eq:muon-spectral-obj}\end{equation}

我们再来捋一捋这里的步骤：

> 1、我们的目标是解方程$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$，找到任意一个解就行；
> 
> 
> 
> 2、$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})$正好是$\Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_*$关于$\lambda$的梯度；
> 
> 
> 
> 3、这就转化为寻找（局部）最小/最大值点问题，因为此处的梯度往往是0；
> 
> 
> 
> 4、可以简单判断没有最大值，所以只能找最小值。

## 梯度下降

确定目标$\eqref{eq:muon-spectral-obj}$后，我们就可以用梯度下降求解了，其中梯度是现成的，即$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})$，那么梯度下降的格式是
\begin{equation}\lambda \quad \leftarrow\quad \lambda - \eta \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\end{equation}
当然我们也可以考虑给$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})$加个$\newcommand{sign}{\mathop{\text{sign}}}\sign$，即SignSGD，这些都可以自由发挥了。从迭代格式上看，对偶梯度下降要比之前我们提的不动点迭代要简单得多，然而，在很多情况下对偶梯度下降所需要的迭代步数也多得多，并且可能需要精调学习率、引入动量机制等，才有可能收敛。

所以，就解方程$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$而言，对偶梯度下降并不算特别理想的方案。但是，我们的最终目标并不是解方程$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$，而是计算$\boldsymbol{\Phi}$作为模型的优化方向。模型的优化本就是一个迭代的过程，我们可以将历史的$\lambda$缓存下来，然后采取$\lambda$与模型参数同步更新的近似策略
\begin{equation}\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \lambda\boldsymbol{\Theta}), \quad \boldsymbol{W}\leftarrow\boldsymbol{W}- \eta_1 \boldsymbol{\Phi},\quad \lambda \leftarrow\lambda - \eta_2 \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\end{equation}
这样一来，每一步训练只需要多算一步近乎免费的$\lambda - \eta_2 \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})$，就得到了原始目标$\eqref{eq:muon-spectral}$的一个近似实现。从形式上来说，它相当于Muon的一种自适应Weight Decay。

## Stiefel上

讨论完相对简单的“Muon\+谱球面”，我们再来看“Muon\+Stiefel”，即目标$\eqref{eq:muon-stiefel}$。此时待定矩阵$\boldsymbol{X}$有约束$\boldsymbol{X}=\boldsymbol{X}^{\top}$，我们通过设$\boldsymbol{X}=\boldsymbol{\Lambda}+\boldsymbol{\Lambda}^{\top}$来去掉约束，其中$\boldsymbol{\Lambda}\in\mathbb{R}^{m\times m}$是任意矩阵，然后可以发现
\begin{equation}\nabla_{\boldsymbol{\Lambda}}\Vert\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}\Vert_* = \boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W} \end{equation}
这里$\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X})$。所以，解方程组$\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W}=\boldsymbol{0}$同样可以转化为找函数的$\Vert\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}\Vert_*$的最小值点，然后用梯度下降解决：
\begin{equation}\boldsymbol{\Lambda} \quad\leftarrow\quad \boldsymbol{\Lambda} - \eta(\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W}) \end{equation}
由于$\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W}$必然是对称的，所以直接$\boldsymbol{X} \leftarrow\boldsymbol{X} - \eta(\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W})$也是可取的，将它跟$\boldsymbol{W}$放一起来同步迭代，我们得到
\begin{equation}\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \boldsymbol{W}\boldsymbol{X}), \quad \boldsymbol{W}\leftarrow\boldsymbol{W}- \eta_1 \boldsymbol{\Phi},\quad \boldsymbol{X} \leftarrow\boldsymbol{X} - \eta_2(\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W})\end{equation}
这样就实现了目标$\eqref{eq:muon-stiefel}$的一个近似，每一步多出来的$\boldsymbol{X} - \eta_2(\boldsymbol{W}^{\top}\boldsymbol{\Phi}+\boldsymbol{\Phi}^{\top}\boldsymbol{W})$也是近乎免费的。

## 拉氏乘数

在这两个例子中，它们所要求解的方程，都刚好等于某个核范数目标的梯度，这是单纯的巧合吗？当然不是，我们在“[基本概念](<https://spaces.ac.cn/archives/11388#%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5>)”一节就说了，这是拉格朗日乘数法的自然结果，这一节我们对此做一下展开讨论。

为了方便理解，我们还是以相对简单的目标$\eqref{eq:muon-spectral}$为例，它可以等价地写成
\begin{equation}\max_{\Vert\boldsymbol{\Phi}\Vert_2\leq 1} \min_{\lambda\in\mathbb{R}}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})  + \lambda\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\end{equation}
要理解这个转换，只需要意识到上式必然有$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$，否则$\min$这一步总可以取到负无穷，那么最后的$\max$结果也只能是负无穷；至于$\Vert\boldsymbol{\Phi}\Vert_2 = 1$改为$\Vert\boldsymbol{\Phi}\Vert_2\leq 1$，并不会改变最大值的结果（因为最大值总是在边界取到），但可以使得$\boldsymbol{\Phi}$的可行域变成一个[凸集](<https://en.wikipedia.org/wiki/Convex_set>)。

有了这个等价形式，我们就可以利用[Minimax theorem](<https://en.wikipedia.org/wiki/Minimax_theorem>)去交换$\min$和$\max$的位置：
\begin{equation}\begin{aligned}
&\,\max_{\Vert\boldsymbol{\Phi}\Vert_2\leq 1} \min_{\lambda\in\mathbb{R}}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})  + \lambda\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi}) \\
=&\, \min_{\lambda\in\mathbb{R}}\max_{\Vert\boldsymbol{\Phi}\Vert_2\leq 1}\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi})  + \lambda\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi}) \\
=&\, \min_{\lambda\in\mathbb{R}} \Vert\boldsymbol{G} + \lambda \boldsymbol{\Theta}\Vert_*
\end{aligned}\end{equation}
其中在$\Vert\boldsymbol{\Phi}\Vert_2\leq 1$上取$\max$这步，是Muon推导的[基本结果](<https://spaces.ac.cn/archives/11215#%E5%9F%BA%E6%9C%AC%E7%BB%93%E6%9E%9C>)，所以先求$\max$并没有困难。这样我们就得到了原问题$\eqref{eq:muon-spectral}$的对偶目标$\Vert\boldsymbol{G} + \lambda \boldsymbol{\Theta}\Vert_*$。

可能有些读者疑问：你这的拉格朗日乘数法，怎么跟我学的好像不一样？因为这里的拉格朗日乘数法推广到了一般凸集，并且严格讨论了$\min,\max$的可交换性，以保证最终结果是我们想要的。而我们一般学的拉格朗日乘数法，只是在$\mathbb{R}^n$中求约束优化问题的一套启发式求解流程，并没有太多去讨论理论保证方面的细节。

## 文章小结

这篇文章我们介绍了通过对偶梯度下降来寻找流形上的最速下降方向的思路，它也是前段时间Thinking Machines Lab的博客[《Modular Manifolds》](<https://thinkingmachines.ai/blog/modular-manifolds/>)用来求解Stiefel流形上的Muon的方法。
