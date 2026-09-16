# 从几何视角来理解模型参数的初始化策略

> 作者：苏剑林 · 科学空间 · 2020-01-16
>
> 原文：<https://spaces.ac.cn/archives/7180>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

对于复杂模型来说，参数的初始化显得尤为重要。糟糕的初始化，很多时候已经不单是模型效果变差的问题了，还更有可能是模型根本训练不动或者不收敛。在深度学习中常见的自适应初始化策略是Xavier初始化，它是从正态分布$\mathcal{N}\left(0,\frac{2}{fan_{in} + fan_{out}}\right)$中随机采样而构成的初始权重，其中$fan_{in}$是输入的维度而$fan_{out}$是输出的维度。其他初始化策略基本上也类似，只不过假设有所不同，导致最终形式略有差别。

标准的初始化策略的推导是基于概率统计的，大概的思路是<u>假设输入数据的均值为0、方差为1，然后期望输出数据也保持均值为0、方差为1，然后推导出初始变换应该满足的均值和方差条件</u>。这个过程理论上没啥问题，但在笔者看来依然不够直观，而且推导过程的假设有点多。本文则希望能从<strong>几何视角</strong>来理解模型的初始化方法，给出一个更直观的推导过程。

## 信手拈来的正交

前者时间笔者写了[《n维空间下两个随机向量的夹角分布》](<https://spaces.ac.cn/archives/7076>)，其中的一个推论是

> <strong>推论1</strong>： 高维空间中的任意两个随机向量几乎都是垂直的。

事实上，推论1正是本文整个几何视角的出发点！它的一个更进一步的推论是：

> <strong>推论2</strong>： 从$\mathcal{N}(0, 1/n)$中随机选取$n^2$个数，组成一个$n\times n$的矩阵，这个矩阵近似为正交矩阵，且$n$越大，近似程度越好。

不信的读者也可以数值验证一下：



<pre><code class="language-python">import numpy as np

n = 100
W = np.random.randn(n, n) / np.sqrt(n)
X = np.dot(W.T, W)  # 矩阵乘以自身的转置
print(X)  # 看看是否接近单位阵
print(np.square(X - np.eye(n)).mean())  # 计算与单位阵的mse
</code></pre>



相信对于多数读者来说，第一次看到这个推论2或多或少都会觉得惊讶。正交矩阵是指满足$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$的矩阵，也就是说它的逆等于转置。一般矩阵的逆和转置的求解难度差得不是一点点，所以给我们的感觉是“逆=转置”是一个很苛刻的条件才对，但推论2却告诉我们随机采样而来的矩阵就已经接近正交矩阵了，不得不说有点反直觉。当初笔者刚意识到这一点时，也是感觉挺惊讶的。

## 其实也没那么难理解

不过，当我们习惯了推论1“高维空间中的任意两个随机向量几乎都是垂直的”这个事实后，我们确实可以很快地理解并导出这个结果。快速推导的时候，我们可以先考虑标准正态分布$\mathcal{N}(0,1)$，注意到推论1要求采样的方向均匀，而标准正态分布正好满足这个要求。从$\mathcal{N}(0,1)$中采样到一个$n\times n$的矩阵，我们可以把它看成是$n$个$n$维向量，那既然这$n$个向量都是随机向量，所以它们两两之间自然就接近正交了。

当然，两两正交还不是正交矩阵，因为正交矩阵还要求每个向量的模长为1，而我们有$\mathbb{E}_{x\sim \mathcal{N}(0,1)}\left[x^2\right]=1$，所以这意味着从$\mathcal{N}(0,1)$中采样出的$n$维向量模长近似为$\sqrt{n}$，所以为了接近正交，还需要将每个元素除以$\sqrt{n}$，这等价于采样方差由1变成了$1/n$。

此外，采样分布还不一定要是正态分布，比如均匀分布$U\left[-\sqrt{3/n}, \sqrt{3/n}\right]$也行。事实上我们有

> <strong>推论3</strong>： 从任意的均值为0、方差为$1/n$的分布$p(x)$中独立重复采样出来的$n\times n$矩阵，都接近正交矩阵。

我们可以从一个更数学化的角度来理解推论3：假设$\boldsymbol{x}=(x_1,x_2,\dots,x_n),\boldsymbol{y}=(y_1,y_2,\dots,y_n)$都是从$p(x)$中采样出来的，那么有
\begin{equation}\begin{aligned}\langle \boldsymbol{x}, \boldsymbol{y}\rangle =&\, n\times \frac{1}{n}\sum_{k=1}^n x_k y_k\\
\approx&\, n\times \mathbb{E}_{x\sim p(x),y\sim p(x)}[xy]\\
=&\, n\times \mathbb{E}_{x\sim p(x)}[x]\times \mathbb{E}_{y\sim p(x)}[y]\\
=&\,0\end{aligned}\end{equation}
以及
\begin{equation}\begin{aligned}\Vert\boldsymbol{x}\Vert^2 =&\, n\times \frac{1}{n}\sum_{k=1}^n x_k^2\\
\approx&\, n\times \mathbb{E}_{x\sim p(x)}\left[x^2\right]\\
=&\, n\times \left(\mu^2 + \sigma^2\right)\\
=&\,1\end{aligned}\end{equation}
所以任意两个向量都是接近正交归一的，因此采样出来的矩阵也接近正交矩阵。

## 现在可以说初始化了

说了那么多正交矩阵的内容，其实本质上都是为理解初始化方法的几何意义做铺垫。如果读者对线性代数还有印象的话，那么应该还记得正交矩阵的重要意义在于它在变换过程中保持了向量的模长不变。用数学公式来表达，就是设$\boldsymbol{W}\in \mathbb{R}^{n\times n}$是一个正交矩阵，而$\boldsymbol{x}\in\mathbb{R}^n$是任意向量，则$\boldsymbol{x}$的模长等于$\boldsymbol{W}\boldsymbol{x}$的模长：
\begin{equation}\Vert\boldsymbol{W}\boldsymbol{x}\Vert^2 = \boldsymbol{x}^{\top}\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{x}=\boldsymbol{x}^{\top}\boldsymbol{x}=\Vert\boldsymbol{x}\Vert^2\end{equation}
考虑全连接层：
\begin{equation}\boldsymbol{y}=\boldsymbol{W}\boldsymbol{x} + \boldsymbol{b}\end{equation}
深度学习模型本身上就是一个个全连接层的嵌套，所以为了使模型最后的输出不至于在初始化阶段就过于“膨胀”或者“退化”，一个想法就是让模型在初始化时能保持模长不变。

这个想法形成的一个自然的初始化策略就是“以全零初始化$\boldsymbol{b}$，以随机正交矩阵初始化$\boldsymbol{W}$”。而推论2就已经告诉我们，从$\mathcal{N}(0, 1/n)$采样而来的$n\times n$矩阵就已经接近正交矩阵了，所以我们可以从$\mathcal{N}(0, 1/n)$采样来初始化$\boldsymbol{W}$。这便是Xavier初始化策略了，有些框架也叫Glorot初始化，因为作者叫Xavier Glorot～此外，采样分布也不一定是$\mathcal{N}(0, 1/n)$，前面推论3说了你可以从任意均值为0、方差为$1/n$的分布中采样。

上面说的是输入和输出维度都是$n$的情况，如果输入是$n$维，输出是$m$维呢？这时候$\boldsymbol{W}\in\mathbb{R}^{m\times n}$，保持$\boldsymbol{W}\boldsymbol{x}$模长不变的条件依然是$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$。然而，当$m < n$时，这是不可能的；当$m \geq n$时，这是有可能成立的，并且根据前面相似的推导，我们可以得到

> <strong>推论4</strong>： 当$m \geq n$时，从任意的均值为0、方差为$1/m$的分布$p(x)$中独立重复采样出来的$m\times n$矩阵，近似满足$\boldsymbol{W}^{\top}\boldsymbol{W}=\boldsymbol{I}$。

所以，如果$m > n$，那么只需要把采样分布的方差改为$1/m$就好，至于$m < n$时，虽然没有直接的推导，但仍然可以沿用这个做法，毕竟合理的策略应该是普适的。注意，这个改动跟Xavier初始化的原始设计有点不一样，它是“LeCun初始化”的对偶版本（LeCun初始化方差是$1/n$），而Xavier初始化的方差则是$2/(m+n)$，这平均了前向传播和反向传播的直觉做法，而我们这里主要考虑的是前向传播。

可能还会有读者疑问：你这里只是考虑了没有激活函数的场景，就算$\boldsymbol{y}$的模长跟$\boldsymbol{x}$一样，但$\boldsymbol{y}$经过激活函数后就不一样了。确实是存在这样的情况，而且这时候只能针对具体问题具体分析。比如$\tanh(x)$在$x$比较小的时候有$\tanh(x)\approx x$，所以可以认为Xavier初始化直接适用于$\tanh$激活；再比如$\text{relu}$时可以认为$\text{relu}(\boldsymbol{y})$会有大约一半的元素被置零，所以模长大约变为原来的$1/\sqrt{2}$，而要保持模长不变，可以让$\boldsymbol{W}$乘上$\sqrt{2}$，也就是说初始化方差从$1/m$变成$2/m$，这就是何恺明大神提出来的针对$\text{relu}$的初始化策略。

当然，事实上很难针对每一个激活函数都做好方差的调整，所以一个更通用的做法就是直接在激活函数后面加上一个类似Layer Normalization的操作，直接显式地恢复模长。这时候就轮到各种Normalization技巧登场了～（欢迎继续阅读旧作[《BN究竟起了什么作用？一个闭门造车的分析》](<https://spaces.ac.cn/archives/6992>)。）

## 还有一点小结

本文主要是从“高维空间中的任意两个随机向量几乎都是垂直的”这个结论推导出“均值为0、方差为$1/n$的任意$n\times n$矩阵接近于正交矩阵”，继而给出了相关初始化策略的一个几何视角。窃以为这个几何视角相比纯统计视角要更直观易懂一些。
