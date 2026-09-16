# 初探MuP：超参数的跨模型尺度迁移规律

> 作者：苏剑林 · 科学空间 · 2025-03-13
>
> 原文：<https://spaces.ac.cn/archives/10770>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，完整训练一次大型LLM的成本是昂贵的，这就决定了我们不可能直接在大型LLM上反复测试超参数。一个很自然的想法是希望可以在同结构的小模型上仔细搜索超参数，找到最优组合后直接迁移到大模型上。尽管这个想法很朴素，但要实现它并不平凡，它需要我们了解常见的超参数与模型尺度之间的缩放规律，而MuP正是这个想法的一个实践。

MuP，有时也写$\mu P$，全名是Maximal Update Parametrization，出自论文[《Tensor Programs V: Tuning Large Neural Networks via Zero\-Shot Hyperparameter Transfer》](<https://papers.cool/arxiv/2203.03466>)，随着LLM训练的普及，它逐渐已经成为了科学炼丹的事实标配之一。

## 方法大意

在接入主题之前，必须先吐槽一下MuP原论文写得实在太过晦涩，并且结论的表达也不够清晰，平白增加了不少理解难度，所以接下来笔者尽量以一种（自认为）简明扼要的方式来复现MuP的结论。

先说结论，MuP主要研究超参数跨模型尺度的迁移规律。这里有几个关键词：

> 1、超参数，目前主要指<strong>学习率</strong>；
> 
> 
> 
> 2、模型尺度，目前主要是模型<strong>宽度</strong>；
> 
> 
> 
> 3、这里的核心是“<strong>迁移</strong>”。

请注意，MuP不研究什么是最优的超参数，只研究最优超参数<u>随着模型尺度的变化规律</u>，所以我们需要在某个小模型上搜索最优的超参数组合，然后迁移到大模型上，这就是MuP的使用场景和使用方法。

推导MuP的原理是让模型的<u>前向传播、反向传播、损失增量和特征变化</u>都不随模型尺度的变化而发生明显变化：

> 1、具体做法是分析初始化的数量级，然后认为结论可以代表后续优化的规律；
> 
> 
> 
> 2、说白了就是假设做好初始化，后面就会自动沿着正确的轨迹走（好的开始是成功的一大半？）;
> 
> 
> 
> 3、当然也可以给这个假设讲<strong>大数定律</strong>或<strong>中心极限定理</strong>的故事，但个人认为非必须。

## 前向传播

我们从前向传播开始讨论，因为这是相对简单且成熟的部分。首先，考虑线性层$\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{W}$，其中$\boldsymbol{X}\in\mathbb{R}^{b\times d_{in}},\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$。我们用RMS（Root Mean Square）来作为矩阵尺度的指标，例如
\begin{equation}\text{RMS}(\boldsymbol{W}) = \sqrt{\frac{1}{d_{in} d_{out}}\sum_{i=1}^{d_{in}} \sum_{j=1}^{d_{out}} W_{i,j}^2}\end{equation}

我们知道，要让初始化阶段$\boldsymbol{X}$的RMS跟$\boldsymbol{Y}$的RMS大致相等（简称“<strong>稳定</strong>”），那么$\boldsymbol{W}$要用：

> <strong>LeCun初始化</strong>：“均值为0、方差为$1/d_{in}$”的随机初始化。

这已经算是深度学习的基础结论之一，所以不再展开推导，还不大了解的读者可以参考以往的[《从几何视角来理解模型参数的初始化策略》](<https://spaces.ac.cn/archives/7180>)、[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620>)等博文。

接着，我们考虑非线性层$\boldsymbol{Y}=\phi(\boldsymbol{X}\boldsymbol{W})$，其中$\phi$是Element\-wise的激活函数。如果还是要维持$\boldsymbol{X}$的RMS跟$\boldsymbol{Y}$的RMS近似相等，那么结果会稍有不同，比如$\text{relu}$激活时我们得到

> <strong>Kaiming初始化</strong>：“均值为0、方差为$2/d_{in}$”的随机初始化。

容易看出，<strong>Kaiming初始化</strong>跟<strong>LeCun初始化</strong>相比，只是方差相差一个（跟模型尺度无关的）常数2，可以证明其他激活函数的结果也类似。所以我们可以下一个结论：

> <strong>fan\_in初始化</strong>：要保证前向传播的稳定性，那么应该要用“均值为0、方差<u>正比于</u>$1/d_{in}$”的随机初始化。

这个结论也可以理解为“激活函数的影响是模型尺度无关的”，所以如果我们只想分析模型尺度的效应，那么可以忽略（Element\-wise的）激活函数的存在，由LeCun初始化直接得到缩放规律$\propto 1/d_{in}$。

## 反向传播

现在我们继续分析反向传播（梯度），注意这里约定变量及其梯度具有相同的shape，那么可以算得
\begin{align}
\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}} =&\, \boldsymbol{X}^{\top}\left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\otimes \phi'(\boldsymbol{X}\boldsymbol{W})\right) \\[5pt]
\frac{\partial\mathcal{L}}{\partial \boldsymbol{X}} =&\, \left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\otimes \phi'(\boldsymbol{X}\boldsymbol{W})\right)\boldsymbol{W}^{\top}
\end{align}
第一个公式是当前层内参数的梯度，第二个公式则是该层往前传播的梯度，$\otimes$是Hadamard积，$\phi'$是$\phi$的导函数。

注意到一个事实：我们常用的激活函数，其导数都可以被一个（尺度无关的）常数给Bound住，所以至少在数量级上我们可以写出
\begin{align}
\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}} =&\, \boldsymbol{X}^{\top}\left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\otimes \phi'(\boldsymbol{X}\boldsymbol{W})\right) \sim \boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}} \label{eq:grad-w}\\[5pt]
\frac{\partial\mathcal{L}}{\partial \boldsymbol{X}} =&\, \left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\otimes \phi'(\boldsymbol{X}\boldsymbol{W})\right)\boldsymbol{W}^{\top}\sim \frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\boldsymbol{W}^{\top}\label{eq:grad-x}
\end{align}
我们先来看第二个公式，跟$\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{W}$相比，它右端乘的矩阵变成了$\boldsymbol{W}^{\top}$，那么按照上一节的结论，如果要保持反向传播的RMS稳定性，那么$\boldsymbol{W}$的初始化就应该是：

> <strong>fan\_out初始化</strong>：“均值为0、方差为$1/d_{out}$”的随机初始化。

当$d_{in}\neq d_{out}$时，前向传播和反向传播的要求就出现冲突，这时候有人提了一个折中策略：

> <strong>Xavier初始化</strong>：“均值为0、方差为$2/(d_{in} + d_{out})$”的随机初始化。

这也叫“<strong>fan\_avg初始化</strong>”，因为就是将$d_{in}$和$d_{out}$简单代数平均了一下，其他平均方式也可以考虑，参考[《初始化方法中非方阵的维度平均策略思考》](<https://spaces.ac.cn/archives/8725>)。Xavier初始化看上去同时兼顾了前向和反向，但也可以说两者都没兼顾，更好的办法是设计模型让大部分参数都是方阵，如后面讨论的模型簇$\eqref{eq:model}$。

## 损失增量

有了前向传播和反向传播的铺垫，我们就可以尝试分析损失函数的增量了。考虑$\boldsymbol{W}\to \boldsymbol{W} + \Delta\boldsymbol{W}$时损失函数的变化量
\begin{equation}\Delta \mathcal{L} = \mathcal{L}(\boldsymbol{W} + \Delta\boldsymbol{W}) - \mathcal{L}(\boldsymbol{W})\approx \left\langle\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}, \Delta\boldsymbol{W}\right\rangle_F\end{equation}
这里的$\langle\cdot,\cdot\rangle_F$是Frobenius内积，即把矩阵展平成向量后算向量内积。考虑梯度下降$\Delta\boldsymbol{W} = -\eta \frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}$，这里$\eta$自然是学习率，结合式$\eqref{eq:grad-w}$，我们有
$$\begin{equation}\Delta \mathcal{L}\approx -\eta\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_F^2\sim -\eta \left\Vert\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\right\Vert_F^2\end{equation}$$
事实上，这个式子已经告诉了我们同一个学习率$\eta$不能跨模型尺度使用的原因：

> 1、$\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}$是一个$d_{in}\times d_{out}$的矩阵；
> 
> 
> 
> 2、$\left\Vert\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\right\Vert_F^2$是$d_{in}\times d_{out}$个数的平方和；
> 
> 
> 
> 3、$\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}$正好是前向和反向的乘积；
> 
> 
> 
> 4、如果前向和反向都稳定，那么$\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}$每个元素都是$\Theta(1)$（$\Theta$是“[Big Theta Notation](<https://en.wikipedia.org/wiki/Big_O_notation#Family_of_Bachmann%E2%80%93Landau_notations>)”）；
> 
> 
> 
> 5、所以$\left\Vert\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}\right\Vert_F^2$就是$\Theta(d_{in} d_{out})$。

第4点可能要多加评述一下。$\boldsymbol{X}^{\top}$是一个$d_{in}\times b$矩阵，$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}$是一个$b\times d_{out}$矩阵，两者相乘就是$d_{in} d_{out}$个$b$维向量对做内积，内积是$b$项求和，而损失$\mathcal{L}$通常是对样本求平均（即包含了除以$b$操作），所以如果$\boldsymbol{X}^{\top}$和$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}$都是尺度无关的，那么它们乘起来基本也是尺度无关的【即RMS都是$\Theta(1)$】。

最后的结论表明，如果我们直接将小模型的学习率用于大模型，那么对于足够大的模型，它的每一步损失增量就会随着参数尺度（即$d_{in} d_{out}$）的变大而<u><strong>爆炸</strong></u>，这意味着没法复制小模型的收敛过程，甚至可能因为步子迈得太大导致无法收敛。

此时大家可能想到的一个做法是让$\eta\propto 1/(d_{in} d_{out})$来缩放$\Delta\mathcal{L}$，事实上这个想法已经跟上了MuP的思路，但实际场景中由于前面说的前向和反向的不兼容性，导致第4点“如果前向和反向都稳定，那么$\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}}$每个元素就是$\Theta(1)$”不能总是成立，所以实际情况更为复杂一些，

## 模型假设

现在让我们考虑一个更接近实践的场景。我们的任务是训练一个$\mathbb{R}^{d_{in}}\mapsto \mathbb{R}^{d_{out}}$的模型，其中$d_{in},d_{out}$是数据决定的，不可改变。开头我们就说了，MuP旨在研究超参数随着模型尺度的缩放规律，所以一切固定不变的量，都相当于是常数或者说$\Theta(1)$，比如初始化方差为$1/d_{in}$，等价于说初始化方差为$\Theta(1)$。

我们可以改变的是模型的架构、参数量等部分，但MuP主要考虑宽度的规律，所以我们把模型的架构定一下。这里主要考虑的模型簇是：
\begin{equation}\begin{gathered}
\boldsymbol{Y}_{in} = \boldsymbol{X} \boldsymbol{W}_{in} \\[5pt]
\boldsymbol{Y}_{out} = \text{NN}(\boldsymbol{Y}_{in},\boldsymbol{\Omega}) \\[5pt]
\boldsymbol{Z} = \boldsymbol{Y}_{out} \boldsymbol{W}_{out}
\end{gathered}\label{eq:model}\end{equation}

其中：

> 1、$\boldsymbol{X}\in\mathbb{R}^{b\times d_{in}}$（带上了batch size）；
> 
> 
> 
> 2、$\boldsymbol{W}_{in} \in \mathbb{R}^{d_{in}\times d}, \boldsymbol{W}_{out} \in \mathbb{R}^{d\times d_{out}}$；
> 
> 
> 
> 3、$\text{NN}$是任意$\mathbb{R}^d\mapsto \mathbb{R}^d$的神经网络；
> 
> 
> 
> 4、这里$d$其实就是我们常说的hidden size；
> 
> 
> 
> 5、我们可以随意调大$d$，来提升模型的参数量和潜力；
> 
> 
> 
> 6、MuP就是想研究超参数关于$d$的变化规律。

更具体一点，这里我们考虑的$\text{NN}$是$K$层MLP：
\begin{equation}\begin{aligned}
\boldsymbol{Y}_0 =&\, Y_{in} \\[5pt]
\boldsymbol{Y}_{k+1} =&\, \phi(\boldsymbol{Y}_k \boldsymbol{W}_{k+1}) \\[5pt]
\boldsymbol{Y}_{out} =&\, \boldsymbol{Y}_K
\end{aligned}\end{equation}
这里$\boldsymbol{\Omega}=\{\boldsymbol{W}_1,\boldsymbol{W}_2,\cdots,\boldsymbol{W}_K\}$，$\boldsymbol{W}_k\in\mathbb{R}^{d\times d}$，即都是$d\times d$的<strong>方阵</strong>，全都用<strong>fan\_in初始化</strong>（等价地，也是<strong>fan\_out初始化</strong>）。

补充一下，这里约定所有参数矩阵都是$d\times d$方阵，纯粹是为了简化分析，并不是强制要求。因为这里真正的目的是假设$\text{NN}$的参数里<u>没有尺度无关的形状</u>，比如不允许$d\times 64$这样的形状，因为$64$是一个常数，但$d\times 4d$这样的形状是允许的，因为你不管fan\_in、fan\_out或fan\_avg初始化，方差都是正比于$1/d$。

## 组装起来

确立后具体模型后，我们就可以把前面的结论都组装起来了。要更新的参数分为$\boldsymbol{W}_{in},\boldsymbol{\Omega},\boldsymbol{W}_{out}$三部分，分别求梯度：
\begin{align}
\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}} =&\, \boldsymbol{Y}_{out}^{\top}\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}} \\[6pt]
\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k} =&\, \frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k} \cdot\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}_{out}} = \frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k} \cdot\left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}\boldsymbol{W}_{out}^{\top}\right) \\[6pt]
\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}} =&\, \boldsymbol{X}^{\top} \frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}_{in}} = \boldsymbol{X}^{\top} \left(\frac{\partial\boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}\cdot\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}_{out}}\right) = \boldsymbol{X}^{\top} \left(\frac{\partial\boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}\cdot\left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}\boldsymbol{W}_{out}^{\top}\right)\right) \\[6pt]
\end{align}

这里的$\cdot$运算需要稍微解释一下：$\boldsymbol{Y}_{in},\boldsymbol{Y}_{out}$都是一个矩阵，所以$\frac{\partial\boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}$原则上是一个四阶张量，链式法则$\frac{\partial\boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}\cdot\frac{\partial\mathcal{L}}{\partial \boldsymbol{Y}_{out}}$实际是高阶张量的乘法，但这里不打算展开介绍了，所以简单用一个$\cdot$代替，读者只需要知道它是矩阵乘法的一般推广就行。

现在来观察规律：

> 1、三个式子都有$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}$；
> 
> 
> 
> 2、后两式都有$\boldsymbol{W}_{out}^{\top}$；
> 
> 
> 
> 3、$\boldsymbol{W}_k$里都是方阵，$\frac{\partial\boldsymbol{Y}_{out}}{\partial \boldsymbol{Y}_{in}}$和$\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k}$都是稳定的【RMS是$\Theta(1)$】；
> 
> 
> 
> 4、如果$\boldsymbol{W}_{in}$也用fan\_in初始化，那么$\boldsymbol{Y}_{out}$也是稳定的；
> 
> 
> 
> 5、要想$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}\boldsymbol{W}_{out}^{\top}$稳定，那么初始化方差是$1/d_{out}$，但$d_{out}$是尺度无关的，相当于常数。

这样一来：

> 1、$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}}$的RMS是$\Theta(1)$，$\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}}\right\Vert_F^2$是$d\times d_{out}$个数平方和，所以大小是$\Theta(d\times d_{out})$，别忘了$d_{out}$是常数，所以实际上就是$\Theta(d)$，于是为了得到$\Theta(1)$的$\Delta\mathcal{L}$，它的学习率要满足$\eta_{out}\propto 1/d$；
> 
> 
> 
> 2、$\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}\right\Vert_F^2$是$d^2$个数求和，$\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k}$和$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}$的RMS都是$\Theta(1)$，我们直接将$\boldsymbol{W}_{out}$的初始化方差设为$\propto 1/d^2$，那么$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}$的RMS就是$\Theta(1/d)$，平方求和后就正好是$\Theta(1)$，因此学习率不用变化；
> 
> 
> 
> 3、此时$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}}$的RMS也是$\Theta(1/d)$，但$\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\Vert_F^2$只是$d_{in}\times d$个数平方和，所以结果是$\Theta(1/d)$的，为了得到$\Theta(1)$的$\Delta\mathcal{L}$，学习率反而需要放大$d$倍来抵消这个影响，即$\eta_{in}\propto d$。

## 特征变化

以上结果是没有问题的，但仔细思考我们会发现推导过程的一个问题：上面的第2、3点，都建立在“我们直接将$\boldsymbol{W}_{out}$的初始化方差设为$\propto 1/d^2$”这个设置上，然而这个设置目前来说并没有直接的依据。如果不对此进一步解释，那么推导过程还是不够完备的。

事实上，单看$\Delta \mathcal{L}=\Theta(1)$这个要求的话，确实是无法排除其他选择的可能性的，比如$\boldsymbol{W}_{out}$的初始化方差设为$\propto 1/d$，此时$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}$的RMS是$\Theta(1/\sqrt{d})$，平方求和后是$\Theta(d)$，那么只要学习率$\eta\propto 1/d$同样可以实现$\Delta \mathcal{L}=\Theta(1)$。因此，为了解释“$\boldsymbol{W}_{out}$的初始化方差设为$\propto 1/d^2$”的必要性，那么就需要引入新的条件。

损失函数$\mathcal{L}$是模型的一个宏观指标，或者说外部指标，单看它的变化已经不足以解释全部结果了，那么就需要细化到模型内部了。具体来说，我们希望模型每一层的输出（通常也称为特征，有时也称激活值）变化量也具有尺度不变性。比如线性层$\boldsymbol{Y}_k = \boldsymbol{Y}_{k-1} \boldsymbol{W}_k$，参数$\boldsymbol{W}_k\to \boldsymbol{W}_k + \Delta \boldsymbol{W}_k$带来的输出变化是
\begin{equation}\Delta\boldsymbol{Y}_k = \boldsymbol{Y}_{k-1} (\boldsymbol{W}_k + \Delta \boldsymbol{W}_k) - \boldsymbol{Y}_{k-1} \boldsymbol{W}_k = \boldsymbol{Y}_{k-1} \Delta\boldsymbol{W}_k\end{equation}
注意$\boldsymbol{Y}_{k-1}\in\mathbb{R}^{b\times d},\Delta\boldsymbol{W}_k\in\mathbb{R}^{d\times d}$，所以$\boldsymbol{Y}_{k-1} \Delta\boldsymbol{W}_k$就是$b\times d$个$d$维向量对的内积。注意这里$\Delta\boldsymbol{W}_k$是精心设计的更新量，它不大可能跟初始化那样跟$\boldsymbol{Y}_{k-1}$是独立的，所以“$d$维向量对的内积”更有可能是$\Theta(d)$（$d$维内积共有$d$项求和），因此如果$\Delta\boldsymbol{Y}_{k-1}$的RMS是$\Theta(1)$，那么可以认为$\Delta\boldsymbol{Y}_k$的RMS将是$\Theta(d\times \text{RMS}(\Delta \boldsymbol{W}_k))$。

于是，为了让$\Delta\boldsymbol{Y}_k$的RMS是$\Theta(1)$，我们得到了对$\Delta \boldsymbol{W}_k$的一个额外要求：
\begin{equation}\text{RMS}(\Delta \boldsymbol{W}_k) = \Theta(1 / d)\label{eq:dw-rms}\end{equation}

结合$\Delta \boldsymbol{W}_k = -\eta\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}$和$\Delta\mathcal{L}=\Theta(1)$，我们就可以得到“$\boldsymbol{W}_{out}$的初始化方差设为$\propto 1/d^2$”的结果。

（注：这一节依赖于 [@Chenyu Zheng](<https://spaces.ac.cn/archives/10770/comment-page-1#comment-27212>) 的指点，非常感谢！）

## Adam版本

以上就是SGD的MuP，对于Adam，我们通常用SignSGD近似做数量级分析：

> 1、$\Delta \boldsymbol{W} = -\eta \mathop{\text{sign}}\left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right)$；
> 
> 
> 
> 2、$\Delta \mathcal{L} \approx -\eta \left|\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right|_1$；
> 
> 
> 
> 3、这里的$|\cdot|_1$指每个元素取绝对值然后求和。

关于SignSGD近似本身，读者还可以参考[《当Batch Size增大时，学习率该如何随之变化？》](<https://spaces.ac.cn/archives/10542>)、[《Adam的epsilon如何影响学习率的Scaling Law？》](<https://spaces.ac.cn/archives/10563>)等文章，这里也不展开讨论了。总而言之，SignSGD是分析Adam相关缩放规律时一个常用的近似方式。

现在可以模仿SGD的过程进行分析：

> 1、$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}}$的RMS是$\Theta(1)$，$\left|\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}}\right|_1$是$d\times d_{out}$个数求和，大小是$\Theta(d\times d_{out}) = \Theta(d)$，所以它的学习率要满足$\eta_{out}\propto 1/d$来抵消尺度影响；
> 
> 
> 
> 2、$\left|\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}\right|_1$是$d^2$个数求和，$\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k}$和$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}$的RMS都是$\Theta(1)$，我们将$\boldsymbol{W}_{out}$的初始方差设为$\propto 1/d^2$，那么$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}$的RMS就是$\Theta(1/d)$，$d^2$个数求和后是$\Theta(d)$，所以学习率按照$\eta_k\propto 1/d$变换来抵消尺度影响；
> 
> 
> 
> 3、此时$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}}$的RMS也是$\Theta(1/d)$，但$\left|\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}}\right|_1$只是$d_{in}\times d$个数求和，所以它已经是$\Theta(1)$，从而学习率不用随尺度改变。

（注：读者可以自行检查一下式$\eqref{eq:dw-rms}$是满足的。）

## Muon版本

接下来自然少不了Muon的分析。对于Muon本身，我们已经在[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)、[《Muon续集：为什么我们选择尝试Muon？》](<https://spaces.ac.cn/archives/10739>)做了详细介绍，这里不再重复。跟Adam用SignSGD类似，我们用MSignSGD来近似Muon：

> 1、$\Delta \boldsymbol{W} = -\eta \mathop{\text{msign}}\left(\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right)$；
> 
> 
> 
> 2、$\Delta \mathcal{L} \approx -\eta \left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_*$（证明见[《Muon优化器赏析：从向量到矩阵的本质跨越》](<https://spaces.ac.cn/archives/10592>)）；
> 
> 
> 
> 3、这里的$\Vert\cdot\Vert_*$指[Nuclear范数](<https://en.wikipedia.org/wiki/Nuclear_norm>)，是矩阵的<u>所有奇异值之和</u>；
> 
> 
> 
> 4、Nuclear范数并不好算，但$F$范数好算，它等于矩阵的<u>所有奇异值的平方和的平方根</u>；
> 
> 
> 
> 5、我们用$F$范数作为Nuclear范数近似，因此$\Delta \mathcal{L} \approx -\eta \left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_*\approx -\eta \left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}}\right\Vert_F$；
> 
> 
> 
> 6、$F$范数又等于矩阵的<u>所有元素的平方和的平方根</u>。

那么可以开始分析过程：

> 1、$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}}$的RMS是$\Theta(1)$，所以$\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{out}}\right\Vert_*$大小是$\Theta(\sqrt{d\times d_{out}}) = \Theta(\sqrt{d})$，要消除尺度的影响，那么它的学习率要满足$\eta_{out}\propto 1/\sqrt{d}$；
> 
> 
> 
> 2、$\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}\right\Vert_F$是$d^2$个数的平方和的平方根，$\frac{\partial \boldsymbol{Y}_{out}}{\partial \boldsymbol{W}_k}$和$\frac{\partial\mathcal{L}}{\partial \boldsymbol{Z}}$的RMS都是$\Theta(1)$，我们将$\boldsymbol{W}_{out}$的初始方差设为$\propto 1/d^2$，那么$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_k}$的RMS就是$\Theta(1/d)$，平方和后再平方根，结果是$\Theta(1)$，所以学习率不用变；
> 
> 
> 
> 3、此时$\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}}$的RMS也是$\Theta(1/d)$，但$\left\Vert\frac{\partial\mathcal{L}}{\partial \boldsymbol{W}_{in}}\right\Vert_F$只是$d_{in}\times d$个数的平方和平方根，所以它是$\Theta(1/\sqrt{d})$的，学习率反而需要放大$\sqrt{d}$倍来抵消这个影响，即$\eta_{in}\propto \sqrt{d}$。

（注：这里Muon的结论是对的，但它不满足条件$\eqref{eq:dw-rms}$，因为式$\eqref{eq:dw-rms}$要细说的话还依赖于一个更新量是Element\-wise的假设，而Muon不符合这个假设，所以实际上不可用。这里没有仔细展开相关讨论，而是直接沿用了“$\boldsymbol{W}_{out}$的初始化方差设为$\propto 1/d^2$”的结论，回避了式$\eqref{eq:dw-rms}$。）

## 结论汇总

将上述结论汇总在一起是：
\begin{array}{c|c|c|c|c|c|c}
\hline
& \boldsymbol{W}_{in}\text{方差} & \boldsymbol{W}_{in}\text{学习率} & \boldsymbol{W}_k\text{方差} & \boldsymbol{W}_k\text{学习率} & \boldsymbol{W}_{out}\text{方差} & \boldsymbol{W}_{out}\text{学习率} \\
\hline
\text{SGD} & 1/d_{in} & d & 1 / d & 1 & 1/d^2 & 1 / d\\
\text{Adam} & 1/d_{in} & 1 & 1 / d & 1 / d & 1/d^2 & 1 / d\\
\text{Muon} & 1/d_{in} & \sqrt{d} & 1 / d & 1 & 1/d^2 & 1 / \sqrt{d} \\
\hline
\end{array}

这里的$\boldsymbol{W}_k$指的是除$\boldsymbol{W}_{in},\boldsymbol{W}_{out}$外的所有参数，还有要强调的是，这里的关系都是“正比于”而不是“等于”。另外实践中可以根据具体需求稍作变化，比如实际我们用Muon时，$\boldsymbol{W}_{in}$和$\boldsymbol{W}_{out}$的优化通常不用Muon而是用Adam，这将导致两个变化：

> 1、$\eta_{out}\propto 1/d$；
> 
> 
> 
> 2、$\eta_{in}$不变。

如果结合我们在[《Muon is Scalable for LLM Training》](<https://papers.cool/arxiv/2502.16982>)所提的Adujst LR的话，那么学习率要多乘一个$\sqrt{\max(n, m)}$，$n\times m$是参数矩阵的形状，我们已经假设了$\text{NN}$部分的参数总等比例缩放，所以$\sqrt{\max(n, m)}\propto \sqrt{d}$。因此，如果要抵消Adujst LR带来的尺度影响，那么就需要

> 3、$\eta_k\propto 1/\sqrt{d}$ 。

## 文章小结

本文以尽可能简明清晰的方式介绍了MuP（Maximal Update Parametrization），这是旨在研究超参数跨模型尺度的迁移规律的工作。基于MuP，我们可以在小模型上以相对较小的成本仔细搜索超参数（这里主要是学习率和初始化），然后迁移到大模型上，降低大模型的炼丹成本。

客观来讲，这里的介绍和分析还比较初步，比如没有考虑Bias项、没有评估结论在MLP以外架构的通用性、也没有仔细考虑Normalization和残差的作用等。没有考虑Bias项这个单纯是偷懒，权当留给读者的习题了；至于不同架构下的MuP，一般分析起来比较麻烦，但由于神经网络的相似性，结论大致上是相同的，我们可以不加证明地用着。个人认为比较关键的改进点是Normalization和残差的影响，尤其是Normalization，它使得不依赖特殊的初始化就可以稳定前向传播，带来了更大的自由度和可能性。

当然，这些都留给后续分析了。
