# 通过梯度近似寻找Normalization的替代品

> 作者：苏剑林 · 科学空间 · 2025-04-02
>
> 原文：<https://spaces.ac.cn/archives/10831>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

不知道大家有没有留意到前段时间的[《Transformers without Normalization》](<https://papers.cool/arxiv/2503.10622>)？这篇论文试图将Transformer模型中的Normalization层用一个Element\-wise的运算DyT替代，以期能提高速度并保持效果。这种基础架构的主题本身自带一点吸引力，加之Kaiming He和Yann LeCun两位大佬挂名，所以这篇论文发布之时就引起了不少围观，评价也是有褒有贬。

无独有偶，上周的一篇新论文[《The Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions》](<https://papers.cool/arxiv/2503.21708>)从梯度分析和微分方程的视角解读了DyT，并提出了新的替代品。个人感觉这个理解角度非常本质，遂学习和分享一波。

## 写在前面

DyT全称是Dynamic Tanh，它通过如下运算来替代Normalization层：
\begin{equation}\mathop{\text{DyT}}(\boldsymbol{x}) = \boldsymbol{\gamma} \odot \tanh(\alpha \boldsymbol{x}) + \boldsymbol{\beta}\end{equation}
其中$\alpha,\boldsymbol{\beta},\boldsymbol{\gamma}$都是可学参数，$\boldsymbol{\beta},\boldsymbol{\gamma}$是Normalization层本来就有的，所以这里的关键是用$\tanh(\alpha \boldsymbol{x})$替代了Normalize运算。$\tanh$是逐元素的运算，免除了均值、方差这两个统计量的计算。

关于DyT，笔者曾在知乎[《如何评价 Meta 新论文 Transformers without Normalization？》](<https://www.zhihu.com/question/14925347536/answer/124434065689>)发表过一些看法，简单来说就是不大看好。理由是Normalization无脑地稳定了模型的前向传播，那么就留了更多的自由度和可能性给模型的其他方面（比如效果），所以笔者不认为比有Normalization更简化的通用操作能实现更好的效果（No Free Lunch）。

事实上早在2021年的[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620#%E6%AE%8B%E5%B7%AE%E8%BF%9E%E6%8E%A5>)我们就讨论过去掉Normalization这个话题，相关工作有[SkipInit](<https://papers.cool/arxiv/2002.10444>)、[ReZero](<https://papers.cool/arxiv/2003.04887>)、[Fixup](<https://papers.cool/arxiv/1901.09321>)等。当时笔者试了一些方案，发现它们即便在某些方面能够追平Normalization，但仍会存在另一些方面的不足，比如预训练效果尚可，但微调效果较差等，所以就没再深究下去了。

因此，笔者现在对类似工作都只视为简化维度上的极限探索来欣赏，正如[《nGPT: Normalized Transformer with Representation Learning on the Hypersphere》](<https://papers.cool/arxiv/2410.01131>)几乎将每一处能Normalize的地方都加上Normalize一样，都属于某个方向的极限探索。

## 梯度计算

当然，不看好归不看好，不妨碍我们的学习和分析。要想寻找Normalization的替代或者说近似，最直接的思路就是从梯度入手，因为深度学习说到底也就是前向传播和反向传播那点事，反向传播也就是求梯度，往往扮演着比较本质的角色。

接下来我们只考虑RMS Norm，它的关键运算是
\begin{equation}\boldsymbol{y} = \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert_{RMS}} = \sqrt{d}\times \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert}\label{eq:rms-norm}\end{equation}
其中$\boldsymbol{x}\in\mathbb{R}^d$，以及
\begin{equation}\Vert\boldsymbol{x}\Vert_{RMS} = \frac{\Vert\boldsymbol{x}\Vert}{\sqrt{d}},\qquad \Vert\boldsymbol{x}\Vert = \sqrt{\boldsymbol{x}^2} =  \sqrt{\sum_{i=1}^d x_i^2}\end{equation}
所以要求$\boldsymbol{x} / \Vert\boldsymbol{x}\Vert_{RMS}$的梯度，等价于求$\boldsymbol{x} / \Vert\boldsymbol{x}\Vert$的梯度，我们可以通过如下方式计算：
\begin{equation}\frac{\boldsymbol{x}+\Delta\boldsymbol{x}}{\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert} = \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert} + \frac{\Delta\boldsymbol{x}}{\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert} \approx \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert} + \frac{\Delta\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert}\label{eq:exp-1}\end{equation}
比较复杂的地方是展开$\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert = \sqrt{(\boldsymbol{x}+\Delta\boldsymbol{x})^2}$：
\begin{equation}\begin{aligned}
&\,\sqrt{(\boldsymbol{x}+\Delta\boldsymbol{x})^2} \\
\approx&\, \sqrt{\Vert\boldsymbol{x}\Vert^2+2\boldsymbol{x}\cdot\Delta\boldsymbol{x}} \\
=&\, \Vert\boldsymbol{x}\Vert\sqrt{1+2\boldsymbol{x}\cdot\Delta\boldsymbol{x}/\Vert\boldsymbol{x}\Vert^2} \\
=&\, \Vert\boldsymbol{x}\Vert (1+\boldsymbol{x}\cdot\Delta\boldsymbol{x}/\Vert\boldsymbol{x}\Vert^2)
\end{aligned} \quad \Rightarrow \quad
\begin{aligned}
\frac{\boldsymbol{x}}{\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert} \approx&\, \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert}(1-\boldsymbol{x}\cdot\Delta\boldsymbol{x}/\Vert\boldsymbol{x}\Vert^2)
\end{aligned}\end{equation}
代入式$\eqref{eq:exp-1}$得：
\begin{equation}\frac{\boldsymbol{x}+\Delta\boldsymbol{x}}{\Vert\boldsymbol{x}+\Delta\boldsymbol{x}\Vert} - \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert} \approx \frac{\Delta\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert} - \frac{(\boldsymbol{x}\cdot\Delta\boldsymbol{x})\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert^3}\quad\Rightarrow\quad\nabla_{\boldsymbol{x}} \frac{\boldsymbol{x}}{\Vert\boldsymbol{x}\Vert} = \frac{\boldsymbol{I}}{\Vert\boldsymbol{x}\Vert} - \frac{\boldsymbol{x}\boldsymbol{x}^{\top}}{\Vert\boldsymbol{x}\Vert^3}\end{equation}
最后代回式$\eqref{eq:rms-norm}$得
\begin{equation}\nabla_{\boldsymbol{x}} \boldsymbol{y} = \sqrt{d}\left(\frac{\boldsymbol{I}}{\Vert\boldsymbol{x}\Vert} - \frac{\boldsymbol{x}\boldsymbol{x}^{\top}}{\Vert\boldsymbol{x}\Vert^3}\right) = \frac{1}{\Vert\boldsymbol{x}\Vert_{RMS}}\left(\boldsymbol{I} - \frac{\boldsymbol{y}\boldsymbol{y}^{\top}}{d}\right)\label{eq:rms-norm-grad}\end{equation}

## DyT现！

注意$\boldsymbol{x},\boldsymbol{y}$都是一个向量，所以$\nabla_{\boldsymbol{x}} \boldsymbol{y}$是一个矩阵（雅可比矩阵）。现在我们考虑给RMS Norm找一个Element\-wise近似，即每个分量是独立运算的：
\begin{equation}f(\boldsymbol{x}) = [f(x_1),f(x_2),\cdots,f(x_d)]\end{equation}
这个独立性意味着它的雅可比矩阵一定是对角阵！我们希望这个近似能尽可能保留RMS Norm的梯度，所以我们考虑保留式$\eqref{eq:rms-norm-grad}$的对角线部分：
\begin{equation}\frac{dy_i}{dx_i} = \frac{1}{\Vert\boldsymbol{x}\Vert_{RMS}}\left(1 - \frac{y_i^2}{d}\right)\label{eq:ode-1}\end{equation}
如果我们进一步假设$\rho = \Vert\boldsymbol{x}\Vert_{RMS}$是常数，那么可以直接求解上述微分方程得到
\begin{equation}y_i = \sqrt{d}\tanh\left(\frac{x_i}{\rho\sqrt{d}}\right)\end{equation}
这样我们就得到了DyT的T（$\tanh$），其中求解过程选择的初值条件为$y_i(0)=0$。

DyT相当于将前面的$\sqrt{d}$吸收到$\boldsymbol{\gamma}$参数里，然后将括号内的$\frac{1}{\rho\sqrt{d}}$视为训练参数$\alpha$，缓解“$\rho = \Vert\boldsymbol{x}\Vert_{RMS}$是常数”这一假设带来的限制。不过在笔者看来，显式保留$\sqrt{d}$可能会更有价值，只要将$\frac{1}{\rho}$部分视为可训练参数就好。

## DyISRU

不知道大家有没有留意到，对于RMS Norm我们恒有$y_i = x_i / \Vert\boldsymbol{x}\Vert_{RMS}$，所以方程$\eqref{eq:ode-1}$的$\Vert\boldsymbol{x}\Vert_{RMS}$我们可以换成$x_i/y_i$，从而得到
\begin{equation}\frac{dy_i}{dx_i} = \frac{y_i}{x_i}\left(1 - \frac{y_i^2}{d}\right)\label{eq:ode-2}\end{equation}
这是一个只有$x_i,y_i$的方程，免除了对$\Vert\boldsymbol{x}\Vert_{RMS}$的近似处理。求解该方程得
\begin{equation}y_i = \frac{\sqrt{d}x_i}{\sqrt{x_i^2 + C}}\end{equation}
其中$C$是任意常数。这种形式有个名字叫做ISRU（Inverse Square Root Unit，我们之前也叫过[SoftSign](<https://spaces.ac.cn/archives/10563#SoftSign>)），出自论文[《Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)》](<https://papers.cool/arxiv/1710.09967>)。如果将$C$视为可训练参数，那么就可以类比DyT称为DyISRU（Dynamic ISRU）。

从梯度$\eqref{eq:rms-norm-grad}$到方程$\eqref{eq:ode-1}$再到$\eqref{eq:ode-2}$来看，DyISRU是我们用Element\-wise函数能做到的最好结果，因为除对角线假设外没有再加额外近似了。从形式上看，DyISRU其实也比DyT更直观，因为$\Vert\boldsymbol{x}\Vert_{RMS}^2$即$\mathbb{E}[x_i^2]$，既然要寻求Element\-wise的运算，只好将$\mathbb{E}[x_i^2]$换成$x_i^2$了，最后加$C$乘$\sqrt{d}$算是平滑操作：
\begin{equation}\frac{x_i}{\sqrt{\color{red}{\frac{1}{d}\sum\limits_{i=1}^d x_i^2}}}\quad\to\quad \frac{x_i}{\sqrt{\color{green}{x_i^2}}}\quad\to\quad \frac{\color{orange}{\sqrt{d}} x_i}{\sqrt{\color{green}{x_i^2} + \color{orange}{C}}}\end{equation}

## 相关工作

$\tanh$和ISRU都可以视为符号函数的光滑近似，而基于它们，我们可以构建$\mathop{\text{clip}}$运算的光滑近似，例如
\begin{equation}\mathop{\text{clip}}(x, -t, t) = \left\{
\begin{aligned}t,&\,\,\, x > t \\ x,&\,\,\, x\in[-t,t] \\ -t,&\,\,\, x < -t\end{aligned}
\right.\quad\approx\quad t\tanh\left(\frac{x}{t}\right)\triangleq \mathop{\text{softcap}}(x, t)\end{equation}
由此，我们也可以将DyT理解为引入（光滑的）$\mathop{\text{clip}}$操作来防止前向传播的爆炸，从而稳定模型。

$\mathop{\text{softcap}}$提出自Google的[Gemma2](<https://papers.cool/arxiv/2408.00118>)，当时的用途是加在Softmax前的Attention Logits矩阵上，防止出现过大的Logits值。然而，我们实测中发现，尽管$\mathop{\text{softcap}}$之后的Logits不会爆炸，但$\mathop{\text{softcap}}$之前的Logits仍有爆炸风险，所以用$\mathop{\text{softcap}}$防止Logits爆炸纯粹是将问题换了个出处，治标不治本。

不知道是否Google后来也意识到了这个问题，他们在最新的[Gemma3](<https://papers.cool/arxiv/2503.19786>)中，选择去掉$\mathop{\text{softcap}}$而改用[QK\-norm](<https://spaces.ac.cn/archives/9859>)。我们自己的实验也显示，QK\-norm可以更好地抑制Attention Logits的增长。这个改动和结论实际上再次间接传递了一个悲观信号：DyT等$\mathop{\text{softcap}}$类操作在实践中很难完全取代Normalization。

## 文章小结

本文从梯度近似角度来分析什么样的Element\-wise的激活函数才能（一定程度上）替代Normalization层，从中我们可以推出DyT以及新的结果。
