# Transformer升级之路：3、从Performer到线性Attention

> 作者：苏剑林 · 科学空间 · 2021-04-22
>
> 原文：<https://spaces.ac.cn/archives/8338>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

看过笔者之前的文章[《线性Attention的探索：Attention必须有个Softmax吗？》](<https://spaces.ac.cn/archives/7546>)和[《Performer：用随机投影将Attention的复杂度线性化》](<https://spaces.ac.cn/archives/7921>)的读者，可能会觉得本文的标题有点不自然，因为是先有线性Attention然后才有Performer的，它们的关系为“Performer是线性Attention的一种实现，在保证线性复杂度的同时保持了对标准Attention的近似”，所以正常来说是“从线性Attention到Performer”才对。

然而，本文并不是打算梳理线性Attention的发展史，而是打算反过来思考Performer给线性Attention所带来的启示，所以是“从Performer到线性Attention”。

## 激活函数

线性Attention的常见形式是
\begin{equation}Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})_i = \frac{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \text{sim}(\boldsymbol{q}_i, \boldsymbol{k}_j)} = \frac{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)\boldsymbol{v}_j}{\sum\limits_{j=1}^n \phi(\boldsymbol{q}_i)^{\top} \varphi(\boldsymbol{k}_j)}\end{equation}
其中$\phi(\cdot)$、$\varphi(\cdot)$是值域非负的激活函数。那么如何选取这个激活函数呢？Performer告诉我们，应该选择指数函数
\begin{equation}\phi(x)=\varphi(x)=e^x\end{equation}

首先，我们来看它跟已有的结果有什么不一样。在[《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》](<https://papers.cool/arxiv/2006.16236>)给出的选择是：
\begin{equation}\phi(x)=\varphi(x)=1 + \text{elu}(x) = \left\{\begin{aligned}1 + x,\, x \geq 0\\ e^x,\, x < 0\end{aligned}\right.\end{equation}
我们知道$1+x$正是$e^x$在$x=0$处的一阶泰勒展开，因此$1+\text{elu}(x)$这个选择其实已经相当接近$e^x$了。

此外，$\phi(x)=\varphi(x)=e^x$这个方案还跟[《Efficient Attention: Attention with Linear Complexities
》](<https://papers.cool/arxiv/1812.01243>)一文中引入的双重softmax来构建线性Attention的设计很相似，在那种设计中有$\phi(\boldsymbol{q})=softmax(\boldsymbol{q}),\varphi(\boldsymbol{k})=e^{\boldsymbol{k}}$，相比直接$\phi(x)=\varphi(x)=e^x$只不过归一化的位置有所不同。

## 简单推导

为什么说Performer告诉我们激活函数的最佳选择是$e^x$呢？我们来看Performer找到的将标准Attention线性化的映射：
\begin{equation}\begin{aligned}
e^{\boldsymbol{q}\cdot \boldsymbol{k}}&=\mathbb{E}_{\boldsymbol{\omega}\sim \mathcal{N}(\boldsymbol{\omega};0,\boldsymbol{1}_d)}\left[e^{\boldsymbol{\omega}\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2} \times e^{\boldsymbol{\omega}\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2}\right]\\[6pt]
&\approx\underbrace{\frac{1}{\sqrt{m}}\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}-\Vert \boldsymbol{q}\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{q}}}
\cdot  \underbrace{\frac{1}{\sqrt{m}}\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2} \end{pmatrix}}_{\tilde{\boldsymbol{k}}}
\end{aligned}\end{equation}
简单来说，Performer找到了一个映射，使得$d$维向量$\boldsymbol{q},\boldsymbol{k}$被映射为了$m$维向量$\tilde{\boldsymbol{q}},\tilde{\boldsymbol{k}}$，并且满足近似关系$e^{\boldsymbol{q}\cdot \boldsymbol{k}}\approx \tilde{\boldsymbol{q}}\cdot\tilde{\boldsymbol{k}}$，此时
\begin{equation}a_{i,j} = \frac{e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}}{\sum\limits_j e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}}\approx \frac{\tilde{\boldsymbol{q}}_i\cdot\tilde{\boldsymbol{k}}_j}{\sum\limits_j \tilde{\boldsymbol{q}}_i\cdot\tilde{\boldsymbol{k}}_j} = \frac{(\lambda(\tilde{\boldsymbol{q}}_i)\tilde{\boldsymbol{q}}_i)\cdot\tilde{\boldsymbol{k}}_j}{\sum\limits_j (\lambda(\tilde{\boldsymbol{q}}_i)\tilde{\boldsymbol{q}}_i)\cdot\tilde{\boldsymbol{k}}_j}\end{equation}
最后一个等式表明，往$\tilde{\boldsymbol{q}}$里边乘以一个常数（哪怕这个常数跟$\tilde{\boldsymbol{q}}$有关），Performer的结果完全不改变，这意味着将映射改为
\begin{equation}
\tilde{\boldsymbol{q}} = \begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}} \end{pmatrix},\qquad
\tilde{\boldsymbol{k}}=\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{k}-\Vert \boldsymbol{k}\Vert^2 / 2} \end{pmatrix}
\end{equation}
Performer的结果不会有任何变化。当然，这里$\Vert \boldsymbol{k}\Vert^2$这一项还不能去掉，但是如果我们假设$\Vert \boldsymbol{k}\Vert^2$不会波动太大，它并不是Attention的主要因素，那么这一项也相当于一个常数，于是最终的映射（近似地）等价为
\begin{equation}
\tilde{\boldsymbol{q}} = \begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{q}} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{q}}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{q}} \end{pmatrix},\qquad
\tilde{\boldsymbol{k}}=\begin{pmatrix}e^{\boldsymbol{\omega}_1\cdot \boldsymbol{k}} \\
e^{\boldsymbol{\omega}_2\cdot \boldsymbol{k}}\\
\vdots\\
e^{\boldsymbol{\omega}_m\cdot \boldsymbol{k}} \end{pmatrix}
\end{equation}
这个看上去已经简化很多的映射该怎么理解呢？其实$m$个随机向量$\boldsymbol{\omega}_1,\boldsymbol{\omega}_2,\cdots,\boldsymbol{\omega}_m$拼成了一个$d\times m$的矩阵，它将$d$维的$\boldsymbol{q},\boldsymbol{k}$映射为了$m$维的向量，然后加上激活函数$e^x$得到了$\tilde{\boldsymbol{q}},\tilde{\boldsymbol{k}}$。我们知道Attention的$\boldsymbol{q},\boldsymbol{k}$都有一个全连接层变换，如果我们将这个$d\times m$的映射矩阵整合到全连接层中，那么剩下的就是一个激活函数$e^x$了！

所以这就是最优激活函数$e^x$的来源了，只要我们将$\boldsymbol{q},\boldsymbol{k}$的输出维度从$d$维改为$m$维，然后配合$e^x$的激活函数，那么理论上它就有Performer的拟合能力，甚至更强，因为Performer的$d\times m$矩阵是一个固定的随机矩阵，而这里我们相当于把该矩阵也设为可训练了，还去掉了低秩约束，空间是比Performer更大的。

## 低秩问题

不管是本文的主角Performer，还是之前在[《Nyströmformer：基于矩阵分解的线性化Attention方案》](<https://spaces.ac.cn/archives/8180>)介绍的Nyströmformer，它们的思路都是“寻找一个能逼近标准Attention的线性Attention”。那么一个很自然的问题就是：标准Attention有什么好的？哪里值得大家向它对齐？

从信息损失的角度来看，标准Attention矩阵的“秩”可能更大，即更接近可逆矩阵，这意味着它能保留更多有效信息。具体来说，Attention矩阵是一个$n\times n$的矩阵，它由$\boldsymbol{Q},\boldsymbol{K}\in\mathbb{R}^{n\times d}$通过$softmax(\boldsymbol{Q}\boldsymbol{K}^{\top})$而来，要注意的是，这里的$d$是Attention的key\_size，比如对于BERT base来说它只是64，而$n$往往比较大，这说明$\boldsymbol{Q}\boldsymbol{K}^{\top}$的秩不超过$d$，而且$d\ll n$，即离满秩还远得很。不过，$softmax$的关键运算是$e^{\boldsymbol{Q}\boldsymbol{K}^{\top}}$，一个矩阵如果每个元素取指数的话，那么新矩阵的秩是可能增加的！所以标准Attention矩阵有升秩的可能性，意味着它蕴含了更有效处理信息的能力。

相比之下，线性Attention矩阵是$\tilde{\boldsymbol{Q}}\tilde{\boldsymbol{K}}^{\top}$的形式，所以线性Attention矩阵的秩一定不超过$m$，而为了弥补秩的损失，所以一般要设置$m > d$，在Performer的实验中选择的是$m = 4d$，也就是key\_size扩大为4倍，秩的重要性可见一斑。当然，扩大了key\_size，一个直接的后果是处理短序列的时候，线性Attention还比标准Attention要慢，这是线性Attention的固有瓶颈。

关于Attention矩阵的秩的理论分析，也有一些论文可以参考，比如[《Low\-Rank Bottleneck in Multi\-head Attention Models》](<https://papers.cool/arxiv/2002.07028>)就指出哪怕在标准Attention中，低秩性也是一个严重的瓶颈，增大key\_size可以提升性能；上个月的[《Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth》](<https://papers.cool/arxiv/2103.03404>)则指出，如果没有残差和FFN，那么标准Attention有极大的风险退化为秩等于1的简单变换。连标准Attention这个有“升秩潜力”的模型都有低秩问题，更不用说线性Attention这种本身秩就有上限的模型了。

所以，一句话就是：用线性Attention需要用更大的key\_size来维持矩阵的秩。

## 集中注意

我们还可以从稀疏性角度来理解标准Attention的好处。直观来想，既然是“注意力机制”，那么肯定需要“集中注意力”，如果太分散，那么可能就相当于平均池化了，而“集中注意力”，意味着每个token应该只能显著地关联到若干个token，用数学的话说，那就是意味着Attention矩阵是稀疏的，或者说至少要具备变得稀疏的可能性。

对于标准Attention来说，它通过softmax来归一化
\begin{equation}a_{i,j} = \frac{e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}}{\sum\limits_j e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}}\end{equation}
其中指数函数$e^x$起到了一个放大的作用，只要各个$\boldsymbol{q}_i\cdot \boldsymbol{k}_j$本身能拉开一定差距，那么$e^{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}$会进一步放大这种差距，结果就是归一化之后除了最大值的那几个位置之外，剩下的概率都很接近于0了，这说明标准Attention是有潜力“集中注意力”的。而对于线性Attention来说，它是直接内积的结果，没有得到$e^x$的进一步放大，所以它的注意力是比较稠密的，在序列长度较大的时候，它往往就很接近平均池化了。要缓解这一点，还是需要增大key\_size，来放大差距，直观来说，就是$n$向量放到一个低维空间太“挤”了，换到更高维的空间就“松”一些了。

怎么样验证稀疏的重要性呢？笔者曾经尝试过，将线性Attention的Attention矩阵先算出来，然后强行截断Attention矩阵（也就是每个token只跟前后几个token做attention，变成局部形式的Attention）让它变得稀疏，结果发现这种截断后的线性Attention效果明显好于全矩阵的线性Attention。这就肯定了稀疏的重要性了，当然，这样把Attention矩阵先算出来然后前行截断的方式，使得线性Attention的复杂度不再是线性的了，因此不具备实用价值，仅用于理论验证。

还有一个实验现象可以辅助证明稀疏的重要性，那就是线性Attention做语言模型或者解码器的时候，效果是跟标准Attention差不了多少的，这时候线性Attention变成了单向的RNN（参考[《Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention》](<https://papers.cool/arxiv/2006.16236>)），等价于Attention矩阵变成了下三角阵，也是更稀疏了。相比之下，如果用不稀疏的双向的线性Attention直接做MLM模型，则掉点会相当明显。

更重要的是，稀疏性和前一节提到的秩是有密切关联的，甚至可以说它们是“一体两面”：适当的稀疏化方法能提高矩阵的秩！比如做语言模型的下三角Attention矩阵，只要对角线元素非零（往往都能达到），那么这时候的矩阵直接就是满秩可逆阵了！还有笔者实验的局部Attention截断，也能增加矩阵的秩，比如极端情况下，每个token只跟自身做attention，那么Attention矩阵就是满秩的单位阵了！

## 文章小结

本文从Performer出发思考了线性Attention的一些问题，包括关于线性Attention的激活函数选择，以及线性Attention的瓶颈所在（低秩性、稀疏性），总的结论是，线性Attention的最佳激活函数应当是指数函数，而有效的Attention机制应当具备更高的秩和更大的稀疏性。
