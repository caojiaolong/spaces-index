# 我们可以无损放大一个Transformer模型吗（一）

> 作者：苏剑林 · 科学空间 · 2021-06-02
>
> 原文：<https://spaces.ac.cn/archives/8444>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

看了标题，可能读者会有疑惑，大家不都想着将大模型缩小吗？怎么你想着将小模型放大了？其实背景是这样的：通常来说更大的模型加更多的数据确实能起得更好的效果，然而算力有限的情况下，从零预训练一个大的模型时间成本太大了，如果还要调试几次参数，那么可能几个月就过去了。

这时候“穷人思维”就冒出来了（土豪可以无视）：能否先训练一个同样层数的小模型，然后放大后继续训练？这样一来，预训练后的小模型权重经过放大后，就是大模型一个起点很高的初始化权重，那么大模型阶段的训练步数就可以减少了，从而缩短整体的训练时间。

那么，小模型可以无损地放大为一个大模型吗？本文就来从理论上分析这个问题。

## 含义

有的读者可能想到：这肯定可以呀，大模型的拟合能力肯定大于小模型呀。的确，从拟合能力角度来看，这件事肯定是可以办到的，但这还不是本文关心的“无损放大”的全部。

以BERT为例，预训练阶段主要就是一个MLM模型，那么“无损放大”的含义就是：

> 是否可以通过某种变换，把一个小模型直接变换成一个大模型，并且输出完全不改变？

这里的变换，指的是对权重做一些确定性的变换，而不用通过梯度下降来继续训练；输出完全不改变，指的是对于同一个输入，小模型和大模型给出的预测结果是完全一致的，也就是说它们表面上看起来不一样，但数学上它们是完全一致的函数，所以称为“无损放大”。由于是无损放大，我们至少可以保证大模型不差于小模型，所以继续预训练理论上有正的收益。至于先小后大这样预训练在效果上能不能比得上一开始就从大训练，这个需要实验来确定，并不是本文关心的问题。

直觉来想，这种放大也不困难，比如通过“重复”、“补零”等操作就可以实现模型权重的自然放大。事实上尝试的方向也是如此，但难点在于我们需要仔细分析模型的每一个模块在被放大之后所产生的后果，以确保最终的结果是无损的。

## 尝试

下面我们以“将一个BERT放大为2倍”为例子进行分析尝试，来确定最终的变换形式。这里的“放大”指的是<u>仅仅扩大隐层向量的维度，并不改变模型的层数，也不改变多头注意力机制的头数</u>。

### Embedding

首先，输入层是Embedding层，因此先要解决的是Embedding层的放大问题。这也是其中最简单的一环，就是直接将每个token的向量维度都放大为2倍即可，主要就是“重复”、“补零”两种操作：
\begin{equation}\begin{array}{ll}
\text{重复：} & [x_1,x_2,x_3,x_4] \to [x_1, x_1, x_2, x_2, x_3, x_3, x_4, x_4]\\
\text{补零：} & [x_1,x_2,x_3,x_4] \to [x_1,x_2,x_3,x_4,0,0,0,0]
\end{array}\end{equation}
两种方案都可以作为候选方案，但直觉上来想，补零这种方式引入了太多的零，会导致过度稀疏和同一个值重复次数过多，不利于权重的多样性，因此我们还是选择了重复这种方案。不过，就算只看重复，也不指上述一种方式，比如$[x_1,x_2,x_3,x_4,x_1,x_2,x_3,x_4]$也是一种方案，但后面关于Attention层的分析表明，后一种方案是不可取的。

除此之外，我们通常还希望变换是正交的，这通常能最大程度上保证模型的稳定性，具体来说，正交变换的最基本性质是不改变向量的模型，所以我们将最终的重复变换调整为：
\begin{equation}\begin{pmatrix}x_1 \\ x_2 \\ \vdots \\ x_d\end{pmatrix}\quad \to\quad \begin{pmatrix}\tilde{x}_1 \\ \tilde{x}_2 \\ \tilde{x}_3 \\ \tilde{x}_4 \\ \vdots \\ \tilde{x}_{2d-1} \\ \tilde{x}_{2d}\end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix}x_1 \\ x_1 \\ x_2 \\ x_2 \\ \vdots \\ x_d \\ x_d \end{pmatrix}\label{eq:vt}\end{equation}
或者简记成$\tilde{x}_i = x_{\lceil i/2\rceil} / \sqrt{2}$，其中$\lceil \cdot\rceil$是上取整运算，我们称之为“<strong>重复再除以$\sqrt{2}$</strong>”。

### LayerNorm

Embedding的下一层就是LayerNorm了，变换前，LayerNorm的运算为
\begin{equation}y_i = \frac{x_i - \mu}{\sigma}\times \gamma_i + \beta_i\quad \mu = \frac{1}{d}\sum_{i=1}^d x_i\quad \sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2}\end{equation}
而变换后，我们有
\begin{equation}\begin{aligned}
&\tilde{\mu} = \frac{1}{2d}\sum_{i=1}^{2d} \tilde{x}_i = \frac{1}{d}\sum_{i=1}^{d} \frac{x_i}{\sqrt{2}} = \frac{\mu}{\sqrt{2}}\\
&\tilde{\sigma} = \sqrt{\frac{1}{2d}\sum_{i=1}^{2d} (\tilde{x}_i-\tilde{\mu})^2}=\sqrt{\frac{1}{d}\sum_{i=1}^{d} \left(\frac{x_i}{\sqrt{2}}-\frac{\mu}{\sqrt{2}}\right)^2}=\frac{\sigma}{\sqrt{2}}\\
&\frac{\tilde{x}_i-\tilde{\mu}}{\tilde{\sigma}} = \frac{x_{\lceil i/2\rceil} / \sqrt{2} - \mu/\sqrt{2}}{\sigma/\sqrt{2}} = \frac{x_{\lceil i/2\rceil} - \mu}{\sigma}
\end{aligned}\end{equation}
这也就是说，“减均值除以标准差”这一步自动帮我们消去了$1/\sqrt{2}$这个因子，其结果是放大前结果的直接重复。如果我们将参数向量$\beta,\gamma$也按照公式$\eqref{eq:vt}$进行变换，那么结果将是$\tilde{y}_i = y_{\lceil i/2\rceil} / \sqrt{2}$，跟Embedding层的变换结果一致，而我们就是要<u>尽量使得每一层“净变换”都是同样的一个简单变换：“重复再除以$\sqrt{2}$”</u>。

### FeedForward

按照顺序，接下来本来应该分析Attention层才对，不过FeedForward层相对简单一点，并且FeedForward层的分析结果也对后面理解Attention层的变换有所帮助，因此这里先来考虑FeedForward层的变换。

FeedForward层只是两个全连接层的复合，所以我们只需要分析单个全连接层：
\begin{equation} y_j = \mathcal{A}\left(\sum_{i=1}^d x_i w_{i,j} + b_j\right)\end{equation}
这里的$\mathcal{A}(\cdot)$是激活函数。鉴于之前的经验，我们尝试如下变换
\begin{equation}\tilde{w}_{i,j}=\frac{1}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil},\quad \tilde{b}_j=\frac{1}{\sqrt{2}}b_{\lceil j/2\rceil}\label{eq:wt}\end{equation}
也就是将$b_j$按照式$\eqref{eq:vt}$进行变换，而对于$w_{i,j}$则尝试使用形式下述变换：
\begin{equation}\begin{pmatrix}w_{1,1} & w_{1,2} & \cdots & w_{1,D} \\ w_{2,1} & w_{2,2} & \cdots & w_{2,D} \\ \vdots & \vdots & \ddots & \vdots \\ w_{d,1} & w_{d,2} & \cdots & w_{d,D}\end{pmatrix} \quad\to\quad \frac{1}{2}\left(\begin{array}{cc:cc:c:cc} w_{1,1} & w_{1,1} & w_{1,2} & w_{1,2} & \cdots & w_{1,D} & w_{1,D} \\ w_{1,1} & w_{1,1} & w_{1,2} & w_{1,2} & \cdots & w_{1,D} & w_{1,D} \\ \hdashline w_{2,1} & w_{2,1} & w_{2,2} & w_{2,2} & \cdots & w_{2,D} & w_{2,D} \\ w_{2,1} & w_{2,1} & w_{2,2} & w_{2,2} & \cdots & w_{2,D} & w_{2,D} \\ \hdashline \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots\\ \hdashline w_{d,1} & w_{d,1} & w_{d,2} & w_{d,2} & \cdots & w_{d,D} & w_{d,D} \\ w_{d,1} & w_{d,1} & w_{d,2} & w_{d,2} & \cdots & w_{d,D} & w_{d,D}\end{array}\right)\end{equation}
这里的$D$就是输出维度大小，这里我们假设模型放大2倍后，$D$也放大2倍。不难看出，该变换其实就是对变换矩阵$w_{i,j}$行列两个方向都分别执行变换$\eqref{eq:vt}$。此时
\begin{equation}\begin{aligned}
\sum_{i=1}^{2d} \tilde{x}_i \tilde{w}_{i,j} + \tilde{b}_j =&\, 2\sum_{i=1}^d \frac{x_i}{\sqrt{2}} \frac{w_{i,\lceil j/2\rceil}}{2} + \frac{b_{\lceil j/2\rceil}}{\sqrt{2}} \\
=&\, \frac{1}{\sqrt{2}}\left(\sum_{i=1}^d x_i w_{i,\lceil j/2\rceil} + b_{\lceil j/2\rceil}\right)
\end{aligned}\end{equation}
这说明变换$\eqref{eq:wt}$对于线性变换层来说，能够满足我们的理想追求——放大后的结果就是“重复再除以$\sqrt{2}$”。然而，这还不够，因为全连接层还有个激活函数$\mathcal{A}(\cdot)$，现在的问题在于$\mathcal{A}(x/\sqrt{2})$未必等于$\mathcal{A}(x)/\sqrt{2}$，而如果不等，我们就没法让整体的变换等价于“重复再除以$\sqrt{2}$”。

事实上，BERT用的[GeLU激活函数](<https://spaces.ac.cn/archives/7309>)就不满足该恒等式；线性激活函数（不加激活函数）显然是满足这个等式的，而满足这个等式一个常见的非线性激活函数便是ReLU（也包括LeakyReLU）函数，因此一个直接的解决方式就是FeedForward层换用ReLU激活函数。事实上，这也已经是预训练模型的一个常见选择了，百度的Ernie和Google的T5模型，它们的FeedForward层激活函数都是用ReLU。

那么，像BERT这样的非ReLU激活函数的FeedForward层就没办法了吗？那也不至于，因为FeedForward层是两个全连接层的复合，我们只需要在变换第一个全连接的时候少除以一个$\sqrt{2}$，变换第二个全连接的时候多除以一个$\sqrt{2}$就行了。具体来说，第一个全连接权重变为：
\begin{equation}
\tilde{w}_{i,j}=\frac{1}{\sqrt{2}}w_{\lceil i/2\rceil,\lceil j/2\rceil},\quad \tilde{b}_j=b_{\lceil j/2\rceil}\label{eq:wt-2}\end{equation}
此时就有
\begin{equation}\mathcal{A}\left(\sum_{i=1}^{2d} \tilde{x}_i \tilde{w}_{i,j} + \tilde{b}_j\right) = \mathcal{A}\left(\sum_{i=1}^d x_i w_{i,\lceil j/2\rceil} + b_{\lceil j/2\rceil}\right)
\end{equation}
此时结果就是原结果的直接重复，没有除以$\sqrt{2}$，既然如此，后面紧接着的全连接层多除以一个$\sqrt{2}$就行了，即后面的全连接层权重变换为
\begin{equation}
\tilde{w}_{i,j}=\frac{1}{2\sqrt{2}}w_{\lceil i/2\rceil,\lceil j/2\rceil},\quad \tilde{b}_j=\frac{1}{2}b_{\lceil j/2\rceil}\end{equation}
这样整个FeedForward层的效果就等价于“重复再除以$\sqrt{2}$”了。

### Attention

现在到了最难啃的“硬骨头”——Attention层的变换。Attention层首先通过三个线性层将每个输入向量变换为$q,k,v$：
\begin{equation}
q_j = \sum_{i=1}^d x_i w_{i,j}^{(q)} + b_j^{(q)}, \quad k_j = \sum_{i=1}^d x_i w_{i,j}^{(k)} + b_j^{(k)}, \quad v_j = \sum_{i=1}^d x_i w_{i,j}^{(v)} + b_j^{(v)}
\end{equation}
根据前面对FeedForward层的分析可以得知，如果要想$q,k,v$都达到“重复再除以$\sqrt{2}$”的效果，只需要按照变换$\eqref{eq:wt}$进行。但Attention层不是单纯的全连接层，变换完之后，我们要检查Attention矩阵是否不变，我们来算内积：
\begin{equation}\sum_{i=1}^{2d'} \tilde{q}_i \tilde{k}_i = 2\sum_{i=1}^{d'} \frac{q_i}{\sqrt{2}}\frac{k_i}{\sqrt{2}} = \sum_{i=1}^{d'} q_i k_i\end{equation}
其中$d'$是对应的head\_size。这个结果告诉我们，上述变换保持了内积不变，所以应该也保持Attention矩阵不变。但是，这里有一个陷阱！如果是T5这样的模型，它的内积之后是没有尺度缩放的，所以这样的确完事了；然而像BERT这样的模型，它是内积之后除了个$\sqrt{d'}$再做Softmax的，，而一旦放大模型后，除以$\sqrt{d'}$变成了除以$\sqrt{2d'}$，内积不变也不能保持Attention矩阵不变，而应当还需要往$q,k$的权重分别再乘个$\sqrt[4]{2}$，所以最终的变换应该是
\begin{equation}\begin{aligned}
&\tilde{w}_{i,j}^{(q)}=\frac{\sqrt[4]{2}}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(q)},\quad \tilde{b}_j^{(q)}=\frac{\sqrt[4]{2}}{\sqrt{2}}b_{\lceil j/2\rceil}^{(q)}\\
&\tilde{w}_{i,j}^{(k)}=\frac{\sqrt[4]{2}}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(k)},\quad \tilde{b}_j^{(k)}=\frac{\sqrt[4]{2}}{\sqrt{2}}b_{\lceil j/2\rceil}^{(k)}\\
&\tilde{w}_{i,j}^{(v)}=\frac{1}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(v)},\quad \tilde{b}_j^{(v)}=\frac{1}{\sqrt{2}}b_{\lceil j/2\rceil}^{(v)}
\end{aligned}\end{equation}
经过这样变换后，Attention矩阵不变，而$\tilde{v}_i = v_{\lceil i/2\rceil} / \sqrt{2}$，所以最终的输出结果也是$\tilde{o}_i = o_{\lceil i/2\rceil} / \sqrt{2}$。

上述内容只是针对Attention的单个头进行分析，事实上Attention有多个头，多个头的输出结果还要拼接起来再接一个全连接层。当然，由于每个头都是平等的、独立的，因此上述结论基本不变，最后全连接层也只需要按照式$\eqref{eq:wt}$进行变换，就可以让Attention的变换效果。但是，多头带来的一个效应是，我们在重复的时候，必须局部地进行重复。

具体来说，我们在实现多头的时候，并非是真的做了多个全连接运算，而是做了一个大的全连接运算后再reshape，这样一来我们可以比较两种不同的重复方式的reshape结果：
$$\begin{array}{c:c}
[x_1,x_2,x_3,x_4,x_5,x_6] & [x_1,x_2,x_3,x_4,x_5,x_6] \\
\downarrow & \downarrow \\
[x_1,x_1,x_2,x_2,x_3,x_3,x_4,x_4,x_5,x_5,x_6,x_6] & [x_1,x_2,x_3,x_4,x_5,x_6,x_1,x_2,x_3,x_4,x_5,x_6] \\
\downarrow & \downarrow \\
\begin{pmatrix}x_1,x_1,x_2,x_2 \\ x_3,x_3,x_4,x_4 \\ x_5,x_5,x_6,x_6\end{pmatrix} & \begin{pmatrix}x_1,x_2,x_3,x_4 \\ x_5,x_6,x_1,x_2 \\ x_3,x_4,x_5,x_6\end{pmatrix} \\
\end{array}$$
注意放大前reshape结果是$\begin{pmatrix}x_1,x_2 \\ x_3,x_4 \\ x_5,x_6\end{pmatrix}$，所以对比两种不同的重复方式的reshape结果，我们发现第二种重复方式reshape之后的结果全乱了，不等价于每个头分别重复。因此我们只能选择前一种重复方式。

### 输出概率分布

通过以上分析，我们可以使得整个Encoder在放大到2倍之后，实现“重复再除以$\sqrt{2}$”的效果。最后剩下的就是输出部分，即将Encoder的输出向量转化为token的概率分布，这里边包含几种情况。

像GPT、T5等模型，它们是直接在Encoder输出后面乘以了Embedding矩阵的转置来做作为概率分布的logits（当然有可能还有个偏置），由于Embedding矩阵本身就包含了“重复再除以$\sqrt{2}$”的操作，而Encoder的输出也是“重复再除以$\sqrt{2}$”，两者结合刚好抵消，所以从概率分布角度看，输出是完全不变的。

不过BERT多了一层全连接，也就是说它先接了一个GeLU激活的全连接层，然后才乘以Embedding矩阵的转置并加上偏置项作为logitis。在“FeedForward”那一节我们已经讨论了，非ReLU激活的全连接层无法实现“重复再除以$\sqrt{2}$”的效果，而只能通过变换$\eqref{eq:wt-2}$来实现单纯的“重复”效果，所以为了再达到“除以$\sqrt{2}$”的效果，它后面接的LayerNorm在变换的时候就要多除以一个$\sqrt{2}$了。

当然，如果是ReLU激活，那么按照式$\eqref{eq:wt}$来变换，那么可以实现完全不改变了。此外，如果是像mT5那样，最后转为logits的变换矩阵跟Embedding层不共享，那么也可以通过调整最后的变换矩阵来实现输出的完全不变。

### RoPE位置编码

前面的分析都只适用于每个神经元都是不相关的情形，也就是说向量的任意两个分量$x_i,x_j$是没啥关联的。但如果我们在模型中用了“[旋转式位置编码（RoPE）](<https://spaces.ac.cn/archives/8265>)”，那么这个假设就不成立了，因为RoPE是以每两个分量为一组进行变换的，即$[x_1,x_2]$为一组、$[x_3,x_4]$为一组，依此类推。

如果还是按照之前式$\eqref{eq:vt}$进行重复变换，那么变换之后就变成了$[x_1,x_1]$为一组、$[x_2,x_2]$为一组、\.\.\.，跟原来的分组不一致，所以会带来很大的偏差。这种情况下，重复的时候也应当按照两个为一组来进行：
\begin{equation}\begin{array}{c}
[x_1,x_2,x_3,x_4,\cdots,x_{d-1},x_d] \\
\downarrow\\
\frac{1}{\sqrt{2}}[x_1,x_2,x_1,x_2,x_3,x_4,x_3,x_4,\cdots,x_{d-1},x_d,x_{d-1},x_d]
\end{array}\label{eq:vt-2}\end{equation}

当然，由于默认的RoPE是没有可训练权重的，它是按照固定的方式进行渐变的，所以哪怕按照该方式进行重复，那不能完全保证结果一致。也就是说，如果使用了RoPE，那么基本上不能实现无损放大。不过实际测试结果表明，按照该方式进行重复放大后，对应的RoFormer虽然性能有所损失，但不多，可以很快通过继续训练恢复。

## 结论

现在我们可以确认，对于BERT来说，如果非线性激活函数用ReLU，那么BERT是可以直接无损放大的，如果非线性激活函数不是ReLU，那么可以实现MLM准确率无损的放大（事实上经过更精细的调整，也可以实现完全无损放大，但每个层的变换有点不统一了，不够优雅）；对于GPT、T5等模型来说，不管激活函数用啥（包括mT5用的GLU激活，也可以定制适当），其实都可以实现无损放大。

其中，将BERT权重进行放大为2倍的变换汇总如下：
$$\begin{array}{l|l}
\hline
\text{Embedding} & \tilde{x}_i = \frac{1}{\sqrt{2}} x_{\lceil i/2\rceil} \\
\hline
\text{LayerNorm} & \tilde{\beta}_i = \frac{1}{\sqrt{2}} \beta_{\lceil i/2\rceil},\quad \tilde{\gamma}_i = \frac{1}{\sqrt{2}} \gamma_{\lceil i/2\rceil} \\
\hline
\text{Attention} & \begin{array}{l}
\tilde{w}_{i,j}^{(q)}=\frac{\sqrt[4]{2}}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(q)},\quad \tilde{b}_j^{(q)}=\frac{\sqrt[4]{2}}{\sqrt{2}}b_{\lceil j/2\rceil}^{(q)}\\
\tilde{w}_{i,j}^{(k)}=\frac{\sqrt[4]{2}}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(k)},\quad \tilde{b}_j^{(k)}=\frac{\sqrt[4]{2}}{\sqrt{2}}b_{\lceil j/2\rceil}^{(k)}\\
\tilde{w}_{i,j}^{(v)}=\frac{1}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(v)},\quad \tilde{b}_j^{(v)}=\frac{1}{\sqrt{2}}b_{\lceil j/2\rceil}^{(v)} \\
\tilde{w}_{i,j}^{(o)}=\frac{1}{2}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(o)},\quad \tilde{b}_j^{(o)}=\frac{1}{\sqrt{2}}b_{\lceil j/2\rceil}^{(o)}
\end{array} \\
\hline
\text{FeedForward} & \begin{array}{l}
\tilde{w}_{i,j}^{(1)}=\frac{1}{\sqrt{2}}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(1)},\quad \tilde{b}_j^{(1)}=b_{\lceil j/2\rceil}^{(1)} \\
\tilde{w}_{i,j}^{(2)}=\frac{1}{2\sqrt{2}}w_{\lceil i/2\rceil,\lceil j/2\rceil}^{(2)},\quad \tilde{b}_j=\frac{1}{2}b_{\lceil j/2\rceil}^{(2)}
\end{array} \\
\hline
\text{输出概率分布} & \tilde{w}_{i,j}=\frac{1}{\sqrt{2}}w_{\lceil i/2\rceil,\lceil j/2\rceil},\quad \tilde{b}_j=b_{\lceil j/2\rceil} \\
\hline
\end{array}$$

如果是其他略有不同的模型，那么就模仿前面的思想进行类似的分析即可。如果是RoPE，那么将重复的方案改为式$\eqref{eq:vt-2}$就好；如果是扩大$k$倍，那么将表格中的多数2换为$k$就好。简单来说，如果Attention没有尺度缩放（除以$\sqrt{d'}$），以及FeedForward的激活函数是ReLU（或者LeakyReLU），那么放大$k$倍的变换就最简单的，将权重的每一维都执行“重复$k$次并除以$\sqrt{k}$”就好了。

## 小结

本文从数学上分析了直接放大Transformer模型的可能性，最终得到了若干可用的变换，确定了无损放大Transformer模型的可行性，为实现大模型的渐进式训练提供了参考思路。
