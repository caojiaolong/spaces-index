# 脑洞大开：非线性RNN居然也可以并行计算？

> 作者：苏剑林 · 科学空间 · 2023-09-26
>
> 原文：<https://spaces.ac.cn/archives/9783>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

近年来，线性RNN由于其可并行训练以及常数推理成本等特性，吸引了一定研究人员的关注（例如笔者之前写的[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](<https://spaces.ac.cn/archives/9554>)），这让RNN在Transformer遍地开花的潮流中仍有“一席之地”。然而，目前看来这“一席之地”只属于线性RNN，因为非线性RNN无法高效地并行训练，所以在架构之争中是“心有余而力不足”。

不过，一篇名为[《Parallelizing Non\-Linear Sequential Models over the Sequence Length》](<https://papers.cool/arxiv/2309.12252>)的论文有不同的看法，它提出了一种迭代算法，宣传可以实现非线性RNN的并行训练！真有如此神奇？接下来我们一探究竟。

## 求不动点

原论文对其方法做了非常一般的介绍，而且其侧重点是PDE和ODE，这里我们直接从RNN入手。考虑常见的简单非线性RNN：
\begin{equation}x_t = \tanh(Ax_{t-1} + u_t)\label{eq:rnn}\end{equation}
由于$\tanh$的存在，它只能串行计算。现在我们在两边都减去$Ax_{t-1}$：
\begin{equation}x_t - Ax_{t-1} = \tanh(Ax_{t-1} + u_t) - Ax_{t-1}\end{equation}
当然，这改变不了它是非线性RNN的实质。然而我们可以发现，假如右端的$x_{t-1}$换成像$u_t$那样的给定向量，那么这就是一个线性RNN了，根据[《Google新作试图“复活”RNN：RNN能否再次辉煌？》](<https://spaces.ac.cn/archives/9554>)的结果，它是可以并行计算的。此时，敏捷的读者可能已经猜到后面的步骤了——迭代求解！

首先，将上述RNN更改成
\begin{equation}x_t^{(n)} - Ax_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)}\label{eq:rnn-iter}\end{equation}
从给定$x_t^{(0)}$出发，反复迭代上式，理想情况下，它会收敛于一个不动点$x_t^*$，这就是原来非线性RNN的计算结果。当然，理论上通过式$\eqref{eq:rnn-iter}$迭代的总计算量是比直接通过式$\eqref{eq:rnn}$递归计算要大的，但由于每一步迭代都是可并行的线性RNN，并且如果收敛速度比较快时迭代步数不需要太多，那么总的耗时通常都会快于直接非线性RNN递归（尤其是序列长度很大时）。

## 简化形式

事实上，非线性RNN之所以慢，无法并行计算还是次要的，最关键是它包含了大量的非element\-wise运算，比如式$\eqref{eq:rnn}$的$\tanh$里边的矩阵运算$Ax_{t-1}$；而线性RNN之所以快，除了它允许并行训练之外，更关键的是它能通过对角化来将矩阵乘法变换为element\-wise的乘法——对于element\-wise乘法来说，即便是串行计算也不会太慢。

当我们通过式$\eqref{eq:rnn-iter}$将非线性RNN转为线性RNN的迭代之后，同样享受线性RNN可对角化的“待遇”，从而提高计算速度。具体来说，在复数域中将$A$对角化为$P\Lambda P^{-1}$，那么式$\eqref{eq:rnn-iter}$变为
\begin{equation}x_t^{(n)} - P\Lambda P^{-1} x_{t-1}^{(n)} = \tanh(P\Lambda P^{-1} x_{t-1}^{(n-1)} + u_t) - P\Lambda P^{-1} x_{t-1}^{(n-1)}\end{equation}
两端都左乘$P^{-1}$：
\begin{equation}P^{-1} x_t^{(n)} - \Lambda P^{-1} x_{t-1}^{(n)} = P^{-1}\tanh(P\Lambda P^{-1} x_{t-1}^{(n-1)} + u_t) - \Lambda P^{-1} x_{t-1}^{(n-1)}\end{equation}
令$y_t = P^{-1} x_t$，那么上式可以简化为
\begin{equation}y_t^{(n)} - \Lambda y_{t-1}^{(n)} = P^{-1}\tanh(P\Lambda y_{t-1}^{(n-1)} + u_t) - \Lambda y_{t-1}^{(n-1)}\end{equation}
由于RNN之后一般都还要接个投影层，所以$x_t = P y_t$的$P$原则上可以合并到外接的投影层里边，也就是说，上式理论上具备跟原来的$\eqref{eq:rnn}$具备同等的表达能力，但由于$\Lambda$是对角阵，递归的计算量会明显降低。上式还出现了逆矩阵$P^{-1}$，不单计算量大，而且不利于优化，所以我们可以干脆将$P^{-1}$和$P\Lambda$换成两个不相关的参数矩阵：
\begin{equation}y_t^{(n)} - \Lambda y_{t-1}^{(n)} = P\tanh(Q y_{t-1}^{(n-1)} + u_t) - \Lambda y_{t-1}^{(n-1)}\end{equation}
只要初始化是$PQ=\Lambda$就行。

## 摄动思想

假定$x_t^{(0)}=0$，那么式$\eqref{eq:rnn-iter}$其实就是将原本的非线性RNN就分解为一系列线性RNN：
\begin{equation}\begin{array}{c}
x_t^{(1)} - Ax_{t-1}^{(1)} = \tanh(u_t)\\
x_t^{(2)} - Ax_{t-1}^{(2)} = \tanh(Ax_{t-1}^{(1)} + u_t) - Ax_{t-1}^{(1)} \\
\vdots \\
x_t^{(n)} - Ax_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - Ax_{t-1}^{(n-1)} \\
\vdots \\
\end{array}\label{eq:rnns}\end{equation}
而假设$x_{t-1},u_t$都是小量，那么对式$\eqref{eq:rnn}$右端利用$\tanh x \approx x$得到：
\begin{equation}x_t = \tanh(Ax_{t-1} + u_t) \approx Ax_{t-1} + u_t \approx Ax_{t-1} + \tanh(u_t)\label{eq:rnn-approx}\end{equation}
这正好是$\eqref{eq:rnns}$中的第一个方程，因此如果假设成立，那么$x_t^{(1)}$或许已经足够接近理想的$x_t^*$，后面的每一步迭代都在快速逼近它。从这里我们可以看出，“两边同时减去$Ax_{t-1}$”是关键之处，这使得$\eqref{eq:rnn-iter}$的第一步迭代就接近于原本非线性RNN的一阶线性近似，这可以提高收敛速度，也是数学物理中的经典操作，名曰“[摄动](<https://spaces.ac.cn/tag/%E6%91%84%E5%8A%A8/>)”。

## 加快收敛

根据摄动法的思想，提高收敛速度的关键就是提高近似展开的精度，比如较为简单的改进是只假设$x_{t-1}$是小量，那么根据一阶泰勒展开有（将$u_t$视为列向量，这里的$\circ$是Hadamard积分）
\begin{equation}x_t = \tanh(Ax_{t-1} + u_t) \approx \tanh(u_t) + (\text{sech}^2 u_t\circ A)x_{t-1}\end{equation}
于是改进的结果就是式$\eqref{eq:rnn-iter}$变为
\begin{equation}x_t^{(n)} - A_t x_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - A_t x_{t-1}^{(n-1)}\label{eq:iter-plus1}\end{equation}
其中$A_t = \text{sech}^2 u_t\circ A$。更精细的改进是在每一步迭代时，都在前一步迭代结果的基础上进行展开：
\begin{equation}\begin{aligned}
x_t =&\, \tanh(Ax_{t-1} + u_t) \\
\approx&\, \tanh(Ax_{t-1}^{(n-1)} + u_t) + (\text{sech}^2 (Ax_{t-1}^{(n-1)} + u_t)\circ A)(x_{t-1} - x_{t-1}^{(n-1)})
\end{aligned}\end{equation}
于是式$\eqref{eq:rnn-iter}$变为
\begin{equation}x_t^{(n)} - A_t^{(n)} x_{t-1}^{(n)} = \tanh(Ax_{t-1}^{(n-1)} + u_t) - A_t^{(n)} x_{t-1}^{(n-1)}\label{eq:iter-plus2}\end{equation}
其中$A_t^{(n)}=\text{sech}^2 (Ax_{t-1}^{(n-1)} + u_t)\circ A$。最后的这个迭代格式，实际上就是求方程数值解的“[牛顿法](<https://en.wikipedia.org/wiki/Newton%27s_method>)”，它具有二次收敛速度。

## 何必收敛

理论上来说，$\eqref{eq:iter-plus1}$、$\eqref{eq:iter-plus2}$两个改进确实能提高收敛速度，然而它们使得每一步线性递归的矩阵$A$变得跟$t$甚至$n$有关了，这其实会大大增加并行的复杂度，也不能利用“[简化形式](<https://spaces.ac.cn/archives/9783#%E7%AE%80%E5%8C%96%E5%BD%A2%E5%BC%8F>)”一节的对角化技巧来加速。另一方面，如果保持$\eqref{eq:rnn-iter}$这样的迭代格式，虽然有诸多效率上的好处，但收敛方面确实无法得到很好的保障。

难道这两者的矛盾就无法调和了吗？事实上，按照笔者的观点，最直接的做法是“别去管它”——借助非线性RNN导出了$\eqref{eq:rnn-iter}$后，就忘记原本的非线性RNN，将式$\eqref{eq:rnn-iter}$作为基本模型。也就是说，何必忧虑式$\eqref{eq:rnn-iter}$会不会收敛到原来的非线性RNN？直接将它作为新的出发点不好吗？梯度下降学到怎样的结果就是怎样的结果，如果梯度下降学到的结果是不收敛到原来的非线性RNN，那么就意味着不收敛到原来的RNN是更适合的。

抛开这一层思维束缚后，其实很多问题会变得豁然开朗起来。首先，即便是式$\eqref{eq:iter-plus2}$在理论上拥有非常好的收敛速度，但也是有条件的，而且在深度学习的背景下，要保证这些条件会显得很奢侈。换言之，即便是式$\eqref{eq:iter-plus2}$的收敛性也没有绝对保证，所以何必“五十步笑百步”去苛责式$\eqref{eq:rnn-iter}$？其次，将式$\eqref{eq:rnn-iter}$视为新的出发点后，我们可以将它单纯地理解为线性RNN的一种新用法，或者说解决线性RNN缺陷（比如线性RNN不是图灵完备的）的一个思路，这样操作性更强。

总的来说，不去管它的收敛性，似乎更能打破思维僵局，探索更一般的结果。

## 一般情形

前面的“长篇大论”，都只围绕着简单的非线性RNN也就是式$\eqref{eq:rnn}$进行讨论，对于更常用的LSTM、GRU，结果又如何呢？

以GRU为例，它原本的形式为
\begin{equation}\begin{aligned} z_{t} & = \sigma \left( W_{z} x_{t} + U_{z} h_{t - 1} + b_{z} \right) \\
r_{t} & = \sigma \left( W_{r} x_{t} + U_{r} h_{t - 1} + b_{r} \right) \\
\hat{h}_t & = \tanh \left( W_{h} x_{t} + U_{h} (r_t \circ h_{t - 1}) + b_{c} \right)\\
h_{t} & = \left(1 - z_{t}\right) \circ h_{t - 1} + z_{t} \circ \hat{h}_t \end{aligned}\end{equation}
初始阶段，所有门控都可以近似视为$\frac{1}{2}$，那么模仿式$\eqref{eq:rnn-approx}$有
\begin{equation}\begin{aligned}
h_{t} &\, = \left(1 - z_{t}\right) \circ h_{t - 1} + z_{t} \circ \hat{h}_t \\
&\, \approx \frac{1}{2} h_{t - 1} + \frac{1}{2} \hat{h}_t \\
&\, \approx \frac{1}{2} h_{t - 1} + \frac{1}{2} \left(\tanh ( W_{h} x_{t} + b_{c} ) + \frac{1}{2}U_{h} h_{t - 1}\right) \\
&\, = \frac{1}{2} \left(I + \frac{1}{2}U_{h}\right)h_{t - 1} + \frac{1}{2} \tanh ( W_{h} x_{t} + b_{c} ) \\
\end{aligned}\end{equation}
所以可以选取$A=\frac{1}{2} \left(I + \frac{1}{2}U_{h}\right)$，将GRU改写为迭代
\begin{equation}\begin{aligned} z_{t}^{(n)} & = \sigma \left( W_{z} x_{t} + U_{z} h_{t - 1}^{(n-1)} + b_{z} \right) \\
r_{t}^{(n)} & = \sigma \left( W_{r} x_{t} + U_{r} h_{t - 1}^{(n-1)} + b_{r} \right) \\
\hat{h}_t^{(n)} & = \tanh \left( W_{h} x_{t} + U_{h} (r_t^{(n)} \circ h_{t - 1}^{(n-1)}) + b_{c} \right)\\
h_{t}^{(n)} & = Ah_{t-1}^{(n)} - Ah_{t-1}^{(n - 1)} + \left(1 - z_{t}^{(n)}\right) \circ h_{t - 1}^{(n-1)} + z_{t}^{(n)} \circ \hat{h}_t^{(n)} \end{aligned}\end{equation}

总的来说，这种将非线性RNN变为线性RNN迭代的转换，从实践的角度来看，就是以非线性RNN为引，导出一种多层线性RNN的参数共享和组合方法，迭代了几次，那么就有几层线性RNN的计算量。这样自然而言就引发了一个思考：除非可以证明GRU、LSTM等非线性RNN有绝对的优势，否则直接叠加几层“线性RNN\+MLP”不好吗？

## 文章小结

本文简单探讨了非线性RNN的并行计算问题——通过数学物理中的“摄动”思想，我们可以将非线性RNN转化为线性RNN的迭代，从而利用线性RNN的可并行性来实现非线性RNN的并行。
