# 动量的新理解：逼近特征层面的梯度下降

> 作者：苏剑林 · 科学空间 · 2026-08-23
>
> 原文：<https://spaces.ac.cn/archives/11875>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

一个以动量为状态变量的优化器，基本形式如下：
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta \boldsymbol{M}_{t-1} + (1 - \beta) \boldsymbol{G}_t \\[4pt]
\boldsymbol{W}_t =&\, \phi(\boldsymbol{W}_{t-1}, \boldsymbol{M}_t, \boldsymbol{G}_t, t)
\end{aligned}\end{equation}
不同的优化器差别主要体现在更新函数$\phi$上，如SGDM、SignSGD、Muon等，新的探索也基本围绕$\phi$进行，因为第一个式子的动量太简单，简单到大家不认为它有什么修改空间。

而本文的主角正是动量。我们将给动量机制带来一个新的理解视角：动量不仅仅可以看作梯度的平均，它还可以被视为一个在线回归问题的解。由此出发，我们可以自然地引出最近的一些工作。

## 基本概念

跟Muon一样，本文主要考虑线性层的矩阵参数：设有线性层$\boldsymbol{Y}=\boldsymbol{X}\boldsymbol{W}$，其中$\boldsymbol{X}\in\mathbb{R}^{b\times d_{in}}$是输入，$\boldsymbol{W}\in\mathbb{R}^{d_{in}\times d_{out}}$是权重，$\boldsymbol{Y}\in\mathbb{R}^{b\times d_{out}}$是输出。损失函数记为$\mathcal{L}(\boldsymbol{Y})=\mathcal{L}(\boldsymbol{X}\boldsymbol{W})$，那么
\begin{equation}\boldsymbol{G} = \frac{\partial \mathcal{L}}{\partial\boldsymbol{W}} = \boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}\end{equation}
最简单的优化器是（参数层面的）梯度下降
\begin{equation}\boldsymbol{W}\quad\leftarrow\quad \boldsymbol{W} - \eta\frac{\partial \mathcal{L}}{\partial\boldsymbol{W}} = \boldsymbol{W} - \eta \boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}\end{equation}
然而，根据[《为什么我们偏爱各向同性？基于最速下降的理解》](<https://spaces.ac.cn/archives/11549>)的思想，我们认为参数本质上是模型的副产品，模型特征层面的变化，才是跟模型效果最相关的。理想情况下，我们希望实现特征层面的梯度下降
\begin{equation}\boldsymbol{Y}\quad\leftarrow\quad \boldsymbol{Y} - \eta\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}\label{eq:gd-y}\end{equation}
问题在于，$\boldsymbol{Y}$不是可以随意修改的变量，我们能直接修改的只有$\boldsymbol{W}$，所以只能设法通过修改$\boldsymbol{W}$来间接实现这一效果。

## 回归目标

怎么间接实现呢？设最终的更新规则为$\boldsymbol{W}\leftarrow\boldsymbol{W} - \eta\boldsymbol{\Phi}$，那么$\boldsymbol{Y}\leftarrow\boldsymbol{Y}-\eta \boldsymbol{X}\boldsymbol{\Phi}$，我们希望它能尽可能接近式$\eqref{eq:gd-y}$的效果，即希望$\boldsymbol{X}\boldsymbol{\Phi}\approx\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}$，所以我们考虑最小化
\begin{equation}\min_{\boldsymbol{\Phi}} \frac{1}{2}\left\Vert\boldsymbol{X}\boldsymbol{\Phi} - \frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}\right\Vert_F^2 + \frac{\lambda}{2}\Vert\boldsymbol{\Phi}\Vert_F^2 \label{eq:obj}\end{equation}
其中$\lambda > 0$是正则系数。事实上，这只不过是一个线性回归问题，可以直接解得
\begin{equation}\boldsymbol{\Phi}^* = (\boldsymbol{X}^{\top}\boldsymbol{X} + \lambda \boldsymbol{I})^{-1} \boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}\end{equation}
留意到$\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}$正好是$\boldsymbol{G}=\frac{\partial \mathcal{L}}{\partial\boldsymbol{W}}$，$(\boldsymbol{X}^{\top}\boldsymbol{X} + \lambda \boldsymbol{I})^{-1}$是基于输入数据的Preconditioner，为了综合不同batch的贡献，我们考虑对$\boldsymbol{X}^{\top}\boldsymbol{X}$和$\boldsymbol{X}^{\top}\frac{\partial \mathcal{L}}{\partial\boldsymbol{Y}}$都做EMA，这将得到一个SGDM变体：
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta \boldsymbol{M}_{t-1} + (1 - \beta) \boldsymbol{G}_t \\[4pt]
\boldsymbol{Z}_t =&\, \beta \boldsymbol{Z}_{t-1} + (1 - \beta) (\boldsymbol{X}_t^{\top}\boldsymbol{X}_t + \lambda \boldsymbol{I}) \\[4pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta \boldsymbol{Z}_t^{-1}\boldsymbol{M}_t
\end{aligned}\end{equation}

## 切换视角

现在我们知道，通过校正的梯度$(\boldsymbol{X}^{\top}\boldsymbol{X} + \lambda \boldsymbol{I})^{-1}\boldsymbol{G}$，可以实现特征层面的梯度下降，那么我们可以将它视为某种“更靠谱的梯度”，同理，相应的$\boldsymbol{Z}_t^{-1}\boldsymbol{M}_t$我们可以视为“更靠谱的动量”。

在这个视角之下，我们可以尝试用$\boldsymbol{Z}_t^{-1}\boldsymbol{M}_t$替换掉各种动量优化器中的动量，比如Muon：
\begin{equation}\begin{aligned}\newcommand{msign}{\mathop{\text{msign}}}
\boldsymbol{M}_t =&\, \beta \boldsymbol{M}_{t-1} + (1 - \beta) \boldsymbol{G}_t \\[4pt]
\boldsymbol{Z}_t =&\, \beta \boldsymbol{Z}_{t-1} + (1 - \beta) (\boldsymbol{X}_t^{\top}\boldsymbol{X}_t + \lambda \boldsymbol{I}) \\[4pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta \msign(\boldsymbol{Z}_t^{-1}\boldsymbol{M}_t)
\end{aligned}\end{equation}
这大致上就是[Newton\-Muon](<https://papers.cool/arxiv/2604.01472>)优化器。之所以说“大致”，是因为原始Newton\-Muon其实是使用了先校正后EMA的方案（即对$(\boldsymbol{X}^{\top}\boldsymbol{X} + \lambda \boldsymbol{I})^{-1}\boldsymbol{G}$做EMA作为动量），这样原则上可以省去一组状态变量。至于矩阵求逆，可以参考[《矩阵r次方根和逆r次方根的高效计算》](<https://spaces.ac.cn/archives/11175>)。

如果输入是各向同性的，那么我们可以期望$\boldsymbol{Z}_t=\sigma^2\boldsymbol{I}$，此时Newton\-Muon退化为Muon，换句话说，Muon可以理解为各向同性假设下的特征梯度下降，这再次回到了文章[《为什么我们偏爱各向同性？基于最速下降的理解》](<https://spaces.ac.cn/archives/11549>)的结论。此外，当$\lambda\to\infty$时，Newton\-Muon也退化为Muon，所以我们可以通过调节$\lambda$来调节它与Muon的近似程度。

## 增量更新

一个更普适且更可玩的做法是：不设法求解析解，而是直接用梯度下降来优化目标$\eqref{eq:obj}$！首先可以求得式$\eqref{eq:obj}$关于$\boldsymbol{\Phi}$的梯度是：
\begin{equation}(\boldsymbol{X}^{\top}\boldsymbol{X}+\lambda\boldsymbol{I})\boldsymbol{\Phi}-\boldsymbol{X}^{\top}\frac{\partial\mathcal{L}}{\partial\boldsymbol{Y}} = (\boldsymbol{X}^{\top}\boldsymbol{X}+\lambda\boldsymbol{I})\boldsymbol{\Phi}-\boldsymbol{G}\end{equation}
如果用梯度下降来更新$\boldsymbol{\Phi}$，我们将得到
\begin{equation}\begin{aligned}
\boldsymbol{\Phi}_t =&\, \boldsymbol{\Phi}_{t-1} - \gamma[(\boldsymbol{X}_t^{\top}\boldsymbol{X}_t+\lambda\boldsymbol{I})\boldsymbol{\Phi}_{t-1}-\boldsymbol{G}_t] \\[4pt]
=&\, (1-\gamma\lambda)\boldsymbol{\Phi}_{t-1} + \gamma[\boldsymbol{G}_t - (\boldsymbol{X}_t^{\top}\boldsymbol{X}_t)\boldsymbol{\Phi}_{t-1}]
\end{aligned}\end{equation}
其中$\gamma > 0$是这个“内层”优化的学习率。延续上一节的思想，如果说$\boldsymbol{Z}_t^{-1}\boldsymbol{M}_t$是某种更靠谱的动量，那么上式就是基于梯度下降思想推出来的“更靠谱的动量更新规则”！这便是[DeltaMomentum](<https://papers.cool/arxiv/2608.19491>)，它将Delta Rule引入到了动量中，与旧动量相比，相当于[GDN](<https://papers.cool/arxiv/2412.06464>)与Vanilla Linear Attention的区别，而$\boldsymbol{Z}_t^{-1}\boldsymbol{M}_t$则相当于[MesaNet](<https://papers.cool/arxiv/2506.05233>)。

当然，为了跑通DeltaMomentum，还需要留意一些细节，主要是关于$\boldsymbol{X}$的归一化和$\gamma$的选择，请大家自行看原论文。原论文的一些处理笔者也不是特别认可，建议大家看的时候斟酌参考。但不论如何，DeltaMomentum提供了一个动量机制的新的尝试方向，还不需要求逆等昂贵运算，值得细细回味。

## 相关工作

事实上，近段时间以来，基于输入来构建Preconditioner的探索越来越多，除了Newton\-Muon和DeltaMomentum外，此前谢天同学在[《特征空间的最速下降》](<https://zhuanlan.zhihu.com/p/2027475941614531067>)还发布过另一个稍微不同的结果，用我们这里的记号，大致上是
\begin{equation}\begin{aligned}
\boldsymbol{M}_t =&\, \beta \boldsymbol{M}_{t-1} + (1 - \beta) \boldsymbol{G}_t \\[4pt]
\boldsymbol{Z}_t =&\, \beta \boldsymbol{Z}_{t-1} + (1 - \beta) (\boldsymbol{X}_t^{\top}\boldsymbol{X}_t + \lambda \boldsymbol{I}) \\[4pt]
\boldsymbol{W}_t =&\, \boldsymbol{W}_{t-1} - \eta \boldsymbol{Z}_t^{-1/2} \msign(\boldsymbol{Z}_t^{-1/2}\boldsymbol{M}_t)
\end{aligned}\end{equation}
即$\boldsymbol{Z}_t$分别以$-1/2$次幂的形式作用于$\msign$内外，其推导原理正是特征层面的谱范数约束下的最速下降，跟本文的中心思想一致。

类似的工作可能还有，但笔者着实是想不起来其他了，欢迎读者在评论区补充（注：有读者补充了[DoPr](<https://papers.cool/arxiv/2606.06418>)）。从实验结果上看，有些尝试确实也取得了正面的结果，比如Newton\-Muon在Speedrun上有着比Muon更好的表现（参考[这里](<https://github.com/zhehangdu/Newton-Muon>)），所以整体看来，这个方向还是有一定的可取之处。

## 延伸思考

然而，这种面向输入的Preconditioner设计，也有一些让人不满意的地方。

第一是实现层面的：它需要我们在前向计算的时候记录自相关矩阵$\boldsymbol{X}_t^{\top}\boldsymbol{X}_t$，然后传入到优化器中，这就打破了优化器的独立性——优化器不再是拿到梯度就能干活的“黑盒”，需要耦合模型定义层面的细节，实现上很难说得上优雅，更不用说额外引入的通信和计算开销了。

第二是理论层面的：以实际输入来构建Preconditioner，可能让优化器的探索受限于输入所张成的子空间，对其余方向的探索不足，从而效果不佳。比较简单的缓解策略是增加$\lambda$来降低Precondition力度——由于$\lambda\to\infty$时相当于没有Precondition，所以选择适当的$\lambda$，总有机会把效果调优，但这也多了一个需要仔细调节的超参数。

总的来说，这个方向仍处于方兴未艾的阶段——雏形已现，但工程与理论上的问题都还没有完全解决，问题并不比结论少。

## 文章小结

本文从让“参数梯度下降”逼近“特征梯度下降”的思想出发，将动量重新诠释为一个在线回归问题的解，由此引出了一些相关工作。有趣的是，这条路线跟线性注意力从Vanilla到DeltaNet再到MesaNet的演变极其一致，两者看起来有颇多可以相互借鉴的地方。
