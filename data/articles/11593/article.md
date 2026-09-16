# Adam优化器的最优超参数是β1=β2 ？

> 作者：苏剑林 · 科学空间 · 2026-02-04
>
> 原文：<https://spaces.ac.cn/archives/11593>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近笔者刷到论文[《Why Adam Works Better with β1=β2: The Missing Gradient Scale Invariance Principle》](<https://papers.cool/arxiv/2601.21739>)，顾名思义，它声称Adam在$\beta_1=\beta_2$时表现更优。经同事提醒，去年论文[《In Search of Adam's Secret Sauce》](<https://papers.cool/arxiv/2505.21829>)也表达了相同的观点。无独有偶，昨天刚出来的[《The Effect of Mini\-Batch Noise on the Implicit Bias of Adam》](<https://papers.cool/arxiv/2602.01642>)也有类似发现。

\begin{equation}\text{Adam}\color{skyblue}{\text{W}}:=\left\{\begin{aligned}
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\
&\boldsymbol{u}_t =\hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.\\
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})
\end{aligned}\right.\end{equation}

众多论文都指向了$\beta_1=\beta_2$，它有什么理论上的好处呢？本文我们来学习一下相关推导。

## 在线估计

按照时间顺序，我们先来看[《In Search of Adam's Secret Sauce》](<https://papers.cool/arxiv/2505.21829>)，这篇论文的故事线应该是通过实验发现Adam在$\beta_1=\beta_2$时的最优解跟无约束时的最优解非常接近，然后尝试为这个结果构建一个理论解释——当$\beta_1=\beta_2=\beta$时，$\hat{\boldsymbol{m}}_t$和$\hat{\boldsymbol{v}}_t$可以视为梯度的一阶矩和二阶矩的一个在线估计。具体来说，对$\hat{\boldsymbol{m}}_t$、$\hat{\boldsymbol{v}}_t$展开，我们得到
\begin{equation}\hat{\boldsymbol{m}}_t = \frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} \boldsymbol{g}_k,\qquad \hat{\boldsymbol{v}}_t = \frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} \boldsymbol{g}_k^2\end{equation}
容易证明系数之和$\frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k}$恒等于1，所以它们分别是$\boldsymbol{g}_t$和$\boldsymbol{g}_t^2$的同一种加权平均，所以分别具有一阶矩和二阶矩的意义。进一步地，我们有
\begin{equation}\begin{aligned}
\hat{\boldsymbol{v}}_t =&\, \frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} (\hat{\boldsymbol{m}}_t + \boldsymbol{g}_k - \hat{\boldsymbol{m}}_t)^2 \\
=&\, \underbrace{\frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} \hat{\boldsymbol{m}}_t^2}_{\hat{\boldsymbol{m}}_t^2} + \frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} (\boldsymbol{g}_k - \hat{\boldsymbol{m}}_t)^2 + \underbrace{\frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} 2\hat{\boldsymbol{m}}_t (\boldsymbol{g}_k - \hat{\boldsymbol{m}}_t)}_{\boldsymbol{0}} \\
=&\, \hat{\boldsymbol{m}}_t^2 + \frac{1-\beta}{1-\beta^t}\sum_{k=1}^t \beta^{t-k} (\boldsymbol{g}_k - \hat{\boldsymbol{m}}_t)^2 \\
\end{aligned}\end{equation}
最后一项显然具有方差的形式，所以我们不妨记为$\hat{\boldsymbol{\sigma}}_t^2$，即$\hat{\boldsymbol{v}}_t=\hat{\boldsymbol{m}}_t^2+\hat{\boldsymbol{\sigma}}_t^2$，这正是二阶矩与均值、方差之间的关系。特别说明，上述运算中涉及到向量的加减乘除和乘方，都是Element\-wise意义下的运算，即逐分量对应地加减乘除或乘方，输出仍是一个向量。

## 信噪感知

在这些新记号之下，Adam的更新量可以写成（简单起见，假设$\epsilon=0$）：
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}}\boldsymbol{u}_t = \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t}} =  \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{m}}_t^2+\hat{\boldsymbol{\sigma}}_t^2}} =  \frac{\sign(\hat{\boldsymbol{m}}_t)}{\sqrt{1 +\hat{\boldsymbol{\sigma}}_t^2/\hat{\boldsymbol{m}}_t^2}}\end{equation}
这个形式的更新量有一些好处。首先最明显的是，它的每个分量都是有界的，被约束在$[-1, 1]$内，所以我们不用担心更新量爆炸的问题。其次，$\hat{\boldsymbol{\sigma}}_t^2/\hat{\boldsymbol{m}}_t^2$正好具有信噪比倒数的形式，所以它还可以理解为是“信噪比感知的最速下降”。

根据我们在[《流形上的最速下降：1\. SGD \+ 超球面》](<https://spaces.ac.cn/archives/11196>)的推导，$\sign(\hat{\boldsymbol{m}}_t)$可以视为如下优化问题的解
\begin{equation}\max_{\boldsymbol{u}} \langle\hat{\boldsymbol{m}}_t,\boldsymbol{u}\rangle\qquad \text{s.t.}\qquad \Vert\boldsymbol{u}\Vert_{\infty} = 1\end{equation}
其中$\Vert\boldsymbol{u}\Vert_{\infty} = 1$代表$\boldsymbol{u}$的分量绝对值最大是1。如果我们将$\hat{\boldsymbol{m}}_t$视为更准确的梯度，那么$\sign(\hat{\boldsymbol{m}}_t)$就是无穷范数下的最速下降方向。但现在这个界是静态的，我们可以合理地认为，如果梯度波动小（信噪比大），那么该区域比较平坦，可以适当增大更新量，反之则要缩小，所以根据信噪比给每个分量构建动态边界$\frac{1}{\sqrt{1 +\hat{\boldsymbol{\sigma}}_t^2/\hat{\boldsymbol{m}}_t^2}}$，更能体现出自适应学习的效果，而此时的最速下降方向，正好是
\begin{equation}\max_{\boldsymbol{u}} \langle\hat{\boldsymbol{m}}_t,\boldsymbol{u}\rangle\quad \text{s.t.}\quad |\boldsymbol{u}| \leq \frac{1}{\sqrt{1 +\hat{\boldsymbol{\sigma}}_t^2/\hat{\boldsymbol{m}}_t^2}} \qquad\Rightarrow\qquad
\boldsymbol{u}^* = \frac{\sign(\hat{\boldsymbol{m}}_t)}{\sqrt{1 +\hat{\boldsymbol{\sigma}}_t^2/\hat{\boldsymbol{m}}_t^2}}\end{equation}
这里的绝对值$||$和小于等于$\leq$都是Element\-wise的。

## 一阶展开

再来看[《Why Adam Works Better with β1=β2: The Missing Gradient Scale Invariance Principle》](<https://papers.cool/arxiv/2601.21739>)，它将Adam连续化成ODE来处理，然而，对于Mini\-Batch优化来说，梯度噪声是不可忽略的，连续化成SDE还说得过去，ODE实在不够科学，因此笔者认为这篇论文的出发点就非常勉强。

按照原论文的思想，笔者对证明过程稍微做了一些调整。我们将$\hat{\boldsymbol{v}}_t$中每个$\boldsymbol{g}_k$都写成$\hat{\boldsymbol{m}}_t + (\boldsymbol{g}_k - \hat{\boldsymbol{m}}_t)$，然后视$\boldsymbol{g}_k - \hat{\boldsymbol{m}}_t$为小量，做一阶展开得
\begin{equation}\begin{aligned}
\hat{\boldsymbol{v}}_t =&\, \frac{1-\beta_2}{1-\beta_2^t}\sum_{k=1}^t \beta_2^{t-k} (\hat{\boldsymbol{m}}_t + \boldsymbol{g}_k - \hat{\boldsymbol{m}}_t)^2 \\
\approx &\, \frac{1-\beta_2}{1-\beta_2^t}\sum_{k=1}^t \beta_2^{t-k} \hat{\boldsymbol{m}}_t^2  + \frac{1-\beta_2}{1-\beta_2^t}\sum_{k=1}^t \beta_2^{t-k} 2\hat{\boldsymbol{m}}_t (\boldsymbol{g}_k - \hat{\boldsymbol{m}}_t) \\
\approx &\, \hat{\boldsymbol{m}}_t^2  + 2\hat{\boldsymbol{m}}_t \left(\frac{1-\beta_2}{1-\beta_2^t}\sum_{k=1}^t \beta_2^{t-k} \boldsymbol{g}_k - \hat{\boldsymbol{m}}_t\right) \\
\end{aligned}\end{equation}
然后希望$\hat{\boldsymbol{v}}_t$应该尽量接近$\hat{\boldsymbol{m}}_t^2$，所以一阶项应该为零，从而得出$\beta_2 = \beta_1$。为什么要让$\hat{\boldsymbol{v}}_t$接近$\hat{\boldsymbol{m}}_t^2$呢？正是希望更新量$\boldsymbol{u}_t = \hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$能更接近$\sign(\hat{\boldsymbol{m}}_t)$，因为$\sign$是有界的，所以它更能抵御$\boldsymbol{g}_t$的尺度变化带来的扰动，提高训练稳定性。

需要指出的是，这里的证明相比原文是极度简化的，但已经抓住了它的核心思想，并进行了修正。原论文先连续化成ODE，这一点本身就有一定的事实错误，然后做一些不严谨的近似处理，最后得到一个$\boldsymbol{u}_t = \sign(\boldsymbol{g}_t)(1 + \cdots)$的展开式，这精度还不如我们这里直接在$\hat{\boldsymbol{m}}_t$展开要靠谱。

## 双重优化

这两篇论文有一个思想是共通的，那就是让$\boldsymbol{u}_t$有界，以提高训练的稳定性。这启发笔者反过来思考：假设给定$\beta_1$，那么$\beta_2$取什么值，才能让$|\boldsymbol{u}_t|$尽可能小呢？由于$\boldsymbol{u}_t = \hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$，直观来想就应该让$\hat{\boldsymbol{v}}_t$尽可能大，这启发笔者思考如下的双重优化问题：
\begin{equation}\max_{\beta_2} \min_{\boldsymbol{g}_1,\cdots,\boldsymbol{g}_t}\underbrace{\frac{1-\beta_2}{1-\beta_2^t}\sum_{k=1}^t \beta_2^{t-k} \boldsymbol{g}_k^2}_{\hat{\boldsymbol{v}}_t},\qquad \text{s.t.}\qquad  \frac{1-\beta_1}{1-\beta_1^t}\sum_{k=1}^t \beta_1^{t-k} \boldsymbol{g}_k = \hat{\boldsymbol{m}}_t\end{equation}
这里的写法并不是特别严谨，总之大家逐分量去理解就行，目标中对$\boldsymbol{g}_1,\cdots,\boldsymbol{g}_t$取$\min$是不能去掉的，它代表着所选的$\beta_2$在任意梯度序列下都能尽可能优。这个优化问题看起来复杂，实际上不难，用柯西不等式逐层求解就行。先求内层的最小化问题，我们有
\begin{equation}\sum_{k=1}^t \frac{p_k^2}{q_k}\times \sum_{k=1}^t q_k \boldsymbol{g}_k^2 \geq  \left(\sum_{k=1}^t p_k \boldsymbol{g}_k\right)^2 = \hat{\boldsymbol{m}}_t^2\end{equation}
其中$p_k = \frac{1-\beta_1}{1-\beta_1^t}\beta_1^{t-k},q_k = \frac{1-\beta_2}{1-\beta_2^t}\beta_2^{t-k}$。这样一来，内层最小化的结果就是$\frac{\hat{\boldsymbol{m}}{}_t^2}{\sum_{k=1}^t p_k^2/q_k}$，想要求它的最大值，就是要求$\sum_{k=1}^t p_k^2/q_k$的最小值，再次利用柯西不等式
\begin{equation}\sum_{k=1}^t p_k^2/q_k = \sum_{k=1}^t q_k \times \sum_{k=1}^t p_k^2/q_k \geq \left(\sum_{k=1}^t p_k\right)^2 = 1\end{equation}
等号在$q_k=p_k$取到，这意味着$\beta_2 = \beta_1$。简而言之，取$\beta_2=\beta_1$具有最佳的训练稳定性，能够保证$\hat{\boldsymbol{v}}_t\geq \hat{\boldsymbol{m}}_t^2$，这一次我们是从双重优化目标中直接求解出来的，而不是已知$\beta_1=\beta_2$后推演出来的。

## 相关工作

可能有些读者会疑惑，Adam的论文[《Adam: A Method for Stochastic Optimization》](<https://papers.cool/arxiv/1412.6980>)推荐的默认值是$\beta_1=0.9,\beta_2=0.999$，以及此前我们在[《从Hessian近似看自适应学习率优化器》](<https://spaces.ac.cn/archives/10588>)的分析也指出应该$\beta_2 > \beta_1$，这些结论之间不就相互矛盾了吗？

并非如此，它们只是在不同大背景下的不同结论。早年我们训练的都只能算是小模型、小Batch Size，每一步的噪声会比较大，设置更大的$\beta_2$能让$\hat{\boldsymbol{v}}_t$变化更缓慢一些，从而让Adam更像SGD，邻近步的更新可以线性叠加，以达到进一步降噪的效果。论文[《The Effect of Mini\-Batch Noise on the Implicit Bias of Adam》](<https://papers.cool/arxiv/2602.01642>)也有类似发现，小Batch Size时$\beta_1 < \beta_2$，大Batch Size则让它们更接近。

至于[《从Hessian近似看自适应学习率优化器》](<https://spaces.ac.cn/archives/10588>)，则是在理想的终点附近做近似分析的结果。然而，现在的情况是，我们要训练的模型越来越大，Batch Size也越来越大，我们会发现，一个模型训练了一个月，Loss感觉还不错了，但如果我们算力、数据充足，那么再训练一个月、两个月，Loss其实还能再降，也就是说不管我们训练一个月还是两个月，模型都离真正的最优点还远着。

在这个新背景下，我们训练模型的核心逻辑变成了“稳中求快”，“稳”是大前提，我们相信只要能稳定地、持续地训练下去，就一定能得到越来越好的效果，而$\beta_1=\beta_2$能够让更新量有界，正好满足了我们对“稳”的期望。事实上，在LLM时代，Adam的默认参数已经逐渐从$\beta_1=0.9,\beta_2=0.999$切换到了$\beta_1=0.9,\beta_2=0.95$，更接近$\beta_1=\beta_2$，也佐证了我们的推论。

## 文章小结

在这篇文章中，我们对Adam优化器的$\beta_1,\beta_2$参数做了一些分析，从稳定性角度表明$\beta_1=\beta_2$通常是一个较优的选择，它可以理解为信噪比感知下的最速下降。
