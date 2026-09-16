# 隐藏在动量中的梯度累积：少更新几步，效果反而更好？

> 作者：苏剑林 · 科学空间 · 2021-08-24
>
> 原文：<https://spaces.ac.cn/archives/8634>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们知道，梯度累积是在有限显存下实现大batch\_size训练的常用技巧。在之前的文章[《用时间换取效果：Keras梯度累积优化器》](<https://spaces.ac.cn/archives/6794>)中，我们就简单介绍过梯度累积的实现，大致的思路是新增一组参数来缓存梯度，最后用缓存的梯度来更新模型。美中不足的是，新增一组参数会带来额外的显存占用。

这几天笔者在思考优化器的时候，突然意识到：梯度累积其实可以内置在带动量的优化器中！带着这个思路，笔者对优化了进行了一些推导和实验，最后还得到一个有意思但又有点反直觉的结论：少更新几步参数，模型最终效果可能会变好！

> <strong>注：本文下面的结果，几乎原封不动且不加引用地出现在了Google的论文[《Combined Scaling for Zero\-shot Transfer Learning》](<https://papers.cool/arxiv/2111.10050>)中。</strong>

## SGDM

在正式讨论之前，我们定义函数
\begin{equation}\chi_{t/k} = \left\{ \begin{aligned}&1,\quad t \equiv 0\,(\text{mod}\, k) \\
&0,\quad t \not\equiv 0\,(\text{mod}\, k)
\end{aligned}\right.\end{equation}
也就是说，$t$是一个整数，当它是$k$的倍数时，$\chi_{t/k}=1$，否则$\chi_{t/k}=0$，这其实就是一个$t$能否被$k$整除的示性函数。在后面的讨论中，我们将反复用到这个函数。

现在，我们来讨论带动量的梯度下降SGDM，它的一般形式为
\begin{equation}\left\{\begin{aligned}
m_t =&\, \beta m_{t-1} + \left(1 - \beta\right) g_t\\
\theta_t =&\, \theta_{t-1} - \alpha_t m_t
\end{aligned}\right.\end{equation}
这里$g_t$是第$t$步的梯度，$\theta$是参数，$\beta$是滑动系数。假设我们累积$k$步梯度，那么实际上就是每$k$步才更新一次，然后每次更新都使用$k$步的梯度平均，即
\begin{equation}\left\{\begin{aligned}
m_{kt} =&\, \beta m_{k(t-1)} + \left(1 - \beta\right) \frac{1}{k}\sum_{i=1}^k g_{k(t-1) + i}\\
\theta_{kt} =&\, \theta_{k(t-1)} - \alpha_{kt} m_{kt}
\end{aligned}\right.\end{equation}
我们可以将动量$m$的更新分拆开来：
\begin{equation}\begin{aligned}
m_{k(t-1)+1} &\,= \beta m_{k(t-1)} + \frac{1}{k}\left(1 - \beta\right)g_{k(t-1) + 1}\\
m_{k(t-1)+2} &\,= m_{k(t-1) + 1} + \frac{1}{k}\left(1 - \beta\right)g_{k(t-1) + 2} \\
m_{k(t-1)+3} &\,= m_{k(t-1) + 2} + \frac{1}{k}\left(1 - \beta\right)g_{k(t-1) + 3} \\
&\,\,\,\vdots \\
m_{kt} &\,= m_{kt-1} + \frac{1}{k}\left(1 - \beta\right)g_{kt} \\
\end{aligned}\label{eq:m-part}\end{equation}
或者写成一个通式：
\begin{equation}m_t = \big[(\beta - 1)\chi_{(t - 1)/k} + 1\big] m_{t-1} + \frac{1}{k}\left(1 - \beta\right) g_t\end{equation}
同理，参数$\theta$的更新也可以分拆开：
\begin{equation}\begin{aligned}
\theta_{k(t-1)+1} &\,= \theta_{k(t-1)}\\
\theta_{k(t-1)+2} &\,= \theta_{k(t-1) + 1}\\
&\,\,\,\vdots \\
\theta_{kt-1} &\,= \theta_{kt-2}\\
\theta_{kt} &\,= \theta_{kt-1} - \alpha_{kt} m_{kt} \\
\end{aligned}\end{equation}
或者写成一个通式：
\begin{equation}\theta_t = \theta_{t-1} - \chi_{t/k} \alpha_t m_t \end{equation}
所以，带动量的梯度下降，如果要累积$k$步梯度，那么只需要按照如下方式更新：
\begin{equation}\left\{\begin{aligned}
m_t =&\, \big[(\beta - 1)\chi_{(t - 1)/k} + 1\big] m_{t-1} + \frac{1}{k}\left(1 - \beta\right) g_t\\
\theta_t =&\, \theta_{t-1} - \chi_{t/k} \alpha_t m_t
\end{aligned}\right.\end{equation}
而不需要引入新的一组参数。

## Adam

对于Adam来说，它的更新公式如下：
\begin{equation}\left\{\begin{aligned}
m_t =&\, \beta_1 m_{t-1} + \left(1 - \beta_1\right) g_t\\
v_t =&\, \beta_2 v_{t-1} + \left(1 - \beta_2\right) g_t^2\\
\hat{m}_t =&\, m_t\left/\left(1 - \beta_1^t\right)\right.\\
\hat{v}_t =&\, v_t\left/\left(1 - \beta_2^t\right)\right.\\
\theta_t =&\, \theta_{t-1} - \alpha_t \hat{m}_t\left/\sqrt{\hat{v}_t + \epsilon}\right.
\end{aligned}\right.\end{equation}
动量$m$的处理方式同前述SGDM一致，所以关键在于二阶矩$v$。按照定义，在梯度累积$k$步的情况下，$v$的更新公式为
\begin{equation}
v_{kt} = \beta_2 v_{k(t-1)} + \left(1 - \beta_2\right) \left(\frac{1}{k}\sum_{i=1}^k g_{k(t-1) + i}\right)^2\end{equation}
很遗憾，由于平方的存在，$v$无法做到像式$\eqref{eq:m-part}$那样的分拆，所以严格来说，不新增一组缓存参数是无法实现Adam的梯度累积的。

但是，如果我们假设“平均的平方”可以由“平方的平均”来估计，即假设
\begin{equation}\left(\frac{1}{k}\sum_{i=1}^k g_{k(t-1) + i}\right)^2\sim \frac{1}{k}\sum_{i=1}^k g_{k(t-1) + i}^2\end{equation}
这里的$\sim$代表有比较紧密的线性相关关系，但不一定是近似相等，比如可以差一个倍数。那么我们依然可以像$m$那样对$v$的公式进行修改：
\begin{equation}v_t = \big[(\beta_2 - 1)\chi_{(t - 1)/k} + 1\big] v_{t-1} + \frac{1}{k}\left(1 - \beta_2\right) g_t^2\end{equation}
综合起来，就得到修改后的Adam公式：
\begin{equation}\left\{\begin{aligned}
m_t =&\, \big[(\beta_1 - 1)\chi_{(t - 1)/k} + 1\big] m_{t-1} + \frac{1}{k}\left(1 - \beta_1\right) g_t\\
v_t =&\, \big[(\beta_2 - 1)\chi_{(t - 1)/k} + 1\big] v_{t-1} + \frac{1}{k}\left(1 - \beta_2\right) g_t^2\\
\hat{m}_t =&\, m_t\left/\left(1 - \beta_1^{t/k}\right)\right.\\
\hat{v}_t =&\, v_t\left/\left(1 - \beta_2^{t/k}\right)\right.\\
\theta_t =&\, \theta_{t-1} - \chi_{t/k}\alpha_t \hat{m}_t\left/\sqrt{\hat{v}_t + \epsilon}\right.
\end{aligned}\right.\end{equation}
笔者实验后发现，这样修改后的Adam，确实能起到跟梯度累积相似的效果。

## 结论反思

总的来说，在上面的两个修改版优化器中，主要包含两个改动：

> 1、修改$m$和$v$的更新公式；
> 
> 
> 
> 2、参数改为每$k$步更新一次（或者学习率由$\alpha_t$改为$\chi_{t/k}\alpha_t$）。

其中$m,v$的更新公式改为了（不失一般性，以$m$为例）：
\begin{equation}m_t = \big[(\beta - 1)\chi_{(t - 1)/k} + 1\big] m_{t-1} + \frac{1}{k}\left(1 - \beta\right) g_t\end{equation}
如果要对应上原来的$m_t = \beta m_{t-1} + (1 - \beta) g_t$的格式，我们猜测上述迭代可能相当于将$\beta$换成了$\tilde{\beta}=1 - \frac{1}{k}(1-\beta)$。

事实上确实如此，我们可以证明将$\beta$换成$\tilde{\beta}=1 - \frac{1}{k}(1-\beta)$后的滑动平均动量，近似于原始$\beta$时累积$k$步梯度的滑动平均动量：
\begin{equation}\begin{aligned}
m_{kt} =&\, \tilde{\beta}m_{kt-1} + (1 - \tilde{\beta})g_{kt} \\
=&\, \tilde{\beta}^2 m_{kt-2} + \tilde{\beta}(1 - \tilde{\beta})g_{kt - 1} + (1 - \tilde{\beta})g_{kt}\\
=&\,\cdots\\
=&\, \tilde{\beta}^k m_{k(t-1)} + (1 - \tilde{\beta})\sum_{i=1}^k \tilde{\beta}^{i-1} g_{kt-i+1}
\end{aligned}\end{equation}
由于$\tilde{\beta}$通常相当接近于1，所以我们近似认为$\tilde{\beta}^{i-1}\approx 1, \forall i=1,2,\cdots,k$，并且
\begin{equation}\tilde{\beta}^k = (1 - (1 - \tilde{\beta}))^k \approx 1 - k(1 - \tilde{\beta}) = \beta\end{equation}
所以有近似
\begin{equation}\begin{aligned}
m_{kt} \approx &\, \beta m_{k(t-1)} + (1 - \tilde{\beta})\sum_{i=1}^k g_{kt-i+1} \\
= &\, \beta m_{k(t-1)} + (1 - \beta) \frac{1}{k}\sum_{i=1}^k g_{kt-i+1}
\end{aligned}\end{equation}
这就是梯度累积形式。

所以，如果你面临着batch\_size比较小而导致性能不好的问题，但又不想自己实现梯度累积或者修改优化器，那么可以尝试如下操作：

> 1、将所有$\beta$参数改为$1-\frac{1}{k}(1-\beta)$；
> 
> 
> 
> 2、学习率乘以$\chi_{t/k}$，使得参数每$k$步才更新一次。

有意思的是，笔者发现第2点比第1点更重要：哪怕我们不修改$\beta$，直接将参数更新频率改为每$k$步更新，效果也有所改善（前提是batch\_size小是模型当前瓶颈）。这就是标题所提到的“反直觉”现象：

> 啥都不用改，少更新几次，效果反而更好了。

## 文章小结

本文介绍了笔者在调试模型中的一个发现——梯度累积隐藏在优化器的动量中，然后对此作了进一步的简单分析和实验，发现通过调整滑动系数和学习率也能达到近似的梯度累积效果，进而发现“少更新几次，效果反而更好”的反直觉现象。
