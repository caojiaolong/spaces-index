# 简述无偏估计和有偏估计

> 作者：苏剑林 · 科学空间 · 2019-06-19
>
> 原文：<https://spaces.ac.cn/archives/6747>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

对于大多数读者（包括笔者）来说，他们接触到的第一个有偏估计量，应该是方差
\begin{equation}\hat{\sigma}^2_{\text{有偏}} = \frac{1}{n}\sum_{i=1}^n \left(x_i - \hat{\mu}\right)^2,\quad \hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i\label{eq:youpianfangcha}\end{equation}
然后又了解到对应的无偏估计应该是
\begin{equation}\hat{\sigma}^2_{\text{无偏}} = \frac{1}{n-1}\sum_{i=1}^n \left(x_i - \hat{\mu}\right)^2\label{eq:wupianfangcha}\end{equation}
在很多人的眼里，公式$\eqref{eq:youpianfangcha}$才是合理的，怎么就有偏了？公式$\eqref{eq:wupianfangcha}$将$n$换成反直觉的$n-1$，反而就无偏了？

下面试图用尽量清晰的语言讨论一下无偏估计和有偏估计两个概念。

假如，我们可以采样无穷无尽的样本，那么理论上下面的估计就是精确的：
\begin{equation}\begin{aligned}\sigma^2 =&\, \mathbb{E}\left[(x - \mu)^2\right]=\lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^n \left(x_i - \hat{\mu}\right)^2\\
\mu =&\, \mathbb{E}[x]=\lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^n x_i\end{aligned}\end{equation}
这也可以理解为，当样本数趋于无穷时，有偏估计和无偏估计等价。

问题是，我们实际计算中，只能采样一批样本来计算，也就是说$n$是一个固定的数字，比如我们随机梯度下降时，用一个batch的样本的平均梯度，来作为整体样本的梯度估计。另一方面，我们也不是估计一次就完事了，我们可能会估计很多次，即首先采样$n$个样本，算一次得到$\hat{\mu}_{1}$和$\hat{\sigma}^2_{\text{有偏},1}$，再随机采样$n$个样本算一次得到$\hat{\mu}_{2}$和$\hat{\sigma}^2_{\text{有偏},2}$，依此类推得到$\left(\hat{\mu}_{3},\hat{\sigma}^2_{\text{有偏},3}\right),\left(\hat{\mu}_{4},\hat{\sigma}^2_{\text{有偏},4}\right),\dots$，我们想知道的是：
\begin{equation}\begin{aligned}\mu &\xlongequal{?}\mathbb{E}\left[\hat{\mu}\right] = \lim_{N\to\infty}\frac{1}{N}\sum_{i=1}^N \hat{\mu}_{i}\\
\sigma^2 &\xlongequal{?}\mathbb{E}\left[\hat{\sigma}^2_{\text{有偏}}\right]=\lim_{N\to\infty}\frac{1}{N}\sum_{i=1}^N \hat{\sigma}^2_{\text{有偏},i}
\end{aligned}\end{equation}
也就是说，“有限平均”的“无限平均”，是否等于我们最终要求的平均？

前面已经说了，本文着重讨论和理解而不是推导，所以不打算完成一般的证明。在这里，我们只用一个最简单的例子：假设$n=2$，即用$\eqref{eq:youpianfangcha}$或$\eqref{eq:wupianfangcha}$进行估计时，每次只采样两个样本。这时候，我们要回答的问题是：
\begin{equation}\begin{aligned}\mu &\xlongequal{?}\mathbb{E}_{x_1,x_2}\left[\frac{x_1 + x_2}{2}\right]\\
\sigma^2 &\xlongequal{?}\mathbb{E}_{x_1,x_2}\left[\frac{1}{2}\left(\left(x_1 - \frac{x_1 + x_2}{2}\right)^2 + \left(x_2 - \frac{x_1 + x_2}{2}\right)^2\right)\right]
\end{aligned}\end{equation}
由于这种情况比较简单，我们可以很容易验证，比如
\begin{equation}\mathbb{E}_{x_1,x_2}\left[\frac{x_1 + x_2}{2}\right] = \mathbb{E}_{x_1}\left[\frac{x_1}{2}\right] + \mathbb{E}_{x_2}\left[\frac{x_2}{2}\right] = \frac{\mu}{2} + \frac{\mu}{2} = \mu\end{equation}
所以用两个样本进行的均值的估计，就是均值的无偏估计了，多个样本也是如此。

但是方差却不一样：
\begin{equation}\begin{aligned}&\mathbb{E}_{x_1, x_2} \left[\frac{1}{2}\left(\left(x_1 - \frac{x_1 + x_2}{2}\right)^2 + \left(x_2 - \frac{x_1 + x_2}{2}\right)^2\right)\right]\\
=&\frac{1}{4}\mathbb{E}_{x_1, x_2} \left[\left(x_1 - x_2\right)^2\right]\\
=&\frac{1}{4}\mathbb{E}_{x_1, x_2} \left[x_1^2 + x_2^2 - 2 x_1 x_2\right]\\
=&\frac{1}{4}\Big(\mathbb{E}_{x_1} \left[x_1^2\right] + \mathbb{E}_{x_2} \left[x_2^2\right] - 2 \mathbb{E}_{x_1} \left[x_1\right] \mathbb{E}_{x_2} \left[x_2\right]\Big)\\
=&\frac{1}{4}\Big(2\mathbb{E}_{x} \left[x^2\right] - 2 \mu^2\Big)\\
=&\frac{1}{2}\Big(\mathbb{E}\left[x^2\right] - \mu^2\Big)
\end{aligned}\end{equation}
注意方差的准确表达式应该是$\mathbb{E}\left[x^2\right] - \mu^2$，所以两个样本的$\hat{\sigma}^2_{\text{有偏}}$是对方差的一个有偏估计，在重复估计取平均后，它仍然低估了真实方差。如果对$n$个样本的估计进行分析，那么前面的因子是$(n-1)/n$。所以，只需要乘以$n/(n-1)$，就得到方差的无偏估计，最终结果就是$\eqref{eq:wupianfangcha}$。

直观来看，用有限样本的式$\eqref{eq:youpianfangcha}$来估计方差，由于样本少了，波动也会变小，所以方差估计也会偏小，这就是所谓的有偏。极端情况下，如果只采样一个样本进行估计呢？用式$\eqref{eq:youpianfangcha}$估计出来的方差就是0了，不管怎么重复实验，结果还是0，我们总不能说整批样本的方差一定就是0吧？这便是有偏估计的最简单例子。

从理论上，有偏估计的产生机制也很容易理解，因为方差的计算公式等价于：
\begin{equation}\mathbb{E}\left[x^2\right] - \mathbb{E}\left[x\right]^2\end{equation}
其中期望运算$\mathbb{E}$是一个线性算子，所以上式关于$\mathbb{E}$是非线性的（二次的，即$\mathbb{E}\left[x\right]^2$这一行），只要一个估计量关于期望运算$\mathbb{E}$是非线性的（注意：这里强调的是关于期望运算的非线性，不是随机变量的非线性），直接有限样本估计就很可能产生偏差，因为线性运算与线性运算的复合，依然还是线性运算，而线性运算与非线性运算的复合，却不是原来的非线性运算了。

并不是所有的有偏估计都可以像方差一样，简单将$n$换成$n-1$就变为无偏估计了。一般情形下，我们想要估计的量，连估计本身都很难，更不要说有偏还是无偏了，所以要对一般的估计量消除偏差，都得具体问题具体分析了。
