# 熵不变性Softmax的一个快速推导

> 作者：苏剑林 · 科学空间 · 2022-04-11
>
> 原文：<https://spaces.ac.cn/archives/9034>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《从熵不变性看Attention的Scale操作》](<https://spaces.ac.cn/archives/8823>)中，我们推导了一版具有熵不变性质的注意力机制：
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\kappa \log n}{d}QK^{\top}\right)V\label{eq:a}\end{equation}
可以观察到，它主要是往Softmax里边引入了长度相关的缩放因子$\log n$来实现的。原来的推导比较繁琐，并且做了较多的假设，不利于直观理解，本文为其补充一个相对简明快速的推导。

## 推导过程

我们可以抛开注意力机制的背景，直接设有$s_1,s_2,\cdots,s_n\in\mathbb{R}$，定义
$$p_i = \frac{e^{\lambda s_i}}{\sum\limits_{i=1}^n e^{\lambda s_i}}$$
显然这就是$s_1,s_2,\cdots,s_n$同时乘上缩放因子$\lambda$后做Softmax的结果。现在我们算它的熵
\begin{equation}\begin{aligned}H =&\, -\sum_{i=1}^n p_i \log p_i = \log\sum_{i=1}^n e^{\lambda s_i} - \lambda\sum_{i=1}^n p_i s_i \\
=&\, \log n + \log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i} - \lambda\sum_{i=1}^n p_i s_i
\end{aligned}\end{equation}
第一项的$\log$里边是“先指数后平均”，我们用“先平均后指数”（平均场）来近似它：
\begin{equation}
\log\frac{1}{n}\sum_{i=1}^n e^{\lambda s_i}\approx \log\exp\left(\frac{1}{n}\sum_{i=1}^n \lambda s_i\right) = \lambda \bar{s}
\end{equation}
然后我们知道Softmax是会侧重于$\max$的那个（参考[《函数光滑化杂谈：不可导函数的可导逼近》](<https://spaces.ac.cn/archives/6620#softmax>)），所以有近似
\begin{equation}\lambda\sum_{i=1}^n p_i s_i \approx \lambda s_{\max}\end{equation}
所以
\begin{equation}H\approx \log n - \lambda(s_{\max} - \bar{s})\end{equation}
所谓熵不变性，就是希望尽可能地消除长度$n$的影响，所以根据上式我们需要有$\lambda\propto \log n$。如果放到注意力机制中，那么$s$的形式为$\langle \boldsymbol{q}, \boldsymbol{k}\rangle\propto d$（$d$是向量维度），所以需要有$\lambda\propto \frac{1}{d}$，综合起来就是
\begin{equation}\lambda\propto \frac{\log n}{d}\end{equation}
这就是文章开头式$\eqref{eq:a}$的结果。

## 文章小结

为之前提出的“熵不变性Softmax”构思了一个简单明快的推导。
