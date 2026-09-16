# 注意力和Softmax的两点有趣发现：鲁棒性和信息量

> 作者：苏剑林 · 科学空间 · 2023-04-25
>
> 原文：<https://spaces.ac.cn/archives/9593>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近几周笔者一直都在思考注意力机制的相关性质，在这个过程中对注意力及Softmax有了更深刻的理解。在这篇文章中，笔者简单分享其中的两点：

> 1、Softmax注意力天然能够抵御一定的噪声扰动；
> 
> 
> 
> 2、从信息熵角度也可以对初始化问题形成直观理解。

## 鲁棒性

基于Softmax归一化的注意力机制，可以写为
\begin{equation}o = \frac{\sum\limits_{i=1}^n e^{s_i} v_i}{\sum\limits_{i=1}^n e^{s_i}}\end{equation}
有一天笔者突然想到一个问题：如果往$s_i$中加入独立同分布的噪声会怎样？为此，我们考虑
\begin{equation}\tilde{o} = \frac{\sum\limits_{i=1}^n e^{s_i+\varepsilon_i} v_i}{\sum\limits_{i=1}^n e^{s_i+\varepsilon_i}}\end{equation}
其中$\varepsilon_i$是独立同分布的噪声。然而，简单分析后笔者发现结论是“不怎么样”，注意力机制天然能抵御这类噪声，即$\tilde{o}\approx o$。

为了理解这一点，只需要意识到：
\begin{equation}\tilde{o} = \frac{\frac{1}{n}\sum\limits_{i=1}^n e^{s_i+\varepsilon_i} v_i}{\frac{1}{n}\sum\limits_{i=1}^n e^{s_i+\varepsilon_i}}=\frac{\mathbb{E}_i[e^{s_i+\varepsilon_i} v_i]}{\mathbb{E}_i[e^{s_i+\varepsilon_i}]}\approx \frac{\mathbb{E}_i[e^{s_i}v_i]\mathbb{E}[e^{\varepsilon}]}{\mathbb{E}_i[e^{s_i}]\mathbb{E}[e^{\varepsilon}]}=\frac{\mathbb{E}_i[e^{s_i}v_i]}{\mathbb{E}_i[e^{s_i}]}=o\end{equation}
约等号是利用了$\varepsilon_i$跟$s_i,v_i$相互独立，所以积的期望等于期望的积。

## 信息量

如果我们记$p_i = e^{s_i}\left/\sum\limits_{i=1}^n e^{s_i}\right.$，那么$p_i$描述了一个离散型概率分布，我们可以算信息熵
\begin{equation}H = -\sum_{i=1}^n p_i\log p_i\quad\in[0,\log n]\end{equation}
在[《“熵”不起：从熵、最大熵原理到最大熵模型（一）》](<https://spaces.ac.cn/archives/3534>)中我们讨论过，熵是不确定性的度量，也是信息量的度量。怎么理解两者的联系呢？熵本质上是均匀度的度量，越均匀越不确定，所以熵是不确定性的度量，熵的下界是0，所以不确定性也意味着它是我们从“不确定”到“完全确定”所能获得的最大信息量。

我们知道，如果将$s_i$初始化得非常大，那么$p_i$就会接近一个one hot分布，此时就会由于梯度消失而无法训练（参考[《浅谈Transformer的初始化、参数化与标准化》）](<https://spaces.ac.cn/archives/8620>)。笔者发现从信息量的角度也可以很直观理解这一点：模型训练本身就是从不确定（随机模型）到确定（训练模型）的过程，优化器负责从随机模型中“榨取”信息，而one hot分布的信息量为0，优化器“无利可图”，说不准还要“倒贴”，自然也就没法优化好了。所以我们要将模型初始化得尽量均匀，以保证可以“榨取”的信息量最大。

当然，除了要保证信息量的上界足够大外，还要保证信息量的下界足够小，才能保证可以“榨取”的信息量尽量大。之前在介绍对比学习中，有读者不理解温度参数的意义，其实也可以从信息量来理解。记
\begin{equation}p_i = \frac{e^{(\cos\theta_i) / \tau}}{\sum\limits_{i=1}^n e^{(\cos\theta_i)/\tau}}\end{equation}
如果$\tau=1$，那么信息熵的上界为$\log n$，但是下界约为$\log n - 0.4745$（参考[评论区](<https://spaces.ac.cn/archives/9593/comment-page-1#comment-28363>)），能获得的信息量太少，所以我们要缩小$\tau$，使得信息熵的下界接近0，从而增加能够获得的信息量。

## 简言之

简单水了一篇博客。可以看出，最终的结论还是——[《听说Attention与Softmax更配哦～》](<https://spaces.ac.cn/archives/9019>)。
