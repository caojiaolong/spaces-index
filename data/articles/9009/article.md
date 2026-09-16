# 为什么Pre Norm的效果不如Post Norm？

> 作者：苏剑林 · 科学空间 · 2022-03-29
>
> 原文：<https://spaces.ac.cn/archives/9009>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

Pre Norm与Post Norm之间的对比是一个“老生常谈”的话题了，本博客就多次讨论过这个问题，比如文章[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620>)、[《模型优化漫谈：BERT的初始标准差为什么是0\.02？》](<https://spaces.ac.cn/archives/8747>)等。目前比较明确的结论是：同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。Pre Norm更容易训练好理解，因为它的恒等路径更突出，但为什么它效果反而没那么好呢？

笔者之前也一直没有好的答案，直到前些时间在知乎上看到 [@唐翔昊](<https://www.zhihu.com/question/519668254/answer/2371885202>) 的一个回复后才“恍然大悟”，原来这个问题竟然有一个非常直观的理解！本文让我们一起来学习一下。

## 基本结论

Pre Norm和Post Norm的式子分别如下：
\begin{align}
\text{Pre Norm: } \quad \boldsymbol{x}_{t+1} = \boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t))\\
\text{Post Norm: }\quad \boldsymbol{x}_{t+1} = \text{Norm}(\boldsymbol{x}_t + F_t(\boldsymbol{x}_t))
\end{align}
在Transformer中，这里的$\text{Norm}$主要指Layer Normalization，但在一般的模型中，它也可以是Batch Normalization、Instance Normalization等，相关结论本质上是通用的。

在笔者找到的资料中，显示Post Norm优于Pre Norm的工作有两篇，一篇是[《Understanding the Difficulty of Training Transformers》](<https://papers.cool/arxiv/2004.08249>)，一篇是[《RealFormer: Transformer Likes Residual Attention》](<https://papers.cool/arxiv/2012.11747>)。另外，笔者自己也做过对比实验，显示Post Norm的结构迁移性能更加好，也就是说在Pretraining中，Pre Norm和Post Norm都能做到大致相同的结果，但是Post Norm的Finetune效果明显更好。

可能读者会反问[《On Layer Normalization in the Transformer Architecture》](<https://papers.cool/arxiv/2002.04745>)不是显示Pre Norm要好于Post Norm吗？这是不是矛盾了？其实这篇文章比较的是在完全相同的训练设置下Pre Norm的效果要优于Post Norm，这只能显示出Pre Norm更容易训练，因为Post Norm要达到自己的最优效果，不能用跟Pre Norm一样的训练配置（比如Pre Norm可以不加Warmup但Post Norm通常要加），所以结论并不矛盾。

## 直观理解

为什么Pre Norm的效果不如Post Norm？知乎上 [@唐翔昊](<https://www.zhihu.com/question/519668254/answer/2371885202>) 给出的答案是：Pre Norm的深度有“水分”！也就是说，一个$L$层的Pre Norm模型，其实际等效层数不如$L$层的Post Norm模型，而层数少了导致效果变差了。

具体怎么理解呢？很简单，对于Pre Norm模型我们迭代得到：
\begin{equation}\begin{aligned}
\boldsymbol{x}_{t+1} =&\,\boldsymbol{x}_t + F_t(\text{Norm}(\boldsymbol{x}_t)) \\
=&\, \boldsymbol{x}_{t-1} + F_{t-1}(\text{Norm}(\boldsymbol{x}_{t-1})) +  F_t(\text{Norm}(\boldsymbol{x}_t)) \\
=&\, \cdots \\
=&\, \boldsymbol{x}_0 + F_0 (\text{Norm}(\boldsymbol{x}_0)) + \cdots + F_{t-1}(\text{Norm}(\boldsymbol{x}_{t-1})) +  F_t(\text{Norm}(\boldsymbol{x}_t))
\end{aligned}\end{equation}
其中每一项都是同一量级的，那么有$\boldsymbol{x}_{t+1}=\mathcal{O}(t+1)$，也就是说第$t+1$层跟第$t$层的差别就相当于$t+1$与$t$的差别，当$t$较大时，两者的相对差别是很小的，因此
\begin{equation}\begin{aligned}
&\,F_t(\text{Norm}(\boldsymbol{x}_t)) + F_{t+1}(\text{Norm}(\boldsymbol{x}_{t+1})) \\
\approx&\,F_t(\text{Norm}(\boldsymbol{x}_t)) + F_{t+1}(\text{Norm}(\boldsymbol{x}_t)) \\
=&\, \begin{pmatrix} 1 & 1\end{pmatrix}\begin{pmatrix} F_t \\ F_{t+1}\end{pmatrix}(\text{Norm}(\boldsymbol{x}_t))
\end{aligned}\end{equation}
这个意思是说，当$t$比较大时，$\boldsymbol{x}_t,\boldsymbol{x}_{t+1}$相差较小，所以$F_{t+1}(\text{Norm}(\boldsymbol{x}_{t+1}))$与$F_{t+1}(\text{Norm}(\boldsymbol{x}_t))$很接近，因此原本一个$t$层的模型与$t+1$层和，近似等效于一个更宽的$t$层模型，所以在Pre Norm中多层叠加的结果更多是增加宽度而不是深度，层数越多，这个层就越“虚”。

说白了，Pre Norm结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。而Post Norm刚刚相反，在[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620>)中我们就分析过，它每Norm一次就削弱一次恒等分支的权重，所以Post Norm反而是更突出残差分支的，因此Post Norm中的层数更加“足秤”，一旦训练好之后效果更优。

## 相关工作

前段时间号称能训练1000层Transformer的[DeepNet](<https://spaces.ac.cn/archives/8978>)想必不少读者都听说过，在其论文[《DeepNet: Scaling Transformers to 1,000 Layers》](<https://papers.cool/arxiv/2203.00555>)中对Pre Norm的描述是：

> However, the gradients of Pre\-LN at bottom layers tend to be larger than at top layers, leading to a degradation in performance compared with Post\-LN\.

不少读者当时可能并不理解这段话的逻辑关系，但看了前一节内容的解释后，想必会有新的理解。

简单来说，所谓“the gradients of Pre\-LN at bottom layers tend to be larger than at top layers”，就是指Pre Norm结构会过度倾向于恒等分支（bottom layers），从而使得Pre Norm倾向于退化（degradation）为一个“浅而宽”的模型，最终不如同一深度的Post Norm。这跟前面的直观理解本质上是一致的。

## 文章小结

本文主要分享了“为什么Pre Norm的效果不如Post Norm”的一个直观理解。
