# GlobalPointer下的“KL散度”应该是怎样的？

> 作者：苏剑林 · 科学空间 · 2022-04-15
>
> 原文：<https://spaces.ac.cn/archives/9039>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近有读者提到想测试一下[GlobalPointer](<https://spaces.ac.cn/archives/8373>)与[R\-Drop](<https://spaces.ac.cn/archives/8496>)结合的效果，但不知道GlobalPointer下的KL散度该怎么算。像R\-Drop或者[虚拟对抗训练](<https://spaces.ac.cn/archives/7466>)这些正则化手段，里边都需要算概率分布的KL散度，但GlobalPointer的预测结果并非一个概率分布，因此无法直接进行计算。

经过一番尝试，笔者给出了一个可用的形式，并通过简单实验验证了它的可行性，遂在此介绍笔者的分析过程。

## 对称散度

KL散度是关于两个概率分布的函数，它是不对称的，即$KL(p\Vert q)$通常不等于$KL(q\Vert p)$，在实际应用中，我们通常使用对称化的KL散度：
\begin{equation}D(p,q) = KL(p\Vert q) + KL(q\Vert p)\end{equation}
代入KL散度的定义$KL(p\Vert q)=\sum\limits_i p_i\log\frac{p_i}{q_i}$，可以化简得到
\begin{equation}D(p,q) = \sum_i (p_i - q_i)(\log p_i - \log q_i)\end{equation}
考虑到$p,q$通常由softmax得到，我们定义
\begin{equation}p_i = \frac{e^{s_i}}{\sum\limits_j e^{s_j}},\quad q_i = \frac{e^{t_i}}{\sum\limits_j e^{t_j}}\end{equation}
代入后得到
\begin{equation}\begin{aligned}
D(p,q) =&\, \sum_i (p_i - q_i)(s_i - t_i) + \sum_i (p_i - q_i)\left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right) \\
=&\, \sum_i (p_i - q_i)(s_i - t_i) + \left(\sum_i p_i - \sum_i q_i\right)\left(\log\sum_j e^{t_j} - \log\sum_j e^{s_j}\right) \\
=&\, \sum_i (p_i - q_i)(s_i - t_i)
\end{aligned}\label{eq:kl-0}\end{equation}

## 类比结果

可以看到，从logits层面看，对称KL散度具有以下的形式
\begin{equation}D(s, t) = \sum_i (f(s_i) - f(t_i))(s_i - t_i) = \langle f(s) - f(t), s -t \rangle\label{eq:kl}\end{equation}
其中$f$是softmax操作，$\langle\cdot,\cdot\rangle$表示向量的内积。从形式上来看，它是两个向量的内积，其中一个向量是logits的差，第二个向量则是logits经过$f$变换后的差。变换$f$有什么特点呢？我们知道，softmax实际上是$\text{onehot}(\text{argmax}(\cdot))$的光滑近似（参考[《函数光滑化杂谈：不可导函数的可导逼近》](<https://spaces.ac.cn/archives/6620>)），对于分类来说，最大值就是要输出的目标类，所以说白了，它实际上是“将目标类置为1、非目标类置为0”的光滑近似。

有了这个抽象视角，我们就可以类比地构建GlobalPointer的“KL散度”了。GlobalPointer的输出也可以理解为是logits，但它所用的损失函数是[《将“Softmax\+交叉熵”推广到多标签分类问题》](<https://spaces.ac.cn/archives/7359>)提出的多标签交叉熵，因此这本质上是多标签交叉熵中如何算KL散度的问题，最后GlobalPointer输出的目标类别亦并非logits最大的那个类，而是所有logits大于0的类别。

所以，对于GlobalPointer来说，其对称散度可以保留式$\eqref{eq:kl}$的形式，但$f$应该换成“将大于0的置为1、将小于0的置为0”的光滑近似，而sigmoid函数$\sigma(x)=1/(1+e^{-x})$正好是满足这一性质的函数，因此我们可以将GlobalPointer的对称KL散度可以设计为
\begin{equation}D(s, t) = \sum_i (\sigma(s_i) - \sigma(t_i))(s_i - t_i) = \langle \sigma(s) - \sigma(t), s -t \rangle\label{eq:gp-kl}\end{equation}

## 峰回路转

有意思的是，笔者事后发现，式$\eqref{eq:gp-kl}$实际上等价于每个logits分别用$\sigma$激活后，各自单独算二元概率的KL散度然后求和。

要证明这一点很简单，留意到$\sigma$函数构建的二元分布$[\sigma(s),1 - \sigma(s)]$，跟用$[s, 0]$为logits加上softmax构建的二元分布是等价的，即$[\sigma(s),1 - \sigma(s)]=softmax([s, 0])$，所以根据公式$\eqref{eq:kl-0}$，我们直接有
\begin{equation}\begin{aligned}
&\,D\big([\sigma(s_i),1 - \sigma(s_i)],[\sigma(t_i),1 - \sigma(t_i)]\big) \\
=&\,(\sigma(s_i)-\sigma(t_i))(s_i - t_i) + \big((1-\sigma(s_i))-(1-\sigma(t_i))\big)(0 - 0)\\
=&\,(\sigma(s_i)-\sigma(t_i))(s_i - t_i)
\end{aligned}\end{equation}
将每个分量加起来，就得到式$\eqref{eq:gp-kl}$

这个等价性说明，虽然我们做多标签分类时作为多个二分类问题来做的话会带来类别不平衡问题，但是如果只是用来评估结果连续性的话，就不存在所谓的类别不平衡问题了（因为根本就不是分类），所以此时仍然可以将其看成多个二分类问题，然后算其常规的KL散度。

## 实验结果

笔者和网友分别做了简单的对比实验，结果显示用式$\eqref{eq:gp-kl}$作为KL散度，将R\-Drop应用到GlobalPointer中，确实能轻微提升效果，而如果对GlobalPointer的logits直接做softmax然后算常规的KL散度，结果反而不好，这就体现了式$\eqref{eq:gp-kl}$的合理性。

但需要指出的是，式$\eqref{eq:gp-kl}$只是提供了一种在GlobalPointer中用R\-Drop或者虚拟对抗训练的方案，但具体情况下效果会不会有提升，这是无法保证的，就好比常规的分类问题配合R\-Drop也未必能取得效果提升一样。这需要多去实验尝试，尤其是需要精调正则项的权重系数。

## 文末小结

本文主要讨论了GlobalPointer下的“KL散度”计算问题，为GlobalPointer应用R\-Drop或者虚拟对抗训练等提供一个可用的KL散度形式。
