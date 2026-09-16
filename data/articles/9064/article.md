# 多标签“Softmax\+交叉熵”的软标签版本

> 作者：苏剑林 · 科学空间 · 2022-05-07
>
> 原文：<https://spaces.ac.cn/archives/9064>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>（注：本文的相关内容已整理成论文[《ZLPR: A Novel Loss for Multi\-label Classification》](<https://papers.cool/arxiv/2208.02955>)，如需引用可以直接引用英文论文，谢谢。）</strong>

在[《将“Softmax\+交叉熵”推广到多标签分类问题》](<https://spaces.ac.cn/archives/7359>)中，我们提出了一个用于多标签分类的损失函数：
\begin{equation}\log \left(1 + \sum\limits_{i\in\Omega_{neg}} e^{s_i}\right) + \log \left(1 + \sum\limits_{j\in\Omega_{pos}} e^{-s_j}\right)\label{eq:original}\end{equation}
这个损失函数有着单标签分类中“Softmax\+交叉熵”的优点，即便在正负类不平衡的依然能够有效工作。但从这个损失函数的形式我们可以看到，它只适用于“硬标签”，这就意味着label smoothing、[mixup](<https://spaces.ac.cn/archives/5693>)等技巧就没法用了。本文则尝试解决这个问题，提出上述损失函数的一个软标签版本。

## 巧妙联系

多标签分类的经典方案就是转化为多个二分类问题，即每个类别用sigmoid函数$\sigma(x)=1/(1+e^{-x})$激活，然后各自用二分类交叉熵损失。当正负类别极其不平衡时，这种做法的表现通常会比较糟糕，而相比之下损失$\eqref{eq:original}$通常是一个更优的选择。

在之前文章的评论区中，读者 [@wu\.yan](<https://spaces.ac.cn/archives/7359/comment-page-2#comment-14196>) 揭示了多个“sigmoid\+二分类交叉熵”与式$\eqref{eq:original}$的一个巧妙的联系：多个“sigmoid\+二分类交叉熵”可以适当地改写成
\begin{equation}\begin{aligned}
&\,-\sum_{j\in\Omega_{pos}}\log\sigma(s_j)-\sum_{i\in\Omega_{neg}}\log(1-\sigma(s_i))\\
=&\,\log\prod_{j\in\Omega_{pos}}(1+e^{-s_j})+\log\prod_{i\in\Omega_{neg}}(1+e^{s_i})\\
=&\,\log\left(1+\sum_{j\in\Omega_{pos}}e^{-s_j}+\cdots\right)+\log\left(1+\sum_{i\in\Omega_{neg}}e^{s_i}+\cdots\right)
\end{aligned}\label{eq:link}\end{equation}
对比式$\eqref{eq:original}$，我们可以发现式$\eqref{eq:original}$正好是上述多个“sigmoid\+二分类交叉熵”的损失去掉了$\cdots$所表示的高阶项！在正负类别不平衡时，这些高阶项占据了过高的权重，加剧了不平衡问题，从而效果不佳；相反，去掉这些高阶项后，并没有改变损失函数的作用（希望正类得分大于0、负类得分小于0），同时因为括号内的求和数跟类别数是线性关系，因此正负类各自的损失差距不会太大。

## 形式猜测

这个巧妙联系告诉我们，要寻找式$\eqref{eq:original}$的软标签版本，可以尝试从多个“sigmoid\+二分类交叉熵”的软标签版本出发，然后尝试去掉高阶项。所谓软标签，指的是标签不再是0或1，而是0～1之间的任意实数都有可能，表示属于该类的可能性。而对于二分类交叉熵，它的软标签版本很简单：
\begin{equation}-t\log\sigma(s)-(1-t)\log(1-\sigma(s))\end{equation}
这里$t$就是软标签，而$s$就是对应的打分。模仿过程$\eqref{eq:link}$，我们可以得到
\begin{equation}\begin{aligned}
&\,-\sum_i t_i\log\sigma(s_i)-\sum_i (1-t_i)\log(1-\sigma(s_i))\\
=&\,\log\prod_i(1+e^{-s_i})^{t_i}+\log\prod_i (1+e^{s_i})^{1-t_i}\\
=&\,\log\prod_i(1+t_i e^{-s_i} + \cdots)+\log\prod_i (1+(1-t_i)e^{s_i}+\cdots)\\
=&\,\log\left(1+\sum_i t_i e^{-s_i}+\cdots\right)+\log\left(1+\sum_i(1-t_i)e^{s_i}+\cdots\right)
\end{aligned}\end{equation}
如果去掉高阶项，那么就得到
\begin{equation}\log\left(1+\sum_i t_i e^{-s_i}\right)+\log\left(1+\sum_i(1-t_i)e^{s_i}\right)\label{eq:soft}\end{equation}
它就是式$\eqref{eq:original}$的软标签版本的候选形式，可以发现当$t_i\in\{0,1\}$时，正好是退化为式$\eqref{eq:original}$的。

## 证明结果

就目前来说，式$\eqref{eq:soft}$顶多是一个“候选”形式，要将它“转正”，我们需要证明在$t_i$为0～1浮点数时，式$\eqref{eq:soft}$能学出有意义的结果。所谓有意义，指的是理论上能够通过$s_i$来重构$t_i$的信息（$s_i$是模型预测结果，$t_i$是给定标签，所以$s_i$能重构$t_i$是机器学习的目标）。

为此，我们记式$\eqref{eq:soft}$为$l$，并求$s_i$的偏导数：
\begin{equation}\frac{\partial l}{\partial s_i} = \frac{-t_i e^{-s_i}}{1+\sum\limits_i t_i e^{-s_i}}+\frac{(1-t_i)e^{s_i}}{1+\sum\limits_i(1-t_i)e^{s_i}}\end{equation}
我们知道$l$的最小值出现在所有$\frac{\partial l}{\partial s_i}$都等于0时，直接去解方程组$\frac{\partial l}{\partial s_i}=0$并不容易，但笔者留意到一个神奇的“巧合”：当$t_i e^{-s_i}=(1-t_i)e^{s_i}$时，每个$\frac{\partial l}{\partial s_i}$自动地等于0！所以$t_i e^{-s_i}=(1-t_i)e^{s_i}$应该就是$l$的最优解了，解得
\begin{equation}t_i = \frac{1}{1+e^{-2s_i}}=\sigma(2s_i)\end{equation}
这是一个很漂亮的结果，它告诉我们几个信息：

> 1、式$\eqref{eq:soft}$确实是式$\eqref{eq:original}$合理的软标签推广，它能通过$s_i$完全重建$t_i$的信息，其形式也刚好与sigmoid相关；
> 
> 
> 
> 2、如果我们要将结果输出为0～1的概率值，那么正确的做法应该是$\sigma(2s_i)$而不是直觉中的$\sigma(s_i)$；
> 
> 
> 
> 3、既然最后的概率公式也具有sigmoid的形式，那么反过来想，也可以理解为我们依旧还是在学习多个sigmoid激活的二分类问题，只不过损失函数换成了式$\eqref{eq:soft}$。

## 实现技巧

式$\eqref{eq:soft}$的实现可以参考bert4keras的代码[multilabel\_categorical\_crossentropy](<https://github.com/bojone/bert4keras/blob/5f5d493fe7be9ff2bd0e303e78ed945d386ed8fd/bert4keras/backend.py#L331>)，其中有个小细节值得跟大家一起交流一下。

首先，我们将式$\eqref{eq:soft}$可以等价地改写成
\begin{equation}\log\left(1+\sum_i e^{-s_i + \log t_i}\right)+\log\left(1+\sum_i e^{s_i + \log (1-t_i)}\right)\label{eq:soft-log}\end{equation}
所以看上去，只需要将$\log t_i$加到$-s_i$、将$\log(1-t_i)$加到$s_i$上，补零后做常规的$\text{logsumexp}$即可。但实际上，$t_i$是有可能取到$0$或$1$的，对应的$\log t_i$或$\log(1-t_i)$就是负无穷，而框架无法直接处理负无穷，因此通常在$\log$之前需要clip一下，即选定$\epsilon > 0$后定义
\begin{equation}\text{clip}(t)=\left\{\begin{aligned}&\epsilon, &t < \epsilon \\
&t, &\epsilon\leq t\leq 1-\epsilon\\
&1-\epsilon, &t > 1-\epsilon\end{aligned}\right.\end{equation}

但这样一clip，问题就来了。由于$\epsilon$不是真的无穷小，比如$\epsilon=10^{-7}$，那么$\log\epsilon$大约是$-16$左右；而像GlobalPointer这样的场景中，我们会提前把不合理的$s_i$给mask掉，方式是将对应的$s_i$置为一个绝对值很大的负数，比如$-10^7$；然而我们再看式$\eqref{eq:soft-log}$，第一项的求和对象是$e^{-s_i + \log t_i}$，所以$-10^7$就会变成$10^7$，如果$t_i$没有clip，那么理论上$\log t_i$是$\log 0 = -\infty$，可以把$-s_i + \log t_i$重新变回负无穷，但刚才我们已经看到进行了clip之后的$\log t_i$顶多就是$-16$，远远比不上$-s_i$的$10^7$，所以$-s_i + \log t_i$依然是一个大正数。

为了解决这个问题，我们不止要对$t_i$进行clip，我们还要找出原本小于$\epsilon$的$t_i$，手动将对应的$-s_i$置为绝对值很大的负数，同样还要找出大于$1-\epsilon$的$t_i$，将对应的$s_i$置为绝对值很大的负数，这样做就是将小于$\epsilon$的按照绝对等于0额外处理，将大于$1-\epsilon$的按照绝对等于1处理。

## 文章小结

本文主要将笔者之前提出的多标签“Softmax\+交叉熵”推广到软标签场景，有了对应的软标签版本后，我们就可以将它与label smoothing、[mixup](<https://spaces.ac.cn/archives/5693>)等技巧结合起来了，像GlobalPointer等又可以多一个炼丹方向。
