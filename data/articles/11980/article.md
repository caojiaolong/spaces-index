# 连续化 Softmax Attention

> 作者：苏剑林 · 科学空间 · 2026-10-19
>
> 原文：<https://spaces.ac.cn/archives/11980>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

Softmax Attention想必大家已经不陌生，它是定义在有限向量集合下的一个加权混合算子。近日，论文[《Continuum Attention》](<https://papers.cool/arxiv/2610.20586>)提出了一个很有意思的想法——将注意力连续化。这让笔者想起，之前在[《为什么DeltaNet要加L2 Normalize？》](<https://spaces.ac.cn/archives/11486>)介绍的[EFLA](<https://papers.cool/arxiv/2512.12602>)也是将离散RNN连续化，然后又重新离散化，得到一些有意思的结果，类似的还有早期的[SSM](<https://spaces.ac.cn/tag/ssm/>)系列工作。

这样看来，“连续化—离散化”似乎是一条屡试不爽的探索路径。那么，Softmax Attention连续化又能得到什么有意思的结果呢？让我们一起来品鉴一番。

## 问题描述

标准的Softmax Attention定义为
\begin{equation}\boldsymbol{o}_t = \frac{\sum_{i=1}^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i} \boldsymbol{v}_i}{\sum_{i=1}^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_i}}\end{equation}
所谓连续化，是想将它改成
\begin{equation}\boldsymbol{o}_t = \frac{\int_0^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s} \boldsymbol{v}_s ds}{\int_0^t e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s} ds}\label{eq:csa}\end{equation}
其中非整数点的$\boldsymbol{k},\boldsymbol{v}$，定义为线性插值，即对于任意$s \in [i-1, i], i\in\mathbb{N}_+$，定义
\begin{equation}\boldsymbol{k}_s = (i - s)\boldsymbol{k}_{i-1} + (s - i + 1) \boldsymbol{k}_i,\qquad \boldsymbol{v}_s = (i - s)\boldsymbol{v}_{i-1} + (s - i + 1) \boldsymbol{v}_i
\end{equation}
该定义需要额外引入$\boldsymbol{k}_0,\boldsymbol{v}_0$两个向量（用于$s\in[0,1]$），其取值我们最后再讨论。

注意，式$\eqref{eq:csa}$的定义跟原论文略有不同，原论文出于对称考虑，将积分区间改为了$[1/2, t + 1/2]$，但笔者感觉这样调整虽然更对称了，但对Causal场景反而没那么自然，所以本文改用积分区间$[0,t]$。当然，接下来的推导其实跟区间的选择关系不大，技术上都是一样的。

## 准备工作

式$\eqref{eq:csa}$只是一个形式化定义，想要实际使用它，就必须把积分显式地计算出来。根据积分的可加性，我们只需要将每一段$\int_i^{i+1}$都分别算出来。记$a_{t,i} = \boldsymbol{q}_t\cdot\boldsymbol{k}_i, \delta_{t,i} = a_{t,i} - a_{t,i-1}$，不难得到
\begin{align}&\,\begin{aligned}
\int_{i-1}^i e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s} \boldsymbol{v}_s ds =&\, \int_0^1 e^{(1-s) a_{t,i-1} + s a_{t,i}} ((1-s)\boldsymbol{v}_{i-1} + s\boldsymbol{v}_i) ds \\
=&\, \boldsymbol{v}_i\int_0^1 e^{a_{t,i-1} + s \delta_{t,i}} ds  + (\boldsymbol{v}_{i-1} - \boldsymbol{v}_i)\int_0^1 e^{a_{t,i-1} + s \delta_{t,i}} (1-s) ds \\[4pt]
=&\, e^{a_{t,i-1}}\varphi_1(\delta_{t,i}) \boldsymbol{v}_i + e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) (\boldsymbol{v}_{i-1} - \boldsymbol{v}_i) \end{aligned} \\[10pt]
&\,\int_{i-1}^i e^{\boldsymbol{q}_t\cdot\boldsymbol{k}_s} ds = \int_0^1 e^{(1-s) a_{t,i-1} + s a_{t,i}} ds = \int_0^1 e^{a_{t,i-1} + s \delta_{t,i}} ds = e^{a_{t,i-1}}\varphi_1(\delta_{t,i})
\end{align}
这里
\begin{equation}\varphi_1(x) \triangleq \int_0^1 e^{s x} ds = \frac{e^x - 1}{x},\qquad \varphi_2(x) \triangleq \int_0^1 e^{s x} (1-s) ds = \frac{e^x - 1 - x}{x^2}\end{equation}
其中$x=0$时按照极限来处理，分别有$\varphi_1(0)=1,\varphi_2(0)=1/2$。它们还成立如下恒等关系
\begin{equation}\varphi_1(x) - \varphi_2(x) = \frac{x e^x - e^x + 1}{x^2} = e^x \cdot \frac{e^{-x} - 1 - (-x)}{(-x)^2} = e^x \varphi_2(-x)\end{equation}
这个恒等式在下一节化简时会用到。另外，从积分定义可以看出，$\varphi_1(x)$和$\varphi_2(x)$都是恒正的，这样后面的取对数运算就不会有未定义的问题。

## 组装化简

将上述积分代入到式$\eqref{eq:csa}$，得到
\begin{equation}\boldsymbol{o}_t = \frac{\sum_{i=1}^t e^{a_{t,i-1}}\varphi_1(\delta_{t,i}) \boldsymbol{v}_i + e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) (\boldsymbol{v}_{i-1} - \boldsymbol{v}_i)}{\sum_{i=1}^t e^{a_{t,i-1}}\varphi_1(\delta_{t,i})}\end{equation}
先化简分子
\begin{equation}\begin{aligned}
&\, \sum_{i=1}^t e^{a_{t,i-1}}\varphi_1(\delta_{t,i}) \boldsymbol{v}_i + e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) (\boldsymbol{v}_{i-1} - \boldsymbol{v}_i) \\
=&\, \sum_{i=1}^t e^{a_{t,i-1}}(\varphi_1(\delta_{t,i})-\varphi_2(\delta_{t,i}))\boldsymbol{v}_i + e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) \boldsymbol{v}_{i-1} \\
=&\, \sum_{i=1}^t e^{a_{t,i-1} + \delta_{t,i}}\varphi_2(-\delta_{t,i})\boldsymbol{v}_i + e^{a_{t,i-1}}\varphi_2(\delta_{t,i}) \boldsymbol{v}_{i-1} \\
=&\, \sum_{i=1}^t e^{a_{t,i}}\varphi_2(-\delta_{t,i})\boldsymbol{v}_i + \sum_{i=0}^{t-1} e^{a_{t,i}}\varphi_2(\delta_{t,i+1}) \boldsymbol{v}_i \\
=&\, e^{a_{t,0}}\varphi_2(\delta_{t,1}) \boldsymbol{v}_0 + e^{a_{t,t}}\varphi_2(-\delta_{t,t})\boldsymbol{v}_t + \sum_{i=1}^{t-1} e^{a_{t,i}} (\varphi_2(-\delta_{t,i}) + \varphi_2(\delta_{t,i+1})) \boldsymbol{v}_i
\end{aligned}\end{equation}
由此可见分子具有$\sum_{i=0}^t e^{a_{t,i} + b_{t,i}} \boldsymbol{v}_i$的形式，其中
\begin{equation}a_{t,i} = \boldsymbol{q}_t\cdot\boldsymbol{k}_i,\qquad e^{b_{t,i}} = \left\{\begin{aligned}
&\,\varphi_2(a_{t,1} - a_{t,0})&\, i = 0 \\
&\,\varphi_2(a_{t,i-1} - a_{t,i}) + \varphi_2(a_{t,i+1} - a_{t,i})&\, 1\leq i\leq t-1 \\
&\,\varphi_2(a_{t,t-1} - a_{t,t})&\, i = t \\
\end{aligned}\right.\end{equation}
注意分母正好是全体$\boldsymbol{v}_i$换成$1$的结果，所以直接可得分母为$\sum_{i=0}^t e^{a_{t,i} + b_{t,i}}$，于是
\begin{equation}\boldsymbol{o}_t = \frac{\sum_{i=0}^t e^{a_{t,i} + b_{t,i}} \boldsymbol{v}_i}{\sum_{i=0}^t e^{a_{t,i} + b_{t,i}}}\end{equation}

## 结果分析

相比标准的Softmax Attention
\begin{equation}\boldsymbol{o}_t = \frac{\sum_{i=1}^t e^{a_{t,i}} \boldsymbol{v}_i}{\sum_{i=1}^t e^{a_{t,i}}}\end{equation}
连续化带来的变化，就是引入了额外的$\boldsymbol{k}_0,\boldsymbol{v}_0$，以及给每个$a_{t,i}$加上了修正项$b_{t,i}$。多出来的$\boldsymbol{k}_0,\boldsymbol{v}_0$其实不算很本质的改动，我们等会再简单讨论一下，现在主要看$b_{t,i}$

对于一般的$1\leq i\leq t-1$，$b_{t,i}$同时依赖于$a_{t,i}$以及左右的$a_{t,i-1},a_{t,i+1}$，这在形式上很像是给Attention矩阵加上了Short Conv，尤其是对$b_{t,i}$做泰勒展开后，这一点更为明显
\begin{equation}b_{t,i} = \log (\varphi_2(a_{t,i-1} - a_{t,i}) + \varphi_2(a_{t,i+1} - a_{t,i})) = \frac{a_{t,i-1} + a_{t,i+1} - 2a_{t,i}}{6} + \cdots\end{equation}
其中$a_{t,i-1} + a_{t,i+1} - 2a_{t,i}$正是$a_{t,i}$的二阶差分，属于一种简单的Short Conv运算。所以，连续化Softmax Attention确实与“给Attention矩阵加Short Conv”有颇多异曲同工之处。

至于$\boldsymbol{k}_0,\boldsymbol{v}_0$，其实就相当于我们为解决Attention Sink问题引入的Sink token，对于连续化来说，主要是为了补足$t$个区间，是一个单纯的数学技巧，我们一般有三种选择：全零、$\boldsymbol{k}_0=\boldsymbol{k}_1,\boldsymbol{v}_0=\boldsymbol{v}_1$以及可训练，笔者在下面的实验中选择了全零。

## 简单实验

$b_{t,i}$同时依赖于$a_{t,i}$及其相邻值，也意味着它无法复用现有的Flash Attention，但实现原理上是通用的，并且理论FLOPs也基本一致。笔者让K3简单写了一个Kernel，速度只有Softmax Attention的1/4左右，主要是里边细碎的运算太多了，难以加速，不过总归可以先跑一下。

第一个实验结果是，连续化虽然引入了局域先验，但它无法完全取代位置编码，在全Full Attention的架构下，连续化Softmax Attention如果不加RoPE，会比标准Softmax Attention加RoPE要明显差，相反如果加上RoPE，则能够追回这个差距。

至于训练动态，在大家都加RoPE的情况下，连续化Softmax Attention前期会比标准Softmax Attention明显好，但差距逐渐缩窄，等到训练结束时，两者基本重合，优势消失。这里猜测核心原因是，自然语言并不是真正的时间序列，连续化并不能带来实质好处，也许对于真正的时间序列，才有机会测出优势。

笔者还做了从标准Softmax Attention中途切换连续化版本的实验，发现起跳Loss并不算高，最后也是慢慢收敛到跟标准版基本一致的效果。

## 文章小结

本文简单推导了一下连续化的Softmax Attention，其结果类似于对Attention矩阵加Short Conv操作。虽然在自然语言任务上实测并没有发现这种连续化带来的优势，但它仍不失作为一个有趣的结果来赏析。
