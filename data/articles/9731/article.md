# Transformer升级之路：14、当HWFA遇见ReRoPE

> 作者：苏剑林 · 科学空间 · 2023-08-24
>
> 原文：<https://spaces.ac.cn/archives/9731>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章[《Transformer升级之路：13、逆用Leaky ReRoPE》](<https://spaces.ac.cn/archives/9728>)中，笔者尝试通过在训练阶段逆用Leaky ReRoPE的思路，使得推理阶段的位置编码变为正常的RoPE，从而在达到长度外推的同时解决ReRoPE推理变慢的缺点。遗憾的是，从实验结果来看，“Leaky ReRoPE → RoPE”的效果并不如“RoPE → ReRoPE/Leaky ReRoPE”，因此这个问题尚未完全解决。

此时，笔者想到此前在[《Transformer升级之路：9、一种全局长度外推的新思路》](<https://spaces.ac.cn/archives/9603>)提出的HWFA本身就具有一定的长度外推能力，如果跟ReRoPE“强强联合”，是否会有更好的效果？更关键是，HWFA的加入可以大幅度降低推理成本，从而弥补ReRoPE的不足！

## 温故

首先，“例行公事”地回顾一下HWFA。HWFA（Hybird Window\-Full Attention）并非一个具体的模型，而是一种Attention的组合方式，能够在基本保持效果不变的前提下，增强Attention模型的长度外推能力，同时还能降低训练和推理成本。

具体来说，HWFA是“$L-1$层Window RoPE Attention \+ $1$层Full NoPE Attention”，即前面$L-1$层Attention都加上[RoPE](<https://spaces.ac.cn/archives/8265>)，并通过window限制感受野，这样一来推理成本就变为常数，并且基于block parallel进行优化的话，也可以提升训练速度；至于最后一层Attention，则保留global的形式，但去掉位置编码（NoPE），同时加上[$\log n$缩放](<https://spaces.ac.cn/archives/8823>)。经过这样修改，并且适当选择window之后，模型的训练效果只有轻微下降，同时呈现出优秀的长度外推能力。

无独有偶，后来Google提出了[FOT（Focused Transformer）](<https://papers.cool/arxiv/2307.03170>)，它跟HWFA有很多异曲同工之处：同样是$L-1$层Local Attention加$1$层Full Attention，Full Attention同样是NoPE的，不同的是FOT把Full Attention放在中间，并且Local Attention没有严格限制感受野，所以无法直接长度外推，因此它提出了crossbatch training来拓展模型长度。事后，笔者实验过在HWFA上使用crossbatch training，也有不错的效果。

## 知新

回到本文的主题，HWFA如何跟ReRoPE“强强联合”呢？我们知道，ReRoPE是用在Full RoPE Attention上的，就是在推理阶段截断一下相对位置矩阵：
$$\begin{pmatrix}0 & \\
1 & 0 & \\
2 & 1 & 0 &\\
\ddots & 2 & 1 & 0 & \\
\ddots & \ddots & 2 & 1 & 0 & \\
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\
\small{L - 2} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\
\small{L - 1} & \small{L - 2} & \ddots & \ddots & \ddots & 2 & 1 & 0 & \\
\end{pmatrix} \,\to\, \begin{pmatrix}
\color{red}{0} & \\
\color{red}{1} & \color{red}{0} & \\
\color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\
\color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\
\color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\ddots} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\
\color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{1} & \color{red}{0} & \\
\end{pmatrix}$$
出人意料的是，这样的事后处理体现出极佳的长度外推能力。然而，由于RoPE的特殊性，原始的ReRoPE实现需要算两次Attention矩阵，并且不兼容主流的Flash Attention加速等。总的来说，推理阶段的成本增加略有点大。

不过，HWFA的加入将会极大地缓解这个问题！综上所述，ReRoPE只用在Full RoPE Attention上，HWFA则大部分都是Window RoPE Attention，所以“HWFA\+ReRoPE”的方案就呼之欲出了：训练阶段将HWFA原本的Full NoPE Attention换成Full RoPE Attention，然后推理阶段则改为Full ReRoPE Attention。这样一来推理阶段切换ReRoPE带来的额外成本就会变得非常少，而且其他层换为Window Attention带来的收益更加显著。

除此之外，“HWFA\+ReRoPE”还可以弥补原本HWFA的效果损失。此前，为了保证长度外推能力，HWFA的Full Attention要去掉位置编码（即NoPE），同时Window Attention的感受野$\tilde{w}$要满足$(\tilde{w}-1)(L-1)+1 = \alpha N$（其中$L$是层数，$N$是训练长度，$0 < \alpha \leq 1$），这些约束限制了模型的表达能力，导致了训练效果变差。而引入ReRoPE之后，Window Attention的感受野可以适当取大一些，Full Attention也可以用RoPE，还可以将它放到中间层而不单是最后一层，甚至也可以多于$1$层Full Attention。这些变化都可以弥补效果损失，并且得益于ReRoPE，长度外推能力并不会有所下降。

为了区别最初版的HWFA，我们也可以将“HWFA\+ReRoPE”的组合，称为“HWFA2”。

## 实验

下面分享一些“HWFA\+ReRoPE（HWFA2）”的实验结果。由于引入ReRoPE之后，HWFA的自由度就大了很多，因此下降只是挑笔者认为比较符合直觉的组合进行实验，无法充分验证所有排列组合。

实验模型跟之前的HWFA、ReRoPE的一样，都是1亿参数的GAU模型，512的训练长度。注意这里有两个window参数：一个是ReRoPE本身有个$w$参数，此前ReRoPE实验显示这个影响不大，所以下面统一取256；另一个是HFWA的Window Attention的感受野，上面记为$\tilde{w}$，这是可调的。所以，“HWFA\+ReRoPE”的主要参数就是Window Attention的$\tilde{w}$，以及Full Attention的层数和分布位置。此前笔者做了一些对比实验，显示从训练效果来看，Full Attention放在中间要比放在末尾效果更好，所以如果是1层Full Attention，那么它默认的放置位置是(index = num\_layers / 2)的层，如果是2层Full Attention，那么默认的放置位置是(index = num\_layers / 3)和(index = 2 \* num\_layers / 3)的层，依此类推。

部分实验结果如下：
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\
\hline
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\
\hline
\text{ReRoPE-w256} & 49.41\% & 77.90\% & 48.48\% \\
\text{ReRoPE-w256-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.40\% & 48.85\% \\
\text{ReRoPE-w256-}\log n & 49.40\% & \boldsymbol{85.12\%} & 49.07\% \\
\hline
\text{InvLeaky ReRoPE-w128-}\log n & 49.38\% & 82.25\% & 48.32\% \\
\text{InvLeaky ReRoPE-w128-b8-}\log n & 49.62\% & 81.15\% & 48.85\% \\
\hline
\text{HFWA} & 48.70\% & 80.84\% & 48.15\% \\
\hline
\text{HFWA-ReRoPE-w32-f1} & 49.29\% & 83.13\% & 49.34\% \\
\text{HFWA-ReRoPE-w64-f1} & 49.32\% & 82.41\% & \boldsymbol{49.37\%} \\
\text{HFWA-ReRoPE-w128-f1} & 49.21\% & 80.18\% & 48.99\% \\
\text{HFWA-ReRoPE-w256-f1} & 49.00\% & 54.94\% & 47.64\% \\
\text{HFWA-ReRoPE-w32-f2} & \boldsymbol{49.50}\% & 84.09\% & 49.35\% \\
\text{HFWA-ReRoPE-w64-f2} & 49.46\% & 84.43\% & 49.36\% \\
\text{HFWA-ReRoPE-w128-f2} & 49.35\% & 83.09\% & 48.97\% \\
\text{HFWA-ReRoPE-w256-f2} & 49.37\% & 75.24\% & 48.42\% \\
\hline
\end{array}

上表中$\text{w}$后的数字就是Window Attention的感受野$\tilde{w}$的大小，$\text{f}$后的数字就是Full Attention的层数。原本的HFWA由于各种约束，$\tilde{w}$只取到了16，再大的话长度外推能力就会明显下降。而从上表可以看到，增大了$\tilde{w}$后，训练性能可以迅速对齐baseline，并且进一步增加Full Attention还超过了baseline。至于外推效果，$\text{w32},\text{w64}$这两个case都相当不错，明显超过了HFWA。总的来看，HFWA\-ReRoPE的最佳组合是$\text{w64-f2}$，训练效果和不重复的外推效果都超过了原本的ReRoPE，再结合训练长度$N$是512、层数$L$是24来看，猜测$\tilde{w}$的最佳取值应该是$N/L$的$2\sim 4$倍左右。

## 小结

本文提出了HWFA与ReRoPE的组合使用方式，小规模的实验结果显示，这种组合能够在不损失训练效果的同时，达到近乎最佳的长度外推效果，并且得益于HFWA的设计，还可以明显地降低推理成本，有效地缓解了ReRoPE原本的推理成本增加的缺点。
