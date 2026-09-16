# 随机分词再探：从Viterbi Sampling到完美采样算法

> 作者：苏剑林 · 科学空间 · 2023-10-16
>
> 原文：<https://spaces.ac.cn/archives/9811>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《随机分词浅探：从Viterbi Decoding到Viterbi Sampling》](<https://spaces.ac.cn/archives/9768>)中，笔者提出了一种名为“Viterbi Sampling”的随机分词算法，它只是在求最优解的Viterbi Decoding基础上进行小修改，保留了Viterbi算法的简单快速的特点，相比于已有的[Subword Regularization](<https://papers.cool/arxiv/1804.10959>)明显更加高效。不过，知乎上的读者 [@鶴舞](<https://www.zhihu.com/people/11f5cd888268129be2b1d9b298387f0d>) 指出，当前的采样算法可能会在多次二选一“稀释”了部分方案的出现概率，直接后果是原本分数最高的切分并不是以最高概率出现。

经过仔细思考后，笔者发现相应的问题确实存在，当时为了尽快得到一种新的采样算法，在细节上的思考和处理确实比较粗糙。为此，本文将进一步完善Viterbi Sampling算法，并证明完善后的算法在效果上可以跟Subword Regularization等价的。

## 问题分析

首先，我们来看一下[评论原话](<https://zhuanlan.zhihu.com/p/658440073>)：

> subword regularization中可以保证按概率数据（具有temperature超参数）。提出的方法中对于每个e，第一个算出的route会被多次1v1“挑战”，最终概率分布会不会和已有算法差蛮多的。
> 
>   举个例子，watching三种分法watch ing，wat ching，和w atching概率都是三分之一，提出的方案的采样概率概率就会变成，前两个的概率是四分之一，第三个的概率是二分之一，是这样的吗？

其实评论里边已经说得很清晰了，如果读者还不理解的话，这里笔者稍微再展开一下。假设有三种切分方案，每种方案的得分都一样，那么我们自然是期望采样过程中每种方案的出现概率都是$1/3$。然而，Viterbi Sampling是将多选一的采样过程转化为多步的二选一：
\begin{equation}
r_i = \left\{\begin{aligned}&\,1\,, \,\, s_i > s_{i-1} \\
&\,0\,, \,\, \text{else}\end{aligned}\right.\qquad\longrightarrow\qquad
r_i = \left\{\begin{aligned}&\,1\,, \,\, \varepsilon < \sigma(\alpha(s_i - s_{i-1})) \\
&\,0\,, \,\, \text{else}\end{aligned}\right.
\end{equation}
这样一来，前面的两种切分方案先二选一，概率都是$\frac{1/3}{1/3+1/3}=1/2$；选出来一个结果之后，又跟第三种方案放一起来二选一，由于概率是按照各自得分来算的，所以这时候各自的概率还是$1/2$。于是，在完整的采样过程中，前两种方案出现的概率是$1/4$，后一种方案出现的概率是$1/2$，越晚出现的方案相对来说越“占便宜”，而越早出现的方案概率被稀释得越严重。而很不巧的是，按照BytePiece的AC自动机的返回顺序，越长的词（通常来说得分越高）出现的次序会越靠前，所以在Viterbi Sampling中，得分越高的方案反而更容易被稀释概率。

## 解决办法

现在看来，其实解决办法也很简单，每次进行二选一后，同时缓存累积概率就可以了，而从第二步开始，每次二选一时新进来的候选者不是跟已有候选者得分二选一，而是跟累积概率得分二选一，这就是俗称“[水塘采样（Reservoir sampling）](<https://en.wikipedia.org/wiki/Reservoir_sampling>)”的算法。

用前面的例子来说，先进来两种切分方案，按照$\frac{1/3}{1/3+1/3}=1/2$的概率选出一种，然后它们总的累积概率是$2/3$；接下来被选者跟新方案选一，新出现的方案被选到的概率应该是$\frac{1/3}{2/3+1/3}=1/3$，也就是说要跟累积概率比，而不是跟被选者自己的概率比，这样完整的采样流程下来，每种切分方案出现的概率都是$1/3$。

对于Viterbi Sampling来说，每个终点位置会有多个切分方案，我们要对其进行多选一采样，被选中的概率是由各自的得分构造出来的$p_i = e^{\alpha s_i}/Z$，$Z$是归一化因子。因为我们是递归处理的，所以我们不知道多选一的“多”是多少，也无法计算$Z$，不过这不重要，知道$e^{\alpha s_i}$就够了，因为计算每一步的条件采样概率其实也用不到完整的$Z$，而是需要递归的$Z_i$：
\begin{array}{c|c|c}
\hline
\text{Viterbi Decoding} & \text{旧版 Viterbi Sampling} & \text{新版 Viterbi Sampling} \\
\hline
r_i = \left\{\begin{aligned}&\,1\,, \,\, s_i > s_{i-1} \\
&\,0\,, \,\, \text{else}\end{aligned}\right. &
r_i = \left\{\begin{aligned}&\,1\,, \,\, \varepsilon < \sigma(\alpha(s_i - s_{i-1})) \\
&\,0\,, \,\, \text{else}\end{aligned}\right. &
\begin{aligned}Z_i =&\, Z_{i - 1} + e^{\alpha s_i} \\[1pt]
r_i =&\, \left\{\begin{aligned}&\,1\,, \,\, \varepsilon < e^{\alpha s_i} / Z_i \\
&\,0\,, \,\, \text{else}\end{aligned}\right.\end{aligned} \\
\hline
\end{array}
实际计算时，由于指数爆炸的原因，直接缓存$Z_i$大概率会有溢出风险，所以我们一般缓存的是它的对数$Z^{\log}_i$，并利用$\text{logsumexp}$函数避免溢出：
\begin{equation}
\begin{aligned}&\,Z^{\log}_i = \text{logsumexp}(Z^{\log}_{i-1}, \alpha s_i) \\
&\qquad e^{\alpha s_i} / Z_i \to e^{\alpha s_i - Z^{\log}_i}
\end{aligned},\qquad \text{logsumexp}(x,y) = \left\{\begin{aligned}&\,x + \log(1+e^{y-x}),\,\, x \geq y \\
&\,y + \log(1 + e^{x-y}),\,\,x < y
\end{aligned}\right.
\end{equation}
相应的实现已经内置在<code>bytepiece&gt;=0.5.0</code>中。

## 完美采样

总的来说，出现旧版Viterbi Sampling的缺陷，还是因为之前操之过急了，所以现在认真地给新版Viterbi Sampling补上数学证明。有意思的是，可以证明更新后的Viterbi Sampling跟Subword Regularization一样都是“完美采样”算法。

之前我们介绍过，Subword Regularization的做法非常“粗暴”，直接找出得分最高的$k$个切分方案，然后通过$p_i = e^{\alpha s_i}/Z$的方式计算被选中的概率，其中$s_i$是第$i$种方案的得分。这种做法除了复杂度高外没有任何毛病，当$k$不做限制（即找出全部切分方案）时，我们得到<u>所有切分方案的一个随机采样</u>，而每种方案被采样到的概率正比于$e^{\alpha s_i}$——是得分$s_i$的单调增函数，即<u>采样概率与得分的大小排序都是一样的</u>，满足这两个条件的，笔者称之为“完美采样”。

### Decoding

为了证明新版Viterbi Sampling也是“完美采样”，我们先来回顾一下Viterbi Decoding。设有一个长度为$l$的字节串$c_1,c_2,\cdots,c_l$，用$S^*(c_1,c_2,\cdots,c_l)$表示最优切分方案的得分，假设我们知道$c_k,c_{k+1}$之间一定会分开，那么必然有
\begin{equation}S^*(c_1,c_2,\cdots,c_l) = S^*(c_1,c_2,\cdots,c_k) + S^*(c_{k+1},c_{k+2},\cdots,c_l)\end{equation}
也就是说，最优切分方案的子串，一定也是对应的子字节串的最优切分方案，这是动态规划的根本依据。当然，事实上我们不能预知哪一处会被切开，所以只能用枚举的方式：
\begin{equation}S^*(c_1,c_2,\cdots,c_l) = \max\left\{\begin{aligned}
&\,\color{green}{s\left(\overline{c_1,\cdots,c_l}\right)} \\
\color{red}{S^*(c_1)} \,+&\, \color{green}{s\left(\overline{c_2,\cdots,c_l}\right)} \\
\color{red}{S^*(c_1,c_2)} \,+&\, \color{green}{s\left(\overline{c_3,\cdots,c_l}\right)} \\
\vdots \\
\color{red}{S^*(c_1,\cdots,c_{l-2})} \,+&\, \color{green}{s\left(\overline{c_{l-1},c_l}\right)} \\
\color{red}{S^*(c_1,\cdots,c_{l-1})} \,+&\, \color{green}{s\left(\overline{c_l}\right)}
\end{aligned}\right\}\label{eq:core}\end{equation}
其中$s\left(\overline{c_1,\cdots,c_l}\right)$是指字节串$c_1, \cdots,c_l$作为一个token时的得分（如果它不是词表中的token，那么记为$-\infty$）。这样一来，$S^*(c_1,c_2,\cdots,c_l)$的计算就转化为$S^*(c_1),S^*(c_1,c_2),\cdots,S^*(c_1,\cdots,c_{l-1})$的计算，依此类推，$S^*(c_1,c_2,\cdots,c_{l-1})$的计算又可以转化为$S^*(c_1),S^*(c_1,c_2),\cdots,S^*(c_1,\cdots,c_{l-2})$的计算，等等，也就是$S^*$的结果是可以复用的。所以，整个流程总结下来就是一句话：

> 扫描到每一个位置时，都记录到当前位置的最优切分方案及其得分。

当然，直接按照式$\eqref{eq:core}$进行递归的话，理论上复杂度是$\mathcal{O}(l^2)$，但事实上不可能每个子字节串都是词表中的一个token，所以可以用Trie树、AC自动机等方法根据词表提前扫描好所有可能出现的token，那么复杂度就正比于搜索出来的候选token数，关于$l$是线性的，如果非要估计一个数值，那么假设词表中token的最大长度为$m$，那么长度为$l\geq m$的字节串扫描出来的token数就不超过
\begin{equation}l + (l - 1) + \cdots + (l - m + 1) = lm - \frac{1}{2}m(m-1) = \mathcal{O}(lm)\end{equation}

### Sampling

有了Decoding部分做铺垫后，理解Sampling就相对容易一些了。其实关键还是在式$\eqref{eq:core}$，我们用$Z(c_1,c_2,\cdots,c_l)$表示字节串$c_1,c_2,\cdots,c_l$的所有切分方案的归一化因子（完美采样），那么有
\begin{equation}Z(c_1,c_2,\cdots,c_l) = \sum\left\{\begin{aligned}
&\,\color{green}{e^{\alpha\cdot s\left(\overline{c_1,\cdots,c_l}\right)}} \\
\color{red}{Z(c_1)} &\, \color{green}{e^{\alpha\cdot s\left(\overline{c_2,\cdots,c_l}\right)}} \\
\color{red}{Z(c_1,c_2)} &\, \color{green}{e^{\alpha\cdot s\left(\overline{c_3,\cdots,c_l}\right)}} \\
\vdots \\
\color{red}{Z(c_1,\cdots,c_{l-2})} &\, \color{green}{e^{\alpha\cdot s\left(\overline{c_{l-1},c_l}\right)}} \\
\color{red}{Z(c_1,\cdots,c_{l-1})} &\, \color{green}{e^{\alpha\cdot s\left(\overline{c_l}\right)}}
\end{aligned}\right\}\label{eq:core-2}
\end{equation}
这个等式也表明，要实现从$c_1,c_2,\cdots,c_l$的所有切分方案中按$e^{\alpha s}$的比重采样，可以从$c_1,\cdots,c_{l-1}$的所有切分方案中随机选一个然后接上token $\overline{c_l}$、从$c_1,\cdots,c_{l-2}$的所有切分方案中随机选一个然后接上token $\overline{c_{l-1},c_l}$、从$c_1,\cdots,c_{l-3}$的所有切分方案中随机选一个然后接上token $\overline{c_{l-2},c_{l-1},c_l}$、\.\.\.，得到这$l$个采样结果后，分别再以权重$Z(c_1,\cdots,c_{l-1}) e^{\alpha\cdot s\left(\overline{c_l}\right)}$、$Z(c_1,\cdots,c_{l-2}) e^{\alpha\cdot s\left(\overline{c_{l-1},c_l}\right)}$、$Z(c_1,\cdots,c_{l-3}) e^{\alpha\cdot s\left(\overline{c_{l-2},c_{l-1},c_l}\right)}$、\.\.\.从中选一个。

接下来跟Decoding情形一样，$Z(c_1,\cdots,c_{l-1})$的计算又可以重用$Z(c_1),Z(c_1,c_2),\cdots,Z(c_1,\cdots,c_{l-2})$的结果，$Z(c_1,\cdots,c_{l-2})$的计算又可以重用$Z(c_1),Z(c_1,c_2),\cdots,Z(c_1,\cdots,c_{l-3})$的结果，等等，以及采样结果也都是可以重用的。于是类似地，那么整个Sampling算法也可以总结为一句话：

> 扫描到每一个位置时，都对以当前位置为终点的所有切分方案按照$e^{\alpha s}$权重进行采样，记录采样结果以及累积权重$Z$。

如果两边取对数，那么式$\eqref{eq:core-2}$可以等价地改写成
\begin{equation}Z^{\log}(c_1,c_2,\cdots,c_l) = \text{logsumexp}\left\{\begin{aligned}
&\,\color{green}{\alpha\cdot s\left(\overline{c_1,\cdots,c_l}\right)} \\
\color{red}{Z^{\log}(c_1)} \,+&\, \color{green}{\alpha\cdot s\left(\overline{c_2,\cdots,c_l}\right)} \\
\color{red}{Z^{\log}(c_1,c_2)} \,+&\, \color{green}{\alpha\cdot s\left(\overline{c_3,\cdots,c_l}\right)} \\
\vdots \\
\color{red}{Z^{\log}(c_1,\cdots,c_{l-2})} \,+&\, \color{green}{\alpha\cdot s\left(\overline{c_{l-1},c_l}\right)} \\
\color{red}{Z^{\log}(c_1,\cdots,c_{l-1})} \,+&\, \color{green}{\alpha\cdot s\left(\overline{c_l}\right)}
\end{aligned}\right\}
\end{equation}

跟Viterbi Decoding的式$\eqref{eq:core}$区别就是$Z^{\log}$代替了$S^*$，$\text{logsumexp}$代替了$\max$，而$\text{logsumexp}$正好是$\max$的光滑近似，所以$\alpha\to\infty$时能退化为Viterbi Decoding。另一方面，在实际计算时，同一终点的多个切分方案是逐一到达而不是一次性到达的，所以就需要将单步的“多选一”转化为多步的“二选一”，这就是“[解决办法](<https://spaces.ac.cn/archives/9811#%E8%A7%A3%E5%86%B3%E5%8A%9E%E6%B3%95>)”一节所讨论的内容。至此，我们证明了（或者说从Viterbi Decoding出发重新推导了）修改后的Viterbi Sampling实际是跟Subword Regularization一样的完美采样算法。

## 文章小结

本文完善了之前提出的随机分词算法Viterbi Sampling，并从数学上证明了它在效果上跟Subword Regularization一样都是“完美采样”算法，而在使用上有着比Subword Regularization明显更高的效率。
