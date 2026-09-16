# Transformer升级之路：15、Key归一化助力长度外推

> 作者：苏剑林 · 科学空间 · 2023-11-20
>
> 原文：<https://spaces.ac.cn/archives/9859>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

大体上，我们可以将目前Transformer的长度外推技术分为两类：一类是事后修改，比如[NTK\-RoPE](<https://spaces.ac.cn/archives/9675>)、[YaRN](<https://papers.cool/arxiv/2309.00071>)、[ReRoPE](<https://spaces.ac.cn/archives/9708>)等，这类方法的特点是直接修改推理模型，无需微调就能达到一定的长度外推效果，但缺点是它们都无法保持模型在训练长度内的恒等性；另一类自然是事前修改，如[ALIBI](<https://spaces.ac.cn/archives/9431#ALIBI>)、[KERPLE](<https://spaces.ac.cn/archives/9431#KERPLE>)、[XPOS](<https://spaces.ac.cn/archives/9431#XPOS>)以及[HWFA](<https://spaces.ac.cn/archives/9603>)等，它们可以不加改动地实现一定的长度外推，但相应的改动需要在训练之前就引入，因此无法不微调地用于现成模型，并且这类方法是否能够Scale Up还没得到广泛认可。

在这篇文章中，笔者将介绍一种意外发现的长度外推方案——“KeyNorm”——对Attention的Key序列做L2 Normalization，很明显它属于事前修改一类，但对Attention机制的修改非常小，因此看上去非常有希望能够Scale Up。

## 最初动机

之所以说“意外发现”，是因为该改动的原始动机并不是长度外推，而是尝试替换Scaled Dot\-Product Attention中的Scale方式。我们知道，Attention的标准定义是（本文主要考虑Causal场景）
\begin{equation}\boldsymbol{o}_i = \frac{\sum_{j = 1}^i\exp\left(\frac{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}{\sqrt{d}}\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\frac{\boldsymbol{q}_i\cdot \boldsymbol{k}_j}{\sqrt{d}}\right)},\quad \boldsymbol{q}_i,\boldsymbol{k}_j\in\mathbb{R}^d\label{eq:sdpa}\end{equation}
其中，Scale因子$\frac{1}{\sqrt{d}}$我们已经多次进行过解释甚至推广，比如[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620#NTK%E5%8F%82%E6%95%B0%E5%8C%96>)、[《从熵不变性看Attention的Scale操作》](<https://spaces.ac.cn/archives/8823>)、[《从梯度最大化看Attention的Scale操作》](<https://spaces.ac.cn/archives/9812>)等。标准的推导是在“$\boldsymbol{q}_i,\boldsymbol{k}_j$均独立地采样自“均值为0、方差为1”的分布”的假设下进行的，而在该假设之下，我们还有
\begin{equation}\Vert\boldsymbol{q}_i\Vert\approx \sqrt{d},\quad \Vert\boldsymbol{k}_j\Vert\approx \sqrt{d}\end{equation}
这是因为
\begin{equation}\Vert\boldsymbol{x}\Vert^2 = \sum_{i=1}^d x_i^2 = d\times\frac{1}{d}\sum_{i=1}^d x_i^2\approx d\,\mathbb{E}_{x\sim\mathcal{N}(0,1)}[x^2] = d\end{equation}
相关推广还可以参考[《让人惊叹的Johnson\-Lindenstrauss引理：理论篇》](<https://spaces.ac.cn/archives/8679#%E5%BC%95%E7%90%86%E7%9A%84%E5%BC%95%E7%90%86>)。这个近似式意味着，在Attention的初始阶段式$\eqref{eq:sdpa}$与下面两个变体有着相同的效果：
\begin{align}\color{red}{\text{Q}}\text{uery}\color{red}{\text{N}}\text{orm:}\quad\boldsymbol{o}_i =&\, \frac{\sum_{j = 1}^i\exp\left(\tilde{\boldsymbol{q}}_i\cdot \boldsymbol{k}_j\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\tilde{\boldsymbol{q}}_i\cdot \boldsymbol{k}_j\right)},\qquad \tilde{\boldsymbol{q}}_i = \frac{\boldsymbol{q}_i}{\Vert\boldsymbol{q}_i\Vert} \\[5pt]
\color{red}{\text{K}}\text{ey}\color{red}{\text{N}}\text{orm:}\quad\boldsymbol{o}_i =&\, \frac{\sum_{j = 1}^i\exp\left(\boldsymbol{q}_i\cdot \tilde{\boldsymbol{k}}_j\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\boldsymbol{q}_i\cdot \tilde{\boldsymbol{k}}_j\right)},\qquad \tilde{\boldsymbol{k}}_j = \frac{\boldsymbol{k}_j}{\Vert\boldsymbol{k}_j\Vert}
\end{align}
因此，就有了验证这两个变体与标准的式$\eqref{eq:sdpa}$哪个更优的想法。为了描述的方便，我们可以相应地称为“Query/Key\-Normalized Dot\-Product Attention”，分别简称为“QNA”和“KNA”。

此外，既然可以QueryNorm和KeyNorm，那么自然也可以考虑两者都Norm一下，所以我们将如下“Scaled Cosine Attention（CosA）”也一并进行实验：
\begin{equation}\boldsymbol{o}_i = \frac{\sum_{j = 1}^i\exp\left(\lambda\,\tilde{\boldsymbol{q}}_i\cdot \tilde{\boldsymbol{k}}_j\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\lambda\,\tilde{\boldsymbol{q}}_i\cdot \tilde{\boldsymbol{k}}_j\right)} = \frac{\sum_{j = 1}^i\exp\left(\lambda\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)\right)\boldsymbol{v}_j}{\sum_{j = 1}^i\exp\left(\lambda\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)\right)}
\end{equation}
其中$\lambda$采用[《从梯度最大化看Attention的Scale操作》](<https://spaces.ac.cn/archives/9812>)中的结果，即$\lambda = 4\log n$（原文是3\.5，但下面训练长度比较小，改为4更精准一些），其中$n$固定为训练长度的一半，或者动态取位置id加1。

## 先看结果

沿着[之前](<https://spaces.ac.cn/archives/9731#%E5%AE%9E%E9%AA%8C>)做长度外推的实验设置，都是1亿参数的小模型，[GAU](<https://spaces.ac.cn/archives/9052>)架构，训练相同的步数（时间有限，这个步数下其实模型还没训练充分），训练长度512，并考虑外推到4096长度，实验结果如下表。其中Baseline就是式$\eqref{eq:sdpa}$，$\text{-}\log n$就是加入[《从熵不变性看Attention的Scale操作》](<https://spaces.ac.cn/archives/8823>)介绍的长度相关的缩放因子。评价指标是语言模型的逐token准确率，越大越好。
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复}) \\
\hline
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\
\hline
\text{QNA} & 49.55\% & 22.45\% & 22.18\% \\
\text{QNA-}\log n & 49.42\% & 19.55\% & 18.74\% \\
\text{KNA} & 49.60\% & 61.08\% & 47.69\% \\
\text{KNA-}\log n & 49.58\% & 63.17\% & 46.40\%\\
\text{CosA} & 49.73\% & 58.90\% & 46.98\% \\
\text{CosA-}\log n & 49.67\% & 64.74\% & 48.95\% \\
\hline
\end{array}
从表格中我们可以看出：1、不管是QueryNorm还是KeyNorm，都在训练长度上取得了更好的效果，虽然这个优势非常微弱，大概率随着训练的进一步推进可以忽略不计，但这个优势非常稳定，暗示着让训练更加平稳的可能性；2、<strong>KeyNorm对长度外推的提升非常明显</strong>，这就是实验结果中的“意外之喜”！

注意，跟NTK\-RoPE、YaRN需要修改推理阶段的模型不同，这里的KNA和CosA的长度外推在推理阶段是完全不做改动的。因此，可能有读者想知道，既然KNA和CosA推理时不加改动外推效果都这么好了，如果配合NTK\-RoPE、YaRN等外推技巧，效果会不会“更上一层楼”？对此，笔者也进行了测试，结果如下表：
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复}) \\
\hline
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\
\text{Baseline-NTK} & 49.41\% & 60.57\% & 42.20\% \\
\text{Baseline-YaRN} & 49.41\% & 80.10\% & 47.45\% \\
\text{Baseline-ReRoPE} & 49.41\% & 76.11\% & 47.82\% \\
\hline
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\
\text{Baseline-}\log n\text{-NTK} & 49.40\% & 75.86\% & 47.06\% \\
\text{Baseline-}\log n\text{-YaRN} & 49.40\% & 82.57\% & 46.52\% \\
\text{Baseline-}\log n\text{-ReRoPE} & 49.40\% & 85.47\% & 48.87\% \\
\hline
\text{QNA} & 49.55\% & 22.45\% & 22.18\% \\
\text{QNA-NTK} & 49.55\% & 52.28\% & 39.88\% \\
\text{QNA-YaRN} & 49.55\% & 82.53\% & 47.50\% \\
\text{QNA-ReRoPE} & 49.55\% & 78.22\% & 47.72\% \\
\hline
\text{QNA-}\log n & 49.42\% & 19.55\% & 18.74\% \\
\text{QNA-}\log n\text{-NTK} & 49.42\% & 57.44\% & 41.56\% \\
\text{QNA-}\log n\text{-YaRN} & 49.42\% & 80.08\% & 45.16\% \\
\text{QNA-}\log n\text{-ReRoPE} & 49.42\% & 84.71\% & 48.31\% \\
\hline
\text{KNA} & 49.60\% & 61.08\% & 47.69\% \\
\text{KNA-NTK} & 49.60\% & 64.44\% & 43.02\% \\
\text{KNA-YaRN} & 49.60\% & 84.19\% & 47.44\% \\
\text{KNA-ReRoPE} & 49.60\% & 77.76\% & 47.73\% \\
\hline
\text{KNA-}\log n & 49.58\% & 63.17\% & 46.40\%\\
\text{KNA-}\log n\text{-NTK} & 49.58\% & 79.05\% & 47.43\%\\
\text{KNA-}\log n\text{-YaRN} & 49.58\% & 83.95\% & 47.16\%\\
\text{KNA-}\log n\text{-ReRoPE} & 49.58\% & 85.48\% & 48.78\%\\
\hline
\text{CosA} & 49.73\% & 58.90\% & 46.98\% \\
\text{CosA-NTK} & 49.73\% & 62.50\% & 42.77\% \\
\text{CosA-YaRN} & 49.73\% & 83.40\% & 47.80\% \\
\text{CosA-ReRoPE} & 49.73\% & 77.82\% & 47.80\% \\
\hline
\text{CosA-}\log n & 49.67\% & 64.74\% & 48.39\% \\
\text{CosA-}\log n\text{-NTK} & 49.67\% & 78.97\% & 47.46\% \\
\text{CosA-}\log n\text{-YaRN} & 49.67\% & 82.28\% & 45.72\% \\
\text{CosA-}\log n\text{-ReRoPE} & 49.67\% & 85.67\% & 48.39\% \\
\hline
\end{array}
这个表比较啰嗦，主要是为了让大家对主流长度外推技巧的效果差异有一个全面的感知，大家选择自己感兴趣的维度比较就好，但要注意如果看长度外推效果的话，应该以“不重复”一列为主，“重复”一列为辅。从上表看，结果着实有点让人意外，KeyNorm似乎“免疫”了已有的RoPE外推技巧，NTK、YaRN等技巧叠加上去并没有明显提升，甚至可能会下降，不过总体来看“重复”一列还是有显著提升的，不显著的是“不重复”一列。这些结果表明，KeyNorm依然有着无法有效识别超出训练长度的位置（所以“重复”的结果不高）的问题，但有效地避免了PPL爆炸问题（所以“不重复”的结果还不错）。

这对做Long Context的同学来说可能是个好消息：一方面，KeyNorm不像ALIBI、KERPLE等，它的长度外推不用加Local约束，训练完成后也不做任何修改，纯属是“免费的午餐”，甚至看上去加了KeyNorm后训练效果都变好了；另一方面，也因为它是非Local的，所以可以更长文本继续训练，并且继续训练时再也不用纠结是选[PI](<https://papers.cool/arxiv/2306.15595>)还是[ABF](<https://papers.cool/arxiv/2309.16039>)了，对于KeyNorm来说，啥也不改就行。

## 原理分析

尽管这是个意外发现，但我们仍需要尝试去解释它，不然它就一直只是个意外。所以这一节我们尝试来思考，为什么KeyNorm会有助于长度外推。

让我们重新回到式$\eqref{eq:sdpa}$，第$i$个token与第$j$个token的相关性打分由内积完成：
\begin{equation}s(j|i) = \boldsymbol{q}_i\cdot \boldsymbol{k}_j = \Vert\boldsymbol{q}_i\Vert \Vert\boldsymbol{k}_j\Vert \cos(\boldsymbol{q}_i,\boldsymbol{k}_j),\quad p(j|i) = \frac{\exp\left(\frac{s(j|i)}{\sqrt{d}}\right)}{\sum_{j=1}^i \exp\left(\frac{s(j|i)}{\sqrt{d}}\right)}\end{equation}
第二个等号，我们从几何意义出发，将它分解为了各自模长与夹角余弦的乘积。注意力$p(j|i)$是一个条件概率，$\Vert\boldsymbol{q}_i\Vert$只跟当前位置$i$有关，它不改变注意力的相对大小，而只改变[稀疏程度](<https://spaces.ac.cn/archives/9595>)；$\Vert\boldsymbol{k}_j\Vert$则有能力改变$p(j|i)$的相对大小，但它不涉及到$i,j$的交互，可以用来表达一些绝对信号，比如[Scissorhands](<https://papers.cool/arxiv/2305.17118>)表明某些绝对位置的token的注意力一直都会很高，这就有可能用$\Vert\boldsymbol{k}_j\Vert$来表达；剩下的$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$就是用来表达$i,j$的交互，它是自由度最大的一项。

很明显，为了提高某个位置$j$的相对重要性，模型有两个选择：1、增大模长$\Vert\boldsymbol{k}_j\Vert$；2、增大$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$，即缩小$\boldsymbol{q}_i,\boldsymbol{k}_j$的夹角大小。然而，由于“[维度灾难](<https://spaces.ac.cn/archives/7076>)”的存在，在高维空间中显著地改变夹角大小相对来说没有那么容易，所以如果能靠增大模长$\Vert\boldsymbol{k}_j\Vert$完成的，模型会优先选择通过增大模长$\Vert\boldsymbol{k}_j\Vert$来完成，这导致的直接后果是：$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练可能并不充分。

这里笔者作出一个断言（猜测）：

> $\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练不充分是Attention无法长度外推的主要原因。

$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练不充分，是指被训练过的$\boldsymbol{q}_i,\boldsymbol{k}_j$的夹角只是一个有限的集合，而进行长度外推时，它要面对一个更大的集合，从而无法进行正确的预测。仔细思考[YaRN](<https://papers.cool/arxiv/2309.00071>)一文的推导就会发现，NTK、YaRN之所以有效，是因为修改了推理阶段RoPE的实现，使得$\boldsymbol{q}_i,\boldsymbol{k}_j$的夹角落到原本训练阶段的有限集合中，避免面对没见过的更大的集合，转外推为内插；ReRoPE则更加干脆，直接截断Window以外的相对位置，这使得推理阶段的位置编码都不会“面生”。这些技巧一定程度上都间接地验证了这个断言。

从这个断言出发，KeyNorm的长度外推起因就变得简单了。不论是只进行KeyNorm的KNA，还是QueryNorm、KeyNorm都做的CosA，它们都将$\Vert\boldsymbol{k}_j\Vert$从Attention的定义中排除掉了，于是为了改变$j$的相对重要性，模型就只有“调整$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$”这一个选择，这将会使得模型更加充分地训练和利用$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$，从而间接促进了长度外推性。此外，笔者也实验过“KeyNorm \+ NoPE”的组合，但并没有发现长度外推性，这说明RoPE也在KeyNorm的长度外推中担任重要角色。事实上这也不难理解，RoPE对$\boldsymbol{q}_i,\boldsymbol{k}_j$进行旋转，更有助于扩大训练期间$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的范围，从而使得$\cos(\boldsymbol{q}_i,\boldsymbol{k}_j)$的训练更为充分。

有没有工作已经尝试过QueryNorm和KeyNorm了呢？有。2020年的论文[《Query\-Key Normalization for Transformers》](<https://papers.cool/arxiv/2010.04245>)曾实验过CosA，论文还提出了一个类似的长度对数的Scale因子，但没有讨论到长度外推问题。此外，今年初Google的论文[《Scaling Vision Transformers to 22 Billion Parameters》](<https://papers.cool/arxiv/2302.05442>)也在Query和Key加了Norm，但加的是LayerNorm，LayerNorm或者RMSNorm都带有可学的gamma参数，这使得Norm之后的向量模长未必为常数，因此并不好说是否能达到本文一样的长度外推效果。

## 文章小结

本文介绍了笔者意外发现的一种长度外推方案“KeyNorm”——对Attention的Key序列进行L2归一化，在训练长度上取得了更好的效果，并在长度外推方面表现出显著的提升。它属于“事前修改”方案，跟其他事前修改方案如ALIBI、KERPLE等相比，它没有Local约束，因此更有希望能够Scale Up；相比于NTK\-RoPE、YaRN等“事后修改”方案，它在外推的时候则不会损失训练长度内的性能。
