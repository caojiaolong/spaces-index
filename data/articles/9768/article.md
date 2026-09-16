# 随机分词浅探：从Viterbi Decoding到Viterbi Sampling

> 作者：苏剑林 · 科学空间 · 2023-09-16
>
> 原文：<https://spaces.ac.cn/archives/9768>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章[《大词表语言模型在续写任务上的一个问题及对策》](<https://spaces.ac.cn/archives/9762>)发布后，很快就有读者指出可以在训练阶段引入带有随机性的分词结果来解决同样的问题，并且已经有论文和实现。经过进一步查阅学习，笔者发现这是一个名为[Subword Regularization](<https://papers.cool/arxiv/1804.10959>)的技巧，最早应用在NMT（机器翻译）中，目前SentencePiece也有相应的实现。看起来这个技巧确实能缓解前述问题，甚至有助于增强语言模型的容错能力，所以就有了将它加进去[BytePiece](<https://spaces.ac.cn/archives/9752>)的想法。

那么问题来了，如何将确定性分词改为随机性分词呢？BytePiece是基于Unigram模型的，它通过Viterbi算法找最大概率的分词方案，既然有概率，是否就可以自然地导出随机采样？本文来讨论这个问题，并分享自己的解决方案。

## 要点分析

现阶段，Unigram分词是直接输出最大概率的切分方案，通常这是一个确定性的输出。具体来说，假设$\boldsymbol{w}=(w_1,w_2,\cdots,w_k)$代表一个切分方案，对应的打分为$P(\boldsymbol{w})=p(w_1)p(w_2)\cdots p(w_k)$，$\Omega(S)$代表句子$S$所有可能的切分方案的集合，那么分词算法可以描述为
\begin{equation}\boldsymbol{w}^* = \mathop{\text{argmax}}_{\boldsymbol{w}\in \Omega(S)}P(\boldsymbol{w})\end{equation}
这可以通过Viterbi算法在线性时间内来完成，所以这个过程我们也称之为“Viterbi Decoding”。看起来，Unigram模型天然带有概率，所以似乎并不难将它改为依概率采样的形式，但细想之下才发现这并非一个平凡的问题，有很多细节上的困难需要克服。

笔者设想是模仿自回归语言模型设计一个递归采样流程，但这里最困难的地方是如何尽量保持原来的候选切分方案的排序不变，或者就算不能保持所有的排序不变，也至少满足最大概率不变，即Viterbi解码的最大概率路径$\boldsymbol{w}^*$应该对应所设计的递归采样算法的最大概率采样结果。由于所有切分方案$\Omega(S)$构成一个有向无环图（DAG，Directed Acyclic Graph），笔者一开始以为直接在有向无环图上随机游走是一个可行方案，但再思考后发现很难设计适当的转移概率来保证最大概率路径不变（因为同一起点的不同边不是平权的，不能简单按照边的频率为权重做采样）。

## 已有方案

由于一时半会没有新想法，所以笔者决定去翻翻“参考答案”——看看Subword Regularization的原始论文[《Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates》](<https://papers.cool/arxiv/1804.10959>)是怎么做的。

然而，这个“标准答案”却让笔者有点哭笑不得。原来Subword Regularization的思路非常简单直接：先搜索出$P(\boldsymbol{w})$最大的$n$个分词方案$\boldsymbol{w}^*_1,\boldsymbol{w}^*_2,\cdots,\boldsymbol{w}^*_n$（$n$\-best segmentations），然后构建如下分布
\begin{equation}p_i = \frac{P(\boldsymbol{w}^*_i)^{\alpha}}{\sum\limits_{j=1}^n P(\boldsymbol{w}^*_j)^{\alpha}}\end{equation}
对这$n$个方案进行依概率采样，其中$\alpha > 0$是一个超参数。该算法已经集成在SentencePiece中，读者可以自行测试（使用方法参考[这里](<https://github.com/google/sentencepiece/tree/master#subword-regularization-and-bpe-dropout>)）。

问题是，“简单直接”不代表着“高效”，尽管搜索top\-$n$个分词方案最优方案的复杂度也是线性的（有兴趣的读者可以自行找找N\-best Viterbi的资料)，但明显比只找top1的Viterbi Decoding要大很多（理论上是$n$倍复杂度），所以直接的后果是开启了随机采样后，会比确定性的分词要慢很多，所以这并非是笔者心中的理想采样方法。

## 个人思路

思路再度陷入了僵局。一筹莫展之际，笔者决定把思路再捋一捋：我们的目标是想要找到复杂度类似Viterbi Decoding的随机采样算法，既然如此，Viterbi Decoding本身应该是一个不错的突破点。于是，笔者再次翻开了分词代码，[当时的分词函数](<https://github.com/bojone/bytepiece/blob/b65716b76938b3ac4124661a3367fc1c270373fa/bytepiece/faster.pyx>)长这个样：



<pre><code class="language-python">def _tokenize(self, bytes text):
    cdef int e, k, s
    cdef double v, score
    cdef list routes = [(0, None)] + [(-INFINITY, None) for _ in text]
    cdef list tokens = []
    for e, (k, v) in self._automaton.iter(text):
        s, e = e - k + 1, e + 1
        score = routes[s][0] + v
        if score &gt; routes[e][0]:
            routes[e] = score, s
    while text:
        s = routes[e][1]
        tokens.append(text[s:e])
        text, e = text[:s], s
    return tokens[::-1]</code></pre>



反复读了几遍，总算有了灵感：Viterbi Decoding的关键是<code>if score &gt; routes[e][0]:</code>这一句，它代表保留截止到当前位置的最优切分方案，其中<code>score</code>是新切分方案分数（概率对数），<code>routes[e][0]</code>是历史最优分数，如果新方案更优则覆盖。这让笔者联想到了[MCMC算法](<https://spaces.ac.cn/archives/8084>)的接受率设计，如果在这里引入随机采样，不就可以将分词结果随机化了？

我们用$r\in \{1, 0\}$表示接受/拒绝新方案，由于这一步只是一个二元选择，所以将它概率化也非常简单：
\begin{equation}
r_i = \left\{\begin{aligned}&\,1\,, \,\, s_i > s_{i-1} \\
&\,0\,, \,\, \text{else}\end{aligned}\right.\qquad\longrightarrow\qquad
r_i = \left\{\begin{aligned}&\,1\,, \,\, \varepsilon < \sigma(\alpha(s_i - s_{i-1})) \\
&\,0\,, \,\, \text{else}\end{aligned}\right.
\end{equation}
这里$\varepsilon\sim U[0,1]$是均匀随机数，$\alpha > 0$是超参数，$\sigma(t)=1/(1+e^{-t})$是Sigmoid函数，$s_i,s_{i-1}$分别是新旧方案的得分（概率对数）。不难发现，左端的确定性采样对应$\alpha\to\infty$的随机性采样。

这样，在Viterbi解码的基础上我们得到了一个非常自然、非常轻量级的随机采样算法，这里称之为“Viterbi Sampling”，实现它只需要将<code>if score &gt; routes[e][0]:</code>这一判据换成带随机数的版本。由于Sigmoid函数的单调性，当$s_i > s_{i-1}$时，它自然会给新方案分配更大的概率，所以很明显原来的的最大概率切分在Viterbi Sampling之下也是最大概率结果，并且当$s_i - s_{i-1}$越大，$\sigma(\alpha(s_i - s_{i-1}))$也越大，这意味着原本得分越大的方案被采样到的概率也越高，一定程度上保持了切分方案的排序不变（尽管还没有证明一定严格保序，但从应用角度看，近似保序就够了）。

## 简单测试

从0\.4\.0版本开始，Viterbi Sampling就内置在BytePiece的分词函数中，只需要在<code>tokenizer.tokenize</code>或者<code>tokenizer.encode</code>时加入大于0的alpha参数，结果就是随机的：



<pre><code class="language-python">import bytepiece
assert bytepiece.__version__ &gt;= '0.4.0'

tokenizer = bytepiece.Tokenizer('bytepiece_160k.model')
text = '今天天气不错'
print(tokenizer.tokenize(text))  # alpha默认值为-1，alpha≤0 都代表确定性分词
for i in range(5):
    print(tokenizer.tokenize(text, alpha=0.1))

# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6', b'\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9\xe6\xb0\x94', b'\xe4\xb8', b'\x8d', b'\xe9\x94', b'\x99']
# [b'\xe4\xbb\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9', b'\xe6\xb0\x94', b'\xe4\xb8\x8d\xe9\x94\x99']
# [b'\xe4\xbb', b'\x8a\xe5\xa4\xa9', b'\xe5\xa4\xa9', b'\xe6\xb0\x94\xe4\xb8\x8d', b'\xe9\x94', b'\x99']
</code></pre>



下面对比一下SentencePiece的Subword Regularization和BytePiece的Viterbi Sampling的速度（随机性分词时都设$\alpha=0.1$）：
\begin{array}{c|cc}
\hline
& \text{确定性分词} & \text{随机性分词} & \\
\hline
\text{SP-BPE} & \text{1.36M bytes/sec} & \text{1.25M bytes/sec}  \\
\text{SP-Unigram} & \text{5.65M bytes/sec} & \text{1.28M bytes/sec} \\
\text{BytePiece} & \text{1.95M bytes/sec} & \text{1.36M bytes/sec}\\
\hline
\end{array}
可以看到，Subword Regularization（“SP\-Unigram”这一行）开启之后，分词速度不到原来的1/4，这表明Subword Regularization的采样算法是相当低效的。相比之下，本文提出的Viterbi Sampling只下降了30%左右，效率显然更高，下降的部分在于随机数的生成和Sigmoid函数的计算，如果能进一步优化这两部分，速度还能进一步提升。至于BPE模型，它的随机分词叫做[BPE Dropout](<https://papers.cool/arxiv/1910.13267>)，这是专属于BPE模型的方法，有兴趣的读者自行了解，这里就不介绍了。

## 文章小结

本文主要探讨了将Unigram分词模型的确定性分词改为随机性分词的策略。尽管已有名为“Subword Regularization”的方法可以实现这一目标，但其效率相对较低。为此，笔者提出了一种更高效的采样算法Viterbi Sampling，它仅需对确定性的Viterbi Decoding进行简单的修改，从而基本保持了原有的效率。实验证明，新的算法采样速度明显超越了Subword Regularization。相应的实现已经内置在BytePiece最新版中。
