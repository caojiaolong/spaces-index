# Bytepiece 0\.8\.0：更高效的Unigram Tokenizer

> 作者：苏剑林 · 科学空间 · 2026-11-28
>
> 原文：<https://spaces.ac.cn/archives/12024>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从算法上来看，常用的Tokenizer可以分为BPE和Unigram两种。当前的主流无疑是BPE，大体上是因为在主流实现上（比如[SentencePiece](<https://github.com/google/sentencepiece>)），BPE的压缩率通常更高。然而，我们三年前的[BytePiece](<https://spaces.ac.cn/archives/9752>)已经否定了这个结果，在笔者所提的BNLM训练算法上，Unigram有着比BPE更高的压缩率和更快的训练速度！

不过，受限于笔者的编程能力，当年BytePiece的实现本身还是很低效的，没能推动Unigram的普及。现在有了AI，编程自然不是困难了，所以这两天“一时兴起”，决定让Kimi好好升级一下BytePiece。

> <strong>Github：[https://github\.com/bojone/bytepiece](<https://github.com/bojone/bytepiece>)</strong>

## 做了什么

BytePiece的细节我们就不重复展开了，感兴趣的读者自行翻看[《BytePiece：更纯粹、更高压缩率的Tokenizer》](<https://spaces.ac.cn/archives/9752>)。简单来说，BytePiece是一个以Byte为单位的Unigram Tokenizer，其中Unigram不是新的，包括[结巴分词](<https://github.com/fxsjy/jieba>)等都是基于这个原理，BytePiece的主要贡献是提供了一种高效的、不需要预边界的训练方案。

此外，Unigram是一个概率模型，天然支持采样（随机分词），但此前的实现（如SentencePiece）效率较低。借助于我们之前提出的[《随机分词浅探：从Viterbi Decoding到Viterbi Sampling》](<https://spaces.ac.cn/archives/9768>)和[《随机分词再探：从Viterbi Sampling到完美采样算法》](<https://spaces.ac.cn/archives/9811>)，BytePiece实现了显著更快的随机分词。

问题在于，训练BytePiece需要按Byte去建立一个统计N\-Gram模型，这并不是Python所擅长的，当初笔者勉强写了一版，自然算不上高效（但也能与当时的SentencePiece掰掰手腕了）。处理Byte流更高效的方式是C\+\+，这次主要也是让Kimi用C\+\+重写了整个训练过程以及分词函数，更高效的同时摆脱了pyahocorasick、Cython的依赖。

具体改动包括：用C\+\+写的双数组Trie替代旧的AC自动机库，来构建分词阶段的核心索引；BNLM训练方面，基本全改为在C\+\+内完成，用紧凑Hash、共享内存多线程等技术提速并降低内存需求。其实这些都是NLP时代很经典很成熟的技术，只不过如今在Kimi的帮助下搬到了BytePiece的优化中。

这次的版本号，直接从旧版本的0\.6\.3，跳到了0\.8\.0，表明这是一次巨大的变化。除了加速之外，0\.8\.0还参考SentencePiece、HF Tokenizer等主流Tokenizer做了扩充完善，补齐大模型生产管线能力，使其成为一个功能相对完备的Tokenizer实现。

## 效果如何

当然了，这三年来主流的Tokenizer实现也在进步，一个明显的区别是，当年我们对比的SentencePiece还是0\.1\.99版，BPE的训练速度极慢，现在最新是0\.2\.3版，BPE和Unigram的结论也有所变化。所以升级完成后，我们还需要跟各种主流实现好好对比一下。

当初的实验语料已经找不到了，所以这次重新用HF上的开源语料做所有实验，也方便大家复现对比。首先是0\.8\.0和0\.6\.3两个新旧版本的对比（在96M语料上训练50k的词表）：
$$\begin{array}{c|cccc}
\hline
& \text{训练时间}\downarrow & \text{峰值内存}\downarrow & \text{压缩率}\uparrow & \text{分词速度 (中/英/代码)}\uparrow \\
\hline
\textbf{BytePiece 0.8.0} & \textbf{3.60} & \textbf{890} & \textbf{4.410} & \textbf{57.1 / 25.1 / 32.8} \\
\text{BytePiece 0.6.3} & \text{75.38} & \text{42,387} & 4.410 & \text{3.9 / 2.7 / 2.7} \\
\hline
\text{提升倍数} & {20.9\times} & {47.6\times} & \text{一致} & {14.6\times}\,/\,{9.3\times}\,/\,{12.1\times} \\
\hline
\end{array}$$
其中训练时间的单位是秒（s），内存的单位是MB，压缩率的单位是“bytes/token”，分词速度的单位是“MB/s”，可见新版是全方面数量级的提升！注意，这次升级是纯粹实现层面的效率提升，没有做算法改进，训练和分词效果是跟0\.6\.3版完全对齐的。

（注：0\.6\.3版内存达到42G，是因为开了多进程，这里汇总了所有进程所用的内存之和，主进程所用内存大概是2G多一点）

接下来是在同一份语料上，跟主流的训练和分词实现对比：
$$\begin{array}{c|cccc}
\hline
& \text{训练时间}\downarrow & \text{峰值内存}\downarrow & \text{压缩率}\uparrow & \text{分词速度 (中/英/代码)}\uparrow \\
\hline
\textbf{BytePiece 0.8.0} & \textbf{3.60} & \textbf{890} & \textbf{4.410} & \textbf{57.1 / 25.1 / 32.8} \\
\hline
\text{SP-Unigram 0.1.99 (旧版)} & \text{51.34} & \text{3,142} & 3.266 & \text{11.8 / 7.8 / 6.9} \\
\text{SP-Unigram 0.2.3 (新版)} & \text{60.99} & \text{1,884} & 3.269 & \text{31.9 / 17.4 / 18.9} \\
\text{SP-BPE 0.1.99 (旧版默认)} & \text{625.31} & \text{2,356} & 3.429 & \text{6.3 / 2.2 / 2.5} \\
\text{SP-BPE 0.2.3 (新版默认)} & \text{9.24} & \text{1,683} & 3.431 & \text{17.6 / 4.1 / 4.6} \\
\text{SP-BPE-Raw 0.2.3 (关空白切分)} & \text{22.77} & \text{3,577} & 4.167 & \text{17.8 / 3.6 / 4.1} \\
\hline
\text{HF tokenizers BPE 0.23.3} & \text{28.00} & \text{2,931} & 3.893 & \text{3.2 / 2.7 / 2.4} \\
\text{HF tokenizers Unigram 0.23.3} & \text{44.72} & \text{2,842} & 3.319 & \text{2.1 / 2.1 / 1.6} \\
\hline
\text{参考：tiktoken o200k_base} & - & - & 3.762 & \text{12.8 / 20.7 / 19.1} \\
\hline
\end{array}$$
由此可见，SentencePiece（SP）新版比旧版有明显提升，但还是比不过新版的BytePiece；HF Tokenizer的训练和分词速度也没有优势；Tiktoken没有训练模块，只能用它现成的词表，比一下压缩率和分词速度了。

## 延伸讨论

总的来说，BPE和Unigram都是基于最大化压缩率思想的Tokenizer，区别在于，它们对“压缩率”的定义有所不同。在BPE中，每个Token的代价固定为$1$，所以最大化压缩率实际上是最短序列长度。BPE的标准训练算法是从细到粗的贪心合并，但这不是唯一的，比如还有[《Tokenisation via Convex Relaxations》](<https://papers.cool/arxiv/2605.22821>)提过基于线性规划的理论上更优的算法。

BPE的压缩率，也正是本文测试中的压缩率指标，这个指标也跟如今LLM的训练和推理效率直接相关，所以它用作最大化的目标无可厚非。但是，从传统的压缩学角度看，这个压缩率的定义是不妥的，它没有考虑Token实际的存储成本——词表越大，每个Token的平均表示成本越大，不可能一直是固定的1。

理论上，最高效的压缩方式是给出现概率为$p$的Token，分配$-\log p$的描述长度，所以我们应该最小化总描述长度
$$-\sum_{w\in \text{语料}} \log p_w$$
其含义是熵，这就引出了“[最小熵](<https://spaces.ac.cn/tag/%E6%9C%80%E5%B0%8F%E7%86%B5/>)”系列工作，这也是Unigram的优化目标。说白了，BPE和Unigram的核心区别是每个Token的代价是固定的$1$还是动态的$-\log p$。值得指出的是，BytePiece的训练算法是后者，但即便改用BPE追求的等权压缩率指标居然也更好，这对于BytePiece来说其实是意外之喜。

体现两者区别的经典例子是：假设训练语料的每个字节是完全均匀随机的，那么Unigram的最佳词表大小就是256，它不会鼓励加大词表，但对于BPE来说，加大词表总能缩小长度，哪怕两个没有关联的字节，也可能会被合并起来，但这不一定利于下游效果。

那么问题来了，谁对LLM的最终效果更有利？从压缩率角度看，BytePiece的Unigram自然已经胜出，但压缩率越高，并不代表着端到端的LLM效果一定更好，这是个很复杂的问题，笔者也不知道答案。顺便说，LLM的训练目标是交叉熵，这实际上也可以理解为带$-\log p$的压缩目标，跟Unigram更为一致，这种一致性是否有利于端到端效果，暂时也不得而知。

只是，从传统NLP的无监督学习经验来看，最小熵带来的收获要比最短长度带来的收获更多，并且更符合人类理解。如果我们认为这些经验对LLM也成立，那么可能是Unigram更好，而且在我们的新训练算法下，Unigram的等权压缩率也不差，所以大家不妨试试BytePiece。

## 文章小结

时隔三年，在Kimi的帮助下，总算让BytePiece也跟上了最先进的性能，也算是对当初所提的算法有了个更好的交代，希望Unigram模型能由此获得更多的关注。后面如果有更新的想法，也会继续把这个项目更新下去的。
