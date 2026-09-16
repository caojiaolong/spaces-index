# BytePiece：更纯粹、更高压缩率的Tokenizer

> 作者：苏剑林 · 科学空间 · 2023-09-07
>
> 原文：<https://spaces.ac.cn/archives/9752>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

目前在LLM中最流行的Tokenizer（分词器）应该是Google的[SentencePiece](<https://github.com/google/sentencepiece>)了，因为它符合Tokenizer的一些理想特性，比如语言无关、数据驱动等，并且由于它是C\+\+写的，所以Tokenize（分词）的速度很快，非常适合追求效率的场景。然而，它也有一些明显的缺点，比如训练速度慢（BPE算法）、占用内存大等，同时也正因为它是C\+\+写的，对于多数用户来说它就是黑箱，也不方便研究和二次开发。

事实上，Tokenizer的训练就相当于以往的“新词发现”，而笔者之前也写过[中文分词](<https://spaces.ac.cn/search/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D%E7%B3%BB%E5%88%97/>)和[最小熵](<https://spaces.ac.cn/tag/%E6%9C%80%E5%B0%8F%E7%86%B5/>)系列文章，对新词发现也有一定的积累，所以很早之前就有自己写一版Tokenizer的想法。这几天总算腾出了时间初步完成了这件事情，东施效颦SentencePiece，命名为“BytePiece”。

> <strong>Github：[https://github\.com/bojone/bytepiece](<https://github.com/bojone/bytepiece>)</strong>

## 理想特性

既然要重写Tokenizer，那么我们就要思考一个理想的Tokenizer应该是怎样的，这样才能判断最终是否达到了预期。照笔者看来，Tokenizer至少应该具备如下基本特性：

> 1、<strong>无损重构</strong>：分词结果应该可以无损还原为输入；
> 
> 
> 
> 2、<strong>高压缩率</strong>：词表大小相同时，同一批数据的tokens数应该尽可能少；
> 
> 
> 
> 3、<strong>语言无关</strong>：基于统计，训练和分词过程都不应引入语言特性；
> 
> 
> 
> 4、<strong>数据驱动</strong>：可以直接基于原始语料进行无监督训练；
> 
> 
> 
> 5、<strong>训练友好</strong>：能够在合理的时间和配置上完成训练过程。

最后，还有一些加分项，比如分词速度快、代码易读、方便二次拓展等，这些满足自然最好，但笔者认为可以不列入基本特性里边。

对于笔者来说，SentencePiece最大的槽点就是“无损重构”和“训练友好”。首先，SentencePiece默认会进行[NFKC normalization](<https://github.com/google/sentencepiece/blob/master/doc/normalization.md>)，这会导致“全角逗号转半角逗号”等不可逆变化，所以默认情况下它连“无损重构”都不满足，所以很长时间里它都不在笔者的候选名单中，直到后来发现，在训练时添加参数<code>--normalization_rule_name=identity</code>就可以让它不做任何转换。所以SentencePiece算是支持无损重构，只不过要特别设置。

至于训练方面，就更让人抓狂了。SentencePiece支持BPE和Unigram两种主流算法，Unigram训练速度尚可，但压缩率会稍低一些，BPE的压缩率更高，但是训练速度要比Unigram慢上一个数量级！而且不管是BPE还是Unigram，训练过程都极费内存。总而言之，用较大的语料去训练一个SentencePiece模型真不是一种好的体验。

## 模型构思

一个新Tokenizer的构建，可以分解为三个部分：1、基本单元；2、分词算法；3、训练算法。确定这三个部分后，剩下的就只是编程技巧问题了。下面逐一介绍BytePiece对这些问题的思考。

### 基本单元

我们知道，Python3的默认字符串类型是Unicode，如果以Unicode为基本单位，我们称之为Char\-based。Char\-based很直观方便，汉字表现为长度为1的单个字符，但不同语言的Char实在太多，即便只是覆盖单字都需要消耗非常大的vocab\_size，更不用说引入Word。所以BytePiece跟主流的Tokenizer一样，以Byte为基本单位。

回到Byte之后，很多问题都“豁然开朗”了。因为不同的单Byte只有256个，所以只要词表里包含了这256个单Byte，那么就可以杜绝OOV（Out of Vocabulary），这是它显而易见的好处。此外，我们知道汉字的平均信息熵要比英文字母的平均信息熵要大，如果我们选择Char\-based，那么虽然每个Char表面看起来长度都是1，但“内在”的颗粒度不一样，这会导致统计结果有所偏置。相比之下，每个Byte的信息熵则更加均匀【比如，大部分汉字的UTF\-8编码对应3个Byte，而汉字的平均信息熵正好是英文字母（对应一个Byte）的2～3倍左右】，因此用Byte的统计结果会更加无偏，这将会使得模型更加“语言无关”。

在Byte\-based方面，BytePiece比SentencePiece更彻底，SentencePiece是先以Char\-based进行处理，然后遇到OOV再以Byte\-based处理，BytePiece则是在一开始就将文本通过<code>text.encode()</code>转为Bytes，然后才进行后续操作，相比之下更加纯粹。

### 分词算法

基于词典进行分词的算法无非就那几种，比如最大匹配、最短路径、最大概率路径等，有兴趣追溯的读者，可以参考Matrix67之前写的[《漫话中文自动分词和语义识别（上）：中文分词算法》](<http://www.matrix67.com/blog/archives/4212>)，

跟jieba等中文分词工具一样，BytePiece选择的是最大概率路径分词，也称“一元文法模型”，即Unigram。选择Unigram有三方面的考虑：第一，Unigram的最大概率换言之就是最大似然，而LLM的训练目标也是最大似然，两者更加一致；第二，从压缩的角度看，最大概率实际上就是最短编码长度（也叫最小描述长度），是压缩率最大化的体现，这也跟“压缩就是智能”的信仰一致；第三，Unigram求最优分词方案可以通过Viterbi算法在线性复杂度内完成，这是理论最优的复杂度了。

当然，既然有“一元文法模型”，自然也有更复杂的“二元文法模型”、“三元文法模型”等，但它们的复杂度增加远大于它能带来的收益，所以我们通常不考虑这些高阶模型。

### 训练算法

之所以先讨论分词算法在讨论训练算法，是因为只有分词算法确定下来后，才能确定训练的优化目标，从而研究对应的训练算法。

开头就提到，Tokenizer的训练本质上就是以往的“新词发现”，而笔者之前也提了好几种新词发现算法，如[《基于切分的新词发现》](<https://spaces.ac.cn/archives/3913>)、[《基于语言模型的无监督分词》](<https://spaces.ac.cn/archives/3956>)、[《更好的新词发现算法》](<https://spaces.ac.cn/archives/4256>)。现在看来，跟Unigram分词算法最契合、最有潜力的，应该是[《基于语言模型的无监督分词》](<https://spaces.ac.cn/archives/3956>)，BytePiece的训练就是基于它实现的，这里称之为<strong>Byte\-based N\-gram Language Model（BNLM）</strong>。

具体来说，对于Unigram分词，如果一个长度为$l$的字节串$c_1, c_2, \dots, c_l$，最优分词结果为$w_1, w_2, \dots, w_m$，那么概率乘积$p(w_1)p(w_2)\dots p(w_m)$应该是所有切分中最大的。设$w_1,w_2,\cdots,w_m$的长度分别为$l_1,l_2,\cdots,l_m$，那么根据条件分解公式
\begin{equation}\prod_{i=1}^m p(w_i) = \prod_{i=1}^m \prod_{j=L_{i-1} + 1}^{j=L_{i-1} + l_i} p(c_j|c_{L_{i-1} + 1},\cdots,c_{j-1})\end{equation}
这里$L_i=l_1+l_2+\cdots+l_i$。只考虑$n$\-gram模型，将$j\gt L_{i-1} + n$的$p(c_j|c_{L_{i-1} + 1},\cdots,c_{j-1})$统一用$p(c_j|c_{j - n + 1},\cdots,c_{j-1})$近似，那么Unigram分词就转化为一个字（节）标注问题，而Tokenizer的训练则转化为$n$\-gram语言模型的训练（推荐$n=6$），可以直接无监督完成。更详细的介绍请读者移步原文[《基于语言模型的无监督分词》](<https://spaces.ac.cn/archives/3956>)。

（注意：$n=6$只是说BytePiece的统计信息最多到6\-gram，但并非最大只能生成长度为6的piece，因为大于$6$的$n$\-gram条件概率我们会用6\-gram的近似，所以它是可以做到任意阶的，即理论上可以生成任意长度piece。）

## 代码实现

原理确定之后，剩下的就是枯燥的开发工作了。幸不辱命，勉强写出了一套可用的代码：

> <strong>Github：[https://github\.com/bojone/bytepiece](<https://github.com/bojone/bytepiece>)</strong>

代码很简单，单文件，里边就<code>Trainer</code>和<code>Tokenizer</code>两个类，分别对应分词两部分。分词借助[pyahocorasick](<https://github.com/WojciechMula/pyahocorasick>)来构建AC自动机来稍微提了一下速，能凑合用，但还是会比SentencePiece慢不少，毕竟速度方面纯Python跟C\+\+确实没法比。训练则分为四个主要步骤：1、$n$\-gram计数；2、$n$\-gram剪枝；3、预分词；4、预分词结果剪枝。其中1、3、4都是计算密集型，并且都是可并行的，所以编写了相应的多进程实现。在开足够多的进程（笔者开了64进程，每个进程的使用率基本上都是满的）下，训练速度能媲美SentencePiece的Unigram训练速度。

这里特别要提一下结果剪枝方面。剪枝最基本的依据自然是频数和vocab\_size，但这还不够，因为有时候会出现$p(w_1)p(w_2) > p(w_1\circ w_2)$（$w_1\circ w_2$指两个词拼接）且$w_1,w_2,w_1\circ w_2$三个词都在词表中，这种情况下$w_1\circ w_2$这个词永远不会切分出来，所以将它放在词表中是纯粹浪费空间的，因此剪枝过程也包含了这类结果的排除。

## 效果测试

到了大家喜闻乐见的测试环节，是骡子是马总要拉出来遛遛。首先做个小规模的测试，从悟道之前开源的数据集里边随机采样10万条作为训练集（导出来的文件大概330MB），然后另外采样1千作为测试集，训练一个vocab\_size=50k的词表，结果对比如下：
\begin{array}{c|ccc}
\hline
& \text{训练时间}\downarrow & \text{最大内存占用}\downarrow & \text{压缩率}\uparrow \\
\hline
\text{SP-BPE} & \text{55.3分钟} & \text{5.2GB} & 4.80 \\
\text{SP-Unigram} & \text{1.6分钟} & \text{2.5GB} & 4.73 \\
\text{BytePiece} & \text{6.5分钟} & \text{4.3GB} & 5.05 \\
\hline
\end{array}
解释一下，这里SP\-BPE、SP\-Unigram分别指SentencePiece的model\_type设为BPE和Unigram，训练代码分别是



<pre><code class="language-python">spm.SentencePieceTrainer.train('--input=wudao.txt --model_prefix=wudao_m --vocab_size=50000 --model_type=bpe --train_extremely_large_corpus=true --normalization_rule_name=identity')

spm.SentencePieceTrainer.train('--input=wudao.txt --model_prefix=wudao_m2 --vocab_size=50000 --model_type=unigram --train_extremely_large_corpus=true --normalization_rule_name=identity')</code></pre>



压缩率的单位是“bytes/token”，即平均每个token对应的字节数。可见，BytePiece能够在训练时间和内存都比较折中的情况下，获得最大的压缩率。

接下来进行一个更大规模的测试。从中英比例大致为3:5的混合语料库中，抽取出10万条样本训练vocab\_size=100k的Tokenizer。这个语料库的文本都比较长，所以这时候10万条导出来的文件已经13GB了，测试集包含两部分，一部分是同样的语料库中采样出1000条（即同源），另一部分是刚才采样出来的1000条悟道数据集（代表不同源）。结果如下：
\begin{array}{c|cccc}
\hline
& \text{训练时间}\downarrow & \text{最大内存占用}\downarrow & \text{压缩率(同源)}\uparrow & \text{压缩率(异源)}\uparrow \\
\hline
\text{SP-BPE} & \text{19.21小时} & \text{97GB} & 4.52 & 4.46 \\
\text{SP-Unigram} & \text{2.02小时} & \text{384GB} & 4.51 & 4.48 \\
\text{BytePiece} & \text{2.24小时} & \text{51GB} & 5.39 & 4.51\\
\hline
\end{array}

不管是训练时间、内存还是压缩率，看起来训练数据量越大，BytePiece越有优势！

## 未完待续

就目前的结果看来，BytePiece在训练方面是有一定优势的，分词效果也尚可，不过吃了纯Python的亏，分词速度只有SentencePiece的1/10左右，这是未来的一个优化方向之一，期待有C/C\+\+大牛能参与进来，帮助提升BytePiece的分词速度。（注：从0\.2\.0版开始，使用Cython加速了分词函数，目前BytePiece的分词速度已经接近BPE，并且在文本足够长时能优于BPE。）

实际上，如果采用随机采样、动态剪枝等技术，BytePiece的训练速度和内存都还可以进一步优化。目前BytePiece为了保证结果的确定性，直到所有结果都统计完毕才进行剪枝，这样不管是单进程还是多进程，都能保证结果的一致性。如果随机打乱输入，并且定时进行剪枝，那么可以进一步控制内存的占用量，同时还能加快统计速度，并且可以预期对最终效果的影响也不大。这部分工作，也在后面根据用户体验进一步引入。

除了以上这些，BytePiece细节之处还有不少需要完善的地方，以及可能还有未发现的错漏之处，敬请大家海涵且反馈

## 文章小结

本文介绍了笔者自行开发的Tokenizer——BytePiece，它是Byte\-based的Unigram分词器，纯Python实现，更加易读和易拓展。由于采用了新的训练算法，所以压缩率通常比现有tokenizer更高，同时支持多进程加速训练。此外，它直接操作文本的utf\-8 bytes，几乎不进行任何的预处理，所以更加纯粹和语言无关。
