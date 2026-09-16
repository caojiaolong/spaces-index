# 那个屠榜的T5模型，现在可以在中文上玩玩了

> 作者：苏剑林 · 科学空间 · 2020-11-06
>
> 原文：<https://spaces.ac.cn/archives/7867>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

不知道大家对Google去年的屠榜之作T5还有没有印象？就是那个打着“万事皆可Seq2Seq”的旗号、最大搞了110亿参数、一举刷新了GLUE、SuperGLUE等多个NLP榜单的模型，而且过去一年了，T5仍然是[SuperGLUE](<https://super.gluebenchmark.com/>)榜单上的第一，目前还稳妥地拉开着第二名2%的差距。然而，对于中文界的朋友来说，T5可能没有什么存在感，原因很简单：没有中文版T5可用。不过这个现状要改变了，因为Google最近放出了多国语言版的T5（mT5），里边当然是包含了中文语言。虽然不是纯正的中文版，但也能凑合着用一下。

[![“万事皆可Seq2Seq”的T5](<https://spaces.ac.cn/usr/uploads/2020/11/4211830191.png>)](<https://spaces.ac.cn/usr/uploads/2020/11/4211830191.png>)

“万事皆可Seq2Seq”的T5

本文将会对T5模型做一个简单的回顾与介绍，然后再介绍一下如何在bert4keras中调用mT5模型来做中文任务。作为一个原生的Seq2Seq预训练模型，mT5在文本生成任务上的表现还是相当不错的，非常值得一试。

## T5

跟BERT一样，T5也是Google出品的预训练模型，来自论文为[《Exploring the Limits of Transfer Learning with a Unified Text\-to\-Text Transformer》](<https://papers.cool/arxiv/1910.10683>)，Github为[text\-to\-text\-transfer\-transformer](<https://github.com/google-research/text-to-text-transfer-transformer>)。T5的理念就是“万事皆可Seq2Seq”，它使用了标准的Encoder\-Decoder模型，并且构建了无监督/有监督的文本生成预训练任务，最终将效果推向了一个新高度。

### 训练

T5的预训练包含无监督和有监督两部分。无监督部分使用的是Google构建的近800G的语料（论文称之为C4），而训练目标则跟BERT类似，只不过改成了Seq2Seq版本，我们可以将它看成一个高级版的完形填空问题：

> <strong>输入：</strong>明月几时有，&#91;M0&#93;问青天，不知&#91;M1&#93;，今夕是何年？我欲&#91;M2&#93;归去，又恐琼楼玉宇，高处&#91;M3&#93;；起舞&#91;M4&#93;清影，何似在人间。
> <strong>输出：</strong>&#91;M0&#93;把酒&#91;M1&#93;天上宫阙&#91;M2&#93;乘风&#91;M3&#93;不胜寒&#91;M4&#93;弄

而有监督部分，则是收集了常见的NLP监督任务数据，并也统一转化为SeqSeq任务来训练。比如情感分类可以这样转化：

> <strong>输入：</strong>识别该句子的情感倾向：这趟北京之旅我感觉很不错。
> <strong>输出：</strong>正面

主题分类可以这样转化：

> <strong>输入：</strong>下面是一则什么新闻？八个月了，终于又能在赛场上看到女排姑娘们了。
> <strong>输出：</strong>体育

阅读理解可以这样转化：

> <strong>输入：</strong>阅读理解：特朗普与拜登共同竞选下一任美国总统。根据上述信息回答问题：特朗普是哪国人？
> <strong>输出：</strong>美国

可以看到，这种转化跟GPT2、GPT3、PET的思想都是一致的，都是希望用文字把我们要做的任务表达出来，然后都转化为文字的预测，读者还可以翻看旧作[《必须要GPT3吗？不，BERT的MLM模型也能小样本学习》](<https://spaces.ac.cn/archives/7764>)了解相关内容。总的来说，在我们的内部实验里边，模型够大、数据够多以及有监督预训练都是T5成功的关键因素，“万事皆可Seq2Seq”则提供了有效地融合这些关键因素的方案。

### 结果

T5的主要战绩汇总如下表：

[![T5的战绩汇总](<https://spaces.ac.cn/usr/uploads/2020/11/358540148.png>)](<https://spaces.ac.cn/usr/uploads/2020/11/358540148.png>)

T5的战绩汇总

除了屠了多个榜单之外，T5还对整个训练流程中很多可调的超参数都调试了一遍，比如模型架构究竟用标准的Encoder\-Decoder好还是UniLM那种结构好，无监督预训练任务究竟是BERT的方式好还是其他方式好，随机Mask的比例是不是15%最好，等等，最后给出了如下的表格，并还很遗憾地表达了一句“其实我们觉得T5的实验做得还不是很充分”，颇有一种“走别人的路，让别人无路可走”的感觉。当然，不管怎样，这些炼丹结果还是值得每一位要做语言模型的同学好好看看，或许能让我们少走一些弯路。

[![T5那巨细无遗的“炼丹宝典”（点击可以看大图）](<https://spaces.ac.cn/usr/uploads/2020/11/8208956.png>)](<https://spaces.ac.cn/usr/uploads/2020/11/8208956.png>)

T5那巨细无遗的“炼丹宝典”（点击可以看大图）

## mT5

至于mT5，即Multilingual T5，T5的多国语言版，出自最近的论文[《mT5: A massively multilingual pre\-trained text\-to\-text transformer》](<https://papers.cool/arxiv/2010.11934>)，Github为[multilingual\-t5](<https://github.com/google-research/multilingual-t5>)，这也是将多语种NLP任务的榜单推到了一个新高度了。当然，对我们来说，最重要的是mT5里边包含了中文，因此我们终于有机会在中文任务中尝试下T5了。

### T5\.1\.1

总的来说，mT5跟T5一脉相承的，整体基本一样，但在模型结构方面，mT5使用的是T5\.1\.1方案，在此对它做个基本的介绍。

很多人都不知道的是，自从在去年10月发布后，T5在今年还经历了一次低调的小升级，具体细节可以查看[Github链接](<https://github.com/google-research/text-to-text-transfer-transformer/blob/master/released_checkpoints.md>)，官方把升级前的T5称为T5\.1\.0，而升级后的叫做T5\.1\.1。它主要的改动来自论文[《GLU Variants Improve Transformer》](<https://papers.cool/arxiv/2002.05202>)，主要是借用了[《Language Modeling with Gated Convolutional Networks》](<https://papers.cool/arxiv/1612.08083>)的GLU（Gated Linear Unit）来增强FFN部分的效果。具体来说，原来T5的FFN为（T5没有Bias）
\begin{equation}\text{FFN}(x)=\text{relu}(xW_1)W_2\end{equation}
现在改为了
\begin{equation}\text{FFN}_{\text{GEGLU}}(x)=\big(\text{gelu}(xW_1)\otimes xW_2\big)W_3\end{equation}
也就是把relu激活的第一个变化层改为了gelu激活的门控线性单元，这样FFN层增加了50%参数，但是从论文效果看效果明显增加。此外，T5\.1\.1还对Embedding层做了改动，原来在T5\.1\.0中，Encoder和Decoder的Embedding层、Decoder最后预测概率分布的Softmax层都是共享同一个Embedding矩阵的，现在T5\.1\.1只让Encoder和Decoder的Embedding层共享，而Decoder最后预测概率分布的Softmax层则用了一个独立的Embedding矩阵，当然这会让参数量大大增加，但Google的结论说这样做效果会更好，其结论被总结在最近的论文[《Rethinking embedding coupling in pre\-trained language models》](<https://papers.cool/arxiv/2010.12821>)中。还有最后一点改动，T5\.1\.1在预训练阶段去掉了Dropout，而只有在下游微调阶段才使用Dropout。

经过这些调整后，Google重新训练并开放了全系列的T5\.1\.1模型，其下载地址可以在刚才的Github链接找到，注意T5\.1\.1只做了无监督预训练，但效果依然相当出色。由于T5\.1\.1提升明显，所以mT5也就继续使用了T5\.1\.1结构了

### 结果

mT5其实就是重新构建了多国语言版的数据集mC4，然后使用T5\.1\.1方案训练了一波，技术路线上没有什么明显的创新。关于训练细节，大家观察下原论文就好，论文其实也不长，毕竟T5已经把路都给铺好了。

至于mT5的战绩，主要就是集中在下面这张表内了：

[![mT5的“战绩”](<https://spaces.ac.cn/usr/uploads/2020/11/3733950397.png>)](<https://spaces.ac.cn/usr/uploads/2020/11/3733950397.png>)

mT5的“战绩”

读者可能会有疑问，这种多国语言版的该用什么方式评测？简单的话，我们可以直接在此基础上finetune一个跨语种的机器翻译任务，看看效果的提升。但事实上，对于多国语言版模型，研究人员更关心的是它在跨语种任务上的Zero Shot表现，说白了，就是同一种任务，在一个语种上进行finetune，其模型能不能直接用于其余语种？这也是上图中“Cross\-lingual zero\-shot transfer (models fine\-tuned on English data only)”的含义了，可以看到，mT5的表现还是相当出色的。

## 实践

终于到了大家喜闻乐见的实践时间了，这里我们简单介绍一下在bert4keras上使用mT5模型来做中文文本生成任务的流程和技巧。bert4keras从0\.9\.1版本开始支持调用mT5模型，如果要进行下述实验的读者，请先将bert4keras升级到0\.9\.1版或以上。

<strong>Github链接：[https://github\.com/bojone/t5\_in\_bert4keras](<https://github.com/bojone/t5_in_bert4keras>)</strong>

### 基本

用bert4keras把mT5模型加载到keras中的基本代码为



<pre><code class="language-python"># 模型路径
config_path = '/root/kg/bert/mt5/mt5_small/t5_config.json'
checkpoint_path = '/root/kg/bert/mt5/mt5_small/model.ckpt-1000000'
spm_path = '/root/kg/bert/mt5/sentencepiece.model'

# 加载分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='&lt;/s&gt;')

# 加载模型
t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model</code></pre>



可以看到跟在bert4keras中加载BERT没太大区别，其中<code>t5_config.json</code>的构建、<code>model.ckpt-1000000</code>的下载在Github上都有详细介绍，大家请移步去看。完整代码（训练和解码细节）在Github上也可以找到，这里就不展开了。

值得一提的是，对于中文来说，tokenizer给出的结果是带有词的，即对于中文来说mT5是以词为单位的，只不过词颗粒度会比较少。这进一步说明了我们之前的工作[《提速不掉点：基于词颗粒度的中文WoBERT](<https://spaces.ac.cn/archives/7758>)》的改进方向是正确的。

### 中文

相信看本博客的读者多数都只关心中文任务，部分读者可能也会关心英文任务，应该鲜有读者会关心中英文以外的任务了。然而，mT5涵盖了101种语言，总词表有25万，而且它采用的T5\.1\.1结构的Softmax还不共享参数，这就导致了Embedding层占用了相当多的参数量，比如mT5 small的参数量为3亿，其中Embedding相关的就占了2\.5亿，关键是里边的大部分参数我们都用不上，纯粹是不必要的浪费。因此，对于主要关心中文任务的我们来说，有必要精简一下这个Embedding层了。

对模型的精简很简单，只需要在两个Embedding矩阵中删除不需要的行就行了，关键在于如何决定要保留的token，以及如何得到一个精简后的sentencepiece模型。决定要保留的token，简单来想就是把中文的token保留下来，但是也不只是中文，英文的也要保留一部分，看上去似乎只是一个正则表达式的问题，实际上没那么简单，用英文字母的也不一定是英语，用中文字的也不一定是中文，这是个让人纠结的事情。于是笔者想了另外一个办法：用这个25万token的tokenizer对笔者收集的几十G中文语料分词，统计分词结果，然后按照词频选择前面的部分（最后保留了3万多个token）。这样虽然费时一些，但是比较靠谱，能确保把我们比较需要的token保留下来。决定词表后，就要修改得到一个新的sentencepiece模型，这也有点麻烦，但最终经过搜索后还是把这个事情解决了，处理方法都分享在Github上。

经过这样处理后，要构建新的模型，则只需要多加三行代码<code>keep_tokens</code>相关的代码，所需要的显存就大大降低，并且中文生成的效果基本不变了：



<pre><code class="language-python"># 模型路径
config_path = '/root/kg/bert/mt5/mt5_base/t5_config.json'
checkpoint_path = '/root/kg/bert/mt5/mt5_base/model.ckpt-1000000'
spm_path = '/root/kg/bert/mt5/sentencepiece_cn.model'
keep_tokens_path = '/root/kg/bert/mt5/sentencepiece_cn_keep_tokens.json'

# 加载分词器
tokenizer = SpTokenizer(spm_path, token_start=None, token_end='&lt;/s&gt;')
keep_tokens = json.load(open(keep_tokens_path))

# 加载模型
t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model</code></pre>



### 效果

最后，大家应该是关心折腾了这么久，生成效果究竟有没有提升，有没有使用的价值？这样说吧，用mT5 small版本finetune出来的CSL标题生成模型，BLEU指标能持平基于WoBERT的UniLM模型，并且解码速度快130%；而用mT5 base版本finetune出来的CSL标题生成模型，指标能超过基于WoBERT的UniLM模型1%以上，并且解码速度也能快60%。

\begin{array}{c}
\text{CSL摘要生成实验结果 (beam size=1)}\\
{\begin{array}{c|cccc|c}
\hline
& \text{Rouge-L} & \text{Rouge-1} & \text{Rouge-2} & \text{BLEU} & \text{解码速度}\\
\hline
\text{BERT base} & 63.81 & 65.45 & 54.91 & 45.52 & \text{1x}\\
\text{WoBERT base} & 66.38 & 68.22 & 57.83 & 47.76 & \text{1.1x}\\
\hline
\text{mT5 small} & 65.14 & 67.08 & 56.71 & 47.69 & \text{2.3x}\\
\text{mT5 base} & \textbf{66.81} & \textbf{68.94} & \textbf{58.49} & \textbf{49.49} & \text{1.6x}\\
\hline
\end{array}}
\end{array}

说白了，确实是又快又好。至于设备要求，平时跑过BERT base的同学，基本都应该能跑起mT5 small/base版，甚至large版也可以尝试一下，至于XL和XXL，那就比较难搞了，建议还是放弃吧。更多的惊喜，还是大家自己去挖掘吧～～对了，顺便需要提醒一下，微调T5模型的时候，学习率要比微调BERT大10倍以上才行（即$10^{-4}$级别，BERT一般是$10^{-5}$级别），这是两者模型架构差异决定的。

## 小结

本文回顾了一下Google去年发布的T5模型，然后介绍了最近发布的多国语言版的mT5，最后介绍了如何在bert4keras中微调mT5来做中文任务，结果显示mT5在中文生成上有着很不错的表现，值得做文本生成任务的同学一试。
