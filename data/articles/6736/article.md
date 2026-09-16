# 当Bert遇上Keras：这可能是Bert最简单的打开姿势

> 作者：苏剑林 · 科学空间 · 2019-06-18
>
> 原文：<https://spaces.ac.cn/archives/6736>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[Bert](<https://papers.cool/arxiv/1810.04805>)是什么，估计也不用笔者来诸多介绍了。虽然笔者不是很喜欢Bert，但不得不说，Bert确实在NLP界引起了一阵轩然大波。现在不管是中文还是英文，关于Bert的科普和解读已经满天飞了，隐隐已经超过了当年Word2Vec刚出来的势头了。有意思的是，<u>Bert是Google搞出来的，当年的word2vec也是Google搞出来的</u>，不管你用哪个，都是在跟着Google大佬的屁股跑啊～

Bert刚出来不久，就有读者建议我写个解读，但我终究还是没有写。一来，Bert的解读已经不少了，二来其实Bert也就是基于Attention的搞出来的大规模语料预训练的模型，本身在技术上不算什么创新，而关于[Google的Attention](<https://spaces.ac.cn/archives/4765>)我已经写过解读了，所以就提不起劲来写了。

[![Bert的预训练和微调（图片来自Bert的原论文）](<https://spaces.ac.cn/usr/uploads/2019/06/3616885925.png>)](<https://spaces.ac.cn/usr/uploads/2019/06/3616885925.png>)

Bert的预训练和微调（图片来自Bert的原论文）

总的来说，我个人对Bert一直也没啥兴趣，直到上个月末在做信息抽取比赛时，才首次尝试了Bert。因为后来想到，即使不感兴趣，终究也是得学会它，毕竟用不用是一回事，会不会又是另一回事。再加上在Keras中使用（fine tune）Bert，似乎还没有什么文章介绍，所以就分享一下自己的使用经验。

## 当Bert遇上Keras

很幸运的是，已经有大佬封装好了Keras版的Bert，可以直接调用官方发布的预训练权重，对于已经有一定Keras基础的读者来说，这可能是最简单的调用Bert的方式了。所谓“站在巨人的肩膀上”，就是形容我们这些Keras爱好者此刻的心情了。

### keras\-bert

个人认为，目前在Keras下对Bert最好的封装是：

> <strong>keras\-bert：</strong>[https://github\.com/CyberZHG/keras\-bert](<https://github.com/CyberZHG/keras-bert>)

本文也是以此为基础的。

顺便一提的是，除了keras\-bert之外，CyberZHG大佬还封装了很多有价值的keras模块，比如keras\-gpt\-2（你可以用像用bert一样用gpt2模型了）、keras\-lr\-multiplier（分层设置学习率）、keras\-ordered\-neurons（就是前不久介绍的[ON\-LSTM](<https://spaces.ac.cn/archives/6621>)）等等，汇总可以看[这里](<https://github.com/CyberZHG/summary>)。看来也是一位Keras铁杆粉丝啊～致敬大佬。

事实上，有了keras\-bert之后，再加上一点点keras基础知识，而且keras\-bert所给的demo已经足够完善，调用、微调Bert都已经变成了意见没有什么技术含量的事情了。所以后面笔者只是给出几个中文的例子，来让读者上手keras\-bert的基本用法。

### Tokenizer

正式讲例子之前，还有必要先讲一下Tokenizer相关内容。我们导入Bert的Tokenizer并重构一下它：



<pre><code class="language-python">from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs


config_path = '../bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../bert/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)
tokenizer.tokenize(u'今天天气不错')
# 输出是 ['[CLS]', u'今', u'天', u'天', u'气', u'不', u'错', '[SEP]']
</code></pre>



这里简单解释一下<code>Tokenizer</code>的输出结果。首先，默认情况下，分词后句子首位会分别加上&#91;CLS&#93;和&#91;SEP&#93;标记，其中&#91;CLS&#93;位置对应的输出向量是能代表整句的句向量（反正Bert是这样设计的），而&#91;SEP&#93;则是句间的分隔符，其余部分则是单字输出（对于中文来说）。

本来<code>Tokenizer</code>有自己的<code>_tokenize</code>方法，我这里重写了这个方法，是要<u>保证tokenize之后的结果，跟原来的字符串长度等长（如果算上两个标记，那么就是等长再加2）</u>。<code>Tokenizer</code>自带的<code>_tokenize</code>会自动去掉空格，然后有些字符会粘在一块输出，导致tokenize之后的列表不等于原来字符串的长度了，这样如果做序列标注的任务会很麻烦。而为了避免这种麻烦，还是自己重写一遍好了～主要就是用&#91;unused1&#93;来表示空格类字符，而其余的不在列表的字符用&#91;UNK&#93;表示，其中&#91;unused\*&#93;这些标记是未经训练的（随即初始化），是Bert预留出来用来增量添加词汇的标记，所以我们可以用它们来指代任何新字符。

## 三个例子

这里包含keras\-bert的三个例子，分别是<strong>文本分类</strong>、<strong>关系抽取</strong>和<strong>主体抽取</strong>，都是在官方发布的预训练权重基础上进行微调来做的。

> <strong>Bert官方Github：</strong>[https://github\.com/google\-research/bert](<https://github.com/google-research/bert>)
> 
> 
> 
> <strong>官方的中文预训练权重：</strong>[chinese\_L\-12\_H\-768\_A\-12\.zip](<https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip>)
> 
> 
> 
> <strong>例子所在Github：</strong>[https://github\.com/bojone/bert\_in\_keras/](<https://github.com/bojone/bert_in_keras/>)

根据官方介绍，这份权重是用中文维基百科为语料进行训练的。

（2019年6月20日更新：哈工大讯飞联合实验室发布了一版新权重，也可以用keras\_bert加载，详情请看[这里](<https://github.com/ymcui/Chinese-BERT-wwm>)。）

### 文本分类

作为第一个例子，我们做一个最基本的文本分类任务，熟悉做这个基本任务之后，剩下的各种任务都会变得相当简单了。这次我们以之前已经讨论过多次的[文本感情分类任务](<https://spaces.ac.cn/archives/3863>)为例，所用的[标注数据](<https://spaces.ac.cn/archives/3414>)也是以前所整理的。

让我们来看看模型部分全貌（完整代码见[这里](<https://github.com/bojone/bert_in_keras/blob/master/sentiment.py>)）：



<pre><code class="language-python"># 注意，尽管可以设置seq_len=None，但是仍要保证序列长度不超过512
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()</code></pre>



在Keras中调用Bert来做情感分类任务就这样写完了～写完了～～

是不是感觉还没有尽兴，模型代码就结束了？Keras调用Bert就这么简短。事实上，真正调用Bert的也就只有<code>load_trained_model_from_checkpoint</code>那一行代码，剩下的只是普通的Keras操作（再次感谢CyberZHG大佬）。所以，如果你已经入门了Keras，那么调用Bert是无往不利啊。

如此简单的调用，能达到什么精度？经过5个epoch的fine tune后，验证集的最好准确率是95\.5%\+！之前我们在[《文本情感分类（三）：分词 OR 不分词》](<https://spaces.ac.cn/archives/3863>)中死调烂调，也就只有90%上下的准确率；而用了Bert之后，寥寥几行，就提升了5个百分点多的准确率！也难怪Bert能在NLP界掀起一阵热潮\.\.\.

> 在这里，用笔者的个人经历先回答读者可能关心的两个问题。
> 
> 
> 
> 第一个问题应该是大家都很关心的，那就是“要多少显存才够？”。事实上，这没有一个标准答案，显存的使用取决于三个因素：句子长度、batch size、模型复杂度。像上面的情感分析例子，在笔者的GTX1060 6G显存上也能跑起来，只需要将batch size调到24即可。所以，如果你的显存不够大，将句子的maxlen和batch size都调小一点试试。当然，如果你的任务太复杂，再小的maxlen和batch size也可能OOM，那就只有升级显卡了。
> 
> 
> 
> 第二个问题是“有什么原则来指导Bert后面应该要接哪些层？”。答案是：用尽可能少的层来完成你的任务。比如上述情感分析只是一个二分类任务，你就取出第一个向量然后加个Dense(1)就好了，不要想着多加几层Dense，更加不要想着接个LSTM再接Dense；如果你要做序列标注（比如NER），那你就接个Dense\+CRF就好，也不要多加其他东西。总之，额外加的东西尽可能少。一是因为Bert本身就足够复杂，它有足够能力应对你要做的很多任务；二来你自己加的层都是随机初始化的，加太多会对Bert的预训练权重造成剧烈扰动，容易降低效果甚至造成模型不收敛～

### 关系抽取

假如读者已经有了一定的Keras基础，那么经过第一个例子的学习，其实我们应该已经完全掌握了Bert的fine tune了，因为实在是简单到没有什么好讲了。所以，后面两个例子主要是提供一些参考模式，让读者能体会到如何“用尽可能少的层来完成你的任务”。

在第二个例子中，我们介绍基于Bert实现的一个极简的关系抽取模型，其标注原理跟[《基于DGCNN和概率图的轻量级信息抽取模型》](<https://spaces.ac.cn/archives/6671>)介绍的一样，但是得益于Bert强大的编码能力，我们所写的部分可以大大简化。在笔者所给出的一种参考实现中，模型部分如下（完整模型见[这里](<https://github.com/bojone/bert_in_keras/blob/master/relation_extract.py>)）：



<pre><code class="language-python">t = bert_model([t1, t2])
ps1 = Dense(1, activation='sigmoid')(t)
ps2 = Dense(1, activation='sigmoid')(t)

subject_model = Model([t1_in, t2_in], [ps1, ps2]) # 预测subject的模型


k1v = Lambda(seq_gather)([t, k1])
k2v = Lambda(seq_gather)([t, k2])
kv = Average()([k1v, k2v])
t = Add()([t, kv])
po1 = Dense(num_classes, activation='sigmoid')(t)
po2 = Dense(num_classes, activation='sigmoid')(t)

object_model = Model([t1_in, t2_in, k1_in, k2_in], [po1, po2]) # 输入text和subject，预测object及其关系


train_model = Model([t1_in, t2_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in],
                    [ps1, ps2, po1, po2])</code></pre>



如果读者已经读过[《基于DGCNN和概率图的轻量级信息抽取模型》](<https://spaces.ac.cn/archives/6671>)一文，了解到不用Bert时的模型架构，那么就会理解到上述实现是多么的简介明了。

可以看到，我们引入了Bert作为编码器，然后得到了编码序列$t$，然后直接接两个Dense(1)，这就完成了subject的标注模型；接着，我们把传入的s的首尾对应的编码向量拿出来，直接加到编码向量序列$t$中去，然后再接两个Dense(num\_classes)，就完成object的标注模型（同时标注出了关系）。

这样简单的设计，最终F1能到多少？答案是：线下dev能接近82%，线上我提交过一次，结果是85%\+（都是单模型）！相比之下，[《基于DGCNN和概率图的轻量级信息抽取模型》](<https://spaces.ac.cn/archives/6671>)中的模型，需要接CNN，需要搞全局特征，需要将s传入到LSTM进行编码，还需要相对位置向量，各种拍脑袋的模块融合在一起，单模型也只比它好一点点（大约82\.5%）。要知道，这个基于Bert的简单模型我只写了一个小时就写出来了，而各种技巧和模型融合在一起的DGCNN模型，我前前后后调试了差不多两个月！Bert的强悍之处可见一斑。

（注：这个模型的fine tune最好有8G以上的显存。另外，因为我在比赛即将结束的前几天才接触的Bert，才把这个基于Bert的模型写出来，没有花心思好好调试，所以最终的提交结果并没有包含Bert。）

> 用Bert做关系抽取的这个例子，跟前面情感分析的简单例子，有一个明显的差别是学习率的变化。
> 
> 
> 
> 情感分析的例子中，只是用了恒定的学习率（$10^{-5}$）训练了几个epoch，效果就还不错了。在关系抽取这个例子中，第一个epoch的学习率慢慢从$0$增加到$5\times 10^{-5}$（这样称为warmup），第二个epoch再从$5\times 10^{-5}$降到$10^{-5}$，总的来说就是先增后减，Bert本身也是用类似的学习率曲线来训练的，这样的训练方式比较稳定，不容易崩溃，而且效果也比较好。

### 事件主体抽取

最后一个例子来自[CCKS 2019 面向金融领域的事件主体抽取](<https://biendata.com/competition/ccks_2019_4/>)，这个比赛目前还在进行，不过我也已经没有什么动力和兴趣做下去了，所以放出我现在的模型（准确率为89%\+）供大家参考，祝继续参赛的选手取得更好的成绩。

简单介绍一下这个比赛的数据，大概是这样的

> 输入：“公司A产品出现添加剂，其下属子公司B和公司C遭到了调查”， “产品出现问题”
> 
> 
> 
> 输出： “公司A”

也就是说，这是个双输入、单输出的模型，输入是一个query和一个事件类型，输出一个实体（有且只有一个，并且是query的一个片段）。其实这个任务可以看成是[SQUAD 1\.0](<https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/>)的简化版，根据这个输出特性，输出应该用指针结构比较好（两个softmax分别预测首尾）。剩下的问题是：双输入怎么搞？

前面两个例子虽然复杂度不同，但它们都是单一输入的，双输入怎么办呢？当然，这里的实体类型只有有限个，直接Embedding也行，只不过我使用一种更能体现Bert的简单粗暴和强悍的方案：直接用连接符将两个输入连接成一个句子，然后就变成单输入了！比如上述示例样本处理成：

> 输入：“\_\_\_产品出现问题\_\_\_公司A产品出现添加剂，其下属子公司B和公司C遭到了调查”
> 
> 
> 
> 输出： “公司A”

然后就变成了普通的单输入抽取问题了。说到这个，这个模型的代码也就没有什么好说的了，就简单几行（完整代码请看[这里](<https://github.com/bojone/bert_in_keras/blob/master/subject_extract.py>)）：



<pre><code class="language-python">x = bert_model([x1, x2])
ps1 = Dense(1, use_bias=False)(x)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
ps2 = Dense(1, use_bias=False)(x)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x1_in, x2_in], [ps1, ps2])</code></pre>



另外加上一些解码的trick，还有模型融合，提交上去，就可以做到89%\+了。在看看目前排行榜，发现最好的结果也就是90%多一点点，所以估计大家都差不多是这样做的了\.\.\.（这个代码重复实验时波动比较大，大家可以多跑几次，取最优结果。）

> 这个例子主要告诉我们，用Bert实现自己的任务时，最好能整理成单输入的模式，这样一来比较简单，二来也更加高效。
> 
> 
> 
> 比如做句子相似度模型，输入两个句子，输出一个相似度，有两个可以想到的做法，第一种是两个句子分别过同一个Bert，然后取出各自的&#91;CLS&#93;特征来做分类；第二种就是像上面一样，用个记号把两个句子连接在一起，变成一个句子，然后过一个Bert，然后将输出特征做分类，后者显然会更快一些，而且能够做到特征之间更全面的交互。

## 文章小结

本文介绍了Keras下Bert的基本调用方法，其中主要是提供三个参考例子，供大家逐步熟悉Bert的fine tune步骤和原理。其中有不少是笔者自己闭门造车的经验之谈，如果有所偏颇，还望读者指正。

事实上有了CyberZHG大佬实现的keras\-bert，在Keras下使用Bert也就是小菜一碟，大家折腾个半天，也就上手了。最后祝大家用得痛快～
