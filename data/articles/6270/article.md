# 基于CNN和序列标注的对联机器人

> 作者：苏剑林 · 科学空间 · 2019-01-14
>
> 原文：<https://spaces.ac.cn/archives/6270>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## 缘起

前几天在量子位公众号上看到了[《这个脑洞清奇的对联AI，大家都玩疯了》](<https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247511349&idx=1&sn=8e5968c0d58c70bab29ab11216b2ac4f&chksm=e8d01a47dfa7935158b0a195c3ff70fa1ebcfac6a62d6c7d5f0db5c6290e4375b87cae14f80e&mpshare=1&scene=23&srcid=0114ebz0Q6v4SqrUdwyEUD1Q#rd>)一文，觉得挺有意思，难得的是作者还整理并公开了数据集，所以决定自己尝试一下。

## 动手

“对对联”，我们可以看成是一个句子生成任务，可以用seq2seq完成，跟笔者之前写的[《玩转Keras之seq2seq自动生成标题》](<https://spaces.ac.cn/archives/5861>)一样，稍微修改一下输入即可。上面提到的文章所用的方法也是seq2seq，可见这算是标准做法了。

### 分析

然而，我们再细想一下就会发现，相对于一般的句子生成任务，“对对联”有规律得多：1、上联和下联的字数一样；2、上联和下联的每一个字几乎都有对应关系。如此一来，其实对对联可以直接看成一个序列标注任务，跟分词、命名实体识别等一样的做法即可。这便是本文的出发点。

说到这，其实本文就没有什么技术含量了，序列标注已经是再普通不过的任务了，远比一般的seq2seq来得简单。所谓序列标注，就是指输入一个向量序列，然后输出另外一个通常长度的向量序列，最后对这个序列的“每一帧”进行分类。相关概念来可以在[《简明条件随机场CRF介绍（附带纯Keras实现）》](<https://spaces.ac.cn/archives/5542>)一文进一步了解。

### 模型

本文直接边写代码边介绍模型。如果需要进一步了解背后的基础知识的读者，还可以参考[《【中文分词系列】 4\. 基于双向LSTM的seq2seq字标注》](<https://spaces.ac.cn/archives/3924>)、[《【中文分词系列】 6\. 基于全卷积网络的中文分词》](<https://spaces.ac.cn/archives/4195>)、[《基于CNN和VAE的作诗机器人：随机成诗》](<https://spaces.ac.cn/archives/5332>)。

我们所用的模型代码如下：



<pre><code class="language-python">x_in = Input(shape=(None,))
x = x_in
x = Embedding(len(chars)+1, char_size)(x)
x = Dropout(0.25)(x)

x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)
x = gated_resnet(x)

x = Dense(len(chars)+1, activation='softmax')(x)

model = Model(x_in, x)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam')</code></pre>



其中<code>gated_resnet</code>是笔者定义的门卷积模块（在[《基于CNN的阅读理解式问答模型：DGCNN》](<https://spaces.ac.cn/archives/5409>)一文也介绍过这个模块）：



<pre><code class="language-python">def gated_resnet(x, ksize=3):
    # 门卷积 + 残差
    x_dim = K.int_shape(x)[-1]
    xo = Conv1D(x_dim*2, ksize, padding='same')(x)
    return Lambda(lambda x: x[0] * K.sigmoid(x[1][..., :x_dim]) \
                            + x[1][..., x_dim:] * K.sigmoid(-x[1][..., :x_dim]))([x, xo])</code></pre>



仅此而已～

就这样完了，剩下的都是数据预处理的事情了。当然，读者也可以尝试也可以把<code>gated_resnet</code>换成普通的双向LSTM，但我实验中发现双向LSTM并没有<code>gated_resnet</code>效果好，而且LSTM相对来说也更慢，所以LSTM在这里就被抛弃了。

### 效果

训练的数据集来自：[https://github\.com/wb14123/couplet\-dataset](<https://github.com/wb14123/couplet-dataset>)，感谢作者的整理。

<strong>完整代码：</strong>
[https://github\.com/bojone/seq2seq/blob/master/couplet\_by\_seq\_tagging\.py](<https://github.com/bojone/seq2seq/blob/master/couplet_by_seq_tagging.py>)

<strong>训练过程：</strong>

[![对联机器人训练过程](<https://spaces.ac.cn/usr/uploads/2019/01/144956437.png>)](<https://spaces.ac.cn/usr/uploads/2019/01/144956437.png>)

对联机器人训练过程

<strong>部分效果：</strong>

> 上联：晚风摇树树还挺，下联：夜雨敲花花更香
> 
> 
> 
> 上联：今天天气不错，下联：昨日人情无明
> 
> 
> 
> 上联：鱼跃此时海，下联：鸟鸣何日人
> 
> 
> 
> 上联：只有香如故，下联：不无月若新
> 
> 
> 
> 上联：科学空间，下联：文明大中

看起来还是有点味道的。注意“晚风摇树树还挺”是训练集的上联，标准下联是“晨露润花花更红”，而模型给出来的是“夜雨敲花花更香”，说明模型并不是单纯地记住训练集的，还是有一定的理解能力；甚至我觉得模型对出来的下联更生动一些。

总的来说，基本的字的对应似乎都能做到，就缺乏一个整体感。总体效果没有下面两个好，但作为一个小玩具，应该能让人满意了。

> 王斌版AI对联：[https://ai\.binwang\.me/couplet/](<https://ai.binwang.me/couplet/>)
> 
> 
> 
> 微软对联：[https://duilian\.msra\.cn/default\.htm](<https://duilian.msra.cn/default.htm>)

## 结语

最后，也没有什么好总结的。我就是觉得这个对对联应该算是一个序列标注任务，所以就想着用一个序列标注的模型来试试看，结果感觉还行～当然，要做得更好，需要在模型上做些调整，还可以考虑引入Attention等，然后解码的时候，还需要引入更多的先验知识，保证结果符合我们对对联的要求。这些就留给有兴趣做下去的读者继续了。
