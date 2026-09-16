# 基于Conditional Layer Normalization的条件文本生成

> 作者：苏剑林 · 科学空间 · 2019-12-14
>
> 原文：<https://spaces.ac.cn/archives/7124>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从文章[《从语言模型到Seq2Seq：Transformer如戏，全靠Mask》](<https://spaces.ac.cn/archives/6933>)中我们可以知道，只要配合适当的Attention Mask，Bert（或者其他Transformer模型）就可以用来做无条件生成（Language Model）和序列翻译（Seq2Seq）任务。

可如果是有条件生成呢？比如控制文本的类别，按类别随机生成文本，也就是Conditional Language Model；又比如传入一副图像，来生成一段相关的文本描述，也就是Image Caption。

## 相关工作

八月份的论文[《Encoder\-Agnostic Adaptation for Conditional Language Generation》](<https://papers.cool/arxiv/1908.06938>)比较系统地分析了利用预训练模型做条件生成的几种方案；九月份有一篇论文[《CTRL: A Conditional Transformer Language Model for Controllable Generation》](<https://papers.cool/arxiv/1909.05858>)提供了一个基于条件生成来预训练的模型，不过这本质还是跟GPT一样的语言模型，只能以文字输入为条件；而最近的论文[《Plug and Play Language Models: a Simple Approach to Controlled Text Generation》](<https://papers.cool/arxiv/1912.02164>)将$p(x|y)$转化为$p(x)p(y|x)$来探究基于预训练模型的条件生成。

不过这些经典工作都不是本文要介绍的。本文关注的是以一个固定长度的向量作为条件的文本生成的场景，而方法是<strong>Conditional Layer Normalization</strong>——把条件融合到Layer Normalization的$\beta$和$\gamma$中去。

## 思路细节

Conditional Layer Normalization的想法来源于图像中流行的条件GAN的思路——条件BN（Conditional Batch Normalization），相关内容可以参考[《从DCGAN到SELF\-MOD：GAN的模型架构发展一览》](<https://spaces.ac.cn/archives/6549>)。条件BN还有一个变种，称之为AdaIN（Adaptive Instance Normalization）。条件BN、AdaIN都是将已有的Normalization方法中的$\beta$和$\gamma$变成输入条件的函数，从而可以通过条件来控制生成的行为。

在Bert等Transformer模型中，主要的Normalization方法是Layer Normalization，所以很自然就能想到将对应的$\beta$和$\gamma$变成输入条件的函数，来控制Transformer模型的生成行为，这就是Conditional Layer Normalization的线索思路。（但目前还没有看到同样思路的工作出现，所以这算是笔者闭门造车出来的新鲜玩意了。）

[![条件Normalization示意图](<https://spaces.ac.cn/usr/uploads/2019/12/2102002684.png>)](<https://spaces.ac.cn/usr/uploads/2019/12/2102002684.png>)

条件Normalization示意图

对于已经预训练好的模型来说，已经有现成的、无条件的$\beta$和$\gamma$了，它们都是长度固定的向量。我们可以通过两个不同的变换矩阵，将输入条件变换到跟$\beta,\gamma$一样的维度，然后将两个变换结果分别加到$\beta$和$\gamma$上去。为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。

## 代码实现

直觉上，这种以文本生成为目的的finetune应该要用GPT等自回归预训练模型才能提升效果，但事实上，之前的文章[《从语言模型到Seq2Seq：Transformer如戏，全靠Mask》](<https://spaces.ac.cn/archives/6933>)已经表明，哪怕你加载Bert的预训练权重来做生成任务，表现依然良好。所以不管哪种Transformer\-based的预训练模型，都可以考虑用来finetune做文本生成模型来。而本文还是以预训练Bert为基础模型进行实验。

至于代码，本文所描述的Conditional Layer Normalization技巧，也已经被集成到笔者所开发的[bert4keras](<https://github.com/bojone/bert4keras>)中了，现在基础函数<code>build_transformer_model</code>新增了如下参数：

> 1、layer\_norm\_cond：如果该参数非None，则意味着它是一个张量，shape=&#91;batch\_size, cond\_size&#93;，用来作为Layer Normalization的条件；
> 
> 
> 
> 2、layer\_norm\_cond\_size：如果该参数非None且layer\_norm\_cond为None，则意味着它是一个整数，自行构建一个shape=&#91;batch\_size, layer\_norm\_cond\_size&#93;的输入层作为Layer Normalization的条件；
> 
> 
> 
> 3、layer\_norm\_cond\_hidden\_size：如果该参数非None，则意味着它是一个整数，用于先将输入条件投影到更低维空间，这是因为输入的条件可能维度很高，直接投影到hidden\_size（比如768）的话，参数可能过多，所以可以先投影到更低维空间，然后升维；
> 
> 
> 
> 4、layer\_norm\_cond\_hidden\_act：投影到更低维空间时的激活函数，如果为None，则不加激活函数（线性激活）；
> 
> 
> 
> 5、additional\_input\_layers：额外的输入层，如果外部传入了张量作为条件，则需要把条件张量所依赖的所有输入层都添加进来，作为输入层，才能构建最终的模型。

## 实验效果

介绍再多，其实还不如看例子来得实际。笔者做了两个实验来验证Conditional Layer Normalization的效果。一个是通过情感极性来控制文本生成，也就是情感分类的反问题，这直接通过类的Embedding来作为Layer Normalization的条件；另一个是图像描述生成（Image Caption），通过预训练的imagenet模型将图片编码为一个固定长度的向量作为Layer Normalization的条件。

这两个代码分别放在[task\_conditional\_language\_model\.py](<https://github.com/bojone/bert4keras/blob/master/examples/task_conditional_language_model.py>)和[task\_image\_caption\.py](<https://github.com/bojone/bert4keras/blob/master/examples/task_image_caption.py>)中。

### 情感文本生成

情感文本生成就是用的训练集是笔者之前收集整理的[情感分类语料](<https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip>)，将输入文本和标签反过来用即可。最后生成的时候按概率随机采样，从而能生成不同的文本。

部分输出：

> <strong>正面采样:</strong>
> &#91;u'外观时尚、漂亮、性价比高。', u'外观漂亮，配置均衡，比较满意，性价比高，外观漂亮，性能较高。', u'我是在大学的时候看到这本书的，所以一直在买。书中的作者是林静蕾，她用自己的口吻写出了一个孩子成长中的心路历程，让我看到了她们成长中的不同之处，以及她们成长过程中的不同境界。让我很欣赏！', u'我想这是一本能够告诉读者什么是坏的，而不是教你怎样说话，告诉我什么是错。这里我推荐了《我要讲故事》，这本书是我很喜欢的一本书，我认为它的理由很多，但是，我相信我。如果你从中得到一些改进，或者你已经有了一个明智的决定。', u'我们一家五口住的是标间，大床房，大床的床很舒服；而我们在携程网上订了两套大床房，这个酒店的价格还是比较合理的；但是房间的隔音效果不太理想，有点响的声音；酒店门口的地铁在施工中，不方便；但是酒店的门口的出租车不知道是哪个车的，打车不是很方便；酒店外面的停'&#93;
> 
> 
> 
> <strong>负面采样:</strong>
> &#91;u'不知道是不是因为电池不太好，不是我不喜欢。', u'看了评论才买的\. 结果发现不是那么便宜, 价格也不便宜\.', u'1、外壳不容易沾手印，不容易洗洗2、屏幕有点旧，不能下载铃声', u'我是7月6日订购了《杜拉拉升职记》并已通过银行付款，为什么订单下了两周多至今还未到货？是收货时间太快了，可能就这么过去了吧？', u'这本书我是在网上先看了一遍，后来我再看了一遍。感觉作者的文笔实在太烂了，特别是在写他的博客时特别别扭，写得很不专业，特别是他写股票时那个情绪调节的小男孩，简直就是自作聪明的样子，简直就是自作聪明的一种表现！'&#93;

### Image Caption

Image Caption以[COCO数据集](<http://cocodataset.org/#download>)为例，这个数据集的图片场景比较丰富一些。另外2017年的challenger\.ai也举办过一个[图像中文描述生成竞赛](<https://challenger.ai/dataset/caption?lan=zh>)，里边也包含了一个不错的数据集（读者自己自行想办法收集），不过图片的场景相对来说单调一些。

部分输出：

[![模型预测: a baseball game in progress with the batter up to plate.](<https://spaces.ac.cn/usr/uploads/2019/12/3163471200.jpg>)](<https://spaces.ac.cn/usr/uploads/2019/12/3163471200.jpg>)

模型预测: a baseball game in progress with the batter up to plate\.

[![模型预测: a train that is sitting on the tracks.](<https://spaces.ac.cn/usr/uploads/2019/12/3673525199.jpg>)](<https://spaces.ac.cn/usr/uploads/2019/12/3673525199.jpg>)

模型预测: a train that is sitting on the tracks\.

> <strong>image\_id:</strong> COCO\_val2014\_000000524611\.jpg
> <strong>url:</strong> [http://images\.cocodataset\.org/val2014/COCO\_val2014\_000000524611\.jpg](<http://images.cocodataset.org/val2014/COCO_val2014_000000524611.jpg>)
> <strong>predict:</strong> a train that is sitting on the tracks\.
> <strong>references:</strong> &#91;u'A train carrying chemical tanks traveling past a water tower\.', u'Dual train tracks with a train on one of them and a water tower in the background\.', u'a train some trees and a water tower ', u'Train on tracks with water tower for Davis Junction in the rear\.', u'A train on a train track going through a bunch of trees\.'&#93;
> 
> 
> 
> <strong>image\_id:</strong> COCO\_val2014\_000000202923\.jpg
> <strong>url:</strong> [http://images\.cocodataset\.org/val2014/COCO\_val2014\_000000202923\.jpg](<http://images.cocodataset.org/val2014/COCO_val2014_000000202923.jpg>)
> <strong>predict:</strong> a baseball game in progress with the batter up to plate\.
> <strong>references:</strong> &#91;u'Batter, catcher, and umpire anticipating the next pitch\.', u'A baseball player holding a baseball bat in the game\.', u'A baseball player stands ready at the plate\.', u'Baseball players on the field ready for the pitch\.', u'A view from behind a mesh fence of a baseball game\.'&#93;

## 文章小结

提出了利用Conditional Layer Normalization来将外部条件融入到预训练模型中的思路，其直接应用就是条件文本生成，但其实也不单单可以用于生成模型，也可以用于分类模型等场景（外部条件可能是其他模态的信息，来辅助分类）。最后基于bert4keras给出了代码实现以及两个例子。
