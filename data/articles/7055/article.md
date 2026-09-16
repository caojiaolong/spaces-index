# Keras：Tensorflow的黄金标准

> 作者：苏剑林 · 科学空间 · 2019-11-06
>
> 原文：<https://spaces.ac.cn/archives/7055>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这两周投入了比较多的精力去做[bert4keras](<https://github.com/bojone/bert4keras>)的开发，除了一些API的规范化工作外，其余的主要工作量是构建预训练部分的代码。在昨天，预训练代码基本构建完毕，并同时在TPU/多GPU环境下测试通过，从而有志（有算力）改进预训练模型的同学多了一个选择。——这可能是目前最为清晰易懂的bert及其预训练代码。

<strong>预训练代码链接： [https://github\.com/bojone/bert4keras/tree/master/pretraining](<https://github.com/bojone/bert4keras/tree/master/pretraining>)</strong>

经过这两周的开发（填坑），笔者的最大感想就是：Keras已经成为了tensorflow的黄金标准了。只要你的代码按照Keras的标准规范写，那可以轻松迁移到tf\.keras中去，继而可以非常轻松地在TPU或多GPU环境下训练，真正的几乎是一劳永逸。相反，如果你的写法过于灵活，包括像笔者之前介绍的很多“移花接木”式的Keras技巧，就可能会有不少问题，甚至可能出现的一种情况是：就算你已经在多GPU上跑通了，在TPU上你也死活调不通。

[![Keras和Tensorflow](<https://spaces.ac.cn/usr/uploads/2019/11/4154078254.png>)](<https://spaces.ac.cn/usr/uploads/2019/11/4154078254.png>)

Keras和Tensorflow

## 不遗余力的支持

大家都说tensorflow 2\.0主推tf\.keras，但事实上，从tensorflow 1\.14开始，tf\.keras就已经成为了它的黄金标准了，所以如果大家要体验Google爸爸是如何<u>不遗余力</u>支持keras的，只需要tensorflow 1\.14\+就行了，不一定要升级到2\.0。目前bert4keras的代码同时支持原版keras以及tf\.keras，在通常的单卡finetune任务里，大家可以用keras或者tf\.keras都行，但如果要用多GPU训练甚至用TPU训练，那最好的选择还是tf\.keras。

要入门tf\.keras，首先建议参考一个很不错的网站：[https://tf\.wiki/](<https://tf.wiki/>)

在tf\.keras中，将一个模型从单卡变成多卡训练，代码非常简单：



<pre><code class="language-python">strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_a_model()
    model.compile(loss='mse', optimizer='adam')

model.fit(train_x, train_y, epochs=10)
</code></pre>



也就是说，只要定义一个<code>strategy</code>，然后在这个<code>strategy</code>的<code>scope</code>下建立的模型，就是多卡的模型了。多卡训练，从未如此简单～

顺便提一下，Keras本身自带了<code>multi_gpu_model</code>函数来实现多GPU训练，但笔者亲身测试<code>multi_gpu_model</code>并不好用，有时候还无法生效。总之还是推荐用tf\.keras。另外，上面是单机多卡的写法，多机多卡类似，但我没有相应的环境测试，所以就没有给出例子，要测试的朋友，请参考[https://tf\.wiki/](<https://tf.wiki/>)里边的介绍。

那TPU呢？一样简单，把<code>strategy</code>替换一下就行了（tensorflow 2\.0有小改动）：



<pre><code class="language-python">resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
tf.config.experimental_connect_to_host(resolver.master())
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)
</code></pre>



有没有感觉到简单得不可思议？这时候大家总该明白我前面说的“不遗余力”支持keras的含义了吧。从tensorflow 1\.14开始，只要你用标准的keras写法，那么几乎可以无往而不利。

## 怎样才算是标准？

前面不断强调用标准的keras写法，那怎样才算是标准呢？这里汇总一些经验。

> 1、尽可能全用keras自带的层、loss函数和优化器实现我们所需要的功能，如果一个模型全都是用keras内置的层、loss函数和优化器，那么基本上可以保证在多GPU或TPU上都能跑通了。
> 
> 
> 
> 2、如果要自定义层，要严格按照规范来，尤其是要把get\_config方法写好。测试自己写得规范与否的一个方法是：用你自定义的层去构建一个模型，然后看看能不能被clone\_model函数成功克隆该模型，如果可以，说明你这个层的定义已经规范了。
> 
> 
> 
> 3、如果要用TPU训练，不要在模型的最后用add\_loss自定义loss，也不要用add\_metric来添加metric。如果需要自定义复杂的loss或metric，请将它们定义为一个层的输出，参考[这种写法](<https://spaces.ac.cn/archives/4493>)。
> 
> 
> 
> 4、如果要用TPU训练，切忌在训练过程中使用动态（变长）的写法，比如使用tf\.where时参数x,y不能为None，否则tf\.where的结果长度不确定，还有tf中几乎所有带有dynamic字眼的函数都不能用。

可以看到，所谓的“标准”，其实就是最大程度上模仿着keras已有的写法了，尽量少自己创造。只要做到1、2点，就可以在tf\.keras下轻易用多GPU训练了；后面3、4两点是针对TPU填的坑，总的来说就是一切要是静态的。

尽管tensorflow 2\.0开始已经默认动态图了，但笔者并不推荐使用，个人认为我们应当去习惯静态图的模型构建流程。动态图虽然方便调试，但会让我们对这种即时的输出结果产生严重依赖，降低我们面对复杂问题时的debug能力。类似地，我也不推荐使用代码补全、代码提示等工具，这些工具会让我们产生过多依赖，使得我们不真正去了解你要用的函数本身。（个人观点，不喜勿喷。）

## 人性化的胜利

如果没有记错，笔者是2015年初接触的Keras，当时也没有太多深度学习框架，只是想找个趁手的工具实现几个简单的模型，于是就找到了Keras，一直用到现在。不仅仅是笔者，也许Keras的作者们当时都没有想到，在今天Keras居然成为了tensorflow的黄金标准。

笔者觉得这并非偶然。tensorflow曾有过不少上层API框架，比如tensorlayer，比如tf\.slim，比如TFLearn，但为啥最终选择了Keras？除了Keras的“历史悠久”之外，还在于Keras真正当之无愧为一个优雅的封装实现。这一年来，笔者时不时会去读Keras源码，每次读Keras源码都会被它的严密性、优雅性所震撼，它是一件当之无愧的艺术品，一个人性化的创造。

所以，这是人性化的胜利。
