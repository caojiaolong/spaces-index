# 跟风玩玩目前最大的中文GPT2模型（bert4keras）

> 作者：苏剑林 · 科学空间 · 2020-11-20
>
> 原文：<https://spaces.ac.cn/archives/7912>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

相信不少读者这几天都看到了清华大学与智源人工智能研究院一起搞的“清源计划”（相关链接[《中文版GPT\-3来了？智源研究院发布清源 CPM —— 以中文为核心的大规模预训练模型》](<https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw>)），里边开源了目前最大的中文GPT2模型CPM\-LM（26亿参数），据说未来还会开源200亿甚至1000亿参数的模型，要打造“中文界的GPT3”。

[![官方给出的CPM-LM的Few Shot效果演示图](<https://spaces.ac.cn/usr/uploads/2020/11/259176451.png>)](<https://spaces.ac.cn/usr/uploads/2020/11/259176451.png>)

官方给出的CPM\-LM的Few Shot效果演示图

我们知道，GPT3不需要finetune就可以实现Few Shot，而目前CPM\-LM的演示例子中，Few Shot的效果也是相当不错的，让人跃跃欲试，笔者也不例外。既然要尝试，肯定要将它适配到自己的[bert4keras](<https://github.com/bojone/bert4keras>)中才顺手，于是适配工作便开始了。本以为这是一件很轻松的事情，谁知道踩坑踩了快3天才把它搞好，在此把踩坑与测试的过程稍微记录一下。

## 模型介绍

该计划发布的第一个模型称为CPM\-LM，参数大约为26亿，预训练中文数据规模100GB，是一个单向的语言模型，其他细节大家自行到下面的链接阅读就好。这么大的参数量，一般我们都是直接使用而不考虑去finetune它的了，它所具备的能力就是无条件地随机生成文本，当然我们也可以实现给它一些引导，然后用它来实现文本续写，至于Few Shot之类的应用，本质上也都是文本续写的变式。

> 主页：[https://cpm\.baai\.ac\.cn/](<https://cpm.baai.ac.cn/>)
> 
> 
> 
> GitHub：[https://github\.com/TsinghuaAI/CPM\-Generate](<https://github.com/TsinghuaAI/CPM-Generate>)
> 
> 
> 
> 公众号：[https://mp\.weixin\.qq\.com/s/oI2Ak\-M57MSuycLVpVEiHw](<https://mp.weixin.qq.com/s/oI2Ak-M57MSuycLVpVEiHw>)

这里说一下模型结构的问题，也是笔者在适配过程中踩的第一个坑。CPM\-LM的模型架构跟OpenAI的GPT2是一样的，所以说白了这就是一个26亿参数的中文GPT2模型。开始笔者没认真看，然后又被[CPM\-LM\-TF2项目](<https://github.com/qhduan/CPM-LM-TF2/>)稍微误导了一下，导致在前期以为它的结构跟[GPT2\_ML](<https://github.com/imcaspar/gpt2-ml>)一样（GPT2\_ML既不是GPT，也不是GPT2，它介乎两者之间），很久都没调出合理的结果。而意识到这个问题后，重新搭建GPT2模型并且适配对应的权重，也不是什么难事了，包括权重转换到tf格式，有了[CPM\-LM\-TF2项目](<https://github.com/qhduan/CPM-LM-TF2/>)的参照，也不算困难。

## Tokenizer

在适配过程中踩到的第二个坑，是关于tokenizer的。不得不说，CPM\-LM所写的tokenizer，在笔者看来实在是难登大雅之堂，至今仍然我耿耿于怀。

该tokenizer实际上就是在Google的sentencepiece的基础上包装了一下，但是又包得特别不优雅，简直是强迫症患者的噩梦。具体来说，像BERT的tokenizer或者sentencepiece这类分词工具，都会默认去掉空格、换行符等分隔符，但是CPM\-LM想要将空格和换行符保留，所以在送入tokenizer之前将它们替换为了别的符号（目前是空格替换为“▂”、换行符替换为“▃”），最后输出之前再替换回来。这是一种常见的做法，是可以理解的，但我最不能理解的是换行符的替代符号“▃”居然不在它的sentencepiece模型的词表中！为了避免“▃”变成\<unk\>，CPM\-LM又将它替换为\<cls\>，也就是做了二次替换，才得到换行符的id\.\.\.

笔者第一次看到这样的设计时，内心是简直要崩溃的：往sentencepiece里边多加入个字符有这么难吗，何至于写成这样\.\.\.没办法，开源模型的才是大佬，只能想办法适配它了。笔者想了很久很久，对bert4keras原有的SpTokenizer修修补补了一番，总算勉强把它搞好了。

## 使用测试

吐槽就到这里吧，总之，经过笔者两天多的折腾，从0\.9\.3版本开始，bert4keras就可以加载CPM\-LM模型了，单跑预测估计需要16G以上的显存（我自己是22G的RTX）。模型权重的转换过程与基本的加载方案可见：

> <strong>GitHub：[https://github\.com/bojone/CPM\_LM\_bert4keras](<https://github.com/bojone/CPM_LM_bert4keras>)</strong>

一些Few Shot效果（输出结果会有一定的随机性，如果只关心Few Shot效果，可以考虑将解码方式换为beam search）：



<pre><code class="language-python"># 常识推理
# 本例输出：北京
query = u"""
美国的首都是华盛顿
法国的首都是巴黎
日本的首都是东京
中国的首都是
"""
print(text_expansion.generate(query[1:-1], 1)[0])

# 单词翻译
# 本例输出：bird
query = u"""
狗 dog
猫 cat
猪 pig
鸟 
"""
print(text_expansion.generate(query[1:-1], 1)[0])

# 主语抽取
# 本例输出：杨振宁
query = u"""
从1931年起，华罗庚在清华大学边学习边工作 华罗庚
在一间简陋的房间里，陈景润攻克了“哥德巴赫猜想” 陈景润
在这里，丘成桐得到IBM奖学金 丘成桐
杨振宁在粒子物理学、统计力学和凝聚态物理等领域作出里程碑性贡献 
"""
print(text_expansion.generate(query[1:-1], 1)[0])

# 三元组抽取
# 本例输出：张红,体重,140斤
query = u"""
姚明的身高是211cm，是很多人心目中的偶像。 -&gt;姚明，身高，211cm
虽然周杰伦在欧洲办的婚礼，但是他是土生土长的中国人-&gt;周杰伦，国籍，中国
小明出生于武汉，但是却不喜欢在武汉生成，长大后去了北京。-&gt;小明，出生地，武汉
吴亦凡是很多人的偶像，但是他却是加拿大人，另很多人失望-&gt;吴亦凡，国籍，加拿大
武耀的生日在5月8号，这一天，大家都为他庆祝了生日-&gt;武耀，生日，5月8号
《青花瓷》是周杰伦最得意的一首歌。-&gt;周杰伦，作品，《青花瓷》
北京是中国的首都。-&gt;中国，首都，北京
蒋碧的家乡在盘龙城，毕业后去了深圳工作。-&gt;蒋碧，籍贯，盘龙城
上周我们和王立一起去了他的家乡云南玩昨天才回到了武汉。-&gt;王立，籍贯，云南
昨天11月17号，我和朋友一起去了海底捞，期间服务员为我的朋友刘章庆祝了生日。-&gt;刘章，生日，11月17号
张红的体重达到了140斤，她很苦恼。-&gt;
"""
print(text_expansion.generate(query[1:-1], 1)[0])</code></pre>



## 文章小结

文章简单介绍了一个清华大学新开源的26亿参数的GPT2模型CPM\-LM，并将它适配到了bert4keras框架内，稍微吐槽了一下转换过程中遇到的坑，最后演示了一下CPM\-LM还不错的Few Shot效果。
