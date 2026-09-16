# 【搜出来的文本】⋅（三）基于BERT的文本采样

> 作者：苏剑林 · 科学空间 · 2021-01-22
>
> 原文：<https://spaces.ac.cn/archives/8119>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从这一篇开始，我们就将前面所介绍的采样算法应用到具体的文本生成例子中。而作为第一个例子，我们将介绍如何利用BERT来进行文本随机采样。所谓文本随机采样，就是从模型中随机地产生一些自然语言句子出来，通常的观点是这种随机采样是GPT2、GPT3这种单向自回归语言模型专有的功能，而像BERT这样的双向掩码语言模型（MLM）是做不到的。

事实真的如此吗？当然不是。利用BERT的MLM模型其实也可以完成文本采样，事实上它就是上一篇文章所介绍的Gibbs采样。这一事实首先由论文[《BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model》](<https://papers.cool/arxiv/1902.04094>)明确指出。论文的标题也颇为有趣：“BERT也有嘴巴，所以它得说点什么。”现在就让我们看看BERT究竟能说出什么来～

## 采样流程

首先，我们再次回顾上一篇文章所介绍的Gibbs采样流程：

> <strong>Gibbs采样</strong>
> 
> 
> 
> 初始状态为$\boldsymbol{x}_0=(x_{0,1},x_{0,2},\cdots,x_{0,l})$，$t$时刻状态为$\boldsymbol{x}_t=(x_{t,1},x_{t,2},\cdots,x_{t,l})$。
> 
> 
> 
> 通过如下流程采样出$\boldsymbol{x}_{t+1}$：
>   1、均匀地从$1,2,\cdots,l$中采样一个$i$；
>   2、计算$p(y|\boldsymbol{x}_{t,-i})=\frac{p(x_{t,1},\dots,x_{t,i-1},y,x_{t,i+1},\cdots,x_{t,l})}{\sum\limits_y p(x_{t,1},\dots,x_{t,i-1},y,x_{t,i+1},\cdots,x_{t,l})}$；
>   3、采样$y\sim p(y|\boldsymbol{x}_{t,-i})$；
>   4、$\boldsymbol{x}_{t+1} = {\boldsymbol{x}_t}_{[x_{t,i}=y]}$（即将$\boldsymbol{x}_t$的第$i$个位置替换为$y$作为$\boldsymbol{x}_{t+1}$）。

其中最关键的一步，就是$p(y|\boldsymbol{x}_{-i})$的计算了，它的具体含义是“通过除去第$i$个元素后的所有$l-1$个元素来预测第$i$个元素的概率”，了解BERT的读者应该都能明白过来：这不正是BERT的MLM模型所要做的事情吗？所以，MLM模型与Gibbs采样结合起来，其实就可以实现文本的随机采样了。

所以，将上述Gibbs采样流程落实到基于MLM的文本采样中，流程如下：

> <strong>MLM模型随机采样</strong>
> 
> 
> 
> 初始句子为$\boldsymbol{x}_0=(x_{0,1},x_{0,2},\cdots,x_{0,l})$，$t$时刻句子为$\boldsymbol{x}_t=(x_{t,1},x_{t,2},\cdots,x_{t,l})$。
> 
> 
> 
> 通过如下流程采样出新的句子$\boldsymbol{x}_{t+1}$：
>   1、均匀地从$1,2,\cdots,l$中采样一个$i$，将第$i$的位置的token替换为&#91;MASK&#93;，得到序列$\boldsymbol{x}_{t,-i}=(x_{t,1},\dots,x_{t,i-1},\text{[MASK]},x_{t,i+1},\cdots,x_{t,l})$；
>   3、将$\boldsymbol{x}_{t,-i}$送入到MLM模型中，算出第$i$个位置的概率分布，记为$p_{t+1}$；
>   3、从$p_{t+1}$采样一个token，记为$y$；
>   4、将$\boldsymbol{x}_{t}$的第$i$个token替换成$y$来作为$\boldsymbol{x}_{t+1}$。

读者或许留意到了，该采样流程只能采样出固定长度的句子，不会改变句子长度。确实如此，因为<u>Gibbs采样只能实现用某个分布中进行采样，而不同长度的句子事实上已经是属于不同的分布了，它们理论上是不存在交集的</u>，只不过通常我们我们建立语言模型时，是直接用自回归模型统一建模不同长度的句子分布，以至于我们没有察觉到“不同的句子事实上属于不同的概率分布”这个事实。

当然，要解决这一点也不是不可能，[《BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model》](<https://papers.cool/arxiv/1902.04094>)原论文就指出，可以将初始句子设为全部都是&#91;MASK&#93;的序列，这样我们就可以事先随机采样一个长度$l$，然后以$l$个&#91;MASK&#93;为初始句子来开始Gibbs采样过程，从而得到不同长度的句子。

## 参考代码

有了现成的MLM模型后，实现上述Gibbs采样其实是一件很简单的事情了，下面是基于bert4keras实现的参考代码：

> <strong>Gibbs采样参考代码：[basic\_gibbs\_sampling\_via\_mlm\.py](<https://github.com/bojone/bert4keras/blob/master/examples/basic_gibbs_sampling_via_mlm.py>)</strong>

下面是一些例子：

> <strong>初始句子：</strong>
> 科学技术是第一生产力。
> <strong>采样结果：</strong>
> 荣耀笔记本开箱怎么样？
> 微信记录没用被怎么办？
> 无法安装浏览器怎么办？
> epf转换器a7l怎么用？
> 没有安装浏览器怎么办？
> 荣耀笔记本充电怎么用？
> 无法打开asp\. net怎么办？
> 没有安装浏览器怎么办？
> 无法重启浏览器怎么办？
> ro汉巴换mac tv版怎么用？
> 
> 
> 
> <strong>初始句子：</strong>
> 北京新增3例本地确诊病例和1例无症状感染者
> <strong>采样结果：</strong>
> 澳门录得233宗h1n1感染案例和13宗放射性感染。
> 庆祝仪式是学院绘画、钢铁工参与创作的盛会。
> 庆祝仪式后吉卜力平台其他游戏也参加了庆祝。
> 临床试验发现中的g染色体多来自胃肠道感染。
> 临床试验发现，人们通常真正享受阴蒂的快感。
> 庆祝模式在吉卜力平台其他游戏中更加与庆祝。
> 庆祝模式在吉卜力，或其他游戏上更新和庆祝。
> 澳门录得20宗h1n1感染病例，2宗放射性感染。
> 临床试验发现女性的染色体常来自胃肠道感染。
> 临床试验发现90% 感染病例为m型胃肠道感染。
> 
> 
> 
> <strong>初始句子：</strong>
> 连续9个&#91;MASK&#93;
> <strong>采样结果：</strong>
> 你每天学你妈妈啊！
> 那晚，眼前白茫茫。
> 層層青翠綠意盎然。
> 幼儿园想做生意了。
> 究竟如何才能入官？
> 老师、同学，您好！
> 云山重重，两茫茫。
> 梅雨，窗外霧茫茫。
> 那时，眼前白茫茫。
> 還是很棒的切蛋糕！

笔者实验所使用的是Google开源的中文BERT base。可以看到，采样出来的句子还是比较丰富的，并且具有一定的可读性，这对于一个base版本的模型来说已经算是不错了。

对于连续&#91;MASK&#93;作为初始值来说，重复实验可能会得到很不一样的结果：

> <strong>初始句子：</strong>
> 连续17个&#91;MASK&#93;
> <strong>采样结果：</strong>
> 其他面瘫吃什么？其他面瘫吃什么好？
> 小儿面瘫怎么样治疗？面瘫吃什么药？
> 幼儿面瘫怎么样治疗？面瘫吃什么好？
> 儿童头痛是什么原因荨麻疹是什么病？
> 其他面瘫吃什么・ 其他面瘫吃什么好？
> 竟然洁具要怎么装进去水龙头怎么接？
> 其他面瘫吃什么好其他面瘫吃什么好？
> 孩子头疼是什么原因荨麻疹是什么病？
> 竟然厨房柜子挑不进去水龙头怎么插？
> 不然厨房壁橱找不到热水龙头怎么办？
> 
> 
> 
> <strong>初始句子：</strong>
> 连续17个&#91;MASK&#93;
> <strong>采样结果：</strong>
> フロクのツイートは下記リンクからこ覧下さい。
> 天方町に運行したいシステムをこ利用くたさい。
> エリアあります2つクロカー専門店からこ案内まて！
> 当サイトては割引に合うシステムを採用しています！
> 同時作品ては表面のシステムを使用する。
> メーカーの品は真面まてシステムを使用しています。
> 掲示板こ利用いたたシステムこ利用くたさい。
> 住中方は、生産のシステムを使用しています。
> エアウェアの住所レヘルをこ一覧下さい。
> フロクのサホートは下記リンクてこ覧下さい。

很神奇，日语都采样出来了，而且笔者用百度翻译看了一下，这些日语还算是可读的。一方面，这体现了随机采样结果的多样性，另一方面，这也体现了Goole版的中文BERT并没有做好去噪，训练语料应该夹杂了不少非中英文本的。

## 吃瓜思考

前段时间，Google、斯坦福、OpenAI等合作发表了一篇文章[《Extracting Training Data from Large Language Models》](<https://papers.cool/arxiv/2012.07805>)，指出GPT2等语言模型完全是可以重现（暴露）训练数据情况的，这个不难理解，因为语言模型本质上就是在背诵句子。而基于MLM的Gibbs采样表明，其实这个问题不仅GPT2这种显式的语言模型存在，像MLM这样的双向语言模型其实也是存在的。从我们上述的采样例子就可以看出些端倪了，比如采样出日语来，说明原始语料并没有做特别完善的去噪，而我们从“北京新增3例本地确诊病例和1例无症状感染者”出发，采样出了一些h1n1相关的结果，这反映出训练语料的时代性。这些结果都意味着，如果你不想开源模型暴露你的隐私，那么对预训练的语料就要做好清理工作了。

此外，关于[《BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model》](<https://papers.cool/arxiv/1902.04094>)这篇论文，还有一个瓜可以吃，那就是原论文说MLM模型是一个Markov Random Field，但事实上这是不对的，后来作者也在自己的主页上做了澄清，有兴趣的读者可以看[《BERT has a Mouth and must Speak, but it is not an MRF》](<https://sites.google.com/site/deepernn/home/blog/amistakeinwangchoberthasamouthanditmustspeakbertasamarkovrandomfieldlanguagemodel>)。总的来说，MLM用来做随机采样是没问题的，但它不能对上Markov Random Field。

## 本文小结

本文介绍了基于BERT的MLM所进行的文本随机采样，它实际上是Gibbs采样的自然应用。总的来说，本文只是一个相当简单的例子。对于已经对Gibbs采样有所了解的读者来说，本文几乎是没有技术难度的；如果还不是很了解Gibbs采样的读者，正好也可以通过这个具体的例子，来进一步去理解Gibbs采样的流程。
