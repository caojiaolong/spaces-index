# bert4keras在手，baseline我有：百度LIC2020

> 作者：苏剑林 · 科学空间 · 2020-04-02
>
> 原文：<https://spaces.ac.cn/archives/7321>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

百度的“[2020语言与智能技术竞赛](<http://lic2020.cipsc.org.cn/>)”开赛了，今年有五个赛道，分别是机器阅读理解、推荐任务对话、语义解析、关系抽取、事件抽取。每个赛道中，主办方都给出了基于PaddlePaddle的baseline模型，这里笔者也基于[bert4keras](<https://github.com/bojone/bert4keras>)给出其中三个赛道的个人baseline，从中我们可以看到用bert4keras搭建baseline模型的方便快捷与简练。

> <strong>地址：</strong>[https://github\.com/bojone/lic2020\_baselines](<https://github.com/bojone/lic2020_baselines>)

## 思路简析

这里简单分析一下这三个赛道的任务特点以及对应的baseline设计。

### 阅读理解

样本示例：



<pre><code class="language-python">{
    "context": "这位朋友你好,女性出现妊娠反应一般是从6-12周左右,也就是女性怀孕1个多月就会开始出现反应,第3个月的时候,妊辰反应基本结束。 而大部分女性怀孕初期都会出现恶心、呕吐的感觉,这些症状都是因人而异的,除非恶心、呕吐的非常厉害,才需要就医,否则这些都是刚怀孕的的正常症状。1-3个月的时候可以观察一下自己的皮肤,一般女性怀孕初期可能会产生皮肤色素沉淀或是腹壁产生妊娠纹,特别是在怀孕的后期更加明显。 还有很多女性怀孕初期会出现疲倦、嗜睡的情况。怀孕三个月的时候,膀胱会受到日益胀大的子宫的压迫,容量会变小,所以怀孕期间也会有尿频的现象出现。月经停止也是刚怀孕最容易出现的症状,只要是平时月&gt;      经正常的女性,在性行为后超过正常经期两周,就有可能是怀孕了。 如果你想判断自己是否怀孕,可以看看自己有没有这些反应。当然这也只是多数人的怀孕表现,也有部分女性怀孕表现并不完全是这样,如果你无法确定自己是否怀孕,最好去医院检查一下。",
    "qas": [
        {
            "question": "怀孕多久会有反应",
            "id": "f2843cffb845aad0100062841222023e",
            "answers": [
                {
                    "text": "6-12周左右",
                    "answer_start": -1
                },
                {
                    "text": "6-12周",
                    "answer_start": -1
                },
                {
                    "text": "1个多月",
                    "answer_start": -1
                }
            ]
        }
    ]
}
</code></pre>



这个baseline其实没什么好说的，就是经过BERT之后，接两个全连接\+Softmax分别预测答案的首和尾。有些训练样本标注了多个答案，但预测的时候只需要预测一个答案，所以训练阶段每次只随机选取一个答案进行训练。

### 关系抽取

样本示例：



<pre><code class="language-python">{
    "text": "谢霆锋扮演的花无缺2004年由大导演王晶执导的40集电视连续剧《绝代双骄》的改版《小鱼儿与花无缺》上映，小编不是针对导演和演员，这部电视连续剧确实可说是天雷滚滚的魔改版，得亏全部主演凭借颜值和演技把观众拴住了，故事情节改的一塌糊涂",
    "spo_list": [
        {
            "predicate": "导演",
            "object": {
                "@value": "王晶"
            },
            "subject": "小鱼儿与花无缺"
        },
        {
            "predicate": "主演",
            "object": {
                "@value": "谢霆锋"
            },
            "subject": "小鱼儿与花无缺"
        },
        {
            "predicate": "主角",
            "object": {
                "@value": "花无缺"
            },
            "subject": "绝代双骄"
        },
        {
            "predicate": "饰演",
            "object": {
                "inWork": "小鱼儿与花无缺",
                "@value": "花无缺"
            },
            "subject": "谢霆锋"
        }
    ]
}
</code></pre>



关系抽取就是去年的三元组抽取，只不过今年做了一些升级。升级的地方在于考虑同一个predicate的多义性，比如“饰演”，有可能是指饰演了哪部电视剧，也有可能指饰演了电视剧中的哪个角色，如果同一个句子中含有多个不同的被饰演的对象（object），那么要全部抽取出来才算对。说是说升级，但其实没有本质变化，我们只需要把predicate和object对应的前缀拼接起来作为不同的predicate，就退化为常规的三元组抽取问题了，比如“饰演\_@value”、“饰演\_inWork”当成两个不同的predicate分别抽取。笔者baseline模型依然是基于去年的“半指针\-半标注”设计，详情可参考[《基于DGCNN和概率图的轻量级信息抽取模型》](<https://spaces.ac.cn/archives/6671>)。

### 事件抽取

样本示例：



<pre><code class="language-python">{
    "text": "雀巢裁员4000人：时代抛弃你时，连招呼都不会打！",
    "id": "409389c96efe78d6af1c86e0450fd2d7",
    "event_list": [
        {
            "event_type": "组织关系-裁员",
            "trigger": "裁员",
            "trigger_start_index": 2,
            "arguments": [
                {
                    "argument_start_index": 0,
                    "role": "裁员方",
                    "argument": "雀巢",
                    "alias": [
                        
                    ]
                },
                {
                    "argument_start_index": 4,
                    "role": "裁员人数",
                    "argument": "4000人",
                    "alias": [
                        
                    ]
                }
            ],
            "class": "组织关系"
        }
    ]
}</code></pre>



事件抽取是一个比较新的任务，要抽取出事件类型以及描述该事件的一些元素，同一个句子可能有多个事件，同一个实体可以同时描述多个事件（比如“XX月XX日”可能同时是多个事件发生时间）。本身事件抽取是比较复杂的任务，但是这次比赛主办方只评测(event\_type, role, argument) 构成的三元组，也就是这样的三元组匹配上了，就加1分。而event\_type、role都是离散的类别，argument则是原文中的一个实体，所以官方这样的评测指标就将这个任务退化为普通的实体标注问题了，因此可以用常规的序列标注模型来解决，笔者的baseline以及官方的baseline，都是转化为序列标注任务给出的。

## 匹配原序列

上面三个比赛，本质上都是抽取问题，也就是说输出的实体都是原文中的片段。但是原始文本经过BERT的tokenizer后，不一定跟原始文本对得上了，存在小幅度的“增”、“删”、“改”的可能性（比如转小写、空格数目变化、部分字符转写），这些小幅度的改动对于工程上的评估是无关紧要的，但是对于这种比赛或学术的评测却是很重要的，因为如果不同的字符哪怕看起来一样，都会匹配错误。比如读者可以把下属代码复制到python运行一下：



<pre><code class="language-python">u'à' == u'à'</code></pre>



为了将分词后的结果对应回原序列，笔者专门花了点时间，给bert4keras的<code>Tokenizer</code>补充了<code>rematch</code>方法，只要传入原始文本和分词后的结果，那么将会返回token到原始文本的映射关系，有了这个映射关系，就可以直接在原始文本中切片了。具体操作大家直接看baseline即可。

## 文章小结

写了三个baseline，又水了一篇博客～
