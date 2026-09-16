# 【中文分词系列】 2\. 基于切分的新词发现

> 作者：苏剑林 · 科学空间 · 2016-08-18
>
> 原文：<https://spaces.ac.cn/archives/3913>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章讲的是基于词典和AC自动机的快速分词。基于词典的分词有一个明显的优点，就是便于维护，容易适应领域。如果迁移到新的领域，那么只需要添加对应的领域新词，就可以实现较好地分词。当然，好的、适应领域的词典是否容易获得，这还得具体情况具体分析。本文要讨论的就是新词发现这一部分的内容。

这部分内容在去年的文章[《新词发现的信息熵方法与实现》](<https://spaces.ac.cn/archives/3491/>)已经讨论过了，算法是来源于matrix67的文章[《互联网时代的社会语言学：基于SNS的文本数据挖掘》](<http://www.matrix67.com/blog/archives/5044>)。在那篇文章中，主要利用了三个指标——频数、凝固度（取对数之后就是我们所说的互信息熵）、自由度（边界熵）——来判断一个片段是否成词。如果真的动手去实现过这个算法的话，那么会发现有一系列的难度。首先，为了得到$n$字词，就需要找出$1\sim n$字的切片，然后分别做计算，这对于$n$比较大时，是件痛苦的时间；其次，最最痛苦的事情是边界熵的计算，边界熵要对每一个片段就行分组统计，然后再计算，这个工作量的很大的。本文提供了一种方案，可以使得新词发现的计算量大大降低。

### 算法

回顾matrix67的算法做新词发现的过程，应该可以认识到，新词发现做的事情，就是根据语料判断给定片段是不是真的成词了，而所谓成词，就是它相对独立，不可切分。<strong>那为什么不反过来呢？为什么我们不去找一下哪些片段不能成词呢？根据前面的说法，我们说片段的凝固度大于一定程度时，片段可能成词（接下来要去考虑它的边界熵）。那这不就是说，如果片段的凝固度低于一定程度时，这个片段就不可能成词了吗？那么我们就可以在原来的语料中把它断开了。</strong>

我们可以做适当的简化，如果$a,b$是语料中相邻两字，那么可以统计$(a,b)$成对出现的次数$\#(a,b)$，继而估计它的频率$P(a,b)$，然后我们分别统计$a,b$出现的次数$\#a,\#b$，然后估计它们的频率$P(a),P(b)$，如果
$$\frac{P(a,b)}{P(a)P(b)} < \alpha \quad (\alpha\text{是给定的大于1的阈值})$$
那么就应该在原来的语料中把这两个字断开。这个操作本质上就是——我们根据这个指标，对原始语料进行初步的分词！在完成初步分词后，我们就可以统计词频了，然后根据词频来筛选。

<strong>对比matrix67文章中的三个指标，我们现在只用了两个：频数和凝固度，去掉了计算量最大的边界熵，而且，在计算凝固度时，我们只需要计算二字片段的凝固度，省掉了更多字片段的凝固度计算，但是，由于我们是基于切分的方式做的，因此我们少了很多计算量，但理论上却能够得任意长度的词语！</strong>

### 实现

看上去很完美——计算量少了，功能更强了。实际效果如何呢？跟matrix67文章中的算法的结果有多少出入？这个还得真的自己试过才能说了算。不过，我用了30万篇微信公众号的文章（约1GB）进行实验，发现效果是可以让人满意的，用时10分钟左右。下面给出实现代码，很短，纯Python，并且不用第三方库的支持，而且内存非常友好，这里的texts可以是一个列表，也可以是一个迭代器（每次返回一篇文章），配合tqdm库，可以方便地显示进度。最后，在统计时，用到了加$\gamma$平滑法，以缓解出现不合理的词。以前做这些统计计算的时候，不用想就用Pandas了，最近尝试了一下Python原生的一些库，发现也相当好用呢～



<pre><code class="language-python">import pymongo

db = pymongo.MongoClient().baike.items
def texts():
    for a in db.find(no_cursor_timeout=True).limit(1000000):
        yield a['content']

from collections import defaultdict #defaultdict是经过封装的dict，它能够让我们设定默认值
from tqdm import tqdm #tqdm是一个非常易用的用来显示进度的库
from math import log
import re

class Find_Words:
    def __init__(self, min_count=10, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int) #如果键不存在，那么就用int函数
                                                                  #初始化一个值，int()的默认结果为0
        self.total = 0.
    def text_filter(self, texts): #预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        for a in tqdm(texts):
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a): #这个正则表达式匹配的是任意非中文、
                                                              #非英文、非数字，因此它的意思就是用任
                                                              #意非中文、非英文、非数字的字符断开句子
                if t:
                    yield t
    def count(self, texts): #计数函数，计算单字出现频数、相邻两字出现的频数
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1
                self.total += 1
        self.chars = {i:j for i,j in self.chars.items() if j &gt;= self.min_count} #最少频数过滤
        self.pairs = {i:j for i,j in self.pairs.items() if j &gt;= self.min_count} #最少频数过滤
        self.strong_segments = set()
        for i,j in self.pairs.items(): #根据互信息找出比较“密切”的邻字
            _ = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
            if _ &gt;= self.min_pmi:
                self.strong_segments.add(i)
    def find_words(self, texts): #根据前述结果来找词语
        self.words = defaultdict(int)
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments: #如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1 #否则断开，前述片段作为一个词来统计
                    s = text[i+1]
            self.words[s] += 1 #最后一个“词”
        self.words = {i:j for i,j in self.words.items() if j &gt;= self.min_count} #最后再次根据频数过滤

fw = Find_Words(16, 1)
fw.count(texts())
fw.find_words(texts())

import pandas as pd
words = pd.Series(fw.words).sort_values(ascending=False)</code></pre>



Python流式读取SQL数据的参考代码：



<pre><code class="language-python">from sqlalchemy import *

def sql_data_generator():
    db = create_engine('mysql+pymysql://user:password@123.456.789.123/yourdatabase?charset=utf8')
    result = db.execution_options(stream_results=True).execute(text('select content from articles'))
    for t in result:
        yield t[0]</code></pre>



### 分析

当然，这个算法不能说完全没有缺点，还是有些问题值得探讨的。一般情况下，为了得到更细粒度的词语（避免分出太多无效的长词），我们可以选择较大的$\alpha$，比如$\alpha=10$，但是这带来一个问题：一个词语中相邻两个字的凝固度不一定很大。一个典型的例子是“共和国”，“和”跟“国”都是很频繁的字，“和国”两个字的凝固度并不高（在微信文本中大概为3左右），如果$\alpha$太大就会导致切错了这个词语（事实上，是“共和”跟“国”的凝固度高），这些例子还有很多，比如“林心如”的“心如”凝固度就不大（当然，如果语料来源于娱乐圈，那又另当别论）。而如果设置$\alpha=1$，则需要更大的语料库才能使得词库完备起来。这是在使用本算法时需要仔细考虑的。

### 微信词典

最后分享一份我从最近的30万微信公众号文章（1G左右， 3亿多字）中提取的一份词表，设置了最小凝固度为1，最小频数为100。从表中也可以发现，跟微信具有明显联系的词语都已经被提取出来，并且，这是最新的公众号文章，因此，最近的热点——奥运、王宝强——相关的词语也被提取出来了。

<strong>微信词典：[dict\.txt](<https://spaces.ac.cn/usr/uploads/2016/08/627400080.txt>)</strong>

### 参考链接

《非主流自然语言处理——遗忘算法系列（二）：大规模语料词库生成》：[http://www\.52nlp\.cn/forgetnlp2](<http://www.52nlp.cn/forgetnlp2>)
