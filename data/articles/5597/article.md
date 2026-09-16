# 基于最小熵原理的NLP库：nlp zero

> 作者：苏剑林 · 科学空间 · 2018-05-31
>
> 原文：<https://spaces.ac.cn/archives/5597>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

陆陆续续写了几篇[最小熵](<https://spaces.ac.cn/tag/%E6%9C%80%E5%B0%8F%E7%86%B5/>)原理的博客，致力于无监督做NLP的一些基础工作。为了方便大家实验，把文章中涉及到的一些算法封装为一个库，供有需要的读者测试使用。

由于面向的是无监督NLP场景，而且基本都是NLP任务的基础工作，因此命名为nlp zero。

## 地址

Github: [https://github\.com/bojone/nlp\-zero](<https://github.com/bojone/nlp-zero>)
Pypi: [https://pypi\.org/project/nlp\-zero/](<https://pypi.org/project/nlp-zero/>)

可以直接通过



<pre><code class="language-bash">pip install nlp-zero==0.1.6</code></pre>



进行安装。整个库纯Python实现，没有第三方调用，支持Python2\.x和3\.x。

## 使用

### 默认分词

库内带了一个词典，可以作为一个简单的分词工具用



<pre><code class="language-python">from nlp_zero import *

t = Tokenizer()
t.tokenize(u'扫描二维码，关注公众号')
</code></pre>



自带的词典加入了一些通过新词发现挖掘出来的新词，并且经过笔者的人工优化，质量相对来说还是比较高的。

### 词库构建

通过大量的原始语料来构建词库。

首先我们需要写一个迭代容器，这样就不用一次性把所有语料加载到内存中了。迭代器的写法很灵活，比如我的数据存在MongoDB中，那就是：



<pre><code class="language-python">import pymongo
db = pymongo.MongoClient().weixin.text_articles

class D:
    def __iter__(self):
        for i in db.find().limit(10000):
            yield i['text']
</code></pre>



如果数据存在文本文件中，大概就是



<pre><code class="language-python">class D:
    def __iter__(self):
        with open('text.txt') as f:
            for l in f:
                yield l.strip() # python2.x还需要转编码
</code></pre>



然后就可以执行了



<pre><code class="language-python">from nlp_zero import *
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')

f = Word_Finder(min_proba=1e-8)
f.train(D()) # 统计互信息
f.find(D()) # 构建词库
</code></pre>



通过Pandas查看结果：



<pre><code class="language-python">import pandas as pd

words = pd.Series(f.words).sort_values(ascending=False)
</code></pre>



直接用统计出来的词库建立一个分词工具：



<pre><code class="language-python">t = f.export_tokenizer()

t.tokenize(u'今天天气不错')</code></pre>



### 句模版构建

跟前面一样，同样要写一个迭代器，这里不再重复。

因为构建句模版是基于词来统计的，因此还需要一个分词函数，可以用自带的分词器，也可以用外部的，比如结巴分词。



<pre><code class="language-python">from nlp_zero import *
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')

tokenize = Tokenizer().tokenize # 使用自带的分词工具
# 通过 tokenize = jieba.lcut 可以使用结巴分词

f = Template_Finder(tokenize, window=3)
f.train(D())
f.find(D())
</code></pre>



通过Pandas查看结果：



<pre><code class="language-python">import pandas as pd

templates = pd.Series(f.templates).sort_values(ascending=False)
idx = [i for i in templates.index if not i.is_trivial()]
templates = templates[idx] # 筛选出非平凡的模版
</code></pre>



每个模版已经被封装为一个类了。

### 层次分解

基于句模版来进行句子结构解析。



<pre><code class="language-python">from nlp_zero import *

# 建立一个前缀树，并加入模版
# 模版可以通过tuple来加入，
# 也可以直接通过“tire[模版类]=10”这样来加入
trie = XTrie()
trie[(None, u'呢')] = 10
trie[(None, u'可以', None, u'吗')] = 9
trie[(u'我', None)] = 8
trie[(None, u'的', None, u'是', None)] = 7
trie[(None, u'的', None, u'是', None, u'呢')] = 7
trie[(None, u'的', None)] = 12
trie[(None, u'和', None)] = 12

tokenize = Tokenizer().tokenize # 使用自带的分词工具
p = Parser(trie, tokenize) # 建立一个解析器

p.parse(u'鸡蛋可以吃吗') # 对句子进行解析
"""输出：
&gt;&gt;&gt; p.parse(u'鸡蛋可以吃吗')
+---&gt; (鸡蛋)可以(吃)吗
|     +---&gt; 鸡蛋
|     |     +---&gt; 鸡蛋
|     +---&gt; 可以
|     +---&gt; 吃
|     |     +---&gt; 吃
|     +---&gt; 吗
"""
</code></pre>



为了方便对结果进行调用以及可视化，输出结果已经被封装为一个SentTree类。这个类有三个属性：template（当前主模版）、content（当前主模版覆盖的字符串）、modules（语义块的list，每个语义块也是用SentTree来描述）。总的来说，就是按照[《最小熵原理（三）：“飞象过河”之句模版和语言结构》](<https://spaces.ac.cn/archives/5577>)一文中我们对语言结构的假设来设计的。

## 待续

如果有必要，请阅读源码寻求答案～后续有更新会继续在这里演示。
