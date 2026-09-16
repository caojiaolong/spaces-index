# 重新写了之前的新词发现算法：更快更好的新词发现

> 作者：苏剑林 · 科学空间 · 2019-09-09
>
> 原文：<https://spaces.ac.cn/archives/6920>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

新词发现是NLP的基础任务之一，主要是希望通过无监督发掘一些语言特征（主要是统计特征），来判断一批语料中哪些字符片段可能是一个新词。本站也多次围绕“新词发现”这个话题写过文章，比如：

> [《新词发现的信息熵方法与实现》](<https://spaces.ac.cn/archives/3491>)
> [《【中文分词系列】 2\. 基于切分的新词发现》](<https://spaces.ac.cn/archives/3913>)
> [《【中文分词系列】 5\. 基于语言模型的无监督分词》](<https://spaces.ac.cn/archives/3956>)
> [《【中文分词系列】 7\. 深度学习分词？只需一个词典！》](<https://spaces.ac.cn/archives/4245>)
> [《【中文分词系列】 8\. 更好的新词发现算法》](<https://spaces.ac.cn/archives/4256>)
> [《分享一次专业领域词汇的无监督挖掘》](<https://spaces.ac.cn/archives/6540>)

在这些文章之中，笔者觉得理论最漂亮的是《基于语言模型的无监督分词》，而作为新词发现算法来说综合性能比较好的应该是《更好的新词发现算法》，本文就是复现这篇文章的新词发现算法。

## 背景

当时写[《【中文分词系列】 8\. 更好的新词发现算法》](<https://spaces.ac.cn/archives/4256>)的时候，笔者就已经给出过一个基本的实现并且做过验证了。然而，当初的版本是纯Python写的，并且当时只是想着快速验证效果，所以写得相当随意，会存在比较严重的效率问题。最近把这事想起来了，觉得不想浪费这个算法，所以就重写了一遍，利用了一些工具和技巧，把速度提上去了。

顺便说一下，“新词发现”是一个比较通俗的叫法，更准确的叫法应该是“无监督构建词库”，因为原则上它能完整地构建一个词库出来，而不仅仅是“新词”。当然，你可以将它跟常用词库进行对比，删掉常见词，就可以得到新词了。

## 细节

主要改动如下：

> 1、使用了语言模型工具[kenlm](<https://github.com/kpu/kenlm/>)的count\_ngrams程序来统计ngram。由于kenlm是用C\+\+写的，速度有保证，并且它还做了优化，所以对内存很友好。
> 
> 
> 
> 2、在第二次遍历词库以得到候选词的时候，使用了Trie树结构来加速搜索字符串是否出现过某个ngram。Trie树或者其变种基本上是所有基于词典的分词工具的标配，就是因为它可以加快搜索字符串中是否出现过词典中的词。

## 使用

本次开源地址位于：

> [https://github\.com/bojone/word\-discovery](<https://github.com/bojone/word-discovery>)

注意这个脚本应该只能在Linux系统下使用。如果你想要在Windows下使用，应该需要做些修改，具体做哪些修改，我也不知道，请自行解决。<strong>注意算法本身理论上能适用于任意语言，但本文的实现原则上只适用于以“字”为基本单位的语言。</strong>

<strong>在决定使用该库之前，还望读者能花点时间读一下[《【中文分词系列】 8\. 更好的新词发现算法》](<https://spaces.ac.cn/archives/4256>)，对算法步骤有个基本了解，以便更好地弄懂下述使用步骤。</strong>

Github中核心的脚本是[word\_discovery\.py](<https://github.com/bojone/word-discovery/blob/master/word_discovery.py>)，它包含了完整的实现和使用例子。下面我们简单梳理一下这个例子。

首先，写一个语料的生成器，逐句返回语料：



<pre><code class="language-python">import re
import glob

# 语料生成器，并且初步预处理语料
# 这个生成器例子的具体含义不重要，只需要知道它就是逐句地把文本yield出来就行了
def text_generator():
    txts = glob.glob('/root/thuctc/THUCNews/*/*.txt')
    for txt in txts:
        d = open(txt).read()
        d = d.decode('utf-8').replace(u'\u3000', ' ').strip()
        yield re.sub(u'[^\u4e00-\u9fa50-9a-zA-Z ]+', '\n', d)</code></pre>



读者不需要看懂我这个生成器究竟在做什么，只需要知道这个生成器就是在<u>逐句地把原始语料给<code>yield</code>出来</u>就行了。如果你还不懂生成器怎么写，请自己去学。请不要在此文章内讨论“语料格式应该是怎样的”、“我要怎么改才能适用我的语料”这样的问题，谢谢。

顺便提一下，因为是无监督训练，语料一般都是越大越好，几百M到几个G都可以，但其实如果你只要几M的语料（比如一部小说），也可以直接测试，也能看到基本的效果（但可能要修改下面的参数）。

有了生成器之后，配置一些参数，然后就可以逐个步骤执行了：



<pre><code class="language-python">min_count = 32
order = 4
corpus_file = 'thucnews.corpus' # 语料保存的文件名
vocab_file = 'thucnews.chars' # 字符集保存的文件名
ngram_file = 'thucnews.ngrams' # ngram集保存的文件名
output_file = 'thucnews.vocab' # 最后导出的词表文件名


write_corpus(text_generator(), corpus_file) # 将语料转存为文本
count_ngrams(corpus_file, order, vocab_file, ngram_file) # 用Kenlm统计ngram
ngrams = KenlmNgrams(vocab_file, ngram_file, order, min_count) # 加载ngram
ngrams = filter_ngrams(ngrams.ngrams, ngrams.total, [0, 2, 4, 6]) # 过滤ngram</code></pre>



注意，kenlm需要一个<u>以空格分词的、纯文本格式的</u>语料作为输入，而<code>write_corpus</code>函数就是帮我们做这件事情的，然后<code>count_ngrams</code>就是调用kenlm的count\_ngrams程序来统计ngram。所以，你需要自行编译好kenlm，并把它的count\_ngrams放到跟word\_discovery\.py同一目录下。如果有Linux环境，那kenlm的编译相当简单，笔者之前在[这里](<https://spaces.ac.cn/archives/3956#%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%AD%E7%BB%83>)也讨论过kenlm，可以参考。<code>count_ngrams</code>执行完毕后，结果会保存在一个二进制文件中，而<code>KenlmNgrams</code>就是读取这个文件的，如果你输入的语料比较大，那么这一步会需要比较大的内存。最后<code>filter_ngrams</code>就是过滤ngram了，&#91;0, 2, 4, 6&#93;是互信息的阈值，其中第一个0无意义，仅填充用，而2, 4, 6分别是2gram、3gram、4gram的互信息阈值，基本上单调递增比较好。

至此，我们完成了所有的“准备工作”，现在可以着手构建词库了。首先构建一个ngram的Trie树，然后用这个Trie树就可以做一个基本的“预分词”：



<pre><code class="language-python">ngtrie = SimpleTrie() # 构建ngram的Trie树

for w in Progress(ngrams, 100000, desc=u'build ngram trie'):
    _ = ngtrie.add_word(w)

candidates = {} # 得到候选词
for t in Progress(text_generator(), 1000, desc='discovering words'):
    for w in ngtrie.tokenize(t): # 预分词
        candidates[w] = candidates.get(w, 0) + 1</code></pre>



这个预分词的过程在[《【中文分词系列】 8\. 更好的新词发现算法》](<https://spaces.ac.cn/archives/4256>)就介绍过了，总之有点像最大匹配，由ngram片段连接成尽可能长的候选词。

最后，再把候选词过滤一下，就可以把词库保存下来了：



<pre><code class="language-python"># 频数过滤
candidates = {i: j for i, j in candidates.items() if j &gt;= min_count}
# 互信息过滤(回溯)
candidates = filter_vocab(candidates, ngrams, order)

# 输出结果文件
with open(output_file, 'w') as f:
    for i, j in sorted(candidates.items(), key=lambda s: -s[1]):
        s = '%s %s\n' % (i.encode('utf-8'), j)
        f.write(s)</code></pre>



## 评测

读者之前老说我写的这些算法没有标准评测，这次我就做了一个简单的评测，评测脚本是[evaluate\.py](<https://github.com/bojone/word-discovery/blob/master/evaluate.py>)。

具体来说，以[THUCNews](<http://thuctc.thunlp.org/>)为基础语料，就用上述脚本构建一个词库（总用时约40分钟），只保留<u>前5万个词</u>，用结巴分词加载这个5万词的词库（不用它自带的词库，并且关闭新词发现功能），这就构成了一个基于无监督词库的分词工具，然后用这个分词工具去分[bakeoff 2005](<http://sighan.cs.uchicago.edu/bakeoff2005/>)提供的测试集，并且还是用它的测试脚本评测，最终在PKU这个测试集上得分是：
$$\begin{array}{c|c|c}
\hline
\text{RECALL} & \text{PRECISION} & \text{F1}\\
\hline
0.777 & 0.711 & 0.742\\
\hline\end{array}$$
也就是说能做到<strong>0\.742</strong>的F1。这是个什么水平呢？ICLR 2019有一篇文章叫做[《Unsupervised Word Discovery with Segmental Neural Language Models》](<https://openreview.net/forum?id=r1NDBsAqY7>)，里边提到了它在同一个测试集上的结果为F1=0\.731，照这样看这个算法的结果还不差于顶会的最优结果呢～读者可以自行下载[THUCNews](<http://thuctc.thunlp.org/>)语料来完整地复现上述结果。

另外，更多的语料可以实现更好的效果。这是我从500万篇微信公众号文章（保存为文本之后是20多G）提取出来的词库：[wx\.vocab\.zip](<https://spaces.ac.cn/usr/uploads/2019/09/1023754363.zip>)，供读者有需使用。保留这个词库的前5万词，然后进行同样的评测，F1明显超过了顶会的结果：
$$\begin{array}{c|c|c}
\hline
\text{RECALL} & \text{PRECISION} & \text{F1}\\
\hline
0.799 & 0.734 & 0.765\\
\hline\end{array}$$

（注：这里是为了给效果提供一个直观感知，比较可能是不公平的，因为我不确定这篇论文中的训练集用了哪些语料。但我感觉在相同时间内本文算法会优于论文的算法，因为直觉论文的算法训练起来会很慢。作者也没有开源，所以有不少不确定之处，如有错谬，请读者指正。）

## 总结

本文复现了笔者之前提出了新词发现（词库构建）算法，主要是做了速度上的优化，然后做了做简单的效果评测。但具体效果读者还是得在使用中慢慢调试了。

祝大家使用愉快～Enjoy it！
