# 更别致的词向量模型(六)：代码、分享与结语

> 作者：苏剑林 · 科学空间 · 2017-11-19
>
> 原文：<https://spaces.ac.cn/archives/4681>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## 列表

> [更别致的词向量模型(一)：simpler glove](<https://spaces.ac.cn/archives/4667/>)
> 
> 
> 
> [更别致的词向量模型(二)：对语言进行建模](<https://spaces.ac.cn/archives/4669/>)
> 
> 
> 
> [更别致的词向量模型(三)：描述相关的模型](<https://spaces.ac.cn/archives/4671/>)
> 
> 
> 
> [更别致的词向量模型(四)：模型的求解](<https://spaces.ac.cn/archives/4675/>)
> 
> 
> 
> [更别致的词向量模型(五)：有趣的结果](<https://spaces.ac.cn/archives/4677/>)
> 
> 
> 
> [更别致的词向量模型(六)：代码、分享与结语](<https://spaces.ac.cn/archives/4681/>)

## 代码

本文的实现位于：[https://github\.com/bojone/simpler\_glove](<https://github.com/bojone/simpler_glove>)

源码修改自斯坦福的[glove原版](<https://github.com/stanfordnlp/GloVe>)，笔者仅仅是小修改，因为主要的难度是在统计共现词频这里，感谢斯坦福的前辈们提供了这一个经典的、优秀的统计实现案例。事实上，笔者不熟悉C语言，因此所作的修改可能难登大雅之台，万望高手斧正。

此外，为了实现上一节的“有趣的结果”，在github中我还补充了simpler\_glove\.py，里边封装了一个类，可以直接读取C版的simple glove所导出的模型文件（txt格式），并且附带了一些常用函数，方便调用。

## 分享

这里有一份利用本文的模型训练好的中文词向量，预料训练自百科百科，共100万篇文章，约30w词，词向量维度为128。其中分词时做了一个特殊的处理：把所有数字和英文都拆成单个的数字和字母了。如果需要实验的朋友可以下载：

> 链接:[http://pan\.baidu\.com/s/1jIb3yr8](<http://pan.baidu.com/s/1jIb3yr8>)
> 
> 
> 
> 密码:1ogw

## 结语

本文算是一次对词向量模型比较完整的探索，也算是笔者的理论强迫症的结果，幸好最后也得到了一个理论上比较好看的模型，初步治愈了我这个强迫症。而至于实验效果、应用等等，则有待日后进一步使用验证了。

本文的大多数推导，都可以模仿地去解释word2vec的skip gram模型的实验结果，读者可以尝试。事实上，word2vec的skip gram模型确实跟本文的模型有着类似的表现，包括词向量的模型性质等。

<strong>总的来说，理论与实验结合是一件很美妙的事情，当然，也是一件很辛苦的事情，因为就以上这些东西，就花了我几个月思考时间。</strong>
