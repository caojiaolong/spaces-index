# Cool Papers更新：简单搭建了一个站内检索系统

> 作者：苏剑林 · 科学空间 · 2024-05-07
>
> 原文：<https://spaces.ac.cn/archives/10088>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

自从[《更便捷的Cool Papers打开方式：Chrome重定向扩展》](<https://spaces.ac.cn/archives/9978>)之后，[Cool Papers](<http://papers.cool>)有两次比较大的变化，一次是引入了venue分支，逐步收录了一些会议历年的论文集，如ICLR、ICML等，这部分是动态人工扩充的，欢迎有心仪的会议的读者提更多需求；另一次就是本文的主题，前天新增加的站内检索功能。

本文将简单介绍一下新增功能，并对搭建站内检索系统的过程做个基本总结。

## 简介

在Cool Papers的首页，我们看到搜索入口：

[![Cool Papers（2024.05.07）](<https://spaces.ac.cn/usr/uploads/2024/05/1948170921.png>)](<https://spaces.ac.cn/usr/uploads/2024/05/1948170921.png>)

Cool Papers（2024\.05\.07）

搜索功能的特点如下：

> 1、只搜索title和summary两个字段，暂不支持指定；
> 
> 
> 
> 2、可以指定搜索arxiv分支或者venue分支，不支持两个分支混合搜索；
> 
> 
> 
> 3、搜索query中的特殊字符（非英文字母和数字）都会被去掉；
> 
> 
> 
> 4、搜索query的单词不会自动主干化，这意味你搜索images不会命中image；
> 
> 
> 
> 5、在搜索结果页面，可以跟原本的页面内搜索功能混合使用。

总的来说，目前只是一个非常简单的文本搜索功能，先满足一部分用户的简单需求。对于更复杂的需求，后面再逐步更新。想象中会逐步引入的功能，包括指定字段、搜索Kimi FAQ内容、按stars排序、指定日期/分类（对于arxiv）、指定会议（对于venue）甚至放开像普通搜索引擎那样的加减法运算（可以用排除某些关键字）等，这些看用户的后续反馈吧，没有固定排期～

## 总结

事实上，站内搜索这个需求，在年初Cool Papers刚面向公众开放时就有用户提出了，之所以一直迟迟未上，主要原因是Cool Papers的论文是逐日收录的，一开始的论文数并不多，站内搜索没有太大意义。经过四个多月的累积，Cool Papers收录的Arxiv论文数到了8万多篇，加上venue分支的会议论文也达到了8万多，目前近17万篇论文，可以拿来搜一搜了。

确定了可以做，那么接下来就是怎么做了。基于关键字检索文章内容的检索系统，我们称为“全文检索（Full\-text Search）”，一般都是基于倒排索引和BM25相似度构建的，也就是说在算法上是成熟的。在实现上，Cool Papers的后端是[BottlePy](<https://bottlepy.org>)，所以我们要寻找Python下可用的全文检索库，才比较方便整合到Cool Papers中。

“Python \+ Full\-text Search”的组合选择并不多，最经典的是一个名为[Whoosh](<https://pypi.org/project/Whoosh/>)的库，从功能上来看它也确实能满足Cool Papers的需求。但Whoosh的问题是自从2016年4月后就再也没更新，所以总让人担心它会有什么隐患。另外一种选择是直接换带有全文检索功能的数据库来存数据，比如MongoDB。如果一开始就是用MongoDB来存数据的话，这无异是最简捷的方案，但Cool Papers选择的是Python自带的key\-value数据库Shelve，如果现在切换到MongoDB工程量太大，而且在Cool Papers的简单场景，MongoDB的速度也比不上Shelve。

经过多次寻觅无果之后，笔者在某次搜索中意外发现了一个Whoosh的非常小巧但强大的替代品——[tantivy](<https://tantivy-py.readthedocs.io/en/latest/>)，这是一个Rust写的全文检索库，但提供了python\-binding可以作为一个Python库用，API跟Whoosh差不多，但依然在持续更新。众所周知Rust以高效著称，所以可以说tantivy满足了笔者对全文检索库的所有完美想象——速度、小巧、简洁。

选定了全文检索库之后，剩下的就是前端工作了。在[《新年快乐！记录一下 Cool Papers 的开发体验》](<https://spaces.ac.cn/archives/9920>)中，笔者就已经说过自己是个完全没有艺术细胞的前端小白，设计UI这种工作对笔者来说可谓异常艰难，只能不断搜索查找、复制粘贴以及求助GPT4和Kimi。在各种东拼西凑和修修补补之下，总算勉强做出来了一个可用的界面。在开发过程中，还顺手把原本自带的页面搜索优化了一下，现在使用页面搜索应该速度会明显感觉到更快了。

## 末尾

在大家都在看KAN（[Kolmogorov\-Arnold Networks](<https://papers.cool/arxiv/2404.19756>)）的五一假期，笔者偷了个懒，没有看论文，而是给Cool Papers补上站内搜索功能。不敢说“千呼万唤始出来”，但也是部分用户催了很久的特性，在此做个简单介绍，并总结一下搭建经验。
