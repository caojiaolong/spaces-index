# 短文本匹配Baseline：脱敏数据使用预训练模型的尝试

> 作者：苏剑林 · 科学空间 · 2021-03-05
>
> 原文：<https://spaces.ac.cn/archives/8213>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近凑着热闹玩了玩全球人工智能技术创新大赛中的“[小布助手对话短文本语义匹配](<https://tianchi.aliyun.com/competition/entrance/531851/introduction>)”赛道，其任务就是常规的短文本句子对二分类任务，这任务在如今各种预训练Transformer“横行”的时代已经没啥什么特别的难度了，但有意思的是，这次比赛脱敏了，也就是每个字都被影射为数字ID了，我们无法得到原始文本。

在这种情况下，还能用BERT等预训练模型吗？用肯定是可以用的，但需要一些技巧，并且可能还需要再预训练一下。本文分享一个baseline，它将分类、预训练和半监督学习都结合在了一起，能够用于脱敏数据任务。

## 模型概览

整个模型的思路，其实就是之前的文章[《必须要GPT3吗？不，BERT的MLM模型也能小样本学习》](<https://spaces.ac.cn/archives/7764>)所介绍的PET（Pattern\-Exploiting Training）的变体，用一个MLM模型来完成所有，示意图如下：

[![本文模型示意图](<https://spaces.ac.cn/usr/uploads/2021/03/3949869211.png>)](<https://spaces.ac.cn/usr/uploads/2021/03/3949869211.png>)

本文模型示意图

可以看到，全模型就只是一个MLM模型。具体来说，我们在词表里边添加了&#91;YES&#93;和&#91;NO&#93;两个标记，用来表示句子之间的相似性，通过&#91;CLS&#93;对应的输出向量来预测句子对的标签（&#91;YES&#93;或&#91;NO&#93;），然后构建语料的方式，就是常规的把句子对拼接起来，两个句子随机mask掉一些token，然后在对应的输出位置预测这个token。

这样一来，我们同时做了句子对的分类任务（&#91;CLS&#93;的预测结果），也做了MLM的预训练任务（其他被mask掉的token），而且没有标签的样本（比如测试集）也可以扔进去训练，只要不预测&#91;CLS&#93;就行了。于是我们通过MLM模型，把分类、预训练和半监督都结合起来了～

## 重用BERT

脱敏数据还可以用BERT吗？当然是可以的，脱敏数据对于BERT来说，其实就是Embedding层不一样而已，其他层还是很有价值的。所以重用BERT主要还是通过预训练重新对齐Embedding层。

在这个过程中，初始化很重要。首先，我们把BERT的Embedding层中的&#91;UNK&#93;、&#91;CLS&#93;、&#91;SEP&#93;等特殊标记拿出来，这部分不变；然后，我们分别统计密文数据和明文数据的字频，明文数据指的是任意的开源通用语料，不一定要密文数据对应的明文数据；接着按照频率简单对齐明文字表和密文字表。这样一来，我们就可以按照明文的字来取出BERT的Embedding层来作为相应的初始化。

简单来说，就是我用最高频的明文字对应的BERT Embedding，来初始化最高频的密文字，依此类推来做一个基本的字表对齐。个人的对比实验表明，这个操作可以明显加快模型的收敛速度。

## 代码分享

说到这里，模型就基本介绍完了，这样的操作我目前使用base版本的bert，在排行榜上的分数是0\.866，线下则已经是0\.952了（单模型，没做K\-fold融合，大家的线上线下差距貌似都蛮大）。这里分享自己的bert4keras实现：

> <strong>Github地址：[https://github\.com/bojone/oppo\-text\-match](<https://github.com/bojone/oppo-text-match>)</strong>

关于明文数据的词频，我已经实现统计好一份，也同步在Github了，大家直接用就好。建议大家训练完100个epoch，在3090上大概要6小时。

对了，如果你想用Large版本的BERT，不建议用哈工大开源的[RoBERTa\-wwm\-ext\-large](<https://github.com/ymcui/Chinese-BERT-wwm>)，理由在[《必须要GPT3吗？不，BERT的MLM模型也能小样本学习》](<https://spaces.ac.cn/archives/7764>)已经说过了，该版本不知道为啥随机初始化了MLM部分的权重，而我们需要用到MLM权重。需要用Large版本的，推荐用腾讯UER开源的[BERT Large](<https://share.weiyun.com/5G90sMJ>)。

## 文本小结

也没啥，就是分享了个比赛的简单baseline，顺便水了篇博客而已，希望对大家有所帮助～
