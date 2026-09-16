# 万能的seq2seq：基于seq2seq的阅读理解问答

> 作者：苏剑林 · 科学空间 · 2019-12-05
>
> 原文：<https://spaces.ac.cn/archives/7115>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

今天给[bert4keras](<https://github.com/bojone/bert4keras>)新增加了一个例子：阅读理解式问答（[task\_reading\_comprehension\_by\_seq2seq\.py](<https://github.com/bojone/bert4keras/blob/master/examples/task_reading_comprehension_by_seq2seq.py>)），语料跟之前一样，都是用[WebQA和SogouQA](<https://spaces.ac.cn/archives/6906>)，最终的得分在0\.77左右（单模型，没精调）。

## 方法简述

由于这次主要目的是给bert4keras增加demo，因此效率就不是主要关心的目标了。这次的目标主要是通用性和易用性，所以用了最万能的方案——seq2seq来实现做阅读理解。

用seq2seq做的话，基本不用怎么关心模型设计，只要把篇章和问题拼接起来，然后预测答案就行了。此外，seq2seq的方案还自然地包括了判断篇章有无答案的方法，以及自然地导出一种多篇章投票的思路。总而言之，不考虑效率的话，seq2seq做阅读理解是一种相当优雅的方案。

这次实现seq2seq还是用UNILM的方案，如果还不了解的读者，可以先阅读[《从语言模型到Seq2Seq：Transformer如戏，全靠Mask》](<https://spaces.ac.cn/archives/6933>)了解相应内容。

## 模型细节

用UNILM方案搭建一个seq2seq模型在bert4keras中基本就是一行代码的事情，所以这个例子的主要工作在并不在模型的建立上，而是在输入输出的处理上面。

### 输入格式

首先是输入，输入格式很简单，一张图可以表达清楚：

[![用seq2seq做阅读理解的模型图示](<https://spaces.ac.cn/usr/uploads/2019/12/4085505815.png>)](<https://spaces.ac.cn/usr/uploads/2019/12/4085505815.png>)

用seq2seq做阅读理解的模型图示

### 输出处理

如果输入单个篇章和单个问题进行回答，那么直接按照seq2seq常规的处理方案——即beam search——来解码即可。

但是，WebQA和SogouQA面对的是搜索场景，即同时存在多篇文章来对同一个问题进行回答，这就涉及到投票方案的选择了。一种朴素的思路是：每个篇章结合问题单独用beam search解码，并且给出置信度，最后再按照[《基于CNN的阅读理解式问答模型：DGCNN》](<https://spaces.ac.cn/archives/5409>)的投票方式进行。这种方式的困难之处在于对每个答案给出一个合理的置信度，它相比我们后面给出的思路则显得不够自然，并且效率也稍低些。

这里我们给出一种跟beam search更加“契合”的方案：

> 先排除没有答案的篇章，然后在解码答案的每一个字时，直接将所有篇章预测的概率值（按照某种方式）取平均。

具体来说，所有篇章分别和问题拼接起来，然后给出各自的第一个字的概率分布。那些第一个字就给出&#91;SEP&#93;的篇章意味着它是没有答案的，排除掉它们。排除掉之后，将剩下的篇章的第一个字的概率分布取平均，然后再保留topk（beam search的标准流程）。预测第二个字时，每个篇章与topk个候选值分别组合，预测各自的第二个字的概率分布，然后再按照篇章将概率平均后，再给出topk。依此类推，直到出现&#91;SEP&#93;。（其实就是在普通的beam search基础上加上按篇章平均，如果实在弄不明白，那就只能去看源码了～）

此外，生成答案的方式应该有两种，一种是抽取式的，这种模式下答案只能是篇章的一个片段，另外一种是生成式的，即不需要考虑答案是不是篇章的片段，直接解码生成答案即可。这两种方式在本文的解码中都有相应的判断处理。

## 实验代码

代码链接：[task\_reading\_comprehension\_by\_seq2seq\.py](<https://github.com/bojone/bert4keras/blob/master/examples/task_reading_comprehension_by_seq2seq.py>)

最终在[SogouQA自带的评估脚本](<https://github.com/bojone/dgcnn_for_reading_comprehension>)上，valid集的分数大概是0\.77 (Accuracy=0\.7259005836184343，F1=0\.813860036706151，Final=0\.7698803101622926)，单模型成绩远超过了之前的[《开源一版DGCNN阅读理解问答模型（Keras版）》](<https://spaces.ac.cn/archives/6906>)模型。当然，提升是有代价的——预测速度大大降低，每秒只能预测2条数据左右。

（模型没精细调优，估计还有提升空间，当前还是以demo为主。）

## 文章小结

本文主要是给出了一个基于bert和seq2seq思路的阅读理解例子，并且给出了一种多篇章投票的beam search策略，供读者参考和测试～
