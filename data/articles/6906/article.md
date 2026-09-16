# 开源一版DGCNN阅读理解问答模型（Keras版）

> 作者：苏剑林 · 科学空间 · 2019-08-20
>
> 原文：<https://spaces.ac.cn/archives/6906>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

去年写过[《基于CNN的阅读理解式问答模型：DGCNN》](<https://spaces.ac.cn/archives/5409>)，介绍了一个纯卷积的简单的问答模型。当时是用Tensorflow实现的，而且没有开源，这几天抽空用Keras复现了一下，决定开源。

## 模型综述

关于DGCNN的基本介绍，这里不再赘述。本文的模型并不是之前模型的重复实现，而是有所改动，这里只介绍一下被改动的地方。

> 1、这里放出的模型，线下验证集的分数大概是0\.72（之前大约是0\.75）；
> 
> 
> 
> 2、本次模型以字为单位，使用笔者之前探索出来的“[字词混合Embedding](<https://spaces.ac.cn/archives/6671#%E5%AD%97%E8%AF%8D%E6%B7%B7%E5%90%88Embedding>)”（之前是以词为单位）；
> 
> 
> 
> 3、本次模型完全去掉了人工特征（之前用了8个人工特征）；
> 
> 
> 
> 4、本次模型去掉了位置Embedding（之前将位置Embedding拼接到输入上）；
> 
> 
> 
> 5、模型架构和训练细节有所微调。

其中，以字为单位是为了让模型的标注更灵活（避免分词错误）；去掉人工特征也增强了模型的灵活性，并且提高了预测速度；至于去掉位置Embedding，是因为测了几次发现位置Embedding并没有明显提升；其它调整，比如用了最新的[RAdam优化器](<https://github.com/bojone/keras_radam>)进行训练，等等。

这次开放模型并没有精细追求分数的提高，只是纯粹引入一个Keras版本供大家参考，笔者相信提升空间还有不少，有兴趣的朋友自行调试即可（代码和数据集均已开放）。

## 开源地址

Github地址：[https://github\.com/bojone/dgcnn\_for\_reading\_comprehension](<https://github.com/bojone/dgcnn_for_reading_comprehension>)
（运行环境：Python 2\.7 \+ Tensorflow 1\.8 \+ Keras 2\.2\.4。运行环境问题勿扰，谢谢！）

词向量：[https://pan\.baidu\.com/s/1YYE2T3f\-lPyLBrJuUowAsA](<https://pan.baidu.com/s/1YYE2T3f-lPyLBrJuUowAsA>)，密码:5p0h

数据集：[https://pan\.baidu\.com/s/11C21BAupOpiYWoOx23J7Mg](<https://pan.baidu.com/s/11C21BAupOpiYWoOx23J7Mg>)，密码:dh9w

（数据集的开源如有不当之处，敬请来信告知，笔者会尽快删除。）

## 最后的话

祝大家试用愉快，欢迎多多交流～
