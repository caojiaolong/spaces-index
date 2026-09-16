# SimBERTv2来了！融合检索和生成的RoFormer\-Sim模型

> 作者：苏剑林 · 科学空间 · 2021-06-11
>
> 原文：<https://spaces.ac.cn/archives/8454>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

去年我们放出了[SimBERT](<https://spaces.ac.cn/archives/7427>)模型，它算是我们开源的比较成功的模型之一，获得了不少读者的认可。简单来说，SimBERT是一个融生成和检索于一体的模型，可以用来作为句向量的一个比较高的baseline，也可以用来实现相似问句的自动生成，可以作为辅助数据扩增工具使用，这一功能是开创性的。

近段时间，我们以[RoFormer](<https://spaces.ac.cn/archives/8265>)为基础模型，对SimBERT相关技术进一步整合和优化，最终发布了升级版的RoFormer\-Sim模型。

## 简介

RoFormer\-Sim是SimBERT的升级版，我们也可以通俗地称之为“SimBERTv2”，而SimBERT则默认是指旧版。从外部看，除了基础架构换成了RoFormer外，RoFormer\-Sim跟SimBERT没什么明显差别，事实上它们主要的区别在于训练的细节上，我们可以用两个公式进行对比：
\begin{array}{c}
\text{SimBERT} = \text{BERT} + \text{UniLM} + \text{对比学习} \\[5pt]
\text{RoFormer-Sim} = \text{RoFormer} + \text{UniLM} + \text{对比学习} + \text{BART} + \text{蒸馏}\\
\end{array}
除此之外，RoFormer\-Sim用到了更多的训练数据，并且拓展到了一般句式，也就是说，不同于SimBERT仅仅局限于疑问句，RoFormer\-Sim可以用来做一般句子的相似句生成，适用场景更大。其他训练细节还包括RoFormer\-Sim用了更大的batch\_size和maxlen等，这些在后面我们都会进一步介绍。

> <strong>开源地址：[https://github\.com/ZhuiyiTechnology/roformer\-sim](<https://github.com/ZhuiyiTechnology/roformer-sim>)</strong>

## 语料

SimBERT和RoFormer\-Sim的关键之处，都是在于训练语料的构建。RoFormer\-Sim的训练语料包括两部分：1、疑问类型相似句；2、通用类型相似句。对于疑问类相似句，我们还是像SimBERT一样，通过收集百度知道的相似问句，然后通过规则进一步清洗，这部分对我们来说已经很成熟了；对于通用类相似句，我们没有现成的地方可以搜集，于是我们提出了两种方案，一定程度上可以无监督地构建（伪）相似句对。

> <strong>第一个方案</strong>是基于“同一个问题的答案是相似的”思想，假如我们有现成的问答语料，该语料对于同一个问题有多个答案，那么我们可以将每个答案分句，然后用一个现成的相似度函数来比较答案之间的相似度，挑出相似度超过某个阈值的句对作为相似句对使用；
> 
> 
> 
> <strong>第二个方案</strong>则是基于“同一篇章的句子是相似的”思想，它更加简单直接一点，就是将每个篇章分句，然后用一个现成的相似度函数两两计算相似度，挑出相似度超过某个阈值的句对作为相似句对使用，显然该方案的合理性更弱，所以它的阈值也更高。

这里涉及到一个“现成的相似度函数”，我们是直接使用Jaccard相似度的一个变体，换言之只需要一个规则的、字符级别的相似度就好了，语义上的关联，则通过篇章内部的关联以及预训练模型本身的泛化能力来获得。通过第一个方案，我们从几个阅读理解数据集中构建了约450万个（伪）相似句对；通过第二个方案，我们从30多G的平行预料中构建了约470万个（伪）相似句对；而爬取的问句则达到了约3000万个相似句组（一组可以构成多对）。从这个角度看来，问句的数目是远超于一般句式的，所以我们按照1:1的方式从中采样，使得每种句式的样本都均衡。

## 生成

RoFormer\-Sim的训练方式跟SimBERT基本一样，如下图所示。稍微不同的是，为了增强模型的生成能力，在构造训练语料的时候，我们还随机地将输入句子的部分token替换为&#91;MASK&#93;，这种预训练方法首先由[BART](<https://papers.cool/arxiv/1910.13461>)提出。而我们跟BART的区别在于：BART是“输入带噪声的句子，输出原句子”，我们是“输入带噪声的句子，输出原句子的一个相似句”，理论上我们的任务还更难。

[![SimBERT训练方式示意图](<https://spaces.ac.cn/usr/uploads/2020/05/2840550561.png>)](<https://spaces.ac.cn/usr/uploads/2020/05/2840550561.png>)

SimBERT训练方式示意图

生成效果没什么特别好的评测指标，我们直接目测一些例子就好：



<pre><code class="language-python">gen_synonyms(u'广州和深圳哪个好？')
[
    '深圳和广州哪个好？',
    '广州和深圳哪个好',
    '广州和深圳哪个更好？',
    '深圳和广州哪个更好？',
    '深圳和广州，那个更好？',
    '深圳和广州哪个好一些呢？',
    '深圳好还是广州好？',
    '广州和深圳哪个地方好点？',
    '广州好还是深圳好？',
    '广州和深圳哪个好一点',
    '广州和深圳哪个发展好？',
    '深圳好还是广州好',
    '深圳和广州哪个城市更好些',
    '深圳比广州好吗？',
    '到底深圳和广州哪个好？为什么呢？',
    '深圳究竟好还是广州好',
    '一般是深圳好还是广州好',
    '广州和深圳那个发展好点',
    '好一点的深圳和广州那边好？',
    '深圳比广州好在哪里？'
]

gen_synonyms(u'科学技术是第一生产力。')
[
    '科学技术是第一生产力！',
    '科学技术是第一生产力',
    '一、科学技术是第一生产力。',
    '一是科学技术是第一生产力。',
    '第一，科学技术是第一生产力。',
    '第一生产力是科学技术。',
    '因为科学技术是第一生产力。',
    '科学技术是第一生产力知。',
    '也即科学技术是第一生产力。',
    '科学技术是第一生产力吗',
    '科技是第一生产力。',
    '因此，科学技术是第一生产力。',
    '其次，科学技术是第一生产力。',
    '科学技术才是第一生产力。',
    '科学技术是第一生产力吗？',
    '第二，科学技术是第一生产力。',
    '所以说科学技术是第一生产力。',
    '科学技术确实是第一生产力。',
    '科学技术还是第一生产力',
    '科学技术是第一生产力对吗？'
]</code></pre>



总的来说，初步实现了任意句式的相似扩增，但问句的扩增效果优于一般句型，这是因为训练语料中问句的质量就明显高于一般句型。由于进行了仿BART式训练，所以除了直接进行相似句生成外，我们还可以自行把某些部分mask掉，让模型自行发散扩充，比如：



<pre><code class="language-python">gen_synonyms(u'科学技术是第一生产力。', mask_idxs=[6, 7])  # mask掉“第一”
[
    "科学技术是第一生产力",
    "2、科学技术是第一生产力。",
    "科学技术是第一生产力，也是第二生产力。",
    "科学技术是第一生产力，科学发展是第二生产力。",
    "9、科学技术是第一生产力。",
    "第一，科学技术是一种生产力。",
    "科学技术是生产力。",
    "科学技术是第二生产力。",
    "科学技术是第一生产力”现在提出来的。",
    "一、科学技术是一种生产力。",
    "科学技术是第一生产力是什么意思",
    "科学技术是一种主要生产力。",
    "一：科学技术是最高生产力。",
    "指科学技术不是第一生产力。",
    "科学技术是第二生产力，第一生产力又是第二生产力。",
    "二、科学技术是一种生产力。",
    "世界上第一种生产力是科学技术。",
    "科学技术是社会主义生产力之一。",
    "第二，科学技术也是第二生产力。",
    "科技是一切生产力。"
]</code></pre>



更多玩法，请大家自行挖掘了。

## 检索

增加一般句式的语料、引入仿BART式训练，这些改动都相对来说提升了生成模型的效果。然而，我们意外地发现，检索模型（即句子编码模型）的效果却降低了。估计的原因，可能是更多的语料、更大的噪声虽然加大了生成模型的难度，但对于对比学习来说，这些不同句式的或者带噪声的样本作为负样本，反而是难度降低了。比如，如果一个batch同时有疑问句和陈述句，那么模型可以简单地通过句式（而不是语义）就可以识别出不少负样本，从而降低了对语义的理解能力。

当然，SimBERT和RoFormer\-Sim的本质定位都是相似句扩增模型，检索模型只是它的“副产品”，但我们仍然希望这个“副产品”能尽可能好一些。为此，我们在RoFormer\-Sim训练完之后，进一步通过蒸馏的方式把SimBERT的检索效果转移到RoFormer\-Sim上去，从而使得RoFormer\-Sim的检索效果基本持平甚至优于SimBERT。蒸馏的方式很简单，假如对于同一批句子，SimBERT出来的句向量为$u_1, u_2, \cdots, u_n$，RoFormer\-Sim出来的句向量为$v_1, v_2, \cdots,v_n$，那么就以
\begin{equation}\mathcal{L}_{\text{sim}} = \frac{\lambda}{n^2}\sum_{i=1}^n\sum_{j=1}^n (\cos(u_i,u_j)-\cos(v_i,v_j))^2\end{equation}
为loss进行学习，这里$\lambda=100$。当然，为了防止模型“遗忘”掉生成模型，蒸馏的同时还要加上生成损失，即$\mathcal{L}=\mathcal{L}_{\text{sim}}+\mathcal{L}_{\text{gen}}$。base版的蒸馏不需要很多步，大致5000步左右就可以训练完成了。

跟[《无监督语义相似度哪家强？我们做了个比较全面的评测》](<https://spaces.ac.cn/archives/8321>)一样，我们用同样的任务对比了SimBERT和RoFormer的检索效果（其中每一格的三个数据，分别代表“不加whitening”、“加whitening”、“加whitening\-256”的效果，同之前的评测）：
$$\small{\begin{array}{l|ccccc}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{STS-B} \\
\hline
\text{V1}\text{-P1} & 38.50 / \color{red}{23.64} / \color{red}{30.79}  & 48.54 / \color{red}{31.78} / \color{red}{40.01}  & 76.23 / \color{red}{75.05} / \color{red}{74.50}  & 15.10 / \color{green}{18.49} / \color{green}{15.64}  & 74.14 / \color{red}{73.37} / \color{green}{75.29}  \\
\text{V1}\text{-P2} & 38.93 / \color{red}{27.06} / \color{red}{30.79}  & 49.93 / \color{red}{35.38} / \color{red}{40.14}  & 75.56 / \color{red}{73.45} / \color{red}{74.39}  & 14.52 / \color{green}{18.51} / \color{green}{15.74}  & 73.18 / \color{green}{73.43} / \color{green}{75.12}  \\
\text{V1}\text{-P3} & 36.50 / \color{red}{31.32} / \color{red}{31.24}  & 45.78 / \color{red}{29.17} / \color{red}{40.98}  & 74.42 / \color{red}{73.79} / \color{red}{73.43}  & 15.33 / \color{green}{18.39} / \color{green}{15.87}  & 67.31 / \color{green}{70.70} / \color{green}{72.00}  \\
\text{V1}\text{-P4} & 33.53 / \color{red}{29.04} / \color{red}{28.78}  & 45.28 / \color{red}{34.70} / \color{red}{39.00}  & 73.20 / \color{red}{71.22} / \color{red}{72.09}  & 14.16 / \color{green}{17.32} / \color{green}{14.39}  & 66.98 / \color{green}{70.55} / \color{green}{71.43}  \\
\hline
\text{V2}\text{-P1} & 39.52 / \color{red}{25.31} / \color{red}{31.10}  & 50.26 / \color{red}{33.47} / \color{red}{40.16}  & 76.02 / \color{red}{74.92} / \color{red}{74.58}  & 14.37 / \color{green}{19.31} / \color{green}{14.81}  & 74.46 / \color{red}{71.00} / \color{green}{76.29}  \\
\text{V2}\text{-P2} & 39.71 / \color{red}{32.60} / \color{red}{30.89}  & 50.80 / \color{red}{37.62} / \color{red}{40.12}  & 75.83 / \color{red}{73.45} / \color{red}{74.52}  & 13.87 / \color{green}{19.50} / \color{green}{14.88}  & 73.47 / \color{green}{74.56} / \color{green}{76.40}  \\
\text{V2}\text{-P3} & 39.55 / \color{red}{24.61} / \color{red}{31.82}  & 50.25 / \color{red}{29.59} / \color{red}{41.43}  & 74.90 / \color{red}{73.95} / \color{red}{74.06}  & 14.57 / \color{green}{18.85} / \color{green}{15.26}  & 68.89 / \color{green}{71.40} / \color{green}{73.36}  \\
\text{V2}\text{-P4} & 36.02 / \color{red}{29.71} / \color{red}{29.61}  & 48.22 / \color{red}{35.02} / \color{red}{39.52}  & 73.76 / \color{red}{71.19} / \color{red}{72.68}  & 13.60 / \color{green}{16.67} / \color{green}{13.86}  & 68.39 / \color{green}{71.04} / \color{green}{72.43}  \\
\hline
\end{array}}$$
从表中可以看出，不管加不加whiteining，RoFormer\-Sim在大部分任务上都超过了SimBERT，可见蒸馏之后的RoFormer\-Sim的检索效果确实能获得提高，这个“副产品”也不至于太差了。

用同样的方法，我们也搞了个small版的RoFormer\-Sim，这时候蒸馏用的是base版的RoFormer\-Sim作为teacher模型，但蒸馏的步数需要比较多（50万左右），最终效果如下：
$$\small{\begin{array}{l|ccccc}
\hline
& \text{ATEC} & \text{BQ} & \text{LCQMC} & \text{PAWSX} & \text{STS-B} \\
\hline
\text{V1}_{\text{small}}\text{-P1} & 30.68 / \color{red}{27.56} / \color{red}{29.07}  & 43.41 / \color{red}{30.89} / \color{red}{39.78}  & 74.73 / \color{red}{73.21} / \color{red}{73.50}  & 15.89 / \color{green}{17.96} / \color{green}{16.75}  & 70.54 / \color{green}{71.39} / \color{green}{72.14}  \\
\text{V1}_{\text{small}}\text{-P2} & 31.00 / \color{red}{29.14} / \color{red}{29.11}  & 43.76 / \color{red}{36.86} / \color{red}{39.84}  & 74.21 / \color{red}{73.14} / \color{red}{73.67}  & 16.17 / \color{green}{18.12} / \color{green}{16.81}  & 70.10 / \color{green}{71.40} / \color{green}{72.28}  \\
\text{V1}_{\text{small}}\text{-P3} & 30.03 / \color{red}{21.24} / \color{red}{29.30}  & 43.72 / \color{red}{31.69} / \color{red}{40.81}  & 72.12 / \color{red}{70.27} / \color{red}{70.52}  & 16.93 / \color{green}{21.68} / \color{green}{18.75}  & 66.55 / \color{red}{66.11} / \color{green}{69.19}  \\
\text{V1}_{\text{small}}\text{-P4} & 29.52 / \color{red}{28.41} / \color{red}{28.57}  & 43.52 / \color{red}{36.56} / \color{red}{40.49}  & 70.33 / \color{red}{68.75} / \color{red}{69.01}  & 15.39 / \color{green}{21.57} / \color{green}{16.34}  & 64.73 / \color{green}{68.12} / \color{green}{68.24}  \\
\hline
\text{V2}_{\text{small}}\text{-P1} & 37.33 / \color{red}{23.59} / \color{red}{31.31}  & 47.90 / \color{red}{29.21} / \color{red}{42.07}  & 74.72 / \color{green}{74.94} / \color{red}{74.69}  & 13.41 / \color{green}{15.30} / \color{green}{13.61}  & 71.48 / \color{red}{69.01} / \color{green}{75.10}  \\
\text{V2}_{\text{small}}\text{-P2} & 37.42 / \color{red}{31.25} / \color{red}{31.18}  & 49.15 / \color{red}{38.01} / \color{red}{41.98}  & 75.21 / \color{red}{73.47} / \color{red}{74.78}  & 13.38 / \color{green}{15.87} / \color{green}{13.69}  & 72.06 / \color{green}{73.92} / \color{green}{75.69}  \\
\text{V2}_{\text{small}}\text{-P3} & 36.71 / \color{red}{30.33} / \color{red}{31.25}  & 49.73 / \color{red}{31.03} / \color{red}{42.74}  & 74.25 / \color{red}{72.72} / \color{red}{74.19}  & 14.58 / \color{green}{18.68} / \color{red}{14.40}  & 69.12 / \color{green}{71.07} / \color{green}{72.68}  \\
\text{V2}_{\text{small}}\text{-P4} & 32.80 / \color{red}{27.87} / \color{red}{29.65}  & 46.80 / \color{red}{36.93} / \color{red}{41.31}  & 72.30 / \color{red}{69.94} / \color{green}{72.38}  & 13.45 / \color{green}{16.93} / \color{red}{13.38}  & 67.21 / \color{green}{70.42} / \color{green}{71.39}  \\
\hline
\end{array}}$$

## 总结

本文介绍和发布了我们SimBERT的升级版——RoFormer\-Sim（SimBERTv2），它既可以用来扩充相似句子，也是语义相似度问题的一个较高的baseline。相比SimBERT，它最大的特点是将句型拓展到了一般类型，不再局限于相似问句。更多的玩法欢迎读者进一步挖掘和分享～
