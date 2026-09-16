# GAU\-α：尝鲜体验快好省的下一代Attention

> 作者：苏剑林 · 科学空间 · 2022-04-22
>
> 原文：<https://spaces.ac.cn/archives/9052>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《FLASH：可能是近来最有意思的高效Transformer设计》](<https://spaces.ac.cn/archives/8934>)中，我们介绍了GAU（Gated Attention Unit，门控线性单元），在这里笔者愿意称之为“目前最有潜力的下一代Attention设计”，因为它真正达到了“更快（速度）、更好（效果）、更省（显存）”的特点。

然而，有些读者在自己的测试中得到了相反的结果，比如收敛更慢、效果更差等，这与笔者的测试结果大相径庭。本文就来分享一下笔者自己的训练经验，并且放出一个尝鲜版“GAU\-α”供大家测试。

> <strong>开源地址：[https://github\.com/ZhuiyiTechnology/GAU\-alpha](<https://github.com/ZhuiyiTechnology/GAU-alpha>)</strong>

## GAU\-α

首先介绍一下开源出来的“GAU\-α”在CLUE任务上的成绩单：
$$\small{\begin{array}{c|ccccccccccc}
\hline
& \text{iflytek} & \text{tnews} & \text{afqmc} & \text{cmnli} & \text{ocnli} & \text{wsc} & \text{csl} & \text{cmrc2018} & \text{c3} & \text{chid} & \text{cluener}\\
\hline
\text{BERT} & 60.06 & 56.80 & 72.41 & 79.56 & 73.93 & 78.62 & 83.93 & 56.17 & 60.54 & 85.69 & 79.45 \\
\text{RoBERTa} & 60.64 & \textbf{58.06} & 74.05 & 81.24 & 76.00 & \textbf{87.50} & 84.50 & 56.54 & 67.66 & 86.71 & 79.47\\
\text{RoFormer} & 60.91 & 57.54 & 73.52 & 80.92 & \textbf{76.07} & 86.84 & 84.63 & 56.26 & 67.24 & 86.57 & 79.72\\
\text{RoFormerV2}^* & 60.87 & 56.54 & 72.75 & 80.34 & 75.36 & 80.92 & 84.67 & 57.91 & 64.62 & 85.09 & \textbf{81.08}\\
\hline
\text{GAU-}\alpha & \textbf{61.41} & 57.76 & \textbf{74.17} & \textbf{81.82} & 75.86 & 79.93 & \textbf{85.67} & \textbf{58.09} & \textbf{68.24} & \textbf{87.91} & 80.01\\
\hline
\end{array}}$$

所有的模型都是Base版，上表显示的是CLUE任务上验证集上的结果，大家的运行方式和比较都是公平的，作为一个相对比较来说是合理的。另外，这里的RoFormerV2<sup>\*</sup>并非[《RoFormerV2：自然语言理解的极限探索》](<https://spaces.ac.cn/archives/8998>)中的多任务版本，而是仅仅进行了MLM预训练的版本（该版本没开源），这样对比是因为GAU\-α也仅仅进行了MLM预训练。

从表中可以看出，除了WSC这个数据量极少的“异类”外，GAU\-α在多数任务上都有优势，并且除了WSC外的平均成绩是最好的。其中，RoFormerV2<sup>\*</sup>与GAU\-α的比较是最为公平的，因为它们的训练脚本、训练数据、整体结构都是一样的，唯一不同就是GAU\-α是将RoFormerV2<sup>\*</sup>中的Attention\+FFN组合换成了两层GAU，两者对比充分显示出了GAU设计“更好”的特点。

此外，我们在[《RoFormerV2：自然语言理解的极限探索》](<https://spaces.ac.cn/archives/8998>)介绍过RoFormerV2对结构进行了简化，从而获得更快的速度，具有同样整体结构的GAU\-α也是如此，所以GAU\-α的速度是比表中的BERT、RoBERTa、RoFormer都要快的，但平均效果却更胜一筹。更进一步的测试显示，当序列长度超过512时，GAU\-α的速度开始超过同样精简过的RoFormerV2，并且显存占用更低，越长则对GAU\-α更有利。

## 训练

现在介绍一下模型的训练细节，完整的代码已经开源到Github中，如有疑惑可以对照着代码来读。

<strong>模型架构</strong>： GAU\-α就是将RoFormerV2的Attention\+FFN换成了两层GAU，在[之前的文章](<https://spaces.ac.cn/archives/8934>)中我们比较过两层GAU的计算量和参数量大致相当于Attention\+FFN组合，所以这样的替换是合理的；RoFormerV2的特点是保留了Post Norm结构，去掉了所有的Bias项，并且Layer Norm换成了RMS Norm的最简单变体，在GAU\-α中也是如此。

<strong>归一化</strong>： 在[《听说Attention与Softmax更配哦～》](<https://spaces.ac.cn/archives/9019>)中我们讨论过Attention的归一化问题，GAU\-α的Attention归一化选取了其中笔者自行提出的具有较好外推能力的[熵不变性Softmax](<https://spaces.ac.cn/archives/8823>)（在bert4keras中暂称为softmax\_plus）。

<strong>训练方式</strong>： 在初始化方面笔者按照[《训练1000层的Transformer究竟有什么困难？》](<https://spaces.ac.cn/archives/8978>)进行了调整，因此无须Wamrup就可以直接训练，优化器用的是LAMB，学习率分段线性衰减；预训练任务用的是全词MLM，分词工具用百度的LAC，这些跟RoFormerV2都是对齐的。

好像值得一提的也就这么多了，确实没进行多大的改变。除了在归一化方式上花了点时间进行测试，其他方面也没多费时间，直接训练就得到了不错的效果。

## 小结

GAU是笔者认为的“目前最有潜力的下一代Attention设计”，本文分享了GAU的一些训练经验，并开源了一个尝鲜版“GAU\-α”。
