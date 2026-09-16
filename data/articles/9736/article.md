# Lion/Tiger优化器训练下的Embedding异常和对策

> 作者：苏剑林 · 科学空间 · 2023-08-28
>
> 原文：<https://spaces.ac.cn/archives/9736>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

打从在[《Tiger：一个“抠”到极致的优化器》](<https://spaces.ac.cn/archives/9512>)提出了Tiger优化器之后，Tiger就一直成为了我训练模型的“标配”优化器。最近笔者已经尝试将Tiger用到了70亿参数模型的预训练之中，前期效果看上来尚可，初步说明Tiger也是能Scale Up的。不过，在查看训练好的模型权重时，笔者发现Embedding出现了一些异常值，有些Embedding的分量达到了$\pm 100$的级别。

经过分析，笔者发现类似现象并不会在Adam中出现，这是Tiger或者Lion这种带符号函数$\text{sign}$的优化器特有的问题，对此文末提供了两种参考解决方案。本文将记录笔者的分析过程，供大家参考。

## 现象

接下来，我们的分析都以Tiger优化器为例，但分析过程和结论同样适用于Lion。

首先，笔者观察到的现象是：

> 1、部分Token的Embedding分量变成了$\pm 100$；
> 
> 
> 
> 2、还有一小部分Token的Embedding分量正在趋于$\pm 100$；
> 
> 
> 
> 3、这些token看上去都是相当低频的token；
> 
> 
> 
> 4、整个Embedding矩阵的最大值就是100，最小值就是\-100；
> 
> 
> 
> 5、除Embedding外，其他权重没有这个问题；
> 
> 
> 
> 6、模型的总体表现（比如训练Loss、生成测试）都正常。

可能有读者想问，既然模型表现正常，那还管它干嘛呢？在笔者看来，至少有两方面的原因：第一，如果后面想要微调，有可能某些低频Token重新变得高频，如果这些Token的Embedding太糟糕，那么微调也救不回来；第二，有些能力在Loss体现不出来，比如中英的预训练模型，通常因为训练语料夹杂着非常少的多语种语料，就体现出一定的多语种能力，很明显这种能力依赖于低频Token的Embedding质量，如果被优化器所连累而失去这种能力，就“亏大发”了。

当然，不管是什么优化器，都有可能训着训着就把模型训崩了，这并不让人意外，很多时候也难以深究。但这里最耐人寻味的地方是“崩”得这么有规律——刚好是整齐的$\pm 100$，这不能不让笔者想要进一步找出它背后的原因。

## 思考

根据以上观察结果，初步可以得出这些异常值只出现在“低频Token的Embedding”上，这让笔者不禁联想到[《Keras实现两个优化器：Lookahead和LazyOptimizer》](<https://spaces.ac.cn/archives/6869#LazyOptimizer>)讨论过的带动量的优化器会导致Embedding层过度优化问题。

具体来说，只要一个token出现过，那么该token的Embedding对应的动量就被更新为非零（假设该token的梯度不会正好是零），于是在后面的更新中，即便当前样本没有出现过该token（梯度为零），但该token的Embedding依然会被更新（动量不为零），这就是低频token的过度优化问题。这个问题会出现在所有带动量的优化器中，包括Adam和Tiger，不过在Adam中，这可能不会有明显感知，因为Adam的更新量跟动量成正比，如果一个token长期不重复出现，那么动量就会指数下降，所以很快就趋于零了，换句话说更新量也很快趋于零，即过度更新很快就会消失。

然而，在Tiger中情况有点不一样。Tiger的更新量是跟动量的符号函数$\text{sign}(\boldsymbol{m}_t)$成正比，尽管动量$\boldsymbol{m}_t$会指数下降，但符号函数不会，在$\boldsymbol{m}_t$由于舍入误差变成0之前，$\text{sign}(\boldsymbol{m}_t)$都保持$\pm 1$的值不变，也就是更新量一直都是常数，所以Tiger的Embedding过度更新问题更加严重。“屋漏偏逢连夜雨”的是，一个token的Embedding由于过度更新偏向了某个方向之后，它的梯度可能会适应并助长这种变化，也就是说下一次它出现时的梯度是同一方向而不是相反方向，这就导致了它长期在同一方向上过度更新，最终导致了异常值。

## 计算

那么异常值为什么偏偏是$\pm 100$呢？这就要邀请权重衰减登场了。Tiger总的优化公式是：
\begin{equation}\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \left[\text{sign}(\boldsymbol{m}_t) + \lambda \boldsymbol{\theta}_{t-1}\right]\end{equation}
也就是说，除了动量的符号函数外，还有一个权重衰减项。在文章开头提到的异常实验中，衰减率$\lambda$设为了0\.01。

不难发现，如果$\text{sign}(\boldsymbol{m}_t)$长期为常量，那么上述迭代公式将会有一个平衡点，它出现在$\text{sign}(\boldsymbol{m}_t) + \lambda \boldsymbol{\theta}^*=\boldsymbol{0}$时，即
\begin{equation}\boldsymbol{\theta}^* = -\frac{\text{sign}(\boldsymbol{m}_t)}{\lambda}\end{equation}
这正好对应一个元素是$\pm 100$的向量，这就解释了异常值为$\pm 100$的结果。如果有兴趣，读者还可以假设$\eta_t$也是常数，那么可以直接求出$\boldsymbol{\theta}_t$的解析式，从而进一步分析收敛速度等。这里笔者就不继续展开了。

## 对策

既然问题出现在对低频Token的Embedding的过度更新，那么一个自然的解决方案就是像[《Keras实现两个优化器：Lookahead和LazyOptimizer》](<https://spaces.ac.cn/archives/6869#LazyOptimizer>)所提的那样，将Embedding的更新Lazy化，即只有当Token出现过的时候，才更新相应的Embedding。如果能获取到所有的输入Token Ids的集合，那么直接只更新这些Token的Embedding即可，如果不能，我们可以通过判断Embedding的梯度模长是否非零，来判断该Embedding是否需要被更新。

另一方面，从更一般的视角看，该问题是Lion/Tiger优化器对于梯度稀疏的参数的共同缺陷，包括但不限于Embedding层。于是，解决问题的另一个思路是将Embedding的梯度变得不再稀疏，为此我们可以考虑Tied Embeddings，即输入和输出的Embedding共享，这样由于输出端重用了整个Embedding矩阵，因此整个Embedding矩阵都有非零梯度，从而让$\boldsymbol{m}_t$不至于长期为常量。当然Tied Embedding可能会带来另外的一些问题，相应的解决方案可以参考[《语言模型输出端共享Embedding的重新探索》](<https://spaces.ac.cn/archives/9698>)。在笔者的实验中，使用将模型特征的channels对半交换的Tied Embedding，能解决以上问题，并且效果似乎比Untied Embedding还要好一点。

最后，笔者也就此问题请教了Lion优化器的作者，得到的回复是他们之前也留意到了这个问题，他们的解决方案是混合优化器，比如Embedding层就用Adam，其他层才用Lion/Tiger。呃，这个解决方案是笔者没想到的，感觉不是特别优雅，但也确实能解决，读者自行选择就好。

## 小结

本文介绍了Lion/Tiger优化器训练下的Embedding异常现象，并分析了背后的原因，最后给出了参考的解决方案。
