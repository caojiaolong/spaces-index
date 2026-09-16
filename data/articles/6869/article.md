# Keras实现两个优化器：Lookahead和LazyOptimizer

> 作者：苏剑林 · 科学空间 · 2019-07-30
>
> 原文：<https://spaces.ac.cn/archives/6869>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近用Keras实现了两个优化器，也算是有点实现技巧，遂放在一起写篇文章简介一下（如果只有一个的话我就不写了）。这两个优化器的名字都挺有意思的，一个是look ahead（往前看？），一个是lazy（偷懒？），难道是两个完全不同的优化思路么？非也非也～只能说发明者们起名字太有创意了。

## Lookahead

首先登场的是Lookahead优化器，它源于论文[《Lookahead Optimizer: k steps forward, 1 step back》](<https://papers.cool/arxiv/1907.08610>)，是最近才提出来的优化器，有意思的是大牛Hinton和Adam的作者之一Jimmy Ba也出现在了论文作者列表当中，有这两个大神加持，这个优化器的出现便吸引了不少目光。

Lookahead的思路很朴素，准确来说它并不是一个优化器，而是一个使用现有优化器的方案。简单来说它就是下面三个步骤的循环执行：

> 1、备份模型现有的权重$\theta$；
> 
> 
> 
> 2、从$\theta$出发，用指定优化器更新$k$步，得到新权重$\tilde{\theta}$；
> 
> 
> 
> 3、更新模型权重为$\theta \leftarrow \theta + \alpha\left(\tilde{\theta} - \theta\right)$。

下面则是我的Keras实现，写法在之前的[《“让Keras更酷一些！”：小众的自定义优化器》](<https://spaces.ac.cn/archives/5879>)一文中就提到过了，属于一种“侵入式”的写法：

> [https://github\.com/bojone/keras\_lookahead](<https://github.com/bojone/keras_lookahead>)

用法就很简单了：



<pre><code class="language-python">model.compile(optimizer=Adam(1e-3), loss='mse') # 用你想用的优化器
lookahead = Lookahead(k=5, alpha=0.5) # 初始化Lookahead
lookahead.inject(model) # 插入到模型中</code></pre>



至于效果，原论文中做了不少实验，有些有轻微提高（cifar10和cifar100那两个），有些提升还比较明显（LSTM做语言模型那个），我自己简单实验了一下，结果是没什么变化。我一直觉得优化器是一个很玄乎的存在，有时候非得SGD才能达到最优效果，有时候又非得Adam才能收敛得下去，总之不能指望单靠换一个优化器就能大幅度提升模型效果。Lookahead的出现，也就是让我们多一种选择罢了，训练时间充足的读者，多去尝试一下就好。

附：[《机器之心的Lookahead的介绍》](<https://mp.weixin.qq.com/s/3J-28xd0pyToSy8zzKs1RA>)

## LazyOptimizer

LazyOptimizer优化器基本上就是为NLP准备的，当然更准确来说是为Embedding层准备的。

LazyOptimizer指出所有带动量的优化器（自然也就包括Adam以及带动量的SGD）都存在一个问题：当前batch中没被采样到的词，依然会使用历史动量来更新，这可能导致Embedding层过拟合（参考[知乎讨论](<https://www.zhihu.com/question/265357659/answer/580469438>)）。具体来说，当一个词的被采样过后，它的Embedding的梯度不为0，这个梯度也会被记录在动量中，实际更新是用动量去更新的；在后面的batch中，假如该词没有被采样到，它的Embedding的梯度为0，但是它的动量并不为0，所以该词还是被更新了。这样一来就算没有被反复采样的词，对应的Embedding也被反复更新了，就导致了过拟合。

所以，一个改进的方案是只有当该词被采样过才更新，这就是LazyOptimizer的基本原理了。

在实现上，我们要如何判断一个词有没有被采样过呢？当然终极方法肯定是传入被采样过的词的index了，但这使用上不够友好。我这里使用了一个近似的方法：判断该词的Embedding对应的梯度是否为0，如果为0意味着它“很可能”在当前batch没有被采样到。背后的原理在于，如果它没有被采样到，那么梯度一定为0，如果它被采样到了，那么梯度为0的概率是非常小的，毕竟那么多分量，同时为0的可能性很小，所以这样实现也够用了。

我的Keras实现位于：

> [https://github\.com/bojone/keras\_lazyoptimizer](<https://github.com/bojone/keras_lazyoptimizer>)

这个用法也很简单，就是包装一个带动量的优化器，传入所有Embedding层，使得它成为一个新的Lazy版的优化器：



<pre><code class="language-python">model.compile(
    loss='mse',
    optimizer=LazyOptimizer(Adam(1e-3), embedding_layers)
)</code></pre>



Github中还带有一个[IMDB的例子](<https://github.com/bojone/keras_lazyoptimizer/blob/master/imdb_lstm_test.py>)，在这个例子中，如果直接用<code>Adam(1e-3)</code>做优化器，那么验证准确率最高只有83\.7%左右，而如果用<code>LazyOptimizer(Adam(1e-3), embedding_layers)</code>，那么基本上最优验证准确率能跑到84\.9%以上，效果可见一斑。总的来说，我觉得Embedding层很大的模型（尤其是以词为单位的模型）都可以试一下，总的来说就是因为Embedding层参数量太大了，减少更新频率，让模型重点去优化其余部分。

> 注：这个LazyOptimizer和标准的LazyOptimizer有点不一样。标准的LazyOptimizer对没有采样过的词，所有相关的缓存量（比如动量等）也不去更新，但我这个实现中，就算该词没有被采样到，该词对应的所有缓存量还是会被更新的，有评测说这样做其实效果会更好些。

## 文末小结

也没啥内容，就用Keras实现了两个优化器，让用Keras的朋友可以及时尝尝鲜，或者用Keras用得更有意思一些。
