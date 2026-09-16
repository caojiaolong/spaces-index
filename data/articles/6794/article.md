# 用时间换取效果：Keras梯度累积优化器

> 作者：苏剑林 · 科学空间 · 2019-07-08
>
> 原文：<https://spaces.ac.cn/archives/6794>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

> 现在Keras中你也可以用小的batch size实现大batch size的效果了——只要你愿意花$n$倍的时间，可以达到$n$倍batch size的效果，而不需要增加显存。

Github地址：[https://github\.com/bojone/accum\_optimizer\_for\_keras](<https://github.com/bojone/accum_optimizer_for_keras>)

## 扯淡

在一两年之前，做NLP任务都不用怎么担心OOM问题，因为相比CV领域的模型，其实大多数NLP模型都是很浅的，极少会显存不足。幸运或者不幸的是，Bert出世了，然后火了。Bert及其后来者们（GPT\-2、XLNET等）都是以足够庞大的Transformer模型为基础，通过足够多的语料预训练模型，然后通过fine tune的方式来完成特定的NLP任务。

即使你很不想用Bert，但现在的实际情况是：<u>你精心设计的复杂的模型，效果可能还不如简单地fine tune一下Bert好</u>。所以不管怎样，为了跟上时代，总得需要学习一下[Bert的fine tune](<https://spaces.ac.cn/archives/6736>)。问题是“不学不知道，一学吓一跳”，只要任务稍微复杂一点，或者句子长度稍微长一点，显存就不够用了，batch size急剧下降——32？16？8？一跌再跌都是有可能的。

这不难理解，Transformer基于Attention，而Attention理论上空间和时间复杂度都是$\mathcal{O}(n^2)$，虽然在算力足够强的时候，Attention由于其并行性还是可以表现得足够快，但是显存占用量是省不了了，$\mathcal{O}(n^2)$意味着当你句子长度变成原来的2倍时，显存占用基本上就需要原来的4倍，这个增长比例肯定就容易OOM了～

而更不幸的消息是，大家都在fine tune预训练Bert的情况下，你batch\_size=8可能比别人batch\_size=80低好几个千分点甚至是几个百分点，显然这对于要刷榜的读者是很难受的。难道除了加显卡就没有别的办法了吗？

## 正事

有！通过梯度缓存和累积的方式，用时间来换取空间，最终训练效果等效于更大的batch size。因此，只要你跑得起batch\_size=1，只要你愿意花$n$倍的时间，就可以跑出$n$倍的batch size了。

梯度累积的思路，在之前的文章[《“让Keras更酷一些！”：小众的自定义优化器》](<https://spaces.ac.cn/archives/5879>)已经介绍了，当时称之为“软batch（soft batch）”，本文还是沿着主流的叫法称之为“梯度累积（accumulate gradients）”好了。所谓梯度累积，其实很简单，我们梯度下降所用的梯度，实际上是多个样本算出来的梯度的平均值，以batch\_size=128为例，你可以一次性算出128个样本的梯度然后平均，我也可以每次算16个样本的平均梯度，然后缓存累加起来，算够了8次之后，然后把总梯度除以8，然后才执行参数更新。当然，必须累积到了8次之后，用8次的平均梯度才去更新参数，不能每算16个就去更新一次，不然就是batch\_size=16了。

刚才说了，在之前的文章的那个写法是有误的，因为用到了



<pre><code class="language-python">K.switch(cond, K.update(p, new_p), p)</code></pre>



来控制更新，但事实上这个写法不能控制更新，因为<code>K.switch</code>只保证结果的选择性，不保证执行的选择性，事实上它等价于



<pre><code class="language-python">cond * K.update(p, new_p) + (1 - cond) * p</code></pre>



也就是说不管<code>cond</code>如何，两个分支都是被执行了。事实上Keras或Tensorflow“几乎”不存在只执行一个分支的条件写法（说“几乎”是因为在一些比较苛刻的条件下可以做到），所以此路不通。

不能这样写的话，那只能在“更新量”上面下功夫，如前面所言，每次算16个样本的梯度，每次都更新参数，只不过8次中有7次的更新量是0，而只有1次是真正的梯度下降更新。很幸运的是，这种写法还可以无缝地接入到现有的Keras优化器中，使得我们不需要重写优化器！详细写法请看：

> [https://github\.com/bojone/accum\_optimizer\_for\_keras](<https://github.com/bojone/accum_optimizer_for_keras/blob/master/accum_optimizer.py>)

具体的写法无外乎就是一些移花接木的编程技巧，真正有技术含量的部分不多。关于写法本身不再细讲，如果有疑问欢迎讨论区讨论。

<strong>（注：这个优化器的修改，使得小batch size能起到大batch size的效果，前提是模型不包含Batch Normalization，因为Batch Normalization在梯度下降的时候必须用整个batch的均值方差。所以如果你的网络用到了Batch Normalization，想要准确达到大batch size的效果，目前唯一的方法就是加显存/加显卡。）</strong>

## 实验

至于用法则很简单：



<pre><code class="language-python">opt = AccumOptimizer(Adam(), 10) # 10是累积步数
model.compile(loss='mse', optimizer=opt)
model.fit(x_train, y_train, epochs=10, batch_size=10)</code></pre>



这样一来就等价于batch\_size=100的Adam优化器了，代价就是你每个epoch的速度会更慢了（因为batch size更小了），好处是你只需要用到batch\_size=10的显存量。

可能读者想问的一个问题是：你怎么证明你的写法生效了？也就是说你怎么证明你的结果确实是batch\_size=100而不是batch\_size=10？为此，我做了个比较极端的实验，代码在这里：
[https://github\.com/bojone/accum\_optimizer\_for\_keras/blob/master/mnist\_mlp\_example\.py](<https://github.com/bojone/accum_optimizer_for_keras/blob/master/mnist_mlp_example.py>)

代码很简单，就是用多层MLP做MNIST分类，用Adam优化器，<code>fit</code>的时候batch\_size=1。优化器有两个选择，第一个是直接<code>Adam()</code>，第二个是<code>AccumOptimizer(Adam(), 100)</code>：

> 如果是直接Adam()，那loss一直在0\.4上下徘徊，后面loss越来越大了（训练集都这样），val的准确率也没超过97%；
> 
> 
> 
> 如果是AccumOptimizer(Adam(), 100)，那么训练集的loss越来越低，最终降到0\.02左右，val的最高准确率有98%\+；
> 
> 
> 
> 最后我比较了直接Adam()但是batch\_size=100的结果，发现跟AccumOptimizer(Adam(), 100)但是batch\_size=1时表现差不多。

这个结果足以表明写法生效了，达到了预期的目的。如果这还不够说服力，我再提供一个训练结果作为参考：在某个Bert的fine tune实验中，直接用Adam()加batch\_size=12，我跑到了70\.33%的准确率；我用AccumOptimizer(Adam(), 10)加batch\_size=12（预期等效batch size是120），我跑到了71%的准确率，提高了0\.7%，如果你在刷榜，那么这0\.7%可能是决定性的。

## 结论

终于把梯度累积（软batch）正式地实现了，以后用Bert的时候，也可以考虑用大batch\_size了哈～
