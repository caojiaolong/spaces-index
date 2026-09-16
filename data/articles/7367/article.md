# 节省显存的重计算技巧也有了Keras版了

> 作者：苏剑林 · 科学空间 · 2020-04-29
>
> 原文：<https://spaces.ac.cn/archives/7367>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

不少读者最近可能留意到了公众号文章[《BERT重计算：用22\.5%的训练时间节省5倍的显存开销（附代码）》](<https://mp.weixin.qq.com/s/CmIVwGFqrSD0wcSN_hgH1A>)，里边介绍了一个叫做“重计算”的技巧，简单来说就是用来省显存的方法，让平均训练速度慢一点，但batch\_size可以增大好几倍。该技巧首先发布于论文[《Training Deep Nets with Sublinear Memory Cost》](<https://papers.cool/arxiv/1604.06174>)，其实在2016年就已经提出了，只不过似乎还没有特别流行起来。

## 探索

公众号文章提到该技巧在pytorch和paddlepaddle都有原生实现了，但tensorflow还没有。但事实上从tensorflow 1\.8开始，tensorflow就已经自带了该功能了，当时被列入了<code>tf.contrib</code>这个子库中，而从tensorflow 1\.15开始，它就被内置为tensorflow的主函数之一，那就是<code>tf.recompute_grad</code>。

找到<code>tf.recompute_grad</code>之后，笔者就琢磨了一下它的用法，经过一番折腾，最终居然真的成功地用起来了，居然成功地让<code>batch_size</code>从48增加到了144！然而，在继续整理测试的过程中，发现这玩意居然在tensorflow 2\.x是失效的\.\.\.于是再折腾了两天，查找了各种资料并反复调试，最终算是成功地补充了这一缺陷。

最后是笔者自己的开源实现：

> <strong>Github地址：[https://github\.com/bojone/keras\_recompute](<https://github.com/bojone/keras_recompute>)</strong>

该实现已经内置在[bert4keras](<https://github.com/bojone/bert4keras>)中，使用bert4keras的读者可以升级到最新版本（0\.7\.5\+）来测试该功能。

## 使用

笔者的实现也命名为<code>recompute_grad</code>，它是一个装饰器，用于自定义Keras层的<code>call</code>函数，比如



<pre><code class="language-python">from recompute import recompute_grad

class MyLayer(Layer):
    @recompute_grad
    def call(self, inputs):
        return inputs * 2
</code></pre>



对于已经存在的层，可以通过继承的方式来装饰：



<pre><code class="language-python">from recompute import recompute_grad

class MyDense(Dense):
    @recompute_grad
    def call(self, inputs):
        return super(MyDense, self).call(inputs)
</code></pre>



自定义好层之后，在代码中嵌入自定义层，然后在执行代码之前，加入环境变量<code>RECOMPUTE=1</code>来启用重计算。

注意：不是在总模型里插入了<code>@recompute_grad</code>，就能达到省内存的目的，而是要在每个层都插入<code>@recompute_grad</code>才能更好地省显存。简单来说，就是插入的<code>@recompute_grad</code>越多，就省显存。具体原因请仔细理解重计算的原理。

## 效果

bert4keras 0\.7\.5\+已经内置了重计算，直接传入环境变量<code>RECOMPUTE=1</code>就会启用重计算，读者可以自行尝试，大概的效果是：

> 1、在BERT Base版本下，batch\_size可以增大为原来的3倍左右；
> 
> 
> 
> 2、在BERT Large版本下，batch\_size可以增大为原来的4倍左右；
> 
> 
> 
> 3、平均每个样本的训练时间大约增加25%；
> 
> 
> 
> 4、理论上，层数越多，batch\_size可以增大的倍数越大。

## 环境

在下面的环境下测试通过：

> tensorflow 1\.14 \+ keras 2\.3\.1
> 
> 
> 
> tensorflow 1\.15 \+ keras 2\.3\.1
> 
> 
> 
> tensorflow 2\.0 \+ keras 2\.3\.1
> 
> 
> 
> tensorflow 2\.1 \+ keras 2\.3\.1
> 
> 
> 
> tensorflow 2\.0 \+ 自带tf\.keras
> 
> 
> 
> tensorflow 2\.1 \+ 自带tf\.keras

确认不支持的环境：

> tensorflow 1\.x \+ 自带tf\.keras

欢迎报告更多的测试结果。

顺便说一下，<strong>强烈建议用keras 2\.3\.1配合tensorflow 1\.x/2\.x来跑，强烈不建议使用tensorflow 2\.x自带的tf\.keras来跑</strong>。

## 参考

最后，笔者的实现主要参考自如下两个源码，在此表示感谢。

> [https://github\.com/davisyoshida/tf2\-gradient\-checkpointing](<https://github.com/davisyoshida/tf2-gradient-checkpointing>)
> 
> 
> 
> [https://github\.com/tensorflow/tensorflow/blob/v2\.1\.0/tensorflow/python/ops/custom\_gradient\.py\#L454\-L499](<https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/custom_gradient.py#L454-L499>)
