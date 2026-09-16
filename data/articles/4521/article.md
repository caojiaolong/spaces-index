# 【备忘】谈谈dropout

> 作者：苏剑林 · 科学空间 · 2017-08-08
>
> 原文：<https://spaces.ac.cn/archives/4521>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

其实这只是一篇备忘\.\.\.

dropout是深度学习中防止过拟合的一项有效措施，当然，就其思想而言，dropout其实也不仅仅可以用在深度学习中，还可以用在传统的机器学习方法中，只不过在深度学习的神经网络框架下，dropout显得更为自然罢了。

## 做了什么

dropout是怎么操作的？一般来做，对于输入的张量$x$，dropout就是将部分元素置零，然后将置零后的结果做一个尺度变换。具体来说，以Keras的Dropout(0\.6)(x)为例，实际上等价于numpy做的这件事情



<pre><code class="language-python">import numpy as np

x = np.random.random((10,100)) #模拟一个batch_size=10、维度为100的输入
def Dropout(x, drop_proba):
    return x*np.random.choice(
                              [0,1], 
                              x.shape,  
                              p=[drop_proba,1-drop_proba]
                             )/(1.-drop_proba)

print Dropout(x, 0.6)
</code></pre>



也就是说将60%的元素置0，然后将剩下的40%元素，放大为原来的1/40%=2\.5倍。要提醒的是，Keras中Dropout(0\.6)(x)的0\.6表示丢弃比例，而如果用tensorflow，tf\.nn\.dropout(x, 0\.6)中的0\.6则表示保留比例。具体什么含义，需要具体框架具体分析（当然，如果dropout的比例是0\.5，那就可以不用管了^\_^）

## 有什么用

一般我们会将dropout理解为“一种低成本的集成策略”，这是对的。具体过程大概可以这样理解。

经过上述置零操作后，我们可以认为0的那部分是被丢弃了，丢失了一部分信息。然而虽然信息丢失了，但生活还得继续呀，不对，是训练还得继续，所以就逼着模型用剩下的信息去拟合目标了。然而每次dropout是随机的，我们就不能侧重于某些节点了。所以总的来说就是——每次逼着模型用少量的特征学习，每次被学习的特征有不同，那么就是说，每个特征都应该对模型的预测有所贡献（而不是侧重部分特征，导致过拟合）。

最后预测的时候，就不dropout了，所以就等价于所有局部特征的平均（这次终于用上所有的信息了），理论上效果就变好了，过拟合也不严重了（因为风险平摊到了每个特征而不是部分特征上面）。

## 更灵活些

有时候我们希望对dropout的方式做一些约束，比如，<u>同一个batch内所有样本我想要同样的方式dropout，而不是第一个样本只dropout了第一个节点，第二个样本则只dropout第二个节点；又比如，用LSTM做文本分类时，我希望按照词向量来dropout，也就是每次要不把整个词向量都dropout掉，要不就整个词向量都保留，而不是只dropout掉一个词向量的某些节点；再再比如，用CNN多RGB图像分类时，我希望按照通道来dropout，每幅图像要dropout掉R、G、B三通道的任意一个通道（有点像颜色变换，或者RGB扰动），而不是只dropout掉图像上的某些像素（这个等价于图像加噪音）</u>。

为了实现这些需求，需要用到dropout中的<strong>noise\_shape参数</strong>，这个参数在Keras的Dropout层以及tf\.nn\.dropout都有（我只会这两个框架），两者的含义是一样的。然而，网上对dropout的这个参数鲜有提及，就算说了也含糊不清。下面以tf\.nn\.dropout(x, 0\.5, noise\_shape)为例。首先noise\_shape是一个一维张量，说白了，就是一个一维数组（可以是list或者tuple），长度必须跟x\.shape一样。而且，noise\_shape里边的元素，只能是1或者是x\.shape里边对应的元素。比如，x\.shape=(3,4,5)，那么noise\_shape就只有下面8种允许情况

> (3,4,5)、(1,4,5)、(3,1,5)、(3,4,1)、(1,1,5)、(1,4,1)、(3,1,1)、(1,1,1)

那每一种的含义是什么呢？可以这样理解：<strong>哪个轴为1，哪个轴就会被一致地dropout。</strong>比如(3,4,5)时就是普通的dropout，没任何约束，(1,4,5)就是说batch中的每个样本要按同样的方式dropout（可以理解为，加在每个样本的噪音都是一样的），如果(3,4,5)是代表(句子数,每个句子的词数,每个词向量的维度)，那么(3,4,1)就表示按照词向量来dropout（可以理解为随机跳过某些词）。

对于有些读者来说，可能用numpy的代码来理解可能会直观一些，那就是：



<pre><code class="language-python">def Dropout(x, drop_proba, noise_shape):
    return x*np.random.choice(
                              [0,1], 
                              noise_shape, 
                              p=[drop_proba,1-drop_proba]
                             )/(1.-drop_proba)
</code></pre>
