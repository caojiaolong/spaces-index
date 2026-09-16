# “让Keras更酷一些！”：分层的学习率和自由的梯度

> 作者：苏剑林 · 科学空间 · 2019-03-10
>
> 原文：<https://spaces.ac.cn/archives/6418>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

高举“[让Keras更酷一些！](<https://spaces.ac.cn/search/%E8%AE%A9Keras%E6%9B%B4%E9%85%B7%E4%B8%80%E4%BA%9B/>)”大旗，让Keras无限可能～

今天我们会用Keras做到两件很重要的事情：分层设置学习率和灵活操作梯度。

首先是<strong><u>分层设置学习率</u></strong>，这个用途很明显，比如我们在fine tune已有模型的时候，有些时候我们会固定一些层，但有时候我们又不想固定它，而是想要它以比其他层更低的学习率去更新，这个需求就是分层设置学习率了。对于在Keras中分层设置学习率，网上也有一定的探讨，结论都是要通过重写优化器来实现。显然这种方法不论在实现上还是使用上都不友好。

然后是<strong><u>操作梯度</u></strong>。操作梯度一个最直接的例子是梯度裁剪，也就是把梯度控制在某个范围内，Keras内置了这个方法。但是Keras内置的是全局的梯度裁剪，假如我要给每个梯度设置不同的裁剪方式呢？甚至我有其他的操作梯度的思路，那要怎么实施呢？不会又是重写优化器吧？

本文就来为上述问题给出尽可能简单的解决方案。

## 分层的学习率

对于分层设置学习率这个事情，重写优化器当然是可行的，但是太麻烦。如果要寻求更简单的方案，我们需要一些数学知识来指导我们怎么进行。

### 参数变换下的优化

首先我们考虑梯度下降的更新公式：
\begin{equation}\boldsymbol{\theta}_{n+1}=\boldsymbol{\theta}_{n}-\alpha \frac{\partial L(\boldsymbol{\theta}_{n})}{\partial \boldsymbol{\theta}_n}\label{eq:sgd-1}\end{equation}
其中$L$是带参数$\boldsymbol{\theta}$的loss函数，$\alpha$是学习率，$\frac{\partial L(\boldsymbol{\theta}_{n})}{\partial \boldsymbol{\theta}_n}$是梯度，有时候我们也写成$\nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_{n})$。记号是很随意的，关键是理解它的含义～

然后我们考虑变换$\boldsymbol{\theta}=\lambda \boldsymbol{\phi}$，其中$\lambda$是一个固定的标量，$\boldsymbol{\phi}$也是参数。现在我们来优化$\boldsymbol{\phi}$，相应的更新公式为：
\begin{equation}\begin{aligned}\boldsymbol{\phi}_{n+1}=&\boldsymbol{\phi}_{n}-\alpha \frac{\partial L(\lambda\boldsymbol{\phi}_{n})}{\partial \boldsymbol{\phi}_n}\\
=&\boldsymbol{\phi}_{n}-\alpha \frac{\partial L(\boldsymbol{\theta}_{n})}{\partial \boldsymbol{\theta}_n}\frac{\partial \boldsymbol{\theta}_{n}}{\partial \boldsymbol{\phi}_n}\\
=&\boldsymbol{\phi}_{n}-\lambda\alpha \frac{\partial L(\boldsymbol{\theta}_{n})}{\partial \boldsymbol{\theta}_n}\end{aligned}\end{equation}
其中第二个等号其实就是链式法则。现在我们在两边乘上$\lambda$，得到
\begin{equation}\lambda\boldsymbol{\phi}_{n+1}=\lambda\boldsymbol{\phi}_{n}-\lambda^2\alpha \frac{\partial L(\boldsymbol{\theta}_{n})}{\partial \boldsymbol{\theta}_n}\quad\Rightarrow\quad\boldsymbol{\theta}_{n+1}=\boldsymbol{\theta}_{n}-\lambda^2\alpha \frac{\partial L(\boldsymbol{\theta}_{n})}{\partial \boldsymbol{\theta}_n}\label{eq:sgd-2}\end{equation}
对比$\eqref{eq:sgd-1}$和$\eqref{eq:sgd-2}$，大家能明白我想说什么了吧：

> 在SGD优化器中，如果做参数变换$\boldsymbol{\theta}=\lambda \boldsymbol{\phi}$，那么等价的结果是学习率从$\alpha$变成了$\lambda^2\alpha$。

不过，在自适应学习率优化器（比如RMSprop、Adam等），情况有点不一样，因为自适应学习率使用梯度（作为分母）来调整了学习率，抵消了一个$\lambda$，从而（请有兴趣的读者自己推导一下）

> 在RMSprop、Adam等自适应学习率优化器中，如果做参数变换$\boldsymbol{\theta}=\lambda \boldsymbol{\phi}$，那么等价的结果是学习率从$\alpha$变成了$\lambda\alpha$。

### 移花接木调整学习率

有了前面这两个结论，我们就只需要想办法实现参数变换，而不需要自己重写优化器，来实现逐层设置学习率了。

实现参数变换的方法也不难，之前我们在[《 “让Keras更酷一些！”：随意的输出和灵活的归一化》一文](<https://spaces.ac.cn/archives/6311#%E7%81%B5%E6%B4%BB%E7%9A%84%E6%9D%83%E9%87%8D%E5%BD%92%E4%B8%80%E5%8C%96>)讨论权重归一化的时候已经讲过方法了。因为Keras在构建一个层的时候，实际上是分开了<code>build</code>和<code>call</code>两个步骤，我们可以在<code>build</code>之后插一些操作，然后再调用<code>call</code>就行了。

下面是一个封装好的实现：



<pre><code class="language-python">import keras.backend as K

class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)</code></pre>



使用示例：



<pre><code class="language-python">x_in = Input(shape=(None,))
x = x_in

# 默认情况下是x = Embedding(100, 1000, weights=[word_vecs])(x)
# 下面这一句表示：后面将会用自适应学习率优化器，并且Embedding层以总体的十分之一的学习率更新。
# word_vecs是预训练好的词向量
x = SetLearningRate(Embedding(100, 1000, weights=[word_vecs]), 0.1, True)(x)

# 后面部分自己想象了～
x = LSTM(100)(x)

model = Model(x_in, x)
model.compile(loss='mse', optimizer='adam') # 用自适应学习率优化器优化</code></pre>



几个注意事项：

> 1、目前这种方式，只能用于自己动手写代码来构建模型的时候插入，无法对建立好的模型进行操作。
> 
> 
> 
> 2、如果有预训练权重，有两种加载方法。第一种是像刚才的使用示例一样，在定义层的时候通过weights参数传入；第二种方法是建立好模型后（已经在相应的地方插入好SetLearningRate），用model\.set\_weights(weights)来赋值，其中weights是“在SetLearningRate的位置已经被除以了$\lambda$或$\sqrt{\lambda}$的原来模型的预训练权重”。
> 
> 
> 
> 3、加载预训练权重的第二种方法看起来有点不知所云，但如果你已经理解了这一节的原理，那么应该能知道我在说什么。因为设置学习率是通过weight \* lamb来实现的，所以weight的初始化要变为weight / lamb。
> 
> 
> 
> 4、这个操作基本上不可逆，比如你一开始设置了Embedding层以总体的1/10比例的学习率来更新，那么<strong><u>很难</u></strong>在这个基础上，再将它改为1/5或者其他比例。（当然，如果你真的彻底搞懂了这一节的原理，并且也弄懂了加载预训练权重的第二种方法，那么还是有办法的，那时候相信你也能搞出来）。
> 
> 
> 
> 5、这种做法有以上限制，是因为我们不想通过修改或者重写优化器的方式来实现这个功能。如果你决定要自己修改优化器，请参考[《“让Keras更酷一些！”：小众的自定义优化器》](<https://spaces.ac.cn/archives/5879>)。

## 自由的梯度操作

在这部分内容中，我们将学习对梯度的更为自由的控制。这部分内容涉及到对优化器的修改，但不需要完全重写优化器。

### Keras优化器的结构

要修改优化器，必须先要了解Keras优化器的结构。在[《“让Keras更酷一些！”：小众的自定义优化器》](<https://spaces.ac.cn/archives/5879>)一文我们已经初步看过了，现在我们重新看一遍。

Keras优化器的代码在
[https://github\.com/keras\-team/keras/blob/master/keras/optimizers\.py](<https://github.com/keras-team/keras/blob/master/keras/optimizers.py>)

随便观察一个优化器，就会发现你要自定义一个优化器，只需要继承<code>Optimizer</code>类，然后定义<code>get_updates</code>方法。但本文我们不想做新的优化器，只是想要对梯度有所控制。可以看到，梯度的获取其实是在父类<code>Optimizer</code>的<code>get_gradients</code>方法中：



<pre><code class="language-python">    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm') and self.clipnorm &gt; 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue &gt; 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads</code></pre>



其中方法中的第一句就是获取原始梯度的，后面则提供了两种梯度裁剪方法。不难想到，只需要重写优化器的<code>get_gradients</code>方法，就可以实现对梯度的任意操作了，而且这个操作不影响优化器的更新步骤（即不影响<code>get_updates</code>方法）。

### 处处皆对象：覆盖即可

怎么能做到只修改<code>get_gradients</code>方法呢？这得益于Python的哲学——“处处皆对象”。Python是一门面向对象的编程语言，Python中几乎你能碰到的一切变量都是一个对象。我们说<code>get_gradients</code>是优化器的一个方法，也可以说<code>get_gradients</code>的一个属性（对象），既然是属性，直接覆盖赋值即可。

我们来举一个最粗暴的例子（恶作剧）：



<pre><code class="language-python">def our_get_gradients(loss, params):
    return [K.zeros_like(p) for p in params]

adam_opt = Adam(1e-3)
adam_opt.get_gradients = our_get_gradients

model.compile(loss='categorical_crossentropy',
              optimizer=adam_opt)</code></pre>



其实这个例子很无聊，就是把所有梯度置零了（然后你怎么优化它都不动了\.\.\.），但这个恶作剧例子已经足够有代表性了——你可以将所有梯度置零，你也可以将梯度做任意你喜欢的操作。比如将梯度按照$l_1$范数而非$l_2$范数裁剪，又或者做其他调整～

假如我只想操作部分层的梯度怎么办？那也简单，你在定义层的时候需要起一个能区分的名字，然后根据<code>params</code>的名字做不同的操作即可。都到这一步了，我相信基本是“一法通，万法皆通”的了。

## 飘逸的Keras

也许在很多人眼中，Keras就是一个好用但是封装得很“死”的高层框架，但在我眼里，我只看到了它无限的灵活性～

那是一个无懈可击的封装。
