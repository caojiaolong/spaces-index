# “让Keras更酷一些！”：中间变量、权重滑动和安全生成器

> 作者：苏剑林 · 科学空间 · 2019-04-28
>
> 原文：<https://spaces.ac.cn/archives/6575>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

继续“[让Keras更酷一些](<https://spaces.ac.cn/search/%E8%AE%A9Keras%E6%9B%B4%E9%85%B7%E4%B8%80%E4%BA%9B/>)”之旅。

今天我们会用Keras实现灵活地输出任意中间变量，还有无缝地进行权重滑动平均，最后顺便介绍一下生成器的进程安全写法。

首先是<strong><u>输出中间变量</u></strong>。在自定义层时，我们可能希望查看中间变量，这些需求有些是比较容易实现的，比如查看中间某个层的输出，只需要将截止到这个层的部分模型保存为一个新模型即可，但有些需求是比较困难的，比如在使用Attention层时我们可能希望查看那个Attention矩阵的值，如果用构建新模型的方法则会非常麻烦。而本文则给出一种简单的方法，彻底满足这个需求。

接着是<strong><u>权重滑动平均</u></strong>。权重滑动平均是稳定、加速模型训练甚至提升模型效果的一种有效方法，很多大型模型（尤其是GAN）几乎都用到了权重滑动平均。一般来说权重滑动平均是作为优化器的一部分，所以一般需要重写优化器才能实现它。本文介绍一个权重滑动平均的实现，它可以无缝插入到任意Keras模型中，不需要自定义优化器。

至于<strong><u>生成器的进程安全写法</u></strong>，则是因为Keras读取生成器的时候，用到了多进程，如果生成器本身也包含了一些多进程操作，那么可能就会导致异常，所以需要解决这个这个问题。

## 输出中间变量

这一节以基本模型



<pre><code class="language-python">x_in = Input(shape=(784,))
x = x_in

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(x_in, x)
</code></pre>



为例，逐步深入地介绍如何获取Keras的中间变量。

### 作为一个新模型

假如模型训练完成后，我想要获取<code>x = Dense(256, activation='relu')(x)</code>对应的输出，那可以在定义模型的时候，先把对应的变量存起来，然后重新定义一个模型：



<pre><code class="language-python">x_in = Input(shape=(784,))
x = x_in

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
y = x
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(x_in, x)
model2 = Model(x_in, y)
</code></pre>



将<code>model</code>训练完成后，直接用<code>model2.predict</code>就可以查看对应的256维的输出了。这样做的前提是<code>y</code><u>必须是某个层的输出，不能是随意一个张量</u>。

### K\.function！

有时候我们自定义了一个比较复杂的层，比较典型的就是[Attention层](<https://github.com/bojone/attention/blob/master/attention_keras.py>)，我们希望查看层的一些中间变量，比如对应的Attention矩阵，这时候就比较麻烦了，如果想要用前面的方式，那么就要把原来的Attention层分开为两个层定义才行，因为前面已经说了，新定义一个Keras模型时输入输出都必须是Keras层的输入输出，不能是随意一个张量。这样一来，如果想要分别查看层的多个中间变量，那就要将层不断地拆开为多个层来定义，显然是不够友好的。

其实Keras提供了一个终极的解决方案：<code>K.function</code>！

介绍<code>K.function</code>之前，我们先写一个简单示例：



<pre><code class="language-python">class Normal(Layer):
    def __init__(self, **kwargs):
        super(Normal, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1,),
                                      initializer='zeros',
                                      trainable=True)
        self.built = True
    def call(self, x):
        self.x_normalized = K.l2_normalize(x, -1)
        return self.x_normalized * self.kernel


x_in = Input(shape=(784,))
x = x_in

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
normal = Normal()
x = normal(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(x_in, x)
</code></pre>



在上面的例子中，<code>Normal</code>定义了一个层，层的输出是<code>self.x_normalized * self.kernel</code>，不过我想在训练完成后获取<code>self.x_normalized</code>的值，而它是跟输入有关，并且不是一个层的输出。这样一来前面的方法就没法用了，但用<code>K.function</code>就只是一行代码：



<pre><code class="language-python">fn = K.function([x_in], [normal.x_normalized])</code></pre>



<code>K.function</code>的用法跟定义一个新模型类似，要把所有跟<code>normal.x_normalized</code>相关的输入张量都传进去，但是不要求输出是一个层的输出，允许是任意张量！返回的<code>fn</code>是一个具有函数功能的对象，所以只需要



<pre><code class="language-python">fn([x_test])</code></pre>



就可以获取到<code>x_test</code>对应的<code>x_normalized</code>了！比定义一个新模型简单通用多了～

事实上<code>K.function</code>就是Keras底层的基础函数之一，它直接封装好了后端的输入输出操作，换句话说，你用Tensorflow为后端时，<code>fn([x_test])</code>就相当于



<pre><code class="language-python">sess.run(normal.x_normalized, feed_dict={x_in: x_test})</code></pre>



了，所以<code>K.function</code>的输出允许是任意张量，因为它本来就在直接操作后端了～

## 权重滑动平均

权重滑动平均是提供训练稳定性的有效方法，通过滑动平均可以几乎零额外成本地提高解的性能。权重滑动平均一般就是指“Exponential Moving Average”，简称EMA，这是因为一般滑动平均时会使用指数衰减作为权重的比例。它已经被主流模型所接受，尤其是GAN，在很多GAN论文中我们通常会看到类似的描述：

> we use an exponential moving average with decay 0\.999 over the weight \.\.\.

这就意味着GAN模型使用了EMA。此外，普通模型也会使用，比如[《QANet: Combining Local Convolution with Global Self\-Attention for Reading Comprehension》](<https://papers.cool/arxiv/1804.09541>)就在训练过程中用了EMA，衰减率是0\.9999。

### 滑动平均的格式

滑动平均的格式其实非常简单：假设每次优化器的更新为：
\begin{equation}\boldsymbol{\theta}_{n+1} = \boldsymbol{\theta}_n - \Delta \boldsymbol{\theta}_n \end{equation}
这里的$\Delta \boldsymbol{\theta}_n$就是优化器带来的更新，优化器可以是SGD、Adam等任意一种。而滑动平均则是维护一组新的新的变量$\boldsymbol{\Theta}$：
\begin{equation}\boldsymbol{\Theta}_{n+1} = \alpha \boldsymbol{\Theta}_n + (1-\alpha) \boldsymbol{\theta}_{n+1}\end{equation}
其中$\alpha$是一个接近于1的正常数，称为“衰减率（decay rate）”。

权重滑动平均也叫Polyak averaging。注意，尽管在形式上有点相似，但它跟动量加速不一样：EMA不改变原来优化器的轨迹，即原来优化器怎么走，现在依然是同样的走法，只不过它维护一组新变量，来平均原来优化器的轨迹；而动量加速则是改变了原来优化器的轨迹。

再次强调，<strong>权重滑动平均不改变优化器的走向，只不过它降优化器的优化轨迹上的点做了平均后，作为最终的模型权重</strong>。

关于权重滑动平均的原理和效果，可以进一步参考[《从动力学角度看优化算法（四）：GAN的第三个阶段》](<https://spaces.ac.cn/archives/6583#%E6%9D%83%E9%87%8D%E6%BB%91%E5%8A%A8%E5%B9%B3%E5%9D%87>)一文。

### 巧妙的注入实现

实现EMA的要点是如何在原来优化器的基础上引入一组新的平均变量，并且在每次参数更新后执行平均变量的更新。这需要对Keras的源码及其实现逻辑有一定的了解。

在此给出的参考实现如下：



<pre><code class="language-python">class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))</code></pre>



使用方法很简单：



<pre><code class="language-python">EMAer = ExponentialMovingAverage(model) # 在模型compile之后执行
EMAer.inject() # 在模型compile之后执行

model.fit(x_train, y_train) # 训练模型</code></pre>



训练完成后：



<pre><code class="language-python">EMAer.apply_ema_weights() # 将EMA的权重应用到模型中
model.predict(x_test) # 进行预测、验证、保存等操作

EMAer.reset_old_weights() # 继续训练之前，要恢复模型旧权重。还是那句话，EMA不影响模型的优化轨迹。
model.fit(x_train, y_train) # 继续训练
</code></pre>



现在翻看实现过程，可以发现主要的一点是引入了<code>K.moving_average_update</code>操作，并且插入到<code>model.metrics_updates</code>中，在训练过程中，模型会读取并执行<code>model.metrics_updates</code>的所有算子，从而完成了滑动平均。

## 进程安全生成器

一般来说，当训练数据无法全部载入内存，或者需要动态生成训练数据时，就会用到<code>generator</code>。一般来说，Keras模型的<code>generator</code>的写法是：



<pre><code class="language-python">def data_generator():
    while True:
        x_train = something
        y_train = otherthing
        yield x_train, y_train
</code></pre>



但如果<code>someting</code>或<code>otherthing</code>里边包含了多进程操作，就可能出问题。这时候有两种解决方法，一是<code>fit_generator</code>时将设置参数<code> use_multiprocessing=False, worker=0</code>；另一种方法就是通过继承<code>keras.utils.Sequence</code>类来写生成器。

### 官方参考例子

官方对<code>keras.utils.Sequence</code>类的介绍在[这里](<https://keras.io/utils/#sequence>)。官方强调：

> <code>Sequence</code> are a safer way to do multiprocessing\. This structure guarantees that the network will only train once on each sample per epoch which is not the case with generators\.

总之，就是对于多进程来说它是安全的，可以放心用。官方提供的例子如下：



<pre><code class="language-python">from skimage.io import imread
from skimage.transform import resize
import numpy as np

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)</code></pre>



就是按格式定义好<code>__len__</code>和<code>__getitem__</code>方法就行了，<code>__getitem__</code>方法直接返回一个batch的数据。

### bert as service例子

我第一次发现<code>Sequence</code>的必要性，是在试验[bert as service](<https://github.com/hanxiao/bert-as-service>)的时候。[bert as service](<https://github.com/hanxiao/bert-as-service>)是肖涵大佬搞的一个快速获取bert编码向量的服务组件，我曾经想用它获取字向量，然后传入到Keras中训练，但发现总会训练着训练着就卡住了。

经过搜索，确认是Keras的<code>fit_generator</code>所带的多进程，和bert\-as\-service自带的多进程冲突问题，具体怎么冲突我也比较模糊，就不深究了～而[这里](<https://github.com/hanxiao/bert-as-service/issues/29#issuecomment-442362241>)提供了一个参考的解决方案，用的就是继承<code>Sequence</code>类来写生成器。

（PS：就调用bert as service而言，后面肖涵大佬提供了协程版的<code>ConcurrentBertClient</code>，可以取代原来的<code>BertClient</code>，这样哪怕在原始生成器也不会有问题了。）

## 清流般的Keras

在我眼里，Keras就是深度学习框架中的一股清流，就好比Python是所有编程语言中的一股清流一样。用Keras实现所需要做的事情，就好比一次次惬意的享受。
