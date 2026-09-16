# “让Keras更酷一些！”：精巧的层与花式的回调

> 作者：苏剑林 · 科学空间 · 2018-08-06
>
> 原文：<https://spaces.ac.cn/archives/5765>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## Keras伴我走来

回想起进入机器学习领域的这两三年来，Keras是一直陪伴在笔者的身边。要不是当初刚掉进这个坑时碰到了Keras这个这么易用的框架，能快速实现我的想法，我也不确定我是否能有毅力坚持下来，毕竟当初是theano、pylearn、caffe、torch等的天下，哪怕在今天它们对我来说仍然像天书一般。

后来为了拓展视野，我也去学习了一段时间的tensorflow，用纯tensorflow写过若干程序，但不管怎样，仍然无法割舍Keras。随着对Keras的了解的深入，尤其是花了一点时间研究过Keras的源码后，我发现Keras并没有大家诟病的那样“欠缺灵活性”。事实上，Keras那精巧的封装，可以让我们轻松实现很多复杂的功能。我越来越感觉，Keras像是一件非常精美的艺术品，充分体现了Keras的开发者们深厚的创作功力。

本文介绍Keras中自定义模型的一些内容，相对而言，这属于Keras进阶的内容，刚入门的朋友请暂时忽略。

## 层的自定义

这里介绍Keras中自定义层及其一些运用技巧，在这之中我们可以看到Keras层的精巧之处。

### 基本定义方法

在Keras中，自定义层的最简单方法是通过Lambda层的方式：



<pre><code class="language-python">from keras.layers import *
from keras import backend as K

x_in = Input(shape=(10,))
x = Lambda(lambda x: x+2)(x_in) # 对输入加上2
</code></pre>



有时候，我们希望区分训练阶段和测试阶段，比如训练阶段给输入加入一些噪声，而测试阶段则去掉噪声，这需要用K\.in\_train\_phase实现，比如



<pre><code class="language-python">def add_noise_in_train(x):
    x_ = x + K.random_normal(shape=K.shape(x)) # 加上标准高斯噪声
    return K.in_train_phase(x_, x)

x_in = Input(shape=(10,))
x = Lambda(add_noise_in_train)(x_in) # 训练阶段加入高斯噪声，测试阶段去掉
</code></pre>



当然，Lambda层仅仅适用于不需要增加训练参数的情形，如果想要实现的功能需要往模型新增参数，那么就必须要用到自定义Layer了。其实这也不复杂，相比于Lambda层只不过代码多了几行，官方文章已经写得很清楚了：
[https://keras\.io/layers/writing\-your\-own\-keras\-layers/](<https://keras.io/layers/writing-your-own-keras-layers/>)

这里把它页面上的例子搬过来：



<pre><code class="language-python">class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim # 可以自定义一些属性，方便调用
        super(MyLayer, self).__init__(**kwargs) # 必须

    def build(self, input_shape):
        # 添加可训练参数
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

    def call(self, x):
        # 定义功能，相当于Lambda层的功能函数
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        # 计算输出形状，如果输入和输出形状一致，那么可以省略，否则最好加上
        return (input_shape[0], self.output_dim)</code></pre>



### 双输出的层

平时我们碰到的所有层，几乎都是单输出的，包括Keras中自带的所有层，都是一个或者多个输入，然后返回一个结果输出的。那么Keras可不可以定义双输出的层呢？答案是可以，但要明确定义好output\_shape，比如下面这个层，简单地将输入切开分两半，并且同时返回。



<pre><code class="language-python">class SplitVector(Layer):

    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)

    def call(self, inputs):
        # 按第二个维度对tensor进行切片，返回一个list
        in_dim = K.int_shape(inputs)[-1]
        return [inputs[:, :in_dim//2], inputs[:, in_dim//2:]]

    def compute_output_shape(self, input_shape):
        # output_shape也要是对应的list
        in_dim = input_shape[-1]
        return [(None, in_dim//2), (None, in_dim-in_dim//2)]


x1, x2 = SplitVector()(x_in) # 使用方法</code></pre>



### 层与loss的结合

有了[《Keras中自定义复杂的loss函数》](<https://spaces.ac.cn/archives/4493>)一文经验的读者可以知道，Keras中对loss的基本定义是一个输入为y\_true和y\_pred函数。但在比较复杂的情况下，它不仅仅是预测值和目标值的函数，还可以结合权重进行复杂的运算。

这里再次以center loss为例，介绍一种基于自定义层的写法。



<pre><code class="language-python">class Dense_with_Center_loss(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Dense_with_Center_loss, self).__init__(**kwargs)

    def build(self, input_shape):
        # 添加可训练参数
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='zeros',
                                    trainable=True)
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer='glorot_normal',
                                       trainable=True)

    def call(self, inputs):
        # 对于center loss来说，返回结果还是跟Dense的返回结果一致
        # 所以还是普通的矩阵乘法加上偏置
        self.inputs = inputs
        return K.dot(inputs, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, y_true, y_pred, lamb=0.5):
        # 定义完整的loss
        y_true = K.cast(y_true, 'int32') # 保证y_true的dtype为int32
        crossentropy = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        centers = K.gather(self.centers, y_true[:, 0]) # 取出样本中心
        center_loss = K.sum(K.square(centers - self.inputs), axis=1) # 计算center loss
        return crossentropy + lamb * center_loss


f_size = 2

x_in = Input(shape=(784,))
f = Dense(f_size)(x_in)

dense_center = Dense_with_Center_loss(10)
output = dense_center(f)

model = Model(x_in, output)
model.compile(loss=dense_center.loss,
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

# 这里是y_train是类别的整数id，不用转为one hot
model.fit(x_train, y_train, epochs=10)</code></pre>



## 花式回调器

除了修改模型，我们还可能在训练过程中做很多事情，比如每个epoch结束后，算一下验证集的指标，保存最优模型，还有可能在多少个epoch后就降低学习率，或者修改正则项参数，等等，这些都可以通过回调器来实现。

回调器官方页：[https://keras\.io/callbacks/](<https://keras.io/callbacks/>)

### 保存最优模型

在Keras中，根据验证集的指标来保留最优模型，最简便的方法是通过自带的ModelCheckpoint，比如



<pre><code class="language-python">checkpoint = ModelCheckpoint(filepath='./best_model.weights',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

model.fit(x_train,
          y_train,
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[checkpoint])</code></pre>



然而，这种方法虽然简单，但是有一个明显的缺点，就是里边的指标是由compile的metrics来确定的，而Keres中自定义一个metric，需要写成张量运算才行，也就是说如果你期望的指标并不能写成张量运算（比如bleu等指标），那么就没法写成一个metric函数了，也就不能用这个方案了。

于是，一个万能的方案就出来了：自己写回调器，爱算什么就算什么。比如：



<pre><code class="language-python">from keras.callbacks import Callback

def evaluate(): # 评测函数
    pred = model.predict(x_test)
    return np.mean(pred.argmax(axis=1) == y_test) # 爱算啥就算啥


# 定义Callback器，计算验证集的acc，并保存最优模型
class Evaluate(Callback):

    def __init__(self):
        self.accs = []
        self.highest = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate()
        self.accs.append(acc)
        if acc &gt;= self.highest: # 保存最优模型权重
            self.highest = acc
            model.save_weights('best_model.weights')

        # 爱运行什么就运行什么
        print 'acc: %s, highest: %s' % (acc, self.highest)


evaluator = Evaluate()
model.fit(x_train,
          y_train,
          epochs=10,
          callbacks=[evaluator])</code></pre>



### 修改超参数

训练过程中还有可能对超参数进行微调，比如最常见的一个需求是根据epoch来调整学习率，这可以简单地通过LearningRateScheduler来实现，它也属于回调器之一。



<pre><code class="language-python">from keras.callbacks import LearningRateScheduler


def lr_schedule(epoch):
    # 根据epoch返回不同的学习率
    if epoch &lt; 50:
        lr = 1e-2
    elif epoch &lt; 80:
        lr = 1e-3
    else:
        lr = 1e-4
    return lr


lr_scheduler = LearningRateScheduler(lr_schedule)

model.fit(x_train,
          y_train,
          epochs=10,
          callbacks=[evaluator, lr_scheduler])</code></pre>



如果是其他超参数呢？比如前面center loss的lamb，或者是类似的正则项。这种情况下，我们需要将lamb设为一个Variable，然后自定义一个回调器来动态赋值。比如当初我定义的一个loss：



<pre><code class="language-python">def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2</code></pre>



如果要动态改变参数e，那么可以改为



<pre><code class="language-python">e = K.variable(0.1)

def mycrossentropy(y_true, y_pred):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2


model.compile(loss=mycrossentropy,
              optimizer='adam')


class callback4e(Callback):
    def __init__(self, e):
        self.e = e
    def on_epoch_end(self, epoch, logs={}):
        if epoch &gt;= 100: # 100个epoch之后设为0.01 
            K.set_value(self.e, 0.01)

model.fit(x_train,
          y_train,
          epochs=10,
          callbacks=[callback4e(e)])</code></pre>



注意Callback类共支持六种在不同阶段的执行函数：on\_epoch\_begin、on\_epoch\_end、on\_batch\_begin、on\_batch\_end、on\_train\_begin、on\_train\_end，每个函数所执行的阶段不一样（根据名字很容易判断），可以组合起来实现很复杂的功能。比如warmup，就是指设定了默认学习率后，并不是一开始就用这个学习率训练，而是在前几个epoch中，从零慢慢增加到默认的学习率，这个过程可以理解为在为模型调整更好的初始化。参考代码：



<pre><code class="language-python">class Evaluate(Callback):
    def __init__(self):
        self.num_passed_batchs = 0
        self.warmup_epochs = 10
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.params['steps'] == None:
            self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
        else:
            self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs &lt; self.steps_per_epoch * self.warmup_epochs:
            # 前10个epoch中，学习率线性地从零增加到0.001
            K.set_value(self.model.optimizer.lr,
                        0.001 * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
            self.num_passed_batchs += 1</code></pre>



## Keras无限可能

Keras还有很多可圈可点的技巧，比如可以直接利用model\.add\_loss来灵活地增加loss，还有模型嵌套调用、纯粹作为tensorflow的简单上层api，等等，就不一一整理了，欢迎有疑问、有兴趣的读者留言讨论。

通常我们认为Keras这样的高度封装的库，灵活性是比较欠缺的，但事实上不然。要知道，Keras并不是简单地调用tensorflow或者theano中现成的上层函数，而仅仅是通过backend来封装了一些基本的函数，然后把所有的东西（各种层、优化器等）用自己的backend重写了一遍！也正是如此，它才能支持切换不同的后段。

能做到这个程度，Keras的灵活性是不容置喙的，但是这种灵活性在帮助文档和普通的案例中比较难体现，很多时候要阅读源码，才能感觉到Keras那样的写法已经无可挑剔了。我感觉，用Keras实现复杂的模型，既是一种挑战，又像是一种艺术创作，当你成功时，你就会陶醉于你创造出来的艺术品了。
