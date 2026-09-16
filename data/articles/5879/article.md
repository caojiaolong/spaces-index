# “让Keras更酷一些！”：小众的自定义优化器

> 作者：苏剑林 · 科学空间 · 2018-09-08
>
> 原文：<https://spaces.ac.cn/archives/5879>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<div data-spaces-format="html-v1">
<p data-spaces-tag="p" data-spaces-attrs="{}">沿着之前的<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/5765/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/5765/">《“让Keras更酷一些！”：精巧的层与花式的回调》</a>写下去～</p><p data-spaces-tag="p" data-spaces-attrs="{}">今天我们来看一个小众需求：自定义优化器。</p><p data-spaces-tag="p" data-spaces-attrs="{}">细想之下，不管用什么框架，自定义优化器这个需求可谓真的是小众中的小众。一般而言，对于大多数任务我们都可以无脑地直接上Adam，而调参炼丹高手一般会用SGD来调出更好的效果，换言之不管是高手新手，都很少会有自定义优化器的需求。</p><p data-spaces-tag="p" data-spaces-attrs="{}">那这篇文章还有什么价值呢？有些场景下会有一点点作用。<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">比如通过学习Keras中的优化器写法，你可以对梯度下降等算法有进一步的认识，你还可以顺带看到Keras的源码是多么简洁优雅。此外，有时候我们可以通过自定义优化器来实现自己的一些功能，比如给一些简单的模型（例如Word2Vec）重写优化器（直接写死梯度，而不是用自动求导），可以使得算法更快；自定义优化器还可以实现诸如“软batch”的功能。</font></p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;Keras优化器&quot;}">Keras优化器</h2><p data-spaces-tag="p" data-spaces-attrs="{}">我们首先来看Keras中自带优化器的代码，位于：<br data-spaces-tag="br" data-spaces-attrs="{}">
<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://github.com/keras-team/keras/blob/master/keras/optimizers.py&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://github.com/keras-team/keras/blob/master/keras/optimizers.py">https://github.com/keras-team/keras/blob/master/keras/optimizers.py</a></p><p data-spaces-tag="p" data-spaces-attrs="{}">简单起见，我们可以先挑SGD来看。当然，Keras中的SGD算法已经把momentum、nesterov、decay等整合进去了，这使用起来方便，但不利于学习。所以我稍微简化了一下，给出一个纯粹的SGD算法的例子：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras import backend as K


class SGD(Optimizer):
    """Keras中简单自定义SGD优化器
    """

    def __init__(self, lr=0.01, **kwargs):
        super(SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """主要的参数更新算法
        """
        grads = self.get_gradients(loss, params) # 获取梯度
        self.updates = [K.update_add(self.iterations, 1)] # 定义赋值算子集合
        self.weights = [self.iterations] # 优化器带来的权重，在保存模型时会被保存
        for p, g in zip(params, grads):
            # 梯度下降
            new_p = p - self.lr * g
            # 如果有约束，对参数加上约束
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            # 添加赋值
            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr))}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">应该不是解释了吧？有没有特别简单的感觉？定义一个优化器也不是特别高大上的事情嘛～</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;实现“软batch”&quot;}">实现“软batch”</h2><p data-spaces-tag="p" data-spaces-attrs="{}">现在来实现一个稍微复杂一点的功能，就是所谓的“软batch”，不过我不大清楚是不是就叫这个名字，姑且先这样叫着吧。大概的场景是：假如模型比较庞大，自己的显卡最多也就能跑batch size=16，但我又想起到batch size=64的效果，那可以怎么办呢？一种可以考虑的方案是，每次算batch size=16，然后把梯度缓存起来，4个batch后才更新参数。也就是说，每个小batch都算梯度，但每4个batch才更新一次参数。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">2019年7月8日更新：以下实现是有误的，无法达到预期效果，修正的实现请看<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/6794&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/6794">《Keras梯度累积优化器：用时间换取效果》</a>。</font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}"><del data-spaces-tag="del" data-spaces-attrs="{}">如果真的有这个需求，那么就只能通过修改优化器来解决了。在前面的SGD的基础上，参考代码如下：</del></p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">class7 MySGD(Optimizer):
    """Keras中简单自定义SGD优化器
    每隔一定的batch才更新一次参数
    """
    def __init__(self, lr=0.01, steps_per_update=1, **kwargs):
        super(MySGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.steps_per_update = steps_per_update # 多少batch才更新一次

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        """主要的参数更新算法
        """
        shapes = [K.int_shape(p) for p in params]
        sum_grads = [K.zeros(shape) for shape in shapes] # 平均梯度，用来梯度下降
        grads = self.get_gradients(loss, params) # 当前batch梯度
        self.updates = [K.update_add(self.iterations, 1)] # 定义赋值算子集合
        self.weights = [self.iterations] + sum_grads # 优化器带来的权重，在保存模型时会被保存
        for p, g, sg in zip(params, grads, sum_grads):
            # 梯度下降
            new_p = p - self.lr * sg / float(self.steps_per_update)
            # 如果有约束，对参数加上约束
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            cond = K.equal(self.iterations % self.steps_per_update, 0)
            # 满足条件才更新参数
            self.updates.append(K.switch(cond, K.update(p, new_p), p))
            # 满足条件就要重新累积，不满足条件直接累积
            self.updates.append(K.switch(cond, K.update(sg, g), K.update(sg, sg+g)))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'steps_per_update': self.steps_per_update}
        base_config = super(MySGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}"><del data-spaces-tag="del" data-spaces-attrs="{}">应该也很容易理解吧。如果带有动量的情况，写起来复杂一点，但也是一样的。重点就是引入多一个变量来储存累积梯度，然后引入cond来控制是否更新，原来优化器要做的事情，都要在cond为True的情况下才做（梯度改为累积起来的梯度）。对比原始的SGD，改动并不大。</del></p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;“侵入式”优化器&quot;}">“侵入式”优化器</h2><p data-spaces-tag="p" data-spaces-attrs="{}">上面实现优化器的方案是标准的，也就是按Keras的设计规范来做的，所以做起来很轻松。然而我曾经想要实现的一个优化器，却不能用这种方式来实现，经过阅读源码，得到了一种“侵入式”的写法，这种写法类似“外挂”的形式，可以实现我需要的功能，但不是标准的写法，在此也跟大家分享一下。</p><p data-spaces-tag="p" data-spaces-attrs="{}">原始需求来源于之前的文章<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/5655&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/5655">《从动力学角度看优化算法（一）：从SGD到动量加速》</a>，里边指出梯度下降优化器可以看成是微分方程组的欧拉解法，进一步可以联想到，微分方程组有很多比欧拉解法更高级的解法呀，能不能用到深度学习中？比如稍微高级一点的有“<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://en.wikipedia.org/wiki/Heun%27s_method&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://en.wikipedia.org/wiki/Heun%27s_method">Heun方法</a>”：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}\tilde{p}_{i+1} =&amp; p_i + \epsilon g(p_i)\\<br data-spaces-tag="br" data-spaces-attrs="{}">
p_{i+1} =&amp; p_i + \frac{1}{2}\epsilon \big[g(p_i)+g(\tilde{p}_{i+1})\big]<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{aligned}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
其中$p$是参数（向量），$g$是梯度，$p_i$表示$p$的第$i$次迭代时的结果。这个算法需要走两步，大概意思就是普通的梯度下降先走一步（探路），然后根据探路的结果取平均，得到更精准的步伐，等价地可以改写为：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}\tilde{p}_{i+1} =&amp; p_i + \epsilon g(p_i)\\<br data-spaces-tag="br" data-spaces-attrs="{}">
p_{i+1} =&amp; \tilde{p}_{i+1} + \frac{1}{2}\epsilon \big[g(\tilde{p}_{i+1}) - g(p_i)\big]<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{aligned}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这样就清楚显示出后面这一步实际上是对梯度下降的微调。</p><p data-spaces-tag="p" data-spaces-attrs="{}">但是实现这类算法却有个难题，要计算两次梯度，一次对参数$p_i$，另一次对参数$\tilde{p}_{i+1}$。而前面的优化器定义中get_updates这个方法却只能执行一步（对应到tf框架中，就是执行一步sess.run，熟悉tf的朋友知道单单执行一步sess.run很难实现这个需求），因此实现不了这种算法。经过研究Keras模型的训练源码，我发现可以这样写：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">#! -*- coding: utf-8 -*-

from keras.optimizers import Optimizer
from keras import backend as K


class InjectOptimizer(Optimizer):
    """定义注入式优化器的基类
        需要传入模型，直接修改模型的训练函数，而不按常规流程使用优化器，所以称为“侵入式”
        其实下面的大部分代码，都是直接抄自keras的源码：
        https://github.com/keras-team/keras/blob/master/keras/engine/training.py#L497
        也就是keras中的_make_train_function函数。
    """
    def get_updates(self, loss, params):
        return []
    def get_grouped_updates(self, loss, params):
        raise NotImplementedError
    def inject(self, model):
        """传入模型做注入
        """
        if not hasattr(model, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        model._check_trainable_weights_consistency()
        if model.train_function is None:
            inputs = (model._feed_inputs +
                      model._feed_targets +
                      model._feed_sample_weights)
            if model._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]
            with K.name_scope('training'):
                train_functions = []
                with K.name_scope(model.optimizer.__class__.__name__):
                    grouped_training_updates = self.get_grouped_updates(
                        params=model._collected_trainable_weights,
                        loss=model.total_loss)
                for i, updates in enumerate(grouped_training_updates[1:]):
                    f = K.function(
                        inputs,
                        [model.total_loss],
                        updates=updates,
                        name='train_function_%s' % (i + 1),
                        **model._function_kwargs)
                    train_functions.append(f)
                # Gets loss and metrics. Updates weights at each call.
                first_updates = (model.updates +
                                 grouped_training_updates[0],
                                 model.metrics_updates)
                first_train = K.function(
                    inputs,
                    [model.total_loss] + model.metrics_tensors,
                    updates=first_updates,
                    name='train_function',
                    **model._function_kwargs)
                def F(inputs):
                    R = first_train(inputs)
                    for f in train_functions:
                        f(inputs)
                    return R
                model.train_function = F


class HeunOptimizer(InjectOptimizer):
    """Heun优化器
    ( https://en.wikipedia.org/wiki/Heun%27s_method )
    """
    def __init__(self, lr, **kwargs):
        super(HeunOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
    def get_updates_1(self, loss, params, cache_grads):
        updates = []
        grads = self.get_gradients(loss, params)
        for p, g, cg in zip(params, grads, cache_grads):
            updates.append(K.update(cg, g))
            updates.append(K.update(p, p - self.lr * g))
        return updates
    def get_updates_2(self, loss, params, cache_grads):
        updates = []
        grads = self.get_gradients(loss, params)
        for p, g, cg in zip(params, grads, cache_grads):
            updates.append(K.update(p, p - 0.5 * self.lr * (g - cg)))
        return updates
    def get_grouped_updates(self, loss, params):
        cache_grads = [K.zeros(K.int_shape(p)) for p in params]
        return [
            self.get_updates_1(loss, params, cache_grads),
            self.get_updates_2(loss, params, cache_grads)
        ]</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">用法是：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">opt = HeunOptimizer(0.1)
model.compile(loss='mse', optimizer=opt)
opt.inject(model) # 必须执行这步

model.fit(x_train, y_train, epochs=100, batch_size=32)</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">其中关键思想在代码中已经注释了，主要是Keras的优化器最终都会被包装为一个train_function，所以我们只需要参照Keras的源码设计好train_function，并在其中插入我们自己的操作。在这个过程中，需要留意到K.function所定义的操作相当于一次sess.run就行了。</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">注：类似地还可以实现RK23、RK45等算法。遗憾的是，这种优化器效果并不好～</font></strong></blockquote><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;优雅的Keras&quot;}">优雅的Keras</h2><p data-spaces-tag="p" data-spaces-attrs="{}">本文讲了一个非常非常小众的需求：自定义优化器，介绍了一般情况下Keras优化器的写法，以及一种“侵入式”的写法。如果真有这么个特殊需求，可以参考使用。</p><p data-spaces-tag="p" data-spaces-attrs="{}">通过Keras中优化器的分析研究，我们进一步可以观察到Keras整体代码实在是非常简洁优雅，难以挑剔～</p>
</div>
