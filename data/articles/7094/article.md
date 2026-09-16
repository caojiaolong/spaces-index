# 6个派生优化器的简单介绍及其实现

> 作者：苏剑林 · 科学空间 · 2019-11-25
>
> 原文：<https://spaces.ac.cn/archives/7094>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

优化器可能是深度学习最“玄学”的一个模块之一了：有时候换一个优化器就能带来明显的提升，有时候别人说提升很多的优化器用到自己的任务上却一丁点用都没有，理论性质好的优化器不一定工作得很好，纯粹拍脑袋而来的优化器也未必就差了。但不管怎样，优化器终究也为热爱“深度炼丹”的同学提供了多一个选择。

近几年来，关于优化器的工作似乎也在慢慢增多，很多论文都提出了对常用优化器（尤其是[Adam](<https://papers.cool/arxiv/1412.6980>)）的大大小小的改进。本文就汇总一些优化器工作或技巧，并统一给出了代码实现，供读者有需调用。

## 基本形式

所谓“派生”，就是指相关的技巧都是建立在已有的优化器上的，任意一个已有的优化器都可以用上这些技巧，从而变成一个新的优化器。

已有的优化器的基本形式为：
\begin{equation}\begin{aligned}\boldsymbol{g}_t =&\, \nabla_{\boldsymbol{\theta}} L\\
\boldsymbol{h}_t =&\, f(\boldsymbol{g}_{\leq t})\\
\boldsymbol{\theta}_{t+1} =&\, \boldsymbol{\theta}_t - \gamma \boldsymbol{h}_t
\end{aligned}\end{equation}
其中$\boldsymbol{g}_t$即梯度，而$\boldsymbol{g}_{\leq t}$指的是截止到当前步的所有梯度信息，它们经过某种运算$f$（比如累积动量、累积二阶矩校正学习率等）后得到$\boldsymbol{h}_t$，然后由$\boldsymbol{h}_t$来更新参数，这里的$\gamma$就是指学习率。

## 变式杂烩

下面介绍优化器的6个变式，也可以理解为用优化器时的一些技巧。这些技巧有时候会很有效，有时候也可能无效甚至反作用，不能一概而论，只能理解为多一种选择就多一种可能。

### 权重衰减

权重衰减指的是直接在优化器每一步更新后面加上一个衰减项：
\begin{equation}\begin{aligned}\boldsymbol{g}_t =&\, \nabla_{\boldsymbol{\theta}} L\\
\boldsymbol{h}_t =&\, f(\boldsymbol{g}_{\leq t})\\
\boldsymbol{\theta}_{t+1} =&\, \boldsymbol{\theta}_t - \gamma \boldsymbol{h}_t - \gamma \lambda \boldsymbol{\theta}_t
\end{aligned}\end{equation}
其中$\lambda$称为“衰减率”。在SGD中，权重衰减等价于往loss里边加入$l_2$正则项$\frac{1}{2}\lambda \Vert \boldsymbol{\theta}\Vert_2^2$，但在Adagrad、Adam等带有自适应学习率的优化器中，$f$变成了非线性，所以两者不在等价。[《Decoupled Weight Decay Regularization》](<https://papers.cool/arxiv/1711.05101>)一文特别指出权重衰减的防过拟合能力优于对应的$l_2$正则，推荐大家使用权重衰减而不是$l_2$正则。

### 层自适应

在优化器中，最后的更新量由$\boldsymbol{h}_t$和学习率$\gamma$决定，有时候$\boldsymbol{h}_t$的模长会大于参数$\boldsymbol{\theta}_t$的模长，这可能会导致更新的不稳定。所以，一个直接的想法是：每一层的参数的更新幅度应该由$\boldsymbol{\theta}_t$的模长的模长来调控。这个直接的想法就导致了如下的优化器变体：
\begin{equation}\begin{aligned}\boldsymbol{g}_t =&\, \nabla_{\boldsymbol{\theta}} L\\
\boldsymbol{h}_t =&\, f(\boldsymbol{g}_{\leq t})\\
\boldsymbol{\theta}_{t+1} =&\, \boldsymbol{\theta}_t - \gamma \boldsymbol{h}_t\times \frac{\Vert\boldsymbol{\theta}_t\Vert_2}{\Vert\boldsymbol{h}_t\Vert_2}
\end{aligned}\end{equation}
如果基础优化器是Adam，那么上述优化器就是LAMB。论文[《Large Batch Optimization for Deep Learning: Training BERT in 76 minutes》](<https://papers.cool/arxiv/1904.00962>)指出LAMB在batch size较大（成千上万）的时候比Adam效果要好。

### 分段线性学习率

学习率也是优化器中的一个迷之存在，通常来说精调学习率策略能够获得一定提升，而不恰当的学习率甚至可能导致模型不收敛。常见的学习率策略有warmup、指数衰减、断层式下降（比如某个epoch后直接降到原来的1/10）等，比较迷的还有cos型学习率策略、多项式型学习率策略等。

考虑到常见的函数都可以用分段线性函数逼近，所以笔者干脆引入了一个分段线性学习率的策略，供大家随便玩。形式如下：
\begin{equation}\begin{aligned}\boldsymbol{g}_t =&\, \nabla_{\boldsymbol{\theta}} L\\
\boldsymbol{h}_t =&\, f(\boldsymbol{g}_{\leq t})\\
\boldsymbol{\theta}_{t+1} =&\, \boldsymbol{\theta}_t - \gamma \rho_t\boldsymbol{h}_t
\end{aligned}\end{equation}
其中$\rho_t$是某个以步数$t$为自变量的分段线性函数。

### 梯度累积

梯度累积之前在[《用时间换取效果：Keras梯度累积优化器》](<https://spaces.ac.cn/archives/6794>)一文中也介绍过了，其实不算优化器的变式，但可以写到优化器中，通过小batch size达到大batch size的效果，实现时间换空间。更大的batch size有时候能提升效果，尤其是基准batch size过小的情况下（8以下？）。

重述[《用时间换取效果：Keras梯度累积优化器》](<https://spaces.ac.cn/archives/6794>)一文关于梯度下降的描述：

> 所谓梯度累积，其实很简单，我们梯度下降所用的梯度，实际上是多个样本算出来的梯度的平均值，以batch\_size=128为例，你可以一次性算出128个样本的梯度然后平均，我也可以每次算16个样本的平均梯度，然后缓存累加起来，算够了8次之后，然后把总梯度除以8，然后才执行参数更新。当然，必须累积到了8次之后，用8次的平均梯度才去更新参数，不能每算16个就去更新一次，不然就是batch\_size=16了。

### Lookahead

Lookahead优化器来自论文[《Lookahead Optimizer: k steps forward, 1 step back》](<https://papers.cool/arxiv/1907.08610>)，在之前的文章[《Keras实现两个优化器：Lookahead和LazyOptimizer》](<https://spaces.ac.cn/archives/6869>)也介绍过。Lookahead的含义是用常用优化器预先往前摸索几步，然后根据摸索结果来更新，流程如下：
\begin{equation}\begin{aligned}&\boldsymbol{g}_t =\, \nabla_{\boldsymbol{\theta}} L\\
&\boldsymbol{h}_t =\, f(\boldsymbol{g}_{\leq t})\\
&\boldsymbol{\theta}_{t+1} =\, \boldsymbol{\theta}_t - \gamma\boldsymbol{h}_t\\
&\text{如果}t\,\text{mod}\,k = 0\text{:}\\
&\qquad\boldsymbol{\Theta}_{t+1} = \boldsymbol{\Theta}_t + \alpha (\boldsymbol{\theta}_{t+1}- \boldsymbol{\Theta}_t)\\
&\qquad\boldsymbol{\theta}_{t+1} = \boldsymbol{\Theta}_{t+1} \,(\text{即覆盖原来的}\boldsymbol{\theta}_{t+1})
\end{aligned}\end{equation}

其实这个优化器叫Lookaback也无妨，也就是每走几步就往后看看，跟几步前的权重做个插值。

### Lazy优化器

Lazy优化器在刚才的文章[《Keras实现两个优化器：Lookahead和LazyOptimizer》](<https://spaces.ac.cn/archives/6869>)中也介绍过，其本意就是Embedding层的更新应当稀疏化一些，这有助于防止过拟合。（参考[知乎讨论](<https://www.zhihu.com/question/265357659/answer/580469438>)）

## 参考实现

前面的介绍比较简单，事实上这些变式本身确实不难理解，关键还是代码实现。从前面的介绍中大家可以发现，这6个变式并无矛盾之处，因此良好的实现应当能让我们能搭积木般组合其中一个或多个变式使用；另外，目前keras也有两个分支：纯keras和tf\.keras，良好的实现应当能同时兼容它们（或者同时给出两种实现）。

### 终极移花接木

虽然部分变式在之前也实现过了，但这里还是用新的方式重新实现了它们。这里的实现方式源于笔者意外发现的一种移花接木技巧。

假设我们有这样一个类：



<pre><code class="language-python">import numpy as np

class A(object):
    def __init__(self):
        self.a = np.ones(1)
        self.b = np.ones(2)
        self.c = np.ones(3)</code></pre>



然后假设我们要继承A类得到一个B类，B类是要将<code>__init__</code>方法的所有<code>np.ones</code>替换为<code>np.zeros</code>，其余都不变。由于<code>__init__</code>可能是一个很复杂的流程，如果将它完整复制过来然后改写显然太冗余了。

有没有直接写几行代码就能替换所有的呢？还真有！



<pre><code class="language-python">class B(A):
    def __init__(self):
        _ = np.ones
        np.ones = np.zeros
        super(B, self).__init__()
        np.ones = _</code></pre>



有了这个demo，我们就可以“魔改”已有优化器了。在keras中，参数的更新都是通过<code>K.update</code>来实现的（参考[keras的optimizers\.py](<https://github.com/keras-team/keras/blob/master/keras/optimizers.py>))，我们只需要用上述方式重新定义一下<code>K.update</code>就好。

如果是tf\.keras呢？很遗憾，这种方式不可行，因为tf\.keras中常用优化器的迭代流程都被写到C里边去了（参考[tf\.keras的adam\.py](<https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/adam.py>)），我们看不到代码，也就不能用这种方法改了。一个解决办法是我们自己重新实现一个优化器如Adam，将迭代流程暴露出来，这样我们就能用上述方式魔改了。

### 使用示例

根据上述思路统一实现的6个优化器变体，都被放在笔者的bert4keras项目中：[<strong>bert4keras\.optimizers</strong>](<https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py>)。

所有函数会根据keras还是tf\.keras来进行正确的导入，从而实现keras/tf\.keras都能用同样的方式使用。里边自带了一个Adam实现，这个Adam是专门写给tf\.keras的。<strong>对于tf\.keras，如果想要实现上述变式，那只能用bert4keras自带的优化器（目前只有Adam），不能用tf\.keras内置的优化器。</strong>

参考代码：



<pre><code class="language-python">from bert4keras.optimizers import *

# 变成带权重衰减的Adam
AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=0.001, weight_decay_rate=0.01)

# 变成带分段线性学习率的Adam
AdamLR = extend_with_piecewise_linear_lr(Adam, 'AdamLR')
# 实现warmup，前1000步学习率从0增加到0.001
optimizer = AdamLR(learning_rate=0.001, lr_schedule={1000: 1.})

# 变成带梯度累积的Adam
AdamGA = extend_with_gradient_accumulation(Adam, 'AdamGA')
optimizer = AdamGA(learning_rate=0.001, grad_accum_steps=10)

# 组合使用
AdamWLR = extend_with_piecewise_linear_lr(AdamW, 'AdamWLR')
# 带权重衰减和warmup的优化器
optimizer = AdamWLR(learning_rate=0.001,
                    weight_decay_rate=0.01,
                    lr_schedule={1000: 1.})
</code></pre>



<strong>（注：一下实现这么多个优化器，而且同时要兼容keras和tf\.keras，难免可能会有错漏之处，如有发现，万望不吝指正。）</strong>

## 写在最后

炼丹不易，且炼且珍惜。
