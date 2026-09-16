# 记录一次半监督的情感分析

> 作者：苏剑林 · 科学空间 · 2017-05-04
>
> 原文：<https://spaces.ac.cn/archives/4374>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

本文是一次不怎么成功的半监督学习的尝试：在IMDB的数据集上，用随机抽取的1000个标注样本训练一个文本情感分类模型，并且在余下的49000个测试样本中，测试准确率为73\.48%。

## 思路

本文的思路来源于OpenAI的这篇文章：
[《OpenAI新研究发现无监督情感神经元：可直接调控生成文本的情感》](<http://jiqizhixin.com/article/2612?from=singlemessage>)

文章里边介绍了一种无监督（实际上是半监督）做情感分类的模型的方法，并且实验效果很好。然而文章里边的实验很庞大，对于个人来说几乎不可能重现（在4块Pascal GPU花了1个月时间训练）。不过，文章里边的思想是很简单的，根据里边的思想，我们可以做个“山寨版”的。思路如下：

我们一般用深度学习做情感分类，比较常规的思路就是Embedding层\+LSTM层\+Dense层(Sigmoid激活)，我们常说的词向量，相当于预训练了Embedding层（这一层的参数量最大，最容易过拟合），而OpenAI的思想就是，为啥不连LSTM层一并预训练了呢？预训练的方法也是用语言模型来训练。当然，为了使得预训练的结果不至于丢失情感信息，LSTM的隐藏层节点要大一些。

如果连LSTM层预训练了，那么剩下的Dense层参数就不多了，因此可以用少量标注样本就能够训练完备了。这就是整个半监督学习的思路了。至少OpenAI文章说的什么情感神经元，那不过是形象的描述罢了。

当然，从情感分析这个任务上来看，本文的73\.48%准确率实在难登大雅之台，随便一个“词典\+规则”的方案，都有80%以上的准确率。我只是验证了这种实验方案的可行性。我相信，如果规模能做到OpenAI那么大，效果应该会更好的。<strong>而且，本文想描述的是一种建模策略，并非局限于情感分析，同样的思想可以用于任意的二分类甚至多分类问题</strong>。

## 过程

首先，加载数据集并且重新划分训练集和测试集：



<pre><code class="language-python">from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.datasets import imdb
from keras import backend as K
import numpy as np

max_features = 10000 #保留前max_features个词
maxlen = 100 #填充/阶段到100词
batch_size = 1000
nb_grams = 10 #训练一个10-gram的语言模型
nb_train = 1000 #训练样本数

#加载内置的IMDB数据集
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_lm_ = np.append(x_train, x_test)

#构造用来训练语言模型的数据
#这里只用了已有数据，实际环境中，可以补充其他数据使得训练更加充分
x_lm = []
y_lm = []
for x in x_lm_:
		for i in range(len(x)):
			x_lm.append([0]*(nb_grams - i + max(0,i-nb_grams))+x[max(0,i-nb_grams):i])
			y_lm.append([x[i]])

x_lm = np.array(x_lm)
y_lm = np.array(y_lm)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x = np.vstack([x_train, x_test])
y = np.hstack([y_train, y_test])

#重新划分训练集和测试集
#合并原来的训练集和测试集，随机挑选1000个样本，作为新的训练集，剩下为测试集
idx = range(len(x))
np.random.shuffle(idx)
x_train = x[idx[:nb_train]]
y_train = y[idx[:nb_train]]
x_test = x[idx[nb_train:]]
y_test = y[idx[nb_train:]]</code></pre>



然后搭建模型



<pre><code class="language-python">embedded_size = 100 #词向量维度
hidden_size = 1000 #LSTM的维度，可以理解为编码后的句向量维度。

#encoder部分
inputs = Input(shape=(None,), dtype='int32')
embedded = Embedding(max_features, embedded_size)(inputs)
lstm = LSTM(hidden_size)(embedded)
encoder = Model(inputs=inputs, outputs=lstm)

#完全用ngram模型训练encode部分
input_grams = Input(shape=(nb_grams,), dtype='int32')
encoded_grams = encoder(input_grams)
softmax = Dense(max_features, activation='softmax')(encoded_grams)
lm = Model(inputs=input_grams, outputs=softmax)
lm.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
#用sparse交叉熵，可以不用事先将类别转换为one hot形式。

#情感分析部分
#固定encoder，后面接一个简单的Dense层（相当于逻辑回归）
#这时候训练的只有hidden_size+1=1001个参数
#因此理论上来说，少量标注样本就可以训练充分
for layer in encoder.layers:
    layer.trainable=False

sentence = Input(shape=(maxlen,), dtype='int32')
encoded_sentence = encoder(sentence)
sigmoid = Dense(10, activation='relu')(encoded_sentence)
sigmoid = Dropout(0.5)(sigmoid)
sigmoid = Dense(1, activation='sigmoid')(sigmoid)
model = Model(inputs=sentence, outputs=sigmoid)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])</code></pre>



好了，训练语言模型，这部分工作比较耗时：
\#训练语言模型，比较耗时，一般迭代两三次就好



<pre><code class="language-python">lm.fit(x_lm, y_lm,
       batch_size=batch_size,
       epochs=3)</code></pre>



语言模型的训练结果为

> Epoch 1/3
> 11737946/11737946 &#91;==============================&#93; \- 2400s \- loss: 5\.0376
> Epoch 2/3
> 11737946/11737946 &#91;==============================&#93; \- 2404s \- loss: 4\.5587
> Epoch 3/3
> 11737946/11737946 &#91;==============================&#93; \- 2404s \- loss: 4\.3968

接着，开始用1000个样本训练情感分析模型。由于前面已经预训练好，因此这里需要训练的参数不多，配合Dropout的话，因此1000个样本也不会导致严重过拟合。
\#训练情感分析模型



<pre><code class="language-python">model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=200)</code></pre>



训练结果为

> Epoch 198/200
> 1000/1000 &#91;==============================&#93; \- 0s \- loss: 0\.2481 \- acc: 0\.9250
> Epoch 199/200
> 1000/1000 &#91;==============================&#93; \- 0s \- loss: 0\.2376 \- acc: 0\.9330
> Epoch 200/200
> 1000/1000 &#91;==============================&#93; \- 0s \- loss: 0\.2386 \- acc: 0\.9350

接着来评估一下模型：



<pre><code class="language-python">#评估一下模型的效果
model.evaluate(x_test, y_test, verbose=True, batch_size=batch_size)</code></pre>



准确率73\.04％，一般般～～试试迁移学习，把训练集连同测试集的预测结果一起进行训练：



<pre><code class="language-python">#把训练集连同测试集的预测结果（即可能包含有误的数据），重新训练模型
y_pred = model.predict(x_test, verbose=True, batch_size=batch_size)
y_pred = (y_pred.reshape(-1) &gt; 0.5).astype(int)
xt = np.vstack([x_train, x_test])
yt = np.hstack([y_train, y_pred])

model.fit(xt, yt,
          batch_size=batch_size,
          epochs=10)

#评估一下模型的效果
model.evaluate(x_test, y_test, verbose=True, batch_size=batch_size)</code></pre>



训练结果为

> Epoch 8/10
> 50000/50000 &#91;==============================&#93; \- 27s \- loss: 0\.1455 \- acc: 0\.9561
> Epoch 9/10
> 50000/50000 &#91;==============================&#93; \- 27s \- loss: 0\.1390 \- acc: 0\.9590
> Epoch 10/10
> 50000/50000 &#91;==============================&#93; \- 27s \- loss: 0\.1349 \- acc: 0\.9600

这次我们得到了73\.33%的准确率。不难发现，事实上这个过程可以重复迭代，再重复一次，得到73\.33%的准确率，重复第二次，得到73\.47%\.\.\.可以预料，这会趋于一个稳定值，我再重复了5次，稳定在73\.48%。

从刚开始的73\.04％，到迁移学习后的73\.48%，约有0\.44%的提升，看上去不大，但是如果对于做比赛或者写论文的同学来说，0\.44%的提升，可以小书一笔了～

## 点评

文章开头已经说了，这次是个不大成功的尝试，毕竟是“山寨版”的，因此大家不要太纠结准确率不高的问题。而从本文的实验结果来看，这种方案是靠谱的。通过大量的、混合情感的语料训练语言模型，确实能够很好地提取文本的特征，这类似于图像的自编码过程。

本文做得很简单，没有细微调过超参。改进的思路大概有那么几个：增大语言模型的规模、增加情感语料（只需要是情感评论的，不需要标签）、训练细节上的优化。这个就暂时不做了～
