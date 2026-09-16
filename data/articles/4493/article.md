# Keras中自定义复杂的loss函数

> 作者：苏剑林 · 科学空间 · 2017-07-22
>
> 原文：<https://spaces.ac.cn/archives/4493>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

Keras是一个搭积木式的深度学习框架，用它可以很方便且直观地搭建一些常见的深度学习模型。在tensorflow出来之前，Keras就已经几乎是当时最火的深度学习框架，以theano为后端，而如今Keras已经同时支持四种后端：theano、tensorflow、cntk、mxnet（前三种官方支持，mxnet还没整合到官方中），由此可见Keras的魅力。

Keras是很方便，然而这种方便不是没有代价的，最为人诟病之一的缺点就是灵活性较低，难以搭建一些复杂的模型。的确，Keras确实不是很适合搭建复杂的模型，但并非没有可能，而是搭建太复杂的模型所用的代码量，跟直接用tensorflow写也差不了多少。但不管怎么说，Keras其友好、方便的特性（比如那可爱的训练进度条），使得我们总有使用它的场景。这样，如何更灵活地定制Keras模型，就成为一个值得研究的课题了。这篇文章我们来关心自定义loss。

## 输入\-输出设计

Keras的模型是函数式的，即有输入，也有输出，而loss即为预测值与真实值的某种误差函数。Keras本身也自带了很多[loss函数](<https://keras.io/losses/>)，如mse、交叉熵等，直接调用即可。而要自定义loss，最自然的方法就是仿照Keras自带的loss进行改写。

比如，我们做分类问题时，经常用的就是softmax输出，然后用交叉熵作为loss。然而这种做法也有不少缺点，其中之一就是分类太自信，哪怕输入噪音，分类的结果也几乎是非1即0，这通常会导致过拟合的风险，还会使得我们在实际应用中没法很好地确定置信区间、设置阈值。因此很多时候我们也会想办法使得分类别太自信，而修改loss也是手段之一。

如果不修改loss，我们就是使用交叉熵去拟合一个one hot的分布。交叉熵的公式是
$$S(q|p)=-\sum_i q_i \log p_i$$
其中$p_i$是预测的分布，而$q_i$是真实的分布，比如输出为$[z_1,z_2,z_3]$，目标为$[1,0,0]$，那么
$$loss = -\log \Big(e^{z_1}/Z\Big),\, Z=e^{z_1}+e^{z_2}+e^{z_3}$$
只要$z_1$已经是$[z_1,z_2,z_3]$的最大值，那么我们总可以“变本加厉”——通过增大训练步数，使得$z_1,z_2,z_3$增加足够大的比例（等价地，即增大向量$[z_1,z_2,z_3]$的模长），从而$e^{z_1}/Z$足够接近1（等价地，loss足够接近0）。这就是通常softmax过于自信的来源：只要盲目增大模长，就可以降低loss，训练器肯定是很乐意了，这代价太低了。为了使得分类不至于太自信，一个方案就是不要单纯地去拟合one hot分布，分一点力气去拟合一下均匀分布，即改为新loss：
$$loss = -(1-\varepsilon)\log \Big(e^{z_1}/Z\Big)-\varepsilon\sum_{i=1}^n \frac{1}{3}\log \Big(e^{z_i}/Z\Big),\, Z=e^{z_1}+e^{z_2}+e^{z_3}$$
这样，盲目地增大比例使得$e^{z_1}/Z$接近于1，就不再是最优解了，从而可以缓解softmax过于自信的情况，不少情况下，这种策略还可以增加测试准确率（防止过拟合）。

那么，在Keras中应该怎么写呢？其实挺简单的：



<pre><code class="language-python">from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
from keras import backend as K

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64

input = Input(shape=(None,))
embedded = Embedding(nb_features,word_size)(input)
encoder = LSTM(encode_size)(embedded)
predict = Dense(nb_classes, activation='softmax')(encoder)

def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/nb_classes, y_pred)
    return (1-e)*loss1 + e*loss2

model = Model(inputs=input, outputs=predict)
model.compile(optimizer='adam', loss=mycrossentropy)
</code></pre>



也就是自定义一个输入为y\_pred,y\_true的loss函数，放进模型compile即可。这里的mycrossentropy，第一项就是普通的交叉熵，第二项中，先通过K\.ones\_like(y\_pred)/nb\_classes构造了一个均匀分布，然后算y\_pred与均匀分布的交叉熵。就这么简单～

## 并不仅仅是输入输出那么简单

前面已经说了，Keras的模型有固定的输入和输出，并且loss即为预测值与真实值的某种误差函数，然而，很多模型并非这样的，比如问答模型与triplet loss。

这个的问题是指有固定的答案库的FAQ形式的问答。一种常见的做问答模型的方法就是：先分别将答案和问题都encode成为一个同样长度的向量，然后比较它们的cos值，cos越大就越匹配。这种做法很容易理解，是一个比较通用的框架，比如这里的问题和答案都不需要一定是问题，图片也行，反正只不过是encode的方法不一样，最终只要能encode出一个向量来即可。但是怎么训练呢？我们当然希望正确答案的cos值越大越好，错误答案的\\cos值越小越好，但是这不是必要的，合理的要求应该是：正确答案的cos值比所有错误答案的\\cos值都要大，大多少无所谓，一丁点都行。因此，这就导致了triplet loss：
$$loss = \max\Big(0, m+\cos(q,A_{\text{wrong}})-\cos(q,A_{\text{right}})\Big)$$
其中$m$是一个大于零的正数。

怎么理解这个loss呢？要注意我们要最小化loss，所以只看$m+\cos(q,A_{\text{wrong}})-\cos(q,A_{\text{right}})$这部分，我们知道目的是拉大正确与错误答案的差距，但是，一旦$\cos(q,A_{\text{right}})-\cos(q,A_{\text{wrong}}) > m$，也就是差距大于$m$时，由于$\max$的存在，loss就等于0，这时候就自动达到最小值，就不会优化它了。所以，triplet loss的思想就是：只希望正确比错误答案的差距大一点（并不是越大越好），超过$m$就别管它了，集中精力关心那些还没有拉开的样本吧！

我们已经有问题和正确答案，错误答案只要随机挑就行，所以这样训练样本是很容易构造的。不过Keras中怎么实现triplet loss呢？看上去是一个单输入、双输出的模型，但并不是那么简单，Keras中的双输出模型，只能给每个输出分别设置一个loss，然后加权求和，但这里不能简单表示成两项的加权求和。那应该要怎么搭建这样的模型呢？下面是一个例子：



<pre><code class="language-python">from keras.layers import Input,Embedding,LSTM,Dense,Lambda
from keras.layers.merge import dot
from keras.models import Model
from keras import backend as K

word_size = 128
nb_features = 10000
nb_classes = 10
encode_size = 64
margin = 0.1

embedding = Embedding(nb_features,word_size)
lstm_encoder = LSTM(encode_size)

def encode(input):
    return lstm_encoder(embedding(input))

q_input = Input(shape=(None,))
a_right = Input(shape=(None,))
a_wrong = Input(shape=(None,))
q_encoded = encode(q_input)
a_right_encoded = encode(a_right)
a_wrong_encoded = encode(a_wrong)

q_encoded = Dense(encode_size)(q_encoded) #一般的做法是，直接讲问题和答案用同样的方法encode成向量后直接匹配，但我认为这是不合理的，我认为至少经过某个变换。

right_cos = dot([q_encoded,a_right_encoded], -1, normalize=True)
wrong_cos = dot([q_encoded,a_wrong_encoded], -1, normalize=True)

loss = Lambda(lambda x: K.relu(margin+x[0]-x[1]))([wrong_cos,right_cos])

model_train = Model(inputs=[q_input,a_right,a_wrong], outputs=loss)
model_q_encoder = Model(inputs=q_input, outputs=q_encoded)
model_a_encoder = Model(inputs=a_right, outputs=a_right_encoded)

model_train.compile(optimizer='adam', loss=lambda y_true,y_pred: y_pred)
model_q_encoder.compile(optimizer='adam', loss='mse')
model_a_encoder.compile(optimizer='adam', loss='mse')

model_train.fit([q,a1,a2], y, epochs=10)
#其中q,a1,a2分别是问题、正确答案、错误答案的batch，y是任意形状为(len(q),1)的矩阵
</code></pre>



如果第一次看不懂，那么请反复阅读几次，这个代码包含了Keras中实现最一般模型的思路：<strong>把目标当成一个输入，构成多输入模型，把loss写成一个层，作为最后的输出，搭建模型的时候，就只需要将模型的output定义为loss，而compile的时候，直接将loss设置为y\_pred（因为模型的输出就是loss，所以y\_pred就是loss），无视y\_true，训练的时候，y\_true随便扔一个符合形状的数组进去就行了。</strong>最后我们得到的是问题和答案的编码器，也就是问题和答案都分别编码出一个向量来，我们只需要比较$\cos$，就可以选择最优答案了。

## Embedding层的妙用

在读这一段之前，请读者务必确定自己对Embedding层有清晰的认识，如果还没有，请移步阅读[《词向量与Embedding究竟是怎么回事？》](<https://spaces.ac.cn/archives/4122/>)。这里需要反复强调的是，虽然词向量叫Word Embedding，但是，Embedding层不是词向量，跟词向量没有半毛钱关系！！！不要有“怎么就跟词向量扯上关系了”这样的傻问题，Embedding层从来就没有跟词向量有过任何直接联系（只不过在训练词向量时可以用它）。对于Embedding层，你可以有两种理解：<u>1、是one hot输入的全连接层的加速版本，也就是说，它就是一个以one hot为输入的Dense层，数学上完全等价；2、它就是一个矩阵查找操作，输入一个整数，输出对应下标的向量，只不过这个矩阵是可训练的。</u>（你看，哪里跟词向量有联系了？）

这部分我们来关心center loss。前面已经说了，做分类时，一般是softmax\+交叉熵做，用矩阵的写法，softmax就是
$$\text{softmax}\Big(\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}\Big)$$
其中$\boldsymbol{x}$可以理解为提取的特征，而$\boldsymbol{W},\boldsymbol{b}$是最后的全连接层的权重，整个模型是一起训练的。问题是，这样的方案所训练出来的特征模型$\boldsymbol{x}$，具有怎样的形态呢？

有一些情况下，我们更关心特征$\boldsymbol{x}$而不是最后的分类结果，比如人脸识别场景，假如我们有10万个不同的人的人脸数据库，每个人有若干张照片，那么我们就可以训练一个10万分类模型，对于给定的照片，我们可以判断它是10万个中的哪一个。但这仅仅是训练场景，那么怎么应用呢？到了具体的应用环境，比如一个公司内部，可能有只有几百人；在公共安全检测场景，可能有数百万人，所以前面做好的10万分类模型基本上是没有意义的，但是在这个模型softmax之前的特征，也就是前一段所说的$\boldsymbol{x}$，可能还是很有意义的。如果对于同一个人（也就是同一类），$\boldsymbol{x}$基本一样，那么实际应用中，我们就可以把训练好的模型当作特征提取工具，然后把提取出来的特征直接用KNN（最邻近距离）来做就行了。

设想很美好，但事实很残酷，直接训练softmax的话，事实上得到的特征不一定具有<strong>聚类特性</strong>，相反，它们会尽量布满整个空间（没有给其他人留出位置，参考center loss的相关论文和文章，比如[这篇](<https://zhuanlan.zhihu.com/p/23340343>)。）。那么，怎样训练才使得结果有聚类特性呢？center loss使用了一种简单粗暴但是却很有效的方案——加聚类惩罚项。完整地写出来，就是
$$loss = - \log\frac{e^{\boldsymbol{W}_y^{\top}\boldsymbol{x}+b_y}}{\sum\limits_i e^{\boldsymbol{W}_i^{\top}\boldsymbol{x}+b_i}} + \lambda \Big\Vert \boldsymbol{x}-\boldsymbol{c}_y \Big\Vert^2$$
其中$y$对应着正确的类别。可以看到，第一项就是普通的softmax交叉熵，第二项就是额外的惩罚项，它给每个类定义了可训练的中心$\boldsymbol{c}$，要求每个类要跟各自的中心靠得很近。所以，总的来说，<strong>第一项负责拉开不同类之间的距离，第二项负责缩小同一类之间的距离</strong>。

那么，Keras中要怎么实现这个方案？关键是，怎么存放聚类中心？答案就是Embedding层！这部分的开头已经提示了，Embedding就是一个待训练的矩阵罢了，正好可以存放聚类中心参数。于是，模仿第二部分的写法，就得到



<pre><code class="language-python">from keras.layers import Input,Conv2D, MaxPooling2D,Flatten,Dense,Embedding,Lambda
from keras.models import Model
from keras import backend as K

nb_classes = 100
feature_size = 32

input_image = Input(shape=(224,224,3))
cnn = Conv2D(10, (2,2))(input_image)
cnn = MaxPooling2D((2,2))(cnn)
cnn = Flatten()(cnn)
feature = Dense(feature_size, activation='relu')(cnn)
predict = Dense(nb_classes, activation='softmax', name='softmax')(feature) #至此，得到一个常规的softmax分类模型

input_target = Input(shape=(1,))
centers = Embedding(nb_classes, feature_size)(input_target) #Embedding层用来存放中心
l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([feature,centers])

model_train = Model(inputs=[input_image,input_target], outputs=[predict,l2_loss])
model_train.compile(optimizer='adam', loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,0.2], metrics={'softmax':'accuracy'})

model_predict = Model(inputs=input_image, outputs=predict)
model_predict.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_train.fit([train_images,train_targets], [train_targets,random_y], epochs=10)
#TIPS：这里用的是sparse交叉熵，这样我们直接输入整数的类别编号作为目标，而不用转成one hot形式。所以Embedding层的输入，跟softmax的目标，都是train_targets，都是类别编号，而random_y是任意形状为(len(train_images),1)的矩阵。
</code></pre>



读者可能有疑问，为什么不像第二部分的triplet loss模型那样，将整体的loss写成一个单一的输出，然后搭建模型，而是要像目前这样变成双输出呢？

事实上，Keras爱好者钟情于Keras，其中一个很重要的原因就是它的进度条——能够实时显示训练loss、训练准确率。<strong>如果像第二部分那样写，那么就不能设置metrics参数，那么训练过程中就不能显示准确率了，这不能说是一个小遗憾。而目前这样写，我们就依然能够在训练过程中看到训练准确率，还能分别看到交叉熵loss、l2\_loss、总的loss分别是多少，非常舒服</strong>。

## Keras就是这么好玩

有了以上三个案例，读者应该对Keras搭建复杂模型的步骤心中有数了，应当说，也是比较简单灵活的。Keras确实有它不够灵活的地方，但也没有网上评论的那么无能。总的来说，Keras是能够满足大多数人快速实验深度学习模型的需求的。如果你还在纠结深度学习框架的选择，那么请选择Keras吧——当你真正觉得Keras不能满足你的需求时，你已经有能力驾驭任何框架了，也就没有这个纠结了。
