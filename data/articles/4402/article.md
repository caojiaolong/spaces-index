# 【不可思议的Word2Vec】5\. Tensorflow版的Word2Vec

> 作者：苏剑林 · 科学空间 · 2017-05-27
>
> 原文：<https://spaces.ac.cn/archives/4402>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">本文封装了一个比较完整的Word2Vec，其模型部分使用tensorflow实现。本文的目的并非只是再造一次Word2Vec这个轮子，而是通过这个例子来熟悉tensorflow的写法，并且测试笔者设计的一种新的softmax loss的效果，为后面研究语言模型的工作做准备。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;不同的地方&quot;}">不同的地方</h2><p data-spaces-tag="p" data-spaces-attrs="{}">Word2Vec的基本的数学原理，请移步到<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/4299/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/4299/">《【不可思议的Word2Vec】 1.数学原理》</a>一文查看。本文的主要模型还是CBOW或者Skip-Gram，但在loss设计上有所不同。本文还是使用了完整的softmax结构，而不是huffmax softmax或者负采样方案，但是在训练softmax时，使用了<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">基于随机负采样的交叉熵</font>作为loss。这种loss与已有的nce_loss和sampled_softmax_loss都不一样，这里姑且命名为random softmax loss。</p><p data-spaces-tag="p" data-spaces-attrs="{}">另外，在softmax结构中，一般是$\text{softmax}(Wx+b)$这样的形式，考虑到$W$矩阵的形状事实上跟词向量矩阵的形状是一样的，因此本文考虑了<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">softmax层与词向量层共享权重</font>的模型（这时候直接让$b$为0），这种模型等效于原有的Word2Vec的<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">负采样</font>方案<del data-spaces-tag="del" data-spaces-attrs="{}">，也类似于glove词向量的</del><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">词共现矩阵分解</font>，但由于使用了交叉熵损失，理论上收敛更快，而且训练结果依然具有softmax的预测概率意义（相比之下，已有的Word2Vec负样本模型训练完之后，最后模型的输出值是没有意义的，只有词向量是有意义的。）。同时，由于共享了参数，因此词向量的更新更为充分，读者不妨多多测试这种方案。</p><p data-spaces-tag="p" data-spaces-attrs="{}">所以，本文事实上也实现了4个模型组合：CBOW/Skip-Gram，是/否共享softmax层的参数，读者可以选择使用。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;loss是怎么来的&quot;}">loss是怎么来的</h2><p data-spaces-tag="p" data-spaces-attrs="{}">前面已经说了，本文的主要目的之一就是测试新loss的有效性，下面简介一下这个loss的来源和形式。这要从softmax为什么训练难度大开始说起～</p><p data-spaces-tag="p" data-spaces-attrs="{}">假设标签数（本文中也就是词表中词汇量）为$n$，那么<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}(p_1,p_2,\dots,p_n) =&amp; \text{softmax}(z_1,z_2,\dots,z_n)\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp; \left(\frac{e^{z_1}}{Z}, \frac{e^{z_2}}{Z}, \dots, \frac{e^{z_n}}{Z}\right)\end{aligned}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这里$Z = e^{z_1} + e^{z_2} + \dots + e^{z_n}$。如果正确类别标签为$t$，使用交叉熵为loss，则<br data-spaces-tag="br" data-spaces-attrs="{}">
$$L=-\log \frac{e^{z_t}}{Z}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
梯度为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\nabla L=-\nabla z_t + \nabla (\log Z)=-\nabla z_t + \frac{\nabla Z}{Z}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
因为有$Z$的存在，每次梯度下降时，都要计算完整的$Z$来计算$\nabla Z$，也就是每一个样本迭代一次的计算量就是$\mathcal{O}(n)$了，对于$n$比较大的情形，这是难以接受的，所以要寻求近似方案（huffman softmax是其中的一种，但是它写起来会比较复杂，而且huffman softmax的结果通常略差于普通的softmax，还有huffman softmax仅仅是训练快，但是如果预测时要找最大概率的标签，那么反而更加慢）。</p><p data-spaces-tag="p" data-spaces-attrs="{}">让我们进一步计算$\nabla L$：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}\nabla L=&amp;-\nabla z_t + \frac{\sum_i e^{z_i}\nabla z_i}{Z}\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;-\nabla z_t + \frac{\sum_i e^{z_i}}{Z}\nabla z_i\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;-\nabla z_t + \sum_i p_i \nabla z_i\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;-\nabla z_t + \text{E}(\nabla z_i)<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{aligned}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
也就是说，最后的梯度由两项组成：<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">一项是正确标签的梯度，一项是所有标签的梯度的均值，这两项反号，可以理解为这两项在“拉锯战”</font>。计算量主要集中在第二项，因为要遍历所有才能算具体的均值。然而，均值本身就具有概率意义的，那么能不能直接就随机选取若干个来算这个梯度均值，而不用算全部梯度呢？如果可以的话，那每步更新的计算量就固定了，不会随着标签数的增大而快速增加。</p><p data-spaces-tag="p" data-spaces-attrs="{}">但如果这样做的话，需要按概率来随机选取标签，这也并不容易写。然而，有个更巧妙的方法是不用我们直接算梯度，我们可以直接对loss动手脚。所以这就导致了本文的loss：<strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">对于每个“样本-标签”对，随机选取nb_negative个标签，然后与原来的标签组成nb_negative+1个标签，直接在这nb_negative+1个标签中算softmax和交叉熵。选取这样的loss之后再去算梯度，就会发现自然就是按概率来选取的梯度均值了</font></strong>。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;代码实现&quot;}">代码实现</h2><p data-spaces-tag="p" data-spaces-attrs="{}">自我感觉代码还是比较精炼的，单文件包含了所有内容。训练输出模仿了gensim中的Word2Vec。模型代码位于GitHub：<br data-spaces-tag="br" data-spaces-attrs="{}">
<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://github.com/bojone/tf_word2vec/blob/master/Word2Vec.py&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://github.com/bojone/tf_word2vec/blob/master/Word2Vec.py">https://github.com/bojone/tf_word2vec/blob/master/Word2Vec.py</a></p><p data-spaces-tag="p" data-spaces-attrs="{}">使用参考：</p><pre data-spaces-tag="pre" data-spaces-attrs="{&quot;class&quot;: [&quot;line-numbers&quot;]}"><code data-spaces-tag="code" data-spaces-attrs="{&quot;class&quot;: [&quot;language-python&quot;]}">from Word2Vec import *
import pymongo
db = pymongo.MongoClient().travel.articles
class texts:
    def __iter__(self):
        for t in db.find().limit(30000):
            yield t['words']

wv = Word2Vec(texts(), model='cbow', nb_negative=16, shared_softmax=True, epochs=2) #建立并训练模型
wv.save_model('myvec') #保存到当前目录下的myvec文件夹

#训练完成后可以这样调用
wv = Word2Vec() #建立空模型
wv.load_model('myvec') #从当前目录下的myvec文件夹加载模型
</code></pre><p data-spaces-tag="p" data-spaces-attrs="{}">有几点需要说明的：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}">1、训练的输入是分好词的句子，可以是列表，也可以是迭代器（class+__iter__），注意，不能是生成器（函数+yield），这跟gensim版的word2vec的要求是一致的。因为生成器只能遍历一次，而训练word2vec需要多次遍历数据；</p><p data-spaces-tag="p" data-spaces-attrs="{}">2、模型不支持更新式训练，即训练完模型后，不能再用额外的文档更新原来的模型（不是不可以，是没必要，而且意义不大）；</p><p data-spaces-tag="p" data-spaces-attrs="{}">3、训练模型需要tensorflow，推荐用GPU加速，训练完成后，<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">重新加载模型并且使用模型，都不需要tensorflow，只需要numpy</font>；</p><p data-spaces-tag="p" data-spaces-attrs="{}">4、对于迭代次数，一般迭代1～2次就够了，负样本个数10～30即可。其余参数如batch_size，可以自己实验调整。</p></blockquote><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;简单的对比实验&quot;}">简单的对比实验</h2><p data-spaces-tag="p" data-spaces-attrs="{}">tensorflow中，已有的两种近似训练softmax的loss是nce_loss和sampled_softmax_loss，这里简单做一个比较。在一个旅游领域的语料中（两万多篇文章）训练同样的模型，并比较结果。模型cbow，softmax选择不共享词向量层，其余参数都采用相同的默认参数。</p><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;random_softmax_loss&quot;}">random_softmax_loss</h3><p data-spaces-tag="p" data-spaces-attrs="{}">耗时：8分19秒（迭代次数2次，batch_size为8000）</p><p data-spaces-tag="p" data-spaces-attrs="{}">相似度测试结果</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; import pandas as pd<br data-spaces-tag="br" data-spaces-attrs="{}">
&gt;&gt;&gt; pd.Series(wv.most_similar(u'水果'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (食品, 0.767908)<br data-spaces-tag="br" data-spaces-attrs="{}">
1     (鱼干, 0.762363)<br data-spaces-tag="br" data-spaces-attrs="{}">
2     (椰子, 0.750326)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (饮料, 0.722811)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (食物, 0.719381)<br data-spaces-tag="br" data-spaces-attrs="{}">
5    (牛肉干, 0.715441)<br data-spaces-tag="br" data-spaces-attrs="{}">
6     (菠萝, 0.715354)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (火腿肠, 0.714509)<br data-spaces-tag="br" data-spaces-attrs="{}">
8    (菠萝蜜, 0.712546)<br data-spaces-tag="br" data-spaces-attrs="{}">
9    (葡萄干, 0.709274)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'自然'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0      (人文, 0.645445)<br data-spaces-tag="br" data-spaces-attrs="{}">
1      (和谐, 0.634387)<br data-spaces-tag="br" data-spaces-attrs="{}">
2       (包容, 0.61829)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (大自然, 0.601749)<br data-spaces-tag="br" data-spaces-attrs="{}">
4    (自然环境, 0.588165)<br data-spaces-tag="br" data-spaces-attrs="{}">
5       (融, 0.579027)<br data-spaces-tag="br" data-spaces-attrs="{}">
6      (博大, 0.574943)<br data-spaces-tag="br" data-spaces-attrs="{}">
7      (诠释, 0.550352)<br data-spaces-tag="br" data-spaces-attrs="{}">
8      (野性, 0.548001)<br data-spaces-tag="br" data-spaces-attrs="{}">
9      (野趣, 0.545887)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'广州'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0    (上海, 0.749281)<br data-spaces-tag="br" data-spaces-attrs="{}">
1    (武汉, 0.730211)<br data-spaces-tag="br" data-spaces-attrs="{}">
2    (深圳, 0.703333)<br data-spaces-tag="br" data-spaces-attrs="{}">
3    (长沙, 0.683243)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (福州, 0.68216)<br data-spaces-tag="br" data-spaces-attrs="{}">
5    (合肥, 0.673027)<br data-spaces-tag="br" data-spaces-attrs="{}">
6    (北京, 0.669859)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (重庆, 0.653501)<br data-spaces-tag="br" data-spaces-attrs="{}">
8    (海口, 0.647563)<br data-spaces-tag="br" data-spaces-attrs="{}">
9    (天津, 0.642161)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'风景'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0      (景色, 0.825557)<br data-spaces-tag="br" data-spaces-attrs="{}">
1      (美景, 0.763399)<br data-spaces-tag="br" data-spaces-attrs="{}">
2      (景致, 0.734687)<br data-spaces-tag="br" data-spaces-attrs="{}">
3      (风光, 0.727672)<br data-spaces-tag="br" data-spaces-attrs="{}">
4       (景观, 0.57638)<br data-spaces-tag="br" data-spaces-attrs="{}">
5    (湖光山色, 0.573512)<br data-spaces-tag="br" data-spaces-attrs="{}">
6      (山景, 0.555502)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (美不胜收, 0.552739)<br data-spaces-tag="br" data-spaces-attrs="{}">
8      (明仕, 0.535922)<br data-spaces-tag="br" data-spaces-attrs="{}">
9       (沿途, 0.53485)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'酒楼'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (酒家, 0.768179)<br data-spaces-tag="br" data-spaces-attrs="{}">
1     (排挡, 0.731749)<br data-spaces-tag="br" data-spaces-attrs="{}">
2    (火锅店, 0.729214)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (排档, 0.726048)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (餐馆, 0.722667)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (面馆, 0.715188)<br data-spaces-tag="br" data-spaces-attrs="{}">
6    (大排档, 0.709883)<br data-spaces-tag="br" data-spaces-attrs="{}">
7     (名店, 0.708996)<br data-spaces-tag="br" data-spaces-attrs="{}">
8    (松鹤楼, 0.705759)<br data-spaces-tag="br" data-spaces-attrs="{}">
9     (分店, 0.705749)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'酒店'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (万豪, 0.722409)<br data-spaces-tag="br" data-spaces-attrs="{}">
1    (希尔顿, 0.713292)<br data-spaces-tag="br" data-spaces-attrs="{}">
2     (五星, 0.697638)<br data-spaces-tag="br" data-spaces-attrs="{}">
3    (五星级, 0.696659)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (凯莱, 0.694978)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (银泰, 0.693179)<br data-spaces-tag="br" data-spaces-attrs="{}">
6    (大酒店, 0.692239)<br data-spaces-tag="br" data-spaces-attrs="{}">
7      (宾馆, 0.67907)<br data-spaces-tag="br" data-spaces-attrs="{}">
8    (喜来登, 0.668638)<br data-spaces-tag="br" data-spaces-attrs="{}">
9     (假日, 0.662169)</p></blockquote><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;nce_loss&quot;}">nce_loss</h3><p data-spaces-tag="p" data-spaces-attrs="{}">耗时：4分钟（迭代次数2次，batch_size为8000），然而相似度测试结果简直不堪入目，当然，考虑到用时变少了，为了公平，将迭代次数增加到4次，其余参数不变，重复跑一次。相似度测试结果依旧一塌糊涂，比如：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'水果'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0      (口, 0.940704)<br data-spaces-tag="br" data-spaces-attrs="{}">
1      (可, 0.940106)<br data-spaces-tag="br" data-spaces-attrs="{}">
2    (100, 0.939276)<br data-spaces-tag="br" data-spaces-attrs="{}">
3      (变, 0.938824)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (第二, 0.938155)<br data-spaces-tag="br" data-spaces-attrs="{}">
5      (：, 0.938088)<br data-spaces-tag="br" data-spaces-attrs="{}">
6      (见, 0.937939)<br data-spaces-tag="br" data-spaces-attrs="{}">
7     (不好, 0.937616)<br data-spaces-tag="br" data-spaces-attrs="{}">
8      (和, 0.937535)<br data-spaces-tag="br" data-spaces-attrs="{}">
9      (（, 0.937383)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">有点怀疑是不是我使用姿势不对了～于是我再次调整，将nb_negative增加到1000，然后迭代次数调回为3，这样耗时为9分17秒，最后的loss比前面的要小一个数量级，比较相似度的结果有些靠谱了，但还是并非特别好，比如：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'水果'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0    (特产, 0.984775)<br data-spaces-tag="br" data-spaces-attrs="{}">
1    (海鲜, 0.981409)<br data-spaces-tag="br" data-spaces-attrs="{}">
2    (之类, 0.981158)<br data-spaces-tag="br" data-spaces-attrs="{}">
3    (食品, 0.980803)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (。, 0.980371)<br data-spaces-tag="br" data-spaces-attrs="{}">
5    (蔬菜, 0.979822)<br data-spaces-tag="br" data-spaces-attrs="{}">
6     (&amp;, 0.979713)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (芒果, 0.979599)<br data-spaces-tag="br" data-spaces-attrs="{}">
8     (可, 0.979486)<br data-spaces-tag="br" data-spaces-attrs="{}">
9    (比如, 0.978958)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'自然'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (与, 0.985322)<br data-spaces-tag="br" data-spaces-attrs="{}">
1    (地处, 0.984874)<br data-spaces-tag="br" data-spaces-attrs="{}">
2    (这些, 0.983769)<br data-spaces-tag="br" data-spaces-attrs="{}">
3    (夫人, 0.983499)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (里, 0.983473)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (的, 0.983456)<br data-spaces-tag="br" data-spaces-attrs="{}">
6     (将, 0.983432)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (故居, 0.983328)<br data-spaces-tag="br" data-spaces-attrs="{}">
8    (那些, 0.983089)<br data-spaces-tag="br" data-spaces-attrs="{}">
9    (这里, 0.983046)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p></blockquote><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;sampled_softmax_loss&quot;}">sampled_softmax_loss</h3><p data-spaces-tag="p" data-spaces-attrs="{}">有了前面的经验，这次直接将nb_negative设为1000，然后迭代次数为3，这样耗时为8分38秒，相似度比较的结果是：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'水果'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0      (零食, 0.69762)<br data-spaces-tag="br" data-spaces-attrs="{}">
1     (食品, 0.651911)<br data-spaces-tag="br" data-spaces-attrs="{}">
2     (巧克力, 0.64101)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (葡萄, 0.636065)<br data-spaces-tag="br" data-spaces-attrs="{}">
4      (饼干, 0.62631)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (面包, 0.613488)<br data-spaces-tag="br" data-spaces-attrs="{}">
6    (哈密瓜, 0.604927)<br data-spaces-tag="br" data-spaces-attrs="{}">
7     (食物, 0.602576)<br data-spaces-tag="br" data-spaces-attrs="{}">
8     (干货, 0.601015)<br data-spaces-tag="br" data-spaces-attrs="{}">
9     (菠萝, 0.598993)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'自然'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (人文, 0.577503)<br data-spaces-tag="br" data-spaces-attrs="{}">
1    (大自然, 0.537344)<br data-spaces-tag="br" data-spaces-attrs="{}">
2     (景观, 0.526281)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (田园, 0.526062)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (独特, 0.526009)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (和谐, 0.503326)<br data-spaces-tag="br" data-spaces-attrs="{}">
6     (旖旎, 0.498782)<br data-spaces-tag="br" data-spaces-attrs="{}">
7     (无限, 0.491521)<br data-spaces-tag="br" data-spaces-attrs="{}">
8     (秀美, 0.482407)<br data-spaces-tag="br" data-spaces-attrs="{}">
9     (一派, 0.479687)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'广州'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (深圳, 0.771525)<br data-spaces-tag="br" data-spaces-attrs="{}">
1     (上海, 0.739744)<br data-spaces-tag="br" data-spaces-attrs="{}">
2     (东莞, 0.726057)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (沈阳, 0.687548)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (福州, 0.654641)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (北京, 0.650491)<br data-spaces-tag="br" data-spaces-attrs="{}">
6    (动车组, 0.644898)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (乘动车, 0.635638)<br data-spaces-tag="br" data-spaces-attrs="{}">
8     (海口, 0.631551)<br data-spaces-tag="br" data-spaces-attrs="{}">
9     (长春, 0.628518)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'风景'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0        (景色, 0.8393)<br data-spaces-tag="br" data-spaces-attrs="{}">
1      (景致, 0.731151)<br data-spaces-tag="br" data-spaces-attrs="{}">
2      (风光, 0.730255)<br data-spaces-tag="br" data-spaces-attrs="{}">
3      (美景, 0.666185)<br data-spaces-tag="br" data-spaces-attrs="{}">
4      (雪景, 0.554452)<br data-spaces-tag="br" data-spaces-attrs="{}">
5      (景观, 0.530444)<br data-spaces-tag="br" data-spaces-attrs="{}">
6    (湖光山色, 0.529671)<br data-spaces-tag="br" data-spaces-attrs="{}">
7      (山景, 0.511195)<br data-spaces-tag="br" data-spaces-attrs="{}">
8      (路况, 0.490073)<br data-spaces-tag="br" data-spaces-attrs="{}">
9    (风景如画, 0.483742)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object<br data-spaces-tag="br" data-spaces-attrs="{}">
&gt;&gt;&gt; pd.Series(wv.most_similar(u'酒楼'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0      (酒家, 0.766124)<br data-spaces-tag="br" data-spaces-attrs="{}">
1      (菜馆, 0.687775)<br data-spaces-tag="br" data-spaces-attrs="{}">
2      (食府, 0.666957)<br data-spaces-tag="br" data-spaces-attrs="{}">
3      (饭店, 0.664034)<br data-spaces-tag="br" data-spaces-attrs="{}">
4      (川味, 0.659254)<br data-spaces-tag="br" data-spaces-attrs="{}">
5      (饭馆, 0.658057)<br data-spaces-tag="br" data-spaces-attrs="{}">
6      (排挡, 0.656883)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (粗茶淡饭, 0.650861)<br data-spaces-tag="br" data-spaces-attrs="{}">
8     (共和春, 0.650256)<br data-spaces-tag="br" data-spaces-attrs="{}">
9      (餐馆, 0.644265)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p><p data-spaces-tag="p" data-spaces-attrs="{}">&gt;&gt;&gt; pd.Series(wv.most_similar(u'酒店'))<br data-spaces-tag="br" data-spaces-attrs="{}">
0     (宾馆, 0.685888)<br data-spaces-tag="br" data-spaces-attrs="{}">
1    (大酒店, 0.678389)<br data-spaces-tag="br" data-spaces-attrs="{}">
2     (四星, 0.638032)<br data-spaces-tag="br" data-spaces-attrs="{}">
3     (五星, 0.633661)<br data-spaces-tag="br" data-spaces-attrs="{}">
4     (汉庭, 0.619405)<br data-spaces-tag="br" data-spaces-attrs="{}">
5     (如家, 0.614918)<br data-spaces-tag="br" data-spaces-attrs="{}">
6     (大堂, 0.612269)<br data-spaces-tag="br" data-spaces-attrs="{}">
7    (度假村, 0.610618)<br data-spaces-tag="br" data-spaces-attrs="{}">
8    (四星级, 0.609796)<br data-spaces-tag="br" data-spaces-attrs="{}">
9     (天域, 0.598987)<br data-spaces-tag="br" data-spaces-attrs="{}">
dtype: object</p></blockquote><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;总结&quot;}">总结</h3><p data-spaces-tag="p" data-spaces-attrs="{}">这个实验虽然不怎么严谨，但是应该可以说，在相同的训练时间下，从相似度任务来看，<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">感觉上random softmax与sampled softmax效果相当，nce loss的效果最差，进一步压缩迭代次数，调整参数也表明了类似结果</font>，欢迎读者进一步测试。由于本文的random softmax对每一个样本都进行不同的采样，因此所需要采样的负样本数更少，并且采样更加充分。</p><p data-spaces-tag="p" data-spaces-attrs="{}">至于其他任务的比较，只能在以后的实践中进行了。毕竟这不是发论文，我也懒得做了～</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;后续工作&quot;}">后续工作</h2><p data-spaces-tag="p" data-spaces-attrs="{}">一个问题是：为啥跟sampled softmax效果相当，我还要造新的loss？其实原因很简单的，我看sampled softmax的论文和公式，总感觉它不大好看，理论不够漂亮，当然，就效果来看，是我太强迫症了。本文就算是强迫症的产物吧，也算是练手tensorflow。</p><p data-spaces-tag="p" data-spaces-attrs="{}">另外，<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/4374/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/4374/">《记录一次半监督的情感分析》</a>一文表明，语言模型在预训练模型、实现半监督等任务中有着重要的潜力，甚至词向量也不过就是语言模型来预训练的第一层参数而已。所以笔者想抽空深入一下类似的内容。本文就是为了这个所做的准备之一了。</p>
</div>
