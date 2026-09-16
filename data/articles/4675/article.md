# 更别致的词向量模型(四)：模型的求解

> 作者：苏剑林 · 科学空间 · 2017-11-19
>
> 原文：<https://spaces.ac.cn/archives/4675>
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
<h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;损失函数&quot;}">损失函数</h2><p data-spaces-tag="p" data-spaces-attrs="{}">现在，我们来定义loss，以便把各个词向量求解出来。用$\tilde{P}$表示$P$的频率估计值，那么我们可以直接以下式为loss<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\sum_{w_i,w_j}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)^2\tag{16}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
相比之下，无论在参数量还是模型形式上，这个做法都比glove要简单，因此称之为simpler glove。glove模型是<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\sum_{w_i,w_j}\left(\langle \boldsymbol{v}_i, \boldsymbol{\hat{v}}_j\rangle+b_i+\hat{b}_j-\log X_{ij}\right)^2\tag{17}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
在glove模型中，对中心词向量和上下文向量做了区分，然后最后模型建议输出的是两套词向量的求和，据说这效果会更好，这是一个比较勉强的trick，但也不是什么毛病。<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red"><u:0002 data-spaces-tag="u:0002" data-spaces-attrs="{}">最大的问题是参数$b_i,\hat{b}_j$也是可训练的，这使得模型是严重不适定的！</u:0002></font>我们有<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\begin{aligned}&amp;\sum_{w_i,w_j}\left(\langle \boldsymbol{v}_i, \boldsymbol{\hat{v}}_j\rangle+b_i+\hat{b}_j-\log \tilde{P}(w_i,w_j)\right)^2\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;\sum_{w_i,w_j}\left[\langle \boldsymbol{v}_i+\boldsymbol{c}, \boldsymbol{\hat{v}}_j+\boldsymbol{c}\rangle+\Big(b_i-\langle \boldsymbol{v}_i, \boldsymbol{c}\rangle - \frac{|\boldsymbol{c}|^2}{2}\Big)\right.\\<br data-spaces-tag="br" data-spaces-attrs="{}">
&amp;\qquad\qquad\qquad\qquad\left.+\Big(\hat{b}_j-\langle \boldsymbol{\hat{v}}_j, \boldsymbol{c}\rangle - \frac{|\boldsymbol{c}|^2}{2}\Big)-\log X_{ij}\right]^2\end{aligned}\tag{18}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
这就是说，如果你有了一组解，那么你将所有词向量加上任意一个常数向量后，它还是一组解！这个问题就严重了，我们无法预估得到的是哪组解，一旦加上的是一个非常大的常向量，那么各种度量都没意义了（比如任意两个词的cos值都接近1）。事实上，对glove生成的词向量进行验算就可以发现，glove生成的词向量，停用词的模长远大于一般词的模长，也就是说一堆词放在一起时，停用词的作用还明显些，这显然是不利用后续模型的优化的。（虽然从目前的关于glove的实验结果来看，是我强迫症了一些。）</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;互信息估算&quot;}">互信息估算</h2><p data-spaces-tag="p" data-spaces-attrs="{}"></p><p data-spaces-tag="p" data-spaces-attrs="{}">为了求解模型，首先要解决的第一个问题就是$P(w_i,w_j),P(w_i),P(w_j)$该怎么算呢？$P(w_i),P(w_j)$简单，直接统计估计就行了，但$P(w_i,w_j)$呢？怎样的两个词才算是共现了？当然，事实上不同的用途可以有不同的方案，比如我们可以认为同出现在一篇文章的两个词就是碰过一次面了，这种方案通常会对主题分类很有帮助，不过这种方案计算量太大。更常用的方案是选定一个固定的整数，记为window，每个词前后的window个词，都认为是跟这个词碰过面的。</p><p data-spaces-tag="p" data-spaces-attrs="{}">一个值得留意的细节是：<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">中心词与自身的共现要不要算进去？</font>窗口的定义应该是跟中心词距离不超过window的词，那么应该要把它算上的，但如果算上，那没什么预测意义，因为这一项总是存在，如果不算上，那么会降低了词与自身的互信息。所以我们采用了一个小trick：不算入相同的共现项，让模型自己把这个学出来。也就是说，哪怕上下文（除中心词外）也出现了中心词，也不算进loss中，因为数据量本身是远远大于参数量的，所以这一项总可以学习出来。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;权重和降采样&quot;}">权重和降采样</h2><p data-spaces-tag="p" data-spaces-attrs="{}">glove模型定义了如下的权重公式：<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\lambda_{ij}=\Big(\min\{x_{ij}/x_{max}, 1\}\Big)^{\alpha}\tag{19}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
其中$x_{ij}$代表词对$(w_i,w_j)$的共现频数，$x_{max},\alpha$是固定的常数，通常取$x_{max}=100,\alpha=3/4$，也就是说，要对共现频数低的词对降权，它们更有可能是噪音，所以最后Golve的loss是<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\sum_{w_i,w_j}\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle+b_i+b_j-\log \tilde{P}(w_i,w_j)\right)^2\tag{20}\]</p><p data-spaces-tag="p" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red"><u data-spaces-tag="u" data-spaces-attrs="{}">在文本的模型中，继续沿用这一权重，但有所选择</u></font>。首先，对频数作$\alpha$次幂，相当于提高了低频项的权重，这跟word2vec的做法基本一致。值得思考的是$\min$这个截断操作，如果进行这个截断，那么相当于大大降低了高频词的权重，有点像word2vec中的对高频词进行降采样，能够提升低频词的学习效果，但可能带来的后果是：高频词的模长没学好。我们可以在《模长的含义》这一小节中看到这一点。总的来说，不同的场景有不同的需求，因此我们在最后发布的源码中，<u data-spaces-tag="u" data-spaces-attrs="{}">允许用户自定义是否截断这个权重</u>。</p><h2 data-spaces-tag="h2" data-spaces-attrs="{&quot;id&quot;: &quot;Adagrad&quot;}">Adagrad</h2><p data-spaces-tag="p" data-spaces-attrs="{}">跟glove一样，我们同样使用Adagrad算法进行优化，使用Adagrad的原因是因为它大概是目前最简单的自适应学习率的算法。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">但是，我发现glove源码中的Adagrad算法写法是错的！！我不知道glove那样写是刻意的改进，还是笔误（感觉也不大可能笔误吧？），总之，如果我毫不改动它的迭代过程，照搬到本文的simpler glove模型中，很容易就出现各种无解的nan！如果写成标准的Adagrad，nan就不会出现了。</font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">选定一个词对$w_i,w_j$我们得到loss<br data-spaces-tag="br" data-spaces-attrs="{}">
\[L=\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)^2\tag{21}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
它的梯度是<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\begin{aligned}\nabla_{\boldsymbol{v}_i} L=\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)\boldsymbol{v}_j\\<br data-spaces-tag="br" data-spaces-attrs="{}">
\nabla_{\boldsymbol{v}_j} L=\lambda_{ij}\left(\langle \boldsymbol{v}_i, \boldsymbol{v}_j\rangle-\log\frac{\tilde{P}(w_i,w_j)}{\tilde{P}(w_i)\tilde{P}(w_j)}\right)\boldsymbol{v}_i<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{aligned}\tag{22}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
然后根据Adagrad算法的公式进行更新即可，默认的初始学习率选为$\eta=0.1$，迭代公式为<br data-spaces-tag="br" data-spaces-attrs="{}">
\[\left\{\begin{aligned}\boldsymbol{g}_{\gamma}^{(n)} =&amp; \nabla_{\boldsymbol{v}_{\gamma}^{(n)}} L\\<br data-spaces-tag="br" data-spaces-attrs="{}">
\boldsymbol{G}_{\gamma}^{(n)} =&amp; \boldsymbol{G}_{\gamma}^{(n-1)} + \boldsymbol{g}_{\gamma}^{(n)}\otimes \boldsymbol{g}_{\gamma}^{(n)}\\<br data-spaces-tag="br" data-spaces-attrs="{}">
\boldsymbol{v}_{\gamma}^{(n)} =&amp; \boldsymbol{v}_{\gamma}^{(n-1)} - \frac{\boldsymbol{g}_{\gamma}^{(n)}}{\sqrt{\boldsymbol{G}_{\gamma}^{(n-1)}}}\eta<br data-spaces-tag="br" data-spaces-attrs="{}">
\end{aligned}\right.,\,\gamma=i,j<br data-spaces-tag="br" data-spaces-attrs="{}">
\tag{23}\]<br data-spaces-tag="br" data-spaces-attrs="{}">
根据公式可以看出，Adagrad算法基本上是对loss的缩放不敏感的，换句话说，将loss乘上10倍，最终的优化效果基本没什么变化，但如果在随机梯度下降中，将loss乘上10倍，就等价于将学习率乘以10了。</p>
</div>
