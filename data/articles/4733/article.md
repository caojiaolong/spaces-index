# 从loss的硬截断、软化到focal loss

> 作者：苏剑林 · 科学空间 · 2017-12-25
>
> 原文：<https://spaces.ac.cn/archives/4733>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## 前言

今天在QQ群里的讨论中看到了focal loss，经搜索它是Kaiming大神团队在他们的论文[《Focal Loss for Dense Object Detection》](<https://papers.cool/arxiv/1708.02002>)提出来的损失函数，利用它改善了图像物体检测的效果。不过我很少做图像任务，不怎么关心图像方面的应用。本质上讲，focal loss就是一个解决分类问题中类别不平衡、分类难度差异的一个loss，总之这个工作一片好评就是了。大家还可以看知乎的讨论：
[《如何评价kaiming的Focal Loss for Dense Object Detection？》](<https://www.zhihu.com/question/63581984>)

看到这个loss，开始感觉很神奇，感觉大有用途。因为<u>在NLP中，也存在大量的类别不平衡的任务</u>。最经典的就是序列标注任务中类别是严重不平衡的，比如在命名实体识别中，显然一句话里边实体是比非实体要少得多，这就是一个类别严重不平衡的情况。我尝试把它用在我的基于序列标注的问答模型中，也有微小提升。嗯，这的确是一个好loss。

接着我再仔细对比了一下，我发现<u>这个loss跟我昨晚构思的一个loss具有异曲同工之理</u>！这就促使我写这篇博文了。我将从我自己的思考角度出发，来分析这个问题，最后得到focal loss，也给出我昨晚得到的类似的loss。

## 硬截断

整篇文章都是从二分类问题出发，同样的思想可以用于多分类问题。二分类问题的标准loss是交叉熵
$$L_{ce} = -y\log \hat{y} - (1-y)\log(1-\hat{y})=\left\{\begin{aligned}&-\log(\hat{y}),\,\text{当}y=1\\ &-\log(1-\hat{y}),\,\text{当}y=0\end{aligned}\right.$$
其中$y\in\{0,1\}$是真实标签，$\hat{y}$是预测值。当然，对于二分类我们几乎都是用sigmoid函数激活$\hat{y}=\sigma(x)$，所以相当于
$$L_{ce} = -y\log \sigma(x) - (1-y)\log\sigma(-x)=\left\{\begin{aligned}&-\log \sigma(x),\,\text{当}y=1\\ &-\log\sigma(-x),\,\text{当}y=0\end{aligned}\right.$$
（我们有$1-\sigma(x)=\sigma(-x)$。）

我在上半年的博文[《文本情感分类（四）：更好的损失函数》](<https://spaces.ac.cn/archives/4293/>)中，曾经针对“集中精力关注难分样本”这个想法提出了一个“硬截断”的loss，形式为
$$L^\cdot = \lambda(y,\hat{y})\cdot L_{ce}$$
其中
$$\lambda(y,\hat{y})=\left\{\begin{aligned}&0,\,(y=1\text{且}\hat{y} > 0.5)\text{或}(y=0\text{且}\hat{y} < 0.5)\\ &1,\,\text{其他情形}\end{aligned}\right.$$
这样的做法就是：<u>正样本的预测值大于0\.5的，或者负样本的预测值小于0\.5的，我都不更新了，把注意力集中在预测不准的那些样本，当然这个阈值可以调整。</u>这样做能部分地达到目的，但是所需要的迭代次数会大大增加。

原因是这样的：以正样本为例，<strong>我只告诉模型正样本的预测值大于0\.5就不更新了，却没有告诉它<u>要“保持”</u>大于0\.5</strong>，所以下一阶段，它的预测值就很有可能变回小于0\.5了，当然，如果是这样的话，下一回合它又被更新了，这样反复迭代，理论上也能达到目的，但是迭代次数会大大增加。所以，要想改进的话，重点就是“不只是要告诉模型正样本的预测值大于0\.5就不更新了，而是要告诉模型当其大于0\.5后就只需要保持就好了”。（好比老师看到一个学生及格了就不管了，这显然是不行的。如果学生已经及格，那么应该要想办法要他保持目前这个状态甚至变得更好，而不是不管。）

## 软化loss

硬截断会出现不足，关键地方在于因子$\lambda(y,\hat{y})$是不可导的，或者说我们认为它导数为0，因此这一项不会对梯度有任何帮助，从而我们不能从它这里得到合理的反馈（也就是模型不知道“保持”意味着什么）。

解决这个问题的一个方法就是“软化”这个loss，<u>“软化”就是把一些本来不可导的函数用一些可导函数来近似</u>，数学角度应该叫“光滑化”。这样处理之后本来不可导的东西就可导了，类似的算例还有[《梯度下降和EM算法：系出同源，一脉相承》](<https://spaces.ac.cn/archives/4277/>)中的kmeans部分。我们首先改写一下$L^*$。
$$L^\cdot =\left\{\begin{aligned}&-\theta(0.5-\hat{y})\log(\hat{y}),\,\text{当}y=1\\ &-\theta(\hat{y}-0.5)\log(1-\hat{y}),\,\text{当}y=0\end{aligned}\right.$$
这里的$\theta$就是单位阶跃函数
$$\theta(x) = \left\{\begin{aligned}&1, x > 0\\
&\frac{1}{2}, x = 0\\
&0, x < 0\end{aligned}\right.$$
这样的$L^*$跟原来的是完全等价的，它也等价于（因为$\sigma(0)=0.5$）
$$L^\cdot =\left\{\begin{aligned}&-\theta(-x)\log \sigma(x),\,\text{当}y=1\\ &-\theta(x)\log\sigma(-x),\,\text{当}y=0\end{aligned}\right.$$
这时候思路就很明显了，要想“软化”这个loss，就得“软化”$\theta(x)$，而<u>软化它就再容易不过，它就是sigmoid函数</u>！我们有
$$\theta(x) = \lim_{K\to +\infty} \sigma(Kx)$$
所以很显然，我们将$\theta(x)$替换为$\sigma(Kx)$即可：
$$L^{\cdot \cdot }=\left\{\begin{aligned}&-\sigma(-Kx)\log \sigma(x),\,\text{当}y=1\\ &-\sigma(Kx)\log\sigma(-x),\,\text{当}y=0\end{aligned}\right.$$
这就是我昨晚思考得到的loss了，显然实现上也是很容易的。

现在跟focal loss做个比较。

## Focal Loss

Kaiming大神的focal loss形式是
$$L_{fl}=\left\{\begin{aligned}&-(1-\hat{y})^{\gamma}\log \hat{y},\,\text{当}y=1\\ &-\hat{y}^{\gamma}\log (1-\hat{y}),\,\text{当}y=0\end{aligned}\right.$$
如果落实到$\hat{y}=\sigma(x)$这个预测，那么就有
$$L_{fl}=\left\{\begin{aligned}&-\sigma^{\gamma}(-x)\log \sigma(x),\,\text{当}y=1\\ &-\sigma^{\gamma}(x)\log\sigma(-x),\,\text{当}y=0\end{aligned}\right.$$
特别地，<strong>如果$K$和$\gamma$都取1，那么$L^{**}=L_{fl}$！</strong>

事实上$K$和$\gamma$的作用都是一样的，都是调节权重曲线的陡度，只是调节的方式不一样～注意$L^{**}$或$L_{fl}$实际上都已经包含了对不均衡样本的解决方法，或者说，类别不均衡本质上就是分类难度差异的体现。<u>比如负样本远比正样本多的话，模型肯定会倾向于数目多的负类（可以想象全部样本都判为负类），这时候，负类的$\hat{y}^{\gamma}$或$\sigma(Kx)$都很小，而正类的$(1-\hat{y})^{\gamma}$或$\sigma(-Kx)$就很大，这时候模型就会开始集中精力关注正样本。</u>

当然，Kaiming大神还发现对$L_{fl}$做个权重调整，结果会有微小提升
$$L_{fl}=\left\{\begin{aligned}&-\alpha(1-\hat{y})^{\gamma}\log \hat{y},\,\text{当}y=1\\ &-(1-\alpha)\hat{y}^{\gamma}\log (1-\hat{y}),\,\text{当}y=0\end{aligned}\right.$$
通过一系列调参，得到$\alpha=0.25,\gamma=2$（在他的模型上）的效果最好。注意在他的任务中，正样本是属于少数样本，也就是说，本来正样本难以“匹敌”负样本，但经过$(1-\hat{y})^{\gamma}$和$\hat{y}^{\gamma}$的“操控”后，也许形势还逆转了，还要对正样本降权。不过我认为这样调整只是经验结果，理论上很难有一个指导方案来决定$\alpha$的值，如果没有大算力调参，倒不如直接让$\alpha=0.5$（均等）。

## 多分类

focal loss在多分类中的形式也很容易得到，其实就是
$$L_{fl}=-(1-\hat{y}_t)^{\gamma}\log \hat{y}_t$$
$\hat{y}_t$是目标的预测值，一般就是经过softmax后的结果。那我自己构思的$L^{**}$怎么推广到多分类？也很简单：
$$L^{\cdot \cdot }=-\text{softmax}(-Kx_t)\log \text{softmax}(x_t)$$
这里$x_t$也是目标的预测值，但它是softmax前的结果。

## 结语

<strong>什么？你得到了跟Kaiming大神一样想法的东西？不不不，本文只是对Kaiming大神的focal loss的一个介绍而已，更准确地说，是应对分类不平衡、分类难度差异的一些方案的介绍，并尽可能给出自己的看法而已。当然，本文这样的写法难免有附庸风雅、东施效颦之嫌，请读者海涵。</strong>
