# 再谈类别不平衡问题：调节权重与魔改Loss的对比联系

> 作者：苏剑林 · 科学空间 · 2020-08-31
>
> 原文：<https://spaces.ac.cn/archives/7708>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

类别不平衡问题，也称为长尾分布问题，在本博客里已经有好几次相关讨论了，比如[《从loss的硬截断、软化到focal loss》](<https://spaces.ac.cn/archives/4733>)、[《将“Softmax\+交叉熵”推广到多标签分类问题》](<https://spaces.ac.cn/archives/7359>)、[《通过互信息思想来缓解类别不平衡问题》](<https://spaces.ac.cn/archives/7615>)。对于缓解类别不平衡，比较基本的方法就是调节样本权重，看起来“高端”一点的方法则是各种魔改loss了（比如Focal Loss、Dice Loss、Logits Adjustment等），本文希望比较系统地理解一下它们之间的联系。

[![长尾分布：少数类别的样本数目非常多，多数类别的样本数目非常少。](<https://spaces.ac.cn/usr/uploads/2020/08/2288869465.png>)](<https://spaces.ac.cn/usr/uploads/2020/08/2288869465.png>)

长尾分布：少数类别的样本数目非常多，多数类别的样本数目非常少。

## 从光滑准确率到交叉熵

这里的分析主要以sigmoid的2分类为主，但多数结论可以平行推广到softmax的多分类。设$x$为输入，$y\in\{0,1\}$为目标，$p_{\theta}(x) \in [0, 1]$为模型。理想情况下，当然是要评测什么指标，我们就去优化那个指标。对于分类问题来说，最朴素的指标当然就是准确率，但准确率并没有办法提供有效的梯度，所以不能直接来训练。

为此，我们一个光滑化的指标。从之前的文章[《函数光滑化杂谈：不可导函数的可导逼近》](<https://spaces.ac.cn/archives/6620>)，准确率的光滑化近似是
\begin{equation}\text{ACC}_{\text{smooth}}=\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y p_{\theta}(x) + (1 - y)(1 - p_{\theta}(x))\big]\end{equation}
其中$\mathcal{D}$是训练数据集合。所以按道理，我们应该以$-\text{ACC}_{\text{smooth}}$为最小化的目标。但事实上，直接优化这个目标的效果并不好，更好的是去优化交叉熵
\begin{equation}\text{cross_entropy}=\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[-y \log p_{\theta}(x) - (1 - y)\log(1 - p_{\theta}(x))\big]\end{equation}
这就有点耐人寻味了，明明$\text{ACC}_{\text{smooth}}$更接近我们的评测指标，为什么用交叉熵反而对评测指标更有利呢？

这需要用梯度来解释。对于$p_{\theta}(x)$，它通常是经过了sigmoid激活的，也就是$p_{\theta}(x)=\sigma(z_{\theta}(x))$，其中$\sigma(t)=\frac{1}{1+e^{-t}}$，它的导数$\sigma'(t)=\sigma(t)(1 - \sigma(t))$，而$z_{\theta}(x)$就是我们通常称的“logits”。

假设$y$是1，那么对应的$-\text{ACC}_{\text{smooth}}$就是$-p_{\theta}(x)=-\sigma(z_{\theta}(x))$，它的梯度是
\begin{equation}-\nabla_{\theta} p_{\theta}(x) = - p_{\theta}(x) (1 - p_{\theta}(x))\nabla_{\theta}z_{\theta}(x)\end{equation}
刚才说了，$y$是1，所以训练目标是$p_{\theta}(x)\to 1$，因此我们期望当$p_{\theta}(x)$接近于0时（误差较大），会带来一个较大的梯度，当$p_{\theta}(x)$接近于1时（误差较小），会带来一个较小的梯度。但上述$-\nabla_{\theta} p_{\theta}(x)$显然不是如此，它的调节项$p_{\theta}(x) (1 - p_{\theta}(x))$在0\.5处取到最大值，至于0和1都是最小值，这就意味着如果误差太大了，梯度反而也小，这就带来优化效率的低下，最终导致整体效果不好。相反，对于交叉熵来说，有
\begin{equation}-\nabla_{\theta} \log p_{\theta}(x) = - (1 - p_{\theta}(x))\nabla_{\theta}z_{\theta}(x)\end{equation}
刚好把梯度里边带来负面作用的$p_{\theta}(x)$因子去掉了，因此优化效率更高，最终效果也好些。上述分析针对的是$y=1$，如果$y=0$，那么结论也是一样的。

## 从光滑F1到加权交叉熵

从这个过程中，我们可以感觉到，对loss的各种魔改，本质上来说都只是在调整梯度，得到更合理的梯度，我们就能实现更有效的优化，得到更好的模型。此外，我们再思考上述转换过程，本来近似目标的梯度是$-\nabla_{\theta}p_{\theta}(x)$，结果$-\nabla_{\theta}\log p_{\theta}(x)$效果更好。如果我们不去仔细分析背后的原因，直接把$p\to \log p$当作一个“公理”来使用，那能否成立呢？会不会带来一些有意思的结果呢？

举个例子，当负样本远远多于正样本时，我们的评测指标通常都不再是准确率了（不然直接全部输出0准确率就很高了），我们通常关心正类的F1，而F1的直接优化也是不容易的，所以我们也需要一个光滑版，文章[《函数光滑化杂谈：不可导函数的可导逼近》](<https://spaces.ac.cn/archives/6620>)同样也给出了结果：
\begin{equation}\text{F1}_{\text{smooth}}=\frac{2 \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y p_{\theta}(x)\big]}{\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y + p_{\theta}(x)\big]}\end{equation}
所以我们的最小化目标原本是$-\text{F1}_{\text{smooth}}$。根据上述“公理”，我们先直接对$-\text{F1}_{\text{smooth}}$求梯度：
\begin{equation}\begin{aligned}&-\nabla_{\theta}\frac{2 \mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y p_{\theta}(x)\big]}{\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y + p_{\theta}(x)\big]}\\
=&-2\frac{\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y \nabla_{\theta}p_{\theta}(x)\big]}{\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y + p_{\theta}(x)\big]} + 2\frac{\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y p_{\theta}(x)\big]\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\nabla_{\theta}p_{\theta}(x)\big]}{\left(\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y + p_{\theta}(x)\big]\right)^2}\\
=&-\frac{2\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\big(y-\text{F1}_{\text{smooth}}/2\big)\nabla_{\theta}p_{\theta}(x)\big]}{\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y + p_{\theta}(x)\big]}
\end{aligned}\end{equation}
其中$\frac{2}{\mathbb{E}_{(x,y)\sim\mathcal{D}}[y + p_{\theta}(x)]}$是整体的一个缩放因子，我们主要关心的还是每个样本的梯度，所以结果是
\begin{equation}-\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\big(y-\text{F1}_{\text{smooth}}/2\big)\nabla_{\theta}p_{\theta}(x)\big]\end{equation}
根据$p\to \log p$“公理”（负样本则是$-p\to\log(1-p)$），我们得到最后的梯度为
\begin{equation}-\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y\cdot\big(1-\text{F1}_{\text{smooth}}/2\big)\cdot\nabla_{\theta}\log p_{\theta}(x) + (1 - y)\cdot\text{F1}_{\text{smooth}}/2\cdot\nabla_{\theta}\log (1-p_{\theta}(x))\big]\end{equation}
这等价于优化目标
\begin{equation}-\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[y\cdot\big(1-\text{F1}_{\text{smooth}}/2\big)\cdot\log p_{\theta}(x) + (1 - y)\cdot\text{F1}_{\text{smooth}}/2\cdot\log (1-p_{\theta}(x))\big]\end{equation}
的梯度（其中$\text{F1}_{\text{smooth}}$不求梯度），所以这其实就是用$1-\text{F1}_{\text{smooth}}/2$调节正样本的交叉熵，用$\text{F1}_{\text{smooth}}/2$调节负样本的交叉熵。

## 从扩大边界到Logits调整

其实无论评测指标是什么，我们肯定都是希望每一个样本都尽可能预测对。问题在于，样本数目比较少的类别，因为学习得不够充分，所以泛化性能不会太好。

让我们从几何角度来思考这个问题。理想情况下，在编码空间里边，每一类样本都占据着自己的一个“地盘”，不同类的“地盘”是互不相交的。样本数目较少的类别泛化性能不大好，主要就体现为其类别所占据的“地盘”比较小，而且往往还会受到类别数目较多的样本的“打压”，因此“生存”几乎都成了问题，更不用说照顾到训练集没有出现过的新样本了。

怎么解决这个问题呢？其实也很形象，如果样本数目少的类别，里边的样本个个都是“大佬”，一个打十个的那种，那么就算样本少，也能在“地盘之争”中不落下风。让我们考虑一个$n$分类问题，某个样本的编码向量为$f_{\theta}(x)$，类别向量为$u_y$，那么该样本与类别向量的相似度，一般用内积$\langle f_{\theta}(x), u_y\rangle$来度量。假设每个样本能占据半径为$r_y$的“地盘”，这样就是说，满足$\Vert z - f_{\theta}(x)\Vert \leq r_y$的任意$z$都算是该样本的编码向量，这也就意味着，满足这个条件的任意$z$，它跟$u_y$的相似度都应该大于它跟其他类别的相似度。

现在我们考虑
\begin{equation}\langle z, u_y\rangle = \langle f_{\theta}(x), u_y\rangle + \langle z - f_{\theta}(x), u_y\rangle\end{equation}
由于$\Vert z - f_{\theta}(x)\Vert \leq r_y$，所以显然有
\begin{equation}\langle f_{\theta}(x), u_y\rangle - r_y\Vert u_y\Vert\leq\langle z, u_y\rangle \leq \langle f_{\theta}(x), u_y\rangle + r_y\Vert u_y\Vert\end{equation}
所以，为了达到“$z$跟$u_y$的相似度都应该大于它跟其他类别的相似度”这个目的，只需要“$z$跟$u_y$的最小相似度都应该大于它跟其他类别的最大相似度”，因此我们的优化目标变为
\begin{equation}-\log\frac{e^{\langle f_{\theta}(x), u_y\rangle - r_y\Vert u_y\Vert}}{e^{\langle f_{\theta}(x), u_y\rangle - r_y\Vert u_y\Vert}+\sum\limits_{i\neq y} e^{\langle f_{\theta}(x), u_i\rangle + r_y\Vert u_i\Vert}}\end{equation}
可以看到，这其实就相当于am\-softmax、circle loss等带有margin的softmax变种，具体形式其实不重要，只需要为类别小的类设置更大的margin就好（样本少的类别每个样本都更“能打”）。那怎么设计每个类的margin呢？之前的文章[《通过互信息思想来缓解类别不平衡问题》](<https://spaces.ac.cn/archives/7615>)就提供了一个方案：$m_y=-\tau\log p(y)$，这里的$p(y)$是先验分布，那么就有
\begin{equation}-\log\frac{e^{\langle f_{\theta}(x), u_y\rangle + \tau \log p(y)}}{\sum\limits_{i} e^{\langle f_{\theta}(x), u_i\rangle + \tau \log p(i)}}\end{equation}
这样我们就联系到了logit adjustment loss了，或者说给logit adjustment loss提供了一种几何直观理解。本质上来说，logit adjustment也是在调节权重，只不过一般的调节权重是在损失函数的$\log$之后调整，而logit adjustment则是在$\log$之前调整。

## 感觉上可以小结一下了

本文就类别不平衡现象及其对策做了一些思考，主要是希望通过一些相对直观的引导，来揭示一些魔改loss的思路，从中我们也可以发现，其实这些方案本质上都算是在调节样本权重或者类权重。本文的分析思路相对来说比较散漫，基本上是笔者的头脑风暴内容，如果错漏之处，请读者见谅并指出。
