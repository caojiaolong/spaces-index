# VQ\-VAE的简明介绍：量子化自编码器

> 作者：苏剑林 · 科学空间 · 2019-06-24
>
> 原文：<https://spaces.ac.cn/archives/6760>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

印象中很早之前就看到过VQ\-VAE，当时对它并没有什么兴趣，而最近有两件事情重新引起了我对它的兴趣。一是[VQ\-VAE\-2](<https://papers.cool/arxiv/1906.00446>)实现了能够匹配BigGAN的生成效果（来自[机器之心的报道](<https://mp.weixin.qq.com/s/GJr-YtV84eV1KbkyVSkcBA>)）；二是我最近看一篇NLP论文[《Unsupervised Paraphrasing without Translation》](<https://papers.cool/arxiv/1905.12752>)时发现里边也用到了VQ\-VAE。这两件事情表明VQ\-VAE应该是一个颇为通用和有意思的模型，所以我决定好好读读它。

[![个人复现的VQ-VAE在CelebA上的重构效果。可以留意到细节保留得还不错，但稍微放大后能留意到仍有一些模糊感。](<https://spaces.ac.cn/usr/uploads/2019/06/3057025890.png>)](<https://spaces.ac.cn/usr/uploads/2019/06/3057025890.png>)

个人复现的VQ\-VAE在CelebA上的重构效果。可以留意到细节保留得还不错，但稍微放大后能留意到仍有一些模糊感。

## 模型综述

VQ\-VAE（Vector Quantised \- Variational AutoEncoder）首先出现在论[《Neural Discrete Representation Learning》](<https://papers.cool/arxiv/1711.00937>)，跟VQ\-VAE\-2一样，都是Google团队的大作。

### 有趣却玄虚

作为一个自编码器，VQ\-VAE的一个明显特征是它编码出的编码向量是离散的，换句话说，它最后得到的编码向量的每个元素都是一个整数，这也就是“Quantised”的含义，我们可以称之为“量子化”（跟量子力学的“量子”一样，都包含离散化的意思）。

明明整个模型都是连续的、可导的，但最终得到的编码向量却是离散的，并且重构效果看起来还很清晰（如文章开头的图），这至少意味着VQ\-VAE会包含一些有意思、有价值的技巧，值得我们学习一番。不过，读了原论文之后，总感觉原论文写得有点难懂。这种难懂不是像ON\-LSTM原论文那样的晦涩难懂，而是有种“故弄玄虚”的感觉。

首先，你读完整篇论文就会明白，VQ\-VAE其实就是一个AE（自编码器）而不是VAE（变分自编码器），我不知道作者出于什么目的非得用概率的语言来沾VAE的边，这明显加大了读懂这篇论文的难度。其次，VQ\-VAE的核心步骤之一是Straight\-Through Estimator，这是将隐变量离散化后的优化技巧，在原论文中没有稍微详细的讲解，以至于必须看源码才能更好地知道它说啥。最后，论文的核心思想也没有很好地交代清楚，给人的感觉是纯粹在介绍模型本身而没有介绍模型思想。

### PixelCNN

要追溯VQ\-VAE的思想，就不得不谈到自回归模型。可以说，VQ\-VAE做生成模型的思路，源于[PixelRNN](<https://papers.cool/arxiv/1601.06759>)、[PixelCNN](<https://papers.cool/arxiv/1606.05328>)之类的自回归模型，这类模型留意到我们要生成的图像，实际上是离散的而不是连续的。以cifar10的图像为例，它是$32\times 32$大小的3通道图像，换言之它是一个$32\times 32\times 3$的矩阵，矩阵的每个元素是0～255的任意一个整数，这样一来，我们可以将它看成是一个长度为$32\times 32\times 3=3072$的句子，而词表的大小是256，从而用语言模型的方法，来逐像素地、递归地生成一张图片（传入前面的所有像素，来预测下一个像素），这就是所谓的自回归方法：
\begin{equation}p(x)=p(x_1)p(x_2|x_1)\dots p(x_{3n^2}|x_1,x_2,\dots,x_{3n^2-1})\end{equation}
其中$p(x_1),p(x_2|x_1),\dots,p(x_{3n^2}|x_1,x_2,\dots,x_{3n^2-1})$每一个都是256分类问题，只不过所依赖的条件有所不同。

PixelRNN、PixelCNN网上都有一定的资料介绍了，这里不再赘述，我感觉其实也可以蹭着Bert的热潮，去搞个PixelAtt（Attention）来做它。自回归模型的研究主要集中在两方面：<u>一方面是如何设计这个递归顺序</u>，使得模型可以更好地生成采样，因为图像的序列不是简单的一维序列，它至少是二维的，更多情况是三维的，这种情况下你是“从左往右再从上到下”、“从上到下再从左往右”、“先中间再四周”或者是其他顺序，都很大程度上影响着生成效果；<u>另一方面是研究如何加速采样过程</u>。在我读到的文献里，自回归模型比较新的成果是ICLR 2019的工作[《Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling》](<https://papers.cool/arxiv/1812.01608>)。

自回归的方法很稳妥，也能有效地做概率估计，但它有一个最致命的缺点：<strong>慢</strong>。因为它是逐像素地生成的，所以要每个像素地进行随机采样，上面举例的cifar10已经算是小图像的，目前做图像生成好歹也要做到$128\times 128\times 3$的才有说服力了吧，这总像素接近5万个（想想看要生成一个长度为5万的句子），真要逐像素生成会非常耗时。而且这么长的序列，不管是RNN还是CNN模型都无法很好地捕捉这么长的依赖。

原始的自回归还有一个问题，就是<u>割裂了类别之间的联系</u>。虽然说因为每个像素是离散的，所以看成256分类问题也无妨，但事实上连续像素之间的差别是很小的，纯粹的分类问题捕捉到这种联系。更数学化地说，就是我们的目标函数交叉熵是$-\log p_t$，假如目标像素是100，如果我预测成99，因为类别不同了，那么$p_t$就接近于0，$-\log p_t$就很大，从而带来一个很大的损失。但从视觉上来看，像素值是100还是99差别不大，不应该有这么大的损失。

## VQ\-VAE

针对自回归模型的固有毛病，VQ\-VAE提出的解决方案是：<u>先降维，然后再对编码向量用PixelCNN建模。</u>

### 降维离散化

看上去这个方案很自然，似乎没什么特别的，但事实上一点都不自然。

因为PixelCNN生成的离散序列，你想用PixelCNN建模编码向量，那就意味着编码向量也是离散的才行。而我们常见的降维手段，比如自编码器，生成的编码向量都是连续性变量，无法直接生成离散变量。同时，生成离散型变量往往还意味着存在梯度消失的问题。还有，降维、重构这个过程，如何保证重构之后出现的图像不失真？如果失真得太严重，甚至还比不上普通的VAE的话，那么VQ\-VAE也没什么存在价值了。

幸运的是，VQ\-VAE确实提供了有效的训练策略解决了这两个问题。

### 最邻近重构

在VQ\-VAE中，一张$n\times n\times 3$的图片$x$先被传入一个$encoder$中，得到连续的编码向量$z$：
\begin{equation}z = encoder(x)\end{equation}
这里的$z$是一个大小为$d$的向量。另外，VQ\-VAE还维护一个Embedding层，我们也可以称为编码表，记为
\begin{equation}E = [e_1, e_2, \dots, e_K]\end{equation}
这里每个$e_i$都是一个大小为$d$的向量。接着，VQ\-VAE通过最邻近搜索，将$z$映射为这$K$个向量之一：
\begin{equation}z\to e_k,\quad k = \mathop{\text{argmin}}_j \Vert z - e_j\Vert_2\end{equation}
我们可以将$z$对应的编码表向量记为$z_q$，我们认为$z_q$才是最后的编码结果。最后将$z_q$传入一个$decoder$，希望重构原图$\hat{x}=decoder(z_q)$。

整个流程是：
\begin{equation}x\xrightarrow{encoder} z \xrightarrow{\text{最邻近}} z_q \xrightarrow{decoder}\hat{x}\end{equation}
这样一来，因为$z_q$是编码表$E$中的向量之一，所以它实际上就等价于$1,2,\dots,K$这$K$个整数之一，因此这整个流程相当于将整张图片编码为了一个整数。

当然，上述过程是比较简化的，如果只编码为一个向量，重构时难免失真，而且泛化性难以得到保证。所以实际编码时直接用多层卷积将$x$编码为$m\times m$个大小为$d$的向量：
\begin{equation}z = \begin{pmatrix}z_{11} & z_{12} & \dots & z_{1m}\\
z_{21} & z_{22} & \dots & z_{2m}\\
\vdots & \vdots & \ddots & \vdots\\
z_{m1} & z_{m2} & \dots & z_{mm}\\
\end{pmatrix}\end{equation}
也就是说，$z$的总大小为$m\times m\times d$，它依然保留着位置结构，然后每个向量都用前述方法映射为编码表中的一个，就得到一个同样大小的$z_q$，然后再用它来重构。这样一来，<strong>$z_q$也等价于一个$m\times m$的整数矩阵，这就实现了离散型编码</strong>。

### 自行设计梯度

我们知道，如果是普通的自编码器，直接用下述loss进行训练即可：
\begin{equation}\Vert x - decoder(z)\Vert_2^2\end{equation}
但是，在VQ\-VAE中，我们用来重构的是$z_q$而不是$z$，那么似乎应该用这个loss才对：
\begin{equation}\Vert x - decoder(z_q)\Vert_2^2\end{equation}
但问题是$z_q$的构建过程包含了$\text{argmin}$，这个操作是没梯度的，所以如果用第二个loss的话，我们没法更新$encoder$。

换言之，我们的目标其实是$\Vert x - decoder(z_q)\Vert_2^2$最小，但是却不好优化，而$\Vert x - decoder(z)\Vert_2^2$容易优化，但却不是我们的优化目标。那怎么办呢？当然，一个很粗暴的方法是两个都用：
\begin{equation}\Vert x - decoder(z)\Vert_2^2 + \Vert x - decoder(z_q)\Vert_2^2\end{equation}
但这样并不好，因为最小化$\Vert x - decoder(z)\Vert_2^2$并不是我们的目标，会带来额外的约束。

VQ\-VAE使用了一个很精巧也很直接的方法，称为Straight\-Through Estimator，你也可以称之为“直通估计”，它最早源于Benjio的论文[《Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation》](<https://papers.cool/arxiv/1308.3432>)，在VQ\-VAE原论文中也是直接抛出这篇论文而没有做什么讲解。但事实上直接读这篇原始论文是一个很不友好的选择，还不如直接读源代码。

事实上Straight\-Through的思想很简单，就是前向传播的时候可以用想要的变量（哪怕不可导），而反向传播的时候，用你自己为它所设计的梯度。根据这个思想，我们设计的目标函数是：
\begin{equation}\Vert x - decoder(z + sg[z_q - z])\Vert_2^2\end{equation}
其中$sg$是stop gradient的意思，就是不要它的梯度。这样一来，前向传播计算（求loss）的时候，就直接等价于$decoder(z + z_q - z)=decoder(z_q)$，然后反向传播（求梯度）的时候，由于$z_q - z$不提供梯度，所以它也等价于$decoder(z)$，这个就允许我们对$encoder$进行优化了。

顺便说一下，基于这个思想，我们可以为很多函数自己自定义梯度，比如$x + sg[\text{relu}(x) - x]$就是将$\text{relu}(x)$的梯度定义为恒为1，但是在误差计算时又跟$\text{relu}(x)$本身等价。当然，用同样的方法我们可以随便指定一个函数的梯度，至于有没有实用价值，则要具体任务具体分析了。

### 维护编码表

要注意，根据VQ\-VAE的最邻近搜索的设计，我们应该期望$z_q$和$z$是很接近的（事实上编码表$E$的每个向量类似各个$z$的聚类中心出现），但事实上未必如此，即使$\Vert x - decoder(z)\Vert_2^2$和$\Vert x - decoder(z_q)\Vert_2^2$都很小，也不意味着$z_q$和$z$差别很小（即$f(z_1)=f(z_2)$不意味着$z_1 = z_2$）。

所以，为了让$z_q$和$z$更接近，我们可以直接地将$\Vert z - z_q\Vert_2^2$加入到loss中：
\begin{equation}\Vert x - decoder(z + sg[z_q - z])\Vert_2^2 + \beta \Vert z - z_q\Vert_2^2\end{equation}
除此之外，还可以做得更仔细一些。由于编码表（$z_q$）相对是比较自由的，而$z$要尽力保证重构效果，所以我们应当尽量“<u>让$z_q$去靠近$z$</u>”而不是“<u>让$z$去靠近$z_q$</u>”，而因为$\Vert z_q - z\Vert_2^2$的梯度等于对$z_q$的梯度加上对$z$的梯度，所以我们将它等价地分解为
\begin{equation}\Vert sg[z] - z_q\Vert_2^2 + \Vert z - sg[z_q]\Vert_2^2\end{equation}
第一项相等于固定$z$，让$z_q$靠近$z$，第二项则反过来固定$z_q$，让$z$靠近$z_q$。注意这个“等价”是对于反向传播（求梯度）来说的，对于前向传播（求loss）它是原来的两倍。根据我们刚才的讨论，我们希望“让$z_q$去靠近$z$”多于“让$z$去靠近$z_q$”，所以可以调一下最终的loss比例：
\begin{equation}\Vert x - decoder(z + sg[z_q - z])\Vert_2^2 + \beta \Vert sg[z] - z_q\Vert_2^2 + \gamma \Vert z - sg[z_q]\Vert_2^2\end{equation}
其中$\gamma < \beta$，在原论文中使用的是$\gamma = 0.25 \beta$。

（注：还可以用滑动平均的方式更新编码表，详情请看原论文。）

### 拟合编码分布

经过上述一大通设计之后，我们终于将图片编码为了$m\times m$的整数矩阵了，由于这个$m\times m$的矩阵一定程度上也保留了原来输入图片的位置信息，所以我们可以用自回归模型比如PixelCNN，来对编码矩阵进行拟合（即建模先验分布）。通过PixelCNN得到编码分布后，就可以随机生成一个新的编码矩阵，然后通过编码表$E$映射为3维的实数矩阵$z_q$（行\*列\*编码维度），最后经过$deocder$得到一张图片。

一般来说，现在的$m\times m$比原来的$n\times n\times 3$要小得多，比如我在用CelebA数据做实验的时候，原来$128\times 128\times 3$的图可以编码为$32\times 32$的编码而基本不失真，所以用自回归模型对编码矩阵进行建模，要比直接对原始图片进行建模要容易得多。

### 个人的复现

这是自己用Keras实现的VQ\-VAE（Python 2\.7 \+ Tensorflow 1\.8 \+ Keras 2\.2\.4，其中模型部分参考了[这个](<https://github.com/nadavbh12/VQ-VAE>)）：

> <strong>[https://github\.com/bojone/vae/blob/master/vq\_vae\_keras\.py](<https://github.com/bojone/vae/blob/master/vq_vae_keras.py>)</strong>

这个脚本的正文部分只包含VQ\-VAE的编码和重构（文章开头的图就是笔者用这个脚本重构的，可见重构效果还可以），没有包含用PixelCNN建模先验分布。不过最后的注释那里包含了一个用Attention来建模先验分布的例子，用Attention建模先验分布后，随机采样的效果如下：

[![个人用PixelAtt建模先验分布后的随机采样效果（随机挑选的，没有经过筛选）](<https://spaces.ac.cn/usr/uploads/2019/06/2483775767.png>)](<https://spaces.ac.cn/usr/uploads/2019/06/2483775767.png>)

个人用PixelAtt建模先验分布后的随机采样效果（随机挑选的，没有经过筛选）

效果图一定程度上表明这样的随机采样是可行的，但是这样的生成效果不能说很好。我用PixelAtt而不是PixelCNN的原因是在我的复现里PixelCNN效果比PixelAtt还差得多，所以PixelAtt是有一定优势的，但缺点是PixelAtt太耗显存，容易OOM。不过我个人的复现不够好也不意味着这套方法不够好，可能是我没调好的原因，也能使网络不够深之类的。我个人是比较看好这种离散化的编码研究的。

## 最后的总结

到此，总算把VQ\-VAE用自己认为比较好的方式讲清楚了。纵观全文，其实没有任何VAE的味道，所以我说它其实就是一个AE，一个编码为离散型向量的AE。它能重构出比较清晰的图像，则是因为它编码时保留了足够大的feature map～

如果弄懂了VQ\-VAE，那么它新出的2\.0版本也就没什么难理解的了，VQ\-VAE\-2相比VQ\-VAE几乎没有本质上的技术更新，只不过把编码和解码都分两层来做了（一层整体，一层局部），从而使得生成图像的模糊感更少（相比至少是少很多了，但其实你认真看VQ\-VAE\-2的大图，还是有略微的模糊感的）。

不过值得肯定的是，VQ\-VAE整个模型还是挺有意思，离散型编码、用Straight\-Through的方法为梯度赋值等新奇特点，非常值得我们认真学习，能加深我们对深度学习的模型和优化的认识（梯度你都能设计了，还担心设计不好模型吗？）。
