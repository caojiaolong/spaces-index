# 强大的NVAE：以后再也不能说VAE生成的图像模糊了

> 作者：苏剑林 · 科学空间 · 2020-07-10
>
> 原文：<https://spaces.ac.cn/archives/7574>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨天早上，笔者在日常刷arixv的时候，然后被一篇新出来的论文震惊了！论文名字叫做[《NVAE: A Deep Hierarchical Variational Autoencoder》](<https://papers.cool/arxiv/2007.03898>)，顾名思义是做VAE的改进工作的，提出了一个叫NVAE的新模型。说实话，笔者点进去的时候是不抱什么希望的，因为笔者也算是对VAE有一定的了解，觉得VAE在生成模型方面的能力终究是有限的。结果，论文打开了，呈现出来的画风是这样的：

[![NVAE的人脸生成效果](<https://spaces.ac.cn/usr/uploads/2020/07/329826639.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/329826639.png>)

NVAE的人脸生成效果

然后笔者的第一感觉是这样的：

> W\!T\!F\! 这真的是VAE生成的效果？这还是我认识的VAE么？看来我对VAE的认识还是太肤浅了啊，以后再也不能说VAE生成的图像模糊了\.\.\.

不过再看了看作者机构，原来是NVIDIA，这也大概能接受了。最近几年可能大家都留意到NVIDIA通常都在年底发布个生成模型的突破，2017年底是[PGGAN](<https://papers.cool/arxiv/1710.10196>)，2018年底是[StyleGAN](<https://papers.cool/arxiv/1812.04948>)，2019年底是[StyleGAN2](<https://papers.cool/arxiv/1912.04958>)，今年貌似早了些，而且动作也多了些，因为上个月才发了个叫[ADA](<https://papers.cool/arxiv/2006.06676>)的方法，将Cifar\-10的生成效果提到了一个新高度，现在又来了个NVAE。

那这个NVAE究竟有什么特别的地方，可以实现VAE生成效果的突飞猛进呢？

## VAE回顾

可能读者认真观察后会说：

> 好像还是有点假呀，那脸部也太光滑了，好像磨过皮一样，还比不上StyleGAN呀～

是的，这样评价并没有错，生成痕迹还是挺明显的。但如果你没感觉到震惊，那估计是因为你没看过之前的VAE生成效果，一般的VAE生成画风是这样的：

[![一般的VAE的随机生成效果](<https://spaces.ac.cn/usr/uploads/2020/07/1882951247.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/1882951247.png>)

一般的VAE的随机生成效果

所以，你还觉得这不是一个突破吗？

那么，是什么限制了（以前的）VAE的表达能力呢？这一次的突破又是因为改进了哪里呢？让我们继续看下去。

### 基本介绍

VAE，即变分自编码器（Variational Auto\-Encoder），本空间已经有不少文章介绍过了，在右侧搜索栏搜索“[变分自编码器](<https://spaces.ac.cn/search/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8/>)”就能搜到很多相关博文。这里做个简单的回顾和分析。

在笔者对VAE的推导里边，我们是先有一批样本，这批样本代表着一个真实的（但不知道形式的）分布$\tilde{p}(x)$，然后我们构建一个带参数的后验分布$p(z|x)$，两者就组成一个联合分布$p(x,z)=\tilde{p}(x)p(z|x)$。接着，我们再定义一个先验分布$q(z)$，已经定义一个生成分布$q(x|z)$，这样构成另一个联合分布$q(x,z)=q(z)q(x|z)$。最后，我们的目的就是让$p(x,z),q(x,z)$相互接近起来，所以我们去优化两者之间的KL散度：
\begin{equation}\begin{aligned}
KL\big(p(x,z)\big\Vert q(x,z)\big)=&\iint p(x,z)\log \frac{p(x,z)}{q(x,z)} dzdx\\
=&\mathbb{E}_{x\sim \tilde{p}(x)} \Big[\mathbb{E}_{z\sim p(z|x)}\big[-\log q(x|z)\big]+KL\big(p(z|x)\big\Vert q(z)\big)\Big] + \text{常数}
\end{aligned}\end{equation}
这就是VAE的优化目标。

### 困难分析

对$p(z|x),q(z),q(x|z)$的要求是：1、能写出解析表达式；2、方便采样。然而连续型分布的世界里这样的分布并不多，最常用的也就是高斯分布了，而这其中又以“各分量独立的高斯分布”最为简单，所以在一般的VAE里边，$p(z|x),q(z),q(x|z)$都被设为各分量独立的高斯分布：$p(z|x)=\mathcal{N}(z;\mu_1(x),\sigma_1^2(x))$、$q(z)=\mathcal{N}(z;0,1)$以及$q(x|z)=\mathcal{N}(x;\mu_2(z),\sigma_2^2(z))$。

问题是，“各分量独立的高斯分布”不能拟合任意复杂的分布，当我们选定$p(z|x)$的形式后，有可能不管我们怎么调它的参数，$\int \tilde{p}(x)p(z|x)dx$和$\frac{\tilde{p}(x)p(z|x)}{\int \tilde{p}(x)p(z|x)dx}$都不能成为高斯分布，这就意味着$KL\big(p(x,z)\big\Vert q(x,z)\big)$从理论上来说就不可能为0，所以让$p(x,z),q(x,z)$相互逼近的话，只能得到一个大致、平均的结果，这也就是常规VAE生成的图像偏模糊的原因。

### 相关改进

改进VAE的一个经典方向是将VAE与GAN结合起来，比如[CVAE\-GAN](<https://papers.cool/arxiv/1703.10155>)、[AGE](<https://papers.cool/arxiv/1704.02304>)等，目前这个方向最先进结果大概是[IntroVAE](<https://papers.cool/arxiv/1807.06358>)。从理论上来讲，这类工作相当于隐式地放弃了$q(x|z)$是高斯分布的假设，换成了更一般的分布，所以能提升生成效果。不过笔者觉得，将GAN引入到VAE中有点像“与虎谋皮”，借助GAN提升了性能，但也引入了GAN的缺点（训练不稳定等），而且提升了性能的VAE生成效果依然不如纯GAN的。另外一个方向是将VAE跟[flow模型](<https://spaces.ac.cn/search/%E7%BB%86%E6%B0%B4%E9%95%BFflow/>)结合起来，比如[IAF\-VAE](<https://papers.cool/arxiv/1606.04934>)以及笔者之前做的[f\-VAE](<https://spaces.ac.cn/archives/5977>)，这类工作则是通过flow模型来增强$p(z|x)$或$q(x|z)$的表达能力。

还有一个方向是引入离散的隐变量，典型代表就是VQ\-VAE，其介绍可以看笔者的[《VQ\-VAE的简明介绍：量子化自编码器》](<https://spaces.ac.cn/archives/6760>)。VQ\-VAE通过特定的编码技巧将图片编码为一个离散型序列，然后PixelCNN来建模对应的先验分布$q(z)$。前面说到，当$z$为连续变量时，可选的$p(z|x),q(z)$都不多，从而逼近精度有限；但如果$z$是离散序列的话，$p(z|x),q(z)$对应离散型分布，而利用自回归模型（NLP中称为语言模型，CV中称为PixelRNN/PixelCNN等）我们可以逼近任意的离散型分布，因此整体可以逼近得更精确，从而改善生成效果。其后的升级版[VQ\-VAE\-2](<https://papers.cool/arxiv/1906.00446>)进一步肯定了这条路的有效性，但整体而言，VQ\-VAE的流程已经与常规VAE有很大出入了，有时候不大好将它视为VAE的变体。

## NVAE梳理

铺垫了这么久，总算能谈到NVAE了。NVAE全称是<strong>Nouveau VAE</strong>（难道不是Nvidia VAE？），它包含了很多当前CV领域的新成果，其中包括<u>多尺度架构、可分离卷积、swish激活函数、flow模型</u>等，可谓融百家之所长，遂成当前最强VAE～

（提醒，本文的记号与原论文、常见的VAE介绍均有所不同，但与本博客其他相关文章是一致的，望读者不要死记符号，而是根据符号的实际含义来理解文章。）

### 自回归分布

前面我们已经分析了，VAE的困难源于$p(z|x),q(z),q(x|z)$不够强，所以改进的思路都是要增强它们。首先，NVAE不改变$q(x|z)$，这主要是为了保持生成的并行性，然后是通过自回归模型增强了先验分布$q(z)$和后验分布$p(z|x)$。具体来说，它将隐变量分组为$z=\{z_1,z_2,\dots,z_L\}$，其中各个$z_l$还是一个向量（而非一个数），然后让
\begin{equation}q(z)=\prod_{l=1}^L q(z_l|z_{< l}),\quad p(z|x)=\prod_{l=1}^L p(z_l|z_{< l},x)\label{eq:arpq}\end{equation}
而各个组的$q(z_l|z_{< l}),p(z_l|z_{< l},x)$依旧建立为高斯分布，所以总的来说$q(z),p(z|x)$就被建立为自回归高斯模型。这时候的后验分布的KL散度项变为
\begin{equation}KL\big(p(z|x)\big\Vert q(z)\big)=KL\big(p(z_1|x)\big\Vert q(z_1)\big)+\sum_{l=2}^L \mathbb{E}_{p(z_{< l}|x)}\Big[KL\big(p(z_l|z_{< l}, x)\big\Vert q(z_l|z_{< l})\big)\Big]\end{equation}

当然，这个做法只是很朴素的推广，并非NVAE的首创，它可以追溯到2015年的[DRAW](<https://papers.cool/arxiv/1502.04623>)、[HVM](<https://papers.cool/arxiv/1511.02386>)等模型。NVAE的贡献是给式$\eqref{eq:arpq}$提出了一种“相对式”的设计：
\begin{equation}\begin{aligned}&q(z_l|z_{< l})=\mathcal{N}\left(z_l;\mu(z_{< l}),\sigma^2(z_{< l})\right)\\
&p(z_l|z_{< l},x)=\mathcal{N}\left(z_l;\mu(z_{< l})+\Delta\mu(z_{< l},x),\sigma^2(z_{< l})\otimes \Delta\sigma^2(z_{< l}, x)\right)
\end{aligned}\end{equation}
也就是说，没有直接去后验分布$p(z_l|z_{< l},x)$的均值方差，而是去建模的是它与先验分布的均值方差的相对值，这时候我们有（简单起见省去了自变量记号，但不难对应理解）
\begin{equation}KL\big(p(z_l|z_{< l}, x)\big\Vert q(z_l|z_{< l})\big)=\frac{1}{2} \sum_{i=1}^{|z_l|} \left(\frac{\Delta\mu_{(i)}^2}{\sigma_{(i)}^2} + \Delta\sigma_{(i)}^2 - \log \Delta\sigma_{(i)}^2 - 1\right)\end{equation}
原论文指出这样做能使得训练更加稳定。

### 多尺度设计

现在隐变量分成了$L$组$z=\{z_1,z_2,\dots,z_L\}$，那么问题就来了：1、编码器如何一一生成$z_1,z_2,\dots,z_L$？2、解码器如何一一利用$z_1,z_2,\dots,z_L$？也就是说，编码器和解码器如何设计？

[![NVAE中的编码器和解码器架构。其中r代表残差模块，h代表可训练参数，蓝色部分是参数共享的](<https://spaces.ac.cn/usr/uploads/2020/07/728752186.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/728752186.png>)

NVAE中的编码器和解码器架构。其中r代表残差模块，h代表可训练参数，蓝色部分是参数共享的

NVAE巧妙地设计了多尺度的编码器和解码器，如上图所示。首先，编码器经过层层编码，得到最顶层的编码向量$z_1$，然后再慢慢地从顶层往下走，逐步得到底层的特征$z_2,\dots,z_L$；至于解码器，自然也是一个自上往下地利用$z_1,z_2,\dots,z_L$的过程，而这部分刚好也是与编码器生成$z_1,z_2,\dots,z_L$的过程有共同之处，所有NVAE直接让对应的部分参数共享，这样既省了参数量，也能通过两者间的相互约束提高泛化性能。

这种多尺度设计在当前最先进的生成模型都有体现，比如[StyleGAN](<https://papers.cool/arxiv/1812.04948>)、[BigGAN](<https://papers.cool/arxiv/1809.11096>)、[VQ\-VAE\-2](<https://papers.cool/arxiv/1906.00446>)等，这说明多尺度设计的有效性已经得到比较充分的验证。此外，为了保证性能，NVAE还对残差模块的设计做了仔细的筛选，最后才确定了如下的残差模块，这炼丹不可谓不充分极致了：

[![NVAE中的残差模块](<https://spaces.ac.cn/usr/uploads/2020/07/2829736148.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/2829736148.png>)

NVAE中的残差模块

### 其他提升技巧

除了以上两点比较明显的特征外，其实NVAE还包含了很多对性能有一定提升的技巧，这里简单列举一些。

<strong>BN层的改进。</strong>当前很多生成模型已经弃用BN（Batch Normalization）了，多数会改用IN（Instance Normalization）或WN（Weight Normalization），因为发现用BN会损失性能。NVAE通过实验发现，其实BN对训练还是有帮助的，但对预测有害，原因是预测阶段所使用的滑动平均得来的均值方差不够好，所以NVAE在模型训练完后，通过多次采样同样batch\_size的样本来重新估算均值方差，从而保证了BN的预测性能。此外，为了保证训练的稳定性，NVAE还给BN的$\gamma$的模长加了个正则项。

<strong>谱正则化的应用。</strong>我们知道，任意两个分布的KL散度是无上界的，所以VAE里边的KL散度项也是无上界的，而优化这种无上界的目标是很“危险”的，说不准啥时候就发散了。所以同样是为了稳定训练，NVAE给每一个卷积层都加了谱正则化，其概念可以参考笔者之前的[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)。加谱归一化可以使得模型的Lipschitz常数变小，从而使得整个模型的Landscape更为光滑，更利于模型稳定训练。

<strong>flow模型增强分布。</strong>通过自回归模型，NVAE增强了模型对分布的拟合能力。不过这个自回归只是对组间进行的，对于组内的单个分布$p(z_l|z_{< l}, x)$和$q(z_l|z_{< l})$，依然假设为各分量独立的高斯分布，这说明拟合能力依然有提升空间。更彻底的方案是，对于组内的每个分量也假设为自回归分布，但是这样一来在采样的时候就巨慢无比了（所有的分量串联递归采样）。NVAE提供了一个备选的方案，通过将组内分布建立为[flow模型](<https://spaces.ac.cn/search/%E7%BB%86%E6%B0%B4%E9%95%BFflow/>)来增强模型的表达能力，同时保持组内采样的并行性。实验结果显示这是有提升的，但笔者认为引入flow模型会大大增加模型的复杂度，而且提升也不是特别明显，感觉能不用就不用为好。

<strong>节省显存的技巧。</strong>尽管NVIDIA应该不缺显卡，但NVAE在实现上还是为省显存下了点功夫。一方面，它采用了混合精度训练，还顺带在论文推了一波自家的[APEX库](<https://github.com/NVIDIA/apex>)。另一方面，它在BN那里启用了[gradient check\-point](<https://papers.cool/arxiv/1604.06174>)（也就是重计算）技术，据说能在几乎不影响速度的情况下节省18%的显存。总之，比你多卡的团队也比你更会省显存～

### 更多效果图

到这里，NVAE的技术要点基本上已经介绍完毕了。如果大家还觉得意犹未尽的话，那就多放几张效果图吧，让大家更深刻地体会NVAE的惊艳之处。

[![NVAE在CelebA HQ和FFHQ上的生成效果。值得注意的是，NVAE是第一个FFHQ数据集上做实验的VAE类模型，而且第一次就如此惊艳了](<https://spaces.ac.cn/usr/uploads/2020/07/2284812415.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/2284812415.png>)

NVAE在CelebA HQ和FFHQ上的生成效果。值得注意的是，NVAE是第一个FFHQ数据集上做实验的VAE类模型，而且第一次就如此惊艳了

[![基于NVAE的图像检索实验。左边是随机生成的样本，右边是训练集的最相似样本，主要是检验模型是否只是单纯地记住了训练集](<https://spaces.ac.cn/usr/uploads/2020/07/4186540346.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/4186540346.png>)

基于NVAE的图像检索实验。左边是随机生成的样本，右边是训练集的最相似样本，主要是检验模型是否只是单纯地记住了训练集

[![CelebA HQ的更多生成效果](<https://spaces.ac.cn/usr/uploads/2020/07/1271285742.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/1271285742.png>)

CelebA HQ的更多生成效果

## 个人收获

从下述训练表格来看，我们可以看到训练成本还是蛮大的，比同样分辨率的StyleGAN都要大，并且纵观整篇论文，可以发现有很多大大小小的训练trick（估计还有不少是没写在论文里边的，当然，其实在StyleGAN和BigGAN里边也包含了很多类似的trick，所以这不算是NVAE的缺点），因此对于个人来说，估计是不容易复现NVAE的。那么，对于仅仅是有点爱好生成模型的平民百姓（比如笔者）来说，从NVAE中可以收获到什么呢？

[![NVAE的训练参数与成本](<https://spaces.ac.cn/usr/uploads/2020/07/3405620191.png>)](<https://spaces.ac.cn/usr/uploads/2020/07/3405620191.png>)

NVAE的训练参数与成本

对于笔者来说，NVAE带来的思想冲击主要有两个。

第一，就是自回归的高斯模型可以很有力地拟合复杂的连续型分布。以前笔者以为只有离散分布才能用自回归模型来拟合，所以笔者觉得在编码时，也需要保持编码空间的离散型，也就是VQ\-VAE那一条路。而NVAE证明了，哪怕隐变量是连续型的，自回归高斯分布也能很好地拟合，所以不一定要走VQ\-VAE的离散化道路了，毕竟连续的隐变量比离散的隐变量更容易训练。

第二，VAE的隐变量可以不止一个，可以有多个的、分层次的。我们再次留意上表，比如FFHQ那一列，关于隐变量$z$的几个数据，他一共有$4+4+4+8+16=36$组，每组隐变量大小还不一样，分别是$\left\{8^2,16^2,32^2,64^2,128^2\right\}\times 20$，如此算来，要生成一个$256\times 256$的FFHQ图像，需要一个总大小有
\begin{equation}\left(4\times 8^2 + 4\times 16^2 + 4\times 32^2 + 8\times 64^2 + 16\times 128^2 \right)\times 20=6005760\end{equation}
维的随机向量，也就是说，采样一个600万维的向量，生成一个$256\times 256\times 3 = 196608$（不到20万）维的向量。这跟传统的VAE很不一样，传统的VAE一般只是将图片编码为单个（几百维）的向量，而这里的编码向量则多得多，有点全卷积自编码器的味道了，所以清晰度提升也是情理之中。

## Nouveau是啥？

最后，笔者饶有性质地搜索了一下Nouveau的含义，以下是来自[维基百科的解释](<https://zh.wikipedia.org/wiki/Nouveau>)：

> nouveau (/nuːˈvoʊ/) 是一个自由及开放源代码显卡驱动程序，是为Nvidia的显卡所编写，也可用于属于系统芯片的NVIDIA Tegra系列，此驱动程序是由一群独立的软件工程师所编写，Nvidia的员工也提供了少许帮助。
> 
> 该项目的目标为利用逆向工程Nvidia的专有Linux驱动程序来创造一个开放源代码的驱动程序。由让freedesktop\.org托管的X\.Org基金会所管理，并以Mesa 3D的一部分进行散布，该项目最初是基于只有2D绘图能力的“nv”自由与开放源代码驱动程序所开发的，但红帽公司的开发者Matthew Garrett及其他人表示原先的代码被混淆处理过了。nouveau以MIT许可证许可。
> 
> 项目的名称是从法文的“nouveau”而来，意思是“新的”。这个名字是由原作者的的IRC客户端的自动取代功能所建议的，当他键入“nv”时就被建议改为“nouveau”。

这是不是说，其实 Nouveau VAE 跟 Nvidia VAE 算是同义词了呢？原来咱们开始的理解也并没有错呀～

## 文章小结

本文介绍了NVIDIA新发表的一个称之为NVAE的升级版VAE，它将VAE的生成效果推向了一个新的高度。从文章可以看出，NVAE通过自回归形式的隐变量分布提升了理论上限，设计了巧妙的编码\-解码结构，并且几乎融合了当前所有生成模型的最先进技术，打造成了当前最强的VAE～
