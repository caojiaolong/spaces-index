# 不用L约束又不会梯度消失的GAN，了解一下？

> 作者：苏剑林 · 科学空间 · 2018-11-20
>
> 原文：<https://spaces.ac.cn/archives/6163>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

不知道从什么时候开始，我发现我也掉到了GAN的大坑里边了，唉，争取早日能跳出来\.\.\.

这篇博客介绍的是我最近提交到arxiv的一个关于GAN的新框架，里边主要介绍了一种对概率散度的新理解，并且基于这种理解推导出了一个新的GAN。整篇文章比较偏理论，对这个GAN的相关性质都做了完整的论证，自认为是一个理论完备的结果。

文章链接：[https://papers\.cool/arxiv/1811\.07296](<https://papers.cool/arxiv/1811.07296>)

先摆结论：

> 1、论文提供了一种分析和构造概率散度的直接思路，从而简化了构建新GAN框架的过程。
> 
> 
> 
> 2、推导出了一个称为GAN\-QP的GAN框架$\eqref{eq:gan-gp-gd}$，这个GAN不需要像WGAN那样的L约束，又不会有SGAN的梯度消失问题，实验表明它至少有不逊色于、甚至优于WGAN的表现。

[![GAN-QP效果图](<https://spaces.ac.cn/usr/uploads/2018/11/3038288324.jpg>)](<https://spaces.ac.cn/usr/uploads/2018/11/3038288324.jpg>)

GAN\-QP效果图

论文的实验最大做到了512x512的人脸生成（CelebA HQ），充分表明了模型的有效性（效果不算完美，但是模型特别简单）。有兴趣的朋友，欢迎继续阅读下去。

## 直面对偶空间

我们现在要构建一个GAN框架，一般包含三个步骤

> 1、寻求一种良好的概率散度；
> 
> 
> 
> 2、找出它的对偶形式；
> 
> 
> 
> 3、转化为极小\-极大游戏（min\-max game）。

问题是：真正对训练过程有用的是第二、第三步，第一步并不是那么必要。

事实上，从原空间要定义一个新的散度很难，定义了之后也不一定容易转化为对偶形式。然而，我们可以直接在对偶空间分析，由此可以发现一批新的、形态良好的散度。换言之，我们其实可以直接在对偶空间中论述一个式子是否满足散度的定义，从而直接给出可优化的目标，而不需要关心它具体是JS散度还是W距离了。

下面我们来举例说明这个思路。

### 散度

首先我们来给出散度的定义：

> 如果$\mathcal{D}[p, q]$是关于$p,q$的标量函数，并且满足：
> 
> 
> 
> 1、$\mathcal{D}[p, q]\geq 0$恒成立；
> 2、$\mathcal{D}[p, q]=0\Leftrightarrow p=q$。

那么称$\mathcal{D}[p, q]$为$p,q$的一个散度，散度与“距离”的主要差别是散度不用满足三角不等式，也不用满足对称性。但是散度已经保留了度量差距的最基本的性质，所以我们可以用它来度量$p,q$之间的差异程度。

### SGAN

#### 基本定义

我们先来看SGAN中的判别器loss，定义
\begin{equation}\mathcal{D}[p(x),q(x)] = \max_T\, \frac{1}{2}\mathbb{E}_{x\sim p(x)}[\log \sigma(T(x))] + \frac{1}{2}\mathbb{E}_{x\sim q(x)}[\log (1 - \sigma(T(x)))] + \log 2\label{eq:js-dual-2}\end{equation}
这其实就是JS散度的对偶形式。但是我们可以直接基于这个定义来证明它是一个散度，然后讨论这个散度本身的性质，而根本不需要知道它是JS散度。

怎么证明？只需要证明这个结果满足刚才说的散度的两点要求。注意，按照我们的逻辑，我们不知道它是JS散度，但我们可以从数学角度证明它是一个散度。

其实如果读者真的明白了式$\eqref{eq:js-dual-2}$的含义，证明就不困难了。式$\eqref{eq:js-dual-2}$先定义了一个期望的式子，然后对$T$取最大（用更准确的说法是求“上确界”），取最大的结果才是散度。再强调一遍，“取最大之后的结果才是散度”，$\frac{1}{2}\mathbb{E}_{x\sim p(x)}[\log \sigma(T(x))] + \frac{1}{2}\mathbb{E}_{x\sim q(x)}[\log (1 - \sigma(T(x)))] + \log 2$这个式子并不是散度。

具体的证明过程略微冗长，就不完整摆出来了，请读者自行去看原文的附录。或者看下面的WGAN的部分，因为WGAN的部分相对简单。

#### 对抗网络

假如有了散度之后，我们就可以通过缩小两个概率分布的散度，来训练生成模型了。也就是说接下来要做的事情应该是
\begin{equation}\min_{G} \mathcal{D}[p(x),q(x)]\end{equation}
注意$\mathcal{D}[p(x),q(x)]$是通过$\max_{T}$操作实现的，所以组合起来就是一个min\-max的过程，比如前面的例子，等价地就是：
\begin{equation}G,T = \mathop{\text{argmin}}_G\mathop{\text{argmax}}_T\, \mathbb{E}_{x\sim p(x)}[\log \sigma(T(x))]+ \mathbb{E}_{x=G(z),z\sim q(z)}[\log (1 - \sigma(T(x)))]\label{eq:js-min-max}\end{equation}
这就是SGAN。

所以我们发现，GAN的过程其实就两步：1、通过$\max$定义一个散度；2、通过$\min$缩小两个分布的散度。这里的新观点，就是将$\max$直接作为散度的定义的一部分。

#### 性能分析

我们知道SGAN可能有梯度消失的风险，这是为什么呢？我们考察一个极端情形：
\begin{equation}p(x)=\delta(x-\alpha),q(x)=\delta(x-\beta)\end{equation}
其中$\alpha\neq\beta$。这样一来，两个分布分别只是单点分布，完全没有交集。这种情况下代入$\eqref{eq:js-dual-2}$，结果就是
\begin{equation}\mathcal{D}[p(x),q(x)] = \max_T\, \frac{1}{2}[\log \sigma(T(\alpha))] + \frac{1}{2}[\log (1 - \sigma(T(\beta)))] + \log 2\end{equation}
注意我们对$T$没有任何约束，所以为了取最大，我们可以让$T(\alpha)\to +\infty,T(\beta)\to -\infty$，从而得到上确界是一个常数$\log 2$。即这种情况下$\mathcal{D}[p(x),q(x)]=\log 2$。

这就是说，对于两个几乎没有交集的分布，式$\eqref{eq:js-dual-2}$定义的散度给出的度量结果是常数$\log 2$，常数就意味着梯度是0，无法优化。而WGAN的那两篇文章则表明，“没有交集”理论上在GAN中是很常见的，所以这是SGAN的固有毛病。

#### 一般的f散度

上面的几个小节已经完整了呈现了这种理解的流程：

> 1、我们通过$\max$定义一个数学式子，然后可以从数学角度直接证明这是一个散度，而不用关心它叫什么名字；
> 
> 
> 
> 2、通过$\min$最小化这个散度，组合起来就是一个min\-max的过程，就得到了一种GAN；
> 
> 
> 
> 3、为了检查这种散度在极端情况下的表现，我们可以用$p(x)=\delta(x-\alpha),q(x)=\delta(x-\beta)$去测试它。

上述关于SGAN的论述过程，可以平行地推广到所有的f\-GAN中（参考[《f\-GAN简介：GAN模型的生产车间》](<https://spaces.ac.cn/archives/6016>)），各种f散度其实没有本质上的差异，它们有同样的固有毛病（要不就梯度消失，要不就梯度爆炸）。

### WGAN

#### 基本定义

现在我们转向一类新的散度：Wasserstein距离。注意Wasserstein距离是一个严格的、满足公理化定义的距离，不过我们这里只关心它的散度性质。定义
\begin{equation}\mathcal{W}[p(x),q(x)] = \max_{T,\,\Vert T\Vert_L \leq 1}\, \mathbb{E}_{x\sim p(x)}[T(x)] - \mathbb{E}_{x\sim q(x)}[T(x)]\label{eq:wd-dual}\end{equation}
这里
\begin{equation}\Vert T\Vert_L = \max_{x\neq y} \frac{|T(x)-T(y)|}{d(x,y)}\end{equation}
而$d(x, y)$是任意一种现成的距离。

可以直接证明它是一个散度。这个证明还算经典，所以将它写在这里：

> 1、不管是什么$p(x),q(x)$，只要让$T(x)\equiv 0$，我们就得到$\mathbb{E}_{x\sim p(x)}[T(x)] - \mathbb{E}_{x\sim q(x)}[T(x)]=0$，因为散度的定义是要遍历所有的$T$取最大的，所以它至少不会小于0，这就证明了第一点非负性。
> 
> 
> 
> 2、证明$p(x)=q(x)$时，$\mathcal{W}[p(x),q(x)]=0$，也就是$\mathcal{W}[p(x),p(x)]=0$，这几乎是显然成立的了。
> 
> 
> 
> 3、证明$p(x)\neq q(x)$时（严格来讲是它们不等的测度大于0），$\mathcal{W}[p(x),q(x)] > 0$。这个相对难一点，但其实也很简单，只需要令$T_0(x) = \text{sign}(p(x) - q(x))$，那么显然有
> \begin{equation}\begin{aligned}&\mathbb{E}_{x\sim p(x)}[T_0(x)] - \mathbb{E}_{x\sim q(x)}[T_0(x)] \\
=&\int (p(x)-q(x))\cdot \text{sign}(p(x) - q(x)) dx > 0\end{aligned}\end{equation}
> 
> 
> 
> 这样我们就直接地证明了$\mathcal{W}[p(x),q(x)]$是满足散度的定义的。

#### 对抗网络

同样地，有了新散度，就可以定义新GAN了：
\begin{equation}G,T = \mathop{\text{argmin}}_G\mathop{\text{argmax}}_{T,\,\Vert T\Vert_L \leq 1}\, \mathbb{E}_{x\sim p(x)}[T(x)] - \mathbb{E}_{x=G(z),z\sim q(z)}[T(x)]\label{eq:wd-min-max}\end{equation}
这就是WGAN，本博客的参考资料有[《互怼的艺术：从零直达WGAN\-GP》](<https://spaces.ac.cn/archives/4439>)、[《WGAN\-div：一个默默无闻的WGAN填坑者》](<https://spaces.ac.cn/archives/6139>)。

#### 性能分析

同样地，用$p(x)=\delta(x-\alpha),q(x)=\delta(x-\beta)$去测试$\mathcal{W}[p(x),q(x)]$散度的性能，我们得到
\begin{equation}\mathcal{W}[p(x),q(x)] = \max_{T,\,\Vert T\Vert_L \leq 1} T(\alpha) - T(\beta)\end{equation}
注意我们有L约束$\Vert T\Vert_L \leq 1$，这意味着$|T(\alpha) - T(\beta)| \leq d(\alpha, \beta)$，等号可以取到，所以
\begin{equation}\mathcal{W}[p(x),q(x)] = d(\alpha,\beta)\end{equation}
结果不是常数，所以即使在这种极端情况下我们可以也拉近两个分布的距离。所以从这一点看，WGAN要比SGAN要好。

#### L约束

WGAN的遗留问题就是如何往判别器加入L约束，目前有三种方案：参数裁剪、梯度惩罚、谱归一化，请参考[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)和[《WGAN\-div：一个默默无闻的WGAN填坑者》](<https://spaces.ac.cn/archives/6139>)

参数裁剪基本已经被弃用了。梯度惩罚原则上只是一个经验方法，有它的不合理之处，而且要算梯度通常很慢。谱归一化看起来最优雅，目前效果也挺好，不过也有限制的太死的可能性。进一步讨论请看[《WGAN\-div：一个默默无闻的WGAN填坑者》](<https://spaces.ac.cn/archives/6139>)。

## 新散度，新GAN

现在的结论是：SGAN可能有梯度消失的风险，WGAN虽然很好，但需要额外的L约束。那么很自然就会问：有没有不需要L约束，又不会梯度消失的GAN？鱼与熊掌能否兼得？

还真的可以，下面带你找一个。不对，其实不止一个，带你找一批都行。

### 平方势散度

#### 基本定义

下面要给出的散度，形式是这样的：
\begin{equation}\begin{aligned}&\mathcal{L}[p(x),q(x)] \\
=& \max_{T}\, \mathbb{E}_{(x_r,x_f)\sim p(x_r)q(x_f)}\left[T(x_r,x_f)-T(x_f,x_r) - \frac{(T(x_r,x_f)-T(x_f,x_r))^2}{2\lambda d(x_r,x_f)}\right]\end{aligned}\label{eq:qp-dual}\end{equation}
其中$\lambda > 0$是一个超参数，$d$可以是任意距离。

这个形式好像就在WGAN的基础上加了一个平方形式的势能，所以称为平方势散度（QP\-div，quadratic potential divergence）。

论文的附录已经证明了式$\eqref{eq:qp-dual}$确实是一个散度。

#### 性能分析

用$p(x)=\delta(x-\alpha),q(x)=\delta(x-\beta)$去测试这个散度，结果是
\begin{equation}\mathcal{L}[p(x),q(x)] = \max_{T}\, T(\alpha,\beta)-T(\beta,\alpha) - \frac{(T(\alpha,\beta)-T(\beta,\alpha))^2}{2\lambda d(\alpha,\beta)}\end{equation}
设$z = T(\alpha,\beta)-T(\beta,\alpha)$就得到$z - \frac{z^2}{2\lambda d(\alpha,\beta)}$，很熟悉有没有？这只是个二次函数的最大值问题呀，最大值是$\frac{1}{2}\lambda d(\alpha,\beta)$呀，所以我们就有
\begin{equation}\mathcal{L}[p(x),q(x)] = \frac{1}{2}\lambda d(\alpha,\beta)\end{equation}
这不就跟WGAN差不多了嘛，哪怕对于极端分布，也不会有梯度消失的风险。鱼与熊掌真的可以兼得。

### GAN\-QP

#### 对抗网络

有了散度就可以构建对抗网络，我们最终给出的形式为
\begin{equation}\begin{aligned}&T= \mathop{\text{argmax}}_T\, \mathbb{E}_{(x_r,x_f)\sim p(x_r)q(x_f)}\left[T(x_r,x_f)-T(x_f,x_r) - \frac{(T(x_r,x_f)-T(x_f,x_r))^2}{2\lambda d(x_r,x_f)}\right] \\
&G = \mathop{\text{argmin}}_G\,\mathbb{E}_{(x_r,x_f)\sim p(x_r)q(x_f)}\left[T(x_r,x_f)-T(x_f,x_r)\right]
\end{aligned}\label{eq:gan-gp-gd}\end{equation}
我在论文中称之为GAN\-QP。

注意不要把二次项$-\frac{(T(x_r,x_f)-T(x_f,x_r))^2}{2\lambda d(x_r,x_f)}$这一项加入到生成器的loss中（理论上不成问题，但是用梯度下降优化时会有问题。），因为这一项的分母是$d(x_r,x_f)$，一旦最小化二次项，等价于最小化$d(x_r,x_f)$，也就是用$d(x_r,x_f)$来度量图片的差距，这是不科学的。

#### 解的分析

通过变分法可以证明（还是在附录），判别器的最优解是：
\begin{equation}\frac{p(x_r)q(x_f) - p(x_f)q(x_r)}{p(x_r)q(x_f) + p(x_f)q(x_r)} = \frac{T(x_r,x_f)-T(x_f,x_r)}{\lambda d(x_r, x_f)}\label{eq:opt-t}\end{equation}

由这个最优解，我们可以得到两点结论。首先，不难证明最优解满足
\begin{equation}-1 \leq \frac{T(x_r,x_f)-T(x_f,x_r)}{\lambda d(x_r, x_f)}\leq 1\end{equation}
也就是说最优解自动满足L约束。所以我们可以认为GAN\-QP是一种自适应L约束的方案。

其次，将最优解代入生成器的loss，那么得到总的目标是：
\begin{equation}\lambda\iint p(x_r)q(x_f)\frac{p(x_r)q(x_f) - p(x_f)q(x_r)}{p(x_r)q(x_f) + p(x_f)q(x_r)} d(x_r, x_f) dx_r dx_f\end{equation}
这也是一个概率散度，并且我们也从理论上证明了它不会梯度消失/爆炸（跟柯西不等式有关）。此外，还可以看到$\lambda$只是一个缩放因子，事实上并不重要，从而这个GAN\-QP对$\lambda$是鲁棒的，$\lambda$不会明显影响模型的效果。

## 实验结果

论文在CelebA HQ数据集上，比较了多种GAN与GAN\-QP的效果，表明GAN\-QP能媲美甚至超越当前最优的模型。

注意，模型$\eqref{eq:gan-gp-gd}$中，$T$是$(x_r,x_f)$的二元函数，但实验表明，取最简单的一元特例$T(x_r,x_f)\equiv T(x_r)$即可，即$T(x_r,x_f) - T(x_f,x_r)$用$T(x_r) - T(x_f)$就够了，改成二元函数并没有明显提升（但也可能是我没调好）。这样的话，形式上就跟WGAN\-GP非常相似了，但理论更完备。

代码开源：[https://github\.com/bojone/gan\-qp](<https://github.com/bojone/gan-qp>)

### 128x128

在128x128分辨率上，我们进行了较为全面的比较，定量指标是FID。结果如下图：

[![不同GAN的FID定量曲线](<https://spaces.ac.cn/usr/uploads/2018/11/2370552731.png>)](<https://spaces.ac.cn/usr/uploads/2018/11/2370552731.png>)

不同GAN的FID定量曲线

以及下表
$$\begin{array}{c|ccccc}
\hline
\hline
& \text{GAN-QP-L1/L2} & \text{WGAN-GP} & \text{WGAN-SN} & \text{SGAN-SN} & \text{LSGAN-SN}  \\
\hline
\text{Best FID} & 45.0 / 44.7 & 55.5 & 47.8 & 44.5 & 45.8\\
\hline
\text{Speed} & 1\text{x} / 1\text{x} & 1.5\text{x} & 1\text{x} & 1\text{x} & 1\text{x}\\
\hline
\hline
\end{array}$$

### 256与512

在128分辨率上，最好的表现是GAN\-QP和SGAN\-SN，不过在256x256分辨率上，它们的表现就拉开了差距：
$$\begin{array}{c|ccccc}
\hline
\hline
& \text{GAN-QP} & \text{SGAN-SN}  \\
\hline
\text{Best FID} & 22.7 & 27.9\\
\hline
\hline
\end{array}$$
我最大把GAN\-QP的实验做到了512x512的人脸生成，效果还是不错的，最终的FID是26\.44：

[![512x512人脸效果图](<https://spaces.ac.cn/usr/uploads/2018/11/3924177162.jpg>)](<https://spaces.ac.cn/usr/uploads/2018/11/3924177162.jpg>)

512x512人脸效果图

## 论文综述

这篇文章源于我对概率散度的思考，企图得到一种更直接的理解概率散度的方案，其中还受启发于[WGAN\-div](<https://spaces.ac.cn/archives/6139>)。

幸好，最后把这条路走通了，还得到了一些新结果，遂提交到arxiv上，供各位参考，希望得到各位前辈高手的指点。事实上，基于类似的思路，我们可以构造很多类似的散度，比如将平方换成4次、6次方等，只不过理论分析起来就会困难一些了。

限于算力，加之我不是专门研究GAN的，所以实验方面可能做得不够完善，基本能论证结论即可，请大家体谅，当然也欢迎各位的指导。
