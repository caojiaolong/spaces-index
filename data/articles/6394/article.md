# 非对抗式生成模型GLANN的简单介绍

> 作者：苏剑林 · 科学空间 · 2019-02-26
>
> 原文：<https://spaces.ac.cn/archives/6394>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前段时间看到facebook发表了一个非对抗的生成模型GLANN（去年12月挂在arxiv上），号称用非对抗的方式也能生成1024的高清人脸，于是饶有兴致地阅读了一番，确实有点收获，但也有点失望。至于为啥失望，大家阅读下去就明白了。

<strong>原论文：</strong>[《Non\-Adversarial Image Synthesis with Generative Latent Nearest Neighbors》](<https://papers.cool/arxiv/1812.08985>)

<strong>机器之心介绍：</strong>[《为什么让GAN一家独大？Facebook提出非对抗式生成方法GLANN》](<https://mp.weixin.qq.com/s/FJA8Tctq_p4Mj-KgNn_OGg>)

<strong>效果图：</strong>

[![GLANN效果图](<https://spaces.ac.cn/usr/uploads/2019/02/4149978803.png>)](<https://spaces.ac.cn/usr/uploads/2019/02/4149978803.png>)

GLANN效果图

下面是对GLANN模型相关内容的一个简单梳理。

## 隐式最大似然

GLANN整个方法的基础是“隐式最大似然估计”，出自文章[《Implicit Maximum Likelihood Estimation》](<https://papers.cool/arxiv/1809.09087>)，简称“IMLE”。这是去年九月份才挂在arxiv上的一篇文章，让我非常意外。因为这个算法非常简单，我两年前就已经使用过，我一直觉得那是一个显然成立的方法，然而居然这么迟才被发布\.\.\.（感觉错过了几千万）

### 直接估算概率分布

IMLE的附录上给出了一大通复杂的数学推导，但笔者觉得其实没有什么必要。IMLE实际上就是狄拉克分布积分近似的结果而已。

总的来说，如下的采样过程：
\begin{equation}z\sim q(z),\quad x = G(z)\end{equation}
实际上就是假设$x$的分布为
\begin{equation}q(x)=\int \delta\big(x-G(z)\big)q(z)dz\end{equation}
其中$q(z)$一般取正态分布或者均匀分布，而$\delta(\cdot)$代表着（多元的）狄拉克函数。

注意$q(x)$也可以写成
\begin{equation}q(x)=\mathbb{E}_{z\sim q(z)}\big[\delta\big(x-G(z)\big)\big]\end{equation}
而$\delta(\cdot)$实际上就是方差趋于0的高斯分布：
\begin{equation}\delta(x)=\lim_{\sigma\to 0}\frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left(-\frac{\Vert x\Vert^2}{2\sigma^2}\right)\end{equation}
这样一来，我们不妨就让$\sigma$取个有限值，算完之后再让$\sigma\to 0$，即
\begin{equation}q(x)=\lim_{\sigma\to 0}\mathbb{E}_{z\sim q(z)}\left[\frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left(-\frac{\Vert x - G(z)\Vert^2}{2\sigma^2}\right)\right]\end{equation}
然后，我们做最大似然，即以$-\int p(x)\log q(x)dx$为loss，$p(x)$是真实样本的分布：
\begin{equation}\begin{aligned}loss=&-\int p(x)\log \left\{\mathbb{E}_{z\sim q(z)}\left[\frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left(-\frac{\Vert x - G(z)\Vert^2}{2\sigma^2}\right)\right]\right\}dx\\
=&\mathbb{E}_{x\sim p(x)}\left[-\log \left\{\mathbb{E}_{z\sim q(z)}\left[\frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left(-\frac{\Vert x - G(z)\Vert^2}{2\sigma^2}\right)\right]\right\}\right]\\
\sim &\mathbb{E}_{x\sim p(x)}\left[-\log \left\{\mathbb{E}_{z\sim q(z)}\left[\exp\left(-\frac{\Vert x - G(z)\Vert^2}{2\sigma^2}\right)\right]\right\}\right]\end{aligned}\end{equation}
在最后一个式子中，我们已经省去了与优化无关的常数。

现在我们将$\mathbb{E}$转化为采样，即把$x_1,x_2,\dots,x_M\sim p(x)$和$z_1,z_2,\dots,z_N\sim q(z)$代入loss：
\begin{equation}\begin{aligned}loss\sim& -\frac{1}{M}\sum_{i=1}^M \log \left\{\frac{1}{N}\sum_{j=1}^N\exp\left(-\frac{\Vert x_i - G(z_j)\Vert^2}{2\sigma^2}\right)\right\}\\
\sim& -\frac{1}{M}\sum_{i=1}^M \log \left\{\sum_{j=1}^N\exp\left(-\frac{\Vert x_i - G(z_j)\Vert^2}{2\sigma^2}\right)\right\}\end{aligned}\end{equation}
从[《寻求一个光滑的最大值函数》](<https://spaces.ac.cn/archives/3290>)一文我们可以知道，$\text{logsumexp}$（指数、求和、然后取对数）实际上是$\max$的光滑近似，当$\sigma\to 0$时它就是$\max$，加上了负号就是$\min$，所以最终$\sigma\to 0$时的最简形式为：
\begin{equation}loss\sim \frac{1}{M}\sum_{i=1}^M \left(\min_{j=1}^N \Vert x_i - G(z_j)\Vert^2\right)\end{equation}
这便是IMLE所用的loss。（推导过程略长，只是因为写得详细，其实不难～）

因此，IMLE的具体流程中：

> 1、采样一批真样本$x_1,x_2,\dots,x_M$；
> 
> 
> 
> 2、采样一批噪声$z_1,z_2,\dots,z_N$，得到一批假样本$\hat{x}_1,\hat{x}_2,\dots,\hat{x}_N$；
> 
> 
> 
> 3、给每个真样本$x_i$找到它最接近的假样本$\hat{x}_{\rho(i)}$；
> 
> 
> 
> 4、最小化平均距离$\frac{1}{M}\sum\limits_{i=1}^M \Vert x_i - \hat{x}_{\rho(i)}\Vert^2$。

### 效果分析与讨论

抛开推导过程不讲，其实这个算法是很朴素的：假如每一张真样本都可以在假样本中找到足够接近的那个，那不就说明假样本生成得也很不错了吗？所以我说这个算法怎么这么迟才成文，太不可思议了。

然后看看效果。算法的原理没毛病，但问题在于上面的算法中“最接近”用到了l2距离，而对于图像来说l2并不是一个好的距离，所以可以想象这个方法跟VAE一样有着模糊的毛病。事实上如果看CelebA的效果，它连VAE都比不上：

[![IMLE在CelebA上的效果图](<https://spaces.ac.cn/usr/uploads/2019/02/2597958082.png>)](<https://spaces.ac.cn/usr/uploads/2019/02/2597958082.png>)

IMLE在CelebA上的效果图

代码：[https://github\.com/bojone/gan/blob/master/imle\.py](<https://github.com/bojone/gan/blob/master/imle.py>)

其实这个思想还可以推广到一般的散度优化，比如我们可以用
\begin{equation}KL(q(x)\Vert p(x))=\int q(x)\log \frac{q(x)}{p(x)}dx=\mathbb{E}_{x\sim q(x)}\big[\log q(x)-\log p(x)\big]\end{equation}
做优化目标，然后$\log q(x)、\log p(x)$按照同样的方法进行处理，那么结果是：
\begin{equation}loss\sim -\frac{1}{M}\sum_{i=1}^M \left(\min_{j=1}^N \Vert G(z_i) - x_j\Vert^2 - \min_{j=1}^K \Vert G(z_i) - G(z_j)\Vert^2\right)\end{equation}
或者设置一个margin $m$，会使得效果更好些：
\begin{equation}loss\sim -\frac{1}{M}\sum_{i=1}^M \left(\min_{j=1}^N \Vert G(z_i) - x_j\Vert^2 + \text{relu}\left(m - \min_{j=1}^K \Vert G(z_i) - G(z_j)\Vert^2\right)\right)\end{equation}
注意这里要采样两批假样本，否则第二项就没有意义了（同一批样本内，第二项总是0），第二项是用来防止mode callopse的。这个新算法的流程是：

> 1、采样一批真样本$x_1,x_2,\dots,x_M$；
> 
> 
> 
> 2、采样一批噪声$z_1,z_2,\dots,z_N$，得到一批假样本$\hat{x}_1,\hat{x}_2,\dots,\hat{x}_N$；
> 
> 
> 
> 3、采样另一批噪声$z_{N+1},z_{N+2},\dots,z_{N+K}$，得到另一批假样本$\hat{x}_{N+1},\hat{x}_{N+2},\dots,\hat{x}_{N+K}$；
> 
> 
> 
> 4、给每个假样本$\hat{x}_i$（$1\leq i\leq N$）找到它最接近的真样本$x_{\rho_1(i)}$；
> 
> 
> 
> 5、给每个假样本$\hat{x}_i$（$1\leq i\leq N$）在$\hat{x}_{N+1},\hat{x}_{N+2},\dots,\hat{x}_{N+K}$中找到它最接近的假样本$x_{N+\rho_2(i)}$；
> 
> 
> 
> 6、最小化“真\-假”距离，同时最大化“假\-假”距离，即上面的loss。

效果图：

[![另一种IMLE在CelebA上的效果图](<https://spaces.ac.cn/usr/uploads/2019/02/4200250233.png>)](<https://spaces.ac.cn/usr/uploads/2019/02/4200250233.png>)

另一种IMLE在CelebA上的效果图

都差不多\.\.\.

## 从IMLE到GLANN

回到IMLE本身的讨论，可以想象，IMLE效果比较差的主要原因是用了l2距离。那么，如果换成其他距离呢？对于图像真实性来说，有没有现成的loss函数可以用呢？

### perceptual loss

还真有，它就是perceptual loss！这个perceptual loss来源于风格迁移，来源应该是[《Perceptual Losses for Real\-Time Style Transfer and Super\-Resolution》](<https://papers.cool/arxiv/1603.08155>)。这个perceptual loss算起来有点复杂，它需要一个训练好的ImageNet模型，简单起见一般用VGG，然后算出它的最后面几个隐藏层向量，然后分别算隐层向量的l2距离（或者l1距离）和Gram矩阵的l2距离（或者l1距离），最后加起来。

这个距离在风格迁移任务中效果还不错，但是算起来有些复杂，而且更像是一个工程产物，而不是理论推导的结果，因此我并不喜欢，也就没有兴趣详细写下来了。总之，可以用perceptual loss结合IMLE的方式：
\begin{equation}loss\sim \frac{1}{M}\sum_{i=1}^M \left(\min_{j=1}^N d_{perceptual}\big(x_i,G(z_j)\big)\right)\label{eq:perceptual-1}\end{equation}

### 过度产物：GLO

直接优化目标$\eqref{eq:perceptual-1}$理论上没有什么问题，但是计算量很大，因为我们说了perceptual loss算起来很复杂，需要用一个现成的ImageNet模型来计算，假如batch size是64，那么我们每个batch要算$64^2=4096$个perceptual loss，然后取最小，将会慢到难以接受，甚至根本跑不起来。

所以，GLANN还用到了一个称为GLO的技巧，它来自文章[《Optimizing the latent space of generative networks》](<https://papers.cool/arxiv/1707.05776>)。

GLO其实也是一个极为简单的东西，然后又写了一篇文章～～GLO根本没有打算做生成模型，GLO只想得到一个低维嵌入。假设真样本集为$x_1,x_2,\dots,x_M$，那么GLO的优化目标是
\begin{equation}\mathop{\text{argmin}}_{G,\hat{z}_1,\dots,\hat{z}_M}\frac{1}{M}\sum_{i=1}^M d\big(x_i, G(\hat{z}_i)\big)\quad \text{s.t.}\quad \Vert z_i\Vert=1\end{equation}
GLO把$\hat{z}_1,\dots,\hat{z}_M$都拿去优化了，这相当于一个Embedding层，训练出每张图片的Embedding。整个模型可以变化的地方是对Embedding的约束以及所用的度量$d$。对于GLANN来说，用的就是perceptual loss：
\begin{equation}\mathop{\text{argmin}}_{G,\hat{z}_1,\dots,\hat{z}_M}\frac{1}{M}\sum_{i=1}^M d_{perceptual}\big(x_i, G(\hat{z}_i)\big)\quad \text{s.t.}\quad \Vert z_i\Vert=1\end{equation}
这样就算batch size是64，每个batch我也只需要算64个perceptual loss，因为它没涉及到两两比较。

### 最后结果：GLANN

现在有了$\hat{z}_1,\dots,\hat{z}_M$，那么$G(\hat{z}_i)$就能生成图片。现在只需要将$\hat{z}_1,\dots,\hat{z}_M$当作原始图片，然后用IMLE搞一下（这时候就可以用l2了，因为只是在隐空间中）：
\begin{equation}\mathop{\text{argmin}}_{T}\frac{1}{M}\sum_{i=1}^M \left(\min_{j=1}^N \Vert \hat{z}_i - T(z_j)\Vert^2\right)\end{equation}
就得到生成模型了。完整的生成过程是：
\begin{equation}z\sim q(z)\quad\xrightarrow{\quad T\quad }\quad \hat{z}_i \quad \xrightarrow{\quad G\quad }\quad x_i\end{equation}

## 个人评价

到此，我们把GLANN这个模型给讲完了。总的来说，这是个综合了几个技巧的模型，至于效果，看开头的图就行了，其实我觉得效果也不是特别好，背景总是很花。当然，比纯IMLE或GLO要好，这是毋庸置疑的，而且应该比GAN容易训练很多。

GLANN的主要改进是用perceptual loss替换了l2距离，其实这个替换在很多模型都应该可以使用，我猜也可以用到VAE中。另一方面，perceptual loss的做法太工程化，我感觉没有什么意思，所以就没兴趣再深入研究它了。还有，GLANN论文中报告了它在某些数据集上的FID的优势，这看起来很不错，但实际上是不公平的。因为FID的计算本来就借助于ImageNet模型，而GLANN的loss也用到了ImageNet模型，所以GLANN的生成肯定是对FID有优势的。

其实，完全可以用FID作为loss，然后训练一个GLO，然后再训练IMLE，得到一个生成模型。这样做出来的生成模型的FID肯定很低，然而这并没有什么意义，因为图片的真实性都未能得到保证。
