# 变分自编码器（四）：一步到位的聚类方案

> 作者：苏剑林 · 科学空间 · 2018-09-17
>
> 原文：<https://spaces.ac.cn/archives/5887>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

由于VAE中既有编码器又有解码器（生成器），同时隐变量分布又被近似编码为标准正态分布，因此VAE既是一个生成模型，又是一个特征提取器。在图像领域中，由于VAE生成的图片偏模糊，因此大家通常更关心VAE作为图像特征提取器的作用。提取特征都是为了下一步的任务准备的，而下一步的任务可能有很多，比如分类、聚类等。本文来关心“聚类”这个任务。

一般来说，用AE或者VAE做聚类都是分步来进行的，即先训练一个普通的VAE，然后得到原始数据的隐变量，接着对隐变量做一个K\-Means或GMM之类的。但是这样的思路的整体感显然不够，而且聚类方法的选择也让我们纠结。本文介绍基于VAE的一个“一步到位”的聚类思路，它同时允许我们完成无监督地完成聚类和条件生成。

## 理论

### 一般框架

回顾VAE的loss（如果没印象请参考[《变分自编码器（二）：从贝叶斯观点出发》](<https://spaces.ac.cn/archives/5343>)）：
$$KL\Big(p(x,z)\Big\Vert q(x,z)\Big) = \iint p(z|x)\tilde{p}(x)\ln \frac{p(z|x)\tilde{p}(x)}{q(x|z)q(z)} dzdx\tag{1}$$
通常来说，我们会假设$q(z)$是标准正态分布，$p(z|x),q(x|z)$是条件正态分布，然后代入计算，就得到了普通的VAE的loss。

然而，也没有谁规定隐变量一定是连续变量吧？这里我们就将隐变量定为$(z, y)$，其中$z$是一个连续变量，代表编码向量；$y$是离散的变量，代表类别。直接把$(1)$中的$z$替换为$(z,y)$，就得到
$$KL\Big(p(x,z,y)\Big\Vert q(x,z,y)\Big) = \sum_y \iint p(z,y|x)\tilde{p}(x)\ln \frac{p(z,y|x)\tilde{p}(x)}{q(x|z,y)q(z,y)} dzdx\tag{2}$$
这就是用来做聚类的VAE的loss了。

### 分步假设

啥？就完事了？呃，是的，如果只考虑一般化的框架，$(2)$确实就完事了。

不过落实到实践中，$(2)$可以有很多不同的实践方案，这里介绍比较简单的一种。首先我们要明确，在$(2)$中，我们只知道$\tilde{p}(x)$（通过一批数据给出的经验分布），其他都是没有明确下来的。于是为了求解$(2)$，我们需要设定一些形式。一种选取方案为
$$p(z,y|x)=p(y|z)p(z|x),\quad q(x|z,y)=q(x|z),\quad q(z,y)=q(z|y)q(y)\tag{3}$$
代入$(2)$得到
$$KL\Big(p(x,z,y)\Big\Vert q(x,z,y)\Big) = \sum_y \iint p(y|z)p(z|x)\tilde{p}(x)\ln \frac{p(y|z)p(z|x)\tilde{p}(x)}{q(x|z)q(z|y)q(y)} dzdx\tag{4}$$
其实$(4)$式还是相当直观的，它分布描述了编码和生成过程：

> 1、从原始数据中采样到$x$，然后通过$p(z|x)$可以得到编码特征$z$，然后通过分类器$p(y|z)$对编码特征进行分类，从而得到类别；
> 
> 
> 
> 2、从分布$q(y)$中选取一个类别$y$，然后从分布$q(z|y)$中选取一个随机隐变量$z$，然后通过生成器$q(x|z)$解码为原始样本。

### 具体模型

$(4)$式其实已经很具体了，我们只需要沿用以往VAE的做法：$p(z|x)$一般假设为均值为$\mu(x)$方差为$\sigma^2(x)$的正态分布，$q(x|z)$一般假设为均值为$G(z)$方差为常数的正态分布（等价于用MSE作为loss），$q(z|y)$可以假设为均值为$\mu_y$方差为1的正态分布，至于剩下的$q(y),p(y|z)$，$q(y)$可以假设为均匀分布（它就是个常数），也就是希望每个类大致均衡，而$p(y|z)$是对隐变量的分类器，随便用个softmax的网络就可以拟合了。

最后，可以形象地将$(4)$改写为
$$\mathbb{E}_{x\sim\tilde{p}(x)}\Big[-\log q(x|z) + \sum_y p(y|z) \log \frac{p(z|x)}{q(z|y)} + KL\big(p(y|z)\big\Vert q(y)\big)\Big],\quad z\sim p(z|x) \tag{5}$$
其中$z\sim p(z|x)$是重参数操作，而方括号中的三项loss，各有各的含义：

> 1、$-\log q(x|z)$希望重构误差越小越好，也就是$z$尽量保留完整的信息；
> 
> 
> 
> 2、$\sum_y p(y|z) \log \frac{p(z|x)}{q(z|y)}$希望$z$能尽量对齐某个类别的“专属”的正态分布，就是这一步起到聚类的作用；
> 
> 
> 
> 3、$KL\big(p(y|z)\big\Vert q(y)\big)$希望每个类的分布尽量均衡，不会发生两个几乎重合的情况（坍缩为一个类）。当然，有时候可能不需要这个先验要求，那就可以去掉这一项。

## 实验

实验代码自然是Keras完成的了（^\_^），在mnist和fashion\-mnist上做了实验，表现都还可以。实验环境：Keras 2\.2 \+ tensorflow 1\.8 \+ Python 2\.7。

### 代码实现

代码位于：[https://github\.com/bojone/vae/blob/master/vae\_keras\_cluster\.py](<https://github.com/bojone/vae/blob/master/vae_keras_cluster.py>)

其实注释应该比较清楚了，而且相比普通的VAE改动不大。可能稍微有难度的是$\sum_y p(y|z) \log \frac{p(z|x)}{q(z|y)}$这个怎么实现。首先我们代入
$$\begin{aligned}p(z|x)&=\frac{1}{\prod\limits_{i=1}^d\sqrt{2\pi\sigma_i^2(x)}}\exp\left\{-\frac{1}{2}\left\Vert\frac{z - \mu(x)}{\sigma(x)}\right\Vert^2\right\}\\
q(z|y)&=\frac{1}{(2\pi)^{d/2}}\exp\left\{-\frac{1}{2}\left\Vert z - \mu_y\right\Vert^2\right\}\end{aligned}\tag{6}$$
得到
$$\log \frac{p(z|x)}{q(z|y)}=-\frac{1}{2}\sum_{i=1}^d \log \sigma_i^2(x)-\frac{1}{2}\left\Vert\frac{z - \mu(x)}{\sigma(x)}\right\Vert^2 + \frac{1}{2}\left\Vert z - \mu_y\right\Vert^2 \tag{7}$$
注意其实第二项是多余的，因为重参数操作告诉我们$z = \varepsilon\otimes \sigma(x) + \mu(x),\,\varepsilon\sim \mathcal{N}(0,1)$，所以第二项实际上只是$-\Vert \varepsilon\Vert^2/2$，跟参数无关，所以$$\log \frac{p(z|x)}{q(z|y)}\sim -\frac{1}{2}\sum_{i=1}^d \log \sigma_i^2(x) + \frac{1}{2}\left\Vert z - \mu_y\right\Vert^2 \tag{8}$$
然后因为$y$是离散的，所以事实上$\sum_y p(y|z) \log \frac{p(z|x)}{q(z|y)}$就是一个矩阵乘法（相乘然后对某个公共变量求和，就是矩阵乘法的一般形式），用K\.batch\_dot实现。

其他的话，读者应该清楚普通的VAE的实现过程，然后才看本文的内容和代码，不然估计是一脸懵的。

### mnist

这里是mnist的实验结果图示，包括类内样本图示和按类采样图示。最后还简单估算了一下，以每一类对应的数目最多的那个真实标签为类标签的话，最终的test准确率大约有83%，对比这篇文章[《Unsupervised Deep Embedding for Clustering Analysis》](<https://papers.cool/arxiv/1511.06335>)的结果（最高也是84%左右），感觉应该很不错了。

#### 聚类图示

[![聚类类别_0](<https://spaces.ac.cn/usr/uploads/2018/09/4248374979.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/4248374979.png>)

聚类类别\_0

[![聚类类别_1](<https://spaces.ac.cn/usr/uploads/2018/09/329866791.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/329866791.png>)

聚类类别\_1

[![聚类类别_2](<https://spaces.ac.cn/usr/uploads/2018/09/3108328353.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3108328353.png>)

聚类类别\_2

[![聚类类别_3](<https://spaces.ac.cn/usr/uploads/2018/09/899104104.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/899104104.png>)

聚类类别\_3

[![聚类类别_4](<https://spaces.ac.cn/usr/uploads/2018/09/2482483906.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2482483906.png>)

聚类类别\_4

[![聚类类别_5](<https://spaces.ac.cn/usr/uploads/2018/09/1771972054.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1771972054.png>)

聚类类别\_5

[![聚类类别_6](<https://spaces.ac.cn/usr/uploads/2018/09/2644294598.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2644294598.png>)

聚类类别\_6

[![聚类类别_7](<https://spaces.ac.cn/usr/uploads/2018/09/2069954609.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2069954609.png>)

聚类类别\_7

[![聚类类别_8](<https://spaces.ac.cn/usr/uploads/2018/09/1670484768.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1670484768.png>)

聚类类别\_8

[![聚类类别_9](<https://spaces.ac.cn/usr/uploads/2018/09/686914600.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/686914600.png>)

聚类类别\_9

#### 按类采样

[![类别采样_0](<https://spaces.ac.cn/usr/uploads/2018/09/2546735984.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2546735984.png>)

类别采样\_0

[![类别采样_1](<https://spaces.ac.cn/usr/uploads/2018/09/3190615297.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3190615297.png>)

类别采样\_1

[![类别采样_2](<https://spaces.ac.cn/usr/uploads/2018/09/2314146861.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2314146861.png>)

类别采样\_2

[![类别采样_3](<https://spaces.ac.cn/usr/uploads/2018/09/1874435472.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1874435472.png>)

类别采样\_3

[![类别采样_4](<https://spaces.ac.cn/usr/uploads/2018/09/2691912108.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2691912108.png>)

类别采样\_4

[![类别采样_5](<https://spaces.ac.cn/usr/uploads/2018/09/2918254071.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2918254071.png>)

类别采样\_5

[![类别采样_6](<https://spaces.ac.cn/usr/uploads/2018/09/2516384089.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2516384089.png>)

类别采样\_6

[![类别采样_7](<https://spaces.ac.cn/usr/uploads/2018/09/7857165.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/7857165.png>)

类别采样\_7

[![类别采样_8](<https://spaces.ac.cn/usr/uploads/2018/09/2985301778.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2985301778.png>)

类别采样\_8

[![类别采样_9](<https://spaces.ac.cn/usr/uploads/2018/09/3572399239.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3572399239.png>)

类别采样\_9

### fashion\-mnist

这里是[fashion\-mnist](<https://github.com/zalandoresearch/fashion-mnist>)的实验结果图示，包括类内样本图示和按类采样图示，最终的test准确率大约有58\.5%。

#### 聚类图示

[![聚类类别_0](<https://spaces.ac.cn/usr/uploads/2018/09/1339561177.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1339561177.png>)

聚类类别\_0

[![聚类类别_1](<https://spaces.ac.cn/usr/uploads/2018/09/146942591.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/146942591.png>)

聚类类别\_1

[![聚类类别_2](<https://spaces.ac.cn/usr/uploads/2018/09/3615056204.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3615056204.png>)

聚类类别\_2

[![聚类类别_3](<https://spaces.ac.cn/usr/uploads/2018/09/2403232492.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2403232492.png>)

聚类类别\_3

[![聚类类别_4](<https://spaces.ac.cn/usr/uploads/2018/09/1673594546.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1673594546.png>)

聚类类别\_4

[![聚类类别_5](<https://spaces.ac.cn/usr/uploads/2018/09/3613423903.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3613423903.png>)

聚类类别\_5

[![聚类类别_6](<https://spaces.ac.cn/usr/uploads/2018/09/4281226969.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/4281226969.png>)

聚类类别\_6

[![聚类类别_7](<https://spaces.ac.cn/usr/uploads/2018/09/2533807020.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2533807020.png>)

聚类类别\_7

[![聚类类别_8](<https://spaces.ac.cn/usr/uploads/2018/09/1313968800.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1313968800.png>)

聚类类别\_8

[![聚类类别_9](<https://spaces.ac.cn/usr/uploads/2018/09/953394026.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/953394026.png>)

聚类类别\_9

#### 按类采样

[![类别采样_0](<https://spaces.ac.cn/usr/uploads/2018/09/2540178160.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2540178160.png>)

类别采样\_0

[![类别采样_1](<https://spaces.ac.cn/usr/uploads/2018/09/3474770774.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3474770774.png>)

类别采样\_1

[![类别采样_2](<https://spaces.ac.cn/usr/uploads/2018/09/1122977125.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1122977125.png>)

类别采样\_2

[![类别采样_3](<https://spaces.ac.cn/usr/uploads/2018/09/4243792424.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/4243792424.png>)

类别采样\_3

[![类别采样_4](<https://spaces.ac.cn/usr/uploads/2018/09/769094682.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/769094682.png>)

类别采样\_4

[![类别采样_5](<https://spaces.ac.cn/usr/uploads/2018/09/211310512.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/211310512.png>)

类别采样\_5

[![类别采样_6](<https://spaces.ac.cn/usr/uploads/2018/09/2624154310.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/2624154310.png>)

类别采样\_6

[![类别采样_7](<https://spaces.ac.cn/usr/uploads/2018/09/1358956369.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/1358956369.png>)

类别采样\_7

[![类别采样_8](<https://spaces.ac.cn/usr/uploads/2018/09/3424192865.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3424192865.png>)

类别采样\_8

[![类别采样_9](<https://spaces.ac.cn/usr/uploads/2018/09/3361607879.png>)](<https://spaces.ac.cn/usr/uploads/2018/09/3361607879.png>)

类别采样\_9

## 总结

文章简单地实现了一下基于VAE的聚类算法，算法的特点就是一步到位，结合“编码”、“聚类”和“生成”三个任务同时完成，思想是对VAE的loss的一般化。

感觉还有一定的提升空间，比如式$(4)$只是式$(2)$的一个例子，还可以考虑更加一般的情况。代码中的encoder和decoder也都没有经过仔细调优，仅仅是验证想法所用。
