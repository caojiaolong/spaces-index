# HSIC简介：一个有意思的判断相关性的思路

> 作者：苏剑林 · 科学空间 · 2019-08-26
>
> 原文：<https://spaces.ac.cn/archives/6910>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前几天，在机器之心看到这样的一个推送[《彻底解决梯度爆炸问题，新方法不用反向传播也能训练ResNet》](<https://mp.weixin.qq.com/s/jT6WU-XlcvP-n428oWyHQQ>)，当然，媒体的标题党作风我们暂且无视，主要看内容即可。机器之心的这篇文章，介绍的是论文[《The HSIC Bottleneck: Deep Learning without Back\-Propagation》](<https://papers.cool/arxiv/1908.01580>)的成果，里边提出了一种通过HSIC Bottleneck来训练神经网络的算法。

坦白说，这篇论文笔者还没有看明白，因为对笔者来说里边的新概念有点多了。不过论文中的“HSIC”这个概念引起了笔者的兴趣。经过学习，终于基本地理解了这个HSIC的含义和来龙去脉，于是就有了本文，试图给出HSIC的一个尽可能通俗（但可能不严谨）的理解。

## 背景

HSIC全称“Hilbert\-Schmidt independence criterion”，中文可以叫做“希尔伯特\-施密特独立性指标”吧，跟互信息一样，它也可以用来衡量两个变量之间的独立性。

### 度量相关

我们知道，互信息的基本形式是
$$\begin{equation}I(X,Y)=\iint p(x,y)\log \frac{p(x, y)}{p(x)p(y)}dxdy\label{eq:i}\end{equation}$$
如果$I(X,Y)=0$那么就说明$p(x, y)\equiv p(x)p(y)$，也就是两个变量是相互独立的，否则是相关的。但$\log \frac{p(x, y)}{p(x)p(y)}$这一项意味着我们要用某种方式对概率密度进行估计。

HSIC的作用跟互信息类似，但是跟互信息不一样的是，它不需要估计两个变量的概率密度，而是直接转化为采样的形式。

长期关注本博客的读者都知道，“互信息”是本博客经常出现的概念，我们可以用互信息做新词发现（比如[《基于切分的新词发现》](<https://spaces.ac.cn/archives/3913>)），也可以用互信息做无监督学习（比如[《深度学习的互信息：无监督提取特征》](<https://spaces.ac.cn/archives/6024>)），互信息的重要性可见一斑。如果说有一个指标可以取代互信息、比互信息还方便，那肯定是笔者必须去学习的对象了。

### 问题定义

一般来说，我们将问题定义为：

> 有数据$(x_1, y_1),(x_2, y_2),\dots,(x_n,y_n)\sim p(x, y)$，判断$p(x, y)$是否恒等于$p(x), p(y)$，即$x,y$是否独立。

严格来讲，如果是对于连续变量，这里的“恒等于”指的是“几乎处处等于”，但我们这里不严格声明这一点。

为了描述的规范，这里设$x\in X, y\in Y$，而$f(x),g(y)\in \mathbb{R}$。注意$x,y$可能是两个含义完全不一样的变量，比如$x$可能是“星期一”，$y$可能是“上班”，$p(x,y)$就是“<u>今天是星期一，且今天要上班</u>”的概率。鉴于此，$X,Y$可能是两个完全不一样的域。

基本的思路是去计算互信息$\eqref{eq:i}$，但很多问题中我们都无法很好地估计概率或概率密度。一种可能的方案是转化为对偶问题，用类似对抗的思路去学习互信息（[infomax](<https://spaces.ac.cn/archives/6024>)的思路），但这种方法可能会不稳定，而且受到采样方案的影响。最好的方案就是能有一个类似“相关系数”的指标，让我们可以显式地计算和优化这个指标。

HSIC就是冲着这个目标而来的～

## HSIC

这里我们尽可能清晰地引入HSIC的概念。然而，“尽可能清晰”不等价于篇幅尽可能短，事实上，下面的篇幅依然会比较长，而且有不少数学公式，但是相对于标准教程里边一上来就引入希尔伯特空间、再生核、各种算子等做法，这里的介绍应该算是对很多不了解相关概念的读者来说都是友好的了。

### 基本思想

HSIC留意到：

> $p(x, y)\equiv p(x)p(y)$当且仅当对于任意的$f,g$，式
> $$\begin{equation}\begin{aligned}C[f,g]=&\iint p(x,y)f(x)g(y)dxdy - \iint p(x)p(y)f(x)g(y)dxdy\\
=&\mathbb{E}_{(x,y)\sim p(x,y)}[f(x)g(y)]-\mathbb{E}_{x\sim p(x)}[f(x)]\mathbb{E}_{y\sim p(y)}[g(y)]\end{aligned}\end{equation}$$
> 都等于0。

这个结论显然不难理解。有意思的是，第二个等号右边是采样的形式，也就是说我们将这个指标转化为了采样的形式，避免了直接估算概率密度。

这样一来，我们就有一个判断独立性的方法：选取“足够多”的$f,g$，然后计算
$$\begin{equation}L_H=\sum_{f,g} \big(C[f,g]\big)^2\label{eq:l0}\end{equation}$$
看$L_H$与0的接近程度；反过来，如果在优化问题中，我们希望特征$x,y$尽可能相互独立，那么我们就可以将$L_H$加入到损失函数中。

### 抽丝剥茧

其实$L_H$的形式已经很好地体现了HSIC的判别思想。下面我们就沿着这个思路，继续抽丝剥茧，逐步地走向HSIC最终的形式。

首先我们把$\big(C[f,g]\big)^2$算一算：
$$\begin{equation}\begin{aligned}\big(C[f,g]\big)^2=&\big(\mathbb{E}_{(x,y)\sim p(x,y)}[f(x)g(y)]\big)^2 + \big(\mathbb{E}_{x\sim p(x)}[f(x)]\big)^2 \big(\mathbb{E}_{y\sim p(y)}[g(y)]\big)^2\\
& - 2\big(\mathbb{E}_{(x,y)\sim p(x,y)}[f(x)g(y)]\big)\big(\mathbb{E}_{x\sim p(x)}[f(x)]\big)\big(\mathbb{E}_{y\sim p(y)}[g(y)]\big)\end{aligned}\end{equation}$$

然后我们用一个技巧：我们知道$\mathbb{E}_{x\sim p(x)}[f(x)]=\mathbb{E}_{x'\sim p(x')}[f(x')]$，说明了这个期望值的结果跟随机变量的记号没啥关系。所以我们有
$$\begin{equation}\begin{aligned}\big(\mathbb{E}_{x\sim p(x)}[f(x)]\big)^2=&\big(\mathbb{E}_{x_1\sim p(x)}[f(x_1)]\big)\big(\mathbb{E}_{x_2\sim p(x)}[f(x_2)]\big)\\
=&\mathbb{E}_{x_1\sim p(x),x_2\sim p(x)}[f(x_1)f(x_2)]\end{aligned}\end{equation}$$

把其余的项都这样变换，最终我们就可以得到
$$\begin{equation}\begin{aligned}\big(C[f,g]\big)^2=&\mathbb{E}_{(x_1,y_1)\sim p(x,y),(x_2,y_2)\sim p(x,y)}[f(x_1)f(x_2)g(y_1)g(y_2)] \\
& + \mathbb{E}_{x_1\sim p(x),x_2\sim p(x),y_1\sim p(y),y_2\sim p(y)}[f(x_1)f(x_2)g(y_1)g(y_2)]\\
& - 2 \mathbb{E}_{(x_1,y_1)\sim p(x,y),x_2\sim p(x),y_2\sim p(y)}[f(x_1)f(x_2)g(y_1)g(y_2)]\end{aligned}\end{equation}\label{eq:c}$$
这样一来，每一项都是$f(x_1)f(x_2)g(x_1)g(x_2)$的期望，只不过变量的采样分布不一样。

### 特征函数

现在的问题是：要选择哪些$f,g$呢？怎样才算“足够多”呢？

类比向量空间的知识，所有可能的$f(x)$能组成一个向量空间$\mathcal{F}$，所有的$g(y)$也一样组成一个向量空间$\mathcal{G}$。如果能把这两个空间的所有“基底”都遍历一遍，那肯定就够了。那问题就是：如何找到所有的基底呢？

这时候“核函数”就登场了。所谓核函数，那就是——呃，其实说起来很复杂，我也不大懂。简单来说，核函数是类似于线性代数中“正定矩阵”的存在，就是一个定义在$X\times X$的二元对称函数$K(x_1, x_2)=K(x_2, x_1)$，然后我们把一元函数$f(x)$类比为一个向量，那么
$$\begin{equation}\int K(x_1,x_2) f(x_2)dx_2\end{equation}$$
就相当于一个矩阵乘以向量的矩阵运算。跟矩阵的特征值和特征向量一样，核函数也能定义特征值和特征函数，满足下述恒等式的一元函数$\psi$就称为这个核函数的特征函数：
$$\begin{equation}\int K(x_1,x_2) \psi(x_2)dx_2=\alpha \psi(x_1)\end{equation}$$

上面的内容都是铺垫的，其严格定义则是属于“再生核希尔伯特空间“范畴。后面我们用到的，实际上是两点性质：

> 1、核函数的所有特征函数$\psi_1,\psi_2,\dots$，构成该空间的一组正交基；
> 
> 
> 
> 2、核函数的所有特征值$\alpha_1,\alpha_2,\dots$都是正的，且满足
> $$\begin{equation}K(x_1,x_2)=\sum_i \alpha_i \psi_i(x_1)\psi_i(x_2)\end{equation}\label{eq:k}$$

### HSIC登场

经过上述铺垫，HSIC基本上就可以登场了～

首先，假如我们已经有定义在$X\times X$的核函数$K_X(x_1,x_2)$，那么我们就可以算出$K_X(x_1,x_2)$对应的特征值$\alpha_1,\alpha_2,\dots$和特征函数$\psi_1,\psi_2,\dots$；同样地，有了定义在$Y\times Y$的核函数$K_Y(y_1,y_2)$后，也可以算出$K_Y(y_1,y_2)$对应的特征值$\beta_1,\beta_2,\dots$和特征函数$\phi_1,\phi_2,\dots$。

然后，因为特征函数构成了基底，所以在$\eqref{eq:l0}$中，我们可以把$f,g$换成对应特征函数$\psi_i,\phi_j$
$$\begin{equation}L_H=\sum_{i,j}\big(C[\psi_i, \phi_j]\big)^2\end{equation}$$
因为所有的特征值都是正的，所以我们还可以用特征值为权重进行加权求和，而不改变$L_H$的作用：
$$\begin{equation}L_H=\sum_{i,j}\alpha_i \beta_j\cdot\big(C[\psi_i, \phi_j]\big)^2\end{equation}$$
现在我们把$\eqref{eq:c}$代入到上面去，就得到
$$\begin{equation}\begin{aligned}L_H=&\mathbb{E}_{(x_1,y_1)\sim p(x,y),(x_2,y_2)\sim p(x,y)}\left[\sum_{i,j}\alpha_i \beta_j\psi_i(x_1)\psi_i(x_2)\phi_j(y_1)\phi_j(y_2)\right] \\
& + \mathbb{E}_{x_1\sim p(x),x_2\sim p(x),y_1\sim p(y),y_2\sim p(y)}\left[\sum_{i,j}\alpha_i \beta_j\psi_i(x_1)\psi_i(x_2)\phi_j(y_1)\phi_j(y_2)\right]\\
& - 2 \mathbb{E}_{(x_1,y_1)\sim p(x,y),x_2\sim p(x),y_2\sim p(y)}\left[\sum_{i,j}\alpha_i \beta_j\psi_i(x_1)\psi_i(x_2)\phi_j(y_1)\phi_j(y_2)\right]
\end{aligned}\end{equation}$$

最后，再利用等式$\eqref{eq:k}$，方括号里边的实际上就是$K_X(x_1,x_2)K_Y(y_1,y_2)$，于是，HSIC就登场了：
$$\begin{equation}\begin{aligned}HSIC(X,Y)=&\mathbb{E}_{(x_1,y_1)\sim p(x,y),(x_2,y_2)\sim p(x,y)}\left[K_X(x_1,x_2)K_Y(y_1,y_2)\right] \\
& + \mathbb{E}_{x_1\sim p(x),x_2\sim p(x),y_1\sim p(y),y_2\sim p(y)}\left[K_X(x_1,x_2)K_Y(y_1,y_2)\right]\\
& - 2 \mathbb{E}_{(x_1,y_1)\sim p(x,y),x_2\sim p(x),y_2\sim p(y)}\left[K_X(x_1,x_2)K_Y(y_1,y_2)\right]\end{aligned}\end{equation}\label{eq:hsic}$$
这就是我们最重要寻找的度量相关性的指标，它纯粹是采样的形式，而且$K_X,K_Y$都是事先给定的、通常是可微的，因此这就是一个可以明确采样计算、可以直接优化的指标！

在实际计算中，我们可选的核函数有很多，比较常用的是
$$\begin{equation}K(x_1, x_2) = \exp\left(-\frac{\Vert x_1 - x_2\Vert_2^2}{\sigma^2}\right)\end{equation}\label{eq:gk}$$
其中$\sigma > 0$是一个常数，本文开头提到的论文[《The HSIC Bottleneck: Deep Learning without Back\-Propagation》](<https://papers.cool/arxiv/1908.01580>)也是用这个核函数。不同的核函数效果有点不一样，但是都能保证$HSIC(X,Y)=0 \Leftrightarrow p(x,y)\equiv p(x)p(y)$。

### 矩阵形式

最后，我们来推导一下$\eqref{eq:hsic}$在有限样本下的矩阵形式。

按照采样求期望的思想，$\mathbb{E}_{(x_1,y_1)\sim p(x,y)}$实际上就是对所有的样本对$(x_i,y_i)$的结果求平均，而$\mathbb{E}_{(x_1,y_1)\sim p(x,y),(x_2,y_2)\sim p(x,y)}$其实就是将这个平均操作做两次，所以
$$\begin{equation}\mathbb{E}_{(x_1,y_1)\sim p(x,y),(x_2,y_2)\sim p(x,y)}\left[K_X(x_1,x_2)K_Y(y_1,y_2)\right]=\frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n \left[K_X(x_i,x_j)K_Y(y_i,y_j)\right]\end{equation}$$
其中$K_X(x_i,x_j),K_Y(y_i,y_j)$实际上都是$n\times n$的对称矩阵，分别记为$K_X,_y$，那么上述运算可以写成矩阵乘法$\frac{1}{n^2}\text{Tr}(K_X K_Y)$，其中$\text{Tr}$表示矩阵的迹。基于同样的思想，第二项实际上就是“$K_X$所有元素的平均乘以$K_Y$所有元素的平均”，如果非要写成矩阵形式的话，那就是$\frac{1}{n^4}\text{Tr}(K_X \boldsymbol{1}K_Y \boldsymbol{1})$，其中加粗的$\boldsymbol{1}$表示大小为$n\times n$的全1矩阵。相应地，最后一项是“$K_X K_Y$所有元素平均值的$1/n$的两倍”，即$\frac{2}{n^3}\text{Tr}(K_X K_Y \boldsymbol{1})$。

所以，如果用矩阵形式表示HSIC，那就是
$$\begin{equation}\begin{aligned}HSIC(X,Y)=&\frac{1}{n^2}\text{Tr}(K_X K_Y)+\frac{1}{n^4}\text{Tr}(K_X \boldsymbol{1}K_Y \boldsymbol{1})-\frac{2}{n^3}\text{Tr}(K_X K_Y \boldsymbol{1})\\
=&\frac{1}{n^2}\text{Tr}(K_X J K_Y J)
\end{aligned}\end{equation}$$
这里的$J = \boldsymbol{I}-\boldsymbol{1}/n$，而$\boldsymbol{I}$是$n$阶单位矩阵。跟[《简述无偏估计和有偏估计》](<https://spaces.ac.cn/archives/6747>)一文讨论的类似，这其实是一个有偏估计，而将前面的$1/n$换成$1/(n-1)$，就得到无偏估计：
$$\begin{equation}HSIC(X,Y)=\frac{1}{(n-1)^2}\text{Tr}(K_X J K_Y J)\end{equation}\label{eq:hsic-m}$$
这就是最终的矩阵形式的HSIC公式（注意$J$里边的$1/n$不用换成$1/(n-1)$）。

## 其它

这里先给出一个参考实现，并做一个简单的实验，来验证HSIC的有效性，然后在后一节中，我们思考一下HSIC可能存在的问题。

### 参考实现

假如已知核矩阵$K_X,K_Y$的情况下，HSIC的计算实现参考如下：



<pre><code class="language-python">import numpy as np


def hsic(Kx, Ky):
    Kxy = np.dot(Kx, Ky)
    n = Kxy.shape[0]
    h = np.trace(Kxy) / n**2 + np.mean(Kx) * np.mean(Ky) - 2 * np.mean(Kxy) / n
    return h * n**2 / (n - 1)**2</code></pre>



注意这里的实现是根据$\eqref{eq:hsic}$每一项的含义来的，并非根据矩阵形式$\eqref{eq:hsic-m}$，事实上矩阵形式$\eqref{eq:hsic-m}$效率并不高（涉及到三次矩阵乘法）。

下面做一个简单的实验，验证HSIC的有效性：



<pre><code class="language-python"># 产生两组独立无关的随机变量
x = np.random.randn(1000)
y = np.random.randn(1000)

Kx = np.expand_dims(x, 0) - np.expand_dims(x, 1)
Kx = np.exp(- Kx**2) # 计算核矩阵

Ky = np.expand_dims(y, 0) - np.expand_dims(y, 1)
Ky = np.exp(- Ky**2) # 计算核矩阵

print(hsic(Kx, Ky)) # 计算HSIC</code></pre>



输出结果大概是0\.0002左右，如果将$x,y$改为



<pre><code class="language-python">x = np.random.randn(1000)
y = x + 0.1 * np.random.randn(1000)
</code></pre>



这意味着$x,y$有比较强的相关性，而HSIC的结果也表明了这一点，约等于0\.096，比0\.0002大了两个数量级以上，这表明了HSIC确实是有效的。（注意，HSIC的输出值一般只有相对比较的意义，其绝对值没有明显含义。）

### 个人思考

显然，由$\eqref{eq:hsic}$给出的HSIC的计算结果取决于核函数的选择。不管用哪个核函数，理论上都可以保证
$$\begin{equation}HSIC(X,Y)=0 \Leftrightarrow p(x,y)\equiv p(x)p(y)\end{equation}$$
但问题是，当$HSIC(X,Y) > 0$时，$X,Y$究竟有多相关呢？

这就相当依赖核函数选择和原始问题的背景了。从常用核函数$\eqref{eq:gk}$的形式我们大概可以感知到，核函数相当于两个样本之间的相似度，问题是什么样的相似度定义才是真正符合问题背景的，这并没有标准答案，而且通常都是很困难的问题。

例如，假设$x_1,x_2,x_3$分别代表三张图片，我们知道$\Vert x_1 - x_2\Vert_2 = 0$的话意味着$x_1,x_2$这两张图片完全一样，但是当$\Vert x_1 - x_2\Vert_2,\Vert x_1 - x_3\Vert_2$都不等于0时，我们不能因为$\Vert x_1 - x_2\Vert_2 < \Vert x_1 - x_3\Vert_2$就说图片$x_2$一定比图片$x_3$“看起来”更像$x_1$，因为范数$\Vert\cdot\Vert_2$不是我们视觉的一个完美的度量。

其实笔者认为这是所有核方法的通病，即核方法只能保证当某个指标等于0时就是我们的理想追求，但是当这个指标不等于0时，这个指标无法很好地度量我们跟理想的差距。良好的度量是要根据具体问题精心设计的，或者根据数据集通过类似GAN的方法自动把这个度量学习出来。

当然，也不能就此说HSIC就没有价值了，HSIC的价值在于它可以作为辅助目标来优化，就好比我们要训练一个图片自编码器，就算我们用GAN的思想，我们也通常会把原图片和重构图片的MSE作为辅助loss来使用。

## 文末

总的来说，本文以一种较为通俗直白的方法介绍了HSIC这个概念，介绍过程中涉及到了一些数学内容，但省去了严格的数学定义和论述，尽量只保持了比较核心的部分，私以为这种处理会使得读者更容易接受一些。对于追求严谨的读者，请多多包涵～
