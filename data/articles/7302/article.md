# AdaFactor优化器浅析（附开源实现）

> 作者：苏剑林 · 科学空间 · 2020-03-23
>
> 原文：<https://spaces.ac.cn/archives/7302>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

自从GPT、BERT等预训练模型流行起来后，其中一个明显的趋势是模型越做越大，因为更大的模型配合更充分的预训练通常能更有效地刷榜。不过，理想可以无限远，现实通常很局促，有时候模型太大了，大到哪怕你拥有了大显存的GPU甚至TPU，依然会感到很绝望。比如GPT2最大的版本有15亿参数，最大版本的T5模型参数量甚至去到了110亿，这等规模的模型，哪怕在TPU集群上也没法跑到多大的batch size。

这时候通常要往优化过程着手，比如使用混合精度训练（tensorflow下还可以使用一种叫做bfloat16的新型浮点格式），即省显存又加速训练；又或者使用更省显存的优化器，比如RMSProp就比Adam更省显存。本文则介绍<strong>AdaFactor</strong>，一个由Google提出来的新型优化器，首发论文为[《Adafactor: Adaptive Learning Rates with Sublinear Memory Cost》](<https://papers.cool/arxiv/1804.04235>)。<u>AdaFactor具有自适应学习率的特性，但比RMSProp还要省显存，并且还针对性地解决了Adam的一些缺陷。</u>

## Adam

首先我们来回顾一下常用的Adam优化器的更新过程。设$t$为迭代步数，$\alpha_t$为当前学习率，$L(\theta)$是损失函数，$\theta$是待优化参数，$\epsilon$则是防止溢出的小正数，那么Adam的更新过程为
\begin{equation}\left\{\begin{aligned}&g_t = \nabla_{\theta} L(\theta_{t-1})\\
&m_t = \beta_1 m_{t-1} + \left(1 - \beta_1\right) g_t\\
&v_t = \beta_2 v_{t-1} + \left(1 - \beta_2\right) g_t^2\\
&\hat{m}_t = m_t\left/\left(1 - \beta_1^t\right)\right.\\
&\hat{v}_t = v_t\left/\left(1 - \beta_2^t\right)\right.\\
&\theta_t = \theta_{t-1} - \alpha_t \hat{m}_t\left/\left(\sqrt{\hat{v}_t} + \epsilon\right)\right.
\end{aligned}\right.\end{equation}

要省显存，就首先得知道显存花在哪里的。首先，计算量和显存的大头肯定都是$\nabla_{\theta} L(\theta_{t-1})$，也就是说，计算梯度是很费资源的，这也是为啥“ALBERT相比BERT参数量虽然少了那么多，但训练速度也没见快多少”的原因了；除此之外，显存的消耗主要是$m,v$了，我们要维护两组缓存变量，来滑动计算梯度的前两阶矩（也就是$m$和$v$），用以计算参数的更新量。这两组变量每一组都跟训练参数本身一样大，因此对于参数比较多的模型，两组缓存变量所消耗的显存也不少。

## AdaFactor

在这一节中，我们会相对详细地介绍一些AdaFactor优化器，介绍中会设计比较多的公式和推导。如果只求一个大致了解的读者，可以自行跳过部分数学内容～

### 抛弃动量

我们知道，CV模型很多时候要靠“SGD\+动量”来炼出最优效果来，自适应学习率优化器通常训练不出最好的效果。但对于NLP模型来说，情况有点相反，自适应学习率显得更重要一些，很少听到由纯靠SGD调NLP模型的案例。因此，作为省显存的第一步，我们可以抛弃Adam里边的动量，这样就少一组缓存参数了，自然也就省了显存：
\begin{equation}\left\{\begin{aligned}&g_t = \nabla_{\theta} L(\theta_{t-1})\\
&v_t = \beta_2 v_{t-1} + \left(1 - \beta_2\right) g_t^2\\
&\hat{v}_t = v_t\left/\left(1 - \beta_2^t\right)\right.\\
&\theta_t = \theta_{t-1} - \alpha_t g_t\left/\sqrt{\hat{v}_t + \epsilon}\right.
\end{aligned}\right.\end{equation}
这其实就是RMSProp的变种，比RMSProp多了$\hat{v}_t = v_t\left/\left(1 - \beta_2^t\right)\right.$这一步。

### 低秩分解

去掉$m$之后，缓存变量直接减少了一半，但AdaFactor还不满意，它希望保留自适应学习率功能，但把缓存变量$v$的参数量再压一压。这一次，它用到了矩阵的低秩分解。

#### 广义KL散度

在SGD中，所有参数都是共用一个标量学习率；在Adam中，则是每一个参数都有自己的学习率$\alpha_t\left/\sqrt{\hat{v}_t + \epsilon}\right.$。我们知道通过精调学习率，SGD其实也能有不错的效果，这表明“每一个参数都有自己的学习率”这件事情都不是特别重要，或者换一种说法，就是“精调每一个参数自己的学习率”并不是特别重要。

这启发我们，将$\hat{v}_t$换一种参数更少的近似可能也就足够了。而“参数更少的近似”，我们就不难想到低秩分解了。对于$m\times n$的矩阵$C$，我们希望找到$m\times k$的矩阵$A$和$k\times n$的矩阵$B$，使得
\begin{equation}AB \approx C\end{equation}
当$k$足够小时，$A$、$B$的参数总量就小于$C$的参数量。为了“省”到极致，AdaFactor直接让$k=1$，即寻找$\{a_i\}_{i=1}^m$和$\{b_j\}_{j=1}^n$，使得
\begin{equation}a_i b_j \approx c_{i,j}\end{equation}
既然要近似，就要有一个度量的标准。很容易想到的标准是欧氏距离，即
\begin{equation}\sum_{i,j} (a_i b_j - c_{i,j})^2\end{equation}
但在这个距离之下，$a_i,b_j$并没有解析解；此外，在优化过程中$c_{i,j}$（即$\hat{v}_t$）是非负的，而通过上述目标优化出来的$a_i b_j$无法保证非负，因此很可能扰乱优化过程。

原论文的作者们很机智地换了一个度量标准，使得$a_i,b_j$有解析解。具体来说，它使用了“广义KL散度”，又称“I散度（[I\-Divergence](<https://arxiv.org/abs/math/0412070>)）”，其形式为：
\begin{equation}l = \sum_{i,j} c_{i,j}\log \frac{c_{i,j}}{a_i b_j} - c_{i,j} + a_i b_j \label{eq:i-div}\end{equation}
这个度量源自不等式$x\log x\geq x - 1(\forall x > 0)$，当且仅当$x=1$时等号成立。所以代入$x = p / q\,(p,q > 0)$，然后两端乘以$q$，我们有
\begin{equation}p\log \frac{p}{q} - p + q \geq 0\end{equation}
当且仅当$p=q$成立，如果$p,q$有多个分量，那么对多个分量的结果求和即可，这就得到了度量$\eqref{eq:i-div}$

显然，广义KL散度是概率的KL散度的自然推广，但它不要求$c_{i,j}$和$a_i b_j$满足归一化，只要求它们非负，这正好对应了AdaFactor的场景。而且巧妙的是，这种情形配上这个目标，刚好有解析解：
\begin{equation}a_i = \sum\limits_{j}c_{i,j},\quad b_j = \frac{\sum\limits_{i}c_{i,j}}{\sum\limits_{i,j}c_{i,j}}\label{eq:aibj}\end{equation}
其实这个解析解也很形象，就是行、列分别求和，然后相乘，再除以全体的和。

#### 推导过程

直接对$\eqref{eq:i-div}$求偏导数并让偏导数等于0，得
\begin{equation}\left\{\begin{aligned}
&\frac{\partial l}{\partial a_i}=\sum_j -\frac{c_{i,j}}{a_i} + b_j = 0\\
&\frac{\partial l}{\partial b_j}=\sum_i -\frac{c_{i,j}}{b_j} + a_i = 0
\end{aligned}\right.\end{equation}
整理得
\begin{equation}\left\{\begin{aligned}
&a_i \sum_{j} b_j = \sum_j c_{i,j}\\
&b_j \sum_{i} a_i = \sum_i c_{i,j}
\end{aligned}\right.\end{equation}
注意到如果$(a_i,b_j)$是一组最优解，那么$(\lambda a_i,b_j/\lambda)$也是，说白了，所有的$a_i$乘以一个常数，所有的$b_j$也除以这个常数，$a_i b_j$是不变的。那么我们就可以随意指定$\sum\limits_{i} a_i$或$\sum\limits_{j} b_j$，因为它们就只是一个缩放标量而已。不失一般性，我们指定$\sum\limits_{j} b_j=1$，那么就解得$\eqref{eq:aibj}$。

#### 直观理解

我们也可以从另一个角度理解结果$\eqref{eq:aibj}$。由于$c_{i,j}$是非负的，我们可以将它归一化，变成具有概率分布的特性，即$\hat{c}_{i,j}=\frac{c_{i,j}}{\sum\limits_{i,j}c_{i,j}}$，然后我们试图完成分解$\hat{c}_{i,j}\approx \hat{a}_i \hat{b}_j$，由于$\hat{c}_{i,j}$现在相当于一个二元联合概率分布，那么$\hat{a}_i,\hat{b}_j$就相当于它们的边缘分布，即
\begin{equation}\hat{a}_i = \sum_j \hat{c}_{i,j} = \frac{\sum\limits_{j}c_{i,j}}{\sum\limits_{i,j} c_{i,j}},\quad \hat{b}_j = \sum_i \hat{c}_{i,j} = \frac{\sum\limits_{i}c_{i,j}}{\sum\limits_{i,j}c_{i,j}}\end{equation}
现在$\hat{c}_{i,j}$到$c_{i,j}$还需要乘上一个$\sum\limits_{i,j}c_{i,j}$，我们可以把它乘到$\hat{a}_i$或$\hat{b}_j$中，不失一般性，我们假设乘到$\hat{a}_i$上，那么就得到$\eqref{eq:aibj}$。

#### AdaFactor雏形

有了结果$\eqref{eq:aibj}$后，我们就可以用它来构建更省内存的优化器了，这就是AdaFactor的雏形。简单来说，当参数$\theta$是普通一维向量时，优化过程保持不变；但$\theta$是$m\times n$的矩阵时，算出来的梯度$g_t$也是矩阵，从而$g_t^2$也是矩阵，这时候我们对$g_t^2$做低秩分解，然后维护两组缓存变量$v^{(r)}_t\in \mathbb{R}^m,v^{(c)}_t\in\mathbb{R}^n$，分别滑动平均低秩分解后的结果，最后用$v^{(r)}_t,v^{(c)}_t$共同调整学习率：
\begin{equation}\left\{\begin{aligned}&g_{i,j;t} = \nabla_{\theta} L(\theta_{i,j;t-1})\\
&v^{(r)}_{i;t} = \beta_2 v^{(r)}_{t-1;i} + \left(1 - \beta_2\right) \sum\limits_{j}\left(g_{i,j;t}^2+\epsilon\right)\\
&v^{(c)}_{j;t} = \beta_2 v^{(c)}_{t-1;j} + \left(1 - \beta_2\right) \sum\limits_{i}\left(g_{i,j;t}^2+\epsilon\right)\\
&v_{i,j;t} = v^{(r)}_{i;t} v^{(c)}_{j;t}\left/\sum\limits_{j}v^{(c)}_{j;t}\right.\\
&\hat{v}_t = v_t\left/\left(1 - \beta_2^t\right)\right.\\
&\theta_t = \theta_{t-1} - \alpha_t g_t\left/\sqrt{\hat{v}_t}\right.
\end{aligned}\right.\end{equation}
（把$\epsilon$加到$g_t^2$上去而不是$\hat{v}_t$上去，这是AdaFactor整出来的形式，不是笔者的锅～）

### 滑动权重

在Adam以及上述AdaFactor雏形中，滑动权重$\beta_2$都是恒为常数，AdaFactor指出这是不科学的，并提出新的策略。

#### 等价形式

为了认识到这一点，我们重写一下Adam的$\hat{v}_t$的更新过程：
\begin{equation}\begin{aligned}
\hat{v}_t =& v_t\left/\left(1 - \beta_2^t\right)\right.\\
=&\frac{\beta_2 v_{t-1} + (1-\beta_2) g_t^2}{1 - \beta_2^t}\\
=&\frac{\beta_2 \hat{v}_{t-1}\left(1 - \beta_2^{t-1}\right) + (1-\beta_2) g_t^2}{1 - \beta_2^t}\\
=&\beta_2\frac{1 - \beta_2^{t-1}}{1 - \beta_2^t}\hat{v}_{t-1} + \left(1 - \beta_2\frac{1 - \beta_2^{t-1}}{1 - \beta_2^t}\right)g_t^2
\end{aligned}\end{equation}
所以如果设$\hat{\beta}_{2,t}=\beta_2\frac{1 - \beta_2^{t-1}}{1 - \beta_2^t}$，那么更新公式就是
\begin{equation}\hat{v}_t =\hat{\beta}_{2,t}\hat{v}_{t-1} + \left(1 - \hat{\beta}_{2,t}\right)g_t^2\end{equation}
问题是这个$\hat{\beta}_{2,t}$够不够合理呢？答案是可能不大够。当$t=1$时$\hat{\beta}_{2,t}=0$，这时候$\hat{v}_t$就是$g_t^2$，也就是用实时梯度来校正学习率，这时候校正力度最大；当$t\to\infty$时，$\hat{\beta}_{2,t}\to \beta_2$，这时候$v_t$是累积梯度平方与当前梯度平方的加权平均，由于$\beta_2 < 1$，所以意味着当前梯度的权重$1 - \beta_2$不为0，这可能导致训练不稳定，因为训练后期梯度变小，训练本身趋于稳定，校正学习率的意义就不大了，因此学习率的校正力度应该变小，并且$t\to\infty$，学习率最好恒定为常数（这时候相当于退化为SGD），这就要求$t\to\infty$时，$\hat{\beta}_{2,t}\to 1$。

#### 新的衰减策略

为了达到这个目的，AdaFactor采用如下的衰减策略
\begin{equation}\hat{\beta}_{2,t} =1 - \frac{1}{t^c}\label{eq:beta2}\end{equation}
它满足$\hat{\beta}_{2,1}=0,\lim\limits_{t\to\infty} \hat{\beta}_{2,t}=1$。但即便如此，也不是任何$c$都适合，必须有$0 < c <1$。$c > 0$好理解，那为什么要$c < 1$呢？原论文包含了对它的分析，大家可以去读读，但笔者觉得原论文的推导过于晦涩，所以这里给出自己的理解。

首先，对于$\hat{v}_t$来说，一个很容易想到的方案是所有梯度平方的平均，即：
\begin{equation}\hat{v}_t = \frac{1}{t}\sum_{i=1}^t g_i^2=\frac{t-1}{t}\hat{v}_{t-1} + \frac{1}{t}g_t^2\end{equation}
所以这等价于让$\hat{\beta}_{2,t} =1 - \frac{1}{t}$。这个方案美中不足的一点是，每一步梯度都是平权的，这不符合直觉，因为正常来说越久远的梯度应该越不重要才对，所以应该适当降低历史部分权重，而当$c < 1$时，$1 - \frac{1}{t^c} < 1 - \frac{1}{t}$，因此一个简洁的方案是在式$\eqref{eq:beta2}$中取$c < 1$，AdaFactor默认的$c$是$0.8$。

### 层自适应

最后，我们还可以进一步根据参数的模长来校正更新量，这个思路来自[LAMB优化器](<https://papers.cool/arxiv/1904.00962>)，在之前的文章[《6个派生优化器的简单介绍及其实现》](<https://spaces.ac.cn/archives/7094#%E5%B1%82%E8%87%AA%E9%80%82%E5%BA%94>)中也介绍过。简单来说，它就是将最后的更新量标准化，然后乘以参数的模长，说白了，就是不管你怎么折腾，最后的更新量我只要你的方向，而大小由参数本身的模长和预先设置学习率共同决定，使得所有层所有参数的相对变化程度保持一致。

### AdaFactor完整版

至此，我们终于可以写出完整版AdaFactor的更新过程了：
\begin{equation}\left\{\begin{aligned}&g_{i,j;t} = \nabla_{\theta} L(\theta_{i,j;t-1})\\
&\hat{\beta}_{2,t} =1 - t^{-c}\\
&v^{(r)}_{i;t} = \hat{\beta}_{2,t} v^{(r)}_{t-1;i} + \left(1 - \hat{\beta}_{2,t}\right) \sum\limits_{j}\left(g_{i,j;t}^2+\epsilon_1\right)\\
&v^{(c)}_{j;t} = \hat{\beta}_{2,t} v^{(c)}_{t-1;j} + \left(1 - \hat{\beta}_{2,t}\right) \sum\limits_{i}\left(g_{i,j;t}^2+\epsilon_1\right)\\
&\hat{v}_{i,j;t} = v^{(r)}_{i;t} v^{(c)}_{j;t}\left/\sum\limits_{j}v^{(c)}_{j;t}\right.\\
&u_t = g_t\left/\sqrt{\hat{v}_t}\right.\\
&\hat{u}_t = u_t \left/\max\left(1, \left. RMS(u_t)\right/d\right)\right.\times \max\left(\epsilon_2, RMS(\theta_{t-1})\right)\\
&\theta_t = \theta_{t-1} - \alpha_t \hat{u}_t
\end{aligned}\right.\end{equation}
其中$RMS(x)=\sqrt{\frac{1}{n}\sum\limits_{i=1}^n x_i^2}$是模长的变种，$\max\left(1, \left. RMS(u_t)\right/d\right)$这一步相当于做了个截断，即$RMS(u_t) > d$时才执行归一化。原论文中的默认参数为
$$\begin{array}{c|c}
\hline
\epsilon_1 & 10^{-30}\\
\hline
\epsilon_2 & 10^{-3}\\
\hline
d & 1\\
\hline
\hat{\beta}_{2,t} & 1 - t^{-0.8}\\
\hline
\end{array}$$
如果参数是一维向量而不是矩阵，那么$\hat{v}_t$使用普通的更新公式$\hat{v}_t = \hat{\beta}_{2,t} v_{t-1} + \left(1 - \hat{\beta}_{2,t}\right) \left(g_t^2+\epsilon_1\right)$就行了。此外，论文还提出如果没有传入学习率，那么可以使用$a_t = \min\left(10^{-2},\frac{1}{\sqrt{t}}\right)$为默认学习率，但笔者看源码的时候发现这个默认学习率很少使用，基本上还是需要自己传入学习率的。

## 开源实现

为了方便大家使用，笔者开源了自己实现的AdaFactor：

> <strong>Github地址：[https://github\.com/bojone/adafactor](<https://github.com/bojone/adafactor>)</strong>

开源包括纯keras版和tf\.keras版，使用方法跟普通keras优化器一样，tf\.keras版也可以当做一个普通的tensorflow优化器使用。开源实现参考了[mesh\_tensorflow版的源码](<https://github.com/tensorflow/mesh/blob/63754cf4524cb96282ac0dfe453a15076a76589f/mesh_tensorflow/optimize.py#L204>)，在此表示感谢。优化器也已经内置在[bert4keras](<https://github.com/bojone/bert4keras/blob/master/bert4keras/optimizers.py>)中，方便大家调用。

需要提醒的是，用AdaFactor的时候，batch\_size最好大一些，因为本身低秩分解会带来误差，而如果batch\_size过小，那么梯度估算本身也带来较大的误差，两者叠加优化过程可能还不收敛。对于预训练模型来说，batch\_size通常还是很大的，所以现在不少预训练模型开始用AdaFactor优化器了；对于普通的下游任务来说，AdaFactor也可以尝试，但可能需要多炼炼丹，才能搞出优于无脑Adam的效果。对了，还要提醒一下，用AdaFactor的时候，学习率要设大一点，大概是$10^{-3}$级别为好，哪怕是finetune阶段也是如此。

## 文章小结

本文介绍了Google提出来的AdaFactor优化器，一个旨在减少显存占用的优化器，并且针对性地分析并解决了Adam的一些缺陷。笔者认为，AdaFactor针对Adam所做的分析相当经典，值得我们认真琢磨体味，对有兴趣研究优化问题的读者来说，更是一个不可多得的分析案例。

当然，没有什么绝对能有效的方法，有的只是

> 方法虽好，要想实际有效，依然要用心炼丹。
