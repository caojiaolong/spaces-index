# Softmax后传：寻找Top\-K的光滑近似

> 作者：苏剑林 · 科学空间 · 2024-09-19
>
> 原文：<https://spaces.ac.cn/archives/10373>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

Softmax，顾名思义是“soft的max”，是$\max$算子（准确来说是$\text{argmax}$）的光滑近似，它通过指数归一化将任意向量$\boldsymbol{x}\in\mathbb{R}^n$转化为分量非负且和为1的新向量，并允许我们通过温度参数来调节它与$\text{argmax}$（的one hot形式）的近似程度。除了指数归一化外，我们此前在[《通向概率分布之路：盘点Softmax及其替代品》](<https://spaces.ac.cn/archives/10145>)也介绍过其他一些能实现相同效果的方案。

我们知道，最大值通常又称Top\-1，它的光滑近似方案看起来已经相当成熟，那读者有没有思考过，一般的Top\-$k$的光滑近似又是怎么样的呢？下面让我们一起来探讨一下这个问题。

## 问题描述

设向量$\boldsymbol{x}=(x_1,x_2,\cdots,x_n)\in\mathbb{R}^n$，简单起见我们假设它们两两不相等，即$i\neq j \Leftrightarrow x_i\neq x_j$。记$\Omega_k(\boldsymbol{x})$为$\boldsymbol{x}$最大的$k$个分量的下标集合，即$|\Omega_k(\boldsymbol{x})|=k$以及$\forall i\in \Omega_k(\boldsymbol{x}), j \not\in \Omega_k(\boldsymbol{x})\Rightarrow x_i > x_j$。我们定义Top\-$k$算子$\mathcal{T}_k$为$\mathbb{R}^n\mapsto\{0,1\}^n$的映射：
\begin{equation}
[\mathcal{T}_k(\boldsymbol{x})]_i = \left\{\begin{aligned}1,\,\, i\in \Omega_k(\boldsymbol{x}) \\ 0,\,\, i \not\in \Omega_k(\boldsymbol{x})\end{aligned}\right.
\end{equation}
说白了，如果$x_i$属于最大的$k$个元素之一，那么对应的位置变成1，否则变成0，最终结果是一个Multi\-Hot向量，比如$\mathcal{T}_2([3,2,1,4]) = [1,0,0,1]$。

从$\boldsymbol{x}$到$\mathcal{T}_k(\boldsymbol{x})$实际上是一种硬指派（Hard Assignment）运算，它本质上是不连续的，没有保留关于$\boldsymbol{x}$的有效梯度，因此无法集成到模型中进行端到端训练。为了解决这个问题，我们就需要构造一个能够提供有效梯度信息的$\mathcal{T}_k(\boldsymbol{x})$的光滑近似——在有些文献中也称为“可微Top\-$k$算子（Differentiable Top\-$k$ Operator）”。

具体来说，我们定义集合
\begin{equation}\Delta_k^{n-1} = \left\{\boldsymbol{p}=(p_1,p_2,\cdots,p_n)\left|\, p_1,p_2,\cdots,p_n\in[0,1],\sum_{i=1}^n p_i = k\right.\right\}\end{equation}
那么我们要做的事情就是构建$\mathbb{R}^n\mapsto \Delta_k^{n-1}$的一个映射$\mathcal{ST}_k(\boldsymbol{x})$，并尽量满足如下性质
\begin{align}&{\color{red}{单调性}}:\quad [\mathcal{ST}_k(\boldsymbol{x})]_i \geq [\mathcal{ST}_k(\boldsymbol{x})]_j \,\,\Leftrightarrow\,\, x_i \geq x_j \\[8pt]
&{\color{red}{不变性}}:\quad \mathcal{ST}_k(\boldsymbol{x}) = \mathcal{ST}_k(\boldsymbol{x} + c),\,\,\forall c\in\mathbb{R} \\[8pt]
&{\color{red}{趋近性}}:\quad \lim_{\tau\to 0^+}\mathcal{ST}_k(\boldsymbol{x}/\tau) = \mathcal{T}_k(\boldsymbol{x}) \\
\end{align}
可以验证作为$\mathcal{ST}_1(\boldsymbol{x})$的Softmax是满足如上性质的，所以提出上述性质实际就是希望构建出来的$\mathcal{ST}_k(\boldsymbol{x})$能够成为Softmax的自然推广。当然，构建Top\-$k$的光滑近似本身就比Top\-1的要难，所以如果遇到困难的话，不必要严格遵循以上性质，只要能表明所构建的映射确实具备$\mathcal{T}_k(\boldsymbol{x})$的光滑近似的特性就行。

## 迭代构造

事实上，笔者很早之前就关注过该问题，首次讨论于2019年的文章[《函数光滑化杂谈：不可导函数的可导逼近》](<https://spaces.ac.cn/archives/6620>)中，当时称之为$\text{soft-}k\text{-max}$，并给出过一个迭代构造方案：

> 输入为$\boldsymbol{x}$，初始化$\boldsymbol{p}^{(0)}$为全0向量；
> 执行$\boldsymbol{x} = \boldsymbol{x} - \min(\boldsymbol{x})$（保证所有元素非负）
> 
> 
> 
> 对于$i=1,2,\dots,k$，执行：
>      $\boldsymbol{y} = (1 - \boldsymbol{p}^{(i-1)})\otimes\boldsymbol{x}$;
>      $\boldsymbol{p}^{(i)} = \boldsymbol{p}^{(i-1)} + \text{softmax}(\boldsymbol{y})$
> 
> 
> 
> 返回$\boldsymbol{p}^{(k)}$。

其实这个迭代的构造思路很简单，我们可以先将$\text{softmax}(\boldsymbol{y})$替换为$\mathcal{T}_1(\boldsymbol{y})$来理解，此时算法流程就是先保证分量非负，然后识别出Top\-1，再将Top\-1置零（最大值变成了最小值），接着识别出剩下的Top\-1，依此类推，最终的$\boldsymbol{p}_k$正好就是$\mathcal{T}_k(\boldsymbol{x})$。而$\text{softmax}(\boldsymbol{y})$作为$\mathcal{T}_1(\boldsymbol{y})$的光滑近似，在迭代过程中使用$\text{softmax}(\boldsymbol{y})$自然也就得到$\mathcal{T}_k(\boldsymbol{x})$的光滑近似了。

无独有偶，笔者发现在Stack Exchange上的提问[《Is there something like softmax but for top k values?》](<https://stats.stackexchange.com/a/454788>)中有回复者提出过一个相同思路的方案，它先定义了加权Softmax：
\begin{equation}[\text{softmax}(\boldsymbol{x};\boldsymbol{w})]_i = \frac{w_i e^{x_i}}{\sum\limits_{i=1}^n w_i e^{x_i}}\end{equation}
然后构建的迭代过程为

> 输入为$\boldsymbol{x}$，初始化$\boldsymbol{p}^{(0)}$为全0向量；
> 
> 
> 
> 对于$i=1,2,\dots,k$，执行：
>      $\boldsymbol{p}^{(i)} = \boldsymbol{p}^{(i-1)} + \text{softmax}(\boldsymbol{x}; 1 - \boldsymbol{p}^{(i-1)})$
> 
> 
> 
> 返回$\boldsymbol{p}^{(k)}$。

这跟笔者所提的迭代过程在思路上是完全一样的，只不过笔者把$1 - \boldsymbol{p}_{i-1}$乘到了$\boldsymbol{x}$上而它则乘到了$e^{\boldsymbol{x}}$上，借助$e^{\boldsymbol{x}}$本身的非负性简化了流程。然而，这个迭代实际上是错误的，它<u>不符合“趋近性”</u>，比如当$k=2$时，代入$\boldsymbol{x}/\tau$取$\tau\to 0^+$的极限并不是Multi\-Hot向量，而是最大值变为1\.5、次大值变为0\.5、其余都变为0的向量，这是因为$1-p_{\max}$跟$e^{-x_{\max}}$大致上是同阶的，$1-p_{\max}$乘到$e^{x_{\max}}$上并不能完全消除最大值。

## 作为梯度

迭代构造全凭经验，可能会隐藏一些难以发现的问题，比如看起来更简单的加权Softmax迭代实际上不合理。由于没有更贴近本质的原理指导，这些方案往往也难以理论分析，比如笔者构造的迭代，虽然测起来没有问题，但我们很难证明$\boldsymbol{p}_k$分量都在$[0,1]$内，也很难判断它是否满足单调性。

所以，我们希望得到一个更高观点的原理去指导和设计这个光滑近似。就在前几天，笔者突然意识到一个关键的事实
\begin{equation}\mathcal{T}_k(\boldsymbol{x}) = \nabla_{\boldsymbol{x}} \sum_{i\in\Omega_k(\boldsymbol{x})} x_i\end{equation}
也就是说，最大的$k$个分量的和，它的梯度正好是$\mathcal{T}_k(\boldsymbol{x})$。所以我们似乎可以改为找$\sum\limits_{i\in\Omega_k(\boldsymbol{x})} x_i$的光滑近似，然后求梯度就得到$\mathcal{T}_k(\boldsymbol{x})$的光滑近似了，前者是一个标量，它的光滑近似找起来更容易一些，比如利用恒等式
\begin{equation}\sum_{i\in\Omega_k(\boldsymbol{x})} x_i = \max_{i_1 < \cdots < i_k} (x_{i_1} + \cdots + x_{i_k})\end{equation}
即遍历所有$k$个分量之和取最大值，这样一来，问题变成了找$\max$的光滑近似，这个我们早已解决（参考[《寻求一个光滑的最大值函数》](<https://spaces.ac.cn/archives/3290>)），答案是$\text{logsumexp}$：
\begin{equation}\max_{i_1 < \cdots < i_k} (x_{i_1} + \cdots + x_{i_k})\approx \log\sum_{i_1 < \cdots < i_k} e^{x_{i_1} + \cdots + x_{i_k}}\triangleq \log Z_k\end{equation}
对它求梯度，我们就得到$\mathcal{ST}_k(\boldsymbol{x})$的一个形式：
\begin{equation}[\mathcal{ST}_k(\boldsymbol{x})]_i = \frac{\sum\limits_{i_2 < \cdots < i_k} e^{x_i+x_{i_2} + \cdots + x_{i_k}}}{\sum\limits_{i_1 < \cdots < i_k} e^{x_{i_1} +x_{i_2}+ \cdots + x_{i_k}}}\triangleq \frac{Z_{k,i}}{Z_k}\label{eq:k-max-grad}\end{equation}
分母是所有$k$分量和的指数和，分子则是所有包含$x_i$的$k$分量和的指数和。根据这个形式，我们可以轻松证明
\begin{equation}0 < [\mathcal{ST}_k(\boldsymbol{x})]_i < 1,\quad \sum_{i=1}^n [\mathcal{ST}_k(\boldsymbol{x})]_i = k\end{equation}
所以这样定义的$\mathcal{ST}_k(\boldsymbol{x})$确实是属于$\Delta_k^{n-1}$的。事实上，我们还可以证明它同时满足单调性、不变性和趋近性，并且$\mathcal{ST}_1(\boldsymbol{x})$它就是Softmax，这些特性显示它确实是Softmax对于Top\-$k$算子的自然推广，我们暂且称之为“<strong>GradTopK</strong>（Gradient\-guided Soft Top\-k operator）”。

不过还没到庆祝时刻，因为式$\eqref{eq:k-max-grad}$的数值计算问题还没解决。如果直接按照式$\eqref{eq:k-max-grad}$来计算的话，分母就涉及到$C_n^k$项指数求和，计算量非常可观，所以必须找到一个高效的计算方法。我们已经分别记了分子、分母为$Z_{k,i},Z_k$，可以观察到分子$Z_{k,i}$满足递归式
\begin{equation}Z_{k,i} = e^{x_i}(Z_{k-1} - Z_{k-1,i})\end{equation}
结合$Z_{k,i}$对$i$求和等于$kZ_k$的事实，我们可以构建一个递归计算过程：
\begin{equation}\begin{aligned}
\log Z_{k,i} =&\, x_i + \log(e^{\log Z_{k-1}} - e^{\log Z_{k-1,i}}) \\
\log Z_k =&\, \left(\log\sum_{i=1}^n e^{\log Z_{k,i}}\right) - \log k \\
\end{aligned}\end{equation}
其中$\log Z_{1,i} = x_i$，为了减少溢出风险，我们对两端都取了对数。现在只需要迭代$k$步就可以完成$\mathcal{ST}_k(\boldsymbol{x})$的计算，效率上是可以接受的。然而，即便做了对数化处理，上述递归也只能对小方差的$\boldsymbol{x}$或者比较小的$k$算一下，反之$\log Z_{k-1}$与最大的$\log Z_{k-1,i}$就会相当接近，当数值上无法区分时就会出现$\log 0$的Bug，个人认为这是这种递归转化的根本困难。

一个非常简陋的参考实现：



<pre><code class="language-python">import numpy as np

def GradTopK(x, k):
    for i in range(1, k + 1):
        logZs = x if i == 1 else x + logZ + np.log(1 - np.exp(logZs - logZ))
        logZ = np.logaddexp.reduce(logZs) - np.log(i)
    return np.exp(logZs - logZ)

k, x = 10, np.random.randn(100)
GradTopK(x, k)</code></pre>



## 待定常数

上一节通过梯度来构建Top\-$k$光滑近似的思路，确实能给人一种高屋建瓴的美感，但可能也会有读者觉得它过于抽象，缺乏一种由表及里的直观感，同时对于大方差的$\boldsymbol{x}$或者比较大的$k$的数值不稳定性，也让我们难以完全满意现有结果。所以，接下来我们将探究一种自下而上的构建思路。

### 方法大意

该思路来源于Stack Exchange上的另一个帖子[《Differentiable top\-k function》](<https://math.stackexchange.com/a/4506773>)的回复。设$f(x)$是任意$\mathbb{R}\mapsto [0,1]$的、光滑的、单调递增的函数，并且满足$\lim\limits_{x\to\infty}f(x) = 1,\lim\limits_{x\to-\infty}f(x) = 0$。看上去条件很多，但实际上这种函数构造起来没有什么难度，比如经典的Sigmoid函数$\sigma(x)=1/(1+e^{-x})$，还有$\text{clip}(x,0,1)$、$\min(1, e^x)$等。接着我们考虑
\begin{equation}f(\boldsymbol{x}) = [f(x_1),f(x_2),\cdots,f(x_n)]\end{equation}
$f(\boldsymbol{x})$跟我们想要的$\mathcal{ST}_k(\boldsymbol{x})$差多远呢？每个分量都在$[0,1]$内肯定是满足了，但是分量之和等于$k$无法保证，所以我们引入一个跟$\boldsymbol{x}$相关的待定常数$\lambda(x)$来保证这一点：
\begin{equation}\mathcal{ST}_k(\boldsymbol{x}) \triangleq f(\boldsymbol{x} - \lambda(\boldsymbol{x})),\quad \sum_{i=1}^n f(x_i - \lambda(\boldsymbol{x})) = k\end{equation}
也就是通过分量之和为$k$来反解出$\lambda(\boldsymbol{x})$，我们可以称之为“<strong>ThreTopK</strong>（Threshold\-adjusted Soft Top\-k operator）”，如果读者已经阅读过[《通向概率分布之路：盘点Softmax及其替代品》](<https://spaces.ac.cn/archives/10145>)，就会发现这个做法跟Sparsemax、Entmax\-$\alpha$是一样的。

ThreTopK会是我们理想的$\mathcal{ST}_k(\boldsymbol{x})$吗？还真是！首先我们假设了$f$的单调性，所以单调性是满足的，其次$f(\boldsymbol{x} - \lambda(\boldsymbol{x}))=f(\boldsymbol{x}+c - (c+\lambda(\boldsymbol{x})))$，也就是常数可以收纳到$\lambda(\boldsymbol{x})$里边，所以不变性也是满足的。最后，当$\tau\to 0^+$时，我们可以找到一个适当的阈值$\lambda(\boldsymbol{x}/\tau)$，使得$\boldsymbol{x}/\tau-\lambda(\boldsymbol{x}/\tau)$最大的$k$个分量趋于$\infty$，剩下的分量趋于$-\infty$，从而$f(\boldsymbol{x}/\tau-\lambda(\boldsymbol{x}/\tau))$就等于$\mathcal{T}_k(\boldsymbol{x})$，也就是说满足趋近性。

### 解析求解

既然已经证明了ThreTopK的理论优越性，那么接下来要解决的就是$\lambda(\boldsymbol{x})$的计算了，这个大部份情况下都只能诉诸数值计算方法，不过对于$f(x)=\min(1, e^x)$，我们可以求得一个解析解。

求解的思路跟之前的Sparsemax是一样的。不失一般性，假设$\boldsymbol{x}$的分量已经从大到小排好顺序，即$x_1 > x_2 > \cdots > x_n$，接着假设我们已经知道$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$，那么此时
\begin{equation}k = \sum_{i=1}^n \min(1, e^{x_i - \lambda(\boldsymbol{x})}) = m + \sum_{i=m+1}^n e^{x_i - \lambda(\boldsymbol{x})}\end{equation}
由此解得
\begin{equation}\lambda(\boldsymbol{x})=\log\left(\sum_{i=m+1}^n e^{x_i}\right) - \log(k-m)\end{equation}
由此我们可以看出，当$k=1$时，$m$只能取$0$，此时可以发现ThreTopK正好就是Softmax；当$k > 1$时，我们无法事先确定$m$的值，所以只能遍历$m=0,1,\cdots,k-1$，根据上式计算$\lambda(\boldsymbol{x})$，寻找满足$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$的$\lambda(\boldsymbol{x})$。下面同样是一个非常简陋的参考实现：



<pre><code class="language-python">import numpy as np

def ThreTopK(x, k):
    x_sort = np.sort(x)
    x_lamb = np.logaddexp.accumulate(x_sort)[-k:] - np.log(np.arange(k) + 1)
    x_sort_shift = np.pad(x_sort[-k:][1:], (0, 1), constant_values=np.inf)
    lamb = x_lamb[(x_lamb &lt;= x_sort_shift) &amp; (x_lamb &gt;= x_sort[-k:])]
    return np.clip(np.exp(x - lamb), 0, 1)

k, x = 10, np.random.randn(100)
ThreTopK(x, k)</code></pre>



### 通用结果

从原理和代码都可以看出，$f(x)=\min(1, e^x)$时的ThreTopK几乎不会出现数值稳定性问题，并且$k=1$时能退化为Softmax，这些都是它的优势。然而，$\min(1, e^x)$实际上也算不上完全光滑（除了当$k=1$时$\min$不起作用），它在$x=0$处是不可导的。如果介意这一点，那么我们就需要选择处处可导的$f(x)$，比如$\sigma(x)$。

下面我们以$f(x)=\sigma(x)$为例，此时我们无法求出$\lambda(\boldsymbol{x})$的解析解，不过由于$\sigma(x)$的单调递增性，所以函数
\begin{equation}F(\lambda)\triangleq \sum_{i=1}^n \sigma(x_i - \lambda)\end{equation}
关于$\lambda$是单调递减的，因此$F(\lambda(\boldsymbol{x}))=k$的数值求解并不困难，二分法、牛顿法均可。以二分法为例，不难看出$\lambda(\boldsymbol{x})\in[x_{\min} - \sigma^{-1}(k/n), x_{\max} - \sigma^{-1}(k/n)]$，这里$\sigma^{-1}$是$\sigma$的逆函数，从这个初始区间出发，逐步二分到指定精度即可：



<pre><code class="language-python">import numpy as np

def sigmoid(x):
    y = np.exp(-np.abs(x))
    return np.where(x &gt;= 0, 1, y) / (1 + y)

def sigmoid_inv(x):
    return np.log(x / (1 - x))

def ThreTopK(x, k, epsilon=1e-4):
    low = x.min() - sigmoid_inv(k / len(x))
    high = x.max() - sigmoid_inv(k / len(x))
    while high - low &gt; epsilon:
        lamb = (low + high) / 2
        Z = sigmoid(x - lamb).sum()
        low, high = (low, lamb) if Z &lt; k else (lamb, high)
    return sigmoid(x - lamb)

k, x = 10, np.random.randn(100)
ThreTopK(x, k)</code></pre>



因此，$\lambda(\boldsymbol{x})$的数值计算并没有太大困难，真正的困难是当我们用数值方法去计算$\lambda(\boldsymbol{x})$时，往往会丢失$\lambda(\boldsymbol{x})$关于$\boldsymbol{x}$的梯度，从而影响端到端的训练。针对这个问题，我们可以手动把$\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})$算出来，然后自定义反向传播过程。具体来说，我们对
\begin{equation}\sum_{i=1}^n \sigma(x_i - \lambda(\boldsymbol{x})) = k\end{equation}
两边求某个$x_j$的偏导数，得到
\begin{equation}\sigma'(x_j - \lambda(\boldsymbol{x}))-\sum_{i=1}^n \sigma'(x_i - \lambda(\boldsymbol{x}))\frac{\partial\lambda(\boldsymbol{x})}{\partial x_j} = 0\end{equation}
那么
\begin{equation}\frac{\partial\lambda(\boldsymbol{x})}{\partial x_j} = \frac{\sigma'(x_j - \lambda(\boldsymbol{x}))}{\sum\limits_{i=1}^n \sigma'(x_i - \lambda(\boldsymbol{x}))}\end{equation}
其中$\sigma'$是$\sigma$的导函数。现在我们有了$\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})$的表达式，它的每一项都是可计算的（$\lambda(\boldsymbol{x})$也已经由数值方法求出），我们可以直接指定它作为反向传播的结果。一个比较简单且通用的实现方法是利用$\text{stop_gradient}$（下面简称$\text{sg}$）技巧，即在实现模型时将$\lambda(\boldsymbol{x})$替换为
\begin{equation}\boldsymbol{x}\cdot\text{sg}[\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})] + \text{sg}[\lambda(\boldsymbol{x}) - \boldsymbol{x}\cdot\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})]\end{equation}
其中$\cdot$是向量的内积。这样一来，前向传播时等价于$\text{sg}$不存在，所以结果是$\lambda(\boldsymbol{x})$，反向传播时被$\text{sg}$的部份梯度都是零，所以梯度就是给定的$\nabla_{\boldsymbol{x}}\lambda(\boldsymbol{x})$，这样我们就自定义了$\lambda(\boldsymbol{x})$的梯度，跟$\lambda(\boldsymbol{x})$如何计算得到的无关。

### 二者兼之

现在我们看到，$f(x)=\min(1,e^x)$有解析解，但它并非全局光滑；$f(x)=\sigma(x)$倒是足够光滑了，但它求解起来比较复杂。有没有兼顾两者优点的选择呢？还真有，笔者发现下述的$f(x)$是全局光滑并且$\lambda(\boldsymbol{x})$可以解析求解的：
\begin{equation}f(x) = \left\{\begin{aligned}1 - e^{-x}/2,\quad x\geq 0 \\ e^x / 2,\quad x < 0\end{aligned}\right.\end{equation}
也可以写成$f(x) = (1 - e^{-|x|})\text{sign}(x)/2+1/2$。可以验证$f(x)$其实也是一个S型函数，虽然它是分段函数，但它本身及其导函数在$x=0$处都是连续的，因此足够光滑了。

求解思路跟之前一样，不失一般性设$x_1 > x_2 > \cdots > x_n$，假设已经知道$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$，那么
\begin{equation}\begin{aligned}
k =&\, \sum_{i=1}^m (1 - e^{-(x_i - \lambda(\boldsymbol{x}))}/2) + \sum_{i=m+1}^n e^{x_i - \lambda(\boldsymbol{x})}/2 \\
=&\, m - \frac{1}{2}e^{\lambda(\boldsymbol{x})}\sum_{i=1}^m e^{-x_i} + \frac{1}{2}e^{-\lambda(\boldsymbol{x})}\sum_{i=m+1}^n e^{x_i}
\end{aligned}\end{equation}
由此解得
\begin{equation}\lambda(\boldsymbol{x})=\log\sum_{i=m+1}^n e^{x_i} - \log\left(\sqrt{(k-m)^2 + \left(\sum_{i=1}^m e^{-x_i}\right)\left(\sum_{i=m+1}^n e^{x_i}\right)}+(k-m)\right)\end{equation}
然后遍历$m=0,1,\cdots,n-1$，寻找满足$x_m \geq \lambda(\boldsymbol{x}) \geq x_{m+1}$的$\lambda(\boldsymbol{x})$即可。读者还可以尝试证明一下，当$k=1$时此$f(x)$下的ThreTopK也正好退化为Softmax。

参考实现：



<pre><code class="language-python">import numpy as np

def ThreTopK(x, k):
    x_sort = np.sort(x)
    lse1 = np.logaddexp.accumulate(x_sort)
    lse2 = np.pad(np.logaddexp.accumulate(-x_sort[::-1])[::-1], (0, 1), constant_values=-np.inf)[1:]
    m = np.arange(len(x) - 1, -1, -1)
    x_lamb = lse1 - np.log(np.sqrt((k - m)**2 + np.exp(lse1 + lse2)) + (k - m))
    x_sort_shift = np.pad(x_sort[1:], (0, 1), constant_values=np.inf)
    lamb = x_lamb[(x_lamb &lt;= x_sort_shift) &amp; (x_lamb &gt;= x_sort)]
    return (1 - np.exp(-np.abs(x - lamb))) * np.sign(x - lamb) * 0.5 + 0.5

k, x = 10, np.random.randn(100)
ThreTopK(x, k)</code></pre>



## 文章小结

本文探讨了Top\-k算子的光滑近似问题，它是Softmax等Top1的光滑近似的一般推广，提出了迭代构造、梯度指引、待定常数三种构造思路，并分析了它们的优缺点。
