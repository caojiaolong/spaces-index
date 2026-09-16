# MoE环游记：6、最优分配促均衡

> 作者：苏剑林 · 科学空间 · 2026-02-22
>
> 原文：<https://spaces.ac.cn/archives/11619>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我们知道，负载均衡（Load Balance）是MoE架构中基本且关键的一环，直接影响模型的效率和性能。本系列已经有两篇文章介绍了两种实现负载均衡的主流思路，分别是[《MoE环游记：2、不患寡而患不均》](<https://spaces.ac.cn/archives/10735>)介绍的经典方案Aux Loss，以及[《MoE环游记：3、换个思路来分配》](<https://spaces.ac.cn/archives/10757>)中的由DeepSeek提出的Loss\-Free方案。两者各有所长，亦各有局限。

本文将探讨第三种思路：<strong>最优分配</strong>，它将负载均衡视为等式约束下的线性规划问题。从最终形式上看，它仍属于Loss\-Free，但基于截然不同的原理，提供了更准确且无超参的更新方式。

## 方法回顾

现有的两种方法中，Aux Loss的思路相对朴素，核心是“哪里不稳罚哪里”，通过正则项对负载不均施加惩罚。然而，Aux Loss有两个问题：首先，惩罚系数不好调，过大会干扰主Loss的优化，过小则均衡效果差；其次，Aux Loss的背后是[STE（Straight\-Through Estimator）](<https://spaces.ac.cn/archives/6760#%E8%87%AA%E8%A1%8C%E8%AE%BE%E8%AE%A1%E6%A2%AF%E5%BA%A6>)，这意味着它的梯度是次优的，它可能会带来负载均衡以外的未知影响。

为此，DeepSeek提出了第二种方案Loss\-Free，它引入额外的偏置项来辅助排序，如下式所示：
\begin{equation}\boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho}} \rho_i \boldsymbol{e}_i\qquad\to\qquad \boldsymbol{y} = \sum_{i\in \mathop{\text{argtop}}_k \boldsymbol{\rho} + \boldsymbol{b}} \rho_i \boldsymbol{e}_i\end{equation}
注意$\boldsymbol{b}$只用于调整Expert的排序，乘到Expert上的还是$\rho_i$，所以它不直接参与模型的计算，也不会产生干扰梯度。不过，$\boldsymbol{b}$没有梯度也意味着我们需要给它单独定制更新规则，思路也很直观：$b_i$越大，第$i$个Expert被选中的概率就越大，所以它先统计出当前负载分布$\boldsymbol{F}$，若$F_i$超过期望值$1/n$则缩小$b_i$，否则增大$b_i$，即
\begin{equation}\newcommand{sign}{\mathop{\text{sign}}}\boldsymbol{b} \leftarrow \boldsymbol{b} - \gamma\sign(\boldsymbol{F} - 1/n)\end{equation}
总的来说，Loss\-Free对模型的“侵入”更少，确实称得上更简洁优雅，但也不尽完美。它虽然没有Aux Loss的惩罚系数，但也有一个$\gamma$参数要调，这相当于$\boldsymbol{b}$的学习率，论文推荐$\gamma=10^{-3}$，这实际上跟$\boldsymbol{\rho}$用Sigmoid激活紧密耦合，一旦换个激活函数，$\gamma$就需要重新调。

此外，即便我们用Sigmoid激活，但有些层的$\boldsymbol{\rho}$分布比较“畸形”，此时模型对$\gamma$就会比较敏感，恒定$\gamma$很难实现负载均衡。这种情况并不鲜见，比如模型前几层如果用MoE，往往就不容易均衡，所以才有了“first\_k\_dense”操作；又比如当模型比较大或Expert总数$n$比较大时，也容易出现个别层难以均衡的现象。

## 线性规划

让我们更准确地描述一下待解问题：假设有$m$个Token，第$i$个Token的Router对$n$个Expert的打分是$\boldsymbol{s}_i = (s_{i,1},s_{i,2},\cdots,s_{i,n})$，那么共有$mn$个分数这些分数可能有正有负，也不一定在预先给定的值域内。我们希望基于这些分数制定一个分配方案，决定每个Token应该激活哪些Expert。

我们约定每个Token要选$k$个Expert，那么一个基本方案就是选分数最高的$k$个Expert，但这个方案可能出现负载失衡问题，即某些Expert的激活次数可能会明显偏多/偏少，所以我们进一步约定每个Expert恰好被激活$mk/n$次。在这两个约束之下，我们求总分最高的分配方案，形式化为
\begin{equation}\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} = k,\quad \sum_i x_{i,j} = \frac{mk}{n}\label{eq:target}\end{equation}
其中$x_{i,j}=1$表示第$i$的Token选中了第$j$个Expert，$x_{i,j}=0$则表示不选，另外注意$mk/n$必须是整数，才能让两个等式约束严格成立，我们先假设这一点成立。这里每个Expert都严格激活$mk/n$次，是最理想的绝对均匀状态，实际上当然可以宽松一些，但理论推导中我们采用最严格的约束。

上式中$x_{i,j}$只能取$0$或$1$，这属于整数规划问题，离散优化通常都比较困难，我们考虑它的松弛版本：
\begin{equation}\max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} = k,\quad \sum_i x_{i,j} = \frac{mk}{n}\label{eq:relax}\end{equation}
现在$x_{i,j}$可以取$[0,1]$中的任意实数，其他条件则不变。注意到$x_{i,j}$关于目标函数和约束等式都是线性的，所以这就是有界区域内的一个线性规划问题。

## 极大极小

约束优化通常也不容易，所以我们考虑去掉约束的$\max\text{-}\min$形式（亦即“[拉格朗日乘子法](<https://en.wikipedia.org/wiki/Lagrange_multiplier>)”）：
\begin{equation}\max_{x_{i,j}\in[0,1]}\min_{\alpha_i,\beta_j} \sum_{i,j} x_{i,j}s_{i,j} - \sum_i \alpha_i\left(\sum_j x_{i,j} - k\right) - \sum_j \beta_j\left(\sum_i x_{i,j} - \frac{mk}{n}\right)\label{eq:relax-max-min}\end{equation}
如果$\sum_j x_{i,j} = k$且$\sum_i x_{i,j} = mk/n$，那么上式等价于式$\eqref{eq:relax}$，最大值是一个有限值；但若它们之一不满足，那么$\min$这一步就可以取到负无穷，最大值将是负无穷。显然负无穷不如有限值大，因此只能是前一种情况，即它等价于式$\eqref{eq:relax}$。

目标$\eqref{eq:relax-max-min}$关于$x_{i,j},\alpha_i,\beta_j$都是线性的，线性函数既凸又凹，且$[0,1]$是一个凸集，所以它满足[Minimax theorem](<https://en.wikipedia.org/wiki/Minimax_theorem>)的条件，我们可以交换$\max$和$\min$的顺序，即它等于
\begin{equation}\min_{\alpha_i,\beta_j} \max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}(s_{i,j} - \alpha_i - \beta_j) +  k\sum_i \alpha_i + \frac{mk}{n}\sum_j \beta_j\label{eq:relax-min-max}\end{equation}
上式中我们已经将$x_{i,j},\alpha_i,\beta_j$的项单独分离了出来。仔细观察可以发现，$\max$这一步其实可以直接求出来：当$s_{i,j} - \alpha_i - \beta_j > 0$时，我们直接取$x_{i,j}=1$以最大化目标；当它小于0时，我们则取$x_{i,j}=0$以最大化目标；当它等于0时，$x_{i,j}$取任意值结果都一样。即
\begin{equation}\left\{\begin{aligned}&\,x_{i,j}^* = 1, &\, s_{i,j} - \alpha_i - \beta_j > 0 \\
&\,x_{i,j}^* = 0, &\, s_{i,j} - \alpha_i - \beta_j < 0 \\
&\,x_{i,j}^* \in  [0,1], &\, s_{i,j} - \alpha_i - \beta_j = 0
\end{aligned}\right.\end{equation}
其中$s_{i,j} - \alpha_i - \beta_j = 0$属于比较特殊的情况，假设它的出现概率可以忽略不计，那么$x_{i,j}^*$就是非0即1；当然，即便$s_{i,j} - \alpha_i - \beta_j = 0$，由于$x_{i,j}^*$的任意性，我们也可以根据约束条件让$x_{i,j}^*$取0或1。由此可见，形式$\eqref{eq:relax}$虽然对原始问题$\eqref{eq:target}$进行了松弛，但它的最优解也是原始问题的最优解，两者完全等价。

## 分而治之

将上述$x_{i,j}^*$代入到式$\eqref{eq:relax-min-max}$得$x_{i,j}^*(s_{i,j} - \alpha_i - \beta_j) = \max(0, s_{i,j} - \alpha_i - \beta_j)$，所以优化目标$\eqref{eq:relax-min-max}$可简化成
\begin{equation}\min_{\alpha_i,\beta_j} \sum_{i,j} \max(0, s_{i,j} - \alpha_i - \beta_j) +  k\sum_i \alpha_i + \frac{mk}{n}\sum_j \beta_j\end{equation}
我们将使用交替最小化的思路去求解它：先固定$\beta_j$来求解$\alpha_i$，然后固定$\alpha_i$来求解$\beta_j$，交替执行。由于$\alpha_i,\beta_j$具有明显的对称性，所以这两个步骤实际上是同一个问题。我们先看固定$\beta_j$来求解$\alpha_i$，那么问题等价于
\begin{equation}\min_{\alpha_i} \sum_{i,j} \max(0, s_{i,j} - \alpha_i - \beta_j) +  k\sum_i \alpha_i\end{equation}
我们还可以观察到，每一项$\alpha_i$都是独立地累加起来的，所以我们可以将它分解为$m$个独立的子优化问题，这意味着可以暂时省略掉下标$i$，将问题进一步简化成
\begin{equation}\min_{\alpha}  k\alpha + \sum_j \max(0, s_j - \beta_j - \alpha)\end{equation}
将全体$s_j - \beta_j$从大到小排列成$s_{\sigma_1} - \beta_{\sigma_1} \geq s_{\sigma_2} - \beta_{\sigma_2} \geq \cdots \geq s_{\sigma_n} - \beta_{\sigma_n}$，即第$j$大的元素为$s_{\sigma_j} - \beta_{\sigma_j}$，然后假设我们已知$s_{\sigma_l} - \beta_{\sigma_l} \geq \alpha \geq s_{\sigma_{l+1}} - \beta_{\sigma_{l+1}}$，那么目标函数等于
\begin{equation}k\alpha + \sum_{j=1}^l (s_{\sigma_j} - \beta_{\sigma_j} - \alpha) = \left\{\begin{aligned}
&\,\sum_{j=1}^k (s_{\sigma_j} - \beta_{\sigma_j}) + \sum_{j=k+1}^l \underbrace{(s_{\sigma_j} - \beta_{\sigma_j} - \alpha)}_{\geq 0},&\, l \geq k \\
&\,\sum_{j=1}^k (s_{\sigma_j} - \beta_{\sigma_j}) - \sum_{j=l+1}^k \underbrace{(s_{\sigma_j} - \beta_{\sigma_j} - \alpha)}_{\leq 0},&\, l \leq k \\
\end{aligned}\right.\end{equation}
这表明无论$l > k$还是$l < k$都会放大目标函数，那么目标函数的最小值只能在$l=k$取到，此时结果不依赖于$\alpha$的具体取值，即$\alpha^*$可以是$s_j - \beta_j$的第$k$大和第$k+1$大元素之间的任意数，习惯起见，我们取$\alpha^*$为第$k+1$大元素。

## 交替迭代

恢复下标$i$，我们将得到：对任意给定$i$，$\alpha_i^*$是将全体$s_{i,j} - \beta_j$从大到小排列后的第$k+1$个元素。类似地，固定$\alpha_i$求$\beta_i$的结果是：对任意给定$j$，$\beta_j^*$是将全体$s_{i,j} - \alpha_i$从大到小排列后的第$mk/n+1$个元素。我们将这两个步骤交替执行，作为最终的求解算法。

假设我们已经求得了足够准确的$\boldsymbol{\alpha}^*$和$\boldsymbol{\beta}^*$，那么根据前面的分析，$x_{i,j}^*$会自动满足非0即1以及约束条件，而$x_{i,j}^*=1$对应于$s_{i,j} - \alpha_i^* - \beta_j^* > 0$，再结合约束条件$\sum_j x_{i,j}^* = k$，可以得出对于每个Token $i$，它选出的Expert必然是$\boldsymbol{s}_i - \boldsymbol{\beta}^*$的Top\-$k$。

这告诉我们，推理阶段只需用到$\boldsymbol{\beta}^*$，$\boldsymbol{\alpha}^*$只是求解过程的中间变量，训练完成后可以忽略不管。这一点至关重要，因为$\boldsymbol{\beta}$的大小是固定的$n$，而$\boldsymbol{\alpha}$的大小是$m$，$m$是全局Batch Size，它会动态变化，所以并非一个科学的推理格式。利用上这个特性的求解流程如下图所示，其中$\mathop{\text{desc_sort}}$表示降序排序。

$$\begin{array}{|l|}
\hline
\text{Quantile Balancing (QB):  问题}\eqref{eq:target}\text{的交替求解算法} \\[4pt]
\hline
\text{输入: 打分矩阵 }\boldsymbol{s}\in\mathbb{R}^{m\times n} \\
\text{输出: 分配方案 }\boldsymbol{x}\in\{0,1\}^{m\times n} \\[4pt]
\hline
\begin{array}{ll}
1: & \text{Initialize }\boldsymbol{\beta} = \boldsymbol{0}_{1\times n} \\
2: & \textbf{For }t=1,2,\cdots,T\textbf{ do } \\
3: & \qquad \boldsymbol{\alpha} \leftarrow \mathop{\text{desc_sort}}(\boldsymbol{s} - \boldsymbol{\beta}, \text{axis=1})_{[:, k:k+1]} \\
4: & \qquad \boldsymbol{\beta} \leftarrow \mathop{\text{desc_sort}}(\boldsymbol{s} - \boldsymbol{\alpha}, \text{axis=0})_{[mk/n:mk/n+1]} \\
5: & \text{Output } x_{i,j}=1 \text{ if } j\in\mathop{\text{argtop}}_k \boldsymbol{s}_i - \boldsymbol{\beta} \text{ else } 0
\end{array} \\
\hline
\end{array}$$

还有一个改进点是，我们可以用“分位数（Quantile）”的概念，将“$n$个数中的第$k+1$大元素、$m$个数中的第$mk/n+1$大元素”统一起来，它们实际上都是各自维度的“$1-k/n$分位数”，Numpy、Jax、Torch等数值框架都有“quantile”函数实现，利用它能避免对数据进行全排序操作，节省一些复杂度。

也正是为此，我们称这个算法为“<strong>Quantile Balancing (QB)</strong>”。

## 小心陷阱

但，还没到庆祝的时候，这里有一个不大起眼的陷阱，一旦掉进去就可能得到本质错误的结果，需要特别小心。

刚才说了，QB推理只用到$\boldsymbol{\beta}$，训练过程只存$\boldsymbol{\beta}$也够了，所以从Loss\-Free的角度看，QB是提供了一个新的Bias更新方法，称“Quantile Bias”也未尝不可。不管是原本的SignSGD还是本文的Quantile，它们都依赖于全体Token的打分，所以不能弄错顺序：必须用旧的$\boldsymbol{\beta}$选出当前批数据的Expert，然后再更新$\boldsymbol{\beta}$值，这样才能保证不会存在信息泄漏问题。

可能有读者疑惑：一个不直接参与前向计算的Bias向量，能泄漏什么东西？确实，直觉来想能泄漏的信息确实很少，但风险终究是存在的。也许训小模型我们可以试试泄漏版，但训大模型不应该冒这个风险，因为它大，能力强，强到可以放大任何细微的Bug。所以，保持训推一致、杜绝任何信息泄漏风险，是训练大模型的基本意识。

结合实际训练场景，QB还能再调整一下，原本是零初始化迭代$T$次，可以改为从上一步的Bias出发，每步只迭代一次，这样可以避免过拟合到当前批次上，同时降低计算量。基于$\boldsymbol{s} - \boldsymbol{\beta}$选Top\-$k$是原本MoE就要做的，现在只不过改为了选Top\-$(k+1)$，几乎不增加成本，所以多出的一步是用$\boldsymbol{s} - \boldsymbol{\alpha}$来找$\text{axis=0}$的“$1-k/n$分位数”。

$$\begin{array}{|l|}
\hline
\text{Quantile Balancing (QB) 实际使用形式} \\[4pt]
\hline
\text{输入: 打分矩阵 }\boldsymbol{s}\in\mathbb{R}^{m\times n}\text{， 上一步 }\boldsymbol{\beta}\in\mathbb{R}^n \\
\text{输出: 分配方案 }\boldsymbol{x}\in\{0,1\}^{m\times n}\text{， 新的 }\boldsymbol{\beta}\in\mathbb{R}^n \\[4pt]
\hline
\begin{array}{ll}
1: & x_{i,j}=1 \text{ if } j\in\mathop{\text{argtop}}_k \boldsymbol{s}_i - \boldsymbol{\beta} \text{ else } 0 \\
2: & \boldsymbol{\alpha} \leftarrow \mathop{\text{desc_sort}}(\boldsymbol{s} - \boldsymbol{\beta}, \text{axis=1})_{[:, k:k+1]} \\
3: & \boldsymbol{\beta} \leftarrow \mathop{\text{desc_sort}}(\boldsymbol{s} - \boldsymbol{\alpha}, \text{axis=0})_{[mk/n:mk/n+1]} \\
4: & \text{Output } \boldsymbol{x},\boldsymbol{\beta}
\end{array} \\
\hline
\end{array}$$

不过，即便只迭代一次，这新增的一步还是比较昂贵的，它需要在$m$个元素中找第$mk/n+1$大元素，$m$等于“全局样本数 \* 序列长度”，一般都是百万起步，在各种并行策略和梯度累积之下，精确实现通常难以接受。一个折中方案是，将样本分为我们能接受的最大的小批次，每个小批次单独按照公式算一个$\boldsymbol{\beta}$，然后将它们平均作为最终的结果。

解决这些问题后，QB几乎就都是优点了，首先它没有学习率这样的超参数要调，其次它均衡的速度非常快，尤其擅长处理极端例子，比如用它训练一个全MoE模型，第一层MoE也会变得特别均衡。不过，对于原本SignSGD规则就能均衡的层，QB通常不会更有优势。

[![第一层MoE的MaxVio对比图](<https://spaces.ac.cn/usr/uploads/2026/02/3461539179.png>)](<https://spaces.ac.cn/usr/uploads/2026/02/3461539179.png>)

第一层MoE的MaxVio对比图

## 演示代码

这里给出一段演示代码，有兴趣的读者可以自行体验一下：



<pre><code class="language-python">import numpy as np

def quantile_bias(s, k, T=5):
    """交替quantile求最优bias
    原理：https://kexue.fm/archives/11619
    """
    m, n = s.shape
    beta = np.zeros((1, n))
    for _ in range(T):
        alpha = np.quantile(s - beta, 1 - k / n, axis=1, keepdims=True)
        # alpha = alpha.clip(0, np.inf)  # BIP会多出这一步
        beta = np.quantile(s - alpha, 1 - k / n, axis=0, keepdims=True)
        # beta = beta.clip(0, np.inf)  # BIP会多出这一步
    return beta

def max_min_avg_vio(s, k):
    """计算max_vio、min_vio和avg_vio
    其中 max_vio ≥ 0, avg_vio ≥ 0, -1 ≤ min_vio ≤ 0，三者都是越接近于0表示越均衡
    """
    m, n = s.shape
    topk = np.argsort(-s, axis=1)[:, :k]
    f = np.bincount(topk.reshape(-1), minlength=n)
    f = f / f.sum() * n - 1
    return f.max(), f.min(), np.abs(f).mean()

m, n, k = 100000, 256, 8
s = np.random.rand(m, n) + np.random.rand(n)  # 模拟一个不均匀的打分

b = quantile_bias(s, k, 5)
max_min_avg_vio(s, k)  # 直接取top-k的max_vio、min_vio和avg_vio
max_min_avg_vio(s - b, k)  # 减去偏置后top-k的max_vio、min_vio和avg_vio</code></pre>



## 相关工作

从最优分配视角看待MoE负载均衡的做法最早见于论文[《BASE Layers: Simplifying Training of Large, Sparse Models》](<https://papers.cool/arxiv/2103.16716>)，但完整地提出一般解法应该是论文[《Binary\-Integer\-Programming Based Algorithm for Expert Load Balancing in Mixture\-of\-Experts Models》](<https://papers.cool/arxiv/2502.15451>)（简称BIP），QB也正是改进自BIP。

相比QB的目标$\eqref{eq:target}$，BIP是将等式约束改为了不等式约束：
\begin{equation}\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} \leq k,\quad \sum_i x_{i,j} \leq \frac{mk}{n}\label{eq:target-leq}\end{equation}
然后重复前面一系列推导，在$\max\text{-}\min$形式这一步，会多出一个$\alpha_i\geq 0, \beta_j\geq 0$的约束，即我们要考虑
\begin{equation}\max_{x_{i,j}\in[0,1]}\min_{\alpha_i\geq 0,\beta_j\geq 0} \sum_{i,j} x_{i,j}s_{i,j} - \sum_i \alpha_i\left(\sum_j x_{i,j} - k\right) - \sum_j \beta_j\left(\sum_i x_{i,j} - \frac{mk}{n}\right)\end{equation}
才能使得它跟问题$\eqref{eq:target-leq}$的松弛版本等价。接着，$\max$和$\min$依然可以交换，重复相似的推导过程，最终$\boldsymbol{\alpha},\boldsymbol{\beta}$的迭代规则后会多出一个$\max(0,\cdot)$的截断操作，确保它们非负：
\begin{equation}\begin{aligned}
\boldsymbol{\alpha} \leftarrow &\, \max(0, \mathop{\text{desc_sort}}(\boldsymbol{s} - \boldsymbol{\beta}, \text{axis=1})_{[:, k:k+1]}) \\
\boldsymbol{\beta} \leftarrow &\, \max(0, \mathop{\text{desc_sort}}(\boldsymbol{s} - \boldsymbol{\alpha}, \text{axis=0})_{[mk/n:mk/n+1]})
\end{aligned}\end{equation}
然而，笔者测试发现，这个截断操作会明显降低负载均衡的速度，而且经常出现只能“劫富”、不能“济贫”的例子（频率极高的Expert会被打压下来，但频率极低的Expert则无法抢救过来），说明这个截断操作完全是反向优化。因此，QB直接将原始问题改为等式约束，这可以去掉$\boldsymbol{\alpha},\boldsymbol{\beta}$的非负约束，使得求解最简化。

此外，BIP看上去犯了前一节说的错误，它的做法是先更新$\boldsymbol{\beta}$再选Top\-$k$，这违反了因果法则（泄漏未来信息）和训推一致原则（跨样本）。但尽管如此，BIP从最优分配角度对负载均衡所做的完整分析，仍极具启发意义。

经过搜索，笔者发现在BIP之后，还有一些工作沿着这个方向做了一些探索，它们跟本文有一些重叠之处，但都不完全一致，这里就不一一展开讨论了：

> [《Maximum Score Routing For Mixture\-of\-Experts》](<https://papers.cool/arxiv/2508.12801>)
> 
> 
> 
> [《Selective Sinkhorn Routing for Improved Sparse Mixture of Experts》](<https://papers.cool/arxiv/2511.08972>)
> 
> 
> 
> [《MicroMoE: Fine\-Grained Load Balancing for Mixture\-of\-Experts with Token Scheduling》](<https://papers.cool/arxiv/2511.16947>)
> 
> 
> 
> [《A Theoretical Framework for Auxiliary\-Loss\-Free Load Balancing of Sparse Mixture\-of\-Experts in Large\-Scale AI Models》](<https://papers.cool/arxiv/2512.03915>)

## 梯度下降

最后，我们再介绍一个介乎Loss\-Free和QB之间的求解方案。我们知道，在QB的迭代格式中，$\boldsymbol{\alpha}$的计算是比较便宜的，真正昂贵的是$\boldsymbol{\beta}$的计算，它需要跨全体Token来做某种排序。假设给定$\boldsymbol{\alpha}$，那么$\boldsymbol{\beta}$的优化目标是
\begin{equation}\min_{\beta_j} \underbrace{\sum_{i,j} \max(0, s_{i,j} - \alpha_i - \beta_j) +  \frac{mk}{n}\sum_j \beta_j}_{\text{记为}\ell}\end{equation}
除了用Quantile去求它的最优解外，还有没有更便宜的方案可以求它的近似解呢？还真有！很明显目标函数$\ell$是可导的，所以我们完全可以考虑梯度下降，它的梯度是
\begin{equation}\frac{\partial\ell}{\partial\beta_j} = \frac{mk}{n} - \sum_{i=1}^m \chi(s_{i,j} - \alpha_i - \beta_j > 0)\end{equation}
其中$\chi$是示性函数，$\chi(\text{True})=1,\chi(\text{False})=0$。显然这个梯度的计算也相对便宜，有了梯度后，就可以进行梯度下降了，为了跟Loss\-Free对齐，我们考虑SignSGD
\begin{equation}\beta_j \leftarrow \beta_j - \gamma\sign\left(\frac{\partial\ell}{\partial\beta_j}\right)\end{equation}
用它来替换QB中Quantile更新$\boldsymbol{\beta}$这一步，就可以做到跟Loss\-Free相近的成本，实测效果也介乎两者之中（跟Loss\-Free都是Sigmoid激活，并且取同一个$\gamma$的情况下）。事实上，我们还可以证明，如果分数矩阵$\boldsymbol{s} - \boldsymbol{\beta}$每一行的第$k$大和第$k+1$大元素都不相等，那么这个方案严格等同于Loss\-Free。

## 文章小结

这篇文章中我们从最优分配角度探讨了MoE的负载均衡问题，得出了一种新的无Aux Loss的负载均衡算法Quantile Balancing，它比已有的Loss\-Free方案更稳定和准确，适用于任意值域的Router Scores，并且没有额外的超参数要调。
