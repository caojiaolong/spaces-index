# MoE环游记：7、动态激活极简解

> 作者：苏剑林 · 科学空间 · 2026-02-23
>
> 原文：<https://spaces.ac.cn/archives/11626>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章[《MoE环游记：6、最优分配促均衡》](<https://spaces.ac.cn/archives/11619>)中，我们通过求解如下最优分配问题来实现负载均衡
\begin{equation}\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_j x_{i,j} = k,\quad \sum_i x_{i,j} = \frac{mk}{n}\end{equation}
其中$\sum_j x_{i,j} = k$表示每个Token恰好激活$k$个Expert，而$\sum_i x_{i,j} = mk/n$表示每个Expert恰好被激活$mk/n$次。然而，仔细思考就会发现，其实前者对训练和推理都不是必要的，我们真正需要的是后者，它意味着“<u>平均来说</u>每个Token激活$k$个Expert”以及每个Expert的负载均衡，这足以达成MoE的目标，所以本文考虑更简化的问题
\begin{equation}\max_{x_{i,j}\in\{0,1\}} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_i x_{i,j} = \frac{mk}{n}\label{eq:target-dyn}\end{equation}

## 动态激活

假设读者已经了解上一篇文章，所以本文的推导会相对简略一些，如果对某些步骤有疑惑，可以回到上篇复习一下。跟上篇一样，我们先考虑目标$\eqref{eq:target-dyn}$的松弛版本
\begin{equation}\max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}s_{i,j} \qquad\text{s.t.}\qquad \sum_i x_{i,j} = \frac{mk}{n}\end{equation}
然后考虑它的等价$\max\text{-}\min$形式：
\begin{equation}\max_{x_{i,j}\in[0,1]}\min_{\beta_j} \sum_{i,j} x_{i,j}s_{i,j}  - \sum_j \beta_j\left(\sum_i x_{i,j} - \frac{mk}{n}\right)\end{equation}
类似地，这里$\max$和$\min$的顺序是可交换的，交换并稍加整理得
\begin{equation}\min_{\beta_j}\max_{x_{i,j}\in[0,1]} \sum_{i,j} x_{i,j}(s_{i,j} - \beta_j) + \frac{mk}{n} \sum_j \beta_j\label{eq:relax-min-max}\end{equation}
$\max$这一步可以先完成，结果是
\begin{equation}\left\{\begin{aligned}&\,x_{i,j}^* = 1, &\, s_{i,j} - \beta_j > 0 \\
&\,x_{i,j}^* = 0, &\, s_{i,j} - \beta_j < 0 \\
&\,x_{i,j}^* \in  [0,1], &\, s_{i,j} - \beta_j = 0
\end{aligned}\right.\end{equation}
这告诉我们，只要$s_{i,j} - \beta_j > 0$的Expert就被激活，激活数量是不确定的，这在形式上正好跟[《MoE环游记：4、难处应当多投入》](<https://spaces.ac.cn/archives/10815>)一致，只不过之前我们是凭直觉设计出来的，而这里是通过更本质的目标$\eqref{eq:target-dyn}$推导出来的。

## 一步求解

将$x_{i,j}^*$代回式$\eqref{eq:relax-min-max}$得$x_{i,j}^*(s_{i,j} - \beta_j) = \max(0, s_{i,j} - \beta_j)$，于是$\beta_j$的优化目标简化成
\begin{equation}\min_{\beta_j} \sum_{i,j} \max(0, s_{i,j} - \beta_j) + \frac{mk}{n} \sum_j \beta_j\label{eq:beta-obj}\end{equation}
每一个$\beta_j$的项都是单独的，所以它可以分解为$m$个独立的子优化问题，这意味着我们可以暂时省掉下标$j$：
\begin{equation}\min_{\beta} \frac{mk}{n}\beta + \sum_i \max(0, s_i - \beta)\end{equation}
将全体$s_i$从大到小排列成$s_{\sigma_1}\geq s_{\sigma_2} \geq \cdots \geq s_{\sigma_m}$，然后假设我们已知$s_{\sigma_l}\geq\beta\geq s_{\sigma_{l+1}}$，那么可以去掉$\max$得
\begin{equation}\frac{mk}{n}\beta + \sum_{i=1}^l (s_{\sigma_i} - \beta) = \left\{\begin{aligned}
&\,\sum_{i=1}^{mk/n} s_{\sigma_i} + \sum_{i=mk/n+1}^l \underbrace{(s_{\sigma_i} - \beta)}_{\geq 0},&\, l \geq mk/n \\
&\,\sum_{i=1}^{mk/n} s_{\sigma_i} - \sum_{\hphantom{ab}i=l+1\hphantom{ab}}^{mk/n} \underbrace{(s_{\sigma_i} - \beta)}_{\leq 0},&\, l \leq mk/n \\
\end{aligned}\right.\end{equation}
这说明$l > mk/n$和$l < mk/n$都会放大目标，因此最小值在$l=mk/n$取到，即$\beta^*$位于$s_i$的第$mk/n$大和第$mk/n+1$大元素之间，习惯起见，我们取第$mk/n+1$大元素。恢复下标$j$，那么对于给定$j$，$\beta_j^*$是将$s_{i,j}$从大到小排列后第$mk/n+1$个元素，或者同样叫做“$1-k/n$分位数”。

注意，现在待求变量只有$\boldsymbol{\beta}$，所以就没有交替迭代的步骤了，直接一步Quantile就得到了绝对均衡的最优解！我们依然称之为“<strong>Quantile Balancing (QB)</strong>”。

## 实践要点

如果读者已经了解[《MoE环游记：6、最优分配促均衡》](<https://spaces.ac.cn/archives/11619>)，那么会发现本文实际上是它的精简版。然而，简约并不简单，它实际上比Top\-$k$形式的MoE更为优雅和高效：首先，它通过$\boldsymbol{s} - \boldsymbol{\beta}$是否大于0来激活Expert，这比选Top\-$k$更加高效；其次，它通过一步Quantile就可以得到负载均衡的最优解，这也更为高效和优雅。

当然，一步最优解极有可能会过拟合当前批训练数据，为了避免这个问题，这里考虑的改进做法是让当前批的最优解跟历史解做EMA（Exponential Moving Average，当然上篇的Top\-$k$版其实也可以考虑EMA）。注意我们同样要小心信息泄漏的陷阱，所以要先用旧的$\boldsymbol{\beta}$激活Expert，然后才去更新$\boldsymbol{\beta}$。

算法流程如下（笔者的实验中$\lambda=0.9$）：
$$\begin{array}{|l|}
\hline
\text{Quantile Balancing (QB) 动态激活版} \\[4pt]
\hline
\text{输入: 打分矩阵 }\boldsymbol{s}\in\mathbb{R}^{m\times n}\text{， 上一步 }\boldsymbol{\beta}\in\mathbb{R}^n\text{， 衰减率 }\lambda \\
\text{输出: 分配方案 }\boldsymbol{x}\in\{0,1\}^{m\times n}\text{， 新的 }\boldsymbol{\beta}\in\mathbb{R}^n \\[4pt]
\hline
\begin{array}{ll}
1: & x_{i,j}=1 \text{ if } s_{i,j} - \beta_j > 0 \text{ else } 0 \\
2: & \boldsymbol{\beta} \leftarrow \lambda\boldsymbol{\beta} + (1-\lambda)\mathop{\text{desc_sort}}(\boldsymbol{s}, \text{axis=0})_{[mk/n:mk/n+1]} \\
3: & \text{Output } \boldsymbol{x},\boldsymbol{\beta}
\end{array} \\
\hline
\end{array}$$

实践中，我们同样面临着找$\boldsymbol{s}$的$1-k/n$分位数比较昂贵的问题，它需要在$m$个元素中找第$mk/n+1$大元素，$m$等于“全局样本数 \* 序列长度”，由于各种并行策略和梯度累积的限制，精确实现通常难以接受。所以我们依然考虑折中方案，把样本分为我们能接受的最大的小批次，每个小批次单独算一个$\boldsymbol{\beta}$，将它们平均作为最终的结果。

## 专家选择

实际上，这个动态版QB有一个非常简单的理解方式，那就是“Expert Choice”：

> 你不是想要每个Expert都恰好被激活$mk/n$次么？那么我让Expert取选Token不就行了？也就是说每个Expert选Top\-$mk/n$个Token，这不就是沿着$\text{axis=0}$找最大的$mk/n$个元素了？

然而，朴素的Expert Choice存在一个致命缺陷：它需要在全局范围内比较所有Token，导致跨样本、跨序列的选择，既违反因果律（训练时看到未来信息），又破坏训推一致性（推理结果依赖于Batch Size）。这正是该方法此前难以落地的原因。

本文的巧妙之处在于将其转化为Bias形式：以每个Expert的第$mk/n+1$大元素（等价地，$1-k/n$分位数）作为阈值$\boldsymbol{\beta}$，我们将Expert Choice重新表述为Token Choice——从“Expert选Top\-$mk/n$”变成“只要$s_{i,j} - \beta_j > 0$即激活”。更关键的是，我们将$\boldsymbol{\beta}$的更新延迟到激活决策之后，完美规避了信息泄漏问题。

令人惊讶的是，这个能修复Expert Choice缺陷的Bias形式，竟自然地蕴含在原始线性规划的对偶问题之中——不得不说，有种让人赏心悦目的美感。

## 初始策略

为了防止信息泄漏，训练时我们需要先用旧的$\boldsymbol{\beta}$去决定每个Token要激活的Expert，然后才去更新$\boldsymbol{\beta}$，这就导致训练前几步会对$\boldsymbol{\beta}$的初始化比较敏感。比如$\boldsymbol{s}$全是正的，但$\boldsymbol{\beta}$用了零初始化，那么训练第一步每个Token都会激活所有Expert，计算将会爆炸。

在[《MoE环游记：4、难处应当多投入》](<https://spaces.ac.cn/archives/10815>)中我们已经提供了一个模拟脚本来估算适合的初始化，但这里我们已经知道$\boldsymbol{\beta}$的最优解是“$1-k/n$分位数”，那么我们可以对初始Scores做一些合理的假设，然后推导出初始化的解析式。具体来说，我们假设Router的初始Logits服从$\mathcal{N}(0,\sigma^2)$，那么它的$1-k/n$分位数是
\begin{equation}\sigma \cdot \Phi^{-1}\left(1 - \frac{k}{n}\right)\end{equation}
这正是这个分布的Logits的最优$\boldsymbol{\beta}$，我们以它为$\boldsymbol{\beta}$的初始化，其中$\Phi^{-1}$是标准正态分布的[分位数函数](<https://en.wikipedia.org/wiki/Cumulative_distribution_function#Inverse_distribution_function_%28quantile_function%29>)，又称为逆累积分布函数。

此外，Router可能会加激活函数，假设激活函数是Element\-wise且单调递增的，比如Sigmoid，那么它不改变排序，所以给上式加上相应的激活函数即可。如果是Softmax则稍微复杂一些，它先指数后归一化，假设归一化的分母近似为常数，那么它就跟前者类似，只需要在上式基础上做指数运算，然后除以分位数模拟采样来估计的分母：
\begin{equation}\frac{\exp\left[\sigma \cdot \Phi^{-1}\left(1 - \frac{k}{n}\right)\right]}{\sum\limits_{i=1}^n \exp\left[\sigma \cdot \Phi^{-1}\left(1 - \frac{i}{n+1}\right)\right]}\end{equation}

至于$\sigma$的估计，其实也很简单，假设Router的输入维度是$d$，并且加了RMS Norm，再假设Router的权重矩阵以$\mathcal{N}(0,\tilde{\sigma}^2)$初始化，那么Router Logits近似服从正态分布，均值约为0、方差则约为$\tilde{\sigma}^2 d$，即$\sigma\approx \tilde{\sigma} \sqrt{d}$，这些都是初始化的经典结果了<sup>&#91;[参考](<https://spaces.ac.cn/archives/8620>)&#93;</sup>。

## 演示代码

这一节依然放一段演示代码，大家可以自行玩玩：



<pre><code class="language-python">import numpy as np

def quantile_bias(s, k):
    """动态版QB，一步quantile即可求最优bias
    原理：https://kexue.fm/archives/11626
    """
    m, n = s.shape
    beta = np.quantile(s, 1 - k / n, axis=0)
    return beta

def max_min_avg_vio(s):
    """计算max_vio、min_vio和avg_vio
    其中 max_vio ≥ 0, avg_vio ≥ 0, -1 ≤ min_vio ≤ 0，三者都是越接近于0表示越均衡
    """
    m, n = s.shape
    f = (s &gt; 0).mean(0)
    f = f / f.sum() * n - 1
    return f.max(), f.min(), np.abs(f).mean()

def avg_std_active(s):
    """计算每个expert被激活次数的平均值和标准差
    avg越接近k越好，std越接近0越好
    """
    m, n = s.shape
    f = (s &gt; 0).mean(0) * n
    return f.mean(), f.std()

m, n, k = 100000, 256, 8
s = np.random.rand(m, n) + np.random.rand(n)  # 模拟一个不均匀的打分

b = quantile_bias(s, k)
max_min_avg_vio(s - b)  # 如无意外是全0
avg_std_active(s - b)  # 如无意外是 (k, 0)


from scipy.stats import norm

sigma = 1
s = np.random.randn(m, n) * sigma  # 模拟初始化
avg_std_active(s)  # 大致是(128, 0.4)，远大于预算k
b0 = np.zeros(n) + norm.ppf(1 - k / n) * sigma  # norm.ppf正是标准正态分布的分位数函数
avg_std_active(s - b0)  # 大致是(8, 0.1)，接近预算k

s2 = 1 / (1 + np.exp(-s))  # Sigmoid激活
b0_2 = 1 / (1 + np.exp(-b0))  # Sigmoid激活
avg_std_active(s2 - b0_2)  # 依然大致是(8, 0.1)，接近预算k

s3 = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)  # Softmax激活
q = (np.arange(n) + 1) / (n + 1)  # 均匀间隔点（分位数）
b0_3 = np.exp(b0) / np.exp(sigma * norm.ppf(q)).sum() # Exp激活并除以模拟的分母
avg_std_active(s3 - b0_3)  # 大致是(7.5, 0.1)，接近预算k</code></pre>



## 梯度下降

最后，如果完全不想Quantile，我们依然可以考虑梯度下降法。首先将目标$\eqref{eq:beta-obj}$的损失函数记为$\ell$：
\begin{equation}\min_{\beta_j} \underbrace{\sum_{i,j} \max(0, s_{i,j} - \beta_j) + \frac{mk}{n} \sum_j \beta_j}_{\text{记为}\ell}\end{equation}
它是可导的，梯度为
\begin{equation}\frac{\partial\ell}{\partial\beta_j} = \frac{mk}{n} - \sum_{i=1}^m \chi(s_{i,j} - \beta_j > 0)\end{equation}
其中$\chi$是示性函数，$\chi(\text{True})=1,\chi(\text{False})=0$。这个梯度的计算比全局Quantile更为便宜，适合对成本比较敏感的读者。有了梯度后，就可以进行梯度下降了，我们依旧考虑SignSGD，跟Loss\-Free对齐
\begin{equation}\beta_j \leftarrow \beta_j - \gamma\mathop{\text{sign}}\left(\frac{\partial\ell}{\partial\beta_j}\right)\end{equation}
它可以用来替换Quantile更新$\boldsymbol{\beta}$这一步，并且因为这本来就是一个长期迭代算法，所以EMA也可以去掉了。事实上，这正是我们在[《MoE环游记：4、难处应当多投入》](<https://spaces.ac.cn/archives/10815>)中所提的公式$(9)$，我们从“最优分配 \+ 对偶目标 \+ 梯度下降”的视角重新推出了它。

## 文章小结

这篇文章接着上篇的Quantile Balancing (QB) 进行探索，通过去掉最优分配问题中的“每个Token只能激活$k$个Expert”的约束，显著简化了解的形式——仅需一步Quantile即可实现负载均衡，每个Token只需通过判断偏置分数的正负，就可以决定要激活的Expert，免除了Top\-$k$排序的开销。
