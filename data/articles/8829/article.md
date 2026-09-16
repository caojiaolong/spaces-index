# 概率分布的熵归一化（Entropy Normalization）

> 作者：苏剑林 · 科学空间 · 2021-12-24
>
> 原文：<https://spaces.ac.cn/archives/8829>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章[《从熵不变性看Attention的Scale操作》](<https://spaces.ac.cn/archives/8823>)中，我们从熵不变性的角度推导了一个新的Attention Scale，并且实验显示具有熵不变性的新Scale确实能使得Attention的外推性能更好。这时候笔者就有一个很自然的疑问：

> 有没有类似L2 Normalization之类的操作，可以直接对概率分布进行变换，使得保持原始分布主要特性的同时，让它的熵为指定值？

笔者带着疑问搜索了一番，发现没有类似的研究，于是自己尝试推导了一下，算是得到了一个基本满意的结果，暂称为“熵归一化（Entropy Normalization）”，记录在此，供有需要的读者参考。

## 幂次变换

首先，假设$n$元分布$(p_1,p_2,\cdots,p_n)$，它的熵定义为
\begin{equation}\mathcal{H} = -\sum_i p_i \log p_i = \mathbb{E}[-\log p_i]\end{equation}
由于$p_i \in [0,1]$，所以$-p_i \log p_i \geq 0$，因此$\mathcal{H} \geq 0$，当某个$p_i$为1、其余$p_i$为0时（one hot），取得最小值0；此外，也可以证明当所有$p_i$等于$1/n$时，$\mathcal{H}$取得最大值$\log n$，所以$\mathcal{H}$的取值范围是$[0,\log n]$。

所以，我们首先要找一种分布的变换，它能够保持分布的主要信息，并且有能力将分布的熵从$0$到$\log n$进行变换。这里选择的是幂次变换
\begin{equation}p_i\quad\to\quad \tilde{p}_i = \frac{p_i^{\gamma}}{\sum\limits_i p_i^{\gamma}}\end{equation}
选择幂次变换的原因之一，是它保持了分布的单调性，即如果$p_i > p_j$，那么也有$\tilde{p}_i > \tilde{p}_j$，个人认为这是分布需要保持的重要性质之一。此外，当各个$p_i$都非零并且两两不相等时，幂次变化确实有能力将熵从$0\sim \log n$进行变化。不失一般性，我们假设$1 > p_1 > p_2 > \cdots > p_n > 0$，显然当$\gamma = 0$时，$\tilde{p_i}=1/n$，此时熵为最大值$\log n$，当$\gamma \to\infty$时，有
\begin{equation}\tilde{p}_1 = \lim_{\gamma\to\infty}\frac{p_1^{\gamma}}{\sum\limits_i p_i^{\gamma}} = \lim_{\gamma\to\infty}\frac{1}{1 + \sum\limits_{i > 1} (p_i/p_1)^{\gamma}}=1\end{equation}
也就是此时为one hot分布$(1,0,\cdots,0)$，对应的熵为最小值0。其实还可以进一步求导证明熵关于$\gamma$是单调递减的，因此当$\gamma$从$0$到$\infty$递增时，熵从$\log n$到$0$递减变化。

## 迭代求解

确定幂次变换确实是一种可用的变换后，我们就要设法求解了，即对于任意给定的$\mathcal{H}^*\in(0,\log n)$，我们需要找到正确的$\gamma$，使得对应的熵为指定值$\mathcal{H}^*$。

首先我们写出
\begin{equation}\mathcal{H}_{\gamma} = -\sum_i\frac{p_i^{\gamma}}{\sum\limits_i p_i^{\gamma}}\log \frac{p_i^{\gamma}}{\sum\limits_i p_i^{\gamma}}=\log\sum_i p_i^{\gamma} - \frac{\gamma\sum\limits_i p_i^{\gamma}\log p_i}{\sum\limits_i p_i^{\gamma}}\end{equation}
最右端结果的复杂性让我们相信应该不存在解析解，所以只能寻求迭代求解算法了。

我们求它在$\gamma=1$处的展开（主要利用$p_i^{\gamma}\approx p_i + (\gamma-1)p_i\log p_i$）：
\begin{equation}\begin{aligned}
\mathcal{H}_{\gamma} \approx &\, -\sum_i p_i\log p_i + \left(\left(\sum_i p_i\log p_i\right)^2-\sum_i p_i\left(\log p_i\right)^2\right)(\gamma - 1)\\
=&\, \mathcal{H}_1 + \left(\mathcal{H}_1^2-\mathbb{E}[\left(\log p_i\right)^2]\right)(\gamma - 1)
\end{aligned}\end{equation}
那么
\begin{equation}\gamma \approx 1 + \frac{\mathcal{H}_{\gamma}-\mathcal{H}_1}{\mathcal{H}_1^2-\mathbb{E}[\left(\log p_i\right)^2]}\end{equation}
根据该结果，我们从$\gamma=1$出发，反复利用上式进行迭代，就可以求出最终的分布：
\begin{equation}
\mathcal{H}\leftarrow -\sum_i p_i \log p_i,\quad
\gamma \leftarrow 1 + \frac{\mathcal{H}^*-\mathcal{H}}{\mathcal{H}^2-\mathbb{E}[\left(\log p_i\right)^2]},\quad p_i \leftarrow \frac{p_i^{\gamma}}{\sum\limits_i p_i^{\gamma}}
\end{equation}
这其实就是求解非线性方程的牛顿法了。在实验时发现，迭代3～4次，就可以取得不错的收敛效果，如果实际使用时只是为了大致地控制一下熵的范围，那么迭代1～2次即可。

Numpy的参考代码：



<pre><code class="language-python">p = np.random.random(100)
p /= p.sum()  # 模拟分布
gamma = 1
H_f = np.log(30)  # 希望达到的熵

for i in range(10):
    H = -(p * np.log(p)).sum()
    gamma = 1 + (H_f - H) / (H**2 - (p * np.log(p)**2).sum())
    p = p**gamma
    p /= p.sum()</code></pre>



## 应用设想

本文主要是觉得“熵归一化”这个概念比较有意思，所以尝试进行了推导。但具体有什么比较好的应用例子，笔者也还没想清楚。

熵越小，意味着概率越集中在几个位置上，换句话说就是其他位置的概率越接近于零，因此某种程度上来说，熵是概率分布的稀疏程度的一种度量，如果我们希望得到比较稀疏的预测结果，那么就可以通过熵归一化进行控制。另一方面，分布越稀疏，也意味着模型越有可能梯度消失，因此反过来也可以通过熵归一化来控制熵不要那么小，从而缓解梯度消失问题。

说到稀疏性，就不免想起[Sparsemax](<https://papers.cool/arxiv/1602.02068>)以及笔者自己构思的[Sparse Softmax](<https://spaces.ac.cn/archives/8046#%E7%A8%80%E7%96%8FSoftmax>)等工作，其中Sparsemax是将熵视为惩罚项来得到的稀疏性，而Sparse Softmax则是通过直接截断而引入的稀疏性，两者皆在某些场景下有更好的解释性或者更好的效果，那么直接通过熵归一化带来的稀疏性有没有效果呢？这可能也是一个值得探究的问题。

另外，在自回归模型的随机采样中，我们经常用top\-$k$、top\-$p$截断，这种截断本质上也是在降低分布的熵，所以相应地，我们也可以通过熵归一化来使得每步采样的分布熵一致，用以取代top\-$k$、top\-$p$采样，这也是一种可能的应用。

使用熵归一化的主要问题是“究竟归一化到哪个值”没有明确的标准，笔者目前也没有比较好的思路，暂时只能想到通过观察已有的实验结果来调参，但终归不是一个理想的答案。

## 文末小结

本文引入了熵归一化（Entropy Normalization）的概念，通过直接的变换使得分布的熵可以为指定值，并构思了一些潜在应用。
