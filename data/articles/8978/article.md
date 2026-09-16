# 训练1000层的Transformer究竟有什么困难？

> 作者：苏剑林 · 科学空间 · 2022-03-09
>
> 原文：<https://spaces.ac.cn/archives/8978>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，现在的Transformer越做越大，但这个“大”通常是“宽”而不是“深”，像GPT\-3虽然参数有上千亿，但也只是一个96层的Transformer模型，与我们能想象的深度相差甚远。是什么限制了Transformer往“深”发展呢？可能有的读者认为是算力，但“宽而浅”的模型所需的算力不会比“窄而深”的模型少多少，所以算力并非主要限制，归根结底还是Transformer固有的训练困难。一般的观点是，深模型的训练困难源于梯度消失或者梯度爆炸，然而实践显示，哪怕通过各种手段改良了梯度，深模型依然不容易训练。

近来的一些工作（如[Admin](<https://papers.cool/arxiv/2004.08249>)）指出，深模型训练的根本困难在于“增量爆炸”，即模型越深对输出的扰动就越大。上周的论文[《DeepNet: Scaling Transformers to 1,000 Layers》](<https://papers.cool/arxiv/2203.00555>)则沿着这个思路进行尺度分析，根据分析结果调整了模型的归一化和初始化方案，最终成功训练出了1000层的Transformer模型。整个分析过程颇有参考价值，我们不妨来学习一下。

## 增量爆炸

原论文的完整分析比较长，而且有些假设或者描述细酌之下是不够合理的。所以在本文的分享中，笔者会尽量修正这些问题，试图以一个更合理的方式来得到类似结果。

假设损失函数为$\mathcal{L}(\boldsymbol{\theta})$，$\boldsymbol{\theta}$是它的参数，考虑参数由$\boldsymbol{\theta}$变为$\boldsymbol{\theta}+\Delta\boldsymbol{\theta}$时损失函数的增量：
\begin{equation}\Delta\mathcal{L} = \mathcal{L}(\boldsymbol{\theta}+\Delta\boldsymbol{\theta}) - \mathcal{L}(\boldsymbol{\theta}) \approx \langle\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}),\Delta\boldsymbol{\theta}\rangle\end{equation}
对于SGD有$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$，那么$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2$。设模型有$N$层，每层有$K$个参数矩阵（$K$接近常数），配合Xavier初始化以及各种Normalization手段，我们可以使得每个参数矩阵的梯度模长是$\mathcal{O}(1)$量级，所以有$\Delta\mathcal{L}=\mathcal{O}(\eta NK)$。因此，模型每一步的更新量是正比于模型深度$N$的，如果模型越深，那么更新量就越大，这意味着初始阶段模型越容易进入不大好的局部最优点，然后训练停滞甚至崩溃，这就是“增量爆炸”问题。

这时候解决方法有两个，一是初始阶段用更小的学习率进行训练（不超过$\eta/N$量级），然后慢慢增大学习率，这就是Warmup技巧；二就是调整初始化方案，使得参数的梯度是$\mathcal{O}(1/\sqrt{N})$量级，这样就自动抵消掉模型深度的影响。

## 量级分析

怎么做到第二种方案呢？我们可以尝试分析Transformer的梯度。然而，精确的梯度求起来比较繁琐，并且事实上我们也不需要精确的梯度，而只是要对梯度做一个量级分析，所以我们可以用如下的“量级分解”技巧转化为标量的导数问题。

对于一个矩阵$\boldsymbol{W}$，我们将其分解为$\boldsymbol{W}=\lambda \boldsymbol{U}$的形式，其中
\begin{equation}\lambda = \mathop{\text{argmin}}_{\kappa > 0} \Vert \boldsymbol{W}\boldsymbol{W}^{\top}/\kappa^2 - \boldsymbol{I}\Vert,\quad \end{equation}
说白了，我们就是要将一个矩阵分解为一个标量$\lambda$与一个尽可能正交的矩阵$\boldsymbol{U}$之积。由于$\boldsymbol{U}$接近正交矩阵，它起到了一个标准参考系的作用，而对应的$\lambda$则代表了矩阵$\boldsymbol{W}$的量级。如果$\boldsymbol{W}$使用Xavier初始化，那么$\lambda$相当于其中的gain参数，即在Xavier初始化的基础上还要再乘一个$\lambda$。这是因为Xavier初始化的结果就接近一个正交矩阵，这一点可以参考[《从几何视角来理解模型参数的初始化策略》](<https://spaces.ac.cn/archives/7180>)。

在此分解之下，我们有
\begin{equation}\frac{\partial \mathcal{L}(\lambda \boldsymbol{U})}{\partial \lambda} = \left\langle\frac{\partial \mathcal{L}(\lambda \boldsymbol{U})}{\partial (\lambda \boldsymbol{U})}, \boldsymbol{U}\right\rangle = \left\langle\frac{\partial \mathcal{L}(\boldsymbol{W})}{\partial \boldsymbol{W}}, \boldsymbol{U}\right\rangle\end{equation}
这意味着$\frac{\partial \mathcal{L}}{\partial \lambda}$跟$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}$在量级上是成正比的，所以对$\frac{\partial \mathcal{L}}{\partial \lambda}$做量级分析就相当于对$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}$做量级分析。这样$\frac{\partial \mathcal{L}}{\partial \lambda}$就相当于$\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}}$量级的一个简单的“探针”，原来的矩阵求导就可以转化为标量求导，降低了分析难度。

## 前馈梯度

很多实验结果都显示虽然Pre Norm比Post Norm更容易训练，但Post Norm的最终效果往往更好些，所以原论文保留了Post Norm结构，并考虑了更一般的形式（DeepNorm）：
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\alpha\boldsymbol{x}_l + F(\boldsymbol{x}_l)) = \text{LN}(\boldsymbol{x}_l + F(\boldsymbol{x}_l)/\alpha)\end{equation}
其中$\alpha > 0$是一个常数。简单起见，我们先考虑FFN层，此时
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \phi(\boldsymbol{x}_l \boldsymbol{W}_1)\boldsymbol{W}_2/\alpha)\end{equation}
这里的$\phi$是激活函数，一般为ReLU或其变体（Swish、GeLU等），它们（近似）满足$\phi(\lambda x) = \lambda \phi(x),\forall \lambda > 0$。使用前一节的量级分解探针，我们得到
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\underbrace{\boldsymbol{x}_l + \lambda_1 \lambda_2 \phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2/\alpha}_{\text{记为}\boldsymbol{z}_{l+1}})\label{eq:ffn}\end{equation}
求$\lambda$的梯度：
\begin{equation}\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \lambda_1} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_1} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_2 \phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2}{\alpha} \\
\frac{\partial \mathcal{L}}{\partial \lambda_2} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_2} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_1 \phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2}{\alpha} \end{aligned}\end{equation}
我们断言$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$都是$\mathcal{O}(1)$的，并且由于$\boldsymbol{U}_1$、$\boldsymbol{U}_2$都接近正交矩阵，所以$\phi(\boldsymbol{x}_l \boldsymbol{U}_1)\boldsymbol{U}_2$也是$\mathcal{O}(1)$的，因此最终有
\begin{equation}\frac{\partial \mathcal{L}}{\partial \lambda_1} = \mathcal{O}\left(\frac{\lambda_2}{\alpha}\right),\quad \frac{\partial \mathcal{L}}{\partial \lambda_2} = \mathcal{O}\left(\frac{\lambda_1}{\alpha}\right)\end{equation}

## 自注意力

现在考虑自Self Attention，作为量级分析，我们考虑单头注意力即可，其形式为
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\boldsymbol{x}_l + \sigma(\boldsymbol{x}_l \boldsymbol{W}_q\boldsymbol{W}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{W}_v\boldsymbol{W}_o/\alpha)\end{equation}
其中$\sigma(\cdot)$是softmax操作的简写，这里省略了Attention的scale操作。对上式进行量级分解后的形式为
\begin{equation}\boldsymbol{x}_{l+1} = \text{LN}(\underbrace{\boldsymbol{x}_l + \lambda_v\lambda_o \sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o/\alpha}_{\text{记为}\boldsymbol{z}_{l+1}})\label{eq:sa}\end{equation}
现在我们可以对各个$\lambda$分别求梯度，而由于softmax的存在，事实上$\lambda_q,\lambda_k$的梯度本身会很小，不会明显影响最终的更新量，所以其实我们考虑$\lambda_v,\lambda_o$的更新量足矣：
\begin{equation}\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \lambda_v} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_v} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_o \sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o}{\alpha} \\
\frac{\partial \mathcal{L}}{\partial \lambda_o} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\partial \boldsymbol{z}_{l+1}}{\partial \lambda_o} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}\frac{\lambda_v \sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o}{\alpha} \end{aligned}\end{equation}
同样断言$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$都是$\mathcal{O}(1)$的，并且注意softmax出来是一个概率分布，然后对$\boldsymbol{x}_l$的各个token做加权平均，通常而言，平均前后的向量会在同一数量级，所以我们认为$\sigma (\lambda_q\lambda_k\boldsymbol{x}_l \boldsymbol{U}_q\boldsymbol{U}_k^{\top}\boldsymbol{x}_l^{\top})\boldsymbol{x}_l\boldsymbol{U}_v\boldsymbol{U}_o$也是$\mathcal{O}(1)$的，因此结果跟FFN层的类似：
\begin{equation}\frac{\partial \mathcal{L}}{\partial \lambda_v} = \mathcal{O}\left(\frac{\lambda_o}{\alpha}\right),\quad \frac{\partial \mathcal{L}}{\partial \lambda_o} = \mathcal{O}\left(\frac{\lambda_v}{\alpha}\right)\end{equation}

## 初步结论

现在不管是FFN还是Self Attention，我们都得到了相似的结论，现在简单起见，假设每个参数的量级（至少在初始化阶段）是一致的，即所有的$\lambda$取同一个值，那么总的结论是
\begin{equation}\frac{\partial \mathcal{L}}{\partial \lambda} = \mathcal{O}\left(\frac{\lambda}{\alpha}\right)\end{equation}
即梯度的量级是$\mathcal{O}(\lambda/\alpha)$。另一方面，我们说$N$层的Transformer模型，一般是$N$层的Self Attention加$N$层的FFN，所以严格来说层数是$2N$。因此，按照“增量爆炸”一节的分析，我们需要将梯度调整到$\mathcal{O}(1/\sqrt{2N})$，上式告诉我们可以通过让$\lambda/\alpha=1/\sqrt{2N}$来实现。原论文的放缩更为宽松一些，得到的结果是$\lambda/\alpha = 1/\sqrt{4N}$，量级上是等价的。

现在我们得到了$\lambda$与$\alpha$的一个比例关系，但无法直接得到$\lambda$和$\alpha$的具体值。按照论文的说法，是从对称角度出发，让$\lambda=1/\alpha$，从而可以解得
\begin{equation}\alpha = (2N)^{1/4},\quad \lambda = (2N)^{-1/4}\label{eq:result}\end{equation}
然而，单纯对称的解释显然是不够说服力的，我们需要搞清楚不同的选择究竟有什么不同的结果。为此，我们可以比较另外两组解：

> <strong>另解一：</strong>$\alpha=1,\lambda=(2N)^{-1/2}$，此时参数的初始化缩小到原来的$(2N)^{-1/2}$倍，梯度也被缩小到原来的$(2N)^{-1/2}$倍，根据SGD的$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$得出每步的更新量也是原来的$(2N)^{-1/2}$倍，也就是说，调整前后的相对学习幅度是没有变化的，因此有可能刚开始$\lambda=\mathcal{O}((2N)^{-1/2})$级别，但训练集几步后就脱离了这个量级了。
> 
> 
> 
> <strong>另解二：</strong>$\alpha=(2N)^{1/2},\lambda=1$，此时参数的初始化没有缩小，但梯度也被缩小到原来的$(2N)^{-1/2}$倍，根据SGD的$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$得出每步的更新量也是原来的$(2N)^{-1/2}$倍，调整前后的相对学习幅度是明显缩小了，因此有可能出现学习得非常慢的情况。

这两种情况看上去都各有缺点，因此介乎两者之间的式$\eqref{eq:result}$似乎就能解释得通了，它就是保持梯度缩放到原来的$(2N)^{-1/2}$倍的同时，让初始学习步伐稍微慢一些，但又不至于太慢，隐式地起到了Warmup的作用。

## 多种优化

上面的分析都是基于SGD进行的，但事实上我们很少直接用SGD去训练NLP模型，我们更多是自适应学习率优化器，主要有两大类：一是用二阶矩来校正学习率，Adam、AdamW等都属此类；另一类是通过参数模长进一步校正学习率，比如[LAMB](<https://spaces.ac.cn/archives/7094>)、[AdaFactor](<https://spaces.ac.cn/archives/7302>)。原论文的说法是“我们在SGD上进行推导，然后在Adam上验证发现也还可以”，但从理论上来讲，它们并不完全通用，这一节我们就来针对性地做一下分析。

对于Adam类优化器来说，每一步的更新量大约为$\Delta\boldsymbol{\theta}=-\eta\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}))$，所以$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1$，它是正比于梯度的1次方而不是2次方，因此要过要让更新量跟层数无关，那么梯度应该缩小到原来的$1/(2N)$倍才对，即应该有$\lambda/\alpha=1/(2N)$，如果同样让$\lambda=1/\alpha$，那么有
\begin{equation}\alpha = (2N)^{1/2},\quad \lambda = (2N)^{-1/2}\end{equation}

对于LAMB类优化器来说，每一步更新量大约为$\Delta\boldsymbol{\theta}=-\eta\Vert\theta\Vert\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}))$，所以$\Delta\mathcal{L} \approx -\eta\Vert\theta\Vert\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1$，注意到参数的缩放比例是$\lambda$、梯度的缩放比例是$\lambda/\alpha$，所以$\Delta\mathcal{L}=\mathcal{O}(2N\lambda^2/\alpha)$，从而是$\lambda^2/\alpha=1/(2N)$。注意这类优化器每步的相对更新量是一样的（等于学习率$\eta$），不管怎么调整$\alpha,\lambda$其相对更新大小都不会变化，所以我们可以直接取$\alpha=1,\lambda=(2N)^{-1/2}$。

结果汇总对比如下：
\begin{array}{c|cc|cc}
\hline
\text{优化器} & \Delta\boldsymbol{\theta} & \Delta\mathcal{L} & \alpha & \lambda \\
\hline
\text{SGD} & -\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}) & -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2 & (2N)^{1/4} & (2N)^{-1/4}\\
\text{Adam} & -\eta\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})) & -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1 & (2N)^{1/2}& (2N)^{-1/2}\\
\text{LAMB} & -\eta\Vert\theta\Vert\,\text{sign}(\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})) & -\eta\Vert\theta\Vert\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert_1 & 1 & (2N)^{-1/2}\\
\hline
\end{array}

## 事后分析

前面的两节推导过程都用到了断言“$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$都是$\mathcal{O}(1)$的”，那么它是否成立呢？这里我们事后分析一下。

其实也很简单，经过前述调整后，不管是FFN层$\eqref{eq:ffn}$还是Self Attention层$\eqref{eq:sa}$，初始阶段每个残差分支的权重被缩放到原来的$\lambda^2/\alpha$倍，不管是哪种优化器的结果，$\lambda^2/\alpha$都是一个比较小的数字，这意味着初始阶段整个模型其实接近一个恒等函数，因此$\frac{\partial \mathcal{L}}{\partial \boldsymbol{x}_{l+1}}$、$\frac{\partial \boldsymbol{x}_{l+1}}{\partial \boldsymbol{z}_{l+1}}$自然都是$\mathcal{O}(1)$的，所以结论和断言是自洽的。

另外，可能有读者想问同样的分析是否可以用到Pre Norm结构上呢？答案是可以的，并且结论是基本一致的，只是因为Norm放在了残差分支之前，所以就没必要设置$\alpha$参数了，所以结论就是上述关于Post Norm的结果中所有的$\alpha$都等于为1，然后重新计算相应的$\lambda$。

最后，读者可能有疑问的是花了那么多功夫讨论把模型做深，那么模型深度真有那么重要吗？有，原论文给出了一个漂亮的实验结果，用一个200层的“深而窄”的模型（32亿参数），战胜了之前48层“浅而宽”的SOTA模型（120亿参数）：

[![“深而窄”的模型胜于“浅而宽”的模型](<https://spaces.ac.cn/usr/uploads/2022/03/2952207079.png>)](<https://spaces.ac.cn/usr/uploads/2022/03/2952207079.png>)

“深而窄”的模型胜于“浅而宽”的模型

## 文章小结

本文分析了将Transformer做“深”的瓶颈所在并给出了相应的解决方案，文章的主要思路源于微软新出的DeepNet，并对原论文的分析过程做了一定的简化和完善。
