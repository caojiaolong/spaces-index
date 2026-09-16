# Transformer升级之路：12、无限外推的ReRoPE？

> 作者：苏剑林 · 科学空间 · 2023-08-07
>
> 原文：<https://spaces.ac.cn/archives/9708>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

自从在[《Transformer升级之路：11、将β进制位置进行到底》](<https://spaces.ac.cn/archives/9706>)中引入混合进制的思路进一步推广了NTK\-aware Scaled RoPE后，笔者感觉类似思路的效果已经达到了上限，想要更大幅度的提升就必须另辟蹊径了。这时候笔者想起了此前构思过的一个思路，该思路由于复杂度较高所以被搁置下了，既然现在已经遇到了瓶颈，那么“唯一的办法就是最好的办法”，于是便将它重拾起来。

万万没想到的是，尽管该方法增加了一些推理复杂度，但它的实验效果却惊人地好——甚至隐约有无限的长度外推能力！因此，笔者迫不及待地撰写了本文来分享该方法。由于形式上跟ReLU激活函数的相似性，所以笔者将该方法命名为“ReRoPE (Rectified Rotary Position Embeddings)”。

## 重温

我们知道，[RoPE](<https://spaces.ac.cn/archives/8265>)形式上是一种绝对位置编码，但实际上给Attention带来的是相对位置信息，即如下的[Toeplitz矩阵](<https://en.wikipedia.org/wiki/Toeplitz_matrix>)：
\begin{equation}\begin{pmatrix}0 & \\
1 & 0 & \\
2 & 1 & 0 &\\
3 & 2 & 1 & 0 & \\
\ddots & 3 & 2 & 1 & 0 & \\
\ddots & \ddots & 3 & 2 & 1 & 0 & \\
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\
\small{L - 2} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\
\small{L - 1} & \small{L - 2} & \ddots & \ddots & \ddots & 3 & 2 & 1 & 0 & \\
\end{pmatrix}\label{eq:rope}\end{equation}
这里的$L$是当前样本长度。当$L$明显超出了训练长度时，多出来的位置由于没有被训练过，所以无法保证效果，这就是直接外推（Length Extrapolation）表现通常比较差的原因。

后来，研究人员提出了位置内插（Position Interpolation），它相当于将相对位置矩阵改为：
\begin{equation}\begin{pmatrix}0 & \\
\frac{1}{k} & 0 & \\
\frac{2}{k} & \frac{1}{k} & 0 &\\
\frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\
\ddots & \frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\
\ddots & \ddots & \frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\
\ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\
\small{\frac{L-2}{k}} & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots & \ddots \\
\small{\frac{L-1}{k}} & \small{\frac{L-1}{k}} & \ddots & \ddots & \ddots & \frac{3}{k} & \frac{2}{k} & \frac{1}{k} & 0 & \\
\end{pmatrix}\end{equation}
这样一来，只要调整$k$，就可以保证最大的相对位置也不超过训练长度，因此避免了外推。然而，它使得位置信息更加“拥挤”了，所以还需要进行一定步数的微调才能让模型重新工作。而也正因为避免了外推，所以它所需要的微调步数相比直接外推要少得多（神经网络往往更擅长内插而不是外推）。

至于后面提出的NTK\-aware Scaled RoPE，则是“剑走偏锋”，巧妙地将外推压力平摊到每一个维度上，所以它不微调也能有不错的效果，但它终究还是依赖外推，这是神经网络不擅长的事情，所以效果存在上限，在笔者的实验中，它的Long Context表现还无法很接近训练效果。

## 融合

我们也可以从语言模型的局域性来考察这些方法。所谓局域性，是指语言模型在推断下一个token时，明显更依赖于邻近的token。直接外推保持了局域性（0附近位置编码不变），效果差是因为引入了超出训练长度的位置编码；位置内插虽然没有外推位置编码，但扰乱了局域性（0附近位置编码被压缩为$1/k$），所以不微调效果也不好；而NTK\-aware Scaled RoPE通过“高频外推、低频内插”隐含了两者优点，保证了局域性，又没有明显外推位置编码，所以不微调也有不错的效果。

有没有能更直接地结合外推和内插的方法呢？有，我们可以设定一个窗口大小$w$，在窗口内我们使用大小为$1$的位置间隔，在窗口外我们使用大小为$1/k$的位置间隔，整个相对位置矩阵如下：
\begin{equation}\begin{pmatrix}
\color{red}{0} & \\
\color{red}{1} & \color{red}{0} & \\
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\small{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\small{w + \frac{2}{k}}} & \color{green}{\small{w + \frac{1}{k}}} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\ddots} & \color{green}{\small{w + \frac{2}{k}}} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\small{w + \frac{2}{k}}} & \color{green}{\small{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\small{w + \frac{L-1-w}{k}}} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\small{w + \frac{2}{k}}} & \color{green}{\small{w + \frac{1}{k}}} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\end{pmatrix}\label{eq:leaky-rerope}\end{equation}
只要$w$小于训练长度，那么通过控制$k$，我们就可以在精确保持了局域性的前提下，使得所有位置编码不超过训练长度，简单直接地结合了直接外推和位置内插。

特别地，矩阵$\eqref{eq:leaky-rerope}$还有一个特别的case：当$k\to\infty$时，它简化为
\begin{equation}\begin{pmatrix}
\color{red}{0} & \\
\color{red}{1} & \color{red}{0} & \\
\color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{w} & \color{green}{w} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{w} & \color{green}{w} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\ddots} & \color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \color{red}{\ddots} & \\
\color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\color{green}{w} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{\ddots} & \color{green}{w} & \color{green}{w} & \color{green}{w} & \color{red}{\small{w - 1}} & \color{red}{\ddots} & \color{red}{2} & \color{red}{1} & \color{red}{0} & \\
\end{pmatrix}\label{eq:rerope}\end{equation}
在这个case下，不管输入长度是多少，它的位置编码范围都不超过$w$，所以这是一种有可能支持任意长度的Context的方案！

形式上，矩阵$\eqref{eq:rerope}$、$\eqref{eq:leaky-rerope}$与标准RoPE矩阵$\eqref{eq:rope}$的关系，就相当于ReLU、Leaky ReLU与Linear的关系，所以笔者将$\eqref{eq:rerope}$称为“ReRoPE（Rectified RoPE）”，将$\eqref{eq:leaky-rerope}$称为“Leaky ReRoPE”。

## 计算

其实，类似的思路并不难想到，以往基于Attention Bias的相对位置编码（比如[经典相对位置编码](<https://spaces.ac.cn/archives/8130#%E7%BB%8F%E5%85%B8%E5%BC%8F>)、[T5位置编码](<https://spaces.ac.cn/archives/8130#T5%E5%BC%8F>)）经常会出现这样的分块运算。然而跟这些相对位置编码不同，在RoPE中实现这样的分块运算会明显增加计算量，这也是该思路会被笔者搁置的主要原因。

怎么理解增加计算量呢？我们知道RoPE是“通过绝对位置实现相对位置”，这样只能得到线性的相对位置，而矩阵$\eqref{eq:leaky-rerope}$、$\eqref{eq:rerope}$是非线性的（或者说分段线性的），要实现它只能算两次Attention矩阵，然后组合起来。具体来说，首先用标准的RoPE计算一次Attention矩阵（Softmax之前）
\begin{equation}a_{i,j}^{(1)} = \left(\boldsymbol{\mathcal{R}}^i\boldsymbol{q}_i\right)^{\top}\left(\boldsymbol{\mathcal{R}}^j\boldsymbol{k}_j\right) = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{j-i}\boldsymbol{k}_j\end{equation}
这里第一个等号是实现方式，第二个等号是等效结果，其中$\boldsymbol{\mathcal{R}}$就是RoPE的旋转矩阵，简单起见我们省略了Attention的scale因子。接着，我们需要计算间隔为$1/k$的RoPE的Attention矩阵（Leaky ReRoPE）：
\begin{equation}a_{i,j}^{(2)} = \left(\boldsymbol{\mathcal{R}}^{(i-w)/k+w}\boldsymbol{q}_i\right)^{\top}\left(\boldsymbol{\mathcal{R}}^{j/k}\boldsymbol{k}_j\right) = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^{(j-i+w)/k-w}\boldsymbol{k}_j\end{equation}
如果是ReRoPE，那么简单一些：
\begin{equation}a_{i,j}^{(2)} = \left(\boldsymbol{\mathcal{R}}^w\boldsymbol{q}_i\right)^{\top}\boldsymbol{k}_j = \boldsymbol{q}_i^{\top}\boldsymbol{\mathcal{R}}^w\boldsymbol{k}_j\end{equation}
最后，根据$i - j < w$这个条件，将它们合并起来：
\begin{equation}a_{i,j} = \left\{\begin{aligned}
&a_{i,j}^{(1)},\quad (i - j < w) \\[8pt] &a_{i,j}^{(2)}, \quad (i - j \geq w)
\end{aligned}\right.\end{equation}
不管是ReRoPE还是Leaky ReRoPE，都不可避免地计算两次Attention矩阵（如果有更高效的实现方法，请赐教），这便是增加的计算量之一。此外，需要自定义计算Attention矩阵也导致了不能直接套用现成的flash attention实现，因此相对之下又增加了计算成本。

另一方面，同样是由于非线性的相对位置，所以在自回归解码时，Key序列的cache只能存RoPE之前的，然后在每步解码时给整个Key序列补上对应的RoPE，这样的改动也会增加推理计算量。唯一的好消息是，在token by token解码时，从第二步开始Query序列的长度就为1，此时只需要为Key序列定制RoPE，那么可以只算一次Attention矩阵：
\begin{equation}a_{i,j} = \left\{\begin{aligned}
&\boldsymbol{q}_i^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-i,-w)}\boldsymbol{k}_j\right), \quad(\text{ReRoPE})\\[8pt]
&\boldsymbol{q}_i^{\top}\left(\boldsymbol{\mathcal{R}}^{\max(j-i,(j-i+w)/k-w)}\boldsymbol{k}_j\right), \quad(\text{Leaky ReRoPE})
\end{aligned}\right.\end{equation}

## 实验

继续沿着[《Transformer升级之路：11、将β进制位置进行到底》](<https://spaces.ac.cn/archives/9706>)的设置，我们对ReRoPE进行了实验，效果如下表：
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\
\hline
\text{Baseline} & 49.41\% & 24.17\% & 23.16\% \\
\text{Baseline-}\log n & 49.40\% & 24.60\% & 24.02\% \\
\hline
\text{PI-RoPE} & 49.41\% & 15.04\% & 13.54\% \\
\text{PI-RoPE-}\log n & 49.40\% & 14.99\% & 16.51\% \\
\hline
\text{NTK-RoPE-old} & 49.41\% & 51.28\% & 39.27\% \\
\text{NTK-RoPE-}\log n\text{-old} & 49.40\% & 61.71\% & 43.75\% \\
\hline
\text{NTK-RoPE-fixed} & 49.41\% & 51.86\% & 39.61\% \\
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-fixed} & 49.41\% & 55.94\% & 41.11\% \\
\text{NTK-RoPE-}\log n\text{-fixed} & 49.40\% & 62.85\% & 44.14\% \\
\text{NTK-RoPE-mixed} & 49.41\% & 53.09\% & 40.12\% \\
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-mixed} & 49.41\% & 59.11\% & 42.38\% \\
\text{NTK-RoPE-}\log n\text{-mixed} & 49.40\% & 68.91\% & 45.41\% \\
\hline
\text{ReRoPE-w256} & 49.41\% & 77.90\% & 48.48\% \\
\text{ReRoPE-w256-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.40\% & 48.85\% \\
\text{ReRoPE-w256-}\log n & 49.40\% & \boldsymbol{85.12\%} & \boldsymbol{49.07\%} \\
\hline
\text{HFWA} & 48.70\% & 80.84\% & 48.15\% \\
\hline
\end{array}
正如文章开头所说，ReRoPE不微调外推的效果可谓出奇地好，不仅明显超越了此前最优的NTK\-RoPE\-mixed，还明显超过了从零预训练的[HFWA](<https://spaces.ac.cn/archives/9603>)！这里的$\text{w256}$指的$w=256$，$\log n^{\color{red}{\dagger}}$是指预训练没有加入$\log n$缩放（比如LLAMA），测试阶段每个$\boldsymbol{q}_n$都乘上$\max(1, \log_{\text{maxlen}} n)$，$\log n$则是指预训练就加入了$\log n$缩放因子。

以下是一些消融实验，显示出ReRoPE关于$w$还是很鲁棒的，最优值大致是训练长度的$1/4\sim 1/2$左右：
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\
\hline
\text{ReRoPE-w64} & 49.41\% & 69.39\% & 45.19\% \\
\text{ReRoPE-w64-}\log n^{\color{red}{\dagger}} & 49.41\% & 78.58\% & 47.42\% \\
\text{ReRoPE-w64-}\log n & 49.40\% & 84.38\% & 48.14\% \\
\hline
\text{ReRoPE-w128} & 49.41\% & 76.11\% & 47.82\% \\
\text{ReRoPE-w128-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.28\% & 48.78\% \\
\text{ReRoPE-w128-}\log n & 49.40\% & \boldsymbol{85.47\%} & 48.87\% \\
\hline
\text{ReRoPE-w256} & 49.41\% & 77.90\% & 48.48\% \\
\text{ReRoPE-w256-}\log n^{\color{red}{\dagger}} & 49.41\% & 82.40\% & 48.85\% \\
\text{ReRoPE-w256-}\log n & 49.40\% & 85.12\% & \boldsymbol{49.07\%} \\
\hline
\text{ReRoPE-w384} & 49.41\% & 70.72\% & 48.15\% \\
\text{ReRoPE-w384-}\log n^{\color{red}{\dagger}} & 49.41\% & 76.42\% & 48.31\% \\
\text{ReRoPE-w384-}\log n & 49.40\% & 83.24\% & 48.62\% \\
\hline
\text{ReRoPE-w512} & 49.41\% & 7.09\% & 8.25\% \\
\text{ReRoPE-w512-}\log n^{\color{red}{\dagger}} & 49.41\% & 7.08\% & 8.25\% \\
\text{ReRoPE-w512-}\log n & 49.40\% & 15.84\% & 10.83\% \\
\hline
\end{array}

下表则对比了ReRoPE和Leaky ReRoPE：
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\
\hline
\text{ReRoPE-w128-}\log n & 49.40\% & \boldsymbol{85.47\%} & 48.87\% \\
\text{Leaky ReRoPE-w128-k64-}\log n & 49.40\% & 85.29\% & 48.96\% \\
\text{Leaky ReRoPE-w128-k32-}\log n & 49.40\% & 85.31\% & 49.03\% \\
\text{Leaky ReRoPE-w128-k16-}\log n & 49.40\% & 85.15\% & \boldsymbol{49.10\%} \\
\text{Leaky ReRoPE-w128-k8-}\log n & 49.40\% & 80.00\% & 48.11\% \\
\hline
\text{ReRoPE-w256-}\log n & 49.40\% & 85.12\% & 49.07\% \\
\text{Leaky ReRoPE-w256-k64-}\log n & 49.40\% & 84.60\% & 49.03\% \\
\text{Leaky ReRoPE-w256-k32-}\log n & 49.40\% & 84.30\% & 48.97\% \\
\text{Leaky ReRoPE-w256-k16-}\log n & 49.40\% & 83.59\% & 48.87\% \\
\text{Leaky ReRoPE-w256-k8-}\log n & 49.40\% & 69.80\% & 45.72\% \\
\hline
\end{array}

作为ReRoPE的一般化，经过精调的Leaky ReRoPE是有机会超过ReRoPE的，但提升很微弱。此外，当$k$取有限值时，能处理的最大长度也是有限的，因为我们不能提前知道要生成的总长度，所以只能预设一个足够大的$k$，但设定为有限值之后，当输入足够长时，就会因为位置编码超出训练长度而效果大幅下降，相比之下ReRoPE则不会有这个风险。总的来说，精调Leaky ReRoPE相比ReRoPE的价值似乎不大。

以上实验结果都只是在1亿参数的GAU模型上测试的，下面给出基于llama2\-13b的测试结果（指标是loss，越小越好），它代表了在真正的LLM表现：
\begin{array}{c|cc}
\hline
\text{测试长度} & 4096(\text{训练}) & 8192 & 16384\\
\hline
\text{RoPE} & 1.4967 & 8.8615 & \text{-} \\
\text{NTK-RoPE} & 1.6081 & 1.5417 & 1.5163 \\
\text{ReRoPE} & 1.4996 & 1.4267 & 1.4001 \\
\hline
\end{array}
可以看到，ReRoPE真正做到了几乎不损训练效果（RoPE\-4096代表训练效果），并且满足“longer context, lower loss”的理想特点（更多的context应该更加有助于预测）。此外，笔者也在OpenBuddy开源的LLAMA2\-13b微调模型上测试了chat的效果，自我感觉还不错（最多测试过20k tokens的Context）。

最后，分享笔者在transformers的LLAMA模型基础上实现ReRoPE和Leaky ReRoPE的代码，读者也可以自行加载LLAMA系列模型进行测试：

> <strong>Github：[https://github\.com/bojone/rerope](<https://github.com/bojone/rerope>)</strong>

## 小结

在这篇文章中，笔者提出了ReRoPE (Rectified RoPE)，它同样是一种RoPE的后处理方案，实验结果显示它的不微调长度外推能力不仅明显超过了此前的NTK\-aware Scaled RoPE，甚至还超过了之前专门设计的需要从零训练的HFWA。此外，不同于NTK\-aware Scaled RoPE在超过某个长度后能力会大幅下降，ReRoPE似乎在任意长度下都表现良好。除了对比实验外，文章还给出了基于transformers\-llama的参考实现，有兴趣的读者可以自行测试。
