# Transformer升级之路：11、将β进制位置进行到底

> 作者：苏剑林 · 科学空间 · 2023-07-31
>
> 原文：<https://spaces.ac.cn/archives/9706>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《Transformer升级之路：10、RoPE是一种β进制编码》](<https://spaces.ac.cn/archives/9675>)中，我们给出了RoPE的$\beta$进制诠释，并基于进制转化的思路推导了能够在不微调的情况下就可以扩展Context长度的[NTK\-aware Scaled RoPE](<https://spaces.ac.cn/archives/9675#%E8%BF%BD%E6%A0%B9%E6%BA%AF%E6%BA%90>)。不得不说，通过类比$\beta$进制的方式来理解位置编码，确实是一个非常美妙且富有启发性的视角，以至于笔者每次深入思考和回味之时，似乎总能从中得到新的领悟和收获。

本文将重新回顾RoPE的$\beta$进制诠释，并尝试将已有的NTK\-aware Scaled RoPE一般化，以期望找到一种更优的策略来不微调地扩展LLM的Context长度。

## 进制类比

我们知道，RoPE的参数化沿用了[Sinusoidal位置编码](<https://spaces.ac.cn/archives/9675>)的形式。而不知道是巧合还是故意为之，整数$n$的Sinusoidal位置编码，与它的$\beta$进制编码，有很多相通之处。

具体来说，整数$n$的$\beta$进制表示的（从右往左数）第$m$位数字是：
\begin{equation}\left\lfloor\frac{n}{\beta^{m-1}}\right\rfloor\bmod\beta\label{eq:mod}\end{equation}
而它的Sinusoidal位置编码是
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\[5pt]
\theta_m = \frac{n}{\beta^{m-1}},\quad \beta=10000^{2/d}
\label{eq:sinu}\end{equation}
可以看到，两者都有相同的$\frac{n}{\beta^{m-1}}$，并且$\bmod$和$\cos,\sin$同为周期函数，所以两者的唯一差距，只是无关紧要的取整$\lfloor\cdot\rfloor$了。所以说，将RoPE/Sinusoidal位置编码类比为它$\beta$进制表示，是非常直观且合理的结果。

## 修正NTK

沿着[《Transformer升级之路：10、RoPE是一种β进制编码》](<https://spaces.ac.cn/archives/9675>)的思路，直接外推会将外推压力集中在“高位（$m$较大）”上，而位置内插则会将“低位（$m$较小）”的表示变得更加稠密，不利于区分相对距离。而NTK\-aware Scaled RoPE其实就是进制转换，它将外推压力平摊到每一位上，并且保持相邻间隔不变，这些特性对明显更倾向于依赖相对位置的LLM来说是非常友好和关键的，所以它可以不微调也能实现一定的效果。

仔细看式$\eqref{eq:sinu}$，$\cos,\sin$事实上是一个整体，所以它实际只有$d/2$位，也就是说它相当于$n$的$d/2$位$\beta$进制编码。如果我们要扩展到$k$倍Context，将$\beta$进制转换为$\beta\lambda$进制，那么至少应该有
\begin{equation}\lambda^{d/2}=k\quad\Rightarrow\quad\lambda = k^{2/d}\end{equation}
于是新的RoPE变为
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\[5pt]
\theta_m = \frac{n}{(\beta\lambda)^{m-1}},\quad \beta=10000^{2/d},\quad \lambda = k^{2/d}\label{eq:ntk-old}\end{equation}
这就是上一篇文章我们提出的NTK\-RoPE。

然而，后来笔者仔细思考后，发现这其实还不够合理。回到式$\eqref{eq:mod}$，如果要计算$\beta\lambda$进制的第$m$位数字，那么应该是
\begin{equation}\left\lfloor\frac{n}{(\beta\lambda)^{m-1}}\right\rfloor\bmod(\beta\lambda)\end{equation}
也就是说，除了$\frac{n}{\beta^{m-1}}$要换成$\frac{n}{(\beta\lambda)^{m-1}}$之外，求$\bmod$的周期也要扩大$\lambda$倍，这等价于求$\cos,\sin$之前，要多除以一个$\lambda$：
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\[5pt]
\theta_m = \frac{n}{\lambda(\beta\lambda)^{m-1}},\quad \beta=10000^{2/d},\quad \lambda = k^{2/d}\label{eq:ntk-fixed}\end{equation}
在后面的实验中，我们把上一篇文章提出的式$\eqref{eq:ntk-old}$称为“NTK\-RoPE\-old”，而式$\eqref{eq:ntk-fixed}$称为“NTK\-RoPE\-fixed”。

## 混合进制

现在，不妨让我们更加“天马行空”一些——既然我们可以用$\beta$进制来表示位置，那么为何不干脆使用更一般化的“混合进制”呢？这里的混合进制，指的是每一位数字所使用的进位基数不尽相同，这对于我们来说并不鲜见，比如60秒是1分钟、60分是1小时，但24小时是1天、7天是1周，这里的60、60、24、7就是不同进制基数，换句话说秒、分、时、天、周就是一个使用混合进制的例子。

假设从右往左数，第1位使用$\beta_1$进制、第2位使用$\beta_2$进制、第3位使用$\beta_3$进制、\.\.\.，那么求$n$的第$m$位数字，结果是
\begin{equation}\left\lfloor\frac{n}{\beta_1\beta_2\cdots\beta_{m-1}}\right\rfloor\bmod\beta_m\label{eq:mod2}\end{equation}
为什么会考虑到混合进制呢？这是因为某天笔者发现了一个有趣的事实：RoPE本质上是一种相对位置编码，相对位置是[Toeplitz矩阵](<https://en.wikipedia.org/wiki/Toeplitz_matrix>)的一个特例，它长这个样（由于本文主要关心语言模型，所以右上角部分就没写出来了）
\begin{equation}\begin{pmatrix}0 & \\
1 & 0 & \\
2 & 1 & 0 &\\
3 & 2 & 1 & 0 & \\
4 & 3 & 2 & 1 & 0 & \\
5 & 4 & 3 & 2 & 1 & 0 & \\
6 & 5 & 4 & 3 & 2 & 1 & 0 & \\
\end{pmatrix}\end{equation}
从上式我们可以发现，相对位置编码的位置分布是不均衡的！0的出现次数最多、1次之、2再次之，以此类推，即$n$越大出现次数越少。这就意味着，作为一种$\beta$进制编码的RoPE，它的“高位”很可能是训练不充分的，换言之高位的泛化能力很可能不如低位。刚才我们说了，NTK\-RoPE将外推压力平摊到每一位上，如果这里的猜测合理的话，那么“平摊”就不是最优的，应该是低位要分摊更多，高位分摊更少，这就导致了混合进制。

## 分摊优化

具体来说，我们通过将$\beta$进制转换为$\beta_1,\beta_2,\cdots,\beta_{d/2}$混合进制的方式来扩展到$k$倍Context，这里$\beta_m = \beta \lambda_m$。此时式$\eqref{eq:mod2}$变为
\begin{equation}\left\lfloor\frac{n}{\beta^{m-1}(\lambda_1\lambda_2\cdots\lambda_{m-1})}\right\rfloor\bmod(\beta\lambda_m)\end{equation}
式$\eqref{eq:ntk-fixed}$也相应地变成
\begin{equation}\boldsymbol{p}_n=\big[\cos\theta_1,\sin\theta_1,\cos\theta_2,\sin\theta_2,\cdots,\cos\theta_{d/2},\sin\theta_{d/2}\big]\\[5pt]
\theta_m = \frac{n}{\beta^{m-1}(\lambda_1\lambda_2\cdots\lambda_m)},\quad \beta=10000^{2/d}\end{equation}
根据“扩展$k$倍”和“低位要分摊更多”的原则，约束条件是
\begin{equation}\lambda_1\lambda_2\cdots\lambda_{d/2} = k,\quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{d/2} \geq 1\end{equation}
我们讨论如下形式的解（有兴趣的读者也可以试探别的形式的解，这里自由度本身就很大）
\begin{equation}\lambda_1\lambda_2\cdots\lambda_m = \exp(am^b)\end{equation}
当$a > 0, b\leq 1$时，它满足$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_{d/2} \geq 1$的条件，当$b=1$时，实际上就是前面的“NTK\-RoPE\-fixed”，当$b=0$时，就是“[Positional Interpolation（PI）](<https://spaces.ac.cn/archives/9675#%E7%BA%BF%E6%80%A7%E5%86%85%E6%8F%92>)”。$\lambda_1\lambda_2\cdots\lambda_{d/2} = k$给出了约束
\begin{equation}a\left(\frac{d}{2}\right)^b = \log k\end{equation}
所以只有一个自由度可以调。经过简单的二分法搜索，笔者发现在自己的实验中，$b=0.625$能取得平均来说比较好的扩展效果（不同的模型可能会有不同的最优解，请自行调试），这个版本被称为“NTK\-RoPE\-mixed”。

## 实验结果

在[《Transformer升级之路：10、RoPE是一种β进制编码》](<https://spaces.ac.cn/archives/9675>)的实验基础上，笔者补做了“NTK\-RoPE\-fixed”和“NTK\-RoPE\-mixed”的实验，对比如下：
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
\text{NTK-RoPE-}\log n\text{-fixed} & 49.40\% & 62.85\% & 44.14\% \\
\text{NTK-RoPE-mixed} & 49.41\% & 53.09\% & 40.12\% \\
\text{NTK-RoPE-}\log n\text{-mixed} & 49.40\% & \boldsymbol{68.91\%} & \boldsymbol{45.41\%} \\
\hline
\end{array}

可以看到，相比等进制的“NTK\-RoPE\-old”和“NTK\-RoPE\-fixed”，混合进制推导出来的“NTK\-RoPE\-mixed”所带来的提升还是很明显的，而且不用微调，可谓是“免费午餐”了。此外，可以看到$\log n$版的外扩性能确实更好，但是$\log n$技巧需要在预训练阶段就加入，之前就有读者问过像LLAMA这种在预训练阶段并没有加入$\log n$技巧的模型，可否享受到$\log n$的“红利”呢？经过笔者测试，发现它可以通过加入如下scale因子来提升效果：
\begin{equation}\max(1, \log_{\text{maxlen}} n)\label{eq:plogn}\end{equation}
这里的$\text{maxlen}$是预训练的最大长度，在本文的实验中是512，在LLAMA中是2048，LLAMA2则是4096，实现时可以直接给每个$\boldsymbol{q}_n$乘上相应的因子。这样一来，在$\text{maxlen}$之内的部分不受影响，之外的部分则按$\log n$缩放，算是一种简单的过渡，效果如下（加个$\color{red}{\dagger}$区别原来的$\log n$）：
\begin{array}{c|cc}
\hline
\text{测试长度} & 512(\text{训练}) & 4096(\text{重复}) & 4096(\text{不重复})\\
\hline
\text{NTK-RoPE-fixed} & 49.41\% & 51.86\% & 39.61\% \\
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-fixed} & 49.41\% & 55.94\% & 41.11\% \\
\text{NTK-RoPE-mixed} & 49.41\% & 53.09\% & 40.12\% \\
\text{NTK-RoPE-}\log n^{\color{red}{\dagger}}\text{-mixed} & 49.41\% & 59.11\% & 42.38\% \\
\hline
\end{array}
可以看到，这个$\log n^{\color{red}{\dagger}}$也算得上免费的午餐了。总之，如果你打算进行从零预训练，不妨事先就加入$\log n$技巧，如果已经训练完成，那么可以使用式$\eqref{eq:plogn}$替代，最后再加上NTK\-RoPE\-mixed，能够取得较优的拓展Context效果。

## 文章小结

在这篇文章中，我们重温了RoPE的$\beta$进制视角，并尝试对NTK\-aware Scaled RoPE进行推广，在混合进制的启发下，我们得到了一个更优的不微调扩展Context长度的策略，最后通过实验表明了它的有效性。
