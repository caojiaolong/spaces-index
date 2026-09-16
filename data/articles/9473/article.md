# Google新搜出的优化器Lion：效率与效果兼得的“训练狮”

> 作者：苏剑林 · 科学空间 · 2023-02-16
>
> 原文：<https://spaces.ac.cn/archives/9473>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨天在Arixv上发现了Google新发的一篇论文[《Symbolic Discovery of Optimization Algorithms》](<https://papers.cool/arxiv/2302.06675>)，主要是讲自动搜索优化器的，咋看上去没啥意思，因为类似的工作也有不少，大多数结果都索然无味。然而，细读之下才发现别有洞天，原来作者们通过数千TPU小时的算力搜索并结合人工干预，得到了一个速度更快、显存更省的优化器Lion（Evo<strong>L</strong>ved S<strong>i</strong>gn M<strong>o</strong>me<strong>n</strong>tum，不得不吐槽这名字起得真勉强），并在图像分类、图文匹配、扩散模型、语言模型预训练和微调等诸多任务上做了充分的实验，多数任务都显示Lion比目前主流的AdamW等优化器有着更好的效果。

更省显存还更好效果，真可谓是鱼与熊掌都兼得了，什么样的优化器能有这么强悍的性能？本文一起来欣赏一下论文的成果。

## 先说结果

本文主要关心搜索出来的优化器本身，所以关于搜索过程的细节就不讨论了，对此有兴趣读者自行看原论文就好。Lion优化器的更新过程为
\begin{equation}\text{Lion}:=\left\{\begin{aligned}
&\boldsymbol{u}_t = \text{sign}\big(\beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\big) \\
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}}) \\
&\boldsymbol{m}_t = \beta_2 \boldsymbol{m}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t
\end{aligned}\right.\end{equation}
其中$\boldsymbol{g}_t = \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}_{t-1})$是损失函数的梯度，$\text{sign}$是[符号函数](<https://en.wikipedia.org/wiki/Sign_function>)，即正数变为1、负数变为\-1。我们可以对比一下目前的主流优化器[AdamW](<https://papers.cool/arxiv/1711.05101>)的更新过程
\begin{equation}\text{Adam}\color{skyblue}{\text{W}}:=\left\{\begin{aligned}
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\
&\boldsymbol{u}_t =\hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.\\
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})
\end{aligned}\right.\end{equation}
对比很明显，Lion相比AdamW参数更少（少了个$\epsilon$），少缓存了一组参数$\boldsymbol{v}$（所以更省显存），并且去掉了AdamW更新过程中计算量最大的除法和开根号运算（所以更快）。

在此之前，跟Lion最相似的优化器应该是[SIGNUM](<https://papers.cool/arxiv/1802.04434>)，其更新过程为
\begin{equation}\text{SIGNUM}:=\left\{\begin{aligned}
&\boldsymbol{m}_t = \beta \boldsymbol{m}_{t-1} + \left(1 - \beta\right) \boldsymbol{g}_t \\
&\boldsymbol{u}_t = \text{sign}\big(\boldsymbol{m}_t\big) \\
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \boldsymbol{u}_t \end{aligned}\right.\end{equation}
跟Lion一样，SIGNUM也用到了符号函数处理更新量，而且比Lion更加简化（等价于Lion在$\beta_1=\beta_2$和$\lambda_t=0$的特例），但是很遗憾，SIGNUM并没有取得更好的效果，它的设计初衷只是降低分布式计算中的传输成本。Lion的更新规则有所不同，尤其是动量的更新放在了变量的更新之后，并且在充分的实验中显示出了它在效果上的优势。

## 论文实验

本文开头就说了，Lion在相当多的任务上都做了实验，实验结果很多，下面罗列一些笔者认为比较关键的结果。

[![Lion在NLU和NLG任务上的结果，大部分都比AdamW、Adafactor优秀](<https://spaces.ac.cn/usr/uploads/2023/02/3166342404.png>)](<https://spaces.ac.cn/usr/uploads/2023/02/3166342404.png>)

Lion在NLU和NLG任务上的结果，大部分都比AdamW、Adafactor优秀

[![在视觉Transformer上Lion与众多优化器的对比](<https://spaces.ac.cn/usr/uploads/2023/02/3192886331.png>)](<https://spaces.ac.cn/usr/uploads/2023/02/3192886331.png>)

在视觉Transformer上Lion与众多优化器的对比

[![在CV的分类任务上，Lion收敛速度更快](<https://spaces.ac.cn/usr/uploads/2023/02/3511074534.png>)](<https://spaces.ac.cn/usr/uploads/2023/02/3511074534.png>)

在CV的分类任务上，Lion收敛速度更快

[![在NLP的自回归生成上，Lion的收敛速度更快](<https://spaces.ac.cn/usr/uploads/2023/02/4036852564.png>)](<https://spaces.ac.cn/usr/uploads/2023/02/4036852564.png>)

在NLP的自回归生成上，Lion的收敛速度更快

[![上右图是ImageNet上的训练曲线，显示Lion尽管验证集效果更好，但训练集上的效果未必会优于AdamW](<https://spaces.ac.cn/usr/uploads/2023/02/3568656053.png>)](<https://spaces.ac.cn/usr/uploads/2023/02/3568656053.png>)

上右图是ImageNet上的训练曲线，显示Lion尽管验证集效果更好，但训练集上的效果未必会优于AdamW

## 超参设置

看到论文效果如此惊人，笔者也跃跃欲试。在跑实验之前，自然需要了解一下各个超参的设置。首先是$\beta_1,\beta_2$，原论文自动搜索出来的结果是$\beta_1=0.9,\beta=0.99$，并在大部分实验中复用了这个组合，但是在NLP的任务上则使用了$\beta_1=0.95,\beta_2=0.98$这个组合（论文的详细实验配置在最后一页的Table 12）。

比较关键的学习率$\eta$和权重衰减率$\lambda$，由于Lion的更新量$\boldsymbol{u}$每个分量的绝对值都是1，这通常比AdamW要大，所以学习率要缩小10倍以上，才能获得大致相同的更新幅度；而由于学习率降低了，那么为了使权重衰减的幅度保持不变，权重衰减率应该要放大相应的倍数。原论文的最后一页给出了各个实验的超参数参考值，其中小模型（Base级别）上使用的是$\eta = 3\times 10^{-4}$和$\lambda=0.01$，大模型（参数10亿以上）则适当降低了学习率到$\eta = 2\times 10^{-4}$甚至$\eta = 10^{-4}$。

事实上，之前我们在[《基于Amos优化器思想推导出来的一些“炼丹策略”》](<https://spaces.ac.cn/archives/9344>)就推导过学习率和权重衰减率的一个组合方案，参考这个方案来设置是最方便的。在该方案中，更新量写为（记号跟前面的描述略有不同，但不至于混淆，应该就不强行统一了）
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - (\alpha_t \boldsymbol{u}_t + \rho_t\boldsymbol{\theta}_t)\end{equation}
其中
\begin{equation}\alpha_t \approx \frac{\alpha_0\Vert\boldsymbol{\varepsilon}_0\Vert}{\Vert\boldsymbol{u}_t\Vert} \frac{1}{\kappa t + 1},\quad \rho_t \approx \frac{\alpha_0^2}{2q} \frac{1}{\kappa t + 1}\end{equation}
其中$\boldsymbol{u}_t$是原本的更新量；$\alpha_0$是（初始阶段）参数变化的相对大小，一般是$10^{-3}$级别，表示每步更新后参数模长的变化幅度大致是千分之一；$q$是一个超参数，没什么特殊情况可以设为1；$\kappa$是控制学习率衰减速度的超参数，可以根据训练数据大小等设置。

由于$\boldsymbol{u}_t$经过了$\text{sign}$运算，因此$\Vert\boldsymbol{u}_t\Vert=\sqrt{k}$，$k$是参数的维度；$\Vert\boldsymbol{\varepsilon}_0\Vert\approx\sqrt{k}\sigma$，这我们在[《基于Amos优化器思想推导出来的一些“炼丹策略”》](<https://spaces.ac.cn/archives/9344>)已经推导过了，其中$\sigma$是参数的变化尺度，对于乘性矩阵，$\sigma^2$就是它的初始化方差。所以，经过一系列简化之后，有
\begin{equation}\alpha_t \approx \frac{\alpha_0\sigma}{\kappa t + 1},\quad \rho_t \approx \frac{\alpha_0^2}{2(\kappa t + 1)}\end{equation}
这里的$\alpha_t$就是前面的$\eta_t$，而$\lambda_t = \rho_t / \alpha_t = \alpha_0 / 2\sigma$。按照BERT base的$d=768$来算，初始化方差的量级大致在$1/d$左右，于是$\sigma = \sqrt{1/d}\approx 0.036$，假设$\alpha_0$取$1.11 \times 10^{-3}$（为了给结果凑个整），那么按照上式学习率大约是$4\times 10^{-5}$、衰减率大约是$0.015$。在笔者自己的MLM预训练实验中，选取这两个组合效果比较好。

> <strong>个人实现：[https://github\.com/bojone/bert4keras](<https://github.com/bojone/bert4keras/commit/b60e7cfe076c0302473bbc3d63fed7e97f1c377f>)</strong>

## 延伸思考

总体来看，Lion表现可圈可点，不管是原论文还是笔者自己的实验中，跟AdamW相比都有一战之力，再加上Lion更快以及更省显存的特点，或者可以预见未来的主流优化器将有它的一席之地。

自Adam提出以来，由于其快速收敛的特性成为了很多模型的默认优化器。甚至有学者提出，这个现象将反过来导致一个进化效应：所有的模型改进都在往Adam有利的方向发展，换句话说，由于我们选择了Adam作为优化器，那么就有可能将很多实际有效、但是在Adam优化器上无效的改动都抛弃了，剩下的都是对Adam有利的改进，详细的评价可以参考[《NEURAL NETWORKS (MAYBE) EVOLVED TO MAKE ADAM THE BEST OPTIMIZER》](<https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/>)。所以，在此大背景之下，能够发现比Adam更简单且更有效的优化器，是一件很了不起的事情，哪怕它是借助大量算力搜索出来的。

可能读者会有疑问：Lion凭啥可以取得更好的泛化性能呢？原论文的解释是$\text{sign}$这个操作引入了额外的噪声（相比于准确的浮点值），它使得模型进入了Loss更平坦（但未必更小）的区域，从而泛化性能更好。为了验证这一点，作者比较了AdamW和Lion训练出来的模型权重的抗干扰能力，结果显示Lion的抗干扰能力更好。然而，理论上来说，这只能证明Lion确实进入到了更平坦的区域，但无法证明该结果是$\text{sign}$操作造成的。不过，Adam发表这么多年了，关于它的机理也还没有彻底研究清楚，而Lion只是刚刚提出，就不必过于吹毛求疵了。

笔者的猜测是，Lion通过$\text{sign}$操作平等地对待了每一个分量，使得模型充分地发挥了每一个分量的作用，从而有更好的泛化性能。如果是SGD，那么更新的大小正比于它的梯度，然而有些分量梯度小，可能仅仅是因为它没初始化好，而并非它不重要，所以Lion的$\text{sign}$操作算是为每个参数都提供了“恢复活力”甚至“再创辉煌”的机会。事实上可以证明，Adam早期的更新量也接近于$\text{sign}$，只是随着训练步数的增加才逐渐偏离。

Lion是不是足够完美呢？显然不是，比如原论文就指出它在小batch\_size（小于64）的时候效果不如AdamW，这也不难理解，本来$\text{sign}$已经带来了噪声，而小batch\_size则进一步增加了噪声，噪声这个东西，必须适量才好，所以两者叠加之下，很可能有噪声过量导致效果恶化。另外，也正因为$\text{sign}$加剧了优化过程的噪声，所以参数设置不当时容易出现损失变大等发散情况，这时候可以尝试引入Warmup，或者增加Warmup步数。还有，Lion依旧需要缓存动量参数，所以它的显存占用多于[AdaFactor](<https://spaces.ac.cn/archives/7302>)，能不能进一步优化这部分参数量呢？暂时还不得而知。

## 文章小结

本文介绍了Google新提出的优化器Lion，它通过大量算力搜索并结合人工干预得出，相比主流的AdamW，有着速度更快且更省内存的特点，并且大量实验结果显示，它在多数任务上都有着不逊色于甚至优于AdamW的表现。
