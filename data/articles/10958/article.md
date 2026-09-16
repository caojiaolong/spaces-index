# 生成扩散模型漫谈（三十）：从瞬时速度到平均速度

> 作者：苏剑林 · 科学空间 · 2025-05-26
>
> 原文：<https://spaces.ac.cn/archives/10958>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，生成速度慢是扩散模型一直以来的痛点，而为了解决这个问题，大家可谓“八仙过海，各显神通”，提出了各式各样的解决方案，然而长久以来并没一项工作能够脱颖而出，成为标配。什么样的工作能够达到这个标准呢？在笔者看来，它至少满足几个条件：

> 1、数学原理清晰，能够揭示出快速生成的本质所在；
> 
> 
> 
> 2、能够单目标从零训练，不需要对抗、蒸馏等额外手段；
> 
> 
> 
> 3、单步生成接近SOTA，可以通过增加步数提升效果。

根据笔者的阅读经历，几乎没有一项工作能同时满足这三个标准。然而，就在几天前，arXiv出了一篇[《Mean Flows for One\-step Generative Modeling》](<https://papers.cool/arxiv/2505.13447>)（简称“MeanFlow”），看上去非常有潜力。接下来，我们将以此为契机，讨论一下相关思路和进展。

## 现有思路

扩散模型的生成加速工作已经有非常多，本博客前面也简单介绍过一些。总的来说，加速思路大体上可以分为三类。

第一，将扩散模型转化为SDE/ODE，然后研究更高效的求解器，代表作是[DPM\-Solver](<https://papers.cool/arxiv/2206.00927>)及其一系列后续改进。然而，这个思路通常只能将生成的NFE（Number of Function Evaluations）降到10左右，再低就会明显降低生成质量。这是因为求解器的收敛速度通常都是正比于步长的若干次方，当NFE很小时步长就无法很小，所以收敛不够快以至于没法用。

第二，通过蒸馏将训练好的扩散模型转化为更少步数的生成器，由此衍生出来的工作和方案也非常多，我们此前介绍过其中的一种名为[SiD](<https://spaces.ac.cn/archives/10085>)的方案。蒸馏算是比较常规和通用的思路，但缺点也是共同的，即需要额外的训练成本，并非从零训练的方案。有些工作为了蒸馏到单步生成器，还加上了对抗训练等多重优化策略，整个方案往往过于复杂。

第三，基于一致性模型（Consistency Model，CM），包括我们在[《生成扩散模型漫谈（二十八）：分步理解一致性模型》](<https://spaces.ac.cn/archives/10633>)简单介绍的CM、它的连续版本[sCM](<https://papers.cool/arxiv/2410.11081>)以及[CTM](<https://papers.cool/arxiv/2310.02279>)等。CM是自成一派的思路，可以从零训练得到NFE很小的模型，也可以用于蒸馏，但CM的目标依赖于EMA或者stop\_gradient运算，意味着它耦合了优化器动力学，这通常给人一种说不清道不明的感觉。

## 瞬时速度

到目前为止，生成NFE最小的扩散模型，基本上都是ODE，因为确定性模型往往更容易分析和求解。本文同样只关注ODE式扩散，所用框架是[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](<https://spaces.ac.cn/archives/9497>)介绍的ReFlow，它跟[Flow Matching](<https://papers.cool/arxiv/2210.02747>)本质是相通的，但更加直观。

ODE式扩散，是希望学习一个ODE
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\label{eq:ode}\end{equation}
来构建一个$\boldsymbol{x}_1\to \boldsymbol{x}_0$的变换。具体来说，设$\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)$是某个容易采样的随机噪声，$\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)$则是目标分布的真实样本，我们希望能够通过上述ODE，实现随机噪声到目标样本的变换，即随机采样一个$\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)$作为初值，求解上述ODE得到的$\boldsymbol{x}_0$就是$p_0(\boldsymbol{x}_0)$的样本。

如果将$t$看成时间，$\boldsymbol{x}_t$看成位移，那么$d\boldsymbol{x}_t/dt$就是瞬时速度，所以ODE式扩散就是瞬时速度的建模。那怎么训练$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$呢？ReFlow提出了一种非常直观的方法：首先构建$\boldsymbol{x}_0$与$\boldsymbol{x}_1$的任意插值方式，如最简单的线性插值$\boldsymbol{x}_t=(1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1$，那么对$t$求导得
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0\end{equation}
这是个极简单的ODE，但不符合我们的要求，因为$\boldsymbol{x}_0$是我们的目标，它不应该出现在ODE中。对此，ReFlow提出一个非常符合直觉的想法——用$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$去逼近$\boldsymbol{x}_1 - \boldsymbol{x}_0$：
\begin{equation}\mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{x}_1}\left[\Vert\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right]\label{eq:loss-reflow}\end{equation}
这就是ReFlow的目标函数。值得指出的是：1）ReFlow理论上允许$\boldsymbol{x}_0$与$\boldsymbol{x}_1$的任意插值方式；2）ReFlow虽然直观，但理论上也是严格的，可以证明它的最优解确实是我们所求的ODE。相关细节大家请参考[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](<https://spaces.ac.cn/archives/9497>)以及原论文。

## 平均速度

然而，ODE仅仅是一个纯数学形式，实际求解还是需要离散化，比如最简单的欧拉格式：
\begin{equation}\boldsymbol{x}_{t - \Delta t} = \boldsymbol{x}_t - \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) \Delta t\end{equation}
从$1$到$0$的NFE是$1/\Delta t$，想要NFE小等价于$\Delta t$大。然而，ReFlow的理论基础是精确的ODE，即精确求解ODE时才能实现目标样本的生成，这意味着$\Delta t$越小越好，跟我们的期望相背。尽管ReFlow声称使用直线插值可以让ODE的轨迹变得更直，从而允许更大的$\Delta t$，但实际轨迹终究是弯曲的，$\Delta t$很难接近1，所以ReFlow很难实现一步生成。

归根结底，ODE本来就是$\Delta t\to 0$的东西，我们非要将它用于$\Delta t \to 1$，还要求它效果好，这本身就是“强模型所难”了。所以说，更换建模目标，而不是继续“为难”模型，才是实现更快生成的本质思路。为此，我们考虑对式$\eqref{eq:ode}$两端进行积分
\begin{equation}\boldsymbol{x}_t - \boldsymbol{x}_r = \int_r^t \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau},\tau) d\tau = (t-r)\times \frac{1}{t-r}\int_r^t \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau},\tau) d\tau\end{equation}
如果我们可以建模
\begin{equation} \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) \triangleq \frac{1}{t-r}\int_r^t \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau},\tau) d\tau\end{equation}
那么就有$\boldsymbol{x}_0 = \boldsymbol{x}_1 - \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_1, 0, 1)$，即理论上可以精准地实现一步生成，而不必求诸于近似关系。如果说$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$是$t$时刻的瞬时速度，那么很显然$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$是$[r,t]$时间段内的平均速度。也就是说，为了加速生成甚至一步生成，我们的建模目标应该是平均速度，而不是ODE的瞬时速度。

## 恒等变换

当然，从瞬时速度到平均速度的转变并不难想，真正难的地方是如何给它构建损失函数。ReFlow只告诉我们如何给瞬时速度构建损失函数，对平均速度的训练我们是一无所知。

接下来很自然的想法是“化未知为已知”，即以平均速度$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$来为出发点来构建瞬时速度$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$，然后代入ReFlow的目标函数，这需要我们去推导两者之间的恒等变换。从$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$的定义我们得到
\begin{equation} \int_r^t \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau},\tau) d\tau = (t-r)\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) \end{equation}
两边对$t$求导，得到
\begin{equation}\begin{aligned}
\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) =&\, \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\frac{d}{dt}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) \\
=&\, \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\left[\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right]
\end{aligned}\label{eq:id1}\end{equation}
这便是$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$跟$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$的第一个恒等关系。有第一自然就有第二，第二个恒等关系由平均速度的定义得到：
\begin{equation}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) = \lim_{r\to t}\frac{1}{t-r}\int_r^t \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau},\tau) d\tau = \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)\label{eq:id2}\end{equation}
说白了，无限小区间内的平均速度，就等于瞬时速度。

## 第一目标

根据$d\boldsymbol{x}_t/dt = \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$以及恒等式$\eqref{eq:id2}$，我们可以将恒等式$\eqref{eq:id1}$的$d\boldsymbol{x}_t/dt$换成$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$或$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)$，前者是隐式关系，我们后面再谈，我们先看后者，此时有：
\begin{equation}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) = \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\left[\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right]\end{equation}
代入ReFlow，我们得到可以用来训练$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$的第一个目标函数：
\begin{equation}\mathbb{E}_{r,t,\boldsymbol{x}_0,\boldsymbol{x}_1}\left[\left\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\left[\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right] - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\right\Vert^2\right]\label{eq:loss-1}\end{equation}
这是一个非常理想的结果，它满足我们对生成模型目标函数的所有期望：

> 1、单个显式的最小化目标；
> 
> 
> 
> 2、没有EMA、stop\_gradient等运算；
> 
> 
> 
> 3、理论上有保证（ReFlow）。

这些特性意味着，不管我们用什么优化算法，只要我们能找到上式的最小值点，那么它就是我们想要的平均速度模型，即理论上能够实现一步生成的生成模型。换句话说，它具备了扩散模型的训练简单和理论保证，又能像GAN那样一步生成，还不用求神拜佛保佑模型别“想不开”而训崩。

## JVP运算

不过，对于部分读者来说，目标函数$\eqref{eq:loss-1}$的实现还是有点困难的，因为它涉及到普通用户比较少见的“雅可比向量积（Jacobian\-Vector Product，JVP）”。具体来说，我们可以将目标函数内方括号部分写成：
\begin{equation}\underbrace{\left[\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t),0,1\right] \\[10pt]}_{\text{向量}}\cdot\underbrace{\left[\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t), \frac{\partial}{\partial r}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t), \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right] \\[10pt]}_{\text{雅可比矩阵}}\end{equation}
即$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$的雅可比矩阵与给定向量$[\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t),0,1]$的乘法，结果是一个跟$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$大小一致的向量，这种运算就叫做JVP，在Jax、Torch里边都有现成实现，比如Jax的参考代码是：



<pre><code class="language-python">u = lambda xt, r, t: diffusion_model(weights, [xt, r, t])
urt, durt = jax.jvp(u, (xt, r, t), (u(xt, t, t), r * 0, t * 0 + 1))</code></pre>



其中<code>urt</code>就是$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$，而<code>durt</code>就是对应的JVP结果，Torch的用法也类似。了解JVP运算后，目标函数$\eqref{eq:loss-1}$的实现就基本上没有难度了。

## 第二目标

如果要说目标函数$\eqref{eq:loss-1}$的缺点，在笔者看来只有一个，那就是计算量相对偏大。这是因为它要进行两次不同的前向传播$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$和$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)$，然后JVP求了一次梯度，用基于梯度下降优化时还要再求一次梯度，所以它本质上要求二阶梯度，跟以往的[WGAN\-GP](<https://spaces.ac.cn/archives/4439>)类似。

为了降低计算量，我们可以考虑给JVP部分加上stop\_gradient运算（$\newcommand{\sg}[1]{\color{skyblue}{\mathop{\text{sg}}\left[\color{blue}{#1}\right]}}\sg{\cdot}$）：
\begin{equation}\mathbb{E}_{r,t,\boldsymbol{x}_0,\boldsymbol{x}_1}\left[\left\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\sg{\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)} - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\right\Vert^2\right]\label{eq:loss-2}\end{equation}
这样就避免了对JVP再次求梯度（但依然需要两次前向传播）。实测结果显示，相比第一目标$\eqref{eq:loss-1}$，上述目标在梯度优化器下训练速度能够快将近一倍，并且效果目测无损。

注意，这里的stop\_gradient单纯是出于减少计算量的目的，实际优化方向依然是损失函数值越小越好，这跟CM系列模型尤其是sCM是不一样的，它们的损失函数只是具有等效梯度的等效损失，并不一定是越小越好，它们的stop\_gradient往往是必须的，一旦去掉几乎可以肯定会训练崩溃。

## 第三目标

前面我们提到，处理恒等式$\eqref{eq:id1}$中的$d\boldsymbol{x}_t/dt$的另一个方案是将其换成$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$，这将导致
\begin{equation}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) = \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\left[\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right]\end{equation}
如果要从中解出$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$，结果将是
\begin{equation}\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t) = \left[\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right]\cdot\left[\boldsymbol{I} - (t-r)\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right]^{-1}\end{equation}
这需要对非常庞大的矩阵求逆，并不现实。MeanFlow用了一个折中方案：既然$d\boldsymbol{x}_t/dt = \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$的回归目标是$\boldsymbol{x}_1-\boldsymbol{x}_0$，那干脆把$d\boldsymbol{x}_t/dt$换成$\boldsymbol{x}_1-\boldsymbol{x}_0$好了，于是目标函数变成
\begin{equation}\mathbb{E}_{r,t,\boldsymbol{x}_0,\boldsymbol{x}_1}\left[\left\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\left[(\boldsymbol{x}_1-\boldsymbol{x}_0)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)\right] - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\right\Vert^2\right]\end{equation}
然而，此时的$\boldsymbol{x}_1-\boldsymbol{x}_0$既是回归目标，又出现在模型$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t,t)$的定义中，难免会有一种“标签泄漏”的感觉。为了避免这个问题，MeanFlow采取的办法同样是给JVP部分加上stop\_gradient：
\begin{equation}\mathbb{E}_{r,t,\boldsymbol{x}_0,\boldsymbol{x}_1}\left[\left\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + (t-r)\sg{(\boldsymbol{x}_1-\boldsymbol{x}_0)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)} - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\right\Vert^2\right]\label{eq:loss-3}\end{equation}
这就是MeanFlow最终所用的损失函数，这里我们称之为“第三目标”。相比第二目标$\eqref{eq:loss-2}$，它少了一次前向传播$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t)$，所以训练速度会更快一些。但此时“标签泄漏”的引入和stop\_gradient的对策，使得第三目标的训练跟梯度优化器是耦合的，这就跟CM一样，多了一些说不清道不明的神秘感。

论文实验结果表明，加上$\sg{\cdot}$的目标$\eqref{eq:loss-3}$是能训出合理结果的，那如果去掉它呢？笔者向作者请教过，他表明去掉$\sg{\cdot}$后，训练依然能收敛，能多步生成，但没有一步生成能力了。其实这也不难理解，因为$r=t$时不管有没有$\sg{\cdot}$，目标函数都退化为ReFlow：
\begin{equation}\mathbb{E}_{t,\boldsymbol{x}_0,\boldsymbol{x}_1}\left[\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right]\label{eq:loss-reflow-2}\end{equation}
也就是说MeanFlow总有ReFlow在背后“兜底”，因此怎样也不至于太差。而去掉$\sg{\cdot}$后，“标签泄漏”的负面影响加剧，因此就不如加上它了。

## 证明一下

我们能否像ReFlow一样，从理论上证明第三目标$\eqref{eq:loss-3}$的最优解确实是我们期望的平均速度模型呢？让我们尝试一下。首先我们回顾证明ReFlow的两个关键引理：

> 1、$\mathop{\text{argmin}}_{\boldsymbol{\mu}}\mathbb{E}[\Vert\boldsymbol{\mu} - \boldsymbol{x}\Vert^2] = \mathbb{E}[\boldsymbol{x}]$，即最小化$\boldsymbol{\mu}$与$\boldsymbol{x}$的平方误差，最优解是$\boldsymbol{x}$的均值；
> 
> 
> 
> 2、按照分布轨迹$\boldsymbol{x}_t=(1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1$将$\boldsymbol{x}_1$变到$\boldsymbol{x}_0$的ODE形式解是$d\boldsymbol{x}_t/dt = \mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\boldsymbol{x}_1-\boldsymbol{x}_0]$。

其中，引理1的证明比较简单，直接对$\boldsymbol{\mu}$求梯度得$\mathbb{E}[\boldsymbol{\mu} - \boldsymbol{x}] = \boldsymbol{\mu} - \mathbb{E}[\boldsymbol{x}]$，令它等于零即可；引理2的证明细节则需要看[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](<https://spaces.ac.cn/archives/9497>)，其中$\mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\boldsymbol{x}_1-\boldsymbol{x}_0]$是需要先利用$\boldsymbol{x}_t=(1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1$消去$\boldsymbol{x}_1$，得到一个$\boldsymbol{x}_0,\boldsymbol{x}_t$的函数，然后对分布$p_t(\boldsymbol{x}_0|\boldsymbol{x}_t)$求期望，结果是关于$t,\boldsymbol{x}_t$的函数。

利用引理1，我们可以证明ReFlow的目标函数$\eqref{eq:loss-reflow}$的理论最优解就是$\boldsymbol{v}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t,t) = \mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\boldsymbol{x}_1-\boldsymbol{x}_0]$，结合引理2就得到$d\boldsymbol{x}_t/dt=\boldsymbol{v}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t,t)$是我们所求的ODE。第三目标$\eqref{eq:loss-3}$的证明类似，由于里边有$\sg{\cdot}$，对$\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t)$求梯度并让它等于零的结果是
\begin{equation}\begin{aligned}
\boldsymbol{0} =&\, \boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t) + \mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}\left[(t-r)\left[(\boldsymbol{x}_1-\boldsymbol{x}_0)\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)\right] - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\right] \\
=&\, \boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t) + (t-r)\left[\mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\boldsymbol{x}_1-\boldsymbol{x}_0]\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)\right] - \mathbb{E}_{\boldsymbol{x}_0|\boldsymbol{x}_t}[\boldsymbol{x}_1 - \boldsymbol{x}_0] \\
=&\, \boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t) + (t-r)\left[\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)\right] - \frac{d\boldsymbol{x}_t}{dt} \\
=&\, \frac{d}{dt}\left[(t - r) \boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)  - (\boldsymbol{x}_t - \boldsymbol{x}_r)\right] \\
\end{aligned}\end{equation}
所以在适当的边界条件下就有$\boldsymbol{x}_t - \boldsymbol{x}_r = (t - r) \boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)$，即我们期望的平均速度模型。

这个过程的关键是$\sg{\cdot}$的引入避免了对JVP部分求梯度，从而简化了梯度表达式并得到了正确的结果。如果去掉$\sg{\cdot}$的话，上式右端就要多乘一项JVP部分对$\boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)$的雅可比矩阵，结果就是最后无法将$\frac{d}{dt}\left[(t - r) \boldsymbol{u}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, r, t)  - (\boldsymbol{x}_t - \boldsymbol{x}_r)\right]$这一项分离出来，而引入$\sg{\cdot}$的数学意义便是为了解决此问题。

当然，笔者还是那句话，$\sg{\cdot}$的引入也使得整个模型的训练耦合了梯度优化器，多了一丝不清晰的感觉。此时梯度等于零的点，顶多算是一个驻点而非（局部）最小值点，所以稳定性也不明朗，这其实也是所有耦合$\sg{\cdot}$的模型的共性。

## 相关工作

非常有趣的是，我们之前介绍过的两篇加速生成的文章[《生成扩散模型漫谈（二十一）：中值定理加速ODE采样》](<https://spaces.ac.cn/archives/9881>)和[《生成扩散模型漫谈（二十七）：将步长作为条件输入》](<https://spaces.ac.cn/archives/10617>)，也都是以平均速度为核心的，并且思想上可以说是一脉相承的。尽管作者之间未必相互有联系，但他们的工作内容上确实给人一种承前启后的连贯感。

在中值定理篇，作者已经意识到了平均速度
\begin{equation}\frac{1}{t-r}\int_r^t \boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_{\tau},\tau) d\tau\end{equation}
的重要性，但他的做法是类比一维函数的积分中值定理，试图寻找$s\in[r,t]$使得$\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_s,s)$等于平均速度。这本质上还是寻找高阶Solver的思想，但不再是Training\-Free，而是需要少量的蒸馏步骤，对Solver来说算是一个小突破。

而步长输入篇所提的Shortcut模型，则几乎已经触碰到了MeanFlow，因为步长作为额外输入，跟MeanFlow的双时间参数$r,t$实质是等价的，不同的是它是直接以平均速度的性质作为额外的正则项来训练模型。用本文的记号，平均速度应该满足的性质是
\begin{equation}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) = \frac{1}{2}\left[\boldsymbol{u}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_t, s, t\right) + \boldsymbol{u}_{\boldsymbol{\theta}}\left(\boldsymbol{x}_s, r, s\right)\right]\end{equation}
其中$s = (r+t)/2$。所以Shortcut干脆以它来构建正则项
\begin{equation}\left\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, r, t) - \frac{1}{2}\sg{\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, s, t) + \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_s, r, s)}\right\Vert^2\end{equation}
跟ReFlow的目标$\eqref{eq:loss-reflow-2}$混合训练，实际训练中$\boldsymbol{x}_s = \boldsymbol{x}_t - (t-s)\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, s, t)$，$\sg{\cdot}$的引入在笔者看来主要也是为了节省计算量。Shortcut模型其实比MeanFlow更直观，但由于没有恒等变换和ReFlow带来的严格理论支撑，使得它看上去更多是一个过渡期的经验产物。

## 一致模型

最后我们再来讨论一下一致性模型。由于CM、sCM珠玉在前，MeanFlow的成功实际上也借鉴了它们的经验，尤其是给JVP加$\sg{\cdot}$的操作，这在原论文中也有提到。当然，MeanFlow作者之一何恺明老师本身也是操控梯度的大师（比如[SimSiam](<https://spaces.ac.cn/archives/7980>)），所以MeanFlow的出现看起来是非常水到渠成的。

离散的CM我们在[《生成扩散模型漫谈（二十八）：分步理解一致性模型》](<https://spaces.ac.cn/archives/10633>)仔细分析过，如果将其中CM的EMA算符换成stop\_gradient，求梯度并取$\Delta t\to 0$的极限，那么就得到了[《Simplifying, Stabilizing and Scaling Continuous\-Time Consistency Models》](<https://papers.cool/arxiv/2410.11081>)中的sCM的目标函数：
\begin{equation}\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\cdot \frac{d}{dt}\boldsymbol{f}_{\sg{\boldsymbol{\theta}}}(\boldsymbol{x}_t, t) = \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\cdot\sg{\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) + \frac{\partial}{\partial t}\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}\label{eq:loss-scm}\end{equation}
如果将$\frac{d\boldsymbol{x}_t}{dt}$换成$\boldsymbol{x}_1 - \boldsymbol{x}_0$，然后记$\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = \boldsymbol{x}_t - t\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t , 0, t)$，那么它的梯度跟$r=0$时的MeanFlow第三目标$\eqref{eq:loss-3}$是等价的：
\begin{equation}\begin{aligned}
\nabla_{\boldsymbol{\theta}}\eqref{eq:loss-scm} =&\, \nabla_{\boldsymbol{\theta}}\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\cdot \left[\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) + \frac{\partial}{\partial t}\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right] \\[10pt]
=&\, -t\nabla_{\boldsymbol{\theta}}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t)\cdot \left[\frac{d\boldsymbol{x}_t}{dt} - t\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t) - \boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t) - t\frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t)\right] \\[10pt]
=&\, t\nabla_{\boldsymbol{\theta}}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t)\cdot \left[\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t) + t\left[\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t)\right]- \frac{d\boldsymbol{x}_t}{dt}\right] \\[10pt]
=&\, \frac{t}{2}\nabla_{\boldsymbol{\theta}}\left\Vert\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t) + t\sg{\frac{d\boldsymbol{x}_t}{dt}\cdot\frac{\partial}{\partial \boldsymbol{x}_t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t) + \frac{\partial}{\partial t}\boldsymbol{u}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, 0, t)}- \frac{d\boldsymbol{x}_t}{dt}\right\Vert^2 \\[10pt]
\sim &\, \left.\nabla_{\boldsymbol{\theta}}\eqref{eq:loss-3}\right|_{r=0}
\end{aligned}\end{equation}

所以，从这个角度看，sCM是MeanFlow在$r=0$时的一个特例。正如前面所说，引入另外的时间参数$r$可以让ReFlow给MeanFlow“兜底”（$r=t$时），从而更好地避免训崩，这是它的优点之一。当然，从sCM出发其实也可以引入双时间参数，得到跟第三目标完全相同的结果，但从个人的审美来看，CM、sCM的物理意义终究不如MeanFlow平均速度的诠释直观。

此外，平均速度和ReFlow结合的出发点，还可以得到另外的第一目标$\eqref{eq:loss-1}$和第二目标$\eqref{eq:loss-2}$，这对于像笔者这样的stop\_gradient洁癖患者来说是非常舒适和漂亮的结果。在笔者看来，从计算成本出发，我们是可以考虑给损失函数加上stop\_gradient，但推导的第一性原理和基本结果不应该跟stop\_gradient耦合，否则意味着它跟优化器和动力学是强耦合的，这并不是一个本质结果应有的表现。

## 文章小结

本文以最近出来的MeanFlow为中心，讨论了“平均速度”视角下的扩散模型加速生成思路。
