# 生成扩散模型漫谈（二十八）：分步理解一致性模型

> 作者：苏剑林 · 科学空间 · 2024-12-18
>
> 原文：<https://spaces.ac.cn/archives/10633>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

书接上文，在[《生成扩散模型漫谈（二十七）：将步长作为条件输入》](<https://spaces.ac.cn/archives/10617>)中，我们介绍了加速采样的Shortcut模型，其对比的模型之一就是“[一致性模型（Consistency Models）](<https://papers.cool/arxiv/2303.01469>)”。事实上，早在[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](<https://spaces.ac.cn/archives/9497>)介绍ReFlow时，就有读者提到了一致性模型，但笔者总感觉它更像是实践上的Trick，理论方面略显单薄，所以兴趣寥寥。

不过，既然我们开始关注扩散模型加速采样方面的进展，那么一致性模型就是一个绕不开的工作。因此，趁着这个机会，笔者在这里分享一下自己对一致性模型的理解。

## 熟悉配方

还是熟悉的配方，我们的出发点依旧是[ReFlow](<https://spaces.ac.cn/archives/9497>)，因为它大概是ODE式扩散最简单的理解方式。设$\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)$是目标分布的<u>真实样本</u>，$\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)$是先验分布的<u>随机噪声</u>，$\boldsymbol{x}_t =  (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$是加噪样本，那么ReFlow的训练目标是：
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{t\sim U[0,1],\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\left[w(t)\Vert\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right]\label{eq:loss}\end{equation}
其中$w(t)$是可调的权重。训练完成后可以通过求解$d\boldsymbol{x}_t/dt = \boldsymbol{v}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, t)$来实现$\boldsymbol{x}_1$到$\boldsymbol{x}_0$的变换，从而完成采样。

需要指出的是，一致性模型的Noise Schedule是$\boldsymbol{x}_t =  \boldsymbol{x}_0 + t\boldsymbol{x}_1$（当$t$足够大时$\boldsymbol{x}_t$同样接近于纯噪声），跟ReFlow略有不同。不过本文的主要目的，是尝试一步步引导出跟一致性模型相同的训练思想和训练目标，笔者认为ReFlow的更好理解一些，所以还是按照ReFlow的来介绍，至于具体的训练细节大家按需自行调整就好。

利用$\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$，我们可以消去目标$\eqref{eq:loss}$中的$\boldsymbol{x}_1$：
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{t\sim U[0,1],\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t)\Vert \underbrace{\boldsymbol{x}_t - t\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}_{\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)} - \boldsymbol{x}_0\Vert^2\big]\label{eq:loss-2}\end{equation}
其中$\tilde{w}(t) = w(t)/t^2$。注意$\boldsymbol{x}_0$是真实样本，$\boldsymbol{x}_t$是加噪样本，所以ReFlow的训练目标实际上也是在去噪。预测干净样本的模型为$\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)=\boldsymbol{x}_t - t\boldsymbol{v}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)$，这个函数有一个重要特性是恒成立$\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_0, 0)=\boldsymbol{x}_0$，这正是一致性模型的关键约束之一。

## 分步理解

接下来让我们一步步解构ReFlow的训练过程，试图从中找到更好的训练目标。首先我们将$[0,1]$等分为$n$份，每份大小为$1/n$，记$t_k = k/n$，那么$t$就只需从有限集合$\{0,t_1,t_2,\cdots,t_n\}$均匀采样。当然我们也可以选择非均匀的离散化方式，这些都是非关键的细节问题。

由于$t_0=0$是平凡的，我们从$t_1$开始，第一步的训练目标是
\begin{equation}\boldsymbol{\theta}_1^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_1)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_1}, t_1) - \boldsymbol{x}_0\Vert^2\big]\end{equation}
接着，考虑第二步的训练目标，还是按照$\eqref{eq:loss-2}$的话，那么应该是$\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_2}, t_2) - \boldsymbol{x}_0\Vert^2$的期望，但现在我们评估一个新目标：
\begin{equation}\boldsymbol{\theta}_2^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_2)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_2}, t_2) - \boldsymbol{f}_{\boldsymbol{\theta}_1^*}(\boldsymbol{x}_{t_1}, t_1)\Vert^2\big]\end{equation}
也就是说预测对象改为$\boldsymbol{f}_{\boldsymbol{\theta}_1^*}(\boldsymbol{x}_{t_1}, t_1)$而不是$\boldsymbol{x}_0$。为什么要这样改呢？我们分可行性和必要性两方面来讨论。可行性方面，$\boldsymbol{x}_{t_2}$相比$\boldsymbol{x}_{t_1}$加了更多噪声，所以它去噪会更困难，换言之$\boldsymbol{f}_{\boldsymbol{\theta}_2^*}(\boldsymbol{x}_{t_2}, t_2)$的复原程度是不如$\boldsymbol{f}_{\boldsymbol{\theta}_1^*}(\boldsymbol{x}_{t_1}, t_1)$的，所以用$\boldsymbol{f}_{\boldsymbol{\theta}_1^*}(\boldsymbol{x}_{t_1}, t_1)$替换掉$\boldsymbol{x}_0$作为第二步的训练目标完全是可行的。

可即便如此，那又有什么换的必要呢？答案是减少“轨迹交叉”。由于$\boldsymbol{x}_{t_k} = (1-t_k)\boldsymbol{x}_0 + t_k\boldsymbol{x}_1$，因此随着$k$的增大，$\boldsymbol{x}_{t_k}$对$\boldsymbol{x}_0$的依赖会越来越弱，以至于两个不同的$\boldsymbol{x}_0$，它们对应的$\boldsymbol{x}_{t_k}$会很接近，这时候还是以$\boldsymbol{x}_0$为预测目标的话，就会出现“一个输入，多个目标”的困境，这就是“轨迹交叉”。

面对这个困境，ReFlow的策略是事后蒸馏，因为预训练完后求解$d\boldsymbol{x}_t/dt = \boldsymbol{v}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, t)$就可以得到很多$(\boldsymbol{x}_0,\boldsymbol{x}_1)$对，用这些配对的$\boldsymbol{x}_0,\boldsymbol{x}_1$去构建$\boldsymbol{x}_t$就能避免交叉。一致性模型的想法是把预测目标换成$\boldsymbol{f}_{\boldsymbol{\theta}_{k-1}^*}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})$，因为对于“同一$\boldsymbol{x}_1$、不同$\boldsymbol{x}_0$”，$\boldsymbol{f}_{\boldsymbol{\theta}_{k-1}^*}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})$间的差异会比$\boldsymbol{x}_0$间的差异要小，所以也能减少交叉风险。

简单来说，就是$\boldsymbol{f}_{\boldsymbol{\theta}_2^*}(\boldsymbol{x}_{t_2}, t_2)$预测$\boldsymbol{f}_{\boldsymbol{\theta}_1^*}(\boldsymbol{x}_{t_1}, t_1)$比预测$\boldsymbol{x}_0$更容易，并且该达到的效果也能达到，所以调整了预测目标。类似地，我们可以写出
\begin{equation}\begin{gathered}
\boldsymbol{\theta}_3^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_3)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_3}, t_3) - \boldsymbol{f}_{\boldsymbol{\theta}_2^*}(\boldsymbol{x}_{t_2}, t_2)\Vert^2\big] \\
\boldsymbol{\theta}_4^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_4)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_4}, t_4) - \boldsymbol{f}_{\boldsymbol{\theta}_3^*}(\boldsymbol{x}_{t_3}, t_3)\Vert^2\big] \\
\vdots \\[5pt]
\boldsymbol{\theta}_n^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_n)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_n}, t_n) - \boldsymbol{f}_{\boldsymbol{\theta}_{n-1}^*}(\boldsymbol{x}_{t_{n-1}}, t_{n-1})\Vert^2\big]
\end{gathered}\end{equation}

## 一致训练

现在我们已经完成了ReFlow模型的解构，并且得到了一个新的自认为更合理的训练目标，但代价是得到了$n$套参数$\boldsymbol{\theta}_1^*,\boldsymbol{\theta}_2^*,\cdots,\boldsymbol{\theta}_n^*$，这当然不是我们想要的，我们只想要一个模型。于是我们认为所有的$\boldsymbol{\theta}_i^*$可以共用同一套参数，于是我们可以写出训练目标
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{k\sim[n],\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_k)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_k}, t_k) - \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})\Vert^2\big]\label{eq:loss-3}\end{equation}
这里$k\sim[n]$是指$k$从$\{1,2,\cdots,n\}$中均匀采样。上式的问题是，$\boldsymbol{\theta}^*$是我们要求的参数，但它又出现在目标函数中，这显然是不科学的（知道$\boldsymbol{\theta}^*$了我还训练干嘛），因此必须修改上述目标使得它更为可行。

$\boldsymbol{\theta}^*$的意义是理论最优解，考虑到随着训练的推进，$\boldsymbol{\theta}$会慢慢逼近$\boldsymbol{\theta}^*$，所以在目标函数中我们可以将这个条件放宽为“超前解”，即它只要比当前的$\boldsymbol{\theta}$更好就行了。怎么构建“超前解”呢？一致性模型的做法是对历史权重进行[EMA（Exponential Moving Average，指数滑动平均](<https://spaces.ac.cn/archives/6575#%E6%9D%83%E9%87%8D%E6%BB%91%E5%8A%A8%E5%B9%B3%E5%9D%87>)），这往往能得到一个更优秀的解，早些年我们在打比赛时就经常用到这个技巧。

因此，一致性模型最终的训练目标是：
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{k\sim[n],\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_k)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_k}, t_k) - \boldsymbol{f}_{\bar{\boldsymbol{\theta}}}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})\Vert^2\big]\label{eq:loss-4}\end{equation}
其中$\bar{\boldsymbol{\theta}}$是$\boldsymbol{\theta}$的EMA。这就是原论文中的“一致性训练（Consistency Training，CT）”。从实践上来看，我们也可以将$\Vert\cdot - \cdot\Vert^2$换成更一般的度量$d(\cdot, \cdot)$，以更贴合数据特性。

## 采样分析

由于我们是从ReFlow出发一步步“等价变换”过来的，所以训练完成后一种基本的采样方式就是跟ReFlow一样求解ODE
\begin{equation}d\boldsymbol{x}_t/dt = \boldsymbol{v}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, t) = \frac{\boldsymbol{x}_t - \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, t)}{t}\label{eq:ode}\end{equation}
当然，如果费那么大劲得到的是跟ReFlow一样的结果，那么就纯粹是瞎折腾了。幸运的是，一致性训练所得的模型，有一个重要的优势是可以使用更大的采样步长——甚至等于1的步长，这就可以实现单步生成：
\begin{equation}\boldsymbol{x}_0 = \boldsymbol{x}_1 - \boldsymbol{v}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1)\times 1 = \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1)\end{equation}
理由是
\begin{equation}\begin{aligned}
\Vert\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1) - \boldsymbol{x}_0\Vert =&\, \left\Vert\sum_{k=1}^n \Big[\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_k}, t_k) - \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})\Big]\right\Vert \\[5pt]
\leq&\, \sum_{k=1}^n \Vert\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_k}, t_k) - \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})\Vert \\
\end{aligned}\label{eq:f-x1-x0}\end{equation}
可以看到，一致性训练相当于在优化$\Vert\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1) - \boldsymbol{x}_0\Vert$的上界，当损失足够小时，意味着$\Vert\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1) - \boldsymbol{x}_0\Vert$也足够小，因此可以一步生成。

可$\Vert\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1) - \boldsymbol{x}_0\Vert$是原本ReFlow的训练目标，为什么直接优化它会不如优化它的上界呢？这又回到了“轨迹交叉”的问题了，直接训练的话，$\boldsymbol{x}_0,\boldsymbol{x}_1$都是随机采样的，没有一一配对关系，所以无法直接训练出一步生成模型。但训练上界的话，通过多个$\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_k}, t_k),\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})$的传递性，隐含地实现了$\boldsymbol{x}_0,\boldsymbol{x}_1$的配对。

如果单步生成的效果不能让我们满意，我们也可以增加采样步数来提高生成质量，这里边又有两种思路：1、用更小的步长来数值求解$\eqref{eq:ode}$；2、转化为类似SDE的随机迭代。前者比较常规，我们主要讨论后者。

首先注意到式$\eqref{eq:f-x1-x0}$中的$\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1)$换成任意$\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, t)$，也可以得到类似的不等关系，这意味着任意的$\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_t, t)$预测的都是$\boldsymbol{x}_0$，这样一来，我们从$\boldsymbol{x}_1$出发，通过$\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1)$就得到一个初步的$\boldsymbol{x}_0$，但可能不够完美，于是我们通过加噪来“掩饰”这种不完美，得到一个$\boldsymbol{x}_{t_{n-1}}$，代入$\boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_{n-1}}, t_{n-1})$得到一个更好一点的结果，依此类推：
\begin{equation}\begin{aligned}
&\boldsymbol{x}_1\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}) \\
&\boldsymbol{x}_0\leftarrow \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_1, 1) \\
&\text{for }k=n-1,n-2,\cdots,1: \\
&\qquad \boldsymbol{z} \sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I}) \\
&\qquad \boldsymbol{x}_{t_k} \leftarrow (1 - t_k)\boldsymbol{x}_0 + t_k\boldsymbol{z} \\
&\qquad \boldsymbol{x}_0\leftarrow \boldsymbol{f}_{\boldsymbol{\theta}^*}(\boldsymbol{x}_{t_k}, t_k)
\end{aligned}\end{equation}

## 用于蒸馏

一致性模型的训练思想同样可以用于现成扩散模型的蒸馏，结果称为“一致性蒸馏（Consistency Distillation，CD）”，方法是把式$\eqref{eq:loss-4}$中$\boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_k}, t_k)$的学习目标由$\boldsymbol{f}_{\bar{\boldsymbol{\theta}}}(\boldsymbol{x}_{t_{k-1}}, t_{k-1})$换成$\boldsymbol{f}_{\bar{\boldsymbol{\theta}}}(\hat{\boldsymbol{x}}_{t_{k-1}}^{\boldsymbol{\varphi}^*}, t_{k-1})$：
\begin{equation}\boldsymbol{\theta}^* = \mathop{\text{argmin}}_{\boldsymbol{\theta}} \mathbb{E}_{k\sim[n],\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\big[\tilde{w}(t_k)\Vert \boldsymbol{f}_{\boldsymbol{\theta}}(\boldsymbol{x}_{t_k}, t_k) - \boldsymbol{f}_{\bar{\boldsymbol{\theta}}}(\hat{\boldsymbol{x}}_{t_{k-1}}^{\boldsymbol{\varphi}^*}, t_{k-1})\Vert^2\big]\label{eq:loss-5}\end{equation}
其中$\hat{\boldsymbol{x}}_{t_{k-1}}^{\boldsymbol{\varphi}^*}$是由教师扩散模型以$\boldsymbol{x}_{t_k}$为初值所预测的$\boldsymbol{x}_{t_{k-1}}$，比如最简单的欧拉求解器，我们有
\begin{equation}\hat{\boldsymbol{x}}_{t_{k-1}}^{\boldsymbol{\varphi}^*} \approx \boldsymbol{x}_{t_k} - (t_k - t_{k-1})\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{x}_{t_k}, t_k)\end{equation}
这样做的理由也很简单，如果有了预训练好的扩散模型，那么我们就没必要在直线$\boldsymbol{x}_t =  (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_1$上找学习目标了，因为这是人为定义的，终究有交叉的风险，而是改为由预训练好扩散模型来预测轨迹，这样找出来的学习目标可能并不一定是“最直”的，但肯定不会有交叉。

如果不计成本，我们也可以从随机采样的$\boldsymbol{x}_1$出发，加上预训练扩散模型解出的$\boldsymbol{x}_0$，通过成对的$(\boldsymbol{x}_0,\boldsymbol{x}_1)$来构建学习目标，这差不多就是ReFlow的蒸馏思路，缺点是必须对教师模型运行完整的采样过程，费时费力。相比之下，一致性蒸馏只需要运行单步教师模型，计算成本更低。

不过，一致性蒸馏在蒸馏过程中还需要真实样本，这在某些场景下也是一个缺点。如果蒸馏过程既不想运行完整的教师模型采样，又不想提供真实数据，那么有一个选择就是我们之前介绍过的[SiD](<https://spaces.ac.cn/archives/10085>)，代价是模型的推导更加复杂了。

## 文章小结

本文通过逐步解构和优化ReFLow训练流程的方式，提供了一个从ReFlow逐渐过渡到一致性模型（Consistency Models）的直观理解路径。
