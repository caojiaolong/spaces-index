# 生成扩散模型漫谈（二十）：从ReFlow到WGAN\-GP

> 作者：苏剑林 · 科学空间 · 2023-06-28
>
> 原文：<https://spaces.ac.cn/archives/9668>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章[《生成扩散模型漫谈（十九）：作为扩散ODE的GAN》](<https://spaces.ac.cn/archives/9662>)中，我们介绍了如何将GAN理解为在另一个时间维度上的扩散ODE，简而言之，GAN实际上就是将扩散模型中样本的运动转化为生成器参数的运动！然而，该文章的推导过程依赖于Wasserstein梯度流等相对复杂和独立的内容，没法很好地跟扩散系列前面的文章连接起来，技术上显得有些“断层”。

在笔者看来，[《生成扩散模型漫谈（十七）：构建ODE的一般步骤（下）》](<https://spaces.ac.cn/archives/9497>)所介绍的ReFlow是理解扩散ODE的最直观方案，既然可以从扩散ODE的角度理解GAN，那么必定存在一个从ReFlow理解GAN的角度。经过一番尝试，笔者成功从ReFlow推出了类似WGAN\-GP的结果。

## 理论回顾

之所以说“ReFlow是理解扩散ODE的最直观方案”，是因为它本身非常灵活，以及非常贴近实验代码——它能够通过ODE建立任意噪声分布到目标数据分布的映射，而且训练目标非常直观，不需要什么“弯弯绕绕”就可以直接跟实验代码对应起来。

具体来说，假设$\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)$是先验分布采样的随机噪声，$\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)$是目标分布采样的真实样本（注：前面的文章中，普遍都是$\boldsymbol{x}_T$是噪声、$\boldsymbol{x}_0$是目标样本，这里方便起见反过来了），ReFlow允许我们指定任意从$\boldsymbol{x}_0$到$\boldsymbol{x}_1$的运动轨迹。简单起见，ReFlow选择的是直线，即
\begin{equation}\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t \boldsymbol{x}_1\label{eq:line}\end{equation}
现在我们求出它满足的ODE：
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0\end{equation}
这个ODE很简单，但是却不实用，因为我们想要的是通过ODE由$\boldsymbol{x}_0$生成$\boldsymbol{x}_1$，但上述ODE却将我们要生成的目标放在了方程里边，可谓是“因果倒置”了。为了弥补这个缺陷，ReFlow的思路很简单：学一个$\boldsymbol{x}_t$的函数去逼近$\boldsymbol{x}_1 - \boldsymbol{x}_0$，学完之后就用它来取代$\boldsymbol{x}_1 - \boldsymbol{x}_0$，即
\begin{equation}\boldsymbol{\varphi}^* = \mathop{\text{argmin}}_{\boldsymbol{\varphi}} \mathbb{E}_{\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0),\boldsymbol{x}_1\sim p_1(\boldsymbol{x}_1)}\left[\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t) - (\boldsymbol{x}_1 - \boldsymbol{x}_0)\Vert^2\right]\label{eq:s-loss}\end{equation}
以及
\begin{equation}\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{x}_1 - \boldsymbol{x}_0\quad\Rightarrow\quad\frac{d\boldsymbol{x}_t}{dt} = \boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{x}_t, t)\label{eq:ode-core}\end{equation}
之前我们已经证明过，在$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t)$具有无限拟合能力的假设下，新的ODE确实能够实现从分布$p_0(\boldsymbol{x}_0)$到分布$p_1(\boldsymbol{x}_1)$的样本变换。

## 相对运动

ReFlow的重要特性之一，是它没有限制先验分布$p_0(\boldsymbol{x}_0)$的形式，这意味着我们可以将先验分布换成任意我们想要的分布，比如，由一个生成器$\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$变换而来的分布：
\begin{equation}\boldsymbol{x}_0\sim p_0(\boldsymbol{x}_0)\quad\Leftrightarrow\quad \boldsymbol{x}_0 = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}),\,\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})\end{equation}
代入式$\eqref{eq:s-loss}$训练完成后，我们就可以利用式$\eqref{eq:ode-core}$，将任意$\boldsymbol{x}_0 = \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z})$变换为真实样本$\boldsymbol{x}_1$了。

然而，我们并不满足于此。前面说过，GAN是将扩散模型中样本的运动转化为生成器参数的运动，这个ReFlow的框架中同样可以如此：假设生成器当前参数为$\boldsymbol{\theta}_{\tau}$，我们期望$\boldsymbol{\theta}_{\tau}\to \boldsymbol{\theta}_{\tau+1}$的变化能模拟式$\eqref{eq:ode-core}$前进一小步的效果
\begin{equation}\boldsymbol{\theta}_{\tau+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}\Big[\big\Vert \boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}) - \boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}) - \epsilon\,\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z}), 0)\big\Vert^2\Big]\label{eq:g-loss}\end{equation}
要注意，式$\eqref{eq:s-loss}$和式$\eqref{eq:ode-core}$中的$t$跟参数$\boldsymbol{\theta}_{\tau}$中的$\tau$不是同一含义，前者是ODE的时间参数，后者是训练进度，所以这里用了不同记号。此外，$\boldsymbol{g}_{\boldsymbol{\theta}_{\tau}}(\boldsymbol{z})$是作为ODE的$\boldsymbol{x}_0$出现的，所以往前推一小步时，得到的是$\boldsymbol{x}_{\epsilon}$，$\boldsymbol{v}_{\boldsymbol{\varphi}^*}(\boldsymbol{x}_t, t)$中要代入的时间$t$是$0$。

现在，我们有了新的$\boldsymbol{g}_{\boldsymbol{\theta}_{\tau+1}}(\boldsymbol{z})$，理论上它产生的分布更加接近真实分布一些（因为往前推了一小步），接着把它当作新的$\boldsymbol{x}_0$代入到式$\eqref{eq:s-loss}$训练，训练完成后又可以代入到式$\eqref{eq:g-loss}$优化生成器，以此类推，就是一个类似GAN的交替训练过程。

## WGAN\-GP

那么，能否将这个过程定量地跟已有的GAN联系起来呢？能！还是带梯度惩罚的[WGAN\-GP](<https://spaces.ac.cn/archives/4439>)。

首先我们来看损失函数$\eqref{eq:s-loss}$，将求期望的部分展开，结果是
\begin{equation}\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t)\Vert^2 - \langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t, t),\boldsymbol{x}_1 - \boldsymbol{x}_0\rangle + \frac{1}{2}\Vert\boldsymbol{x}_1 - \boldsymbol{x}_0\Vert^2\end{equation}
第三项跟参数$\boldsymbol{\varphi}$无关，去掉也不影响结果。现在我们假设$\boldsymbol{v}_{\boldsymbol{\varphi}}$有足够强的拟合能力，以至于我们不需要显式输入$t$，那么上式作为损失函数，等价于
\begin{equation}\frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t),\boldsymbol{x}_1 - \boldsymbol{x}_0\rangle = \frac{1}{2}\Vert\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \left\langle\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t),\frac{d\boldsymbol{x}_t}{dt}\right\rangle\end{equation}
$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$是一个输入输出维度相同的向量函数，我们进一步假设它是某个标量函数$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$的梯度，即$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)=\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$，那么上式就是
\begin{equation}\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \left\langle\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t),\frac{d\boldsymbol{x}_t}{dt}\right\rangle = \frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - \frac{d D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)}{dt}\end{equation}
假设$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$的变化比较平稳，那么$\frac{d D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)}{dt}$应该与它在$t=0,t=1$两点处的差分$D_{\boldsymbol{\varphi}}(\boldsymbol{x}_1)-D_{\boldsymbol{\varphi}}(\boldsymbol{x}_0)$比较接近，于是上述损失函数近似于
\begin{equation}\frac{1}{2}\Vert\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)\Vert^2 - D_{\boldsymbol{\varphi}}(\boldsymbol{x}_1) + D_{\boldsymbol{\varphi}}(\boldsymbol{x}_0)\end{equation}
熟悉GAN的读者应该会觉得很眼熟，它正是带梯度惩罚的WGAN的判别器损失函数！甚至连梯度惩罚项的$\boldsymbol{x}_t$的构造方式$\eqref{eq:line}$都一模一样（在真假样本之间线性插值）！唯一不同的是原始WGAN\-GP的梯度惩罚是以1为中心，这里是以零为中心，但事实上[《WGAN\-div：一个默默无闻的WGAN填坑者》](<https://spaces.ac.cn/archives/6139>)、[《从动力学角度看优化算法（四）：GAN的第三个阶段》](<https://spaces.ac.cn/archives/6583>)等文章已经表明以零为中心的梯度惩罚通常效果更好。

所以说，在特定的参数化和假设之下，损失函数$\eqref{eq:s-loss}$其实就等价于WGAN\-GP的判别器损失。至于生成器损失，在上一篇文章[《生成扩散模型漫谈（十九）：作为扩散ODE的GAN》](<https://spaces.ac.cn/archives/9662>)中我们已经证明了当$\boldsymbol{v}_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)=\nabla_{\boldsymbol{x}_t} D_{\boldsymbol{\varphi}}(\boldsymbol{x}_t)$时，式$\eqref{eq:g-loss}$单步优化的梯度等价于
\begin{equation}\boldsymbol{\theta}_{\tau+1} = \mathop{\text{argmin}}_{\boldsymbol{\theta}}\mathbb{E}_{\boldsymbol{z}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})}[-D(\boldsymbol{g}_{\boldsymbol{\theta}}(\boldsymbol{z}))]\end{equation}
的梯度，而这正好也是WGAN\-GP的生成器损失。

## 文章小结

在这篇文章中，笔者尝试从ReFlow出发推导了WGAN\-GP与扩散ODE之间的联系，这个角度相对来说更加简单直观，并且避免了Wasserstein梯度流等相对复杂的概念。
