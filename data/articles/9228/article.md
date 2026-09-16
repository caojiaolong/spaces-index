# 生成扩散模型漫谈（六）：一般框架之ODE篇

> 作者：苏剑林 · 科学空间 · 2022-08-08
>
> 原文：<https://spaces.ac.cn/archives/9228>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章[《生成扩散模型漫谈（五）：一般框架之SDE篇》](<https://spaces.ac.cn/archives/9209>)中，我们对宋飏博士的论文[《Score\-Based Generative Modeling through Stochastic Differential Equations》](<https://papers.cool/arxiv/2011.13456>)做了基本的介绍和推导。然而，顾名思义，上一篇文章主要涉及的是原论文中SDE相关的部分，而遗留了被称为“概率流ODE（Probability flow ODE）”的部分内容，所以本文对此做个补充分享。

事实上，遗留的这部分内容在原论文的正文中只占了一小节的篇幅，但我们需要新开一篇文章来介绍它，因为笔者想了很久后发现，该结果的推导还是没办法绕开Fokker\-Planck方程，所以我们需要一定的篇幅来介绍Fokker\-Planck方程，然后才能请主角ODE登场。

## 再次反思

我们来大致总结一下上一篇文章的内容：首先，我们通过SDE来定义了一个前向过程（“拆楼”）：
\begin{equation}d\boldsymbol{x} = \boldsymbol{f}_t(\boldsymbol{x}) dt + g_t d\boldsymbol{w}\label{eq:sde-forward}\end{equation}
然后，我们推导了相应的逆向过程的SDE（“建楼”）：
\begin{equation}d\boldsymbol{x} = \left[\boldsymbol{f}_t(\boldsymbol{x}) - g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x}) \right] dt + g_t d\boldsymbol{w}\label{eq:sde-reverse}\end{equation}
最后，我们推导了用神经网络$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$来估计$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$的损失函数（得分匹配）：
\begin{equation}\mathbb{E}_{\boldsymbol{x}_0,\boldsymbol{x}_t \sim p(\boldsymbol{x}_t|\boldsymbol{x}_0)\tilde{p}(\boldsymbol{x}_0)}\left[\left\Vert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{x}_t|\boldsymbol{x}_0)\right\Vert^2\right] \end{equation}
至此，我们完成了扩散模型的训练、预测的一般框架，可以说，它是DDPM的非常一般化的推广了。但正如[《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](<https://spaces.ac.cn/archives/9181>)中介绍的DDIM是DDPM的高观点反思结果，SDE作为DDPM的推广，有没有相应的“高观点反思结果”呢？有，其结果就是本文主题“概率流ODE”。

## Dirac函数

DDIM做了什么反思呢？很简单，DDIM发现DDPM的训练目标主要跟$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$有关，而跟$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1})$无关，所以它以$p(\boldsymbol{x}_t|\boldsymbol{x}_0)$为出发点，去推导更一般的$p(\boldsymbol{x}_{t-1}|\boldsymbol{x}_t,\boldsymbol{x}_0)$和$p(\boldsymbol{x}_t|\boldsymbol{x}_{t-1},\boldsymbol{x}_0)$。概率流ODE做的反思是类似的，它想知道在SDE框架中，对于固定的$p(\boldsymbol{x}_t)$，能找出哪些不同的$p(\boldsymbol{x}_{t+\Delta t}|\boldsymbol{x}_t)$（或者说找到不同的前向过程SDE）。

我们先写出前向过程$\eqref{eq:sde-forward}$的离散形式
\begin{equation}\boldsymbol{x}_{t+\Delta t} = \boldsymbol{x}_t + \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})\label{eq:sde-discrete}\end{equation}
这个等式描述的是随机变量$\boldsymbol{x}_{t+\Delta t},\boldsymbol{x}_t,\boldsymbol{\varepsilon}$之间的关系，我们可以方便地对两边求期望，然而我们并非想求期望，而是想求分布$p(\boldsymbol{y}_t)$（所满足的关系式）。怎么将分布转换成期望形式呢？答案是[Dirac函数](<https://spaces.ac.cn/archives/1870>)：
\begin{equation}p(\boldsymbol{x}) = \int \delta(\boldsymbol{x} - \boldsymbol{y}) p(\boldsymbol{y}) d\boldsymbol{y} = \mathbb{E}_{\boldsymbol{y}}[\delta(\boldsymbol{x} - \boldsymbol{y})]\end{equation}
Dirac函数严格定义是属于泛函分析的内容，但我们通常都是当它是普通函数来处理，一般都能得到正确的结果。由上式还可以得知，对于任意$f(\boldsymbol{x})$，成立
\begin{equation}p(\boldsymbol{x})f(\boldsymbol{x}) = \int \delta(\boldsymbol{x} - \boldsymbol{y}) p(\boldsymbol{y})f(\boldsymbol{y}) d\boldsymbol{y} = \mathbb{E}_{\boldsymbol{y}}[\delta(\boldsymbol{x} - \boldsymbol{y}) f(\boldsymbol{y})]\end{equation}
直接对上式两边求偏导数，得到
\begin{equation}\nabla_{\boldsymbol{x}}[p(\boldsymbol{x}) f(\boldsymbol{x})] = \mathbb{E}_{\boldsymbol{y}}\left[\nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{y}) f(\boldsymbol{y})\right] = \mathbb{E}_{\boldsymbol{y}}\left[f(\boldsymbol{y})\nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{y})\right]\end{equation}
这是后面要用到的性质之一，可以发现它本质上是狄拉克函数的导数能够通过积分转移到所乘函数上去。

## F\-P方程

经过上述铺垫，现在我们根据式$\eqref{eq:sde-discrete}$写出
\begin{equation}\begin{aligned}
&\,\delta(\boldsymbol{x} - \boldsymbol{x}_{t+\Delta t}) \\[5pt]
=&\, \delta(\boldsymbol{x} - \boldsymbol{x}_t - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t - g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}) \\
\approx&\, \delta(\boldsymbol{x} - \boldsymbol{x}_t) - \left(\boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}\right)\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t) + \frac{1}{2} \left(g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}\cdot \nabla_{\boldsymbol{x}}\right)^2\delta(\boldsymbol{x} - \boldsymbol{x}_t)
\end{aligned}\end{equation}
这里当$\delta(\cdot)$是普通函数那样做了泰勒展开，只保留了不超过$\mathcal{O}(\Delta t)$的项。现在我们两边求期望：
\begin{equation}\begin{aligned}
&\,p_{t+\Delta t}(\boldsymbol{x}) \\[6pt]
=&\,\mathbb{E}_{\boldsymbol{x}_{t+\Delta t}}\left[\delta(\boldsymbol{x} - \boldsymbol{x}_{t+\Delta t})\right] \\
\approx&\, \mathbb{E}_{\boldsymbol{x}_t, \boldsymbol{\varepsilon}}\left[\delta(\boldsymbol{x} - \boldsymbol{x}_t) - \left(\boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t + g_t \sqrt{\Delta t}\boldsymbol{\varepsilon}\right)\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t) + \frac{1}{2} \left(g_t\sqrt{\Delta t}\boldsymbol{\varepsilon}\cdot \nabla_{\boldsymbol{x}}\right)^2\delta(\boldsymbol{x} - \boldsymbol{x}_t)\right] \\
=&\, \mathbb{E}_{\boldsymbol{x}_t}\left[\delta(\boldsymbol{x} - \boldsymbol{x}_t) - \boldsymbol{f}_t(\boldsymbol{x}_t) \Delta t\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t) + \frac{1}{2} g_t^2\Delta t \nabla_{\boldsymbol{x}}\cdot \nabla_{\boldsymbol{x}}\delta(\boldsymbol{x} - \boldsymbol{x}_t)\right] \\
=&\,p_t(\boldsymbol{x}) - \nabla_{\boldsymbol{x}}\cdot\left[\boldsymbol{f}_t(\boldsymbol{x})\Delta t\, p_t(\boldsymbol{x})\right] + \frac{1}{2}g_t^2\Delta t \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})
\end{aligned}\end{equation}
两边除以$\Delta t$，并取$\Delta t\to 0$的极限，结果是
\begin{equation}\frac{\partial}{\partial t} p_t(\boldsymbol{x}) = - \nabla_{\boldsymbol{x}}\cdot\left[\boldsymbol{f}_t(\boldsymbol{x}) p_t(\boldsymbol{x})\right] + \frac{1}{2}g_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\label{eq:fp}
\end{equation}
这就是式$\eqref{eq:sde-forward}$所对应的“F\-P方程”（[Fokker\-Planck方程](<https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation>)），它是描述边际分布的偏微分方程。

## 等价变换

大家看到偏微分方程不用担心，因为这里并没有打算去研究怎么求解偏微分方程，只是借助它来引导一个等价变换而已。对于任意满足$\sigma_t^2\leq g_t^2$的函数$\sigma_t$，F\-P方程$\eqref{eq:fp}$完全等价于
\begin{equation}\begin{aligned}
\frac{\partial}{\partial t} p_t(\boldsymbol{x}) =&\, - \nabla_{\boldsymbol{x}}\cdot\left[\boldsymbol{f}_t(\boldsymbol{x})p_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})\right] + \frac{1}{2}\sigma_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x}) \\
=&\,- \nabla_{\boldsymbol{x}}\cdot\left[\left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right)p_t(\boldsymbol{x})\right] + \frac{1}{2}\sigma_t^2 \nabla_{\boldsymbol{x}}\cdot\nabla_{\boldsymbol{x}}p_t(\boldsymbol{x})
\end{aligned}\label{eq:fp-2}\end{equation}
形式上该F\-P方程又相当于原来的F\-P的$\boldsymbol{f}_t(\boldsymbol{x})$换成了$\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$、$g_t$换成了$\sigma_t$，根据式$\eqref{eq:fp}$对应于式$\eqref{eq:sde-forward}$，上式则对应于
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 - \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}\label{eq:sde-forward-2}\end{equation}
但是别忘了式$\eqref{eq:fp}$跟式$\eqref{eq:fp-2}$是完全等价的，所以这意味着式$\eqref{eq:sde-forward}$和式$\eqref{eq:sde-forward-2}$这两个随机微分方程所对应的边际分布$p_t(\boldsymbol{x})$是完全等价的！这个结果告诉我们存在不同方差的前向过程，它们产生的边际分布是一样的。这个结果相当于DDIM的升级版，后面我们还会证明，当$\boldsymbol{f}_t(\boldsymbol{x})$是关于$\boldsymbol{x}$的线性函数时，它就完全等价于DDIM。

特别地，根据上一篇SDE的结果，我们可以写出式$\eqref{eq:sde-forward-2}$对应的反向SDE：
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}(g_t^2 + \sigma_t^2)\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt + \sigma_t d\boldsymbol{w}\label{eq:sde-reverse-2}\end{equation}

## 神经ODE

式$\eqref{eq:sde-forward-2}$允许我们改变采样过程的方差，这里我们特别考虑$\sigma_t = 0$的极端情形，此时SDE退化为ODE（常微分方程）：
\begin{equation}d\boldsymbol{x} = \left(\boldsymbol{f}_t(\boldsymbol{x}) - \frac{1}{2}g_t^2\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})\right) dt\label{eq:flow-ode}\end{equation}
这个ODE称为“概率流ODE（Probability flow ODE）”，由于实践中的$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$需要用神经网络$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$近似，所以上式也对应一个“神经ODE”。

为什么要特别研究方差为0的情形呢？因为此时传播过程不带噪声，从$\boldsymbol{x}_0$到$\boldsymbol{x}_T$是一个确定性变换，所以我们直接反向求解ODE就能得到由$\boldsymbol{x}_T$变换为$\boldsymbol{x}_0$的逆变换，这也是一个确定性变换（直接在式$\eqref{eq:sde-reverse-2}$中代入$\sigma_t=0$也可以发现前向和反向的方程是一样的）。这个过程和[flow模型](<https://spaces.ac.cn/tag/flow/>)是一致的（即通过一个可逆的变换将噪声变换成样本），所以概率流ODE允许我们将扩散模型的结果与flow模型相关结果对应起来，比如原论文提到概率流ODE允许我们做精确的似然计算、获得隐变量表征等，这些本质上都是flow模型的好处。由于flow模型的可逆性，它还允许我们在隐变量空间对原图做一些图片编辑等。

另一方面，从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的变换由一个ODE描述，这意味着我们可以通过各种高阶的ODE数值算法来加速从$\boldsymbol{x}_T$到$\boldsymbol{x}_0$的变换过程。当然，原则上SDE的求解也有一些加速方法，但SDE的加速研究远远不如ODE的容易和深入。总的来说，相比SDE，ODE在理论分析和实际求解中都显得更为简单直接。

## 回顾DDIM

在[《生成扩散模型漫谈（四）：DDIM = 高观点DDPM》](<https://spaces.ac.cn/archives/9181>)的最后，我们推导了DDIM的连续版本对应于ODE
\begin{equation}\frac{d}{ds}\left(\frac{\boldsymbol{x}(s)}{\bar{\alpha}(s)}\right) = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}\left(\boldsymbol{x}(s), t(s)\right)\frac{d}{ds}\left(\frac{\bar{\beta}(s)}{\bar{\alpha}(s)}\right)\label{eq:ddim-ode}\end{equation}
接下来我们可以看到，该结果其实就是本文的式$\eqref{eq:flow-ode}$在$\boldsymbol{f}_t(\boldsymbol{x})$取线性函数$f_t \boldsymbol{x}$时的特例：在[《生成扩散模型漫谈（五）：一般框架之SDE篇》](<https://spaces.ac.cn/archives/9209>)的末尾，我们推导过对应的关系
\begin{equation}\left\{\begin{aligned}
&f_t = \frac{1}{\bar{\alpha}_t}\frac{d\bar{\alpha}_t}{dt} \\
&g^2 (t) = 2\bar{\alpha}_t \bar{\beta}_t \frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right) \\
&\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t) = -\frac{\boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)}{\bar{\beta}_t}
\end{aligned}\right.\end{equation}
将这些关系代入到式$\eqref{eq:flow-ode}$【$\nabla_{\boldsymbol{x}}\log p_t(\boldsymbol{x})$替换为$\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t)$】后，整理得到
\begin{equation}\frac{1}{\bar{\alpha}_t}\frac{d\boldsymbol{x}}{dt} - \frac{\boldsymbol{x}}{\bar{\alpha}_t^2}\frac{d\bar{\alpha}_t}{dt} = \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\frac{d}{dt}\left(\frac{\bar{\beta}_t}{\bar{\alpha}_t}\right)\end{equation}
左端可以进一步整理得到$\frac{d}{dt}\left(\frac{\boldsymbol{x}}{\bar{\alpha}_t}\right)$，因此上式跟式$\eqref{eq:ddim-ode}$完全等价。

## 文章小结

本文在SDE篇的基础上，借助F\-P方程推导了更一般化的前向方程，继而推导出了“概率流ODE”，并证明了DDIM是它的一个特例。
