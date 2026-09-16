# AdamW的Weight RMS的渐近估计（下）

> 作者：苏剑林 · 科学空间 · 2025-11-17
>
> 原文：<https://spaces.ac.cn/archives/11404>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在博客[《AdamW的Weight RMS的渐近估计（上）》](<https://spaces.ac.cn/archives/11307>)中，我们推导了AdamW训练出来的模型权重的RMS渐近表达式。不过，那会我们假设了Weight Decay和学习率在整个训练过程中是固定的，这跟实际训练并不完全吻合，所以这篇文章我们将之前的结论推广成动态版。

所谓动态版，即允许Weight Decay和学习率都随着训练步数的增加而变化，比如经典的Cosine Decay、WSD（Warmup Stable Decay）等，从而让结论更为通用。

## 步骤之一

我们的出发点还是AdamW的定义：
\begin{equation}\text{Adam}\color{skyblue}{\text{W}}:=\left\{\begin{aligned}
&\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + \left(1 - \beta_1\right) \boldsymbol{g}_t\\
&\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + \left(1 - \beta_2\right) \boldsymbol{g}_t^2\\
&\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t\left/\left(1 - \beta_1^t\right)\right.\\
&\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t\left/\left(1 - \beta_2^t\right)\right.\\
&\boldsymbol{u}_t =\hat{\boldsymbol{m}}_t\left/\left(\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon\right)\right.\\
&\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t (\boldsymbol{u}_t \color{skyblue}{ + \lambda_t \boldsymbol{\theta}_{t-1}})
\end{aligned}\right.\end{equation}
由于$\eta_t\lambda_t\ll 1$，我们可以写出
\begin{equation}\boldsymbol{\theta}_t = (1 - \eta_t\lambda_t)\boldsymbol{\theta}_{t-1} -\eta_t\boldsymbol{u}_t \approx e^{- \eta_t\lambda_t}\boldsymbol{\theta}_{t-1} -\eta_t\boldsymbol{u}_t\end{equation}
记$\kappa_t = \sum_{i=1}^t \eta_i\lambda_i$，那么直接展开得到
\begin{equation}\boldsymbol{\theta}_t \approx e^{-\kappa_t}\boldsymbol{\theta}_0 - \sum_{i=1}^t e^{-(\kappa_t - \kappa_i)}\eta_i\boldsymbol{u}_i  = e^{-\kappa_t}\left(\boldsymbol{\theta}_0 - \sum_{i=1}^t e^{\kappa_i}\eta_i\boldsymbol{u}_i\right)\end{equation}
然后设$z_t = \sum_{i=1}^t e^{\kappa_i}\eta_i$，那么由平均场近似得
\begin{equation}\bar{\boldsymbol{u}}_t\triangleq\frac{1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i \boldsymbol{u}_i = \frac{1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i \frac{\boldsymbol{m}_i}{\sqrt{\boldsymbol{v}_i}}\approx \frac{\bar{\boldsymbol{m}}_t \,\,\triangleq\,\, \frac{1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i\boldsymbol{m}_i}{\sqrt{\bar{\boldsymbol{v}}_t \,\,\triangleq\,\, \frac{1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i\boldsymbol{v}_i}}\end{equation}
这样一来
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0 - z_t \bar{\boldsymbol{u}}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{-2\kappa_t}z_t^2\Vert\bar{\boldsymbol{u}}_t\Vert_{RMS}^2\end{equation}

## 步骤之二

按照之前的思路，为了估计$\Vert \bar{\boldsymbol{u}}_t\Vert_{RMS}^2$，我们需要假设$\boldsymbol{g}_j$独立同分布地服从$\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2)$，然后求
\begin{equation}\mathbb{E}[\bar{\boldsymbol{u}}_t^2] \approx \mathbb{E}\left[\frac{\bar{\boldsymbol{m}}_t^2}{\bar{\boldsymbol{v}}_t}\right] \approx \frac{\mathbb{E}[\bar{\boldsymbol{m}}_t^2]}{\mathbb{E}[\bar{\boldsymbol{v}}_t]}\end{equation}
最后再对$\mathbb{E}[\bar{\boldsymbol{u}}_t^2]$的各个分量求平均，结果就可以作为$\Vert \bar{\boldsymbol{u}}_t\Vert_{RMS}^2$的一个近似。

对$\boldsymbol{m}_t,\boldsymbol{v}_t$展开得到
\begin{equation}\boldsymbol{m}_t = (1 - \beta_1)\sum_{i=1}^t \beta_1^{t-i}\boldsymbol{g}_i,\qquad \boldsymbol{v}_t = (1 - \beta_2)\sum_{i=1}^t \beta_2^{t-i}\boldsymbol{g}_i^2\label{eq:mv-roll}\end{equation}
同时我们还有恒等式
\begin{equation}\sum_{i=1}^t \sum_{j=1}^i a_i b_j = \sum_{j=1}^t \sum_{i=j}^t a_i b_j\end{equation}
利用这两个结果，我们可以写出
\begin{gather}
\bar{\boldsymbol{m}}_t = \frac{1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i\boldsymbol{m}_i = \frac{1 - \beta_1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i\sum_{j=1}^i \beta_1^{i-j}\boldsymbol{g}_j = \sum_{j=1}^t\boldsymbol{g}_j\underbrace{\frac{1 - \beta_1}{z_t}\sum_{i=j}^t e^{\kappa_i}\beta_1^{i-j}\eta_i}_{\text{记为}\bar{\beta}_1(j,t)} \\
\bar{\boldsymbol{v}}_t = \frac{1}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i\boldsymbol{v}_i = \frac{1 - \beta_2}{z_t}\sum_{i=1}^t e^{\kappa_i}\eta_i\sum_{j=1}^i \beta_2^{i-j}\boldsymbol{g}_j^2 = \sum_{j=1}^t\boldsymbol{g}_j^2\underbrace{\frac{1 - \beta_2}{z_t}\sum_{i=j}^t e^{\kappa_i}\beta_2^{i-j}\eta_i}_{\text{记为}\bar{\beta}_2(j,t)} \\
\end{gather}

## 步骤之三

先求分母，当$t$足够大时（$\beta_1^t, \beta_2^t$足够小），$\sum_{j=1}^t \bar{\beta}_1(j,t)$和$\sum_{j=1}^t \bar{\beta}_2(j,t)$ 会足够接近于1（因为本来就是双重加权平均的形式，只不过交换了求和符号），所以有
\begin{equation}\mathbb{E}[\bar{\boldsymbol{v}}_t] = \sum_{j=1}^t\bar{\beta}_2(j,t) \mathbb{E}[\boldsymbol{g}_j^2] = \sum_{j=1}^t\bar{\beta}_2(j,t) (\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2) \approx \boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2 \end{equation}
$\mathbb{E}[\bar{\boldsymbol{m}}_t]$同理，结果是$\boldsymbol{\mu}$，而$\mathbb{E}[\bar{\boldsymbol{m}}_t^2] = \mathbb{E}[\bar{\boldsymbol{m}}_t]^2 + \mathbb{V}ar[\bar{\boldsymbol{m}}_t]$，利用方差的平方可加性得
\begin{equation}\mathbb{V}ar[\bar{\boldsymbol{m}}_t] = \sum_{j=1}^t\bar{\beta}_1(j,t)^2 \mathbb{V}ar[\boldsymbol{g}_j] = \sum_{j=1}^t\bar{\beta}_1(j,t)^2 \boldsymbol{\sigma}^2\end{equation}
所以
\begin{equation}\mathbb{E}[\bar{\boldsymbol{u}}_t^2] \approx \frac{\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2\sum_{j=1}^t\bar{\beta}_1(j,t)^2}{\boldsymbol{\mu}^2 + \boldsymbol{\sigma}^2}\end{equation}
以及
\begin{equation}\Vert\bar{\boldsymbol{u}}_t\Vert_{RMS}^2 \approx \frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + \sum_{j=1}^t\bar{\beta}_1(j,t)^2}{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + 1} \end{equation}
最终有
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{-2\kappa_t}z_t^2\frac{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + \sum_{j=1}^t\bar{\beta}_1(j,t)^2}{\Vert\boldsymbol{\mu}\Vert^2/\Vert\boldsymbol{\sigma}\Vert^2 + 1}\end{equation}
如果读者是直接看这篇文章的话，那么可能会觉得某些步骤会有些跳跃，这时候不妨先重温一下[《AdamW的Weight RMS的渐近估计（上）》](<https://spaces.ac.cn/archives/11307>)，熟悉每步近似背后的思想。

## 例子之一

先考虑$\boldsymbol{\mu}=\boldsymbol{0}$，那么代入$\bar{\beta}_1(j,t)$的表示式到上式得
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{-2\kappa_t}(1-\beta_1)^2\sum_{j=1}^t\left(\sum_{i=j}^t e^{\kappa_i}\beta_1^{i-j}\eta_i\right)^2\label{eq:w-rms-mu0}\end{equation}
再考虑其中的简单例子$\lambda_t=0$，即没有Weight Decay，此时
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1-\beta_1)^2\sum_{j=1}^t\left(\sum_{i=j}^t \beta_1^{i-j}\eta_i\right)^2\end{equation}
如果$\beta_1\to 0$，那么立马就有$\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx \Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + \sum_{j=1}^t\eta_j^2$，这表明在没有Weight Decay且训练步数$t\to\infty$时，要想Weight RMS不爆炸，那么学习率序列的平方和要收敛，这也是传统优化理论的经典条件之一。事实上，即便$0 < \beta_1 < 1$，这个条件也是充要的，即
\begin{equation}\sum_{j=1}^{\infty}\left(\sum_{i=j}^{\infty} \beta_1^{i-j}\eta_i\right)^2 < \infty \qquad\Leftrightarrow\qquad \sum_{j=1}^{\infty}\eta_j^2 < \infty\end{equation}
证明不算难，我们将左端做一下变换：
\begin{equation}\begin{aligned}
\sum_{j=1}^{\infty}\left(\sum_{i=j}^{\infty} \beta_1^{i-j}\eta_i\right)^2 = \sum_{j=1}^{\infty}\left(\sum_{i=0}^{\infty} \beta_1^i\eta_{i+j}\right)^2 =&\, \sum_{j=1}^{\infty}\left(\sum_{i_1=0}^{\infty} \beta_1^{i_1}\eta_{i_1+j}\right)\left(\sum_{i_2=0}^{\infty} \beta_1^{i_2}\eta_{i_2+j}\right) \\
=&\, \sum_{i_1=0}^{\infty}\sum_{i_2=0}^{\infty} \beta_1^{i_1 + i_2}\sum_{j=1}^{\infty}\eta_{i_1+j}\eta_{i_2+j}
\end{aligned}\end{equation}
这表明，如果左端收敛，那么对于$\forall i_1, i_2$，求和$\sum_{j=1}^{\infty}\eta_{i_1+j}\eta_{i_2+j}$都会收敛，这自然也意味着$\sum_{j=1}^{\infty}\eta_j^2$收敛，必要性得证。至于充分性，我们可以从上式出发，利用柯西不等式：
\begin{equation}\begin{aligned}
\sum_{i_1=0}^{\infty}\sum_{i_2=0}^{\infty} \beta_1^{i_1 + i_2}\sum_{j=1}^{\infty}\eta_{i_1+j}\eta_{i_2+j} \leq&\, \sum_{i_1=0}^{\infty}\sum_{i_2=0}^{\infty} \beta_1^{i_1 + i_2}\sqrt{\left(\sum_{j=1}^{\infty}\eta_{i_1+j}^2\right)\left(\sum_{j=1}^{\infty}\eta_{i_2+j}^2\right)} \\
\leq&\, \sum_{i_1=0}^{\infty}\sum_{i_2=0}^{\infty} \beta_1^{i_1 + i_2}\sqrt{\left(\sum_{j=1}^{\infty}\eta_j^2\right)\left(\sum_{j=1}^{\infty}\eta_j^2\right)} \\
=&\, \frac{1}{(1-\beta_1)^2} \sum_{j=1}^{\infty}\eta_j^2
\end{aligned}\end{equation}
所以$\sum_{j=1}^{\infty}\eta_j^2$收敛推出左侧收敛，充分性得证。

## 例子之二

接下来，我们考虑Weight Decay是常数、学习率可变的情况，此时$\kappa_t = \lambda\sum_{i=1}^t \eta_i$。如果我们想要无限训练下去，得到尽可能接近理论最优的解，那么学习率应当满足$\sum_{i=1}^{\infty} \eta_i \to \infty$，这样才能通过第一项$e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2$把初始化彻底“忘掉”（理论最优解应当是跟初始化无关的）。有趣的是，这也是传统优化理论的经典条件之一。

对于一般的情况，计算式$\eqref{eq:w-rms-mu0}$是比较困难的，但我们可以根据实际情况再考虑一下近似。实际训练中，一般都有$\lambda_t \eta_t \ll 1$，所以$e^{\kappa_i}$的增长速度会远远慢于$\beta_1^i$的衰减速度，同时学习率$\eta_i$相比$\beta_1^i$通常也是缓变的，因此可以考虑近似
\begin{equation}\sum_{i=j}^t e^{\kappa_i}\beta_1^{i-j}\eta_i \approx \sum_{i=j}^t e^{\kappa_j}\beta_1^{i-j}\eta_j = e^{\kappa_j}\eta_j\sum_{i=j}^t\beta_1^{i-j}\approx e^{\kappa_j}\eta_j\sum_{i=j}^{\infty}\beta_1^{i-j} = \frac{e^{\kappa_j}\eta_j}{1-\beta_1}\end{equation}
将这个近似代回式$\eqref{eq:w-rms-mu0}$得
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{- 2\kappa_t}\sum_{j=1}^t e^{2\kappa_j}\eta_j^2\label{eq:w-rms-simp}\end{equation}
接下来就只能具体$\eta_j$具体计算了，比如$\lambda_j,\eta_j$都是常数时，可以算得$\kappa_t = \lambda\eta t$，以及
\begin{equation}\begin{aligned}
\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx&\, e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{- 2\kappa_t}\sum_{j=1}^t e^{2\kappa_j}\eta_j^2 \\
=&\, e^{-2\lambda\eta t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{-2\lambda\eta t}\sum_{j=1}^t e^{2\lambda\eta j}\eta^2 \\
=&\, e^{-2\lambda\eta t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + \frac{e^{2\lambda\eta}(1 - e^{-2\lambda\eta t})}{e^{2\lambda\eta} - 1}\eta^2 \\
\approx&\, e^{-2\lambda\eta t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1 - e^{-2\lambda\eta t} )\frac{\eta}{2\lambda}
\end{aligned}\end{equation}
这跟上一篇文章的结果一致。

## 微分方程

对数值计算来说，式$\eqref{eq:w-rms-simp}$其实已经相当简洁了，但如果我们想要得到一般的$\lambda_t,\eta_t$的解析结果，通常还是很困难的，所以还需要进一步寻找新的计算手段。

考虑到积分通常会比求和更容易计算一些，所以我们可以尝试用积分来近似求和：
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{- 2\kappa_t}\sum_{j=1}^t e^{2\kappa_j}\eta_j^2\approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + e^{- 2\kappa_t}\int_0^t e^{2\kappa_s}\eta_s^2 ds \label{eq:w-rms-int}\end{equation}
其中$\kappa_t = \int_0^t \lambda_s\eta_s ds$。记$\rho_t = \Vert\boldsymbol{\theta}_t\Vert_{RMS}^2$，两边乘$e^{2\kappa_t}$然后求导得$\frac{d}{dt}(e^{2\kappa_t}\rho_t) \approx e^{2\kappa_t}\eta_t^2$，整理可得
\begin{equation}\frac{d}{dt}\rho_t \approx -2\lambda_t\eta_t\rho_t + \eta_t^2\end{equation}
这便是RMS平方所服从的微分方程，应该说并不算复杂。如果$t\to\infty$时$\rho_t$收敛到常数，那么左侧等于0，于是
\begin{equation}\lim_{t\to\infty} \rho_t \approx \lim_{t\to\infty} \frac{\eta_t}{2\lambda_t}\end{equation}
这告诉我们，对于Decay型学习率序列，终点学习率不要设为0，否则长期训练下模型权重有坍缩的风险；当然，我们也可以选择设置$\lambda_t\propto \eta_t$（例如[AdamC](<https://papers.cool/arxiv/2506.02285>)），以避免权重坍缩。

## 平均之场

能够视为$t\to\infty$的，一般是Multi\-Epoch的监督训练，而在预训练场景中，训练通常都是Single\-Epoch的，这时候的$\kappa_t$往往是$\mathcal{\Theta}(1)$的，因为训练初期的样本权重跟$\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2$项的权重$e^{-2\kappa_t}$相当，过大的$\kappa_t$有可能会把早期的训练样本“忘掉”。

在$\kappa_t=\mathcal{\Theta}(1)$的假设下，我们可以考虑平均场近似。还是从积分形式$\eqref{eq:w-rms-int}$出发，根据定义，在$[0,t]$内，$\kappa_s$是一个以$0$为起点、以$\kappa_t$为终点的单调递增函数，于是我们$e^{\kappa_s} \geq 1$以及$e^{\kappa_s - \kappa_t} \leq 1$，那么
\begin{equation}e^{- 2\kappa_t}\int_0^t \eta_s^2 ds \leq e^{- 2\kappa_t}\int_0^t e^{2\kappa_s}\eta_s^2 ds = \int_0^t e^{2\kappa_s- 2\kappa_t}\eta_s^2 ds \leq \int_0^t \eta_s^2 ds\end{equation}
也就是目标积分本身就夹在$e^{- 2\kappa_t} \nu_t$和$\nu_t$之间，其中$\nu_t = \int_0^t \eta_s^2 ds$。而当$\kappa_t=\mathcal{\Theta}(1)$时，$e^{-2\kappa_t}$不会比1小太多，这意味着$\nu_t$本身就可能是一个良好近似了。当然，我们也可以做得更仔细一些，给$\nu_t$估计一个合理的倍数：
\begin{equation}e^{- 2\kappa_t}\int_0^t e^{2\kappa_s}\eta_s^2 ds \approx  e^{- 2\kappa_t}\int_0^t e^{2\kappa_s} (\nu_t / t) ds = \frac{\nu_t e^{- 2\kappa_t}}{t}\int_0^t e^{2\kappa_s} ds\end{equation}
考虑到$\kappa_s$是一个从$0$到$\kappa_t$的单调递增函数，我们用$(\kappa_t/t)s$去近似它：
\begin{equation}e^{- 2\kappa_t}\int_0^t e^{2\kappa_s}\eta_s^2 ds \approx \frac{\nu_t e^{- 2\kappa_t}}{t}\int_0^t e^{2\kappa_s} ds \approx \frac{\nu_t e^{- 2\kappa_t}}{t}\int_0^t e^{2(\kappa_t/t)s} ds = \frac{\nu_t}{2\kappa_t}(1 - e^{- 2\kappa_t})\end{equation}
代入到式$\eqref{eq:w-rms-int}$得
\begin{equation}\Vert\boldsymbol{\theta}_t\Vert_{RMS}^2 \approx e^{-2\kappa_t}\Vert\boldsymbol{\theta}_0\Vert_{RMS}^2 + (1 - e^{- 2\kappa_t})\frac{\nu_t}{2\kappa_t}\end{equation}

## 例子之三

再次回到我们主要关心的“Weight Decay固定、学习率可变”这一常见设置，我们来算几个具体例子的$\kappa_t,\nu_t$。首先是线性学习率：
\begin{equation}\eta_s = \eta_a + (\eta_b - \eta_a) s / t\end{equation}
这里的$\eta_a,\eta_b$分别是初始学习率和终止学习率，可能$\eta_b > \eta_a$（比如Warmup），也可能$\eta_b < \eta_a$（线性Decay），$t$是预期的总训练步数。积分得
\begin{gather}
\kappa_t = \int_0^t \lambda\eta_s ds= \lambda (\eta_a + \eta_b) t / 2  \\
\nu_t = \int_0^t \eta_s^2 ds = (\eta_a^2 + \eta_a \eta_b + \eta_b^2) t / 3
\end{gather}
然后是Cosine Decay：
\begin{equation}\eta_s = \eta_{\min} + (\eta_{\max} - \eta_{\min})\left(\frac{1}{2} + \frac{1}{2}\cos \frac{s\pi}{t}\right)\end{equation}
积分得
\begin{gather}
\kappa_t = \int_0^t \lambda\eta_s ds= \lambda (\eta_{\min} + \eta_{\max}) t / 2  \\
\nu_t = \int_0^t \eta_s^2 ds = (3\eta_{\min}^2 + 2\eta_{\min} \eta_{\max} + 3\eta_{\max}^2 ) t / 8
\end{gather}
最后是WSD（Warmup Stable Decay）：
\begin{equation}\eta_s = \left\{\begin{aligned} \frac{s}{t_1}\eta_{\max}, \quad s \in [0, t_1] \\[5pt]
\eta_{\max} , \quad s \in [t_1, t_2] \\[5pt]
\frac{t-s}{t-t_2}\eta_{\max}, \quad j \in [t_2, t]
\end{aligned}\right.\end{equation}
有
\begin{gather}
\kappa_t = \int_0^t \lambda\eta_s ds= \lambda \eta_{\max} (t + t_2 - t_1) / 2  \\
\nu_t = \int_0^t \eta_s^2 ds = \eta_{\max}^2 (t + 2t_2 - 2t_1) / 3
\end{gather}

## 模拟验证

我们也可以通过数值模拟验证上面的各种近似：



<pre><code class="language-python">import numpy as np

N, T = 10000, 10000
beta1, beta2 = 0.9, 0.95
m, v = 0, 0
w = np.random.randn(N) * (init_std := 0.1)
lr_max, lr_min, wd = 0.001, 0.0001, 0.1
lr = lr_min + (lr_max - lr_min) * (1 + np.cos(np.arange(T) / T * np.pi)) / 2
for i in range(T):
    g = np.random.randn(N)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    w = w - lr[i] * (m / v**0.5 + wd * w)

# 直接计算 ≈ 0.0744
weight_rms = (w**2).mean()**0.5

# 级数近似 ≈ 0.0742
kappa = wd * lr.cumsum()
approx1 = ((np.exp(kappa * 2) * lr**2).sum() + init_std**2)**0.5 * np.exp(-kappa[-1])

# 平均场近似 ≈ 0.0760
kappa = wd * (lr_max + lr_min) / 2 * T
nu = (3 * lr_max**2 + 2 * lr_max * lr_min + 3 * lr_min**2) / 8 * T
approx2 = ((np.exp(kappa * 2) - 1) * nu / kappa / 2 + init_std**2)**0.5 * np.exp(-kappa)

print(weight_rms)
print(approx1)
print(approx2)</code></pre>



## 文章小结

本文将上篇的结果推广到了动态版，允许我们估计随时间变化的学习率和Weight Decay之下的AdamW的Weight RMS。
