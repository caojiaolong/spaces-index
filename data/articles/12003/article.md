# 让炼丹更科学一些（十一）：无调度学习率新思路

> 作者：苏剑林 · 科学空间 · 2026-11-09
>
> 原文：<https://spaces.ac.cn/archives/12003>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前面的多篇文章中，我们已经表明了学习率调度对终点收敛的重要性。与此同时，“无调度（Schedule\-Free）学习率”一直也是训练人的理想追求：不预设学习率曲线，也不预设训练步数，训练过程随时终止，都能得到截止到当前步数的近乎最优的解。

笔者此前对这条路线的态度是比较悲观的，参考[《让炼丹更科学一些（八）：多阶段训练的学习率》](<https://spaces.ac.cn/archives/11879>)。但近日，论文[《Conic\-Lifting for Learning\-rate\-free Convex Optimization》](<https://papers.cool/arxiv/2610.39531>)通过一个精妙的提升构造实现了这一点！可以证明，在有界域和凸优化假设下，它真的能无调度地以理想速度收敛到最优解！

它究竟是怎么做到的？深度学习的训练能否从中借鉴什么吗？本文一起来学习一下。

## 基础回顾

考虑常数学习率的投影更新规则
\begin{equation}\boldsymbol{\theta}_{t+1} = \Pi_{\boldsymbol{\Theta}}(\boldsymbol{\theta}_t - \eta \boldsymbol{g}_t)\end{equation}
其中$\boldsymbol{\Theta}$是$\boldsymbol{\theta}$的可行集，要求是一个凸闭集，$\Pi_{\boldsymbol{\Theta}}$定义为
\begin{equation}\Pi_{\boldsymbol{\Theta}} (\boldsymbol{\varphi}) = \mathop{\text{argmin}}_{\boldsymbol{\theta}\in\boldsymbol{\Theta}}\Vert\boldsymbol{\varphi}-\boldsymbol{\theta}\Vert\end{equation}
那么对于$\forall \boldsymbol{\varphi}\in \boldsymbol{\Theta}$，成立不等式
\begin{equation}\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi})\leq \frac{\Vert\boldsymbol{\theta}_1-\boldsymbol{\varphi}\Vert^2}{2\eta}+\frac{\eta}{2}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert^2\label{eq:regret-base}\end{equation}
左端的式子，我们称为“线性化遗憾（Regret）”。由于我们只考虑常数学习率，因此该不等式的证明特别简单，由第一篇[《让炼丹更科学一些（一）：SGD的平均损失收敛》](<https://spaces.ac.cn/archives/9902>)的方法即可完成。同样也是常学习率的缘故，这个结论也不要求$\boldsymbol{\Theta}$有界，只要求它是一个凸闭集。

在上述不等式中，$\boldsymbol{g}_t$也不要求是真正的梯度。但当$\boldsymbol{g}_t$是梯度时，结合损失函数的凸性假设，我们有
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\varphi}) \leq \frac{1}{T}\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi})\end{equation}
继而将前述结论跟损失函数联系起来，这是“$\boldsymbol{g}_t$是梯度”和凸性假设的唯一用途。最后代入$\boldsymbol{\varphi}=\boldsymbol{\theta}^*$，就得到一个平均收敛结论。通过最小化式$\eqref{eq:regret-base}$右端，我们可以得到最优学习率，它至少依赖于总训练步数$T$。

## 提升向量

接下来我们约定$\boldsymbol{\Theta}$是一个有界凸闭集，并引入标量序列$\lambda_t\geq 0$，常数$K > 0$，定义提升向量：
\begin{align}
\check{\boldsymbol{\theta}}_t \triangleq&\, (\lambda_t \boldsymbol{\theta}_t, \lambda_t K) \\
\check{\boldsymbol{g}}_t \triangleq&\, (\boldsymbol{g}_t, -\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t/K) \\
\check{\boldsymbol{\varphi}} \triangleq&\, (\boldsymbol{\varphi},K) \\
\end{align}
为什么要限制$\boldsymbol{\Theta}$有界呢？因为只有$\boldsymbol{\Theta}$是有界凸闭集时，$\check{\boldsymbol{\theta}}$的可行域
\begin{equation}\check{\boldsymbol{\Theta}} \triangleq \Big\{(\lambda \boldsymbol{\theta}, \lambda K)\,\Big|\,\forall \boldsymbol{\theta}\in\boldsymbol{\Theta},\,\forall \lambda\geq 0\Big\}\end{equation}
才是闭集（形状是一个锥，所以称为“锥提升”）。接下来我们需要往$\check{\boldsymbol{\Theta}}$投影，如果$\boldsymbol{\Theta}$无界，那$\check{\boldsymbol{\Theta}}$就不一定是闭集，投影运算无法良好定义。类似地，约束$\lambda\geq 0$也是为了让$\check{\boldsymbol{\Theta}}$为凸集，实际上后面的推导并没有显式要求$\lambda\geq 0$，但如果允许$\lambda$任意，那么$\check{\boldsymbol{\Theta}}$就非凸，证明就无法进行下去。

为什么要这样提升我们先不讨论，等会大家在证明过程中自己领悟。注意这三个提升向量有如下特点：首先，对于任意$\lambda_t > 0$，$\check{\boldsymbol{\theta}}_t$可以通过前$\dim(\boldsymbol{\theta}_t)$个维度除以最后一个维度再乘以$K$，来恢复$\boldsymbol{\theta}_t$（这一步称为“解码”），所以一般情况下$\check{\boldsymbol{\theta}}_t$已经包含了$\boldsymbol{\theta}_t$；其次，可以验证这样的构造下，恒成立$\check{\boldsymbol{g}}_t\cdot\check{\boldsymbol{\theta}}_t = 0$，于是
\begin{equation}\sum_{t=1}^T \check{\boldsymbol{g}}_t\cdot(\check{\boldsymbol{\theta}}_t-\check{\boldsymbol{\varphi}}) = -\sum_{t=1}^T \check{\boldsymbol{g}}_t\cdot\check{\boldsymbol{\varphi}} = \sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi})\label{eq:regret-id}\end{equation}
即提升前后的遗憾完全相等，但提升后的向量多了个可以灵活伸缩的自由度，这一多出来的自由度正是消去学习率的关键。

## 代入结论

考虑提升向量的更新规则
\begin{equation}\check{\boldsymbol{\theta}}_{t+1} = \Pi_{\check{\boldsymbol{\Theta}}}(\check{\boldsymbol{\theta}}_t - \eta \check{\boldsymbol{g}}_t)\end{equation}
那么对于任意$\rho > 0$，有
\begin{equation}\begin{aligned}
\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi}) =&\, \frac{1}{\rho}\sum_{t=1}^T \check{\boldsymbol{g}}_t\cdot(\check{\boldsymbol{\theta}}_t-\rho\check{\boldsymbol{\varphi}}) \\
\leq&\, \frac{\Vert\check{\boldsymbol{\theta}}_1-\rho\check{\boldsymbol{\varphi}}\Vert^2}{2\eta\rho}+\frac{\eta}{2\rho}\sum_{t=1}^T\Vert\check{\boldsymbol{g}}_t\Vert^2 \\
=&\, \frac{\Vert\lambda_1\boldsymbol{\theta}_1-\rho\boldsymbol{\varphi}\Vert^2 + K^2(\lambda_1 - \rho)^2}{2\eta\rho}+\frac{\eta}{2\rho}\sum_{t=1}^T\Big(\Vert\boldsymbol{g}_t\Vert^2 + (\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t/K)^2\Big)
\end{aligned}\end{equation}
第一个“$=$”是恒等式$\eqref{eq:regret-id}$的推论，“$\leq$”则是代入了基础结论$\eqref{eq:regret-base}$，第二个“$=$”则是代入了$\check{\boldsymbol{\theta}}_t$、$\check{\boldsymbol{g}}_t$的定义。注意根据$\check{\boldsymbol{\Theta}}$的定义，$\rho\boldsymbol{\varphi}\in\check{\boldsymbol{\Theta}}$对任意$\rho > 0$都成立，这也是上式可行的关键。

在上式中，左端是固定的目标，右端是它的上界，有$\lambda_1,\rho,\eta,K$四个参数，简单起见，我们取$\lambda_1=0$，此时$\check{\boldsymbol{\theta}}_1$是全零向量，这也是后续推导的关键。当前整理得
\begin{equation}\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi}) \leq \frac{\rho\big(\Vert\boldsymbol{\varphi}\Vert^2 + K^2\big)}{2\eta}+\frac{\eta}{2\rho}\sum_{t=1}^T\Big(\Vert\boldsymbol{g}_t\Vert^2 + (\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t/K)^2\Big)\end{equation}

## 固定步长

现在还剩下$\eta,\rho,K$三个参数。一个关键观测是：上式右端关于$\rho,\eta$是零齐次的，即它只依赖于比值$\rho/\eta$。这意味着我们可以固定$\eta=1$，而不改变右端的最小值。那可以固定$\eta$为某个值吗？还真可以！可以证明锥$\check{\boldsymbol{\Theta}}$的欧氏投影成立正齐次性，即对于任意$\alpha > 0$有
\begin{equation}\Pi_{\check{\boldsymbol{\Theta}}}(\alpha\boldsymbol{x}) = \alpha\Pi_{\check{\boldsymbol{\Theta}}}(\boldsymbol{x})\end{equation}
基于此，我们可以归纳地证明学习率为$\eta$和学习率为$1$的两条轨迹
\begin{equation}\check{\boldsymbol{\theta}}_{t+1}^{(\eta)} = \Pi_{\check{\boldsymbol{\Theta}}}(\check{\boldsymbol{\theta}}_t^{(\eta)} - \eta \check{\boldsymbol{g}}_t)\qquad\text{和}\qquad \check{\boldsymbol{\theta}}_{t+1}^{(1)} = \Pi_{\check{\boldsymbol{\Theta}}}(\check{\boldsymbol{\theta}}_t^{(1)} - \check{\boldsymbol{g}}_t)\end{equation}
恒成立
\begin{equation}\check{\boldsymbol{\theta}}_t^{(\eta)} = \eta\check{\boldsymbol{\theta}}_t^{(1)}\end{equation}
即$\check{\boldsymbol{\theta}}$的轨迹会等比例缩放$\eta$倍。然而我们关心的是$\boldsymbol{\theta}$，它取决于$\check{\boldsymbol{\theta}}$的前$\dim(\boldsymbol{\theta}_t)$个维度与最后一个维度之比，正好消除了这个倍数，所以$\eta$固定为1不改变实际参数轨迹。由此可见，锥提升加比值解码让学习率变成了不可观测的整体尺度。

固定$\eta=1$后，得到
\begin{equation}\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi}) \leq \frac{\rho\big(\Vert\boldsymbol{\varphi}\Vert^2 + K^2\big)}{2}+\frac{1}{2\rho}\sum_{t=1}^T\Big(\Vert\boldsymbol{g}_t\Vert^2 + (\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t/K)^2\Big)\end{equation}

## 超球体内

接下来我们选取一个特殊但有代表性的$\boldsymbol{\Theta}$。由于$\boldsymbol{\Theta}$被限定为有界凸闭集，我们将它选择为简单的超球体$\Vert\boldsymbol{\theta}\Vert\leq R$，此时根据约定也有$\Vert\boldsymbol{\varphi}\Vert\leq R$，且
\begin{equation}(\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t)^2 \leq \Vert\boldsymbol{g}_t\Vert^2 \Vert\boldsymbol{\theta}_t\Vert^2 \leq \Vert\boldsymbol{g}_t\Vert^2 R^2\end{equation}
代入，然后取$K=R$，得到
\begin{equation}\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi}) \leq \rho R^2 +\frac{1}{\rho}\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert^2\end{equation}
最后，取$\rho = R^{-1}\sqrt{\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert^2}$，得到
\begin{equation}\sum_{t=1}^T \boldsymbol{g}_t\cdot(\boldsymbol{\theta}_t-\boldsymbol{\varphi}) \leq 2R\sqrt{\sum_{t=1}^T\Vert\boldsymbol{g}_t\Vert^2}\end{equation}
右端已经是最理想的收敛速度。如果进一步假设$\Vert\boldsymbol{g}_t\Vert$有界，那么右端$\sim\mathcal{O}(\sqrt{T})$，两端除以$T$后，收敛速度$\sim\mathcal{O}(1/\sqrt{T})$，在不引入进一步假设下，已经是最佳速度。当然，这里给出的是平均收敛结论，实际训练可能更关注终点收敛结论，两者不完全等价，但不管怎样，这确实是一个很漂亮的结果了。

## 显式规则

最为重要的是，达到这个收敛速度的更新规则是
\begin{equation}\check{\boldsymbol{\theta}}_{t+1} = \Pi_{\check{\boldsymbol{\Theta}}}(\check{\boldsymbol{\theta}}_t - \check{\boldsymbol{g}}_t)\end{equation}
没有任何学习率超参数，实现了真正的“Schedule\-Free”！针对超球体$\Vert\boldsymbol{\theta}\Vert\leq R$，我们还可以将更新规则显式写出来。首先定义
\begin{equation}\begin{aligned}
(\lambda_{t+1}\boldsymbol{\theta}_{t+1}, \lambda_{t+1}R) =&\, \Pi_{\check{\boldsymbol{\Theta}}} (\tilde{\lambda}_{t+1}\tilde{\boldsymbol{\theta}}_{t+1}, \tilde{\lambda}_{t+1}R) \\
(\tilde{\lambda}_{t+1}\tilde{\boldsymbol{\theta}}_{t+1}, \tilde{\lambda}_{t+1}R) =&\, (\lambda_t \boldsymbol{\theta}_t - \boldsymbol{g}_t , \lambda_t R + \boldsymbol{g}_t\cdot\boldsymbol{\theta}_t/R)
\end{aligned}\end{equation}
投影运算$\Pi_{\check{\boldsymbol{\Theta}}}$的定义是
\begin{equation}\mathop{\text{argmin}}_{\boldsymbol{\theta}, \lambda} \Vert \lambda\boldsymbol{\theta} - \tilde{\lambda}\tilde{\boldsymbol{\theta}}\Vert^2 + (\lambda - \tilde{\lambda})^2 R^2\qquad\text{s.t}\qquad \Vert\boldsymbol{\theta}\Vert\leq R,\quad \lambda\geq 0\end{equation}
解得
\begin{equation}\lambda^* = \max\left\{0, \tilde{\lambda}, \frac{\Vert\tilde{\lambda}\tilde{\boldsymbol{\theta}}\Vert + \tilde{\lambda} R}{2R}\right\},\qquad \boldsymbol{\theta}^* = \Pi_{\boldsymbol{\Theta}}\left(\frac{\tilde{\lambda}\tilde{\boldsymbol{\theta}}}{\lambda^*}\right)\end{equation}
注意，当$\lambda^*=0$时，对应的$\boldsymbol{\theta}^*$实际上可以是满足$\Vert\boldsymbol{\theta}\Vert\leq R$的任意向量，上式给出了一个比较自然的选择（按$\lambda^*\to 0^+$理解），原论文则直接约定返回初始值$\boldsymbol{\theta}_1$，这些选择都不影响证明过程和结论。特别地，如果没有触发投影，那么更新公式可以简化成
\begin{equation}\begin{aligned}
\lambda_{t+1} =&\, \lambda_t + \frac{\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t}{R^2} \\
\boldsymbol{\theta}_{t+1} =&\, \frac{\lambda_t}{\lambda_{t+1}} \boldsymbol{\theta}_t - \frac{1}{\lambda_{t+1}}\boldsymbol{g}_t
\end{aligned}\end{equation}

## 延伸思考

以上推导过程经过笔者自己的调整，没有完全照搬原论文；原论文的结果还更为丰富，但本篇博客就先介绍到这里。接下来，谈谈笔者对它的一些思考。

刚看到的这个结果时，笔者的第一反应是不可思议，哪怕在凸优化假设下，居然真的有完全Schedule\-Free的算法能达到最理想的收敛速度。随后，笔者认真过了一遍推导，发现过程并没有什么问题，但也产生了一些疑惑：这个新算法说白了就是基于$\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t$来调节学习率，如果它为零怎么办？

当然，如果$\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t$偶尔为零，那么一般不碍事，但如果$\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t$恒等于零呢？这意味着迭代压根就不会正常启动。那$\boldsymbol{g}_t\cdot\boldsymbol{\theta}_t$有可能恒等于零吗？有可能，比如$L(\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert)$这样的损失函数就是。但前面的推导又没有错，问题在哪呢？答案就是——凸假设——形如$L(\boldsymbol{\theta}/\Vert\boldsymbol{\theta}\Vert)$的函数不可能是关于$\boldsymbol{\theta}$的凸函数！

这也就意味着，这个新的Schedule\-Free思路，看起来强依赖于凸优化假设，笔者的初步判断是，对深度学习中的非凸优化的参考价值可能很有限。这跟本系列前面的文章有点不一样，之前的文章虽然也是基于凸优化假设的结果，但很多结论还是对非凸优化有所启发的。

比如AdaGrad优化器，它也可以说是凸优化的结果，但它本身以及衍生出来的RMSProp、Adam优化器，对深度学习模型的训练也很有效；又比如前面我们推出的终点收敛结论依赖于类似Linear Decay，这在实践中也很有效。虽然很多结论都是“马后炮”式的强行解释，但我们确实能感知到这种启发性的差异。

## 文章小结

本文介绍了Schedule\-Free方向的一个新工作，它在原本的参数空间中新增一个维度，通过巧妙的构造，让提升前后的遗憾形式相等，然后在提升后的空间应用已有的收敛结果，在有界域与凸优化假设下，得到无需显式的学习率设置就可以自动达到最佳收敛速度的结论。
