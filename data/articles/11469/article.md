# 让炼丹更科学一些（二）：将结论推广到无界域

> 作者：苏剑林 · 科学空间 · 2025-12-12
>
> 原文：<https://spaces.ac.cn/archives/11469>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

两年前，笔者打算开一个“科学炼丹”专题，本想着系统整理一下优化器的经典理论结果，但写了第一篇[《让炼丹更科学一些（一）：SGD的平均损失收敛》](<https://spaces.ac.cn/archives/9902>)后，就一直搁置至今。主要原因在于，笔者总觉得这些经典优化结论所依赖的条件过于苛刻，跟实际应用相去甚远，尤其是进入LLM时代后，这些结论的参考价值似乎更加有限，所以就没什么动力继续写下去。

然而，近期在思考Scaling Law的相关问题时，笔者发现这些结论并非想象中那么“没用”，它可以为一些经验结果提供有益的理论洞见。因此，本文将重启该系列，继续推进这个专题文章的撰写，“偿还”之前欠下的“债务”。

## 结论回顾

记号方面我们沿用第一篇文章的，所以不再重复记号的介绍。第一篇文章的主要结论是：在适当的假设之下，SGD成立
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)\leq \frac{R^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T\eta_t\label{leq:avg-1}\end{equation}
其中$R, G$是优化轨迹无关的常数，而“适当的假设”包括：

> 1、$\boldsymbol{\Theta}$是一个有界凸集，$R=\max\limits_{\boldsymbol{\theta}_1,\boldsymbol{\theta}_2\in \boldsymbol{\Theta}}\Vert\boldsymbol{\theta}_1-\boldsymbol{\theta}_2\Vert < \infty$；
> 
> 
> 
> 2、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，$L(\boldsymbol{x},\boldsymbol{\theta})$都是关于$\boldsymbol{\theta}$的凸函数；
> 
> 
> 
> 3、对于任意$\boldsymbol{\theta}\in \boldsymbol{\Theta}$以及任意$\boldsymbol{x}$，都有$\Vert\nabla_{\boldsymbol{\theta}}L(\boldsymbol{x},\boldsymbol{\theta})\Vert\leq G < \infty$；
> 
> 
> 
> 4、学习率$\eta_t$是关于$t$的单调递减函数（即$\eta_t\geq \eta_{t+1}$）；

这里比较“别扭”的地方可能是第一点“有界凸集”中的“有界”，因为SGD的优化轨迹原则上是没法保证有界的，所以为了保证有界性我们要将SGD修改为投影SGD：
\begin{equation}\boldsymbol{\theta}_{t+1} = \Pi_{\boldsymbol{\Theta}}\big(\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\big)\in \boldsymbol{\Theta}\label{eq:sgd-p}\end{equation}
当然，有界性在实践中完全没有问题，适当的有界性甚至还能增强优化算法的稳定性。但是从“理论洁癖”的角度看，多加一个约束总是不舒服的。所以，本文我们先把有界性这个条件去掉，以得到更简捷、更具启发性的证明

## 新的思路

顺便说，这一节以及本文接下来章节的证明，主要参考了博客[Parameter\-free](<https://parameterfree.com/>)的文章[《Last Iterate of SGD Converges (Even in Unbounded Domains)》](<https://parameterfree.com/2020/08/07/last-iterate-of-sgd-converges-even-in-unbounded-domains/>)，这是一个关于优化理论的经典博客，除了这篇外上面还有很多优化器方面的理论介绍，有不少还是作者原创的，值得一读。

跟上一篇文章的证明一样，我们出发点都是恒等式
\begin{equation}\begin{aligned}
\Vert\boldsymbol{\theta}_{t+1} - \boldsymbol{\varphi}\Vert^2=&\, \Vert\boldsymbol{\theta}_t - \eta_t \boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)- \boldsymbol{\varphi}\Vert^2 \\
=&\, \Vert\boldsymbol{\theta}_t - \boldsymbol{\varphi}\Vert^2 - 2\eta_t (\boldsymbol{\theta}_t- \boldsymbol{\varphi})\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) + \eta_t^2\Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2
\end{aligned}\end{equation}
这里$\boldsymbol{\varphi}$是任意向量，上一篇文章直接取了$\boldsymbol{\varphi}=\boldsymbol{\theta}^*$，但这里我们保留了它取任意值的可能性。这一恒等式之后，上文采用了两边除以$2\eta_t$然后对$t$求和的方案，导致在后面的放缩中需要有界性条件才容易操作；本文则不打算除以$2\eta_t$，而是直接两端对$t$求和，这样中间的$\Vert\boldsymbol{\theta}_t - \boldsymbol{\varphi}\Vert^2$就自然地消去了：
\begin{equation}\begin{aligned}
2\sum_{t=1}^T \eta_t (\boldsymbol{\theta}_t- \boldsymbol{\varphi})\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) =&\, \Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2 - \Vert\boldsymbol{\theta}_{T+1} - \boldsymbol{\varphi}\Vert^2 + \sum_{t=1}^T \eta_t^2 \Vert\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t)\Vert^2 \\
\leq&\, \Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2 + G^2\sum_{t=1}^T \eta_t^2
\end{aligned}\end{equation}
利用$L$的凸性可知$(\boldsymbol{\theta}_t- \boldsymbol{\varphi})\cdot\boldsymbol{g}(\boldsymbol{x}_t,\boldsymbol{\theta}_t) \geq L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})$，代入上式左端整理得
\begin{equation}\sum_{t=1}^T \eta_t [L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})]\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2}{2} + \frac{G^2}{2}\sum_{t=1}^T \eta_t^2\label{leq:avg-2-mid1}\end{equation}
这已经非常接近我们想要的结果了，并且在放缩过程中没有用到每一点的有界性，所以只需要对初始化$\boldsymbol{\theta}_1$与终点$\boldsymbol{\varphi}$的距离作出假设。

## 加个期望

接下来，如果我们能进一步假设$L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})$的非负性，那么就可以结合$\eta_t$的单调递减性得到$\eta_t [L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})] \geq \eta_T [L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})]$，继而代入上式得
\begin{equation}\frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - \frac{1}{T}\sum_{t=1}^T L(\boldsymbol{x}_t,\boldsymbol{\varphi})\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2}{2 T \eta_T} + \frac{G^2}{2T}\sum_{t=1}^T \frac{\eta_t^2}{\eta_T}\end{equation}
问题是，即便我们设$\boldsymbol{\varphi}$是全局最优点$\boldsymbol{\theta}^*$，也没法保证$L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})$非负，因为最优点对全体样本来说是最优点，但对部分样本来说不一定是最优点。为了避开这个麻烦，我们留意到式$\eqref{leq:avg-2-mid1}$右端是$\boldsymbol{x}_t$无关的，而左端是$\boldsymbol{x}_t$相关的，所以对两边求期望，不等式依然成立：
\begin{equation}\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})]\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2}{2} + \frac{G^2}{2}\sum_{t=1}^T \eta_t^2\label{leq:avg-2-mid2}\end{equation}
这里$\mathbb{E}$是$\mathbb{E}_{\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_T}$，即“以不同的采样顺序、无限次重复SGD，对每一条优化轨迹求平均值”。接着，由于$\boldsymbol{\varphi}$是数据无关的任意向量，所以有$\mathbb{E}[L(\boldsymbol{x}_t,\boldsymbol{\varphi})] = \mathbb{E}_{\boldsymbol{x}_t}[L(\boldsymbol{x}_t,\boldsymbol{\varphi})] = L(\boldsymbol{\varphi})$，正好是我们的优化目标；至于$L(\boldsymbol{x}_t,\boldsymbol{\theta}_t)$，注意$\boldsymbol{\theta}_t$是依赖$\boldsymbol{x}_1,\cdots,\boldsymbol{x}_{t-1}$的，所以我们顶多有
\begin{equation}\mathbb{E}[L(\boldsymbol{x}_t,\boldsymbol{\theta}_t)] = \mathbb{E}_{\boldsymbol{x}_1,\cdots,\boldsymbol{x}_t}[L(\boldsymbol{x}_t,\boldsymbol{\theta}_t)] = \mathbb{E}_{\boldsymbol{x}_1,\cdots,\boldsymbol{x}_{t-1}}[\mathbb{E}_{\boldsymbol{x}_t}[L(\boldsymbol{x}_t,\boldsymbol{\theta}_t)]] = \mathbb{E}_{\boldsymbol{x}_1,\cdots,\boldsymbol{x}_{t-1}}[L(\boldsymbol{\theta}_t)] = \mathbb{E}[L(\boldsymbol{\theta}_t)]\end{equation}
由于$\boldsymbol{\varphi}$的数据无关性，$L(\boldsymbol{\varphi})$也可以写成$\mathbb{E}[L(\boldsymbol{\varphi})]$，将它们代入到式$\eqref{leq:avg-2-mid2}$得
\begin{equation}\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\varphi})]\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\varphi}\Vert^2}{2} + \frac{G^2}{2}\sum_{t=1}^T \eta_t^2\label{leq:avg-2-mid3}\end{equation}

## 思想讨论

在进一步推导之前，笔者打算多讨论几句：我们<u>为什么要</u>以及<u>为什么能</u>引入这种期望形式？

“为什么要”的答案很简单——为了让推导能够进行下去。比如，我们想要$L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\varphi})$的非负性，但这是没法保证的，而加了期望后，我们就可以得到$\mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\varphi})]$的非负性，只要$\boldsymbol{\varphi}$取理论最优点就行。总的来说，通过求期望可以消去对数据变量$\boldsymbol{x}_t$的依赖，简化后续推导，是这一步骤的主要目的。

那“为什么能”呢？可能会有部分读者认为，我们又不会真的去重复训练很多次然后求平均，为何要关心这么一个期望结果呢？要回答这个问题，需要先思考一下：我们本质上想要什么？事实上，我们只想要一个SGD在某种程度上的收敛结论，由于目标函数限定了凸函数，我们希望这个结论在数学上是严格的。

对于一个复杂的随机过程而言，数学期望往往是我们能算的最基本结果，甚至可能是唯一能算的。因此，对两端加期望，虽然让结论与实践有所偏离，但它依然不失为一个具有参考价值的收敛结论。众所周知，实践通常更为复杂，所以只要能跟实践有一定联系、能够描述一些现象、产生一些启发，都是有价值的理论结果。

## 单调放缩

言归正传。现在有了式$\eqref{leq:avg-2-mid3}$，我们设$\boldsymbol{\varphi}$为理论最优点$\boldsymbol{\theta}^*$，就可以得到$\mathbb{E}[L(\boldsymbol{x}_t,\boldsymbol{\theta}_t) - L(\boldsymbol{x}_t,\boldsymbol{\theta}^*)]$的非负性，然后继续假设$\eta_t$的单调递减性，那么就可以将$\eta_t$换成$\eta_T$，最后两边除以$T\eta_T$得到
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2T\eta_T} + \frac{G^2}{2T}\sum_{t=1}^T \frac{\eta_t^2}{\eta_T}\label{leq:avg-2}\end{equation}
对比式$\eqref{leq:avg-1}$，两者在关于$T$和$\eta_t$的依赖关系是相似的：右端第一项$R^2$换成了$\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2$，这相对来说是缩小的；而右端第二项求和的$\eta_t$换成了$\eta_t^2/\eta_T$，这通常是放大的。一小一大之下，区别大吗？我们观察两个例子。第一例是常数学习率$\frac{\alpha}{\sqrt{T}}$，此时结论$\eqref{leq:avg-1}$和$\eqref{leq:avg-2}$本质是一样的，以$\eqref{leq:avg-2}$为例，代入得到
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{1}{2\sqrt{T}}\left(\frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{\alpha} + G^2\alpha\right)\end{equation}
也就是都能达到$\mathcal{O}(1/\sqrt{T})$的收敛速率。第二例是动态学习率$\eta_t = \frac{\alpha}{\sqrt{t}}$，上一篇文章我们已经证明了，代入到结论$\eqref{leq:avg-1}$，也能得到$\mathcal{O}(1/\sqrt{T})$的收敛速率，但如果代入到$\eqref{leq:avg-2}$，结果是
\begin{equation}\frac{1}{T}\sum_{t=1}^T \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)] \leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2 \alpha \sqrt{T}} + \frac{G^2\alpha}{2\sqrt{T}}\sum_{t=1}^T \frac{1}{t} \sim \mathcal{O}\left(\frac{\ln T}{\sqrt{T}}\right)\end{equation}
因此，推广到无界域的代价是收敛速率从$\mathcal{O}(1/\sqrt{T})$略微放大成了$\mathcal{O}(\ln T/\sqrt{T})$，这种程度的放大对于实践来说并无区别。此外，这个结果仅仅是对于$\eta_t = \frac{\alpha}{\sqrt{t}}$而言的，通过设$\eta_t = \frac{\alpha}{\sqrt{t\ln (t+1)}}$，还可以将它逼近$\mathcal{O}(\sqrt{\ln T/T})$（差一个$\ln\ln T$），但这实际上并无本质区别，更多是一种推导游戏。

## 加权平均

事实上，对于式$\eqref{leq:avg-2-mid3}$，还有一种更简单的处理方式：直接两边除以$\sum_{t=1}^T \eta_t$，然后代入$\boldsymbol{\varphi}=\boldsymbol{\theta}^*$得
\begin{equation}\frac{\sum_{t=1}^T \eta_t \mathbb{E}[L(\boldsymbol{\theta}_t) - L(\boldsymbol{\theta}^*)]}{\sum_{t=1}^T \eta_t}\leq \frac{\Vert\boldsymbol{\theta}_1 - \boldsymbol{\theta}^*\Vert^2}{2\sum_{t=1}^T \eta_t} + \frac{G^2}{2}\frac{\sum_{t=1}^T \eta_t^2}{\sum_{t=1}^T \eta_t}\label{leq:avg-3}\end{equation}
这样不需要对学习率做单调递减假设，可以说是相当宽松的一个收敛结论了，它描述的是优化轨迹上的加权平均的收敛性质。可能有读者质疑为什么要加权呢？还是刚才的答案——“我们只想要一个SGD在某种程度上的收敛结论”——所以从逼近最优损失值来看，加不加权又有什么所谓呢？

某种程度上，式$\eqref{leq:avg-3}$给我们带来的启发要多于式$\eqref{leq:avg-2}$。比如，要想右端收敛得尽可能快，那么学习率之和$\sum_{t=1}^T \eta_t$要尽可能大，但平方和$\sum_{t=1}^T \eta_t^2$要尽可能小，如果想要$T\to\infty$时收敛到最优点，那么学习率应当满足
\begin{equation}\sum_{t=1}^{\infty} \eta_t = \infty \qquad\text{和}\qquad \sum_{t=1}^{\infty} \eta_t^2 \bigg/ \sum_{t=1}^{\infty} \eta_t = 0\end{equation}
这就得到了关于学习率策略的两个经典条件。此外，式$\eqref{leq:avg-3}$不显式依赖于终点学习率$\eta_T$，从而也允许$\eta_T\to 0$，因此我们可以更灵活调度学习率。总而言之，式$\eqref{leq:avg-3}$会给我们一种更“从容”的感觉，审美上也更为舒服。

## 文章小结

这篇文章我们重启了“科学炼丹”专题，将上一篇文章的SGD在有界域收敛的结论推广到了无界域，得到了更丰富的结果。
