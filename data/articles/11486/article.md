# 为什么DeltaNet要加L2 Normalize？

> 作者：苏剑林 · 科学空间 · 2025-12-23
>
> 原文：<https://spaces.ac.cn/archives/11486>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《线性注意力简史：从模仿、创新到反哺》](<https://spaces.ac.cn/archives/11033>)中，我们介绍了DeltaNet，它把Delta Rule带进了线性注意力中，成为其强有力的工具之一，并构成[GDN](<https://papers.cool/arxiv/2412.06464>)、[KDA](<https://papers.cool/arxiv/2510.26692>)等后续工作的基础。不过，那篇文章我们主要着重于DeltaNet的整体思想，并未涉及到太多技术细节——这篇文章我们来讨论其中之一：DeltaNet及其后续工作都给$\boldsymbol{Q}、\boldsymbol{K}$加上了L2 Normalize，这是为什么呢？

当然，直接从特征值的角度解释这一操作并不困难，但个人总感觉还差点意思。前几天笔者在论文[《Error\-Free Linear Attention is a Free Lunch: Exact Solution from Continuous\-Time Dynamics》](<https://papers.cool/arxiv/2512.12602>)学习到了一个新理解，感觉也有可取之处，特来分享一波。

## 基本解释

DeltaNet的递归格式是
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1} - \eta_t (\boldsymbol{S}_{t-1} \boldsymbol{k}_t - \boldsymbol{v}_t)\boldsymbol{k}_t^{\top} = \boldsymbol{S}_{t-1}(\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \eta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\label{eq:delta}\end{equation}
从TTT的角度看，这是用SGD优化器、以$\eta_t$的学习率对损失$\frac{1}{2}\Vert\boldsymbol{S}\boldsymbol{k} - \boldsymbol{v}\Vert^2$做在线优化（训练参数是$\boldsymbol{S}$）。我们知道优化器往往对学习率比较敏感，尤其是SGD这种非自适应学习率优化器，而在DeltaNet中则表现为对转移矩阵$\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{k}_t^{\top}$的一些额外要求。

具体来说，由于不同时间的转移矩阵在递归过程中是连乘起来的，所以为了避免数值爆炸，转移矩阵不能出现大于1或小于\-1的特征值。而对于矩阵$\boldsymbol{I} - \eta_t \boldsymbol{k}_t\boldsymbol{k}_t^{\top}$来说，它的特征值有一个是$1 - \eta_t\Vert\boldsymbol{k}_t\Vert^2$、剩下都是1（请证明这一点），由此我们可以得到约束
\begin{equation}-1 \leq 1 - \eta_t\Vert\boldsymbol{k}_t\Vert^2 \leq 1\label{eq:cond}\end{equation}
为了实现该约束，常见做法是给$\boldsymbol{k}_t$加L2 Normalize、给$\eta_t$加Sigmoid，这样全体特征值就都在$(0, 1]$内了，这便是$\boldsymbol{K}$加L2 Normalize的来源。至于$\boldsymbol{Q}$的L2 Normalize本质上不是必要的，更多是出于对称性考虑“顺手”加上的，这跟Short Conv的情况类似，给$\boldsymbol{K}$加Short Conv才是最关键的<sup>&#91;[参考](<https://spaces.ac.cn/archives/11320>)&#93;</sup>。

## 补充说明

顺便提一下，在很长时间内，大家习惯了让特征值都在$(0, 1]$内，所以选择给$\eta_t$加Sigmoid，后来[《Unlocking State\-Tracking in Linear RNNs Through Negative Eigenvalues》](<https://papers.cool/arxiv/2411.12537>)指出负特征值能增强DeltaNet的状态跟踪能力，于是提出将DeltaNet改为
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1}(\boldsymbol{I} - 2\eta_t \boldsymbol{k}_t\boldsymbol{k}_t^{\top}) + \eta_t \boldsymbol{v}_t \boldsymbol{k}_t^{\top}\end{equation}
然后还是给$\boldsymbol{k}_t$加L2 Normalize、给$\eta_t$加Sigmoid，这样转移矩阵$\boldsymbol{I} - 2\eta_t \boldsymbol{k}_t\boldsymbol{k}_t^{\top}$的特征值范围就扩大到$(-1, 1]$了。不过，状态跟踪是一个偏向于特殊语法（比如代码）的能力，因此如果我们修改之后只是在自然语言上训练和测试，那么不见得能测出明显的变化。

还有一个要注意的细节，就是当$\eta_t=1$时，转移矩阵$\boldsymbol{I} - 2\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$是正交矩阵，理论上没问题，但实际上不行，因为出于效率考虑，我们在实现中通常都至少使用BF16计算，而BF16精度较低，导致$\boldsymbol{I} - 2\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$的特征值有概率小于\-1，在长期累乘之下依然有爆炸风险，所以还需要控制$\eta_t$不能太接近于1。

事实上，上面的解释已经很完整了，也不复杂，所以对它的挑刺主要是出于个人的审美：实现条件$\eqref{eq:cond}$的方式并不是唯一的，比如还可以像[Longhorn](<https://papers.cool/arxiv/2407.14207>)那样引入类似[Capsule](<https://spaces.ac.cn/archives/4819>)的Squash操作，因此我们无法自然地导出L2 Normalize，只能说它是一个可用的方案。

## 连续视角

接下来介绍论文[《Error\-Free Linear Attention is a Free Lunch: Exact Solution from Continuous\-Time Dynamics》](<https://papers.cool/arxiv/2512.12602>)的思路，笔者认为它也是一条很优雅的推导途径，当然这也取决于每个人的审美。它将式$\eqref{eq:delta}$看成是如下微分方程在$[t-\eta_t, t]$区间内的欧拉离散化：
\begin{equation}\frac{d}{dt}\boldsymbol{S}_t = \boldsymbol{S}_t\underbrace{(-\boldsymbol{k}_t\boldsymbol{k}_t^{\top})}_{\boldsymbol{A}_t} + \underbrace{\boldsymbol{v}_t \boldsymbol{k}_t^{\top}}_{\boldsymbol{B}_t}\label{eq:ode}\end{equation}
然后指出之所以会出现数值爆炸的异常，是因为离散化格式的精度不够高，所以提出直接用求解微分方程来构建递归，而不是近似地离散化。因为$[t-\eta_t,t]$区间内$\boldsymbol{A}_t$和$\boldsymbol{B}_t$都是常量，所以求从$t-\eta_t$到$t$的递归形式相当于解一个常系数线性微分方程，一般结果是
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-\eta_t} e^{\eta_t \boldsymbol{A}_t} + \boldsymbol{B}_t \boldsymbol{A}_t^{-1}(e^{\eta_t \boldsymbol{A}_t} - \boldsymbol{I})\label{eq:S-t-eta}\end{equation}
重新将$\boldsymbol{S}_{t+\eta_t}$换回记号$\boldsymbol{S}_{t-1}$，然后代入$\boldsymbol{A}_t,\boldsymbol{B}_t$的表达式，化简得到
\begin{equation}\boldsymbol{S}_t = \boldsymbol{S}_{t-1} \left(\boldsymbol{I} - \frac{1 - e^{-\eta_t\Vert\boldsymbol{k}_t\Vert^2}}{\Vert\boldsymbol{k}_t\Vert^2}\boldsymbol{k}_t\boldsymbol{k}_t^{\top}\right) + \frac{1 - e^{-\eta_t\Vert\boldsymbol{k}_t\Vert^2}}{\Vert\boldsymbol{k}_t\Vert^2}\boldsymbol{v}_t \boldsymbol{k}_t^{\top}\label{eq:ode-deltanet}\end{equation}
这便是我们要推的最终结果，原论文称之为“EFLA（Error\-Free Linear Attention）”，它相当于将$\eta_t$换成了$\frac{1 - e^{-\eta_t\Vert\boldsymbol{k}_t\Vert^2}}{\Vert\boldsymbol{k}_t\Vert^2}$，$\Vert\boldsymbol{k}_t\Vert^2$自然地出现在了分母中，跟$\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$相乘正好表现为对$\boldsymbol{K}$的L2 Normalize。

## 数学细节

上一节我们快速介绍了EFLA的结果，省略了很多数学细节，本节我们补充一些讨论。篇幅所限，这里也只能简略提一下推导要点，无法详细展开介绍。

上一节的核心结果是式$\eqref{eq:S-t-eta}$，它是微分方程$d\boldsymbol{S}_t/dt=\boldsymbol{S}_t \boldsymbol{A} + \boldsymbol{B}$的解，为了避免混乱，这里省掉了$\boldsymbol{A},\boldsymbol{B}$的下标，因为求解区间内它们确实也是常量。如果$\boldsymbol{B}=\boldsymbol{0}$，那么直接可以写出$\boldsymbol{S}_t=\boldsymbol{S}_0 e^{t\boldsymbol{A}}$，其中$e^{t\boldsymbol{A}}$是[矩阵指数](<https://en.wikipedia.org/wiki/Matrix_exponential>)；当$\boldsymbol{B}\neq \boldsymbol{0}$时，将方程改写成$d(\boldsymbol{S}_t + \boldsymbol{B}\boldsymbol{A}^{-1})/dt=(\boldsymbol{S}_t + \boldsymbol{B}\boldsymbol{A}^{-1})\boldsymbol{A}$，然后利用$\boldsymbol{B}=\boldsymbol{0}$时的解即得
\begin{equation}\boldsymbol{S}_t = (\boldsymbol{S}_0 + \boldsymbol{B}\boldsymbol{A}^{-1})e^{t\boldsymbol{A}} - \boldsymbol{B}\boldsymbol{A}^{-1} = \boldsymbol{S}_0 e^{t\boldsymbol{A}} + \boldsymbol{B}\boldsymbol{A}^{-1}(e^{t\boldsymbol{A}} - \boldsymbol{I})\end{equation}
最后将起点改为$t-\eta_t$、终点改为$t$，以及给$\boldsymbol{A},\boldsymbol{B}$补回下标$t$，就可以得到式$\eqref{eq:S-t-eta}$。注意最后一项出现了逆矩阵$\boldsymbol{A}^{-1}$，但实际上不要求$\boldsymbol{A}$可逆，它按照将$(e^x-1)/x$展开为幂级数后再代入$x = \boldsymbol{A}$来理解。现在再次聚焦式$\eqref{eq:S-t-eta}$，对于DeltaNet有$\boldsymbol{A}_t = -\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$是一个秩1矩阵，这提供了进一步的化简空间：
\begin{equation}f(\boldsymbol{x}\boldsymbol{y}^{\top}) = \sum_{n=0}^{\infty} a_n (\boldsymbol{x}\boldsymbol{y}^{\top})^n = a_0\boldsymbol{I} + \sum_{n=1}^{\infty} a_n (\boldsymbol{x}\boldsymbol{y}^{\top})^n = f(0)\boldsymbol{I} + \boldsymbol{x}\underbrace{\left(\sum_{n=1}^{\infty} a_n(\boldsymbol{y}^{\top}\boldsymbol{x})^{n-1}\right)}_{\frac{f(\boldsymbol{y}^{\top}\boldsymbol{x})-f(0)}{\boldsymbol{y}^{\top}\boldsymbol{x}}}\boldsymbol{y}^{\top}\end{equation}
注意$\boldsymbol{y}^{\top}\boldsymbol{x}$是一个标量，所以化简的要义是将矩阵函数变成了标量函数，由此可得
\begin{equation}e^{\eta_t \boldsymbol{A}_t} = \boldsymbol{I} - \frac{1 - e^{-\eta_t\Vert\boldsymbol{k}_t\Vert^2}}{\Vert\boldsymbol{k}_t\Vert^2}\boldsymbol{k}_t\boldsymbol{k}_t^{\top},\qquad \boldsymbol{B}_t \boldsymbol{A}_t^{-1}(e^{\eta_t \boldsymbol{A}_t} - \boldsymbol{I})=\frac{1 - e^{-\eta_t\Vert\boldsymbol{k}_t\Vert^2}}{\Vert\boldsymbol{k}_t\Vert^2}\boldsymbol{v}_t \boldsymbol{k}_t^{\top}\end{equation}

## 个人思考

到这里，我们对EFLA的介绍就结束了，原论文还有一些实验内容，显示EFLA相比原始DeltaNet有一些优势。但从式$\eqref{eq:ode-deltanet}$可以看出，EFLA仍然是DeltaNet的形式，所以原则上不能期望它会“突飞猛进”，那为什么EFLA普遍稍好一些呢？DeltaNet通过L2 Noramlize直接舍去$\boldsymbol{K}$的模长，而式$\eqref{eq:ode-deltanet}$的$\boldsymbol{v}_t \boldsymbol{k}_t^{\top}$是依赖于$\Vert\boldsymbol{k}_t\Vert$的，所以EFLA实际多了一个自由度，理论上限会更高一些。

此外，EFLA中用微分方程精确解来构造递归的做法不是新的，我们在[《重温SSM（二）：HiPPO的一些遗留问题》](<https://spaces.ac.cn/archives/10137>)中介绍SSM时就提到过，关键结果式$\eqref{eq:S-t-eta}$在[HiPPO](<https://papers.cool/arxiv/2008.07669>)中就已经出现了。EFLA主要是针对DeltaNet这个特例做了展开计算，得到了简化可用的结果。

一个更值得思考的问题是，微分方程作为出发点有什么好处？不难看出，式$\eqref{eq:ode-deltanet}$的转移矩阵特征值自动在$(0, 1]$内，也就说求解微分方程$\eqref{eq:ode}$得到的递归形式，天然有更好的稳定性。因为微分方程伴随着连续性约束，加上矩阵$-\boldsymbol{k}_t\boldsymbol{k}_t^{\top}$是一个半负定矩阵，根据微分方程的相关理论，它的解是稳定的。

数学建模上有个经典例子是Logistic方程$dx/dt = \alpha x - \beta x^2$，它的解很简单，就是Logistic函数，但对应的差分方程$x_{t+1} - x_t = \alpha x_t - \beta x_t^2$却会在某些设置下出现混沌行为（对初值极其敏感以至于不可预测）。所以，以微分方程为出发点，能自动规避一些异常行为。

## 文章小结

这篇文章围绕DeltaNet的L2 Normalize进行讨论，主要介绍了以微分方程为出发点对DeltaNet重新参数化的思路，它也可以视作DeltaNet中$\boldsymbol{K}$的L2 Normalize运算的一种解释。
