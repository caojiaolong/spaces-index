# 除了交叉熵，LM Loss还有什么选择？

> 作者：苏剑林 · 科学空间 · 2026-08-09
>
> 原文：<https://spaces.ac.cn/archives/11854>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

一直以来，交叉熵（Cross Entropy）都是LLM预训练和微调的标准损失函数。那这个“标准”可以改吗？如果想改，那又有哪些选择呢？改完之后又会带来什么影响呢？

可能很多读者从未认真推敲过这些问题。一方面，交叉熵简洁有效，又有信息论诠释作为背书，让我们觉得它非常“理所当然”，以至于欣然接受；另一方面，换损失函数是“牵一发而动全身”的事情——换了之后意味着所有基于损失的比较都不再有效，我们只能去比下游任务效果了，工程量太大。

然而，“理所当然”不等于“别无选择”，将背后的原理思考清楚，不仅有助于我们更好地理解模型的优化过程，也可能为提升效果带来新的改进视角。

## 分析

更准确地说，交叉熵是分类问题的标准损失函数，而LLM的训练看上去像是逐Token的分类问题，所以沿用了交叉熵损失。这样看来，只要我们换用别的分类损失函数就行了？

很遗憾，并不对。跟常规分类问题不同，自然语言的规律是一对多的，比如“白切”后面不仅可以接“鸡”，还可接“鸭”、“狗”、“羊”等，所以我们要建模的是完整的分布，而不只是预测单个标签。换句话说，我们需要把“白切”后面接“鸡”、“鸭”、“狗”、“羊”的概率都估计出来，而不单单给出一个正确答案——正确答案并不唯一。

而这里的核心难题在于，训练语料是“零散”地送入模型中的，这次来了个“白切鸡”，下次可能来个“白切鸭”，后面可能再来个“白切鸡”，我们无法事先统计出完整的频率分布来，这就要求损失函数具有转变为采样形式的能力。用数学的话说，这要求损失函数关于目标分布$\boldsymbol{p}$是线性的，即
\begin{equation}\newcommand{argmin}{\mathop{\text{argmin}}} L(\boldsymbol{p}, \boldsymbol{q}) = \sum_{i=1}^n p_i S(\boldsymbol{q}, i) = \mathbb{E}_{i\sim\boldsymbol{p}}[S(\boldsymbol{q}, i)]\qquad\text{s.t.}\qquad \boldsymbol{p} = \argmin_{\boldsymbol{q}\in\Delta^{n-1}} L(\boldsymbol{p}, \boldsymbol{q})\label{eq:obj}\end{equation}
其中$\boldsymbol{p}=(p_1,p_2,\cdots,p_n),\boldsymbol{q}=(q_1,q_2,\cdots,q_n)$分别代表目标分布和预测分布，$\text{s.t.}$后的条件表明固定$\boldsymbol{p}$后，$L(\boldsymbol{p}, \boldsymbol{q})$的最小值点是$\boldsymbol{q}^*=\boldsymbol{p}$，这是损失函数的基本要求。线性约束排除了很多常见的概率度量，比如[Total Variation](<https://en.wikipedia.org/wiki/F-divergence#Common_examples_of_f-divergences>)：
\begin{equation}TV(\boldsymbol{p}, \boldsymbol{q}) = \sum_{i=1}^n |p_i - q_i| = \sum_{i=1}^n p_i\left|1 - \frac{q_i}{p_i}\right| = \mathbb{E}_{i\sim \boldsymbol{p}}\left[\left|1 - \frac{q_i}{p_i}\right|\right]\end{equation}
为了采样估计它，我们需要计算$|1 - q_i/p_i|$，然而$p_i$我们是无法预知的，所以Total Variation无法改造成用于LLM训练的形式。

## 推导

现在我们来求解目标$\eqref{eq:obj}$。根据损失函数的基本要求，固定$\boldsymbol{p}$后，$L(\boldsymbol{p}, \boldsymbol{q})$的最小值是$H(\boldsymbol{p})\triangleq L(\boldsymbol{p}, \boldsymbol{p})$，于是我们可以写出
\begin{equation}L(\boldsymbol{p}, \boldsymbol{q})\geq H(\boldsymbol{p})\end{equation}
现在我们将注意力集中在变量$\boldsymbol{p}$上，显然$L(\boldsymbol{p}, \boldsymbol{q})$关于$\boldsymbol{p}$是线性的，所以当我们固定$\boldsymbol{q}$时，左端描述了一个超平面，右端则描述了一个超曲面，它们在$\boldsymbol{p}=\boldsymbol{q}$处相交。如果我们进一步假设或者说要求最小值点是唯一的，那么相交实际上就是相切，即$L(\boldsymbol{p}, \boldsymbol{q})$是$H(\boldsymbol{p})$在$\boldsymbol{p}=\boldsymbol{q}$处的切平面！

由于上述不等式是恒成立的，所以这等价于说$H(\boldsymbol{p})$总在其切平面的下方，这正好是凹函数的定义！所以我们可以确定$H(\boldsymbol{p})$是一个凹函数。反过来，任选一个关于$\boldsymbol{p}$的凹函数$H(\boldsymbol{p})$，它在$\boldsymbol{p}=\boldsymbol{q}$处的切平面为
\begin{equation}H(\boldsymbol{q}) + (\boldsymbol{p}-\boldsymbol{q})\cdot\nabla_{\boldsymbol{q}} H(\boldsymbol{q}) = \boldsymbol{p}\cdot\big[H(\boldsymbol{q}) + \nabla_{\boldsymbol{q}} H(\boldsymbol{q}) - \boldsymbol{q}\cdot\nabla_{\boldsymbol{q}} H(\boldsymbol{q})\big]\end{equation}
其中等号利用上了$\boldsymbol{p},\boldsymbol{q}\in\Delta^{n-1}$这一约束（各分量之和为1），“$\cdot$”表示内积，向量加标量按Element\-wise相加处理。根据前面的推导，上式便是我们期望的$L(\boldsymbol{p}, \boldsymbol{q})$，于是可以直接读出
\begin{align}S(\boldsymbol{q},i) =&\, H(\boldsymbol{q}) + \partial_i H(\boldsymbol{q}) - \boldsymbol{q}\cdot\nabla_{\boldsymbol{q}} H(\boldsymbol{q}) \label{eq:S-q-i-1} \\[5pt]
=&\, H(\boldsymbol{q}) + (\boldsymbol{e}_i - \boldsymbol{q})\cdot\nabla_{\boldsymbol{q}} H(\boldsymbol{q})\label{eq:S-q-i-2}\end{align}
这便是$S(\boldsymbol{q},i)$的一般形式（允许加减一个常数以及乘一个正常数），其中$\partial_i H(\boldsymbol{q})$表示$\nabla_{\boldsymbol{q}} H(\boldsymbol{q})$的第$i$个分量，$\boldsymbol{e}_i$是第$i$位为1的One Hot向量。不难看出$S(\boldsymbol{q},i)$关于$H(\boldsymbol{q})$是线性的，并且两个凹函数的线性插值依然是凹函数，所以两个评分函数的线性插值也还是一个评分函数。

## 评分

上述结果其实有一个专门的名字，叫做[恰当评分规则（Proper Scoring Rules）](<https://en.wikipedia.org/wiki/Scoring_rule>)，这个名字怎么来的我们就不追溯了，下面给出几个经典的例子：
$$\newcommand{\rs}{\rule[-1.2ex]{0pt}{3.5ex}}
\begin{array}{c|c|c}
\hline
\rs\text{名称} & H(\boldsymbol{p}) & S(\boldsymbol{q},i) \\
\hline
\rs\text{对数评分 (交叉熵)} & -\sum_i p_i\log p_i & -\log q_i \\
\hline
\rs\text{Brier评分 (平方损失)} & 1-\sum_i p_i^2 & \|\boldsymbol{q}-\boldsymbol{e}_i\|^2 \\
\hline
\rs\text{Tsallis评分 }(\alpha > 0) & \frac{1-\sum_i p_i^\alpha}{\alpha-1} & \sum_j q_j^\alpha - \frac{\alpha}{\alpha-1}q_i^{\alpha-1} + \frac{1}{\alpha-1} \\
\hline
\rs\text{球面评分 }(\alpha > 0) & \frac{1-\Vert\boldsymbol{p}\Vert_\alpha}{\alpha-1} & \frac{1}{\alpha-1}\left(1 -\frac{q_i^{\alpha-1}}{\Vert\boldsymbol{q}\Vert_\alpha^{\alpha-1}}\right) \\
\hline
\rs\text{Rényi评分 }(0 < \alpha < 1) & \frac{1}{1-\alpha}\log\sum_i p_i^\alpha & \frac{1}{1-\alpha}\left(\log\sum_j q_j^\alpha+\alpha \frac{q_i^{\alpha-1}}{\sum_j q_j^\alpha}-\alpha\right) \\
\hline
\end{array}$$

值得指出的是，后三个评分函数都在$\alpha\to 1$时退化成对数评分，这意味着它们都是交叉熵的某种推广。特别地，如果进一步要求$S(\boldsymbol{q},i)$只依赖于$q_i$，即$S(\boldsymbol{q},i)=S(q_i)$，那么只有交叉熵$-\log q_i$这一个选择。这个不难证明，此时$H(\boldsymbol{q}) = \sum_i q_i S(q_i)$，代入式$\eqref{eq:S-q-i-1}$得
\begin{equation}\require{cancel}\cancel{S(q_i)} = \cancel{S(q_i)} + q_i S'(q_i) - \sum_j q_j^2 S'(q_j)\end{equation}
其中$\sum_j q_j^2 S'(q_j)$这一项对单个$q_i$来说也相当于常数，所以这个方程等价于$q_i S'(q_i) = -c$，容易解得$S(q_i) = - c \log q_i$，这就得到了对数评分。

> <strong>注：</strong>这里的论述其实稍欠严谨性。由于约束$\sum_i q_i = 1$的存在，“只依赖于$q_i$”这件事本身就不是那么朴素，比如$S(q_n) = S(1 - q_1 - \cdots - q_{n-1})$，我们不能简单地声称$S(q_n)$只依赖于$q_n$。
> 
> 
> 
> 类似的困惑也出现在“$\sum_j q_j^2 S'(q_j)$是否依赖于$q_i$”的论证中，这里相对严谨的表述是，只要求前$n-1$个$q_i$满足$q_i S'(q_i) = \sum_j q_j^2 S'(q_j)$，$q_n$由$q_n=1 - q_1 - \cdots - q_{n-1}$进行消元，这样就只剩下$n-1$个相对独立的变量，$q_i S'(q_i) = \sum_j q_j^2 S'(q_j)$意味着前$n-1$个$q_i S'(q_i)$等于同一个包含$q_1,\cdots,q_{n-1}$的式子，但$q_i S'(q_i)$又至多依赖于$q_i$，所以这个式子只能是常数，于是得到$q_i S'(q_i) = -c$。

这些评分函数都能推广到连续分布，只不过是将离散变量$i$换成连续变量$\boldsymbol{x}$，将求和换成积分。但连续分布的困难往往是无法计算归一化因子，这些评分函数都需要显式的概率密度，所以通常不那么“好用”，这种场景下更需要不依赖归一化因子的评分函数（通常要借助梯度，这也是连续分布特有的），但我们就不展开了。

## 梯度

刚才说到，后三个评分函数都是交叉熵的推广，那直觉来想，只要我们精调一下$\alpha$，就有机会在下游任务中取得提升？然而，事情没那么简单。

一般情况下，模型只能预测无界的Logits向量$\boldsymbol{z}\in\mathbb{R}^n$，我们需要加一个激活函数，才能将它投影成概率分布$\boldsymbol{q}$，而激活函数通常的选择便是Softmax。由于我们只能用基于梯度的优化器，那么损失函数关于$\boldsymbol{z}$的凸性和梯度性质便显得尤其重要。以对数评分和Brier评分为例，在Softmax激活函数下损失对$\boldsymbol{z}$的梯度分别是
\begin{align}\newcommand{diag}{\mathop{\text{diag}}}
\text{对数评分 (交叉熵) :}&\qquad \nabla_{\boldsymbol{z}} S(\boldsymbol{q}, i) = \boldsymbol{q} - \boldsymbol{e}_i \\[5pt]
\text{Brier评分 (平方损失) :}&\qquad \nabla_{\boldsymbol{z}} S(\boldsymbol{q}, i) = 2(\diag(\boldsymbol{q})-\boldsymbol{q}\boldsymbol{q}^\top)(\boldsymbol{q}-\boldsymbol{e}_i) \\
\end{align}
显然，交叉熵的梯度看上去更加“干净”，当且仅当$\boldsymbol{q}=\boldsymbol{e}_i$时梯度才为零，这说明只要还没达到目标，它就能提供有效梯度，并且距离目标越远，梯度越大；平方损失多出一项变换$\diag(\boldsymbol{q})-\boldsymbol{q}\boldsymbol{q}^\top$，当$\boldsymbol{q}=\boldsymbol{e}_j\neq \boldsymbol{e}_i$时，这一项也会为零，说明当模型“自信地错误”时，它也会梯度消失。

这个特性是双面的：前中期模型大部分预测都不准确，这意味着用平方损失学习效率极低；但后期模型已经基本稳定，这时候依然还“自信地错误”的可能是极难或者错误的样本，跳过它们可能会更利于整体效果。所以，交叉熵学习效率更高，应当作为主损失，但平方损失有着更好的抵御噪声的能力，后期可以尝试。

这些也可以用$S(\boldsymbol{q},i)$关于$\boldsymbol{z}$的凸性来理解。可以证明，在Softmax激活下：交叉熵关于$\boldsymbol{z}$是凸的，这意味着最优点是唯一的，任意处的梯度都指向目标点$\boldsymbol{e}_i$；但平方损失关于$\boldsymbol{z}$是非凸的，这些良好性质都不保证存在，模型可能会进入“既错误又走不出去”的饱和困境。

当然，即便对$\boldsymbol{z}$是凸的，但$\boldsymbol{z}$还有参数，而在深度模型中损失函数对参数一般只能是非凸的，我们要求损失函数对$\boldsymbol{z}$的凸性，更多是出于不额外给模型“添乱”的考虑——深度学习的优化已经够难了，没必要在最后一层再增加障碍。

## 反推

上一节的结果都有个前提——Softmax激活——但如果不是Softmax激活呢？或者反过来，对于给定的评分函数，我们能不能推导出适用于该评分函数的“最优”的激活函数呢？

首先要思考的问题是：“最优”该如何定义呢？参考“Softmax\+交叉熵”组合，它的优点有两个，一是对$\boldsymbol{z}$的凸性，二是梯度$\nabla_{\boldsymbol{z}} S(\boldsymbol{q}, i) = \boldsymbol{q} - \boldsymbol{e}_i$比较“干净”。实际上第二点更强也更实用，所以我们从第二点出发考虑。即我们希望寻找一个变换$\boldsymbol{q} = \sigma(\boldsymbol{z})\in\Delta^{n-1}$，使得对于给定$S(\boldsymbol{q}, i)$，成立
\begin{equation}\nabla_{\boldsymbol{z}} S(\boldsymbol{q}, i) = \boldsymbol{q} - \boldsymbol{e}_i\end{equation}
注意$\boldsymbol{e}_i= \nabla_{\boldsymbol{z}} z_i$，所以上式也可以写成$\nabla_{\boldsymbol{z}} (S(\boldsymbol{q}, i) + z_i) = \boldsymbol{q}$，这就表明，存在某个跟$i$无关的标量函数$\Phi(\boldsymbol{z})$，使得
\begin{equation}\Phi(\boldsymbol{z}) = S(\boldsymbol{q}, i) + z_i,\qquad \boldsymbol{q} = \nabla_{\boldsymbol{z}}\Phi(\boldsymbol{z})\end{equation}
移项得$\Phi(\boldsymbol{z}) - z_i = S(\boldsymbol{q}, i)$，两端乘$p_i$然后求和得
\begin{equation}\Phi(\boldsymbol{z}) - \boldsymbol{p}\cdot\boldsymbol{z} = L(\boldsymbol{p},\boldsymbol{q})\geq H(\boldsymbol{p})\end{equation}
继续移项得$\Phi(\boldsymbol{z}) \geq \boldsymbol{p}\cdot\boldsymbol{z} + H(\boldsymbol{p})$，这是对任意$\boldsymbol{p}$都成立，所以$\Phi(\boldsymbol{z})$是全体$\boldsymbol{p}\cdot\boldsymbol{z} + H(\boldsymbol{p})$的上界，然后代入$\boldsymbol{p}=\boldsymbol{q}$到上式得$\Phi(\boldsymbol{z}) - \boldsymbol{q}\cdot\boldsymbol{z} = L(\boldsymbol{q},\boldsymbol{q}) = H(\boldsymbol{q})$，即$\boldsymbol{p}=\boldsymbol{q}$时能取到等号，所以$\Phi(\boldsymbol{z})$是全体$\boldsymbol{p}\cdot\boldsymbol{z} + H(\boldsymbol{p})$的“上确界”，即
\begin{equation}\newcommand{argmax}{\mathop{\text{argmax}}}\Phi(\boldsymbol{z}) = \max_{\boldsymbol{p}\in\Delta^{n-1}} \boldsymbol{p}\cdot \boldsymbol{z} + H(\boldsymbol{p}),\qquad \boldsymbol{q} = \argmax_{\boldsymbol{p}\in\Delta^{n-1}} \boldsymbol{p}\cdot \boldsymbol{z} + H(\boldsymbol{p})\end{equation}
这正好是凸函数$-H(\boldsymbol{p})$的[凸共轭](<https://en.wikipedia.org/wiki/Convex_conjugate>)，而这几节的内容加起来，正好是经典的[Fenchel\-Young Losses](<https://papers.cool/arxiv/1901.02324>)框架。

## 激活

这一节我们同样把前面提到的几个评分函数对应的最优激活函数推一推。不难发现，它们对应的$H(\boldsymbol{q})$都是同一个结构$g(\sum_i q_i^{\alpha})$，所以我们可以统一求解。设$t = \sum_i q_i^{\alpha}$，那么$H(\boldsymbol{q})=g(t)$，其中
\begin{equation}g(t)=\frac{1-t}{\alpha-1}\ (\text{Tsallis/Brier}),\quad g(t)=\frac{1-t^{1/\alpha}}{\alpha-1}\ (\text{球面}),\quad g(t)=\frac{\log t}{1-\alpha}\ (\text{Rényi}) \end{equation}
引入拉格朗日函数$\boldsymbol{q}\cdot \boldsymbol{z} + H(\boldsymbol{q}) - \lambda(\sum_i q_i - 1)$，对$q_i$求导并让它等于0得
\begin{equation}z_i + \alpha g'(t) q_i^{\alpha-1} = \lambda \qquad\Rightarrow\qquad q_i^{\alpha-1} = \frac{\lambda - z_i}{\alpha g'(t)}\end{equation}
对单个分量来说，$\lambda$和$\alpha g'(t)$都是共用的“常数”，我们就是要调节这两个常数，使$\boldsymbol{q}$成为一个适合的分布。注意$q_i$只有$q_i > 0$和$q_i = 0$两种可能，后者是平凡的，所以我们只需分析前者。又留意到$\alpha > 1$时有$g'(t) < 0$，$\alpha < 1$时，$g'(t) > 0$，因此可以写出
\begin{equation}q_i = \left\{\begin{aligned}
&\, e^{z_i - \lambda},&\, \alpha \to 1 \\
&\,\left[\frac{z_i - \lambda}{-\alpha g'(t)}\right]_+^{\frac{1}{\alpha-1}},&\, \alpha \neq 1 \\
\end{aligned}\right.\end{equation}
其中$[x]_+ = \max(x, 0)$，但$\alpha < 1$时该截断是多余的，因为此时指数小于0，零的负幂没有意义，所以截断必然不生效，这也说明$\alpha > 1$对应稀疏分布，$\alpha < 1$则是稠密分布，至于$\lambda$由方程$\sum_i q_i = 1$决定。显然，$\alpha\to 1$时正是经典的Softmax，当$\alpha\neq 1$时，对于Tsallis评分有$g'(t)=1/(1-\alpha)$，于是
\begin{equation}q_i = \left[\frac{\alpha-1}{\alpha}(z_i - \lambda)\right]_+^{\frac{1}{\alpha-1}}\end{equation}
$\lambda$可以通过条件$\sum_i q_i = 1$用二分法求解。当$\alpha=2$时，结果正是[Sparsemax](<https://papers.cool/arxiv/1602.02068>)，其余情况则称为[Entmax\-$\alpha$](<https://papers.cool/arxiv/1905.05702>)，当$\alpha=2$和$\alpha=1.5$时，$\lambda$有比二分法更高效的精确解法，这些我们在[《通向概率分布之路：盘点Softmax及其替代品》](<https://spaces.ac.cn/archives/10145>)也曾介绍过。至于其他评分的结果稍微复杂一些，就留给大家尝试了。

## 小结

本文从“学习分布”和“允许采样”两个角度，推导出了LM Loss的一般构造方式。随后，我们结合预测分布所用的激活函数，计算这些Loss的梯度和凸性，以此简单判断了它们的优劣。最后，我们尝试从给定Loss出发，反推出其配套的最优激活函数，而交叉熵的最优激活函数正是Softmax——这解释了为什么这二者几乎总是配套出现。
