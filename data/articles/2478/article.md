# 一本对称闯物理：相对论力学(一)

> 作者：苏剑林 · 科学空间 · 2014-03-19
>
> 原文：<https://spaces.ac.cn/archives/2478>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>简单说说</strong>

[![《可畏的对称》](<https://spaces.ac.cn/usr/uploads/2014/03/411546316.jpg>)](<https://spaces.ac.cn/usr/uploads/2014/03/411546316.jpg>)

《可畏的对称》

笔者最近陶醉于从李对称的角度来理解力学和场论，并且计算得到一些比较有趣的结果，遂想在此与大家分享。我发现，仅仅需要一个描述对称的无穷小生成元和一些最基本的假设，几乎就可以完成地推导出整个相对论力学来，甚至推导出整个（经典）场论理论来。这确实是不可思议的，我现在能基本体会到当年徐一鸿大师写的《可畏的对称》的含义了。对称的威力如此之大，以至于我们真的不得不敬畏它。而在构思本文题目的时候，我也曾想到过用“可畏的对称”为题，但不免有抄袭和老套之嫌。后来想到曾有一部漫画叫《一本漫画闯天涯》，遂将“漫画”改成“对称”，“天涯”改成“物理”，似乎也能表达我对“对称”的感觉。

对称就是在某种变换下保持不变的性质，比如狭义相对论要求所有物理定律在所有惯性系中保持不变，这相对于要求描述物理定律的方程在匀速运动的坐标变换下保持不变，结合光速不变的要求，我们就可以推导出洛伦兹变换，从而完成地描述了狭义相对论里边的对称。然而，并不是任何时候都可以想推导洛伦兹变换那样，能够把一个完整的变换推导出来的。幸好，李对称的不需要完整的对称描述，它只需要“无穷小变换”（意味着我们可以忽略掉高阶项），对应地产生一个“无穷小生成元”，用这个无穷小生成元，就足以完整构建出我们所需要的物理来。这种“无穷小”决定“广泛”、“局部”决定“全局”的奇妙至今仍让我觉得不可思议。（关于李对称、无穷小生成元的基本概念，不妨先阅读：[《求解微分方程的李对称方法》](<https://spaces.ac.cn/archives/2185/>)）

<strong>狭义相对论的生成元</strong>

爱因斯坦说所有物理定律在所有惯性系中保持不变，这实际暗示着一个单参数变换群（当然，这个群在牛顿力学中也有），因为速度是任意的（现在还没有谈到光速不变），所以这是一个以速度为参数的单参数变换群。我们考虑我们是一个惯性系$(x,t)$，并且存在一个以无穷小的速度$\epsilon$离我们远去的惯性系$(x',t')$，伽利略变换告诉我们
$$x'=x+\epsilon t$$
我们知道在狭义相对论中这不正确。但是假设我们并不知道正确的变换是什么，只知道以上变换是需要修正的。但是无论如何，上述变换已经经过了大量的“考验”，因此我们认为，至少在精确到“一阶”时，它是正确的。

狭义相对论还要求光速不变，这是对$t'=t$的一个修正。这相当于要求
$$\frac{dx'}{dt'}=c\Leftrightarrow=\frac{dx}{dt}=c$$
类比$x'=x+\epsilon t$，我们只能假设
$$t'=t+\epsilon kx$$
其中$k$是常数，即
$$\begin{aligned}
c=\frac{dx'}{dt'}&=\frac{dx+\epsilon dt}{dt+\epsilon kdx}=\frac{dx/dt+\epsilon }{1+\epsilon kdx/dt}\\
&=\frac{c+\epsilon }{1+\epsilon kc}\approx (c+\epsilon )(1-\epsilon kc)\\
&\approx c+\epsilon(1-kc^2)\end{aligned}$$
这说明在一阶的近似下，要求$1-kc^2=0,k=1/c^2$，即
$$t'=t+\epsilon \frac{x}{c^2}$$

于是我们找到了对应狭义相对论所要求的对称性的无穷小生成元
$$X=\frac{x}{c^2}\frac{\partial}{\partial t}+t\frac{\partial}{\partial x}$$

读者可能会有这样的想法：$\frac{1}{c^2}$也是无穷小量，应当舍去！但这是不对的，因为$\frac{1}{c^2}$是“小量”而不是“无穷小量”。我们说$\epsilon$是无穷小量，是因为它表示的是参考系的相对速度，它确实可以任意小。而在选定了单位长度之后，$\frac{1}{c^2}$就有确定的值了，它不能任意小。因此，这个项是不能忽略的。换句话说，我们觉得$\frac{1}{c^2}$小，只不过选取的单位不对而已，如果选取“光秒”为单位，那么c=1，此时$\frac{1}{c^2}=1$，很明显就“不小”了。

<strong>一阶延拓和二阶延拓</strong>

力学定律通常都包含着速度和加速度这些变量，因此，上述只包含位移和时间的生成元还不足够，我们需要将其进行延拓，至少延拓至位移对时间的二阶导数。

我们有（记$\frac{dx}{dt}=\dot{x}$）
$$\begin{aligned}
\frac{dx'}{dt'}&=\frac{dx+\epsilon dt}{dt+\epsilon dx/c^2}=\frac{\dot{x}+\epsilon }{1+\epsilon \dot{x}/c^2}\\
&\approx (\dot{x}+\epsilon )(1-\epsilon \dot{x}/c^2)\\
&\approx \dot{x}+\epsilon(1-\dot{x}^2/c^2)\end{aligned}$$
这说明一阶延拓是
$$X^{(1)}=\frac{x}{c^2}\frac{\partial}{\partial t}+t\frac{\partial}{\partial x}+\left(1-\frac{\dot{x}^2}{c^2}\right)\frac{\partial}{\partial \dot{x}}$$

同理
$$\begin{aligned}
\frac{d^2 x'}{dt'^2}&=\frac{d[\dot{x}+\epsilon(1-\dot{x}^2/c^2)]}{dt+\epsilon dx/c^2}=\frac{\ddot{x}-2\epsilon \dot{x}\ddot{x}/c^2 }{1+\epsilon \dot{x}/c^2}\\
&\approx (\ddot{x}-2\epsilon \dot{x}\ddot{x}/c^2 )(1-\epsilon \dot{x}/c^2)\\
&\approx \ddot{x}-3\epsilon \dot{x}\ddot{x}/c^2\end{aligned}$$
这说明二阶延拓是
$$X^{(2)}=\frac{x}{c^2}\frac{\partial}{\partial t}+t\frac{\partial}{\partial x}+\left(1-\frac{\dot{x}^2}{c^2}\right)\frac{\partial}{\partial \dot{x}}-\frac{3\dot{x}\ddot{x}}{c^2}\frac{\partial}{\partial \ddot{x}}$$
下面我们将会使用这些算子，来将狭义相对论力学构造出来。当然，这放到下一篇文章中，现在姑且留个悬念。
