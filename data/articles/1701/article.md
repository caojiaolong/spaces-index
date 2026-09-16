# 电偶极子浅探（2）

> 作者：苏剑林 · 科学空间 · 2012-08-18
>
> 原文：<https://spaces.ac.cn/archives/1701>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在上一篇文章中，我们已经得到了电偶极子的等势面和电场线方程，这应该可以让我们对电偶极子的力场情况有个大致的了解了。当然，我们还是希望能够求出在这样的一个受力情况下，一个带电粒子是如何运动的。简单起见，在下面的探讨中，我们假定带电粒子的质量和电荷量均为1，至于电荷的正负，可以通过改变在$U=-\frac{k \cos\theta}{r^2}$中的k值的正负来控制。我们使用的工具依旧是理论力学中的欧拉\-拉格朗日方程。

也许不少读者始终对公式感到头疼，更不用说是博大精深的理论力学了。但是请相信我，如果你花一点点心思去弄懂用变分法研究力学（或其他物理系统，但我目前只会用于力学）的基本思路和步骤，那么对你的物理研究是大有裨益的。因为在我眼中，学习了一丁点的理论力学知识后，我看到的只有物理的简洁与和谐。有兴趣的朋友可以看看我的那几篇《自然极值》等相关文章。

首先写出动能的表达式：$T=\frac{1}{2} (\dot{r}^2+r^2 \dot{\theta}^2)$

还有势能：$U=-\frac{k \cos\theta}{r^2}$

拉格朗日量：$L=T-U=\frac{1}{2}(\dot{r}^2+r^2 \dot{\theta}^2)+\frac{k \cos\theta}{r^2}$

求关于r的微分方程：
$$\begin{aligned}\frac{\partial L}{\partial \dot{r}}=\dot{r} \\ \frac{\partial L}{\partial r}=r\dot{\theta}^2-\frac{2kcos\theta}{r^3}\end{aligned}$$

我们得到：
$$\ddot{r}=r\dot{\theta}^2-\frac{2kcos\theta}{r^3}\tag{10}$$
继续对θ求偏导数我们可以求出关于θ的微分方程，但是我们不这样做，因为保守力场中必然存在着能量守恒：
$$T+U=h=\frac{1}{2}(\dot{r}^2+r^2 \dot{\theta}^2)-\frac{k \cos\theta}{r^2}\tag{11}$$
用(10)和(11)联立就可以得出所有的解。原题本质上是关于两个函数的二阶微分方程，所以应该包含4个积分常数，而(11)式已经包含了一个积分常数，所以还剩下三个。

用心摆弄一会儿，我们会发现一件很巧的事情：
$$h=\frac{1}{2}\dot{r}^2+\frac{1}{2}(r^2 \dot{\theta}^2-\frac{2k \cos\theta}{r^2})=\frac{1}{2}(\dot{r}^2+r\ddot{r})\tag{12}$$
即$$h=\frac{1}{2}(\dot{r}^2+r\ddot{r})\tag{12}$$
太巧了，而且这个结果也太简洁了！乘胜追击，得到
$$\begin{aligned}r\dot{r}=2ht+C_1 \\ \frac{1}{2}r^2=ht^2+C_1 t+C_2\end{aligned}\tag{13}$$
至此，我们已经得到了三个积分常数。让我们用点心思分析一下这个解。从(13)式可以知道，$C_2 >0$（一开始不能就在原点）。如果 h \> 0 ，那么带电粒子最终将会飞到无穷远处，同时这必然要求了$\Delta=C_1^2-4hC_2 \leq 0$；如果 h \< 0 ，该粒子最终会回到原点，撞到电偶极子上。这也符合我们的知识，具有正能量的系统可以脱离束缚，具有负能量的系统无法脱离束缚。当h为0时，又需要进一步讨论。

剩下最后一个积分常数，我们已经求出了r关于时间t的演化，能量守恒也给出了速度的演化，最后一个是什么呢？答案有很多，但我们有兴趣的是——轨道的形状是什么？以下是简单的过程

$$\begin{aligned}\dot{r}^2+r^2 \dot{\theta}^2=2h+\frac{2kcos\theta}{r^2} \\ 1+r^2(\frac{d\theta}{dr})^2=\frac{2h+\frac{2kcos\theta}{r^2}}{\dot{r}^2}\end{aligned}\tag{14}$$
下面想办法将(14)式中的$\dot{r}^2$消去，根据(13)，我们有
$$\begin{aligned}r^2 \dot{r}^2=4h^2 t^2+C_1^2+4hC_1 t \\ 2hr^2=4h^2 t^2 +4hC_1 t+4hC_2 \\ r^2 \dot{r}^2-C_1^2=2hr^2-4hC_2 \\ \dot{r}^2=2h+\frac{C_1^2-4hC_2}{r^2}\end{aligned}$$

代入

$$\begin{aligned}1+r^2(\frac{d\theta}{dr})^2=\frac{2hr^2+2kcos\theta}{2hr^2+C_1^2-4hC_2} \\ r^2(\frac{d\theta}{dr})^2=\frac{2kcos\theta-C_1^2+4hC_2}{2hr^2+C_1^2-4hC_2} \\ (\frac{d\theta}{dr})^2=\frac{2kcos\theta-C_1^2+4hC_2}{(2hr^2+C_1^2-4hC_2)r^2} \\ \frac{dr}{r\sqrt{2hr^2+C_1^2-4hC_2}}=\frac{d\theta}{\sqrt{2kcos\theta-C_1^2+4hC_2}}\end{aligned}\tag{15}$$
最后一个式子已经包含了轨道的形状在内了，两边同时积分即可。很遗憾，这不是一个简单的显式解，它只能够用椭圆积分来表达。一个容易想到的法子是有没有一些什么简单的特例，让我们可以快速看看曲线的形状？好吧，我们令$k=0.5,-C_1^2+4hC_2=1,h=0.5$，那么(15)就变成
$$\frac{dr}{r\sqrt{r^2-1}}=\frac{d\theta}{\sqrt{\cos\theta+1}}$$

得到$arccos(\frac{1}{r})=\frac{1}{\sqrt{2}}ln(tan(\frac{\pi}{4}+\frac{\theta}{4}))+Const$

也就是说$r=\frac{\sqrt{2}}{cos(ln(tan(\frac{\pi}{4}+\frac{\theta}{4})))}$

这些曲线的形状让我也非常头疼\.\.\.由于对相关知识研究还不够深入，不再多说\.\.\.权当设置一个悬念，希望有兴趣的读者继续完成\.
