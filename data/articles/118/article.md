# 微积分学习（二）：导数

> 作者：苏剑林 · 科学空间 · 2009-09-12
>
> 原文：<https://spaces.ac.cn/archives/118>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

自从上次写了关于微积分中的极限学习后，就很长的时间没有与大家探讨微积分的学习了（估计有20多天了吧）。启事，我自己也是从今年的9月下旬才开始系统地学习微积分的，到现在也就一个月的时间吧。学习的内容有：集合、函数、极限、导数、微分、积分。不过都是一元微积分，多元的微积分正在紧张地进修中\.\.\.\.\.\.

现在不妨和大家探讨一下关于微积分中的最基本内容——“导数”的学习。

其实，用最简单的说法，如果存在函数$f(x)$，那么它的导数（一阶导数）为
$$\lim_{\Delta x->0} f'(x)=\frac{f(x+\Delta x)-f(x)}{\Delta x}$$

<strong>其中</strong>
$$\lim_{\Delta x->-0} f'(x_0)=\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$$
叫做函数$f(x)$在$x_0$处的“左导数”，而
$$\lim_{\Delta x->+0} f'(x_0)=\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}$$
叫做函数$f(x)$在$x_0$处的“右导数”。

只有当“<strong>左导数=右导数</strong>”的时候，我们才能够说函数$f(x)$在$x_0$处“<strong>可导</strong>”。

<strong>导数（一阶导数）的应用意义：</strong>

<strong>（一）瞬时速度、瞬时变化率</strong>

如果在直线运动中，路程s与时间t的关系为$s=f(t)$，那么t秒与$(t+\Delta t)$秒之间通过的路程为$f(t+\Delta t)-f(t)$，在这段时间内的平均速度为$\frac{f(t+\Delta t)-f(t)}{\Delta t}$。令$\Delta t$尽可能地少（即$\Delta t->0$），那么所得的结果（即$f'(x)$）就是在t时刻的瞬时速度。

<strong>（二）曲线的切率</strong>

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090911210650.JPG>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090911210650.JPG>)

如图，在曲线$f(x)$上，由点$(x,f(x))$和点$(x+h,f(x+h))$所确定的直线的斜率（即正切值）为
$$\frac{f(x+h)-f(x)}{h}$$

h越小，这条直线就越接近该曲线在x处的切线，相应的，$\frac{f(x+h)-f(x)}{h}$也越接近切线的斜率。于是，当$h->0$时，该直线无限接近切线，而$\frac{f(x+h)-f(x)}{h}=f'(x)$就是曲线$f(x)$在x处的切线的斜率值。

<strong>求导方法：</strong>

<strong>1、基本求导公式</strong>
[http://web\.nuist\.edu\.cn/courses/gdsx/calculus1/chap2/section2/2\.2\.4\.1\.HTM](<http://web.nuist.edu.cn/courses/gdsx/calculus1/chap2/section2/2.2.4.1.HTM>)

<u>其中，我想谈一下其中的几个基本求导公式的推导过程。</u>

<strong>1\.1 $(a^x)'=a^x ln a(a>0,a!=1)$</strong>

$$(a^x)'=\lim_{\Delta x->0} \frac{a^{x+\Delta x}-a^x}{\Delta x}=a^x \lim_{\Delta x->0} \frac{a^{\Delta x}-1}{\Delta x}$$

令$a^{\Delta x}-1=\beta<\Rightarrow a^{\Delta x}=\beta+1$，两边取对数，有$\Delta x=\frac{ln(\beta+1)}{ln a}$，故原式变为

$$\begin{aligned}(a^x)'=a^x lim_{\beta->0} \frac{\beta ln a}{ln(1+\beta)} \\ =a^x ln a lim_{\beta->0} \frac{1}{\frac{1}{\beta}ln(1+\beta)} \\ =a^x ln a lim_{\beta->0} \frac{1}{ln(1+\beta)^{\frac{1}{\beta}}}\end{aligned}$$

因为$lim_{\beta->0} (1+\beta)^{\frac{1}{\beta}}=e$，所以$lim_{\beta->0} \frac{1}{ln(1+\beta)^{\frac{1}{\beta}}}=1$
即$(a^x)'=a^x ln a$\.

<strong>1\.2 $(log_a x)'=\frac{1}{x ln a}$</strong>

$$\begin{aligned}(log_a x)'=\lim_{\Delta x->0} \frac{log_a (x+\Delta x)-log_a x}{\Delta x} \\ =\lim_{\Delta x->0} \frac{log_a (1+\frac{\Delta x}{x})}{\Delta x} \\ =\lim_{\Delta x->0} log_a (1+\frac{1}{x}\cdot \Delta x)^{\frac{1}{\Delta x}} \\ =log_a e^{(1/x)}=\frac{log_a e}{x}\end{aligned}$$

换底，即得$(log_a x)'=\frac{1}{x ln a}$

<strong>从以上的方法可以看出，“求导”实际上就是求极限的一种，如果从最基本的定义出发来推导导数，那么就需要动用变量$\Delta x$，并且令其趋于零。同时要致力于变成我们常见的极限（如e），这就需要我们去了解一些基本的极限公式。</strong>

<strong>当然，这远远不够。因为从定义出发求导数是极其麻烦的，我们必须找到一些[方便我们求导的法则](<http://zh.wikibooks.org/wiki/%E5%BE%AE%E7%A7%AF%E5%88%86%E5%AD%A6/%E5%AF%BC%E6%95%B0#.E5.AF.BC.E6.95.B0.E7.9A.84.E6.B1.82.E5.AF.BC.E6.B3.95.E5.88.99>)</strong>

具体的内容在这里就一笔带过了，读者应该要有一定的微积分基础，而且建议初学者不要立即去阅读维基上的微积分内容。

另外，我们以上所讨论的都是“一阶导数”，其实还有“二阶导数”、“n阶导数”等。其实定义也很简单，函数的<strong>“(n\-1)阶导数的导数”就叫做该函数的“n阶导数”</strong> 。$f(x)$的n阶导数记为$f^{(n)} (x)$。
