# 《方程与宇宙》:二体问题的来来去去(一)

> 作者：苏剑林 · 科学空间 · 2010-03-20
>
> 原文：<https://spaces.ac.cn/archives/549>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![二体问题的轨道模拟](<https://spaces.ac.cn/usr/uploads/2010/03/15136248.gif>)](<https://spaces.ac.cn/attachment/550/>)

二体问题的轨道模拟

为了让大家能够查询到“天体力学”方面的内容，同时锻炼我的表达和计算能力，BoJone构思了《方程与宇宙》这个主题，主要是写一些关于使用数学相对深入地讨论一些天文问题。其实我一直觉得，不用公式是无法完美地描述科学的（当然也不能纯公式），我记得霍金的《时间简史》以及《果壳中的宇宙》等之类的书，都力求不用或者尽可能少用数学公式来表达自己的观点。这种模式对于对于公众来说是很好的，但是对于希望深入研究的朋友来说却难以进行。所以我主张：<strong>宇宙是算出来的！</strong>

这个主题每一个字都是由BoJone敲击出来的，其中包括引用了《天体力学引论》里面的一些内容，以及加入了BoJone个人的一些见解。由于篇幅长及时间有限问题，BoJone打算分若干次撰写发布，并且尽可能写得通俗一点，力求让有一点微积分基础的朋友就可以弄懂。这里首先发布第一部分。由于时间匆忙等原因，可能会出现一些疏忽，欢迎大家挑错！

二体问题使假设只有两个天体，不考虑其他天体的干扰，在万有引力（牛顿经典力学）作用下如何运动的问题。这是天体力学中最简单的一类问题，也是目前唯一能够精确求解的类型。如果考虑三个或者更多天体的作用，那么只能够用到近似分析或者数值算法。

设只有太阳S和某一个行星P，建立任一空间坐标系O\-xyz，$\boldsymbol{r}_s$是太阳的位置向量，$\boldsymbol{r}_p$是行星的位置向量。$\boldsymbol{r}$是行星相对于太阳的位置向量。以M和m表示太阳和行星的质量。太阳受到行星的引力
$$\boldsymbol{F}_s=G\frac{Mm}{r^2}\frac{\boldsymbol{r}}{|\boldsymbol{r}|}=G\frac{Mm}{r^3}\boldsymbol{r}$$
同时行星也受到太阳的引力
$$\boldsymbol{F}_p=G\frac{Mm}{r^2}\frac{-\boldsymbol{r}}{|\boldsymbol{r}|}=-G\frac{Mm}{r^3}\boldsymbol{r}$$
由牛顿第二定律：太阳和行星都在力的作用下产生运动
$$\boldsymbol{F}=\frac{d(mv)}{dt}=m\frac{d^2 \boldsymbol{r}}{dt^2}$$
应当有
$$\begin{aligned}\boldsymbol{F}_s=G\frac{Mm}{r^3}\boldsymbol{r}=M\frac{d^2 \boldsymbol{r}_s}{dt^2} \\ \boldsymbol{F}_p=-G\frac{Mm}{r^3}\boldsymbol{r}=m\frac{d^2 \boldsymbol{r}_p}{dt^2}\end{aligned}$$
因为$\boldsymbol{r}=\boldsymbol{r}_p-\boldsymbol{r}_s$，所以有
$$\frac{d^2 \boldsymbol{r}}{dt^2}=\frac{d^2}{dt^2}(\boldsymbol{r}_p-\boldsymbol{r}_s)=-\frac{\mu}{r^3}\boldsymbol{r}$$
其中$\mu=G(M+m)$。如果建立以太阳为原点的x\-y\-z直角坐标系，可以获得二体问题的微分方程组为：

$$\begin{aligned}\frac{d^2 x}{dt^2}=-\frac{\mu}{r^3}x \\ \frac{d^2 y}{dt^2}=-\frac{\mu}{r^3}y \\ \frac{d^2 z}{dt^2}=-\frac{\mu}{r^3}z\end{aligned}$$

式中$r^2=x^2+y^2+z^2$。下面我们来看一下这道方程组的解。在理论力学可以得知，行星的运动是有心力的作用，在有心力作用下的运动都是平面运动，当然我们可以从上述微分方程证实。由微分方程后两式可得：
$$y\ddot z-z\ddot y=0\Rightarrow \frac{d}{dt}(y\dot z-z\dot y)=0$$
两边积分：
$$y\dot z-z\dot y=A\tag{1}$$其中A为积分常数，同理可以得到：
$$z\dot x-x\dot z=B\tag{2}$$$$x\dot y-y\dot x=C\tag{3}$$(1),(2),(3)被称为<strong>“动量矩积分”</strong>。

用x,y,z分别乘以(1),(2),(3)式后相加，得到：$Ax+By+Cz=0$
<u>这是一个通过原点（太阳）的平面方程，这表明行星和太阳始终在同一个平面上！</u>所以我们可以只考虑O\-xy平面的方程，即
$\ddot x=-\frac{\mu x}{r^3}$，$$\ddot y=-\frac{\mu y}{r^3}\tag{4}$$动量矩积分变为一个：$$x\dot y-y\dot x=h\tag{5}$$
现在我们用极坐标讨论：设$x=rcos\theta,y=rsin\theta$，将(4)变换成
$$\ddot{r} -r\dot{\theta}^2=-\frac{\mu}{r^2}\tag{6}$$(5)变换成：
$$r^2 \dot{\theta}=h\tag{7}$$(7)式称为<strong>“面积积分”</strong>，<u>h就是向径所扫过的面积速度的两倍</u>。由于面积速度一定，所以当然在相同时间内所扫过的面积相等啦，于是我们就证明了<strong>“开普勒第二定律”</strong>

> <strong>附：这里简述一下变换过程：</strong>
> 
> 
> 
> 设$x=r*cos\theta,y=r*sin\theta$\.\.\.\.\.\.\.\.(00)
> 有$x^2+y^2=r^2$代入原方程
> $\ddot{x}=-\frac{\mu cos\theta}{r^2}$，$\ddot{y}=-\frac{\mu sin\theta}{r^2}$\.\.\.\.\.\.\.\.(0)
> 对(00)关于t求导
> $\dot{x}=\dot{r}*cos\theta-\dot{\theta}*r*sin\theta$\.\.\.\.\.\.\.\.\.\.(01)
> $\dot{y}=\dot{r}*sin\theta+\dot{\theta}*r*cos\theta$\.\.\.\.\.\.\.\.\.\.(02)
> (01),(02)再次对t求导
> $\ddot{x}=\ddot{r}*cos\theta-2*\dot{r}*\dot{\theta}*sin\theta-\ddot{\theta}*r*sin\theta-\dot{\theta}^2*r*cos\theta$\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.(03)
> $\ddot{y}=\ddot{r}*sin\theta+2*\dot{r}*\dot{\theta}*cos\theta+\ddot{\theta}*r*cos\theta-\dot{\theta}^2*r*sin\theta.$\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.\.(04)
> $(03)*cos\theta+(04)*sin\theta$得
> $\ddot{x}cos\theta+\ddot{y}sin\theta=\ddot{r}-\dot{\theta}^2*r$\.\.\.\.\.(05)
> 又由(0)有
> $\ddot{x}cos\theta+\ddot{y}sin\theta=-{\mu}/r^2$\.\.\.\.\.\.\.(06)
> 对比(05),(06)即有
> $$\ddot{r}-\dot{\theta}^2\cdot r=-{\mu}/r^2$$
> 
> 
> 
> 至于从(5)式变换成(7)式，就不需要那么多技巧性的东西，直接代入求导数，然后合并就行了。

现在我们来求解由(6),(7)构成的方程组，如果希望找到行星轨道的曲线类型，那么我们需要找到r和$\theta$之间的关系式；如果我们希望计算某时刻的行星位置，那么我们得找到r或$\theta$关于时间t的关系式。

首先我们寻求轨道的曲线类型，这里我直接摘录《天体力学引论》里面的内容。
令$u=1/r$，则(7)变为：$\dot{\theta}=hu^2$，并且有
$$\begin{aligned}\dot{r}=\frac{dr}{d\theta}\frac{d\theta}{dt}=\frac{d(1/u)}{d\theta}\dot{\theta}=-1/u^2\cdot \frac{du}{d\theta}\cdot hu^2=-h\frac{du}{d\theta} \\ \ddot{r}=-h\frac{d}{d\theta}(\frac{du}{d\theta})\dot{\theta}=-h^2 u^2\frac{d^2 u}{d\theta^2}\end{aligned}$$
代入(6)得到
$$\frac{d^2 u}{d\theta^2}+u=\frac{\mu}{h^2}$$
这是一个二阶线性微分方程。它的通解是：
$$u=\frac{\mu}{h^2}[1+e \cos(\theta-\omega)]\tag{8}$$其中e和$\omega$是待定常数，还原为r，则变为：
$$r=\frac{h^2//\mu}{1+e \cos(\theta-\omega)}\tag{9}$$
由解析几何可以知道，这是一条以焦点为原点的圆锥曲线（椭圆、双曲线、抛物线，可以参考[维基百科](<http://zh.wikipedia.org/wiki/%E5%9C%86%E9%94%A5%E6%9B%B2%E7%BA%BF>)）。至此我们证明了<strong>“开普勒第一定律”</strong>，而且更加广泛（不仅包括椭圆，还包括双曲线、抛物线）。由此可以推出$h^2=\mu a(1-e^2)$，a就是圆锥曲线的半长轴，而e则是偏心率，并且$\theta=\omega$时，r最小，即行星在近日点，所以$\omega$为近日点角）

> <strong>附：如何解这个微分方程？</strong>
> 具体求解线性微分方程可以参考[维基百科](<http://zh.wikipedia.org/zh-cn/%E7%BA%BF%E6%80%A7%E5%BE%AE%E5%88%86%E6%96%B9%E7%A8%8B>)
> 
> 
> 
> 根据解法，我们可以求出$\frac{d^2 u}{d\theta^2}+u=0$的通解为$y=C_1 cos\theta+C_2 sin\theta$，加上右边的$\frac{\mu}{h^2}$即为该微分方程的通解，但是如何转换成(8)的类型呢？
> 不难发现其实$C_1 cos\theta+C_2 sin\theta+\frac{\mu}{h^2}$与(8)是等价的，因为
> $$e \cos(\theta-\omega)=ecos\theta \cos\omega+e \sin\theta \sin\omega$$
> 即$ecos\omega=C_1 h^2/{\mu},esin\omega=C_2 h^2/{\mu}$

至此，我们已经完成了大部分的工作，还有最后一点点，解答二体问题就大功告成了，就是找到r或者$\theta$关于时间t的函数，才能计算某时刻的行星位置。

> <strong>附：t与r的关系式</strong>
> 由(7)得到$\dot{\theta}=h/r^2$，代入(6)得到：
> $$\ddot{r} -h^2/r^3=-\frac{\mu}{r^2}\tag{10}$$这是一个二阶微分方程，它的解很容易找出，但是这个积分太复杂：
> $$\dot{r}\frac{d\dot{r}}{dr}=h^2/r^3-\frac{\mu}{r^2}$$
> $\dot{r}d\dot{r}=(h^2/r^3-\frac{\mu}{r^2})dr$，两端积分
> $$\dot{r}^2={2\mu}/r-h^2/r^2+K_1\tag{11}$$$$\begin{aligned}\Rightarrow {dt}/{dr}=\frac{r}{\sqrt{K_1 r^2+2\mu r-h^2}} \\ t=\int \frac{r}{\sqrt{K_1 r^2+2\mu r-h^2}}dr\end{aligned}$$
> 这个积分很容易的（可以参考本站的[积分表](<https://spaces.ac.cn/archives/354/>)），缺点是但是最终的结果形式太复杂了！

《天体力学引论》找出的是$\theta$与t的关系（即开普勒方程）\.\.\.\.\.\.（待续）
