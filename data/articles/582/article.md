# 《方程与宇宙》:抛物线与双曲线轨道(三)

> 作者：苏剑林 · 科学空间 · 2010-04-03
>
> 原文：<https://spaces.ac.cn/archives/582>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![圆锥曲线](<https://spaces.ac.cn/usr/uploads/2010/04/3676400487.png>)](<https://spaces.ac.cn/attachment/581/>)

圆锥曲线

经过上两回的讨论，我们已经基本摸清了二体问题的运动情况。我们已经找到了二体问题在轨道为椭圆的时候的所有积分，给出了“活力公式”等常用公式的证明，并且留下了一些没有解答的问题。那就是在轨道为抛物线和双曲线时的最后一个积分还没有找出来，现在我们解决这两个问题。其中的关键积分依旧是
$\dot{r}^2={2\mu}/r-{\mu a(1-e^2)}/r^2-\frac{\mu}{a}$——(12)

<strong>一、抛物线轨道</strong>

对于抛物线轨道，我们有$e=1,a->\infty$，但是近日点的关系式依然没有变，也就是说近日距依旧是$a(1-e)=p$，而由于a趋向无穷大，那么$\frac{\mu}{a}$可以忽略掉，于是(12)可以简化成：
$$\dot{r}^2=\frac{2\mu}{r}-\frac{2p\mu}{r^2}$$
不难推出
$$\sqrt{2\mu}dt=\frac{rdr}{\sqrt{r-p}}$$
并且令$R^2=r-p$，则上式变为
$$\sqrt{2\mu}dt=\frac{(R^2+p)d(R^2+p)}{R}=(2R^2+2p)dR$$
两端积分
$$\sqrt{2\mu}t=2/3 R^3+2pR+K_3$$
取t=0为近日点的时刻，那么可以得到$K_3=0$，于是抛物线的轨道问题就解决了：
$\sqrt{\frac{\mu}{2}}t=1/3 R^3+pR$，$$r=R^2+p\tag{16}$$
<u>（附：这个不同于《天体力学引论》等教程的解法，这是直接求出向径r的长度，然后再求角度，过程刚好与原著相反。个人认为这种方法更好理解，而且更符合我们的思维。另外这是一道关于R的三次方程，我们也不必去寻找它的求根公式，只要使用方程的迭代解法针对某一数据就可以求出相应的R）</u>

<strong>二、双曲线轨道</strong>

对于双曲线，有$e>1,a<0$，所以我们令$A=-a$，避免负数开平方的情况。(12)变为
$$\dot{r}^2={2\mu}/r-{\mu A(e^2-1)}/r^2+\frac{\mu}{A}\tag{17}$$从而推出
$$\sqrt{\frac{\mu}{A}}dt=\frac{rdr}{\sqrt{(A+r)^2-A^2 e^2}}$$
这个与椭圆轨道的方程类似，方法也是用换元法，但是不是用三角函数，而是用“双曲函数”。令$A+r=Ae cosh(E)$，则可以变换成
$$\sqrt{\frac{\mu}{A^3}}dt=(e \cos h E -1)dE$$
两端积分
$$\sqrt{\frac{\mu}{A^3}}t=e \sin h E -E+K_4$$
同样取t=0为近日点的时刻，那么可以得到$K_4=0$，于是可以得到双曲线的开普勒方程
$$\sqrt{\frac{\mu}{A^3}}t=e \sin h E -E\tag{18}$$这与椭圆轨道的开普勒方程有点类似，同样是超越方程。

> <strong>附：简介一下双曲函数</strong>
> 
> 
> 
> 根据欧拉的表示法，我们可以把余弦表示成
> $$\cos \theta=\frac{e^{i\theta}+e^{-i\theta}}{2}$$
> 这里的$e=2.71828....,i^2=-1$，要是我们把里面的“i”去掉，可以定义一种新的函数，称为“双曲余弦函数”，记为\\cos h，也就是说
> $$\cos h \theta=\frac{e^{\theta}+e^{-\theta}}{2}$$
> 类似的：
> $$\begin{aligned}\sin h \theta=\frac{e^{\theta}-e^{-\theta}}{2} \\ \tan h \theta=\frac{\sin h \theta}{\cos h \theta}\end{aligned}$$
> 对于求导数
> $$\begin{aligned}(\sin h \theta)'=\cos h \theta \\ (\cos h \theta)'=\sin h \theta\end{aligned}$$
> 更多内容可以参考：
> [http://zh\.wikipedia\.org/wiki/%E5%8F%8C%E6%9B%B2%E5%87%BD%E6%95%B0](<http://zh.wikipedia.org/wiki/%E5%8F%8C%E6%9B%B2%E5%87%BD%E6%95%B0>)

<strong>最后，二体问题部分就讲到这里了，还有其他一些诸如怎样求解开普勒方程的小问题，大家其实不用想的那么复杂。对于一道难以直接求解的方程的求解，数学上有通用的数值方法，可以用来求任意精确度的特解，对于天文应用来说来说，这已经足够了。另外《方程与宇宙》这个主题可能会有很长时间不更新了，因为要准备天文奥赛，以及想钻研一下三体问题的摄动解法，所以希望大家等待了。当然，blog还会坚持更新的。</strong>
