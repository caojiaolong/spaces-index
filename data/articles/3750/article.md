# 路径积分系列：2\.随机游走模型

> 作者：苏剑林 · 科学空间 · 2016-05-30
>
> 原文：<https://spaces.ac.cn/archives/3750>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

随机游走模型形式简单，但通过它可以导出丰富的结果，它是物理中各种扩散模型的基础之一，它也等价于随机过程中的布朗运动\.

<strong>笔者所阅的文献表明，数学家已经对对称随机游走问题作了充分研究&#91;2&#93;，也探讨了随机游走问题与偏微分方程的关系&#91;3&#93;，并且还研究过不对称随机游走问题&#91;4&#93;\. 然而，已有结果的不足之处有：1、在推导随机游走问题的概率分布或者偏微分方程之时，所用的方法不够简洁明了；2、没有研究更一般的不对称随机游走问题\.</strong>

<strong>本章弥补了这一不足，首先通过母函数和傅里叶变换的方法，推导出了不对称随机游走问题所满足的偏微分方程，并且提出，由于随机游走容易通过计算机模拟，因此通过随机游走来模拟偏微分方程的解是一种有效的数值途径\. </strong>

### 模型简介

本节通过一个本质上属于二项分布的走格子问题来引入随机游走\.

> 考虑实数轴上的一个粒子，在$t=0$时刻它位于原点，每秒钟它以相等的概率向前或向后移动一格（$+1$或$-1$），问$n$秒后它所处位置的概率分布\.

这是一个独立重复实验，可以用母函数法来解决，其中每秒的行走可用函数描述为$\frac{1}{2}z+\frac{1}{2}z^{-1}$，其中$\frac{1}{2}z$项表示以$\frac{1}{2}$的概率向正方向移动1，$\frac{1}{2}z^{-1}$则是以$\frac{1}{2}$ 的概率向负方向移动1\. 由于运动是独立重复的，所以$n$秒后的运动分布情况可以用
$$\left(\frac{z+z^{-1}}{2}\right)^n=\sum_{k=0}^{n}\frac{1}{2^n}\binom{n}{k}z^{2k-n}\tag{1}$$
来描述，$z^{n-2k}$的系数$\frac{1}{2^n}\binom{n}{k}$表示粒子位于$n-2k$的概率是$\frac{1}{2^n}\binom{n}{k}$\. 可见，这是一个二项分布问题\.

随机游走模型是上面的问题的细致化：

> 考虑实数轴上的一个粒子，在$t=0$时刻它位于原点，每$\Delta t$秒以相等的概率向前或向后移动$\Delta s$格（$+\Delta s$或$-\Delta s$），考虑$\Delta t,\Delta s\to 0$，问$t$秒后它所处位置的概率分布\.

同样地，用母函数技巧，我们得到
$$\left(\frac{z^{\Delta s}+z^{-\Delta s}}{2}\right)^{t/\Delta t},\tag{2}$$
用$e^{-i\omega}$代替$z$，得到用傅里叶变换描述的母函数：
$$\left(\frac{e^{i\omega\Delta s}+e^{-i\omega\Delta s }}{2}\right)^{t/\Delta t},\tag{3}$$
同样地，$e^{-i\omega x}$的系数表示概率，但是这时候系数要通过傅里叶逆变换求得\. 由于$\Delta t,\Delta s\to 0$，使用欧拉公式进一步化简得
$$\left(\frac{e^{i\omega\Delta s }+e^{-i\omega\Delta s }}{2}\right)^{t/\Delta t}= \cos^{t/\Delta t}\left(\omega\Delta s \right)\approx\left(1-\frac{\omega^2 \Delta s ^2 }{2}\right)^{t/\Delta t},\tag{4}$$
为了得到意义明显的结果，取$\Delta s^2 =\alpha \Delta t,\Delta t\to 0$，得到
$$\exp\left(\frac{-\omega^2 \alpha t }{2}\right),\tag{5}$$
上式就是随机游走问题概率分布的傅里叶变换结果\. 也就是说，假如$t$秒后，粒子位于$[x,x+dx]$处的概率是$ P (x,t)dx$，那么就有
$$\exp\left(\frac{-\omega^2 \alpha t }{2}\right)=\int_{-\infty}^{+\infty} P (x,t) \exp\left(-i\omega x\right)dx,\tag{6}$$
通过傅里叶变换的逆变换，得到
$$ P (x,t)=\frac{1}{\sqrt{2\pi \alpha t}}\exp\left(-\frac{x^2}{2\alpha t }\right).\tag{7}$$

这就是随机游走的概率分布，跟已有文献&#91;2&#93;的结果是一样的\. 结果表明粒子的位置服从正态分布\. 在概率论中，我们已经知道正态分布应用广泛，因此这从侧面反映了随机游走模型的重要意义\.

### 非对称随机游走

上一节的例子中，每一步都是独立重复试验，即向左向右的概率都是$\frac{1}{2}$，所以我们可以通过求极限的方法得到概率分布\. 现在考虑粒子处于$(x,t)$时，它以概率$\frac{1-p(x,t)\Delta s/\alpha}{2}$向左移动$\Delta s$，以概率$\frac{1+p(x,t)\Delta s/\alpha}{2}$向右移动$\Delta s$（为什么不直接将$/\alpha$整合到$p(x,t)$的定义中呢？这里事实上涉及到量纲的问题，笔者这样定义，目的是让$p(x,t)$有速度的量纲，从而跟后面随机微分方程的结果一致\. ）\. 这时上述过程就不能描述为独立重复试验，但是傅里叶变换方法依然适用\.

方便起见，记$ P (x,t)$的傅里叶变换为
$$
\mathcal{F}_{ P } (t)=\mathcal{F}[ P (x,t)]=\int_{-\infty}^{+\infty} P (x,t)e^{-i\omega x}dx
.\tag{8}$$
假如粒子目前位于$(x,t)$，那么可以通过下面的母函数（近似）描述它下一步的随机游走
$$
\begin{aligned}&\left(\frac{1-p(x,t)\Delta s/\alpha}{2}\right)e^{i\omega \Delta s}+\left(\frac{1+p(x,t)\Delta s/\alpha}{2}\right)e^{-i\omega \Delta s}\\
=&\cos\omega \Delta s-i p(x,t)\Delta s \sin\omega\Delta s/\alpha\\
\approx &1-\frac{\omega^2 \Delta s^2}{2}-i \omega p(x,t)\Delta s^2/\alpha\\
=&1-\frac{\omega^2 \alpha \Delta t}{2}-i \omega p(x,t)\Delta t\end{aligned}
.\tag{9}$$
母函数的乘法对应着概率的合成，因此
$$\mathcal{F}_{ P } (t+\Delta t)\approx\int_{-\infty}^{+\infty}\left[1-\frac{\omega^2 \alpha \Delta t}{2}-i \omega p(x,t)\Delta t\right] P (x,t)e^{-i\omega x}dx
,\tag{10}$$
取极限即
$$\frac{\partial \mathcal{F}_{ P }(t)}{\partial t}= -\frac{\omega^2 \alpha }{2}\mathcal{F}_{ P }(t)-i \omega \mathcal{F}[p(x,t) P (x,t)]
,\tag{11}$$
进行逆傅里叶变换，就得
$$\frac{\partial P }{\partial t}= \frac{\alpha }{2}\frac{\partial^2 P }{\partial x^2}-\frac{\partial}{\partial x}(p P )
.\tag{12}$$
特别地，对称随机游走有$p\equiv 0$，那么上述方程变为扩散方程
$$\frac{\partial P }{\partial t}= \frac{\alpha }{2}\frac{\partial^2 P }{\partial x^2}.
.\tag{13}$$

### 简化形式

考虑简化问题，可以通过变换消去$ P $对$x$的一阶偏导数项\. 由$(12)$ 得
$$\alpha\frac{\partial P }{\partial t}= \frac{\alpha^2}{2}\frac{\partial^2 P }{\partial x^2}-\alpha \frac{\partial p}{\partial x} P -\alpha p\frac{\partial P }{\partial x}
,\tag{14}$$

设$ P (x,t)=\phi(x,t) \xi(x,t)$，代入得
$$\begin{aligned}
\alpha\frac{\partial \phi}{\partial t}\xi=& \frac{\alpha^2}{2}\left(\frac{\partial^2 \phi}{\partial x^2}\xi+2\frac{\partial \phi}{\partial x}\frac{\partial \xi}{\partial x}+\phi\frac{\partial^2 \xi}{\partial x^2}\right)\\
&-\alpha \frac{\partial p}{\partial x}\phi\xi-\alpha p\left(\frac{\partial \phi}{\partial x}\xi+\frac{\partial \xi}{\partial x}\phi\right)-\alpha\phi\frac{\partial \xi}{\partial t}\\
=&\left(\frac{\alpha^2}{2}\frac{\partial^2 \phi}{\partial x^2}+\eta(x,t)\frac{\partial \phi}{\partial x}+V(x,t)\phi\right)\xi\end{aligned}
,\tag{15}$$
其中
$$\begin{aligned}&\eta(x,t)=\alpha^2\frac{1}{\xi}\frac{\partial \xi}{\partial x}-\alpha p,\\
&V(x,t)=-\frac{1}{\xi}\left(\alpha\frac{\partial p}{\partial x}\xi+\alpha p\frac{\partial \xi}{\partial x}-\frac{1}{2}\alpha^2\frac{\partial^2 \xi}{\partial x^2}+\alpha\frac{\partial \xi}{\partial t}\right)\end{aligned},\tag{16}$$
让$\eta(x)\equiv 0$，得到
$$\xi(x,t)=\exp\left(\frac{1}{\alpha}\int p(x,t) dx\right),\tag{17}$$
并且求得
$$V(x,t)=-\frac{1}{2}\left(\alpha\frac{\partial p}{\partial x}+p^2\right)-\int \frac{\partial p}{\partial t} dx
,\tag{18}$$
此时，关于$\phi$的方程就是
$$\alpha\frac{\partial \phi}{\partial t}=\frac{\alpha^2}{2}\frac{\partial^2 \phi}{\partial x^2}+ V\phi
.\tag{19}$$
这是研究得比较多、且相对来说易于研究的形式，对应于量子力学的薛定谔方程\. 当然，可以把$\phi(x,t)$也想象为相对概率分布，但必须明确，$ P $才是真实的概率分布\.

如果$p$与$t$无关，那么
$$V(x)=-\frac{1}{2}\left(\alpha\frac{\partial p}{\partial x}+p^2\right),\tag{20}$$
特别地，如果$p=0$，那么将导致$V=0$，于是得到扩散方程
$$\frac{\partial \phi}{\partial t}=\frac{\alpha }{2}\frac{\partial^2 \phi}{\partial x^2}.\tag{21}$$

### 计算机模拟

既然随机游走对应着偏微分方程$(19)$，而随机游走在编程上是容易实现的，那么我们通过模拟随机游走的方法求解偏微分方程$(19)$的数值近似，这在很多时候是非常有效的\. 肖柳青和周石鹏的《随机模拟方法与应用》一书中关于随机游走的章节，就有这样的案例。
