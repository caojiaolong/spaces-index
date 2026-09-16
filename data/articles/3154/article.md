# 鬼斧神工：求n维球的体积

> 作者：苏剑林 · 科学空间 · 2014-12-23
>
> 原文：<https://spaces.ac.cn/archives/3154>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

今天早上同学问了我有关伽马函数和$n$维空间的球体积之间的关系，我记得我以前想要研究，但是并没有落实。既然她提问了，那么就完成这未完成的计划吧。

### 标准思路

简单来说，$n$维球体积就是如下$n$重积分
$$V_n(r)=\int_{x_1^2+x_2^2+\dots+x_n^2\leq r^2}dx_1 dx_2\dots dx_n$$
用更加几何的思路，我们通过一组平行面（$n-1$维的平行面）分割，使得$n$维球分解为一系列近似小柱体，因此，可以得到递推公式
$$V_n (r)=\int_{-r}^r V_{n-1} \left(\sqrt{r^2-t^2}\right)dt$$
设$t=r\sin\theta_1$，就有
$$V_n (r)=r\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} V_{n-1} \left(r\cos\theta_1\right)\cos\theta_1 d\theta_1$$
迭代一次就有
$$V_n (r)=r^2\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} V_{n-2} \left(r\cos\theta_1\cos\theta_2\right)\cos\theta_1\cos^2\theta_2 d\theta_1 d\theta_2$$
迭代$n-1$次
$$\begin{aligned}V_n (r)=&r^{n-1}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\dots\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} V_1\left(r\cos\theta_1\cos\theta_2\dots \cos\theta_{n-1}\right)\times\\
&\cos\theta_1\cos^2\theta_2\dots\cos^{n-1}\theta_{n-1} d\theta_1 d\theta_2\dots d\theta_{n-1}\end{aligned}$$
其中$V_1 (r)=2r$，即两倍半径长的线段。从而
$$V_n (r)=2r^{n}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\dots\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos^2\theta_1\cos^3\theta_2\dots\cos^{n}\theta_{n-1} d\theta_1 d\theta_2\dots d\theta_{n-1}$$
完成这个积分，最终就得到$n$维球体积的公式，这个积分自然是可以求出来的（只是$n-1$个一维积分的乘积）。但是这样的步骤太不容易了，为了将其跟伽马函数联系起来，还要做很多工作。总的来说，这是一个不容易记忆、也不怎么漂亮的标准方法。

### 绝妙思路

有一个利用高斯积分的绝妙技巧，能够帮助我们直接将球体积跟伽马函数联系起来，整个过程堪称鬼斧神工，而且给人“仅此一家，别无分号”的感觉。据说这个技巧为物理系学生所知晓，我是从[百读文库](<http://doc.baidu.com/view/9b4e929c680203d8ce2f24bb.html>)看到的，原始来源则是《热力学与统计力学》顾莱纳(德)，例5\.2 理想气体的熵的统计计算。

这一绝妙的思路，始于我们用两种不同的思路计算高斯积分
$$G(n)=\int_{-\infty}^{+\infty}\dots\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} \exp\left(-x_1^2-x_2^2-\dots-x_n^2\right)dx_1 dx_2 \dots dx_n\tag{1}$$
一方面，将$(1)$当作$n$次累次积分，因为我们已经算得（可以参考[这里](<https://spaces.ac.cn/archives/1967/>)）
$$\int_{-\infty}^{+\infty}\exp(-t^2)dt=\sqrt{\pi}$$
而$(1)$只不过是这样的$n$个积分的乘积，因此
$$G(n)=\pi^{n/2}\tag{2}$$
另一方面，将$(1)$当作$n$重积分，由于积分变量只是跟径向长度$r=\sqrt{x_1^2+x_2^2+\dots+x_n^2}$有关的变量，因此很容易联想到球坐标，在$n$维空间中，可以称为“超球坐标”，不需要将超球坐标完整写出来，只需要注意到，<u>球内的积分，可以化为先对“球壳”进行积分，然后再对球半径进行积分。</u>
$$G(n)=\int_{0}^{+\infty}dr\int_{S_n(r)}\exp\left(-r^2\right)dS_n\tag{3}$$
这里的$S_n(r)$是半径为$r$的$n$维球体表面（以及表面积，在不至于混淆的情况下，这里不作区分）。但是注意到，被积函数只跟$r$有关，因此对球表面进行积分，等价于原函数乘以球的表面积而已，因此$(2)$式的结果为
$$G(n)=\int_{0}^{+\infty}dr\exp\left(-r^2\right)S_n(r)\tag{4}$$
虽然我们不知道$n$维球的体积和表面积公式，但是我们可以肯定，$n$维球的体积一定正比于$r^n$，即有
$$V_n (r)=V_n(1)r^n$$
球的表面积，就是球体积的一阶导数（考虑球壳分割），那么
$$S_n (r)=n V_n(1)r^{n-1}$$
代入$(4)$，得到
$$\begin{aligned}G(n)=&n V_n(1)\int_{0}^{+\infty}r^{n-1}\exp\left(-r^2\right)dr\\
=&\frac{1}{2}n V_n(1)\int_{0}^{+\infty}(r^2)^{n/2-1}\exp\left(-r^2\right)d(r^2)\\
=&\frac{1}{2}n V_n(1)\int_{0}^{+\infty}z^{n/2-1}\exp\left(-z\right)dz\quad\left(z=r^2\right)\\
=&\frac{1}{2}n V_n(1)\Gamma\left(\frac{n}{2}\right)\end{aligned}\tag{5}$$
结合$(2)$得
$$\pi^{n/2}=G(n)=\frac{1}{2}n V_n(1)\Gamma\left(\frac{n}{2}\right)$$
从而
$$V_n(1)=\frac{\pi^{n/2}}{\frac{1}{2}n\Gamma\left(\frac{n}{2}\right)}=\frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2}+1\right)}$$
最后
$$V_n(r)=\frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2}+1\right)}r^n$$
就这样得到了$n$维球体积公式！！对$r$求导得到$n$维球表面积公式
$$S_n(r)=\frac{2\pi^{n/2}}{\Gamma\left(\frac{n}{2}\right)}r^{n-1}$$
结合前后两个方法，就得到
$$\frac{\pi^{n/2}}{\Gamma\left(\frac{n}{2}+1\right)}=2\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\dots\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}}\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos^2\theta_1\cos^3\theta_2\dots\cos^{n}\theta_{n-1} d\theta_1 d\theta_2\dots d\theta_{n-1}$$

### 简单评述

<strong>该技巧相当漂亮、简洁，其中高斯积分、球坐标变换这些都是物理系学生很熟悉的，只需简单峰回路转，就把结果给算了出来，这俨然就是只有物理系学生才能想出来的绝妙思路！</strong>

<strong>更妙的是，我们发现这一思路如此奇妙，以至于我们想用它来做更多的事情，但是稍微研究之后就会得到结论：不能再做什么了！也就是说，整个过程似乎就只为计算$n$维球体积而订制的！真的是“只此一家，别无分号”！妙哉\~\~</strong>
