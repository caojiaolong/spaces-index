# 旋转的弹簧将如何伸长？ 

> 作者：苏剑林 · 科学空间 · 2010-07-30
>
> 原文：<https://spaces.ac.cn/archives/782>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![旋转的弹簧](<https://spaces.ac.cn/usr/uploads/2010/07/3603508682.png>)](<https://spaces.ac.cn/attachment/784/>)

旋转的弹簧

> 一根均匀的弹簧长度l<sub>0</sub>
> 
> ，线密度λ
> 
> <sub>0</sub>
> 
> ，劲度系数k，总质量M。现在没有重力的环境下，绕其一端作角速度ω的旋转（角速度恒定），则此时其长度变为多少？

这是网友“宇宙为家”在几天前提出的问题。期间我曾做过多次解答，犯了若干次错误，经过修修补补，得出了最后的答案，在此感谢“宇宙为家”朋友的多次提醒。如果下面的答案依旧有错误，望各位读者发现并指出。

首先要把题目理解清楚，弹簧在旋转前是均匀的，但是旋转后由于不同点受到的“惯性离心力”不同，所以每一部分弹簧必将不均匀地伸长，导致密度不再是常数。由于弹簧变得不均匀，因此以“<strong>长度比</strong>”来衡量劲度系数比已经不可靠了，应该以“<strong>质量比</strong>”来表达。设在旋转之后密度函数为$\lambda=\lambda(r)$，r是弹簧上的一个横断面到旋转重心O的距离，弹簧上每一点的<strong><u>惯性离心力</u></strong>为$dF_c=r\omega^2 dm=\lambda r\omega^2 dr$，那么距离圆心为r处的弹簧的断面受到的惯性离心力为$F_c=\int_r^{l}\lambda r\omega^2 dr$（注：$F_c$即Centrifugal force，离心力，这是为了避免与下面的F混淆）。每一段长度为dr弹簧的<strong><u>劲度系数</u></strong>为$\frac{M}{dm}k$，由于惯性离心力的作用，<strong><u>伸长量</u></strong>为$\frac{\int_r^{l}\lambda r\omega^2 dr}{Mk/dm}=\frac{\omega^2 dm}{Mk}\int_r^{l}\lambda r dr$，旋转前这一段的长度为$\frac{dm}{\lambda_0}$，显然这一段的<strong><u>密度</u></strong>为$\lambda=\frac{dm}{\frac{dm}{\lambda_0}+\frac{\omega^2 dm}{Mk}\int_r^{l}\lambda r dr}=\frac{1}{\frac{1}{\lambda_0}+\frac{\omega^2}{Mk}\int_r^{l}\lambda r dr}$

请看最后一步，我们已经是列出了关于密度函数的“积分方程”：
$$\lambda=\frac{1}{\frac{1}{\lambda_0}+\frac{\omega^2}{Mk}\int_r^{l}\lambda r dr}$$

这是BoJone解答这道题目的关键。对于BoJone来说，解积分方程是很困难的，通过变换，可以将其变为微分方程（其实对BoJone而言，解微分方程也不容易）。
设$F=\int_r^{l}\lambda r dr$，则$\lambda =-\frac{dF}{rdr}=-\frac{\dot{F}}{r}$，代入原来的方程，得到：
$$\begin{aligned}-\frac{\dot{F}}{r}=\frac{1}{\frac{1}{\lambda_0}+\frac{\omega^2}{Mk}F} \\ -r=\frac{\dot{F}}{\lambda_0}+\frac{\omega^2}{Mk}F\dot{F}\end{aligned}$$

这道微分方程显得如此简单，积分一次就得到：
$$C-\frac{1}{2} r^2=\frac{F}{\lambda_0}+\frac{\omega^2}{2Mk}F^2$$

C是积分常数，根据F的定义，可以得出当r=l时，应该有F=0，所以推出：$C=\frac{1}{2} l^2$，并且从中解出F，得到
$$F=\pm \sqrt{\frac{Mk}{\omega^2}(l^2-r^2)+(\frac{Mk}{\omega^2\lambda_0})^2}-\frac{Mk}{\omega^2\lambda_0}$$
舍去负值（自己想原因？）
可以计算
$$\lambda =-\frac{dF}{rdr}=\frac{Mk}{\omega^2\sqrt{\frac{M^2 k^2}{\omega^4 \lambda_0^2}+\frac{Mk (l^2-r^2) }{\omega^2}}}=\frac{1}{\sqrt{\frac{1}{\lambda_0^2}+\frac{\omega^2(l^2-r^2)}{Mk}}}$$

根据密度函数的定义，应该有
$$\int_0^l \lambda dr=\int_0^l \frac{dr}{\sqrt{\frac{1}{\lambda_0^2}+\frac{\omega^2(l^2-r^2)}{Mk}}}=M$$
这个积分的结果是
$$\frac{\sqrt{Mk}}{\omega}arcsin( \frac{r\omega}{\sqrt{l^2\omega^2+\frac{Mk}{\lambda_0^2}}})$$

于是我们有
$$M=\frac{\sqrt{Mk}}{\omega}arcsin( \frac{l\omega}{\sqrt{l^2\omega^2+\frac{Mk}{\lambda_0^2}}})$$

为了方便计算，根据$arcsin(\frac{a}{b})=arctg\frac{a}{\sqrt{b^2-a^2}}$，可以将上式变成
$$M=\frac{\sqrt{Mk}}{\omega}arctg(\frac{l\omega\lambda_0}{\sqrt{Mk}})$$

可以解出
$$l=\frac{\sqrt{Mk}}{\lambda_0 \omega}tg(\omega\sqrt{M/k})=l_0\cdot \frac{1}{\omega}\sqrt{k/M}tg(\omega\sqrt{M/k})$$

<strong>这个答案很漂亮！首先$\frac{1}{\omega}\sqrt{k/M}$和$\omega\sqrt{M/k})$互为倒数，这两个具有<u>对称的美</u>。而且M的量纲是$M$，k的量纲是$\frac{M}{T^2}$，那么$\sqrt{M/k}$的量纲就是$T$，而$\omega$的量纲是$1/T$，那么$\frac{1}{\omega}\sqrt{k/M}$和$\omega\sqrt{M/k})$均是无量纲的量，这与答案的运算过程是<u>自洽</u>的！再者，当$\omega\sqrt{M/k}->\pi/2$时，$l->\infty$，换句话说，$\omega\sqrt{M/k}$不能超过$\pi/2$，这给出了衡量弹簧性能的一个参数：$\omega=\pi/2 \sqrt{k/M}$，如果$\omega$越大，那么就说明弹簧“受折腾”能力越强，也就是性能越好！</strong>
