# 精确自由落体运动定律的讨论(二)

> 作者：苏剑林 · 科学空间 · 2010-01-09
>
> 原文：<https://spaces.ac.cn/archives/337>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

之前在[这篇文章](<https://spaces.ac.cn/archives/330/>)中，我们使用过一个牛顿引力场中的自由落体公式：
\begin{equation}t=\sqrt{\frac{r_0}{2GM}}\left(r_0 \cdot \arctan \sqrt{\frac{r_0 -r}{r}}+\sqrt{r(r_0 -r)}\right)\label{eq:1}\end{equation}

我们来尝试一下推导出这个公式来。

[![跳伞过程中的自由落体阶段](<https://spaces.ac.cn/usr/uploads/2010/01/2788496468.jpg>)](<https://spaces.ac.cn/attachment/338/>)

跳伞过程中的自由落体阶段

同时，站长在逐渐深入研究的过程中，发现微分方程极其重要。以前一些我认为不可能解决的问题，都用微分方程逐渐解决了。在以后的文章里，我们将会继续体验到微分方程的伟大魔力！因此，建议各位有志研究物理学的朋友，一定要掌握微分方程，更加深入的，需要用到偏微分方程！

首先，质量为$m$的物理在距离地心$r$处的引力为$\frac{GMm}{r^2}$，根据牛顿第二定律$F=ma$，自然下落的物体所获得的加速度为$\frac{GM}{r^2}$。假设物体从距离地心$r$开始向地心自由下落，求位移$s$关于$t$的函数$s=s(t)$\.

根据根据加速度的定义，我们有：$\frac{d^2 s}{dx^2}=a=\frac{GM}{(r-s)^2}$，于是问题实质就是解常微分方程
\begin{equation}s''=\frac{GM}{(r-s)^2}\end{equation}

接着我们令$s'=v$，则$s''=v(\frac{dv}{ds})$，代入上式：$GM(r-s)^{-2} ds=vdv$，两端积分：
\begin{equation}\begin{gathered}\int vdv = \int GM(r - s)^{ - 2} ds = - \int GM( r - s)^{ - 2}d(r - s) \\
\Downarrow\\
\frac{1}{2} v^2 = GM[(r - s)^{ - 1} + C_1]\end{gathered}\end{equation}

根据实际情况，当$t=0$时，$v=s=0$，推出$C_1=-r^{-1}$，即
\begin{equation}\frac{1}{2}v^2 = GM[(r - s)^{ - 1} - r^{ - 1}] \quad\Rightarrow\quad \frac{ds}{dt} = v = \sqrt {\frac{2GM}{r}} \sqrt {\frac{s}{r - s}}\end{equation}
两端积分：
\begin{equation}\int dt = \sqrt {\frac{r}{2GM}} \int (\sqrt {\frac{r - s}{s}} )ds =2\sqrt {\frac{r}{2GM}} \int (\sqrt {r - s} )d(s^{0.5}) \end{equation}

令$s^{0.5}=P$，则
\begin{equation}2\int (\sqrt {r - s} )d(s^{0.5}) =2\int (\sqrt {r - p^2} )dp\end{equation}
根据积分公式：
\begin{equation}\int \sqrt{a^2-x^2} dx =\frac{a^2}{2} \arcsin \frac{x}{a} +x/2 \sqrt{a^2-x^2}+C\end{equation}
得出
\begin{equation}2\int (\sqrt {r - p^2} )dp=r\cdot \arcsin \frac{p}{\sqrt{r}}+p \sqrt{r-p^2}+C\end{equation}
换回$s$，有：
\begin{equation}2\int (\sqrt {r - s} )d(s^{0.5}) =r\cdot \arcsin \sqrt{\frac{s}{r}}+\sqrt{s(r-s)}+C\end{equation}
则：
\begin{equation}t=\sqrt{\frac{r}{2GM}}\left(r\cdot \arcsin\sqrt{\frac{s}{r}}+\sqrt{s(r-s)}+C\right)\end{equation}
当$s=0$时，$t=0$，则$C=-\pi r / 2$，得出：
\begin{equation}t=\sqrt{\frac{r}{2GM}}\left(r\cdot \arcsin\sqrt{\frac{s}{r}}+\sqrt{s(r-s)}\right)\end{equation}
而根据反三角函数公式：$\arcsin \frac{a}{b} = \arctan\sqrt{\frac{a^2}{b^2-a^2}}$，即得
\begin{equation}t=\sqrt{\frac{r}{2GM}}\left(r\cdot\arctan\sqrt{\frac{s}{r-s}}+\sqrt{s(r-s)}\right)\label{eq:2}\end{equation}

不难看出，$\eqref{eq:1}$和$\eqref{eq:2}$两式是等价的，$\eqref{eq:2}$的$r$实际上是$\eqref{eq:1}$的$r_0$，而$s$则相当于$r_0 - r$。
