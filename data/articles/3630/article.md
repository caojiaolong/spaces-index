# 积分估计的极值原理——变分原理的初级版本

> 作者：苏剑林 · 科学空间 · 2016-02-15
>
> 原文：<https://spaces.ac.cn/archives/3630>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

如果一直关注科学空间的朋友会发现，笔者一直对极值原理有偏爱。比如，之前曾经写过一系列《自然极值》的文章，介绍一些极值问题和变分法；在物理学中，笔者偏爱最小作用量原理的形式；在数据挖掘中，笔者也因此对基于最大熵原理的最大熵模型有浓厚的兴趣；最近，在做《量子力学与路径积分》的习题中，笔者也对第十一章所说的变分原理产生了很大的兴趣。

对于一样新东西，笔者的学习方法是以一个尽可能简单的例子搞清楚它的原理和思想，然后再逐步复杂化，这样子我就不至于迷失了。对于变分原理，它是估算路径积分的一个很强大的方法，路径积分是泛函积分，或者说，无穷维积分，那么很自然想到，对于有限维的积分估计，比如最简单的一维积分，有没有类似的估算原理呢？事实上是有的，它并不复杂，弄懂它有助于了解变分原理的核心思想。很遗憾，我并没有找到已有的资料描述这个简化版的原理，可能跟我找的资料比较少有关。

### 从高斯型积分出发

变分原理本质上是Jensen不等式的应用。我们从下述积分出发
$$\begin{equation}\label{jifen}I(\epsilon)=\int_{-\infty}^{\infty}e^{-x^2-\epsilon x^4}dx\end{equation}$$
我们之前已经从微扰展开的角度研究过这个积分。这个积分通常被视为复杂的路径积分的一维版本，通常来说可以通过研究它来获得一些研究路径积分的有效思路。为了估算上述积分，我们会有一个很自然的想法，就是上述积分能不能用一个简单的高斯积分来表示呢？
$$\begin{equation}\label{jinsi}\int_{-\infty}^{\infty}e^{-px^2}dx=\sqrt{\frac{\pi}{p}}\end{equation}$$
问题的关键是如何选取$p$使得$\eqref{jinsi}$尽可能接近$\eqref{jifen}$。我们将$\eqref{jifen}$改写为
$$\begin{equation}\label{bianhuan}I(\epsilon)=\sqrt{\frac{\pi}{p}}\int_{-\infty}^{\infty}e^{(p-1)x^2-\epsilon x^4}\left(\sqrt{\frac{p}{\pi}}e^{-px^2}\right)dx\end{equation}$$
怎么看待$\eqref{bianhuan}$呢？我已经刻意写成了特定的形式——用统计的角度看，它就是$e^{(p-1)x^2+q-\epsilon x^4}$在权重$\sqrt{\frac{p}{\pi}}e^{-px^2}$下的平均，因此，我们不妨将它写成
$$\begin{equation}\label{pingjun}I(\epsilon)=\sqrt{\frac{\pi}{p}}\left\langle e^{(p-1)x^2-\epsilon x^4}\right\rangle\end{equation}$$
再次提醒，这个积分是在权重$\sqrt{\frac{p}{\pi}}e^{-px^2}$下的平均。到这里我们所做的只不过是形式的变换，并没有实质的进展，要算的积分我们还没算出来。

现在我们回想一下Jensen不等式，它说，如果$f(x)$是凸函数（关于凹凸性，可能不同的书的定义会有所不同，这里的凸函数，是指函数的二阶导数非负），那么
$$\begin{equation}\label{jensen}f(\langle x \rangle)\leq \langle f(x) \rangle\end{equation}$$
可能读者平时看到的Jensen不等式并不是这样子的，但是请读者仔细回味，Jensen不等式是不是就表达了这个意思？Jensen不等式有一个明显的物理意义，它说：如果若干个点位于一个凸函数的图像上面，那么这些点的质心位于凸函数的上方。

利用Jensen不等式，我们可以对$\eqref{pingjun}$对进一步的简化，因为很明显$e^x$是凸函数，因此
$$\begin{equation}\label{pingjun-jensen}I(\epsilon)=\sqrt{\frac{\pi}{p}}\left\langle e^{(p-1)x^2-\epsilon x^4}\right\rangle \geq \sqrt{\frac{\pi}{p}}e^{\left\langle (p-1)x^2-\epsilon x^4\right\rangle}\end{equation}$$
其中$\left\langle (p-1)x^2-\epsilon x^4\right\rangle$我们可以算出来
$$\left\langle (p-1)x^2-\epsilon x^4\right\rangle=\int_{-\infty}^{\infty}\left[(p-1)x^2-\epsilon x^4\right]\left(\sqrt{\frac{p}{\pi}}e^{-px^2}\right)dx=\frac{2 (p-1) p-3 \epsilon}{4 p^2}$$
于是
$$\begin{equation}\label{pingjun-guji}I(\epsilon)\geq\sqrt{\frac{\pi}{p}}\exp\left[\frac{2 (p-1) p-3 \epsilon}{4 p^2}\right]\end{equation}$$
不管$p$怎么取，最右边都会小于$I(\epsilon)$，那么很自然，为了最接近$I(\epsilon)$，要选取$p$使得不等号右边取最大值！<strong>这就是极值原理的来源了</strong>。求导算得，右边取最大值时，有
$$p=\frac{1}{2} \left(\sqrt{12 \epsilon+1}+1\right)$$
代进$\eqref{pingjun-guji}$得到
$$\begin{equation}\label{pingjun-guji2}I(\epsilon)\geq \sqrt{\frac{2 \pi }{\sqrt{12 \epsilon+1}+1}}\exp\left[\frac{3 \epsilon}{\left(\sqrt{12 \epsilon+1}+1\right)^2}\right]=\hat{I}(\epsilon)\end{equation}$$
下面是一些结果
$$\begin{array}{c|cccc}
\hline
& \epsilon=0 & \epsilon=1 & \epsilon=10 & \epsilon=100 \\
\hline
I(\epsilon) & 1.77245 & 1.36843 & 0.921961 & 0.554577 \\
\hat{I}(\epsilon) & 1.77245 & 1.34547 & 0.891204 & 0.531509 \\
\hline \end{array}$$
对比[http://kexue\.fm/archives/3280/](<https://spaces.ac.cn/archives/3280/>)中的结果，可以发现$\hat{I}(\epsilon)$是一个相当不错的估计，而且可以看到确实有$\hat{I}(\epsilon)\leq I(\epsilon)$。

### 我的总结

从上面的过程我们可以看到极值原理的一些威力：它给出了积分的一个良好估计，这个估计是稳定的，不管参数$\epsilon$大小，都可以得到一个合理的估计。其次，它不仅仅给出了积分的估计，而且估计的结果是积分的下界，这是很难得的。通常来说，我们可以通过各种各样的方法，得到一个式子的有效估计，但是往往很难判断估计值与精确值的大小关系（除非很大程度上牺牲精确度），而通过极值原理则一起做到了这两点——既保留了较高精确度，又实现了界的估计。

本文所涉及到的极值原理，源于Jensen不等式，而核心地方在于$e^x$是凸函数，而且$e^{\left\langle h(x)\right\rangle}$往往比$\left\langle e^{h(x)}\right\rangle$更好算。因此，推广的方向是很明显的，换成其他的权重，或者把$e^x$换成一般的凸函数，只要$f(\left\langle h(x)\right\rangle)$比$\left\langle f(h(x))\right\rangle$更好算即可。

当然，极值原理的缺点也非常明显。首先，要找到好的并且可积的近似函数并不容易；其次，即便能积分出来，接下来求极值的过程也不一定简单，因为这涉及到了解非线性方程组，而一旦使用近似，往往又退化为微扰展开的结果，失去了变分法独有的优点。最后，极值原理的估计还有一个不那么明显的缺点，就是很难进一步提高精确度。
