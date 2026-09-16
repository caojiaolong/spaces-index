# 伽马函数的傅里叶变换之路

> 作者：苏剑林 · 科学空间 · 2014-12-08
>
> 原文：<https://spaces.ac.cn/archives/3108>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<div data-spaces-format="html-v1">
<p data-spaces-tag="p" data-spaces-attrs="{}">伽马函数<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(x)=\int_0^{+\infty}t^{x-1}e^{-t}dt$$<br data-spaces-tag="br" data-spaces-attrs="{}">
作为阶乘的推广，会让很多初学者感到困惑，对于笔者来说也不例外。一个最自然的问题就是：这般复杂的推广公式是如何得到的？</p><p data-spaces-tag="p" data-spaces-attrs="{}">在cos.name的文章<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://\\cos.name/2013/01/lda-math-gamma-function/&quot;, &quot;target&quot;: &quot;_blank&quot;}">《神奇的伽马函数》</a>中，有比较详细地对伽马函数的历史介绍，笔者细读之后也获益匪浅。但美中不足的是，笔者还是没能从中找到引出伽马函数的一种“自然”的办法。所谓“自然”，并不是说最简单的，而是根据一些基本的性质和定义，直接把伽马函数的表达式反解出来。它的过程和运算也许并不简单，但是思想应当是直接而简洁的。当然，我们不能苛求历史上伽马函数以这种方式诞生，但是作为事后探索是有益的，有助于我们了解伽马函数的特性。于是笔者尝试了以下途径，得到了一些结果，可是也得到了一些困惑。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">在实数的世界里：干不了！</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">伽马函数作为阶乘函数的推广，自然要满足阶乘函数所具有的性质，比如阶乘函数满足<br data-spaces-tag="br" data-spaces-attrs="{}">
$$n!=n\times (n-1)!\tag{1}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
现在我们定义伽马函数$\Gamma(x)$，在$x$是正整数的时候有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(x)=(x-1)!\tag{2}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
于是根据$(1)$有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(x+1)=x\Gamma(x)\tag{3}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
上式开始只对$x$为正整数时成立，而作为推广，我们要求$(3)$对于所有的实数都成立（这正是伽马函数的性质），而为了求出$\Gamma(x)$的具体表达式，我们就得从$(3)$中解出$\Gamma(x)$，这是函数方程的求解问题。根据<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/3018/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/3018/">《算符的艺术：差分、微分与伯努利数》</a>中的思想，记$D\equiv \frac{d}{dx}$，我们就可以将$(3)$写成<br data-spaces-tag="br" data-spaces-attrs="{}">
$$e^D\Gamma(x)=x\Gamma(x)\tag{4}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
利用傅里叶变换来求解它，两边作傅里叶变换，其中<br data-spaces-tag="br" data-spaces-attrs="{}">
$$D\to i\omega,\quad x\to i\frac{d}{d\omega}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
从而$(4)$变为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$e^{i\omega}\mathcal{F}[\Gamma(x)]=i\frac{d}{d\omega}\mathcal{F}[\Gamma(x)]$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这方程是很容易解的，解为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\mathcal{F}[\Gamma(x)]=C\exp\left(-e^{i\omega}\right)$$<br data-spaces-tag="br" data-spaces-attrs="{}">
最后作逆变换得到<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(x)=\frac{1}{2\pi}\int_{-\infty}^{+\infty} C\exp\left(-e^{i\omega}\right)e^{i\omega x}d\omega\tag{5}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
似乎挺完美的，但是请注意，积分$(5)$不会得到一个良好的函数（除非$C=0$），比如说$x$是整数的时候，被积函数是一个周期函数！非零的周期函数自然不会积出一个有意义的结果来。严格来说，上述积分并不存在，它的结果只能算是一个广义函数。因此，我们走这条路失败了。事实上，这表明$\Gamma(x)$的傅里叶变换不是一个普通的函数，因此不能像研究普通函数那样研究$\Gamma(x)$的傅里叶变换，所以也就就没有从它的傅里叶变换反过来求伽马函数的做法了。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">在复数的世界里：柳暗花明</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">数学中往往有这样的现象：在某个范围内解决某个特殊的问题显得特别困难，但是把问题推广了，把范围扩大了，一次性解决更多的问题，反而显得更加简洁，如前不久我们谈到过的<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/3101/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/3101/">《正弦级数和余弦级数》</a>就是一例。而在这篇文章中，我们把$\Gamma(x)$看成是阶乘函数对一般实数的推广，但是前面我们没有办法从$(3)$式中反解出$\Gamma(x)$来。有一个奇妙的技巧，让我们可以从中得到有意义的结果，那就是干脆把$\Gamma(x)$延拓到复数域中去，把它认为是复数域中的函数$\Gamma(z)$！这样做能够让我们从$(3)$式中反解出$\Gamma(x)$来，经历了实数世界中的“山重水复疑无路”之后，复数的世界给我们“柳暗花明又一村”的感觉。</p><p data-spaces-tag="p" data-spaces-attrs="{}">设$x$是一个实数变量，那么根据$(2)$式，对于纯虚数$xi$，我们有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma\left((x-i)i\right)=\Gamma(x i+1)=x i\Gamma(x i)\tag{6}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
同样设$D\equiv\frac{d}{dx}$，那么类似$(4)$，有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$e^{-i D}\Gamma(x i)=x i\Gamma(x i)\tag{7}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
同样利用傅里叶变换，此时得到<br data-spaces-tag="br" data-spaces-attrs="{}">
$$e^{\omega}\mathcal{F}[\Gamma(x i)]=-\frac{d}{d\omega}\mathcal{F}[\Gamma(x i)]$$<br data-spaces-tag="br" data-spaces-attrs="{}">
解得<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\mathcal{F}[\Gamma(x i)]=C\exp\left(-e^{\omega}\right)$$<br data-spaces-tag="br" data-spaces-attrs="{}">
作逆变换得<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(x i)=\frac{1}{2\pi}\int_{-\infty}^{+\infty} C\exp\left(-e^{\omega}\right)e^{i\omega x}d\omega\tag{8}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
那么<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(z)=\frac{1}{2\pi}\int_{-\infty}^{+\infty} C\exp\left(-e^{\omega}\right)e^{\omega z}d\omega\tag{9}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
注意不同于$(5)$式，$(9)$式是一个简单的实积分，它可以继续化简<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}\Gamma(z)=&amp;\frac{1}{2\pi}\int_{-\infty}^{+\infty} C\exp\left(-e^{\omega}\right)e^{\omega z}d\omega\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;\frac{C}{2\pi}\int_{-\infty}^{+\infty} \exp\left(-e^{\omega}\right)e^{\omega (z-1)}e^{\omega}d\omega\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;\frac{C}{2\pi}\int_{0}^{+\infty} \exp\left(-e^{\omega}\right)e^{\omega (z-1)}de^{\omega}\\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;\frac{C}{2\pi}\int_{0}^{+\infty} e^{-t} t^{z-1}dt\end{aligned}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
最后根据$\Gamma(1)=0!=1$，确定$\frac{C}{2\pi}=1$，从而<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\Gamma(z)=\int_{0}^{+\infty} e^{-t} t^{z-1}dt\tag{10}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
$(10)$式正好是我们要寻求的伽马函数的表达式。</p><p data-spaces-tag="p" data-spaces-attrs="{}">这表明，伽马函数作为复自变量的虚部的一个实函数（也就是说，固定自变量的实部，虚部为变量），才具有意义普通函数性质的傅里叶变换。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">简述</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">本文是笔者试探式的结果，通过求解函数方程$(3)$来反解伽马函数，我们看到了傅里叶变换在其中的作用，但也看到了某些不足，一些步骤需要一些特定的技巧才能进一步求解下去。当然，更重要的是不断尝试，不断探索，才能真正去体味到其中的乐趣。本文的后半部分，也就是在复数中求解的思路，也是经过长时间的尝试无果，在今天突然想到的。总的来说，多多尝试，总会有收获的。</p>
</div>
