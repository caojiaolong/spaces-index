# (原创)切抛物线法解方程

> 作者：苏剑林 · 科学空间 · 2010-03-06
>
> 原文：<https://spaces.ac.cn/archives/504>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>牛顿法</strong>使用的是函数切线的方程的零点来逼近原函数的零点，他所使用的是“切直线”，要是改为同曲率的“切抛物线”，则有更稳定的收敛效果以及更快的收敛速度

设函数$y=f(x)$在$(x_0,y_0)$处有一条<u>“切抛物线”</u>$y=ax^2+bx+c$，则应该有

$a(x_0+\Delta x)^2+b(x_0+\Delta x)+c=f(x_0+\Delta x)$\-\-\-\-\-\-\-(A)
$ax_0^2+bx_0+c=f(x_0)$\-\-\-\-\-\-\-(B)
$a(x_0-\Delta x)^2+b(x_0-\Delta x)+c=f(x_0-\Delta x)$\-\-\-\-\-\-\-(C)

其中$lim_{\Delta x->0}$

这道方程组是有解的。首先(A)\-(B)得到：
$a(\Delta x^2+2x_0\Delta x)+b\Delta x=f(x_0+\Delta x)-f(x_0)$\-\-\-\-(D)
(B)\-(C)得到
$a(-\Delta x^2+2x_0\Delta x)+b\Delta x=f(x_0)-f(x_0-\Delta x)$\-\-\-\-(E)
(D)\-(E)得到
$$\begin{aligned}2\Delta x^2 a=[f(x_0+\Delta x)-f(x_0)]-[f(x_0)-f(x_0-\Delta x)] \\ \Rightarrow \\ a=lim_{\Delta x->0}\frac{\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}-\frac{f(x_0)-f(x_0-\Delta x)}{\Delta x}}{2\Delta x} \\ =lim_{\Delta x->0}\frac{f'(x_0)-f'(x_0-\Delta x)}{2\Delta x} \\ =lim_{\Delta x->0} \frac{f''(x_0-\Delta x)}{2}=\frac{f''(x_0)}{2}\end{aligned}$$

即是$a=\frac{y_0''}{2}$，进而有：$b=y_0'-y_0''x_0$，$c=y_0-y_0'+\frac{y_0''x_0^2}{2}$

至此，我们已经求出了任意函数$y=f(x)$的切抛物线方程，利用其可以有效地逼近原来函数的零点。设$y=f(x)$，令上述抛物线方程等于0，解这道方程：
$$x=x_0-\frac{y_0'}{y_0''}+-\sqrt{(\frac{y_0'}{y_0''})^2-\frac{2y_0}{y_0''}}$$

一般地，这是比$x_0$更接近方程零点的值（这里$+-$待定，一般取的是正号），我们可以取这个解继续求切抛物线，继续进行上述步骤，于是得出递推公式为：
$$x_{n+1}=x_n-\frac{y_n'}{y_n''}+-\sqrt{(\frac{y_n'}{y_n''})^2-\frac{2y_n}{y_n''}}$$

不可否认，这比牛顿法计算量大了很多；但也有一定优点，收敛更快，$x_0$的取值范围更广一些。

<u>上面是从纯几何的角度来推导出这种方法的，几何意义明显。</u>从<strong>泰勒级数</strong>可以更方便推出。根据泰勒级数有：
$$f(x)=f(x_n)+f'(x_n)(x-x_n)+\frac{f''(x_n)}{2!}(x-x_n)^2 +\frac{f'(x_n)}{3!}(x-x_n)^3 +...$$

牛顿法是取$f(x)=f(x_n)+(x-x_n)f'(x_n)$作为$f(x)=0$的近似方程，而“切抛物线”则是取$f(x)=f(x_n)+(x-x_n)f'(x_n)+(x-x_n)^2 \frac{f''(x_n)}{2!}$，这是关于x的二次方程，于是可以得出：
$$x_{n+1}=x_n-\frac{y_n'}{y_n''}+-\sqrt{(\frac{y_n'}{y_n''})^2-\frac{2y_n}{y_n''}}$$

过程是不是简单了很多？可为什么一开始就从复杂的几何角度来推导呢？其实，这样推导，正是体现了解析几何的要点——代数与几何相结合，有助于我们理解一个方法的几何含义，进而推导出更多的方法来。例如我们可以不用“切抛物线”，我们可以用“切双曲线”、“切椭圆”等等的方法，进而取得更多的成果。例如函数“切双曲线”的结果则是
$$\begin{aligned}y=k/x+b \\ k=-y_0'x_0^2 \\ b=y_0+y_0'x_0\end{aligned}$$
递推公式为：$x_{n+1}=\frac{y_n' x_n^2}{y_n+y_n' x_n}$
（这是“弱切双曲线”，收敛比牛顿法慢；完整切双曲线应该是$y=k/{x-l}+b$形式的，需要用到二阶导数。）
