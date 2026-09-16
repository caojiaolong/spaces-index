# 函数图像旋转公式（“想当然”的教训）

> 作者：苏剑林 · 科学空间 · 2010-02-09
>
> 原文：<https://spaces.ac.cn/archives/416>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>阅读小提示：</strong>亲爱的读者，你可以选择不读这篇文章，但如果你选择了阅读，请一定要阅读完。BoJone对“半途而废”所造成的后果一概不负责任^\_^。

[![函数图像旋转](<https://spaces.ac.cn/usr/uploads/2010/02/2096919062.png>)](<https://spaces.ac.cn/attachment/412/>)

函数图像旋转

我们来考虑下一个旋转问题：将某一函数图像y=f(x)，绕点(p,q)逆时针旋转了θ角之后，得到的图象的解析式。

<u><strong>首先，由于仅仅是通过了旋转，所以函数的整体图像并没有变化，因此一定也是原来的函数f</strong></u>，我们在原来的函数图像上随便选取一点(x,y)，对应在旋转后的图像为(x’,y’)，那么新的图像解析式应该是y’=f(x’)\.而且由于图像是绕(p,q)旋转的，所以(x,y)和(x’,y’) 两个点到(p,q)的距离应该相等，设这个距离为r，即$\sqrt{(x-p)^2+(y-q)^2}=r$；令(x,y)\-(p,q)（这是指两个点之间的连线，下同）与<u>x轴的“过(p,q)的平行线</u>”所成的角为α，于是我们有：
$\sin\alpha=\frac{y-q}{r}$，$\cos\alpha=\frac{x-p}{r}$

继而：
$$\begin{aligned}\sin(\alpha +\theta)=\sin\alpha \cos\theta+\sin\theta \cos\alpha= \frac{y-q}{r} \cos\theta+\frac{x-p}{r}\sin\theta \\ \cos(\alpha +\theta)=\cos\alpha \cos\theta-\sin\theta \sin\alpha= \frac{x-p}{r} \cos\theta-\frac{y-q}{r}\sin\theta\end{aligned}$$

于是很显然：
$$\begin{aligned}y' =(\frac{y-q}{r} \cos\theta+\frac{x-p}{r}\sin\theta)r+q=(y-q)\cos\theta+(x-p)\sin\theta+q \\ x' =(\frac{x-p}{r} \cos\theta-\frac{y-q}{r}\sin\theta)r+p=(x-p)\cos\theta-(y-q)\sin\theta+p\end{aligned}$$

到此，问题解决了，新函数的解析式为：
$$(y-q)\cos\theta+(x-p)\sin\theta+q=f[(x-p)\cos\theta-(y-q)\sin\theta+p]$$

特别地，绕<u>原点</u>旋转的方程为：
$$y \cos\theta+x \sin\theta=f(x \cos\theta-y \sin\theta)$$

例如：y=6\-x，绕(0,0)逆时针旋转45°后，结果为
$$\begin{aligned}\frac{\sqrt{2}}{2}(x+y)=6-[\frac{\sqrt{2}}{2}(x-y)] \\ x=3\sqrt{2}\end{aligned}$$

注意，问题来了！明明是$y=3\sqrt{2}$，怎么变成了…？哈哈，大家和我一样，掉进了“陷阱”了！再仔细推敲，发现似乎有点问题；再推一下，又好像没有呀。难道…？<u>其实，问题在一开始的时候就出现了！</u>

<strong>一开始我们就设新图像的函数是f（注意划线），这是毫无根据的、而且是错误的。</strong><u><strong>我们已经知道了原来图像中的函数为y=f(x)，然后可以确定(x,y)与(x’,y’)之间的关系，求的是x’与y’之间的关系。正确的做法是：分别求出x、y关于x’、y’的表达式，然后代入y=f(x)，结果就是x’与y’的关系了！</strong></u>原来的思考过程是没有错误的，只要修改一下原来的过程，就可以得出答案了：

设β=α\+θ，有
$$\begin{aligned}\sin(\beta-\theta)=\sin\beta \cos\theta-\sin\theta \cos\beta= \frac{y'-q}{r} \cos\theta-\frac{x'-p}{r}\sin\theta \\ \cos(\beta -\theta)=\cos\beta \cos\theta+\sin\theta \sin\beta= \frac{x'-p}{r} \cos\theta+\frac{y'-q}{r}\sin\theta\end{aligned}$$

于是很显然：
$$\begin{aligned}y =(\frac{y'-q}{r} \cos\theta-\frac{x'-p}{r}\sin\theta)r+q=(y'-q)\cos\theta-(x'-p)\sin\theta+q \\ x =(\frac{x'-p}{r} \cos\theta+\frac{y'-q}{r}\sin\theta)r+p=(x'-p)\cos\theta+(y'-q)\sin\theta+p\end{aligned}$$

代入y=f(x)，就有

到此，我们终于得出了新函数的解析式<u>（逆时针）</u>：
$$(y-q)\cos\theta-(x-p)\sin\theta+q=f[(x-p)\cos\theta+(y-q)\sin\theta+p]$$
要是<u>顺时针</u>旋转的话：
$$(y-q)\cos\theta+(x-p)\sin\theta+q=f[(x-p)\cos\theta-(y-q)\sin\theta+p]$$

特别地，绕<u>原点</u>旋转的方程为：
$y cos\theta-x sin\theta=f(x cos\theta+y sin\theta)$（逆时针）
$y cos\theta+x sin\theta=f(x cos\theta-y sin\theta)$（顺时针）

这一次没有错误了吧？y=6\-x，绕(0,0)逆时针旋转45°后，结果为
$$\begin{aligned}\frac{\sqrt{2}}{2}(y-x)=6-[\frac{\sqrt{2}}{2}(x+y)] \\ y=3\sqrt{2}\end{aligned}$$
费了一番周折，答案终于出来……

<strong>感悟：</strong>

这时候读者明白我为什么要强调读完这篇文章的缘故了吧？并不是这篇文章特别地重要，只是如果一旦读到一半，就把公式或者方法抄了去，以后用的时候发现总是出错，那时候就不好了^\_^。

为什么我一开始要把读者们都引入一个“陷阱”呢？其实我相信这样的错误很多人都犯过，这篇文章描述了BoJone对这个问题的整个思考过程，从思考、错误到修正错误，错误的根源在于“想当然”、“应该是”之类的想法。要是无法冲破这个牢笼，就难以在数学、物理领域前进。这篇文章既是让自己引以为鉴，也希望读者们不要“覆前人之车”。记住：数理没有想当然！
