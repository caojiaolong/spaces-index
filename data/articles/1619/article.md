# 费曼积分法——积分符号内取微分(2)

> 作者：苏剑林 · 科学空间 · 2012-06-12
>
> 原文：<https://spaces.ac.cn/archives/1619>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章我对“费曼积分法”做了一个简单的介绍，并通过举例来初步展示了它的操作步骤。但是，要了解一个方法，除了知道它能够干什么之外，还必须了解它的原理和方法，这样我们才能够更好地掌握它。因此，我们需要建立“积分符号内取微分”的一般理论，为进一步的应用奠基。

<strong>一般原理</strong>

我们记
$$G(a)=\int_{m(a)}^{n(a)} f(x,a)dx$$

在这里，f(x,a)是带有参数a的关于x的函数，而积分区间是关于参数a的两个函数，这样的积分也叫变限积分，可以理解为是普通定积分的推广。我们记F(x,a)为f(x,a)的原函数，也就是说$\frac{\partial F(x,a)}{\partial x}=f(x,a)$，那么按照微积分基本定理，我们就有：
$$G(a)=F(n(a),a)-F(m(a),a)$$

对$a$求导：

$$\begin{aligned}\frac{d G(a)}{da}= &\frac{\partial F(n(a),a)}{\partial n(a)} \times \frac{d n(a)}{da}+\frac{\partial F(n(a),a)}{\partial a}\\
&-\frac{\partial F(m(a),a)}{\partial m(a)} \times \frac{d m(a)}{da}-\frac{\partial F(m(a),a)}{\partial a}\end{aligned}$$

这个这么长的式子实则告诉我们：
$$G'(a)=\int_{m(a)}^{n(a)} \frac{\partial f(x,a)}{\partial a} dx + f(n(a),a) \times \frac{d n(a)}{da}-f(m(a),a) \times \frac{d m(a)}{da}$$

这就是“积分符号内取微分”的法则！如果m,n是一个常数，那么直接变成了

$$G'(a)=\int_{m}^{n} \frac{\partial f(x,a)}{\partial a} dx $$

<u>“费曼积分法”之所以能够奏效，关键是“求导”这一步看似化简为繁，实则在求导过程中让许多繁琐的东西暂时消失了，得到一个相对简单的结果，然后再还原，这在某种意义上来说是分步处理的思想。</u>

<strong>更多的例子</strong>

为了进一步说明费曼积分法的应用，下面BoJone将展示更多的例子，需要说明的是，前面展示的一个例子是运算量相对较低的，而一般的定积分运算往往都比较复杂，不论是哪种方法，都要借助某个变换将其变成我们熟悉的积分，因此，需要积累一定的定积分结果，知道一些基本的可积函数等等。（当然啦，必要时还可以查[《积分表》](<https://spaces.ac.cn/sci/integral/index.html>) ）

另一方面，<strong>费曼积分法的核心之处在于参数的选择</strong>，并不是每一个要求的积分都是带有参数的，比如$\int_0^{\frac{\pi}{2}} \frac{x}{tan x}dx$，而且即使给出了，给出的参数也不一定符合要求，因此，我们需要逐渐领悟怎么去为被积函数“改装”成带有参数的形式，而原定积分则作为参数取某个特殊值时的情况。至于怎么增加参数，参数的形式怎么确定，BoJone对此也是一知半解，有待进一步提高。

<strong>例子1：</strong>

$$\int_0^{\frac{\pi}{2}} \frac{x}{\tan x}dx$$

这还算时一种比较常见的形式，它的基本形状为$\int \frac{x}{f(x)}dx$，我们的办法是试着（每一次运算都是一次尝试）将它改装成$\int \frac{f^{-1}(a \times f(x))}{f(x)}dx$，其中$f^{-1}(x)$是$f(x)$的反函数，原积分就等价于a=1时的情况。为什么要这样做呢？只要你将它一求导就会明白了^\_^

对于这道题目，我们将它变成

$$G(a)=\int_0^{\frac{\pi}{2}} \frac{arctan(a \times \tan x)}{\tan x}dx \\ f(x,a)=\frac{arctan(a \times \tan x)}{\tan x}$$

那么
$$\begin{aligned}\frac{\partial f(x,a)}{\partial a}=&\frac{1}{a^2 \tan^2 x+1} \\
=&\frac{\cos^2 x}{a^2+(1-a^2)\cos^2 x} \\
=&\frac{1}{1-a^2} \left[1-\frac{2a^2}{(1+a^2)+(1-a^2)\cos 2x}\right]\end{aligned}$$

等等等等等等，首先得弄清楚我们究竟在干什么，我们究竟想干什么。求导后的下一步就是要积分，什么形式的积分可以被简单地积分出来呢？查查[《积分表》](<https://spaces.ac.cn/sci/integral/11.html>)就可以知道，形式$\int \frac{1}{a+b \times \cos x}dx$类型的积分是可以积分出来的，因此我们要尽量将它往这个形式靠拢！

最终得到：
$$\begin{aligned}G'(a)=&\frac{1}{1-a^2} \int_0^{\frac{\pi}{2}} \left[1-\frac{2a^2}{(1+a^2)+(1-a^2)\cos 2x}\right]dx \\
=&\frac{1}{1-a^2} \Big[x-a \times \arctan(a \times \tan x)\Big]\Big|_0^{\pi/2} \\
=&\frac{1}{1-a^2}\left(\frac{\pi}{2}-a \times \frac{\pi}{2}\right) \\
=&\frac{\pi}{2(1+a)}\end{aligned}$$

再积分就得到
$$G(a)=\frac{\pi}{2}[ln(1+a)+C]$$

当$a=0$的时候，$f(x,a)=f(x,0)=0$，所以$G(0)=\int_0^{\frac{\pi}{2}} 0dx=0$，由此得出$C=0$。

顺便提及一下，关于首步的积分（对x变量的）结果，有经验的读者会从
$$\frac{1}{a^2 \tan^2 x+1}$$
得到灵感，猜测原函数会是$\arctan(a \times \tan x)$的形式，对它求导，我们会发现：
$$\begin{aligned}\frac{d \arctan(a \times \tan x)}{dx}=&\frac{a}{a^2 \tan^2 x+1} \times \frac{1}{\cos^2 x} \\
=&\frac{a}{a^2 \tan^2 x+1} \times (1+\tan^2 x) \\
=&\frac{a}{a^2 \tan^2 x+1} \times \left(\frac{1}{a^2}+\tan^2 x+1-\frac{1}{a^2}\right) \\
=&\frac{1}{a}+\frac{a-1/a}{a^2 \tan^2 x+1}\end{aligned}$$

这与要求的只相差一个常数，所以可以很快得出原函数$\frac{1}{1-a^2} [x-a\times\arctan(a \times \tan x)]$了。当然，这对数学思维有一定的要求。

<strong>说明：</strong>也许读者看到上面的过程后会感觉很头痛，疑问：这哪里简单了，明明这么复杂！其实不论用哪种方法，微积分运算的过程都是相当复杂的，关键是一个算法的操作性和可行性。换句话说，我们知道可行了，就可以按部就班地算下去。运算过程的繁琐不是一个问题的难度所在，<strong>好问题的难度在于：<u>如何得到这一个灵感，让我们得到这一个过程！</u></strong>

为了使文章不至于过长，更多的例子就放在下一篇文章了。
