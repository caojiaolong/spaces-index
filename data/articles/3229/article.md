# 从Knotsevich在黑板上写的级数题目谈起

> 作者：苏剑林 · 科学空间 · 2015-02-27
>
> 原文：<https://spaces.ac.cn/archives/3229>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

某天在浏览高教社的“i数学”编辑的微博时候，发现上面有一道Knotsevich在黑板上写的他认为很有意思的题目，原始网址是：[http://weibo\.com/3271276117/BBrL5foVz](<http://http://weibo.com/3271276117/BBrL5foVz>)。

[![Knotsevich在黑板上写的级数题目](<https://spaces.ac.cn/usr/uploads/2015/02/44461549.jpg>)](<https://spaces.ac.cn/usr/uploads/2015/02/44461549.jpg>)

Knotsevich在黑板上写的级数题目

题目是这样的
$$\sum_{n=0}^{\infty} \frac{n! (20n)!}{(4n)!(7n)!(10n)!}x^n\tag{1}$$
大概的目的是找出原函数的表达式吧。

### 渐近级数

要注意，这里的级数不是十分“正常”，因为只需要简单的分析就可以得出，系数$\frac{n! (20n)!}{(4n)!(7n)!(10n)!}$增长得非常快，甚至超过了$n!$的增长速度，而大家知道指数函数的增长速度远弱于阶乘，所以按照一般的数学分析理论，级数$(1)$的收敛区域实际上只有一个点：$x=0$！

然后，上述函数也可能是某个良好函数的渐近级数形式。所谓渐近级数，是指误差估计项具有增长得非常快的系数，比如，如果函数$f(x)$的泰勒展开式$\sum_{n=0}^{\infty} a_n x^n$满足
$$(N-1)! |x|^N < \left|f(x)-\sum_{n=0}^{N} a_n x^n\right| < N! |x|^N$$
在取定前$N$项的时候，就有一个误差的估计，由于$|x|^N$的存在，所以当$|x|$充分小时，误差总可以充分小，但是由于带有$(N-1)!$的下界的存在，所以为了达到同样的精度，当$N$增大的时候，$|x|$的取值范围便随之缩小。当$N\to\infty$时，$|x|$的收敛范围就只有一个0了。这就是渐近级数的概念了，取的项越多，不一定越精确。而为了得到合理的结果，要不就缩小定义域，即$|x|$的范围；要不就少取一些项，但是少取一些项是有代价的，那就是不够精确，我们可能永远不知道小数点后面第十位的数字。

### Borel再求和

虽然渐近级数的实际收敛区域只有原点，但事实上渐近级数的原始函数，也就是作展开前的表达式，性态很可能是很好的，甚至延拓到复平面，在整个复平面上（的大部分区域）都有比较好的性态。为了还原渐近级数的原始表达式，我们需要一些“再求和”技术。我们通过求级数
$$\sum_{n=0}^{\infty} (n!)x^n \tag{2}$$
的原始表达式来展示这种再求和级数。首先我们有
$$\sum_{n=0}^{\infty} a_n x^n=\sum_{n=0}^{\infty} \frac{a_n}{n!} (n!) x^n$$
然后由[伽马函数](<https://spaces.ac.cn/archives/3108/>)的表达式，我们有$n!=\int_0^{\infty} t^n e^{-t} dt$，代入上式得
$$\sum_{n=0}^{\infty} a_n x^n=\sum_{n=0}^{\infty} \frac{a_n}{n!} \left(\int_0^{\infty} t^n e^{-t} dt \right) x^n$$
交换积分号和求和号
$$\sum_{n=0}^{\infty} a_n x^n=\int_0^{\infty} e^{-t} dt\sum_{n=0}^{\infty} \frac{a_n}{n!} (tx)^n \tag{3}$$
这样子，级数变成了$\sum_{n=0}^{\infty} \frac{a_n}{n!} (tx)^n $，如果这是一个正常的收敛级数，那么就可以求出原始函数，然后再积分。总的来说，就是通过将系数除以$n!$，以降低系数的发散速度，“逼”级数收敛，更一般地，还有
$$\sum_{n=0}^{\infty} a_n x^n=\int_0^{\infty} e^{-t} dt\sum_{n=0}^{\infty} \frac{a_n}{(kn)!} (t^k x)^n \tag{4}$$
这种求和技术叫做Borel再求和。

通过Borel再求和，我们很快可以将$(1)$式化为
$$\int_0^{+\infty}ds\int_0^{+\infty}dt e^{-s-t}\sum_{n=0}^{\infty}\frac{1}{(4n)!(7n)!(10n)!}\left(ts^{20}x\right)^n\tag{5}$$

为了方便后面的处理，还需要作进一步转化
$$\int_0^{+\infty}dr\int_0^{+\infty}ds\int_0^{+\infty}dt e^{-r-s-t}\sum_{n=0}^{\infty}\frac{1}{n! (4n)!(7n)!(10n)!}\left(rst^{20}x\right)^n\tag{6}$$

### 阶乘的倒数

现在$(5)$式或者$(6)$式中的求和部分怎么处理呢？我们通过Borel求和，消去了系数中分子的阶乘部分，可是分母呢？反思Borel求和法的推导，我们发现，其核心之处是用了阶乘的积分形式——伽马函数来替换它。那么，阶乘的倒数有没有类似的形式呢？有！不过需要借助复变函数了。
$$\frac{1}{n!}=\frac{1}{2\pi i}\oint_{|z|=1} \frac{e^z}{z^{n+1}}dz\tag{7}$$
$(7)$式的推导很简单，只不过是把$e^z$的级数展开式的系数用复积分的柯西积分公式来表达了一下而已。有意思的是，它与伽马函数的表达式
$$\Gamma(x)=\int_0^{+\infty} e^{-t}t^{x-1}dt$$
在形式上非常相似。考虑到$n!$与$\frac{1}{n!}$的关系，可以将$(7)$视为伽马函数的对偶形式？

通过$(7)$式，仿照$(4)$式，我们有
$$\sum_{n=0}^{\infty} a_n x^n=\frac{1}{2\pi i}\oint_{|z|=1} e^{z} z^{-1} dz\sum_{n=0}^{\infty} a_n (kn)! (z^{-k} x)^n \tag{8}$$

利用这个结果，我们就可以将$(6)$式转化为
$$\begin{aligned}&\left(\frac{1}{2\pi i}\right)^3\left(\int\dots\int\right) e^{-r-s-t+u+v+w}(uvw)^{-1}\times\sum_{n=0}^{\infty}\frac{1}{n!}\left(rst^{20}u^{-4}v^{-7}w^{-10} x\right)^n\\
=&\left(\frac{1}{2\pi i}\right)^3\left(\int\dots\int\right) e^{-r-s-t+u+v+w+rst^{20}u^{-4}v^{-7}w^{-10} x}(uvw)^{-1}
\end{aligned}$$
这里的$\left(\int\dots\int\right)$指的是
$$\oint_{|u|=1}du\oint_{|v|=1}dv\oint_{|w|=1}dw\int_0^{+\infty}dr\int_0^{+\infty}ds\int_0^{+\infty}dt$$
前面三个积分是复积分，后面三个积分是实积分。不知道这是不是想要的结果？
