# 从费马大定理谈起（十）：x^3\+y^3=z^3\+w^3

> 作者：苏剑林 · 科学空间 · 2014-10-10
>
> 原文：<https://spaces.ac.cn/archives/2972>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![Ramanujan](<https://spaces.ac.cn/usr/uploads/2014/10/345307851.jpg>)](<https://spaces.ac.cn/usr/uploads/2014/10/345307851.jpg>)

Ramanujan

在正式开始数学之前，我们不妨先说一个关于印度著名数学天才——拉马努金的轶事。拉马努金病重，哈代前往探望。哈代说：“我乘出租车来，车牌号码是1729，这数真没趣，希望不是不祥之兆。”拉马努金答道：“不，那是个有趣得很的数。可以用两个立方之和来表达而且有两种表达方式的数之中，1729是最小的。”（即$1729 = 1^3+12^3 = 9^3+10^3$，后来这类数称为的士数。）利特尔伍德回应这宗轶闻说：“每个整数都是拉马努金的朋友。”（来自[维基百科](<http://zh.wikipedia.org/zh-cn/%E6%96%AF%E9%87%8C%E5%B0%BC%E7%93%A6%E7%91%9F%C2%B7%E6%8B%89%E9%A9%AC%E5%8A%AA%E9%87%91>)）

从这则轶事中，我们发现，确实存在的某些整数，可以表示为两种不同的立方和，换句话说，不定方程：
$$x^3+y^3=z^3+w^3$$
存在$(x,y,z,w)$两两不等的整数解（正负均可）。借助艾森斯坦整数，我们可以求出通解的表达式来。这里得出的通解，比教科书上现有的通解更为复杂，但是能够由整数参数得到所有的整数解。（存在简化版的公式，也能够得到所有的整数解，但是为了得到某些整数解，需要一个分数形式的参数，这对于如何产生所有整数解会存在疑问。）

首先将两端分别在艾森斯坦整数内分解
$$(x+y)(x+y\omega)(x+y\omega^2)=(z+w)(z+w\omega)(z+w\omega^2)$$
将$x+y\omega$记为$\xi$，将$z+w\omega$记为$\eta$，我们得到
$$\xi\bar{\xi}\left(\omega\xi+\omega^2\bar{\xi}\right)=\eta\bar{\eta}\left(\omega\eta+\omega^2\bar{\eta}\right)$$
稍加分析就知道，$\xi$在艾森斯坦整数环中一定是个合数，不妨设
$$\xi=\alpha\beta$$
那么上式左端变为
$$\alpha\beta\bar{\alpha}\bar{\beta}\left(\omega\alpha\beta+\omega^2\bar{\alpha}\bar{\beta}\right)$$
要注意括号里边的数一定是个实数，因此一般地，可以设（$\lambda$是个实数）
$$\omega\alpha\beta+\omega^2\bar{\alpha}\bar{\beta}=\lambda\gamma\bar{\gamma}$$
那么等式左边就是
$$\alpha\beta\bar{\alpha}\bar{\beta}\left(\lambda\gamma\bar{\gamma}\right)$$
根据右边的特征，它也应该是跟左边类似，因此右边各数只能是左边各因数的一个重排，不失一般性，一个重排为：
$$\alpha\gamma\bar{\alpha}\bar{\gamma}\left(\lambda\beta\bar{\beta}\right)$$
也就是说
$$\left\{\begin{aligned}&\eta=\alpha\gamma\\
&\bar{\eta}=\bar{\alpha}\bar{\gamma}\\
&\omega\eta+\omega^2\bar{\eta}=\lambda\beta\bar{\beta}\end{aligned}\right.$$
结合已知的
$$\left\{\begin{aligned}&\xi=\alpha\beta\\
&\bar{\xi}=\bar{\alpha}\bar{\beta}\\
&\omega\xi+\omega^2\bar{\xi}=\lambda\gamma\bar{\gamma}\end{aligned}\right.$$
留意到有
$$\left\{\begin{aligned}&\omega\alpha\gamma+\omega^2\bar{\alpha}\bar{\gamma}=\lambda\beta\bar{\beta}\\
&\omega\alpha\beta+\omega^2\bar{\alpha}\bar{\beta}=\lambda\gamma\bar{\gamma}\end{aligned}\right.$$
关键的一步来了，我们要倒过来，把它们看成是关于$\alpha,\bar{\alpha}$的方程组，那只不过是二元一次线性方程组！解出来得到
$$\alpha=\lambda\omega^2\frac{\beta\bar{\beta}^2-\gamma\bar{\gamma}^2}{\gamma\bar{\beta}-\beta\bar{\gamma}}$$

如此一来，我们便完成了解的构造：给定参数$\lambda,\beta,\gamma$，就可以算出$\alpha$，然后根据
$$\xi=\alpha\beta,\quad\eta=\alpha\gamma$$
可以求出$\xi,\eta$，继而求出$x,y,z,w$。细心的读者会留意到：$\lambda$是个实数，$\beta,\gamma$是艾森斯坦整数，从实数来看，它们两个都带有两个自由参数，因此我们得到的是五个自由参数的解！未知数只有四个，参数却有五个，原则上来说参数存在冗余。但是为了从整数参数中得到所有整数解，却又不得不这样子取，实在奇妙。

设$\beta=a+b\omega,\,\gamma=c+d\omega$，借助Mathematica可以导出（$\lambda$属于平凡参数，遂没写出）
$$\left\{\begin{aligned}x&=\frac{a^3 (c+d)-3 a^2 b c+3 a b^2 c+b^3 (d-2 c)-(c^2-c d+d^2)^2}{3 b c-3 a d}\\[2ex]
y&=\frac{a^3 (c-2 d)+3 a^2 b d-3 a b^2 d+b^3 (c+d)-(c^2-c d+d^2)^2}{3 a d-3 b c}\\[2ex]
z&=\frac{a^4-2 a^3 b+3 a^2 b^2-a (2 b^3+c^3+d^3)+b (b^3+2 c^3-3 c^2 d+3 c d^2-d^3)}{3 a d-3 b c}\\[2ex]
w&=\frac{-a^4+2 a^3 b-3 a^2 b^2+a (2 b^3+c^3-3 c^2 d+3 c d^2-2 d^3)+b (-b^3+c^3+d^3)}{3 a d-3 b c}
\end{aligned}\right.$$
一个复杂的参数解\~\~

其中$a=-1,b=-2,c=3,d=3$时，给出$x=9,y=-12,z=1,w=-10$，也就是$1^3+12^3=9^3+10^3$，这是最小的整数解。

也许有兴趣的读者会想试试$x^4+y^4=z^4+w^4$的解。的确，这道方程也存在两两不等的整数解，然而，它远比三次方的困难，到目前为止，我们甚至都不知道它任意一个形式的通解。
