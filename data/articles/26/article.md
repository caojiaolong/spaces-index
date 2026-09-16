# 三次方程的根式求解（通俗版本）

> 作者：苏剑林 · 科学空间 · 2009-07-19
>
> 原文：<https://spaces.ac.cn/archives/26>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>(说明：由于本文章含有较多的根号，推荐使用IE直接阅读，或者使用IE\+MathPlayer。火狐浏览器对根号的显示是相当的差。)</strong>

大家知道，1到4次的代数方程都有求根公式（尽管未必是最简单的方法），对于1次和2次方程的求根，大家可能滚瓜烂熟了。但是你了解三次方程的解法吗？
$$ax^3+bx^2+cx+d=0\,(a\neq0)$$

网上有不少关于这方面的资料，但是却有着两个缺点：一是缺乏描述专业数学公式的相关程序（很多网站都是这样）；二是语言过于专业，不能大众化（如维基百科）。

要了解三次方程的求根公式，首先要知道，<u>一般地，n次代数方程有n个根</u>。而对于最基本的三次方程$x^3+p=0$，我们有：
$$x_1=-\sqrt[3]{p}$$
同时根据<strong>韦达定理</strong>，我们有$x_1+x_2+x_3=0,x_1\cdot x_2\cdot x_3=-p$，我们已经知道$x_1=-\sqrt[3]{p}$，现在就变成了关于$x_2,x_3$的二次方程组，可以求解($i^2=-1$，虚数单位)：
$$x_2=-\frac{1}{2}(-1+\sqrt{3}i)\sqrt[3]{p}, \quad x_3=-\frac{1}{2}(-1-\sqrt{3}i)\sqrt[3]{p}$$
特别地，一般会将$\frac{1}{2}(-1+\sqrt{3}i)$写成$\omega$，于是
$$x_2=-\sqrt[3]{p} \omega,\quad x_3=-\sqrt[3]{p}\omega ^2$$

[![图片说明：塔塔利亚](<https://spaces.ac.cn/usr/uploads/2009/07/20090719092206.jpg>)](<https://spaces.ac.cn/usr/uploads/2009/07/20090719092206.jpg>)

图片说明：塔塔利亚

<strong>下面进入一般的三次方程的求解：</strong>

对于一道三次方程$ax^3+bx^2+cx+d=0\,(a\neq 0)$，我们有可以用换元法，设$y=x+\frac{b}{3a}$，将原方程变为关于y的三次方程：
$$y^3+py+q=0 \Leftrightarrow y^3+py=-q$$
其中
$$\begin{aligned}&y=x+\frac{b}{3a}\\
&p=\frac{c}{a}-\frac{b^2}{3a^2}\\
&q=\frac{2{b^3}}{27a^3}-\frac{bc}{3a^2}+\frac{d}{a}\end{aligned}$$

(卡丹的证明)由于$(A-B)^3+3AB(A-B)=A^{3}-B^{3}$，所以$3AB=p,A^{3}-B^{3}=-q,y=A-B$，于是变成关于$A,B$的六次方程组，而这一道六次方程组很简单，通过换元法，就变成了一道二次方程组，可以求解。最后我们得到的结果为：
$$\begin{aligned}&A=\sqrt[3]{-\frac{q}{2}+\sqrt{\left(\frac{q}{2}\right)^2+\left(\frac{p}{3}\right)^3}},
&B=\sqrt[3]{\frac{q}{2}+\sqrt{\left(\frac{q}{2}\right)^2+\left(\frac{p}{3}\right)^3}}\end{aligned}$$

接下来就很容易得出原方程的解了。

[![图片说明：卡丹，又译卡尔达诺](<https://spaces.ac.cn/usr/uploads/2009/07/20090719092220.jpg>)](<https://spaces.ac.cn/usr/uploads/2009/07/20090719092220.jpg>)

图片说明：卡丹，又译卡尔达诺

最终，我们得出关于方程$y^3+px+q=0$的求根公式为
$$\begin{aligned}&x_1=\sqrt[3]{-\frac{q}{2}+\sqrt{(\frac{q}{2})^2+(\frac{p}{3})^3}}+\sqrt[3]{-\frac{q}{2}-\sqrt{(\frac{q}{2})^2+(\frac{p}{3})^3}}\\
&x_2=\omega \sqrt[3]{-\frac{q}{2}+\sqrt{(\frac{q}{2})^2+(\frac{p}{3})^3}}+\omega^2\sqrt[3]{-\frac{q}{2}-\sqrt{(\frac{q}{2})^2+(\frac{p}{3})^3}}\\
&x_3=\omega^2\sqrt[3]{-\frac{q}{2}+\sqrt{(\frac{q}{2})^2+(\frac{p}{3})^3}}+\omega \sqrt[3]{-\frac{q}{2}-\sqrt{(\frac{q}{2})^2+(\frac{p}{3})^3}}\\
&\omega=\frac{1}{2}(-1+\sqrt{3}i)\end{aligned}$$

看到这里，如果想挑战自己的你，请写出$ax^3+bx^2+cx+d=0\,(a\neq 0)$的一般求根公式吧^\_^

至于四次方程，有时间也会写一下。

<strong>参考资料：</strong>
[http://zh\.wikipedia\.org/w/index\.php?title=%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B&variant=zh\-cn](<http://zh.wikipedia.org/w/index.php?title=%E4%B8%89%E6%AC%A1%E6%96%B9%E7%A8%8B&variant=zh-cn>)
[http://baike\.baidu\.com/view/521598\.htm](<http://baike.baidu.com/view/521598.htm>)
[http://baike\.baidu\.com/view/1315076\.htm](<http://baike.baidu.com/view/1315076.htm>)
[http://www\.oursci\.org/archive/magazine/200112/011208\.htm](<http://www.oursci.org/archive/magazine/200112/011208.htm>)
