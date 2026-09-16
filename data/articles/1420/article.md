# &#91;更正&#93;一道经典不等式的美妙证明

> 作者：苏剑林 · 科学空间 · 2011-07-20
>
> 原文：<https://spaces.ac.cn/archives/1420>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">在数学竞赛中，很多题目都专门设置了一种技巧，这种技巧在很大程度上是不怎么理所当然的，换句话说，难以“顺理成章”地想下去，或者是说方法不成系统的，这也是我有点不喜欢数学竞赛题目的一个原因。当然，另一方面，个人认为数学竞赛比物理竞赛更能锻炼一个人的思维能力，尤其是在抽象思维以及几何想象能力等，因此做一些这样的题目也会有好处的。</p><p data-spaces-tag="p" data-spaces-attrs="{}">下面就是一道很经典的竞赛题，它是在韩国举行的第42届IMO中的题目：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">设a,b,c都是正实数，求证：<br data-spaces-tag="br" data-spaces-attrs="{}">
$\frac{a}{\sqrt{a^2+8bc}}+ \frac{b}{\sqrt{b^2+8ac}} + \frac{c}{\sqrt{c^2+8ab}} \geq 1$</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}"></p><p data-spaces-tag="p" data-spaces-attrs="{}">这道题我看到过几个不同证明，在《国际数学奥林匹克》376页中就给出了两个不同的证明，其中一个证明如下：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 500px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/attachment/1421/&quot;, &quot;title&quot;: &quot;经典不等式-证明.jpg&quot;}" href="https://spaces.ac.cn/attachment/1421/" title="经典不等式-证明.jpg"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;经典不等式-证明&quot;, &quot;src&quot;: &quot;/usr/uploads/2011/07/3910535316.jpg&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2011/07/3910535316.jpg" alt="经典不等式-证明"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">经典不等式-证明</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">另外一个证明思路虽然不同，但是我感觉还是难以思考得到。昨天在浏览数联天地时，发现了一个绝妙的证明，很符合常规的思路。它利用的主要工具是<strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">Jensen不等式</font></strong>：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}">若函数f(x)在一个区间内满足$f(a_1 x_1+a_2 x_2) \geq a_1 f(x_1)+a_2 f(x_2)$，则称其为上凸函数，简称凸函数，其中$a_1,a_2$非负且$a_1+a_2=1$。判断函数的凸性还可以通过二阶导数来判断，若在区间内$f''(x) &lt; 0$，则该函数在该区间为上凸函数。（将不等号方向改变，则函数称为下凸函数，或称凹函数）</p><p data-spaces-tag="p" data-spaces-attrs="{}">Jensen不等式其实将上述关系推广了，即若函数对任意非负的$a_1+a_2=1$满足$f(a_1 x_1+a_2 x_2) \geq a_1 f(x_1)+a_2 f(x_2)$，则对于任意的满足$a_1+a_2+...+a_n=1$的非负数$a_1,a_2,...,a_n$，都有<br data-spaces-tag="br" data-spaces-attrs="{}">
$f(a_1 x_1+a_2 x_2+...+a_n x_n) \geq a_1 f(x_1)+a_2 f(x_2)+...+a_n f(x_n)$</p></blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">这是<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">一条强有力的不等式</font>，几乎可以用它来推导所有的不等式，如平均不等式、柯西不等式等等。它的证明方法在此略过，最直接的方法是用泰勒级数展开，具体可以参考网络资料。现在回到本文章所讲的题目。本题的证明涉及到几个证明不等式常用的方法。</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><del data-spaces-tag="del" data-spaces-attrs="{}">首先是<strong data-spaces-tag="strong" data-spaces-attrs="{}">形式变换</strong>，然后是<strong data-spaces-tag="strong" data-spaces-attrs="{}">挖掘隐含条件</strong>。<br data-spaces-tag="br" data-spaces-attrs="{}">
<br data-spaces-tag="br" data-spaces-attrs="{}">
<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">形式变换：</font><br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{1}{\sqrt{1+\frac{8bc}{a^2}}}+ \frac{1}{\sqrt{1+\frac{8ac}{b^2}}} + \frac{1}{\sqrt{1+\frac{8ab}{c^2}}} \geq 1$$<br data-spaces-tag="br" data-spaces-attrs="{}">
<br data-spaces-tag="br" data-spaces-attrs="{}">
<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">挖掘隐含条件：</font><br data-spaces-tag="br" data-spaces-attrs="{}">
令$\frac{bc}{a^2}=x,\frac{ac}{b^2}=y,\frac{ab}{c^2}=z$，则变为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{1}{\sqrt{1+8x}}+ \frac{1}{\sqrt{1+8y}} + \frac{1}{\sqrt{1+8z}} \geq 1$$<br data-spaces-tag="br" data-spaces-attrs="{}">
<br data-spaces-tag="br" data-spaces-attrs="{}">
其中隐含条件是$xyz=1$<br data-spaces-tag="br" data-spaces-attrs="{}">
<br data-spaces-tag="br" data-spaces-attrs="{}">
接着一部堪称妙绝！因为用Jensen不等式涉及到<u data-spaces-tag="u" data-spaces-attrs="{}">求和</u>，而已知条件则是<u data-spaces-tag="u" data-spaces-attrs="{}">积</u>的形式，我们要将其变成和的形式，那么我们容易想到指数运算：令$x=e^u,y=e^v,z=e^w$，则已知条件变为$u+v+w$。题目变为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{1}{\sqrt{1+8e^u}}+ \frac{1}{\sqrt{1+8e^v}} + \frac{1}{\sqrt{1+8e^w}} \geq 1$$<br data-spaces-tag="br" data-spaces-attrs="{}">
<br data-spaces-tag="br" data-spaces-attrs="{}">
这个变换的最绝之处在于它不仅仅进行了可行的变换，而且直接包含了题目本身就带有的限制：x,y,z都是正数!<br data-spaces-tag="br" data-spaces-attrs="{}">
<br data-spaces-tag="br" data-spaces-attrs="{}">
接下来利用Jensen不等式就很简单了：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{1}{\sqrt{1+8e^u}}+ \frac{1}{\sqrt{1+8e^v}} + \frac{1}{\sqrt{1+8e^w}} \geq \frac{3}{\sqrt{1+8e^(\frac{u+v+w}{3})}} =1$$<br data-spaces-tag="br" data-spaces-attrs="{}">
（这是一个凹函数）</del></blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">本文原来提供的证明有缺陷，我们无法证明在整个R上$\frac{1}{\sqrt{1+8e^x}}$都是凹函数。因此需要把证明改正：</p><p data-spaces-tag="p" data-spaces-attrs="{}">同样是用Jensen不等式，但是对函数$f(x)=\frac{1}{\sqrt{x}}$使用，可以证明$f''(x) &gt;0$对于所有正数x都成立，于是Jensen不等式可以使用（前提）。</p><p data-spaces-tag="p" data-spaces-attrs="{}">我们不难得出：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}(\frac{a}{a+b+c})\frac{1}{\sqrt{a^2+8bc}}+ (\frac{b}{a+b+c})\frac{1}{\sqrt{b^2+8ac}} + (\frac{c}{a+b+c})\frac{1}{\sqrt{c^2+8ab}} \\ \geq \frac{1}{\sqrt{(\frac{a}{a+b+c})(a^2+8bc)+(\frac{b}{a+b+c})(b^2+8ac)+(\frac{c}{a+b+c})(c^2+8ab)}} \\ =\sqrt{\frac{a+b+c}{a^3+b^3+c^3+24abc}}\end{aligned}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">于是：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{a}{\sqrt{a^2+8bc}}+ \frac{b}{\sqrt{b^2+8ac}} + \frac{c}{\sqrt{c^2+8ab}} \geq \sqrt{\frac{(a+b+c)^3}{a^3+b^3+c^3+24abc}}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">下面只需要证明$(a+b+c)^3 \geq a^3+b^3+c^3+24abc$就行了，利用平均不等式很容易得出结果，不再详述。</p><p data-spaces-tag="p" data-spaces-attrs="{}">以上证明来源于《美国奥数生》</p>
</div>
