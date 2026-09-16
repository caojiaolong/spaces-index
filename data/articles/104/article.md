# 正十七边形的尺规作图

> 作者：苏剑林 · 科学空间 · 2009-08-28
>
> 原文：<https://spaces.ac.cn/archives/104>
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
<p data-spaces-tag="p" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Red&quot;}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">为何正17边形能够用尺规作出来？要如何作？</strong>先别急，请看下面的解释：</span></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Blue&quot;}">一个正质数多边形可以用标尺作图的充分和必要条件是，该多边形的边数必定是一个费马质数。</span></strong>换句话说，只有正三边形、正五边形、正十七边形、正257边形和正63357边形可以用尺规作出来，其它的正质数多边形就不可以了。（除非我们再发现另一个费马质数。）</p><p data-spaces-tag="p" data-spaces-attrs="{}">正17边形的尺规作法是高斯在1796年得出的，他也因此决心要成为数学家。<span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Green&quot;}">关于费马质数，是指形如$2^{2^n}+1$的质数，一开始费马认为对于所有的n，这种形式的数都是质数。可是这似乎是上天的玩笑，<strong data-spaces-tag="strong" data-spaces-attrs="{}">目前</strong>只发现了当n=0,1,2,3,4的时候$2^{2^n}+1$是质数，其余都是合数。</span></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">黎西罗给出了正257边形的尺规作法，写满了整整80页纸。盖尔梅斯给出了正65537边形的尺规作法，此手稿整整装满了一只手提箱，现存于德国哥廷根大学。这是有史以来最繁琐的尺规作图。</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">要证明正17边形可用尺规作出，其实很简单，因为我们有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\cos\frac{2\pi}{17}= \frac{-1+\sqrt{17}+\sqrt{34-2\sqrt{17}}+2\sqrt{17+3\sqrt{17}-\sqrt{34-2\sqrt{17}}-2\sqrt{34+2\sqrt{17}}}}{16}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Red&quot;}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">尺规作图必要和充分的条件是：</strong>所作长度能够从1出发，通过有限次的加、减、乘、除和开平方运算得出，这条长度就能够以尺规作图的形式作出来。</span>而$\cos\frac{2\pi}{17}$符合这个条件，因此正17边形可以用尺规作出来。<span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Blue&quot;}">（至于$cos\frac{2\pi}{17}$，是通过一系列的三角函数计算得出，读者不妨试试，我还没有找到详细过程）</span></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Blue&quot;}">说了这么多，要进入正题了，尺规作图的方法来了：</span></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">1、GIF版本（这个貌似太复杂了）</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 100%&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/sci/r17/r17.gif&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/sci/r17/r17.gif" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;&quot;, &quot;src&quot;: &quot;/sci/r17/r17.gif&quot;, &quot;style&quot;: &quot;max-width:100%&quot;}" src="https://spaces.ac.cn/sci/r17/r17.gif" alt=""></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}"></p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">2、Flash (1)</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
<details data-spaces-opaque="1"><summary>原文嵌入内容（embed；预览不执行）</summary><pre><code>&lt;embed height="400" src="/sci/r17/r17_1.swf" type="application/x-shockwave-flash" width="500"/&gt;</code></pre></details></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">3、Flash (2)</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
<details data-spaces-opaque="1"><summary>原文嵌入内容（embed；预览不执行）</summary><pre><code>&lt;embed height="400" src="/sci/r17/r17_2.swf" type="application/x-shockwave-flash" width="500"/&gt;</code></pre></details></p>
</div>
