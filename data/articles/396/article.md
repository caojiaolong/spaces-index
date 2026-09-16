# 大气光学质量(Airmass)

> 作者：苏剑林 · 科学空间 · 2010-02-04
>
> 原文：<https://spaces.ac.cn/archives/396>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">天文学中有一个名词Airmass，注意这并非Air mass（空气质量），这是<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;blue&quot;}" color="blue"><strong data-spaces-tag="strong" data-spaces-attrs="{}">指天顶距等于z的方向上大气光学厚度和天顶方向大气光学厚度之比</strong></font><del data-spaces-tag="del" data-spaces-attrs="{}">，我目前也找不到它的中文名称究竟是什么，反正觉得如果译成“大气质量”很怪，就暂且翻译成“大气厚度指数”好了。</del>现在知道它叫做“大气光学质量”了，一般用X表示，如下图中，$X={BC}/{AC}$。<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 100%&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/attachment/398/&quot;, &quot;title&quot;: &quot;星光传播示意图.JPG&quot;}" href="https://spaces.ac.cn/attachment/398/" title="星光传播示意图.JPG"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;星光传播示意图&quot;, &quot;src&quot;: &quot;/usr/uploads/2010/02/1077770125.jpg&quot;, &quot;style&quot;: &quot;max-width:100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2010/02/1077770125.jpg" alt="星光传播示意图"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">星光传播示意图</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">在一片较小的区域内，大气层和地面都可以视为平行平面，这时有一个很好的近似公式：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=\sec z$$<br data-spaces-tag="br" data-spaces-attrs="{}">
对于现在的中学教材来说，有的读者可能不了解\sec为何物，实际上：$\sec z=\frac{1}{\cos z}$<br data-spaces-tag="br" data-spaces-attrs="{}">
在60°的时候，airmass约等于2。然而地球是不平坦的，根据不同的精度要求，z的峰值为60-75°。随着天顶角的变大，该公式的准确度迅速降低；该式会在地平线的时候趋于无穷大，而根据实际的弯曲大气情况，airmass通常不会大于40。</p><p data-spaces-tag="p" data-spaces-attrs="{}">实际上，如果<u data-spaces-tag="u" data-spaces-attrs="{}">不考虑大气折射，把地球看成一个球体，而大气层也视为高度为y的空心球体，只从几何的角度来看，则从天顶角z发出来的光线要穿透的大气厚度为</u>（$R_E$为地球半径）：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s=\sqrt{R_E^2 \cos^2 z+2R_E \cdot y+y^2}-R_E \cdot\cos z$$<br data-spaces-tag="br" data-spaces-attrs="{}">
或者写成<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s=\sqrt{(R_E +y)^2 - R_E^2 \sin^2 z}-R_E \cdot\cos z$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">但实际中还必须考虑折射等因素，所以很多<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;blue&quot;}" color="blue"><u data-spaces-tag="u" data-spaces-attrs="{}">插值公式</u></font>便出来了。例如， <strong data-spaces-tag="strong" data-spaces-attrs="{}">Young和Irvine</strong>于1967年在$X=\sec z$基础上加入了一个修正因子：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=\sec z_t [1-0.0012(\sec^2 z_t -1)]$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这里的$z_t$是真实的天顶角，也就是修正了大气折射后的天顶角。这样处理后，天顶角z的峰值可以达到80°。但同样随着天顶角的变大准确度也迅速降低；该式会在z=86.6°时达到11.13的最大值，而在地平线的时候趋于负无穷大。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">Hardie</strong>在1962年使用了$\sec z-1$的多项式来修正：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=\sec z-0.0018167(\sec z-1)-0.002875(\sec z-1)^2 -0.0008083(\sec z-1)^3$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这提供了高达85°的峰值，不过和上式一样，该式会达到一个最大值，然后在地平线的时候趋于负无穷大。</p><p data-spaces-tag="p" data-spaces-attrs="{}">1966年<strong data-spaces-tag="strong" data-spaces-attrs="{}">Rozenberg</strong>提出了<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=(\cos z+0.025e^{-11\cos z})^{-1}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
该法在地平线的时候依旧能够得到合理的值（z=90°时airmass约等于40）</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">Kasten和Young</strong>在1989年发展成<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=\frac{1}{\cos z+0.50572(96.07995-z)^{-1.6364}}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
公式在z接近90°仍然相当合理，在地平线时的结果约为38，注意这里的z必须使用角度为单位。</p><p data-spaces-tag="p" data-spaces-attrs="{}">1994年<strong data-spaces-tag="strong" data-spaces-attrs="{}">Young</strong>得出<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=\frac{1.002432\cos^2 z_t + 0.148386\cos z_t + 0.0096467}{\cos^3 z_ t + 0.149864\cos^2 z_t + 0.0102963\cos z_t + 0.000303978}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
同样$z_t$为真实天顶角。这条公式可以将误差（即使是地平线的结果）控制在0.0037内。</p><p data-spaces-tag="p" data-spaces-attrs="{}">2002年<strong data-spaces-tag="strong" data-spaces-attrs="{}">Pickering</strong>派生出了公式<br data-spaces-tag="br" data-spaces-attrs="{}">
$$X=\frac{x}{\sin(h+\frac{244}{165+47h^{1.1}})}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
h=90°-z</p><p data-spaces-tag="p" data-spaces-attrs="{}">下图的曲线比较了不同的插值公式准确度：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 500px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/attachment/397/&quot;, &quot;title&quot;: &quot;不同的插值公式准确度比较&quot;}" href="https://spaces.ac.cn/attachment/397/" title="不同的插值公式准确度比较"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;不同的插值公式准确度比较&quot;, &quot;src&quot;: &quot;/usr/uploads/2010/02/3634003498.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2010/02/3634003498.png" alt="不同的插值公式准确度比较"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">不同的插值公式准确度比较</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">差值公式是从不同的运算过程中导出的近似公式，它只考虑了z这个因素，因为在该问题中，我们只有z这个变量。在一定程度上，插值公式方便了我们的计算。尽管看起来有些公式很麻烦，但对于计算机时代来说，公式的关键是：有效、准确。</font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">更多内容可以参考：<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://en.wikipedia.org/wiki/Airmass&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;维基百科Airmass&quot;}" href="http://en.wikipedia.org/wiki/Airmass" title="维基百科Airmass">http://en.wikipedia.org/wiki/Airmass</a></p>
</div>
