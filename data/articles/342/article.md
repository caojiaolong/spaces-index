# 【竖直上抛】炮弹能够射多高(第二宇宙速度)？

> 作者：苏剑林 · 科学空间 · 2010-01-17
>
> 原文：<https://spaces.ac.cn/archives/342>
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
<blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">一枚炮弹以速度$v_0$向上射出，只考虑重力因素，请问炮弹到达多远的距离后就会开始自由下落？</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}"></p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;], &quot;style&quot;: &quot;float:right; margin: 5px auto 2px 1px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;display: inline-block; background-color: #fff&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 210px; margin: 0 0 0 8px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/attachment/343/&quot;, &quot;title&quot;: &quot;大炮的发射.jpg&quot;}" href="https://spaces.ac.cn/attachment/343/" title="大炮的发射.jpg"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;大炮的发射&quot;, &quot;src&quot;: &quot;/usr/uploads/2010/01/2807531250.jpg&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2010/01/2807531250.jpg" alt="大炮的发射"></a></div><p data-spaces-tag="p" data-spaces-attrs="{}">大炮的发射</p></div></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">对于这个问题，我们首先采取的是高中生的做法。考虑地球重力，也就是说炮弹在做加速度为<fong data-spaces-tag="fong" data-spaces-attrs="{&quot;color&quot;: &quot;blue&quot;}" color="blue">-g（-9.8m/s<sup data-spaces-tag="sup" data-spaces-attrs="{}">2</sup>）的匀变速运动。根据公式$v_t^2-v_0^2=2as$，可得$s=\frac{v_0^2}{2g}$。<br data-spaces-tag="br" data-spaces-attrs="{}">
此即炮弹能够走得最远距离。</fong></p><p data-spaces-tag="p" data-spaces-attrs="{}">但是看了这条式子，我们会发现，这个“距离”始终是有限的。换一句话说，只要$v_0$不趋于无穷大，s就不会无穷大。但是我们还听到过牛顿这样说过：<b data-spaces-tag="b" data-spaces-attrs="{}">假如炮弹以某个速度（就是我们现在所所说的第二宇宙速度）飞离地球，它就永远不会回来了。</b>两者不是矛盾吗？</p><p data-spaces-tag="p" data-spaces-attrs="{}">看了之前我写的<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/337/&quot;}" href="https://spaces.ac.cn/archives/337/">这篇文章</a>的朋友，也是马上就有头绪了。这个加速度a并不是恒定的。恭喜你，答对了！但具体的情况是怎样的呢？请继续往下看——</p><p data-spaces-tag="p" data-spaces-attrs="{}">设炮弹的路程为s，则在运动过程中：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s''=v'=-\frac{GM}{(r+s)^2}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">令$s''=v \frac{dv}{ds}$，代入<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">（这个过程是多么的熟悉）</font><br data-spaces-tag="br" data-spaces-attrs="{}">
$$vdv=-GM(r+s)^{-2} ds$$<br data-spaces-tag="br" data-spaces-attrs="{}">
两端积分：$\int vdv=\int -GM(r+s)^{-2} ds$<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{1}{2} v^2=GM(r+s)^{-1} +C$$<br data-spaces-tag="br" data-spaces-attrs="{}">
<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">下面的处理有些不同：</font><br data-spaces-tag="br" data-spaces-attrs="{}">
当$v=v_0$时，有s=0，则<br data-spaces-tag="br" data-spaces-attrs="{}">
$$C=\frac{1}{2} v_0^2 - GMr^{-1}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
得出：$v^2=v_0^2+2GM[(r+s)^{-1}-r^{-1}]$<br data-spaces-tag="br" data-spaces-attrs="{}">
炮弹走最远即当v=0时的s的值。于是有$0=v_0^2+2GM[(r+s)^{-1}-r^{-1}]$<br data-spaces-tag="br" data-spaces-attrs="{}">
$s=\frac{v_0^2 r^2}{2GM-v_0^2 r}$————（A）</p><p data-spaces-tag="p" data-spaces-attrs="{}">其中又有$GM=r^2 g$，g是1kg物体在地球表面所受到的重力，代入<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s=\frac{v_0^2 r}{2rg-v_0^2}=\frac{v_0^2}{2g-\frac{v_0^2}{r}}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">若$\frac{v_0^2}{r}$很小，则可以忽略，得到低速近似的伽利略公式：$s=\frac{v_0^2}{2g}$</font><br data-spaces-tag="br" data-spaces-attrs="{}">
<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">这与文章开头的结果是一致的。</font></p><p data-spaces-tag="p" data-spaces-attrs="{}"><b data-spaces-tag="b" data-spaces-attrs="{}">我们发现，当$v=\sqrt{\frac{2GM}{r}}$时，就会出现分母为0的情况，也就是说$s-&gt;\infty$，这时也就是牛顿所说的一去不复返（只是不复返回地球）。于是我们就自然而然地推导出了第二宇宙速度！$v=\sqrt{\frac{2GM}{r}}$</b></p><p data-spaces-tag="p" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red"><b data-spaces-tag="b" data-spaces-attrs="{}">Yeah！欢呼吧！科学应该要这样，尽管一点的成就，也应该雀跃。但是记住，不要沾沾自喜！</b></font></p>
</div>
