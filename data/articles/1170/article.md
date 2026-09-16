# 不可能事件——一道经典电磁感应题的错误

> 作者：苏剑林 · 科学空间 · 2011-01-09
>
> 原文：<https://spaces.ac.cn/archives/1170>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">相信高二理科的学生都会做过这样的一道题目：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}"></p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 500px&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/attachment/1171/&quot;, &quot;title&quot;: &quot;光滑导轨-电磁感应.PNG&quot;}" href="https://spaces.ac.cn/attachment/1171/" title="光滑导轨-电磁感应.PNG"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;光滑导轨-电磁感应&quot;, &quot;src&quot;: &quot;/usr/uploads/2011/01/3869165784.png&quot;, &quot;style&quot;: &quot;max-width: 100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2011/01/3869165784.png" alt="光滑导轨-电磁感应"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">光滑导轨-电磁感应</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">水平放置于匀强磁场中的光滑导轨上，磁感应强度为B，平衡导轨的距离为L，有一根导体棒ab，用恒力F作用在ab上，由静止开始运动，回路总电阻为R，求ab的最大速度。</p></blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">对于高二学生来说，这样的题目是很好解决的。只要列出<br data-spaces-tag="br" data-spaces-attrs="{}">
$E=BLv,I=\frac{E}{R},f_1=BIL$，并根据当匀速运动时速度最大，由受力平衡有$f_1=F$，解得<br data-spaces-tag="br" data-spaces-attrs="{}">
（E：感应电动势；I：感应电流；f<sub data-spaces-tag="sub" data-spaces-attrs="{}">1</sub>：安培力）<br data-spaces-tag="br" data-spaces-attrs="{}">
$$v=\frac{FR}{B^2 L^2}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">看似非常合理，可是BoJone在进一步分析这个问题时，却发现了很奇怪的现象：它根本不可能达到匀速运动状态！</p><p data-spaces-tag="p" data-spaces-attrs="{}">我们从高等数学角度分析一下这个问题。设ab质量为m，其受力是$F-f_1=F-BIL=F-\frac{B^2 L^2 v}{R}$，由牛顿第二定律得到$ma=F-\frac{B^2 L^2 v}{R}$，记$p=\frac{F}{m},q=\frac{B^2 L^2 }{mR}$，得到<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\ddot{s}=p-q\dot{s}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">这是一个二阶微分方程，积分一次就得到$\dot{s}=pt-qs+C_1$，并且开始时候位移和速度都为0，有$C_1=0$</p><p data-spaces-tag="p" data-spaces-attrs="{}">解方程$\dot{s}=pt-qs$得：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s = C_2 e^{-q t}+\frac{p}{q}t-\frac{p}{q^2}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">同样根据开始时候位移为0得到$C_2=\frac{p}{q^2}$，于是就有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$v=\dot{s}=\frac{p}{q}(1-e^{-qt})$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">显然，最大速度为$V_{max}=\frac{p}{q}=\frac{FR}{B^2 L^2}$，结果和高中的经典做法一样。可是，我们惊讶地发现，这个结果的取值，需要$t \to \infty$。换句话说，<b data-spaces-tag="b" data-spaces-attrs="{}"><u data-spaces-tag="u" data-spaces-attrs="{}">根本就达不到这个状态！</u></b></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">思考</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">显然，这个例子说明了教材的命题者并没有经过进一步的思考，草率地出了这一道题目，犯了“想当然”的错误。BoJone不禁提出一个疑问：<u data-spaces-tag="u" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">对于一个中国高中生来说，进行上述分析的确过于苛刻，但是对于一个教材命题者来说，上述过程显然是可以轻而易举地完成的（受力分析，解答二阶线性微分方程），但为什么这一道“谬题”一出再出？<del data-spaces-tag="del" data-spaces-attrs="{}">难道<strong data-spaces-tag="strong" data-spaces-attrs="{}">功利性</strong>足以掩盖科学的<strong data-spaces-tag="strong" data-spaces-attrs="{}">严谨性</strong>？还是高中物理不需要严谨？还是欺负高中生不懂数学分析而随便“蒙混过关”？</del>（BoJone承认，这评论过激了，抱歉...）</font></u></p><p data-spaces-tag="p" data-spaces-attrs="{}">另一方面，这也告诉我们，不能盲目迷信权威。当我们拥有了更充足的科学知识时，不妨回头看看，对我们曾经“解决”过的问题进行再次分析，或许会有意外的收获！</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">后注：</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
<u data-spaces-tag="u" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">物理定律始终有个精确度和适用范围问题，虽然本题的时间为无穷大，但是只要给定精确度，就存在一个有限大的时间内可以达到这个精确度，因此本题也无可厚非。本文（原文）写于高中的“血气方刚”之时，正好发现这现象，稍微激动了一些。也正是因为这道题，让我对物理的意义了解更为深入了一些。正所谓人生处处皆学问呀。（2011.02.12）</font></u></p>
</div>
