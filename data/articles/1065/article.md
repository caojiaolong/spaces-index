# 《自然极值》系列——1\.前言

> 作者：苏剑林 · 科学空间 · 2010-11-27
>
> 原文：<https://spaces.ac.cn/archives/1065>
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
<p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red"><strong data-spaces-tag="strong" data-spaces-attrs="{}">附：</strong>期中考过后，课程紧了，自由时间少了，因此科学空间的更新也放缓了。不过BoJone也会尽量地更新一些内容，和大家一同分享学习的乐趣。</font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}"></p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 100%&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/attachment/1066/&quot;, &quot;title&quot;: &quot;闭区间[a,b]上的连续函数?(x)，其最大值为红色点，最小值为蓝色点&quot;}" href="https://spaces.ac.cn/attachment/1066/" title="闭区间[a,b]上的连续函数?(x)，其最大值为红色点，最小值为蓝色点"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;闭区间[a,b]上的连续函数?(x)，其最大值为红色点，最小值为蓝色点&quot;, &quot;src&quot;: &quot;/usr/uploads/2010/11/3941873990.png&quot;, &quot;style&quot;: &quot;max-width:100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2010/11/3941873990.png" alt="闭区间[a,b]上的连续函数?(x)，其最大值为红色点，最小值为蓝色点"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">闭区间[a,b]上的连续函数?(x)，其最大值为红色点，最小值为蓝色点</p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}">上一周和这一周的时间里，BoJone将自己学习物理和极值的一些内容进行了总结和整合，写成了<strong data-spaces-tag="strong" data-spaces-attrs="{}">《自然极值》</strong>一文。因此从今天起，到十二月的大多数时间里，科学空间将和大家讲述并讨论关于“极值”的问题，希望读者会喜欢这部分内容。当然，我不是专业的研究人员，更不是经验丰富的物理和数学教师，甚至可以说是一个“乳臭未干的小子”，因此，错误在所难免，只希望同好不吝指出，<u data-spaces-tag="u" data-spaces-attrs="{}">更希冀能够起到我抛出的这一块“砖”能够引出美妙的“玉”。</u></p><p data-spaces-tag="p" data-spaces-attrs="{}">《自然极值》主要包括以下内容：</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">1、费马原理（光学部分）<br data-spaces-tag="br" data-spaces-attrs="{}">
2、平衡态公理（势能最小原理）<br data-spaces-tag="br" data-spaces-attrs="{}">
3、费马点问题（以上两个原理的应用）<br data-spaces-tag="br" data-spaces-attrs="{}">
4、最速降线问题<br data-spaces-tag="br" data-spaces-attrs="{}">
5、悬链线问题<br data-spaces-tag="br" data-spaces-attrs="{}">
6、极值的初步剖析（导出变分学的一道基本公式）</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;colo&quot;: &quot;blue&quot;}">以上内容都将尽可能初等地讲述（为什么“尽可能初等”？其中一个很重要的原因是：高等的讲述我不会，呵呵^_^）</font></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">前言部分：</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">在数学上，求极值大概是我们处理得最多的问题之一了。对于数学工作者来说，求极值就是把函数输入计算机，然后求导数或者偏导数（我们所学的拉格朗日乘数法），令其为0，继而把结果“暴力”出来。诚然，在许多情况下，这种模式是必须的，它适应了“效率型”社会。不过，对于热爱物理，或者是热衷于寻找科学之美的朋友来说，这个处理过程尤显单调乏味。同时，很多计算的问题都是为了解决实际问题，如果其解法能够回归到物理范畴，显然是美不胜收的。<u data-spaces-tag="u" data-spaces-attrs="{}">正如我们宁愿象爱迪生一样，用装水测量的方法计量梨形灯泡的体积，而不是像阿普拉那样进行繁琐的积分。</u></p><p data-spaces-tag="p" data-spaces-attrs="{}">因此，我们都希望能够从自然界中寻找到一些科学事实（一些可以称为“公理”或“原理”的东西），变成解决数理问题的工具，甚至将其发展成实在而完善的理论。于是，BoJone尝试写了《自然极值》一文，介绍一下这样的一个过程。</p>
</div>
