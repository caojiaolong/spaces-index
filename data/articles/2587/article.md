# &#91;问题解答&#93;运煤车的最大路程（更正）

> 作者：苏剑林 · 科学空间 · 2014-05-04
>
> 原文：<https://spaces.ac.cn/archives/2587>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">刚刚在浏览<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://weibo.com/ChanghaiNews&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://weibo.com/ChanghaiNews">卢昌海大师的微博</a>时，发现他微博上有一道比较有趣的题目，于是饶有兴致地思考了一翻，构思了一个答案，希望读者们看看这个答案有问题不？</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}"><p data-spaces-tag="p" data-spaces-attrs="{}">五一”长假微博很闷，出一道题给博友们解闷：</p><p data-spaces-tag="p" data-spaces-attrs="{}">用重载列车运煤，每次可装1万吨，每行驶1公里耗煤1吨，起点处共有N万吨煤（简单起见N为正整数），请问最远可运至何处（是国营煤老板，成本不计，只要运到的数量大于0就算成功）？并求$N\to\infty$时的渐进形式。</p></blockquote><p data-spaces-tag="p" data-spaces-attrs="{}"></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">分析：</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
说白了，这个问题就是问火车最远可以走多远，问题的关键是要尽可能地消耗完所有的煤。<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">为了达成这个目的，需要来回多趟，并且尽可能用满列车的运载量。还有一个要求，就是不要在同一段路上消耗太多的煤。</font>（即来回的次数尽可能少，下面的错误解法的错误就在于在前面的路消耗了太多的煤。）</p><p data-spaces-tag="p" data-spaces-attrs="{}">得益于<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://wuling.name/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://wuling.name/">伍岭</a>博友的启示，<strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;green&quot;}" color="green">最优的做法应该是“逐步推进”，也就是说先把煤集中运到前方的一个点上（运输过程中会有消耗），然后再从前方的那个点出发，把煤运输到更前方的一个点上，如此重复下去，就得到最远的距离。</font></strong>下面对这个过程进行分析。</p><p data-spaces-tag="p" data-spaces-attrs="{}">题目假设了$N$是正整数，当然，就算$N$不是整数也可以进行同样的分析。（事实上，假设$N$是整数并没有带来太大的方便。）我们这里用到“上取整函数”$\lceil x \rceil$，定义为不小于$x$的最小整数。</p><p data-spaces-tag="p" data-spaces-attrs="{}">从原点出发，<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">每次运输1<strong data-spaces-tag="strong" data-spaces-attrs="{}">万吨</strong>，运输到前方$y$<strong data-spaces-tag="strong" data-spaces-attrs="{}">万公里</strong>处，共来回运输$\lceil N \rceil-1$次，每次卸下一部分煤后，回到原点刚好消耗完。最后一次不用返回去</font>，所以经过这样一折腾，火车在距离原点$y$处，并且那里剩下的煤量为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$(\lceil N \rceil-1)(1-2y)+N+1-\lceil N \rceil-y=N+y-2y\lceil N \rceil$$<br data-spaces-tag="br" data-spaces-attrs="{}">
每一段路都是重复这个过程，用$N_i$表示每次剩下的煤量，用$y_i$表示每次前进的路程，于是<br data-spaces-tag="br" data-spaces-attrs="{}">
$$N_i=N_{i-1}+y_{i-1}-2y_{i-1}\lceil N_{i-1} \rceil;\ i=1,2,\dots,N;\ N_0=N$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这是一个带有取整函数的递推问题，同时带有可调参数$y_i$。<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red"><u data-spaces-tag="u" data-spaces-attrs="{}">稍微经过分析就可以发现，每一段路中，都是越靠近原点的那一段路，平均耗煤量越大。因此，每个$y_i$都是越小越好，因此，最优的情况是所有的$y_i$都有$y_i\to 0$。</u></font></p><p data-spaces-tag="p" data-spaces-attrs="{}">这是这样的一个过程：<strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;green&quot;}" color="green">运煤，然后走一丁点的距离，卸煤，回去，然后运煤，走一丁点的距离，卸煤，回去。严格意义上来说，这种运作是无法实现的。但是，这不妨碍我们在数学上理解它。</font></strong>另一方面，连续的结果也可以作为离散的一个近似。那么<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{N_i-N_{i-1}}{y_{i-1}}=1-2\lceil N_{i-1} \rceil$$<br data-spaces-tag="br" data-spaces-attrs="{}">
当$y_{i-1}\to 0$时，记$N_i=N(s)$，$s$是到起点的距离，由导数的定义得<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{dN(s)}{ds}=1-2\lceil N(s) \rceil$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这是一道带有取整函数的微分方程。虽然取整函数使得我们不能直接应用我们教科书上的通解公式，但是某种意义上来说，这种方程更加简单。因为经过分段之后，它就是一道$\frac{dy}{dx}=Constant$的最简单的方程了。</p><p data-spaces-tag="p" data-spaces-attrs="{}">以上的推导并没有基于$N$是整数。当然，现在看来，$N$是整数能够给出形式上比较简单的解。在$N(s)\in (N-1,N]$时，有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{dN(s)}{ds}=1-2N$$<br data-spaces-tag="br" data-spaces-attrs="{}">
它的解是<br data-spaces-tag="br" data-spaces-attrs="{}">
$$N(s)=(1-2N)s+N$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这个解的生效区间是$N(s)\in (N-1,N]$，即只维持一个单位，所以在第一步，火车走了$\frac{1}{2N-1}$。<br data-spaces-tag="br" data-spaces-attrs="{}">
第二步，在$N(s)\in (N-2,N-1]$时，有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{dN(s)}{ds}=1-2(N-1)$$<br data-spaces-tag="br" data-spaces-attrs="{}">
它的解是（每一步都重新选择原点）<br data-spaces-tag="br" data-spaces-attrs="{}">
$$N(s)=(3-2N)s+N-1$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这个解的生效区间是$N(s)\in (N-2,N-1]$，所以在第二步，火车走了$\frac{1}{2N-3}$。我们发现，这个过程是可以递推的，最终的结果是<br data-spaces-tag="br" data-spaces-attrs="{}">
$$S=1+\frac{1}{3}+\dots+\frac{1}{2N-1}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
可以估算<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{1}{2}\ln(2N-1) &lt; S &lt; \frac{1}{2} \ln(2N-3)+1$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">当$N$不是整数时，记忆$\tilde{N}=\lceil N \rceil -1$，则<br data-spaces-tag="br" data-spaces-attrs="{}">
$$S=\frac{N-\tilde{N}}{2\tilde{N}+1}+1+\frac{1}{3}+\dots+\frac{1}{2\tilde{N}-1}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">望各位读者继续指正。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">以下是站长昨天（2014年5月4日）的错误解答：</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
===========================================<br data-spaces-tag="br" data-spaces-attrs="{}">
说白了，这个问题就是问火车最远可以走多远，问题的关键是要尽可能地消耗完所有的煤。为了达成这个目的，需要来回多趟，并且尽可能用满列车的运载量。在N是整数的假设下，比较理想的情况是走N趟，每趟的出发都运1万吨，然后中途放下一部分，再回去，回去到起点时，列车上的煤刚好用完，再运一万吨，以此类推。在最后一趟时，由于不用再回去了，所以尽可能地消耗所有的煤，再遇到第一堆之前卸下的煤时，装上去，刚好装满；遇到下一堆煤时，装上去，刚好装满...这样子就完全消耗了所有的煤。下面分析这种情况。</p><p data-spaces-tag="p" data-spaces-attrs="{}">设走$N$趟（从起点出发一次叫一趟），每趟装1万吨，走$a_i$路程（万公里），路程也就是耗煤量$a_i$（万吨），$i=1,2,\dots,N-1$，每趟在途中卸下的煤量为$1-2a_i$。考虑最后一趟，根据上面的假设，它跟之前卸下的煤相遇时，把煤装上，刚好装满，于是有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$(1-2a_i)+[1-(a_i-a_{i-1})]=1,i=1,2,\dots,N-1,a_0=0$$<br data-spaces-tag="br" data-spaces-attrs="{}">
也就是<br data-spaces-tag="br" data-spaces-attrs="{}">
$$1+a_{i-1}=3a_i$$<br data-spaces-tag="br" data-spaces-attrs="{}">
解这个递推得<br data-spaces-tag="br" data-spaces-attrs="{}">
$$a_i=\frac{1}{2}\left(1-\frac{1}{3^i}\right)$$<br data-spaces-tag="br" data-spaces-attrs="{}">
于是最后一趟走的路程为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s=a_{N-1}+1=\frac{1}{2}\left(1-\frac{1}{3^{N-1}}\right)+1$$<br data-spaces-tag="br" data-spaces-attrs="{}">
其渐进形式（$N\to\infty$）为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$s=\frac{3}{2}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">闭门造车之解答，望各位读者指正。^_^</p>
</div>
