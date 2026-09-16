# 一道级数求和证明题(非数学归纳法)

> 作者：苏剑林 · 科学空间 · 2009-08-02
>
> 原文：<https://spaces.ac.cn/archives/49>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">今天在<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://bbs.emath.ac.cn/thread-1670-1-1.html&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://bbs.emath.ac.cn/thread-1670-1-1.html">数学研发论坛</a>看到了一道题目：</p><p data-spaces-tag="p" data-spaces-attrs="{}">$$\sum_{j=0}^{j=n} (jx^j)={nx^{n+2}-(n+1)x^{n+1}+x}/{(x-1)^2}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">这道题实际是求$x+2x^2+3x^3+...+nx^n$的求和公式而已。</p><p data-spaces-tag="p" data-spaces-attrs="{}">本来呢用数学归纳法是十分简单的（数学归纳法对于证明简单，对于推导就不行了），但是题目说不能用数学归纳法。只好用以下方法了。</p><p data-spaces-tag="p" data-spaces-attrs="{}">我们把它改写成：</p><p data-spaces-tag="p" data-spaces-attrs="{}">$$\begin{aligned}x(1+2x+3x^2+...+nx^{n-1}) \\ \Rightarrow x[1+x+x^2+...+x^{n-1}+x(1+x+x^2+...+x^{n-2})+...+x^{n-1}] \\ \Rightarrow x[{x^n-1+x(x^{n-1}-1)+x^2(x^{n-2}-1)+...+x^{n-1}(x-1)}/{x-1}] \\ \Rightarrow x[{nx^n-(1+x+x^2+...+x^{n-1})}/{x-1}] \\ \Rightarrow x[{nx^n-{x^n-1}/{x-1}}/{x-1}]\end{aligned}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
$\Rightarrow x[{nx^{n+1}-(n+1)x^n+1}/{(x-1)^2}]</p><p data-spaces-tag="p" data-spaces-attrs="{}">剩下来的就不写了。不用说了吧？</p>
</div>
