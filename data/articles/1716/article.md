# 均值不等式的两个巧妙证明

> 作者：苏剑林 · 科学空间 · 2012-09-26
>
> 原文：<https://spaces.ac.cn/archives/1716>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">记得几年前，BoJone提供过一个证明均值不等式（代数—几何平均不等式）的方法，但是其中的证明有点长，有点让人眼花缭乱的感觉（虽然里边的思想还是挺简单的）。昨天在上《数学分析》课程的时候，老师讲到了这个不等式，也讲了他的证明，用的是数学归纳法，感觉还是没有那种简洁美和巧妙美。但这让我回想起了之前我研究过的两种巧妙证明方法，可是在昨天划了一整天，都没有把这两种方法回忆起来。直到今天才回想起来，所以就放在这里与大家分享，同时也作备忘之用。</p><p data-spaces-tag="p" data-spaces-attrs="{}">对于若干个非负数$x_i$，我们有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{x_1+x_2+...+x_n}{n} \geq \sqrt[n]{x_1 x_2 ... x_n}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">记为$A_n \geq G_n$</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">证明1：数学归纳法</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
这个方法不算简单，但是非常巧妙，它从n递推到n+1的过程让人拍案叫绝。用数学归纳法证明詹森不等式也是同样的递推思路，而均值不等式不过是詹森不等式的一个特例而已。</p><p data-spaces-tag="p" data-spaces-attrs="{}">假设$A_n \geq G_n$成立，要证$A_{n+1} \geq G_{n+1}$。我们有</p><p data-spaces-tag="p" data-spaces-attrs="{}">$$\begin{aligned}&amp;2n A_{n+1}=(n+1)A_{n+1}+(n-1)A_{n+1} \\<br data-spaces-tag="br" data-spaces-attrs="{}">
=&amp;[x_1 + x_2 +...+x_n]+[x_{n+1}+(n-1)A_{n+1}] \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\geq &amp;nG_n+n(x_{n+1}\cdot A_{n+1}^{n-1})^{\frac{1}{n}} \\<br data-spaces-tag="br" data-spaces-attrs="{}">
\geq &amp;2n(G_{n+1}^{n+1}\cdot A_{n+1}^{n-1})^{\frac{1}{2n}}\end{aligned}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">化简即得：$A_{n+1} \geq G_{n+1}$</p><p data-spaces-tag="p" data-spaces-attrs="{}">这就完成了从n到n+1的递推。其他细节略。</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">证明2：对数方法</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">这个方法更巧妙，更简单，有可能是我见过的最简单方法了。它利用的一个很简单的公式是对于所有<font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red"><del data-spaces-tag="del" data-spaces-attrs="{}"><u data-spaces-tag="u" data-spaces-attrs="{}">非负数</u></del></font>x，有$e^x \geq 1+x$，并且当$x\geq -1$时，两边都是非负数。</p><p data-spaces-tag="p" data-spaces-attrs="{}">（为了方便排版，记$exp(x)=e^x$）</p><p data-spaces-tag="p" data-spaces-attrs="{}">$$\begin{aligned}&amp;\exp\left(\frac{n A_n}{G_n}-n\right) \\ =&amp;\exp\left(\frac{x_1}{G_n}-1\right)\cdot \exp\left(\frac{x_2}{G_n}-1\right)\dots\exp\left(\frac{x_n}{G_n}-1\right) \\ \geq &amp;\frac{x_1}{G_n}\cdot \frac{x_2}{G_n}...\frac{x_n}{G_n} \quad\text{(前面的各项指数部分显然都大于等于-1)}\\ =&amp;1\end{aligned}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">也就是说<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\exp\left(n\frac{A_n}{G_n}-n\right) \geq 1$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">稍稍化简就有$A_n \geq G_n$</p>
</div>
