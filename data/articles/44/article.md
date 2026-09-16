# 关于无理数及其和的证明

> 作者：苏剑林 · 科学空间 · 2009-07-31
>
> 原文：<https://spaces.ac.cn/archives/44>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

![](<https://spaces.ac.cn/usr/uploads/2009/07/20090731094451.jpg>)在中学，有理数的定义为整数和分数的集合，统一来说就是能够写成两个整数之比的数。那相对地，无理数自然就是不能写成两个整数之比的数了，也就是无限不循环小数，比如$\pi,\sqrt{2}$等等。历史上无理数的发现带来了第一次数学危机，并生下了一颗“金蛋”，不过发现者却因此丢掉了生命。让我们永远铭记——希帕索斯(Hippasus)。

<strong>历史：</strong>

> [http://baike\.baidu\.com/view/1167\.htm\#2](<http://baike.baidu.com/view/1167.htm#2>)

在这里对无理数就不多说些什么了，主要是谈谈相关的证明而已。
先说明，以下是我自己的证明方法，当然我相信有一种方法是通用的，但是我没有找出来。

<strong>1\. $\sqrt{2}$是无理数。</strong>

> <strong>证：</strong>我们假设$\sqrt{2}$是有理数，并且令$\sqrt{2}=p/q$，$p/q$是最简分数。由于$\sqrt{2}$不是整数，所以q\>1。
> 
> 
> 
> 我们将两边平方，得到$2={p^2}/{q^2}$，因为$p/q$是最简分数，所以${p^2}/{q^2}$也是最简分数。2的最简分数只能够是$2/1$，由此得出q=1，这与q\>1矛盾。所以假设不成立，$\sqrt{2}$是无理数。

<strong>2\. $\sqrt{2}+\sqrt{3}$是无理数。</strong>

> <strong>证：</strong>这个简单了，相信大家都会。
> 
> 
> 
> 我们假设$\sqrt{2}+\sqrt{3}=p$是有理数，两边平方得到$5+2\sqrt{6}=p^2\Rightarrow \sqrt{6}={p^2-5}/2$，于是$\sqrt{6}$是有理数，矛盾。因此假设不成立。

<strong>3\. $\sqrt{2}+\sqrt{3}+\sqrt{5}$是无理数。</strong>

证：这个可不怎么简单了，我可是想了好久。

> (1) 我们假设$\sqrt{2}+\sqrt{3}+\sqrt{5}=p$是有理数，两边平方得到
> $10+2\sqrt{6}+2\sqrt{10}+2\sqrt{15}=p^2\Rightarrow \sqrt{6}+\sqrt{10}+\sqrt{15}={p^2-10}/2$，
> 于是$\sqrt{6}+\sqrt{10}+\sqrt{15}$是有理数。两边再次平方，得：
> $31+10\sqrt{6}+6\sqrt{10}+4\sqrt{15}={(p^2-10)^2}/4$，
> 于是$5\sqrt{6}+3\sqrt{10}+2\sqrt{15}={{(p^2-10)^2}/4-31}/2$
> 由于$\sqrt{6}+\sqrt{10}+\sqrt{15}$是有理数，所以
> $$3\sqrt{6}+\sqrt{10}={{(p^2-10)^2}/4-31}/2-2(\sqrt{6}+\sqrt{10}+\sqrt{15})$$
> 的左边是一个有理数，但由方法2可证明$3\sqrt{6}+\sqrt{10}$为无理数，矛盾，于是假设不成立。

> (2) 同样假设$\sqrt{2}+\sqrt{3}+\sqrt{5}=p$是有理数，我们将式子变成
> $\sqrt{2}+\sqrt{3}=p-\sqrt{5}$，两边平方：
> $$\begin{aligned}(\sqrt{2}+\sqrt{3})^2=(p-\sqrt{5})^2 \\ \Rightarrow 5+2\sqrt{6}=p^2+5-2p\sqrt{5}\end{aligned}$$
> 于是得出$\sqrt{6}+p\sqrt{5}$是有理数，也是矛盾的。

<strong>4\. $\sqrt{2}+\sqrt{3}+\sqrt{5}+\sqrt{7}$是无理数。</strong>

> <strong>证：</strong>在报纸上画了一整天，终于有头绪了。
> 
> 
> 
> 同样假设$\sqrt{2}+\sqrt{3}+\sqrt{5}+\sqrt{7}=p$是有理数，我们将式子变成
> $\sqrt{2}+\sqrt{3}+\sqrt{5}=p-\sqrt{7}$，两边平方得到：
> $$10+2\sqrt{6}+2\sqrt{10}+2\sqrt{15}=p^2+7-2p\sqrt{7}$$
> $\Rightarrow \sqrt{6}+\sqrt{10}+\sqrt{15}+p\sqrt{7}=(p^2-3)/2$是一个有理数。
> 两边继续平方：
> $$\begin{aligned}(\sqrt{6}+\sqrt{10}+\sqrt{15})^2=((p^2-3)/2-p\sqrt{7})^2 \\ \Rightarrow 31+10\sqrt{6}+6\sqrt{10}+4\sqrt{15}=((p^2-3)/2)^2+7p^2-p(p^2-3)\sqrt{7} \\ \Rightarrow 2\sqrt{10}+4\sqrt{6}+p(p^2-7)\sqrt{7}=((p^2-3)/2)^2+7p^2-4(\sqrt{6}+\sqrt{10}+\sqrt{15}+p\sqrt{7})\end{aligned}$$
> 由于$\sqrt{6}+\sqrt{10}+\sqrt{15}+p\sqrt{7}=(p^2-3)/2$是有理数，所以$2\sqrt{10}+4\sqrt{6}+p(p^2-7)\sqrt{7}$是有理数，矛盾。

<strong>以上是关于平方根和的证明，证明比较粗糙、繁琐，不过原理极其容易理解的。方法范围有限，最多支持到4个平方根和，通用的方法正在寻找中\.\.\.\.\.\.</strong>
