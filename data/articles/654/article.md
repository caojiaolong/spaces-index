# 数学魔术——漂亮的近似

> 作者：苏剑林 · 科学空间 · 2010-05-29
>
> 原文：<https://spaces.ac.cn/archives/654>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

$$e\approx\Big(1+3^{-2^{85}}\Big)^{9^{4^{6\times 7}}}$$

这个e的近似表达很漂亮，它恰好用到了1到9这9个数字。而且漂亮的不仅仅是这一点，大家猜猜看它的有效数字是多少位？10 位？100 位？1000 位？10000 位？

结果是吓人的，他的近似程度远远超出了我们的想象—— 它能精确到小数点后<u><strong>1,315,266,887,768,832,673,579,363</strong></u> 位！

显然，这绝对不是一个巧合，或者说，这也是一个巧合。它的秘密就在于：$e=\lim\limits_{n\to\infty}(1+1/n)^n$，而 $9^{4^{6\times 7}}$恰好就等于$3^{2^{85}}$。于是后果就很严重了，这个指数相当大，Mathematica 直接就报 Overflow 了，所以它能精确到e的小数点后那么多位。

据说，这个神一般的近似表达最早来源于[这里](<http://www2.stetson.edu/~efriedma/mathmagic/0804.html>)。

由于这个魔术实在太漂亮了，为了防止内容丢失，我特意把网页内容原封不动地保存了下来。本网站提供地址：[http://kexue\.fm/sci/Math\-Magic/Math\-Magic\.htm](<https://spaces.ac.cn/sci/Math-Magic/Math-Magic.htm>)

本文转自：[数学研发论坛(bbs\.emath\.ac\.cn)](<http://bbs.emath.ac.cn/viewthread.php?tid=2388&extra=&page=1>)
