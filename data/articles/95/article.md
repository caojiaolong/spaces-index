# 关于a,b的极限证明题目

> 作者：苏剑林 · 科学空间 · 2009-08-24
>
> 原文：<https://spaces.ac.cn/archives/95>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>证明下列极限：</strong>
$$\lim_{x \to 0}\left(\frac{a^x+b^x}{2}\right)^{3/x}=ab\sqrt{ab}$$

<strong>解：</strong>
这是我认为比较难的极限题目之一，由麦克劳林公式可以推出：
$$a^x=1+x \ln a+\frac{x^2 \ln^2 a}{2!}+\frac{x^3 \ln^3 a}{3!}+...$$

于是原式可以变成
$$\lim_{x \to 0}\left(\frac{2+x \ln a+\frac{x^2 \ln^2 a}{2!}+...+x \ln b+\frac{x^2 \ln^2 b}{2!}+...}{2}\right)^{3/x}$$

我们有一个简单的极限：$\lim\limits_{x\to 0}(a+x^2)^{1/x}=a^{1/x}$，因此，在上式中，$\frac{x^2 \ln^2 a}{2!}$及其后面的项可以忽略，只考虑
$$\begin{aligned}
&\,\lim_{x \to 0}\left(\frac{2+x \ln a+x \ln b}{2}\right)^{3/x}\\
=&\,\lim_{x \to 0} \left\{\left[1+\left(\frac{\ln a+\ln b}{2}\right)x\right]^{1/x}\right\}^3\\
=&\,e^{\frac{3(\ln a+ \ln b)}{2}}\\
=&\,(ab)^{3/2}\\
=&\,ab\sqrt{ab}
\end{aligned}$$

同理，有
$$\lim_{x \to 0}\left(\frac{a_1^x+a_2^x+...+a_n^x}{n}\right)^{1/x}=\sqrt[n]{a_1 a_2...a_n}$$
