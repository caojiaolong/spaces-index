# cos 1°的根式表达式

> 作者：苏剑林 · 科学空间 · 2011-06-26
>
> 原文：<https://spaces.ac.cn/archives/1385>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

BoJone记得自己第一次接触三角函数大概是小学五、六年级的时候，那时候我拿来了表姐的初中数学书来看。看到三角函数一章后，饶有兴致，希望能够找到一个根据角度来求三角函数值的方法，可是书本上只是教我去用计算器算和查表，这让我这个爱好计算的孩子大失所望。这个问题直到高一才得以解决，原来这已经涉及到了微积分中的泰勒级数了\.\.\.

我记得为了求任意角度的三角函数值，我曾经根据30°、45°和60°的正弦值拟合过一条近似公式出来：
$$\sin A \approx \sqrt{\frac{A}{60}-1/4}$$

其中A以角度为单位，大致适用于25°\~60°，精度好像有两位小数。当然，这个结果在今天看来是很粗糙的，但是这毕竟是我的“小学的作品”！在此留念一翻。

后来上了初中，了解到了一些高中的知识后，我就尝试求更多的角度的三角函数表达式。除了30°、45°和60°外，我们还有18°的三角函数表达式：

$$\begin{aligned}\sin 18^{\circ}=\frac{\sqrt{5}-1}{4} \\ \cos 18^{\circ}=\frac{\sqrt{2\sqrt{5}+10}}{4}\end{aligned}$$

同时，根据$cos A$和$cos B$求$cos (A-B)$的方法：
$$\cos(A-B)=\cos A \cos B+\sqrt{(1-\cos^2 A)(1-\cos^2 B)}$$

根据$cos A$求$cos \frac{A}{2}$的方法：
$$\cos\frac{A}{2}=\sqrt{\frac{\cos A+1}{2}}$$

根据$cos A$求$cos \frac{A}{3}$的方法：
$$\begin{aligned}\cos A=4(\cos\frac{A}{3})^3-3\cos\frac{A}{3} \\ \cos \frac{A}{3} = \frac{1}{2} ( -\sqrt[3]{\sqrt{\cos^2 A-1}-\cos A} +\sqrt[3]{\sqrt{\cos^2 A-1}+\cos A})\end{aligned}$$

于是我们可以写出：$cos 15^{\circ}=\frac{\sqrt{\sqrt{3}+2}}{2}$

$$\begin{aligned}\cos 3^{\circ}=\cos(18^{\circ}-15^{\circ})= \\ \frac{\sqrt{2\sqrt{15}+4\sqrt{5}+10\sqrt{3}+20}+\sqrt{2\sqrt{15}-4\sqrt{5}-6\sqrt{3}+12}}{8}\end{aligned}$$

于是就不难写出cos 1°了。当然现在看来，这个东西只能作为一种存在性证明，没有用多大的实用价值。
