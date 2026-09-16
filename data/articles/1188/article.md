# &#91;更新&#93;将向量乘法“退化”到复数

> 作者：苏剑林 · 科学空间 · 2011-02-04
>
> 原文：<https://spaces.ac.cn/archives/1188>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

向量有两个乘法：点乘和叉乘，其结果又分别叫做数量积和向量积。在很多情况下，用这两个定义的乘法运算都能够给我们带来很大的方便（其实它就是在实际问题中抽象出来的）。不过，也有相当一部分的二维问题用复数来描述更为简洁。于是，为了整合两者的巧妙之处，有必要把向量的两个乘法运算“退化”到复数中去（为什么用“退化”？因为向量是多维的，可以是3维、4维等，而复数运算只是二维的，很明显这是一种“退化”而不是“拓展”^\_^）

<strong>运算法则：</strong>

<strong>点乘：</strong>
总法则：$Z_1 \cdot Z_2=|Z_1||Z_2|\cos(arg\frac{Z_2}{Z_1})$
$$\begin{aligned}1\cdot i=0 \\ i\cdot i=1 \\ \exp(i\theta)\cdot \exp(i\varphi)=\cos(\varphi -\theta) \\ iexp(i\theta)\cdot \exp(i\varphi)=-\sin(\theta-\varphi ) \\ Z_1 \cdot Z_2=Z_1 \bar{Z}_2+Z_2 \bar{Z}_1\end{aligned}$$

<strong>叉乘：</strong>
由于二维向量的叉积都指向第三维，所以可以认为复数的叉积结果都是一个数。
总法则：$Z_1 \times Z_2=|Z_1| |Z_2| sin(arg\frac{Z_2}{Z_1})$
$$\begin{aligned}1\times i=1 \\ i\times i=0 \\ \exp(i\theta) \times \exp(i\varphi)=\sin(\varphi-\theta ) \\ iexp(i\theta) \times \exp(i\varphi)=-\cos(\theta-\varphi ) \\ Z_1 \times Z_2=(Z_1 \bar{Z}_2-Z_2 \bar{Z}_1)i\end{aligned}$$

<strong>变换关系：</strong>
$$\begin{aligned}Z_1 \times Z_2=-Z_2 \times Z_1 \\ Z_1 \times (i Z_2)=Z_1\cdot Z_2 \\ Z_1 \cdot (i Z_2)=-Z_1 \times Z_2\end{aligned}$$

<strong>微分恒等关系：</strong>
$$Z\cdot dZ=|Z| d|Z|$$
$Z*(i dZ)=-(iZ)*dZ=+-|Z|\sqrt{dZ*dZ-(d|Z|)^2}$（正负待定）

于是，复数便有了三种乘法了。它们代表的意义都不一样！复数原来的乘法是一种旋转和伸长，而点积和叉积分别是有关两个复数的夹角的余弦和正弦。

这是用复数研究三体问题周期轨道时所悟到的思想，特此记录！
