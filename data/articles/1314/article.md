# 备忘：椭圆坐标与复三角函数

> 作者：苏剑林 · 科学空间 · 2011-04-10
>
> 原文：<https://spaces.ac.cn/archives/1314>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

椭圆坐标系是一种二维正交坐标系。与直角坐标的转换关系为
$$\begin{aligned}x = a \cos h \mu \cos \nu \\ y = a \sin h \mu \sin \nu\end{aligned}$$

其中$(-a,0)$和$(a,0)$是两个焦点。

参看：[http://zh\.wikipedia\.org/wiki/椭圆坐标系](<http://zh.wikipedia.org/wiki/%E6%A9%A2%E5%9C%93%E5%9D%90%E6%A8%99%E7%B3%BB>)

[![Elliptical_coordinates_grid](<https://spaces.ac.cn/usr/uploads/2011/04/2023863032.svg>)](<https://spaces.ac.cn/usr/uploads/2011/04/2023863032.svg>)

Elliptical\_coordinates\_grid

另外，考虑$sin z$，其中$z=x +iy$，有
$$\sin z= i \cos x \sin h y + \sin x \cos h y$$

<strong>证明：</strong>

> $2i sinz = e^{iz}-e^{-iz}= e^{ix-y} - e^{-ix+y}$
> $$\begin{aligned}= e^(ix-y) - e^(ix+y) + e^(ix+y) - e^(-ix+y) \\ = e^{ix} (e^{-y} - e^y) + e^y (e^{ix} - e^{-ix}) \\ = -2 e^{ix} \sin h y + 2i e^y \sin x \\ = -2 (\cos x + i \sin x) \sin h y + 2i (e^y) \sin x\end{aligned}$$
> 
> 
> 
> 然后再乘上$-\frac{1}{2}i$：
> $$\begin{aligned}\sin z = i (\cos x + i \sin x) \sin h y + (e^y) \sin x \\ = i \cos x \sin h y - \sin x \sin h y + e^y \sin x \\ = i \cos x \sin h y - \sin x( \sin h y - e^y) \\ = i \cos x \sin h y - (\frac{e^y - e^{-y} - 2e^y}{2}) \sin x \\ = i \cos x \sin h y + (\frac{e^y + e^{-y}}{2})\sin x \\ = i \cos x \sin h y + \sin x \cos h y\end{aligned}$$
> 
> 
> 
> 证明来源：[http://au\.answers\.yahoo\.com/question/index?qid=20100418003726AAyRTR8](<http://au.answers.yahoo.com/question/index?qid=20100418003726AAyRTR8>)

类似的还有
$$\begin{aligned}\cos z= -i \sin x \sin h y + \cos x \cos h y \\ \sin h z=i \sin y \cos h x + \cos y \sin h x\end{aligned}$$
$cosh z=i sin y sinh x + cos y cosh x$（这完全和椭圆坐标系对应起来了）

惊叹它们与椭圆坐标的相似性！

还有一点有趣的东西
$$\begin{aligned}\sin iz=i \sin h z \\ \sin h iz=i \sin z \\ \cos iz=i \cos h z \\ \cos h iz= i \cos z \\ |\sin z|=\sqrt{\sin^2 x+\sin h^2 y} \\ |\cos z|=\sqrt{\cos^2 x+\sin h^2 y} \\ |\sin h z|=\sqrt{\sin h^2 x+\sin^2 y} \\ |\cos h z|=\sqrt{\cos h^2 x-\sin^2 y}\end{aligned}$$

参考书籍：<strong>Applied complex variables for scientists and engineers</strong>
