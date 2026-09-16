# 正十七边形的尺规作图存在之证明

> 作者：苏剑林 · 科学空间 · 2009-09-20
>
> 原文：<https://spaces.ac.cn/archives/133>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在网上查找到的，好像有三个不同的版本，全部摘录在此。

关于正17边形的尺规作图方法，请看：
[http://kexue\.fm/article\.asp?id=104](<https://spaces.ac.cn/article.asp?id=104>)

本文章只是证明它的存在（就是求出$\cos ({2\pi}/{17})$）。

$$\cos \frac{2\pi}{17}=\frac{-1+\sqrt{17}+\sqrt{34-2\sqrt{17}}+2\sqrt{17+3\sqrt{17}-\sqrt{34-2\sqrt{17}}-2\sqrt{34+2\sqrt{17}}}}{16}$$

<strong>版本一（来自[数学研发论坛](<http://bbs.emath.ac.cn/thread-944-1-1.html>)）：</strong>

设正17边形中心角为$\theta$,则$17\theta=2\pi$,即$16\theta=2\pi-\theta$
故$\sin 16\theta=-\sin \theta$,而
$$\begin{aligned}\sin 16\theta=2\sin 8\theta \cos 8\theta=2^2\sin 4\theta \cos 4\theta \cos 8\theta \\ =2^4 \sin \theta \cos \theta \cos 2\theta \cos 4\theta \cos 8\theta\end{aligned}$$
因$\sin \theta\neq 0$,两边除之有:
$$16\cos \theta \cos 2\theta \cos 4\theta \cos 8\theta=-1$$
又由$2\cos \theta \cos 2\theta=\cos \theta+\cos 3\theta$等,有
$$2(\cos \theta+\cos 2\theta+...+\cos 8\theta)=-1$$
注意到 $\cos 15\theta=\cos 2\theta,\cos 12\theta=\cos 5\theta$,令
$$\begin{aligned}x=\cos \theta+\cos 2\theta+\cos 4\theta+\cos 8\theta \\ y=\cos 3\theta+\cos 5\theta+\cos 6\theta+\cos 7\theta\end{aligned}$$
有:
$$x+y=-1/2$$
又$xy=(\cos \theta+\cos 2\theta+\cos 4\theta+\cos 8\theta)(\cos 3\theta+\cos 5\theta+\cos 6\theta+\cos 7\theta)$
$$=1/2(\cos 2\theta+\cos 4\theta+\cos 4\theta+\cos 6\theta+...+\cos \theta+\cos 15\theta)$$
经计算知$xy=-1$
又有
$$x=(-1+\sqrt{17})/4,y=(-1-\sqrt{17})/4$$
其次再设:$x_1=\cos \theta+\cos 4\theta,x_2=\cos 2\theta+\cos 8\theta$
$$y_1=\cos 3\theta+\cos 5\theta,y_2=\cos 6\theta+\cos 7\theta$$
故有$x_1+x_2=(-1+\sqrt{17})/4$
$$y_1+y_2=(-1-\sqrt{17})/4$$
解之可有: (大家自己解解吧\~\~\~\~)
最后,由$\cos \theta+\cos 4\theta=x_1,\cos \theta\cos 4\theta=(y_1)/2$
可求$\cos \frac{2\pi}{17}$，它是数的加减乘除平方根的组合, 故正17边形可用尺规作出

<strong>版本二（不知道来自哪个论坛了）：</strong>

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145250.JPG>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145250.JPG>)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145255.JPG>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145255.JPG>)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145300.JPG>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145300.JPG>)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145305.JPG>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090920145305.JPG>)

<strong>版本三（PDF文件，也来源于网络）：</strong>
[正十七边形作图的思路和方法介绍\.zip](<https://spaces.ac.cn/sci/r17/r17.zip>)
