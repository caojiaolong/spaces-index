# 关于行星留周期的几何讨论

> 作者：苏剑林 · 科学空间 · 2010-10-02
>
> 原文：<https://spaces.ac.cn/archives/958>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

关于行星留的周期的计算，我们之前[已经讨论过这个问题](<https://spaces.ac.cn/archives/608/>)，利用的是微积分的方法。也许不少还没有高数基础的朋友会感到很头晕，因此在这里给出一个从几何方面讨论的推导。

关于留，很多人认为就是行星相对于地球的速度为0的时刻，其实这个说法稍欠准确，严格来讲应该要将速度改为“角速度”或“切向速度”（天文的切向就是指与视线方向垂直的方向）。实际的运动中，没有哪一瞬间行星相对于地球的运动速度是为0的。根据这句话，我们可以作出下面的图（依旧只考虑正圆运动）：

[![行星留-运动分析](<https://spaces.ac.cn/usr/uploads/2010/10/2064192587.png>)](<https://spaces.ac.cn/attachment/959/>)

行星留\-运动分析

并且列出：
$$V_p \cos\varphi=V_e \cos(\theta+\varphi)=V_e \cos\theta \cos \varphi-V_e \sin\theta \sin\varphi$$
即
$$V_p=V_e \cos \theta-V_e \tan \varphi \sin\theta$$
同时，根据图上可以列出
$tan\varphi=\frac{r_e sin\theta}{r_p-r_e cos\theta}$，代入上式得到
$$\begin{aligned}V_p(r_p-r_e \cos\theta)=V_e r_p \cos\theta-V_e r_e \\ V_p r_p+V_e r_e=(V_p r_e+V_e r_p)\cos\theta\end{aligned}$$
反三角函数得到
$$\theta=arccos(\frac{V_p r_p+V_e r_e}{V_p r_e+V_e r_p})$$
冲日前后不大于$\theta$角的一段时间里，行星都处于逆行状态，因此逆行总时间为
$T=\frac{2\theta}{\omega_e-\omega_p}$
