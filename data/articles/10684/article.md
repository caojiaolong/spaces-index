# 三个球的交点坐标（三球交会定位）

> 作者：苏剑林 · 科学空间 · 2025-01-28
>
> 原文：<https://spaces.ac.cn/archives/10684>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前几天笔者在思考一个问题时，联想到了三球交点问题，即给定三个球的球心坐标和半径，求这三个球的交点坐标。按理说这是一个定义清晰且简明的问题，并且具有鲜明的应用背景（比如卫星定位），应该早已有人给出“标准答案”才对。但笔者搜了一圈，发现不管是英文资料还是中文资料，都没有找到标准的求解流程。

当然，这并不是说这个问题有多难以至于没人能求解出来，事实上这是个早已被人解决的经典问题，笔者只是意外于似乎没有人以一种可读性比较好的方式将求解过程写到网上，所以本文试图补充这一点。

## 特殊情形

首先，设三个球的方程分别是
\begin{align}
&\text{球1:}\quad (\boldsymbol{x} - \boldsymbol{o}_1)^2 = r_1^2 \label{eq:s1} \\
&\text{球2:}\quad (\boldsymbol{x} - \boldsymbol{o}_2)^2 = r_2^2 \label{eq:s2} \\
&\text{球3:}\quad (\boldsymbol{x} - \boldsymbol{o}_3)^2 = r_3^2 \label{eq:s3} \\
\end{align}
我们要做的事情，就是联立这三个方程求解$\boldsymbol{x}$。该方程的一般求解是比较困难的，但它有一个比较简单的例子——当$\boldsymbol{o}_1=(0,0,0),\boldsymbol{o}_2=(a,0,0),\boldsymbol{o}_3=(b,c,0)$时，方程是
\begin{align}
&\text{球1:}\quad x^2+y^2+z^2 = r_1^2 \label{eq:s4} \\
&\text{球2:}\quad (x-a)^2+y^2+z^2 = r_2^2 \label{eq:s5} \\
&\text{球3:}\quad (x-b)^2+(y-c)^2+z^2 = r_3^2 \label{eq:s6} \\
\end{align}
将式$\eqref{eq:s1}$、$\eqref{eq:s2}$相减，我们就可以求出$x$，然后将式$\eqref{eq:s1}$、$\eqref{eq:s3}$相减，就可以求出$y$，最后就可以求$z$：
\begin{align}
&x = \frac{r_1^2 - r_2^2 + a^2}{2a} \label{eq:s7} \\[5pt]
&y = \frac{r_1^2 - r_3^2 + b^2 - 2bx + c^2}{2c} \label{eq:s8} \\[5pt]
&z = \pm \sqrt{r_1^2 - x^2 - y^2} \label{eq:s9}
\end{align}
最后的$\pm$表明交点如果存在的话，一般情况下都会有两个。

## 一般情形

上述例子的意义是：$\boldsymbol{o}_1$位于原点，$\boldsymbol{o}_2$位于$x$轴，$\boldsymbol{o}_3$位于$x,y$平面。当它们不满足这个条件时，我们可以为它们重新选择一个坐标系。

首先，全体坐标都减去$\boldsymbol{o}_1$，那么我们就实现了以$\boldsymbol{o}_1$为原点，然后我们记$\boldsymbol{o}_{ij} = \boldsymbol{o}_j - \boldsymbol{o}_i$，并以
\begin{equation}\boldsymbol{u}=\frac{\boldsymbol{o}_{12}}{\Vert\boldsymbol{o}_{12}\Vert}\end{equation}
为$x$轴，那么$\boldsymbol{o}_2$就落在$x$轴上，并且有$a=\Vert\boldsymbol{o}_{12}\Vert=\boldsymbol{o}_{12}\cdot\boldsymbol{u}$。接着再以
\begin{equation}\boldsymbol{v}=\frac{\boldsymbol{o}_{13} - (\boldsymbol{o}_{13}\cdot \boldsymbol{u})\boldsymbol{u}}{\Vert\boldsymbol{o}_{13} - (\boldsymbol{o}_{13}\cdot \boldsymbol{u})\boldsymbol{u}\Vert}\end{equation}
为$y$轴，那么$\boldsymbol{o}_3$就落在$x,y$平面上，且$b = \boldsymbol{o}_{13}\cdot \boldsymbol{u},c = \boldsymbol{o}_{13}\cdot \boldsymbol{v}$。最后补上$\boldsymbol{w}=\boldsymbol{u}\times \boldsymbol{v}$，那么$\boldsymbol{u},\boldsymbol{v},\boldsymbol{w}$就构成了一组新的标准正交基。

现在有了$a,b,c$，我们就可以代入式$\eqref{eq:s7},\eqref{eq:s8},\eqref{eq:s9}$计算交点坐标$(x,y,z)$了。但要注意这个交点坐标是在新坐标系$(\boldsymbol{u},\boldsymbol{v},\boldsymbol{w})$下得到的，恢复到原坐标下的交点是
\begin{equation}\boldsymbol{x}=x \boldsymbol{u} + y\boldsymbol{v} + z \boldsymbol{w} + \boldsymbol{o}_1\end{equation}

## 总的流程

以上求解过程，参考自维基百科的“[True\-range multilateration](<https://en.wikipedia.org/wiki/True-range_multilateration#Three_Cartesian_dimensions,_three_measured_slant_ranges>)”条目，但上面的过程并不算完整，所以本文将它补充完整了。

将所有求解步骤汇集在一起是
\begin{equation}\left\{\begin{aligned}
&\,\boldsymbol{o}_{ij} = \boldsymbol{o}_j - \boldsymbol{o}_i \\[5pt]
&\,\boldsymbol{u}=\frac{\boldsymbol{o}_{12}}{\Vert\boldsymbol{o}_{12}\Vert},\,\,\boldsymbol{v}=\frac{\boldsymbol{o}_{13} - (\boldsymbol{o}_{13}\cdot \boldsymbol{u})\boldsymbol{u}}{\Vert\boldsymbol{o}_{13} - (\boldsymbol{o}_{13}\cdot \boldsymbol{u})\boldsymbol{u}\Vert},\,\,\boldsymbol{w}=\boldsymbol{u}\times \boldsymbol{v}\\[5pt]
&\,a=\boldsymbol{o}_{12}\cdot\boldsymbol{u},\,\,b = \boldsymbol{o}_{13}\cdot \boldsymbol{u},\,\,c = \boldsymbol{o}_{13}\cdot \boldsymbol{v} \\[5pt]
&\,x = \frac{r_1^2 - r_2^2 + a^2}{2a} \\[5pt]
&\,y = \frac{r_1^2 - r_3^2 + b^2 - 2bx + c^2}{2c} \\[5pt]
&\,z = \pm \sqrt{r_1^2 - x^2 - y^2} \\[5pt]
&\,\boldsymbol{x} = x \boldsymbol{u} + y\boldsymbol{v} + z \boldsymbol{w} + \boldsymbol{o}_1
\end{aligned}\right.\end{equation}

参考实现：



<pre><code class="language-python">import numpy as np

def trilaterate(o1, o2, o3, r1, r2, r3):
    o12, o13 = o2 - o1, o3 - o1
    u = o12 / (o12**2).sum()**0.5
    v = (v := o13 -  o13.dot(u) * u) / (v**2).sum()**0.5
    w = np.cross(u, v)
    a, b, c = o12.dot(u), o13.dot(u), o13.dot(v)
    x = (r1**2 - r2**2 + a**2) / (2 * a)
    y = (r1**2 - r3**2 + b**2 - 2 * b * x + c**2) / (2 * c)
    z = (r1**2 - x**2 - y**2)**0.5
    p = x * u + y * v + o1
    return p + z * w, p - z * w

o1 = np.array([1, 2, -3])
o2 = np.array([2, 1, -1])
o3 = np.array([-3, 0, 2])
r1, r2, r3 = 4, 5, 6
trilaterate(o1, o2, o3, r1, r2, r3)</code></pre>



## 文章小结

本文整理了三球交点问题的一个相对简明的求解方案。
