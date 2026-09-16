# 弹簧双体运动

> 作者：苏剑林 · 科学空间 · 2011-07-10
>
> 原文：<https://spaces.ac.cn/archives/1397>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这也是我们期末考的题目，是理综的物理题之一。

> 一个零质量的理想弹簧两端分别系着一个质量为m的质点物体（A左B右），现给A一个向右的速度v<sub>0</sub>，使得整体开始运动。问弹簧压缩到最短时弹性势能是多少？以及B质点的最大速度是多少？

高中生是通过结合动量守恒和能量守恒来求解的。而我希望通过微分方程把握这个运动的整体信息，顺便验证弹簧能否将A的速度v<sub>0</sub>完全传递给B。

首先选择向右为正方向。由于A、B都在同一直线上运动，因此这是一个一维的运动问题，令A的坐标为x，B的坐标为y，弹簧原长为l<sub>0</sub>，劲度系数为k。显然，弹簧的伸长量为$\Delta l=y-x-l$。根据胡克定律，A所受到的弹力为$F=k(y-x-l_0)$。因此可以列出微分方程组

$$m\ddot{x}=F\tag{1}$$$$m\ddot{y}=-F\tag{2}$$$$F=k(y-x-l_0)\tag{3}$$
其中初始条件是：
当t=0时，x=0，y=l<sub>0</sub>，v<sub>A</sub>=v<sub>0</sub>，v<sub>B</sub>=0

(1)\+(2)得到：$\ddot{x}+\ddot{y}=0$
即
$$\begin{aligned}\dot{x}+\dot{y}=v_0 \\ x+y=v_0 t+l_0\end{aligned}\tag{4}$$（这里已经根据初始条件确定了积分常数）

(2)\-(1)得到：
$$m(y-x)''=-2F$$
或者写成：
$$(y-x-l_0)''=-\frac{2k}{m}(y-x-l_0)\tag{5}$$令$y-x-l_0=z$，则(5)变成了
$$z''=-\frac{2k}{m}z$$
这是一道二阶常系数线性微分方程。特征根为$+-i\sqrt{\frac{2k}{m}}$，于是通解可以写成：
$$z=C_1 \sin(\sqrt{\frac{2k}{m}} t)+C_2 \cos(\sqrt{\frac{2k}{m}} t)=y-x-l_0\tag{6}$$
当t=0时，显然有z=0，于是$C_2=0$。对z求导得到
$$\dot{z}=C_1 \sqrt{\frac{2k}{m}} \cos(\sqrt{\frac{2k}{m}} t)=\dot{y}-\dot{x}$$
当t=0时，显然有$\dot{z}=-v_0$，于是$C_1=-v_0\sqrt{\frac{m}{2k}}$

即$$y-x=l_0-v_0\sqrt{\frac{m}{2k}} \sin(\sqrt{\frac{2k}{m}} t)\tag{7}$$
结合(4)、(7)就可以求出
$$\begin{aligned}x=\frac{v_0 t}{2}+ \frac{v_0}{2}\sqrt{\frac{m}{2k}} \sin(\sqrt{\frac{2k}{m}} t) \\ y=l_0+\frac{v_0 t}{2}- \frac{v_0}{2}\sqrt{\frac{m}{2k}} \sin(\sqrt{\frac{2k}{m}} t)\end{aligned}$$

接下来的一切都会有答案了^\_^。原来B真的可以得到速度v<sub>0</sub>的。

<strong>遇到问题，用自己所学的知识多加分析，才能够广拓渠道，开阔思维^\_^\!</strong>
