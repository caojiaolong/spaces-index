# 《向量》系列——1\.向心力公式证明

> 作者：苏剑林 · 科学空间 · 2010-07-15
>
> 原文：<https://spaces.ac.cn/archives/701>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

向量在几何和物理中都有极其重要的作用，现在就让我们来看如何用向量研究物理中的圆周运动。

首先我们必须了解一些<strong>基础：</strong>

> 1\.在向量中，只要一条“向径”($\vec{r}$)就可以描述出物体的运动，而不需要建立坐标系。这就是向量应用于物理的原因：物理定律不应该依赖于坐标系，而向量恰恰也不依赖于坐标系！
> 2\.牛顿第二定律：$\vec{F}=m\vec{a}$
> 3\.以及一些向量的微积分运算等（可以查阅维基百科或者相关资料）

在下面及以后的文章描述中，为了大家的阅读方便，把向量写成$\vec{r}$的形式，而非把字母加粗。一般情况下，在本站的描述中，有$|\vec{r}|=r,|\dot{\vec{r}}|=v,|\ddot{\vec{r}}|=a$。但是，$\dot{r}=\frac{d|\vec{r}|}{dt} != |\dot{\vec{r}}|$

[![圆周运动示意图](<https://spaces.ac.cn/usr/uploads/2010/07/170518899.png>)](<https://spaces.ac.cn/attachment/702/>)

圆周运动示意图

上图是一个简单的圆周运动示意图，其中我们得到：
$$\begin{aligned}\vec{F}=m\vec{a} \\ \vec{a}=\ddot{\vec{r}} \\ \vec{v}=\dot{\vec{r}}\end{aligned}$$

其中$\vec{F}$是合外力，由于速度方向总是与向径方向垂直，我们就有
$$\vec{r}\cdot \dot{\vec{r}}=0\tag{1}$$并且
$$\frac{d}{dt}(\vec{r}\cdot \dot{\vec{r}})=\dot{\vec{r}}^2+\vec{r}\cdot \ddot{\vec{r}}=0$$
得出
$$\begin{aligned}m\dot{\vec{r}}^2+\vec{r}\cdot m\ddot{\vec{r}}=0 \\ mv^2+\vec{r}\cdot \vec{F}=0\end{aligned}\tag{2}$$
其中$\vec{F}$是<strong>合外力</strong>（请记住，向心力由合外力提供，是合外力的一个分力，不能说物体受到向心力的作用），(1)和(2)结合起来，就是<u>描述圆周运动的基本方程</u>。

如果该运动为匀速圆周运动，v恒定，则应该有$\dot{\vec{r}}^2=const$，即
$$\frac{d}{dt}(\dot{\vec{r}}^2)=2\dot{\vec{r}}\cdot \ddot{\vec{r}}=0$$
即$$\dot{\vec{r}}\cdot \vec{F}=0\tag{3}$$
即合外力方向与速度方向垂直，结合(2)，有
$$\begin{aligned}\vec{F}=-m(\frac{\dot{\vec{r}}^2}{r^2})\vec{r}=-m\omega^2 \vec{r} \\ F=\frac{mv^2}{r}\end{aligned}$$
