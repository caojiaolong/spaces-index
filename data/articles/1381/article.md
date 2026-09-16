# 向量结合复数：常曲率曲线(1)

> 作者：苏剑林 · 科学空间 · 2011-06-19
>
> 原文：<https://spaces.ac.cn/archives/1381>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在之前的[一篇向量系列的文章](<https://spaces.ac.cn/archives/714/>)中，我们通过结合物理与向量来巧妙地推导出了曲线（包括平面和空间的）的曲率半径为
$$R=\frac{v^2}{a_c}=\frac{|\dot{\vec{r}}|^3}{|\dot{\vec{r}}\times \ddot{\vec{r}}|}\tag{1}$$
曲率则是曲率半径的导数：$\rho=\frac{1}{R}$。我们反过来思考一下：曲率恒定的平面曲线是否只有圆？

答案貌似是很显然的，我们需要证明一下。

由于只是考虑平面情况，我们先设$\dot{\vec{r}}=(v cos\theta,v sin\theta)=z=ve^{i\theta}$，代入(1)得到
$\frac{\dot{\theta}}{v}=\rho$————(2)

> <strong>注意，这里我们用到了导数符号$\dot{\theta}$，但依旧还没有说明是对哪个变量求导的。我们发现要是$\dot{\theta}=\frac{d\theta}{dv}$的话，(2)是很容易求解的。于是我们就约定，函数上面的一点表示对变量v求导。这是一种“事后决定法”，因为这里求导的变量是随意的，适当的选择让我们更方便地求解。</strong>

这样，(2)的通解为：$$\theta=1/2 \rho v^2+C_1\tag{2}$$
继续换回x,y变量，我们有：$$\frac{dz}{dv}=ve^{i\theta}=ve^{i(1/2 \rho v^2+C_1)}\tag{3}$$
那么
$$\begin{aligned}z=\int ve^{i\theta}=ve^{i(1/2 \rho v^2+C_1)}dv \\ =\frac{1}{\rho} e^{i(1/2 \rho v^2+C_1)}d(1/2 \rho v^2+C_1)=\frac{1}{\rho} e^{i(1/2 \rho v^2+C_1)}+C_2\end{aligned}\tag{4}$$
选取进行适当的平移使得$C_2=0$。有
$$z=\frac{1}{\rho} e^{i(1/2 \rho v^2+C_1)}\tag{5}$$
不难检验$|z|=\frac{1}{\rho}$。因此，这是一个圆。

<strong>注意这里的“复数”充当了一个“形式上”的作用，它是一个把正交的坐标运算结合为一个整体来运算的工具，而不是一个单纯的数。下一篇文章我们来探讨一下空间曲线和平面的类似问题。</strong>
