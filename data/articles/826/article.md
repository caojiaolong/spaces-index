# 旋转的弹簧将如何伸长(2)？

> 作者：苏剑林 · 科学空间 · 2010-08-07
>
> 原文：<https://spaces.ac.cn/archives/826>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![弹簧](<https://spaces.ac.cn/usr/uploads/2010/08/580421765.jpg>)](<https://spaces.ac.cn/attachment/828/>)

弹簧

[<strong>上一次</strong>](<https://spaces.ac.cn/archives/782/>)我从密度的角度讨论了旋转的弹簧伸长的问题，由于对弹性形变等问题是初涉，所以花了好大功夫。这几天重新认识了一下胡克定律，并且从另外的角度给出了这道题目的一个相对简单的解法。在此把它记录下来，并写写我对弹性形变的一些粗浅看法。

在解答的过程中，我再次体验到了<strong>殊途同归</strong>的感觉，科学就是这样的奇妙，一个目的地往往有着不止一条道路，不同的道路会给我们不同的科学视觉，最终领略到不同的科学美景；多走几条路，更能够让我们从不同的角度领略美不胜收的科学，这也是众多旅游爱好者不辞千里地观赏美景的原因！

<strong>——————————华丽的分割线———————————</strong>

我们之前把胡克定律写成$F=k\cdot \Delta l$，$\Delta l$是弹簧的伸长，这种写法简单而不完美，因为虽说k是常数，但是影响因素太多了，如弹簧长度、厚度等都会影响劲度系数k；而在大学的力学教程中，是这样写胡克定律的：
$$\frac{F_n}{S}=E\frac{\Delta l}{l_0}$$

$F_n$是垂直作用在弹簧横截面的力；S是横截面积；E称为弹性模量(也叫杨氏模量)，这是与材料有关的量，对于拉伸和压缩，E一般不同，但是相差也不大；$\frac{\Delta l}{l_0}$称为(正)应变，由于弹簧的整体应变未必相同，所以严格的应变应该写成$\frac{d(\Delta l)}{dl}$，因此，胡克定律应该写成
$$\frac{F_n}{S}=E\frac{d(\Delta l)}{dl}\tag{1}$$
如果受力是均匀的，那么有：$\frac{F_n}{S*E}dl=d(\Delta l)$，积分便成
$$\frac{F_n}{S\cdot E}l_0=\Delta l\Rightarrow F_n=\frac{E\cdot S}{l_0}\Delta l$$
<u>对比$F=k\Delta l$，可以得出：$k=\frac{E*S}{l_0}$，这个等式很清晰地说明了k与那些量有关（E、S、l分别代表了弹簧的<strong>材料、厚度、长度</strong>）。</u>

<strong>——————————华丽的分割线———————————</strong>

上面的一堆只是BoJone对胡克定律的粗浅看法，下面进入正题，给出另外角度的解决旋转弹簧的伸长问题的方法。

选取弹簧旋转重心为参考点，设原来位于弹簧上坐标为(x,x\+dx)、质量为dm的质量元，旋转后坐标为(y,y\+dy)，那么伸长量为$dy-dx$，应变为$\frac{dy-dx}{dx}=\dot{y}-1$，根据(1)，有
$$F_n=E\cdot S\cdot (\dot{y}-1)=kl_0 (\dot{y}-1)\tag{2}$$
$F_n$即作用在截面上的惯性离心力，可以得到$F_n=F_c=\int_y^l y\omega^2 dm$，l是旋转后的弹簧的长度，由此并结合(2)式可以给出：$y=l,dot{y}=1$。现在我们把(2)式两端微分，得到
$$\begin{aligned}d(F_n)=kl_0 \ddot{y}dx \\ d(F_n)=-y\omega^2 dm=-\lambda_0\omega^2 ydx \\ \ddot{y}=-\frac{\lambda_0\omega^2}{kl_0}y\end{aligned}\tag{3}$$
这就是一个二阶线性齐次微分方程，为了得出形式相对简单的解，我们不采取求通解的方法，而是与之前处理过的问题一样，令$\ddot{y}=\dot{y}\frac{d\dot{y}}{dy}$，代入(3)，分离变量并积分，可以得到
$$\dot{y}^2=-\frac{\lambda_0\omega^2}{kl_0}y^2+C_1$$
根据初始条件$y=l,dot{y}=1$，可以得到$C_1=\frac{\lambda_0\omega^2}{kl_0}l^2+1$。代入上式，继续分离变量，得到
$$dx=\frac{dy}{\sqrt{\frac{\lambda_0\omega^2}{kl_0}l^2+1-\frac{\lambda_0\omega^2}{kl_0}y^2}}$$
积分可以得到
$$x=\frac{1}{\omega}\sqrt{\frac{kl_0}{\lambda_0}}arcsin(\frac{\omega\sqrt{\frac{\lambda_0}{kl_0}}y}{\sqrt{\frac{\lambda_0 \omega^2}{kl_0}l^2+1}})\tag{4}$$（积分常数为0，因为初始条件是y=0,x=0）

(4)便是我们所希望的结果，我们希望求出l的值，那么可以把$x=l_0,y=l$代入(4)，并且利用变换关系$arcsin(a/b)=arctg\frac{a}{\sqrt{b^2-a^2}}$变成
$$l_0=\frac{1}{\omega}\sqrt{\frac{kl_0}{\lambda_0}}arctg(\omega\sqrt{\frac{\lambda_0}{kl_0}}l)$$

即$l=l_0*\frac{1}{\omega}\sqrt{k/{\lambda_0 l_0}}tg(\omega\sqrt{{\lambda_0 l_0}/k})=l_0*\frac{1}{\omega}\sqrt{k/M}tg(\omega\sqrt{M/k})$
