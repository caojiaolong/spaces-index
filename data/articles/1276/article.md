# 线圈感抗和电容容抗的计算

> 作者：苏剑林 · 科学空间 · 2011-02-26
>
> 原文：<https://spaces.ac.cn/archives/1276>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![形形色色的电容](<https://spaces.ac.cn/usr/uploads/2011/02/749071170.jpg>)](<https://spaces.ac.cn/attachment/1277/>)

形形色色的电容

学到人教版高二物理选修3\-2的同学们，眼前会出现许多新的名词，如楞次定律、自感（电感）、感抗、容抗等等。其中对于电感，在[中文维基百科](<http://zh.wikipedia.org/zh-cn/%E7%94%B5%E6%84%9F>)给予的解释为：当电流改变时，因电磁感应而产生抵抗电流改变的<strong>电动势</strong>（EMF，electromotive force）。电路中的任何电流，会产生磁场，磁场的磁通量又作用于电路上。依据楞次定律，此磁通会借由感应出的电压（<strong>反电动势</strong>）而倾向于抵抗电流的改变。磁通改变量对电流改变量的比值称为<strong>自感</strong>，自感通常也就直接称作是这个电路的<strong>电感</strong>。

自感的计算公式为：$U=-L\frac{dI}{dt}$，U是自感电动势，I是电流，负号表示自感电动势反抗原来的电流。L是比例系数，就称为<strong>电感</strong>，对于同一个线圈来说，L是常数，单位是$V\cdot t//A=\Omega \cdot t$，同时也简记为$H$（亨利）。

<strong>感抗</strong>

简单介绍后，进入正题了。我们知道，自感线圈可以“<u>通直流，阻交流</u>”，而电容器相反，可以“<u>通交流，阻直流</u>”。其原理就不多说了，本文主要是证明其感抗和容抗的计算公式。先说感抗，由于交流电的电动势和电流在不断变化，而且周期很小，所以通常用一些“平均值”来进行相关计算，以代替不断变化的量，如有效电压（电流）就是在一个周期内的电压（电流）平方的平均值的平方根。感抗也由此而生，虽然自感线圈是“阻交流”的，但是在一个周期内不同时刻的“阻碍情况”一般不同，因此我们需要计算一个“平均的阻碍情况”，也就是说，把这个自感线圈接入交流电路，相当于在电路中接入了一个多大的电阻。

求一个函数的平均值一般有两种方法，<u>一种是对自变量求平均，再代入函数式求值；一种是直接对函数值求平均。</u>感抗和容抗使用的是前者的方法<strong>（在理解本文后，不妨思考一下为什么只能够选择前者而不选择后者）</strong>，换言之，是$\bar{R}=\frac{\bar{U}}{\bar{I}}$而不是$\bar{R}=\bar{(\frac{U}{I})}$。

而且，设交流电周期为T，有
$$\begin{aligned}\bar{U}=\int_0^T \frac{|U|dt}{T}=L\int_0^T \frac{|dI|}{T} \\ \bar{I}=\int_0^T \frac{|I| dt}{T}\end{aligned}$$

接着再设交流电为正弦式交流电，$I=I_m sin(2\pi f t)$，频率为$f=\frac{1}{T}$，代入上式计算，由于正弦函数的对称性，我们只需要考虑四分之一周期即可，这样就可以把绝对值符号干掉。即
$$\begin{aligned}\bar{U}=L\int_0^{T//4} \frac{d(I_m \sin(2\pi f t))}{T//4}=4I_m L f \\ \bar{I}=\int_0^{T//4} \frac{I_m \sin(2\pi f t) dt}{T//4}=\frac{4I_m}{2\pi}\end{aligned}$$

于是感抗$X_L=\bar{R}=4I_m L f\div \frac{4I_m}{2\pi}=2\pi f L$

<strong>容抗</strong>

容抗和感抗的计算有相当多的相似之处。我们知道电容的定义是$C=\frac{Q}{U}$。C是常数，于是我们可以写出
$$C=\frac{dQ}{dU}=\frac{dQ}{dt}\frac{dt}{dU}$$

其中$\frac{dQ}{dt}$即是电流I，于是$I=C\frac{dU}{dt}$。这和自感的计算公式几乎是同出一辙！类似感抗的计算，很快可以写出
$$\begin{aligned}\bar{I}=\int_0^T \frac{|I|dt}{T}=L\int_0^T \frac{|dU|}{T} \\ \bar{U}=\int_0^T \frac{|U| dt}{T}\end{aligned}$$

同样对正弦交流电考虑四分之一周期，则有（这里直接把结果贴出）
$$\begin{aligned}\bar{I}=4CU_m f \\ \bar{U}=\frac{4U_m}{2\pi}\end{aligned}$$
容抗$X_C=\frac{\bar{U}}{\bar{I}}=\frac{1}{2\pi f C}$
