# 《自然极值》系列——8\.极值分析

> 作者：苏剑林 · 科学空间 · 2010-12-26
>
> 原文：<https://spaces.ac.cn/archives/1134>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![《非线性泛函分析及其应用,第3卷,变分法及最优化》](<https://spaces.ac.cn/usr/uploads/2010/12/1230484808.jpg>)](<https://spaces.ac.cn/attachment/1136/>)

《非线性泛函分析及其应用,第3卷,变分法及最优化》

<strong>本篇文章是《自然极值》系列最后一篇文章，估计也是2010年最后一篇文章了。在这个美好的2010年，想必大家一定收获匪浅，BoJone也在2010年成长了很多。在2010年的尾声，BoJone和科学空间都祝大家在新的一年里更加开心快乐，在科学的道路上更快速地前行。</strong>

在本文，BoJone将与大家讨论求极值的最基本原理。这一探讨思路受到了天才的费恩曼所著《费恩曼物理讲义》的启迪。我们分别对函数求极值（求导）和泛函数极值（变分）进行一些简略的分析。

<strong>一、函数求极值</strong>

对于一个函数$y=f(x)$，设想它在$x=x_0$处取到最大值，那么显然对于很小的增量$\Delta x$，有
$$f(x_0+\Delta x) \leq f(x_0)\tag{3}$$根据泰勒级数，我们有
$f(x_0+\Delta x)=f(x_0)+f'(x_0)\Delta x$————(4)
这里我们略去了二次以及更高次方的项，因为中值定理告诉我们，剩下来的项之和仍然只是一个二次项（二阶无穷小），也就是说，它无法“撼动$f'(x_0)\Delta x$的地位”。于是将(4)代入(3)有
$$f'(x_0)\Delta x \leq 0$$
要注意的是，$f'(x_0)$是一个定值，而$\Delta x$是一个可正可负的变量，于是我们就得到
$$f'(x_0) \leq 0,f'(x_0) \geq 0$$
从而有$f'(x_0)=0$。

我们还可以把上面的名词“最小值”换成“最大值”，把$\leq$和$\geq$互换，同样可以进行类似的讨论，结果是一样的。于是我们得出：$f'(x)=0$是函数$f(x)$的极大（小）值的必要条件。

<strong>二、泛函数求极值</strong>

关于最速降线和悬链线问题的讨论，我们最终都归结为这样的一个问题：

> 求一过$(x_1,y_1),(x_2,y_2)$的函数$y=f(x)$，满足积分$\int_{x_1}^{x_2} F(x,y,\dot{y})dx$为极大（小）值。

设函数$y=y(x)$是所求函数，那么对于y的一个很小的增值函数$\varepsilon=\varepsilon(x)$，其中$\varepsilon(x_1)=\varepsilon(x_2)=0$，那么$y=y(x)+\varepsilon(x)$同样是一个过$(x_1,y_1),(x_2,y_2)$的函数。那么

$$\int_{x_1}^{x_2} F(x,y+\varepsilon,\dot{y}+\dot{\varepsilon})dx \leq \int_{x_1}^{x_2} F(x,y,\dot{y})dx\tag{5}$$
利用多元泰勒级数对$F(x,y+\varepsilon,\dot{y}+\dot{\varepsilon})$进行展开，得
$$F(x,y+\varepsilon,\dot{y}+\dot{\varepsilon})=F(x,y,\dot{y})+\frac{\partial F}{\partial y}\varepsilon+\frac{\partial F}{\partial \dot{y}}\dot{\varepsilon}$$

这里我们同样略去了二次及更高次方的项。代入(5)式得到
$$\int_{x_1}^{x_2}(\frac{\partial F}{\partial y}\varepsilon+\frac{\partial F}{\partial \dot{y}} \dot{\varepsilon})dx \leq 0\tag{6}$$
这里有一个对$\int(\frac{\partial F}{\partial \dot{y}} \dot{\varepsilon})dx$处理的技巧，利用的是《数学分析》中的“分步积分法”，即
$$\int(\frac{\partial F}{\partial \dot{y}} \dot{\varepsilon})dx=\int(\frac{\partial F}{\partial \dot{y}}d\varepsilon)=\frac{\partial F}{\partial \dot{y}}\varepsilon-\int[\frac{d(\frac{\partial F}{\partial \dot{y}})}{dx}\varepsilon] dx$$

代入(6)式得到

$$(\frac{\partial F}{\partial \dot{y}}\varepsilon)|_{x_1}^{x_2}+\int_{x_1}^{x_2}[\frac{\partial F}{\partial y}-\frac{d(\frac{\partial F}{\partial \dot{y}})}{dx}]\varepsilon dx \leq 0$$

由于$\varepsilon(x_1)=\varepsilon(x_2)=0$，所以$(\frac{\partial F}{\partial \dot{y}}\varepsilon)|_{x_1}^{x_2}=0$，同样$\varepsilon$可正可负，因而必定有
$$\int_{x_1}^{x_2}[\frac{\partial F}{\partial y}-\frac{d(\frac{\partial F}{\partial \dot{y}})}{dx}]\varepsilon dx =0$$

这个式子必须对于所有的$\varepsilon=\varepsilon(x)$都成立，因而括号内的值只能为0，于是
$$\frac{\partial F}{\partial y}-\frac{d(\frac{\partial F}{\partial \dot{y}})}{dx}=0\tag{7}$$
把极大值和极小值互换，把$\leq$和$\geq$互换，同样可以进行类似的讨论，结果也是一样的。于是我们得出：(7)式是积分$\int_{x_1}^{x_2} F(x,y,\dot{y})dx$为极值的必要条件。

(7)式就是著名的（二维形式）<strong>欧拉\-拉格朗日方程 (Euler\-Lagrange equation)</strong> 。

利用类似的思路，还可以把方程扩展到更多维度，以及更高的阶（比如F中含有$\ddot{y}$项等）。大家不难发现，里边是思路是一致的：<u>假设极值→设置增量→一阶展开→与原值比较→分析化简→得出等式→解出等式</u>。尽管其中的处理过程有所差别，但是原理并没有变化。因此，可以认为，这是处理极值问题的最根本思路。

由于本文属于思路引导而非专业教程，所以该问题讨论至此已经算是完毕。具体内容大家可以查阅维基百科里边的相关内容。

> 变分：
> [http://zh\.wikipedia\.org/zh/%E5%8F%98%E5%88%86%E6%B3%95](<http://zh.wikipedia.org/zh/%E5%8F%98%E5%88%86%E6%B3%95>)
> 
> 
> 
> 欧拉－拉格朗日方程：
> [http://zh\.wikipedia\.org/zh\-sg/%E6%AD%90%E6%8B%89%EF%BC%8D%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E6%96%B9%E7%A8%8B](<http://zh.wikipedia.org/zh-sg/%E6%AD%90%E6%8B%89%EF%BC%8D%E6%8B%89%E6%A0%BC%E6%9C%97%E6%97%A5%E6%96%B9%E7%A8%8B>)

<strong>《自然极值》告一段落了，2010年也将告一段落了，尽管还有很多的不舍和遗憾，我们还是在2010收获了很多，成长了很多。愿我们都带着最美好的希望，迎接即将到来的2011，在阳光的沐浴和风雨的洗礼中，慢慢成长，渐渐前行，体味科学，领略真理！在科学的道路上，愿继续与众多的科学爱好者共同前行！</strong>

《自然极值》系列终。
