# 【外微分浅谈】2\. 反对称的威力

> 作者：苏剑林 · 科学空间 · 2016-11-04
>
> 原文：<https://spaces.ac.cn/archives/4054>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### 内积与外积

<strong>向量（这里暂时指的是二维或者三维空间中的向量）的强大之处，在于它定义了内积和外积（更多时候称为叉积、向量积等），它们都是两个向量之间的运算，其中，内积被定义为是对称的，而外积则被定义为反对称的，它们都满足分配律。</strong>

沿着书本的传统，我们用$\langle,\rangle$表示内积，用$\land$表示外积，对于外积，更多的时候是用$\times$，但为了不至于出现太多的符号，我们统一使用$\land$。我们将向量用基的形式写出来，比如
$$\boldsymbol{A}=\boldsymbol{e}_{\mu}A^{\mu} \tag{1} $$
其中$\boldsymbol{e}_{\mu}$代表着一组基，而$A^{\mu}$则是向量的分量。我们来计算两个向量$\boldsymbol{A},\boldsymbol{B}$的内积和外积，即
$$\begin{aligned}&\langle \boldsymbol{A}, \boldsymbol{B}\rangle=\langle \boldsymbol{e}_{\mu}A^{\mu}, \boldsymbol{e}_{\nu}B^{\nu}\rangle=\langle\boldsymbol{e}_{\mu},\boldsymbol{e}_{\nu}\rangle A^{\mu}A^{\nu}\\
&\boldsymbol{A}\land \boldsymbol{B}=(\boldsymbol{e}_{\mu}A^{\mu})\land (\boldsymbol{e}_{\nu}B^{\nu})=\boldsymbol{e}_{\mu}\land\boldsymbol{e}_{\nu} A^{\mu}B^{\nu}
\end{aligned} \tag{2} $$

然后呢？没有然后了，因为我们还没给$\langle\boldsymbol{e}_{\mu},\boldsymbol{e}_{\nu}\rangle$和$\boldsymbol{e}_{\mu}\land\boldsymbol{e}_{\nu}$下定义。在解析几何中，我们是这样定义内积的，设$\boldsymbol{e}_{\mu}$是一组标准正交基，那么
$$\langle\boldsymbol{e}_{\mu},\boldsymbol{e}_{\nu}\rangle=\delta_{\mu\nu} \tag{3} $$
当$\mu=\nu$时，$\delta_{\mu\nu}=1$，否则为0。这样，我们就可以对任意两个向量计算它们的内积了，并且有了这个定义，内积成为了判断垂直的工具（两个向量内积为0），也成为了计算模长的工具（向量与它自身做内积）。

再来看外积，在二维空间中，外积是这样定义的，设$\boldsymbol{e}_{\mu}$是一组标准正交基，那么
$$\boldsymbol{e}_1\land\boldsymbol{e}_2=1 \tag{4} $$
注意由反对称性，我们可以得到$\boldsymbol{e}_1\land\boldsymbol{e}_1=\boldsymbol{e}_2\land\boldsymbol{e}_2=0, \boldsymbol{e}_2\land\boldsymbol{e}_1=-1$，因此这样定义已经完整了。此时可以算得
$$\boldsymbol{A}\land \boldsymbol{B}=A^1 B^2 - A^2 B^1 \tag{5} $$
这时候的外积是一个数，其绝对值正是$\boldsymbol{A}$和$\boldsymbol{B}$所张成的平行四边形的面积。

而在三维空间中，则定义
$$\boldsymbol{e}_1\land\boldsymbol{e}_2=\boldsymbol{e}_3,\boldsymbol{e}_2\land\boldsymbol{e}_3=\boldsymbol{e}_1,\boldsymbol{e}_3\land\boldsymbol{e}_1=\boldsymbol{e}_2 \tag{6} $$
这样定义后，三维空间中的内积就是一个向量了，它与原来的两个向量垂直，并且模长等于原来两向量张成的平行四边形的面积。

回顾整个过程，我们可以这样理解，内积和外积本来就是纯粹代数定义的对称和反对称运算，至于几何意义，则是取决于基的内积和外积确定下来后，进一步赋予的。当然，内外积的定义有一定的历史渊源，但由于它本身并不算困难，因此我们忽略对它历史的研究了。可以看到，对于内积，其定义明显地可以推广到高维空间，而外积则不然。不管怎么样，我们可以清楚这一思路：<strong>纯粹代数定义（主要是定义基的内外积）——寻求几何意义——反观历史渊源</strong>。

### 反对称的威力

从我们学算数开始，我们接触的运算基本都是对称的，即满足$ab=ba$的运算，数的加法、乘法都是这样，到了高中，学了向量的内积，还是这样。因为高中其实讲向量外积的学校并不多，所以很多同学直到大学才接触到不可交换（即$ab\neq ba$）的运算，比如矩阵乘法。在所有非交换运算中，反对称运算是比较特殊而且内涵相当丰富的一种。不仅如此，它还带来了运算的方便。比如下面的例子。

考虑质点在固定的引力中心中运动的问题，那么我们有运动方程
$$\ddot{\boldsymbol{x}}=-\frac{\mu\boldsymbol{x}}{|\boldsymbol{x}|^3} \tag{7} $$
两边以$\boldsymbol{x}$外积，即
$$\boldsymbol{x}\land \ddot{\boldsymbol{x}}=-\boldsymbol{x}\land \frac{\mu\boldsymbol{x}}{|\boldsymbol{x}|^3}=0 \tag{8} $$
留意到
$$\frac{d}{dt}(\boldsymbol{x}\land \dot{\boldsymbol{x}}) = \dot{\boldsymbol{x}}\land \dot{\boldsymbol{x}}+\boldsymbol{x}\land \ddot{\boldsymbol{x}}=\boldsymbol{x}\land \ddot{\boldsymbol{x}} \tag{9} $$
所以上式意味着
$$\frac{d}{dt}(\boldsymbol{x}\land \dot{\boldsymbol{x}})=0 \tag{10} $$
那么
$$\boldsymbol{x}\land \dot{\boldsymbol{x}}=\boldsymbol{C} \tag{11} $$
这事实上就是角动量守恒，由于是矢量方程，因此如果用分量形式写出来就是三个方程。这样我们通过简单几个步骤，就得到了三个积分常数。反思根源，就是外积的反对称性$a\land b = -b\land a$决定了$a\land a=0$。这是任何反对称量的性质，也是反对称的威力，它自然地消去了很多本该为0的部分。
