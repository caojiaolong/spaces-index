# 【理解黎曼几何】4\. 联络和协变导数

> 作者：苏剑林 · 科学空间 · 2016-10-16
>
> 原文：<https://spaces.ac.cn/archives/3998>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### 向量与联络

当我们在我们的位置建立起自己的坐标系后，我们就可以做很多测量，测量的结果可能是一个标量，比如温度、质量，这些量不管你用什么坐标系，它都是一样的。当然，有时候我们会测量向量，比如速度、加速度、力等，这些量都是客观实体，但因为测量结果是用坐标的分量表示的，所以如果换一个坐标，它的分量就完全不一样了。

假如所有的位置都使用同样的坐标，那自然就没有什么争议了，然而我们前面已经反复强调，不同位置的人可能出于各种原因，使用了不同的坐标系，因此，当我们写出一个向量$A^{\mu}$时，严格来讲应该还要注明是在$\boldsymbol{x}$位置测量的：$A^{\mu}(\boldsymbol{x})$，只有不引起歧义的情况下，我们才能省略它。

到这里，我们已经能够进行一些计算，比如$A^{\mu}$是在$\boldsymbol{x}$处测量的，而$\boldsymbol{x}$处的模长计算公式为$ds^2 = g_{\mu\nu} dx^{\mu} dx^{\nu}$，因此，$A^{\mu}$的模长为$\sqrt{g_{\mu\nu} A^{\mu}A^{\nu}}$，它是一个客观实体。

[![如图，可以在球面上每一点建立不同的局部坐标系，至少这些坐标系的竖直方向的轴指向是不一样的。](<https://spaces.ac.cn/usr/uploads/2016/10/605188233.jpg>)](<https://spaces.ac.cn/usr/uploads/2016/10/605188233.jpg>)

如图，可以在球面上每一点建立不同的局部坐标系，至少这些坐标系的竖直方向的轴指向是不一样的。

有些时候我们要比较不同位置的两个向量，这就涉及到了作差。特别的是，由于不同位置的坐标系不同，我们如果直接对两个不同位置的向量分量进行相减，是没有意义的。这就好比中国的5元人民币和美国的5元美金，我们不能得出“5元美金\-5元人民币=(5\-5)=0”的结论。不过这种情况只是单位选取不一致产生的，选取相同的单位就没事了，但是向量的测量结果不仅跟单位有关，而且跟坐标系有关。比如，一架飞机从中国飞到美国，经过中国某位置时，测得的速度为$(300,300,300)$，到了美国某位置时，测得的速度也是$(300,300,300)$，单位都用“千米／小时”，但是在中国时是中国人测的，在美国时是美国人测的，大家都知道，美国和中国分别位于地球的两端，它们所建立的局部坐标系肯定是不一样的，所以也不能断定飞机在两地的速度差是$(300-300,300-300,300-300)=(0,0,0)$，因为，还跟方向有关呢！

说白了，产生这个问题的原因就是不同位置使用了不同的坐标系——坐标的单位长度不一样，坐标轴的指向不一样，等等。因此，我们需要将一个位置的向量坐标，变换到另外一个位置的坐标中去，才能对分量进行比较。这里，我们只考虑两个距离为无穷小的位置$\boldsymbol{x}$和$\boldsymbol{x}+d\boldsymbol{x}$的变换，即怎么将位于$\boldsymbol{x}+d\boldsymbol{x}$处测量的向量$A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})$（$\boldsymbol{x}+d\boldsymbol{x}$处有自己的坐标系），用$\boldsymbol{x}$处的坐标系表示出来。很显然，这涉及到一个变换矩阵，因此关键是确定这个变换矩阵。

怎么找到这个矩阵呢？设想一下我们可以在$\boldsymbol{x}$处放一个试探向量，然后把这个向量放到$\boldsymbol{x}+d\boldsymbol{x}$处，看看它变成了什么。只要有足够多的试探向量，我们就可以确定这个变换矩阵了。因此，我们需要一组最自然的试探向量来给我们提供参考，而我们已经知道了测地线这个客观实体，事实上测地线就提供了一个最自然的参照物，没有比它更加自然了。我们可以把测地线的方程改写为
$$d\left(\frac{d x^{\mu} }{ds}\right)=-\Gamma_{\alpha\beta}^{\mu} \frac{d x^{\alpha} }{ds}d x^{\beta} \tag{28} $$
这个公式的意思是，如果在当前位置$\boldsymbol{x}$有个单位向量$\frac{dx^{\mu}}{ds}$，那么它沿着测地线方向往前走了$d\boldsymbol{x}$后，单位向量$\frac{dx^{\mu}}{ds}$的变化量为$d\left(\frac{d x^{\mu} }{ds}\right)$，它也等于$-\Gamma_{\alpha\beta}^{\mu} \frac{d x^{\alpha} }{ds}d x^{\beta}$，即
$$\frac{dx^{\mu}}{ds}\quad\to\quad \frac{dx^{\mu}}{ds}-\Gamma_{\alpha\beta}^{\mu} \frac{d x^{\alpha} }{ds}d x^{\beta} \tag{29} $$
那么，从$\boldsymbol{x}$到$\boldsymbol{x}+d\boldsymbol{x}$的坐标变换矩阵（雅可比矩阵）就为
$$\frac{\partial\left(\frac{dx^{\mu}}{ds}-\Gamma_{\alpha\beta}^{\mu} \frac{d x^{\alpha} }{ds}d x^{\beta}\right)}{\partial \frac{dx^{\nu}}{ds}}=\delta_{\nu}^{\mu}-\Gamma_{\nu\beta}^{\mu} d x^{\beta} \tag{30} $$
这样，如果反过来，从$\boldsymbol{x}+d\boldsymbol{x}$到$\boldsymbol{x}$的坐标变换矩阵就为
$$\label{lianluo}\delta_{\nu}^{\mu}+\Gamma_{\nu\beta}^{\mu} d x^{\beta} \tag{31} $$
这样子看，<strong>系数$\Gamma_{\alpha\beta}^{\mu}$把$\boldsymbol{x}$和$\boldsymbol{x}+d\boldsymbol{x}$两处的坐标联系了起来，因此称之为“联络系数”是非常恰当的</strong>。

知道了坐标变换方式，我们就容易得到，如果将$A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})$放到$\boldsymbol{x}$位置测量，那么结果将是
$$\left(\delta_{\nu}^{\mu}+\Gamma_{\nu\beta}^{\mu} d x^{\beta}\right) A^{\nu}(\boldsymbol{x}+d\boldsymbol{x})=A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})+\Gamma_{\nu\beta}^{\mu} A^{\nu}(\boldsymbol{x}+d\boldsymbol{x}) d x^{\beta} \tag{32} $$

### 协变导数

我们之前提到过，向量有作差的需求，并且在联络的部分已经研究了相距无穷小的两个位置的坐标变换，得到式$(32)$的结果：如果将$A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})$放到$\boldsymbol{x}$位置测量，那么结果将是
$$A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})+\Gamma_{\nu\beta}^{\mu} A^{\nu}(\boldsymbol{x}+d\boldsymbol{x}) d x^{\beta} \tag{33} $$
这样一来，我们就可以直接作差了，因为两个向量已经是在同一位置测量了。
$$A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})+\Gamma_{\nu\beta}^{\mu} A^{\nu}(\boldsymbol{x}+d\boldsymbol{x}) d x^{\beta}-A^{\mu}(\boldsymbol{x}) \tag{34} $$
我们可以研究极限情形：
$$\lim_{d\boldsymbol{x} \to 0} \frac{A^{\mu}(\boldsymbol{x}+d\boldsymbol{x})+\Gamma_{\nu\beta}^{\mu} A^{\nu}(\boldsymbol{x}+d\boldsymbol{x}) d x^{\beta}-A^{\mu}(\boldsymbol{x})}{dx^{\beta}} \tag{35} $$
不难得出，结果将是
$$\frac{\partial A^{\mu}}{\partial x^{\beta}}+\Gamma_{\nu\beta}^{\mu} A^{\nu} \tag{36} $$
这称为向量$A^{\mu}$的<strong>协变导数</strong>，记为
$$A^{\mu}_{;\beta}=\frac{\partial A^{\mu}}{\partial x^{\beta}}+\Gamma_{\nu\beta}^{\mu} A^{\nu} \tag{37} $$
而很自然地，将下述结果称为<strong>协变微分</strong>：
$$D A^{\mu}=dA^{\mu}+\Gamma_{\nu\beta}^{\mu} A^{\nu}dx^{\beta} \tag{38} $$
如果除以线元$ds$，那么得到微分
$$\frac{D A^{\mu}}{Ds}=\frac{dA^{\mu}}{ds}+\Gamma_{\nu\beta}^{\mu} A^{\nu}\frac{dx^{\beta}}{ds} \tag{39} $$
可见结果是跟曲线$x^{\beta}$的选取有关的，我们一般选取为测地线，上式称之为<strong>测地导数（沿着测地线求导）</strong>。

回顾整个过程，我们导出协变导数的出发点是：<strong>直接将两个位置的向量分量相减是没有意义的，如果有必要，则需要把一个位置的向量变换到另外一个位置上去。如果研究极限情况，就得到了协变导数。协变导数是空间中有几何意义的导数定义，因此它也是一个客观实体。</strong>

在一般的张量分析或者黎曼几何教程中，导出协变导数的方式有很多。有的教材采取了这样的思路：因为我们知道直角坐标系下梯度的具体形式，所以从直角坐标系出发，通过变换规律得到其它坐标系下的导数形式。看上去很合理，但事实上隐含了平直空间的假设——也就是说，虽然使用了曲线坐标系，但是能够通过某种变换变回直角坐标系，否则这种做法不成立。这意味着空间是平直的；而弯曲的空间具有相同的结果，某种意义上只是个巧合。如果用同样的思路去导出黎曼曲率，会发现结果恒等于0——因为本来就是平直空间。

还有其他的一些导出方式，总的来说，我认为多数方式怎么看都“不够几何”，更多的是代数的演算。笔者这里使用了尽量几何化的思路。
