# 矩阵化简二次型（无穷小近似处理抛物型）

> 作者：苏剑林 · 科学空间 · 2012-12-25
>
> 原文：<https://spaces.ac.cn/archives/1841>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

（阅读本文最好有一定的线性代数基础，至少对线性代数里边的基本概念有所了解。）

这学期已经接近尾声了，我们的《解析几何》已经讲到化简二次曲线了。可是，对于没有线性代数的其他同学们，直接用转轴和移轴这个计算公式来变换，那计算量会让我们很崩溃的；虽然那个“不变量”方法计算上有些简单，却总让人感到很诡异，总觉得不知从何而来，而且又要记一堆公式。事实上，如果有线性代数的基础，这些东西变得相当好理解的。我追求用统一的方法求解同一种问题，即用统一的方式处理所有的二次型，当然也希望计算量简单一点。

<strong>一般的模型</strong>

一般的二次型可以写成
$$x^T A x + 2 b^T x + c=0$$

其中$x,b$都是n维列向量（各元素为$x_i$和$b_i$），A是n阶方阵（各元素为$a_{ij}$），c是常数。在这里，我们只讨论n=2和n=3的情况。化简二次型的过程，可以归结为A矩阵的简化。

一般的文章在讨论这个问题的时候，把矩阵A拓展成(n\+1)维的矩阵$\widetilde{A}$，用方块矩阵表示为：
$$\widetilde{A}=\left[\begin{array}{c c}A & b\\ b^T & c \end{array}\right]$$

而向量$x$则拓展为(n\+1)维的向量$\widetilde{x}$：
$$\widetilde{x}=\left[\begin{array}{c}x\\1\end{array}\right]=\left[\begin{array}{c}x_1\\x_2\\...\\x_n\\1\end{array}\right]$$

于是二次型变成了
$$\widetilde{x}^T \widetilde{A} \widetilde{x}=0$$

在这种情况下就变成了矩阵$\widetilde{A}$的化简。不过在我看来，这种方法很简明扼要，但是计算量偏大了，比如平面的二次曲线会产生一个三阶方阵，求特征值由原来的二次方程变成了三次方程，而且行列式的计算中每升一阶就会麻烦好多。在这种情况下，简洁带来的好处有点“得不偿失”的感觉。所以我尝试直接用n阶矩阵的方式对n维二次型进行化简，得到以下的结果。

<strong>自己的理想化模型</strong>

考虑一个保距变换矩阵R，所谓保距，就意味着在这个变换之下，向量的“距离”不变。即$v^T v =(Rv)^T (Rv)=v^T (R^T R) v$，所以我们有$R^T R=I$，矩阵的转置等于其逆。在此种变换下，曲线的形状保持不变。最常见的保距变换就是以原点为中心进行旋转。

如果化简后的二次型形式为：
$$x^T M x =k$$

其中M是对角线矩阵，k是常数，这将是二次型最简单的形式，是我们研究最方便的形式。

考虑保距变换与平移的复合，即
$$x \to R(x+x_0)$$

$x_0$也是一个n维常数列向量，代入上式，就可以得到：
$$\begin{eqnarray*} (x+x_0)^T R^T A R(x+x_0)=k \\ x^T(R^T M R)x+x_0^T[(R^T M R)+(R^T M R)^T]x+x_0^T (R^T M R) x_0-k=0 \end{eqnarray*}$$

对应原来的一般二次型，则有
$$\begin{eqnarray*} R^T M R=A \\ \left[(R^T M R)+(R^T M R)^T \right]x_0 =2b \\ x_0^T (R^T M R) x_0 -k=c \end{eqnarray*}$$

其中$R^T A R=R^{-1} A R=M$是一个相似矩阵变换，一般情况下，其最简形式是由矩阵A的所有特征值组成的对角线矩阵。因此，相对来说是很容易求出它的形式的，接下来，有：
$$\begin{eqnarray*} (A+A^T)x_0=2b \\ x_0^T A x_0-k=c \end{eqnarray*}$$

关键是要求出$x_0$的具体形式，这等价于解一道n元一次方程，即
$$x_0=2(A+A^T)^{-1}b$$

关于二次型，有一些非常令人兴奋的结果（证明参考《数学桥》）：

> 1、任何一个实二次型，总有一个实对称矩阵（关于对角线）A与之对应；
> 2、任何一个实对称矩阵A，总有n个实数特征根。

有了这两个保证，我们就不怕诸如“特征根”是复数怎么办的情况了，而且上面的计算还可以进一步化简，因为对称矩阵有$A^T=A$，所以就变成了
$$x_0=A^{-1} b$$

<strong>特殊情形的一般处理</strong>

到此，问题似乎已经全部解决。只要先求出特征根，得出矩阵M；接着求出$x_0=A^{-1} b$，然后代入$x_0^T A x_0-k=c$来求常数项就行了。尤其是对于二次曲线，计算量真的是非常的少。但问题是，要是矩阵A不可逆怎么办？这种情况是有可能的，即特征根出现0的情况，对于平面来说它对应着抛物线的情况。这样求逆阵是不可能的，难道在此失效了？

对于抛物线，其实我们还是可以将它看成双曲线或抛物线的极限的。比如抛物线$2py+x^2=0$，可以看成$2py+x^2+ \varepsilon y^2$，其中$\varepsilon \to 0$，配平方：
$$x^2+ \varepsilon (y+\frac{p}{\varepsilon})^2-\frac{p^2}{\varepsilon}=0$$

还是各项都是平方项的形式。

这给我们提供了一种思路，即可以把原来不可逆的矩阵A加以上一个微小的的偏离，使之成为可逆的矩阵，求出的结果后，再取偏离趋于0的极限。这是一种相当有趣的思路。下面我们沿着这一思路进行探究。

关键是偏离要加在哪里呢？怎么加呢？其实一般来讲，改动原来矩阵任意一个元素都可以，但是为了研究的方便和处理上的统一，我们决定改动对角线上的元素。假如只有一个零根，就把矩阵A的对角线上第一个数加上$\varepsilon_1$，假如有两个零根，就继续在对角线上的第二个数加$\varepsilon_2$，依此类推，这样做目的是统一处理高维情况，记新矩阵为$\bar{A}$。如果有k个零根，那么行列式$\det \bar{A} =(a_{11}+a_{22}+...+a_{nn})\varepsilon_1 \varepsilon_2 ... \varepsilon_k$（即对角线元素之和乘以所有的$\varepsilon_i$，读者可以根据矩阵的“迹”的知识轻易得到这一点。）

于是一切都变回原来的步骤：

原来的n个特征值$\lambda_i$变为了$\lambda_i -\varepsilon_i$，这样就没有了零根的出现（把零根排在前方），然后通过$x_0=(\bar{A})^{-1} b$求出常向量$x_0$，当然这是一个无穷大的向量，这并不重要，保留一阶无穷小项即可，然后我们就可以求出常数项：$k=x_0^T A x_0-c$。它应当是下面的形式：
$$\frac{C_1}{\varepsilon_1}+\frac{C_2}{\varepsilon_2}+...+Const$$

对比
$$x^2+ \varepsilon (y+\frac{p}{\varepsilon})^2-\frac{p^2}{\varepsilon}=0$$

不难发现其实$2\sqrt{C_i}x_i$就是对应的特征根为0的那一项。（抛物项）

<strong>实例：</strong>

1、椭圆（双曲）型。
$$x^2+6xy+y^2+6x+2y-1=0$$

<strong>解：</strong>
上式即
$$\left[\begin{array}{c c}1 & 3\\ 3 & 1 \end{array}\right] \left[\begin{array}{c}x\\ y \end{array}\right]+2[3,1]\left[\begin{array}{c c}x\\ y \end{array}\right]-1=0$$

即
$$A=\left[\begin{array}{c c}1 & 3\\ 3 & 1 \end{array}\right],b=\left[\begin{array}{c}3\\1 \end{array}\right],c=-1$$

求出A的两个特征值：$(1-\lambda)^2-3^2=0$，分别为\-2和4。

接着求
$$\begin{aligned}x_0=\left[\begin{array}{c c}1 & 3\\ 3 & 1 \end{array}\right]^{-1} \left[\begin{array}{c}3\\1 \end{array}\right]=\left[\begin{array}{c}0\\1 \end{array}\right] \\ k=x_0^T A x_0-c=x_0^T b=[0,1]\left[\begin{array}{c}3\\1 \end{array}\right]+1=2\end{aligned}$$

最终结果为
$$\left[\begin{array}{c c}-2 & 0\\0 & 4 \end{array}\right]\left[\begin{array}{c}x\\ y \end{array}\right]=2$$

即：$-x^2+2y^2=1$

2、抛物型
$$x^2-4xy+4y^2+2x-2y-1=0$$

<strong>解：</strong>
上式即
$$\left[\begin{array}{c c}1 & -2\\ -2 & 4 \end{array}\right] \left[\begin{array}{c}x\\ y \end{array}\right]+2[1,-1]\left[\begin{array}{c c}x\\ y \end{array}\right]-1=0$$

即
$$A=\left[\begin{array}{c c}1 & -2\\ -2 & 4 \end{array}\right],b=\left[\begin{array}{c}1\\-1 \end{array}\right],c=-1$$

两个特征根为0和5

将矩阵A变为$\bar{A}$，即
$$\bar{A}=\left[\begin{array}{c c}1+\varepsilon & -2\\ -2 & 4 \end{array}\right]$$

对于二阶方阵，求逆阵的运算比较简单：

$$\left[\begin{array}{c c}a & c\\ b & d\end{array}\right]^{-1}=\frac{\left[\begin{array}{c c}d & -b\\ -c & a\end{array}\right]}{\det \left[\begin{array}{c c} a & c\\ b & d\end{array}\right]}$$

二阶行列式的近似展开也较为简单：
$$\det \left[\begin{array}{c c} a+\varepsilon & c\\ b & d \end{array} \right] =ad-bc+(a+d)\varepsilon$$

因此
$$\bar{A}^{-1}=\frac{1}{5\varepsilon}\left[\begin{array}{c c}4 & 2\\ 2 & 1 \end{array}\right]$$
（精确到一阶无穷小）

$$x_0=\bar{A}^{-1}b=\frac{1}{5\varepsilon}\left[\begin{array}{c c}4 & 2\\ 2 & 1 \end{array}\right]\left[\begin{array}{c c}1\\-1 \end{array}\right]=\frac{1}{5\varepsilon}\left[\begin{array}{c}2\\1 \end{array}\right]$$

求得：
$$k=x_0^T A x_0-c=x_0^T b=\frac{1}{5\varepsilon}[2,1]\left[\begin{array}{c}1\\-1 \end{array}\right]+1=\frac{1}{5\varepsilon}+1$$

对应地产生了$2\sqrt{\frac{1}{5}}y$这一项，即二次曲线化简为
$$5x^2 \pm 2\sqrt{\frac{1}{5}}y=0$$
(这里的加减并不重要，一个向上一个向下而已）

对于三维曲面甚至更高维的情况，这种方法也是以同样的步骤进行的，当然在计算量上有所增加，但是并没有改变其实质。所谓“不变量”方法化简，我想，不变量的寻找，其实也是通过上面的一系列步骤产生的。放到解析几何中，只不过脱去了它的线性代数外衣，（并非故意地）加上了一点令人疑惑的色彩而已。

<strong>同时，我由衷地希望，在解析几何的考试中，应当不限制方法地出题目，不要出现诸如“用不变量方法化简\.\.\.”之类的固定方法的死板题目，这对我们的能力提升没有任何好处。毕竟，掌握书本上的方法并不是必要的，掌握处理某一问题的数学技能才是根本。</strong>
