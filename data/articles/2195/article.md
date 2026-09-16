# 用二次方程判别式判断正定矩阵

> 作者：苏剑林 · 科学空间 · 2013-12-24
>
> 原文：<https://spaces.ac.cn/archives/2195>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

快要学期末了，不少学霸开始忙碌起来了。不过对非学霸的我来说，基本上每天都是一样的，希望把自己感兴趣的东西深入研究下去，因为我觉得，真正学会点有用的东西才是最重要的。数学分析和高等代数老师都要求写课程论文，我也写了我比较感兴趣的“欧拉数学”和“超复数研究”，之后会把这部分内容与大家分享。

虽然学期已经接近尾声了，但是我们的课程还没有上完。事实上，我们的新课一直上到十八周\~随着考试的接近，我们的《高等代数》课程也已经要落幕了。最近在上的是二次型方面的内容，讲到正定二次型和正定矩阵。关于正定矩阵的判别，教科书上提供了两个判别方法，一个是基于定义的初等变换，另外一个就是主子式法。前者无可厚非，但是后者我似乎难以理解——它虽然是正确的，但是它很丑，计算量又大。我还没有想清楚主子式法到底有什么好的？在我看来，本文所探讨的基于二次方程判别式的方法才是简单、快捷的。

<strong>正定二次型</strong>
所谓正定二次型，就是关于n个变量$x_1,x_2,...,x_n$的二次齐次函数，只要$x_i$不全为0，它的值恒为正数。比如
$$2 x_1^2+x_2^2-2 x_1 x_2=x_1^2+(x_2-x_1)^2$$
这是一个比较简单的正定二次型，多元的还有
$$5 x_1^2+x_2^2+5 x_3^2+4 x_1 x_2-8 x_1 x_3-4 x_2 x_3$$
等等。但是上式并不容易看出它的正定性，为了判别，我们可以讲它改写为矩阵形式，然后通过初等变换或者主子式法判别。但是我是通过考虑方程
$$f(t)=5 x_1^2+x_2^2+5 t^2+4 x_1 x_2-(8 x_1 + 4 x_2 )t=0$$
的解来判别的，首先容易检验
$$f(0)=5 x_1^2+x_2^2+4 x_1 x_2=(2 x_1+x_2)^2 +x_1^2 \leq 0$$
因为它的判别式
$$\begin{aligned}
\Delta &=(8 x_1 + 4 x_2 )^2-4\times 5\times(5 x_1^2+x_2^2+4 x_1 x_2)\\
&=-36 x_1^2-4 x_2^2-16 x_1 x_2\\
&=-4[5 x_1^2+(2 x_2 -x_1)^2]
\end{aligned}$$
当$x_1,x_2$不全为0时，判别式为负数，因此原方程无解，$f(t)$恒大于0；当$x_1=x_2=0$时，只有一个解$t=0$，对于$t > 0$，也有$f(t) > 0$。因此上述二次型正定。

<strong>正定矩阵</strong>
把上面的方法系统整理起来，就得到如下的正定矩阵判别法了。

所谓正定矩阵，就是一个n阶实矩阵$\boldsymbol{A}_n$，对于任意的n维非零实列向量$\boldsymbol{x}$，都有
$$\boldsymbol{x}^{T} \boldsymbol{A}_n \boldsymbol{x} > 0$$

把$\boldsymbol{A}_n$写成分块形式
$$\boldsymbol{A}_n=\left[ \begin{matrix}
\boldsymbol{A}_{n-1}& \boldsymbol{b}\\
{\boldsymbol{c}^T}&d
\end{matrix}\right]$$
其中$\boldsymbol{A}_{n-1}$是n\-1阶矩阵。并且将向量$\boldsymbol{x}$写成分块向量
$$\left( {\begin{matrix}
{\boldsymbol{y}}\\t
\end{matrix}} \right)$$
其中$\boldsymbol{y}$是n\-1维向量，它$\boldsymbol{x}$的前n\-1个元素，$t$是$\boldsymbol{x}$的第n个元素。那么
$$\begin{aligned}
f(t)=\boldsymbol{x}^T \boldsymbol{A}_n \boldsymbol{x}&=\left(\boldsymbol{y}^{T},t\right)\left[ {\begin{array}{\cdot {20}{c}}
\boldsymbol{A}_{n-1}& \boldsymbol{b}\\
{\boldsymbol{c}^T}&d
\end{array}} \right] \left( {\begin{array}{\cdot {20}{c}}
{\boldsymbol{y}}\\
t
\end{array}} \right)\\
&=d t^2+\left(\boldsymbol{c}^{T} \boldsymbol{y}+\boldsymbol{y}^{T}\boldsymbol{b}\right)t+\boldsymbol{y}^{T}\boldsymbol{A}_{n-1}\boldsymbol{y}\\
&=d t^2+\left[\left(\boldsymbol{b}+\boldsymbol{c}\right)^{T}\boldsymbol{y}\right]t+\boldsymbol{y}^{T}\boldsymbol{A}_{n-1}\boldsymbol{y}
\end{aligned}$$

如果矩阵$\boldsymbol{A}_n$是正定的，那么$f(0)=\boldsymbol{y}^{T}\boldsymbol{A}_{n-1}\boldsymbol{y} \geq 0$，即$\boldsymbol{A}_{n-1}$是正定的，这是一个必要条件；另外一个必要条件是$d > 0$。

同时$f(t)=0$的判别式必须非正数，即
$$\begin{aligned}
\Delta &=\left[\left(\boldsymbol{b}+\boldsymbol{c}\right)^{T}\boldsymbol{y}\right]^2-4d\boldsymbol{y}^{T}\boldsymbol{A}_{n-1}\boldsymbol{y}\\
&=4\boldsymbol{y}^{T}\left(\boldsymbol{m}\boldsymbol{m}^T-d\boldsymbol{A}_{n-1}\right)\boldsymbol{y}\\
&\leq 0
\end{aligned}$$
其中$\boldsymbol{m}=\frac{1}{2}(\boldsymbol{b}+\boldsymbol{c})$。上式也就是说$\left(d\boldsymbol{A}_{n-1}-\boldsymbol{m}\boldsymbol{m}^T\right)$是正定的。并且注意到矩阵$\boldsymbol{m}\boldsymbol{m}^T$是正定的，因此如果$\left(d\boldsymbol{A}_{n-1}-\boldsymbol{m}\boldsymbol{m}^T\right)$是正定的，那么$\boldsymbol{A}_{n-1}$一定是正定的（两个正定矩阵的和还是正定矩阵），所以后者已经包含在前者里边。那么我们就得到了如下判别法：

> 矩阵$\boldsymbol{A}_n=\left[ \begin{array}{*{20}{c}}
\boldsymbol{A}_{n-1}& \boldsymbol{b}\\
{\boldsymbol{c}^T}&d
\end{array} \right]$是正定的，当且仅当$d >0$且$\left(d\boldsymbol{A}_{n-1}-\boldsymbol{m}\boldsymbol{m}^T\right)$是正定的。其中$\boldsymbol{m}=\frac{1}{2}(\boldsymbol{b}+\boldsymbol{c})$。

这样n阶矩阵的正定性问题就变成了n\-1阶矩阵的正定性问题，类似地可以变成n\-2阶矩阵的正定性判定\.\.\.直到变为单个数的正性判断。

<strong>结束语</strong>
不论怎么看，本文所提供的判别法的计算量应该比初等变换和主子式法要低，而且推导也比较简单。所以我现在还想不明白为什么教材和网络上对此几乎没有提及。事实上，即使在写作业中，我也愿意用判别式的方法而不是常规的方法来判断正定性。因为对于我来说，我自己想出来的、能够解决问题的方法才是好方法。如果我没有接触二次型的矩阵基础知识，我也可以用判别式来判断一个二次型是否正定，而纵使让我探究上几年，估计我也不会想出像主子式那样“丑陋”的方法。

<strong>不过可能让读者感到意外的是，主子式这样“丑陋”的方法，却是跟本文的方法本质一样的。我们先判断对角线最下面的元素是否为正数，然后把矩阵降低一阶，再判断新矩阵的对角线最下面的元素是否为正数，依此类推\.\.\.每一步需要判断的对角线最下面的元素其实就相当于一个主子式。两者没有本质区别，但是如果真的使用计算机编程，那么显然本文的方法在效率上会略胜一筹。</strong>
