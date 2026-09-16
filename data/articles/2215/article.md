# 体积与阿达马不等式

> 作者：苏剑林 · 科学空间 · 2013-12-26
>
> 原文：<https://spaces.ac.cn/archives/2215>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<strong>阿达马不等式</strong>
设有$n$阶实矩阵$\boldsymbol{A}=(a_{ij})_{n\times n}$，那么它的行列式满足阿达马(Hadamard)不等式
$$\begin{equation}
\left(\det \boldsymbol{A}\right)^2 \leq \prod\limits_{i=1}^{n}\left(a_{1i}^2+a_{2i}^2+\dots+a_{ni}^2\right)
\end{equation}$$

这是阿达马在1893年首先发表的。根据体积就是行列式的说法，上述不等式具有相当明显的几何意义。当$n=2$时，它就是说平行四边形的面积不大于两边长的乘积；当$n=3$时，它就是说平行六面体的体积不大于三条棱长的乘积；高维可以类比。这些结论在几何中几乎都是“显然成立”的东西。因此很难理解为什么这个不等式在1893年才被发现。当然，代数不会接受如此笼统的说法，它需要严格的证明。

可是，不论是教科书上的提示还是我在互联网上搜索到的资料（比如本文附件的文章），它们几乎都是千篇一律使用正定矩阵的思想来做的。但是这个几何意义如此明显的、如此基本的不等式，为了证明它却需要正定矩阵知识的奠基，不能不说是一种本末倒置。这也反映了国内不少数学工作者缺乏数学思想、人云亦云的不良作风。（第一个人这么做了，大量的人也就跟着做了，就算方法略有不同，还是没有改变原来的路子。）事实上，完全不需要正定矩阵，只通过一个很几何的方法就可以证明这道不等式。

<strong>最佳坐标系</strong>
将一个矩阵看成是$n$个列向量$\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n$的集合：
$$\begin{equation}
\boldsymbol{A}=[\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n]
\end{equation}$$

为了求得这$n$个列向量所构成的平行$n$维体的体积，我们需要把原来的坐标系稍微转动一下，也就是说，换一个直角坐标系，使得这$n$个向量的形式更为简单。而用代数的语言说，那就是找一个正交矩阵$\boldsymbol{U}$，使得$\boldsymbol{A}=\boldsymbol{U}\boldsymbol{T}$，$\boldsymbol{T}$的形式尽可能简单（容易计算行列式）。

什么坐标系才是最优的呢？设$\boldsymbol{e}_1,\boldsymbol{e}_2,\dots,\boldsymbol{e}_n$为新坐标系的基，它们当然也是两两正交的单位向量。（只有这样才保持体积不变，用代数的语言就是正交变换保持行列式不变。）。很自然的，我们选取其中一个向量，比如$\boldsymbol{e}_1=\frac{\boldsymbol{a}_1}{|\boldsymbol{a}_1|}$，为一条坐标轴，那么$\boldsymbol{a}_1$有最简单的表达式；为使$\boldsymbol{a}_2$的表达式尽可能简单，我们应该让$\boldsymbol{a}_2$位于一个坐标平面上，从构造的角度来讲，那就是从$\boldsymbol{e}_1$和$\boldsymbol{a}_2$出发构造一个向量，使得它就是$\boldsymbol{e}_2$轴，这样子$\boldsymbol{a}_2$就是$\boldsymbol{e}_1$和$\boldsymbol{e}_2$的线性组合。同样地，为使得$\boldsymbol{a}_3$表达式尽可能简单，那么$\boldsymbol{a}_3$应该要表示为$\boldsymbol{e}_1,\boldsymbol{e}_2,\boldsymbol{e}_3$的线性组合，这等价于用$\boldsymbol{a}_1,\boldsymbol{a}_2,\boldsymbol{a}_3$构造出$\boldsymbol{e}_3$ 来。将这个过程一直进行下去，那么其实就是向量$\{\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n\}$的规范正交化过程了！

也就是说由$\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n$规范正交化得到的向量组$\boldsymbol{e}_1,\boldsymbol{e}_2,\dots,\boldsymbol{e}_n$是表示原来向量的一个最佳的直角坐标系之一了。这个向量组构成了一个正交矩阵$\boldsymbol{U}$
$$\begin{equation}
\boldsymbol{U}=[\boldsymbol{e}_1,\boldsymbol{e}_2,\dots,\boldsymbol{e}_n]
\end{equation}$$

在此坐标系之下，有
$$\begin{equation}
\boldsymbol{A}=[\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n]=[\boldsymbol{e}_1,\boldsymbol{e}_2,\dots,\boldsymbol{e}_n]\boldsymbol{T}=\boldsymbol{U}\boldsymbol{T}
\end{equation}$$
其中
$$\begin{equation}
\boldsymbol{T}=\left[ {\begin{array}{\cdot {20}{c}}
{\lambda_1}&{\cdot }&{\dots}&{\cdot }\\
{}&{\lambda_2}&{\dots}&{\cdot }\\
{}&{}&{\ddots}&{\vdots}\\
{0}&{}&{}&{\lambda_n}
\end{array}} \right]
\end{equation}$$
是一个上三角形阵。

<strong>完成证明</strong>
有了上面的结论，那么阿达马不等式的证明就很简单了。因为$\boldsymbol{U}$是正交矩阵，自然有
$$\begin{equation}
\begin{array}{l}
\left| {\boldsymbol{a}_1} \right| = \left| {\left( \lambda_1 \right)} \right|\geq \lambda_1,\left| {\boldsymbol{a}_2} \right| = \left| {\left( {\begin{array}{\cdot {20}{c}}
\cdot \\
\lambda_2
\end{array}} \right)} \right|\geq \lambda_2\\
\dots
\\
\left| {\boldsymbol{a}_n} \right| = \left| {\left( {\begin{array}{\cdot {20}{c}}
\begin{array}{l}
\cdot \\
\cdot \\
\vdots
\end{array}\\
\lambda_n
\end{array}} \right)} \right|\geq \lambda_n
\end{array}
\end{equation}$$

所以
$$\begin{equation}
\begin{aligned}
(\det\boldsymbol{A})^2&=(\det\boldsymbol{T})^2=(\lambda_1 \lambda_2 \dots\lambda_n)^2\\
&\leq |\boldsymbol{a}_1|^2 |\boldsymbol{a}_2|^2 \dots |\boldsymbol{a}_n|^2\\
&=\prod\limits_{i=1}^{n}\left(a_{1i}^2+a_{2i}^2+\dots+a_{ni}^2\right)
\end{aligned}
\end{equation}$$
这就完成了阿达马不等式的证明。一个几何的证明！

<strong>结束语</strong>
学习高等代数，尤其是学习矩阵相关的理论，要紧紧抓住它的几何意义，这是非常有益处且非常有必要的，对此我深有感悟。不论是从记忆还是推理的角度来讲，几何都给了我们最直观的思路，使得我们不至于对一个概念完全没有任何头绪的接受。不管怎样强调都不为过的是：死记硬背在数学中是没有用的。

<strong>附件：</strong>
[阿达玛\_Hadamard\_不等式的证明及几何意义\.pdf](<https://spaces.ac.cn/usr/uploads/2013/12/973083582.pdf>)
