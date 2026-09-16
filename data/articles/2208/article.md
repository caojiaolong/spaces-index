# 《新理解矩阵5》：体积=行列式

> 作者：苏剑林 · 科学空间 · 2013-12-25
>
> 原文：<https://spaces.ac.cn/archives/2208>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《新理解矩阵3》：行列式的点滴](<https://spaces.ac.cn/archives/1770/>)中，笔者首次谈及到了行列式的几何意义，它代表了n维的“平行多面体”的“体积”。然而，这篇文章写于我初学矩阵之时，有些论述并不严谨，甚至有些错误。最近笔者在写期末论文的时候，研究了超复数的相关内容，而行列式的几何意义在我的超复数研究中具有重要作用，因此把行列式的几何意义重新研究了一翻，修正了部分错误，故发此文，与大家分享。

一个$n$阶矩阵$A$可以看成是$n$个$n$维列向量$\boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n$的集合
$$A=(\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n)$$
从代数的角度来看，这构成了一个矩阵；从几何的角度来看，这$n$个向量可以建立一个平行$n$维体。比如：平行四边形就是“平行二维体”，平行六面体就是“平行三维体”，高阶的只需要相应类比，不需要真正想象出高维空间的立体是什么样。

让我们考虑矩阵$A$的行列式$\det A$，我们知道$\det A$有如下性质：

> <strong>行列式性质</strong>
> 
> 
> 
> 1、行列式是$\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n$的一个函数，即$\det A=f(\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n)$；
> 
> 
> 
> 2、（线性1）行列式的某一列乘上常数$\alpha$，则行列式的值也乘上$\alpha$，即$ f(\boldsymbol{x}_1,\dots,\alpha\boldsymbol{x}_i,\dots,\boldsymbol{x}_n)=\alpha f(\boldsymbol{x}_1,\dots,\boldsymbol{x}_i,\dots,\boldsymbol{x}_n)$；
> 
> 
> 
> 3、（线性2）将行列式的某一列写成两列之和，那么行列式也相应地成为两个行列式之和，即$ f(\boldsymbol{x}_1,\dots,\boldsymbol{x}_i,\dots,\boldsymbol{x}_n)= f(\boldsymbol{x}_1,\dots,\boldsymbol{y}_i,\dots,\boldsymbol{x}_n)+f(\boldsymbol{x}_1,\dots,\boldsymbol{z}_i,\dots,\boldsymbol{x}_n)$，其中$\boldsymbol{x}_i=\boldsymbol{y}_i+\boldsymbol{z}_i$，性质二和三表明$f$是关于每个向量的线性函数；
> 
> 
> 
> 4、（反对称）只要有两列相同，那么行列式值为0，即$f(\dots,\boldsymbol{x},\dots,\boldsymbol{x},\dots)=0$；
> 
> 
> 
> 5、（归一）单位矩阵的行列式为1，即$f(I)=1$。

一个惊人的事实是，行列式可以由上面五条性质唯一确定！即由上面五条性质就可以唯一确定一个函数$f$，这个函数就是矩阵的行列式。

从几何的角度来看，用这$n$个向量，可以生成$n$维空间的一个平行$n$维体。让我们来考虑这个平行$n$维体的体积$V$。只在第一卦限讨论，那么体积具有下面的性质（只在第一卦限讨论，限保证了所有的向量和因子都是正数。）

> <strong>体积性质</strong>
> 
> 
> 
> 1、体积是这$n$个向量的一个函数$V(\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_n)$；
> 
> 
> 
> 2、将某个向量乘以$\alpha$，也就是把它的长度变为来说的$\alpha$倍，那么体积也增大$\alpha$倍，即$V(\boldsymbol{x}_1,\dots,\alpha\boldsymbol{x}_i,\dots,\boldsymbol{x}_n)=\alpha V(\boldsymbol{x}_1,\dots,\boldsymbol{x}_i,\dots,\boldsymbol{x}_n)$；
> 
> 
> 
> 3、体积是可加的，即$V(\boldsymbol{x}_1,\dots,\boldsymbol{x}_i,\dots,\boldsymbol{x}_n)= V(\boldsymbol{x}_1,\dots,\boldsymbol{y}_i,\dots,\boldsymbol{x}_n)+V(\boldsymbol{x}_1,\dots,\boldsymbol{z}_i,\dots,\boldsymbol{x}_n)$，其中$\boldsymbol{x}_i=\boldsymbol{y}_i+\boldsymbol{z}_i$；这点需要稍加验证，但它的确是正确的。
> 
> 
> 
> 4、只要有两个向量重合，那么体积自然为0，即$V(\dots,\boldsymbol{x},\dots,\boldsymbol{x},\dots)=0$；比如在三维空间中的一个立体，有两条边重合，那么说明这个立体已经压缩为一个面了，面的体积自然为0。
> 
> 
> 
> 5、由单位矩阵$I$构成的平行$n$维体是一个$n$维的单位立方体，它的体积自然是1，即$V(I)=1$。

比较行列式和体积的性质，可以发现它们是完全相同的，所以在第一卦限中的平行$n$维体的体积就是对应矩阵的行列式！如果将其放到所有卦限中，那只不过是体积概念的推广（允许为负数）。因此，我们不妨这样定义：<strong>体积就是行列式</strong>。

<u>事实上，负体积的引入具有重要意义，它是现在的“外微分”的基础之一。外微分一个典型的用处是它可以把高斯积分公式、斯托克斯积分公式等统一起来。它使微分的理论和形式更完整统一。</u>
