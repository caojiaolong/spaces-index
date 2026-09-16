# 行列式的导数

> 作者：苏剑林 · 科学空间 · 2014-02-28
>
> 原文：<https://spaces.ac.cn/archives/2383>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在讨论曲线坐标系的积分时，通常都会出现行列式这个东西，作为“体积元”的因子。在广义相对论中，爱因斯坦场方程的作用量就带有度规的行列式，而在对其进行变分时，自然也就涉及到了行列式的求导问题。我参考了朗道的《场论》以及《数理物理基础\-\-物理需用线性高等数学导引》，了解到相关结果，遂记录如下。

## 推导

设
\begin{equation}\boldsymbol{A}(t)=\left(a_{ij}(t)\right)_{n\times n}\end{equation}
是一个n阶矩阵，其中每个矩阵元素都是t的函数。其行列式为$|\boldsymbol{A}|$，自然地，考虑
\begin{equation}\frac{d}{dt}|\boldsymbol{A}|\end{equation}
由行列式的基本性质，有
\begin{equation}|\boldsymbol{A}(a_{ij}+\varepsilon)|-|\boldsymbol{A}|=\varepsilon A_{ij}\end{equation}
其中$|\boldsymbol{A}(a_{ij}+\varepsilon)|$是将矩阵$\boldsymbol{A}$的$a_{ij}$换成$a_{ij}+\varepsilon$后的行列式的值，而$A_{ij}$是行列式$|\boldsymbol{A}|$关于$a_{ij}$的代数余子式。上式给出
\begin{equation}\frac{\partial |\boldsymbol{A}|}{\partial a_{ij}}= A_{ij}\end{equation}
也就是说，代数余子式可以表示为行列式的偏导数。

那么
\begin{equation}\frac{d}{dt}|\boldsymbol{A}|=\sum_{i}\sum_{j}\frac{\partial |\boldsymbol{A}|}{\partial a_{ij}}\frac{d a_{ij}}{dt}=\sum_{i}\sum_{j} A_{ij}\frac{d a_{ij}}{dt}\end{equation}
（为了得出第一个等式，只需要给矩阵$\boldsymbol{A}$的每个元素都增加一个无穷小量，然后把增量后的矩阵的行列式展开，保留一阶无穷小项。）

所以
\begin{equation}\frac{d}{dt}|\boldsymbol{A}|=|\boldsymbol{A}|\sum_{i}\sum_{j} \frac{A_{ij}}{|\boldsymbol{A}|}\frac{d a_{ij}}{dt}\end{equation}
或许可以重写一下，会使得过程更加清楚
\begin{equation}\frac{d}{dt}|\boldsymbol{A}|=|\boldsymbol{A}|\sum_{j=k}\sum_{i} \frac{A_{ij}}{|\boldsymbol{A}|}\frac{d a_{ik}}{dt}\end{equation}
其中
\begin{equation}\frac{A_{ji}}{|\boldsymbol{A}|}\end{equation}
正好是矩阵$\boldsymbol{A}$的逆阵$\boldsymbol{A}^{-1}$的第$(i,j)$个元素$(\boldsymbol{A}^{-1})_{ij}$，而$\frac{d a_{ij}}{dt}$则是矩阵$\frac{d \boldsymbol{A}}{dt}$的元素。第一次求和即把两个矩阵相乘，得
\begin{equation}\left(\boldsymbol{A}^{-1}\frac{d \boldsymbol{A}}{dt}\right)_{jk}=\sum_{i} (\boldsymbol{A}^{-1})_{ji}\left(\frac{d\boldsymbol{A}}{dt}\right)_{ik}=\sum_{i} \frac{A_{ij}}{|\boldsymbol{A}|}\frac{d a_{ik}}{dt}\end{equation}
而第二次求和则相当于取矩阵的迹，所以
\begin{equation}\frac{d}{dt}|\boldsymbol{A}|=|\boldsymbol{A}| \text{Tr}\left(\boldsymbol{A}^{-1}\frac{d \boldsymbol{A}}{dt}\right)\end{equation}
用张量分析中的符号，则更加简单了，记$g=\det(g_{\mu\nu}),g_{\mu s} g^{s\nu}=\delta_{\mu}^{\nu}$，则
\begin{equation}dg=g g^{\mu\nu} dg_{\mu\nu}\end{equation}

## 例子

作为一个简单的应用，我们求
\begin{equation}\frac{d}{dt}|\boldsymbol{I}+t\boldsymbol{A}|\end{equation}
其中$\boldsymbol{I}$是单位矩阵，$\boldsymbol{A}$对同样大小的方阵。代入公式得到
\begin{equation}\frac{d}{dt}|\boldsymbol{I}+t\boldsymbol{A}|=|\boldsymbol{I}+t\boldsymbol{A}|\text{Tr}\big[(\boldsymbol{I}+t\boldsymbol{A})^{-1}\boldsymbol{A}\big]\end{equation}
那么
\begin{equation}\left(\frac{d}{dt}|\boldsymbol{I}+t\boldsymbol{A}|\right)_{t=0}=\text{Tr}\,\boldsymbol{A}\end{equation}
那么根据泰勒展开，当$|t|$充分小时，我们就有近似
\begin{equation}|\boldsymbol{I}+t\boldsymbol{A}|\approx 1 + t\times\text{Tr}\,\boldsymbol{A}\end{equation}
