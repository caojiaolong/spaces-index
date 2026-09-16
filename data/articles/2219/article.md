# 高维空间的叉积及其几何意义

> 作者：苏剑林 · 科学空间 · 2013-12-26
>
> 原文：<https://spaces.ac.cn/archives/2219>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

向量之间的运算有点积和叉积（Cross Product，向量积、外积），其中点积是比较简单的，而且很容易推广到高维；但是叉积不同，一般来说它只不过是三维空间中的东西。叉积的难以推广在于它的多重含义性，如果将向量及其叉积放到张量里边来看（这属于微分形式的内容），那么三维以上的向量叉积是不存在的；但是如果只是把叉积看成是“由两个向量生成第三个与其正交的向量”的工具的话，那么叉积也是可以高维推广的，而且推广的技巧非常巧妙，与三维空间的叉积也非常相似。

## 回顾三维空间

为了推广三维空间的叉积，首先回顾三维空间的叉积来源是有益的。叉积起源于四元数乘法，但是从目的性来讲，我们希望构造一个向量$\boldsymbol{w}=(w_1,w_2,w_3)$，使得它与已知的两个不共线的向量$\boldsymbol{u}=(u_1,u_2,u_3),\boldsymbol{v}=(v_1,v_2,v_3)$垂直（正交）。从普适性的角度来讲，我们还希望构造出来的向量没有任何“奇点”，为此，我们只用乘法构造。至于叉积的几何意义，则是后话，毕竟，先达到基本的目的再说。

这里，为了构造出这样的向量，行列式发挥着巨大作用！我们考虑行列式
$$\begin{vmatrix}
u_1 & v_1 & w_1 \\
u_2 & v_2 & w_2 \\
u_3 & v_3 & w_3
\end{vmatrix}$$

由行列式的性质可以知道，如果$(w_1,w_2,w_3)=(u_1,u_2,u_3)$，那么行列式就为0，展开来就是：
$$u_1 \begin{vmatrix} u_2 & v_2\\u_3 &v_3\end{vmatrix}-u_2 \begin{vmatrix} u_1 & v_1\\u_3 &v_3\end{vmatrix}+u_3 \begin{vmatrix} u_1 & v_1\\u_2 &v_2\end{vmatrix}=0$$

同理，如果$(w_1,w_2,w_3)=(v_1,v_2,v_3)$，那么行列式也为0，展开来得到
$$v_1 \begin{vmatrix} u_2 & v_2\\u_3 &v_3\end{vmatrix}-v_2 \begin{vmatrix} u_1 & v_1\\u_3 &v_3\end{vmatrix}+v_3 \begin{vmatrix} u_1 & v_1\\u_2 &v_2\end{vmatrix}=0$$

这样看来，向量
$$\left(\begin{vmatrix} u_2 & v_2\\u_3 &v_3\end{vmatrix},-\begin{vmatrix} u_1 & v_1\\u_3 &v_3\end{vmatrix}, \begin{vmatrix} u_1 & v_1\\u_2 &v_2\end{vmatrix}\right)$$
自动地与向量$\boldsymbol{u}=(u_1,u_2,u_3),\boldsymbol{v}=(v_1,v_2,v_3)$垂直。而上式就是我们目前定义的向量叉积，因此叉积来源可见一斑。

在理论分析中，我们一般将叉积写成
$$\boldsymbol{u}\times\boldsymbol{v}=\begin{vmatrix}
u_1 & v_1 & \boldsymbol{e}_1 \\
u_2 & v_2 & \boldsymbol{e}_2 \\
u_3 & v_3 & \boldsymbol{e}_3
\end{vmatrix}$$

其中$\boldsymbol{e}_1,\boldsymbol{e}_2,\boldsymbol{e}_3$是三维空间的基。

## 高维的叉积

有了上面的基础后，高维空间的叉积也就不难定义了，比如说四维空间中的三个向量$\boldsymbol{x}=(x_1,x_2,x_3,x_4),\boldsymbol{y}=(y_1,y_2,y_3,y_4),\boldsymbol{z}=(z_1,z_2,z_3,z_4)$，它的叉积可以定义为
$$Cross(\boldsymbol{x},\boldsymbol{y},\boldsymbol{z})=\begin{vmatrix}
x_1 & y_1 & z_1 &\boldsymbol{e}_1 \\
x_2 & y_2 & z_2 &\boldsymbol{e}_2 \\
x_3 & y_3 & z_3 &\boldsymbol{e}_3 \\
x_4 & y_4 & z_4 &\boldsymbol{e}_4
\end{vmatrix}$$
高维的空间向量也可以类似定义。

现在来考虑它的模的几何意义。记$Cross(\boldsymbol{x},\boldsymbol{y},\boldsymbol{z})=\boldsymbol{w}=(\omega_1,\omega_2,\omega_3,\omega_4)$，其中
$$\begin{aligned}\omega_1=\begin{vmatrix}
x_2 & y_2 & z_2 \\
x_3 & y_3 & z_3 \\
x_4 & y_4 & z_4
\end{vmatrix},\omega_2=-\begin{vmatrix}
x_1 & y_1 & z_1 \\
x_3 & y_3 & z_3 \\
x_4 & y_4 & z_4
\end{vmatrix}&,\\
\omega_3=\begin{vmatrix}
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2 \\
x_4 & y_4 & z_4
\end{vmatrix},\omega_4=-\begin{vmatrix}
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2 \\
x_3 & y_3 & z_3
\end{vmatrix}&.
\end{aligned}$$

那么行列式
$$\begin{vmatrix}
x_1 & y_1 & z_1 &\omega_1 \\
x_2 & y_2 & z_2 &\omega_2 \\
x_3 & y_3 & z_3 &\omega_3 \\
x_4 & y_4 & z_4 &\omega_4
\end{vmatrix}$$
就表示$\boldsymbol{x},\boldsymbol{y},\boldsymbol{z},\boldsymbol{w}$所组成的平行四维体的四维超体积。而且由于$\boldsymbol{w}$垂直于另外三个向量，那么这个平行四维体其实就是四维空间的一个“四维直棱柱”（想像三维的直棱柱），它的“底部”是由$\boldsymbol{x},\boldsymbol{y},\boldsymbol{z}$所组成的三维空间的平行六面体。根据“体积=底面积乘以高”的类比，上述行列式应该等于“平行六面体的体积乘以$\boldsymbol{w}$的模长”。<u>当模长为1时，4维柱体的超体积在数值上就等于三维平行六面体的体积。</u>于是行列式
$$\begin{vmatrix}
x_1 & y_1 & z_1 &\frac{\omega_1}{|\boldsymbol{w}|} \\
x_2 & y_2 & z_2 &\frac{\omega_2}{|\boldsymbol{w}|} \\
x_3 & y_3 & z_3 &\frac{\omega_3}{|\boldsymbol{w}|} \\
x_4 & y_4 & z_4 &\frac{\omega_4}{|\boldsymbol{w}|}
\end{vmatrix}$$
就等于由$\boldsymbol{x},\boldsymbol{y},\boldsymbol{z}$所组成的三维空间的平行六面体的体积，对最后一列展开它之后就得到$|\boldsymbol{w}|$。这就是模的几何意义，它和三维空间的叉积类似。更高维空间叉积及其模长的几何意义也可以相应类比，n维空间的(n\-1)个线性无关的向量可以定义叉积，它的模长就是原来(n\-1)个向量所构成的(n\-1)维体的超体积。

回首一看，就可以发现，这个结果甚至对于2维空间都是成立的。由向量$(a,b)$自构造一个叉积，即垂直于它的向量，只需要考虑行列式
$$\begin{vmatrix}
a &\boldsymbol{e}_1 \\
b &\boldsymbol{e}_2 \\
\end{vmatrix}=-b \boldsymbol{e}_1+a\boldsymbol{e}_2=(-b,a)$$

可见，叉积为生成垂直向量提供了最自然的描述。
