# 构造一个显式的、总是可逆的矩阵

> 作者：苏剑林 · 科学空间 · 2019-03-01
>
> 原文：<https://spaces.ac.cn/archives/6407>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从[《恒等式 det(exp(A)) = exp(Tr(A)) 赏析》](<https://spaces.ac.cn/archives/6377>)一文我们得到矩阵$\exp(\boldsymbol{A})$总是可逆的，它的逆就是$\exp(-\boldsymbol{A})$。问题是$\exp(\boldsymbol{A})$只是一个理论定义，单纯这样写没有什么价值，因为它要把每个$\boldsymbol{A}^n$都算出来。

有没有什么具体的例子呢？有，本文来构造一个显式的、总是可逆的矩阵。

其实思路非常简单，假设$\boldsymbol{x},\boldsymbol{y}$是两个$k$维列向量，那么$\boldsymbol{x}\boldsymbol{y}^{\top}$就是一个$k\times k$的矩阵，我们就来考虑
\begin{equation}\begin{aligned}\exp\left(\boldsymbol{x}\boldsymbol{y}^{\top}\right)=&\sum_{n=0}^{\infty}\frac{\left(\boldsymbol{x}\boldsymbol{y}^{\top}\right)^n}{n!}\\
=&\boldsymbol{I}+\boldsymbol{x}\boldsymbol{y}^{\top}+\frac{\boldsymbol{x}\boldsymbol{y}^{\top}\boldsymbol{x}\boldsymbol{y}^{\top}}{2}+\frac{\boldsymbol{x}\boldsymbol{y}^{\top}\boldsymbol{x}\boldsymbol{y}^{\top}\boldsymbol{x}\boldsymbol{y}^{\top}}{6}+\dots\end{aligned}\end{equation}
注意到
\begin{equation}\boldsymbol{y}^{\top}\boldsymbol{x}=\langle \boldsymbol{x},\boldsymbol{y}\rangle\end{equation}
实际上就只是个标量，因此可以继续化简
\begin{equation}\begin{aligned}\exp\left(\boldsymbol{x}\boldsymbol{y}^{\top}\right)=&\boldsymbol{I}+\boldsymbol{x}\boldsymbol{y}^{\top}\left(1+\frac{\langle \boldsymbol{x},\boldsymbol{y}\rangle}{2}+\frac{\langle \boldsymbol{x},\boldsymbol{y}\rangle^2}{6}+\dots\right)\\
=&\boldsymbol{I}+\boldsymbol{x}\boldsymbol{y}^{\top}\left(\frac{e^{\langle \boldsymbol{x},\boldsymbol{y}\rangle}-1}{\langle \boldsymbol{x},\boldsymbol{y}\rangle}\right)
\end{aligned}\end{equation}
现在这个矩阵就很具体了，它可以很简单地算出来，因为它只涉及到标量的指数运算。括号里的$(e^x - 1)/x$有个可去的间断点$x=0$，当$x=0$时它的值为1。

根据恒等式$\det(\exp(\boldsymbol{A})) = \exp(\text{Tr}(\boldsymbol{A}))$，这个矩阵的行列式为
\begin{equation}\det\left(\exp\left(\boldsymbol{x}\boldsymbol{y}^{\top}\right)\right)= e^{\langle \boldsymbol{x},\boldsymbol{y}\rangle}\end{equation}
它的逆矩阵为
\begin{equation}\exp\left(-\boldsymbol{x}\boldsymbol{y}^{\top}\right)=\boldsymbol{I}-\boldsymbol{x}\boldsymbol{y}^{\top}\left(\frac{1 - e^{-\langle \boldsymbol{x},\boldsymbol{y}\rangle}}{\langle \boldsymbol{x},\boldsymbol{y}\rangle}\right)\end{equation}
也是一个显式的结果。

当然，一般的矩阵有$k^2$个独立参数，而这里两个向量构造出来的矩阵只有$2k$个参数，就表达能力来说应该是不足够的。为了增强表达能力，可以考虑多个这样的矩阵乘起来：
\begin{equation}\exp\left(\boldsymbol{x}_1\boldsymbol{y}_1^{\top}\right)\exp\left(\boldsymbol{x}_2\boldsymbol{y}_2^{\top}\right)\exp\left(\boldsymbol{x}_3\boldsymbol{y}_3^{\top}\right)\dots\end{equation}
注意它一般不等于
\begin{equation}\exp\left(\boldsymbol{x}_1\boldsymbol{y}_1^{\top}+\boldsymbol{x}_2\boldsymbol{y}_2^{\top}+\boldsymbol{x}_3\boldsymbol{y}_3^{\top}+\dots\right)\end{equation}

好了，推了这么多，那它究竟有什么用？

呃\.\.\.我也不知道它有什么用，权当欣赏吧。

（其实我的初衷是想要通过构造法来直接构造出可逆的神经网络出来。按照这个现成的矩阵，构造出可逆的全连接网络并不困难，但我没想好怎么推广到卷积层。等我搞定了这个推广，再来继续讨论这个矩阵的应用吧～）
