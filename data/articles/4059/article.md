# 【外微分浅谈】4\. 微分不微

> 作者：苏剑林 · 科学空间 · 2016-11-05
>
> 原文：<https://spaces.ac.cn/archives/4059>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### 外微分

向量的外积一般只定义于不超过3维的空间。为了在更高维空间中使用反对称运算，我们需要下面描述的微分形式与外微分。

我们知道，任意$x$的函数的微分都可以写成$dx^{\mu}$的线性组合，在这里，各$dx^{\mu}$实则上扮演了一个基的角色，因此，我们不妨把$dx^{\mu}$看成是一组基，并且把任意函数称为微分0形式，而诸如$\omega_{\mu}dx^{\mu}$的式子，称为微分1形式。

在$dx^{\mu}$这组基之上，我们定义外积$\land$，即有反对称的运算$dx^{\mu}\land dx^{\nu}$，并且把诸如$\omega_{\mu\nu}dx^{\mu}\land dx^{\nu}$的式子，称为微分2形式。注意到这是$n$维空间中的外积，$dx^{\mu}\land dx^{\nu}$事实上是一个新空间的基，而不能用$dx^{\mu}$的线性组合来表示。

接着，允许$\land$重复执行，即允许$dx^{\mu}\land dx^{\nu}\land dx^{\lambda}$，并且称诸如$\omega_{\mu\nu\lambda}dx^{\mu}\land dx^{\nu}\land dx^{\lambda}$的式子为微分3形式。相应地，可以定义一般的微分$p$形式。至于它的几何意义，我们后面再谈。

最后，定义一个外微分算符$d$，它允许我们从一个微分$p$形式，产生一个微分$p+1$形式
$$\begin{aligned}&d\left(\omega_{\mu_1 \mu_2 \dots \mu_p} dx^{\mu_1}\land dx^{\mu_2} \land \dots\land dx^{\mu_p}\right)\\
=&\frac{\partial \omega_{\mu_1 \mu_2 \dots \mu_p}}{\partial x^{\mu_{1+1}}} dx^{\mu_{p+1}}\land dx^{\mu_1}\land dx^{\mu_2} \land \dots\land dx^{\mu_p}\end{aligned} \tag{25} $$
事实上，这个算符$d$形式上跟普通的微分算符是一致的，只不过允许重复执行。但是，我们不难证明：对于任意微分形式$\omega$，均有
$$d^2 \omega = 0 \tag{26} $$
因此，外微分算符最多也就是两次作用于微分形式。此外，下面的恒等式也是不难证明的：若$\alpha,\beta$分别是微分$p,q$形式，则
$$d(\alpha\land \beta)=d\alpha\land \beta + (-1)^p \alpha\land d\beta \tag{27} $$
$(-1)^p$的出现，正是由于反对称性。虽然说是外“微”分，可内涵一点也不“微”。

### 立竿见影的应用

我们知道行列式可以用来判断$n$个$n$维向量是否线性相关。但是$k$个呢？

外积可以帮助我们！考虑$k$个向量$\alpha_{\mu}^1,\alpha_{\mu}^2,\dots,\alpha_{\mu}^k$，可以依次构造微分形式$\alpha_{\mu}^1 dx^{\mu},\alpha_{\mu}^2 dx^{\mu},\dots,\alpha_{\mu}^k dx^{\mu}$，然后考虑外积
$$(\alpha_{\mu}^1 dx^{\mu}) \land (\alpha_{\mu}^2 dx^{\mu}) \land \dots \land (\alpha_{\mu}^k dx^{\mu}) \tag{28} $$
如果这$k$个向量线性相关，也就是其中一个能表示为剩下的$k-1$个的线性组合，比如假设
$$\alpha_{\mu}^1 = \sum_{i=2}^{k} b_i \alpha_{\mu}^i \tag{29} $$
那么
$$\alpha_{\mu}^1 dx^{\mu} = \sum_{i=2}^{k} b_i \alpha_{\mu}^i dx^{\mu} \tag{30} $$
这样这$k$个微分形式的外积必然为0，其逆命题也成立。也就是说，$k$个向量线性相关，当且仅当
$$(\alpha_{\mu}^1 dx^{\mu}) \land (\alpha_{\mu}^2 dx^{\mu}) \land \dots \land (\alpha_{\mu}^k dx^{\mu})=0 \tag{31} $$

当然，严格来讲这是反对称运算的结果，可见反对称性质的内涵相当丰富。
