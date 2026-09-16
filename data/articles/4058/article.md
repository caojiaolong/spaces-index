# 【外微分浅谈】3\. 正交标架

> 作者：苏剑林 · 科学空间 · 2016-11-05
>
> 原文：<https://spaces.ac.cn/archives/4058>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

众所周知，要掌握黎曼几何，需要强烈的几何直观感。但除此之外，用分量语言描述的黎曼几何，也需要很好的分析能力才能梳理清楚，因为有$N$多的指标在表示着分量和求和，咋看上去处处皆指标。这种繁琐的分量语言并不总讨人喜欢，甚至在不少地方是声名狼籍的。

在分量的语言中，我们本质上可以在局部建立任意形式的坐标系，也就是采用任意形式的基底$\{\boldsymbol{e}_{\mu}\}$，或者说自然标架。但不可否认，在正交标架（标准正交基）之下，很多方程会简单不少，并且得益于我们对欧氏空间的熟练，我们对正交标架下的研究可能会更有感觉。因此，如果条件允许的话，我们应当使用正交标架$\{\hat{\boldsymbol{e}}_{\mu}\}$，哪怕是活动的，这里我们用$\hat{}$标记正交标架。

比如，我们有微元
$$d\boldsymbol{r} = \boldsymbol{e}_{\mu}dx^{\mu} \tag{12} $$
是在一般标架下测量的，那么就可以得到黎曼度量
$$ds^2 = \langle d\boldsymbol{r}, d\boldsymbol{r}\rangle= g_{\mu\nu}dx^{\mu} dx^{\nu} \tag{13} $$
其中
$$g_{\mu\nu} = \langle \boldsymbol{e}_{\mu}, \boldsymbol{e}_{\nu}\rangle \tag{14} $$
也许是一个带有复杂函数的矩阵。我们把黎曼度量写成矩阵形式
$$g_{\mu\nu}dx^{\mu} dx^{\nu}=d\boldsymbol{x}^T \boldsymbol{g}d\boldsymbol{x} \tag{15} $$
然后试图做这样的分解
$$\boldsymbol{g}=\boldsymbol{h}^T \boldsymbol{\eta}\boldsymbol{h} \tag{16} $$
$\boldsymbol{h},\boldsymbol{\eta}$都是跟$\boldsymbol{g}$同一形状的矩阵，那么
$$ds^2 = (\boldsymbol{h}d\boldsymbol{x})^T\boldsymbol{\eta}(\boldsymbol{h}d\boldsymbol{x}) \tag{17} $$
写成分量语言是
$$ds^2 = \eta_{\mu\nu}(h_{\alpha}^{\mu} dx^{\alpha} )(h_{\beta}^{\nu} dx^{\beta}) \tag{18} $$
我们记
$$\omega^{\mu} = h_{\alpha}^{\mu} dx^{\alpha} \tag{19} $$
事实上$h_{\alpha}^{\mu}$就是一个变换矩阵，将原来的任意标架$\{\boldsymbol{e}_{\mu}\}$变换到了（活动）正交标架$\{\hat{\boldsymbol{e}}_{\mu}\}$，即
$$\hat{\boldsymbol{e}}_{\mu} = \boldsymbol{e}_{\alpha}(h^{-1})^{\alpha}_{\mu} , \quad \boldsymbol{e}_{\mu} = \hat{\boldsymbol{e}}_{\alpha} h^{\alpha}_{\mu} \tag{20} $$
此时我们有
$$d\boldsymbol{r} = \boldsymbol{e}_{\mu} dx^{\mu} =\hat{\boldsymbol{e}}_{\mu} \omega^{\mu} \tag{21} $$
这表明正交标架中的$\omega^{\mu}$相当于一般标架中的$dx^{\mu}$，以及
$$ds^2 = \eta_{\mu\nu} \omega^{\mu} \omega^{\nu} \tag{22} $$

上式表明正交标架有助于简化黎曼度量，现在度规张量是更为简单的$\eta_{\mu\nu}$。需要指出的是，理想的分解是$\boldsymbol{\eta}$为单位阵，但我们如果考虑一般的黎曼度量（尤其是广义相对论中的），并且仅限于实数范围内的话，这个理想并不一定能够达到，比如简单的$\boldsymbol{g}=\begin{pmatrix} -1 & 0 \\ 0 & 1\end{pmatrix}$就做不到，因此我们只希望$\boldsymbol{\eta}$尽可能简单，比如说是常数对角矩阵，但不要求它一定是单位阵。这样的分解总是可以做到的，尤其是在很多实用的情形下，$\boldsymbol{g}$一个对角阵，这时候就相当容易实现了。因此，这里假设$\eta_{\mu\nu}$是一个对角矩阵，其对角元素是1或\-1。

接着我们写出
$$ds^2 = \langle d\boldsymbol{r},d\boldsymbol{r}\rangle=\langle \hat{\boldsymbol{e}}_{\mu},\hat{\boldsymbol{e}}_{\nu}\rangle \omega^{\mu}\omega^{\nu} \tag{23} $$
也就是
$$\eta_{\mu\nu} = \langle \hat{\boldsymbol{e}}_{\mu},\hat{\boldsymbol{e}}_{\nu}\rangle \tag{24} $$
这里$\hat{\boldsymbol{e}}$就是在上式意义下的“正交标架”了。
