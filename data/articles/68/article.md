# 无穷级数求和的积分审敛法

> 作者：苏剑林 · 科学空间 · 2009-08-12
>
> 原文：<https://spaces.ac.cn/archives/68>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这是我研究级数求和的时候的一个猜测，现在已经发现为正确的。

存在级数$\sum_{x=1}^{\infty} f(x)$，若有

$\lim_{x -> \infty } \int f(x)dx -> \infty $，则该级数发散。

如果$\lim_{x -> \infty } \int f(x)dx $收敛，则该级数收敛。

<strong>例如：</strong>

级数$\sum_{x=1}^{\infty} 1/x$，由于$\int (1/x)dx=ln x$，$\lim_{x -> \infty}ln x -> \infty$，因此该级数发散。

级数$\sum_{x=1}^{\infty} 1/{x^2}$，由于$\int (1/{x^2})dx=-1/x$，$\lim_{x -> \infty}-1/x -> 0$，因此该级数收敛。

<strong>原来这个结论已经存在的，现在以肯定形式给出来</strong>
