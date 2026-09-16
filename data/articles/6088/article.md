# 变分自编码器 = 最小化先验分布 \+ 最大化互信息

> 作者：苏剑林 · 科学空间 · 2018-10-10
>
> 原文：<https://spaces.ac.cn/archives/6088>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这篇文章很简短，主要描述的是一个很有用、也不复杂、但是我居然这么久才发现的事实～

在[《深度学习的互信息：无监督提取特征》](<https://spaces.ac.cn/archives/6024>)一文中，我们通过先验分布和最大化互信息两个loss的加权组合来得到Deep INFOMAX模型最后的loss。在那篇文章中，虽然把故事讲完了，但是某种意义上来说，那只是个拼凑的loss。而本文则要证明那个loss可以由变分自编码器自然地导出来。

## 过程

不厌其烦地重复一下，变分自编码器（VAE）需要优化的loss是
\begin{equation}\begin{aligned}&KL(\tilde{p}(x)p(z|x)\Vert q(z)q(x|z))\\
=&\iint \tilde{p}(x)p(z|x)\log \frac{\tilde{p}(x)p(z|x)}{q(x|z)q(z)} dzdx\end{aligned}\end{equation}
相关的论述在本博客已经出现多次了。VAE中既包含编码器，又包含解码器，如果我们只需要编码特征，那么再训练一个解码器就显得很累赘了。所以重点是怎么将解码器去掉。

其实再简单不过了，把VAE的loss分开两部分
\begin{equation}\begin{aligned}&KL(\tilde{p}(x)p(z|x)\Vert q(z)q(x|z))\\
=&\iint \tilde{p}(x)p(z|x)\log \frac{p(z|x)}{q(z)} dzdx-\iint \tilde{p}(x)p(z|x)\log \frac{q(x|z)}{\tilde{p}(x)} dzdx\end{aligned}\end{equation}
第一项是先验分布的KL散度，第二项的$\log \frac{q(x|z)}{\tilde{p}(x)}$其实不也就是$x,z$的点互信息吗？假如$q(x|z)$具有无限的拟合能力，最终必然也会有$\tilde{p}(x)p(z|x) = q(x|z)p(z)$（贝叶斯公式），所以第二项也就是
\begin{equation}KL(q(x|z)p(z)\Vert \tilde{p}(x)p(z))=KL(\tilde{p}(x)p(z|x)\Vert \tilde{p}(x)p(z))\end{equation}
就是$x,z$两个随机变量的互信息了，前面的负号意味着我们要最大化互信息。

剩下的处理过程就跟[《深度学习的互信息：无监督提取特征》](<https://spaces.ac.cn/archives/6024>)一样了，略。

## 结语

开头已经说了，这篇文章会很简短，没有什么内容。主要目的就是给出变分自编码器的loss的新理解（最小化先验分布 \+ 最大化互信息），然后就可以自然而言地导出Deep INFOMAX的loss。

如果我还没有写[《深度学习的互信息：无监督提取特征》](<https://spaces.ac.cn/archives/6024>)，那么我肯定会用这个出发点来讲解Deep INFOMAX，不过既然那篇文章都写了好几天了，所以只好另开这个简短的小文，来补充说明一下～
