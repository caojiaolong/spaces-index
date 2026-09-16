# 哈勃定律——宇宙各向同性的体现

> 作者：苏剑林 · 科学空间 · 2010-10-04
>
> 原文：<https://spaces.ac.cn/archives/964>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![universe_mystery_expand](<https://spaces.ac.cn/usr/uploads/2010/10/1799387840.jpg>)](<https://spaces.ac.cn/attachment/967/>)

universe\_mystery\_expand

> 1929年哈勃(Edwin Hubble)对河外星系的视向速度与距离的关系进行了研究。当时只有46个河外星系的视向速度可以利用，而其中仅有24个有推算出的距离，哈勃得出了视向速度与距离之间大致的线性正比关系。

不少宇宙学的书籍中都提到了标题，那么，为什么哈勃定律是宇宙各向同性的体现？或者说为什么宇宙各向同性就必然导致哈勃定律？

首先我们得需要了解一下宇宙学原理，它告诉我们宇宙在大尺度范围是均匀的、各向同性的。基于这个原理，我们会得到一些很奇怪的东西，如宇宙中的每一点都是宇宙的中心。另外，我们还可以得到：宇宙的（整体）运动情况在每一个方向都应该取相同的形式。

[![宇宙膨胀-简单图示](<https://spaces.ac.cn/usr/uploads/2010/10/3945159214.png>)](<https://spaces.ac.cn/attachment/965/>)

宇宙膨胀\-简单图示

就让我们以这个宇宙学原理来研究宇宙的膨胀问题吧。假设在C处的观测者看到A天体的向径为$\vec{r}$，速度为$\dot{\vec{r}}=f(\vec{r})$；看到B天体的向径为$\vec{R}$，根据各向同性原理，B的速度$\dot{\vec{R}}$也是关于$\vec{R}$的$f()$函数，因此也有$\dot{\vec{R}}=f(\vec{R})$。

现在换B点的观测者看C天体了。显然向径为$\vec{R}-\vec{r}$，速度为$\frac{d(\vec{R}-\vec{r})}{dt}=\dot{\vec{R}}-\dot{\vec{r}}$，由于各向同性，从B点和A点观测C的运动情况应该是一样的，因此速度$\dot{\vec{R}}-\dot{\vec{r}}$也是关于$\vec{R}-\vec{r}$的$f()$函数，即$\dot{\vec{R}}-\dot{\vec{r}}=f(\vec{R}-\vec{r})=f(\vec{R})-f(\vec{r})$

接下来我们将看到，满足上式的，只有正比例函数。其实这等价于解数学中的函数方程$f(x-y)=f(x)-f(y)$，取x=y=0，立即得到f(0)=0，取x=a\+1,y=1得$f(a+1)-f(a)=f(1)$，这类似与我们所学的等差数列，于是可以写出$f(x)=(x-1)\cdot f(1)+f(1)=f(1)\cdot x=kx$，其中k是常数。虽然结果的得出只是针对x是自然数，但是单从自然数这一“片面”已经得到了唯一的结果。也就是说，我们证明了满足$f(x-y)=f(x)-f(y)$的只有正比例函数。同样，满足$f(\vec{R}-\vec{r})=f(\vec{R})-f(\vec{r})$的就只有正比例函数$\vec{v}(\vec{r})=H_0 \vec{r}$了，于是哈勃定律是必然的。

[![各向同性膨胀](<https://spaces.ac.cn/usr/uploads/2010/10/725727941.jpg>)](<https://spaces.ac.cn/attachment/966/>)

各向同性膨胀
