# 当概率遇上复变：从二项分布到泊松分布

> 作者：苏剑林 · 科学空间 · 2015-01-13
>
> 原文：<https://spaces.ac.cn/archives/3188>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

泊松分布，适合于描述单位时间内随机事件发生的次数的概率分布，如某一服务设施在一定时间内受到的服务请求的次数、汽车站台的候客人数等。[<sup>&#91;维基百科&#93;</sup>](<http://zh.wikipedia.org/zh-cn/%E6%B3%8A%E6%9D%BE%E5%88%86%E4%BD%88>)泊松分布也可以作为小概率的二项分布的近似，其推导过程在一般的概率论教材都会讲到。可是一般教材上给出的证明并不是那么让人赏心悦目，如《概率论与数理统计教程》（第二版，茆诗松等编）的第98页就给出的证明过程。那么，哪个证明过程才更让人点赞呢？我认为是利用母函数的证明。

二项分布的母函数为
$$\begin{equation}(q+px)^n,\quad q=1-p\end{equation}$$

当试验次数非常大而概率很小的时候，可以考虑它的近似，此时$\lambda=pn$是一个不算大的数字，$p=\frac{\lambda}{n}$很小，也是根据概率公式
$$\begin{equation}(q+px)^n=\left(1+\frac{\lambda}{n}(x-1)\right)^n\end{equation}$$
由于
$$\begin{equation}\lim_{n\to\infty}\left(1+\frac{x}{n}\right)^n=e^x\end{equation}$$
当$n$相当大但不是无穷大时，上式与$e^x$相差还是很小的，于是有近似
$$\begin{equation}\left(1+\frac{\lambda}{n}(x-1)\right)^n\approx e^{\lambda x-\lambda}\end{equation}$$
这就是泊松分布的母函数。

要注意，$e^{\lambda x-\lambda}|_{x=1}=1$，这是作为概率的母函数必须服从的条件，这是非常巧的。因为我们做了一个近似，近似之前的函数，确实是一个概率的母函数，但是近似之后的函数，还恰好是另一个概率的母函数（而不用作修正），不得不说这是一个非常漂亮的巧合。

直接将其展开为级数，就得到各项的概率
$$\begin{equation}e^{\lambda x-\lambda}=e^{-\lambda}\sum_{k=0}^{\infty}\frac{\lambda^k}{k !}x^k\end{equation}$$
也就是说
$$\begin{equation}P(X=k)=e^{-\lambda}\frac{\lambda^k}{k !}\end{equation}$$
于是我们得到了泊松分布的概率公式。

下面给出一张刚找到的各种概率的联系图，与读者分享。

[![各种概率分布的联系](<https://spaces.ac.cn/usr/uploads/2015/01/2615058286.png>)](<https://spaces.ac.cn/usr/uploads/2015/01/2615058286.png>)

各种概率分布的联系

原图来自：[http://www\.math\.wm\.edu/\~leemis/2008amstat\.pdf](<http://www.math.wm.edu/~leemis/2008amstat.pdf>)
