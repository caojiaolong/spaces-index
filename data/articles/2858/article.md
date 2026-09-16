# 从费马大定理谈起（六）：n=4（2）

> 作者：苏剑林 · 科学空间 · 2014-08-19
>
> 原文：<https://spaces.ac.cn/archives/2858>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[上一篇文章](<https://spaces.ac.cn/archives/2831/>)中，笔者提到似乎证明n=4时必须要证明$x^4+y^4=z^2$无解而不能只证明$x^4+y^4=z^4$无解。不过，在今天中午研究的时候，笔者发现了另外一个n=4的证明，它同样是在$\mathbb{Z}[i]$中，但是，证明的则是指数全是4的形式，但是，又不单单是$x^4+y^4=z^4$的形式，而是$\varepsilon x^4+y^4=z^4$，$\varepsilon$是单位数。这个证明过程，我觉得应该更接近n等于其他奇素数时的证明，遂补充了这篇文章，供大家参考。读者可以对比着上一篇文章进行比较阅读。

## 引理

用$\varepsilon_1,\varepsilon_2,\varepsilon_3,\varepsilon$表示$\mathbb{Z}[i]$中的单位数，下面先证明

> 如果方程$\varepsilon_1 x'^4 +\varepsilon_2 y'^4+\varepsilon_3 z'^4=0$在$\mathbb{Z}[i]$中有全不为0的解，那么在经过适当的化简和整理之后，方程必有形式$\varepsilon x^4+y^4=z^4$，其中$(x,y,z)$是$(x',y',z')$的某个置换，$\xi^2|x$。

证明过程跟上一篇文章是类似的。首先证明$\xi|x'y'z'$，若否，则有
$$\begin{aligned}x'^4 &\equiv 1(\bmod\, 8)\\
y'^4 &\equiv 1(\bmod\, 8)\\
z'^4 &\equiv 1(\bmod\, 8)
\end{aligned}$$
那么就有$8|(\varepsilon_1+\varepsilon_2+\varepsilon_3)$，不管$\varepsilon_1,\varepsilon_2,\varepsilon_3$如何取值都不是成立的。

那么假设$\xi|x'$，并且假设$\xi\nmid y'z'$，那么
$$\begin{aligned}x'^4 &\equiv 0(\bmod\, \xi^4)\\
y'^4 &\equiv 1(\bmod\, 8)\\
z'^4 &\equiv 1(\bmod\, 8)
\end{aligned}$$
由原方程整理得
$$(\varepsilon_1/\varepsilon_2) x'^4 +y'^4+(\varepsilon_3/\varepsilon_2) z'^4=0$$
各项模$\xi^4$（$-8i=\xi^6$），得到
$$\xi^4|(1+\varepsilon_3/\varepsilon_2)$$
于是$\varepsilon_3/\varepsilon_2=-1$，取$\varepsilon_1/\varepsilon_2=\varepsilon,(x',y',z')=(x,y,z)$，即得
$$\varepsilon x^4+y^4=z^4$$
由于$z^4-y^4\equiv x^4\equiv 0(\bmod\,8),\ \xi^6=-8i$，所以至少有$\xi^2|x$。

## 证明

下面假设$(x,y,z)$是$\varepsilon x^4+y^4=z^4,\ \xi^2|x$的一组两两互质的解，并且这组解是$x$所含的$\xi$次数最少的一组解，要注意的是，此时$\varepsilon$是任意的单位数，是不确定的，因此这里最小是遍历所有可能的$\varepsilon$后取$\xi$次数最小的解。此时，记$\xi^m|x,\xi^{m+1}\nmid x,m\geq 2$。

在$\mathbb{Z}[i]$中，可以完全分解：
$$\varepsilon x^4=(z+y)(z-y)(z+yi)(z-yi)$$
注意到
$$\begin{aligned}(z+y)+(z-y)=2z=-i\xi^2 z\\
(z+y)-(z-y)=2y=-i\xi^2 y\\
(z-y)i+(z+yi)=\xi z\\
(z-y)-(z+yi)=-\xi y\\
(z+yi)+(z-yi)=2z=-i\xi^2 z\\
(z+yi)-(z-yi)=2y=-i\xi^2 y
\end{aligned}$$
以上的计算表明，$(z+y),(z-y)$最多有公约数$\xi^2$，$(z-y),(z+yi)$最多有公约数$\xi$，$(z+yi),(z-yi)$最多有公约数$\xi^2$。而左边至少有约数$\xi^8$，因此右边四项之中，至少有一项有约数$\xi^2$，但是，不可能所有项都有约数$\xi^2$，因为这与$(z-y),(z+yi)$最多有公约数$\xi$矛盾。那么，至少有两项的$\xi$次数为1，因为$z+y$有约数$\xi^2$必然推出$z-y$有约数$\xi^2$，$z+yi$有约数$\xi^2$必然推出$z-yi$有约数$\xi^2$，反之亦然，如果只有一项的$\xi$次数为1，便意味着其他三项都是至少是2，矛盾。那么四项之中，有一项的次数至少为4，由此得到，$\xi$在右边的次数分别是$\geq 4,2,1,1$。不失一般性，设
$$\begin{aligned}x&=\xi \chi\\
z+y&=\xi^{4m-4} u'\\
z-y&=\xi^2 v'\\
z+yi&=\xi s'\\
z-yi&=\xi t'
\end{aligned}$$
那么
$$\varepsilon\chi^4=u'v's't'$$
且$u',v',s',t'$两两互质，那么它们都是某个四次方数的伴随，同时它们都不能有约数$\xi$，否则与$\xi^{m+1}\nmid x$矛盾。于是可以设
$$\begin{aligned}u'=\varepsilon_1 u^4,\ v'=\varepsilon_2 v^4\\
s'=\varepsilon_3 s^4,\ t'=\varepsilon_4 t^4
\end{aligned}$$
此时
$$\begin{aligned}2z&=\xi^{4m-4} u^4 +\xi^2 v^4=\xi s^4+\xi t^4\\
2y&=\xi^{4m-4} u^4 -\xi^2 v^4=(\xi s^4-\xi t^4)(-i)
\end{aligned}$$
亦即
$$\begin{aligned}\varepsilon_1 \xi^{4m-5} u^4 +\varepsilon_2 \xi v^4&=\varepsilon_3 s^4+\varepsilon_4 t^4\\
\varepsilon_1\xi^{4m-5} u^4 -\varepsilon_2\xi v^4&=(\varepsilon_3 s^4-\varepsilon_4 t^4)(-i)
\end{aligned}$$
两式左右相加
$$2\varepsilon_1 \xi^{4m-5} u^4=(1-i)\varepsilon_3 s^4+(1+i)\varepsilon_4 t^4$$
各项除以$\xi$，整理得
$$(-i)(\varepsilon_1/\varepsilon_4) \xi^{4m-4} u^4 +i (\varepsilon_3/\varepsilon_4) s^4=t^4$$
由引理得
$$i (\varepsilon_3/\varepsilon_4)=1$$
记$(-i)(\varepsilon_1/\varepsilon_4)=\varepsilon'$，那么
$$\varepsilon' \xi^{4m-4} u^4 +s^4=t^4$$
这说明$(\xi^{m-1} u,s,t)$是单位数取$\varepsilon'$时的一组解，$\xi\nmid u$，但是$m-1 < m$，与假设矛盾，因此方程$\varepsilon x^4+y^4=z^4$在$\mathbb{Z}[i]$内无解。

## 评述

上面的过程绕来绕去，似乎难以捉摸，但这正是揭示了一般的n时的证明规律。其中，最关键的一步时，利用扩展后的数域，对$z^n-y^n$进行完全分解。如果唯一分解定理成立，那么就可以导出矛盾来。同时，基本的同余分析是很必要的，同余分析后的结果，会“恰好地”帮我们把系数凑合好，让我们可以“刚好”可以用费马的无穷下降法导出矛盾。这一切，似乎都是那么巧合，又好像那么必然\.\.\.
