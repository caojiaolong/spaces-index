# 从费马大定理谈起（九）：n=3

> 作者：苏剑林 · 科学空间 · 2014-09-01
>
> 原文：<https://spaces.ac.cn/archives/2910>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

现在可以开始$n=3$的证明了。在实整数范围内n=3的证明看起来相当复杂，而且跟n=4的证明似乎没有相通之处。然而，如果我们在$\mathbb{Z}[\omega]$中考虑$x^3+y^3+z^3=0$无解的证明，就会跟n=4时有很多类似的地方，而且事实上证明比n=4时简单（要注意在实整数范围内的证明，n=4比n=3简单。费马完成了n=4的证明，但是没完成n=3的证明。）。我想，正是这样的类似之处，才让当初还没有完成证明的数学家拉梅就自信他从这条路可以完成费马大定理的证明。（不过，这自信却是失败的案例：拉梅的路不能完全走通，而沿着这条路走得更远的当属库默，但即便这样，库默也没有证明费马大定理。）

证明跟$n=4$的第二个证明是类似的。我们先往方程中添加一个单位数，然后证明无论单位数是什么，方程在$\mathbb{Z}[\omega]$中都无解。这是一个很妙的技巧，让我们证明了更多的方程无解，但是却用到了更少的步骤。事实上，存在着只证明$x^3+y^3+z^3=0$无解的证明，但需要非常仔细地分析里边的单位数情况，这是相当麻烦的。本证明是我参考了[Fermats last theorem blogspot](<http://fermatslasttheorem.blogspot.com/>)上的证明，然后结合本系列[n=4的第二个证明](<https://spaces.ac.cn/archives/2858/>)，简化而来，主要是减少了对单位数的仔细分析。

## 引理

> 在这篇文章中，用$\varepsilon_1,\varepsilon_2,\varepsilon_3,\varepsilon$表示$\mathbb{Z}[\omega]$中的单位数，并且记$\xi=1-\omega$。那么如果方程$\varepsilon_1 x^3+\varepsilon_2 y^3 +\varepsilon_3 z^3=0,\,\xi|x,\xi\nmid yz$在$\mathbb{Z}[\omega]$中有解，那么方程可以化为
> $$\varepsilon x^3+ y^3 +z^3=0$$

证明很简单，将方程各项都除以$\varepsilon_3$，得到$(\varepsilon_1/\varepsilon_3) x^3+(\varepsilon_2/\varepsilon_3) y^3 + z^3=0$，然后考虑模$\xi$。由于$\xi\nmid yz$，那么$y^3\equiv\pm 1(\bmod\,9),z^3\equiv \pm 1(\bmod\,9)$，注意$9=\xi^4 \omega$，因此考虑模$\xi^3$，则有
$$0\pm (\varepsilon_2/\varepsilon_3)\pm 1\equiv 0(\bmod\,\xi^3)$$
因此$\varepsilon_2/\varepsilon_3=\pm 1$，将$y^3$前面的\-1因子（如果是负号的话）纳入到$y$之中（$(-1)^3=-1$），并且设$\varepsilon_1/\varepsilon_3=\varepsilon$，则有$\varepsilon x^3+ y^3 +z^3=0$。

## 证明

好，下面可以开始证明了。

> 方程$x^3+y^3+z^3=0$在$\mathbb{Z}[\omega]$中无$xyz\neq 0$的解。

如果方程$x^3+y^3+z^3=0$在$\mathbb{Z}[\omega]$中有$xyz\neq 0$的解，那么必有$\xi|xyz$，否则$\xi\nmid x,\xi\nmid y,\xi\nmid z$，这样子就有
$$\begin{aligned}x^3\equiv \pm 1(\bmod\,9)\\
y^3\equiv \pm 1(\bmod\,9)\\
z^3\equiv \pm 1(\bmod\,9)\end{aligned}$$
由此得到$\pm 1\pm 1\pm 1(\bmod\,9)$，不管正负号怎么取，都是不能成立的，因此必有$\xi|xyz$。由于此时$x,y,z$地位均是一样的，不失一般性，设$\xi|x$。完成这个步骤的目的是表明费马大定理在n=3时，也属于$\varepsilon x^3+y^3 +z^3=0,\xi|x$类型的方程，这样就可以进一步进行相应的同余分析。

下面的步骤跟n=4时几乎一样了，甚至更简单。假如某个$\varepsilon x^3+y^3 +z^3=0,\xi|x,\xi\nmid yz$类型的方程有解，那么设$(x,y,z)$是两两互质的、$N(x)$最小的一组解。同样要注意这里的$N(x)$既要遍历所有的$\varepsilon$（也就是六个单位数），又要遍历在$\varepsilon$固定时的所有的解，最后挑选出$N(x)$最小的解，可能不止一组这样的解，但是只取任意一组就行了。

首先对$x$中$\xi$的次数作一个判断。由于$-\varepsilon x^3=y^3 +z^3$，设$y^3\equiv e(\bmod\,9),z^3\equiv f(\bmod\,9)$，$e,f\in\{-1,1\}$，那么考虑两边模$\xi^3$，就有$e+f\equiv 0(\bmod\,\xi^3)$，也就是$e+f=0$，这样一来$y^3+z^3$至少是9的倍数，但是$9=\xi^4\omega$，因此$x$的$\xi$次数至少是2，即$\xi^2|x$。

然后核心部分是分解：
$$-\varepsilon x^3=(y+z)(y+z\omega)(y+z\omega^2)$$
右边的三项有下面的关系
$$\begin{aligned}(y+z)-(y+z\omega)=(1-\omega)z=\xi z\\
(y+z)\omega-(y+z\omega)=(\omega-1)y=-\xi y\\
(y+z\omega)-(y+z\omega^2)=\omega(1-\omega)z=\omega\xi z\\
(y+z\omega)\omega-(y+z\omega^2)=(\omega-1)y=-\xi y
\end{aligned}$$
由于$y,z$互质，因此$y+z$和$y+z\omega$、$y+z\omega$和$y+z\omega^2$最多只有公约数$\xi$，而左边有约数$\xi^6$，所以右边至少有一项有约数$\xi$，一旦其中有一项有了约数$\xi$，另外两项必然也有约数$\xi$，但是三者的公约数最多是$\xi$，因此其中两项的$\xi$的次数只能是1，剩下一项则“包揽”了余下的所有$\xi$的次数（大于等于4）。然而$y+z,y+z\omega,y+z\omega^2$这三项地位其实是平等的，因为可以在$z$中乘上$\omega$的若干次方，以达到置换。因此不失一般性，设$\xi^4|y+z$，那么可以设
$$\begin{aligned}x=\xi^2 \chi\\
y+z=\xi^4 r'\\
y+z\omega=\xi s'\\
y+z\omega^2=\xi t'
\end{aligned}$$
那么$-\varepsilon\chi^3=r's't'$，$r',s',t'$两两互质，因此它们在相差一个单位数因子的情况之下，都是立方数，也就是立方数的伴随，于是可以设
$$r'=\varepsilon_1 r^3,\ s'=\varepsilon_2 s^3,\ t'=\varepsilon_3 t^3$$

留意到
$$(y+z)+(y+z\omega)\omega+(y+z\omega^2)\omega^2=0$$
从而得到
$$\varepsilon_1 \xi^3 r^3+(\varepsilon_2 \omega) s^3+(\varepsilon_3 \omega^2) t^3=0$$
其中$\varepsilon_1,\varepsilon_2 \omega,\varepsilon_3 \omega^2$都是单位数，根据引理，上述方程具有形式
$$\varepsilon' \xi^3 r^3+s^3+t^3=0$$
那么$(\xi r,s,t)$就是单位数取$\varepsilon'$时的一组解，而且显然有$N(\xi r) < N(x)$（因为$\xi^4 r^3$是$x^3$的一个因子，$N(\xi) > 1$），因此跟假设矛盾。所以方程$\varepsilon x^3+y^3 +z^3=0,\xi|x,\xi\nmid yz$在$\mathbb{Z}[\omega]$中无解，从而$x^3+y^3+z^3=0$在$\mathbb{Z}[\omega]$中无解。证毕。
