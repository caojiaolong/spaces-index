# 素数之美2：Bertrand假设的证明

> 作者：苏剑林 · 科学空间 · 2014-08-09
>
> 原文：<https://spaces.ac.cn/archives/2800>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

有了上一篇文章的$\prod\limits_{p\leq n}p < 4^{n-1}$的基础，我们其实已经很接近Bertrand假设的证明了。Bertrand假设的证明基于对二项式系数$C_n^{2n}$的素因子次数的细致考察，而在本篇文章中，我们先得到一个关于素数之积的下限公式，然后由此证明一个比Bertrand假设稍微弱一点的假设。最后，则通过一个简单的技巧，将我们的证明推动至Bertrand假设。

<strong>二项式系数的素因子</strong>

首先，我们考察$n!$中的素因子$p$的次数，结果是被称为Legendre定理的公式：

> $n$中素因子$p$的次数恰好为$\sum\limits_{k\geq 1}\left\lfloor\frac{n}{p^k}\right\rfloor$。

证明很简单，因为$n!=1\times 2\times 3\times 4\times \dots \times n$，每隔$p$就有一个$p$的倍数，每隔$p^2$就有一个$p^2$的倍数，每隔$p^3$就有一个$p^3$的倍数，每增加一次幂，将多贡献一个$p$因子，所以把每个间隔数叠加即可。注意该和虽然写成无穷形式，但是非零项是有限的。

接着，我们用上面的公式考察$C_{n}^{2n}=\frac{(2n)!}{n! n!}$的素因子$p$的个数，由于只是阶乘的除法，因此，$C_n^{2n}$中素因子$p$的次数为
$$\sum\limits_{k\geq 1}\left(\left\lfloor\frac{2n}{p^k}\right\rfloor-2\left\lfloor\frac{n}{p^k}\right\rfloor\right)$$
注意到括弧里边的数最多是1，因为$\left\lfloor\frac{2n}{p^k}\right\rfloor-2\left\lfloor\frac{n}{p^k}\right\rfloor < \frac{2n}{p^k}-2\left(\frac{n}{p^k}-1\right)=2$，因此，设$C_n^{2n}$中$p$的次数是$\nu_p$，则
$$\nu_p \leq \max \{r,p^r \leq 2n\}$$

这样一来，我们就知道，在$C_n^{2n}$中，大于$\sqrt{2n}$的素因子的次数最多是1，而小于等于$\sqrt{2n}$的素因子的次数则不超过2n，同时小于等于$\sqrt{2n}$的素数不超过$\sqrt{2n}$个，于是，我们有公式
$$C_{n}^{2n} \leq (2n)^{\sqrt{2n}}\prod\limits_{\sqrt{2n} < p\leq 2n}p$$

接着，我们证明$C_{n}^{2n} >\frac{4^n}{2n}$，事实上，这个不等式相当弱，因为由string公式，我们有$C_{n}^{2n} \sim\frac{4^n}{\sqrt{\pi n}}$，因此该不等式的证明不难，我们直接留给读者。利用该不等式，进一步得到
$$4^n \leq (2n)^{1+\sqrt{2n}}\prod\limits_{\sqrt{2n} < p\leq 2n}p$$
如果只是为了证明Bertrand假设，现在就可以直接跳到下面的“Bertrand假设”部分。这里我们简单地先得到
$$\prod\limits_{p\leq 2n}p > \frac{4^n}{(2n)^{1+\sqrt{2n}}}$$
要注意，分母$(2n)^{1+\sqrt{2n}}$比任意$a^n , a>1$的阶都要小。这是关于素数之积的下限的一个（比较弱的）估计。

<strong>弱Bertrand假设</strong>

一个比Bertrand假设稍弱的命题是

> 对于任意$\varepsilon > 0$，存在$N(\varepsilon) > 0$，使得任意$ n > N(\varepsilon)$，在$(1-\varepsilon)n$到$2n$之间都存在一个素数

上面我们说的内容已经足够证明这一命题，假设$(1-\varepsilon)n$到$2n$之间没有一个素数，那么
$$\prod\limits_{p\leq 2n}p=\prod\limits_{p\leq (1-\varepsilon)n}p$$
根据$\prod\limits_{p\leq n}p < 4^{n-1}$，就有
$$\prod\limits_{p\leq 2n}p < 4^{(1-\varepsilon)n-1}$$
结合本文的结果，就有
$$4^{(1-\varepsilon)n-1} > \frac{4^n}{(2n)^{1+\sqrt{2n}}}$$
也就是
$$(2n)^{1+\sqrt{2n}} > 4^{\varepsilon n+1}$$
之前已经提到过，$(2n)^{1+\sqrt{2n}}$比任意$a^n , a>1$的阶都要小，因此对于足够大的n，上述不等式将导致矛盾，因此对于足够大的n，$(1-\varepsilon)n$到$2n$之间至少有一个素数。

<strong>Bertrand假设</strong>

现在我们离Bertrand假设的证明只有一步之遥了。我们要更细致地考虑$C_n^{2n}$的素因子了，我们发现，$C_n^{2n}$中没有$\frac{2n}{3} < p \leq n$的素因子！证明相当简单，因为$C_{n}^{2n}=\frac{(2n)!}{n! n!}$，而$3p > 2n$则表明，分子$(2n)!$包含且只包含了$p$和$2p$，而$p \leq n$则表明，分母$(n!)^2$只含有素因子$p^2$，分子分母一约，次数为0。这样子，我们就有
$$C_{n}^{2n} \leq (2n)^{\sqrt{2n}}\prod\limits_{\sqrt{2n} < p\leq \frac{2n}{3}}p\times \prod\limits_{n < p\leq 2n}p$$
把$C_{n}^{2n} >\frac{4^n}{2n}$和$\prod\limits_{p\leq n}p < 4^{n-1}$代入上式，我们就得到
$$4^n \leq (2n)^{1+\sqrt{2n}}4^{\frac{2n}{3}}\times \prod\limits_{n < p\leq 2n}p$$
如果$n$到$2n$之间没有一个素数，那么最后一项就是1，这样子
$$4^{\frac{n}{3}} \leq (2n)^{1+\sqrt{2n}}$$
该不等式在$n > 4000$时就出现矛盾。因此对于$ n > 4000$，Bertrand假设成立。

剩下的部分，我们写出素数列

> 2, 3, 5, 7, 13, 23, 43, 83, 163, 317, 631, 1259, 2503, 4001

对于任意不大于4000的数字n，必然落在上面的素数列的相邻两数之间，或者是素数列本身某项。这样n与2n之间必然包括上述素数列的某一项，因此不大于4000时，Bertrand假设也成立。

以上证明收录在《数学天书中的证明》之中，是纯粹由初等的技巧所证明的素数区间上限，事实上，这个上限可以进一步改进，但是所用到的技巧就更为复杂了。一个还没有解决的猜想是，对于任意$n$，$n^2$与$(n+1)^2$之间是否一定有一个素数，目前还没有证实或者证伪。
