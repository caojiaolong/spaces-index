# &#91;欧拉数学&#93;素数倒数之和

> 作者：苏剑林 · 科学空间 · 2011-11-19
>
> 原文：<https://spaces.ac.cn/archives/1510>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上一篇文章我通过欧拉数学的方式简单地讲了数论中的“黎曼ζ函数”和“金钥匙”。事实上，这把“金钥匙”与很多问题之间的联系已经被建立了起来，换句话说，<u>“金钥匙”已经插入到了相应的“锁孔”中，数学家的工作就是要把这个金钥匙“拧动”，继而打开数学之门</u>！

接下来我们看看如何证明所有素数的倒数之和发散的。在入正题之前，我们得需要看一个<strong>引理</strong>：

> 无限数列${a_n}$的每一项都大于0，那么$\sum\limits_{n=1}^{\infty} a_n$与$\prod\limits_{n=1}^{\infty} \left(1+a_n\right)$的敛散性相同。换句话说，两者互为充分必要条件！

怎么证明呢？记$S=a_1+a_2+...+a_n,T=\left(1+a_1\right)\left(1+a_2\right)...\left(1+a_n\right)$。首先证明<strong> T \> 1\+S</strong>，这应该很简单，因为$\left(1+a_1\right)\left(1+a_2\right)=1+a_1+a_2+a_1 a_2 > 1+\left(a_1+a_2\right)$，所以$\left(1+a_1\right)\left(1+a_2\right)\left(1+a_3\right) > [1+\left(a_1+a_2\right)]\left(1+a_3\right) > 1+\left(a_1+a_2+a_3\right)$，一直递推下去即可。接着我们回忆一下一条<strong>平均不等式</strong>：倘若$x_i$都是正数，那么：
$$\frac{x_1+x_2+...+x_n}{n} \geq \sqrt[n]{x_1 x_2...x_n}$$
或者写成
$$x_1 x_2...x_n \leq \left(\frac{x_1+x_2+...+x_n}{n}\right)^n$$
这样的话
$$\left(1+a_1\right)\left(1+a_2\right)...\left(1+a_n\right) \leq \left(\frac{n+a_1+a_2+...+a_n}{n}\right)^n =\left(1+\frac{S}{n}\right)^n$$
最后一步由<u>自然对数的底e的定义</u>可以得到$\left(1+\frac{S}{n}\right)^n < e^S$。于是我们实际证明了：$T < e^S$，综合起来就是
$$1+S < T < e^S$$

于是，T的值域完全由S决定了起来。显然，S有限的情况下，T不可能趋于无穷！将n推广至无穷显然也是成立的。证毕。顺道提一下：可以检验发现，$e^S$是$T$的一个相当好的近似值，其精确度远远超越$\left(1+S\right)$，于是在相当多的情况下，可以直接用$T \approx e^S$。

现在拿出我们的“金钥匙”——$\xi \left(s\right)=\prod\limits_{p} \left(1-p^{-s}\right)^{-1}$。这里我们只用到s=1的情况，并且将右端的素数截取到p，即
$$\begin{aligned}
&\left(1+\frac{1}{2}+\frac{1}{2^2}+...\right)\left(1+\frac{1}{3}+\frac{1}{3^2}+...\right)...\left(1+\frac{1}{p}+\frac{1}{p^2}+...\right)\\
=&\frac{2}{2-1}\cdot \frac{3}{3-1}\cdot \frac{5}{5-1}\cdot ...\cdot \frac{p}{p-1}=1+\frac{1}{2}+\frac{1}{3}+\frac{1}{4}+...+\frac{1}{p}+...\\
> &1+\frac{1}{2}+\frac{1}{3}+\frac{1}{4}+...+\frac{1}{p} \\
> &\ln\left(p+1\right)\end{aligned}$$

记第n个素数为$p_n$，不难证明：$\frac{p_n}{p_n -1} < 1+\frac{1}{p_{n-1}}$。

于是
$$\begin{aligned}\ln\left(p+1\right) < &\frac{2}{2-1}\cdot \frac{3}{3-1}\cdot \frac{5}{5-1}\cdot ...\cdot \frac{p}{p-1} \\
< &2\left(1+\frac{1}{2}\right)\left(1+\frac{1}{3}\right)\left(1+\frac{1}{5}\right)...\left(1+\frac{1}{p}\right)\\
< &2e^{\left(\frac{1}{2}+\frac{1}{3}+\frac{1}{5}+...+\frac{1}{p}\right)}\end{aligned}$$

记$Q=\frac{1}{2}+\frac{1}{3}+\frac{1}{5}+...+\frac{1}{p}$，即有$\ln\left(p+1\right) < 2e^Q$，则
$$Q > \ln \ln \left(p+1\right) -\ln2$$

因为存在无穷大的素数（素数的个数无限，且递增），所以所有素数倒数之和发散！证毕。

最后，根据前边的“$e^S$是$T$的一个相当好的近似”，我们就可以认为$ \ln \ln p$是Q的一个相当好的近似。
