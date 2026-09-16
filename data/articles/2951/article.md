# Cantor\-Bernstein 定理（给出双射！）

> 作者：苏剑林 · 科学空间 · 2014-09-19
>
> 原文：<https://spaces.ac.cn/archives/2951>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

学过集合论的朋友应该会知道，按照定义，判断两个集合的“势”相同的最直接的方法就是给出这两个集合之间的一个双射。然而，这样的双射往往不容易想到，比如

> 请给出一个$[0,+\infty)$到$(0,+\infty)$的双射

但是，直观地看，这两个集合的势一定是相当的，而且这两个集合之间的双射有无穷多个。然而，大多数的我们却很难想出一个双射来。当我们看到构造出来的双射时，第一反应往往是：这样的证明是怎么想到的！？比如，上述问题的答案之一是：

> $$h(x)=\left\{\begin{aligned}&x^2+1,\quad x=0,1,2,5,26,677,\dots\\
&x,\quad x\in[0,+\infty)\text{且}x\neq 0,1,5,26,\dots\end{aligned}\right.$$

其中序列$0,1,2,5,26,677,\dots$的后一项是前一项的平方加1。这样构造出来的双射无疑会让我们惊叹！

<strong>直观分析</strong>

怎么样才能构造出这样的双射来呢？想法是很美妙的。首先，假设集合$A$和$B$的势相当，虽然$A\mapsto B$的双射不容易给出，但是$A\mapsto B$的一个单射$f$通常是很容易给出的，也就是给出$A$到$B$的一个子集的双射；同样地，我们也可以给出$B\mapsto A$的一个单射$g$。一个很“经济”的想法是：能否用$f$和$g^{-1}$“拼凑”出$A\mapsto B$的一个双射来？

答案是肯定的，这就是被我们称之为“Cantor\-Bernstein定理”的内容！下面我们来简单描述一下其中的思想。

<strong>Step One：</strong>先考虑$B\mapsto A$的双射$g$，$g$的值域是$g(B)$，也就是说，$g^{-1}$的定义域最多是$g(B)$，这样一来，$A\mapsto B$的双射中，$A-g(B)$的那部分，只能由$f$来定义，记$C_0=A-g(B)$。

<strong>Step Two：</strong>如果$C_0=\emptyset$，那么双射就可以只由$g^{-1}$定义了。如果$C_0\neq\emptyset$，那么$C_0$部分就得由$f$定义。但是，这不意味着剩下部分就由$g^{-1}$定义了。因为剩下部分（也就是$g(B)$）通过$g^{-1}$可以映射到整个$B$，而且$f$又将$C_0$映射到部分的$B$，这样一来构造出来的就不是双射了。

<strong>Step Three：</strong>为了消除Step Two的困难，唯一的办法就是将$f$的定义域扩大一点，将$g^{-1}$的定义域缩小一点。因为既然$C_0$部分已经必须由$f$定义了，那么要更改的只能是增大$f$的定义域。$C_0$在$B$中的像（值域）就是$f(C_0)$，我们不能让$g^{-1}$的像跟它有交集（$g^{-1}$的像就是$g$的定义域，$g^{-1}$的定义域就是$g$的像），那么就是说从$g^{-1}$的定义域（即$g$的像$g(B)$）中扣除$g(f(C_0))=C_1$这部分即可，扣除了之后，$g^{-1}(x)$不再可能属于$f(C_0)$；

<strong>Step Four：</strong>但是，现在扣除了$C_0$的影响了，却多出了$C_1$出来，我们必须用同样的技巧扣除$C_1$的影响，但这必然会产生$C_2=g(f(C_1))$的影响\.\.\.\.\.\.直至无穷。扣除了这无穷的影响后，得到的就是合理的双射了。

用数学的语言表述如下：
$$h(x)=\left\{\begin{aligned}&f(x),\quad x\in C\\
&g^{-1}(x),\quad x\in A-C\end{aligned}\right.$$
其中$C=\bigcup\limits_{n=0}^{\infty} C_n,\quad C_{n+1}=g(f(C_n))$。
可以证明，这就是$A\mapsto B$的一个双射。具体证明这里就不写出来了，读者可以参考维基百科的“[康托尔－伯恩斯坦－施罗德定理](<http://zh.wikipedia.org/zh/%E5%BA%B7%E6%89%98%E5%B0%94%EF%BC%8D%E4%BC%AF%E6%81%A9%E6%96%AF%E5%9D%A6%EF%BC%8D%E6%96%BD%E7%BD%97%E5%BE%B7%E5%AE%9A%E7%90%86>)”条目。

<strong>例子演示</strong>

现在我们来看如何给出$[0,+\infty)$到$(0,+\infty)$的一个双射。记
$$\left\{\begin{aligned}&f(x)=x^2+1,\quad x\in [0,+\infty)\\
&g(x)=x,\quad x\in (0,+\infty)
\end{aligned}\right.$$
$f,g$分别是$[0,+\infty)\mapsto (0,+\infty)$、$(0,+\infty)\mapsto [0,+\infty)$的单射，$g^{-1}(x)=x$。可以算得：
$$\begin{aligned}&C_0=[0,+\infty)-(0,+\infty)=\{0\}\\
&C_1=g(f(C_0))=\{1\}\\
&C_2=g(g(C_1))=\{2\}\\
&\dots
\end{aligned}$$
所以
$$C=\bigcup\limits_{n=0}^{\infty} C_n=\{0,1,2,5,26,677,\dots\}$$
于是可以得到双射
$$h(x)=\left\{\begin{aligned}&x^2+1,\quad x=0,1,2,5,26,677,\dots\\
&x,\quad x\in[0,+\infty)\text{且}x\neq 0,1,5,26,\dots\end{aligned}\right.$$
