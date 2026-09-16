# 自然数集中 N = ab \+ c 时 a \+ b \+ c 的最小值

> 作者：苏剑林 · 科学空间 · 2023-09-20
>
> 原文：<https://spaces.ac.cn/archives/9775>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前天晚上微信群里有群友提出了一个问题：

> 对于一个任意整数$N > 100$，求一个近似算法，使得$N=a\times b+c$（其中$a,b,c$都是非负整数），并且令$a+b+c$尽量地小。

初看这道题，笔者第一感觉就是“这还需要算法？”，因为看上去自由度太大了，应该能求出个解析解才对，于是简单分析了一下之后就给出了个“答案”，结果很快就有群友给出了反例。这时，笔者才意识到这题并非那么平凡，随后正式推导了一番，总算得到了一个可行的算法。正当笔者以为这个问题已经结束时，另一个数学群的群友精妙地构造了新的参数化，证明了算法的复杂度还可以进一步下降！

整个过程波澜起伏，让笔者获益匪浅，遂将过程记录在此，与大家分享。

## 随手的错

抛开$N > 100$这个约束，将原问题可以等价变换为：

> 已知$a,b\in\mathbb{N},ab \leq N$，求$S = N - ab + a + b$的最小值。

很明显当$N$足够大时，$ab$应该占主项，所以直觉上就是$ab$尽量接近$N$和$a,b$尽可能向相等，于是就随手拍了个“$a=b=\left\lfloor\sqrt{N}\right\rfloor$和$a=\left\lfloor\sqrt{N}\right\rfloor,b=\left\lfloor\sqrt{N}\right\rfloor+1$”二选一的答案。然而很快就有群友给出了反例：当$N=130$时，$S$最小值在$a=10,b=13$取到，而根据笔者给出的公式则是$a=b=11$，显然不够优。

细想之下，才发现是笔者低视这个问题了，于是乎认真推导了一番～

## 直接分析

不失一般性，设$a\leq b$。首先，S可以等价变换为
\begin{equation}S = N - (\sqrt{ab}-1)^2 + (\sqrt{a}-\sqrt{b})^2\end{equation}
可以看出，$S$取最小值大致上是两个方向：1、$ab$尽量大（接近$N$）；2、$a,b$尽量接近。为此，暂且设
\begin{equation}b = \left\lfloor\frac{N}{a}\right\rfloor = \frac{N}{a} - \left\{\frac{N}{a}\right\}\end{equation}
那么
\begin{equation}S = N - a\left(\frac{N}{a} - \left\{\frac{N}{a}\right\}\right)+ a + \frac{N}{a} - \left\{\frac{N}{a}\right\} = (a-1)\left\{\frac{N}{a}\right\}
+ a + \frac{N}{a}\label{eq:S}\end{equation}
考虑两个极端：

> 1、最理想情况下$\left\{\frac{N}{a}\right\}=0$，那么$a + \frac{N}{a}$最小值在$a=\sqrt{N}$取到。
> 
> 
> 
> 2、最不理想的情况下，$\left\{\frac{N}{a}\right\}$可以无限接近于1，即
> \begin{equation}S \to a - 1 + a + \frac{N}{a} = 2a + \frac{N}{a} - 1\end{equation}
> 此时最小值在$a=\sqrt{N/2}$取到。

基于以上两点，我们就可以提出一个算法：

> $a$遍历$\big(\sqrt{N/2},\sqrt{N}\big]$的所有整数，令$b=\left\lfloor N/a\right\rfloor$，取$S$最小者。

很明显，这是一个复杂度为$\mathcal{O}(\sqrt{N})$的算法，跟大数分解同复杂度，所以刚推导出来那会笔者对此很震惊，这个看上去不起眼的题目居然跟大数分解是同一复杂度。

## 新参数化

后来笔者将题目分享到一个数学群，经群里的大牛指点后，才发现发现自己前面的分析还是太肤浅了，原来算法的复杂度还可以明显降低。

降低复杂度的关键是引入新的参数化，并精细地放缩不等式来缩小搜索范围。假设$a,b$是同奇偶的，那么我们可以设$a=x-y,b=x+y$，其中$x,y\in\mathbb{N}$，然后得到
\begin{equation}N + y^2 = x^2 + c,\quad S = 2x + c \end{equation}
新参数化的关键是将原本相乘的$ab$变为了相减的$x^2-y^2$，从而可以更清楚地看到变化方向，并且允许更精细的放缩。如果$a,b$是一奇一偶，那么只需要将$x$换成$x+\frac{1}{2}$、$y$换成$y+\frac{1}{2}$，推导是相似的，下面我们分别讨论。

### 同奇偶性

进一步地，我们有
\begin{equation}S = 2x + c = 2x + (N + y^2 - x^2)= N + y^2 + 1 - (x - 1)^2\end{equation}
以及$N + y^2 \geq x^2$。如果$x$已经给定，那么自然是$y$越小$S$也越小。接下来又要分两种情况讨论。

第一种情况是$x^2 \leq N$，那么$y^2 \geq x^2 - N$最小值显然是$y=0$，此时$S=N+1-(x-1)^2$，很明显$x$越大$S$就越小，再结合$x^2 \leq N$，那么最大就只能是$x=\left\lfloor \sqrt{N}\right\rfloor$了。

第二种情况是$x^2 \geq N$，那么$y^2 \geq x^2 - N$的最小值就是$y=\left\lceil \sqrt{x^2 - N}\right\rceil$，此时
\begin{equation}\begin{aligned}
S =&\, 2x + (N + y^2 - x^2)\\
\leq&\, 2x + \left[N + \left(\sqrt{x^2 - N} + 1\right)^2 - x^2\right]\\
=&\,2x + 1 + 2\sqrt{x^2 - N}
\end{aligned}\label{eq:S1}\end{equation}
最后的式子是关于$x$单调递增的，所以为了让它尽可能小，那么$x$应该尽可能小，结合$x^2 \geq N$，那么$x=\left\lceil \sqrt{N}\right\rceil$。注意这是基于$S$的上界算出来的，实际$S$的最小值未必就在$x=\left\lceil \sqrt{N}\right\rceil$取到，但我们可以利用这个结果来缩小搜索范围。代入上述不等式后，进一步得到
\begin{equation}\begin{aligned}
S \leq&\,2x + 1 + 2\sqrt{x^2 - N} \\
\leq&\,2(\sqrt{N}+1) + 1 + 2\sqrt{(\sqrt{N}+1)^2 - N} \\
=&\,2\sqrt{N}+ 3+ 2\sqrt{1+ 2\sqrt{N}} \\
\end{aligned}\label{eq:S2}\end{equation}
这提供了$S$的最小值的一个上界。假设$S$取最小值时$x=x^*,c=c^*$，那么有
\begin{equation}\begin{array}{c} S = 2x^* + c^* \leq 2\sqrt{N}+ 3+ 2\sqrt{1+ 2\sqrt{N}} \\
\Downarrow\\
x^*\leq \sqrt{N} + \frac{3}{2} + \sqrt{1+ 2\sqrt{N}} = \sqrt{N} + \mathcal{O}(\sqrt[4]{N})
\end{array}\end{equation}
这就意味着我们只需要搜索$\big[\sqrt{N},\sqrt{N} + \frac{3}{2} + \sqrt{1+ 2\sqrt{N}}\big]$的整数就可以找到最优解，复杂度是$\mathcal{O}(\sqrt[4]{N})$而不是$\mathcal{O}(\sqrt{N})$！

### 一奇一偶

假设$a,b$是一奇一偶，那么我们可以设$a=\left(x+\frac{1}{2}\right)-\left(y+\frac{1}{2}\right),b=\left(x+\frac{1}{2}\right)+\left(y+\frac{1}{2}\right)$，其中$x,y\in\mathbb{N}$，然后得到
\begin{equation}\begin{aligned}
&\,N + \left(y+\frac{1}{2}\right)^2 = \left(x+\frac{1}{2}\right)^2 + c \\
&\,S = 2\left(x + \frac{1}{2}\right) + c =  N + \left(y+\frac{1}{2}\right)^2 + 1 - \left(x-\frac{1}{2}\right)^2
\end{aligned}\end{equation}
同样要分两种情况讨论。第一，当$\left(x+\frac{1}{2}\right)^2\leq N$时，类似上一节的结果是$y=0,x=\left\lfloor \sqrt{N}-\frac{1}{2}\right\rfloor$。第二，当$\left(x+\frac{1}{2}\right)^2\geq N$时，$y$的最小值是$y=\left\lceil \sqrt{\left(x+\frac{1}{2}\right)^2 - N}-\frac{1}{2}\right\rceil$，那么跟$\eqref{eq:S1},\eqref{eq:S2}$同理，代入$x=\left\lceil \sqrt{N}-\frac{1}{2}\right\rceil$，有
\begin{equation}\begin{aligned}
S =&\, 2\left(x + \frac{1}{2}\right) + \left[N + \left(y + \frac{1}{2}\right)^2 - \left(x+\frac{1}{2}\right)^2\right] \\
\leq&\, 2\left(x + \frac{1}{2}\right) + \left[N + \left(\sqrt{\left(x
+ \frac{1}{2}\right)^2 - N} + 1\right)^2 - \left(x+\frac{1}{2}\right)^2\right]\\
=&\,2\left(x
+ \frac{1}{2}\right) + 1 + 2\sqrt{\left(x
+ \frac{1}{2}\right)^2 - N} \\
\leq&\,2(\sqrt{N}+1) + 1 + 2\sqrt{(\sqrt{N}+1)^2 - N} \\
=&\,2\sqrt{N}+ 3+ 2\sqrt{1+ 2\sqrt{N}} \\
\end{aligned}\end{equation}
因此有
\begin{equation}\begin{array}{c} S = 2\left(x^*+\frac{1}{2}\right) + c^* \leq 2\sqrt{N}+ 3+ 2\sqrt{1+ 2\sqrt{N}} \\
\Downarrow\\
x^*\leq \sqrt{N} + 1 + \sqrt{1+ 2\sqrt{N}} = \sqrt{N} + \mathcal{O}(\sqrt[4]{N})
\end{array}\end{equation}

### 汇总结果

综合以上两节的结果，我们可以将求$S$的最小值流程整理如下：

> 1、如果$N$是平方数，那么返回$x=\sqrt{N},y=0$（$a=b=\sqrt{N},c=0$）；
> 
> 
> 
> 2、否则，记录$x=\left\lfloor \sqrt{N}\right\rfloor,y=0$、$x=\left\lfloor \sqrt{N}-\frac{1}{2}\right\rfloor+\frac{1}{2},y=\frac{1}{2}$中让$S$较小者；
> 
> 
> 
> 3、遍历$\big(\sqrt{N}-1/2,\sqrt{N} + \frac{3}{2} + \sqrt{1+ 2\sqrt{N}}\big]$的所有整数$m$，令$x=m,y=\left\lceil \sqrt{m^2 - N}\right\rceil$以及$x=m+\frac{1}{2},y=\left\lceil \sqrt{\left(m+\frac{1}{2}\right)^2 - N}-\frac{1}{2}\right\rceil + \frac{1}{2}$，如果它们对应的$S$更小，那么覆盖记录的$x,y$；
> 
> 
> 
> 4、返回$a=x-y,b=x+y,c=N-ab$。

## 又一思路

文章发出后，有一位大牛觉得以上分是否同奇偶的讨论过于麻烦，于是又提出了一个新的参数化方式：令$p=a+b,q=b-a$，那么
\begin{equation}4N - p^2 + q^2 = 4c, \quad S = p + c \end{equation}
注意这里$p,q$必须是同奇偶的，才能保证$c$是一个整数。

接下来的分析跟前面就几乎一样了，因为$c\geq 0$，所以$q^2\geq p^2 - 4N$。考虑给定$p$，那么$S$的最小值等价于$c$的最小值，也等价于$q$的最小值。如果$p^2 \leq 4N$，那么$q$的最小值是$0$或$1$，这需要根据$p,q$同奇偶来确定。然后根据
\begin{equation}4S = 4p+4c = 4p + 4N - p^2 + q^2 = 4N + q^2 + 4 - (p - 2)^2 \end{equation}
所以$S$最小对应$p$最大，结合$p^2 \leq 4N$则有$p=\left\lfloor 2\sqrt{N}\right\rfloor$。

如果$p^2 \geq 4N$，那么$q$的最小值是$q=\left\lceil \sqrt{p^2 - 4N} \right\rceil + \varepsilon$，这里$\varepsilon\in\{0,1\}$同样要根据$p,q$同奇偶来确定。然后我们代入
\begin{equation}\begin{aligned}
4S =&\, 4p + 4N - p^2 + q^2 \\
\leq &\, 4p + 4N - p^2 + \left( \sqrt{p^2 - 4N}  + \varepsilon + 1\right)^2 \\
=&\, 4p + (\varepsilon + 1)^2 + 2(\varepsilon + 1)\sqrt{p^2 - 4N}
\end{aligned}\end{equation}
代入$p=\left\lceil 2\sqrt{N}\right\rceil$，我们可以得出$4S$的最小值的一个上界：
\begin{equation}\begin{aligned}
4S \leq&\, 4p + (\varepsilon + 1)^2 + 2(\varepsilon + 1)\sqrt{p^2 - 4N} \\
\leq&\, 4(2\sqrt{N} + 1) + (\varepsilon + 1)^2 + 2(\varepsilon + 1)\sqrt{(2\sqrt{N} + 1)^2 - 4N} \\
=&\, 4(2\sqrt{N} + 1) + (\varepsilon + 1)^2 + 2(\varepsilon + 1)\sqrt{1 + 4\sqrt{N}} \\
\leq&\, 4(2\sqrt{N} + 1) + 4 + 4\sqrt{1 + 4\sqrt{N}} \\
\\
\Rightarrow S \leq&\,2\sqrt{N} + 2 + \sqrt{1 + 4\sqrt{N}}
\end{aligned}\end{equation}
因为$S = p + c \geq p$，所以这也是$p$的一个上界。

综上所述，新的求$S$的最小值流程整理如下：

> 1、如果$N$是平方数，那么返回$p=2\sqrt{N},q=0$（$a=b=\sqrt{N},c=0$）；
> 
> 
> 
> 2、否则，令$p = \left\lfloor 2\sqrt{N}\right\rfloor$，如果$p$是偶数，则$q=0$，否则$q=1$，计算此时的$S$；
> 
> 
> 
> 3、遍历$\big(2\sqrt{N},2\sqrt{N} + 2 + \sqrt{1 + 4\sqrt{N}}\big]$的所有整数为$p$，令$q=\left\lceil \sqrt{p^2 - 4N} \right\rceil + \varepsilon$，$\varepsilon\in\{0,1\}$来保证$p,q$同奇偶，如果它们对应的$S$更小，那么覆盖记录的$p,q$；
> 
> 
> 
> 4、返回$a=\frac{p-q}{2},b=\frac{p+q}{2},c=N-ab$。

## 文章小结

本文分享了一道“以为是个青铜，结果是个王者”的算法题的思考和学习过程。
