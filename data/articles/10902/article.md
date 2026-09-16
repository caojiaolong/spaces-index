# 一道概率不等式：盯着它到显然成立为止！

> 作者：苏剑林 · 科学空间 · 2025-04-30
>
> 原文：<https://spaces.ac.cn/archives/10902>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前两天，QQ群里有群友抛出了一道不等式求证：

[![一道概率相关的不等式，出自《There is no fast single hashing algorithm》](<https://spaces.ac.cn/usr/uploads/2025/04/3939521975.jpg>)](<https://spaces.ac.cn/usr/uploads/2025/04/3939521975.jpg>)

一道概率相关的不等式，出自《There is no fast single hashing algorithm》

简短的题目，加上“easily”的提示，让人觉得这似乎是显然成立的结果，然而提问者却表示尝试了很久仍未果。那么实际情况如何呢？是否真的是显然成立呢？

## 初步尝试

题目等价于证
\begin{equation}\sum_{i=0}^j p^i \leq \sum_{i=0}^j \left(\log\frac{1}{1-p}\right)^i/i!,\qquad p\in[0, 1)\label{eq:q}\end{equation}
所给的提示是
\begin{equation}\log\frac{1}{1-p} = \sum_{i=1}^{\infty} \frac{p^i}{i} = p + \frac{p^2}{2} + \frac{p^3}{3} + \cdots \label{eq:log-1-p}\end{equation}
这个提示似乎在引导我们替换掉式$\eqref{eq:q}$右端的$\log\frac{1}{1-p}$，然后展开成$p$的幂级数来逐项对比。由于$\log\frac{1}{1-p}$的展开式系数全是正数，所以式$\eqref{eq:q}$展开成$p$的幂级数后每一项系数必然也都是正数，因此我们只要证明前$j$项系数都大于等于1。

思路看上去靠谱，然而将一个幂级数的$i$次幂重新展开为幂级数是一件非常繁琐的事情，所以实际上很难走通。笔者又另外尝试了一些思路，比如设$x=\log\frac{1}{1-p}$转为关于$x$的不等式来证，看上去也很有希望，但最终都被卡住，无法完整完成证明。

所以，初次尝试宣布失败。

## 泰勒硬刚

兜兜转转，回到原点。试了大半天，总感觉还是将式$\eqref{eq:q}$右端展开为幂级数的思路靠谱，但直接对式$\eqref{eq:log-1-p}$求幂走不通的话，我们只能回到泰勒展开的原始做法——求导。

简单起见，我们定义
\begin{equation}f_j(x) = \sum_{i=0}^j \frac{x^i}{i!}\end{equation}
原题就是证
\begin{equation}f_j(x) \geq \sum_{i=0}^j p^i,\qquad x = \log\frac{1}{1-p}\end{equation}
我们转化为证
\begin{equation}\left.\frac{d^k}{dp^k}f_j(x)\right|_{p=0} \,\,\geq\,\, \left\{\begin{aligned}&\, 1,\,\, k \leq j \\[5pt]
&\,0,\,\, k > j\end{aligned}\right.\end{equation}
留意到当$j \geq 0$时$f_j(0)=1$，当$j < 0$时我们定义$f_j(x)\equiv 0$，以及成立$f_j'(x)= f_{j-1}(x)$，于是我们有
\begin{gather}\frac{d}{dp}f_j(x) = f_{j-1}(x) \frac{1}{1-p} \\
\frac{d^2}{dp^2}f_j(x) = [f_{j-1}(x) + f_{j-2}(x)]\frac{1}{(1-p)^2} \\
\frac{d^3}{dp^3}f_j(x) = [2f_{j-1}(x) + 3f_{j-2}(x) + f_{j-3}(x)]\frac{1}{(1-p)^3} \\
\vdots \notag\\
\frac{d^k}{dp^k}f_j(x) = [\alpha_1 f_{j-1}(x) + \alpha_2 f_{j-2}(x) + \cdots + \alpha_k f_{j-k}(x)]\frac{1}{(1-p)^k} \\[8pt]
\frac{d^{k+1}}{dp^{k+1}}f_j(x) = \left[\begin{aligned}
\alpha_1 f_{j-2}(x) + \alpha_2 f_{j-3}(x) + \cdots + \alpha_k f_{j-k-1}(x) \\
+ k(\alpha_1 f_{j-1}(x) + \alpha_2 f_{j-2}(x) + \cdots + \alpha_k f_{j-k}(x))
\end{aligned}\right]\frac{1}{(1-p)^{k+1}}
\end{gather}
规律：$\frac{d^k}{dp^k}f_j(x)$是$f_{j-1}(x),f_{j-2}(x),\cdots,f_{j-k}(x)$的线性组合与$\frac{1}{(1-p)^k}$的乘积，注意其实我们只关心$\left.\frac{d^k}{dp^k}f_j(x)\right|_{p=0}$的值，上式代入$p=0$后就等于$f_{j-1}(x),f_{j-2}(x),\cdots,f_{j-k}(x)$中非零项的系数之和（注意$f_{j-1}(0),f_{j-2}(0),\cdots,f_{j-k}(0)$非1即0）。根据$f_j(x)$的求导规律，我们可以得出：当$k \leq j$时，有
\begin{equation}\left.\frac{d^k}{dp^k}f_j(x)\right|_{p=0} = k \times \left.\frac{d^{k-1}}{dp^{k-1}}f_j(x)\right|_{p=0}\end{equation}
所以$\left.\frac{d^k}{dp^k}f_j(x)\right|_{p=0} = k!$，那么根据泰勒展开式$p^k$项的系数为$k!/k!=1$；当$k > j$时，由于此时$f_{j-k}(0)=0$，所以$\left.\frac{d^k}{dp^k}f_j(x)\right|_{p=0} < k!$，所以$p^k$项的系数则小于1，但仍旧大于零，因此所证不等式成立。

## 显然易得

这样看来，原题目并不是像作者说的显然成立了？笔者开始也是这样认为的，直到另一位群友给出了一个极其简单的理解视角，才恍然大悟，这道题还真的是显然成立的！

这个视角充分体现了逆向思维，如下：
\begin{equation}\begin{aligned}
\sum_{i=0}^j \left(\log\frac{1}{1-p}\right)^i/i! =&\, \sum_{i=0}^{\infty} \left(\log\frac{1}{1-p}\right)^i/i! - \sum_{i=j+1}^{\infty} \left(\log\frac{1}{1-p}\right)^i/i! \\
=&\, \exp\left(\log\frac{1}{1-p}\right) - \sum_{i=j+1}^{\infty} \left(\log\frac{1}{1-p}\right)^i/i! \\
=&\, \frac{1}{1-p} - \sum_{i=j+1}^{\infty} \left(\log\frac{1}{1-p}\right)^i/i! \\
=&\, \sum_{i=1}^{\infty} p^i - \sum_{i=j+1}^{\infty} \left(\log\frac{1}{1-p}\right)^i/i! \\
\end{aligned}\end{equation}
注意$\log\frac{1}{1-p}\sim p$，所以后面的$\sum_{i=j+1}^{\infty}$部分，只对$p^{j+1}$及以上的项有贡献，因此我们就近乎显然地证明了右端展开式的前$j+1$项系数都是1！整个证明并不难理解，真正难的地方是勇敢将右端的求和推到无穷项，然后留意到误差部分只对$p^{j+1}$及以上有贡献，这是经典的逆向思维。

数学，妙不可言～
