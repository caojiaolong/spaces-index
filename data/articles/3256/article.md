# 有趣的求极限题：随心所欲的放缩

> 作者：苏剑林 · 科学空间 · 2015-03-28
>
> 原文：<https://spaces.ac.cn/archives/3256>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨天一好友问我以下题目，求证：
$$\lim_{n\to\infty} \frac{1^n + 2^n +\dots + n^n}{n^n}=\frac{e}{e-1}$$
将解答过程简单记录一下。

### 求解

首先可以注意到，当$n$充分大时，
$$\frac{1^n + 2^n +\dots + n^n}{n^n}=\left(\frac{1}{n}\right)^n+\left(\frac{2}{n}\right)^n+\dots+\left(\frac{n}{n}\right)^n$$
的主要项都集中在最后面那几项，因此，可以把它倒过来计算
$$\begin{aligned}\frac{1^n + 2^n +\dots + n^n}{n^n}=&\left(\frac{1}{n}\right)^n+\left(\frac{2}{n}\right)^n+\dots+\left(\frac{n}{n}\right)^n\\
=&\left(\frac{n}{n}\right)^n+\dots+\left(\frac{2}{n}\right)^n+\left(\frac{1}{n}\right)^n\end{aligned}$$
而我们有
$$\left(\frac{n-i}{n}\right)^n=\left(1-\frac{i}{n}\right)^n$$
是关于$n$的增函数，因此有
$$\left(\frac{n-i}{n}\right)^n \leq \lim_{n\to\infty} \left(1-\frac{i}{n}\right)^n =e^{-i}$$
从而
$$\left(\frac{n}{n}\right)^n+\dots+\left(\frac{2}{n}\right)^n+\left(\frac{1}{n}\right)^n \leq e^0 + e^{-1}+e^{-2}+\dots=\frac{e}{e-1}$$
恒成立。

另一边，主要是找到$\left(\frac{n-i}{n}\right)^n$的下界估计。我们有
$$\begin{aligned}&\ln\left[\left(\frac{n-i}{n}\right)^n\right]\\
=&n\ln\left(1-\frac{i}{n}\right)\\
=&n\left[-\frac{i}{n}-\frac{1}{2}\left(\frac{i}{n}\right)^2-\frac{1}{3}\left(\frac{i}{n}\right)^3-\dots\right]\end{aligned}$$

注意到我们有不等式$\ln(1-x) = -x -\frac{1}{2}x^2-\frac{1}{3}x^3-\dots > -x-x^2$，当然，这个不等式不是恒成立的，但是它在$x$较小时成立，可以粗略地估计，它在$0\leq x \leq \frac{1}{2}$时成立，因此，当$i \leq \frac{n}{2}$时，就有
$$\ln\left[\left(\frac{n-i}{n}\right)^n\right]\geq n\left[-\frac{i}{n}-\left(\frac{i}{n}\right)^2\right]=-i-\frac{i^2}{n}$$
或者
$$\left(\frac{n-i}{n}\right)^n \geq e^{-i-i^2/n}\geq e^{-i} \left(1-\frac{i^2}{n}\right)\geq e^{-i} -\frac{i^2}{n}$$

好了，我们已经知道$\sum_{i=0}^m i^2 \sim m^3$，故我们只取$i < n^{1/4}$的部分，也就是说
$$\begin{aligned}&\left(\frac{n}{n}\right)^n+\dots+\left(\frac{2}{n}\right)^n+\left(\frac{1}{n}\right)^n\\
\geq &\sum_{i=0}^{\left\lfloor n^{1/4} \right\rfloor}\left(1-\frac{i}{n}\right)^n\\
\geq &\sum_{i=0}^{\left\lfloor n^{1/4} \right\rfloor} \left(e^{-i} -\frac{i^2}{n}\right)\\
=&\frac{1-e^{-\left\lfloor n^{1/4} \right\rfloor-1}}{1-e^{-1}}-\lambda \frac{\left\lfloor n^{1/4} \right\rfloor ^3}{n}\end{aligned}$$

$\lambda$是一个常数，我们不需要知道它的具体值。取极限，后一项就为0了，得
$$\lim_{n\to\infty}\left(\frac{1}{n}\right)^n+\left(\frac{2}{n}\right)^n+\dots+\left(\frac{n}{n}\right)^n \geq\frac{e}{e-1}$$

结合另一端的不等式，就有
$$\lim_{n\to\infty}\left(\frac{1}{n}\right)^n+\left(\frac{2}{n}\right)^n+\dots+\left(\frac{n}{n}\right)^n = \frac{e}{e-1}$$

整个过程主要就是放缩，而且根据我们的需要随心所欲的放缩——当然，这与题目本身比较弱有关。文中的过程不是最简洁漂亮的，却是相当实用的！
