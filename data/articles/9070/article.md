# logsumexp运算的几个不等式

> 作者：苏剑林 · 科学空间 · 2022-05-10
>
> 原文：<https://spaces.ac.cn/archives/9070>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

$\text{logsumexp}$是机器学习经常遇到的运算，尤其是交叉熵的相关实现和推导中都会经常出现，同时它还是$\max$的光滑近似（参考[《寻求一个光滑的最大值函数》](<https://spaces.ac.cn/archives/3290>)）。设$x=(x_1,x_2,\cdots,x_n)$，$\text{logsumexp}$定义为
\begin{equation}\text{logsumexp}(x)=\log\sum_{i=1}^n e^{x_i}\end{equation}
本文来介绍$\text{logsumexp}$的几个在理论推导中可能用得到的不等式。

## 基本界

记$x_{\max} = \max(x_1,x_2,\cdots,x_n)$，那么显然有
\begin{equation}e^{x_{\max}} < \sum_{i=1}^n e^{x_i} \leq \sum_{i=1}^n e^{x_{\max}} = ne^{x_{\max}}\end{equation}
各端取对数即得
\begin{equation}x_{\max} < \text{logsumexp}(x) \leq x_{\max} + \log n\end{equation}
这是关于$\text{logsumexp}$上下界的最基本结果，它表明$\text{logsumexp}$对$\max$的近似误差不超过$\log n$。注意这个误差跟$x$本身无关，于是我们有
\begin{equation}x_{\max}/\tau < \text{logsumexp}(x/\tau) \leq x_{\max}/\tau + \log n\end{equation}
各端乘以$\tau$得到
\begin{equation}x_{\max} < \tau\text{logsumexp}(x/\tau) \leq x_{\max} + \tau\log n\end{equation}
当$\tau\to 0$时，误差就趋于0了，这告诉我们可以通过降低温度参数来提高对$\max$的近似程度。

## 平均界

我们知道$e^x$是凸函数，满足[詹森不等式](<https://en.wikipedia.org/wiki/Jensen%27s_inequality>)$\mathbb{E}[e^{x}]\geq e^{\mathbb{E}[x]}$，因此
\begin{equation}\frac{1}{n}\sum_{i=1}^n e^{x_i}\geq e^{\bar{x}}\end{equation}
这里$\bar{x}=\frac{1}{n}\sum\limits_{i=1}^n x_i$，两边乘以$n$后取对数得
\begin{equation}\text{logsumexp}(x)\geq \bar{x} + \log n\end{equation}
这是关于$\text{logsumexp}$下界的另一个结果。该结果可以进一步推广到加权平均的情形：设有$p_1,p_2,\cdots,p_n\geq 0$且$\sum\limits_{i=1}^n p_i = 1$，由柯西不等式得
\begin{equation}\left[\sum_{i=1}^n (e^{x_i/2})^2\right]\left[\sum_{i=1}^n p_i^2\right]\geq \left[\sum_{i=1}^n p_i e^{x_i/2}\right]^2\end{equation}
对右端方括号内的式子应用詹森不等式得到
\begin{equation}\left[\sum_{i=1}^n p_i e^{x_i/2}\right]^2\geq \left[e^{\left(\sum\limits_{i=1}^n p_i x_i/2\right)}\right]^2 = e^{\left(\sum\limits_{i=1}^n p_i x_i\right)}\end{equation}
各式两端取对数，整理得到
\begin{equation}\text{logsumexp}(x)\geq \sum_{i=1}^n p_i x_i  - \log\sum_{i=1}^n p_i^2\end{equation}
如果开始不用柯西不等式而是用更一般的[Hölder不等式](<https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality>)，那么还可以得到
\begin{equation}\text{logsumexp}(x)\geq \sum_{i=1}^n p_i x_i  - \frac{1}{t-1}\log\sum_{i=1}^n p_i^t,\quad \forall t > 1\end{equation}
特别地，取$t\to 1$的极限，我们可以得到
\begin{equation}\text{logsumexp}(x)\geq \sum_{i=1}^n p_i x_i  - \sum_{i=1}^n p_i \log p_i\end{equation}
它可以等价地改写为$\sum\limits_{i=1}^n p_i \log \frac{p_i}{e^{x_i}/Z} \geq 0$，其中$Z=e^{\text{logsumexp}(x)}$是归一化因子，所以它实际就是两个分布的$KL$散度。

## L约束

在无穷范数下，$\text{logsumexp}$还满足Lipschitz约束，即
\begin{equation}|\text{logsumexp}(x) - \text{logsumexp}(y)| \leq |x - y|_{\infty}\end{equation}
这里的$|x-y|_{\infty} = \max\limits_i |x_i - y_i|$（其实记为$|x - y|_{\max}$还更直观一些）。证明也不算困难，定义
\begin{equation}f(t) = \text{logsumexp}(tx + (1-t)y),\quad t\in[0, 1]\end{equation}
将它视为关于$t$的一元函数，由[中值定理](<https://en.wikipedia.org/wiki/Mean_value_theorem>)知存在$\varepsilon\in(0, 1)$，使得
\begin{equation}f'(\varepsilon) = \frac{f(1) - f(0)}{1 - 0} = \text{logsumexp}(x) - \text{logsumexp}(y) \end{equation}
不难求出
\begin{equation}f'(\varepsilon) = \frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}(x_i - y_i)}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}} \end{equation}
所以
\begin{equation}\begin{aligned}&\,|\text{logsumexp}(x) - \text{logsumexp}(y)| = \left|\frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}(x_i - y_i)}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}}\right| \\
\leq &\, \frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i} |x_i - y_i|}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}} \leq \frac{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i} |x - y|_{\infty}}{\sum\limits_{i=1}^n e^{\varepsilon x_i + (1-\varepsilon)y_i}} = |x - y|_{\infty}
\end{aligned}\end{equation}

## 凸函数

最后是一个很强的结论：$\text{logsumexp}$还是一个凸函数！这意味着凸函数相关的所有不等式都适用于$\text{logsumexp}$，比如最基本的詹森不等式：
\begin{equation} \mathbb{E}[\text{logsumexp}(x)] \geq \text{logsumexp}(\mathbb{E}[x])\end{equation}

要证明$\text{logsumexp}$是凸函数，就是要证明对于$\forall t\in[0, 1]$，都成立
\begin{equation} t\text{logsumexp}(x) + (1-t)\text{logsumexp}(y)\geq \text{logsumexp}(tx + (1-t)y)\end{equation}
证明过程其实就是[Hölder不等式](<https://en.wikipedia.org/wiki/H%C3%B6lder%27s_inequality>)的基本应用。具体来说，我们有
\begin{equation}t\text{logsumexp}(x) + (1-t)\text{logsumexp}(y) = \log\left(\sum_{i=1}^n e^{x_i}\right)^t \left(\sum_{i=1}^n e^{y_i}\right)^{(1-t)}\end{equation}
现在直接应用Hölder不等式就可以得到
\begin{equation}\log\left(\sum_{i=1}^n e^{x_i}\right)^t \left(\sum_{i=1}^n e^{y_i}\right)^{(1-t)}\geq \log\sum_{i=1}^n e^{tx_i + (1-t)y_i} = \text{logsumexp}(tx + (1-t)y)\end{equation}
这就证明了$\text{logsumexp}$是凸函数。

## 文末结

主要总结了$\text{logsumexp}$运算的相关不等式，以备不时之需。
