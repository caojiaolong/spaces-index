# 圆内随机n点在同一个圆心角为θ的扇形的概率

> 作者：苏剑林 · 科学空间 · 2022-10-25
>
> 原文：<https://spaces.ac.cn/archives/9324>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这几天网上热传了一道“四鸭共半圆”题目：

[![四鸭共半圆问题](<https://spaces.ac.cn/usr/uploads/2022/10/3296253276.png>)](<https://spaces.ac.cn/usr/uploads/2022/10/3296253276.png>)

四鸭共半圆问题

可能有不少读者看到后也尝试做过，就连李永乐老师也专门开了一节课讲这道题（参考[《圆形水池四只鸭子在同一个半圆里，概率有多大？》](<https://www.bilibili.com/read/cv19251490>)）。就这道题目本身而言，答案并不算困难，可以有很多方法算出来。稍微有难度的是它的推广版本，也就是本文标题所描述的，将鸭子的数目一般化为$n$只，将半圆一般化为圆心角为$\theta$的扇形。更有趣的是，当$\theta \leq \pi$时，依然有比较初等的解法，但是当$\theta > \pi$后，复杂度开始“剧增”\.\.\.

## 题目转换

首先要说明的是，这里我们是将鸭子抽象为圆内均匀分布随机抽取的点来处理的，那些诸如“鸭子的占地面积”等推广这里我们就不考虑了。就“四鸭共半圆”而言，我们很容易去将它一般化，比如推广到高维空间中：

> $d$维超球内均匀分布的$n$个点，它们位于同一个$d$维超半球的概率是多少？

这个推广的答案其实在1962年发表的论文[《A Problem in Geometric Probability》](<https://www.mscand.dk/article/view/10655/8676>)就给出了。另一个推广就是本文的标题：

> 圆内均匀分布的$n$个点，它们位于同一个圆心角为$\theta$的扇形的概率是多少？

本文主要就是讨论这个问题，它还可以等价地转换成（怎么转换请读者自行思考）：

> 圆周上均匀随机地选取$n$个点，将圆周分为$n$段圆弧，最长的圆弧所对的圆心角大于$2\pi - \theta$的概率是多少？

进一步地，可以等价地转换为

> 单位长线段上均匀随机地选取$n-1$个点，将线段分为$n$段，最长的线段长度大于$1 - \frac{\theta}{2\pi}$的概率是多少？

## 分布求解

其实，最后一个等价表述所涉及的分布，已经被很好地讨论过了，比如知乎上的[《在 (0, 1) 内随机取点将区间分成 n 段，最长段的长度期望是多少？》](<https://www.zhihu.com/question/50198685/answer/1509793840>)。这里为了方便大家理解，笔者重新组织语言介绍一遍，将会分为几个步骤，整个过程有点长，但这是求通解（适用于$\theta > \pi$）必须的。

顺便指出的是，多个随机变量的最值，属于“[顺序统计量（Order Statistic）](<https://en.wikipedia.org/wiki/Order_statistic>)”之一，在机器学习中也颇为常见，比如[《EAE：自编码器 \+ BN \+ 最大熵 = 生成模型》](<https://spaces.ac.cn/archives/7343#%E7%86%B5%E7%9A%84%E9%87%87%E6%A0%B7%E9%82%BB%E8%BF%91%E4%BC%B0%E8%AE%A1>)介绍的熵的最邻近估计、[《从重参数的角度看离散概率分布的构建》](<https://spaces.ac.cn/archives/9085>)介绍的离散分布的一般构建，都可归结为此类。

### 联合分布

从最后一个等价表述出发，设单位长线段被随机的$n-1$个点分为的$n$段，从左到右的长度依次为$x_1,x_2,\cdots,x_n$（注：只是定了个方向，并没有按大小排列），记它们的联合分布的概率密度函数为$p_n(x_1,x_2,\cdots,x_n)$。知乎链接中的答案是直接基于“$p_n(x_1,x_2,\cdots,x_n)$是均匀分布”这一事实来进行后续求解的，但不知道是不是笔者漏了什么细节，还是哪里没绕过弯，笔者觉得“$p_n(x_1,x_2,\cdots,x_n)$是均匀分布”并不是一件显然成立的事情，所以这里对它进行推导一遍。

首先，留意到两个事实：

> 1、$x_1,x_2,\cdots,x_n$带有约束$x_1 + x_2 + \cdots + x_n = 1$，所以它只有$n-1$个自由度，我们取前$n-1$个变量为自由变量；
> 
> 
> 
> 2、$p_n(x_1,x_2,\cdots,x_n)$是概率密度而非概率，$p_n(x_1,x_2,\cdots,x_n)dx_1 dx_2 \cdots dx_{n-1}$才是概率。

为了理解“$p_n(x_1,x_2,\cdots,x_n)$是均匀分布”这一事实，我们从$n=2$出发，此时就是在$(0, 1)$中均匀随机取一个点将线段分为两部分，由于取点是均匀分布的（概率密度是1），取点跟$(x_1, x_2)$一一对应，所以也有$p_2(x_1, x_2)=1$。

接着考虑$n=3$，对应的概率是$p_3(x_1, x_2, x_3)dx_1 dx_2$，它有两种可能：

> 1、先采样一个点，将线段分为$(x_1, x_2 + x_3)$两段，这部分概率为$p_2(x_1, x_2 + x_3) dx_1$，然后再采样一个点，将$x_2 + x_3$这一段分为$(x_2, x_3)$两段，这部分概率为$dx_2$，所以乘起来是$p_2(x_1, x_2 + x_3) dx_1 dx_2$；
> 
> 
> 
> 2、先采样一个点，将线段分为$(x_1 + x_2, x_3)$两段，这部分概率为$p_2(x_1 + x_2, x_3) dx_2$，然后再采样一个点，将$x_1 + x_2$这一段分为$(x_1, x_2)$两段，这部分概率为$dx_1$，所以乘起来是$p_2(x_1 + x_2, x_3) dx_1 dx_2$；

两者加起来是
\begin{equation}p_3(x_1, x_2, x_3)dx_1 dx_2 = p_2(x_1, x_2 + x_3) dx_1 dx_2 + p_2(x_1 + x_2, x_3) dx_1 dx_2 = 2dx_1 dx_2\end{equation}
即$p_3(x_1, x_2, x_3)=2$也是一个均匀分布。类似地递推，可以得到
\begin{equation}p_n(x_1,x_2,\cdots,x_n) = (n - 1) p_{n-1}(x_1,x_2,\cdots,x_{n-1}) = \cdots = (n - 1)!\end{equation}

### 边缘分布

有了联合分布，那么我们就可以求出任选$k$个变量的边缘分布了，这是为下一节使用“容斥原理”作准备的。由于联合分布就是一个均匀分布，所以各个变量是全对称的，因此不失一般性，我们求前$k$个变量的边缘分布即可。

根据定义
\begin{equation}p_n(x_1,x_2,\cdots,x_k) = \int \cdots \int p_n(x_1,x_2,\cdots,x_n) dx_{k+1}\cdots dx_{n-1}\end{equation}
要注意的是积分上下限，下限自然是$0$，因为约束$x_1 + x_2 + \cdots + x_n = 1$的存在，上限对于每个变量都是不一样的，给定$x_1,x_2,\cdots,x_i$后，$x_{i+1}$的取值就只能是$0\sim 1 - (x_1 + x_2 + \cdots + x_i)$，所以准确形式是
\begin{equation}\int_0^{1 - (x_1 + x_2 + \cdots + x_k)} \cdots \left[\int_0^{1 - (x_1 + x_2 + \cdots + x_{n-2})} p_n(x_1,x_2,\cdots,x_n) dx_{n-1}\right] \cdots dx_{k+1}\end{equation}
由于$p_n(x_1,x_2,\cdots,x_n)=(n-1)!$只是一个常数，所以上式是可以逐次积分出来的，最终结果是
\begin{equation}p_n(x_1,x_2,\cdots,x_k) = \frac{(n-1)!}{(n - k - 1)!}[1 - (x_1 + x_2 + \cdots + x_k)]^{n-k-1}\end{equation}
这时候我们就可以求出$x_1,x_2,\cdots,x_k$分别大于给定阈值$c_1, c_2, \cdots, c_k$（成立$c_1 + c_2 +  \cdots + c_k \leq 1$）的概率：
\begin{equation}\begin{aligned}
&\, \qquad P_n(x_1 > c_1,x_2 > c_2,\cdots,x_k > c_k) = \int_{c_1}^{1 - (c_2 + c_2 + \cdots + c_k)} \cdots
\\
&\, \left[\int_{c_{k-1}}^{1 - (x_1 + x_2 + \cdots + x_{k-2} + c_k)}\left[\int_{c_k}^{1 - (x_1 + x_2 + \cdots + x_{k-1})} p_n(x_1,x_2,\cdots,x_k) dx_k\right]dx_{k-1}\right] \cdots dx_1
\end{aligned}\end{equation}
积分上限跟前面类似，都是在约束$x_1 + x_2 + \cdots + x_n = 1$下进行推导的。跟$p_n(x_1,x_2,\cdots,x_n)$一样，上式也是可以逐次积分出来的，最终结果反而很简单
\begin{equation}P_n(x_1 > c_1,x_2 > c_2,\cdots,x_k > c_k) = [1 - (c_1 + c_2 + \cdots + c_k)]^{n-1}\end{equation}

### 容斥原理

现在一切准备就绪，轮到“容斥原理”来完成“最后一击”了。别忘了我们的目标是求出最长段长度的概率，即对于某个阈值$x$，计算出$P_n(\max(x_1,x_2,\cdots,x_n) > x)$，而$\max(x_1,x_2,\cdots,x_n) > x$这件事本身是多个事件的并集：
\begin{equation}\max(x_1,x_2,\cdots,x_n) > x \quad\Leftrightarrow\quad x_1 > x \,\color{red}{\text{或}}\, x_2 > x \,\color{red}{\text{或}}\, \cdots \,\color{red}{\text{或}}\, x_n > x \end{equation}
关键是，当$x < \frac{1}{2}$时，各个事件之间并非不相交的，所以不能直接将每一部分的概率简单相加，而需要用到“[容斥原理](<https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle>)”：
\begin{equation}\begin{aligned}
\text{存在一点大于}x\text{的概率} = &\,\text{任一点大于}x\text{的概率之和} \\
&\,\quad \color{red}{-} \text{任两点都大于}x\text{的概率之和} \\
&\, \quad\quad \color{red}{+} \text{任三点都大于}x\text{的概率之和} \\
&\, \quad\quad\quad \color{red}{-} \text{任四点都大于}x\text{的概率之和} \\
&\, \quad\quad\quad\quad \color{red}{+} \cdots
\end{aligned}\end{equation}
根据上一节的结果，任选$k$点，每一点都大于$x$的概率为$(1-kx)^{n-1}$，而任选$k$点的组合数为$C_n^k$，所以代入上式得
\begin{equation}\begin{aligned}
P_n(\max(x_1,x_2,\cdots,x_n) > x) =&\, \sum_{k=1, 1 - kx > 0}^n (-1)^{k-1} C_n^k (1-kx)^{n-1} \\
=&\, C_n^1 (1 - x)^{n-1} - C_n^2 (1 - 2x)^{n-1} + \cdots
\end{aligned}\end{equation}

## 答案分析

对于本文开始的题目来说，即$x = 1 - \frac{\theta}{2\pi}$，当$x > \frac{1}{2}$（即$\theta < \pi$）时，后面的$1 - k x$都小于0了，即实际上不存在，因此答案是最简单的：
\begin{equation}C_n^1 (1 - x)^{n-1} = n \left(\frac{\theta}{2\pi}\right)^{n-1}\end{equation}
当$x < \frac{1}{2}$时，看$x$的具体大小来增减项，$x$越小，项数相对来说越多，这就是李永乐老师文章说的“当$\theta$大于180度时，情况将变得非常复杂”了。

我们也可以求它的期望，这知乎上的提问所讨论的问题。还有一个有意思的情况，显然当$x < \frac{1}{n}$时，恒有$1 - kx > 0$，即所有项都需要用上了：
\begin{equation}P_n(\max(x_1,x_2,\cdots,x_n) > x) = \sum_{k=1}^n (-1)^{k-1} C_n^k (1-kx)^{n-1}\end{equation}
然而根据“抽屉原理”我们可以得知，将单位长线段分为$n$段后，最长的一段的长度必然是不小于$\frac{1}{n}$的，这意味着$P_n(\max(x_1,x_2,\cdots,x_n) > \frac{1}{n}) = 1$，因此当$x < \frac{1}{n}$时，必然成立
\begin{equation}\sum_{k=1}^n (-1)^{k-1} C_n^k (1-kx)^{n-1} = 1\end{equation}
又因为左端仅仅是简单的代数多项式，因此它必然是解析的，所以上式对于所有的$x$都恒成立。感兴趣的读者，不妨试试直接从代数角度证明它。

## 文章小结

本文讨论了“四鸭共半圆”的推广问题的一般解法，其中主要思想是容斥原理。
