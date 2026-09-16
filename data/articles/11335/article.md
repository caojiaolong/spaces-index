# 随机矩阵的谱范数的快速估计

> 作者：苏剑林 · 科学空间 · 2025-10-12
>
> 原文：<https://spaces.ac.cn/archives/11335>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《高阶MuP：更简明但更高明的谱条件缩放》](<https://spaces.ac.cn/archives/10795>)的“近似估计”一节中，我们曾“预支”了一个结论：“一个服从标准正态分布的$n\times m$大小的随机矩阵，它的谱范数大致是$\sqrt{n}+\sqrt{m}$。”

这篇文章我们来补充讨论这个结论，给出随机矩阵谱范数的快速估计方法。

## 随机矩阵论

设有随机矩阵$\boldsymbol{W}\in\mathbb{R}^{n\times m}$，每个元素都是从标准正态分布$\mathcal{N}(0,1)$中独立重复地采样出来的，要求估计$\boldsymbol{W}$的谱范数，也就是最大奇异值，我们以$\mathbb{E}[\Vert\boldsymbol{W}\Vert_2]$为最终的估计结果。

首先要指出的是，随机矩阵的性态分析已经形成了一个专门的分支，就正态矩阵的谱范数估计来说，关系到关键词包括“[Marchenko–Pastur分布](<https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution>)”、“Bai\-Yin law”、“Gordon’s theorem”等，它们包含了关于奇异值分布的详细估计结果。不过我们都不打算介绍这些内容，而是介绍一种快速得到$\sqrt{n}+\sqrt{m}$的方法。

该方法由笔者基于[《Introduction to the non\-asymptotic analysis of random matrices》](<https://papers.cool/arxiv/1011.3027>)的“5\.3\.1节”简化而来，它实际上只能算是一个“科普”，让大家快速理解到$\sqrt{n}+\sqrt{m}$的来源。严格的证明需要补充很多繁琐且单调的细节，这里就不展开了。

## 谱范数估计

我们的出发点是恒等式
\begin{equation}\Vert\boldsymbol{W}\Vert_2 = \max_{\Vert \boldsymbol{u}\Vert=1, \Vert \boldsymbol{v}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v}\label{eq:w-uv}\end{equation}
其中$\boldsymbol{u}\in\mathbb{R}^n,\boldsymbol{v}\in\mathbb{R}^m$。直接计算$\Vert\boldsymbol{W}\Vert_2$通常都是不容易的，所以很自然地想到寻找某种近似。我们考虑下面两个“半成品”：
\begin{equation}\max_{\Vert \boldsymbol{u}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v}\qquad\qquad \max_{\Vert \boldsymbol{v}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v}\end{equation}
直观来看，相比式$\eqref{eq:w-uv}$，上面两个式子都只完成了一半的工作，我们做个大胆的假设：它们结果也是完整结果的一半。于是，将它们叠加起来，我们认为是最终结果的一个好近似：
\begin{equation}\Vert\boldsymbol{W}\Vert_2 = \max_{\Vert \boldsymbol{u}\Vert=1, \Vert \boldsymbol{v}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v} \approx \max_{\Vert \boldsymbol{u}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v} + \max_{\Vert \boldsymbol{v}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v} = \Vert \boldsymbol{W}\boldsymbol{v}\Vert + \Vert \boldsymbol{u}^{\top}\boldsymbol{W}\Vert \label{eq:core-approx}\end{equation}
也就是说，从$\mathbb{R}^n,\mathbb{R}^m$的单位超球面上分别采样$\boldsymbol{u},\boldsymbol{v}$，然后根据上式来得到$\boldsymbol{W}$谱范数的一个近似。

## 计算期望值

有了这个近似，我们就可以计算
\begin{equation}\mathbb{E}[\Vert\boldsymbol{W}\Vert_2]\approx\mathbb{E}[\Vert \boldsymbol{W}\boldsymbol{v}\Vert] + \mathbb{E}[\Vert \boldsymbol{u}^{\top}\boldsymbol{W}\Vert] \approx \sqrt{\mathbb{E}[\Vert \boldsymbol{W}\boldsymbol{v}\Vert^2]} + \sqrt{\mathbb{E}[\Vert \boldsymbol{u}^{\top}\boldsymbol{W}\Vert^2]}\end{equation}
其中
\begin{equation}\mathbb{E}[\Vert \boldsymbol{W}\boldsymbol{v}\Vert^2] = \mathbb{E}[ \boldsymbol{v}^{\top}\boldsymbol{W}^{\top}\boldsymbol{W}\boldsymbol{v}] = \boldsymbol{v}^{\top}\mathbb{E}[\boldsymbol{W}^{\top}\boldsymbol{W}]\boldsymbol{v} = \boldsymbol{v}^{\top}(n\boldsymbol{I}_m)\boldsymbol{v} = n\end{equation}
同理$\mathbb{E}[\Vert \boldsymbol{u}^{\top}\boldsymbol{W}\Vert^2]=m$，所以
\begin{equation}\mathbb{E}[\Vert\boldsymbol{W}\Vert_2]\approx\sqrt{n} + \sqrt{m}\end{equation}
这个结果其实非常准（可以通过模拟实验来验证它），具体来说，如果$n=ka,m=kb$，$a,b$都是常数，那么
\begin{equation}\lim_{k\to\infty} \frac{\Vert\boldsymbol{W}\Vert_2}{\sqrt{n} + \sqrt{m}} = 1,\qquad \boldsymbol{W}\sim\mathcal{N}(0,1)\end{equation}
之所以如此准确，是因为我们作了弊——最关键的式$\eqref{eq:core-approx}$本质上是从已知的正确答案反向推测出来的。除了式$\eqref{eq:core-approx}$外，我们还用到的条件只有$\mathbb{E}[\boldsymbol{W}^{\top}\boldsymbol{W}]=n\boldsymbol{I}_m$和$\mathbb{E}[\boldsymbol{W}\boldsymbol{W}^{\top}]=m\boldsymbol{I}_n$，因此可以认为上述近似对于任意0均值、1方差的分布都成立。

## 最小奇异值

谱范数是最大奇异值，事实上，我们还可以用同样的思路估计最小奇异值。当然，这里的最小需要更严格定义一下，设$n\geq m$，这里的最小奇异值，指的是$\boldsymbol{W}$从大到小的第$m$个奇异值，它等于
\begin{equation}\sigma_{\min}(\boldsymbol{W}) = \min_{\Vert \boldsymbol{v}\Vert=1}\max_{\Vert \boldsymbol{u}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v}\end{equation}
注意这里的$\min$和$\max$的位置和对象都不能交换。类似地，我们考虑分别$\min$和$\max$的两个“半成品”之和，作为它的近似
\begin{equation}\sigma_{\min}(\boldsymbol{W}) = \min_{\Vert \boldsymbol{v}\Vert=1}\max_{\Vert \boldsymbol{u}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v}\approx \min_{\Vert \boldsymbol{v}\Vert=1}\boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v} + \max_{\Vert \boldsymbol{u}\Vert=1} \boldsymbol{u}^{\top}\boldsymbol{W} \boldsymbol{v} = -\Vert \boldsymbol{u}^{\top}\boldsymbol{W}\Vert + \Vert \boldsymbol{W}\boldsymbol{v}\Vert \end{equation}
接下来的事情就跟前面一样了，结果是$\mathbb{E}[\sigma_{\min}(\boldsymbol{W})]\approx\sqrt{n}-\sqrt{m}$。

## 最后的提示

本文提供了估计随机矩阵谱范数的一个快速思路，或者更严格地说，是一个科普式、启发式的讲解，而不是严格的、准确的推导。它存在严格化的可能，但需要补上非常多的理论细节，这些我们全都跳过了。
