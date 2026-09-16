# 流形上的最速下降：4\. Muon \+ 谱球面

> 作者：苏剑林 · 科学空间 · 2025-08-21
>
> 原文：<https://spaces.ac.cn/archives/11241>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

看完了前三篇的读者，想必已经熟悉本系列的“套路”——先提出更新量的约束，寻找最速下降方向，接着再给参数也加上约束，寻找新的最速下降方向。在求解参数约束问题时，我们采用的是“一阶近似够用”原则来简化约束形式，这在几何上对应于“切空间”。然后，我们用待定系数法转化无约束形式来写出解析解，最后再数值求解待定系数。

这篇文章我们再来求解一个新例子——谱球面约束下的Muon——它是第一篇文章[《流形上的最速下降：1\. SGD \+ 超球面》](<https://spaces.ac.cn/archives/11196>)的类比推广，当我们希望参数的谱范数始终不变时可以考虑它。当然，也可以单纯作为一道练习题来练手。

## 问题描述

在[《流形上的最速下降：2\. Muon \+ 正交》](<https://spaces.ac.cn/archives/11215>)和[《流形上的最速下降：3\. Muon \+ Stiefel》](<https://spaces.ac.cn/archives/11221>)中，我们已经详细讨论了Muon与正交约束的碰撞，所以相关背景我们就不展开了，直接给出问题形式：
\begin{equation}\newcommand{tr}{\mathop{\text{tr}}}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \Vert\boldsymbol{W}\Vert_2 = 1,\,\,\Vert\boldsymbol{W} - \eta \boldsymbol{\Phi}\Vert_2=1\end{equation}
其中$\boldsymbol{W},\boldsymbol{\Phi}\in\mathbb{R}^{n\times m}(n \geq m)$，$\Vert\cdot\Vert_2$是谱范数。当然，如果我们有兴趣，后面两个谱范数可以改用其他范数，比如$F$范数代表了“Muon \+ 超球面”的组合。

“一阶近似够用”原则需要我们求谱范数的梯度，这我们在[《从谱范数梯度到新式权重衰减的思考》](<https://spaces.ac.cn/archives/10648>)和[《SVD的导数》](<https://spaces.ac.cn/archives/10878>)都已经介绍过了，答案是$\nabla_{\boldsymbol{W}}\Vert\boldsymbol{W}\Vert_2=\boldsymbol{u}_1 \boldsymbol{v}_1^{\top}$，其中$\boldsymbol{u}_1,\boldsymbol{v}_1$是$\boldsymbol{W}$的最大奇异值对应的两个奇异向量，可以由幂迭代求解。这个结果还有个最大奇异值唯一的假设，不唯一的情况我们后面再讨论。

如果是$F$范数，则有$\nabla_{\boldsymbol{W}}\Vert\boldsymbol{W}\Vert_F=\boldsymbol{W}/\Vert\boldsymbol{W}\Vert_F$。总之，不管是哪种范数，都存在一个只依赖于$\boldsymbol{W}$的矩阵$\boldsymbol{\Theta}$，使得$\nabla_{\boldsymbol{W}}\Vert\boldsymbol{W}\Vert=\boldsymbol{\Theta}$，那么由$\Vert\boldsymbol{W}\Vert = 1$和$\Vert\boldsymbol{W} - \eta \boldsymbol{\Phi}\Vert=1$可以得到它的一阶近似版$0 = \langle\boldsymbol{\Theta},\boldsymbol{\Phi}\rangle_F = \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})$。所以，在一阶近似下，此类问题的通用提法是：
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0\end{equation}

## 待定系数

套路还是一样的，引入待定系数$\lambda$，我们有
\begin{equation}\begin{aligned}
\tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) =&\, \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) + \lambda \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi}) \\
=&\, \tr((\boldsymbol{G} + \lambda\boldsymbol{\Theta})^{\top}\boldsymbol{\Phi}) \\
\leq &\,\Vert\boldsymbol{G} + \lambda\boldsymbol{\Theta}\Vert_*
\end{aligned}\end{equation}
最后一个不等式是Muon本身的结果，类似于两个向量的Hölder不等式，等号在
\begin{equation}\boldsymbol{\Phi} = \newcommand{msign}{\mathop{\text{msign}}}\msign(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\end{equation}
接下来的任务是需要解出一个$\lambda$，使得它满足约束条件$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$，就大功告成了。

由于$\msign$的存在，$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0$实际上是一个非线性方程，笔者倾向于它没有解析解，所以寻求数值解法。不过有了[《流形上的最速下降：3\. Muon \+ Stiefel》](<https://spaces.ac.cn/archives/11221>)的经验后，面对此类方程我们也可以从容地构建迭代法了。

## 迭代求解

首先，根据定义$\msign(\boldsymbol{M}) = \boldsymbol{M}(\boldsymbol{M}^{\top}\boldsymbol{M})^{-1/2}$，我们可以写出$\boldsymbol{\Phi}=(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}$，其中$\boldsymbol{Q}=((\boldsymbol{G} + \lambda\boldsymbol{\Theta})^{\top}(\boldsymbol{G} + \lambda\boldsymbol{\Theta}))^{1/2}$，那么
\begin{equation}\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})=0\qquad\Rightarrow\qquad \lambda = -\frac{\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{G}\boldsymbol{Q}^{-1})}{\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta}\boldsymbol{Q}^{-1})}\end{equation}
但要注意，这并不是一个解析解，因为$\boldsymbol{Q}$也是依赖于$\lambda$的。但根据上式我们可以构建一个迭代格式：代入一个初始$\lambda$就可以求$\boldsymbol{Q}= (\boldsymbol{G} + \lambda\boldsymbol{\Theta})^{\top}\boldsymbol{\Phi}$，然后代入上式更新$\lambda$，反复执行直到收敛。

然而，这个迭代方案虽然理论上可行，但需要算$\boldsymbol{Q}^{-1}$，尽管我们在[《矩阵r次方根和逆r次方根的高效计算》](<https://spaces.ac.cn/archives/11175>)已经提过高效的求解算法，但从“勿增实体”的角度看，我们还是希望避免出现$\msign$以外的迭代。所以，笔者尝试寻找另外一个不需要求$\boldsymbol{Q}^{-1}$的迭代方案。为此，我们先写出
\begin{equation}\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi} = \boldsymbol{\Theta}^{\top}(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}\end{equation}
对于我们的目标，上式的迹等于零，我们可以在上式左边显式地减去$\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m$，保证它满足这个条件
\begin{equation}\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi} - \tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m = \boldsymbol{\Theta}^{\top}(\boldsymbol{G} + \lambda\boldsymbol{\Theta})\boldsymbol{Q}^{-1}\end{equation}
这时候两边乘$\boldsymbol{Q}$，然后取迹就可以反解出$\lambda$
\begin{equation}\lambda = \frac{\tr(\boldsymbol{\Theta}^{\top} \boldsymbol{\Phi}\boldsymbol{Q}) - \tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Phi}) \tr(\boldsymbol{Q})/m - \tr(\boldsymbol{\Theta}^{\top}\boldsymbol{G})}{\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})}\end{equation}
这样迭代起来就不用求$\boldsymbol{Q}^{-1}$了。

## 参考代码

我们以$\lambda=-\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{G})/\tr(\boldsymbol{\Theta}^{\top}\boldsymbol{\Theta})$为初始值，测试代码如下：



<pre><code class="language-python">import numpy as np

def msign(g):
    """奇异值分解精确计算msign
    """
    u, s, vh = np.linalg.svd(g, full_matrices=False)
    return u @ np.diag(np.sign(s)) @ vh

def dot(a, b):
    """恒等于 np.trace(a.T @ b)
    """
    return (a * b).sum()

n, m = 100, 50
w = np.random.randn(n, m) / m**0.5
g = np.random.randn(n, m) / m**0.5
u, s, vh = np.linalg.svd(w, full_matrices=False)
theta = u[:, :1] @ vh[:1]

lamb = - dot(theta, g) / dot(theta, theta)
for i in range(10):
    phi = msign(z := g + lamb * theta)
    print('step:', i, ', inner product:', dot(phi, g), ', tangent error:', dot(theta, phi))
    q, x = z.T @ phi, theta.T @ phi
    lamb = (dot(x, q) - np.trace(x) * np.trace(q) / m - dot(theta, g)) / dot(theta, theta)</code></pre>



## 其他细节

同前三篇文章一样，由于使用了“一阶近似够用”原则，所以$\boldsymbol{W} - \eta\boldsymbol{\Phi}$的谱范数准确到$1 + \mathcal{O}(\eta^2)$，通常没法精确到1，所以我们还需要做一次谱归一化（Spectral Normalization）：
\begin{equation}\boldsymbol{W}\quad\leftarrow\quad \frac{\boldsymbol{W} - \eta\boldsymbol{\Phi}}{\Vert\boldsymbol{W} - \eta\boldsymbol{\Phi}\Vert_2}\end{equation}
幸运的是，谱范数可以通过幂迭代来高效计算，所以这并不是特别昂贵的计算（相比$\msign$本身的迭代来说）。

另外可以值得分析一番的是最大奇异值不唯一的情形，实际数值计算时可以忽略这种特殊情形，但从理论的完整性来说应该将它纳入分析范围内。这时候对应的奇异向量也不唯一，等价于说有多个不同的切空间，实际可行空间是这些切空间的交集。我们以两个最大奇异值为例，此时问题变成
\begin{equation}\max_{\boldsymbol{\Phi}} \tr(\boldsymbol{G}^{\top}\boldsymbol{\Phi}) \qquad \text{s.t.}\qquad \Vert\boldsymbol{\Phi}\Vert_2 = 1,\,\, \tr(\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi})=0,\,\, \tr(\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi})=0\end{equation}
这里$\boldsymbol{\Theta}_1=\boldsymbol{u}_1 \boldsymbol{v}_1^{\top}, \boldsymbol{\Theta}_2=\boldsymbol{u}_2 \boldsymbol{v}_2^{\top}$。引入两个待定系数$\lambda_1,\lambda_2$，我们可以解得
\begin{equation}\boldsymbol{\Phi} = \msign(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)\end{equation}
接下来要求解方程组$\tr(\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi})=0,\tr(\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi})=0$。类似地，引入
\begin{equation}\boldsymbol{Q}=((\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)^{\top}(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2))^{1/2} = (\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)^{\top}\boldsymbol{\Phi}\end{equation}
可以写出方程组
\begin{equation}\begin{gathered}
\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi} - \tr(\boldsymbol{\Theta}_1^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m = \boldsymbol{\Theta}_1^{\top}(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)\boldsymbol{Q}^{-1} \\
\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi} - \tr(\boldsymbol{\Theta}_2^{\top} \boldsymbol{\Phi})\boldsymbol{I}/m = \boldsymbol{\Theta}_2^{\top}(\boldsymbol{G} + \lambda_1\boldsymbol{\Theta}_1+ \lambda_2\boldsymbol{\Theta}_2)\boldsymbol{Q}^{-1} \\
\end{gathered}\end{equation}
两边乘$\boldsymbol{Q}$然后取迹，就变成了关于可以求解的$\lambda_1,\lambda_2$的二元一次方程组，那么就可以据此来构建迭代求解格式了。其中细节就不展开讨论了，有兴趣练手的读者自行补充完整就好。

## 文章小结

这篇文章主要考虑了给参数施加谱范数或者一般的范数约束后，对应的Muon形式。在前三篇的基础上，本篇没有明显的技术难点，读者也可以单纯视为一道补充习题来练手。
