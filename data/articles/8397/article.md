# Transformer升级之路：4、二维位置的旋转式位置编码

> 作者：苏剑林 · 科学空间 · 2021-05-10
>
> 原文：<https://spaces.ac.cn/archives/8397>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在之前的文章[《Transformer升级之路：2、博采众长的旋转式位置编码》](<https://spaces.ac.cn/archives/8265>)中我们提出了旋转式位置编码RoPE以及对应的Transformer模型RoFormer。由于笔者主要研究的领域还是NLP，所以本来这个事情对于笔者来说已经完了。但是最近一段时间，Transformer模型在视觉领域也大火，各种Vision Transformer（ViT）层出不穷，于是就有了问题：二维情形的RoPE应该是怎样的呢？

咋看上去，这个似乎应该只是一维情形的简单推广，但其中涉及到的推导和理解却远比我们想象中复杂，本文就对此做一个分析，从而深化我们对RoPE的理解。

## 二维RoPE

什么是二维位置？对应的二维RoPE又是怎样的？它的难度在哪里？在这一节中，我们先简单介绍二维位置，然后直接给出二维RoPE的结果和推导思路，在随后的几节中，我们再详细给出推导过程。

### 二维位置

在NLP中，语言的位置信息是一维的，换句话说，我们需要告诉模型这个词是句子的第几个词；但是在CV中，图像的位置信息是二维的，即我们需要告诉模型这个特征是在第几行、第几列。这里的二维指的是完整描述位置信息需要两个数字，并不是指位置向量的维数。

有读者可能想：简单展平后当作一维的处理不行吗？确实不大行，比如一个$h\times h$的feature map，位置$(x,y)$展平后就变成了$xh + y$，而位置$(x+1,y)$和$(x,y+1)$展平后就分别变成了$xh+y+h$和$xh+y+1$，两者与$xh + y$的差分别是$h$和$1$。然而，按照我们直观的认识，$(x+1,y)$、$(x,y+1)$它们与$(x,y)$的距离应该是一样的才对，但是展平后却得到了不一样的$h$和$1$，这未免就不合理了。

所以，我们需要专门为二维情形设计的位置编码，不能简单地展平为一维来做。

### 标准答案

经过后面的一番推导，得到二维RoPE的一个解为：
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y}=\left(
\begin{array}{cc:cc}
\cos x\theta & -\sin x\theta & 0 & 0 \\
\sin x\theta & \cos x\theta & 0 & 0 \\
\hdashline
0 & 0 & \cos y\theta & -\sin y\theta \\
0 & 0 & \sin y\theta & \cos y\theta \\
\end{array}\right)\label{eq:rope-2d}\end{equation}
其中这个解很容易理解，它是两个一维RoPE组成的分块矩阵，实现上它就是将输入向量分为两半，一半施加$x$的一维RoPE，一半施加$y$的一维RoPE。由此形式我们也不难类比三维、四维等位置的RoPE。

矩阵$\eqref{eq:rope-2d}$是一个正交矩阵，它满足两个关键性质：

> 1、<strong>相对性</strong>：即$\boldsymbol{\mathcal{R}}_{x_1,y_1}^{\top}\boldsymbol{\mathcal{R}}_{x_2,y_2}=\boldsymbol{\mathcal{R}}_{x_2-x_1,y_2-y_1}$，也正是由于这个性质，RoPE才具有通过绝对位置实现相对位置的能力；
> 
> 
> 
> 2、<strong>可逆性</strong>：给定$\boldsymbol{\mathcal{R}}_{x,y}$可以反解出$x,y$，这意味着对位置信息的编码是无损的。

某种意义上来说，式$\eqref{eq:rope-2d}$是满足上述两个性质的最简单解，也就是说，虽然存在略有不同的解满足上述两个性质，但是它们形式上和实现上都相对复杂些。

### 推导思路

事后来看，RoPE其实就是找到了矩阵$\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\ \sin n\theta & \cos n\theta\end{pmatrix}$，使得满足“相对性”条件：
\begin{equation}\boldsymbol{\mathcal{R}}_m^{\top}\boldsymbol{\mathcal{R}}_n=\boldsymbol{\mathcal{R}}_{n-m}\label{eq:re}\end{equation}
所以，不难想到，二维RoPE的基本要求也是满足相对性，即要找到矩阵$\boldsymbol{\mathcal{R}}_{x,y}$，使得它满足二维的相对性条件$\boldsymbol{\mathcal{R}}_{x_1,y_1}^{\top}\boldsymbol{\mathcal{R}}_{x_2,y_2}=\boldsymbol{\mathcal{R}}_{x_2-x_1,y_2-y_1}$。不过，如果仅仅是这个要求的话，可行解就很多了，比如直接让
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y} = \begin{pmatrix}\cos (x+y)\theta & -\sin (x+y)\theta\\ \sin (x+y)\theta & \cos (x+y)\theta\end{pmatrix}\end{equation}
但这个解的问题是，我们无法从$x+y$逆向推出$(x,y)$，这意味着这个选择对位置信息来说是有损的，所以我们需要多一个“可逆性”，保证可以从位置矩阵中无损地重构出原始位置信号。

对此，我们有两个比较自然的途径选择：1、四元数；2、矩阵指数。接下来我们将会逐一介绍它们。

## 四元数

在一维RoPE的推导中，我们主要以复数为工具，而[四元数（Quaternion）](<https://en.wikipedia.org/wiki/Quaternion>)是复数的推广，它同时也保留了复数的很多性质，所以用它来推导二维RoPE也算是一个自然的思路。不过很遗憾，这是一条<u>走不通</u>的途径，但笔者仍然将思考过程放置在此，供大家参考。

### 复数与矩阵

在高中时我们就学习过，复数$a+b\boldsymbol{i}$跟二维向量$(a,b)$是一一对应的（为了跟后面的四元数对齐，这里把虚数单位$\boldsymbol{i}$也加粗了），但这个对应关系只能保持加减法对应（因为向量没有通用的乘法运算）。更精妙的对应是将复数与矩阵对应
\begin{equation}a+b\boldsymbol{i} \quad \leftrightarrow \quad \begin{pmatrix} a & -b \\ b & a \end{pmatrix}\end{equation}
在这个映射下，复数的加减乘除与矩阵的加减乘除一一对应，比如
\begin{equation}\begin{array}{ccc}
(a+b\boldsymbol{i})(c+d\boldsymbol{i}) &=& (ac - bd) + (ad + bc)\boldsymbol{i} \\[5pt]
\begin{pmatrix} a & -b \\ b & a \end{pmatrix}\begin{pmatrix} c & -d \\ d & c \end{pmatrix} &=& \begin{pmatrix} ac - bd & - ad - bc \\ ad + bc & ac - bd \end{pmatrix}
\end{array}\end{equation}
所以，矩阵映射才是复数域的完全同构，而向量映射只是直观的几何理解。

复数的矩阵映射也是RoPE的重要基础，在[《Transformer升级之路：2、博采众长的旋转式位置编码》](<https://spaces.ac.cn/archives/8265>)中我们已经推导得到RoPE的复数表示是$\boldsymbol{q}e^{n\boldsymbol{i}\theta}=(\cos n\theta + \boldsymbol{i}\sin n\theta)\boldsymbol{q}$，所以根据复数的矩阵映射，$\cos n\theta + \boldsymbol{i}\sin n\theta$就对应着矩阵
\begin{equation}\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\ \sin n\theta & \cos n\theta\end{pmatrix}\end{equation}
从而得到了一维RoPE的矩阵形式。

### 四元数简介

前面说了，四元数是复数的一种推广，事实上它还是矩阵的“鼻祖”，从历史上来说，是先有四元数然后才有一般的矩阵运算，并且四元数启发了矩阵的很多运算。早年笔者也写了[《与向量的渊源极深的四元数》](<https://spaces.ac.cn/archives/898>)、[《几何的数与数的几何：超复数的浅探究》](<https://spaces.ac.cn/archives/2291>)等文章来介绍四元数，欢迎读者参考。

如果说复数是一个二维向量，那么四元数就是一个四维向量，表示为$a+b\boldsymbol{i}+c\boldsymbol{j}+d\boldsymbol{k}$，这里的$\boldsymbol{i}^2=\boldsymbol{j}^2=\boldsymbol{k}^2=-1$，但是它们本身都各不相等，几个基之间的运算规则是：
\begin{array}{c|cccc}
\times & 1 & \boldsymbol{i} & \boldsymbol{j} & \boldsymbol{k} \\
\hline
1 & 1 & \boldsymbol{i} & \boldsymbol{j} & \boldsymbol{k} \\
\boldsymbol{i} & \boldsymbol{i} & -1 & \boldsymbol{k} & -\boldsymbol{j} \\
\boldsymbol{j} & \boldsymbol{j} & -\boldsymbol{k} & -1 & \boldsymbol{i} \\
\boldsymbol{k} & \boldsymbol{k} & \boldsymbol{j} & -\boldsymbol{i} & -1 \\
\end{array}
在当时它给人们最大的冲击是非交换性，比如$\boldsymbol{i}\boldsymbol{j}=-\boldsymbol{j}\boldsymbol{i}\neq \boldsymbol{j}\boldsymbol{i}$。但除此之外，它跟复数运算其实是高度相似的。

比如类似复数的欧拉公式
\begin{equation}e^{a+b\boldsymbol{i}+c\boldsymbol{j}+d\boldsymbol{k}} = e^a\left(\cos r + \frac{b\boldsymbol{i}+c\boldsymbol{j}+d\boldsymbol{k}}{r}\sin r\right)\label{eq:euler}\end{equation}
这里$r = \Vert b\boldsymbol{i}+c\boldsymbol{j}+d\boldsymbol{k}\Vert = \sqrt{b^2+c^2+d^2}$。此外，还有类似的矩阵映射
\begin{equation}
a+b\boldsymbol{i}+c\boldsymbol{j}+d\boldsymbol{k} \quad \leftrightarrow \quad \begin{pmatrix}
a & -b & -c & -d \\
b & a & -d & c \\
c & d & a & -b \\
d & -c & b & a
\end{pmatrix}\label{eq:mapping}\end{equation}

### 违背相对性

关于这些公式背后的起源，那就说来话长了，这里也不打算细谈，有兴趣的读者请自行搜索资料阅读。有了欧拉公式和指数映射后，读者或许反应过来：一维RoPE无非是$e^{n\boldsymbol{i}\theta}$对应的矩阵映射，那么二维RoPE的话将$e^{x\boldsymbol{i}\theta + y\boldsymbol{j}\theta}$映射为矩阵形式不就行了？

笔者一开始也是这样想的，很遗憾，这是错的。错在哪里呢？在一维RoPE中，我们利用了内积的复数表示：
\begin{equation}\langle\boldsymbol{q},\boldsymbol{k}\rangle=\text{Re}[\boldsymbol{q}\boldsymbol{k}^*]\end{equation}
该恒等式在四元数中同样成立，因此可以照搬。接着我们利用了复指数：
\begin{equation}\langle\boldsymbol{q}e^{m\boldsymbol{i}\theta},\boldsymbol{k}e^{n\boldsymbol{i}\theta}\rangle=\text{Re}\left[\left(\boldsymbol{q}e^{m\boldsymbol{i}\theta}\right)\left(\boldsymbol{k}e^{n\boldsymbol{i}\theta}\right)^*\right]=\text{Re}\left[\boldsymbol{q}e^{m\boldsymbol{i}\theta}e^{-n\boldsymbol{i}\theta}\boldsymbol{k}^*\right]=\text{Re}\left[\boldsymbol{q}e^{(m-n)\boldsymbol{i}\theta}\boldsymbol{k}^*\right]\end{equation}
前两个等号都可以照搬到四元数中，关键是第三个等号在四元数中并不恒成立！一般地，对于两个四元数$\boldsymbol{p},\boldsymbol{q}$，等式$e^{\boldsymbol{p}+\boldsymbol{q}}=e^{\boldsymbol{p}}e^{\boldsymbol{q}}$并不成立！更广义来说，对于两个乘法不满足交换律的对象，一般都有$e^{\boldsymbol{p}+\boldsymbol{q}}\neq e^{\boldsymbol{p}}e^{\boldsymbol{q}}$。

所以，推导到最后，由于指数乘法无法转换为加法，最后的相对性没法得到保证。因此通过四元数推导这条路，就此夭折了\.\.\.

## 矩阵指数

四元数的矩阵映射表明四元数事实上代表了一簇特定的$4\times 4$矩阵，四元数推导走不通，那么或许使用一般的矩阵分析可以走得通。事实上确实如此，在这一节中，我们将利用矩阵指数给出一个推导结果。

### 矩阵指数

这里的[矩阵指数](<https://en.wikipedia.org/wiki/Matrix_exponential>)，并不是神经网络的用指数函数作为激活函数的逐位运算，而是按照幂级数定义的运算：
\begin{equation}\exp \boldsymbol{B} = \sum_{k=0}^{\infty}\frac{\boldsymbol{B}^k}{k!}\end{equation}
其中$\boldsymbol{B}^k$是指按照矩阵乘法将$k$个$\boldsymbol{B}$连乘。关于矩阵指数，笔者之前曾写过[《恒等式 det(exp(A)) = exp(Tr(A)) 赏析》](<https://spaces.ac.cn/archives/6377>)，也欢迎参考阅读。

矩阵指数是非常重要的一种矩阵运算，它可以直接写出常系数微分方程组$\frac{d}{dt}\boldsymbol{x}_t=\boldsymbol{A}\boldsymbol{x}_t$的解：
\begin{equation}\boldsymbol{x}_t = \big(\exp t\boldsymbol{A}\big)\boldsymbol{x}_0\end{equation}
当然这跟本文的主题关系不大。对于RoPE的推导，我们主要利用到矩阵指数的如下性质：
\begin{equation}\boldsymbol{A}\boldsymbol{B} = \boldsymbol{B}\boldsymbol{A} \quad\Rightarrow\quad \big(\exp \boldsymbol{A}\big)\big(\exp \boldsymbol{B}\big) = \exp \big(\boldsymbol{A} + \boldsymbol{B}\big)\label{eq:expm-ex}\end{equation}
也就是说，如果$\boldsymbol{A},\boldsymbol{B}$的乘法可以交换，那么矩阵指数就可以像数的指数一样将乘法转换为加法。不过要注意这是一个充分不必要条件。

至于怎么把矩阵指数算出来，这里没法再展开介绍了，但是很多软件库已经自带了矩阵指数运算，比如数值计算库scipy和tensorflow都有<code>expm</code>函数，而符号计算的话，Mathematica里边有<code>MatrixExp</code>函数。

### 一维通解

为什么能够将RoPE跟矩阵指数联系起来呢？因为一维的RoPE存在比较简单的指数表达式：
\begin{equation}\boldsymbol{\mathcal{R}}_n=\begin{pmatrix}\cos n\theta & -\sin n\theta\\ \sin n\theta & \cos n\theta\end{pmatrix}=\exp\left\{n\theta\begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}\right\}\label{eq:rope-exp}\end{equation}
于是笔者开始思考如下形式的矩阵作为RoPE的解：
\begin{equation}\boldsymbol{\mathcal{R}}_n=\exp n\boldsymbol{B}\end{equation}
其中$\boldsymbol{B}$是一个跟$n$无关的矩阵。RoPE的必要条件是满足“相对性”条件$\eqref{eq:re}$，于是我们分析
\begin{equation}\big(\exp m\boldsymbol{B}\big)^{\top}\big(\exp n\boldsymbol{B}\big) = \big(\exp m\boldsymbol{B}^{\top}\big)\big(\exp n\boldsymbol{B}\big)\end{equation}
这里先假设$\boldsymbol{B}^{\top},\boldsymbol{B}$是可交换的，那么根据式$\eqref{eq:expm-ex}$有
\begin{equation}\big(\exp m\boldsymbol{B}^{\top}\big)\big(\exp n\boldsymbol{B}\big) = \exp \big(m\boldsymbol{B}^{\top} + n\boldsymbol{B}\big)\end{equation}
要让$m\boldsymbol{B}^{\top} + n\boldsymbol{B}=(n-m)\boldsymbol{B}$，只需要满足
\begin{equation}\boldsymbol{B}^{\top} = - \boldsymbol{B}\end{equation}
这便是“相对性”给出的约束条件，刚才我们还假设了$\boldsymbol{B}^{\top},\boldsymbol{B}$是可交换的，现在可以检验满足这个等式的$\boldsymbol{B}^{\top},\boldsymbol{B}$一定是可交换的，所以结果是自洽的。

这也就是说，对于任何满足$\boldsymbol{B}^{\top} + \boldsymbol{B} = 0$的矩阵$\boldsymbol{B}$，$\exp n\boldsymbol{B}$都是方程$\eqref{eq:re}$的解，并且还可以证明它一定是正交矩阵。当然，根据$\exp n\boldsymbol{B}=\left(\exp \boldsymbol{B}\right)^n$，我们更直接的得到：对于任意正交矩阵$\boldsymbol{O}$，$\boldsymbol{\mathcal{R}}_n=\boldsymbol{O}^n$都是方程$\eqref{eq:re}$的解。

对于$2\times 2$的矩阵来说，$\boldsymbol{B}^{\top} + \boldsymbol{B} = 0$的通解是$\boldsymbol{B}=\begin{pmatrix}0 & -\theta\\ \theta & 0\end{pmatrix}$，于是就有了如式$\eqref{eq:rope-exp}$的解。

### 二维约束

类似地，对于二维RoPE，我们考虑
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y}=\exp \big(x\boldsymbol{B}_1 + y\boldsymbol{B}_2\big)\end{equation}
作为候选解。重复上述关于“相对性”条件的推导：先假设$x_1\boldsymbol{B}_1^{\top} + y_1\boldsymbol{B}_2^{\top}$与$x_2\boldsymbol{B}_1 + y_2\boldsymbol{B}_2$是可交换的，那么我们可以得到如下约束条件：
\begin{equation}\boldsymbol{B}_1^{\top} + \boldsymbol{B}_1 = 0,\quad \boldsymbol{B}_2^{\top} + \boldsymbol{B}_2 = 0\end{equation}
然而，$x_1\boldsymbol{B}_1^{\top} + y_1\boldsymbol{B}_2^{\top}$与$x_2\boldsymbol{B}_1 + y_2\boldsymbol{B}_2$可交换，意味着$(\boldsymbol{B}_1,\boldsymbol{B}_1^{\top})$、$(\boldsymbol{B}_2,\boldsymbol{B}_2^{\top})$、$(\boldsymbol{B}_1,\boldsymbol{B}_2^{\top})$和$(\boldsymbol{B}_2,\boldsymbol{B}_1^{\top})$都可交换，但是上述两个约束只能保证$(\boldsymbol{B}_1,\boldsymbol{B}_1^{\top})$和$(\boldsymbol{B}_2,\boldsymbol{B}_2^{\top})$可交换，不能保证后两者的交换性，所以我们需要把它作为约束条件补充上去，得到：
\begin{equation}\left\{\begin{aligned}
&\boldsymbol{B}_1^{\top} + \boldsymbol{B}_1 = 0\\
&\boldsymbol{B}_2^{\top} + \boldsymbol{B}_2 = 0\\
&\boldsymbol{B}_1 \boldsymbol{B}_2^{\top} = \boldsymbol{B}_2^{\top} \boldsymbol{B}_1
\end{aligned}\right.\label{eq:2d-conds}\end{equation}
不难证明在前两个条件下，新增的约束条件也相当于$\boldsymbol{B}_1 \boldsymbol{B}_2 = \boldsymbol{B}_2 \boldsymbol{B}_1$。

### RoPE现身

由于满足前两个条件的$2\times 2$矩阵只有一个独立参数，不满足“可逆性”，所以我们至少要考虑$3\times 3$矩阵，它有3个独立参数
\begin{equation}\begin{pmatrix}0 & -a & -b \\ a & 0 & -c \\ b & c & 0\end{pmatrix}\end{equation}
为了保证可逆性，我们不妨设$\boldsymbol{B}_1,\boldsymbol{B}_2$是“正交”的，比如设：
\begin{equation}\boldsymbol{B}_1=\begin{pmatrix}0 & -a & 0 \\ a & 0 & 0 \\ 0 & 0 & 0\end{pmatrix},\quad\boldsymbol{B}_2=\begin{pmatrix}0 & 0 & -b \\ 0 & 0 & -c \\ b & c & 0\end{pmatrix}\end{equation}
不失一般性还可以设$a=1$，那么由条件$\eqref{eq:2d-conds}$解得$b=0,c=0$，即$\boldsymbol{B}_2$只能是全零解，这不符合我们的要求。Mathematica的求解代码为：



<pre><code class="language-mathematica">B[a_, b_, c_] = {{0, -a, -b}, {a, 0, -c}, {b, c, 0}};
B1 = B[1, 0, 0];
B2 = B[0, b, c];
Solve[{Dot[B1, B2] == Dot[B2, B1]}, {b, c}]</code></pre>



因此，我们至少要考虑$4\times 4$矩阵，它有6个独立参数，不失一般性，考虑正交分解：
\begin{equation}\boldsymbol{B}_1=\begin{pmatrix}0 & -a & -b & 0 \\ a & 0 & -c & 0 \\ b & c & 0 & 0 \\ 0 & 0 & 0 & 0\end{pmatrix},\quad\boldsymbol{B}_2=\begin{pmatrix}0 & 0 & 0 & -d \\ 0 & 0 & 0 & -e \\ 0 & 0 & 0 & -f \\ d & e & f & 0\end{pmatrix}\end{equation}
解得
\begin{equation}d=cf,\quad e=-bf\end{equation}
求解代码：



<pre><code class="language-mathematica">B[a_, b_, c_, d_, e_, 
   f_] = {{0, -a, -b, -d}, {a, 0, -c, -e}, {b, c, 0, -f}, {d, e, f, 
    0}};
B1 = B[1, b, c, 0, 0, 0];
B2 = B[0, 0, 0, d, e, f];
Solve[{Dot[B1, B2] == Dot[B2, B1]}, {b, c, d, e, f}]</code></pre>



可以发现结果没有对$f$提出约束，所以从最简单起见，我们可以让$f=1$，剩下的$b,c,d,e$全部为0，此时
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y}=\exp \,\begin{pmatrix}0 & -x & 0 & 0 \\ x & 0 & 0 & 0 \\ 0 & 0 & 0 & -y \\ 0 & 0 & y & 0\end{pmatrix}\end{equation}
可以增加个参数$\theta$，完成展开，就得到：
\begin{equation}\boldsymbol{\mathcal{R}}_{x,y}=\exp \,\left\{\begin{pmatrix}0 & -x & 0 & 0 \\ x & 0 & 0 & 0 \\ 0 & 0 & 0 & -y \\ 0 & 0 & y & 0\end{pmatrix}\theta\right\}=\left(
\begin{array}{cc:cc}
\cos x\theta & -\sin x\theta & 0 & 0 \\
\sin x\theta & \cos x\theta & 0 & 0 \\
\hdashline
0 & 0 & \cos y\theta & -\sin y\theta \\
0 & 0 & \sin y\theta & \cos y\theta \\
\end{array}\right)\end{equation}

## 延伸故事

至此，关于二维RoPE的推导介绍完毕。现在读者可能想问的是效果如何？很遗憾，现在还没有很完整的实验结果，毕竟笔者之前也没做过ViT相关的工作，而这个二维RoPE的推导也刚完成没多久，所以进展比较慢，只能说初步的结果显示还是挺有效的。EleutherAI团队的成员也实验过这个方案，效果也比已有的其他位置编码好。

说到EleutherAI团队，这里再多说几句。EleutherAI团队是前段时候比较火的号称要“复现GPT3”的那个团队，我们在文章[《Transformer升级之路：2、博采众长的旋转式位置编码》](<https://spaces.ac.cn/archives/8265>)中提出了RoPE及RoFormer后，有幸得到了EleutherAI团队的关注，他们做了很多补充实验，确认了RoPE比很多其他位置编码都更加有效（参考他们的博客[《Rotary Embeddings: A Relative Revolution》](<https://blog.eleuther.ai/rotary-embeddings/>)），这促使我们完成了英文论文[《RoFormer: Enhanced Transformer with Rotary Position Embedding》](<https://papers.cool/arxiv/2104.09864>)并提交到了Arxiv上。而关于二维RoPE的疑问，最初也是来源于EleutherAI团队。

## 文章小结

本文介绍了我们对RoPE的二维推广，主要以“相对性”、“可逆性”为出发点来确定二维RoPE的最终形式，尝试了四元数和矩阵指数两种推导过程，最终通过矩阵指数来给出了最终的解，从推导过程中我们还可以深化对RoPE的理解。
