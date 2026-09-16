# 从Wasserstein距离、对偶理论到WGAN

> 作者：苏剑林 · 科学空间 · 2019-01-20
>
> 原文：<https://spaces.ac.cn/archives/6280>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![推土机哪家强？成本最低找Wasserstein](<https://spaces.ac.cn/usr/uploads/2019/01/1287001589.jpg>)](<https://spaces.ac.cn/usr/uploads/2019/01/1287001589.jpg>)

推土机哪家强？成本最低找Wasserstein

2017年的时候笔者曾写过博文[《互怼的艺术：从零直达WGAN\-GP》](<https://spaces.ac.cn/archives/4439>)，从一个相对通俗的角度来介绍了WGAN，在那篇文章中，WGAN更像是一个天马行空的结果，而实际上跟Wasserstein距离没有多大关系。

在本篇文章中，我们再从更数学化的视角来讨论一下WGAN。当然，本文并不是纯粹地讨论GAN，而主要侧重于Wasserstein距离及其对偶理论的理解。本文受启发于著名的国外博文[《Wasserstein GAN and the Kantorovich\-Rubinstein Duality》](<https://vincentherrmann.github.io/blog/wasserstein/>)，内容跟它大体上相同，但是删除了一些冗余的部分，对不够充分或者含糊不清的地方作了补充。不管怎样，在此先对前辈及前辈的文章表示致敬。

（<strong>注：</strong>完整理解本文，应该需要多元微积分、概率论以及线性代数等基础知识。还有，本文确实长，数学公式确实多，但是，真的不复杂、不难懂，大家不要看到公式就吓怕了～）

## Wasserstein距离

显然，整篇文章必然围绕着Wasserstein距离（W距离）来展开，而Wasserstein距离的定义又基于最优传输成本，所以我们需要先介绍最优传输成本。假设我们有了两个概率分布$p(\boldsymbol{x}),q(\boldsymbol{x})$，那么最优传输成本的定义为
\begin{equation}\mathcal{C}[p,q]=\inf_{\gamma\in \Pi[p,q]} \iint \gamma(\boldsymbol{x},\boldsymbol{y}) c(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{x}d\boldsymbol{y}\label{eq:ot}\end{equation}
事实上，这也算是最优传输理论中最核心的定义了。

相信我，式$\eqref{eq:ot}$没有想象中那么难理解。我们来逐项看看。

### 成本函数

首先看$c(\boldsymbol{x},\boldsymbol{y})$，它是一个成本函数，代表着从$\boldsymbol{x}$运输到$\boldsymbol{y}$的成本，常见的选择就是欧氏距离的若干次方：
\begin{equation}c(\boldsymbol{x},\boldsymbol{y}) = \Vert\boldsymbol{x}-\boldsymbol{y}\Vert^{\rho}\end{equation}
此时我们记
\begin{equation}\mathcal{W}_{\rho}[p,q]=\left(\mathcal{C}[p,q]\right)^{1/\rho}\end{equation}
$\mathcal{W}_{\rho}[p,q]$就被称为“Wasserstein距离”（更准确来说是“Wasserstein\-$\rho$距离”）。可以看到，最优传输成本$\mathcal{C}[p,q]$的含义比Wasserstein距离$\mathcal{W}_{\rho}[p,q]$更为一般化，所以后面的推导以$\mathcal{C}[p,q]$为主。当$\rho=1$时，最优传输成本等于相应的Wasserstein距离。

一般地，欧氏距离$\Vert\boldsymbol{x}-\boldsymbol{y}\Vert$可以换成更一般的距离，但具体采用哪种距离并不是特别重要，因为很多范数都是相互等价的，范数的等价性表明其实最终定义出来的W距离都差不多。

### 成本最小化

然后来看$\gamma$，条件$\gamma\in \Pi[p,q]$意味着：
\begin{equation}\int \gamma(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{y}=p(\boldsymbol{x})\quad\text{且}\quad\int \gamma(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{x}=q(\boldsymbol{y})\end{equation}
这也就是说，$\gamma$是一个联合分布，它的边缘分布就是原来的$p$和$q$。

事实上$\gamma$就描述了一种运输方案。不失一般性，设$p$是原始分布，设$q$是目标分布，$p(\boldsymbol{x})$的意思是原来在位置$\boldsymbol{x}$处有$p(\boldsymbol{x})$量的货物，$q(\boldsymbol{x})$是指最终$\boldsymbol{x}$处要存放的货物量，如果$p(\boldsymbol{x}) > q(\boldsymbol{x})$，那么就要把$\boldsymbol{x}$处的一部分货物运到别处，反之，如果$p(\boldsymbol{x}) < q(\boldsymbol{x})$，那么就要从别的地方运一些货物到$\boldsymbol{x}$处。而$\gamma(\boldsymbol{x}, \boldsymbol{y})$的意思是指，要从$\boldsymbol{x}$处搬$\gamma(\boldsymbol{x}, \boldsymbol{y})d\boldsymbol{x}$那么多的东西到$\boldsymbol{y}$处。

最后是$\inf$，这表示下确界，简单来说就是取最小，也就是说，要从所有的运输方案中，找出总运输成本$\iint \gamma(\boldsymbol{x},\boldsymbol{y}) c(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{x}d\boldsymbol{y}$最小的方案，这个方案的成本，就是我们要算的$\mathcal{C}[p,q]$。如果将上述比喻中的“货物”换成“沙土”，那么最优传输成本就是在求最省力的“搬土”方案了，所以Wasserstein距离经常也被称为“推土机距离”（Earth Mover's Distance）。

最后改编一张开头提到的国外博文的图片，来展示这个“推土”过程：

[![推土机距离图示。左边p(x)每处的沙土被分为若干部分，然后运输到右端q(x)的同色的位置（或者不动）](<https://spaces.ac.cn/usr/uploads/2019/01/417385384.png>)](<https://spaces.ac.cn/usr/uploads/2019/01/417385384.png>)

推土机距离图示。左边p(x)每处的沙土被分为若干部分，然后运输到右端q(x)的同色的位置（或者不动）

### 矩阵形式

逐项分析完含义之后，现在我们再来完成地重述一下问题，我们实际上在求
\begin{equation}\iint \gamma(\boldsymbol{x},\boldsymbol{y}) c(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{x}d\boldsymbol{y}\label{eq:ot-t}\end{equation}
的最小值，其中$c(\boldsymbol{x},\boldsymbol{y})$是事先给定的，而这个最小值要满足如下约束：
\begin{equation}\int \gamma(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{y}=p(\boldsymbol{x}),\quad\int \gamma(\boldsymbol{x},\boldsymbol{y}) d\boldsymbol{x}=q(\boldsymbol{y}),\quad \gamma(\boldsymbol{x},\boldsymbol{y})\geq 0\label{eq:ot-c}\end{equation}

认真盯着式$\eqref{eq:ot-t}$，考虑到积分只是求和的极限形式，所以我们可以把$\gamma(\boldsymbol{x},\boldsymbol{y})$和$c(\boldsymbol{x},\boldsymbol{y})$离散化，然后看成很长很长的（列）向量$\boldsymbol{\Gamma}$和$\boldsymbol{C}$：
\begin{equation}\boldsymbol{\Gamma}=\begin{pmatrix}
\gamma(\boldsymbol{x}_1, \boldsymbol{y}_1) \\
\gamma(\boldsymbol{x}_1, \boldsymbol{y}_2) \\
\vdots \\ \hline
\gamma(\boldsymbol{x}_2, \boldsymbol{y}_1) \\
\gamma(\boldsymbol{x}_2, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\ \hline
\gamma(\boldsymbol{x}_n, \boldsymbol{y}_1) \\
\gamma(\boldsymbol{x}_n, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\
\end{pmatrix},\quad \boldsymbol{C}=\begin{pmatrix}
c(\boldsymbol{x}_1, \boldsymbol{y}_1) \\
c(\boldsymbol{x}_1, \boldsymbol{y}_2) \\
\vdots \\ \hline
c(\boldsymbol{x}_2, \boldsymbol{y}_1) \\
c(\boldsymbol{x}_2, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\ \hline
c(\boldsymbol{x}_n, \boldsymbol{y}_1) \\
c(\boldsymbol{x}_n, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\
\end{pmatrix}\label{eq:lp-ot-t1}\end{equation}

所以式$\eqref{eq:ot-t}$相当于就是将$\boldsymbol{\Gamma}$和$\boldsymbol{C}$对应位置相乘，然后求和，这不就是<strong><u>内积</u></strong>$\langle\boldsymbol{\Gamma},\boldsymbol{C}\rangle$了吗？

如果还没理解这一点，那么请再好好地盯一会式$\eqref{eq:ot-t}$，头脑中想象着将$\boldsymbol{x},\boldsymbol{y}$分区间离散化的过程，再想想积分的定义，相信这并不难理解；如果已经理解了这一点，那就好办了，我们可以把约束条件$\eqref{eq:ot-c}$也这样看：把$p(\boldsymbol{x}),q(\boldsymbol{x})$分别看成一个长向量，然后还可以拼起来，把积分也看成求和，这时候约束条件$\eqref{eq:ot-c}$也可以写成矩阵形式$\boldsymbol{A}\boldsymbol{\Gamma}=\boldsymbol{b}$：
\begin{equation}\underbrace{\left( \begin{array}{ccc|ccc|c|ccc|c}
1 & 1 & \dots & 0 & 0 & \dots & \dots & 0 & 0 & \dots & \dots \\
0 & 0 & \dots & 1 & 1 & \dots & \dots & 0 & 0 & \dots & \dots \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots & \ddots & \ddots \\
0 & 0 & \dots & 0 & 0 & \dots & \dots & 1 & 1 & \dots & \dots \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots & \ddots & \ddots \\ \hline
1 & 0 & \dots & 1 & 0 & \dots & \dots & 1 & 0 & \dots & \dots \\
0 & 1 & \dots & 0 & 1 & \dots & \dots & 0 & 1 & \dots & \dots \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots & \ddots & \ddots \\
0 & 0 & \dots & 0 & 0 & \dots & \dots & 0 & 0 & \dots & \dots \\
\vdots & \vdots & \ddots & \vdots & \vdots & \ddots & \ddots & \vdots & \vdots & \ddots & \ddots \\
\end{array} \right)}_{\Large\boldsymbol{A}}\,\, \underbrace{\begin{pmatrix}
\gamma(\boldsymbol{x}_1, \boldsymbol{y}_1) \\
\gamma(\boldsymbol{x}_1, \boldsymbol{y}_2) \\
\vdots \\ \hline
\gamma(\boldsymbol{x}_2, \boldsymbol{y}_1) \\
\gamma(\boldsymbol{x}_2, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\ \hline
\gamma(\boldsymbol{x}_n, \boldsymbol{y}_1) \\
\gamma(\boldsymbol{x}_n, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\
\end{pmatrix}}_{\Large\boldsymbol{\Gamma}} \,\,=\,\, \underbrace{\begin{pmatrix}
p(\boldsymbol{x}_1) \\
p(\boldsymbol{x}_2) \\
\vdots \\
p(\boldsymbol{x}_n) \\
\vdots \\ \hline
q(\boldsymbol{y}_1) \\
q(\boldsymbol{y}_2) \\
\vdots \\
q(\boldsymbol{y}_n) \\
\vdots \\
\end{pmatrix}}_{\Large\boldsymbol{b}}\label{eq:lp-ot-t2}\end{equation}

最后不能忘记的是$\boldsymbol{\Gamma}\geq 0$，它表示$\boldsymbol{\Gamma}$的每个分量都大于等于0。

### 线性规划问题

现在问题可以用一行字来描述
\begin{equation}\min_{\boldsymbol{\Gamma}}\big\{\langle\boldsymbol{\Gamma},\boldsymbol{C}\rangle\,\big|\,\boldsymbol{A}\boldsymbol{\Gamma}=\boldsymbol{b},\,\boldsymbol{\Gamma}\geq 0\big\}\label{eq:lp-ot}\end{equation}

这就是“<u>线性约束下的线性函数最小值</u>”的问题，它就是我们在高中时就已经接触过的线性规划问题了～可见，虽然原始问题足够复杂，又有积分又有下确界的，但经过转写，它本质上就是一个并不难理解的线性规划问题（当然，“不难理解”并不意味着“容易求解”）。

## 线性规划与对偶

让我们用更一般的记号，把线性规划问题重写一遍，常见的形式有两种：
\begin{equation}\min_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}\quad\text{或}\quad \min_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}\geq \boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}\end{equation}
这两种形式本质上是等价的，只不过在讨论第一种的时候相对简单一点（真的只是简单一点点，并没有本质差别），而从$\eqref{eq:lp-ot}$式可以知道，我们目前只关心第一种情况。

注意，为了避免混乱，我们必须声明一下各个向量的大小。我们假设每个向量都是列向量，经过转置$^\top$之后就代表一个行向量，$\boldsymbol{x},\boldsymbol{c}\in\mathbb{R}^n$都是$n$维向量，其中$\boldsymbol{c}$也就是权重，$\boldsymbol{c}^{\top}\boldsymbol{x}$就是对$\boldsymbol{x}$的各个分量加权求和；$\boldsymbol{b}\in\mathbb{R}^m$是$m$维向量，自然$\boldsymbol{A}\in\mathbb{R}^{m\times n}$是一个$m\times n$的矩阵了，$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$实际上就是描述了$m$个等式约束。

### 弱对偶形式

在规划和优化问题中，“对偶形式”是一个非常重要的概念。一般情况下，“对偶”是指某种变换，能将原问题转化为一个等价的、但是看起来几乎不一样的新问题，即
\begin{equation}\text{原问题}\quad\xrightarrow{\text{对偶变换}}\quad \text{新问题}\end{equation}
“对偶”之所以称为“对偶”，是因为将新问题进行同样形式的变换后，通常来说能还原为原问题，即
\begin{equation}\text{新问题}\quad\xrightarrow{\text{对偶变换}}\quad \text{原问题}\end{equation}
即“对偶”像是一面镜子，原问题和新问题相当于“原像”和“镜像”，解决了一个问题，就等价于解决了另一个问题。所以就看哪个问题更简单了。

读者可能还有疑问：“对偶”跟数学中诸如“逆否命题”之类的等价描述有什么区别？其实也没有本质区别，简单来说“对偶”和“逆否命题”都是跟原来的命题完全等价的，但是“对偶”看起来跟原命题很不一样，而“逆否命题”仅仅是原命题的一个逻辑变换～从线性代数的角度来看，“对偶”相当于向量空间中的“原空间”和“补空间”之间的关系。

#### 最大 vs 最小

这里我们先介绍“弱对偶形式”，其实它推导起来还是挺简单的。

我们的目标是$\min\limits_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}$，设置最小值在$\boldsymbol{x}^*$处取到，那么我们有$\boldsymbol{A}\boldsymbol{x}^*=\boldsymbol{b}$，我们可以在两边乘以一个$\boldsymbol{y}^{\top}\in\mathbb{R}^m$，使得等式变成一个标量：$\boldsymbol{y}^{\top}\boldsymbol{A}\boldsymbol{x}^*=\boldsymbol{y}^{\top}\boldsymbol{b}$。

如果此时假设$\boldsymbol{y}^{\top}\boldsymbol{A}\leq \boldsymbol{c}^{\top}$，那么$\boldsymbol{y}^{\top}\boldsymbol{A}\boldsymbol{x}^*\leq \boldsymbol{c}^{\top}\boldsymbol{x}^*$（因为$\boldsymbol{x}^* \geq 0$），则$\boldsymbol{y}^{\top}\boldsymbol{b}\leq \boldsymbol{c}^{\top} \boldsymbol{x}^*$。也就是说，在条件$\boldsymbol{y}^{\top}\boldsymbol{A}\leq \boldsymbol{c}^{\top}$下的任意$\boldsymbol{y}^{\top}\boldsymbol{b}$总是不大于$\min\limits_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}$，“总是”意味着即使对于最大那个也一样，所以我们就有
\begin{equation}\max_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\}\leq \min_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}\label{eq:weak-dual}\end{equation}
这便称为“弱对偶形式”，它的形式就是：“左边的最大”还大不过“右端的最小”。

#### 几点评注

对于弱对偶形式，也许下面几点值得进一步说明一下：

> 1、现在我们将原来的最小值问题变成了一个最大值问题$\max\limits_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\}$，这便有了对偶的味道。当然，弱对偶形式之所以“弱”，是因为我们目前找到的对偶形式只是原来问题的一个下界，还没有证明它们二者相等。
> 
> 
> 
> 2、弱对偶形式在很多优化问题中（包括非线性优化）都成立。如果二者真的相等，那么就是真正意义上的对偶了，称为强对偶形式。
> 
> 
> 
> 3、理论上，我们确实需要证明式$\eqref{eq:weak-dual}$左右两端相等才能进一步应用它。但从应用角度，其实弱对偶形式给出的下界都已经够用了，因为深度学习中的问题都很复杂，能有一个近似的目标去优化都已经很不错了。
> 
> 
> 
> 4、读者可能会想问：前面我们为什么要假设$\boldsymbol{y}^{\top}\boldsymbol{A}\leq \boldsymbol{c}^{\top}$而不干脆假设$\boldsymbol{y}^{\top}\boldsymbol{A}=\boldsymbol{c}^{\top}$？假设后者当然简单很多，但问题是后者在实践中很难实现，所以只能假设前者。

### 强对偶形式

上面已经说了，从实践角度其实弱对偶形式已经够用了。但是为了让对完整理论有兴趣的读者也有更多收获，这里继续把“强对偶形式”也论证一番。对于只关心WGAN本身的读者来说，可以考虑跳过这部分。

所谓强对偶形式，也就是
\begin{equation}\max_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\} = \min_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}\label{eq:strong-dual}\end{equation}
注意前面已经说了，弱对偶形式对于很多优化问题都成立，但强对偶形式不一定成立。而对于线性规划来说，强对偶形式是成立的。

#### Farkas引理

强对偶形式的证明，主要用到称之为“Farkas引理”的结论：

> 对于固定的矩阵$\boldsymbol{A}\in\mathbb{R}^{m\times n}$和向量$\boldsymbol{b}\in\mathbb{R}^m$，下面两个选项有且只有一个成立：
> 
> 
> 
> 1、存在$\boldsymbol{x}\in \mathbb{R}^n$且$\boldsymbol{x}\geq 0$，使得$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$；
> 2、存在$\boldsymbol{y}\in \mathbb{R}^m$使得$\boldsymbol{A}^{\top}\boldsymbol{y}\leq 0$且$\boldsymbol{b}^{\top}\boldsymbol{y} > 0$。

什么鬼？又大又小又转置的？能不能说人话？

其实这个引理还真有一个很直观的几何解释，只不过几何解释翻译成代数语言就不简单了。几何解释的出发点是我们去考虑如下的向量集合：
\begin{equation}\big\{\boldsymbol{A}\boldsymbol{x}\big|\boldsymbol{x}\in \mathbb{R}^n\text{且}\boldsymbol{x}\geq 0\big\}\end{equation}
这个集合的含义是：我们将$\boldsymbol{A}$看成是$n$个$m$维列向量的组合
\begin{equation}\boldsymbol{A}=(\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n)\end{equation}
那么上述集合实际上就是所有$\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n$的非负线性组合。那这个集合是个啥呢？答案是：一个锥，如图所示。

[![给定向量的非负线性组合构成了这些向量围成的一个锥](<https://spaces.ac.cn/usr/uploads/2019/01/1913433046.png>)](<https://spaces.ac.cn/usr/uploads/2019/01/1913433046.png>)

给定向量的非负线性组合构成了这些向量围成的一个锥

现在我们随便给定一个向量$\boldsymbol{b}$，那么显然它只有两种可能性，而且必有一种成立：<u>1、在锥内（包括边界）；2、在锥外。</u>（这当然是废话，但是将它翻译成代数语言，那就不是废话了。）

[![给定的向量在锥内，意味着可以表示为非负线性组合](<https://spaces.ac.cn/usr/uploads/2019/01/3528538907.png>)](<https://spaces.ac.cn/usr/uploads/2019/01/3528538907.png>)

给定的向量在锥内，意味着可以表示为非负线性组合

[![给定的向量在锥外，我们可以找到一个“标杆”向量与之对比](<https://spaces.ac.cn/usr/uploads/2019/01/2134595027.png>)](<https://spaces.ac.cn/usr/uploads/2019/01/2134595027.png>)

给定的向量在锥外，我们可以找到一个“标杆”向量与之对比

如果它在锥内，那么根据锥本身的定义，它就可以表示为$\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n$的非负线性组合（表示方式可能不唯一），也就是存在$\boldsymbol{x}\geq 0$，使得$\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b}$，这就是第一种情况。

如果在锥外呢？怎么表示在锥外？当然我们可以直接写出锥内的否命题，但是那实用价值不大。如果向量$\boldsymbol{b}$在锥外，那么我们总是可以找到一个“标杆”向量$\boldsymbol{y}$，它与$\boldsymbol{a}_1,\boldsymbol{a}_2,\dots,\boldsymbol{a}_n$的夹角都大于等于90度，向量表示法就是内积都小于等于零，即$(\boldsymbol{a}_1^{\top}\boldsymbol{y}, \boldsymbol{a}_2^{\top}\boldsymbol{y}, \dots, \boldsymbol{a}_n^{\top}\boldsymbol{y}) \leq 0$，或者写成一个整体$\boldsymbol{A}^{\top}\boldsymbol{y} \leq 0$。找到这个“标杆”后，向量$\boldsymbol{b}$与“标杆”的夹角必然是要小于90度的，即$\boldsymbol{b}^{\top}\boldsymbol{y} > 0$。这样一个大于等于90度，一个小于90度，保证了向量$\boldsymbol{b}$在全体向量构成的锥外。这就是第二种情况。

当然，这不能算是完备的证明，只能算是一个启发式引导，完备的证明还要仔细论证为什么这些向量的非负线性组合构成了一个锥～这些就不在本文的范畴了。<u>Farkas引理的特点是二选一，比如我要证明满足第二点，只需要证明它不满足第一点，反之亦然。这是对问题的一个转化。</u>

#### 从引理到强对偶

有了Farkas引理，我们就可以证明强对偶形式了。证明的思路是：证明$\max$可以任意程度地接近$\min$。

证明还是先假设$\min$的最小值在$\boldsymbol{x}^*$处取到，即最小值为$z^* = \boldsymbol{c}^{\top}\boldsymbol{x}^*$，那么我们考虑：
\begin{equation}\hat{\boldsymbol{A}} = \begin{pmatrix} \boldsymbol{A} \\ -\boldsymbol{c}^{\top} \end{pmatrix}, \quad
\hat{\boldsymbol{b}}_{\epsilon} = \begin{pmatrix} \boldsymbol{b} \\ -z^* + \epsilon \end{pmatrix}, \quad
\hat{\boldsymbol{y}} = \begin{pmatrix} \boldsymbol{y} \\ \alpha \end{pmatrix}\end{equation}
当$\epsilon > 0$时，那么对于任意$\boldsymbol{x} \geq 0$，$\hat{\boldsymbol{A}} \boldsymbol{x}$都不可能等于$\hat{\boldsymbol{b}}_{\epsilon}$，这是因为$\boldsymbol{c}^{\top} \boldsymbol{x}^* = z^*$已经是最小值，所以$-z^*$是$-\boldsymbol{c}^{\top} \boldsymbol{x}$能达到的最大值，它不可能等于更大的$-z^* + \epsilon$。

前面已经说了，不满足第一种情况，那就只能满足第二种情况了，即存在$\hat{\boldsymbol{y}} = \begin{pmatrix} \boldsymbol{y} \\ \alpha \end{pmatrix}$使得$\hat{\boldsymbol{A}}^{\top}\hat{\boldsymbol{y}}\leq 0$且$\hat{\boldsymbol{b}}_{\epsilon}^{\top}\hat{\boldsymbol{y}} > 0$，这等价于
\begin{equation}\boldsymbol{A}^{\top} \boldsymbol{y} \leq \alpha \boldsymbol{c}, \quad \boldsymbol{b}^{\top} \boldsymbol{y} > \alpha(z^* - \epsilon)\label{eq:whocare}\end{equation}

下面我们表明$\alpha$必须大于0。由于已知$0 < \hat{\boldsymbol{b}}_{\epsilon}^{\top}\hat{\boldsymbol{y}} = \hat{\boldsymbol{b}}_{0}^{\top}\hat{\boldsymbol{y}} + \alpha\epsilon$，这里出现了$\hat{\boldsymbol{b}}_{0}^{\top}\hat{\boldsymbol{y}}$，所以我们不妨再看一下$\epsilon=0$的情形：当$\epsilon = 0$时有$\hat{\boldsymbol{A}} \boldsymbol{x}^* = \hat{\boldsymbol{b}}_0$，即满足Farkas引理的第一种情况，那就不满足第二种情况，而不满足第二种情况，意味着“$\forall \hat{\boldsymbol{A}}^{\top}\hat{\boldsymbol{y}}\leq 0,\,\text{都有}\hat{\boldsymbol{b}}_{0}^{\top}\hat{\boldsymbol{y}}\leq 0$”。而刚刚我们已经证明了$\hat{\boldsymbol{b}}_{0}^{\top}\hat{\boldsymbol{y}} + \alpha\epsilon > 0$，所以必须有$\alpha > 0$。

现在确定$\alpha > 0$了，我们就可以从式$\eqref{eq:whocare}$得到
\begin{equation}\boldsymbol{A}^{\top} \big(\boldsymbol{y}/\alpha\big) \leq \boldsymbol{c}, \quad \boldsymbol{b}^{\top} \big(\boldsymbol{y}/\alpha\big) > z^* - \epsilon\end{equation}
这意味着
\begin{equation}\max_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\} > z^* - \epsilon\end{equation}
而弱对偶形式已经告诉我们
\begin{equation}z^* \geq \max_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\}\end{equation}
也就是$\max\limits_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\}$被夹在$z^* - \epsilon$和$z^*$之间，而因为$\epsilon > 0$是任意的，所以两者可以无限接近，从而
\begin{equation}\max_{\boldsymbol{y}}\big\{\boldsymbol{b}^{\top}\boldsymbol{y}\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{y}\leq \boldsymbol{c}\big\}=z^*=\min_{\boldsymbol{x}}\big\{\boldsymbol{c}^{\top}\boldsymbol{x}\,\big|\,\boldsymbol{A}\boldsymbol{x}=\boldsymbol{b},\,\boldsymbol{x}\geq 0\big\}\end{equation}
这便是要证的强对偶形式。

#### 简单说明

Farkas引理和强对偶形式的证明，看上去比较迂回，但实际上是优化理论中非常经典和重要的证明案例，对于初学者来说，它应该是一次非常强烈的思维冲击。因为我们以往的认识中，我们对原命题做变换，仅仅是局限于“逻辑变换”，如否命题、逆否命题等。而对偶形式和Farkas引理却出现了一些“看起来很不一样，却又偏偏等价”的结论

Farkas引理和强对偶形式也可以进一步推广到一般的凸集优化问题，证明手段相似，只不过在对区域和不等关系的讨论上，比上面的线性规划的过程更为细致和复杂。不过我也不是专门搞这些优化问题的，所以只有一个模糊的认识，就不再继续班门弄斧了。

## Wasserstein GAN

好了，进行了一大通的准备工作后，我们终于可以导出Wasserstein GAN了，就本文来看，它只不过最优传输成本的线性规划对偶形式的一个副产品罢了。

### 传输成本的对偶

在推导之前，我们还是再来捋一捋本文的思路：本文介绍了W距离定义所依赖的最优传输成本定义$\eqref{eq:ot}$，然后经过分析，发现它其实就是普通线性规划问题的一个连续版本，转化过程为$\eqref{eq:lp-ot-t1},\eqref{eq:lp-ot-t2}$和$\eqref{eq:lp-ot}$；因此中间我们花了相当一部分篇幅去学习线性规划及其对偶形式，最终得到了结论$\eqref{eq:strong-dual}$。

现在我们要做的事情，就是把整个过程“逆”过来，也就是将找出$\eqref{eq:strong-dual}$对应的连续版本，为最优传输成本找一个对偶表达式。

其实这个过程也不复杂，由结论$\eqref{eq:strong-dual}$和式$\eqref{eq:lp-ot}$，我们得到
\begin{equation}\min_{\boldsymbol{\Gamma}}\big\{\langle\boldsymbol{\Gamma},\boldsymbol{C}\rangle\,\big|\,\boldsymbol{A}\boldsymbol{\Gamma}=\boldsymbol{b},\,\boldsymbol{\Gamma}\geq 0\big\}=\max_{\boldsymbol{F}}\big\{\langle\boldsymbol{b},\boldsymbol{F}\rangle\,\big|\,\boldsymbol{A}^{\top}\boldsymbol{F}\leq \boldsymbol{C}\big\}\end{equation}
注意式$\eqref{eq:lp-ot-t2}$中$\boldsymbol{b}$是由两部分拼起来的，所以我们也可以把$\boldsymbol{F}$类似地写成：
\begin{equation}\boldsymbol{F}=\begin{pmatrix}
f(\boldsymbol{x}_1) \\
f(\boldsymbol{x}_2) \\
\vdots \\
f(\boldsymbol{x}_n) \\
\vdots \\ \hline
g(\boldsymbol{y}_1) \\
g(\boldsymbol{y}_2) \\
\vdots \\
g(\boldsymbol{y}_n) \\
\vdots \\
\end{pmatrix}\end{equation}
现在$\langle\boldsymbol{b},\boldsymbol{F}\rangle$可以写成
\begin{equation}\langle\boldsymbol{b},\boldsymbol{F}\rangle=\sum_n p(\boldsymbol{x}_n) f(\boldsymbol{x}_n) + \sum_n q(\boldsymbol{x}_n) g(\boldsymbol{x}_n)\end{equation}
或者对应的积分形式是
\begin{equation}\langle\boldsymbol{b},\boldsymbol{F}\rangle=\int \big[p(\boldsymbol{x}) f(\boldsymbol{x}) + q(\boldsymbol{x}) g(\boldsymbol{x})\big]d\boldsymbol{x}\end{equation}

别忘了约束条件$\boldsymbol{A}^{\top}\boldsymbol{F}\leq \boldsymbol{C}$：
\begin{equation}\underbrace{\left( \begin{array}{ccccc|ccccc}
1 & 0 & \dots & 0 & \dots & 1 & 0 & \dots & 0 & \dots \\
1 & 0 & \dots & 0 & \dots & 0 & 1 & \dots & 0 & \dots \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots \\ \hline
0 & 1 & \dots & 0 & \dots & 1 & 0 & \dots & 0 & \dots \\
0 & 1 & \dots & 0 & \dots & 0 & 1 & \dots & 0 & \dots \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots \\ \hline
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots \\ \hline
0 & 0 & \dots & 1 & \dots & 1 & 0 & \dots & 0 & \dots \\
0 & 0 & \ddots & 1 & \ddots & 0 & 1 & \ddots & 0 & \ddots \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots \\ \hline
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots & \vdots & \ddots & \vdots & \ddots \\
\end{array} \right)}_{\Large\boldsymbol{A}^{\top}}\,\,\underbrace{\begin{pmatrix}
f(\boldsymbol{x}_1) \\
f(\boldsymbol{x}_2) \\
\vdots \\
f(\boldsymbol{x}_n) \\
\vdots \\ \hline
g(\boldsymbol{y}_1) \\
g(\boldsymbol{y}_2) \\
\vdots \\
g(\boldsymbol{y}_n) \\
\vdots \\
\end{pmatrix}}_{\Large\boldsymbol{F}}\,\,\leq\,\,\underbrace{\begin{pmatrix}
c(\boldsymbol{x}_1, \boldsymbol{y}_1) \\
c(\boldsymbol{x}_1, \boldsymbol{y}_2) \\
\vdots \\ \hline
c(\boldsymbol{x}_2, \boldsymbol{y}_1) \\
c(\boldsymbol{x}_2, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\ \hline
c(\boldsymbol{x}_n, \boldsymbol{y}_1) \\
c(\boldsymbol{x}_n, \boldsymbol{y}_2) \\
\vdots \\ \hline
\vdots \\
\end{pmatrix}}_{\Large \boldsymbol{C}}\end{equation}
代入计算后，可以发现这个诺大的矩阵运算实际上就说了这样的一件事情：
\begin{equation}\forall i,j,\,\,f(\boldsymbol{x}_i) + g(\boldsymbol{y}_j)\leq c(\boldsymbol{x}_i,\boldsymbol{y}_j)\end{equation}
或者直接写成
\begin{equation}\forall \boldsymbol{x},\boldsymbol{y},\,\,f(\boldsymbol{x}) + g(\boldsymbol{y})\leq c(\boldsymbol{x},\boldsymbol{y})\end{equation}

### 从对偶到WGAN

终于，我们就要接近尾声了，现在我们得到了最优传输成本$\eqref{eq:ot}$的一个对偶形式了：
\begin{equation}\mathcal{C}[p,q]=\max_{f,g}\Bigg\{\int \big[p(\boldsymbol{x}) f(\boldsymbol{x}) + q(\boldsymbol{x}) g(\boldsymbol{x})\big]d\boldsymbol{x} \,\Bigg|\,\, f(\boldsymbol{x}) + g(\boldsymbol{y})\leq c(\boldsymbol{x},\boldsymbol{y})\Bigg\}\end{equation}
注意到由$f(\boldsymbol{x}) + g(\boldsymbol{y})\leq c(\boldsymbol{x},\boldsymbol{y})$我们得到
\begin{equation}f(\boldsymbol{x}) + g(\boldsymbol{x})\leq c(\boldsymbol{x},\boldsymbol{x})=0\end{equation}
即$g(\boldsymbol{x}) \leq - f(\boldsymbol{x})$，所以我们有
\begin{equation}\begin{aligned}p(\boldsymbol{x}) f(\boldsymbol{x}) + q(\boldsymbol{x}) g(\boldsymbol{x})&\leq p(\boldsymbol{x}) f(\boldsymbol{x}) + q(\boldsymbol{x}) [-f(\boldsymbol{x})]\\
& = p(\boldsymbol{x}) f(\boldsymbol{x}) - q(\boldsymbol{x}) f(\boldsymbol{x})\end{aligned}\end{equation}
这似乎表明了一个结论：如果$g = -f$，它的最大值不会小于原来的最大值。事实上，这个结论不完全对，除非约定$c(\boldsymbol{x},\boldsymbol{y})$是距离（满足三角不等式），关于这个细节的进一步讨论可以参考[评论区](<https://spaces.ac.cn/archives/6280/comment-page-3#comment-22351>)。接下来我们假设$c(\boldsymbol{x},\boldsymbol{y})$是距离函数，这正是WGAN所考虑的，此时我们可以放心地让$g=-f$，从而
\begin{equation}\mathcal{C}[p,q]=\max_{f}\Bigg\{\int \big[p(\boldsymbol{x}) f(\boldsymbol{x}) - q(\boldsymbol{x}) f(\boldsymbol{x})\big]d\boldsymbol{x} \,\Bigg|\,\, f(\boldsymbol{x}) - f(\boldsymbol{y})\leq c(\boldsymbol{x},\boldsymbol{y})\Bigg\}\label{eq:ot-dual-u}\end{equation}
这便是我们最终要寻找的最优传输成本$\eqref{eq:ot}$的对偶形式了。特别地，当$c(\boldsymbol{x},\boldsymbol{y}) = \Vert \boldsymbol{x}-\boldsymbol{y}\Vert$时，我们有$\mathcal{C}[p,q] = \mathcal{W}_1[p,q]$，即
\begin{equation}\mathcal{W}_1[p,q]=\max_{f}\Bigg\{\int \big[p(\boldsymbol{x}) f(\boldsymbol{x}) - q(\boldsymbol{x}) f(\boldsymbol{x})\big]d\boldsymbol{x} \,\Bigg|\,\, f(\boldsymbol{x}) - f(\boldsymbol{y})\leq \Vert \boldsymbol{x}-\boldsymbol{y}\Vert\Bigg\}\label{eq:wd-dual-u}\end{equation}
这就是WGAN所采用的W距离，其中约束条件我们通常写为$\Vert f\Vert_{L}\leq 1$，称为Lipschitz约束。从这个过程我们也可以看到，理论上WGAN的$c(\boldsymbol{x},\boldsymbol{y})$可以是更一般的距离函数，而不单单是欧氏距离，但由于很多距离都有等价性，而这里的距离的作用只是给判别器加约束，所以选择欧氏距离其实就够了。

由于$p,q$都是概率分布，因此我们可以写成采样形式：
\begin{equation}\mathcal{W}_1[p,q]=\max_{f,\,\Vert f\Vert_{L}\leq 1}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[f(\boldsymbol{x})] - \mathbb{E}_{\boldsymbol{x}\sim q(\boldsymbol{x})}[f(\boldsymbol{x})]\end{equation}
这就是WGAN的判别器所采用的loss了，自然地，整个WGAN的训练过程就是
\begin{equation}\min_{G}\max_{f,\,\Vert f\Vert_{L}\leq 1}\mathbb{E}_{\boldsymbol{x}\sim p(\boldsymbol{x})}[f(\boldsymbol{x})] - \mathbb{E}_{\boldsymbol{z}\sim q(\boldsymbol{z})}[f(G(\boldsymbol{z}))]\end{equation}
千呼万唤的WGAN终于现身，剩下的就是怎么加Lipschitz约束的问题了，可以参考：[《深度学习中的Lipschitz约束：泛化与生成模型》](<https://spaces.ac.cn/archives/6051>)。

## 终于写完了

本文主要介绍了最优传输成本和Wasserstein距离，然后转化为一个线性规划问题，继而介绍了线性规划的对偶理论，最终导出了Wasserstein距离的对偶形式，它可以用作训练生成模型，即WGAN及后面一系列推广。

本文是笔者对线性规划及其对偶理论的一次简单学习总结，对熟悉线性代数后希望从理论上了解WGAN的读者应该会有一定的参考价值。如果对文中内容有什么疑惑或批评，欢迎留言指出。
