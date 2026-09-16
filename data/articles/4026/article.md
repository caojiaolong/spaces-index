# 【理解黎曼几何】6\. 曲率的计数与计算(Python)

> 作者：苏剑林 · 科学空间 · 2016-10-19
>
> 原文：<https://spaces.ac.cn/archives/4026>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### 曲率的独立分量

黎曼曲率张量是一个非常重要的张量，当且仅当它全部分量为0时，空间才是平直的。它也出现在爱因斯坦的场方程中。总而言之，只要涉及到黎曼几何，黎曼曲率张量就必然是核心内容。

已经看到，黎曼曲率张量有4个指标，这也意味着它有$n^4$个分量，$n$是空间的维数。那么在2、3、4维空间中，它就有16、81、256个分量了，可见，要计算它，是一件相当痛苦的事情。幸好，这个张量有很多的对称性质，使得独立分量的数目大大减少，我们来分析这一点。

首先我们来导出黎曼曲率张量的一些对称性质，这部分内容是跟经典教科书是一致的。定义
$$R_{\mu\alpha\beta\gamma}=g_{\mu\nu}R^{\nu}_{\alpha\beta\gamma} \tag{50} $$
定义这个量的原因，要谈及逆变张量和协变张量的区别，我们这里主要关心几何观，因此略过对张量的详细分析。这个量被称为完全协变的黎曼曲率张量，有时候也直接叫做黎曼曲率张量，只要不至于混淆，一般不做区分。通过略微冗长的代数运算（在一般的微分几何、黎曼几何或者广义相对论教材中都有），可以得到
$$\begin{aligned}&R_{\mu\alpha\beta\gamma}=-R_{\mu\alpha\gamma\beta}\\
&R_{\mu\alpha\beta\gamma}=-R_{\alpha\mu\beta\gamma}\\
&R_{\mu\alpha\beta\gamma}=R_{\beta\gamma\mu\alpha}\\
&R_{\mu\alpha\beta\gamma}+R_{\mu\beta\gamma\alpha}+R_{\mu\gamma\alpha\beta}=0
\end{aligned} \tag{51} $$
前两个式子很快就告诉我们，如果$\beta=\gamma$或$\mu=\alpha$，那么$R_{\mu\alpha\gamma\beta}=0$，这是反对称的结果。下面的计数表明，独立分量的数目还可以以进一步减少。我们可以将每个等式看成一个约束，有一个约束，独立分量就减少1，我们要考虑有多少个独立的约束，来得出有多少个独立分量。

首先，$R_{\mu\alpha\beta\gamma}$可以看成一个$n\times n$矩阵（前两个指标），由这个矩阵的每个元素都是一个$n\times n$矩阵（后两个指标），由$R_{\mu\alpha\beta\gamma}=-R_{\mu\alpha\gamma\beta}$可知，作为矩阵每个元素的矩阵，是一个反对称矩阵，因此只有$n(n-1)/2$个独立分量，而由$R_{\mu\alpha\beta\gamma}=-R_{\alpha\mu\beta\gamma}$知道，作为矩阵的矩阵，它也是反对称的，因此也只有$n(n-1)/2$个独立分量，也就是说，根据前两个式子就可以知道总的独立分量数不超过$[n(n-1)/2]^2$。

接着，第三、第四个约束可以为我们进一步减少分量数目。$R_{\mu\alpha\beta\gamma}=R_{\beta\gamma\mu\alpha}$表明，如果将剩下的$[n(n-1)/2]^2$个分量排成$n(n-1)/2\times n(n-1)/2$的矩阵，那么它是对称的，这时候它的独立分量数目就只有
$$\frac{1}{2}\frac{n(n-1)}{2}\left[\frac{n(n-1)}{2}+1\right] \tag{52} $$

最后一个约束$R_{\mu\alpha\beta\gamma}+R_{\mu\beta\gamma\alpha}+R_{\mu\gamma\alpha\beta}=0$是针对四个不相同指标的，因为一旦有一对相同指标，就能够转化为前三个约束的线性组合。因此，独立分量的数目还需要减少$\binom{n}{4}$：
$$\frac{1}{2}\frac{n(n-1)}{2}\left[\frac{n(n-1)}{2}+1\right]-\binom{n}{4}=\frac{n^2(n^2-1)}{12} \tag{53} $$
于是，在2、3、4、5维空间中，黎曼曲率张量的独立分量数目分别是1、6、20、50。特别地，在二维空间中（即研究三维空间中的二维曲面之时），曲率张量只有一个独立分量。

### 曲率张量的计算

<strong>下述代码有误，属于SymPy的bug，已经提交给官方，请等待修正。</strong>

很遗憾，在这里我们只能略去这个曲率张量的人工计算了，对于这个曲率张量，并没有明显简单的计算方法。利用变分方法可以简化$\Gamma^{\mu}_{\alpha\beta}$的计算，进而间接简化了曲率张量的计算。而一个自上而下的简化计算方法需要用到外微分的知识，我们目前在这个系列中没有办法做到。

但在计算机方面，利用Python中的符号计算库SymPy可以帮助我们快速地计算联络系数和曲率张量。SymPy是Python的一个符号计算库，具有小巧、开源、强大等特点。当然，如果跟商业软件Mathematica相比那肯定是没法比的，但是在某些特定的方面，其实尤有过之，也就是“不胜在全、只胜在专”了。SymPy内置了专门用来处理微分几何和张量的模块，这使得它可以非常方便地计算$\Gamma_{\alpha\beta}^{\mu}$和$R_{\alpha\beta\gamma}^{\mu}$。

一个简单的代码如下：



<pre><code class="language-python">from sympy.diffgeom import Manifold, Patch, CoordSystem
from sympy.diffgeom import TensorProduct as TP
from sympy.diffgeom import metric_to_Riemann_components as Riemann
from sympy.diffgeom import metric_to_Christoffel_2nd as Christoffel
from sympy import Symbol, Function, latex, sin

n = 2
M = Manifold('M', n)
P = Patch('P', M)

coord = CoordSystem('coord', P, ['x%s'%i for i in range(n)])
x = coord.coord_functions()
dx = coord.base_oneforms()
g = [[1, 0], [0, sin(x[0])**2]]
metric = sum([g[i][j]*TP(dx[i], dx[j]) for i in range(n) for j in range(n)])

C = Christoffel(metric)
R = Riemann(metric)</code></pre>



这是以二维球面坐标为例的，其中n定义了维数，g定义了度规矩阵，如果读者要修改，只需要修改这两个参数就好。里边的x变量是坐标的集合，dx变量是微分元的集合。为了一致，这里的指标是从0开始的，也就是遍历$0\sim n-1$。得到结果是

> &#91;&#91;&#91;0, 0&#93;, &#91;0, \-\\sin(x0)\\cdot \\cos(x0)&#93;&#93;, &#91;&#91;0, \\cos(x0)/\\sin(x0)&#93;, &#91;\\cos(x0)/\\sin(x0), 0&#93;&#93;&#93;
> &#91;&#91;&#91;&#91;0, 0&#93;, &#91;0, 0&#93;&#93;, &#91;&#91;0, \\sin(x0)\\cdot \\cdot 2&#93;, &#91;\-\\sin(x0)\\cdot \\cdot 2, 0&#93;&#93;&#93;, &#91;&#91;&#91;0, \-1&#93;, &#91;1, 0&#93;&#93;, &#91;&#91;0, 0&#93;, &#91;0, 0&#93;&#93;&#93;&#93;

可以看到，程序会把所有的分量都列出来，如果我们只关心非0分量，那么可以用以下代码输出：



<pre><code class="language-python">for i1 in range(n):
    for i2 in range(n):
        for i3 in range(n):
            if C[i1, i2, i3]:
                print 'Gamma^%s_%s%s:'%(i1, i2, i3), C[i1, i2, i3]

for i1 in range(n):
    for i2 in range(n):
        for i3 in range(n):
            for i4 in range(n):
                if R[i1, i2, i3, i4]:
                    print 'Riemann^%s_%s%s%s:'%(i1, i2, i3, i4), R[i1, i2, i3, i4]</code></pre>



得到

> Gamma^0\_11: \-\\sin(x0)\\cdot \\cos(x0)
> Gamma^1\_01: \\cos(x0)/\\sin(x0)
> Gamma^1\_10: \\cos(x0)/\\sin(x0)
> Riemann^0\_101: \\sin(x0)\\cdot \\cdot 2
> Riemann^0\_110: \-\\sin(x0)\\cdot \\cdot 2
> Riemann^1\_001: \-1
> Riemann^1\_010: 1

注意，这里的曲率张量算的是$R_{\alpha\beta\gamma}^{\mu}$，如果读者关心$R_{\mu\alpha\beta\gamma}$，但SymPy并没有现成的函数，我们可以自己用张量运算库来算，即



<pre><code class="language-python">from sympy.tensor.array import tensorproduct, tensorcontraction
tensorcontraction(tensorproduct(g, R), (1, 2))</code></pre>



其中tensorproduct是用来做张量积的，tensorcontraction是用来缩并的。得到

> &#91;&#91;&#91;&#91;0, 0&#93;, &#91;0, 0&#93;&#93;, &#91;&#91;0, \\sin(x0)\\cdot \\cdot 2&#93;, &#91;\-\\sin(x0)\\cdot \\cdot 2, 0&#93;&#93;&#93;, &#91;&#91;&#91;0, \-\\sin(x0)\\cdot \\cdot 2&#93;, &#91;\\sin(x0)\\cdot \\cdot 2, 0&#93;&#93;, &#91;&#91;0, 0&#93;, &#91;0, 0&#93;&#93;&#93;&#93;

最后如果要输出为LaTeX代码，只需要用latex命令转就行了，比如



<pre><code class="language-python">for i1 in range(n):
    for i2 in range(n):
        for i3 in range(n):
            if C[i1, i2, i3]:
                print 'Gamma^%s_%s%s:'%(i1, i2, i3), \
                      latex(C[i1, i2, i3]).replace('\\mathrm{x_', '{x_').replace('\\boldsymbol{{x_', '{{x_')</code></pre>



最后的replace是做一些简单的微调，使得结果更符合我们平时的写法，结果是

> Gamma^0\_11: \- \\sin\{\\left (\{\{x\_\{0\}\}\} \\right )\} \\cos\{\\left (\{\{x\_\{0\}\}\} \\right )\}
> Gamma^1\_01: \\frac\{\\cos\{\\left (\{\{x\_\{0\}\}\} \\right )\}\}\{\\sin\{\\left (\{\{x\_\{0\}\}\} \\right )\}\}
> Gamma^1\_10: \\frac\{\\cos\{\\left (\{\{x\_\{0\}\}\} \\right )\}\}\{\\sin\{\\left (\{\{x\_\{0\}\}\} \\right )\}\}

最后我们可能需要对含有待定函数的度量求曲率，比如$ds^2=f(x,y)(dx^2+dy^2)$，这时候大概的代码是：



<pre><code class="language-python">f = Function('f')
g = [[f(x[0], x[1]), 0], [0, f(x[0], x[1])]]
metric = sum([g[i][j]*TP(dx[i], dx[j]) for i in range(n) for j in range(n)])

C = Christoffel(metric)
R = Riemann(metric)</code></pre>



这时候在$\Gamma_{\alpha\beta}^{\mu}$中，会出现这样的项：

> Subs(Derivative(f(\_xi\_1, x1), \_xi\_1), (\_xi\_1,), (x0,))/(2\\cdot f(x0, x1))

转换为LaTeX是（跟前面一样做了一下简单的处理，替换掉boldsymbol和mathrm标签）

> \\frac\{\\left\. \\frac\{d\}\{d \\xi\_\{1\}\} f\{\\left (\\xi\_\{1\},\{\{x\_\{1\}\}\} \\right )\} \\right\|\_\{\\substack\{ \\xi\_\{1\}=\{\{x\_\{0\}\}\} \}\}\}\{2 f\{\\left (\{\{x\_\{0\}\}\},\{\{x\_\{1\}\}\} \\right )\}\}

输出效果是
$$\frac{\left. \frac{d}{d \xi_{1}} f{\left (\xi_{1},{{x_{1}}} \right )} \right|_{\substack{ \xi_{1}={{x_{0}}} }}}{2 f{\left ({{x_{0}}},{{x_{1}}} \right )}}$$
实际上就是
$$\frac{ \frac{\partial }{\partial x_0} f\left (x_0, x_1 \right ) }{2 f{\left ({{x_{0}}},{{x_{1}}} \right )}}$$
可见，SymPy的输出还有待改善呀，很多地方还是需要人工优化一下的。更糟糕的是，$R_{\alpha\beta\gamma}^{\mu}$中出现了这样的项：

> \-Subs(Derivative(f(\_\#\_1(\_Dummy\_38), x1), \_\#\_1(\_Dummy\_38), \_\#\_1(\_Dummy\_38)), (\_\#\_1(\_Dummy\_38),), (x0,))/f(x0, x1) \- Subs(Derivative(f(x0, \_\#\_0(\_Dummy\_38)), \_\#\_0(\_Dummy\_38), \_\#\_0(\_Dummy\_38)), (\_\#\_0(\_Dummy\_38),), (x1,))/f(x0, x1) \+ Subs(Derivative(f(\_xi\_1, x1), \_xi\_1), (\_xi\_1,), (x0,))\\cdot \\cdot 2/(2\\cdot f(x0, x1)\\cdot \\cdot 2) \+ Subs(Derivative(f(x0, \_xi\_2), \_xi\_2), (\_xi\_2,), (x1,))\\cdot \\cdot 2/(2\\cdot f(x0, x1)\\cdot \\cdot 2)

转换为LaTeX后是

> \- \\frac\{1\}\{f\{\\left (\{\{x\_\{0\}\}\},\{\{x\_\{1\}\}\} \\right )\}\} \\left\. \\frac\{d^\{2\}\}\{d \\operatorname\{\_\#\_\{1\}\}\{\\left (Dummy\_\{38\} \\right )\}^\{2\}\} f\{\\left (\\operatorname\{\_\#\_\{1\}\}\{\\left (Dummy\_\{38\} \\right )\},\{\{x\_\{1\}\}\} \\right )\} \\right\|\_\{\\substack\{ \\operatorname\{\_\#\_\{1\}\}\{\\left (Dummy\_\{38\} \\right )\}=\{\{x\_\{0\}\}\} \}\} \- \\frac\{1\}\{f\{\\left (\{\{x\_\{0\}\}\},\{\{x\_\{1\}\}\} \\right )\}\} \\left\. \\frac\{d^\{2\}\}\{d \\operatorname\{\_\#\_\{0\}\}\{\\left (Dummy\_\{38\} \\right )\}^\{2\}\} f\{\\left (\{\{x\_\{0\}\}\},\\operatorname\{\_\#\_\{0\}\}\{\\left (Dummy\_\{38\} \\right )\} \\right )\} \\right\|\_\{\\substack\{ \\operatorname\{\_\#\_\{0\}\}\{\\left (Dummy\_\{38\} \\right )\}=\{\{x\_\{1\}\}\} \}\} \+ \\frac\{\\left\. \\frac\{d\}\{d \\xi\_\{1\}\} f\{\\left (\\xi\_\{1\},\{\{x\_\{1\}\}\} \\right )\} \\right\|\_\{\\substack\{ \\xi\_\{1\}=\{\{x\_\{0\}\}\} \}\}^\{2\}\}\{2 f^\{2\}\{\\left (\{\{x\_\{0\}\}\},\{\{x\_\{1\}\}\} \\right )\}\} \+ \\frac\{\\left\. \\frac\{d\}\{d \\xi\_\{2\}\} f\{\\left (\{\{x\_\{0\}\}\},\\xi\_\{2\} \\right )\} \\right\|\_\{\\substack\{ \\xi\_\{2\}=\{\{x\_\{1\}\}\} \}\}^\{2\}\}\{2 f^\{2\}\{\\left (\{\{x\_\{0\}\}\},\{\{x\_\{1\}\}\} \\right )\}\}

事实上这还会报错！这时候我们还需要把诸如\\operatorname\{\_\#\_\{0\}\}\{\\left (Dummy\_\{38\} \\right )\}这些内容替换为普通的符号，比如



<pre><code class="language-python">print latex(R[0,1,0,1]).replace('\\mathrm{x_', '{x_').replace('\\boldsymbol{{x_', '{{x_')\
.replace('\\operatorname{_#_{0}}{\\left (Dummy_{38} \\right )}', '\\xi')\
.replace('\\operatorname{_#_{1}}{\\left (Dummy_{38} \\right )}', '\\xi')</code></pre>



这才得到一个没报错、有点靠谱的输出：
$$\begin{aligned}&- \frac{1}{f{\left ({{x_{0}}},{{x_{1}}} \right )}} \left. \frac{d^{2}}{d \xi^{2}} f{\left (\xi,{{x_{1}}} \right )} \right|_{\substack{ \xi={{x_{0}}} }} - \frac{1}{f{\left ({{x_{0}}},{{x_{1}}} \right )}} \left. \frac{d^{2}}{d \xi^{2}} f{\left ({{x_{0}}},\xi \right )} \right|_{\substack{ \xi={{x_{1}}} }} \\
&+ \frac{\left. \frac{d}{d \xi_{1}} f{\left (\xi_{1},{{x_{1}}} \right )} \right|_{\substack{ \xi_{1}={{x_{0}}} }}^{2}}{2 f^{2}{\left ({{x_{0}}},{{x_{1}}} \right )}} + \frac{\left. \frac{d}{d \xi_{2}} f{\left ({{x_{0}}},\xi_{2} \right )} \right|_{\substack{ \xi_{2}={{x_{1}}} }}^{2}}{2 f^{2}{\left ({{x_{0}}},{{x_{1}}} \right )}}\end{aligned}$$
最后整理为
$$-\frac{\frac{\partial^2 f(x_0, x_1)}{\partial x_0^2}}{f(x_0,x_1)} -\frac{\frac{\partial^2 f(x_0, x_1)}{\partial x_1^2}}{f(x_0,x_1)} + \frac{\left(\frac{\partial f(x_0, x_1)}{\partial x_0}\right)^2}{2f^2 (x_0,x_1)} + \frac{\left(\frac{\partial f(x_0, x_1)}{\partial x_1}\right)^2}{2f^2 (x_0,x_1)}$$
或者
$$-\frac{\nabla^2 f}{f}+\frac{1}{2}\left|\frac{\nabla f}{f}\right|^2$$
