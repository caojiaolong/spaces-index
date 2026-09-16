# 变分与理论力学略览

> 作者：苏剑林 · 科学空间 · 2011-04-04
>
> 原文：<https://spaces.ac.cn/archives/1304>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![拉格朗日](<https://spaces.ac.cn/usr/uploads/2011/04/8556167.jpg>)](<https://spaces.ac.cn/attachment/1306/>)

拉格朗日

BoJone在之前的[《自然极值》系列](<https://spaces.ac.cn/tag/%E6%9E%81%E5%80%BC/>)已经花了一定篇幅来讲述“极值”在自然界中是多么的普遍，它能够引导我们进行某些问题的思考，从而获得简单快捷的解答。接下来，我要说的一个更加令人惊讶的“事实”：<u>“极值”不仅仅在某些数学或物理问题上给予我们创造性的思考，它甚至构建了整个经典力学乃至于整个物理学！</u>这不是夸大其辞，这是物理学中被称为“最小作用量原理”的一个原理，很多物理学家（如费恩曼）被它深深吸引着，甚至认为它就是<u>“上帝创造世界的终极公式”</u>！（关于做小作用量原理，大家不妨看一下[范翔](<http://www.eaglefantasy.com/>)所写的[《最小作用量原理与物理之美》](<https://spaces.ac.cn/usr/uploads/2011/04/800511278.rar>)系列文章）

话说在18世纪，<strong>欧拉</strong>和<strong>拉格朗日</strong>开创了一条独特的道路，即用变分法来研究经典力学，从而使经典力学焕发出了新的活力，也由此衍生出了一个叫“理论力学”或“分析力学”的分支。用变分法研究力学有很多的好处，变分的对象一般都是标量函数，我们只需要写出动力系统的动能与势能表达式，就可以进行一系列的研究，比如<u>列出质点的运动方程、判断平衡点的稳定性、求周期轨道</u>等等（由于BoJone对理论力学研究还不够深入，无法举太多例子，但请相信，其作用远远不止这些），省去了不少繁琐的矢量性分析，这些都是在变分法发明前难以研究的。

理论力学的内容很广泛，其基础是<strong>“最小作用量原理”</strong>。对于某一种力学，只要找出一个“最小作用量”，就可以用变分法来<u>构建出整个力学体系</u>！因此，除了研究经典力学外，量子力学、狭义相对论、广义相对论都能够用类似的方法来研究，因为我们都可以写出它们各自的作用量。例如，经典力学的作用量很简单，它可以写成
$$S=\int_{t_1}^{t_2} L(\vec{r},\dot{\vec{r}},t)dt$$

要注意的是，这里的$\vec{r}$不在具有向量的意义，它只是一组广义坐标组$(x_1,x_2,...x_n)$。你或许要问：<strong>L是什么？</strong>L叫做“<u>拉格朗日函数</u>”，由于只是作“略览”，因此我们只考虑保守系统，即L函数中不出现时间t的情况。这时，有
$$L=T-U$$
T是系统的总动能，U是总位能（势能）。一般而言，$T=T(\dot{\vec{r}}),U=U(\vec{r})$，即T中不显含$\vec{r}$，U中也不显含$\dot{\vec{r}}$。

已经写了这么多关于理论力学的描述了，不过介绍理论力学却不是本文的主要目的，关于更多的变分、力学内容可以参考<u>《变分法、有限元法和外推法》、《经典力学中的数学方法》</u>等书籍。本文主要<u>不加证明</u>（证明方法可以类比《自然极值》最后一篇）地介绍一些变分法中的各种“欧拉\-拉格朗日方程”的形式与特解，并简单结合力学看看应用。

<strong>一、含有各阶导数的泛函变分（多阶）</strong>

若$S=\int_{t_1}^{t_2} F(x,\dot{x},\ddot{x},...,x^{(n)},t)dt$取得极值（即它的变分$\delta S=0$），那么欧拉\-拉格朗日方程为

$$\sum_{i=0}^n (-1)^n \frac{\partial^n}{\partial t^n}(\frac{\partial F}{\partial x^{(n)}})=0\tag{1}$$
对于n=1，有
$$\frac{\partial F}{\partial x}-\frac{d}{dt}(\frac{\partial F}{\partial \dot{x}})=0$$
对于n=2，有
$$\frac{\partial F}{\partial x}-\frac{d}{dt}(\frac{\partial F}{\partial \dot{x}})+\frac{d^2}{dt^2}(\frac{\partial F}{\partial \ddot{x}})=0$$
\.\.\.\.\.\.

由于含有多阶导数的变分在物理上应用并不十分广泛，因而这里不做进一步的讨论。

<strong>二、含有多个变量的泛函变分（多元）</strong>

设$\vec{r}=(x_1,x_2,...,x_n)$，记$F(x_1,x_2,...,x_n,\dot{x}_1,\dot{x}_2,...\dot{x}_n,t)$为$F(\vec{r},\dot{\vec{r}},t)$，若$S=\int_{t_1}^{t_2} F(\vec{r},\dot{\vec{r}},t)$取极值，则

$$\frac{\partial F}{\partial \vec{r}}-\frac{d}{dt}(\frac{\partial F}{\partial \vec{r}})=0\tag{2}$$

> 这里有必要对符号$\frac{\partial }{\partial \vec{r}}$进行一个简单解释。这里的$\frac{\partial F}{\partial \vec{r}}=(\frac{\partial F}{\partial x_1},\frac{\partial F}{\partial x_2},...,\frac{\partial F}{\partial x_n})$，表示一个向量。很显然，$\frac{\partial F}{\partial \vec{r}}*d\vec{r}=dF$（$\frac{\partial }{\partial \dot{\vec{r}}}$可以类比）
> 
> 
> 
> 因此运算$\frac{\partial }{\partial \vec{r}}$可以看作是路径积分的逆运算，即
> $\int f(\vec{r})*d\vec{r}=F(\vec{r})$，那么$\frac{\partial F}{\partial \vec{r}}=f(\vec{r})$，f是一个矢量函数。
> 
> 
> 
> 下面是一些很简单的运算公式：
> $\frac{\partial}{\partial \vec{r}} (\vec{c}*\vec{r})=\vec{c}$（$\vec{c}$是一个常向量）
> $$\begin{aligned}\frac{\partial }{\partial \vec{r}}(\vec{r}\cdot \vec{r})=2\vec{r} \\ \frac{\partial |\vec{r}|}{\partial \vec{r}}=\frac{\partial }{\partial \vec{r}}(\sqrt{\vec{r}\cdot \vec{r}})=(\frac{1}{2\sqrt{\vec{r}\cdot \vec{r}}})\times \frac{\partial }{\partial \vec{r}}(\vec{r}\cdot \vec{r})=\frac{\vec{r}}{|\vec{r}|}\end{aligned}$$
> 其他形式的运算大多数都可以类似于一般的求导运算，所以不用细细说明啦^\_^
> 这里专门把这三个提出来，是因为这三个的原函数和运算结果都可以用$\vec{r}$来表示，不需要进行坐标分解，显得有规律。
> 最后，给出
> $$\frac{\partial }{\partial \vec{r}}[f(g(\vec{r}))]=(\frac{df}{dg})\frac{\partial }{\partial \vec{r}}[g(\vec{r})]$$

<strong>三、经典力学</strong>
有了变分法，牛顿的经典力学可以重新写成
$$\frac{\partial L}{\partial \vec{r}}=\frac{d}{dt}(\frac{\partial L}{\partial \dot{\vec{r}}})\tag{3}$$
L的意义在文章开头已经说明了。这里的$\vec{r}=(x_1,x_2,...,x_n)$是一组<strong>广义坐标</strong>，即不一定要直角坐标系，极坐标、椭圆坐标等都行。

要是你觉得上述形式不好看，我们可以改写成
$$\begin{aligned}\vec{p}=\frac{\partial L}{\partial \dot{\vec{r}}} \\ \dot{\vec{p}}=\frac{\partial L}{\partial \vec{r}}\end{aligned}$$
这是<u>多么对称</u>的表达形式！！由此可见物理之美！！
（证明方法请参考《理论力学》等教程）

<strong>四、一些特解</strong>

要求(2)并没有通用的方法，但是当对函数F给定一定的限制时，(2)能够求出一些首次积分。

<strong>第一种</strong>情况是函数F中不显含时间t的情况。我们可以写出：
$$\begin{aligned}\frac{\partial F}{\partial \vec{r}}-\frac{d}{dt}(\frac{\partial F}{\partial \dot{\vec{r}}})=0 \\ \frac{\partial F}{\partial \vec{r}}\cdot \vec{r}-\frac{d}{dt}(\frac{\partial F}{\partial \vec{r}})\cdot \vec{r}=0 \\ [\frac{\partial F}{\partial \vec{r}}\cdot \dot{\vec{r}}+\frac{\partial F}{\partial \dot{\vec{r}}}\cdot \ddot{\vec{r}}]-[\frac{d}{dt}(\frac{\partial F}{\partial \dot{\vec{r}}})\cdot \dot{\vec{r}}+\frac{\partial F}{\partial \dot{\vec{r}}}\cdot \ddot{\vec{r}}]=0\end{aligned}$$

两个中括号内的式子可以“缩并”成
$$\frac{dF}{dt}-\frac{d}{dt}(\frac{\partial F}{\partial \dot{\vec{r}}}\cdot \dot{\vec{r}})=0$$

积分即得到
$$F-\frac{\partial F}{\partial \dot{\vec{r}}}\cdot \dot{\vec{r}}=C\tag{4}$$
C是常数。如果将F换成经典力学系统的拉格朗日函数L，那么<u>这个积分对应着经典力学的“<strong>能量守恒定律</strong>”</u>（有兴趣者自己演算一下？）

<strong>第二种</strong>情况是F中不显含任意坐标$x_i$的情况（但是允许$\dot{x}_i$存在），那么很显然就有
$\frac{\partial F}{\partial x_i}=0$，即
$$\frac{d}{dt}(\frac{\partial F}{\partial \dot{x}_i})=\frac{\partial F}{\partial x_i}=0$$
于是可以得到积分
$$\frac{\partial F}{\partial \dot{x}_i}=C\tag{5}$$
这时，$x_i$被称为<u>循环坐标</u>。出现一个循环坐标，就对应着一个首次积分。于是求积动力系统就变成了寻找循环坐标的问题了。

<strong>五、总结</strong>
BoJone不才，只是略略浏览了理论力学和变分法的相关教材，加上文笔能力有限，无法将力学之美展示给各位读者。因此，欢迎各位读者能不吝赐教，大家相互交流，相互进步，一起领略无限的物理之美！！我们要告诉大家：<strong><u>物理并不如想象中枯燥、深不可测，而是出处洋溢着美的视角！！</u></strong>
