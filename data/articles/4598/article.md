# 从马尔科夫过程到主方程（推导过程）

> 作者：苏剑林 · 科学空间 · 2017-10-06
>
> 原文：<https://spaces.ac.cn/archives/4598>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

主方程（master equation）是对随机过程进行建模的重要方法，它代表着马尔科夫过程的微分形式，我们的专业主要工具之一就是主方程，说宏大一点，量子力学和统计力学等也不外乎是主方程的一个特例。

然而，笔者阅读了几个著作，比如《统计物理现代教程》，还有我导师的《生物系统的随机动力学》，我发现这些著作对于主方程的推导都很模糊，他们在着力解释结果的意义，但并不说明结果的思想来源，因此其过程难以让人信服。而知乎上有人提问[《如何理解马尔科夫过程的主方程的推导过程？》](<https://www.zhihu.com/question/45471276>)但没有得到很好的答案，也表明了这个事实。

## 马尔可夫过程

主方程是用来描述马尔科夫过程的，而马尔科夫过程可以理解为运动的无记忆性，说通俗点，就是下一刻的概率分布，只跟当前时刻有关，跟历史状态无关。用概率公式写出来就是（这里只考虑连续型概率，因此这里的$p$是概率密度）：
$$\begin{equation}\label{eq:maerkefu}p(x,\tau)=\int p(x,\tau|y,t) p(y,t) dy\end{equation}$$
这里的积分区域是全空间。这里的$p(x,\tau|y,t)$称为跃迁概率，即已经确定了$t$时刻来到了$y$位置后、在$\tau$时刻达到$x$的概率密度，这个式子的物理意义是很明显的，就不多做解释了。

尽管$\eqref{eq:maerkefu}$式很直观，但用它来建模实际上存在两个比较困难的问题

> 1、为了建模就得写出$p(x,\tau|y,t)$，而事实上我们很难直接写出一个合理的跃迁概率出来；
> 
> 
> 
> 2、即使写出来$p(x,\tau|y,t)$，方程是一个积分方程，而我们对积分方程的研究远不如微分方程。

因此，有必要写出它的微分方程形式。

## 主方程

我们设$\tau = t+\epsilon$
$$\begin{equation}\label{eq:maerkefu-2}p(x,t+\epsilon)=\int p(x,t+\epsilon|y,t) p(y,t) dy\end{equation}$$
并且考虑$\epsilon\to 0$的极限，保留到$\epsilon$的一阶项，我们有
$$\begin{equation}\label{eq:yueqian}p(x,t+\epsilon|y,t)=\delta(x-y)+\epsilon \tilde{W}(x,y,t)\end{equation}$$
其中用到了$p(x,t|y,t)=\delta(x-y)$这个事实，而
$$\begin{equation}\tilde{W}(x,y)=\left.\frac{\partial p(x,\tau|y,t)}{\partial \tau}\right|_{\tau=t}\end{equation}$$
我们对$\eqref{eq:maerkefu-2}$式两边都展开$\epsilon$到一阶项，得到
$$\begin{equation}p(x,t)+\epsilon\frac{\partial p(x,t)}{\partial t}=\int \left[\delta(x-y)+\epsilon \tilde{W}(x,y,t)\right] p(y,t) dy\end{equation}$$
故
$$\begin{equation}\label{eq:zhufangcheng-1}\frac{\partial p(x,t)}{\partial t}=\int \tilde{W}(x,y,t) p(y,t) dy\end{equation}$$

## 更便于建模

读者可以发现，$\eqref{eq:zhufangcheng-1}$式并非我们常见的主方程的形式，这是因为它不方便建模。我们再来看$\eqref{eq:yueqian}$式，注意我们有
$$\begin{equation}\int p(x,t+\epsilon|y,t) dx = 1\end{equation}$$
因此结合$\eqref{eq:yueqian}$式，则必须要求
$$\begin{equation}\label{eq:yueshu}\int \tilde{W}(x,y,t)dx = 0\end{equation}$$
科研中的建模过程是反过来的：<u>需要先写出主方程的形式，然后去求解主方程。也就是说，为了用$\eqref{eq:zhufangcheng-1}$式建模，则需要写出$\tilde{W}(x,y,t)$，而$\tilde{W}(x,y,t)$要满足$\eqref{eq:yueshu}$式的约束条件，但我们很难凭空写出一个$\tilde{W}(x,y,t)$同时又满足这个约束。</u>

幸运的是，我们可以用一个技巧来去掉这个约束。首先我们写出一个任意的函数$W(x,y,t)$，然后考虑
$$\begin{equation}\begin{aligned}&\int W(x,y,t)dx\\
=&\iint \delta(y-z) W(x,z,t)dxdz \quad (\text{接下来交换}x,z\text{的位置})\\
=&\iint \delta(y-x) W(z,x,t)dzdx\end{aligned}\end{equation}$$
从而我们可以让
$$\begin{equation}\tilde{W}(x,y,t) = W(x,y,t) - \int \delta(y-x) W(z,x,t)dz\end{equation}$$
那么它自动地满足$\eqref{eq:yueshu}$式，最终主方程的形式变为
$$\begin{equation}\label{eq:zhufangcheng-2}\frac{\partial p(x,t)}{\partial t}=\int \Big[W(x,y,t)p(y,t)-W(y,x,t)p(x,t)\Big] dy\end{equation}$$
这就是我们在教科书上能看到的主方程的形式，这时候对$W(x,y,t)$则没有特别的约束了，因此可以方便地用它来建模。至于对$W(y,x,t)$的物理意义的诠释，则是后话了。类似的推导过程可以平移到离散型的主方程，也不再赘述。

## 一点评价

从上面的过程可以看到，主方程之所以是我们经常看到的形式，是因为那种形式更方便我们进行建模，至于对结果的诠释，本质上来说都是强加的，不能算作推导过程。而多数著作对主方程的推导，侧重于对结果的物理诠释，对方程的形式的来源不加详述，是引起我们理解上的困难的重要原因之一，希望这里的文字可以对此做些补充。
