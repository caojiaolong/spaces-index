# 路径积分系列：4\.随机微分方程

> 作者：苏剑林 · 科学空间 · 2016-06-09
>
> 原文：<https://spaces.ac.cn/archives/3762>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

本章将路径积分用于随机微分方程，并且得到了与不对称随机游走一样的结果，从而证明了它与该模型的等价性\.

<strong>将路径积分用于随机微分方程的研究，这一思路由来已久\. 费曼在他的著作&#91;5&#93;中，已经建立了路径积分与线性随机微分方程的关系\. 而对于非线性的情况，也有不少研究，但比较混乱，如文献&#91;8&#93;甚至给出了错误的结果\. </strong>

本文从路径积分的离散化概念出发，明确地建立了两个路径积分微元的雅可比行列式关系，从而对非线性随机微分方程也建立了路径积分\. 本文的结果跟文献&#91;9&#93;的结果是一致的\.

### 概念

本文所研究的仅仅是随机常微分方程，它与一般的常微分方程的区别在于布朗运动项的引入，如常见的一类随机微分方程为
$$dx(t)=p(x(t),t)dt + \sqrt{\alpha} dW_t.\tag{48}$$
其中$W_t$代表着一个标准的布朗运动\. 由于引入了随机项，所以解$x(t)$不再是确定的，而是有一定的概率分布\.

在对随机微分方程中，感兴趣的量有很多，比如关于$x$的某个量的期望、方差，或者稳定性，等等\. 随机微分方程领域中有各种分析的技巧，但是显然，直接求出$x(t)$的概率分布后对概率分布进行研究，是最理想最容易的方案\. 路径积分正是给出了求概率分布的一个方法\.

### 线性随机微分方程

我们从下述线性随机微分方程出发
$$dx(t)=x(t)dt + \sqrt{\alpha} dW_t,\tag{49}$$
我们希望求解这个方程，我们所关注的是$x$的概率分布，因此求解的结果是找出$t_a$时刻从$x_a$出发、在$t_b$时刻到达$x_b$的概率，也就是<strong>传播子（Green函数）</strong>\. 为了构建路径积分，我们需要找到从$(t_a,x_a)$出发，经由路径$x(t)$达到$(t_b,x_b)$的概率泛函$ P [x(t)]$\.

我们可以从布朗运动的概率泛函出发，由于给定了$W(t)$就可以唯一地算出$x(t)$，因此，给定$x(t)$与给定$W(t)$的几率相同，即
$$ P [x(t)]\mathscr{D}x(t)= P [W(t)]\mathscr{D}W(t).\tag{50}$$
我们已经知道，布朗运动中$W(t)$的概率密度泛函
$$ P [W(t)]=\exp\left(-\int_{t_a}^{t_b} \frac{1}{2}\dot{W}^2(t)dt\right).\tag{51}$$
注意到，我们仍然写成导数的形式，这只是形式上的（路径积分的定义中的导数项本来就只是形式上的而已），便于理解和记忆的符号\. 现在有
$$\begin{aligned} P [x(t)]\mathscr{D}x(t)=& P [W(t)]\mathscr{D}W(t)\\
=&\exp\left(-\int_{t_a}^{t_b} \frac{1}{2}\dot{W}^2(t)dt\right)\mathscr{D}W(t)\\
=&\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}(t)-x(t)\right]^2 dt\right)\mathscr{D}W(t)\end{aligned},\tag{52}$$
到目前为止，我们做的事情都是容易理解的，只不过是变量代换\. 现在缺的是最后一步了，$\mathscr{D}x(t)$与$\mathscr{D}W(t)$的关系\. $\mathscr{D}x(t)$与$\mathscr{D}W(t)$定义为$n-1$重积分的极限，而有限重积分的变换代换，微元之间的变换相差一个雅可比行列式，那么这两者之间也应该相差一个雅可比行列式：
$$\mathscr{D}W(t)=\mathcal{J}[x(t)]\mathscr{D}x(t),\tag{53}$$
由于$\mathscr{D}$的无穷性，因此这里的雅可比行列式是一个$\infty\times\infty$维度的方阵的行列式\.

注意到$W(t)$与$x(t)$是线性的关系，我们已经知道线性变换的雅可比行列式是一个常数，这里也不例外，因此我们可以先不管这个常数，等到求出结果后再通过归一化来求出常数\. 在相差一个常数因子的情况下，我们有
$$\begin{aligned} P [x(t)]\mathscr{D}x(t)=&\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}(t)-x(t)\right]^2 dt\right)\mathscr{D}W(t)\\
=&\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}(t)-x(t)\right]^2 dt\right)\mathscr{D}x(t)\end{aligned},\tag{54}$$
也就是说
$$ P [x(t)]=\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}(t)-x(t)\right]^2 dt\right).\tag{55}$$
有了这个结果，我们就可以用路径积分来算传播子了：
$$\begin{aligned}\int P [x(t)]\mathscr{D}x(t)=&\int\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}^2(t)+x^2(t)-2\dot{x}(t) x(t)\right] dt\right)\mathscr{D}x(t)\\
=&\exp\left(\frac{1}{2\alpha}\left(x_b^2-x_a^2\right)\right)\times\\
&\qquad\int\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}^2(t)+x^2(t)\right] dt\right)\mathscr{D}x(t)\end{aligned}.\tag{56}$$
其中的路径积分部分
$$K(x_b,t_b;x_a,t_a)=\int\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}^2(t)+x^2(t)\right] dt\right)\mathscr{D}x(t)\tag{57}$$
就是基本的高斯型路径积分，我们可以精确求解\.

### 计算雅可比行列式

现在我们来推导一般形式的随机微分方程
$$dx(t)=p(x(t), t)dt + \sqrt{\alpha} dW_t ,\tag{58}$$
的路径积分\. 跟上一节一样，很容易得到
$$\begin{aligned} P [x(t)]\mathscr{D}x(t)=& P [W(t)]\mathscr{D}W(t)\\
=&\exp\left(-\int_{t_a}^{t_b} \frac{1}{2}\dot{W}^2 dt\right)\mathscr{D}W(t)\\
=&\exp\left(-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}-p(x,t)\right]^2 dt\right)\mathscr{D}W(t)\end{aligned}.\tag{59}$$

重点就是$\mathscr{D}x(t)$与$\mathscr{D}W(t)$的关系，在线性的情况下，我们知道它是一个常数，从而可以不管它，等最后归一化的时候再确定；然而在非线性的情况就没办法绕过它，只能想办法算出它的行列式\.

将方程离散化为
$$(x_k - x_{k-1}) - \frac{1}{2}\epsilon\left[p(x_k,t_k)+p(x_{k-1},t_{k-1})\right]=\sqrt{\alpha} (W_k-W_{k-1}),\tag{60}$$
其中$k=0,1,2,\dots,n$，而$W_0\equiv W(t_a),W_n\equiv W(t_b),x_0\equiv x(t_a),x_k\equiv x(t_b)$是预先给定的初始条件，$\epsilon=\frac{t_b-t_a}{n}$是时间间隔，$x_k\equiv x\left(\frac{k\epsilon}{n}\right)$\. 注意我们对$p(x,t)$取了平均\. 这是必要的，不取平均则会得到错误的结果\. 因为$\frac{x(t)-x(t-\epsilon)}{\epsilon}$作为$\dot{x}(t)$的近似，只有零阶的精确度，而事实上它最近于$\dot{x}(t-\epsilon/2)$，这时候它具有$\epsilon$的一阶精确度\. 也就是说，当我们写下$(x_k-x_{k-1})/\epsilon$时，实际上它表达的是$x_k$与$x_{k-1}$之间的中点的导数\. 因此，$p(x,t)$的离散化也要取中点部分，可以取的有$\frac{1}{2}\left[p(x_k,t_k)+p(x_{k-1},t_{k-1})\right]$或$p\left(\frac{x_k+x_{k-1}}{2},\frac{t_k+t_{k-1}}{2}\right)$，两者都具有同等的精确度，得到的结果也是一样的(在积分近似中，我们可以用梯形或者中点矩形来近似曲边梯形的面积，具有类似的含义)\.

我们写出
$$\left\{\begin{aligned}&\mathscr{D}x(t)\approx dx_1 dx_2\dots dx_{k-1}\\
&\mathscr{D}W(t)\approx dW_1 dW_2\dots dW_{k-1}\end{aligned}\right. .\tag{61}$$
然后用有限维微积分的方法求得雅可比行列式（的近似值），注意可以算得，从$W_k$到$x_k$的雅可比矩阵是一个三角阵，因此行列式是对角线的乘积，而对角线元素为
$$\frac{\partial W_k}{\partial x_k}=\frac{1}{\sqrt{\alpha}}\left(1-\frac{1}{2}\epsilon\frac{\partial p(x_k,t_k)}{\partial x_k}\right)\approx \frac{1}{\sqrt{\alpha}}\exp\left(-\frac{1}{2}\epsilon\frac{\partial p(x_k,t_k)}{\partial x_k}\right),\tag{62}$$
因此在相差一个常数的情况下，有
$$\mathcal{J}[x(t)]\approx \prod_{k=1}^{n-1} \left(1-\frac{1}{2}\epsilon\frac{\partial p(x_k,t_k)}{\partial x_k}\right)\approx \exp\left[-\frac{1}{2}\left(\sum_{k=1}^{n-1}\frac{\partial p(x_k,t_k)}{\partial x_k}\right)\epsilon\right],\tag{63}$$
取$n\to\infty$的极限，我们有
$$\mathcal{J}[x(t)]=\exp\left(-\frac{1}{2}\int_{t_a}^{t_b}\frac{\partial p(x)}{\partial x}dt\right),\tag{64}$$
代回得到
$$\begin{aligned} P [x(t)]\mathscr{D}x(t)=&\exp\left\{-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\dot{x}-p(x,t)\right]^2 dt\right\}\\
&\qquad\qquad\exp\left(-\frac{1}{2}\int_{t_a}^{t_b}\frac{\partial p(x,t)}{\partial x}dt\right)\mathscr{D}x(t)\\
=&\exp\left\{-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\left(\dot{x}-p(x,t)\right)^2 + \alpha\frac{\partial p(x,t)}{\partial x} \right]dt\right\}\mathscr{D}x(t)
\end{aligned},\tag{65}$$
从而
$$\begin{aligned} P [x(t)]=&\exp\left\{-\frac{1}{2\alpha}\int_{t_a}^{t_b} \left[\left(\dot{x}-p(x,t)\right)^2 + \alpha\frac{\partial p(x,t)}{\partial x} \right]dt\right\}\\
=&\exp\left\{-\frac{1}{\alpha}\int_{t_a}^{t_b}\left[\frac{1}{2}\dot{x}^2-\dot{x}p(x,t)+\frac{1}{2}p^2(x,t)+\frac{1}{2}\alpha\frac{\partial p(x,t)}{\partial x}\right]dt\right\}
\end{aligned}.\tag{66}$$
注意到
$$\begin{aligned}\frac{d}{dt}\int p(x,t)dx =& \dot{x}\frac{\partial}{\partial x}\int p(x,t)dx+\frac{\partial}{\partial t}\int p(x,t)dx\\
=&\dot{x}p(x,t)+\int \frac{\partial p(x,t)}{\partial t}dx
\end{aligned},\tag{67}$$
利用这个式子就可以上式改写为
$$P [x(t)]=\exp\left[-\frac{1}{\alpha}\int_{t_a}^{t_b}\left(\frac{1}{2}\dot{x}^2-V(x,t)\right)dt\right]\exp\left(\frac{1}{\alpha}\int_{x_a}^{x_b}p(x,t)dx\right).\tag{68}$$
其中
$$V(x,t)=-\frac{1}{2}\left(\alpha\frac{\partial p}{\partial x}+p^2\right)-\int \frac{\partial p}{\partial t}dx.\tag{69}$$
对比式$(16)\sim (19)$的结果，我们可以发现，随机微分方程$(48)$与不对称随机游走的结果是一模一样的\.

### 路径积分方法

现在我们已经证明了随机微分方程与不对称随机游走模型的等价性，并且在第一章中，我们已经分析到了不对称随机游走模型所满足的偏微分方程，那也就意味着，我们立即可以写出随机微分方程所对应的偏微分方程\. 至此，我们已经实现了三者的相互转化，转化的纽带是路径积分方法\. 这也体现了路径积分方法的威力\.
