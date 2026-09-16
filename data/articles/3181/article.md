# 借助变分法变换坐标

> 作者：苏剑林 · 科学空间 · 2015-01-06
>
> 原文：<https://spaces.ac.cn/archives/3181>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

### ODE的坐标变换

熟悉理论力学的读者应该能够领略到变分法在变换坐标系中的作用。比如，如果要将下面的平面二体问题方程
$$\left\{\begin{aligned}\frac{d^2 x}{dt^t}=\frac{-\mu x}{(x^2+y^2)^{3/2}}\\
\frac{d^2 y}{dt^t}=\frac{-\mu y}{(x^2+y^2)^{3/2}}\end{aligned}\right.\tag{1}$$
变换到极坐标系下，如果直接代入计算，将会是一道十分繁琐的计算题。但是，我们知道，上述方程只不过是作用量
$$S=\int \left[\frac{1}{2}\left(\dot{x}^2+\dot{y}^2\right)+\frac{\mu}{\sqrt{x^2+y^2}}\right]dt\tag{2}$$
变分之后的拉格朗日方程，那么我们就可以直接对作用量进行坐标变换。而由于作用量一般只涉及到了一阶导数，因此作用量的变换一般来说比较简单。比如，很容易写出，$(2)$在极坐标下的形式为
$$S=\int \left[\frac{1}{2}\left(\dot{r}^2+r^2\dot{\theta}^2\right)+\frac{\mu}{r}\right]dt\tag{3}$$
对$(3)$进行变分，得到的拉格朗日方程为
$$\left\{\begin{aligned}&\ddot{r}=r\dot{\theta}^2-\frac{\mu}{r^2}\\
&\frac{d}{dt}\left(r^2\dot{\theta}\right)=0\end{aligned}\right.\tag{4}$$
就这样完成了坐标系的变换。如果想直接代入$(1)$暴力计算，那么请参考[《方程与宇宙》:二体问题的来来去去(一)](<https://spaces.ac.cn/archives/549/>)

### PDE的坐标变换

上面说了ODE中的坐标变换，现在来说PDE的。偏微分方程以及向量分析课程中，最常见的练习题就是把拉普拉斯算子变换到极坐标、球坐标或者其他正交坐标系之下，以便适应不同的边界条件。在这时候，变分也能起到一定的作用。以二维拉普拉斯算子为例，我们考虑方程
$$\Delta^2 u=\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}=F\tag{5}$$
其中$F$是常数（引入$F$的作用是什么？请读者思考）。为了考虑$(5)$在极坐标下的形式，我们只需要注意到它是作用量
$$S=\int \left[\frac{1}{2}\left(\left(\frac{\partial u}{\partial x}\right)^2+\left(\frac{\partial u}{\partial y}\right)^2\right)+F u\right] dxdy\tag{6}$$
变分之下的结果。而我们可以直接对$(6)$进行坐标变换，然后再变分，可以很快地写出
$$S=\int \left[\frac{1}{2}\left(\left(\frac{\partial u}{\partial r}\right)^2+\left(\frac{1}{r}\frac{\partial u}{\partial \theta}\right)^2\right)+F u\right] rd\theta dr\tag{7}$$
变分的结果给出（别漏了$d\theta dr $前面的$r$）
$$\frac{\partial}{\partial r}\left(r\frac{\partial u}{\partial r}\right)+\frac{1}{r}\frac{\partial^2 u}{\partial \theta^2}=F r\tag{8}$$
也就是
$$\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial u}{\partial r}\right)+\frac{1}{r^2}\frac{\partial^2 u}{\partial \theta^2}=F$$
对比$(5)$式，就得到
$$\frac{\partial^2 }{\partial x^2}+\frac{\partial^2 }{\partial y^2}=\frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial }{\partial r}\right)+\frac{1}{r^2}\frac{\partial^2 }{\partial \theta^2}$$
计算量比直接代入坐标系变换式进行硬算应该稍低。

<strong>读者可能会指出：你作了弊！你算得快，是因为你提前算好了极坐标的雅可比行列式（重积分变换），而且动能项的变换也没那么显然！如果把这些计算都算进去，计算时间可能会远超出直接代入计算。</strong>

这样考虑的读者的确是正确的，如果先去算个雅可比行列式，然后再化简、变分，计算量也是颇大的。但是笔者在这里强调的是：简单的答案或许会有简单的过程（如果答案本身很复杂，那么不管什么技巧都不会简单），以及不熟悉的问题或许可以转换到熟悉的问题中去。在拉普拉斯算子的变换中，至少对于笔者来说，这个变换过程是陌生的，而相反，极坐标（包括球坐标）的雅可比行列式是熟悉的，变分法也是比较熟悉的。因此，我会宁愿用这种变分的技巧——看起来迂回——来推导拉普拉斯算子的变换。

其次，极坐标和球坐标都是正交坐标系，对于正交坐标系来说，写出它的雅可比行列式并不是特别困难的事。比如极坐标，只需要注意到$r d\theta$和$dr$是极坐标系中的正交微元，就可以立马写出$dxdy=r d\theta dr$以及$\left(\frac{\partial u}{\partial x}\right)^2+\left(\frac{\partial u}{\partial y}\right)^2=\left(\frac{\partial u}{\partial r}\right)^2+\left(\frac{1}{r}\frac{\partial u}{\partial \theta}\right)^2$。

### 又一个例子

再举一个例子，比如说正交坐标系$(\xi,\eta)$
$$\left\{\begin{aligned}&x=\xi^2-\eta^2\\
&y=2\xi\eta\end{aligned}\right.$$
（为什么$(\xi,\eta)$是正交的？这里笔者确实作了弊，这是从$x+yi=(u+vi)^2$展开而来，解析函数的实部和虚部必然正交。）
可以算得
$$ds^2=dx^2+dy^2=\left(\xi^2+\eta^2\right)\left(d\xi^2+d\eta^2\right)$$
从而两个正交微元为$\sqrt{\xi^2+\eta^2}d\xi,\sqrt{\xi^2+\eta^2}d\eta$。因此，很快写出
$$\begin{aligned}&dxdy=\sqrt{\xi^2+\eta^2}d\xi\cdot\sqrt{\xi^2+\eta^2}d\eta=\left(\xi^2+\eta^2\right)d\xi d\eta\\
&\left(\frac{\partial u}{\partial x}\right)^2+\left(\frac{\partial u}{\partial y}\right)^2=\left(\frac{1}{\sqrt{\xi^2+\eta^2}}\frac{\partial u}{\partial \xi}\right)^2+\left(\frac{1}{\sqrt{\xi^2+\eta^2}}\frac{\partial u}{\partial \eta}\right)^2\end{aligned}$$
从而作用量变换成
$$S=\int \left[\frac{1}{2}\left(\left(\frac{\partial u}{\partial \xi}\right)^2+\left(\frac{\partial u}{\partial \eta}\right)^2\right)+F u \left(\xi^2+\eta^2\right) \right] d\xi d\eta$$
变分的结果是
$$\frac{\partial^2 u}{\partial \xi^2}+\frac{\partial^2 u}{\partial \eta^2}=F\cdot\left(\xi^2+\eta^2\right) $$
从而
$$\frac{\partial^2 }{\partial x^2}+\frac{\partial^2 }{\partial y^2}=\frac{1}{\xi^2+\eta^2}\left(\frac{\partial^2 u}{\partial \xi^2}+\frac{\partial^2 u}{\partial \eta^2}\right)$$

### 球坐标的例子

现在来到三维的情况了。这里仅仅说球坐标
$$\left\{\begin{aligned}&x=r\sin\theta\cos\phi\\
&y=r\sin\theta\sin\phi\\
&z=r\cos\theta\end{aligned}\right.$$
可以算出
$$\begin{aligned}ds^2&=\left[d\left(r\sin\theta\right)\right]^2+\left(r\sin\theta\right)^2 d\phi^2+\left[d\left(r\cos\theta\right)\right]^2\\
&=dr^2+r^2 d\theta^2 +\left(r\sin\theta\right)^2 d\phi^2\end{aligned}$$
其中第一步是直接对前两个坐标应用一次极坐标的变换公式（把半径看成$\left(r\sin\theta\right)$），总的来说就是用了两次极坐标的变换公式而已。从而我们得到球面的正交微元$dr,r d\theta,r\sin\theta d\phi$，从而
$$\begin{aligned}&dxdydz=r^2\sin\theta dr d\theta d\phi\\
&\left(\frac{\partial u}{\partial x}\right)^2+\left(\frac{\partial u}{\partial y}\right)^2+\left(\frac{\partial u}{\partial z}\right)^2=\left(\frac{\partial u}{\partial r}\right)^2+\left(\frac{1}{r}\frac{\partial u}{\partial \theta}\right)^2+\left(\frac{1}{r\sin\theta}\frac{\partial u}{\partial \phi}\right)^2\end{aligned}$$
从而作用量为
$$S=\int {\left[\frac{1}{2}\left(\left(\frac{\partial u}{\partial r}\right)^2+\left(\frac{1}{r}\frac{\partial u}{\partial \theta}\right)^2+\left(\frac{1}{r\sin\theta}\frac{\partial u}{\partial \phi}\right)^2\right)+F u\right] r^2\sin\theta dr d\theta d\phi}$$
变分的结果是
$$\frac{\partial}{\partial r}\left(r^2\sin\theta\frac{\partial u}{\partial r}\right)+\frac{\partial}{\partial \theta}\left(\sin\theta\frac{\partial u}{\partial \theta}\right)^2+\frac{1}{\sin\theta}\frac{\partial^2 u}{\partial \phi^2}=F r^2\sin\theta$$
最后整理得
$$\Delta u=\frac{1}{r^2}\frac{\partial}{\partial r}\left(r^2\frac{\partial u}{\partial r}\right)+\frac{1}{r^2\sin\theta}\frac{\partial}{\partial \theta}\left(\sin\theta\frac{\partial u}{\partial \theta}\right)+\frac{1}{r^2\sin^2\theta}\frac{\partial^2 u}{\partial \phi^2}$$

究竟怎么算简单一些？读者自己把握吧。^\_^
