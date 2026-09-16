# 带点电荷的均匀杆

> 作者：苏剑林 · 科学空间 · 2014-02-16
>
> 原文：<https://spaces.ac.cn/archives/2362>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在讨论了倒立单摆的相关分析之后，胡雄大哥（笔者的一位好友）提出了一个问题：一根均匀杆，当然质量不可忽略，只有一个力（简单起见，可以先假设为恒力）作用在其中一个点上（简单起见，可以假设为端点），那么杆是怎么运动的？

其实笔者学了不少的经典力学，也分析了不少问题，但就是对于力矩、角动量等还是模模糊糊的，对于我来说，大多数经典力学问题就是“作用量\+变分”，本题也不例外。为了让题目的实验意义更加明确，不妨将题目改成：

> 一根中性的均匀杆，它的一个端点带有一个点电荷，那么它（仅仅）在一个均匀电场中的运动是怎样的？

在这里，我们进一步简化，只考虑平面问题。杆属于刚体，为了描述杆的运动，我们需要描述杆上一点的运动，以及杆绕这一点的转动，也就是说，即使只考虑平面的情况，该系统也是有三个自由度的。设杆的带电荷那一端点的坐标为$(x,y)$，为了描述杆的转动，以这一端点为中心建立极坐标系，设杆的极角为$\theta$。设电势的函数为$U(x,y)$，因为只有一点带电（受力），因此势能是简单的。

<strong>运动方程</strong>

但是同时存在转动和平动，动能稍微复杂一点。设杆的线密度是$\rho$，长度是$R$，质量为$m=\rho R$，那么杆上到端点的距离为$r$那一点的坐标（相对于一个惯性系）为
$$(X,Y)=(x,y)+(r\cos\theta,r\sin\theta)$$
其速度就是
$$(\dot{X},\dot{Y})=(\dot{x}-r\sin\theta \dot{\theta},\dot{y}+r\cos\theta\dot{\theta})$$
动能就是积分
$$\begin{aligned}
E_k&=\int_0^R \frac{1}{2}\left(\dot{X}^2+\dot{Y}^2\right) \rho dr \\
&=\int_0^R \frac{1}{2}\left(\dot{x}^2+\dot{y}^2+r^2\dot{\theta}^2+2r\dot{\theta}\dot{y}\cos\theta-2r\dot{\theta}\dot{x}\sin\theta\right)\rho dr\\
&=\frac{1}{2}\rho R\left(\dot{x}^2+\dot{y}^2+\frac{1}{3}R^2\dot{\theta}^2+R\dot{\theta}\dot{y}\cos\theta-R\dot{\theta}\dot{x}\sin\theta\right)\\
&=\frac{1}{2}m\left(\dot{x}^2+\dot{y}^2+\frac{1}{3}R^2\dot{\theta}^2+R\dot{\theta}\dot{y}\cos\theta-R\dot{\theta}\dot{x}\sin\theta\right)
\end{aligned}$$
虽然有点复杂，但是现在可以写出作用量了：
$$S=\int \left[\frac{1}{2}m\left(\dot{x}^2+\dot{y}^2+\frac{1}{3}R^2\dot{\theta}^2+R\dot{\theta}\dot{y}\cos\theta-R\dot{\theta}\dot{x}\sin\theta\right)-U(x,y)\right]dt$$
变分它，也就是代入欧拉\-拉格朗日方程，分别得到
$$\begin{aligned}
m\frac{d}{dt}\left(\dot{x}-\frac{1}{2}R\dot{\theta}\sin\theta\right)=-\frac{\partial U}{\partial x}\\
m\frac{d}{dt}\left(\dot{y}+\frac{1}{2}R\dot{\theta}\cos\theta\right)=-\frac{\partial U}{\partial y}\\
m\frac{d}{dt}\left(\frac{1}{3}R^2\dot{\theta}+\frac{1}{2}R\dot{y}\cos\theta -\frac{1}{2}R\dot{x}\sin\theta\right)=\\
-\frac{1}{2}mR\dot{\theta}\dot{y}\sin\theta -\frac{1}{2}mR\dot{\theta}\dot{x}\cos\theta
\end{aligned}$$
即使没有最小作用量原理，我们也很容易列出前两道方程，它只不过是说：作用在系统上的合外力等于系统的总质量乘以系统质心的加速度。说白了，就是如果讨论质心的运动时，系统就相当于一个质点而已。因此，考虑力有可能不是保守力，因此最一般的方程应该写成
$$\begin{aligned}
&m\frac{d}{dt}\left(\dot{x}-\frac{1}{2}R\dot{\theta}\sin\theta\right)=F_x\\
&m\frac{d}{dt}\left(\dot{y}+\frac{1}{2}R\dot{\theta}\cos\theta\right)=F_y
\end{aligned}$$

比较难列出的是关于$\dot{\theta}$的方程，请可以给出力学分析解释的朋友不吝告知，十分感谢。如果把$\ddot{x}$和$\ddot{y}$的表达式代入到关于$\dot{\theta}$的表达式之中，会得到物理意义相对明显一些的结果
$$\begin{aligned}\frac{1}{3}mR^2\ddot{\theta}+\frac{1}{2}mR\left(-\frac{1}{2}R\ddot{\theta}\cos\theta+\frac{1}{2}R\dot{\theta}^2\sin\theta +\frac{F_y}{m}\right)\cos\theta\\
-\frac{1}{2}mR\left(\frac{1}{2}R\ddot{\theta}\sin\theta+\frac{1}{2}R\dot{\theta}^2\cos\theta +\frac{F_x}{m}\right)\sin\theta=0
\end{aligned}$$
即
$$\frac{1}{6}mR\ddot{\theta}+F_y \cos\theta-F_x\sin\theta=0$$
看上去物理意义很明显，但我还是找不到比较准确的物理解释，望各位指教。
至此，我们完成了第一步工作，即列出运动方程。

<strong>恒力</strong>
如果力$F=(F_x,F_y)$是恒力，那么总可以选择适当的坐标系，使得其中一个分量为0，不妨假设$F_y=0$，那么
$$\begin{aligned}
m\frac{d}{dt}\left(\dot{x}-\frac{1}{2}R\dot{\theta}\sin\theta\right)&=F_x\\
m\frac{d}{dt}\left(\dot{y}+\frac{1}{2}R\dot{\theta}\cos\theta\right)&=0\\
\frac{1}{6}mR\ddot{\theta}-F_x\sin\theta &=0
\end{aligned}$$
其中前两道方程是容易积分的
$$\begin{aligned}
x+\frac{1}{2}R\cos\theta &=\frac{F_x}{2m}t^2+C_1 t+C_2\\
y+\frac{1}{2}R\sin\theta &=C_3 t+C_4
\end{aligned}$$
这说明质心的运动类似于平抛。

第三道$\frac{1}{6}mR\ddot{\theta}-F_x\sin\theta=0$类似于单摆的方程，当$F_x$为正时，相当于倒立的单摆；$F_x$为负时，就相当于一般的单摆。为作简单演示，不妨假设初始的$\theta$很小，且$F_x < 0$，这样近似解得
$$\theta=C_5 \cos\omega t+C_6\sin\omega t,\quad \omega=\sqrt{\frac{-6F_x}{mR}}$$

取其中一个特例，绘制其运动动画近似如下：

[![带点电荷的均匀杆](<https://spaces.ac.cn/usr/uploads/2014/02/4133040644.gif>)](<https://spaces.ac.cn/usr/uploads/2014/02/4133040644.gif>)

带点电荷的均匀杆

（受力点为左边端点，力为恒力，方向水平向左，初速度为0。）

[![带点电荷的均匀杆2](<https://spaces.ac.cn/usr/uploads/2014/02/2789803481.gif>)](<https://spaces.ac.cn/usr/uploads/2014/02/2789803481.gif>)

带点电荷的均匀杆2

（受力点为左边端点，力为恒力，方向水平向左，带有一个竖直向下的初速度。）
