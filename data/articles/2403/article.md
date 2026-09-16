# 平面曲线的曲率的复数表示

> 作者：苏剑林 · 科学空间 · 2014-03-04
>
> 原文：<https://spaces.ac.cn/archives/2403>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

开学已经是第二周了，我的《微分几何》也上课两周了，进度比较慢，现在才讲到平面曲线的曲率。在平面曲线$\boldsymbol{t}(t)=(x(t),y(t))$某点上可以找出单位切向量。
$$\boldsymbol{t}=\left(\frac{dx}{ds},\frac{dy}{ds}\right)$$
其中$ds^2 =dx^2+dy^2$，将这个向量逆时针旋转90度之后，就可以定义相应的单位法向量$\boldsymbol{n}$，即$\boldsymbol{t}\cdot\boldsymbol{n}=0$。

<strong>常规写法</strong>

让我们用弧长$s$作为参数来描述曲线方程，$\boldsymbol{t}(s)=(x(s),y(s))$，函数上的一点表示对$s$求导。那么我们来考虑$\dot{\boldsymbol{t}}$，由于$\boldsymbol{t}^2=1$，对s求导得到
$$\boldsymbol{t}\cdot\dot{\boldsymbol{t}}=0$$
也就是说$\dot{\boldsymbol{t}}$与$\boldsymbol{t}$垂直，由于只是在平面上，所以$\dot{\boldsymbol{t}}$与$\boldsymbol{n}$平行。即
$$\dot{\boldsymbol{t}}=\kappa \boldsymbol{n}$$

类似地，有$\dot{\boldsymbol{n}}$与$\boldsymbol{t}$平行。并且对$\boldsymbol{t}\cdot\boldsymbol{n}=0$求导得到
$$\dot{\boldsymbol{t}}\cdot\boldsymbol{n}+\boldsymbol{t}\cdot\dot{\boldsymbol{n}}=0$$
将$\dot{\boldsymbol{t}}=\kappa \boldsymbol{n}$代入上式得到
$$\dot{\boldsymbol{n}}=-\kappa \boldsymbol{t}$$
$\kappa$被称为曲线在该点的曲率。

<strong>复数表示</strong>

以上是教科书的标准写法，但事实上，研究平面曲线的最方便的工具还是复数。将$\boldsymbol{r}(s)$用一个带参数的复数表示$z(s)$，那么上面的两式可以写成更简洁的一个式子
$$\ddot{z}(s)=i\kappa (s) \dot{z}(s) $$

这样写的好处还在于，任意给出曲率函数$\kappa (s) $，我们就可以求出对应的曲线
$$z(s)=\int e^{i\int \kappa (s)ds}ds $$
这是简洁而有效的。

另外，不妨设$dz=ds e^{i\phi}$，那么
$$\dot{z}=e^{i\phi}$$
自然地
$$\ddot{z}=e^{i\phi}\left(i\dot{\phi}\right)$$
所以曲率可以表示为
$$\kappa=\dot{\phi}$$

<strong>各种坐标</strong>

利用它可以很方便地推导出各种坐标系下的曲率表达式，如曲线为一般的参数方程$(x(t),y(t))$时，用函数加一撇表示对t求导，有$ds=\sqrt{x'(t）^2+y'(t)^2}dt,\phi=\arctan\left(\frac{y'(t)}{x'(t)}\right)$，那么
$$\frac{d\phi}{ds}=\frac{\frac{y''(t)}{x'(t)}-\frac{y'(t)x''(t)}{[x'(t)]^2}}{1+\left(\frac{y'(t)}{x'(t)}\right)^2}\div \left(\frac{ds}{dt}\right)$$
代入整理易得
$$\kappa=\frac{y''(t) x'(t)-x''(t) y'(t)}{[x'(t)^2+y'(t)^2]^{3/2}}$$

在极坐标下，设$r=f(\theta)$，则$z=f(\theta)e^{i\theta}$，那么
$$dz=\left(\frac{d f}{d \theta}+i f\right)e^{i\theta}d\theta$$
所以
$$ds=\sqrt{f^2+\left(\frac{d f}{d \theta}\right)^2}d\theta$$
而$\phi=\arctan\frac{f}{\left(\frac{d f}{d \theta}\right)}+\theta$，那么
$$\frac{d\phi}{ds}=\left[\frac{1-\left(\frac{d^2 f}{d \theta^2}\right) f/\left(\frac{d f}{d \theta}\right)^2}{1+f^2/\left(\frac{d f}{d \theta}\right)^2}+1\right]\div \left(\frac{d s}{d \theta}\right)$$
代入整理得
$$\kappa=\frac{2\left(\frac{d f}{d \theta}\right)^2+f^2-\left(\frac{d^2 f}{d \theta^2}\right)f}{\left[\left(\frac{d f}{d \theta}\right)^2+f^2\right]^{3/2}}$$
三维空间有没有类似方便的东西呢？我也正在思考^\_^
