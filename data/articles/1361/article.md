# 地球引力场的悬链线方程

> 作者：苏剑林 · 科学空间 · 2011-05-15
>
> 原文：<https://spaces.ac.cn/archives/1361>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

之前曾在[《自然极值》系列文章](<https://spaces.ac.cn/archives/1128/>)中提到过均匀重力场下的悬链线形状问题，并且在那文章中向读者提出：在一个质点（地球）引力场中的悬链线形状会是怎么样的。说实话，提出这个问题的时候，我还不懂怎么解答这个问题，不过现在会了，回头一看，已经几个月了，时间过得真快\.\.\.

与之前的思路一样，我们依旧采用的是“平衡态公理”，即总势能最小。从天体力学中我们知道，任意两个质点间的势能为$-\frac{Gm_1 m_2}{r}$。对于本题的悬链线问题，我们可以把地球放到坐标原点位置，而悬链的两个固定点分别为$(x_1,y_1)$和$(x_2,y_2)$，链的总长度为l。即
$$\int_{x_1}^{x_2} \sqrt{dx^2+dy^2}=l$$

设悬链的线密度为1（无量纲处理，长度即质量）。那么悬链上的点(x,y)到(x\+dx,y\+dy)一小段的引力势能可以记为
$$dE=-GM\frac{\sqrt{dx^2+dy^2}}{\sqrt{x^2+y^2}}$$

总势能显然是$$E=-GM\int_{x_1}^{x_2} \frac{\sqrt{dx^2+dy^2}}{\sqrt{x^2+y^2}}$$

选取适当的坐标对被积函数进行化简是很必要的，我们采取极坐标系：
$$\frac{\sqrt{dx^2+dy^2}}{\sqrt{x^2+y^2}}=\frac{\sqrt{r^2 d\theta^2+dr^2}}{r}=\frac{\sqrt{r^2 \dot{\theta}^2+1}}{r}dr$$

其中$\dot{\theta}=\frac{d\theta}{dr}$

于是$$E=-GM\int_{r_1}^{r_2} \frac{\sqrt{r^2 \dot{\theta}^2+1}}{r}dr=-GM\int_{r_1}^{r_2} F(\dot{\theta},r)dr$$

代入拉格朗日方程就得到（将r看成自变量，θ作为函数）：
$$0=\frac{\partial F }{\partial \theta}=\frac{d}{dr}(\frac{\partial F }{\partial \dot{\theta}})=\frac{d}{dr}(\frac{r\dot{\theta}}{\sqrt{r^2 \dot{\theta}^2+1}})$$

于是积分得到$\frac{r\dot{\theta}}{\sqrt{r^2 \dot{\theta}^2+1}}=Const$

并且化简得到：$$r^2 \dot{\theta}^2=C_1^2 ( C_1 > 0 )$$

要求$C_1 > 0 $只是为了描述上的方便，即$\pm \dot{\theta}=\frac{C_1}{r} \Rightarrow C_2 \pm \theta=C_1 ln r$

我们约定，在$\theta=C_1 ln r-C_2 < 0$时，上面去负号，反之取正号，则可以写成：
$$C_2 + |\theta|=C_1 ln r$$

可以进一步化简成$r=Ae^{B|\theta|}$的形式。其形状如下图：

[![新悬链线](<https://spaces.ac.cn/usr/uploads/2011/05/3865967782.png>)](<https://spaces.ac.cn/attachment/1360/>)

新悬链线
