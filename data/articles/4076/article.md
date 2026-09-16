# 【外微分浅谈】7\. 有力的计算

> 作者：苏剑林 · 科学空间 · 2016-11-11
>
> 原文：<https://spaces.ac.cn/archives/4076>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这里我们将展示上面一节的方法对于计算黎曼曲率张量的计算是多少的有力！我们再次列出我们得到的所有公式。首先是概念式的
$$\begin{aligned}&\omega^{\mu}=h_{\alpha}^{\mu}dx^{\alpha}\\
&d\boldsymbol{r}=\hat{\boldsymbol{e}}_{\mu} \omega^{\mu}\\
&ds^2 = \eta_{\mu\nu} \omega^{\mu}\omega^{\nu}\\
&\langle \hat{\boldsymbol{e}}_{\mu}, \hat{\boldsymbol{e}}_{\nu}\rangle = \eta_{\mu\nu}\end{aligned} \tag{65} $$
然后是
$$\begin{aligned}&d\eta_{\mu\nu}=\omega_{\nu\mu}+\omega_{\mu\nu}=\eta_{\nu\alpha}\omega_{\mu}^{\alpha}+\eta_{\mu \alpha}\omega_{\nu}^{\alpha}\\
&d\omega^{\mu}+\omega_{\nu}^{\mu}\land \omega^{\nu}=0\end{aligned} \tag{66} $$
这两个可以帮助我们确定$\omega_{\nu}^{\mu}$；接着就是
$$\mathscr{R}_{\nu}^{\mu} = d\omega_{\nu}^{\mu}+\omega_{\alpha}^{\mu} \land \omega_{\nu}^{\alpha} \tag{67} $$
最后你要正交标架下的$\hat{R}^{\mu}_{\nu\beta\gamma}$，就要写出：
$$\mathscr{R}_{\nu}^{\mu}=\sum_{\beta < \gamma} \hat{R}^{\mu}_{\nu\beta\gamma}\omega^{\beta}\land \omega^{\gamma} \tag{68} $$
如果你要原始标架下的$R^{\mu}_{\nu\beta\gamma}$，就要写出
$$(h^{-1})_{\mu'}^{\mu}\mathscr{R}^{\mu'}_{\nu'}h_{\nu}^{\nu'} = \sum_{\beta < \gamma} R^{\mu}_{\nu\beta\gamma}dx^{\beta}\land dx^{\gamma} \tag{69} $$
然后依次读出$R^{\mu}_{\nu\beta\gamma}$，就像制表一样。

### 二维的例子：球面

先用一个二维的例子来热身，我们来计算球面$ds^2 = d\theta^2 + \sin^2 \theta d\phi^2$的黎曼曲率张量。

我们取
$$\omega^1 = d\theta, \quad \omega^2 = \sin\theta d\phi \tag{70} $$
也就是
$$\boldsymbol{h}=\begin{pmatrix}1&0\\0&\sin\theta\end{pmatrix},\quad \boldsymbol{\eta}=\begin{pmatrix}1&0\\0&1\end{pmatrix} \tag{71} $$
由于$\boldsymbol{\eta}$是单位阵，因此$d\eta_{\mu\nu}=\eta_{\alpha \nu}\omega_{\mu}^{\alpha}+\eta_{\mu \alpha}\omega_{\nu}^{\alpha}$告诉我们，$\omega_{\nu}^{\mu}$是反对称矩阵。我们用矩阵的形式写出$d\omega^{\mu}+\omega_{\nu}^{\mu}\land \omega^{\nu}=0$
$$\begin{pmatrix} 0 & \omega_2^1 \\ -\omega_2^1 & 0 \end{pmatrix}\land \begin{pmatrix} d\theta \\ \sin\theta d\phi \end{pmatrix}=-d\begin{pmatrix} d\theta \\ \sin\theta d\phi \end{pmatrix}=-\begin{pmatrix} 0 \\ \cos\theta d\theta\land d\phi \end{pmatrix} \tag{72} $$
由于反对称性，$\omega_{\nu}^{\mu}$只有一个独立分量，不难发现$\omega_2^1=-\cos\theta d\phi$。这个求解过程，用猜测、试探的方法就行了。接着，求$\mathscr{R}_{\nu}^{\mu} = d\omega_{\nu}^{\mu}+\omega_{\alpha}^{\mu} \land \omega_{\nu}^{\alpha}$，即
$$\begin{aligned}\mathscr{R}_{\nu}^{\mu} = &d\begin{pmatrix} 0 & -\cos\theta d\phi \\ \cos\theta d\phi & 0 \end{pmatrix}\\
&+\begin{pmatrix} 0 & -\cos\theta d\phi \\ \cos\theta d\phi & 0 \end{pmatrix}\land \begin{pmatrix} 0 & -\cos\theta d\phi \\ \cos\theta d\phi & 0 \end{pmatrix} \end{aligned} \tag{73} $$
最后矩阵相乘一项显然为0，事实上也可以证明只要在2维空间中就有$\omega_{\alpha}^{\mu} \land \omega_{\nu}^{\alpha}$恒等于0，因此
$$\mathscr{R}_{\nu}^{\mu} = d\begin{pmatrix} 0 & -\cos\theta d\phi \\ \cos\theta d\phi & 0 \end{pmatrix}=\begin{pmatrix} 0 & \sin\theta d\theta\land d\phi \\ -\sin\theta d\theta\land d\phi & 0 \end{pmatrix} \tag{74} $$
又因为
$$\mathscr{R}_{\nu}^{\mu}=\sum_{\beta < \gamma} \hat{R}^{\mu}_{\nu\beta\gamma}\omega^{\beta}\land \omega^{\gamma}=\hat{R}^{\mu}_{\nu 1 2 }\sin\theta d\theta \land d\phi \tag{75} $$
对比即知
$$\hat{R}^{\mu}_{\nu 1 2 } = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix} \tag{76} $$
也就是说在正交标架下有$\hat{R}^{1}_{2 1 2 } = 1, \hat{R}^{2}_{1 1 2 } = -1$。然后根据$(h^{-1})_{\mu'}^{\mu}\mathscr{R}^{\mu'}_{\nu'}h_{\nu}^{\nu'} = \sum_{\beta < \gamma} R^{\mu}_{\nu\beta\gamma}dx^{\beta}\land dx^{\gamma}$计算$R^{\mu}_{\nu\beta\gamma}$，我们有
$$\begin{aligned}&\begin{pmatrix}1&0\\0&\sin\theta\end{pmatrix}^{-1}\begin{pmatrix} 0 & \sin\theta d\theta\land d\phi \\ -\sin\theta d\theta\land d\phi & 0 \end{pmatrix}\begin{pmatrix}1&0\\0&\sin\theta\end{pmatrix}\\
=&R^{\mu}_{\nu 12}d\theta\land d\phi\end{aligned} \tag{77} $$
也就是
$$R^{\mu}_{\nu 1 2 } = \begin{pmatrix} 0 & \sin^2\theta \\ -1 & 0 \end{pmatrix} \tag{78} $$
也就是说有$R^{1}_{2 1 2 } = \sin\theta, R^{2}_{1 1 2 } = -1$。所有过程只不过是涉及到矩阵的乘法，这是我们所熟悉的，比繁琐的多重指标求和不知道省了多少心。对比$\hat{R}^{\mu}_{\nu\beta\gamma}$和$R^{\mu}_{\nu\beta\gamma}$的形式，也可以发现确实正交标架确实起到了简化的作用。

### 四维的例子：史瓦西度规

爱因斯坦场方程的第一个精确解是史瓦西度规，它是通过求解如下形式的度规得到的
$$ds^2= -e^{2\Phi}dt^2 + e^{2\Lambda} dr^2 + r^2 d\theta^2 + r^2 \sin^2\theta d\phi^2 \tag{79} $$
它的出发点是考虑一个各项同性的度规，因此$\Phi, \Lambda$假定为仅仅是$r$的函数。我们来算这种情形下的黎曼曲率张量（难度中上）。

很自然地，我们取
$$\omega^1 = e^{\Phi}dt, \quad \omega^2 = e^{\Lambda }dr, \quad \omega^3 = rd\theta, \quad \omega^4 = r\sin\theta d\phi \tag{80} $$
这时
$$\boldsymbol{h}=\begin{pmatrix}e^{\Phi}&0&0&0\\0&e^{\Lambda }&0&0\\0&0&r&0\\0&0&0&r\sin\theta\end{pmatrix},\quad \boldsymbol{\eta}=\begin{pmatrix}-1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&1\end{pmatrix} \tag{81} $$
由$d\eta_{\mu\nu}=\eta_{\alpha \nu}\omega_{\mu}^{\alpha}+\eta_{\mu \alpha}\omega_{\nu}^{\alpha}$可以知道$\omega_{\mu\nu}=\omega_{\mu}^{\alpha}\eta_{\alpha \nu}$是反对称的，并且结合$\boldsymbol{\eta}$的形式来稍加分析，就可以得出，$\omega_{\nu}^{\mu}$具有以下形式
$$\begin{pmatrix}0&\omega_2^1&\omega_3^1&\omega_4^1\\ \omega_2^1&0&\omega_3^2&\omega_4^2\\ \omega_3^1&-\omega_3^2&0&\omega_4^3\\ \omega_4^1&-\omega_4^2&-\omega_4^3&0\end{pmatrix} \tag{82} $$
它的特点是，作为分块矩阵
$$\left(\begin{array}{c:ccc}0&\omega_2^1&\omega_3^1&\omega_4^1\\ \hdashline \omega_2^1&0&\omega_3^2&\omega_4^2\\ \omega_3^1&-\omega_3^2&0&\omega_4^3\\ \omega_4^1&-\omega_4^2&-\omega_4^3&0\end{array}\right) =\left(\begin{array}{c:c}E & F\\ \hdashline G&H\end{array}\right) \tag{83} $$
来看，它是对称的，但对角线矩阵$E,H$都是反对称的。至于怎么分块，取决于我们将$\boldsymbol{\eta}$分块为$\left(\begin{array}{c:c}-I & 0\\ \hdashline 0&I\end{array}\right)$的方式，其中$I$代表单位阵。

有了$\omega_{\nu}^{\mu}$的具体形式后，我们就可以写出$d\omega^{\mu}+\omega_{\nu}^{\mu}\land \omega^{\nu}=0$：
$$\begin{aligned}&\begin{pmatrix}0&\omega_2^1&\omega_3^1&\omega_4^1\\ \omega_2^1&0&\omega_3^2&\omega_4^2\\ \omega_3^1&-\omega_3^2&0&\omega_4^3\\ \omega_4^1&-\omega_4^2&-\omega_4^3&0\end{pmatrix}\land \begin{pmatrix}e^{\Phi}dt\\ e^{\Lambda }dr\\rd\theta\\r\sin\theta d\phi\end{pmatrix}\\
=&-d\begin{pmatrix}e^{\Phi}dt\\ e^{\Lambda }dr\\rd\theta\\r\sin\theta d\phi\end{pmatrix}=-\begin{pmatrix}e^{\Phi} \dot{\Phi} dr\land dt\\ 0 \\dr\land d\theta\\ \sin\theta dr\land d\phi+ r\cos\theta d\theta\land d\phi \end{pmatrix}\end{aligned} \tag{84} $$
这里的$\dot{}$指的是对$r$求导。充分头脑风暴，就可以快速将答案定位。比如看着第2行恒等于0，那么就可以断定$\omega_2^1,\omega_3^2,\omega_4^2$分别只与$dt,d\theta,d\phi$有关，结合第1行，就可以确定$\omega_2^1 = e^{\Phi-\Lambda}\dot{\Phi} dt$，并且得出$\omega_3^1,\omega_4^1$分别只与$d\theta,d\phi$有关。然后结合第3行，就可以确定$\omega_3^1=0$，还有$\omega_3^2=-e^{-\Lambda}d\theta$，并且得出$\omega_4^3$只与$d\phi$有关，最后看着第4行，很快就确定$\omega_4^1=0, \omega_4^2 = -\sin\theta e^{-\Lambda} d\phi, \omega_4^3 = -\cos\theta d\phi$，最后得到
$$\omega_{\nu}^{\mu}=\begin{pmatrix}0& e^{\Phi-\Lambda}\dot{\Phi} dt & 0 & 0 \\ e^{\Phi-\Lambda}\dot{\Phi} dt &0&-e^{-\Lambda}d\theta&-\sin\theta e^{-\Lambda} d\phi\\ 0 &e^{-\Lambda}d\theta&0&-\cos\theta d\phi\\ 0&\sin\theta e^{-\Lambda} d\phi&\cos\theta d\phi&0\end{pmatrix} \tag{85} $$
现在就可以计算$\mathscr{R}_{\nu}^{\mu} = d\omega_{\nu}^{\mu}+\omega_{\alpha}^{\mu} \land \omega_{\nu}^{\alpha}$了，即
$$\begin{aligned}\mathscr{R}_{\nu}^{\mu} =&d\begin{pmatrix}0& e^{\Phi-\Lambda}\dot{\Phi} dt & 0 & 0 \\ e^{\Phi-\Lambda}\dot{\Phi} dt &0&-e^{-\Lambda}d\theta&-\sin\theta e^{-\Lambda} d\phi\\ 0 &e^{-\Lambda}d\theta&0&-\cos\theta d\phi\\ 0&\sin\theta e^{-\Lambda} d\phi&\cos\theta d\phi&0\end{pmatrix}\\
&+\begin{pmatrix}0& e^{\Phi-\Lambda}\dot{\Phi} dt & 0 & 0 \\ e^{\Phi-\Lambda}\dot{\Phi} dt &0&-e^{-\Lambda}d\theta&-\sin\theta e^{-\Lambda} d\phi\\ 0 &e^{-\Lambda}d\theta&0&-\cos\theta d\phi\\ 0&\sin\theta e^{-\Lambda} d\phi&\cos\theta d\phi&0\end{pmatrix}\\
&\land \begin{pmatrix}0& e^{\Phi-\Lambda}\dot{\Phi} dt & 0 & 0 \\ e^{\Phi-\Lambda}\dot{\Phi} dt &0&-e^{-\Lambda}d\theta&-\sin\theta e^{-\Lambda} d\phi\\ 0 &e^{-\Lambda}d\theta&0&-\cos\theta d\phi\\ 0&\sin\theta e^{-\Lambda} d\phi&\cos\theta d\phi&0\end{pmatrix}
\end{aligned} \tag{86} $$
算得
$$\begin{aligned}&\mathscr{R}_1^1=\mathscr{R}_2^2=\mathscr{R}_3^3=\mathscr{R}_4^4=0\\
&\mathscr{R}^1_2=\mathscr{R}^2_1=-e^{\Phi-\Lambda}(\ddot{\Phi}+\dot{\Phi}^2-\dot{\Phi}\dot{\Lambda}) dt\land dr\\
&\mathscr{R}^1_3=\mathscr{R}^3_1=-e^{\Phi-2\Lambda} \dot{\Phi} dt\land d\theta\\
&\mathscr{R}^1_4=\mathscr{R}^4_1=-e^{\Phi-2\Lambda} \dot{\Phi} \sin\theta dt\land d\phi\\
&\mathscr{R}^2_3=-\mathscr{R}^3_2=e^{-\Lambda}\dot{\Lambda} dr\land d\theta\\
&\mathscr{R}^2_4=-\mathscr{R}^4_2=e^{-\Lambda}\dot{\Lambda}\sin\theta dr\land d\phi\\
&\mathscr{R}^3_4=-\mathscr{R}^4_3=(1-e^{2\Lambda})\sin\theta d\theta\land d\phi\end{aligned} \tag{87} $$
这就可以依次读出$\hat{R}^{\mu}_{\nu\beta\gamma}$了，比如
$$\hat{R}^{1}_{212}=-e^{-2\Lambda}(\ddot{\Phi}+\dot{\Phi}^2-\dot{\Phi}\dot{\Lambda}), \quad\hat{R}^{1}_{312}=-\frac{1}{r}e^{-2\Lambda}\dot{\Phi} \tag{88} $$
等等，如果你愿意，还可以继续求$R^{\mu}_{\nu\beta\gamma}$，因为$\boldsymbol{h}$是一个对角阵，因此并不会增加多少工程。

读者要是亲自去运算，可能仍然会抱怨要花很多时间，也许会觉得并没有什么化简作用。但是，按照上面的步骤一步步来，哪怕是人工，也是可以操作的。这至少是一种人工可能算得出来的方法，上面的过程就是笔者手算出来的，并没有用到mathematica之类的软件。相信没有人从原始表达式$R^{\mu}_{\nu\beta\gamma}=\frac{\partial \Gamma^{\mu}_{\nu\gamma}}{\partial x^{\beta}}-\frac{\partial \Gamma^{\mu}_{\nu\beta}}{\partial x^{\gamma}}+\Gamma^{\mu}_{\alpha\beta}\Gamma^{\alpha}_{\nu\gamma}-\Gamma^{\mu}_{\alpha\gamma}\Gamma^{\alpha}_{\nu\beta}$算过三维以上的黎曼曲率张量吧？别说算了，连搞清楚求和指标都不容易，这样对比起来，外微分的技巧就有效多了。当然，不管什么方法，首先总需要一点时间训练到熟悉；其次，就算已经熟悉了方法，也需要花点时间思考才能做出来，不可能一眼就看出来——除非是计算机。
