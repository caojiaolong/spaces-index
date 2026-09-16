# IMO42\-1，我也会做几何题

> 作者：苏剑林 · 科学空间 · 2011-07-30
>
> 原文：<https://spaces.ac.cn/archives/1453>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

七月再次“农忙”，农村里要插秧了，播下种苗，等待再次收获的季节^\_^

我一直觉得我的数学能力偏向于分析计算而不擅长于几何，纵使遇到几何问题，也是满脑子的解析几何做法，没有纯几何的美。而这几天为了加强数学竞赛题目的能力，我一直在看IMO的题目，并且企图独立做出一些题目，但都无果。我比较感兴趣的是不等式，我感觉一道简单的式子，不用太多的文字就可以讲清楚的题目非不等式莫属，但是IMO的不等式题实在高深，我还没有能够独立做出一道来（参考答案可以看懂，只是想不到思路），或许是我在努力追求统一的方法而不肯研究那些特定的技巧的原因吧。不料今天看了一下2001年IMO的几何题目，发现我可能将它做出来，于是研究了一会，最终很幸运地做了出来。虽然不是最简单的方法，但也与大家分享一下。

[![IMO-42-1](<https://spaces.ac.cn/usr/uploads/2011/07/3670458097.png>)](<https://spaces.ac.cn/attachment/1455/>)

IMO\-42\-1

> 如图，O是锐角三角形ABC的外心，AP是三角形的垂线段，∠B\-∠C不小于30°。证明∠BAC\+∠BOP \< 90°

其实BoJone能做出这道题，是因为这道题目着实不难。我的几何水平依旧停留在初中的阶段，我的证明依旧着重于分析和计算。首先由“∠B\-∠C不小于30°”可以得出∠B \> 60°和∠A\+∠C \< 120°，不然会得出∠C是直角或钝角的结果，这与题目矛盾。

按下图作辅助线：OD垂直于BC，连接O、C，三角形的三个角直接记为A、B、C。

[![IMO-42-1-解答](<https://spaces.ac.cn/usr/uploads/2011/07/2719797395.png>)](<https://spaces.ac.cn/attachment/1454/>)

IMO\-42\-1\-解答

由圆心角是对应圆周角的两倍可以得出：∠BOC=2∠A，于是∠A=∠BOD，题目要证∠A\+∠BOP \< 90°，即证∠BOD\+∠BOP \< 90°，而∠BOD\+∠OBD=90°，等价于证明∠BOP \< ∠OBD，亦等价于证明BP \< OP，即BP<sup>2</sup> \< OP<sup>2</sup>。

设角A、B、C的对边分别为a,b,c，外接圆半径为R，不难看出：
$$BP=c\cdot \cos B$$
由余弦定理：
$$\begin{aligned}OP^2=BP^2+R^2-2R\cdot BP\cdot \cos \angle OBD \\ =BP^2+R^2-2R\cdot BP\cdot \sin \angle BOD=BP^2+R^2-2R\cdot c\cdot \cos B\cdot \sin A\end{aligned}$$

于是证明的结论变为：$R^2-2R*c*cosB*sin A > 0$，而由正弦定理知道：$2R*sin C=c$，所以最终要证明：
$$4\sin A\cdot \sin C\cdot \cos B < 1$$

积化和差：$sin C cos B=1/2 [sin(B+C)-sin(B-C)]$，B\-C ≥ 30°，所以 \\sin(B\-C) ≥ 1/2

因此：
$$\begin{aligned}4\sin A\cdot \sin C\cdot \cos B =2 \sin A [\sin(B+C)-\sin(B-C)] \\ < 2 \sin A (\sin A-1/2) \leq 2\cdot 1\cdot (1-1/2)=1\end{aligned}$$

证毕。

<strong>本文的证明方法属于杀鸡用牛刀\.\.\.只是BoJone的几何能力实在太弱，不能想出更好的证明方法\.\.\.</strong>
