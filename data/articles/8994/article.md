# 为什么需要残差？一个来自DeepNet的视角

> 作者：苏剑林 · 科学空间 · 2022-03-19
>
> 原文：<https://spaces.ac.cn/archives/8994>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《训练1000层的Transformer究竟有什么困难？》](<https://spaces.ac.cn/archives/8978>)中我们介绍了微软提出的能训练1000层Transformer的DeepNet技术。而对于DeepNet，读者一般也有两种反应，一是为此感到惊叹而点赞，另一则是觉得新瓶装旧酒没意思。出现后一种反应的读者，往往是因为DeepNet所提出的两个改进点——增大恒等路径权重和降低残差分支初始化——实在过于稀松平常，并且其他工作也出现过类似的结论，因此很难有什么新鲜感。

诚然，单从结论来看，DeepNet实在算不上多有意思，但笔者觉得，DeepNet的过程远比结论更为重要，它有意思的地方在于提供了一个简明有效的梯度量级分析思路，并可以用于分析很多相关问题，比如本文要讨论的“为什么需要残差”，它就可以给出一个比较贴近本质的答案。

## 增量爆炸

为什么需要残差？答案是有了残差才更好训练深层模型，这里的深层可能是百层、千层甚至万层。那么问题就变成了为什么没有残差就不容易训练深层模型呢？

很多读者的答案应该是梯度消失或梯度爆炸。这确实是两个很重要的问题，然而配合特定的初始化方法和Normalization技术，我们已经可以将普通前馈神经网络的梯度做得很稳定了，但即便如此训练深层前馈神经网络依然不容易。这说明其中的原因不仅有梯度消失/爆炸，还有别的问题，它就是我们在[《训练1000层的Transformer究竟有什么困难？》](<https://spaces.ac.cn/archives/8978>)中已经讨论过的“增量爆炸”。

理解增量爆炸并不困难，假设损失函数为$\mathcal{L}(\boldsymbol{\theta})$，$\boldsymbol{\theta}$是它的参数，当参数由$\boldsymbol{\theta}$变为$\boldsymbol{\theta}+\Delta\boldsymbol{\theta}$时：
\begin{equation}\Delta\mathcal{L} = \mathcal{L}(\boldsymbol{\theta}+\Delta\boldsymbol{\theta}) - \mathcal{L}(\boldsymbol{\theta}) \approx \langle\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}),\Delta\boldsymbol{\theta}\rangle\end{equation}
对于SGD有$\Delta\boldsymbol{\theta}=-\eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$，那么$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2$。设模型有$N$层，每层的平均参数量为$K$，如果解决了梯度消失/爆炸问题，那么可以认为每个参数的梯度是$\mathcal{O}(1)$量级，所以有$\Delta\mathcal{L}=\mathcal{O}(\eta NK)$。因此，模型每一步的更新量是正比于模型深度$N$的（宽度不在本文讨论范围），如果模型越深，那么更新量就越大，这意味着初始阶段模型越容易进入不大好的局部最优点，然后训练停滞甚至崩溃，这就是“增量爆炸”问题。

## 治标之法

简单来说，“增量爆炸”就是在层数变多时，参数的微小变化就会导致损失函数的大变化，这对于模型的训练，特别是初始阶段的训练时尤其不利的。对此，一个直接的应对技巧就是Wamrup，初始阶段先用极小的学习率，然后再慢慢增大，避免在初始阶段学习过快。待模型平稳渡过初始阶段的“危险期”后，就可以正常训练了。

然而，尽管Wamrup能起到一定的作用，但其实是“治标不治本”的，因为“参数的微小变化就会导致损失函数的大变化”意味着模型本身的抖动很大，用更专业的话说就是模型的landscape极度不平滑了，这不是一个好模型应该具备的性质。因此，我们应该通过修改模型来解决这个问题，而不是通过降低学习率这种“表面”方法。

所谓修改模型，就是通过调整模型结构或初始化方式，来自然地抵消层数$N$对更新量的影响。根据前面的结果$\Delta\mathcal{L} \approx -\eta\Vert\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\Vert^2$和$\Delta\mathcal{L}=\mathcal{O}(\eta NK)$，那么要抵消层数的影响，就要使得梯度$\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})$变为$\mathcal{O}(1/\sqrt{N})$量级。换言之，每个参数的梯度要随着层数的增多而变小。

## 稳定传播

如果只是纯粹地缩小梯度，那么很简单，只要尽量降低初始化方差就行。但实际上我们在缩小梯度的同时，必须还要保持前向传播稳定性，因为前向传播的稳定性是我们对所做任务的一种先验知识，它意味着是模型更好的起点。在[《浅谈Transformer的初始化、参数化与标准化》](<https://spaces.ac.cn/archives/8620>)中我们讨论过，前向传播的稳定性可以用二阶矩来衡量，对于简单的线性层
\begin{equation}\boldsymbol{y} = \boldsymbol{x}\boldsymbol{W}, \quad \boldsymbol{x}\in\mathbb{R}^n, \boldsymbol{W}\in\mathbb{R}^{n\times m}\end{equation}
我们已经知道，要想让$\boldsymbol{y}$的二阶矩跟$\boldsymbol{x}$的二阶矩相等，那么需要用均值为零、方差为$1/n$的初始化方法，如果考虑激活函数，那么就多一个常数级别的scale，比如对于$\text{relu}$激活函数来说，方差改为$2/n$。而对于反向传播来说，我们有
\begin{equation}\frac{\partial\mathcal{L}}{\partial \boldsymbol{x}} = \frac{\partial\mathcal{L}}{\partial \boldsymbol{y}}\frac{\partial\boldsymbol{y}}{\partial \boldsymbol{x}} = \frac{\partial\mathcal{L}}{\partial \boldsymbol{y}}\boldsymbol{W}^{\top}\end{equation}
可以看到，反向传播刚好相反，如果要稳定反向传播的二阶矩，那么需要用均值为零、方差为$1/m$的初始化方法。Xavier初始化则取了两者平均$2/(n+m)$，更多细节可以参考[《初始化方法中非方阵的维度平均策略思考》](<https://spaces.ac.cn/archives/8725>)。

换句话说，如果我们要稳定前向传播，那么初始化方差就是$1/n$，而反向传播的二阶矩则是原来的$m/n$倍。$m,n$都是事先选定的超参数，它跟层数没有必然联系，我们不可能通过它来实现梯度降为原来的$1/\sqrt{N}$倍的需求。这就意味着，对于无残差的深层前馈神经网络
\begin{equation}\phi_l(\phi_{l-1}(\phi_{l-2}(\cdots\phi_1(\boldsymbol{x}\boldsymbol{W}_1 + \boldsymbol{b}_1)\cdots)\boldsymbol{W}_{l-1} + \boldsymbol{b}_{l-1})\boldsymbol{W}_l + \boldsymbol{b}_l)\end{equation}
只要它前向传播稳定了，那么反向传播也就固定了，无法使得梯度跟层数相关。因此，我们顶多可以解决深层前馈神经网络的梯度消失和梯度爆炸问题，但无法解决本文开头提到的“增量爆炸”问题，因此深层前馈神经网络必然不好训练。

## 残差初现

这时候残差就可以登场了！不失一般性，假设输入输出维度相等，我们考虑
\begin{equation}\boldsymbol{y} = \boldsymbol{x} + \varepsilon \boldsymbol{f}(\boldsymbol{x};\boldsymbol{\theta})\end{equation}
很显然，只要$\varepsilon$足够小，那么前向传播必然是稳定的；而
\begin{equation}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = \boldsymbol{I} + \varepsilon\frac{\partial \boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{x}}\label{eq:bp}\end{equation}
所以也可以看出，只要$\varepsilon$足够小，那么反向传播也是稳定的。至于参数的梯度
\begin{equation}\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = \frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{\theta}} = \varepsilon\frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}\frac{\partial \boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{\theta}}\end{equation}
这说明我们可以通过控制$\varepsilon$来实现层数相关的梯度缩放！比如要想梯度缩放到$1/\sqrt{N}$，那么让$\varepsilon=1/\sqrt{N}$即可。

有了这个结果，我们就可以回答为什么要用残差了：

> 因为残差结构是可以同时稳定前向传播和反向传播、并且可以缩放参数梯度以解决增量爆炸的一种设计，它能帮助我们训练更深层的模型。

## 足够的小

刚才我们说了两次“$\varepsilon$足够小”，那么多小算足够小呢？$\varepsilon=1/\sqrt{N}$够了没？

假设是一维模型，那么$\frac{\partial y}{\partial x} = 1 + \varepsilon\frac{\partial f}{\partial x}$，一般假设$\frac{\partial f}{\partial x}$是$\mathcal{O}(1)$的，所以我们可以近似地用$\frac{\partial y}{\partial x}=1+\varepsilon$做量级估计，那么传播$N$层后“膨胀系数”近似为$(1+\varepsilon)^N$。而我们知道
\begin{equation}\left(1 + \frac{1}{N}\right)^N < \lim_{N\to\infty} \left(1 + \frac{1}{N}\right)^N = e\end{equation}
也就是说，对于一维模型来说，要使得反向传播不随着层数增加而爆炸，那么至少要$\varepsilon$至少要$\mathcal{O}(1/N)$，$\varepsilon=1/\sqrt{N}$确实不大够。

不过对于高维模型来说，情况有所改观。我们在式$\eqref{eq:bp}$两边同时乘上一个任意向量$\boldsymbol{v}$：
\begin{equation}\boldsymbol{v}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}} = \boldsymbol{v} + \varepsilon\boldsymbol{v}\frac{\partial \boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{x}}\end{equation}
注意在初始阶段$\frac{\boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{x}}$也相当于一个零均值的随机初始化矩阵，在[《从几何视角来理解模型参数的初始化策略》](<https://spaces.ac.cn/archives/7180>)我们讨论过这样的觉得接近一个正交矩阵（的若干倍），所以初始阶段$\boldsymbol{v}$和$\varepsilon\boldsymbol{v}\frac{\partial \boldsymbol{f(\boldsymbol{x};\boldsymbol{\theta})}}{\partial \boldsymbol{x}}$是接近正交的，因此
\begin{equation}\left\Vert\boldsymbol{v}\frac{\partial \boldsymbol{y}}{\partial \boldsymbol{x}}\right\Vert^2 = \mathcal{O}\big((1 + \varepsilon^2)\Vert\boldsymbol{v}\Vert^2\big)\end{equation}
说白了，就是说高维情形每一层的膨胀系数更接近于$1+\varepsilon^2$而不是$1+\varepsilon$，根据一维情况的讨论结果，我们只需要$\varepsilon^2=\mathcal{O}(1/N)$，所以$\varepsilon=1/\sqrt{N}$基本够了。

## 文章小结

本文讨论了“为什么需要残差”这个问题，在DeepNet的启发之下，得到的结论是残差可以同时稳定前向传播和反向传播并解决增量爆炸，从而使得深层模型更容易训练，而无残差的普通前馈神经网络，则无法同时解决这三个问题，因此其深层化后不容易训练。
