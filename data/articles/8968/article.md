# 指数梯度下降 \+ 元学习 = 自适应学习率

> 作者：苏剑林 · 科学空间 · 2022-03-03
>
> 原文：<https://spaces.ac.cn/archives/8968>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前两天刷到了Google的一篇论文[《Step\-size Adaptation Using Exponentiated Gradient Updates》](<https://papers.cool/arxiv/2202.00145>)，在其中学到了一些新的概念，所以在此记录分享一下。主要的内容有两个，一是非负优化的指数梯度下降，二是基于元学习思想的学习率调整算法，两者都颇有意思，有兴趣的读者也可以了解一下。

## 指数梯度下降

梯度下降大家可能听说得多了，指的是对于无约束函数$\mathcal{L}(\boldsymbol{\theta})$的最小化，我们用如下格式进行更新：
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}_t)\end{equation}
其中$\eta$是学习率。然而很多任务并非总是无约束的，对于最简单的非负约束，我们可以改为如下格式更新：
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t \odot \exp\left(- \eta\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}_t)\right)\label{eq:egd}\end{equation}
这里的$\odot$是逐位对应相乘（Hadamard积）。容易看到，只要初始化的$\boldsymbol{\theta}_0$是非负的，那么在整个更新过程中$\boldsymbol{\theta}_t$都会保持非负，这就是用于非负约束优化的“指数梯度下降”。

怎么理解这个“指数梯度下降”呢？也不难，转化为无约束的情形进行推导就行了。如果$\boldsymbol{\theta}$是非负的，那么$\boldsymbol{\varphi}=\log\boldsymbol{\theta}$就是可正可负的了，因此可以设$\boldsymbol{\theta}=e^{\boldsymbol{\varphi}}$转化为关于$\boldsymbol{\varphi}$的无约束优化问题，继而就可以用梯度下降解决：
\begin{equation}\boldsymbol{\varphi}_{t+1} = \boldsymbol{\varphi}_t - \eta\nabla_{\boldsymbol{\varphi}}\mathcal{L}(e^{\boldsymbol{\varphi}_t}) = \boldsymbol{\varphi}_t - \eta e^{\boldsymbol{\varphi}_t}\odot\nabla_{e^{\boldsymbol{\varphi}}}\mathcal{L}(e^{\boldsymbol{\varphi}_t})\end{equation}
我们认为梯度的$e^{\boldsymbol{\varphi}_t}\odot$这部分只起到了调节学习率的作用，所以它不是本质重要的，我们将它舍去得到
\begin{equation}\boldsymbol{\varphi}_{t+1} = \boldsymbol{\varphi}_t - \eta \nabla_{e^{\boldsymbol{\varphi}}}\mathcal{L}(e^{\boldsymbol{\varphi}_t})\end{equation}
两边取指数得
\begin{equation}e^{\boldsymbol{\varphi}_{t+1}} = e^{\boldsymbol{\varphi}_t}\odot\exp\left( - \eta \nabla_{e^{\boldsymbol{\varphi}}}\mathcal{L}(e^{\boldsymbol{\varphi}_t})\right)\end{equation}
换回$\boldsymbol{\theta}=e^{\boldsymbol{\varphi}}$就得到式$\eqref{eq:egd}$。

## 元学习调学习率

对于元学习（Meta Learning），可能多数读者都跟笔者一样听得多，但几乎没接触过。简单来说，普通机器学习跟元学习的关系，就像是数学中“函数”跟“泛函”的关系，泛函是“函数的函数”，元学习则是“学习如何学习（Learning How to Learn）”，也就是说它是关于“学习”本身的方法论，比如接下来要介绍的，就是“用梯度下降去调整梯度下降”。

我们从一般的梯度下降出发，记目标函数$\mathcal{L}$的梯度为$\boldsymbol{g}$，那么更新公式为
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\boldsymbol{g}_t\end{equation}
我们希望给每个分量都调节一下学习率，所以我们引入跟参数一样大小的非负变量$\boldsymbol{\nu}$，修改更新公式为
\begin{equation}\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\boldsymbol{\nu}_{t+1}\odot\boldsymbol{g}_t\label{eq:update}\end{equation}
那么，$\boldsymbol{\nu}$要按照什么规则迭代呢？记住我们最终的目的是最小化$\mathcal{L}$，所以$\boldsymbol{\nu}$的更新规则应该也要是梯度下降，而这里$\boldsymbol{\nu}$要求是非负的，所以我们用指数梯度下降：
\begin{equation}\boldsymbol{\nu}_{t+1} = \boldsymbol{\nu}_t \odot\exp\left(- \gamma\nabla_{\boldsymbol{\nu}_t}\mathcal{L}\right)\label{eq:update-nu}\end{equation}
注意$\mathcal{L}$本来只是$\boldsymbol{\theta}$的函数，但根据$\eqref{eq:update}$，在$t$时刻我们有$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\boldsymbol{\nu}_t\odot\boldsymbol{g}_{t-1}$，所以根据链式法则有
\begin{equation}\nabla_{\boldsymbol{\nu}_t}\mathcal{L} = -\eta\boldsymbol{g}_{t-1} \odot\nabla_{\boldsymbol{\theta}_t}\mathcal{L}= -\eta\boldsymbol{g}_{t-1} \odot\boldsymbol{g}_t\end{equation}
代入到$\nu$的更新公式$\eqref{eq:update-nu}$，得到
\begin{equation}\boldsymbol{\nu}_{t+1} = \boldsymbol{\nu}_t \odot\exp\left( \gamma\eta\boldsymbol{g}_{t-1} \odot\boldsymbol{g}_t\right)\end{equation}
将$\gamma\eta$合成一个参数$\gamma$，于是整个模型的更新公式是：
\begin{equation}\begin{aligned}&\boldsymbol{\nu}_{t+1} = \boldsymbol{\nu}_t \odot\exp\left( \gamma\boldsymbol{g}_{t-1} \odot\boldsymbol{g}_t\right) \\
&\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\boldsymbol{\nu}_{t+1}\odot\boldsymbol{g}_t\end{aligned}\end{equation}
如果$\boldsymbol{\nu}$初始化为全1，那么将有
\begin{equation}\boldsymbol{\nu}_{t+1} = \exp\left(\gamma\sum_{k=1}^t\boldsymbol{g}_{k-1} \odot\boldsymbol{g}_k\right)\end{equation}
可以看到，该方法的学习率调节思路是：如果某分量相邻两步的梯度经常同号，那么对应项的累加结果就是正的，意味着我们可以适当扩大一下学习率；如果相邻两步的梯度经常异号，那么对应项的累加结果很可能是负的，意味着我们可以适当缩小一下学习率。

注意这跟Adam调学习率的思想是不一样的，Adam调节学习率的思想是如果某个分量的梯度长时间很小，那么就意味着该参数可能没学好，所以尝试放大它的学习率。两者也算是各有各的道理吧。

## 简单做个小结

本文主要对“指数梯度下降”和“元学习调学习率”两个概念做了简单笔记，“指数梯度下降”是非负约束优化的一个简单有效的方案，而“元学习调学习率”则是元学习的一个简单易懂的应用。其中在介绍“元学习调学习率”时笔者做了一些简化，相比原论文的形式更为简单一些，但思想是一致的。
