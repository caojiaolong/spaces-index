# 关于NBCE方法的一些补充说明和分析

> 作者：苏剑林 · 科学空间 · 2023-05-31
>
> 原文：<https://spaces.ac.cn/archives/9632>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

上周在[《NBCE：使用朴素贝叶斯扩展LLM的Context处理长度》](<https://spaces.ac.cn/archives/9617>)中，我们介绍了一种基于朴素贝叶斯来扩展LLM的Context长度的方案NBCE（Naive Bayes\-based Context Extension）。由于它有着即插即用、模型无关、不用微调等优点，也获得了一些读者的认可，总的来说目前大家反馈的测试效果还算可以。

当然，部分读者在使用的时候也提出了一些问题。本文就结合读者的疑问和笔者的后续思考，对NBCE方法做一些补充说明和分析。

## 方法回顾

假设$T$为要生成的token序列，$S_1,S_2,\cdots,S_n$是给定的若干个Context，我们需要根据$S_1,S_2,\cdots,S_n$生成$T$，那么就需要估计$p(T|S_1, S_2,\cdots,S_n)$。根据朴素贝叶斯思想，我们得到
\begin{equation}\log p(T|S_1, S_2,\cdots,S_n) = \color{red}{(\beta + 1)\overline{\log p(T|S)}} - \color{green}{\beta\log p(T)} + \color{skyblue}{\text{常数}}\label{eq:nbce-2}\end{equation}
其中$\beta = n - 1$，$\overline{\log p(T|S)} = \frac{1}{n}\sum\limits_{k=1}^n \log p(T|S_k)$，细节请参考上一篇文章。NBCE做了两点改动：1、将$\beta$作为超参数来调；2、将$\overline{\log p(T|S)}$换成一般的Pooling方法$\mathcal{P}$。结果变为
\begin{equation}\log p(T|S_1, S_2,\cdots,S_n) = \color{red}{(\beta + 1)\mathcal{P}[\log p(T|S)]} - \color{green}{\beta\log p(T)} + \color{skyblue}{\text{常数}}\label{eq:nbce-3}\end{equation}
最后，NBCE选取的Pooling方案为“取最小熵者”：
\begin{equation}\begin{aligned}
&\mathcal{P}[\log p(T|S)] = \log p(T|S_{\color{red}{k}}) \\[5pt]
&\color{red}{k} = \mathop{\text{argmin}} \big\{H_1,H_2,\cdots,H_n\big\} \\[5pt]
&H_i = -\sum_T p(T|S_i)\log p(T|S_i)
\end{aligned}\label{eq:min-h}\end{equation}

## 截断预测

式$\eqref{eq:nbce-2}$是标准的朴素贝叶斯结果，但笔者按照它实现时，发现随着$n$的增大，式$\eqref{eq:nbce-2}$的效果慢慢变差直至完全乱码，所以反复调整后，最终选取“取最小熵者”作为NBCE的Pooling方案。但后来仔细一想，式$\eqref{eq:nbce-2}$的这个表现是不正常的，因为朴素贝叶斯的唯一假设是Context之间相互独立，而我所测试的Context是随机找的若干篇新闻，一定程度上满足这个假设的，所以式$\eqref{eq:nbce-2}$再怎么差也不至于完全乱码才对。

百思不得其解之际，微信上[@孔某人](<https://www.zhihu.com/people/hamann-lo>)提醒：<strong>语言模型的训练标签都是One\-Hot，所以预测结果除了头部（概率最高的部分）外其实都是不可信的。</strong>这个提示可谓是一针见血，让答案瞬间水落石出了：由于式$\eqref{eq:nbce-2}$包含了$-\beta\log p(T)$这一项，它会放大尾部的预测结果，如果尾部的预测结果不可靠，那么这个放大作用甚至会完全打乱头部的准确结果。为什么它不会影响“取最小熵者”呢？因为最小熵的结果头部概率往往更大，尾部概率更小，所以即便$-\beta\log p(T)$这一项放大了尾部，也依然无法胜过头部；而对于式$\eqref{eq:nbce-2}$来说，它是所有预测结果的平均，则会弱化头部，以至于加上$-\beta\log p(T)$后尾部胜过了头部。

有了这个线索，解决方法也就很明显了，给每个预测结果都加上Top\-P或者Top\-K截断就行，在Github的代码中，笔者选的是Top\-P截断。

## 无穷处理

然而，事情还没完，经过截断后，$\log p(T|S_k)$和$\log p(T)$的尾部都变成了$-\infty$，这时候式$\eqref{eq:nbce-2}$或式$\eqref{eq:nbce-3}$都有可能出现$(-\infty)-(-\infty)$的无意义运算。总的来说，有下面几种情况：
\begin{array}{c|cc|c}
\hline
& \log p(T|S_k) & \log p(T) & \log p(T|S_k) - \log p(T) \\
\hline
\text{情况一} & > -\infty & > -\infty & > -\infty\\
\text{情况二} & > -\infty & = -\infty & = +\infty \\
\text{情况三} & = -\infty & > -\infty & = -\infty \\
\text{情况四} & = -\infty & = -\infty & \text{NaN}\\
\hline
\end{array}
其中，“情况一”和“情况三”能正常运算，“情况二”虽然也能正常运算，但其结果正无穷是不合理的，“情况四”则是非良定的无意义运算，也就是说，我们需要想办法修正“情况二”和“情况四”，这两种情况正好对应$\log p(T)=-\infty$，所以我们将式$\eqref{eq:nbce-3}$改为：
\begin{equation}\log p(T|S_1, S_2,\cdots,S_n) =\left\{
\begin{aligned}
&\color{red}{\mathcal{P}[\log p(T|S)]}, \quad \text{如果}\color{green}{\log p(T) = -\infty} \\[5pt]
&\color{red}{(\beta + 1)\mathcal{P}[\log p(T|S)]} - \color{green}{\beta\log p(T)}, \quad \text{其他}\\
\end{aligned}\right\} + \color{skyblue}{\text{常数}}\label{eq:nbce-4}\end{equation}
经过上述处理后，标准的朴素贝叶斯式$\eqref{eq:nbce-2}$也能输出正常结果了（虽然最终效果还是不如取最小熵，但至少不会乱码了），并且修改后的代码对Pooling方式和$\beta$都更加鲁棒。

## 转移概率

当用于回答一些观点型问题或者偏向自由创作的问题时，NBCE可能会出现在Context中反复跳转的问题。具体来说，就是模型并没有自信地注意到某个Context中，所以$H_1,H_2,\cdots,H_n$之间相差不大，于是式$\eqref{eq:min-h}$中的$\mathop{\text{argmin}}$结果不稳定，每步生成都选取到了不同的Context上，导致生成结果语义上的不连续甚至完全和Context不相关，加剧LLM的“幻觉”现象。

为了缓解这个问题，我们可以模仿转移概率的概念，适当给上一步所选的Context加权，让模型“非必要不跳转”，具体做法是引入参数$\eta > 0$，将式$\eqref{eq:min-h}$改为
\begin{equation}\color{red}{k} = \mathop{\text{argmin}} \big\{H_1,\cdots,H_{k'-1},H_{k'}\color{red}{-\eta},H_{k'+1},\cdots,H_n\big\}\end{equation}
其中$k'$是上一步生成时选取的Context编号。这样一来，只有$H_k < H_{k'} - \eta$时才会发生Context跳转，降低了跳转概率。

以上提到的修改都已经同步到Github中：

> <strong>Github: [https://github\.com/bojone/NBCE](<https://github.com/bojone/NBCE>)</strong>

## 适用场景

由于朴素贝叶斯所做的独立假设，不少读者可能会怀疑：当Context之间有明显的语义重叠时，NBCE的效果会不会明显下降？或者说，NBCE的适用场景是什么？

事实上，受限于独立假设的是标准的朴素贝叶斯，也就是式$\eqref{eq:nbce-2}$，一般化后的式$\eqref{eq:nbce-3}$和式$\eqref{eq:min-h}$已经基本上不受限于独立假设了。事实上，“取最小熵者”版本的NBCE实质就是用LLM的熵作为相似度对Context进行检索，每步生成都会更新检索结果。所以，NBCE的适用场景是：<strong>假设要预测的答案可以分为若干个片段，每个片段只依赖于一个Context。</strong>

基于这个结论，当我们只有一段长文本作为Context（比如一篇小说）时，我们可以通过有重叠的滑窗自动将长Context划分为多个短Context，而不一定需要人工分割为相对独立的片段，因为刚才的结论告诉我们，NBCE的适用场景跟Context的重叠性无关。至于为什么重叠地滑窗，则只是为了能尽可能依赖单个Context就能输出完整的结果。

以下两种场景用NBCE效果大概率是不会好的：

> <strong>1、有序的Context</strong>：这是指当生成结果强烈依赖于Context的输入顺序（或者更复杂的嵌套结构）时，NBCE通常效果不好，因为NBCE保留了朴素贝叶斯的无序性，这种场景的典型例子是给小说写摘要（小说被切割为多个Context），一个治标不治本的方案是人工给每段Context加上可以识别顺序的标记，如“第xx章”；
> 
> 
> 
> <strong>2、耦合的Context</strong>：这是指必须要结合两个或更多Context构建输出时，NBCE效果不好，因为NBCE每次只选了一个Context，[@孔某人](<https://www.zhihu.com/people/hamann-lo>)给出了一个典型例子“已知$x > 1$，且$x < 0$，求$x$解集”，假设两个条件被分为两个Context，那么必须结合两个Context才能输出正确答案“空集”，只看单个Context无法确定是空集。

如果NBCE还要继续做下去的话，大体上也是围绕以上两个场景进行改进。

## 文章小结

本文介绍了Context长度扩展方案NBCE的一些后续更新及分析，并进一步讨论了NBCE的适用场景。
