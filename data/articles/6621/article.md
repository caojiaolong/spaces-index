# ON\-LSTM：用有序神经元表达层次结构

> 作者：苏剑林 · 科学空间 · 2019-05-28
>
> 原文：<https://spaces.ac.cn/archives/6621>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

今天介绍一个有意思的LSTM变种：ON\-LSTM，其中“ON”的全称是“Ordered Neurons”，即有序神经元，换句话说这种LSTM内部的神经元是经过特定排序的，从而能够表达更丰富的信息。ON\-LSTM来自文章[《Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks》](<https://papers.cool/arxiv/1810.09536>)，顾名思义，将神经元经过特定排序是为了将层级结构（树结构）整合到LSTM中去，从而允许LSTM能自动学习到层级结构信息。这篇论文还有另一个身份：<u>ICLR 2019的两篇最佳论文之一</u>，这表明在神经网络中融合层级结构（而不是纯粹简单地全向链接）是很多学者共同感兴趣的课题。

笔者留意到ON\-LSTM是因为[机器之心的介绍](<https://mp.weixin.qq.com/s/vIL-bKHZK-6eXZYWxrc9vw>)，里边提到它除了提高了语言模型的效果之外，甚至还可以无监督地学习到句子的句法结构！正是这一点特性深深吸引了我，而它最近获得ICLR 2019最佳论文的认可，更是坚定了我要弄懂它的决心。认真研读、推导了差不多一星期之后，终于有点眉目了，遂写下此文。

在正式介绍ON\-LSTM之后，我忍不住要先吐槽一下这篇文章实在是写得太差了，将一个明明很生动形象的设计，讲得异常晦涩难懂，其中的核心是$\tilde{f}_t$和$\tilde{i}_t$的定义，文中几乎没有任何铺垫就贴了出来，也没有多少诠释，开始的读了好几次仍然像天书一样\.\.\.总之，文章写法实在不敢恭维～

## 背景内容

通常来说，在文章的前半部分，都是要先扯一些背景知识的～

### 回顾LSTM

首先来回顾一下普通的LSTM。用常见的记号，普通的LSTM写为：
\begin{equation}\begin{aligned} f_{t} & = \sigma \left( W_{f} x_{t} + U_{f} h_{t - 1} + b_{f} \right) \\
i_{t} & = \sigma \left( W_{i} x_{t} + U_{i} h_{t - 1} + b_{i} \right) \\
o_{t} & = \sigma \left( W_{o} x_{t} + U_{o} h_{t - 1} + b_{o} \right) \\
\hat{c}_t & = \tanh \left( W_{c} x_{t} + U_{c} h_{t - 1} + b_{c} \right)\\
c_{t} & = f_{t} \circ c_{t - 1} + i_{t} \circ \hat{c}_t  \\
h_{t} & = o_{t} \circ \tanh \left( c_{t} \right)\end{aligned}\label{eq:lstm}\end{equation}
如果熟悉了神经网络本身，其实这样的结构没有什么神秘的，$f_t,i_t,o_t$就是三个单层全连接模型，输入是历史信息$h_{t-1}$和当前信息$x_t$，用sigmoid激活，因为sigmoid的结果在0～1之间，所以它们的含义可以诠释为“门（gate）”，分别称为<u>遗忘门、输入门、输出门</u>。不过我个人觉着gate这个名字是不够贴切的，“valve（阀门）”也许更贴切些。

有了门之后，$x_t$被整合为$\hat{c}_t$，然后通过$\circ$运算（对应逐位相乘，有时候也记为$\otimes$）与前面的“门”结合起来，来对$c_{t-1}$和$\hat{c}_t$进行加权求和。

下面是自己画的一个LSTM的示意图：

[![LSTM运算流程示意图](<https://spaces.ac.cn/usr/uploads/2019/05/203965127.png>)](<https://spaces.ac.cn/usr/uploads/2019/05/203965127.png>)

LSTM运算流程示意图

### 语言和序信息

在常见的神经网络中，神经元通常都是无序的，比如遗忘门$f_t$是一个向量，向量的各个元素的位置没有什么规律。如果把LSTM运算过程中涉及到的所有向量的位置按照同一方式重新打乱，权重的顺序也相应地打乱，然后输出结果可以只是原来向量的重新排序（考虑多层的情况下，甚至可以完全不变），信息量不变，不影响后续网络对它的使用。

换言之，LSTM以及普通的神经网络都没有用到神经元的序信息，ON\-LSTM则试图把这些神经元排个序，并且用这个序来表示一些特定的结构，从而把神经元的序信息利用起来。

ON\-LSTM的思考对象是自然语言。一个自然句子通常能表示为一些层级结构，这些结构如果人为地抽象出来，就是我们所说的句法信息，而ON\-LSTM希望能够模型在训练的过程中自然地学习到这种层级结构，并且训练完成后还能把它解析出来（可视化），这就利用到了前面说的神经元的序信息。（曾经做过的相关研究[《最小熵原理（三）：“飞象过河”之句模版和语言结构》](<https://spaces.ac.cn/archives/5577>)）

为了达到这个目标，我们需要有一个层级的概念，层级越低代表语言中颗粒度越小的结构，而层级越高则代表颗粒度越粗的结构，比如在中文句子中，“字”可以认为是最低层级的结构，词次之，再上面是词组、短语等。层级越高，颗粒度越粗，那么它在句子中的跨度就越大。

用原文的图示就是：

[![层级结构导致了不同模型的不同跨度，也就是不同层级信息的传输距离不一样，这最终将引导我们将层级结构进行矩阵化表示，从而融入到神经网络中。](<https://spaces.ac.cn/usr/uploads/2019/05/373971065.png>)](<https://spaces.ac.cn/usr/uploads/2019/05/373971065.png>)

层级结构导致了不同模型的不同跨度，也就是不同层级信息的传输距离不一样，这最终将引导我们将层级结构进行矩阵化表示，从而融入到神经网络中。

## ON\-LSTM

最后一句“层级越高，颗粒度越粗，那么它在句子中的跨度就越大”看起来是废话，但它对于ON\-LSTM的设计有着指导作用。首先，这要求我们在设计ON\-LSTM的编码时能区分高低层级的信息；其次，这也告诉我们，高层级的信息意味着它要在高层级对应的编码区间保留更久（不那么容易被遗忘门过滤掉），而低层级的信息则意味着它在对应的区间更容易被遗忘。

### 设计：分区间更新

有了这个指导之后，我们可以着手建立。假设ON\-LSTM中的神经元都排好序后，向量$c_t$的index越小的元素，表示越低层级的信息，而index越大的元素，则表示越高层级的信息。然后，ON\-LSTM的门结构和输出结构依然和普通的LSTM一样：
\begin{equation}\begin{aligned} f_{t} & = \sigma \left( W_{f} x_{t} + U_{f} h_{t - 1} + b_{f} \right) \\
i_{t} & = \sigma \left( W_{i} x_{t} + U_{i} h_{t - 1} + b_{i} \right) \\
o_{t} & = \sigma \left( W_{o} x_{t} + U_{o} h_{t - 1} + b_{o} \right) \\
\hat{c}_t & = \tanh \left( W_{c} x_{t} + U_{c} h_{t - 1} + b_{c} \right)\\
h_{t} & = o_{t} \circ \tanh \left( c_{t} \right) \end{aligned}\label{eq:update-o0}\end{equation}
不同的是从$\hat{c}_t$到$c_t$的更新机制不一样。

接下来，初始化一个全零的$c_t$，即没有任何记忆，或者想象为一个空的U盘。然后，我们将历史信息和当前输入按一定规律存入到$c_t$中（即更新$c_t$）。每次在更新$c_t$之前，首先预测两个整数$d_f$和$d_i$，分别表示历史信息$h_{t-1}$和当前输入$x_t$的层级：
\begin{equation}\begin{aligned} d_f = F_1\left(x_t, h_{t-1}\right) \\ d_i = F_2\left(x_t, h_{t-1}\right) \end{aligned}\end{equation}
至于$F_1, F_2$的具体结构，我们后面再补充，先把核心思路讲清楚。这便是我不满原论文写作的原因，一上来就定义$\text{cumax}$，事前事后都没能把思想讲清楚。

有了$d_f,d_i$之后，那么有两种可能：

<strong>1、$d_f \leq d_i$，这意味着当前输入$x_t$的层级要高于历史记录$h_{t-1}$的层级，那就是说，两者之间的信息流有交汇，当前输入信息要整合到高于等于$d_f$的层级中</strong>，方法是：
\begin{equation}\begin{aligned} c_t = \begin{pmatrix}\hat{c}_{t,< d_f} \\
f_{t,[d_f:d_i]}\circ c_{t-1,[d_f:d_i]}  + i_{t,[d_f:d_i]}\circ \hat{c}_{t,[d_f:d_i]} \\
c_{t-1,> d_i}
\end{pmatrix}\end{aligned}\label{eq:update-o1}\end{equation}
这个公式是说，由于当前输入层级更高，它影响到了交集$[d_f, d_i]$的部分，这部分由普通的LSTM更新公式来更新，小于$d_f$的部分，直接覆盖为当前输入$\hat{c}_t$对应的部分，大于$d_i$的部分，保持历史记录$c_{t-1}$对应的部分不变。

这个更新公式是符合直觉的，因为我们已经将神经元排好序了，位置越前的神经元储存越低层结构信息。而对于当前输入来说，显然更容易影响低层信息，所以当前输入的“波及”范围是$[0, d_i]$（自下而上），也可以理解为当前输入所需要的储蓄空间就是$[0, d_i]$；而对于历史记录来说，它保留的是高层信息，所以“波及”范围是$[d_f, d_{\max}]$（自上而下，$d_{\max}$是最高层级），或许说历史信息所需要的储蓄空间是$[d_f, d_{\max}]$。在不相交部分，它们“各自为政”，各自保留自己的信息；在相交部分，信息要进行融合，退化为普通的LSTM。

[![ON-LSTM设计图。主要想法是将LSTM的神经元排序，然后分段更新。](<https://spaces.ac.cn/usr/uploads/2019/05/524726490.png>)](<https://spaces.ac.cn/usr/uploads/2019/05/524726490.png>)

ON\-LSTM设计图。主要想法是将LSTM的神经元排序，然后分段更新。

<strong>2、$d_f > d_i$，这意味着历史记录$h_{t-1}$和当前输入$x_t$互不相交，那么对于$(d_i, d_f)$的区间“无人问津”，所以只好保持初始状态（即全零，可以理解为没有东西写入）；而剩下部分，当前输入直接把自己的信息写入到$[0, d_i]$区间，而历史信息直接写入到$[d_f, d_{\max}]$区间。这种情况下，当前输入和历史信息合并起来都不能占满整个储蓄空间，从而空下了一些剩余容易（中间那部分全零）。</strong>：
\begin{equation}\begin{aligned} c_t = \begin{pmatrix}\hat{c}_{t,\leq d_i} \\
0_{(d_i : d_f)} \\
c_{t-1,\geq d_f}
\end{pmatrix}\end{aligned}\label{eq:update-o2}\end{equation}
其中$(d_i : d_f)$表示大于$d_i$、小于$d_f$的区间；而前面的$[d_f : d_i]$表示大于等于$d_f$、小于等于$d_i$的区间。

至此，我们能够理解ON\-LSTM的基本原理了，它将神经元排序之后，通过位置的前后来表示信息层级的高低，然后在更新神经元时，先分别预测历史的层级$d_f$和输入的层级$d_i$，通过这两个层级来对神经元实行分区间更新。

[![ON-LSTM分区间更新图示。图上数字都是随机生成的，最上为历史信息，最下为当前输入，中间为当前整合的输出。最上方黄色部分为历史信息层级（主遗忘门），最下方绿色部分为输入信息层级（主输入门），中间部分黄色则是直接复制的历史信息，绿色则是直接复制的输入信息，紫色是按照LSTM方式融合的交集信息，白色是互不相关的“空白地带”。从历史信息（最上层黄色部分）的复制传递中，我们就可以析出对应的如右图的层次结构（注意右图的层次结构与作图的过程不是精确对应的，仅作粗略示意，读者应侧重对图像与模型的直观感知而不是细抠对应关系）。](<https://spaces.ac.cn/usr/uploads/2019/06/956027511.png>)](<https://spaces.ac.cn/usr/uploads/2019/06/956027511.png>)

ON\-LSTM分区间更新图示。图上数字都是随机生成的，最上为历史信息，最下为当前输入，中间为当前整合的输出。最上方黄色部分为历史信息层级（主遗忘门），最下方绿色部分为输入信息层级（主输入门），中间部分黄色则是直接复制的历史信息，绿色则是直接复制的输入信息，紫色是按照LSTM方式融合的交集信息，白色是互不相关的“空白地带”。从历史信息（最上层黄色部分）的复制传递中，我们就可以析出对应的如右图的层次结构（注意右图的层次结构与作图的过程不是精确对应的，仅作粗略示意，读者应侧重对图像与模型的直观感知而不是细抠对应关系）。

<strong>这样一来，高层信息就可能保留相当长的距离（因为高层直接复制历史信息，导致历史信息可能不断被复制而不改变），而低层信息在每一步输入时都可能被更新（因为低层直接复制输入，而输入是不断改变的），所以就通过信息分级来嵌入了层级结构。更通俗地说就是分组更新，更高的组信息传得更远（跨度更大），更低的组跨度更小，这些不同的跨度就形成了输入序列的层级结构。</strong>

<strong>（请反复阅读这段话，必要时对照上图，直接完全理解为止，这段话称得上是ON\-LSTM的设计总纲。）</strong>

### 成型：分段软化

现在要解决的问题就是，这两个层级怎么预测，即$F_1, F_2$怎么构建。用一个模型来输出一个整数不难，但是这样的模型通常都是不可导的，无法很好地整合到整个模型进行反向传播，所以，更好的方案是进行“软化”，即寻求一些光滑近似。

为了进行软化，我们先对$\eqref{eq:update-o1},\eqref{eq:update-o2}$进行改写。引入记号$1_k$，它表示第$k$位为1、其他都为0的$d_{\max}$维向量（即one hot向量），那么$\eqref{eq:update-o1},\eqref{eq:update-o2}$可以统一地写为
\begin{equation}\begin{aligned}\tilde{f}_t & = \stackrel{\rightarrow}{\text{cs}}\left(1_{d_f}\right), \quad \tilde{i}_t = \stackrel{\leftarrow}{\text{cs}}\left(1_{d_i}\right) \\
\omega_t & = \tilde{f}_t \circ \tilde{i}_t \quad (\text{用来表示交集})\\
c_t & = \underbrace{\omega_t \circ \left(f_{t} \circ c_{t - 1} + i_{t} \circ \hat{c}_t \right)}_{\text{交集部分}} + \underbrace{\left(\tilde{f}_t - \omega_t\right)\circ c_{t - 1}}_{\text{大于}\max\left(d_f, d_i\right)\text{的部分}} + \underbrace{\left(\tilde{i}_t - \omega_t\right)\circ \hat{c}_{t}}_{\text{小于}\min\left(d_f,d_i\right)\text{的部分}}
\end{aligned}\label{eq:update-o3}\end{equation}
其中$\stackrel{\rightarrow}{\text{cs}}$/$\stackrel{\leftarrow}{\text{cs}}$分别是右向/左向的cumsum操作：
\begin{equation}\begin{aligned}\stackrel{\rightarrow}{\text{cs}}([x_1,x_2,\dots,x_n]) & = [x_1, x_1+x_2, \dots,x_1+x_2+\dots+x_n]\\
\stackrel{\leftarrow}{\text{cs}}([x_1,x_2,\dots,x_n]) & = [x_1+x_2+\dots+x_n,\dots,x_n+x_{n-1},x_n]\end{aligned}\end{equation}
注意，这里指的是$\eqref{eq:update-o3}$所给出的结果，跟$\eqref{eq:update-o1},\eqref{eq:update-o2}$分情况给出的结果，是<strong>完全等价</strong>的。这只需要留意到$\tilde{f}_t$给出了一个从$d_f$位开始后面全是1、其他位全是0的$d_{\max}$维向量，而$\tilde{i}_t$给出了一个从0到$d_i$位全是1、其他位全是0的$d_{\max}$维向量，那么$\omega_t = \tilde{f}_t \circ \tilde{i}_t$正好给出了交集部分为1、其余全是0的向量（如果没有交集，那就是全0向量），所以$\omega_t \circ \left(f_{t} \circ c_{t - 1} + i_{t} \circ \hat{c}_t \right)$这部分就是在处理交集部分；而$\left(\tilde{f}_t - \omega_t\right)$得到一个从$\max\left(d_f,d_i\right)$位开始后面全是1、其他位全是0的$d_{\max}$维向量，正好标记了历史信息的范围$[d_f, d_{\max}]$去掉了交集之后的部分；而$\left(\tilde{i}_t - \omega_t\right)$得到一个从$0\sim \min\left(d_f,d_i\right)$位全是1、其他位全是0的$d_{\max}$维向量，正好标记了当前输入的范围$[0, d_i]$去掉了交集之后的部分。

现在，$c_t$的更新公式由式$\eqref{eq:update-o3}$来描述，两个one hot向量$1_{d_f},1_{d_i}$由两个整数$d_f,d_i$决定，而这两个整数本身是由模型$F_1, F_2$预测出来的，所以我们可以干脆直接用模型预测$1_{d_f},1_{d_i}$就是了。当然，就算预测出来两个one hot向量，也没有改变整个更新过程不可导的事实。但是，我们可以考虑将$1_{d_f},1_{d_i}$用一般的浮点数向量来代替，比如：
\begin{equation}\begin{aligned}1_{d_f}\approx& softmax\left( W_{\tilde{f}} x_{t} + U_{\tilde{f}} h_{t - 1} + b_{\tilde{f}} \right)\\
1_{d_i}\approx& softmax\left( W_{\tilde{i}} x_{t} + U_{\tilde{i}} h_{t - 1} + b_{\tilde{i}} \right)
\end{aligned}\end{equation}
这样一来，我们用一个$h_{t-1}$和$x_t$的全连接层，来预测两个向量并且做$softmax$，就可以作为$1_{d_f},1_{d_i}$的近似，并且它是完全可导的，从而我们将它们取代$1_{d_f},1_{d_i}$代入到$\eqref{eq:update-o3}$中，就得到ON\-LSTM的$c_t$的更新公式：
\begin{equation}\begin{aligned}\tilde{f}_t & = \stackrel{\rightarrow}{\text{cs}}\left(softmax\left( W_{\tilde{f}} x_{t} + U_{\tilde{f}} h_{t - 1} + b_{\tilde{f}} \right)\right)\\
\tilde{i}_t & = \stackrel{\leftarrow}{\text{cs}}\left(softmax\left( W_{\tilde{i}} x_{t} + U_{\tilde{i}} h_{t - 1} + b_{\tilde{i}} \right)\right) \\
\omega_t & = \tilde{f}_t \circ \tilde{i}_t \quad (\text{用来表示交集})\\
c_t & = \underbrace{\omega_t \circ \left(f_{t} \circ c_{t - 1} + i_{t} \circ \hat{c}_t \right)}_{\text{交集部分}} + \underbrace{\left(\tilde{f}_t - \omega_t\right)\circ c_{t - 1}}_{\text{大于}\max\left(d_f, d_i\right)\text{的部分}} + \underbrace{\left(\tilde{i}_t - \omega_t\right)\circ \hat{c}_{t}}_{\text{小于}\min\left(d_f,d_i\right)\text{的部分}}
\end{aligned}\end{equation}
把剩余部分（即$\eqref{eq:update-o0}$）也写在一起，整个ON\-LSTM的更新公式就是：
\begin{equation}\begin{aligned} f_{t} & = \sigma \left( W_{f} x_{t} + U_{f} h_{t - 1} + b_{f} \right) \\
i_{t} & = \sigma \left( W_{i} x_{t} + U_{i} h_{t - 1} + b_{i} \right) \\
o_{t} & = \sigma \left( W_{o} x_{t} + U_{o} h_{t - 1} + b_{o} \right) \\
\hat{c}_t & = \tanh \left( W_{c} x_{t} + U_{c} h_{t - 1} + b_{c} \right)\\
\tilde{f}_t & = \stackrel{\rightarrow}{\text{cs}}\left(softmax\left( W_{\tilde{f}} x_{t} + U_{\tilde{f}} h_{t - 1} + b_{\tilde{f}} \right)\right)\\
\tilde{i}_t & = \stackrel{\leftarrow}{\text{cs}}\left(softmax\left( W_{\tilde{i}} x_{t} + U_{\tilde{i}} h_{t - 1} + b_{\tilde{i}} \right)\right) \\
\omega_t & = \tilde{f}_t \circ \tilde{i}_t\\
c_t & = \omega_t \circ \left(f_{t} \circ c_{t - 1} + i_{t} \circ \hat{c}_t \right) + \left(\tilde{f}_t - \omega_t\right)\circ c_{t - 1} + \left(\tilde{i}_t - \omega_t\right)\circ \hat{c}_{t}\\
h_{t} & = o_{t} \circ \tanh \left( c_{t} \right)\end{aligned}\end{equation}
示意图如下。对比LSTM的$\eqref{eq:lstm}$，就可以发现主要改动在哪了。其中新引入的$\tilde{f}_t$和$\tilde{i}_t$被作者称为“主遗忘门（master forget gate）”和“主输入门（master input gate）”。

[![ON-LSTM运算流程示意图。主要是将分段函数用cumax光滑化变成可导。](<https://spaces.ac.cn/usr/uploads/2019/05/1083836927.png>)](<https://spaces.ac.cn/usr/uploads/2019/05/1083836927.png>)

ON\-LSTM运算流程示意图。主要是将分段函数用cumax光滑化变成可导。

> <strong>注：</strong>
> 
> 
> 
> 1、论文中将$\stackrel{\rightarrow}{\text{cs}}(softmax(x))$简记为$\text{cumax}(x)$，这只是记号上的转换而已；
> 
> 
> 
> 2、作为数列来看，$\hat{f}_t$是一个单调递增的数列，而$\hat{i}_t$是一个单调递减的数列；
> 
> 
> 
> 3、对于$\tilde{i}_t$，论文定义为
> \begin{equation}1-\text{cumax}\left( W_{\tilde{i}} x_{t} + U_{\tilde{i}} h_{t - 1} + b_{\tilde{i}} \right)\end{equation}
> 这个选择会产生类似的单调递减的向量，一般情况下没有什么差别，但从对称性的角度来看，我认为我的选择更合理一些。

## 实验与思考

下面简单汇总一下ON\-LSTM的实验，其中包括原作者的实现（PyTroch）以及笔者自己的复现（Keras），最后谈及笔者对此ON\-LSTM的一些思考。

<strong>作者实现：</strong>[https://github\.com/yikangshen/Ordered\-Neurons](<https://github.com/yikangshen/Ordered-Neurons>)

<strong>个人实现：</strong>[https://github\.com/bojone/on\-lstm](<https://github.com/bojone/on-lstm>)

（限于笔者水平，个人的理解、复现可能存在问题，如果读者发现，请不吝指出，谢谢。个人复现目前只保证Python 2\.7 \+ Tensorflow 1\.8 \+ Keras 2\.24能跑通，其他环境不保证。）

### 分组的层级

代表层级的向量$\tilde{f}_t,\tilde{i}_t$要与$f_t$等做$\circ$运算，这意味着它们的维度大小（即神经元数目）要相等。而我们知道，根据不同的需求，LSTM的隐层神经元数可以达到几百甚至几千，这意味着$\tilde{f}_t,\tilde{i}_t$所描述的层级数也有几百甚至几千。而事实上，序列的层级结构（如果存在的话）的总层级数一般不会太大，也就是说两者之间存在一点矛盾之处。

ON\-LSTM的作者想了个比较合理的解决方法，假设隐层神经元数目为$n$，它可以分解为$n=pq$，那么我们可以只构造一个$p$个神经元的$\tilde{f}_t,\tilde{i}_t$，然后将$\tilde{f}_t,\tilde{i}_t$的每个神经元依次重复$q$次，这样就得到一个$n$维的$\tilde{f}_t,\tilde{i}_t$，然后再与$f_t$等做$\circ$运算。例如$n=6=2\times 3$，那么先构造一个2维向量如$[0.1, 0.9]$，然后依次重复3次得到$[0.1, 0.1, 0.1, 0.9, 0.9, 0.9]$。

这样一来，我们既减少了层级的总数，同时还减少了模型的参数量，因为$p$通常可以取得比较小（比$n$小1～2个数量级），因此相比普通的LSTM，ON\-LSTM并没有增加太多参数量。

### 语言模型

作者做了若干个实验，包括语言模型、句法评价、逻辑推理等，不少实验中均达到当前最优效果，普遍超越了普通的LSTM，这证明了ON\-LSTM所引入的层级结构信息是有价值的。其中我比较熟悉的也就只有语言模型了，就放一个语言模型实验的截图好了：

[![ON-LSTM原论文中的语言模型实验效果](<https://spaces.ac.cn/usr/uploads/2019/05/3484159577.png>)](<https://spaces.ac.cn/usr/uploads/2019/05/3484159577.png>)

ON\-LSTM原论文中的语言模型实验效果

### 无监督句法

如果仅仅是在常规的一些语言任务中超过普通LSTM，那么ON\-LSTM也算不上什么突破，但ON\-LSTM的一个令人兴奋的特性是它能够无监督地从训练好的模型（比如语言模型）中提取输入序列的层级树结构。提取的思路如下：

首先我们考虑：
\begin{equation}p_f = softmax\left( W_{\tilde{f}} x_{t} + U_{\tilde{f}} h_{t - 1} + b_{\tilde{f}} \right)\end{equation}
它是$\tilde{f}_t$在$\stackrel{\rightarrow}{\text{cs}}$之前的结果，根据我们前面的推导，它就是历史信息的层级$d_f$的一个软化版本，那么我们可以写出：
\begin{equation}d_f\approx\mathop{\text{argmax}}_{k} p_f(k)\end{equation}
这里的$p_f(k)$就是指向量$p_f$的第$k$个元素。但是，$p_f$中所包含的$softmax$本身就是一个“软化”后的算子，这种情况下我们可能考虑“软化”的$\text{argmax}$比较好（参考[《函数光滑化杂谈：不可导函数的可导逼近》](<https://spaces.ac.cn/archives/6620/comment-page-1#argmax>)），即
\begin{equation}d_f\approx \sum_{k=1}^n k\times p_f(k)=n\left(1 - \frac{1}{n}\sum_{k=1}^n \tilde{f}_t(k)\right)+1\end{equation}
第二个等号是恒等变换，大家可以自行证明一下（原论文$(15)$式是有误的）。这样我们就得到了一个层级的计算公式了，当$n$固定时它直接取决于它$\left(1 - \frac{1}{n}\sum\limits_{k=1}^n \tilde{f}_t(k)\right)$

这样以来，我们就可以用序列
\begin{equation}\left\{d_{f,t}\right\}_{t=1}^{\text{seq_len}}=\left\{\left(1 - \frac{1}{n}\sum\limits_{k=1}^n \tilde{f}_t(k)\right)\right\}_{t=1}^{\text{seq_len}}\end{equation}
来表示输入序列的层级变化。有了这个层级序列后，按照下述贪心算法来析出层次结构：

> 给定输入序列$\left\{x_{t}\right\}$到预训练好的ON\-LSTM，输出对应的层级序列$\left\{d_{f,t}\right\}$，然后找出层级序列中最大值所在的下标，比如$k$，那么就将输入序列分区为$[x_{t < k}, [x_k, x_{t > k}]]$。然后对子序列$x_{t < k}$和$x_{t > k}$重复上述步骤，直到每个子序列长度为1。

算法的大概意思是从最高层级处断开（这意味着当此处包含的历史信息最少，与前面所有内容的联系最为薄弱，最有可能是一个新的子结构的开始），然后递归处理，从而逐渐得到输入序列隐含的嵌套结构。作者是用三层的ON\-LSTM训练了一个语言模型，然后用中间那层ON\-LSTM的$\tilde{f}_t$来计算层级，然后跟标注的句法结构对比，发现准确率颇高。我自己也在中文语料下尝试了一下：[https://github\.com/bojone/on\-lstm/blob/master/lm\_model\.py](<https://github.com/bojone/on-lstm/blob/master/lm_model.py>)

至于效果，因为我没做过也不了解句法分析，我也不知道怎么评价，反正好像看着是那么一回事，但是又好像不大对一样，所以各位读者自己评价好了～近一两年，无监督句法分析其实还有不少研究工作，可能要都读一读才能更深刻地理解ON\-LSTM。

> <strong>输入：苹果的颜色是什么</strong>
> <strong>输出：</strong>
> &#91;
>   &#91;
>     &#91;
>       '苹果',
>       '的'
>     &#93;,
>     &#91;
>       '颜色',
>       '是'
>     &#93;
>   &#93;,
>   '什么'
> &#93;
> 
> 
> 
> <strong>输入：爱真的需要勇气</strong>
> <strong>输出：</strong>
> &#91;
>   '爱',
>   &#91;
>     '真的',
>     &#91;
>       '需要',
>       '勇气'
>     &#93;
>   &#93;
> &#93;

### 思考与发散

文章最后，我们来一起思考几个问题。

> RNN还有研究价值？

首先，有读者可能会困惑，都9102年了，居然还有人研究RNN类模型，还有研究价值吗？近年来，BERT、GPT等基于[Attention](<https://spaces.ac.cn/archives/4765>)和语言模型的预训练模型，在NLP的诸多任务上都提升了效果，甚至有文章直接说“RNN已死”之类的。事实上真的如此吗？我认为，RNN活得好好的，并且在将来的相当长时间内都不会死，原因至少包含下面几个：

第一，BERT之类的模型，以增加好几个数量级的算力为代价，在一些任务上提升了也就一两个百分点的效果，这样的性价比只有在学术研究和比赛刷榜才有价值，在工程上几乎没什么用（至少没法直接用）；第二，RNN类的模型本身具有一些无可比拟的优势，比如它能轻松模拟一个计数函数，在很多序列分析的场景，RNN效果好得很；第三，几乎所有seq2seq模型（哪怕是BERT中）decoder都是一种RNN，因为它们基本都是递归解码的，RNN哪会消失？

> 单向ON\-LSTM就够了？

然后，读者可能会有疑惑：你要析出层级结构，但是只用了单向的ON\-LSTM，这意味着当前的层级分析还不依赖于将来的输入，这显然是不大符合事实的。这个笔者也有同样的困惑，但是作者的实验表明这样做效果已经够好了，可能自然语言的整体结构都倾向于是局部的、单向的（从左往右），所以对于自然语言来说单向也就够了。

如果一般情况下是否用双向比较好呢？双向的话是不是要像BERT那样用masked language model的方式来训练呢？双向的话又怎么计算层级序列呢？这一切都还没有完整的答案。至于无监督析出的结构是不是一定就符合人类自身理解的层级结构呢？这个也说不准，因为比较没有什么监督指引，神经网络就“按照自己的方式去理解”了，而幸运的是，神经网络的“自己的方式”，似乎跟人类自身的方式有不少重叠之处。

> 为什么析出层级考虑的是$d_f$而不是$d_i$？

读者可能会困惑，明明有两个master gate，为什么析出层级用$d_f$而不是$d_i$？要回答这个问题，我们要理解$d_f$的含义。我们说$d_f$是历史信息的层级，换言之，它告诉我们做出当前决策还要用多少历史信息。如果$d_f$很大，意味着当前决策几乎用不着历史信息了，这意味着从当前开始就是一个新层级的开始，与历史输入几乎割断了联系。也就是从这种割断和联系中析出了层级结构，所以只能用$d_f$。

> 能否用到CNN或者Attention？

最后，可能想到的一个困惑是，这种设计能不能用到CNN、Attention之中呢？换句话说能不能将CNN、Attention的神经元也排个序，融入层级结构信息呢？个人感觉是有可能的，但需要重新设计，因为层级结构被假设为连续嵌套的，RNN的递归性正好可以描述了这种连续性，而CNN、Attention的非递归性导致我们很难直接去表达这种连续嵌套结构。

不管怎样，我觉得这是个值得思考的主题，有进一步的思考结果我会和大家分享，当然也欢迎读者们和我分享你的思考。

## 文章总结

<strong>本文梳理了LSTM的一个新变种ON\-LSTM的来龙去脉，主要突出了它在表达层级结构上的设计原理。个人感觉整体的设计还是比较巧妙和有趣的，值得细细思考一番。</strong>

最后，学习和研究都关键是有自己的判断能力，不要人云亦云，更不能轻信媒体的“标题党”。BERT的Transformer固然有它的优势，但是LSTM等RNN模型的魅力依然不可小觑。我甚至觉得，诸如LSTM之类的RNN模型，会在将来的某天，焕发出更强烈的光彩，transformer与之相比将会相当逊色。

让我们拭目以待好了。
