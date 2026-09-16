# Transformer升级之路：20、MLA好在哪里?（上）

> 作者：苏剑林 · 科学空间 · 2025-05-04
>
> 原文：<https://spaces.ac.cn/archives/10907>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

自从DeepSeek爆火后，它所提的Attention变体MLA（<strong>M</strong>ulti\-head <strong>L</strong>atent <strong>A</strong>ttention）也愈发受到关注。MLA通过巧妙的设计实现了MHA与MQA的自由切换，使得模型可以根据训练和推理的不同特性（Compute\-Bound or Memory\-Bound）选择最佳的形式，尽可能地达到效率最大化。

诚然，MLA很有效，但也有观点认为它不够优雅，所以寻找MLA替代品的努力一直存在，包括我们也有在尝试。然而，经过一段时间的实验，我们发现很多KV Cache相同甚至更大的Attention变体，最终效果都不如MLA。这不得不让我们开始反思：MLA的出色表现背后的关键原因究竟是什么？

接下来，本文将详细介绍笔者围绕这一问题的思考过程以及相关实验结果。

## 观察

MLA提出自[DeepSeek\-V2](<https://papers.cool/arxiv/2405.04434>)，本文假设读者已经熟悉MLA，至少了解之前的博客[《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》](<https://spaces.ac.cn/archives/10091>)所介绍的内容，因此MLA自身的细节将不会过多展开。

MLA的主要特点如下：

> 1、MLA在训练阶段是一个qk\_head\_dims=(128\+64)、v\_head\_dims=128的MHA；
> 
> 
> 
> 2、MLA在解码阶段是一个qk\_head\_dims=(512\+64)、v\_head\_dims=512、KV\-Shared的MQA；
> 
> 
> 
> 3、MLA的&#91;qc, qr&#93;、&#91;kc, kr&#93;拼接，可以理解为一种[Partial RoPE](<https://spaces.ac.cn/archives/10122#%E9%83%A8%E5%88%86%E6%97%8B%E8%BD%AC>)。

## 猜测

MHA、GQA常用的head\_dims是128，而对于MLA来说，不管是从训练看的128\+64，还是从推理看的512\+64，都要大于128，再结合[《突破瓶颈，打造更强大的Transformer》](<https://spaces.ac.cn/archives/7325>)的经验，我们有：

> <strong>猜测1</strong>： 增大head\_dims是MLA好的关键之一。

另外，KV\-Shared这个特性，可以在同等KV Cache大小下，增大GQA的head\_dims或者num\_groups，所以有：

> <strong>猜测2</strong>： KV\-Shared是MLA好的关键之一。

最后，此前有一些理论和实验显示Partial RoPE可能会对效果有正面帮助（参考[《Transformer升级之路：18、RoPE的底数选择原则》](<https://spaces.ac.cn/archives/10122#%E9%83%A8%E5%88%86%E6%97%8B%E8%BD%AC>)），所以有

> <strong>猜测3</strong>： Partial RoPE是MLA好的关键之一。

## 实验

现在我们通过实验逐一检验以上猜测。

### 设置

所有实验公共部分的超参数如下：

> 1、类似LLAMA3的Dense模型；
> 
> 
> 
> 2、hidden\_size=2048，num\_layers=12，num\_heads=16；
> 
> 
> 
> 3、优化器是[Muon](<https://spaces.ac.cn/archives/10592>)，Attention部分per head更新；
> 
> 
> 
> 4、训练长度为4096，总tokens数为16B，总训练步数为16k；
> 
> 
> 
> 5、所有实验都是只改变Attention，所以参数量不会严格对齐。

### Part I

MLA的KV Cache大小是512\+64，约等于GQA2\-128（第一个数字是num\_groups，第二个数字是head\_dims），所以对比的baseline为GQA2\-128和GQA1\-256。为了验证Partial RoPE，我们增加了GQA1\-256\-PR，具体做法是将Q、K的256 dims分成192\+64两部分，在64上加RoPE，192不加。

结果如下：
$$\begin{array}{c|ccc}
\hline
& \text{Params} & \text{Loss} & \text{Cache} \\
\hline
\text{MLA} & 894M & 2.721 & 576 \\
\text{GQA2-128} & 842M & 2.75 & 512 \\
\text{GQA1-256} & 943M & 2.72 & 512 \\
\text{GQA1-256-PR} & 943M & 2.711 & 512 \\
\hline
\end{array}$$

即
$$\text{GQA2-128} < \text{MLA} \lesssim \text{GQA1-256} < \text{GQA1-256-PR}$$
初步验证了增大head\_dims和Partial RoPE的作用。这样看来，MLA的设计中，RoPE和NoPE拼接这部分看似无奈的设计，极有可能是它效果优异的关键原因！原论文声称MLA甚至优于MHA，大概率也是因为所对比的MHA的head\_dims只有128。

### Part II

为了进一步验证增大head\_dims的作用，我们另外跑了MHA、GQA2\-192、MLA\-256三个实验，MHA是head\_dims=128的常规MHA，GQA2\-192是直接增大GQA2的head\_dims到192，MLA\-256是将MLA的128\+64提升到192\+64，对照如下

$$\begin{array}{c|ccc}
\hline
& \text{Params} & \text{Loss} & \text{Cache} \\
\hline
\text{MHA} & 931M & 2.721 & 4096 \\
\text{MLA} & 894M & 2.721 & 576 \\
\text{MLA-256} & 989M & 2.705 & 576 \\
\text{GQA2-128} & 842M & 2.75 & 512 \\
\text{GQA2-192} & 899M & 2.729 & 768 \\
\text{GQA1-256} & 943M & 2.72 & 512 \\
\text{GQA1-256-PR} & 943M & 2.711 & 512 \\
\hline
\end{array}$$

可以看到，MHA总参数量更多，KV Cache更是7倍于MLA，但Loss才堪堪追平MLA，这跟DeepSeek\-V2里边的结论接近。此外，GQA2\-192优于GQA2\-128，但不如GQA1\-256；MLA的head\_dims升到(192\+64)后，相比(128\+64)也还能进一步提升效果。这些现象都表明，增加head\_dims远比增加num\_groups更有效。

### Part III

接下来我们验证KV\-Shared，即K、V共享全部或大部分dims。这里我们主要考虑的替代品是head\_dims不超过256的GQA，并且控制KV Cache的总大小跟MLA接近，所以当KV\-Shared时，我们可以至多可以考虑GQA2\-256。

由于KV\-Shared跟RoPE不完全兼容，参考MLA的做法，我们将256分成192\+64两部分，其中

> 1、192部分不加RoPE，在K、V间共享；
> 
> 
> 
> 2、64部分加RoPE，只用于K；
> 
> 
> 
> 3、V另外再投影64 dims，concat到共享的192 dims上去。

这样一来，K、V的head\_dims都是256，KV Cache总大小是(192\+64\+64)\*2=640，略大于MLA的512\+64=576，这个版本我们简记为“GQA2\-(192\+64)\-S1”，其实“S1”是“Shared\-1”的缩写。

### Part IV

另外一种KV\-Shared的方案是：

> 1、192部分不加RoPE，在K、V间共享；
> 
> 
> 
> 2、64部分加RoPE，同样在K、V间共享；
> 
> 
> 
> 3、做Attention，由于V带RoPE，此时是绝对位置编码效果；
> 
> 
> 
> 4、为了保证相对位置编码，将输出分成192\+64两部分，64部分再加一次逆向RoPE。

这种做法是K、V完全共享，KV Cache大小是(192\+64)\*2=512，略小于MLA。这个版本我们称为“GQA2\-(192\+64)\-S2”，“S2”是“Shared\-2”的缩写，背后的原理是笔者新提出的VO\-RoPE，参考[《Transformer升级之路：19、第二类旋转位置编码》](<https://spaces.ac.cn/archives/10862>)。

### Part V

另外，根据同样思路补了几个GQA4和GQA1的实验。所有实验结果汇总如下：
$$\begin{array}{c|ccc|c}
\hline
& \text{Params} & \text{Loss} & \text{Cache} & \text{备注} \\
\hline
\text{MLA} & 894M & 2.721 & 576 & \\
\text{MLA-256} & 989M & 2.705 & 576 & \\
\text{GQA2-(192+64)-S1} & 946M & 2.714 & 640 & \\
\text{GQA2-(192+64)-S2} & 943M & 2.708 & 512 & \text{引入VO-RoPE} \\
\text{GQA4-(64+64)-S2} & 842M & 2.738 & 512 & \\
\text{GQA4-(128+64)-S2} & 899M & 2.713 & 768 & \text{KV Cache最大} \\
\text{GQA1-(512+64)-S3} & 1171M & 2.677 & 576 & \text{head_dims最大} \\
\hline
\end{array}$$

这里“GQA1\-(512\+64)\-S3”是按照MLA的推理形式实现的MQA，形式介乎S1与S2之间，它的主要特点是head\_dims大。

结果解读：

> 1、KV\-Shared的GQA自带Partial RoPE；
> 
> 
> 
> 2、KV\-Shared的GQA2\-256，也能超过MLA；
> 
> 
> 
> 3、VO\-RoPE的引入，似乎有利于效果（S1 ≲ S2）；
> 
> 
> 
> 4、同等KV Cache下，head\_dims越大越好；
> 
> 
> 
> 5、GQA2\-(192\+64)\-S2 略微超过 GQA1\-256\-PR；
> 
> 
> 
> 6、GQA4\-(128\+64)\-S2 的KV Cache最大，但效果不是最优，再次表明head\_dims更关键。

关于KV\-Shared，还有两点观察：

> 1、训练过程中，GQA1\-256\-PR前期是明显领先GQA2\-(192\+64)\-S2，但后期被追平甚至略微反先，猜测GQA1\-256\-PR可能有后劲不足的嫌疑；
> 
> 
> 
> 2、如果没有KV\-Shared，GQA顶多是GQA1\-256，也就是说head\_dims顶天了256，但有KV\-Shared的话，GQA可以做到GQA1\-512\-S，单纯从head\_dims看，KV\-Shared天花板更高。

### Part VI

由于没有严格对齐参数量，可能读者会有“到底是增加参数量还是增加head\_dims更本质”的疑虑，所以这里补充几个对齐参数量的实验。

这里考虑的对齐参数量的方式有三种：

> 1、<strong>double\-heads</strong>：以“GQA2\-128 vs GQA1\-256”为例，将GQA2\-128的num\_heads翻倍，可以让GQA2\-128的参数量跟GQA1\-256相同；
> 
> 
> 
> 2、<strong>缩减MLP</strong>：缩小MLP（SwiGLU）的intermediate\_size，也可以使得GQA1\-256的参数量跟GQA2\-128大致相同；
> 
> 
> 
> 3、<strong>Q&O LoRA</strong>：GQA的主要参数量来自Query和Output的投影矩阵，对这两个矩阵改用LoRA，也可以降低GQA1\-256的参数量。

实验结果如下：
$$\begin{array}{c|ccc|ccc}
\hline
& \text{Params} & \text{Loss} & \text{Cache} & \text{num_heads} & \text{intermediate_size} & \text{qo_lora} \\
\hline
\text{MLA} & 894M & 2.721 & 576 & 16 & 5456 & \text{No}\\
\hline
\text{GQA2-128} & 842M & 2.75 & 512 & 16 & 5456 & \text{No}\\
\text{GQA1-256} & 943M & 2.72 & 512 & 16 & 5456 & \text{No}\\
\hline
\text{GQA2-128} & 943M & 2.723 & 512 & \color{red}{32} & 5456 & \text{No} \\
\text{GQA1-256} & 843M & 2.747 & 512 & 16 & \color{red}{4096} & \text{No} \\
\text{GQA1-256} & 842M & 2.726 & 512 & 16 & 5456 & \color{red}{\text{Yes}} \\
\hline
\text{GQA4-(64+64)-S2} & 842M & 2.738 & 512 & 16 & 5456 & \text{No} \\
\text{GQA2-(192+64)-S2} & 943M & 2.708 & 512 & 16 & 5456 & \text{No} \\
\hline
\text{GQA4-(64+64)-S2} & 943M & 2.711 & 512 & \color{red}{32} & 5456 & \text{No} \\
\text{GQA2-(192+64)-S2} & 843M & 2.733 & 512 & 16 & \color{red}{4096} & \text{No} \\
\text{GQA2-(192+64)-S2} & 842M & 2.708 & 512 & 16 & 5456 & \color{red}{\text{Yes}} \\
\hline
\end{array}$$

结果主要分三块：

> 1、heads翻倍相比head\_dims翻倍，loss稳定差0\.003左右；
> 
> 
> 
> 2、缩小MLP比head\_dims减半，loss稳定优0\.004左右；
> 
> 
> 
> 3、Q&O LoRA性能损失最小，可以实现head\_dims翻倍但参数量不增，且loss明显降。

结论：如果从增加参数量角度看，增大head\_dims可能是效果增益较大的方向，配合Q&O LoRA可以实现参数量几乎不增，但收益仍相当。

## 小结

初步结论是：

> 1、增大head\_dims收益最大；
> 
> 
> 
> 2、Partial RoPE对Loss也有一定帮助；
> 
> 
> 
> 3、KV\-Shared应该也有一定作用。

这样看来，此前我们一直在head\_dims=128下找MLA的替代品，感觉是起点就先天不足了，难怪一直比不上MLA。要想追平MLA，head\_dims应该要192起步了，并辅以Partial RoPE。至于KV\-Shared，也可能有用，但应该还需要更大规模的验证。

## 意义

其实这里边的意义，就看我们换掉MLA的决心有多强。

假设 GQA2\-(192\+64)\-S2 可以替代MLA，但MLA也可以升到256，目前看来 GQA2\-(192\+64)\-S2 比不上 MLA\-256 。那么换掉MLA的唯二好处是：

> 1、结构更简单，可以方便加QK\-Norm；
> 
> 
> 
> 2、解码阶段的head\_dims由512\+64变成了256，同时num\_groups变为2，可以TP。
