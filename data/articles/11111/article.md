# Transformer升级之路：21、MLA好在哪里?（下）

> 作者：苏剑林 · 科学空间 · 2025-07-10
>
> 原文：<https://spaces.ac.cn/archives/11111>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在文章[《Transformer升级之路：20、MLA好在哪里?（上）》](<https://spaces.ac.cn/archives/10907>)中，我们对[MLA](<https://spaces.ac.cn/archives/10091>)相比常见MHA、GQA、MQA的一些变化分别做了消融实验，其中的变化包括“增大head\_dims”、“Partial RoPE”和“KV共享”，实验的初步结果是这三个变化很可能都是MLA效果优异的原因。

本文我们将从一个更加偏理论的角度出发，来理解MLA的成功之处。

## 部分旋转

首先，我们把最终的断言放在前面：

> 在相同训练成本和推理成本下，MLA可能是效果最好的Full Attention变体。

很明显，这个判断把MLA摆在了非常高的位置。这是在比较理想和简化的假设下，根据上一篇文章的实验结果以及本文接下来的理论分析所得的结论。由于实际的训练和推理存在诸多复杂的因素，所以该结论大概率会有所偏差，但我们至少可以得出，MLA应该是走在了正确的改进方向上。

MLA之所以能够表现出色，有一个非常大的前提，那就是部分旋转的[Partial RoPE](<https://spaces.ac.cn/archives/10122>)效果不逊色于甚至可能优于完全体的RoPE。这里的Partial RoPE可以有两种含义：一是我们对Attention的$\boldsymbol{Q}$、$\boldsymbol{K}$加RoPE时，可以只对小部份维度加，剩下的维度保持不变；二是我们可以考虑层间RoPE与NoPE交替出现，并且NoPE的层可以占多数。

说白了，RoPE可以只加“一点点”，但不能不加，完全不加的话效果不行。如果需要理论，笔者比较认同[《Transformer升级之路：18、RoPE的底数选择原则》](<https://spaces.ac.cn/archives/10122>)的解释，大致意思是Partial RoPE使得检索结果更兼顾位置与语义。此外，像[FoX](<https://papers.cool/arxiv/2503.02130>)、[SBA](<https://papers.cool/arxiv/2410.17980>)等新工作也体现出一定潜力，但对于MLA来说，这些变体就相当于NoPE，因此不改变结论。

“Partial RoPE效果不差”的结论，允许我们把Attention的主要计算复杂度放到NoPE部分上，这提供了更大的腾挪空间，MLA便是得益于此。

## 键值共享

Full Attention的变化大致上是从[MHA](<https://spaces.ac.cn/archives/4765>)、[MQA](<https://papers.cool/arxiv/1911.02150>)、[GQA](<https://papers.cool/arxiv/2305.13245>)然后到MLA，虽然MQA可以看作是GQA的特例，但按时间顺序来说确实是GQA在后。在MLA之后，还出现了[MFA](<https://papers.cool/arxiv/2412.19255>)、[TPA](<https://papers.cool/arxiv/2501.06425>)两个变体。这些变体本质上都是在尽量保持效果的前提下，尽可能压榨KV Cache以提高生成速度。

简单来说，Attention模型的复杂度可以分<u>训练、Prefill和Decoding</u>三部分，其中训练和Prefill是相似的，所以本质上是Prefill和Decoding两部分。Prefill是指模型处理输入、直至吐出第一个token的阶段，这部分我们下节再谈；Decoding是指Token by Token的生成阶段，它可以通过KV Cache机制来加速，但同时也导致了KV Cache大小几乎是Decoding速度的唯一瓶颈。

所以，压缩KV Cache就是提高Decoding速度。现在问大家一个问题：<strong>在NoPE背景下，给定KV Cache大小后，效果最好的Attention是什么呢？</strong>如果不考虑参数量差异，只在单层MHA/GQA/MQA内讨论（TPA和MFA我们后面再补充讨论），那么答案将会是：

> <strong>一个head\_dims等于KV Cache大小、K和V共享的MQA。</strong>

看上去是不是让人意外？其实不难理解。因为MHA、MQA都可以看成是GQA的一个特例，所以我们只需要分析GQA，我们在[《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》](<https://spaces.ac.cn/archives/10091>)已经给出了，GQA可以重新表示成一个K、V拼接起来的模型：
\begin{equation}\underbrace{\left[\boldsymbol{k}_i^{(1)},\cdots,\boldsymbol{k}_i^{(g)},\boldsymbol{v}_i^{(1)},\cdots,\boldsymbol{v}_i^{(g)}\right]}_{\boldsymbol{c}_i\in\mathbb{R}^{g(d_k+d_v)}} = \boldsymbol{x}_i \underbrace{\left[\boldsymbol{W}_k^{(1)},\cdots,\boldsymbol{W}_k^{(g)},\boldsymbol{W}_v^{(1)},\cdots,\boldsymbol{W}_v^{(g)}\right]}_{\boldsymbol{W}_c\in\mathbb{R}^{d\times g(d_k+d_v)}}\end{equation}
这里$g(d_k+d_v)$正是单个Token的KV Cache总大小。接着我们算Attention的时候，$\boldsymbol{c}$到$\boldsymbol{k},\boldsymbol{v}$的变换分别吸收到$\boldsymbol{W}_q$和$\boldsymbol{W}_o$那边去，那么就得到了一个K、V都是$\boldsymbol{c}$的MQA。所以说，“head\_dims等于KV Cache大小、K和V共享的MQA”，实际上是给定KV Cache大小后MHA/GQA/MQA的“超集”，那么它自然是理论上效果最好的选择。

## 双重投影

综上所述，如果我们想要在相同Decoding速度下效果最优，那么应该训练一个指定head\_dims的、KV共享的MQA，比如约定KV Cache不超过512，那么head\_dims=512的、KV共享的MQA就是最佳选择。事实上，MLA在Decoding阶段正是KV共享的MQA（NoPE部分），这就是它走在正确方向上的体现之一。

然而，将head\_dims升到512，Decoding是没问题，但训练和Prefill都很难接受，因为它们俩的瓶颈是计算，而影响计算速度的主要因素是num\_heads和head\_dims。为了保证效果，num\_heads变动的空间不大，因此head\_dims大小可以说是计算量的唯一指标，head\_dims升到512意味着计算量要增加到原来的4倍（相比head\_dims=128）。

现在再来问大家一个问题：<strong>同样在NoPE背景下，给定num\_heads和head\_dims后，效果最好的Attention是什么呢？</strong>这个问题的答案我相信大家都能接受，那就是<strong>MHA</strong>，因为它限制最少。所以，单从训练和Prefill成本来看，我们希望的是训练一个head\_dims=128的MHA。

怎么调和Prefill与Decoding这两个不同的期望呢？这就是MLA的“大招”了，它通过两步投影得到K、V：先将输入投影到单个512维的向量，然后将该向量投影到多个128维的向量，然后利用“Attention \+ NoPE”固有的恒等变换性质，可以让模型在MHA\-128和MQA\-512间自由切换。

$$\require{cancel}\begin{array}{c|c}
\text{训练/Prefill} & \text{Decoding} \\
\\
\begin{gathered}
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}, \boldsymbol{o}_t^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\right] \\[10pt]
\boldsymbol{o}_t^{(s)} = \frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)\boldsymbol{v}_i^{(s)}}{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{(s)}{}^{\top}\right)} \\[15pt]
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_q^{(s)}\in\mathbb{R}^{d\times d_k}\\
\boldsymbol{k}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_k},\quad \boldsymbol{W}_k^{(s)}\in\mathbb{R}^{d_c\times d_k} \\
\boldsymbol{v}_i^{(s)} = \boldsymbol{c}_i\boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_v},\quad \boldsymbol{W}_v^{(s)}\in\mathbb{R}^{d_c\times d_v} \\[10pt]
\boldsymbol{c}_i = \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c},\quad \boldsymbol{W}_c\in\mathbb{R}^{d\times d_c}
\end{gathered}
&
\begin{gathered}
\boldsymbol{o}_t = \left[\boldsymbol{o}_t^{(1)}\boldsymbol{W}_v^{(1)}, \boldsymbol{o}_t^{(2)}\boldsymbol{W}_v^{(2)}, \cdots, \boldsymbol{o}_t^{(h)}\boldsymbol{W}_v^{(h)}\right] \\[10pt]
\boldsymbol{o}_t^{(s)} = \frac{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)\boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} }{\sum_{i\leq t}\exp\left(\boldsymbol{q}_t^{(s)} \boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}}{}^{\top}\right)} \\[15pt]
\boldsymbol{q}_i^{(s)} = \boldsymbol{x}_i\boldsymbol{W}_q^{(s)}\boldsymbol{W}_k^{(s)}{}^{\top}\in\mathbb{R}^{d_c}\\
\boldsymbol{k}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{v}_i^{\color{#ccc}{\smash{\bcancel{(s)}}}} = \boldsymbol{c}_i= \boldsymbol{x}_i \boldsymbol{W}_c\in\mathbb{R}^{d_c}
\end{gathered}
\end{array}$$

## 总而言之

我们将前面的推理逻辑做个总结：

> 1、大前提：Partial RoPE的效果不差于甚至可能优于RoPE，这使得我们可以把主要精力放在NoPE上；
> 
> 
> 
> 2、Decoding主要瓶颈是KV Cache，理论效果最优的模型是head\_dims=KV Cache、KV共享的MQA；
> 
> 
> 
> 3、训练和Prefill的主要瓶颈都是head\_dims，理论效果最优的模型是head\_dims为期望值的MHA；
> 
> 
> 
> 4、在NoPE前提下，Attention具有恒等变换性质，可以通过LoRA来尽可能地兼顾两个理想方向，这正好是MLA所做的。

剩下的，就是给K拼接一个共享的低维RoPE，以最小的成本给MLA补充上位置信息，同时还“一箭双雕”：拼接RoPE的做法暗合了“Partial RoPE”，同时也增加了head\_dims，这跟上一篇文章的结论相符。换句话说，有意或者无意之中使用了Partial RoPE和增加了head\_dims，是MLA在极致压缩之下还能媲美MHA的主要原因。

从MQA的角度看，MLA是给Q加了rank=128的LoRA；从MHA的角度看，MLA是给K、V加了rank=512的LoRA。可以说，MLA是一场NoPE结合LoRA、MHA结合MQA的极致“魔术秀”，成功实现了Prefill和Decoding的“双向奔赴”。

当然，上述思考过程肯定有一些过于简化的地方。比如，实际的训练和推理还有诸多细节因素，笼统地归结为head\_dims和KV Cache是不完全准确的，例如MQA在Decoding阶段无法TP（张量并行），这可能会带来新的效率问题；还有，分析过程中我们也没有特别注重参数量的对齐，比如在head\_dims=128时我们也可以考虑增加Q、K、V的投影复杂度来提高性能，而不一定要增大head\_dims；等等。

总之，上下两篇文章旨在提供一些实验和思考，来论证MLA在一定范围内的最优性。当然，MLA是DeepSeek首先提出的，第三方使用MLA总会给人一种复制DeepSeek的感觉，但在更好的变体出现之前，或者在发现严重的缺陷之前，MLA始终是一个相当有竞争力的选择，如果单纯是为了显示自己不“追随”DeepSeek而不用MLA，那是一个相当不明智的选择。

举个例子，现在Linear Attention和Softmax Attention的混合模型也体现出极大的竞争力，但如果我们将Linear Attention跟LLAMA使用的GQA8\-128按3:1混合，那么KV Cache大致上降低到GQA8\-128的1/4，然而MLA本身就能将KV Cache降低到GQA8\-128的1/4了。

## 补充讨论

前面我们都在围绕MHA、GQA、MQA和MLA讨论，这一节我们来简单聊聊两个比较少谈及的Attention变体：TPA和MFA。

TPA全称是Tensor Product Attention，作者给它安了个Tensor Product的名字，显得比较“唬人”，实际上它是一个介乎GQA和MLA的中间产物。我们以目标KV Cache=512为例，TPA先投影得到一个512维向量，然后reshape为(4, 128)，然后分成两个(2,128)分别代表K Cache和V Cache。到目前为止，TPA的做法都跟GQA2\-128一致。

接下来，TPA借鉴了MLA的思想，将(2,128)的K/V重新投影成Multi\-Head，但它不是像MLA那样整个向量投影，而是沿着“2”所在的维度投影，说白了就是将2个128维向量做head\_dims次不同的线性组合。显然，这样TPA的上限是不如MLA直接从整个512维向量出发来投影的。为了缓解这个问题，TPA又引入了data\-dependent的组合系数来增强K、V的表达能力，即便如此，笔者还是认为它上限不如MLA。

为什么TPA要这样设计呢？大体上是为了兼容RoPE，这也是它相比MLA的最大“优点”。然而，这里的“优点”是要加个双引号的，因为在Partial RoPE不逊色甚至还可能更优的背景下，兼容RoPE就有点啼笑皆非的感觉了。还有，TPA这样设计，堵死了它升head\_dims的空间，比如head\_dims想要升到256，那么K Cache、V Cache就只是(1,256)形状了，单个向量没有线性组合的自由度。

再来看MFA，它全称是“Multi\-matrix Factorization Attention”，这个名字看上去也有点“唬人”，它实际上就是一个带有Q\-LoRA的、head\_dims=256的MQA。看到这个配置，是不是有点熟悉？因为这配置跟我们上一篇文章的结论完全吻合——增大head\_dims到256来提升MQA的效果，并且KV Cache跟MLA接近，同时通过Q\-LoRA来控制参数量。

所以，MFA能“打”MLA，笔者并不意外，上一篇文章我们也实验过差不多的做法了。此外，上一篇文章我们还提出另外两个提升MQA效果的方向，一个是本文已经多次提及的Partial RoPE，另一个是通过[QKVO\-RoPE](<https://spaces.ac.cn/archives/10862>)的方式实现完全的KV共享，让MQA变成GQA2\-256，这两点叠加上去，MFA应该还能再涨一点。

## 文章小结

本文在上一篇文章的实验结果基础上，给出一个偏理论的思考过程，以论证MLA在一定范围内的最优性。总的来说，在Partial RoPE的背景下，MLA似乎是一个非常难以超越的Attention变体。
