# 【搜出来的文本】⋅（四）通过增、删、改来用词造句

> 作者：苏剑林 · 科学空间 · 2021-02-25
>
> 原文：<https://spaces.ac.cn/archives/8194>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

“用词造句”是小学阶段帮助我们理解和运用词语的一个经典任务，从自然语言处理的角度来看，它是一个句子扩写或者句子补全任务，它其实要求我们具有不定向地进行文本生成的能力。然而，当前主流的语言模型都是单方向生成的（多数是正向的，即从左往右，少数是反向的，即从右往左），但用词造句任务中所给的若干个词未必一定出现在句首或者句末，这导致无法直接用语言模型来完成造句任务。

本文我们将介绍论文[《CGMH: Constrained Sentence Generation by Metropolis\-Hastings Sampling》](<https://papers.cool/arxiv/1811.10996>)，它使用MCMC采样<u>使得单向语言模型也可以做到不定向生成</u>，通过增、删、改操作模拟了人的写作润色过程，从而能无监督地完成用词造句等多种文本生成任务。

## 问题设置

无监督地进行文本采样，那么直接可以由语言模型来完成，而我们同样要做的，是往这个采样过程中加入一些信号$\boldsymbol{c}$，使得它能生成我们期望的一些文本。在本系列第一篇文章[《【搜出来的文本】⋅（一）从文本生成到搜索采样》](<https://spaces.ac.cn/archives/8062>)的“明确目标”一节中，我们就介绍了本系列的指导思想：把我们要寻找的目标量化地写下来，然后最大化它或者从中采样。

量化的目标没有固定的格式要求，但一般来说，它大概会具有如下的形式$\setCounter{24}$：
\begin{equation}\rho(\boldsymbol{x}, \boldsymbol{c}) = p(\boldsymbol{x})\cdot \text{sim}(\boldsymbol{x}, \boldsymbol{c})\cdot \chi(\boldsymbol{x}, \boldsymbol{c})\end{equation}
其中$\boldsymbol{c}$是给定的条件，而$\boldsymbol{x}$是我们要生成的句子；$p(\boldsymbol{x})$是流畅度打分，一般由语言模型给出，此项即希望我们生成的目标是一个自然句子；$\text{sim}(\boldsymbol{x}, \boldsymbol{c})$是“软”相关性打分，它衡量了$\boldsymbol{x}$与$\boldsymbol{c}$的某种相似性；$\chi(\boldsymbol{x}, \boldsymbol{c})$是“硬”约束，通常由示性函数表示，满足某些条件时它为1，否则为0。这三项并非一定都会出现，也并非一定不超过三项，它只是设计目标的一种思路。

理论上来说，有了$\rho(\boldsymbol{x},\boldsymbol{c})$后，我们就可以构建条件概率
\begin{equation}p(\boldsymbol{x}|\boldsymbol{c}) = \frac{\rho(\boldsymbol{x},\boldsymbol{c})}{\sum\limits_{\boldsymbol{x}} \rho(\boldsymbol{x},\boldsymbol{c})}\end{equation}
然后从中条件采样。但是，直接从$p(\boldsymbol{x}|\boldsymbol{c})$进行采样通常来说是很难进行的，因此第二篇文章[《【搜出来的文本】⋅（二）从MCMC到模拟退火》](<https://spaces.ac.cn/archives/8084>)所介绍的MCMC方法就派上用场了，它允许我们只知道$\rho(\boldsymbol{x},\boldsymbol{c})$的情况下，实现从$p(\boldsymbol{x}|\boldsymbol{c})$的采样。

## 增、删、改

我们主要用到MCMC方法中的MH采样，而构建MH采样需要一个先验的转移概率$q(\boldsymbol{x}\leftarrow\boldsymbol{y})$。所谓转移概率，就是将一个样本修改为另一个样本的概率，理论上来说，几乎任意的转移概率都是可以接受的，但是从[《【搜出来的文本】⋅（二）从MCMC到模拟退火》](<https://spaces.ac.cn/archives/8084>)中的“分析思考”一节我们得知，MH采样的有效性在于我们每一步的转移概率只对当前样本进行“微调”而不是“大改”，因此这里的转移概率的设计也要遵循这样的思想。

本系列的“样本”都是指要生成的句子，我们可以认为句子的基本单位是“词”，因此对句子进行微调的操作，我们可以设计为每次对单个词进行操作，具体包含增（insert）、删（delete）、改（replace）三种：

> 增（insert）：在句子的某个位置插入一个新词；
> 
> 
> 
> 删（delete）：删除句子的某个词；
> 
> 
> 
> 改（replace）：将句子的某个词替换为另一个词。

要注意，这三个操作是存在互逆关系的，增与删互为逆操作，改与改自身也互为逆操作。包含逆操作是必须的，因为接受率的计算公式是$\mathcal{A}(\boldsymbol{y}\leftarrow\boldsymbol{x}) = \min\left(1, \frac{q(\boldsymbol{x}\leftarrow\boldsymbol{y})p(\boldsymbol{y})}{q(\boldsymbol{y}\leftarrow\boldsymbol{x})p(\boldsymbol{x})}\right)$，需要同时$q(\boldsymbol{x}\leftarrow\boldsymbol{y}) > 0$和$q(\boldsymbol{y}\leftarrow\boldsymbol{x}) > 0$，即从原句子变换到新句子的概率以及新句子变换为原句子的概率都不为0，才能比较合理地计算接受率，所以需要定义每个操作及其逆操作的转移概率。

从数学角度来看，这也是转移概率具有唯一平稳分布的要求，在[《【搜出来的文本】⋅（二）从MCMC到模拟退火》](<https://spaces.ac.cn/archives/8084>)我们已经说了，具有唯一平稳分布的条件是任意两个状态都是连通的，这意味着从任意一个句子出发可以慢慢变换到另外一个任意句子，显然如果没有逆操作是无法做到的。

## 转移概率

现在的微调操作有增、删、改三种，我们首先需要定义这三种操作出现的概率$p_{\text{insert}},p_{\text{delete}},p_{\text{replace}}$，由于适当定义的MCMC方法最终都会收敛到同一个平稳分布，因此其实三者的具体大小对最终结果的影响不大，简单起见我们可以直接让$p_{\text{insert}} = p_{\text{delete}} = p_{\text{replace}} = 1/3$。接着我们逐步定义每一种操作的转移概率。

首先要定义的是“改”的转移概率。假设当前句子为$\boldsymbol{x}=(x_1, x_2, \cdots, x_l)$，我们从$1,2,\cdots,l$随机选取一个位置$i$，然后从概率分布
\begin{equation}p(y|\boldsymbol{x}_{-i})=\frac{p(x_1,\dots,x_{i-1},y,x_{i+1},\cdots,x_l)}{\sum\limits_y p(x_1,\dots,x_{i-1},y,x_{i+1},\cdots,x_l)}\end{equation}
选一个新词$y$，然后令$\boldsymbol{y}=(x_1,\dots,x_{i-1},y,x_{i+1},\cdots,x_l)$作为微调后的句子，其中$p(x_1, x_2, \cdots, x_l)$是语言模型给出的句子概率。这就是“改”的转移概率$q_{\text{replace}}(\boldsymbol{y}\leftarrow\boldsymbol{x})$。

看过[《【搜出来的文本】⋅（三）基于BERT的文本采样》](<https://spaces.ac.cn/archives/8119>)的读者都能反应过来，其实$p(y|\boldsymbol{x}_{-i})$就是BERT的MLM模型，但是CGMH是BERT发布之前的工作，因此当时应该还没有MLM的概念，有的只是单向的语言模型。如果用单向的语言模型按上述公式计算$p(y|\boldsymbol{x}_{-i})$的话，那就意味着我们要计算$|V|$个句子的概率（将第$i$个位置遍历词表的所有词），计算量比较大。为此，作者想了个办法，先用语言模型过滤掉概率比较低的部分，具体来说，它用正向语言模型$\stackrel{\rightarrow}{p}(y|x_1,x_2,\cdots,x_{i-1})$和反向语言模型$\stackrel{\leftarrow}{p}(y|x_l,x_{l-1},\cdots,x_{i+1})$分别预测第$i$个位置的词分布，然后用
\begin{equation}\min\big(\stackrel{\rightarrow}{p}(y|x_1,x_2,\cdots,x_{i-1}),\, \stackrel{\leftarrow}{p}(y|x_l,x_{l-1},\cdots,x_{i+1})\big)\end{equation}
为指标，只保留指标最大的$K$个词作为候选集，这样就只需要算$K$个句子的概率，从而降低了计算量了。同样，这个指标没什么硬性要求，比如很多时候我们我们只有正向的语言模型，这时候直接用$\stackrel{\rightarrow}{p}(y|x_1,x_2,\cdots,x_{i-1})$作为指标筛选也无妨。

有了$q_{\text{replace}}(\boldsymbol{y}\leftarrow\boldsymbol{x})$，定义“增”操作的$q_{\text{insert}}(\boldsymbol{y}\leftarrow\boldsymbol{x})$也容易了，它可以作为“改”的变式。设$\boldsymbol{x}=(x_1, x_2, \cdots, x_l)$，从$1,2,\cdots,l,l+1$随机选取一个位置$i$，在$x_{i-1},x_i$之间插入一个新词$\tilde{x}$（任意的）得到$\tilde{\boldsymbol{x}} = (x_1,\dots,x_{i-1},\tilde{x},x_{i},\cdots,x_l)$，然后从$\tilde{\boldsymbol{x}}$出发、$i$为采样位置来计算$q_{\text{replace}}(\boldsymbol{y}\leftarrow\tilde{\boldsymbol{x}})$作为$q_{\text{insert}}(\boldsymbol{y}\leftarrow\boldsymbol{x})$就行了。

最后剩下的“删”操作就最简单了，设$\boldsymbol{x}=(x_1, x_2, \cdots, x_l)$，从$1,2,\cdots,l$随机选取一个位置$i$，那么删除该位置的词得到$\boldsymbol{y} = (x_1,\dots,x_{i-1},x_{i+1},\cdots,x_l)$，然后直接令$q_{\text{delete}}(\boldsymbol{y}\leftarrow\boldsymbol{x})=1$即可。

## 接受概率

前面我们定义了转移概率，本质上它就是我们每一步对句子进行微调的动作空间。上述定义的转移概率是跟任务背景无关的，也就是转移概率不依赖于条件信息$\boldsymbol{c}$，只依赖于一个无监督的语言模型。具体的任务信息包含在定量指标$\rho(\boldsymbol{x},\boldsymbol{c})$的定义中，它通过影响接受率来影响最终结果：
\begin{equation}\begin{aligned}
\mathcal{A}_{\text{replace}}(\boldsymbol{y}\leftarrow\boldsymbol{x}) =&\, \frac{p_{\text{replace}} \cdot \rho(\boldsymbol{y},\boldsymbol{c}) \cdot q_{\text{replace}}(\boldsymbol{x}\leftarrow\boldsymbol{y})}{p_{\text{replace}} \cdot \rho(\boldsymbol{x},\boldsymbol{c}) \cdot q_{\text{replace}}(\boldsymbol{y}\leftarrow\boldsymbol{x})} = \frac{\rho(\boldsymbol{y},\boldsymbol{c}) \cdot q_{\text{replace}}(\boldsymbol{x}\leftarrow\boldsymbol{y})}{\rho(\boldsymbol{x},\boldsymbol{c}) \cdot q_{\text{replace}}(\boldsymbol{y}\leftarrow\boldsymbol{x})} \\[10pt]
\mathcal{A}_{\text{insert}}(\boldsymbol{y}\leftarrow\boldsymbol{x}) =&\, \frac{p_{\text{delete}} \cdot \rho(\boldsymbol{y},\boldsymbol{c}) \cdot q_{\text{delete}}(\boldsymbol{x}\leftarrow\boldsymbol{y})}{p_{\text{insert}} \cdot \rho(\boldsymbol{x},\boldsymbol{c}) \cdot q_{\text{insert}}(\boldsymbol{y}\leftarrow\boldsymbol{x})} = \frac{\rho(\boldsymbol{y},\boldsymbol{c})}{\rho(\boldsymbol{x},\boldsymbol{c}) \cdot q_{\text{insert}}(\boldsymbol{y}\leftarrow\boldsymbol{x})} \\[10pt]
\mathcal{A}_{\text{delete}}(\boldsymbol{y}\leftarrow\boldsymbol{x}) =&\, \frac{p_{\text{insert}} \cdot \rho(\boldsymbol{y},\boldsymbol{c}) \cdot q_{\text{insert}}(\boldsymbol{x}\leftarrow\boldsymbol{y})}{p_{\text{delete}} \cdot \rho(\boldsymbol{x},\boldsymbol{c}) \cdot q_{\text{delete}}(\boldsymbol{y}\leftarrow\boldsymbol{x})} = \frac{\rho(\boldsymbol{y},\boldsymbol{c}) \cdot q_{\text{insert}}(\boldsymbol{x}\leftarrow\boldsymbol{y})}{\rho(\boldsymbol{x},\boldsymbol{c})}
\end{aligned}\end{equation}
对了，上述接受率还少了个$\min(1,\alpha)$的运算，但如果我们的随机数是从$U[0,1]$中采样的，那么加不加这个运算都不影响结果。简单来说，上述接受率就意味着如果微调操作使得$\rho(\boldsymbol{x},\boldsymbol{c})$上升，那么就很有可能被接受，不过就算使得$\rho(\boldsymbol{x},\boldsymbol{c})$下降，那么也有一定的几率被接受，不管是随机采样还是求最大值（模拟退火），这个随机性非常重要，它是跳出局部极值点的关键所在。

以“增”、“删”、“改”为操作的转移概率，结合MH采样的接受率，其实就是对我们写作过程中的反复润色进行模拟了：先改一下，然后看看要不要保留这个修改，好的就尽量保留，差的就尽量不保留。

## 确定目标

作为准备工作的最后一步，我们继续来讨论一下指标$\rho(\boldsymbol{x},\boldsymbol{c})$要如何确定的问题。CGMH做了三种实验：用词造句、无监督句子改写、无监督句子纠错，它们的采样过程都是一样的，区别就在于$\rho(\boldsymbol{x},\boldsymbol{c})$。对于用词造句来说，约束条件是硬约束，即
\begin{equation}\rho(\boldsymbol{x}, \boldsymbol{c}) = p(\boldsymbol{x})\cdot \chi(\boldsymbol{x}, \boldsymbol{c})\end{equation}
其中$\boldsymbol{c}$是所给定的词集，而$\chi(\boldsymbol{x}, \boldsymbol{c})$则只有当句子$\boldsymbol{x}$包含$\boldsymbol{c}$中所有词时才为1，否则为0，在实现的时候，我们可以实现记下这些词所在的位置，在执行“改”和“删”操作时，都不要选到这些位置就行了。如果是无监督句子改写和无监督句子纠错，则$\boldsymbol{c}$是输入句子，$\chi(\boldsymbol{x}, \boldsymbol{c})$替换成某种相似度指标$\text{sim}(\boldsymbol{x}, \boldsymbol{c})$，具体设置请阅读原论文。

这里值得讨论的还有$p(\boldsymbol{x})$的选择，它代表句子的流畅性，通常就是语言模型：
\begin{equation}p(\boldsymbol{x}) = \prod_{t=1}^l p(x_t|\boldsymbol{x}_{< t})\end{equation}
然而，直接使用语言模型会倾向于生成很短的句子，因为通常来说句子越长概率越小。如果要鼓励生成长一点的句子，建议加个幂调整变为$p^{\gamma}(\boldsymbol{x})$，其中$\gamma$是一个稍小于1的数。

## 实验效果

一切准备就绪，就可以进行实验了。这里演示的是用词造句任务，为了跟CGMH介绍的方法尽量一致，这里只用了一个正向语言模型，没有用MLM模型。语言模型用的是华为开源的中文GPT base模型（参考[这里](<https://github.com/bojone/bert4keras/blob/master/examples/basic_language_model_nezha_gen_gpt.py>)）。参考代码分享在

> <strong>Github链接：[https://github\.com/bojone/unsupervised\-text\-generation](<https://github.com/bojone/unsupervised-text-generation/blob/main/gpt_cgmh.py>)</strong>

效果演示：

> <strong>输入词语：广州、美食、开幕</strong>
> 开始状态：广州美食开幕。
> 第0步，执行insert, 输出：广州美食节开幕。
> 第4步，执行insert, 输出：广州美食节开幕式。
> 第11步，执行insert, 输出：广州美食节开幕月式。
> 第13步，执行delete, 输出：广州美食节开幕式。
> 第14步，执行replace, 输出：广州美食节开幕了。
> 第20步，执行delete, 输出：广州美食节开幕。
> 第50步，执行replace, 输出：广州美食网开幕。
> 第52步，执行insert, 输出：广州美食网开幕式。
> 第54步，执行replace, 输出：广州美食节开幕式。
> 第55步，执行delete, 输出：广州美食节开幕。
> 第63步，执行insert, 输出：广州美食节昨开幕。
> 第70步，执行delete, 输出：广州美食节开幕。
> 第74步，执行insert, 输出：广州美食节开幕了。
> 第76步，执行delete, 输出：广州美食节开幕。
> 第84步，执行insert, 输出：广州美食节将开幕。
> 第90步，执行replace, 输出：广州美食节的开幕。
> 第93步，执行insert, 输出：广州美食节的开幕式。
> 第96步，执行delete, 输出：广州美食节开幕式。
> 第99步，执行replace, 输出：广州美食节开幕了。
> 第100步，执行insert, 输出：广州的美食节开幕了。
> 第101步，执行delete, 输出：广州美食节开幕了。
> 第105步，执行insert, 输出：广州美食节也开幕了。
> 第106步，执行insert, 输出：广州的美食节也开幕了。
> 第107步，执行insert, 输出：广州人的美食节也开幕了。
> 第108步，执行replace, 输出：广州市的美食节也开幕了。
> 第123步，执行delete, 输出：广州的美食节也开幕了。
> 第127步，执行replace, 输出：广州的美食节又开幕了。
> 
> 
> 
> <strong>输入词语：科学、空间</strong>
> 开始状态：科学空间。
> 第3步，执行insert, 输出：科学，空间。
> 第5步，执行delete, 输出：科学空间。
> 第9步，执行insert, 输出：科学是空间。
> 第11步，执行insert, 输出：科学是指空间。
> 第12步，执行delete, 输出：科学指空间。
> 第15步，执行delete, 输出：科学空间。
> 第25步，执行insert, 输出：科学空间战。
> 第26步，执行delete, 输出：科学空间。
> 第28步，执行insert, 输出：科学是空间。
> 第29步，执行delete, 输出：科学空间。
> 第32步，执行insert, 输出：科学空间观。
> 第34步，执行delete, 输出：科学空间。
> 第42步，执行insert, 输出：科学，空间。
> 第43步，执行replace, 输出：科学与空间。
> 第44步，执行delete, 输出：科学空间。
> 第50步，执行insert, 输出：科学是空间。
> 第55步，执行delete, 输出：科学空间。
> 第63步，执行insert, 输出：科学有空间。
> 第65步，执行delete, 输出：科学空间。
> 第67步，执行insert, 输出：科学是空间。
> 第68步，执行replace, 输出：科学管空间。
> 第69步，执行insert, 输出：科学管理空间。
> 第70步，执行insert, 输出：科学管理空间大。
> 第71步，执行insert, 输出：科学的管理空间大。
> 第73步，执行delete, 输出：科学管理空间大。
> 第75步，执行delete, 输出：科学管理空间。
> 第78步，执行replace, 输出：科学管控空间。
> 第84步，执行replace, 输出：科学调控空间。
> 第88步，执行insert, 输出：科学调控空间大。
> 第89步，执行insert, 输出：科学调控的空间大。
> 第90步，执行insert, 输出：科学调控的空间很大。
> 第94步，执行replace, 输出：科学调控的空间加大。
> 第104步，执行insert, 输出：科学调控的空间在加大。
> 第110步，执行replace, 输出：科学调控的空间将加大。
> 第125步，执行insert, 输出：科学调控的空间将更加大。
> 
> 
> 
> <strong>输入词语：好吃、零食</strong>
> 开始状态：好吃零食。
> 第4步，执行insert, 输出：好吃的零食。
> 第5步，执行delete, 输出：好吃零食。
> 第6步，执行insert, 输出：好吃零食卖。
> 第7步，执行delete, 输出：好吃零食。
> 第18步，执行insert, 输出：我好吃零食。
> 第20步，执行replace, 输出：最好吃零食。
> 第23步，执行replace, 输出：我好吃零食。
> 第24步，执行replace, 输出：最好吃零食。
> 第26步，执行replace, 输出：有好吃零食。
> 第27步，执行delete, 输出：好吃零食。
> 第29步，执行insert, 输出：好吃的零食。
> 第30步，执行delete, 输出：好吃零食。
> 第32步，执行insert, 输出：好吃的零食。
> 第33步，执行delete, 输出：好吃零食。
> 第35步，执行insert, 输出：好吃的零食。
> 第37步，执行replace, 输出：好吃，零食。
> 第38步，执行replace, 输出：好吃的零食。
> 第40步，执行delete, 输出：好吃零食。
> 第53步，执行insert, 输出：最好吃零食。
> 第54步，执行replace, 输出：我好吃零食。
> 第58步，执行insert, 输出：我好吃零食嘛。
> 第59步，执行replace, 输出：我好吃零食！。
> 第60步，执行delete, 输出：我好吃零食。
> 第61步，执行delete, 输出：好吃零食。
> 第63步，执行insert, 输出：好吃, 零食。
> 第64步，执行replace, 输出：好吃的零食。
> 第66步，执行delete, 输出：好吃零食。
> 第75步，执行insert, 输出：好吃的零食。
> 第76步，执行delete, 输出：好吃零食。
> 第77步，执行insert, 输出：好吃的零食。
> 第81步，执行delete, 输出：好吃零食。
> 第84步，执行insert, 输出：好吃的零食。
> 第89步，执行replace, 输出：好吃，零食。
> 第90步，执行insert, 输出：好吃的，零食。
> 第92步，执行replace, 输出：好吃的新零食。
> 第93步，执行delete, 输出：好吃的零食。
> 第94步，执行delete, 输出：好吃零食。
> 第96步，执行insert, 输出：好吃，零食。
> 第98步，执行replace, 输出：好吃的零食。
> 第99步，执行delete, 输出：好吃零食。
> 第104步，执行insert, 输出：最好吃零食。
> 第105步，执行insert, 输出：最好吃的零食。
> 第106步，执行delete, 输出：最好吃零食。
> 第107步，执行delete, 输出：好吃零食。
> 第124步，执行insert, 输出：最好吃零食。
> 第125步，执行replace, 输出：我好吃零食。
> 第126步，执行insert, 输出：但我好吃零食。
> 第127步，执行replace, 输出：而我好吃零食。

这里每个样本都执行了128步迭代，有些没有对句子进行改动的中间步骤就没有显示出来了，当然了，由于随机性的存在，重复跑同样的输入通常会得到不同的句子，因为MCMC本质上也是随机采样方案。通过这些例子可以看出，这确实是对句子的反复微调的过程，最终也能得到相对可读的、满足条件的句子，这显示了CGMH方法的有效性。

理论上来说，只要我们能写下我们要寻求的目标$\rho(\boldsymbol{x},\boldsymbol{c})$，那么我们就可以套用上述整个过程，从而无监督地完成相应的文本生成任务（但是可能要迭代足够多的步数，而不是像用词造句那样，迭代百来步就显示出效果了）。

## 本文小结

本文介绍并简单复现了一种名为“CGMH”的利用MCMC进行无监督有约束文本生成的方法，可以说它模拟了人写作过程中的增、删、改操作，使得生成过程颇具解释性，是本系列文章中第一个真正意义上有实用价值的例子。原则上来说，CGMH已经包含了通过离散优化进行文本生成的一般过程，只要对应的文本生成任务能够写出不依赖标签的量化评价指标，都可以尝试套用该方法进行无监督生成。
