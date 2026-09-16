# 纠缠的时空（一）：洛仑兹变换的矩阵

> 作者：苏剑林 · 科学空间 · 2013-02-01
>
> 原文：<https://spaces.ac.cn/archives/1889>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

我现在是越来越佩服爱因斯坦了，他的相对论是他天才的思想的充分体现。只有当相对论提出之后，宏观物理的大多数现象和规律才得到了统一的描述。狭义相对论中爱因斯坦对我们速度叠加常识的否定已经显示了他莫大的勇气，而一项头脑风暴性的工作——广义相对论则将他惊人的创造力体现得完美无瑕。我是被量子力学的数学吸引的，于相对论则是被相对论美妙的逻辑体系吸引。当然，其中也有相当美妙的数学。

狭义相对论中的核心内容之一就是被称为洛仑兹变换的东西，这在相对论发表之前已经由洛仑兹推导出来了，只不过他不承认他的物理意义，也就没有就此进行一次物理革命，革命的任务则由爱因斯坦完成。很久前我就已经看过洛仑兹变换的推导，那是直接设一种线性关系来求解的。但是我总感觉那样的推导不够清晰（也许是我的理解方式有问题吧），而且没有说明狭义相对论的两条原理如何体现出现。所以在研究过矩阵之后，我就尝试用矩阵来推导洛仑兹变换，发现效果挺好的，而且我觉得能够体现出相对论中的对称性。

### 两条原理

> 1、<strong>狭义相对性原理：</strong>在所有惯性系中，物理定律有相同的表达形式。这是力学相对性原理的推广，它适用于一切物理定律，其本质是所有惯性系平权。
> 
> 
> 
> 2、<strong>光速不变原理：</strong>所有惯性系中，真空中的光速都等于c=299 792 458 m/s，与光源运动无关。迈克耳孙\-莫雷实验是其有力证明。

### 推导过程

下面将从这两条原理出发，通过矩阵来推导洛仑兹变换。<strong>同样，为了说清楚过程，我们只考虑只有$x,t$两个变量的二维时空。方便起见，设光速$c=1$，而$x'\text{-}t'$惯性系以速度$v$远离$x\text{-}t$惯性系。</strong>

#### 狭义相对性原理

我们要理解的是，每一条原理究竟告诉了我们什么。首先是<strong>狭义相对性原理</strong>，它说一切惯性系都是平权的，也就是说，从一个惯性系到另外一个惯性系的变换并没有改变其本质，这样来说，洛仑兹变换只能够是线性变换。当然，这不算证明，证明需要用到更多的数学知识，但是我们认为，这样理解就够了。<u>即线性变换是没有改变其本质的变换</u>。

所有的线性变换都可以用矩阵来描述，所以我们可以设：
\begin{equation}\left[\begin{array}{c} x\\t \end{array}\right]=\left[\begin{array}{c c}a & c\\ b & d \end{array}\right]\left[\begin{array}{c}x'\\t' \end{array}\right] \end{equation}

也就是说，在普通的直角坐标系中的$[x,t]$向量，变成了$\left[\begin{array}{c c}a & c\\ b & d \end{array}\right]$中的$[x',t']$向量。普通的直角坐标系就是我们所在的时空（姑且称为“静时空”），而$\left[\begin{array}{c c}a & c\\ b & d \end{array}\right]$就是运动者所在的时空（动时空）。我们在自己的时空测量得到$[x,t]$，他们则在自己的时空测量得到$[x',t']$。<u>不管是哪个时空，它们的坐标轴都是直的，并没有谁变成了曲线坐标系，这就是“没有改变其本质”的几何描述</u>。

#### 光速不变原理

下面的关键是<strong>光速不变原理</strong>了。光速不变原理告诉我们在动、静时空的测量到同一束光的光速都是一样的。因此，在动时空测量到的速度为$c=1$的点$\left[\begin{array}{c} 1\\1 \end{array}\right]$，在静时空中也测量到速度为1，即$\left[\begin{array}{c} \lambda_1\\ \lambda_1 \end{array}\right]$。即
\begin{equation}\lambda_1 \left[\begin{array}{c} 1\\1 \end{array}\right]=\left[\begin{array}{c c}a & c\\ b & d \end{array}\right]\left[\begin{array}{c} 1\\1 \end{array}\right] \end{equation}

同理，光速可以为负的，因此：
\begin{equation}\lambda_2 \left[\begin{array}{c} -1\\1 \end{array}\right]=\left[\begin{array}{c c}a & c\\ b & d \end{array}\right]\left[\begin{array}{c} -1\\1 \end{array}\right] \end{equation}

换成矩阵的语言，也就是说$\left[\begin{array}{c} 1\\1 \end{array}\right]$和$\left[\begin{array}{c} -1\\1 \end{array}\right]$是矩阵$\left[\begin{array}{c c}a & c\\ b & d \end{array}\right]$的<strong>特征向量</strong>。这给出：
\begin{equation}a+c=b+d,\quad a-c=d-b\end{equation}
这等价于
\begin{equation}a=d,\quad b=c\end{equation}
这时变换矩阵变成了$\left[\begin{array}{c c}a & b\\ b & a \end{array}\right]$

#### 狭义相对性原理

这时已经完成了一大半的工作了，剩下的要讨论<strong>相对运动</strong>，即讨论速度$v$的影响了。在静时空中速度为$v$的点$\left[\begin{array}{c} v\\1 \end{array}\right]$，在动时空中无疑是$\left[\begin{array}{c} 0\\t' \end{array}\right]$，即是动时空的原点。即
\begin{equation}\lambda \left[\begin{array}{c} v\\1 \end{array}\right]=\left[\begin{array}{c c}a & b\\ b & a \end{array}\right]\left[\begin{array}{c} 0\\t' \end{array}\right] \end{equation}

我们得到：$b=av$。现在变量只剩下一个了：
\begin{equation}\left[\begin{array}{c} x\\t \end{array}\right]=a\left[\begin{array}{c c}1 & v\\ v & 1 \end{array}\right]\left[\begin{array}{c}x'\\t' \end{array}\right] \end{equation}

下面是胜利的一步！还是由于<strong>狭义相对性原理</strong>得来。<u>因为所有惯性系系都是平权的，在静时空看来，动时空是一个速度为$v$远离他们的时空；而在动时空看来，他自己就是静时空，而静时空则是一个速度为$-v$的动时空。因此，从$[x',t']$到$[x,t]$的变换，应该只需要把$[x,t]$到$[x',t']$的变换中的速度$v$改为$-v$即可。</u>所以：
\begin{equation}\left[\begin{array}{c} x'\\t' \end{array}\right]=a\left[\begin{array}{c c}1 & -v\\ -v & 1 \end{array}\right]\left[\begin{array}{c}x\\t \end{array}\right] \end{equation}
这样必然有
\begin{equation}\left(a\left[\begin{array}{c c}1 & v\\ v & 1 \end{array}\right]\right)^{-1}=a\left[\begin{array}{c c}1 & -v\\ -v & 1 \end{array}\right]\end{equation}

于是得到$a^2=\frac{1}{1-v^2}$，即$a=\pm \frac{1}{\sqrt{1-v^2}}$。经过简单的检验就知道应该取正值，即：
\begin{equation}\left[\begin{array}{c} x\\t \end{array}\right]=\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & v\\ v & 1 \end{array}\right]\left[\begin{array}{c}x'\\t' \end{array}\right] \end{equation}

<strong>这就是洛伦兹变化的矩阵形式！！看，在这样的形式和单位之下，时间和空间完全“纠缠”在一起了！读者也许明白为什么时间和空间要成为一个不可分割的整体了，在矩阵$\frac{1}{\sqrt{1-v^2}}\left[\begin{array}{c c}1 & v\\ v & 1 \end{array}\right]$之中，你还能看出时间和空间有什么特别之处吗？它们居然连形式都是一模一样的！太让人意外了！！</strong>

### 完整形式

当然，如果不设光速$c=1$，完整的变换矩阵应该是
\begin{equation}\frac{1}{\sqrt{1-\frac{v^2}{c^2}}}\left[\begin{array}{c c}1 & v\\ \frac{v}{c^2} & 1 \end{array}\right]\end{equation}

可见，伽利略变换对应$c\to \infty$的情况，即矩阵：
\begin{equation}\left[\begin{array}{c c}1 & v\\ 0 & 1 \end{array}\right]\end{equation}

对于四维时空的一般情况，我们只需要对二维时空做一个旋转即可。这用矩阵的思想是非常好理解而且很容易处理的，我们以后将会讨论这一问题。
