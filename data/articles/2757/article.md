# 《新理解矩阵6》：为什么只有方阵有行列式？

> 作者：苏剑林 · 科学空间 · 2014-07-15
>
> 原文：<https://spaces.ac.cn/archives/2757>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

学过线性代数的朋友都知道，方阵和非方阵的一个明显不同是，对于方阵我们可以计算它的行列式，如果不是方阵的话，就没有行列式这个概念了。在追求统一和谐的数学系统中，为什么非方阵却没有行列式？也许对于这个问题最恰当的回答是——因为不够美。对于非方阵，其实也可以类似地定义它的行列式，定义出来的东西，跟方阵的行列式具有同样的性质，比如某行乘上一个常数，行列式值也就乘以一个常数，等等；而且还可以把其几何意义保留下来。但是，非方阵的行列式是不够美的，因为对于一个一般的整数元素的方阵，我们的行列式是一个整数；而对于一个一般的整数元素的非方阵，却导致了一个无理数的行列式值。另外，一个也比较重要的原因是，单单是方阵的行列式也够用了。综合以上两个理由，非方阵的行列式就被舍弃不用了。

<strong>非方阵的行列式不够漂亮</strong>

$n$阶方阵的行列式是每个向量的线性函数，它代表着向量之间的线性相关性；从几何上来讲，它就是向量组成的平行n维体的（有向）体积。我们当然期望非方阵的行列式也保留这些性质，因为只有这样，方阵行列式的那些运算性质才得以保留，比如上面说的，行列式的一行乘上一个常数，行列式值也乘上一个常数。我们考虑$m\times n$的矩阵，其中$ m < n $，我们将它看成是$m$个$n$维向量的组合。最简单的，我们先考虑$1\times 2$矩阵的行列式，也就是二维向量$(a,b)$的行列式。

我们已经知道，$2\times 2$矩阵的行列式的绝对值，就是这两个向量所围成的平行四边形的面积。类似地，二维向量$(a,b)$的行列式的绝对值，就等于该向量自身的长度了，也就是$\det (a,b)=\sqrt{a^2+b^2}$，对于有理的$a,b$，大多数情况都会出现无理的行列式值，这是不协调的。因为它是线性的函数，有理数的线性运算导致无理数，却是不大舒服的。

类似的，可以考虑$2\times 3$矩阵
$$\boldsymbol{A}=\left(\begin{array}{\cdot {20}{c}}
a&b&c\\
d&e&f
\end{array}\right)$$
的行列式。按照几何意义，它的行列式的模就是这两个向量所围成的平行四边形的面积。我们可以利用叉积来计算这个面积，但为了更一般化，我们利用正交化的方法来计算它。

以向量$\boldsymbol{x}_1=(a,b,c)$为出发点，选取
$$\boldsymbol{e}_1=\frac{(a,b,c)}{\sqrt{a^2+b^2+c^2}}$$
那么$\boldsymbol{x}_1=\left|\boldsymbol{x}_1\right|\boldsymbol{e}_1$。对$\boldsymbol{x}_2=(d,e,f)$正交化，得到
$$\boldsymbol{x}_2 - \left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1$$
取
$$\boldsymbol{e}_2=\frac{\boldsymbol{x}_2 - \left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1}{\left|\boldsymbol{x}_2 - \left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1\right|}$$
得到
$$\boldsymbol{x}_2=\left|\boldsymbol{x}_2 - \left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1\right|\boldsymbol{e}_2+\left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1$$
因此，行列式的模，也就是两个向量所形成的平行四边形的面积等于
$$\left|\det\boldsymbol{A}\right|=\left|\boldsymbol{x}_1\right|\cdot \left|\boldsymbol{x}_2 - \left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1\right|$$

最后的表达式出现了两个模，表明两个根号，随便挑个具体例子就可以验证，即使$\boldsymbol{A}$的元素全部是整数，也得不到有理数。这体现了非方阵行列式的不美之处。

<strong>方阵的行列式就够了</strong>

由于非方阵的行列式不够美，那么我们干脆弃之不用了。可是，这样会不会产生什么“副作用”？也就是说，会不会有哪些地方，非用到非方阵的行列式不可？事实上，至少就我目前的认识来说，答案是没有。

比如说判断$m$个$n$维行向量的线性相关性，我们是这样做的：第一种方法，利用初等变换，看变换后的矩阵的秩是否为$m$；如果嫌第一种方法步骤太多，第二种方法相对来说干脆一点，就是检验删除任意$n-m$列后，剩下的方阵行列式是否为0，如果存在一个不为0，就说明线性无关了。所以说，方阵的行列式够用了。

如果要求面积、体积怎么办？事实上，用的就是类似$\left|\boldsymbol{x}_1\right|\cdot \left|\boldsymbol{x}_2 - \left\langle {\boldsymbol{e}_1,\boldsymbol{x}_2} \right\rangle\boldsymbol{e}_1\right|$的公式，这公式不难理解，也不难记忆。由于它带有不可化简的根号，因此，也就没有简单的计算公式了。

最后，一个比较核心的限制了非方阵行列式使用的原因是，非方阵的行列式应用不多。方阵的行列式可以用来求各种因子，比如重积分的坐标变换中的雅可比行列式，等等，这些由于方阵的可逆性，这些行列式都是有直接的应用意义的。相比之下，非方阵没有可逆的说法，所以非方阵的行列式应用也就很窄了。
