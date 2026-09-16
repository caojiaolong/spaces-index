# 四次方程的根式求解（通俗版）

> 作者：苏剑林 · 科学空间 · 2009-09-06
>
> 原文：<https://spaces.ac.cn/archives/114>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前些时间发表了<strong>[三次方程的一般求解](<https://spaces.ac.cn/archives/26/>)</strong> ，并通过了维基百科链接到了这里来，想不到带来了很多的人气，看到大家还是比较需要这方面的资料的。在此之前曾经承诺过会把4次方程的求根公式也写出来，现在终于有时间了，就此一写，希望能够为大家带来帮助。

$$ax^4+bx^3+cx^2+dx+e=0(a!=0)$$

<strong>仍然是这两句话：</strong>网上的资料中，一是缺乏描述专业数学公式的相关程序（很多网站都是这样）；二是语言过于专业，不能大众化（如[维基百科](<http://zh.wikipedia.org/wiki/%E5%9B%9B%E6%AC%A1%E6%96%B9%E7%A8%8B>)）。<u>如果一开始我就去看wiki，那么我保证我到现在还不能弄懂。</u>

话说卡丹从塔塔利亚的口中得到了三次方程的求根公式之后，他又有了一个不平凡的遭遇。一天，一个叫“费拉里”的人成为了他家中的仆人。然而，这个“仆人”却表现出了高超的数学能力，因此，卡丹和费拉里的关系很快地从主仆关系转变为师生关系。通过听卡尔达诺讲课，学习了拉丁语、希腊语和数学，后来，还作出了一项空前的成就，那就是发现了“四次方程的根式求解方法”！

<strong>分析过程：</strong>

> 与卡丹的三次方程解法同出一辙，费拉里先把一般的四次方程
> $ax^4+bx^3+cx^2+dx+e=0$，
> 通过设$y=x+{b}/{4a}$，变成关于y的四次方程
> $y^4+py^2+qy+r=0$ ———(A)
> 
> 
> 
> 在这道方程中，增加一个额外的变量z，并作以下变换：
> $(y^2+p+z)^2=(p+2z)y^2-qy+(p^2-r+2pz+z^2)$ ———(B)
> 
> 
> 
> 可以证明，<strong>(A)</strong>和<strong>(B)</strong>等价。
> 
> 
> 
> 如果我们可以把<strong>(B)</strong>的右边也变成完全平方形式，那么原方程就可以<u>变成关于y的一元二次方程</u>了。究竟怎样才能够把<strong>(B)</strong>的右边变成我们希望的完全平方形式呢？<strong>这取决于z。</strong>
> 
> 
> 
> 我们知道，如果存在一道二次三项式$ax^2+bx+c(a!=0)$，使得这一道二次三项式为完全平方式的充分条件为$b^2-4ac=0$。而<strong>(B)</strong>右端正是这样的一条二次三项式，因此我们不妨令
> $q^2-4(p+2z)(p^2-r+2pz+z^2)=0$ ———(C)
> 
> 
> 
> 那么就实现了我们的目的，而<strong>(C)</strong>是关于z的三次方程，是可以求解的。这样，我们求出了z后，可以把<strong>(B)</strong>两端同时开方，变为关于y的两道二次方程。这样可以求出4个y（一般情形），进而求出4个x。

<strong>因此，根式求解4次方程的一般步骤为：</strong>
1\. 通过设$y=x+{b}/{4a}$，变成关于y的四次方程$y^4+py^2+qy+r=0$。
2\. 解方程$q^2-4(p+2z)(p^2-r+2pz+z^2)=0$。
3\. 解方程$(y^2+p+z)^2=(p+2z)y^2-qy+(p^2-r+2pz+z^2)$
4\. 解方程$y=x+{b}/{4a}$。

<strong>可以想象，四次方程一般情形的求根公式会有多么的复杂！因此，似乎还没有这样的能人把这条公式写出来（或者说没有人愿意去写）！只是借助计算机写了出来</strong>

<strong>一元4次方程的4个根:</strong>
(1)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165823.png>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165823.png>)

(2)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165833.png>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165833.png>)

(3)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165839.png>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165839.png>)

(4)

[![](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165845.png>)](<https://spaces.ac.cn/usr/uploads/2009/09/20090905165845.png>)

可以参考下：
[http://planetmath\.org/encyclopedia/QuarticFormula\.html](<http://planetmath.org/encyclopedia/QuarticFormula.html>)

[http://www\.wolframalpha\.com/input/?i=a\+x%5E4%2Bb\+x%5E3%2Bc\+x%5E2%2Bd\+x%2Be\+%3D\+0](<http://www.wolframalpha.com/input/?i=a+x%5E4%2Bb+x%5E3%2Bc+x%5E2%2Bd+x%2Be+%3D+0>)

<strong>最后，现在看来，卡丹这个“伯乐”发现了费拉里这匹“千里马”，使得他能够在数学上绽放光彩！</strong>
