# 费曼积分法(7)：欧拉数学的综合

> 作者：苏剑林 · 科学空间 · 2013-03-27
>
> 原文：<https://spaces.ac.cn/archives/1946>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在本系列的[第五篇文章](<https://spaces.ac.cn/archives/1942/>)中，BoJone导出了一些看似不合理的公式，而且并没有说明它的应用和来源。其实，这些都是我在研究以下积分的时候总结出来的：

$$\int_{-\infty}^{+\infty} \frac{\cos x}{a^2+x^2}dx$$

## 特例的计算

这道积分该怎么算呢？我一下子也没有好的思路，虽然它已经给出了参数a，但是直接对a求导会使得被积函数变得更加复杂，并没有出现明显可积的情况，所以我决定先研究一个特例，即a=1的情况：

$$\int_{-\infty}^{+\infty} \frac{\cos x}{1+x^2}dx$$

既然对原参数求导并没有起到化简的作用，那么就说明费曼积分法不能直接应用上述参数。我决定<strong>尝试</strong>着把参数改为：
$$F(a)=\int_{-\infty}^{+\infty} \frac{\cos (ax)}{1+x^2}dx$$
明显当a=0时积分值为$F(0)=\pi$，根据[《费曼积分法(5)》](<https://spaces.ac.cn/archives/1942/>)的讨论，当$a\to \infty$时，有$F(\infty)=0$。

求导看看？
$$F'(a)=-\int_{-\infty}^{+\infty} \frac{x \sin (ax)}{1+x^2}dx$$

还是没有明显可积的情况，可是由于三角函数求导的周期性，让我想起了[《费曼积分法(3)》](<https://spaces.ac.cn/archives/1629/>)里边曾<u>用微分方程的思想求积分的方法</u>，于是我再求一次导数：
$$\begin{aligned}F''(a)=&\,-\int_{-\infty}^{+\infty} \frac{x^2 \cos (ax)}{1+x^2}dx \\ =&\,\int_{-\infty}^{+\infty} \frac{\cos (ax)}{1+x^2}dx-\int_{-\infty}^{+\infty}\cos (ax)dx \\
=&\,F(a)-\int_{-\infty}^{+\infty}\cos (ax)dx\end{aligned}$$

真让我们惊喜！！一道二阶线性微分方程出来了，根据[《费曼积分法(5)》](<https://spaces.ac.cn/archives/1942/>)的讨论，$\int_{-\infty}^{+\infty}cos (ax)dx=0$。那么就是说：
$$F''(a)=F(a)$$

那就再好办不过了，直接写出同通解$F(a)=C_1 e^a+C_2 e^{-a}$，并且根据$F(0)=\pi,F(\infty)=0$，直接推出：
$$F(a)=\pi e^{-a}$$

这时我们就可以得到：

$$\begin{aligned}\int_{-\infty}^{+\infty} \frac{\cos x}{a^2+x^2}dx =&\,\frac{1}{a}\int_{-\infty}^{+\infty} \frac{\cos (a\times \frac{x}{a})}{1+(\frac{x}{a})^2}d(\frac{x}{a}) \\ =&\,\frac{1}{a}F(a)=\frac{\pi}{a}e^{-a}\end{aligned}$$

## 思考与疑惑

如果你是自己推导过这个过程的，我相信你也会和我一样为一件事情感到困惑，那就是根据$F'(a)=-\int_{-\infty}^{+\infty} \frac{x \sin (ax)}{1+x^2}dx$会得到$F'(0)=0$，但是我们求出的解显然不符合这个规律，这是为什么呢？

一开始我百思不得其解，可是后来想想，$a=0$是一个比较特殊的情况，如果$a\neq 0$时会怎样呢？我尝试让a取一个很小的具体的数值（比如0\.000001），发现由$F'(a)=-\int_{-\infty}^{+\infty} \frac{x \sin (ax)}{1+x^2}dx$算出的结果跟$F(a)=\pi e^{-a}$算出的结果是一致的。那么最终的结论就是：<u>积分$\int_{-\infty}^{+\infty} \frac{x \sin (ax)}{1+x^2}dx$在a=0处不连续！！</u>其实也可以这样思考：a有一个趋于0的过程，但是x也有一个趋于无穷大的过程，更重要的是，即使$\sin(ax)$趋于0，$x \sin(ax)$也不一定趋于0。总的来说可以归结为一句话：

> <strong>在这道积分中，先对$a$取极限再对$x$取极限，以及先对$x$取极限再对$a$取极限，结果是不同的！</strong>

这告诫我们多元变量函数取极限并不是任何时候都可以交换顺序的！

另一方面，其实在《费曼积分法(5)》中也算是提供了我们求
$$\int_{-\infty}^{+\infty} \frac{\cos x}{a^2+x^2}dx$$
的另外一个思路，那就是将其变成：
$$\int_{-\infty}^{+\infty} \frac{e^{ix}}{a^2+x^2}dx$$
并引入参数使其变成：
$$F(b)=\int_{-\infty}^{+\infty} e^{-b(x^2+a^2)}\frac{e^{ix}}{a^2+x^2}dx$$
对b求导即可。这也算是一种不复杂的思路，但是过程要用到[《费曼积分法(3)》](<https://spaces.ac.cn/archives/1629/>)里边的结论。有兴趣的读者不妨尝试一下？
