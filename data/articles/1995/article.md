# 当Matlab遇上牛顿法

> 作者：苏剑林 · 科学空间 · 2013-05-22
>
> 原文：<https://spaces.ac.cn/archives/1995>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

牛顿法是求方程近似根的一个相当有用而且快捷的方法，我们最近科学计算软件课程（Matlab）的一个作业就是编写求方程近似解的程序，其中涉及到牛顿法。我们要实现的目标是，<u>用户输入一道方程，脚本就自动求出根来。</u>这看起来是一个挺简单的循环迭代程序，但是由于Matlab本身的特殊性，却产生了不少困难。

Matlab是为了数值计算（尤其是矩阵运算）而生的，因此它并不擅长处理符号计算。这就给我们编程带来了困难。在网上随便一搜，就可以发现，<u>网上的Matlab牛顿法程序都是要求用户同时输入方程及其导函数</u>，这显然是不方便的，因为Matlab本身就具备了求导功能。下面我们来分析一下困难在哪里。

我们要实现的最基本功能是<strong>定义一个函数，然后可以根据该函数求具体的函数值，并且自动求该函数的导数，接着求导数值</strong>。这些看起来很基本的功能在Matlab中却很难调和，因为Matlab的“函数”定义很广，一个具有特定功能的M文件叫“函数”，一个运算式$f(x)$也可能是一个函数，显然后者是可以求导的，前者却不行，所以Matlab一刀砍——<strong>不能对函数求导！！</strong>

那Matlab的求导功能是怎样的呢？事实上，它是对函数式的字符串操作的，比如



<pre><code class="language-php">diff(x^2) </code></pre>



其中括号里边是一个<strong>字符串式子</strong>，而不是函数。那Matlab中的函数是怎样的呢？Matlab中简单定义一个函数，可以写



<pre><code class="language-php">f=@(x) x^2 </code></pre>



或



<pre><code class="language-php">f=inline(x^2,'x')</code></pre>



这样输入f(2)求出$2^2$了。这样定义的函数是不能求导的。那么怎么调和两者的矛盾呢？里边有个小技巧：



<pre><code class="language-php">diff(f(x)) </code></pre>



注意<u>这里不能将f(x)改为f，f是一个函数，而f(x)是该函数自变量为x时的值！</u>这一点相当重要。当然，这样求导的结果也是一个字符串，我们要把它重新定义为函数：



<pre><code class="language-php">df=inline(diff(f(x)),'x') </code></pre>



这样我们就可以轻松写出下面的脚本：



<pre><code class="language-php">function f=jfc(g)

syms x
p=0.000001;  %定义精度
n=0;  %计算迭代次数

s=inline(g,'x');  %定义函数
ds=inline(diff(s(x)),'x');   %定义导函数

x=1;  %初始值
e=s(x); 

%下面是迭代过程
while abs(e)&gt;p
    n=n+1;
    x=x-s(x)/ds(x);
    e=s(x);
end

%输出
[x n e]</code></pre>



这是一个简单的程序，还不完善（比如初值等等情况还没有处理好）。
用法：



<pre><code class="language-php">jfc(x^2-2)
</code></pre>



我们还有另外一个思路，就是<u>整个过程纯粹用字符串形式，而不定义任意函数。</u>但是这样我们就不能根据函数求值了，怎么求值呢？有一个巧妙的技巧——Matlab中的“替换函数”。比如$f(x)=x^2$，求$f(2)$：



<pre><code class="language-php">f=subs(x^2,'5',x)
</code></pre>



这将输出f=25。Subs是一个神奇的函数，它等价于将$x^2$中的x替换成2，这是求函数值的方法！于是我们也可写出以下版本的程序：



<pre><code class="language-php">function f=jfc(g)

p=0.000001;
x=1;
n=0;

ds=diff(s);
e=subs(s,'x',x);

while abs(e)&gt;p
    n=n+1;
    ds0=subs(ds,'x',x);
    x=x-e/ds0;
    e=subs(s,'x',x);
end

[x n p]</code></pre>



<strong>结论</strong>

看完本文，比较了解Matlab的读者会感觉这是一个并没有什么了不起的技巧。的确，现在看来这真的很简单，可是我也想不到为什么网上并没有出现类似的程序。我不敢妄下定论。也许是因为我才刚刚入门，一些从最基本的原理出发吧。当然，还有一个小原因，就是当程序现成的功能不够用时，就自己动手编写新的；当你不能适应世界时，就让世界适应你吧。
