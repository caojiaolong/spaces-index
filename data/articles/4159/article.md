# 【备忘】Python中断多重循环的几种思路

> 作者：苏剑林 · 科学空间 · 2016-12-19
>
> 原文：<https://spaces.ac.cn/archives/4159>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## 跳出单循环

不管是什么编程语言，都有可能会有跳出循环的需求，比如枚举时，找到一个满足条件的数就终止。跳出单循环是很简单的，比如



<pre><code class="language-python">for i in range(10):
    if i &gt; 5:
        print i
        break</code></pre>



然而，我们有时候会需要跳出多重循环，而break只能够跳出一层循环，比如



<pre><code class="language-python">for i in range(10):
    for j in range(10):
        if i+j &gt; 5:
            print i,j
            break</code></pre>



这样的代码并非说找到一组i\+j \> 5就停止，而是连续找到10组，因为break只跳出了for j in range(10)这一重循环。那么，怎么才能跳出多重呢？在此记录备忘一下。

## 跳出多重循环

事实上，Python的标准语法是不支持跳出多重循环的，所以只能利用一些技巧，大概的思路有：写成函数、利用笛卡尔积、利用调试。

### 写成函数

在Python中，函数运行到return这一句就会停止，因此可以利用这一特性，将功能写成函数，终止多重循环，例如



<pre><code class="language-python">def work():
    for i in range(10):
        for j in range(10):
            if i+j &gt; 5:
                return i,j

print work()</code></pre>



### 利用笛卡尔积

这种方法的思路就是，既然可以跳出单循环，我就将多重循环改写为单循环，这可以利用itertools中的笛卡尔积函数product，例如



<pre><code class="language-python">from itertools import product

for i,j in product(range(10), range(10)):
    if i+j &gt; 5:
        print i,j
        break</code></pre>



### 利用调试模式

笛卡尔积的方式很巧妙，也很简洁，但它只能用于每次循环的集合都是独立的情形，假如每层循环都与前一层紧密相关，就不能用这种技巧了。这时候可以用第一种方法，将它写成函数，另外，还可以利用调试模式。这个利用了调试模式中，只要出现报错就退出的原理，它伪装了一个错误出来。



<pre><code class="language-python">class Found(Exception):
    pass

try:
    for i in range(10):
        for j in range(i): #第二重循环跟第一重有关
            if i + j &gt; 5:
                raise Found
except Found:
    print i, j</code></pre>
