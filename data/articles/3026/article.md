# 在Python中使用GMP（gmpy2）

> 作者：苏剑林 · 科学空间 · 2014-10-28
>
> 原文：<https://spaces.ac.cn/archives/3026>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

之前笔者曾写过[《初试在Python中使用PARI/GP》](<https://spaces.ac.cn/archives/2775/>)，简单介绍了一下在Python中调用PARI/GP的方法。PARI/GP是一个比较强大的数论库，“针对数论中的快速计算（大数分解，代数数论，椭圆曲线\.\.\.）而设计”，它既可以被C/C\+\+或Python之类的编程语言调用，而且它本身又是一种自成一体的脚本语言。而如果仅仅需要高精度的大数运算功能，那么[GMP](<http://gmplib.org/>)似乎更满足我们的需求。

了解C/C\+\+的读者都会知道GMP（全称是GNU Multiple Precision Arithmetic Library，即GNU高精度算术运算库），它是一个开源的高精度运算库，其中不但有普通的整数、实数、浮点数的高精度运算，还有随机数生成，尤其是提供了非常完备的数论中的运算接口，比如Miller\-Rabin素数测试算法、大素数生成、欧几里德算法、求域中元素的逆、Jacobi符号、legendre符号等<sup>[&#91;来源&#93;](<http://blog.csdn.net/jcwkyl/article/details/3553411>)</sup>。虽然在C/C\+\+中调用GMP并不算复杂，但是如果能在以高开发效率著称的Python中使用GMP，那么无疑是一件快事。这正是本文要说的[gmpy2](<https://pypi.python.org/pypi/gmpy2/2.0.4>)。

<strong>gmpy2介绍</strong>

gmpy2是Python的一个扩展库，是对GMP的封装，它的前身是gmpy。经过作者的调整和封装，使得gmpy2的使用大大简化。根据笔者目前的体会，在Python中使用gmpy2，跟在C/C\+\+中使用GMP相比，至少以下优势：

> 1、简化的函数名：作者对GMP的大量函数名做了简化，比如素数的概率性测试，在C/C\+\+中，命令是mpz\_probab\_prime\_p，在gmpy2中，只需要简单的is\_prime；
> 
> 
> 
> 2、方便的运算符重载：如果在C/C\+\+中，两个mpz整数相加，我们要用mpz\_add(z\_i, z\_i, z\_o)，在gmpy2中，我们只需要用\+号即可，更一般的，将一个mpz整数与int整数相加，也只需要\+号，类似的还有减、乘、除、求模、乘方等。

笔者所学还不深入，因此只能这样片面的评论了，当然，作为第三方语言的调用，gmpy2的效率总比直接在C/C\+\+中直接调用GMP会低一些，但差别是很小的，因为gmpy2就是预先编译好的C语言库而已。具体的gmpy2教程请看：
[http://gmpy2\.readthedocs\.org/en/latest/](<http://gmpy2.readthedocs.org/en/latest/>)

<strong>基本使用</strong>

本文只做简单介绍。以下代码均在Python 3\.4中运行。初始化一个大整数，只需要



<pre><code class="language-python">import gmpy2
n=gmpy2.mpz(1257787) #初始化
gmpy2.is_prime(n) #概率性素性测试</code></pre>



这里跟C/C\+\+是平行的，其实括号里边的参数，可以是整型，也可以是字符串。gmpy2中不仅集成了大整数mpz，还有高精度浮点数mpfr，用法跟C/C\+\+都是平行的。下面是一些基本计算



<pre><code class="language-python">a+2 #求和，结果是一个mpz
a-2 #求差，结果是一个mpz
a*2 #求积，结果是一个mpz
a/2 #求商，结果是一个mpfr
a//2 #求商，结果是一个mpz
a**2 #求平方，结果是一个mpz
a%2 #求模，结果是一个mpz</code></pre>



这些运算符的重载既直观又方便，其中将数字2换成mpz类型变量也是可以的，这自然不用多说。事实上，运算a\+2，就是先将2转换为mpz中的2，然后执行mpz中的加法。

<strong>与Python自身比较</strong>

Python本身支持高精度大整数运算，一般的数字比较小的范围足够了，只是并非“快速”，读者只需要分别运行以下两句代码



<pre><code class="language-python">gmpy2.mpz(1257787)**123456 #gmpy2计算
1257787**123456 #Python自带计算</code></pre>



就能感觉到了（当然，可能读者的配置还不错，两者并没有明显差别，那么请把指数扩大十倍\~笔者的笔记本配置不高，见笑了^\_^）。

<strong>尝试大数分解</strong>

下面来一段自己编写的大数分解的代码：



<pre><code class="language-python">from gmpy2 import *
import time
start=time.clock()
n = mpz(63281217910257742583918406571)
x = mpz(2)
y = x**2 + 1
for i in range(n):
    p = gcd(y-x,n)
    if p != 1:
        print(p)
        break
    else:
        y=(((y**2+1)%n)**2+1)%n
        x=(x**2+1)%n
end=time.clock()
print(end-start)</code></pre>



该代码调用了gmpy2，算法则是Pollard rho方法，只是最单纯地运行，并没做任何优化。该算法在我的笔记本上分解

> 63281217910257742583918406571 = 125778791843321 × 503115167373251

耗时84秒（获得一个因子即停止）。当然这不是什么了不起的成绩，这个数字在Mathematica和PARI/GP中都是瞬间完成的。这个脚本只是纯粹练习和测试Python对gmpy2的调用。

关于Pollard rho方法的步骤：

> 1、给定合数$n$，以及$x_0=2,y_0=x_0^2+1$；
> 
> 
> 
> 2、计算$p=Gcd(x_0-y_0,n)$，如果p非1，则结果为$n$的一个因子，停止；
> 
> 
> 
> 3、如果p=1，则按照$x_{n+1}=x_n^2+1,y_{n+1}=(y_n^2+1)^2+1$，然后重复第2步。

其中，为了减少计算量，每一步$x_n ,y_n$的计算，都要作模$n$运算，以降低计算量。上述步骤只是最简单的计算部分，还没有处理各种可能出现的异常，比如出现$n|(x_n -y_n)$等；也没有对某些细节进行优化，请谨慎引用。

参考网址：
[http://hi\.baidu\.com/pytvzcnbuolrsue/item/ae714592f1fda8d87b7f010a](<http://hi.baidu.com/pytvzcnbuolrsue/item/ae714592f1fda8d87b7f010a>)
