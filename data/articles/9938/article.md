# 旁门左道之如何让Python的重试代码更加优雅

> 作者：苏剑林 · 科学空间 · 2024-01-14
>
> 原文：<https://spaces.ac.cn/archives/9938>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

这篇文章我们讨论一个编程题：如何更优雅地在Python中实现重试。

在文章[《新年快乐！记录一下 Cool Papers 的开发体验》](<https://spaces.ac.cn/archives/9920>)中，笔者分享了开发[Cool Papers](<https://papers.cool/>)的一些经验，其中就提到了Cool Papers所需要的一些网络通信步骤。但凡涉及到网络通信，就有失败的风险（谁也无法保证网络不会间歇性抽风），所以重试是网络通信的基本操作。此外，当涉及到多进程、数据库、硬件交互等操作时，通常也需要引入重试机制。

在Python中，实现重试并不难，但如何更加简单而又不失可读性地实现重试，还是有一定技巧的。接下来笔者分享一下自己的尝试。

## 循环重试

完整的重试流程大致上包含循环重试、异常处理、延时等待、后续操作等部分，其标准写法就是用for循环，用“try \.\.\. except \.\.\.”来捕捉异常，一个参考代码是：



<pre><code class="language-python">import time
from random import random

allright = False  # 执行成功的标记

for i in range(5):  # 最多重试5次
    try:
        # 有概率出错的代码
        x = random()
        if x &lt; 0.5:
            yyyy  # 未定义yyyy，所以会报错
        allright = True
        break
    except Exception as e:
        print(e)  # 打印错误信息
        if i &lt; 4:
            time.sleep(2)  # 延时两秒

if allright:
    # 执行某些操作
    print('执行成功')
else:
    # 执行另一些操作
    print('执行失败')</code></pre>



接下来我们的目的是简化<code>if allright:</code>之前的代码，可以发现，它具有比较固定的格式，即在for循环再加上“try \.\.\. break \.\.\. except \.\.\. sleep \.\.\.”的模版，不难想象它应该有很大的简化空间。

## 函数装饰

for循环的问题在于，如果有很多处地方需要重试，并且异常处理的方法都一样，那么每次都重写一遍<code>except</code>的代码就显得很累赘。这种情况下，标准的推荐方法是将有概率出错的代码写成一个函数，并写一个用来处理异常的装饰器：



<pre><code class="language-python">import time
from random import random

def retry(f):
    """重试装饰器，包装函数加上重试功能
    """
    def new_f(*args, **kwargs):
        for i in range(5):  # 最多重试5次
            try:
                return True, f(*args, **kwargs)
            except Exception as e:
                print(e)  # 打印错误信息
                if i &lt; 4:
                    time.sleep(2)  # 延时两秒
        return False, None
    return new_f

@retry
def f():
    # 有概率出错的代码
    x = random()
    if x &lt; 0.5:
        yyyy  # 未定义yyyy，所以会报错
    return x

allright, _ = f()  # 返回执行状态和执行结果
if allright:
    # 执行某些操作
    print('执行成功')
else:
    # 执行另一些操作
    print('执行失败')</code></pre>



当有多处不同的代码都需要重试时，只需要将它们都分别写成函数并加上装饰器<code>@retry</code>就可以实现相同的重试逻辑，所以装饰器写法确实是一个简明的解决方法，也很直观，所以不难理解能成为标准。目前主流的重试库，比如[tenacity](<https://github.com/jd/tenacity>)，或者更早的[retry](<https://github.com/invl/retry>)、[retrying](<https://github.com/groodt/retrying>)等，都是基于装饰器原理的。

## 理想写法

然而，装饰器的写法虽然是标准，但并不完美。首先，需要把重试代码另外封装为一个函数，在很多情况下会让代码显得不够流畅，有种突然卡顿的感觉；其次，由于代码被封装为一个函数，所以代码中的中间变量没法直接用，需要用的变量得全部写在<code>return</code>中，这显得有点迂回。总的来说，装饰器虽然能够简化重试的代码，但仍然还差点意思。

笔者想象中的完美重试代码，应该是基于上下文管理器的写法，类似于：



<pre><code class="language-python">with Retry(max_tries=5) as retry:
    # 有概率出错的代码
    x = random()
    if x &lt; 0.5:
        yyyy  # 未定义yyyy，所以会报错

if retry.allright:
    # 执行某些操作
    print('执行成功')
else:
    # 执行另一些操作
    print('执行失败')</code></pre>



然而，笔者去研究了一下上下文管理器的原理之后，才发现这种理想写法注定是<strong>无法实现</strong>的。因为上下文管理器只能管理上下文，却不能管理主体代码（即本文中那段“有概率出错的代码”）。具体来说，上下文管理器是一个带有<code>__enter__</code>和<code>__exit__</code>方法的类，它将<code>__enter__</code>插入到代码运行之前（上文），将<code>__exit__</code>插入到代码运行之后（下文），但无法操控中间的代码（比如让它运行多几次）。

所以，基于上下文管理器的一行实现重试的写法宣告失败。

## 挣扎一下

不过，好消息是，上下文管理器虽然不能实现循环，但它的<code>__exit__</code>方法能处理异常，所以它至少能代替“try \.\.\. except \.\.\.”来处理异常，于是我们可以写出：



<pre><code class="language-python">import time
from random import random

class Retry:
    """自定义处理异常的上下文管理器
    """
    def __enter__(self):
        self.allright = False
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.allright = True
        else:
            print(exc_val)
            time.sleep(2)
        return True

for i in range(5):  # 最多重试5次
    with Retry() as retry:
        # 有概率出错的代码
        x = random()
        if x &lt; 0.5:
            yyyy  # 未定义yyyy，所以会报错
        break

if retry.allright:
    # 执行某些操作
    print('执行成功')
else:
    # 执行另一些操作
    print('执行失败')</code></pre>



最新的这个版本，其实写法上已经非常接近前一节的理想写法了，不同点有两个：1、需要多写一句<code>for</code>循环，但这个无法避免，因为前面已经说了上下文管理器是无法启动循环的，所以只能自己额外用<code>for</code>或者<code>while</code>启动循环；2、需要自己显式加一句<code>break</code>，这个倒是可以想办法优化掉。

此外，这个版本还有一个小缺陷，就是假如所有重试都失败了，那么最后一次重试失败之后，依然会启动sleep，这理论上是没有必要的，应当想办法去掉。

## 继续优化

为了优化掉<code>break</code>，那么模型的循环应该要学会自己停止，这有两个办法：第一个办法是可以改用<code>while</code>循环，然后让停止条件根据重试结果改变，这将会导致跟[《Handling exceptions inside context managers》](<https://stackoverflow.com/a/35483446>)相近的结果；第二个办法则是保持<code>for</code>循环，但<code>range(5)</code>这个要换成根据重试结果变化的迭代器。本文主要探讨后一种方案。

经过分析，笔者发现可以通过内置方法<code>__call__</code>和<code>__iter__</code>将<code>retry</code>同时作为一个可变的迭代器，并解决最后一次失败后的非必要sleep问题：



<pre><code class="language-python">import time
from random import random

class Retry:
    """处理异常的上下文管理器 + 迭代器
    """
    def __call__(self, max_tries=5):
        self.max_tries = max_tries
        return self
    def __iter__(self):
        for i in range(self.max_tries):
            yield i
            if self.allright or i == self.max_tries - 1:
                return
            time.sleep(2)
    def __enter__(self):
        self.allright = False
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.allright = True
        else:
            print(exc_val)
        return True

retry = Retry()
for i in retry(5):  # 最多重试5次
    with retry:
        # 有概率出错的代码
        x = random()
        if x &lt; 0.5:
            yyyy  # 未定义yyyy，所以会报错

if retry.allright:
    # 执行某些操作
    print('执行成功')
else:
    # 执行另一些操作
    print('执行失败')</code></pre>



仔细对比的读者可能会疑问：你想办法删了一句<code>break</code>，但多了一句<code>retry = Retry()</code>，总的代码数不变（还把上下文管理器搞复杂了），有这个必要折腾吗？事实上这里的<code>retry</code>是可重用的，用户只需要一次定义<code>retry = Retry()</code>，那么在后面的重试中都只需要：



<pre><code class="language-python">for i in retry(max_tries):
    with retry:
        # 有概率出错的代码
</code></pre>



就行了，因此虽然把上下文的管理器搞复杂了一点，但却已经是无限接近理想写法的实现了。

## 终极版本

不过，“一次定义<code>retry = Retry()</code>，多次重用<code>retry</code>”只适合于单进程，如果是多进程还是要分别定义<code>retry = Retry()</code>的，另外就是这样重用总给人“不同的重试之间似乎没有完全隔离”的感觉。有没有可能将这一句完全去掉呢？笔者再想一下，发现还是有可能的！参考代码如下：



<pre><code class="language-python">import time
from random import random

class Retry:
    """处理异常的上下文管理器 + 迭代器
    """
    def __init__(self, max_tries=5):
        self.max_tries = max_tries
    def __iter__(self):
        for i in range(self.max_tries):
            yield self
            if self.allright or i == self.max_tries - 1:
                return
            time.sleep(2)
    def __enter__(self):
        self.allright = False
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.allright = True
        else:
            print(exc_val)
        return True

for retry in Retry(5):  # 最多重试5次
    with retry:
        # 有概率出错的代码
        x = random()
        if x &lt; 0.5:
            yyyy  # 未定义yyyy，所以会报错

if retry.allright:
    # 执行某些操作
    print('执行成功')
else:
    # 执行另一些操作
    print('执行失败')</code></pre>



这次的改动是将<code>__call__</code>换为<code>__init__</code>，然后<code>__iter__</code>中的<code>yield i</code>改为了<code>yield self</code>，即返回对象本身。这样一来，就不用单独写一行<code>retry = Retry(5)</code>来初始化，而是在<code>for retry in Retry(5):</code>中同时实现了初始化和别名赋值，并且由于每次重试都会重新初始化，因此实现了重试之间的完全隔离，可谓一举两得了。

## 文章小结

本文相对完整地探讨了Python中重试机制的写法，试图得到笔者心目中重试代码的完美实现，最后的结果也勉强达到了心中所想。

不过，不得不承认的是，这篇文章的起因实质只是“强迫症”在作祟，并没有算法效率的实质改进，在编程细节上花太多时间，某种程度上已经是“旁门左道”、“不务正业”了，并不是一件十分值得学习的事情。
