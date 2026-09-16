# 两百万素数之和与“电脑病”

> 作者：苏剑林 · 科学空间 · 2014-10-17
>
> 原文：<https://spaces.ac.cn/archives/2991>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

原则上来讲，同样的算法，如果分别在Python和C\+\+上实现，那么Python的速度肯定比不上C\+\+的。但是Python还被称为“胶水语言”，它允许我们把主要计算的部分用C或C\+\+等高效的语言编写好，然后它作为“粘合剂”把两者粘合在一起。正因为如此，Python才有了各种各样的扩展库，这些库中有不少是用C语言编写的。因此，我们在编写Python程序的时候，如果可以用这些现成的库，速度会快很多。本文就是用Numpy来改进之前的[《两百万前素数之和与前两百万素数之和》](<https://spaces.ac.cn/archives/2612/>)的计算。

算法本身是没有变的，只是用了Numpy来处理数组计算，代码如下：

<strong>小于200万的素数之和：</strong>



<pre><code class="language-python">import time
import numpy as np
start=time.clock()

n=2000000
prime=np.arange(n+1,dtype=np.uint64) 
r=int(np.sqrt(n))

for j in range(2,r+1):
    if prime[j] != 0:
        prime[j*j:n+1:j]=0
print(np.int64(prime.sum()-1))

end=time.clock()
print("time:",end-start)</code></pre>



<strong>前200万个素数之和</strong>



<pre><code class="language-python">import time
import numpy as np
start=time.clock()

n=34000000
prime=np.arange(n+1,dtype=np.uint64) 
r=int(np.sqrt(n))

for j in range(2,r+1):
    if prime[j] != 0:
        prime[j*j:n+1:j]=0
        
prime.sort()
z=np.where(prime==1)[0][0]
print(np.int64(prime[z+1:z+1+2000000].sum()))
end=time.clock()
 
print("time:",end-start)</code></pre>



这样改进之后，计算效率大大提高了，而且代码缩短了\~。在我的电脑上测试（Win 8 \+ Python 3\.4）：

> 两百万前的素数之和：
> 142913828922
> time: 0\.13008331361399986
> 
> 
> 
> 前两百万的素数之和：
> 31381137530481
> time: 5\.534884070836405

速度比之前的代码用PyPy执行还快！

<strong>关于“电脑病”</strong>

谈点外话。感觉自己患上电脑病了，呵呵\~\~下面是来自《别闹了费曼先生》里边的关于“电脑病”的介绍：

> 至于弗兰科呢，这个“程序”是他发明的，但这时候他却跟所有后来的电脑使用者一样，患上了电脑病。这是种很严重的病，甚至干扰到正常工作的进行了。电脑的麻烦，在于你会跟它“玩”。它们是那么的有趣——所有的按钮都在你掌握之中，你这样弄得到某个双数，那样弄就是单数。不久之后，只要你够聪明，能计算的东西便愈来愈多。
> 
> 
> 
> 可是不久之后，我们的系统也崩溃下来了，因为弗兰科无法专心工作，更没用心督导其他人。计算系统运行得很慢很慢，他却坐在房间内，思索如何能让列表机自动算出角度的反正切值。好了，列表机开始动作，画出一行行的线，发出“嗖！嗖！嗖！”的声音，一边画一边计算积分值，然后把所有角度的反正切值列出来，一次完成。
> 
> 
> 
> 这绝对是没用的事情，因为我们早已有反正切函数表了。但如果你用过计算机，你就会充分了解这种病——发现自己有多能干的喜悦。这是他第一次感染上这种病症，好笑的是，那套系统却刚好是这个可怜虫创造出来的！

感觉真患上这种病了\~^\_^
