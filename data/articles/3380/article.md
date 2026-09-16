# 用Pandas实现高效的Apriori算法

> 作者：苏剑林 · 科学空间 · 2015-07-02
>
> 原文：<https://spaces.ac.cn/archives/3380>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最新更新：[《用Numpy实现高效的Apriori算法》](<https://spaces.ac.cn/archives/5525>)

最近在做数据挖掘相关的工作，阅读到了Apriori算法。平时由于没有涉及到相关领域，因此对Apriori算法并不了解，而如今工作上遇到了，就不得不认真学习一下了。Apriori算法是一个寻找关联规则的算法，也就是从一大批数据中找到可能的逻辑，比如“条件A\+条件B”很有可能推出“条件C”（A\+B\-\-\>C），这就是一个关联规则。具体来讲，比如客户买了A商品后，往往会买B商品（反之，买了B商品不一定会买A商品），或者更复杂的，买了A、B两种商品的客户，很有可能会再买C商品（反之也不一定）。有了这些信息，我们就可以把一些商品组合销售，以获得更高的收益。而寻求关联规则的算法，就是关联分析算法。

### 啤酒与尿布

[![啤酒与尿布](<https://spaces.ac.cn/usr/uploads/2015/07/2732992076.png>)](<https://spaces.ac.cn/usr/uploads/2015/07/2732992076.png>)

啤酒与尿布

关联算法的案例中，最为人老生常谈的应该是“啤酒与尿布”了。“啤酒与尿布”的故事产生于20世纪90年代的美国沃尔玛超市中，超市管理人员发现“啤酒与尿布两件看上去毫无关系的商品会经常出现在同一个购物篮中”。经过分析，原来在美国有婴儿的家庭中，一般是母亲在家中照看婴儿，年轻的父亲前去超市购买尿布。父亲在购买尿布的同时，往往会顺便为自己购买啤酒，这样就会出现啤酒与尿布这两件看上去不相干的商品经常会出现在同一个购物篮的现象。因此，沃尔玛尝试将啤酒与尿布摆放在相同的区域，让年轻的父亲可以同时找到这两件商品。事实是效果相当不错！

（注：这个故事的真实性是被人怀疑的，但无论如何，它已经成为了介绍关联分析的著名、显浅的案例。）

### Apriori算法

如何从一大批购买记录中，找到像“啤酒与尿布”之类的关联规则？很有多关联分析算法，其中最简单的应该是Apriori算法了（当然效率也不高，但是作为入门算法，还是相当不错的。）关于Apriori算法的详细内容，本文不作介绍，因为网上已经有太多类似的内容了，而且现有内容都已经不错了。推荐阅读：
[https://zh\.wikipedia\.org/zh\-cn/关联式规则](<https://zh.wikipedia.org/zh-cn/%E5%85%B3%E8%81%94%E5%BC%8F%E8%A7%84%E5%88%99>)

[http://hackerxu\.com/2014/10/18/apriori\.html](<http://hackerxu.com/2014/10/18/apriori.html>)

### Python实现

经过几天的调试，终于用Python实现了一个比较高效的Apriori脚本。当然，这里的高效是就Apriori算法本身而言的，不涉及到对算法本身的改进。算法利用了Pandas库，在保证运行效率的前提下，基本实现了代码最短化。读者可以发现，这里比网上找到的很多Apriori算法的代码（不限于Python代码）都要短，效率都要高。

代码同时兼容Python 2\.x和3\.x，当然前提是安装好Pandas。本代码基本可以处理几万条记录，数十个候选项的关联分析问题，当然，前提是需要耐心等待。

### 算法的效率问题

Apriori算法的运行时间取决于很多因素，比如数据量、最小支持度（但是跟最小置信度没什么关系）、候选项个数等，以购物篮分析为例，首先，运行的时间当然直接取决于购物记录的条数$N$，但是跟$N$的关系仅仅是线性的。其次，最小支持度是几乎具有决定性的，它对运行时间很有影响，但是其影响又得具体问题具体分析，此外它很大程度上也决定了最终产生的规则数目。最后是候选项的数目$k$，也就是所有购物车记录中，总共出现了多少种商品，这个也是决定性的，如果$k$本身比较大，后面依次连接，那么项数将是$k^2$、$k^3$\.\.\.（近似），对速度的的影响是致命的。

因此，Apriori算法思路是简单了，但是效率却不高。

### 代码



<pre><code class="language-python">#-*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd

d = pd.read_csv('apriori.txt', header=None, dtype = object)

print(u'\n转换原始数据至0-1矩阵...')
import time
start = time.clock()
ct = lambda x : pd.Series(1, index = x)
b = map(ct, d.as_matrix())
d = pd.DataFrame(list(b)).fillna(0)
d = (d==1)
end = time.clock()
print(u'\n转换完毕，用时：%0.2f秒' %(end-start))
print(u'\n开始搜索关联规则...')
del b

support = 0.06 #最小支持度
confidence = 0.75 #最小置信度
ms = '--' #连接符，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符

#自定义连接函数，用于实现L_{k-1}到C_k的连接
def connect_string(x, ms):
    x = list(map(lambda i:sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i,len(x)):
            if x[i][:l-1] == x[j][:l-1] and x[i][l-1] != x[j][l-1]:
                r.append(x[i][:l-1]+sorted([x[j][l-1],x[i][l-1]]))
    return r

#寻找关联规则的函数
def find_rule(d, support, confidence):
    import time
    start = time.clock()
    result = pd.DataFrame(index=['support', 'confidence']) #定义输出结果

    support_series = 1.0*d.sum()/len(d) #支持度序列
    column = list(support_series[support_series &gt; support].index) #初步根据支持度筛选
    k = 0

    while len(column) &gt; 1:
        k = k+1
        print(u'\n正在进行第%s次搜索...' %k)
        column = connect_string(column, ms)
        print(u'数目：%s...' %len(column))
        sf = lambda i: d[i].prod(axis=1, numeric_only = True) #新一批支持度的计算函数

        #创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(list(map(sf,column)), index = [ms.join(i) for i in column]).T

        support_series_2 = 1.0*d_2[[ms.join(i) for i in column]].sum()/len(d) #计算连接后的支持度
        column = list(support_series_2[support_series_2 &gt; support].index) #新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []
        
        for i in column: #遍历可能的推理，如{A,B,C}究竟是A+B--&gt;C还是B+C--&gt;A还是C+A--&gt;B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j]+i[j+1:]+i[j:j+1])
        
        cofidence_series = pd.Series(index=[ms.join(i) for i in column2]) #定义置信度序列
        
        for i in column2: #计算置信度序列
            cofidence_series[ms.join(i)] = support_series[ms.join(sorted(i))]/support_series[ms.join(i[:len(i)-1])]
        
        for i in cofidence_series[cofidence_series &gt; confidence].index: #置信度筛选
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort(['confidence','support'], ascending = False) #结果整理，输出
    end = time.clock()
    print(u'\n搜索完成，用时：%0.2f秒' %(end-start))
    print(u'\n结果为：')
    print(result)
    
    return result

find_rule(d, support, confidence).to_excel('rules.xls')</code></pre>



测试数据集：[apriori\.txt](<https://spaces.ac.cn/usr/uploads/2015/07/3424358296.txt>)

运行结果：

[![apriori](<https://spaces.ac.cn/usr/uploads/2015/07/204981067.png>)](<https://spaces.ac.cn/usr/uploads/2015/07/204981067.png>)

apriori
