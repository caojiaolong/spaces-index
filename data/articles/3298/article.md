# 记录一次爬取淘宝/天猫评论数据的过程

> 作者：苏剑林 · 科学空间 · 2015-05-06
>
> 原文：<https://spaces.ac.cn/archives/3298>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

笔者最近迷上了数据挖掘和机器学习，要做数据分析首先得有数据才行。对于我等平民来说，最廉价的获取数据的方法，应该是用爬虫在网络上爬取数据了。本文记录一下笔者爬取天猫某商品的全过程，淘宝上面的店铺也是类似的做法，不赘述。<strong>主要是分析页面以及用Python实现简单方便的抓取。</strong>

笔者使用的工具如下

> <strong>Python 3</strong>——极其方便的编程语言。选择3\.x的版本是因为3\.x对中文处理更加友好。
> 
> 
> 
> <strong>Pandas</strong>——Python的一个附加库，用于数据整理。
> 
> 
> 
> <strong>IE 11</strong>——分析页面请求过程（其他类似的流量监控工具亦可）。
> 
> 
> 
> 剩下的还有<strong>requests,re</strong>，这些都是Python自带的库。

实例页面（美的某热水器）：[http://detail\.tmall\.com/item\.htm?id=41464129793](<http://detail.tmall.com/item.htm?id=41464129793>)

### 评论在哪里？

要抓取评论数据，首先得找到评论究竟在哪里。打开上述网址，然后查看源代码，发现里面并没有评论内容！那么，评论数据究竟在哪里呢？原来天猫使用了ajax加密，它会从另外的页面中读取评论数据。

这时候IE 11就发挥作用了（当然你也可以使用其他的流量监控工具），使用前，先打开上述网址，待页面打开后，清除一下IE 11的缓存、历史文件等，然后按<strong>F12</strong>，会出现如下界面

[![F12](<https://spaces.ac.cn/usr/uploads/2015/05/1032426268.png>)](<https://spaces.ac.cn/usr/uploads/2015/05/1032426268.png>)

F12

这时候点击绿色的三角形按钮，启动网络流量捕获（或者直接按F5），然后点击天猫页面中的“累计评价”：

[![捕获](<https://spaces.ac.cn/usr/uploads/2015/05/3158865010.png>)](<https://spaces.ac.cn/usr/uploads/2015/05/3158865010.png>)

捕获

出现如下结果

[![捕获结果](<https://spaces.ac.cn/usr/uploads/2015/05/4017499246.png>)](<https://spaces.ac.cn/usr/uploads/2015/05/4017499246.png>)

捕获结果

在URL下面出现很多网址，而评论数据正隐藏在其中！我们主要留意类型为“text/html”或者“application/json”的网址，经过测试发现，天猫的评论在下面这个网址之中

> http://rate\.tmall\.com/list\_detail\_rate\.htm?itemId=41464129793&spuId=296980116&sellerId=1652490016&order=3&currentPage=1&append=0&content=1&tagId=&posi=&picture=&ua=166UW5TcyMNYQwiAiwVQX1EeUR5RH5Cd0xiNGI%3D%7CUm5Ockt1SHxBe0B0SXNOdCI%3D%7CU2xMHDJxPk82UjVOI1h2VngRd1snQSJEI107F2gFfgRlAmRKakQYeR9zFGoQPmg%2B%7CVGhXd1llXGJfa1ZsV2NeZFljVGlLdUt2TXFOc0tyT3pHe0Z6QHlXAQ%3D%3D%7CVWldfS0SMgo3FysUNBonHyMdNwI4HStHNkVrPWs%3D%7CVmhIGCIWNgsrFykQJAQ6DzQAIBwiGSICOAM2FioULxQ0DjEEUgQ%3D%7CV25OHjAePgA0DCwQKRYsDDgHPAdRBw%3D%3D%7CWGFBET8RMQ04ACAcJR0iAjYDNwtdCw%3D%3D%7CWWBAED5%2BKmIZcBZ6MUwxSmREfUl2VmpSbVR0SHVLcU4YTg%3D%3D%7CWmFBET9aIgwsECoKNxcrFysSL3kv%7CW2BAED5bIw0tESQEOBgkGCEfI3Uj%7CXGVFFTsVNQw2AiIeJxMoCDQIMwg9az0%3D%7CXWZGFjhdJQsrECgINhYqFiwRL3kv%7CXmdHFzkXNws3DS0RLxciAj4BPAY%2BaD4%3D%7CX2ZGFjgWNgo1ASEdIxsjAz8ANQE1YzU%3D%7CQHtbCyVAOBY2Aj4eIwM%2FAToONGI0%7CQXhYCCYIKBMqFzcLMwY%2FHyMdKRItey0%3D%7CQntbCyULKxQgGDgEPQg8HCAZIxoveS8%3D%7CQ3paCiQKKhYoFDQIMggwEC8SJh8idCI%3D%7CRH1dDSMNLRIrFTUJMw82FikWKxUueC4%3D%7CRX5eDiAOLhItEzMOLhIuFy4VKH4o%7CRn5eDiAOLn5GeEdnW2VeYjQUKQknCSkQKRIrFyN1Iw%3D%3D%7CR35Dfl5jQ3xcYFllRXtDeVlgQHxBYVV1QGBfZUV6QWFZeUZ%2FX2FBfl5hXX1AYEF9XXxDY0J8XGBbe0IU&isg=B2E8ACFC7C2F2CB185668041148A7DAA&\_ksTS=1430908138129\_1993&callback=jsonp1994

是不是感觉长到晕了？不要紧，只需要稍加分析，就发现可以精简为以下部分

> http://rate\.tmall\.com/list\_detail\_rate\.htm?itemId=41464129793&sellerId=1652490016&currentPage=1

我们发现天猫还是很慷慨的，评论页面的地址是很有规律的（像京东就完全没规律了，随机生成。），其中itemId是商品id，sellerid是卖家id，currentPage是页面号。

### 怎么爬取？

费了一番周折，终于找到评论在哪里了，接下来是爬取，怎么爬取呢？首先分析一下页面规律。

[![页面格式](<https://spaces.ac.cn/usr/uploads/2015/05/1422469517.png>)](<https://spaces.ac.cn/usr/uploads/2015/05/1422469517.png>)

页面格式

我们发现页面数据是很规范的，事实上，它是一种被称为JSON的轻量级数据交换格式（大家可以搜索JSON），但它又不是通常的JSON，事实上，页面中的方括号&#91;&#93;里边的内容，才是一个正确的JSON规范文本。

下面开始我们的爬取，我使用Python中的requests库进行抓取，在Python中依次输入：



<pre><code class="language-python">import requests as rq
url='http://rate.tmall.com/list_detail_rate.htm?itemId=41464129793&amp;sellerId=1652490016&amp;currentPage=1'
myweb = rq.get(url)</code></pre>



现在该页面的内容已经保存在myweb变量中了，我们可以用myweb\.text查看文本内容。

接下来就是只保留方括号里边的部分，这需要用到正则表达式了，涉及到的模块有re。



<pre><code class="language-python">import re
myjson = re.findall('\"rateList\":(\[.*?\])\,\"tags\"',myweb.text)[0]</code></pre>



呃，这句代码什么意思？懂Python的读者大概都能读懂它，不懂的话，请先阅读一下相关的正则表达式的教程。上面的意思是，在文本中查找下面标签

> "rateList":&#91;\.\.\.&#93;,"tags"

找到后保留方括号及方括号里边的内容。为什么不直接以方括号为标签呢，而要多加几个字符？这是为了防止用户评论中出现方括号而导致抓取出错。

现在抓取到了myjson，这是一个标准的JSON文本了，怎么读取JSON？也简单，直接用Pandas吧。这是Python中强大的数据分析工具，用它可以直接读取JSON。当然，如果仅仅是为了读取JSON，完全没必要用它，但是我们还要考虑把同一个商品的每个评论页的数据都合并成一个表，并进行预处理等，这时候Pandas就非常方便了。



<pre><code class="language-python">import pandas as pd
mytable = pd.read_json(myjson)</code></pre>



现在mytable就是一个规范的Pandas的DataFrame了：

[![mytable1](<https://spaces.ac.cn/usr/uploads/2015/05/3529976756.png>)](<https://spaces.ac.cn/usr/uploads/2015/05/3529976756.png>)

mytable1

[![mytable2](<https://spaces.ac.cn/usr/uploads/2015/05/2768477366.png>)](<https://spaces.ac.cn/usr/uploads/2015/05/2768477366.png>)

mytable2

如果有两个表mytable1和mytable2需要合并，则只要



<pre><code class="language-python">pd.concat([mytable1, mytable2], ignore_index=True)</code></pre>



等等。更多的操作请参考Pandas的教程。

最后，要把评论保存为txt或者Excel（由于存在中文编码问题，保存为txt可能出错，因此不妨保存为Excel，Pandas也能够读取Excel文件）



<pre><code class="language-python">mytable.to_csv('mytable.txt')
mytable.to_excel('mytable.xls')
</code></pre>



### 一点点结论

让我们看看一共用了几行代码？



<pre><code class="language-python">import requests as rq
import re
import pandas as pd
url='http://rate.tmall.com/list_detail_rate.htm?itemId=41464129793&amp;sellerId=1652490016&amp;currentPage=1'
myweb = rq.get(url)
myjson = re.findall('\"rateList\":(\[.*?\])\,\"tags\"',myweb.text)[0]
mytable = pd.read_json(myjson)mytable.to_csv('mytable.txt')
mytable.to_excel('mytable.xls')
</code></pre>



<strong>九行！十行不到，我们就完成了一个简单的爬虫程序，并且能够爬取到天猫上的数据了！是不是跃跃欲试了？</strong>

当然，这只是一个简单的示例文件。要想实用，还要加入一些功能，比如找出评论共有多少页，逐页读取评论。另外，批量获取商品id也是要实现的。这些要靠大家自由发挥了，都不是困难的问题，本文只希望起到抛砖引玉的作用，为需要爬取数据的读者提供一个最简单的指引。

其中最困难的问题，应该是大量采集之后，有可能被天猫本身的系统发现，然后要你输入验证码才能继续访问的情况，这就复杂得多了，解决的方案有使用代理、使用更大的采集时间间隔或者直接OCR系统识别验证码等等，笔者也没有很好的解决办法。
