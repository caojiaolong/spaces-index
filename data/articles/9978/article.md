# 更便捷的Cool Papers打开方式：Chrome重定向扩展

> 作者：苏剑林 · 科学空间 · 2024-02-02
>
> 原文：<https://spaces.ac.cn/archives/9978>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

## 一些铺垫

自Cool Papers上线以来，很多用户就建议笔者加入搜索功能，后面也确实在前端用JS简单做了个页面内搜索，解决了部分用户的需求，但仍有读者希望引入更完整的全局搜索。诚然，笔者理解这个需求确实是存在，但Cool Papers的数据是逐天累积的，目前才上线一个月，论文数并不多，建立一个大而全的搜索引擎意义不大，其次做搜索也不是笔者的强项，以及并没有很好的利用LLM优化搜索的思路，等等。总而言之，暂时没有条件实现一个全面而又有特色的搜索，所以不如不做（也欢迎大家在评论区集思广益）。

后来，经过和同事讨论，想出了一个“借花献佛”的思路——写一个Chrome的重定向扩展，可以从任意页面重定向到Cool Papers。这样我们可以用任意方式（如Google搜索或者直接Arxiv官方搜索）找到Arxiv上的论文，然后右击一下就转到Cool Papers了。前两周这个扩展已经在Chrome应用商店上线，上周服务器配合做了一些调整，如今大家可以尝试使用了。

> <strong>扩展地址：[Cool Papers Redirector](<https://chromewebstore.google.com/detail/cool-papers-redirector/goopbgbhhhionbpfdkkblnbeopfmlihm>)</strong>

## 使用方式

扩展的使用很简单，在Chrome安装成功后，以任意方式搜索论文，然后右击对应的地方（有时候空白之处也可以），会多处一个“Redirect to Cool Papers”的选项，点击该选项，浏览器会自动检测“所选文字”、“所选链接”或者“网站地址”可能存在的论文ID（检测到即停止），并自动跳转到Cool Papers对应的页面，效果如下：

[![右击重定向效果1](<https://spaces.ac.cn/usr/uploads/2024/02/2162922427.jpg>)](<https://spaces.ac.cn/usr/uploads/2024/02/2162922427.jpg>)

右击重定向效果1

[![右击重定向效果2](<https://spaces.ac.cn/usr/uploads/2024/02/835676026.jpg>)](<https://spaces.ac.cn/usr/uploads/2024/02/835676026.jpg>)

右击重定向效果2

## 放开历史

事实上，实现这样一个重定向扩展还是比较简单的，只需要简单的HTML\+JS就行，当然这前提是得益于GPT4和Kimi的双重指导。

既然扩展的开发不是难度，那么就剩下一个难题——全面开放历史论文访问所带来的压力。很多用户已经留意到，此前Cool Papers就可以通过<code>https://papers.cool/arxiv/&lt;paper_id&gt;</code>访问特定的论文，但仅限于数据库已有的论文，不存在的则会显示Not Found。而如果普及Cool Papers Redirector的使用的话，则必然要放开所有Arxiv历史论文的爬取和访问，否则十篇论文有八篇都是Not Found，那Redirector就几乎没有意义了。

为了在放开历史论文的同时，保证Cool Papers的主线——刷当天最新论文——的正常进行，笔者对Arxiv爬取队列和Kimi对话队列采取了多优先级设计（目前分三级）。首先，“超级VIP”是我们内部权限，通过填入正确的Magic Token进行解锁，这跟普通访客无关；其次，是当天论文优先，Arxiv队列在到点之后，优先获取当天论文列表，然后才处理历史论文请求，同理Kimi队列会优先处理当天论文，当天论文自动“插队”到历史论文前面；最后，历史论文的爬取和Kimi都是第三优先级。

这样一来，基本能够确保不影响当天论文的阅读，并且将闲余资源用于历史论文的处理。

## 其他更新

相比在[《写了个刷论文的辅助网站：Cool Papers》](<https://spaces.ac.cn/archives/9907>)和[《新年快乐！记录一下 Cool Papers 的开发体验》](<https://spaces.ac.cn/archives/9920>)刚发布的时候，如今经过一个月的改进，Cool Papers在功能上已经完善了很多（当然界面上一如既往地简陋），除了本文的开放所有历史论文访问外，其他改动还包括：

> 1、底部增加Bar，可以搜索论文、查看/导出阅读记录等，当然这些功能都仅限页面内；
> 
> 
> 
> 2、支持指定日期的论文列表，可在首页分类右端的日历图标选日期，也可以点击列表页的日期文字来选择日期；
> 
> 
> 
> 3、换用PDF\.js预览PDF，支持移动端刷论文，同时优化PDF文字解析效果，提高&#91;Kimi&#93;的质量；
> 
> 
> 
> 5、在第4个Bar按钮可以将&#91;Kimi&#93;切换为<strong>英文</strong>输出，方便外国网友，或者需要对照英文原文的网友；
> 
> 
> 
> 6、一大堆微小的Bug修复～

总的来说，界面上看起来没怎么变化，但实际上几乎每天都或多或少有些改进，源码相比初版，可谓是“面目全非”了。接下来要做的事情，可能是新增其他论文源，如[OpenReview](<https://openreview.net/>)、[bioRxiv](<https://www.biorxiv.org/>)等，敬请大家期待～

## 文章小结

本文分享了通过Chrome重定向扩展来打开Cool Papers的新方式，并简单回顾了一下近来Cool Papers的变化。
