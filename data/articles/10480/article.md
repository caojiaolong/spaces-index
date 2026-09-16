# Cool Papers浏览器扩展升级至v0\.2\.0

> 作者：苏剑林 · 科学空间 · 2024-10-16
>
> 原文：<https://spaces.ac.cn/archives/10480>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

年初，我们在[《更便捷的Cool Papers打开方式：Chrome重定向扩展》](<https://spaces.ac.cn/archives/9978>)中发布了一个Chrome浏览器插件（Cool Papers Redirector v0\.1\.0），可以通过右击菜单从任意页面中重定向到Cool Papers中，让大家更方便地获取Kimi对论文的理解。前几天我们把该插件升级到了v0\.2\.0，并顺利上架到了Chrome应用商店中，遂在此向大家推送一下。

## 更新汇总

相比旧版v0\.1\.0，当前版v0\.2\.0的主要更新内容如下：

> 1、右键菜单跳转改为在新标签页打开；
> 
> 
> 
> 2、右键菜单支持同时访问多个论文ID；
> 
> 
> 
> 3、右键菜单支持PDF页面；
> 
> 
> 
> 4、右键菜单新增更多论文源（arXiv、OpenReview、ACL、IJCAI、PMLR）；
> 
> 
> 
> 5、右键菜单在搜索不到论文ID时，转入站内搜索（即划词搜索）；
> 
> 
> 
> 6、在某些网站的适当位置插入快捷跳转链接（arXiv、OpenReview，ACL）。

下面对更新内容展开介绍一下。

## 右键菜单

前面五点更新都是关于右键菜单的，这也是Cool Papers扩展程序最主要的功能，它的工作逻辑是：依次检查用户右击鼠标时的所选文字、超链接、页面路径中是否包含论文ID，如果包含则跳转到Cool Papers相应页面。

v0\.1\.0版写在Cool Papers建立之初，当时只支持识别单一的arXiv论文ID，现在支持识别多个论文ID在同一页打开，并且支持arXiv、OpenReview、ACL、IJCAI、PMLR五个论文源（后面再陆续增加，这几个是图方便先补充上去）。还有一点比较关键的改进，就是之前在PDF页面是无法进行跳转的，比如你访问了arXiv的PDF如“https://arxiv/org/pdf/xxxx\.xxxxx”，那么右击时就会发现“Redirect to Cool Papers”根本不显示，这是旧版遗留的一个问题，新版把它解决了。

最后，如果在所选文字、超链接、页面路径中都检测不到论文ID，那么就直接以所选文字为Query，跳转到Cool Papers的站内搜索页，这时候就变成了一个划词搜索功能了。

## 页面链接

最后一点变化，是新引入的除右键菜单外的功能，它在某些页面的适当位置插入了可以跳转到Cool Papers的超链接，这些页面包括arXiv列表页、arXiv详情页、OpenReView全站、ACL详情页。希望这个改动不会影响大家浏览原本的页面，如有打扰，请及时指正。

[![arXiv列表页的快捷链接](<https://spaces.ac.cn/usr/uploads/2024/10/3920073027.png>)](<https://spaces.ac.cn/usr/uploads/2024/10/3920073027.png>)

arXiv列表页的快捷链接

[![arXiv详情页的快捷链接](<https://spaces.ac.cn/usr/uploads/2024/10/1893966955.png>)](<https://spaces.ac.cn/usr/uploads/2024/10/1893966955.png>)

arXiv详情页的快捷链接

[![OpenReview的快捷链接](<https://spaces.ac.cn/usr/uploads/2024/10/3091733289.png>)](<https://spaces.ac.cn/usr/uploads/2024/10/3091733289.png>)

OpenReview的快捷链接

[![ACL详情页的快捷链接](<https://spaces.ac.cn/usr/uploads/2024/10/3569796965.png>)](<https://spaces.ac.cn/usr/uploads/2024/10/3569796965.png>)

ACL详情页的快捷链接

## 文章小结

本文根据作者自己的使用经验简单升级了一下Cool Papers Redirector浏览器拓展，欢迎大家测试。如果有更多改进意见，欢迎留言指出。
