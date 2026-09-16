# 【备忘】维基百科与DNSCrypt

> 作者：苏剑林 · 科学空间 · 2015-05-30
>
> 原文：<https://spaces.ac.cn/archives/3327>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

> 中文维基百科的域名zh\.wikipedia\.org于5月19日被关键字屏蔽和DNS污染，目前从中国已无法访问中文维基百科，中文维基百科的域名也无法解析出正确的IP地址，而英文维基百科目前未受影响，可以正常访问。
> 
> 
> 
> 来自“月光博客”：[http://www\.williamlong\.info/archives/4240\.html](<http://www.williamlong.info/archives/4240.html>)
> 
> 
> 
> 类似的新闻还有：[http://www\.freebuf\.com/news/68011\.html](<http://www.freebuf.com/news/68011.html>)

先是Google被屏蔽，现在轮到维基百科。当然，屏蔽维基百科也不是第一次了，我想也不会是最后一次。其实，在这之前，我还不知道什么叫DNS污染，我觉得，我的翻墙技术、代理技术等，都是我们泱泱大国屏蔽这些优秀网站之后，我等平民被逼无奈的结果。

很想吐槽一句，还让不让搞学术的人活了？难道那个跟垃圾同构的百度百科，能满足我们的需要吗？负责屏蔽的部门，难道就没有一人是搞学术的吗？（激动了点，失态了，抱歉。）

当然，天朝还算发了善心，屏蔽维基百科的方式，远没有屏蔽Google来得严重。我们只要对DNS用DNSCrypt进行加密，就可以访问其HTTPS版本了。使用教程挺简单的，基本上装了就可以直接用了，也没有什么教程。

DNSCrypt：[http://www\.opendns\.com/about/innovations/dnscrypt/](<http://www.opendns.com/about/innovations/dnscrypt/>)

中文维基百科：[https://zh\.wikipedia\.org](<https://zh.wikipedia.org>)

一起维基吧。
