# 【备忘】在自己的电脑上搭建服务器

> 作者：苏剑林 · 科学空间 · 2012-07-19
>
> 原文：<https://spaces.ac.cn/archives/1665>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

宇宙驿站维修期间，BoJone曾经想过用自己的电脑来搭建服务器，建立一个临时页面。但后来发现经常开着电脑不大好，就没有这样做了。不过<u>如何在自己的电脑上搭建服务器</u>，还是值得笔记一下的。

BoJone还在使用WinXP专业版系统，最标准的方法当然是使用IIS，可以一气呵成。但是考虑到IIS需要配置挺多东西的，所以就没有这样做了。所以自己在网上下载一些小软件，“拼凑”成了一个临时服务器。这样的方法也能够很方便地应用到各个Windows系统。

<strong>软件列表：</strong>

&#91;1&#93;<strong>花生壳</strong>：[http://www\.oray\.com/peanuthull/](<http://www.oray.com/peanuthull/>)

<strong>BoJone使用的是ADSL上网，IP是动态的，所以需要一个固定的域名来解析我的动态IP，只有这样才方便将Spaces\.ac\.cn指向我自己的电脑。使用很简单，注册、下载客户端登陆就行了。</strong>

&#91;2&#93;<strong>Abyss Web Server</strong>：[http://www\.aprelium\.com/abyssws/](<http://www.aprelium.com/abyssws/>)

<strong>这是用来对服务器进行基本配置的软件，换句话说让外界能够访问你的电脑，其中包括指定路径、安全设置、主页文件等等，不过它只提供了基本的HTTP/1\.1, CGI scripts和Server Side，目前是英文版本的。</strong>

<strong>要注意的是，如果你是通过路由器上网的，那么需要对路由器进行一定的设置才能够成功让别人浏览你的网页（具体请看帮助文件）。直接Modem拨号的就可以忽略这一点。</strong>

<strong>以下是它的界面之一：</strong>

[![Abyss Web Server](<https://spaces.ac.cn/usr/uploads/2012/07/3879339217.png>)](<https://spaces.ac.cn/attachment/1666/>)

Abyss Web Server

&#91;3&#93;<strong>PHPnow</strong>：[http://phpnow\.org/](<http://phpnow.org/>)

<strong>当前最流行的网页语言当属php了，本软件用来提供php环境。当然官方还提供了一些插件，能够简单地支持Java、Asp、Asp\.net等语言。</strong>
<strong>所使用的软件就只有这么多了，这些软件的使用方法也都很简单，就不一一介绍了。剩下的是域名指向（没有的可以跳过）等基本建站操作了。由于本文只是备忘所用，不详述^\_^有不懂的朋友可以私下跟我讨论</strong>
