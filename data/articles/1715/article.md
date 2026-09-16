# 又折腾网络了\.\.\.\.\.\.

> 作者：苏剑林 · 科学空间 · 2012-09-25
>
> 原文：<https://spaces.ac.cn/archives/1715>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

今晚主要干了两件事情：

<strong>1、实现了在windows 8的情况下，把自己的笔记本当做wifi的信号发射点，共享校园网</strong>（即“笔记本 wifi 热点”那技术，不知道这样会不会折损电脑寿命呀）。主要方法如下：
1\.1、安装\.net 3\.5，安装方法：

> 挂载windows 8的安装光盘，
> 然后右击开始菜单(Win \+ X)的左下角，选择－命令提示符(管理员)，接着然后输入如下命令：
> dism\.exe /online /enable\-feature /featurename:NetFX3 /Source:F:\\sources\\sxs
> 其中F是安装光盘的驱动器符号。
> 
> 
> 
> 接下来是漫长等待，估计会有十多分钟，就会提示安装进度100%了。

1\.2、安装Connectify软件，直接到官网下载最新的精简版就行，有兴趣可以购买专业版。安装后需要重新启动，然后简单地配置一下就行了，不再细说。

<strong>附：</strong>
顺便提一下，我也试过国内的wifi共享精灵，但是发现它会卡在“查找当前配置信息”那里，这折腾了我几个小时，最终还是没有解决\.\.\.所以还是用回外国软件了。

还有网络上一个方法是依次通过管理员的cmd运行：

> netsh wlan set hostednetwork
> mode=allow ssid=您想要的无线网络的名称 key=您想要设置的密码
> netsh wlan start hostednetwork

然后在更改网络适配器那里共享。用这个方法我的手机可以寻找到信号，但是在连接的时候寻找ip失败，原因不明\.\.\.\.\.\.

<strong>2、实现了在ubuntu 12下连接校园网。</strong>

锐捷客户端一般只是用在windows下，而不能用在linux下，所以在ubuntu中需要用mentohust来代替它。可是我这个linux新手一开始却怎么也搞不定在ubuntu安装mentohust。

经过一阵子研究，也算是学会了。首先下载mentohust\_0\.3\.4\-1\_amd64\.deb备用，下载地址：
[http://code\.google\.com/p/mentohust/](<http://code.google.com/p/mentohust/>)

在ubuntu在启动“终端”，由于之前没有安装过软件，所以先要设置超级管理员（root）密码，在终端输入：

> sudo passwd root

然后重复输入密码就可以了。
然后输入

> su

来启动超级管理员账户。

进入到mentohust\_0\.3\.4\-1\_amd64\.deb所在目录，运行

> dpkg \-i mentohust\_0\.3\.4\-1\_amd64\.deb

安装完成！

最后通过

> sudo mentohust

来启动这个程序，配置一下网卡和帐号密码就行了。

如果要修改帐号密码，就通过

> sudo mentohust \-u 学号 \-p 密码

来修改
