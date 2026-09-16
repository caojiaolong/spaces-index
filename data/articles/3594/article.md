# 简单的迅雷VIP账号获取器（Python）

> 作者：苏剑林 · 科学空间 · 2016-01-20
>
> 原文：<https://spaces.ac.cn/archives/3594>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在Windows工作的时候，经常会用迅雷下载东西，如果速度慢或者没资源，尤其是一些比较冷门的视频，迅雷的VIP会员服务总能够帮上大忙。后来无意间发现了有个“迅雷VIP账号获取器”的软件，可以获取一些临时的VIP账号供使用，这可是个好东西，因为开通迅雷会员虽然不贵，但是我又不经常下载，所以老感觉有点浪费，而有了这个之后，我随时下点东西都可以免费用了。

[![简单的迅雷VIP账号获取器](<https://spaces.ac.cn/usr/uploads/2016/01/3260771467.png>)](<https://spaces.ac.cn/usr/uploads/2016/01/3260771467.png>)

简单的迅雷VIP账号获取器

最近转移到了Mac上，而Mac也有迅雷，但那个账号获取器是exe的，不能在Mac运行。本以为获取器的构造会很复杂，谁知道，经过抓包研究，发现那个账号获取器的原理极其简单，说白了，就是一个简单的爬虫，以下这两个网站提供账号，它就到相应的抓取账号而已：

http://yunbo\.xinjipin\.com/
http://www\.fenxs\.com

据此，我也用Python简单写了一个，主要是方便我在Mac使用。读者如果有需要，也可以下载使用，代码兼容2\.x和3\.x的版本。主要的库是requests和re，pandas和sys的使用只不过是为了更加人性化。本来想用Tkinter写一个简单的GUI的，但是想想看，还是没必要了～～



<pre><code class="language-python"># -*- coding:utf-8 -*-
'''
2016.01.21更新：修改了正则表达式的写法，使得更加通用；增加了一个账号来源。
'''
import requests as rq
import re

def get1():
    source = 'http://yunbo.xinjipin.com/articlelist/?33.html'
    web = rq.get(source).text
    url = re.findall('&lt;li&gt;&lt;p&gt;&lt;a href=\"(.*?)\" title=', web)[0]
    web = rq.get('http://yunbo.xinjipin.com%s'%url)
    web.encoding = 'gb2312'
    web = web.text
    return re.findall(u'迅雷.*?([a-zA-z0-9\:]+?)[密码]+?([a-zA-z0-9]+?)&lt;/div&gt;', web)

def get2():
    source = 'http://www.fenxs.com'
    web = rq.get(source).text
    url = re.findall(u'&lt;h2&gt;&lt;a href=\"(.*?)\" title=.*?迅雷会员账号分享.*?&lt;/a&gt;&lt;/h2&gt;', web)[0]
    web = rq.get(url).text
    return re.findall(u'迅雷.*?([a-zA-z0-9\:]+?)[密码]+?([a-zA-z0-9]+?)&lt;br /&gt;', web)

def get3():
    source = 'http://xlfans.com'
    web = rq.get(source).text
    url = re.findall(u'&lt;h2&gt;&lt;a href=\"(.*?)\" title=.*?迅雷会员账号分享.*?&lt;/a&gt;&lt;/h2&gt;', web)[0]
    web = rq.get(url).text
    return re.findall(u'迅雷.*?([a-zA-z0-9\:]+?)[密码]+?([a-zA-z0-9]+?)&lt;br /&gt;', web)

if __name__ == '__main__':
    import pandas as pd #方便输出显示
    import sys #判断系统版本

    print u'\n============简单的迅雷账号获取器============\n           By http://kexue.fm\n'
    while True:
        if sys.version_info[0] &lt; 3:
            s = raw_input(u'请选择数据源(输入s1或s2或s3，输入其他则退出): ')
        else:
            s = input(u'请选择数据源(输入s1或s2或s3，输入其他则退出): ')
        if s == 's1':
            print pd.DataFrame(get1(), columns=[u'账号', u'密码'])
        elif s == 's2':
            print pd.DataFrame(get2(), columns=[u'账号', u'密码'])
        elif s == 's3':
            print pd.DataFrame(get3(), columns=[u'账号', u'密码'])
        else:
            break
</code></pre>
