# 【备忘】访问Google的方法（更新）

> 作者：苏剑林 · 科学空间 · 2014-06-04
>
> 原文：<https://spaces.ac.cn/archives/2611>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<div data-spaces-format="html-v1">
<p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">6月13日：更新了一个新的可用IP，不知道能够用多久。</font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">以前大家顶多看到利用这个技巧访问facebook、youtube之类的网站，现在无奈到连Google都得用这个方法访问了。</p><p data-spaces-tag="p" data-spaces-attrs="{}">近日，笔者发现直接输入<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://www.google.com.hk&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://www.google.com.hk">http://www.google.com.hk</a>无法访问Google搜索，要知道对于学术来说，没有Google是多么严重的事情，很多有用的学术资料，尤其是外文资料，都得靠Google来搜。主观性来说，在学术方面，百度不可能赶得上Google，望其项背都不可能。</p><p data-spaces-tag="p" data-spaces-attrs="{}">上网搜索了一下，发现这并不是我一个人的问题，甚至都已经传出“谷歌全面退出中国？香港域名google.com.hk打不开！”之类的猜测了。当然，不管事实如何，我还是得使用Google，不能直接访问，就得另想办法。其实这方法也是老生常谈了。直接访问http://www.google.com.hk不行，但是可以直接访问某些Google的IP，比如<del data-spaces-tag="del" data-spaces-attrs="{}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://203.208.46.177&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://203.208.46.177">203.208.46.177</a>、<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://203.208.46.178&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://203.208.46.178">203.208.46.178</a>、<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://203.208.46.146&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://203.208.46.146">203.208.46.146</a></del>、<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://173.194.127.19&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://173.194.127.19">173.194.127.19</a>等。找到对应的IP后，修改C:\Windows\System32\drivers\etc目录下的hosts文件即可（在Windows 8中，不能直接用记事本打开修改，需要把这个文件复制到非系统目录中，修改后再复制回来，覆盖原文件。），在hosts下加入</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">173.194.127.19  www.google.com.hk<br data-spaces-tag="br" data-spaces-attrs="{}">
173.194.127.19  google.com.hk<br data-spaces-tag="br" data-spaces-attrs="{}">
173.194.127.19  www.google.com<br data-spaces-tag="br" data-spaces-attrs="{}">
173.194.127.19  google.com</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">这样就可以用<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://www.google.com.hk&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://www.google.com.hk">http://www.google.com.hk</a>访问Google了。不管搜索引擎如何变化，使用Google是不会变化的。</p>
</div>
