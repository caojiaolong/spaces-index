# BLOG评论故障修复，部分数据丢失

> 作者：苏剑林 · 科学空间 · 2009-07-23
>
> 原文：<https://spaces.ac.cn/archives/31>
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
<p data-spaces-tag="p" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Red&quot;}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">今天一早起来，兴致勃勃地发表着日志，却发现评论用不了了，侧边栏也故障了</strong>。</span><br data-spaces-tag="br" data-spaces-attrs="{}">
如图：<br data-spaces-tag="br" data-spaces-attrs="{}">
</p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 100%&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2009/07/20090723132232.JPG&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2009/07/20090723132232.JPG" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;&quot;, &quot;src&quot;: &quot;/usr/uploads/2009/07/20090723132232.JPG&quot;, &quot;style&quot;: &quot;max-width:100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2009/07/20090723132232.JPG" alt=""></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}"></p></div></div><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><u data-spaces-tag="u" data-spaces-attrs="{}">紧接着，尝试更新缓存、重新安装、换空间，都无法解决。</u></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">于是只好到官方网站求助，谁知道发现，<strong data-spaces-tag="strong" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color:Green&quot;}">只要删除cookies就好了</span></strong>（真冤枉，害我丢失了数据）</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">为了一劳永逸，按以下方式进行了修改：</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">解决方法：</strong><br data-spaces-tag="br" data-spaces-attrs="{}">
打开class/cls_article.asp找到</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">Ts_Content = Split(Split(Split(Ts, "|-|")(1), "|\$|")(1), "|+|")(0)</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">改成</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">Ts_Content = Split(Split(Split(Ts, "|-|")(1), "|\$|")(0), "|+|")(0)</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">试验过，删除这行也没有什么影响。</p>
</div>
