# Cool Papers更新：简单适配Zotero Connector

> 作者：苏剑林 · 科学空间 · 2025-08-25
>
> 原文：<https://spaces.ac.cn/archives/11250>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

很早之前就有读者提出希望可以给[Cool Papers](<https://papers.cool/>)增加导入Zotero的功能，但由于笔者没用Zotero，加上又比较懒，所以一直没提上日程。这个周末刚好有点时间，研究了一下，做了个简单的适配。

## 单篇导入

首先，我们需要安装[Zotero](<https://www.zotero.org/>)（这是废话），然后需要给所用浏览器安装[Zotero Connector](<https://www.zotero.org/download/connectors>)插件。安装完成后，我们访问Cool Papers的单篇论文页面，如 [https://papers\.cool/arxiv/2104\.09864](<https://papers.cool/arxiv/2104.09864>) 或 [https://papers\.cool/venue/2024\.naacl\-long\.431@ACL](<https://papers.cool/venue/2024.naacl-long.431@ACL>) ，然后点击Zotero Connector的图标，就会自动把论文导入了，包括PDF文件。

[![单篇论文导入到Zotero](<https://spaces.ac.cn/usr/uploads/2025/08/2053963362.png>)](<https://spaces.ac.cn/usr/uploads/2025/08/2053963362.png>)

单篇论文导入到Zotero

保存的信息包括论文标题、作者、摘要、日期、所属主类别（arXiv）或所属会议（会议论文）。这是通过在网页头部嵌入Metadata实现的，不需要用户做额外配置，缺点是只支持单页面单论文的导入。

## 批量导入

如果想要支持批量导入，那就只能通过[Translator](<https://github.com/zotero/translators>)来实现了，它是实现复杂导入的必要条件。当我们访问arXiv官网时，Zotero Connector能够自动识别多篇论文让我们选择导入，就是通过Translator实现的，只不过arXiv的Translator已经内置到Zotero中，而Cool Papers的自然要自己写了。

Translator实际上就是一段JS代码，笔者已经写好并放到了Github上（[链接](<https://github.com/bojone/papers.cool/blob/main/Zotero/CoolPapers.js>)），读者只需要将它保存下来，并放到Zotero的translators目录中（MacOS是“用户/Zotero/translators”），重启Zotero就生效了。（注意，Translator是配置到Zotero而不是Zotero Connector中的。）

这时候我们访问列表页，如 [https://papers\.cool/arxiv/cs\.AI](<https://papers.cool/arxiv/cs.AI>) 或 [https://papers\.cool/venue/ACL\.2025](<https://papers.cool/venue/ACL.2025>) ，正常来说Zotero Connector的图标就会变成文件夹形状，点击它，就会出现如下的多选框：

[![批量导入到Zotero](<https://spaces.ac.cn/usr/uploads/2025/08/168196517.png>)](<https://spaces.ac.cn/usr/uploads/2025/08/168196517.png>)

批量导入到Zotero

这里显示的论文，并不是当前列表页的所有论文，而是当前页面下我们曾点击过“&#91;PDF&#93;”或“&#91;Kimi&#93;”的论文，这意味着我们可能对这些论文感兴趣，因此可能会需要进一步收藏到Zotero，这便是Cool Papers适配Zotero的逻辑。

## 文章小结

对Cool Papers做了一些简单的调整，并写了一个Translator，从而可以配合Zotero Connector，将Cool Papers的论文导入到Zotero上。
