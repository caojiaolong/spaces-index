# 近乎完美地解决MathJax与Marked的冲突

> 作者：苏剑林 · 科学空间 · 2024-08-26
>
> 原文：<https://spaces.ac.cn/archives/10332>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

在[《让MathJax更好地兼容谷歌翻译和延时加载》](<https://spaces.ac.cn/archives/10320>)我们提到[Cool Papers](<https://papers.cool/>)加入了MathJax来解析LaTeX公式，不过万万没想到引发了诸多兼容性问题，虽然部分问题纯粹是笔者的强迫症作祟，但一个尽可能完美的解决方案终究是让人赏心悦目的，所以还是愿意在上面花一点心思。

上一篇文章我们已经解决了MathJax与谷歌翻译、延时加载的兼容性，这篇文章我们则来解决MathJax与Marked的冲突。

## 问题简述

Markdown是一种轻量级标记语言，允许人们使用易读易写的纯文本格式编写文档，可谓是目前最流行的写作语法之一，Cool Papers中的&#91;Kimi&#93;功能，基本上也是按照Markdown语法输出。然而。Markdown并不是直接面向浏览器的语言，面向浏览器的语言叫做HTML，所以在展示给用户之前，有一个Markdown转HTML的过程（渲染）。

Markdown转HTML也分为两种方式，一种是在后端有服务器将Markdown转好HTML再发送给用户，一种是用户接受到Markdown后再在前端由浏览器转为HTML，本文的例子主要用于后者，但原则上相同思路应该也能简单修改后用于前者。在前端对Markdown进行转换的库有很多，Cool Papers用的是[Marked](<https://github.com/markedjs/marked>)，这是一个比较轻量级的选择。

Marked渲染Markdown的方法很简单，直接对字符串<code>marked.parse</code>一下就行，此时再配合上一节介绍的[MathJax](<https://spaces.ac.cn/archives/10320#%E5%85%AC%E5%BC%8F%E6%B8%B2%E6%9F%93>)，就可以对LaTeX代码进行解析了。然而，Markdown与LaTeX在语法上有一些相交之处，所以Marked可能会先按照Markdown规则对LaTeX代码（如果有的话）进行转换，于是后面的MathJax无法获取原始的LaTeX代码，导致渲染失败。

一个可复现的代码如下：



<pre><code class="language-html">&lt;div id="content"&gt;&lt;/div&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"&gt;&lt;/script&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;
&lt;script&gt;
    var div = document.getElementById('content');
    div.innerHTML = marked.parse('**cannot** render: \\(a^2 + b^2\\), **can** render: \\\\(c^2 + d^2\\\\)');
    MathJax.Hub.Typeset(div);
&lt;/script&gt;</code></pre>



## 已有方案

值得一提的是，流行的博客框架Hexo默认也是用Marked来渲染Markdown，所以如果我们搜索“MathJax Marked 冲突”，能找到不少资料，大部份都是以Hexo为背景展开的，其中总结的比较详细的一篇是[《调教Hexo&#91;2&#93;——Hexo与Mathjax的冲突及解决方案》](<https://www.lizhechen.com/2017/03/08/Hexo%E4%B8%8EMathjax%E7%9A%84%E5%86%B2%E7%AA%81%E5%8F%8A%EF%BC%88%E9%83%A8%E5%88%86%EF%BC%89%E8%A7%A3%E5%86%B3/>)，它将解决思路总结为以下四种：

> 1、<strong>手动转义</strong>：即在写公式时不要写正确的LaTeX代码，而是写“被Marked渲染后才正确”的LaTeX代码，比如原本LaTeX代码是双斜杠\\\\，经过Marked后会变成单斜杆\\，所以干脆一开始就写四斜杆\\\\\\\\，经过Marked就变成双斜杠\\\\了；
> 
> 
> 
> 2、<strong>保护公式</strong>：这个思路就更简单了，利用Marked不会渲染代码的特性，通过代码块标记来将公式保护起来，经过Marked渲染后再把公式提取出来用MathJax解析，如[《解决 MathJax 与 Markdown 的冲突》](<https://liam0205.me/2015/09/09/fix-conflict-between-mathjax-and-markdown/>)，该思路的问题就是比较容易跟正常代码块混淆；
> 
> 
> 
> 3、<strong>更换引擎</strong>：换用更好地支持Markdown与LaTeX混排的渲染引擎，比如在Hexo下通常推荐Pandoc，如[《解决 Hexo 和 Mathjax 的冲突》](<https://piggerzzm.github.io/2019/04/22/%E8%A7%A3%E5%86%B3Hexo%E5%92%8CMathjax%E7%9A%84%E5%86%B2%E7%AA%81/>)，但Pandoc是个后端渲染引擎，而对于前端渲染，笔者并没有找到更好的替代方案；
> 
> 
> 
> 4、<strong>修改引擎</strong>：就是修改Marked的代码，让它不要渲染某些LaTeX代码，从而一定程度上解决问题，如[《Hexo中Marked\.js与MathJax共存问题》](<https://qinjiangbo.com/solution-to-coexistance-of-markerjs-and-mathjax.html>)，这需要我们总结出一些容易被Marked误渲染的规则来逐一处理。

第1、第2个方案需要人为修改公式代码，但Cool Papers的公式是Kimi生成的，不能修改，所以基本上可以否定；由于笔者也没找到更好的Markdown前端渲染引擎，所以方案3也被否定了；方案4虽然一定程度上能解决问题，但过于规则化，不够优雅，而且也只能“头痛医头，脚痛医脚”，没法判断是否还有遗漏的规则没处理好。

## 逆向思路

事实上，这个问题有一个非常简单的解决方案：这本质上就是先Marked后MathJax带来的语法冲突，那我们不反过来，先用MathJax渲染公式，然后再用Marked来渲染Markdown呢？因为MathJax能比较严格地识别出数学公式，并且渲染结果几乎不会再出现Markdown语法，所以先MathJax后Marked可以从根本上解决冲突。

参考代码如下：



<pre><code class="language-html">&lt;div id="content"&gt;&lt;/div&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"&gt;&lt;/script&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;
&lt;script&gt;
    var div = document.getElementById('content');
    div.innerHTML = '**can** render: \\(a^2 + b^2\\)';
    MathJax.Hub.Queue(
        ['Typeset', MathJax.Hub, div],
        function() {
            div.innerHTML = marked.parse(div.innerHTML);
        }
    );
&lt;/script&gt;</code></pre>



## 强迫之症

上述代码的最终显示效果已经是我们所期望的了，但对于有强迫症的读者来说还差点意思，它有两个小缺陷。

第一个缺陷，是它会把原始Markdown文本先显示出来，隔一小会（取决于渲染速度）才会显示最终的渲染效果，注意原始Markdown直接输出到浏览器中，其效果是近乎乱码的，也就是说用户先看到一个近乎乱码的页面，一小会后才看到正式页面，这会影响阅读体验。为此我们可以另外创建一个元素用来渲染，渲染完后才赋值给当前页面：



<pre><code class="language-html">&lt;div id="content"&gt;&lt;/div&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"&gt;&lt;/script&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;
&lt;script&gt;
    var div = document.getElementById('content');
    var div2 = document.createElement('div');
    div2.innerHTML = '**can** render: \\(a^2 + b^2\\)';
    MathJax.Hub.Queue(
        ['Typeset', MathJax.Hub, div2],
        function() {
            div.innerHTML = marked.parse(div2.innerHTML);
        }
    );
&lt;/script&gt;</code></pre>



这样一来用户直接看到的就是渲染完成后的效果，不会出现看上去乱码的过渡内容。第二个缺陷是我们可以发现此时右击公式不会显示如下的MathJax菜单了：

[![正常来说右击公式会显示MathJax的菜单](<https://spaces.ac.cn/usr/uploads/2024/08/692170630.png>)](<https://spaces.ac.cn/usr/uploads/2024/08/692170630.png>)

正常来说右击公式会显示MathJax的菜单

这个稍微了解一下自定义右键菜单的原理就明白了。简单来说，自定义右键菜单需要给元素绑定一个事件监听器，但当我们编辑了元素的innerHTML后，事件监听器就会失效。这个问题笔者也想了很久，最终意外发现当我们再次<code>MathJax.Hub.Typeset</code>命令时，MathJax会自动把公式重新渲染，所以我们只需要在上述代码基础上，先删除原有的公式，然后重新渲染一下公式就行了：



<pre><code class="language-html">&lt;div id="content"&gt;&lt;/div&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"&gt;&lt;/script&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;
&lt;script&gt;
    var div = document.getElementById('content');
    var div2 = document.createElement('div');
    div2.innerHTML = '**can** render: \\(a^2 + b^2\\)';
    MathJax.Hub.Queue(
        ['Typeset', MathJax.Hub, div2],
        function() {
            div.innerHTML = marked.parse(div2.innerHTML);
            div.querySelectorAll('.MathJax').forEach(e =&gt; e.remove());
            MathJax.Hub.Typeset(div);
        }
    );
&lt;/script&gt;</code></pre>



这样就恢复了右键菜单。但这还没完，笔者研究了一下其中的原理，发现第一次<code>Typeset</code>后，公式原始代码就被存在一个<code>script</code>标签中，所以后面删除公式可以重复调用<code>Typeset</code>来渲染。但是，笔者发现Marked居然会渲染<code>script</code>的内容！！为了避免Marked修改公式，我们可以在<code>marked.parse</code>之前，把公式的原始代码保存下来，<code>marked.parse</code>之后再覆盖回去：



<pre><code class="language-html">&lt;div id="content"&gt;&lt;/div&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"&gt;&lt;/script&gt;
&lt;script src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;
&lt;script&gt;
    function parseMarkdown(text) {
        var scripts = text.match(/&lt;script[^&gt;]*&gt;([\s\S]*?)&lt;\/script&gt;/gi);
        text = marked.parse(text);
        return text.replace(/&lt;script[^&gt;]*&gt;([\s\S]*?)&lt;\/script&gt;/gi, m =&gt; scripts.shift());
    }
    var div = document.getElementById('content');
    var div2 = document.createElement('div');
    div2.innerHTML = '**can** render: \\(J\'_\\theta = J_\\theta\\)';
    MathJax.Hub.Queue(
        ['Typeset', MathJax.Hub, div2],
        function() {
            div.innerHTML = parseMarkdown(div2.innerHTML);
            div.querySelectorAll('.MathJax').forEach(e =&gt; e.remove());
            MathJax.Hub.Typeset(div);
        }
    );
&lt;/script&gt;</code></pre>
