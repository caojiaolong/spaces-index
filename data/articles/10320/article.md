# 让MathJax更好地兼容谷歌翻译和延时加载

> 作者：苏剑林 · 科学空间 · 2024-08-15
>
> 原文：<https://spaces.ac.cn/archives/10320>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

很早之前，就有读者提出希望把[Cool Papers](<https://papers.cool/>)上面的数学公式渲染一下，因为很多偏数学的论文，它们的摘要甚至标题上都带有LaTeX代码写的数学公式，如果不把这些公式渲染出来，那么看上去就像是一堆乱码，确实会比较影响阅读体验。然而，之前的测试显示，负责渲染公式的MathJax跟谷歌翻译和延时加载都不大兼容，所以尽管需求存在已久，但笔者一直没有把它加上去。

不过好消息是，经过反复查阅和调试，这两天笔者总算把兼容性问题解决了，所以现在大家看到的Cool Papers已经能够渲染数学公式了。这篇文章总结一下解决方案，供大家参考。

[![摘要带有公式的论文](<https://spaces.ac.cn/usr/uploads/2024/08/3020393852.png>)](<https://spaces.ac.cn/usr/uploads/2024/08/3020393852.png>)

摘要带有公式的论文

## 公式渲染

对于在网页中显示数学公式（LaTeX），目前有两个比较主流的方案，分别是[MathJax](<https://www.mathjax.org/>)和[KaTeX](<https://katex.org/>)。KaTeX相对来说更为轻量级一些，但它对LaTeX的支持不如MathJax全面，加之本博客一直都是用MathJax，所以在考虑给Cool Papers加数学公式支持时，也是第一时间选择MathJax。

跟Python类似，MathJax的3\.x跟2\.x是两套差别比较大的体系（最新版本是3\.2\.2，4\.0版也已经开始测试了），而目前我们能搜到的多数MathJax相关资料都是2\.x版的，所以给Cool Papers上的是MathJax 2\.x的最新版2\.7\.9（也是本博客用的版本，特别地，arXiv官网也用MathJax，用的是2\.7\.3版）。

对于一个普通的网页，给它加上数学公式渲染功能并不难，在网页中加两段代码即可，以下是本博客加的参考代码：



<pre><code class="language-javascript">&lt;script type="text/x-mathjax-config"&gt;
    MathJax.Hub.Config({
        tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
        TeX: {equationNumbers: {autoNumber: ["AMS"], useLabelIds: true}, extensions: ["AMSmath.js", "AMSsymbols.js", "extpfeil.js"]},
        "HTML-CSS": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MathJax_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "CommonHTML": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MJXc-display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "SVG": {linebreaks: {automatic: true, width: "95% container"}, styles: {".MathJax_SVG_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "PreviewHTML": {linebreaks: {automatic: true, width: "95% container"}}
    });
&lt;/script&gt;
&lt;script src="/static/MathJax-2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;</code></pre>



## 翻译之殇

以上代码对于普通需求来说是有效的，可以成功将LaTeX代码转为网页可显示的数学公式。但对于Cool Papers来说，它遇到了两个“拦路虎”：网页翻译和延时加载。这一节我们首先来解决第一个“拦路虎”——网页翻译，这里主要指<u>Chrome浏览器自带的谷歌翻译</u>。

众所周知，Cool Papers的主旨是刷论文，论文的标题和摘要都是英文的，而对于母语是中文的我们来说，很多时候都会打开网页翻译功能来加速阅读。虽然肯定也有读者对此“不屑一顾”，认为直接阅读原始英文描述更精准，但不可否认网页翻译的需求确实存在，而且对于“刷论文”这个目标来说，很多时候即便只看机翻为中文的内容也够用了。

然而，对于带有经MathJax渲染的数学公式的页面，谷歌翻译后的效果可谓“面目全非”，近乎乱码到没法阅读下去。读者可以自行到arXiv找一篇带有公式的论文尝试一下，比如[2408\.07010](<https://papers.cool/arxiv/2408.07010>)的结果如下：

[![带公式的页面翻译前效果](<https://spaces.ac.cn/usr/uploads/2024/08/1687499664.png>)](<https://spaces.ac.cn/usr/uploads/2024/08/1687499664.png>)

带公式的页面翻译前效果

[![带公式的页面翻译后效果](<https://spaces.ac.cn/usr/uploads/2024/08/186421905.png>)](<https://spaces.ac.cn/usr/uploads/2024/08/186421905.png>)

带公式的页面翻译后效果

## 免翻金牌

解决这个问题的思路就是给公式加个“免翻金牌”，即不要翻译公式。经过搜索，发现让谷歌翻译不翻译某个元素的方法有两个，一是给该元素增加一个类名<code>class="notranslate"</code>，二是给该元素增加一个属性<code>translate="no"</code>。增加的方式又有两种，一种是在后端增加，即在服务端输出之前就修改好网页内容，另一种是在前端增加，即浏览器接收完网页内容后再用JS修改。

对于MathJax来说，它是在前端实时渲染的数学公式，后端接触不到渲染后的公式，所以我们只能选择在前端增加的方案。经过测试发现，MathJax会给渲染后的数学公式加上<code>MathJax</code>的class名，所以我们根据该class名称就可以抽取到所有公式，然后就可以通过JS追加<code>class="notranslate"</code>，参考代码如下：



<pre><code class="language-javascript">document.querySelectorAll('.MathJax').forEach(element =&gt; element.classList.add('notranslate'));</code></pre>



但是要注意，这句代码需要在所有数学公式都渲染完毕后运行才有效。如何保证公式都已经渲染完呢？最可靠的方案，是将该代码放置到MathJax的Queue中去（参考[这里](<https://docs.mathjax.org/en/v2.7-latest/advanced/queues.html#using-queues>)）：



<pre><code class="language-javascript">&lt;script type="text/x-mathjax-config"&gt;
    MathJax.Hub.Config({
        tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
        TeX: {equationNumbers: {autoNumber: ["AMS"], useLabelIds: true}, extensions: ["AMSmath.js", "AMSsymbols.js", "extpfeil.js"]},
        "HTML-CSS": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MathJax_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "CommonHTML": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MJXc-display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "SVG": {linebreaks: {automatic: true, width: "95% container"}, styles: {".MathJax_SVG_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "PreviewHTML": {linebreaks: {automatic: true, width: "95% container"}}
    });
    MathJax.Hub.Queue(function() {
        document.querySelectorAll('.MathJax').forEach(element =&gt; element.classList.add('notranslate'));
    });
&lt;/script&gt;
&lt;script src="/static/MathJax-2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;</code></pre>



这样的意思就是让MathJax自己在确保公式渲染完后，再去执行<code>MathJax.Hub.Queue</code>里边定义的函数。经过这样处理，在公式渲染完毕后再打开谷歌翻译，公式就不会被翻译了。

## 延时加载

MathJax的第二个“拦路虎”是延时加载。在Cool Papers的列表页中，由于要显示的论文数可能比较多（数百乃至上千），所以刚开始打开网页的时候，不会一次性加载所有论文，而且只显示前25篇，直到滑动到接近底部了，才继续加载下一个25篇，这样提高了效应速度，同时也几乎不会影响用户体验，这就是延时加载。

很多内容比较多的网站都会采用延时加载，这是一个比较成熟的技术了。但对于MathJax来说，它只会渲染刚打开页面时的所有公式，后面延时加载的内容里边的公式则不会主动渲染，所以要在延时加载之后手动渲染一下。这个倒是不难，手动触发渲染的函数是<code>MathJax.Hub.Typeset</code>，我们只需要在延时加载的函数后面加上这个函数就行，类似于：



<pre><code class="language-javascript">loadMorePapers();
MathJax.Hub.Typeset();</code></pre>



注意单纯的<code>MathJax.Hub.Typeset</code>不包含给公式追加<code>notranslate</code>的操作，要补上这个操作，则需要改为：



<pre><code class="language-javascript">loadMorePapers();
MathJax.Hub.Queue(
    ['Typeset', MathJax.Hub],
    function() {
        document.querySelectorAll('.MathJax').forEach(element =&gt; element.classList.add('notranslate'));
    }
);</code></pre>



## 混合双殇

上面我们分别解决了MathJax与谷歌翻译、延时加载的兼容性问题，然而，当谷歌翻译跟延时翻译一起出现时，新的问题出现了。

假设我们刚进入页面时就打开了谷歌翻译，那么当我们浏览到接近底部时，新的论文被延时加载近来，然后谷歌翻译会被继续触发，进而把新加载的论文也翻译了。如果我们还设置了上一节的手动渲染公式代码，那么公式也会被MathJax渲染出来。由于谷歌翻译和MathJax同时触发，但公式渲染完才会给公式追加<code>class="notranslate"</code>，换句话说翻译开始进行时公式都还没来得及加上<code>class="notranslate"</code>，所以最终公式还是被翻译了。

要解决这个问题，就必须想一个办法，确保在公式渲染完毕且加上<code>class="notranslate"</code>后才执行谷歌翻译。可是，谷歌翻译是Chrome自带的，我们没法通过网站来操纵浏览器的行为。看上去进入了死胡同，不过笔者测试发现，谷歌翻译会随时监控页面的变化，来决定是否触发新的翻译。根据这个特性，我们可以来个“逆向思维”。

怎么个逆向法呢？我们知道，对于Cool Papers来说，要翻译的无非就是论文的标题和摘要，我们可以在一开始就它们加上<code>class="notranslate"</code>，加上后不管是首次浏览还是延时加载，谷歌翻译都不会主动翻译它们。然后，我们可以在公式渲染完成之后，再把标题和摘要的<code>class="notranslate"</code>去掉，此时浏览器就会识别到标题和摘要是可翻译的内容了，然后翻译就会被触发。

这样一来，我们就成功实现了确保公式渲染完成后才触发翻译功能了。参考代码如下：



<pre><code class="language-javascript">loadMorePapers();
MathJax.Hub.Queue(
    ['Typeset', MathJax.Hub],
    function() {
        document.querySelectorAll('.MathJax').forEach(element =&gt; element.classList.add('notranslate'));
        document.querySelectorAll('a.title-link, p.summary').forEach(element =&gt; element.classList.remove('notranslate'));
    }
);</code></pre>



## 总结一下

最后，我们来总结一下我们的解决方案。如果你的网站需要显示数学公式，带有延时加载功能，并且用户有网页翻译功能的需求，那么可以按照如下步骤，达到最大的兼容性：

1、给所有带有公式的内容块，都加上<code>class="notranslate"</code>；

2、按照如下方式，加载MathJax（其中“a\.title\-link”、“p\.summary”是带有数学公式的块的class名之一）：



<pre><code class="language-javascript">&lt;script type="text/x-mathjax-config"&gt;
    MathJax.Hub.Config({
        tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
        TeX: {equationNumbers: {autoNumber: ["AMS"], useLabelIds: true}, extensions: ["AMSmath.js", "AMSsymbols.js", "extpfeil.js"]},
        "HTML-CSS": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MathJax_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "CommonHTML": {linebreaks: {automatic: true, width: "95% container"}, noReflows: false, availableFonts: ["tex"], styles: {".MJXc-display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "SVG": {linebreaks: {automatic: true, width: "95% container"}, styles: {".MathJax_SVG_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "PreviewHTML": {linebreaks: {automatic: true, width: "95% container"}}
    });
    MathJax.Hub.Queue(function() {
        document.querySelectorAll('.MathJax').forEach(element =&gt; element.classList.add('notranslate'));
        document.querySelectorAll('a.title-link, p.summary').forEach(element =&gt; element.classList.remove('notranslate'));
    });
&lt;/script&gt;
&lt;script src="/static/MathJax-2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"&gt;&lt;/script&gt;</code></pre>



3、在延时加载的代码后面，加上如下代码：



<pre><code class="language-javascript">MathJax.Hub.Queue(
    ['Typeset', MathJax.Hub],
    function() {
        document.querySelectorAll('.MathJax').forEach(element =&gt; element.classList.add('notranslate'));
        document.querySelectorAll('a.title-link, p.summary').forEach(element =&gt; element.classList.remove('notranslate'));
    }
);</code></pre>



具体效果欢迎在[Cool Papers](<https://papers.cool/>)里边测试。以上方案笔者在Chrome和Safari下都测试通过，适用于Chrome自带的翻译、Safari自带的翻译以及Cool Papers自带的翻译功能。
