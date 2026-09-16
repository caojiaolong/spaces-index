# 让MathJax的数学公式随窗口大小自动缩放

> 作者：苏剑林 · 科学空间 · 2024-10-15
>
> 原文：<https://spaces.ac.cn/archives/10474>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

随着MathJax的出现和流行，在网页上显示数学公式便逐渐有了标准答案。然而，MathJax（包括其竞品KaTeX）只是负责将网页LaTeX代码转化为数学公式，对于自适应分辨率方面依然没有太好的办法。像本站一些数学文章，因为是在PC端排版好的，所以在PC端浏览效果尚可，但转到手机上看就可能有点难以入目了。

经过测试，笔者得到了一个方案，让MathJax的数学公式也能像图片一样，随着窗口大小而自适应缩放，从而尽量保证移动端的显示效果，在此跟大家分享一波。

## 背景思路

这个问题的起源是，即便在PC端进行排版，有时候也会遇到一些单行公式的长度超出了网页宽度，但又不大好换行的情况，这时候一个解决方案是用HTML代码手动调整一下公式的字体大小，比如



<pre><code class="language-html">&lt;span style="font-size:90%"&gt;
    \begin{equation}一个超长的数学公式\end{equation}
&lt;/span&gt;</code></pre>



这样就将数学公式大小调整为原来的90%，基本能解决大部分问题，但手动调整终究是有点麻烦，90%这个数字需要人工调试几次才能得到最优值，并且只能适应单一宽度。前几天突然意识到一个问题：为什么数学公式不能像图片一样设置一个max\-width来自动缩放大小呢？比如下面的图片代码



<pre><code class="language-html">&lt;img style="width:400;max-width:100%" src="https://example.com/test.jpg"&gt;</code></pre>



实现的效果就是图片大小在不超过上级元素的前提下尽可能接近400px。测试了一翻，MathJax的数学公式是文本的block，没有现成的方法可以实现类似图片的max\-width缩放特性，但我们可以通过JS来计算公式超出上级元素宽度的比例，然后自动给数学公式设置font\-size来实现同样的效果。

## 代码参考

首先给出最终的参考答案，用如下的代码替换掉原本MathJax配置代码：



<pre><code class="language-html">&lt;script type="text/x-mathjax-config"&gt;
    MathJax.Hub.Config({
        tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]},
        TeX: {equationNumbers: {autoNumber: ["AMS"], useLabelIds: true}, extensions: ["AMSmath.js", "AMSsymbols.js", "extpfeil.js"]},
        "HTML-CSS": {noReflows: false, availableFonts: ["tex"], styles: {".MathJax_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "CommonHTML": {noReflows: false, availableFonts: ["tex"], styles: {".MJXc-display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}},
        "SVG": {noReflows: false, availableFonts: ["tex"], styles: {".MathJax_SVG_Display": {margin: "1em 0em 0.7em;", display: "inline-block!important;"}}}
    });
    MathJax.Hub.Queue(function() {
        document.querySelectorAll('span[id^="MathJax-Element"]').forEach(function(e) {
            parentWidth = e.parentNode.offsetWidth;
            if (e.parentNode.className.endsWith('isplay')) {
                parentWidth = e.parentNode.parentNode.offsetWidth;
            }
            if (e.offsetWidth &gt; parentWidth) {
                e.style.fontSize = parentWidth * 100 / e.offsetWidth + '%';
            }
        });
    });
&lt;/script&gt;</code></pre>



相比在[《让MathJax更好地兼容谷歌翻译和延时加载》](<https://spaces.ac.cn/archives/10320>)提到的配置代码，它有两个关键变化。首先是去掉了配置项<code>{linebreaks: {automatic: true, width: "95% container"}</code>，这个配置项的作用是自动换行，原本用意就是提高对分辨率的自适应能力，但实际上用途不大，很多写好的数学公式如果自动换行的话反而不好看了，所以去掉自动换行才能保证数学公式在不同宽度下的一致性。

接下来的关键改动就是<code>MathJax.Hub.Queue</code>部分了，这是公式渲染完毕后执行的函数，它首先找到所有id以“MathJax\-Element”开头的span，这就是所有的数学公式，然后通过<code>e.offsetWidth</code>获取它的宽度，以及通过<code>e.parentNode.offsetWidth</code>或<code>e.parentNode.parentNode.offsetWidth</code>获取它的上级宽度，根据这个结果计算和设置font\-size要缩小的比例。

什么时候用<code>e.parentNode.offsetWidth</code>或<code>e.parentNode.parentNode.offsetWidth</code>呢？数学公式分“行内公式”和“单行公式”，“行内公式”就直接用前者而“单行公式”则用后者。在代码上，“单行公式”多嵌套了一层div，所以祖父节点才是上级元素，而div的class以Display或display结束，所以就加了个<code>e.parentNode.className.endsWith('isplay')</code>的判据。

以上就是参考代码的所有原理。

## 文章小结

本文分享了一种让MathJax公式自动随着窗口大小而缩放的方案，能够尽可能兼容移动端设备的窄屏浏览需求。经过调整之后，即便在小屏幕下也能显示出跟PC端一样的数学公式——除了可能有点费眼睛，但用来应应急总是可以的。

[![修改后的窄屏显示效果](<https://spaces.ac.cn/usr/uploads/2024/10/4189716394.png>)](<https://spaces.ac.cn/usr/uploads/2024/10/4189716394.png>)

修改后的窄屏显示效果
