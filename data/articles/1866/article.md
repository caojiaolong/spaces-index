# 又折腾数学公式插件了

> 作者：苏剑林 · 科学空间 · 2013-01-05
>
> 原文：<https://spaces.ac.cn/archives/1866>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

从2013\.11\.15开始，使用MathJax插件。主要原因是MathJax在兼容性方面比ASCIIMath Image Fallback Scripts做得好很多。而且从长远考虑，用MathJax也是应该的。
官方网站：[http://www\.mathjax\.org/](<http://www.mathjax.org/>)

复制数学公式：[http://www\.mathjax\.org/demos/copy\-and\-paste/](<http://www.mathjax.org/demos/copy-and-paste/>)

<strong>\-\-\-\-\-\-\-以下内容已经过时（写于2013\.01\.05）\-\-\-\-\-\-\-\-</strong>

原来一直是使用“数学研发论坛”完善的数学公式插件来显示数学公式的，使用很简单，载入速度很快，这样一下子就用了三年了。

不过进入大学后，学习的东西越来越多，数学符号也越来越多，郭大哥的插件的不足也暴露出来了。最要命的是它居然无法显示$\hbar$，这叫我这个学习量子力学的孩子情何以堪\.\.\.（不过郭大哥新版的插件已经加入了这个符号）。还有另外一个不足的地方，就是郭大哥的插件进行了大量的化简，使得数学公式的输入简单了不少，但是反而对标准的Latex代码支持不足了。久而久之，会带来一个弊端，就是迁移性不强。万一哪天这个插件无法使用了，就难以找到替代品了。考虑到这些，我写latex代码的时候总是用标准的语法而不用简化语法，后来$\hbar$的问题出来后，一下子用上了MathJax这个强大的插件（考虑过JsMath，但是发现它的行内公式显示效果不大好）。

MathJax使用很标准的Latex语法，效果也很多样化，显示的数学公式也很漂亮。但是我还是感到一点点不足，就是它是利用自身的功能显示数学公式的，所以数学公式并不支持复制，因此不利于知识的传播（虽然我这里也没有什么知识）。另外一个就是载入速度有点慢。郭大哥改好他的插件后我也考虑换回他的插件，但又顾及到标准语法问题，一直犹豫不决\.\.\.\.\.\.

今天我重新找回郭大哥的tex\.js的原型——[ASCIIMath Image Fallback Scripts](<http://dlippman.imathas.com/asciimathtex/AMT.html>)，发现它原来是支持标准语法的，所以眼睛一亮，立马决定修改为这个版本。期间发现了不少不兼容的地方，所以我不断修改、调试、修改、调试\.\.\.其实就是一些很简单的修改，却折腾了我四五个钟，不懂代码的孩子真的伤不起\.\.\.到今天下午四点，终于得出了自己一个比较满意的ASCIIMath Image Fallback Scripts了。其中特点如下：

1、使用相对标准的Latex语法显示数学公式，数学公式用蓝色突出。把Latex代码放进两个美元符号之间（行内公式）、两个双美元符号之间（独立一行显示公式）都行。
2、支持MathML的时候（比如火狐浏览器）使用MathML显示数学公式，这时显示效果相当不错；不支持的时候调用图片，图片来自[MathTex](<http://www.forkosh.com/mathtex.html>)，是MineTex的“大哥”，图片质量还过得去，不过比MathML差。
3、IE本身不支持MathML，可以通过安装MathPlayer来支持MathML。不过我是在Windows 8下使用IE 10，IE装上MathPlayer后无法正常显示公式（兼容模式也不行）。其他版本IE还没有机会测试。

<strong>以下是一些公式展示：</strong>
分数：
$$\sum_{i=1}^{\infty} (\frac{1}{2})^i,\sum_{n=1}^{\infty}\frac{1}{n^2}=\frac{\pi^2}{6}$$
根式：
$$c=\sqrt{a^2+b^2},p= \sqrt[3]{x^2+y^2}$$（这个立方根在MathML下不能正常显示，努力修复中\.\.\.）
矩阵：
$$\left[ \begin{array}{c c} 1&2 \\ 3&4 \end{array}\right],\left[ \begin{array}{c c c} a&b&c \\ d&e&f \\ g&h&i \end{array}\right]$$
多行公式：
$$\begin{eqnarray*}\nabla \times \vec{\boldsymbol{B}} - \frac{1}{c} \frac{\partial\vec{\boldsymbol{E}}}{\partial t} = \frac{4 \pi}{c} \vec{\boldsymbol{j}} \\ \nabla \cdot \vec{\boldsymbol{E}} = 4 \pi \rho \\ \nabla \times \vec{\boldsymbol{E}} + \frac{1}{c} \frac{\partial\vec{\boldsymbol{B}}}{\partial t} = \vec{\boldsymbol{0}} \\ \nabla \cdot \vec{\boldsymbol{B}} = 0 \end{eqnarray*}$$

其中MathTex官网有更多例子以及一个公式编辑器：
[http://www\.forkosh\.com/source\_mathtex\.html\#examples](<http://www.forkosh.com/source_mathtex.html#examples>)

<strong>注：旧版已经失效</strong>
[http://kexue\.fm/archives/83/](<https://spaces.ac.cn/archives/83/>)
