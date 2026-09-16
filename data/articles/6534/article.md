# 分享：用LaTeX\+MathJax画一个三维三阶环方

> 作者：苏剑林 · 科学空间 · 2019-03-28
>
> 原文：<https://spaces.ac.cn/archives/6534>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

昨天看到数学研发论坛在讨论[三维三阶幻方](<https://bbs.emath.ac.cn/thread-16090-1-1.html>)，论坛里的各大牛都已经讨论得差不多了，我也没什么好插话的。然后突发奇想，能不能用纯LaTeX画出一个这样的立体幻方出来？

昨天下午折腾了好一会儿，最后只抛出了个半成品，然后经过论坛的mathe大佬继续完善后，终于成功地画出来了：
$$\begin{array}{ccccccccccc}
& & & & 4 & —& —& — & — & 25 & —& —& — & — & 11
\\
& & & \require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & && &\require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} && &&\require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} &|
\\
& & 14 & — & — & —& — & 22 & — & — & — & —& 7 & & |
\\
& \require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}}& &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & &\require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}}& &  \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}}&&\require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} & | & & | \\
24 & — & —& —& — & 1 &  —& —& — & — & 18 & & | &  & |\\
|& & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & &\color{red}{13} &| &  \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} &\color{red}{27} & |  & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} & | &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}&5\\
|& & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & \require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & | &  & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}}  &\require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} &| & & |&\require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|} &|\\
|& & \color{red}{8} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}& | &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} & \color{red}{12} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}& | &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}&22&&|\\
|&\require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & | &\require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}}& | &\require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|}  & | &&|\\
15  & — & —& —& — & 3 & — & — & —& —& 21 &  & | & &|\\
|& & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} &  & \color{red}{9}  &| &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} &  \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} & \color{red}{26} &|&\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}&|&\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}&6\\
|& & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}}&\require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}}  &  &|  &  &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\vdots}} &\require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}}  &&|&&|&\style{display: inline-block; transform: rotate(45deg)}{|}\\
|& &\color{red}{16} & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}  & \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}} &|&\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}& \color{red}{8} &\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}&\require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}& | &  \require{HTML} \style{display: inline-block; opacity:0.5;}{\color{red}{\cdots}}&17\\
|& \require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}}& & &  &|&  \require{HTML} \style{display: inline-block; transform: rotate(45deg); opacity:0.5;}{\color{red}{\vdots}} &&&& | & \require{HTML} \style{display: inline-block; transform: rotate(45deg)}{|}\\
23 & — & — & — & — & 2  & — & — & — & — & 19\\
\end{array}$$

事实上代码里边还内嵌了一些HTML代码，所以不算是严格的纯LaTeX代码，应该说是LaTeX\+MathJax的结合。
