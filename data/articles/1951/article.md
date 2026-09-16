# 2^29363731\-1不是素数！

> 作者：苏剑林 · 科学空间 · 2013-04-08
>
> 原文：<https://spaces.ac.cn/archives/1951>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![2^29363731-1](<https://spaces.ac.cn/usr/uploads/2013/04/867773871.png>)](<https://spaces.ac.cn/attachment/1952/>)

2^29363731\-1

很小的时候就开始对素数感兴趣了，后来是在一本《未解之谜》上看到了梅森素数、完全数、孪生素数等等东西，觉得甚是好玩。在初中买了计算机之后，就关注到了Prime 95这个梅森素数的分布式计算程序，以前也尝试过运行它，不过由于那时候计算机配置较低，一般都是运行到20%左右就没有坚持下去了。

上大学入手了一台四核的笔记本，就在去年10月份左右再次运行了这个程序，由于是四核，一次性可以同时测试四个数字。经过半年的运行，今天终于测试完了第一个数字：$2^{29363731}-1$。正如预料中的，<strong>这不是一个素数</strong>。不管怎样，它是我第一个完成的测试，也算是自己的一个独立的成果啦，呵呵，自娱自乐一番。

第二、三、四个数字的测试还在进行中，进度不一，应该很快就可以测试完了，到时候再和大家分享。

日志显示，完成于今天早上10点11分：

> &#91;Comm thread Apr 8 10:11&#93; Sending result to server: UID: bojone/bojone, M29363731 is not prime\. Res64: 216CDDA1A8E59DB8\. We4: DD94A916,19701102,00000000, AID:
> &#91;Comm thread Apr 8 10:11&#93;
> &#91;Comm thread Apr 8 10:11&#93; PrimeNet success code with additional info:
> &#91;Comm thread Apr 8 10:11&#93; LL test successfully completes double\-check of M29363731
> &#91;Comm thread Apr 8 10:11&#93; CPU credit is 29\.2006 GHz\-days\.
> &#91;Comm thread Apr 8 10:11&#93; Done communicating with server\.
