# 捉弄计划的失败——单摆周期

> 作者：苏剑林 · 科学空间 · 2010-06-09
>
> 原文：<https://spaces.ac.cn/archives/674>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![老式机械摆钟](<https://spaces.ac.cn/usr/uploads/2010/06/3962377054.jpg>)](<https://spaces.ac.cn/attachment/679/>)

老式机械摆钟

“滴答滴答，滴答滴答——”当我们看到家里的摆钟来回摆动，并且能够准确地报时的时候，有没有想过其中的奥妙呢？

有一天，你想捉弄一下妈妈，把钟摆系上一个重物，心想着钟一定会走得更快，妈妈就会乱套了。可是很快你会失望地发现，摆钟依然准时地走着，没有任何异常，时间仿佛在宣告他的不可控制。你感到非常纳闷：为什么我的计划会失败呢？

据说，世界上第一个研究单摆的人是伽利略，他通过多次实验得出结论：单摆的周期只取决于摆绳的长度，和摆的重量无关。这是你明白了，原来要捉弄妈妈，应该要增加钟摆长度才对\.\.\.^\_^

现在我们来分析一下这个单摆\.\.\.\.

[![单摆示意图](<https://spaces.ac.cn/usr/uploads/2010/06/1321017280.png>)](<https://spaces.ac.cn/attachment/677/>)

单摆示意图

设单摆的长度为l，在下落过程中只考虑重力因素，那么机械能是守恒的。点A是始点，此时速度为0；当摆下落到B点后，速度为v，动能为$1/2 mv^2$，动能是由重力势能转化而来的，设$RQ=h=l(\cos\theta-\cos\theta_0)$，重力势能的减少量为$mgh=mgl(\cos\theta-\cos\theta_0)$。于是我们可以列出
$$\begin{aligned}v^2=2gl(\cos\theta-\cos\theta_0)=(\frac{ds}{dt})^2=(l\frac{d\theta}{dt})^2 \\ \frac{d\theta}{dt}=\sqrt{\frac{2g}{l}}\sqrt{\cos\theta-\cos\theta_0}\end{aligned}$$

利用$cos\theta=1-2sin^2 \frac{\theta}{2}$，可以变换成
$$\frac{dt}{d\theta}=1/2 \sqrt{\frac{l}{g}}\frac{1}{\sqrt{\sin^2 \frac{\theta_0}{2}-\sin^2 \frac{\theta}{2}}}$$

对于摆来说，一个周期T定义为从A点到达C点再回到A点所用的时间，由于从A—P与从P—C所用时间是相同的，所以$T=4T_{AP}$，即
$$T=4\int_0^{\theta_0} 1/2 \sqrt{\frac{l}{g}}\frac{d\theta}{\sqrt{\sin^2 \frac{\theta_0}{2}-\sin^2 \frac{\theta}{2}}}$$

[![单摆动画](<https://spaces.ac.cn/usr/uploads/2010/06/2464705813.gif>)](<https://spaces.ac.cn/attachment/676/>)

单摆动画

实际上，对于不同的$\theta_0$，这个积分的结果是不一样的，但是为什么高中书本上都说周期和角度无关呢？当$\theta$比较小的时候，我们用关系式$sin^2 \frac{\theta}{2}\approx (\frac{\theta}{2})^2$，替换后变成
$$T=4\sqrt{\frac{l}{g}}\int_0^{\theta_0} \frac{d\theta}{\sqrt{\theta_0^2-\theta^2}}$$

由定积分的知识可以知道，结果为$T=2\pi \sqrt{\frac{l}{g}}$，这正是伽利略的结果，与质量、初始角度无关。

如果需要精确的算法，可以从以下方向思考：

令$sin\frac{\theta_0}{2}=k,sin\frac{\theta}{2}=k sin\varphi$，$\varphi \in(0,\frac{\pi}{2})$,则$1/2 cos\frac{\theta}{2}d\theta=kcos\varphi$，即
$$d\theta=\frac{2k \cos\varphi d\varphi }{\sqrt{1-k^2 \sin^2 \varphi}}$$

代入后得到
$$T=4\sqrt{\frac{l}{g}}\int_0^{\frac{\pi}{2}} \frac{1}{\sqrt{1-k^2 \sin^2 \varphi}}$$

这类积分成为“第一类椭圆积分”。其结果为：

$$T=2\pi\sqrt{\frac{l}{g}}(1+\frac{1}{2^2}\sin^2 \frac{\theta}{2}+\frac{1\cdot 3^2}{2^2\cdot 4^2}\sin^4 \frac{\theta}{2}+...)$$

如果只取到第一项，那么就是刚开始的近似公式。推导过程可以查阅大学的《数学分析》教程，或者参考下边的图片。

[![第一类椭圆积分](<https://spaces.ac.cn/usr/uploads/2010/06/4100993903.png>)](<https://spaces.ac.cn/attachment/678/>)

第一类椭圆积分

[数学传播\-单摆的详细分析\.pdf](<https://spaces.ac.cn/usr/uploads/2010/06/3504644286.pdf>)
