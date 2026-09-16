# 有质动力：倒立单摆的稳定性

> 作者：苏剑林 · 科学空间 · 2013-12-29
>
> 原文：<https://spaces.ac.cn/archives/2232>
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
<p data-spaces-tag="p" data-spaces-attrs="{}">前几天在“宇宙的心弦”浏览网页时，发现他更新了一篇很有趣的文章，叫<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://www.physixfan.com/archives/2227&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://www.physixfan.com/archives/2227">《倒立单摆的稳定性与Ponderomotive Force》</a>（果然，物理系的能接触到各种各样有趣的现象），里边谈到通过施加一个运动在单摆上面，倒立的单摆也可以是稳定的。这勾起了我的兴趣，遂也计算了一番。</p><details data-spaces-opaque="1"><summary>原文嵌入内容（embed；预览不执行）</summary><pre><code>&lt;embed align="middle" allowfullscreen="true" allowscriptaccess="always" height="400" quality="high" src="https://player.youku.com/player.php/sid/XNjQ3MDY0OTEy/v.swf" type="application/x-shockwave-flash" width="480"/&gt;</code></pre></details><p data-spaces-tag="p" data-spaces-attrs="{}"></p><p data-spaces-tag="p" data-spaces-attrs="{}">单摆有两个平衡点，一个是垂直向下的稳定平衡点，另一个是垂直向上的不稳定平衡点。用一个形象的说法，那就是如果你有一个垂直向下的单摆，那么你稍微晃一下它，它只是来回摆动；如果那个单摆是垂直向上的，那么如果你晃一下它，它就会迅速倒下到垂直乡下的位置了。如果给单摆施加一个运动，可以让倒立的单摆变成稳定的？咋看之下，很难判断这种运动是否存在，但是只需要想象将拉着单摆向下以$10m/s^2$的加速度向下运动，那么倒立的单摆自然就稳定了，因为这是的力已经反方向了。这其实我们，非惯性运动可能给我们带来一个额外的力场，使得倒立单摆稳定。当然，上面举的加速运动的例子没有太大用处，因为它的运动幅度太大了，以至于那已经不是我们能控制的单摆了。但是还有另外一种运动有可能使得倒立单摆稳定，那就是高频振荡。我们将详细分析它。</p><p data-spaces-tag="p" data-spaces-attrs="{}">Physixfan的文章也对这个问题进行了一番数学分析。但是我总觉得他那样的分析过程是不能让人满意的，他往原系统中的运动方程$m\ddot{x} = -dU/dx$加了一个“高频的振荡项”$f$，变为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$m\ddot{x} = -dU/dx+f$$<br data-spaces-tag="br" data-spaces-attrs="{}">
但是这里的$f$意义是不明确的，如果$x$表示位移，那么$f$可以理解为力，但是$x$也可以是角度或者其他变量，那时就不好把握了。另外，即使将$f$解释为力，那么“将一个高频振荡的力加在单摆上面”与实验而言其实也是不清楚的，更清晰的做法应该是将一个高频振荡直接加在原系统上面，换句话说，把单摆放到一个高频振荡的参考系里边去。（想象一个极端的情况，有一个以极快速度不断一上一下的电梯，单摆就放在里边。）</p><p data-spaces-tag="p" data-spaces-attrs="{}">在这里采用最小作用量原理来分析是最为方便的，以垂直向上为x轴正方向，水平向右为y轴正方向，那么单摆的作用量为：<br data-spaces-tag="br" data-spaces-attrs="{}">
$$S=\int \left[ \frac{1}{2}m(\dot{x}^2+\dot{y}^2)- mgx \right]dt$$<br data-spaces-tag="br" data-spaces-attrs="{}">
其中还有$x^2+y^2=l^2$这一约束，由此可以导出单摆的运动方程。选取这样的坐标系对于分析倒立单摆较为方便。</p><p data-spaces-tag="p" data-spaces-attrs="{}">现在，将一个$x=h(t)$的振荡项加到原单摆中，也就是说让单摆的固定中心以$x=h(t)$的方式运动这，这样一来，单摆的作用量还是<br data-spaces-tag="br" data-spaces-attrs="{}">
$$S=\int \left[ \frac{1}{2}m(\dot{x}^2+\dot{y}^2)- mgx \right]dt$$<br data-spaces-tag="br" data-spaces-attrs="{}">
只不过约束条件变为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$[x-h(t)]^2+y^2=l^2$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">现在就设$x=l\cos\theta+h(t),y=l\sin\theta$，代入作用量，得到<br data-spaces-tag="br" data-spaces-attrs="{}">
$$S=\int \left[ \frac{1}{2}m(l^2\dot{\theta}^2+\dot{h}^2-2l\dot{h}\dot{\theta}\sin\theta)- mg\left(l\cos\theta+h\right) \right]dt$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">由欧拉-拉格朗日方程就得到运动方程<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{d}{dt}\left(ml^2\dot{\theta}-ml\dot{h}\sin\theta\right)=mgl\sin\theta-ml\dot{h}\dot{\theta}\cos\theta$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">即<br data-spaces-tag="br" data-spaces-attrs="{}">
$$l\ddot{\theta}-(\ddot{h}+g)\sin\theta=0$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">可见，稳定点$\theta=0$依然存在，当$h$项不存在时，方程$l\ddot{\theta}-g\sin\theta=0$在原点处是不稳定的，为了认识到这一点，只需用近似$\sin\theta\approx\theta$，得到$l\ddot{\theta}-g\theta=0$，解得<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\theta=Ae^{t\sqrt{g/l}}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
这是发散至无穷的指数解。</p><p data-spaces-tag="p" data-spaces-attrs="{}">只要$(\ddot{h}+g)&lt;0$，就可以让上述方程变为受约束的方程，从而让$\theta=0$处是稳定的。但是我们希望原来的单摆改变的“幅度”不要太大，因此$h$只能够是小幅振荡。不妨假设<br data-spaces-tag="br" data-spaces-attrs="{}">
$$h(t)=h_0 \cos(\omega t)$$<br data-spaces-tag="br" data-spaces-attrs="{}">
有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\ddot{h}=-h_0 \omega^2 \cos(\omega t)$$<br data-spaces-tag="br" data-spaces-attrs="{}">
代入上述方程得<br data-spaces-tag="br" data-spaces-attrs="{}">
$$l\ddot{\theta}+\left[h_0 \omega^2 \cos(\omega t)-g\right]\sin\theta=0$$<br data-spaces-tag="br" data-spaces-attrs="{}">
在$t=0,\theta=0$附近展开，近似有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$l\ddot{\theta}+\left(h_0 \omega^2 -g\right)\theta=0$$<br data-spaces-tag="br" data-spaces-attrs="{}">
可见，为了让$\theta=0$成为稳定平衡点，必须有$h_0 \omega^2 -g \gg 0$，即<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\omega \gg \sqrt{\frac{g}{h_0}}$$<br data-spaces-tag="br" data-spaces-attrs="{}">
如果$h_0=0.01m$，那么估计$\omega \gg 30 s^{-1}$，这只不过是$5Hz$左右，而我们交流电流的频率有$50Hz$，所以上述视频的实验是可能实现的。总共的来说，长期观察而言，就是高频振荡可以产生一项力来使得系统保持它的稳定性，这个力就叫做Ponderomotive Force，中文翻译是有质动力。这种现象在粒子物理中有着重要应用。</p>
</div>
