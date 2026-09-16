# 重提“旋转弹簧伸长”问题（变分解法）

> 作者：苏剑林 · 科学空间 · 2011-04-05
>
> 原文：<https://spaces.ac.cn/archives/1311>
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
<p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">感谢Awank-Newton读者的来信，本文于2013.01.30作了修正，主要是弹性势能的正负号问题。之前连续犯了两个错误，导致得出了正确答案。现在已经修正。参考<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/1902/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/1902/">《平衡态公理的修正与思考》</a></font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">在下面的两篇文章中，BoJone已经介绍了这个“旋转弹簧伸长”的问题，并从两个角度提供了两种解答方法。前者列出了一道积分方程，然后再转变为微分方程来解；后者直接从弹性力学的角度来列出一道二阶微分方程，两者殊途同归。<br data-spaces-tag="br" data-spaces-attrs="{}">
<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/782/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/782/">http://kexue.fm/archives/782/</a></p><p data-spaces-tag="p" data-spaces-attrs="{}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/826/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/826/">http://kexue.fm/archives/826/</a></p><p data-spaces-tag="p" data-spaces-attrs="{}">今天，再经过一段时间的变分法涉猎后，BoJone尝试从变分的角度（总能量最小）来给出一种新的解法。同样设r为旋转达到平衡后弹簧上一点到旋转中心的距离，该点的线密度为$\lambda =\lambda (r)$，该点到中心的弹簧质量为$m=m(r)$，旋转前的长度为$l_0$，旋转平衡后的长度为$l_1$。由于弹簧旋转后已经达到了平衡状态，<u data-spaces-tag="u" data-spaces-attrs="{}">由平衡态公理（参看《自然极值》系列），平衡意味着<del data-spaces-tag="del" data-spaces-attrs="{}">总能量</del>“动能-势能”取极值。</u></p><p data-spaces-tag="p" data-spaces-attrs="{}">该弹簧的总动能为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$E_k=\int_0^{l_1} \frac{1}{2} [v(r)]^2 dm=\frac{1}{2} \int_0^{l_1} \lambda r^2 \omega^2 dr$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">下面来求弹簧的总弹性势能。由于弹簧被拉伸了，因而弹性势能为负数，理想弹簧的势能计算公式为$E_p=\frac{1}{2} k (\Delta l)^2$，$\Delta l$是伸长量。但是旋转后的弹簧并非理想弹簧了（不均匀伸长所致）。弹簧[r,r+dr]处的劲度系数为$\frac{M_0}{dm} k$，伸长量为$(dr-\frac{dm}{\lambda_0})$（$M_0$弹簧总质量，$\lambda_0$是原来的密度，即平均密度），于是总的弹性势能为</p><p data-spaces-tag="p" data-spaces-attrs="{}">$$E_p=\frac{1}{2} \int_0^{l_1} \frac{M_0}{dm} k(dr-\frac{dm}{\lambda_0})^2=\frac{1}{2} \int_0^{l_1} M_0 k(\frac{1}{\lambda}+\frac{\lambda}{\lambda_0^2}-\frac{2}{\lambda_0})dr$$</p><p data-spaces-tag="p" data-spaces-attrs="{}"><del data-spaces-tag="del" data-spaces-attrs="{}">总能量</del>“动能-势能”为<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\begin{aligned}L=\frac{1}{2} \int_0^{l_1} [\lambda r^2 \omega^2 -M_0 k(\frac{1}{\lambda}+\frac{\lambda}{\lambda_0^2}-\frac{2}{\lambda_0})]dr \\ =\frac{1}{2} \int_0^{l_1} [\dot{m}r^2 \omega^2-M_0 k(\frac{1}{\dot{m}}+\frac{\dot{m}}{\lambda_0^2}-\frac{2}{\lambda_0})]dr\end{aligned}$$</p><p data-spaces-tag="p" data-spaces-attrs="{}">现在就好办了，令$F=\dot{m}r^2 \omega^2-M_0 k(\frac{1}{\dot{m}}+\frac{\dot{m}}{\lambda_0^2}-\frac{2}{\lambda_0})$，令其变分为0，有<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{\partial F}{\partial m}-\frac{d}{dt}(\frac{\partial F}{\partial \dot{m}})=0$$<br data-spaces-tag="br" data-spaces-attrs="{}">
由于F不显含$m$，因此可以得到首次积分（<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/1304/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/1304/">参看</a>）<br data-spaces-tag="br" data-spaces-attrs="{}">
$$\frac{\partial F}{\partial \dot{m}}=C$$<br data-spaces-tag="br" data-spaces-attrs="{}">
即<br data-spaces-tag="br" data-spaces-attrs="{}">
$$r^2 \omega^2+\frac{M_0 k}{\dot{m}^2}-\frac{M_0 k}{\lambda_0^2}=C$$<br data-spaces-tag="br" data-spaces-attrs="{}">
当$r=l_0$时，应该有$\dot{m}=\lambda_0$（没有力拉伸这一段，因而密度不变），可以确定$C=l_1^2 \omega^2$.</p><p data-spaces-tag="p" data-spaces-attrs="{}">接下来的处理就和<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/782/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/782/">http://kexue.fm/archives/782/</a>一样了，不再赘述！</p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">通过本文，我们可以稍稍领略到了平衡态公理的普适性，以及用变分法研究的魅力，我们省去了受力分析，只需要考虑能量，而能量的标量性使我们可以将其简单叠加运算，这大大减少了我们的思考量。当然，这仅仅是冰山一角。变分法以其独有的魅力在诸多领域大放着光彩！</font></strong></p>
</div>
