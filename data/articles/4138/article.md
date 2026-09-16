# 端到端的腾讯验证码识别（46%正确率）

> 作者：苏剑林 · 科学空间 · 2016-12-14
>
> 原文：<https://spaces.ac.cn/archives/4138>
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
<p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><font data-spaces-tag="font" data-spaces-attrs="{&quot;color&quot;: &quot;red&quot;}" color="red">最新结果请参考：<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/archives/4503/&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://spaces.ac.cn/archives/4503/">http://kexue.fm/archives/4503/</a></font></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}">前段时间有幸得到了一个网友提供的一批带标签的腾讯验证码样本（验证码样板：<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;http://captcha.qq.com/getimage&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="http://captcha.qq.com/getimage">http://captcha.qq.com/getimage</a>），于是抽了点时间，测试了一下验证码识别的模型。</p><p data-spaces-tag="p" data-spaces-attrs="{}"></p><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;pic-container&quot;]}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption&quot;, &quot;aligncenter&quot;], &quot;style&quot;: &quot;max-width: 100%&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;style&quot;: &quot;margin: 5px&quot;}"><a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;/usr/uploads/2016/12/457376744.jpeg&quot;, &quot;target&quot;: &quot;_blank&quot;, &quot;title&quot;: &quot;点击查看原图&quot;}" href="https://spaces.ac.cn/usr/uploads/2016/12/457376744.jpeg" title="点击查看原图"><img data-spaces-tag="img" data-spaces-attrs="{&quot;alt&quot;: &quot;腾讯验证码&quot;, &quot;src&quot;: &quot;/usr/uploads/2016/12/457376744.jpeg&quot;, &quot;style&quot;: &quot;max-width:100%&quot;}" src="https://spaces.ac.cn/usr/uploads/2016/12/457376744.jpeg" alt="腾讯验证码"></a></div><p data-spaces-tag="p" data-spaces-attrs="{&quot;class&quot;: [&quot;typecho-caption-text&quot;]}">腾讯验证码</p></div></div><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;样本&quot;}">样本</h3><p data-spaces-tag="p" data-spaces-attrs="{}">这批验证码比较简单，4位的英文字母，有大小写，但输入的时候不区分大小写，图案有一定的混淆，传统的基于分割的方案估计比较难办。端到端的方案是，直接将验证码输入，做几个卷积层，然后连接几个分类器（26分类），然后就直接输出四个字母标签了。其实还真没有什么好说的，有样本就能做了，而且这个框架是通用的，可以用到区分大小写的情形（52分类），也可以用到英文数字混合的情形（再加10个类别而已）。</p><p data-spaces-tag="p" data-spaces-attrs="{}">不过，有一个我认为是比较难搞的地方，就是标签不区分大小写。这批样本中，标签全部是小写的，但是图片上的验证码有大写有小写。这样，如果只是26分类的话，那么强行要将A、a归为同一类，而A、a从外形上来看区别还是有点大的，强行归类，似乎有点“强模型所难”...我估计这也是我模型准确率上不去的原因之一。但我也没什么好思路。</p><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;代码&quot;}">代码</h3><p data-spaces-tag="p" data-spaces-attrs="{}">也不多说了，直接上代码<br data-spaces-tag="br" data-spaces-attrs="{}">
<a data-spaces-tag="a" data-spaces-attrs="{&quot;href&quot;: &quot;https://github.com/bojone/n2n-ocr-for-qqcaptcha&quot;, &quot;target&quot;: &quot;_blank&quot;}" href="https://github.com/bojone/n2n-ocr-for-qqcaptcha">https://github.com/bojone/n2n-ocr-for-qqcaptcha</a></p><p data-spaces-tag="p" data-spaces-attrs="{}">模型非常简洁，也很常规（只是单文件，能有多复杂？）</p><p data-spaces-tag="p" data-spaces-attrs="{}">就是用4个卷积层提取了图片特征，然后将这个图片特征分别接4个softmax，每个都分为26类，注意这里不能图方便用TimeDistributed，TimeDistributed是权值共享的，而我们这里要对同一个特征分别输出不同的标签，权值相同结果不就也相同了么？另外要注意，我的Keras是用theano做后端，不是tensorflow，两者对图像的处理有所不同，因此用tensorlfow的朋友要自己调整过来。</p><blockquote data-spaces-tag="blockquote" data-spaces-attrs="{}">_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
Layer (type)                     Output Shape          Param #     Connected to<br data-spaces-tag="br" data-spaces-attrs="{}">
=============================================================<br data-spaces-tag="br" data-spaces-attrs="{}">
input_15 (InputLayer)            (None, 3, 129, 53)    0<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
convolution2d_40 (Convolution2D) (None, 32, 127, 51)   896         input_15[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
maxpooling2d_48 (MaxPooling2D)   (None, 32, 63, 25)    0           convolution2d_40[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
convolution2d_41 (Convolution2D) (None, 32, 61, 23)    9248        maxpooling2d_48[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
maxpooling2d_49 (MaxPooling2D)   (None, 32, 30, 11)    0           convolution2d_41[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
activation_37 (Activation)       (None, 32, 30, 11)    0           maxpooling2d_49[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
convolution2d_42 (Convolution2D) (None, 32, 28, 9)     9248        activation_37[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
maxpooling2d_50 (MaxPooling2D)   (None, 32, 14, 4)     0           convolution2d_42[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
activation_38 (Activation)       (None, 32, 14, 4)     0           maxpooling2d_50[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
convolution2d_43 (Convolution2D) (None, 32, 12, 2)     9248        activation_38[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
maxpooling2d_51 (MaxPooling2D)   (None, 32, 6, 1)      0           convolution2d_43[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
flatten_15 (Flatten)             (None, 192)           0           maxpooling2d_51[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
activation_39 (Activation)       (None, 192)           0           flatten_15[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
dense_63 (Dense)                 (None, 26)            5018        activation_39[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
dense_64 (Dense)                 (None, 26)            5018        activation_39[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
dense_65 (Dense)                 (None, 26)            5018        activation_39[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________<br data-spaces-tag="br" data-spaces-attrs="{}">
dense_66 (Dense)                 (None, 26)            5018        activation_39[0][0]<br data-spaces-tag="br" data-spaces-attrs="{}">
=============================================================<br data-spaces-tag="br" data-spaces-attrs="{}">
Total params: 48712<br data-spaces-tag="br" data-spaces-attrs="{}">
_____________________________________________________________</blockquote><p data-spaces-tag="p" data-spaces-attrs="{}">经过几十轮训练后，得到模型，第1、2、3、4字识别的准确率分别是0.89、0.72、0.73、0.87，这样，四个全对的准确率应该是<br data-spaces-tag="br" data-spaces-attrs="{}">
$$0.89\times0.72\times0.73\times0.87\approx 0.41$$<br data-spaces-tag="br" data-spaces-attrs="{}">
即应该会有41%的全对率，经过实际测试，效果还更好一些，全对率为46%，这样差不多两张就有一张识别正确，应该在不少情形都很实用了。当然，这个准确率只针对这批样本的，实际准确率可能还要更低一些，但是估计10%应该有吧？^_^</p><p data-spaces-tag="p" data-spaces-attrs="{}">训练样本就不方便公开了，模型权重也不好直接公开，有需要的，请私下联系我。</p><h3 data-spaces-tag="h3" data-spaces-attrs="{&quot;id&quot;: &quot;后话&quot;}">后话</h3><p data-spaces-tag="p" data-spaces-attrs="{}">按送我样本的朋友的说法，他现在接入了一个别人提供的接口，全对率有95%以上，我瞬间肃然起敬啊，真想好好向那个人学习，不过那个程序已经商业化了，估计也不可能给我观摩了，估计别人就是长期盯着腾讯验证码识别这个需求做的，不像我，泛而不精。</p><p data-spaces-tag="p" data-spaces-attrs="{}">欢迎读者提供更好的建模思路哈，请大家多多指教。目前的模型才5万个参数不到，可能欠拟合了，有空在好好调整一下。</p>
</div>
