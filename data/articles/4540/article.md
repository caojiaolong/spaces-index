# fashion\-mnist的gan玩具

> 作者：苏剑林 · 科学空间 · 2017-08-26
>
> 原文：<https://spaces.ac.cn/archives/4540>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![fashion_mnist_demo](<https://spaces.ac.cn/usr/uploads/2017/08/1865825983.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/1865825983.png>)

fashion\_mnist\_demo

mnist的手写数字识别数据集一直是各种机器学习算法的试金石之一，最近有个新的数据集要向它叫板，称为fashion\-mnist，内容是衣服鞋帽等分类。为了便于用户往fashion\-mnist迁移，作者把数据集做成了几乎跟mnist手写数字识别数据集一模一样——同样数量、尺寸的图片，同样是10分类，甚至连数据打包和命名都跟mnist一样。看来fashion mnist为了取代mnist，也是拼了，下足了功夫，一切都做得一模一样，最大限度降低了使用成本～这叫板的心很坚定呀。

叫板的原因很简单——很多人吐槽，如果一个算法在mnist没用，那就一定没用了，但如果一个算法在mnist上有效，那它也不见得在真实问题中有效～也就是说，这个数据集太简单，没啥代表性。

fashion\-mnist的github：[https://github\.com/zalandoresearch/fashion\-mnist/](<https://github.com/zalandoresearch/fashion-mnist/>)
公众号的介绍：[《取代MNIST？德国时尚圈的科学家们推出基准数据集，全是衣裤鞋包》](<https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247488402&idx=3&sn=ff4bc5c8666c9d0790dc4dded1f673ca>)

fashion\-mnist共有10个类，分别是：T\-shirt/top、Trouser、Pullover、Dress、Coat、Sandal、Shirt、Sneaker、Bag、Ankle boot。不少人都在跑实验，看把原来在mnist奏效的模型直接搬到这个新的数据集的效果如何～[这里](<http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/>)已经有一些benchmark可以参考了。

我也来凑了个热闹，把之前在mnist有效的wgan\-gp代码（请看[《互怼的艺术：从零直达WGAN\-GP》](<https://spaces.ac.cn/archives/4439/>)）在这个数据集跑了一遍，发现还是work的。用法很非常简单，只需要改一下路径即可～

代码：[https://github\.com/bojone/gan/blob/master/fashion\_mnist\_gangp\.py](<https://github.com/bojone/gan/blob/master/fashion_mnist_gangp.py>)

使用方法



<pre><code class="language-bash">git clone https://github.com/zalandoresearch/fashion-mnist.git
cd fashion-mnist
wget https://raw.githubusercontent.com/bojone/gan/master/fashion_mnist_gangp.py
python fashion_mnist_gangp.py
</code></pre>



对比原来的文件，就只改动了路径。在当前目录下的out目录，可以看到逐渐生成的样本，越来越清晰，并越来越多样化～当然，跟原图差距还是有的，估计是模型容量不足的问题，大家可以自行折腾，笔者仅是抛砖引玉了。

部分生成样本（顺序：从左往右，从上往下。可以发现，中间有个过程图片会变得比之前更模糊，但可以发现之后图片就更加多样化了。）：

[![fashion_mnist_gan_1](<https://spaces.ac.cn/usr/uploads/2017/08/3437015829.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/3437015829.png>)

fashion\_mnist\_gan\_1

[![fashion_mnist_gan_2](<https://spaces.ac.cn/usr/uploads/2017/08/1678228052.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/1678228052.png>)

fashion\_mnist\_gan\_2

[![fashion_mnist_gan_3](<https://spaces.ac.cn/usr/uploads/2017/08/3711757182.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/3711757182.png>)

fashion\_mnist\_gan\_3

[![fashion_mnist_gan_4](<https://spaces.ac.cn/usr/uploads/2017/08/1184928247.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/1184928247.png>)

fashion\_mnist\_gan\_4

[![fashion_mnist_gan_5](<https://spaces.ac.cn/usr/uploads/2017/08/1083299594.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/1083299594.png>)

fashion\_mnist\_gan\_5

[![fashion_mnist_gan_6](<https://spaces.ac.cn/usr/uploads/2017/08/3263927284.png>)](<https://spaces.ac.cn/usr/uploads/2017/08/3263927284.png>)

fashion\_mnist\_gan\_6
