# 在bert4keras中使用混合精度和XLA加速训练

> 作者：苏剑林 · 科学空间 · 2022-04-28
>
> 原文：<https://spaces.ac.cn/archives/9059>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

之前笔者一直都是聚焦于模型的构思和实现，鲜有关注模型的训练加速，像混合精度和XLA这些技术，虽然也有听过，但没真正去实践过。这两天折腾了一番，成功在bert4keras中使用了混合精度和XLA来加速训练，在此做个简单的总结，供大家参考。

本文的多数经验结论并不只限于bert4keras中使用，之所以在标题中强调bert4keras，只不过bert4keras中的模型实现相对较为规整，因此启动这些加速技巧所要做的修改相对更少。

## 实验环境

本文的实验显卡为3090，使用的docker镜像为nvcr\.io/nvidia/tensorflow:21\.09\-tf1\-py3，其中自带的tensorflow版本为1\.15\.5。另外，实验所用的bert4keras版本为0\.11\.3。其他环境也可以参考着弄，要注意有折腾精神，不要指望着无脑调用。

顺便提一下，3090、A100等卡只能用cuda11，而tensorflow官网的1\.15版本是不支持cuda11的，如果还想用tensorflow 1\.x，那么只能用nvidia亲自维护的[nvidia\-tensorflow](<https://github.com/NVIDIA/tensorflow>)，或者用其构建的[docker镜像](<https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html>)。用nvidia而不是google维护的tensorflow，除了能让你在最新的显卡用上1\.x版本外，还有nvidia专门做的一些额外优化，具体文档可以参考[这里](<https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html>)。

不要说“tensorflow都出到2\.8了，怎么还用1\.15”这样的话，你的显卡是nvidia产的，所以哪个版本的tensorflow最好用，你我说了不算，甚至Google说了都不算，nvidia说的才算，nvidia还在维护着1\.15，那说明1\.15才是yyds。

## 混合精度

首先我们来看混合精度训练，简单来说就是模型计算用FP16、参数更新和存储用FP32，FP16的表示范围大致是$6\times 10^{-8}\sim 65504$，其上下界都是我们在实现模型时有可能触碰到的，所以引入FP16后最大的问题就是溢出和精度损失。更详细的原理介绍大家自行搜索就好，本文主要关注怎么用。

nvidia\-tensorflow的帮助文档中对混合精度训练的介绍可见[这里](<https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#tfamp>)，其中启动混合精度训练最简单的方法是脚本的开头添加环境变量：



<pre><code class="language-python">import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'  # 混合精度训练</code></pre>



读者或许留意到，多数教程介绍的是 TF\_ENABLE\_AUTO\_MIXED\_PRECISION 而我这里是 TF\_ENABLE\_AUTO\_MIXED\_PRECISION\_GRAPH\_REWRITE ，它们的区别在于前者会自动添加“动态损失放大（Loss Scaling）”而后者不会，但笔者测试发现“动态损失放大”并不能替代手动调整损失，因此干脆不要这个功能了。

添加完环境变量后，可以重新启动训练脚本看看情况。如果训练开始就出现了NaN，那么可以调整一下infinity和epsilon：



<pre><code class="language-python">from bert4keras.backend import K
K.set_infinity(1e4)
K.set_epsilon(1e-5)
</code></pre>



调整完后通常不会一开始就NaN了（如果还有，那就检查一下模型其他地方有没有用到不受这这两个函数控制的 infinity 和 epsilon 并修改过来），但有可能出现的是loss先降后升最后NaN，这是因为初始化不好，或者是像[DeepNet](<https://spaces.ac.cn/archives/8994>)那样刻意为之，使得模型存在部分参数的梯度极小（小于$10^{-8}$），这时候在FP16的精度内它就直接等于0了，于是这部分参数不会得到更新，或者等价说梯度是不准的，长时间用不准的梯度更新，就容易不收敛。

这时候解决方案就是“损失放大”了。我们可以直接在损失函数上乘上一个放大因子（比如1000，可以自行调试，不出现NaN的前提下越大越好），使得原本很小的梯度就得以放大到FP16范围内，不至于直接置零，避免了梯度的精度损失。而对于我们平时用的Adam、[LAMB](<https://spaces.ac.cn/archives/8978>)等优化器来说，损失函数乘上一个常数并不会改变这些优化器的训练过程，也就是它们完全是兼容“损失放大”的。

事实上，笔者发现“损失放大”技巧不仅仅在混合精度训练场景下有效，即便是全FP32精度训练也会有一定作用：在全FP32精度训练时，如果不进行损失放大，开始阶段模型会停留在某个损失值一段时间，然后才慢慢下降；而如果进行了损失放大，那么开始阶段模型就一直保持缓慢下降趋势，相对来说收敛更快了。

## 代数加速

现在我们来看XLA，全称为“Accelerated Linear Algebra”，即专门用来加速线性代数运算的。简单来说，XLA就是对计算图提前进行编译优化，将能合并的算子进行合并（减少缓存变量以节省内存），将能并行的算子进行并行（提高计算速度）。

在nvidia\-tensorflow中，启动XLA的最简单方式依旧是添加环境变量：



<pre><code class="language-python">import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'  # 启用XLA</code></pre>



但要注意，XLA不是保证有提升的，刚才我们说到，XLA会将能并行的算子尽量并行，很明显这是通过空间换时间的方案，因此启用XLA后可能会消耗更多的显存以导致OOM，甚至并行簇过大时反而会导致性能下降。[官方文档](<https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-best-practices>)对有可能出现的异常做了比较详尽的分析并提出了相应的建议，其中笔者推荐的解决方法是补充<code>--tf_xla_enable_lazy_compilation=false</code>参数：



<pre><code class="language-python">import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'  # 启用XLA
os.environ['TF_XLA_FLAGS'] += ' --tf_xla_enable_lazy_compilation=false'  # 优化XLA</code></pre>



如果这都不能解决，那就换成XLA Lite：



<pre><code class="language-python">import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'  # 启用XLA Lite</code></pre>



如果换成XLA Lite都无法解决，那基本就说明XLA不适合你的模型了。

## 性能比较

在3090上，启动混合精度训练带来的加速大概是10%多一点。这个幅度可能不如大家想象的那么快，笔者猜测这是因为3090、A100等新卡上面，默认的FP32格式实际上用的是一种名为TF32的格式（参考[这里](<https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/>)），TF32某种意义来说本身就是一种“半精度格式”，比FP32更快。换句话说，3090上的FP32本身就相当于已经做过一定的半精度优化了，速度本来就更快，因此换成混合精度后的提升相对来说就小了。

至于XLA带来的提升，大致是15%左右。在笔者的训练脚本中，直接设置环境变量 TF\_XLA\_FLAGS 为<code>--tf_xla_auto_jit=1</code>会OOM，补充<code>--tf_xla_enable_lazy_compilation=false</code>依旧，而改为<code>--tf_xla_auto_jit=fusible</code>则可以正常训练。

最后，最关键的是，混合精度与XLA可以叠加使用！两者一起使用带来的加速大概是30%左右，并且混合精度训练的加入基本上可以抵消XLA带来的显存消耗增加，两者真可谓是相得益彰了。

## 文章小结

本文介绍了在bert4keras中使用混合精度和XLA加速训练的尝试，两者同时启用大概能在3090上加速30%左右。
