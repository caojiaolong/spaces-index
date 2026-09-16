# 用Numpy实现高效的Apriori算法

> 作者：苏剑林 · 科学空间 · 2018-05-10
>
> 原文：<https://spaces.ac.cn/archives/5525>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![关联规则的经典例子：啤酒与尿布](<https://spaces.ac.cn/usr/uploads/2018/05/3503635631.jpg>)](<https://spaces.ac.cn/usr/uploads/2018/05/3503635631.jpg>)

关联规则的经典例子：啤酒与尿布

三年前笔者曾写了[《用Pandas实现高效的Apriori算法》](<https://spaces.ac.cn/archives/3380/>)，里边给出了Apriori算法的Python实现，并得到了一些读者的认可。然而，笔者当时的Python还学得并不好，所以现在看来那个实现并不优雅（但速度还过得去），而且还不支持变长的输入数据。而之前承诺过会重写这个算法，把上述问题解决掉，而现在总算完成了～

关于Apriori算法就不重复介绍了，直接放出代码：



<pre><code class="language-python">#! -*- coding: utf-8 -*-

import numpy as np


class Apriori:

    def __init__(self, min_support, min_confidence):
        self.min_support = min_support # 最小支持度
        self.min_confidence = min_confidence # 最小置信度

    def count(self, filename='apriori.txt'):
        self.total = 0 # 数据总行数
        items = {} # 物品清单

        # 统计得到物品清单
        with open(filename) as f:
            for l in f:
                self.total += 1
                for i in l.strip().split(','): # 以逗号隔开
                    if i in items:
                        items[i] += 1.
                    else:
                        items[i] = 1.

        # 物品清单去重，并映射到ID
        self.items = {i:j/self.total for i,j in items.items() if j/self.total &gt; self.min_support}
        self.item2id = {j:i for i,j in enumerate(self.items)}

        # 物品清单的0-1矩阵
        self.D = np.zeros((self.total, len(items)), dtype=bool)

        # 重新遍历文件，得到物品清单的0-1矩阵
        with open(filename) as f:
            for n,l in enumerate(f):
                for i in l.strip().split(','):
                    if i in self.items:
                        self.D[n, self.item2id[i]] = True

    def find_rules(self, filename='apriori.txt'):
        self.count(filename)
        rules = [{(i,):j for i,j in self.items.items()}] # 记录每一步的频繁项集
        l = 0 # 当前步的频繁项的物品数

        while rules[-1]: # 包含了从k频繁项到k+1频繁项的构建过程
            rules.append({})
            keys = sorted(rules[-2].keys()) # 对每个k频繁项按字典序排序（核心）
            num = len(rules[-2])
            l += 1
            for i in range(num): # 遍历每个k频繁项对
                for j in range(i+1,num):
                    # 如果前面k-1个重叠，那么这两个k频繁项就可以组合成一个k+1频繁项
                    if keys[i][:l-1] == keys[j][:l-1]:
                        _ = keys[i] + (keys[j][l-1],)
                        _id = [self.item2id[k] for k in _]
                        support = 1. * sum(np.prod(self.D[:, _id], 1)) / self.total # 通过连乘获取共现次数，并计算支持度
                        if support &gt; self.min_support: # 确认是否足够频繁
                            rules[-1][_] = support

        # 遍历每一个频繁项，计算置信度
        result = {}
        for n,relu in enumerate(rules[1:]): # 对于所有的k，遍历k频繁项
            for r,v in relu.items(): # 遍历所有的k频繁项
                for i,_ in enumerate(r): # 遍历所有的排列，即(A,B,C)究竟是 A,B -&gt; C 还是 A,B -&gt; C 还是 A,B -&gt; C ？
                    x = r[:i] + r[i+1:]
                    confidence = v / rules[n][x] # 不同排列的置信度
                    if confidence &gt; self.min_confidence: # 如果某种排列的置信度足够大，那么就加入到结果
                        result[x+(r[i],)] = (confidence, v)

        return sorted(result.items(), key=lambda x: -x[1][0]) # 按置信度降序排列
</code></pre>



使用方法：



<pre><code class="language-python">from pprint import pprint

# 测试文件来自 https://kexue.fm/archives/3380
model = Apriori(0.06, 0.75)
pprint(model.find_rules('apriori.txt'))
</code></pre>



输出结果：

> &#91;(('A3', 'F4', 'H4'), (0\.8795180722891566, 0\.07849462365591398)),
> (('C3', 'F4', 'H4'), (0\.875, 0\.07526881720430108)),
> (('B2', 'F4', 'H4'), (0\.7945205479452054, 0\.06236559139784946)),
> (('C2', 'E3', 'D2'), (0\.7543859649122807, 0\.09247311827956989)),
> (('D2', 'F3', 'H4', 'A2'), (0\.7532467532467533, 0\.06236559139784946))&#93;

结果的含义是前n\-1项推出第n项，也就是A3 \+ F4 \-\-\> H4、 D2 \+ F3 \+ H4 \-\-\> A2等。

这次的实现相对简洁一点，去掉了Pandas库的依赖，只用到了Numpy库，变量命名也更清晰一些，编程思路没有变化，所以理论上效率不会降低，甚至有可能会提升。应该兼容Python 2\.x和3\.x，如果有问题欢迎继续指出。
