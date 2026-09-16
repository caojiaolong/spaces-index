# 什么时候多进程的加速比可以大于1？

> 作者：苏剑林 · 科学空间 · 2019-10-27
>
> 原文：<https://spaces.ac.cn/archives/7031>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

多进程或者多线程等并行加速目前已经不是什么难事了，相信很多读者都体验过。一般来说，我们会有这样的结论：多进程的加速比很难达到1。换句话说，当你用10进程去并行跑一个任务时，一般只能获得不到10倍的加速，而且进程越多，这个加速比往往就越低。

要注意，我们刚才说“很难达到1”，说明我们的潜意识里就觉得加速比最多也就是1。理论上确实是的，难不成用10进程还能获得20倍的加速？这不是天上掉馅饼吗？不过我前几天确实碰到了一个加速比远大于1的例子，所以在这里跟大家分享一下。

## 词频统计

我的原始任务是统计词频：我有很多文章，然后我们要对这些文章进行分词，最后汇总出一个词频表出来。一般的写法是这样的：



<pre><code class="language-python">tokens = {}

for text in read_texts():
    for token in tokenize(text):
        tokens[token] = tokens.get(token, 0) + 1</code></pre>



这种写法在我统计[THUCNews](<http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews>)全部文章的词频时，大概花了20分钟。

## 多进程版本

然后，我们来比较一下多进程版的。多进程的写法我在[《Python的多进程编程技巧》](<https://spaces.ac.cn/archives/4231>)一文已经介绍过了，为了方便重复使用，我就将其封装为一个函数了：



<pre><code class="language-python">def parallel_apply(func,
                   iterable,
                   workers,
                   max_queue_size,
                   callback=None,
                   dummy=False):
    """多进程或多线程地将func应用到iterable的每个元素中。
    注意这个apply是异步且无序的，也就是说依次输入a,b,c，但是
    输出可能是func(c), func(a), func(b)。
    参数：
        dummy: False是多进程/线性，True则是多线程/线性；
        callback: 处理单个输出的回调函数；
    """
    if dummy:
        from multiprocessing.dummy import Pool, Queue
    else:
        from multiprocessing import Pool, Queue
    from six.moves import queue

    in_queue, out_queue = Queue(max_queue_size), Queue()

    def worker_step(in_queue, out_queue):
        # 单步函数包装成循环执行
        while True:
            d = in_queue.get()
            r = func(d)
            out_queue.put(r)

    # 启动多进程/线程
    pool = Pool(workers, worker_step, (in_queue, out_queue))

    if callback is None:
        results = []

    # 后处理函数
    def process_out_queue():
        out_count = 0
        for _ in range(out_queue.qsize()):
            d = out_queue.get()
            out_count += 1
            if callback is None:
                results.append(d)
            else:
                callback(d)
        return out_count

    # 存入数据，取出结果
    in_count, out_count = 0, 0
    for d in iterable:
        in_count += 1
        while True:
            try:
                in_queue.put(d, block=False)
                break
            except queue.Full:
                out_count += process_out_queue()
        if in_count % max_queue_size == 0:
            out_count += process_out_queue()

    while out_count != in_count:
        out_count += process_out_queue()

    pool.terminate()

    if callback is None:
        return results
</code></pre>



调用这个函数来多进程统计词频，大致代码如下：



<pre><code class="language-python">def _batch_texts():
    texts = []
    for text in read_texts():
        texts.append(text)
        if len(texts) == 1000:
            yield texts
            texts = []
    if texts:
        yield texts

def _tokenize_and_count(texts):
    tokens = {}
    for text in texts:
        for token in tokenize(text):
            tokens[token] = tokens.get(token, 0) + 1
    return tokens

tokens = {}
def _total_count(result):
    for k, v in result.items()
        tokens[k] = tokens.get(k, 0) + v

# 10进程来完成词频统计
parallel_apply(
    func=_tokenize_and_count,
    iterable=_batch_texts(),
    workers=10,
    max_queue_size=200,
    callback=_total_count,
)
</code></pre>



整个流程是：<code>_batch_texts</code>将文本按批划分，每批为1000个文本；<code>_tokenize_and_count</code>用来对每一批样本进行统计；<code>_total_count</code>对每一批样本的结果进行汇总；最后<code>parallel_apply</code>用10进程实现这个过程。

这个用时多少呢？结果是55秒！这意味着加速20倍，加速比是2！

## 原理分析

为什么能实现大于1的加速比呢？其实，原因在于最开始的单进程实现中，<code>tokens[token] = tokens.get(token, 0) + 1</code>一句会越来越慢，因为随着统计的推进，<code>tokens</code>里边的元素越来越多，对<code>tokens</code>的增删改查就会越来越慢。

而在多进程版本中，<code>tokens[token] = tokens.get(token, 0) + 1</code>一句只对不超过1000个样本执行，显然会一直保持很快的速度，最后的合并统计结果虽然对<code>tokens</code>的读写也很频繁，但远比不上原始实现的读写频率，因此也是很快的。所以多进程版本就可以实现20倍的加速，而不仅仅是理论上的极限10倍。

当然，读者可能已经感觉到，这并不是真正地让加速比超过了1，而是原始的单进程版写得不好的表象，换成下述代码就好了：



<pre><code class="language-python">count = 0
tokens = {}
_tokens = {}

for text in read_texts():
    for token in tokenize(text):
        _tokens[token] = _tokens.get(token, 0) + 1
    count += 1
    if count == 1000:
        for k, v in _tokens.items():
            tokens[k] = tokens.get(k, 0) + v
        count = 0
        _tokens = {}

for k, v in _tokens.items():
    tokens[k] = tokens.get(k, 0) + v
</code></pre>



也就还是分批统计再汇总的做法，只不过这是单进程的，看上去这种写法很迂回，很不直观，但事实上只用了8分钟，大约只是原来版本的三分之一！由此可见，实际上的加速比大约是0\.8。

## 本文小结

文本简单讨论了一下Python的多进程问题，给出了一个看上去加速比可以大于1的例子，然后分析了其原因。从侧面来看，这其实也给我们写类似的代码提了个醒：哪怕在单进程的情况下，分批计算然后在汇总的效率，也通常会高于一整批一次性计算。
