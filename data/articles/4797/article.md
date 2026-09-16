# 增强typecho的搜索功能

> 作者：苏剑林 · 科学空间 · 2018-01-09
>
> 原文：<https://spaces.ac.cn/archives/4797>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

科学空间是使用typecho程序搭建的博客，侧边栏提供了搜索功能，然而typecho内置搜索功能仅仅是基于字符串的全匹配查找，因此导致很多合理的查询都没法得到结果，比如“2018天象”、“新词算法”都没法给出结果，原因就是文章中都不包含这些字符串。

于是就萌生了加强搜索功能的想法，之前也有读者建议过这个事情。这两天搜索了一下，本来计划用Python下的Whoosh库来建立一个全文检索引擎，但感觉整合和后期维护的工作量太大，还是放弃了。后来想到在typecho自身的搜索上加强，在公司同事（大佬）的帮助下，完成了这个改进。

由于是直接修改typecho源文件实现的改进，因此如果typecho升级后就可能被覆盖，因此在这里做个备忘。

## 探索

通过在[Github](<https://github.com/typecho/typecho/>)检索我发现，typecho的搜索功能是在<code>var/Widget/Archive.php</code>中实现的，具体代码大概在1185～1192行：



<pre><code class="language-php">        if (!$hasPushed) {
            $searchQuery = '%' . str_replace(' ', '%', $keywords) . '%';
            /**搜索无法进入隐私项保护归档 */

            $select-&gt;where('table.contents.password IS NULL')
            -&gt;where('table.contents.title LIKE ? OR table.contents.text LIKE ?', $searchQuery, $searchQuery)
            -&gt;where('table.contents.type = ?', 'post');
        }
</code></pre>



可见，搜索结果是通过在SQL中匹配keywords返回的，其中%是SQL中的通配符。因此我们还发现，如果我们输入查询语句是自带空格的话，那么空格也会被替换成通配符，这样搜索起来就灵活一点。

因此很自然的一个想法是，不管查询语句有没有空格，我们人工对查询语句进行分词，然后用通配符连接分词结果，从而实现在没有空格的情况下也更灵活搜索。这确实是我实践的第一个思路。然而这样做存在的问题是：尽管进行了分词，然而还是要匹配完所有的词才出结果，如果有一个词在博客中从未出现过，那么就匹配不到了。于是要想更好，那么需要考虑每个词都只是候选词而不是必选词的做法。

## 实践

为了实现上述目的，我用Python写了个http接口，放到服务器上，这个http接口负责分词并生成SQL语句，然后将<code>$keywords = $this-&gt;request-&gt;filter('url', 'search')-&gt;keywords;</code>替换为<code>$keywords = $this-&gt;request-&gt;keywords;</code>，并改写上述代码为



<pre><code class="language-php">        if (!$hasPushed) {
            $url = 'http://127.0.0.1:7777/token?text=' . $keywords;
            $url = str_replace(' ', '%20', $url);
            $searchQuery = file_get_contents($url);

            /**当接口失效时使用简单全匹配 */
            if (!$searchQuery) {
                $searchQuery = 'SIGN(INSTR(table.contents.title, "' . $keywords . '"))';
                $searchQuery = $searchQuery . ' + SIGN(INSTR(table.contents.text, "' . $keywords . '"))';
            }

            /**搜索无法进入隐私项保护归档 */
            $select-&gt;where('table.contents.password IS NULL')
            -&gt;where($searchQuery . ' &gt; 0')
            -&gt;where('table.contents.type = ?', 'post')
            -&gt;order($searchQuery, Typecho_Db::SORT_DESC);
        }
</code></pre>



其中接口<code>http://127.0.0.1:7777/token?text=</code>是Python程序：



<pre><code class="language-python">#! -*- coding:utf-8 -*-

import bottle
import jieba
jieba.initialize()

def convert(s):
    ws = jieba.cut(s)
    search = []
    for i in ws:
        search.append('2*SIGN(INSTR(table.contents.title, "%s"))'%i)
        search.append('SIGN(INSTR(table.contents.text, "%s"))'%i)
    return '(%s)'%(' + '.join(search))

@bottle.route('/token', method='GET')
def token_home():
    text = bottle.request.GET.get('text')
    if not text:
        text = ''
    return convert(text)

if __name__ == '__main__':
    bottle.run(host='0.0.0.0', port=7777, server='gunicorn')
</code></pre>



这个接口返回的是SQL语句的算分部分，具体算法是：先分词，如果文章标题中包含一个词，那么加2分，如果文章内容包含一个词，加1分，最后算个总分，用到的函数SIGN、INSTR等，大家百度一下就知道了。这里推荐一下，用bottle这个轻量级的库写http接口是非常方便的～

还有要修改的是：因为我们修改的php部分，用了<code>order($searchQuery, Typecho_Db::SORT_DESC);</code>来希望按分数降序排列。然而这不会直接生效，因为typecho中默认全部按时间降序排列，因此我们还要修改同一个文件的1396～1397行，将原来是



<pre><code class="language-php"> $select-&gt;order('table.contents.created', Typecho_Db::SORT_DESC)
        -&gt;page($this-&gt;_currentPage, $this-&gt;parameter-&gt;pageSize);
</code></pre>



改为



<pre><code class="language-php">        if (strpos($select, 'INSTR') === false) {
            $select-&gt;page($this-&gt;_currentPage, $this-&gt;parameter-&gt;pageSize)
            -&gt;order('table.contents.created', Typecho_Db::SORT_DESC);
        } else {
            $select-&gt;page($this-&gt;_currentPage, $this-&gt;parameter-&gt;pageSize);
        }
</code></pre>



大概意思是判断一下是不是搜索语句，如果是的话，那么就不按时间排列；如果不是的话，就按时间排列。直接去掉按时间排列是不行的，因为这一句也包含了首页的输出，首页的输出必须按照时间排序。

## 结语

为什么要用这种Python和PHP结合的方案，而不纯写成PHP版？没错，写成纯PHP也可以，结巴分词的确也有PHP版，然而最重要的问题是我不会PHP！而且PHP版的结巴也需要额外配置，略麻烦。像这样用Python对我来说就简单多了，如果有什么要改进的，修改Python脚本即可。

最后，也许有使用者会担心这么粗暴的解决方法会不会存在效率问题，事实上，如果文章多达几十万的话，那么上述做法肯定有很严重的效率问题，然而对于一个只有几百篇文章的博客来说，这个问题并不需要考虑了。

终于可以用更自由地搜索了～欢迎大家更多的建议。
