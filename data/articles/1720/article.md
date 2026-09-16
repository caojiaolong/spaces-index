# 开始学习数学软件Scilab

> 作者：苏剑林 · 科学空间 · 2012-09-28
>
> 原文：<https://spaces.ac.cn/archives/1720>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

其实很早之前我就想学习一款数学软件的使用，以前很感兴趣的是mathematica，也玩弄过一阵子，但毕竟在高中没有多大需要，也就没有坚持下来。更重要的是，这些软件都是要收费的。上了大学后，听了师兄姐对数学建模的讲述，发现他们基本上也是用mathematica或者matlab的，但这两个软件都是要收费的，我不大想用破解版本。既然我都已经用上了ubuntu了，那么我就该好好利用它。据说命令跟matlab很相似的软件是scilab，还有octave，不同的是这些都是开源免费的。

出于熟悉代码操作和数学软件编程的目的，我选择了学习scilab。虽然网上说octave与matlab的相似程度更高，但是我感觉scilab比octave用的更广一些，所以就用它。所谓“一理通百理明”，先专心学好一个。

下面是我编写的<strong>第一个scialb程序</strong>，利用威尔逊方法来进行素性测试。这个代码的主要目的是练习条件语句和循环语句，以及一些输出输入的技巧而已。程序本身比较丑陋。



<pre><code class="language-python">//我的第一个scilab程序
//完成于2012.09.27

label1=['p:';];  //定义标签
B=x_mdialog(['本程序使用威尔逊方法判断进行素数测试。';'请输入要判断的数'],label1,['127';]);  //输入框
p=evstr(B(1));  //提取输入框里边的数字进行赋值
i=1;
j=1;
q=p-1;
while i&lt;q
    j=j*i;
    j=modulo(j,p);//这个是模函数。
    i=i+1;
end
if j==1
    messagebox(['这是一个素数';],['测试结果']);  //输出，其中后边的“测试结果”是输入框的标题
else
    messagebox(['这是一个合数';],['测试结果']);
end
</code></pre>



这是我的<strong>第二个程序</strong>，主要是为了加强记忆和练习算法而已。程序使用牛顿法来求解三次方程$x^3+ax^2+bx+c=0$，没有进行一些错误判断。因为我觉得细节方面不是我目前的主要任务。



<pre><code class="language-python">//我的第二个scilab程序
//完成于2012.09.28

label1=['a:';'b:';'c:';'初始值';'精确度';];  //定义标签
B=x_mdialog(['本程序使用牛顿法求解x^3+ax^2+bx+c=0';'请输入系数、初始值、精确度'],label1,['1';'1';'1';'1';'0.001';]);  //输入框
a=evstr(B(1));  //提取输入框里边的数字进行赋值
b=evstr(B(2));
c=evstr(B(3));
d=evstr(B(4));
q=evstr(B(5));
x=d;
e=x^3+a*x^2+b*x+c;
while abs(e)&gt;q
    x=x-e/(3*x^2+2*a*x^2+b);
    e=x^3+a*x^2+b*x+c;
end

x//输出答案</code></pre>



下面是我的<strong>第三个scilab程序</strong>。是为了解决“数学研发论坛”上面的一道题目而写的，题目如下：
[http://bbs\.emath\.ac\.cn/viewthread\.php?tid=3731](<http://bbs.emath.ac.cn/viewthread.php?tid=3731>)

主要是输出素数数列，其中第n个的后n\-1位是数列中的第n\-1个数。比如7，17，317，6317，\.\.\.
由于要进行素性测试，但是我在scilab没有找到素数判断函数，所以首先我自定义了一个新函数prime()，使用埃拉托斯特尼筛法来判断素数。



<pre><code class="language-python">//我的第三个scilab程序
//输出素数数列，其中第n个的后n-1位是数列中的第n-1个数
//完成于2012.09.28

function p=prime(x)
    p=1
    for i=3:2:x^0.5;
        if modulo(x,i)==0 then
            p=0;
            break
        end
    end
endfunction
//上面是自定义新函数来进行素数判断。素数则输出1，合数则输出0
//被判断数要大于9

a=7
j=0;
while j&lt;8
    if prime(a)==1
        a
        j=j+1;
        a=a+10^j;
    else
        a=a+10^j;
    end
end</code></pre>



以上都是我刚刚开始学习scilab的拙作，权当纪念之用，中高手请忽略。等熟悉了scilab的各种函数和结构等等之后，会尽可能多把编程用于物理等科学问题上。当然，希望高手多多指教啦。
