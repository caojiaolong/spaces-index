# 年三十折腾极路由之SSH反向代理

> 作者：苏剑林 · 科学空间 · 2016-02-07
>
> 原文：<https://spaces.ac.cn/archives/3604>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

[![猴年快乐！](<https://spaces.ac.cn/usr/uploads/2016/02/560663882.png>)](<https://spaces.ac.cn/usr/uploads/2016/02/560663882.png>)

猴年快乐！

<strong>今天是年三十了，这里简单祝大家除夕快乐，新年快乐！愿大家在新的一年里都晋升为学神。^\_^</strong>

这两天主要在折腾家里的路由器。平时家里只有爸妈两人，所以为了节省，家里只是通过中继隔壁家的网络来上网。本来家里用小米路由器mini，可是小米mini中继模式下功能限制非常多，我又不想刷第三方固件（因为这样会失去app控制功能，不是很方便），所以干脆换了个极路由3。极路由在中继模式下仍然保留了大部分功能（我觉得这样才是正常的，我不理解小米mini在中继之后就没了那么多功能究竟是什么逻辑）。

作为折腾派，一个新路由到手，总有很多东西要配置，极路由本身是基于openwrt的，因此可玩性也很强。首先要完成中继，然后上网，这个很简单就不多说了。其次是获得ssh权限，在极路由那里叫做“申请开发者模式”，或者叫root（感觉极路由想做路由界的苹果，但是在如今这个时代，苹果当初那种发展模式估计很难发展起来了），这个步骤也不难，不过申请之后就会失去极路由的保修资格（不理解这是什么逻辑）。

本文主要介绍了怎么在openwrt（极路由）上安装python，以及建立SSH反向代理（实现内网穿透）。

### 安装python到SD卡

有了ssh权限之后，就可以做很多事情了，首先要做的是安装python，这样就能够做很多事情了。这里费了一点周折。首先要修改openwrt的软件源，要修改下面的三个文件

> /etc/opkg\.conf /etc/opkg\.d/opkg\-secure\.conf /etc/opkg\.d/opkg\-fast\.conf

将第一行



<pre><code class="language-bash">src/gz barrier_breaker https://upgrade.hiwifi.com/upgrade_file/ralink-HC5861/0.9017.1.11380s/packages</code></pre>



注释掉，然后在最底下加入



<pre><code class="language-bash">src/gz barrier_breaker http://downloads.openwrt.org.cn/PandoraBox/ralink/mt7620_old/packages/</code></pre>



删除缓存文件之后就可以更新源，并安装软件了。



<pre><code class="language-bash">rm /var/opkg-lists/barrier_breaker
opkg update</code></pre>



要注意，极路由3的ROM剩下的空间不多，因此最好把python装到SD卡中。为此，首先通过df \-h查看一下SD卡的挂载点（我的是/tmp/storage/mmcblk0p2），然后依旧是修改下面三个文件

> /etc/opkg\.conf /etc/opkg\.d/opkg\-secure\.conf /etc/opkg\.d/opkg\-fast\.conf

在



<pre><code class="language-bash">dest root /</code></pre>



这一行下面加入



<pre><code class="language-bash">dest usb /tmp/storage/mmcblk0p2</code></pre>



此外，要修改系统变量，为此要修改/etc/profile，如（留意红色部分）



<pre><code class="language-bash">export PATH=/tmp/storage/mmcblk0p2/usr/bin:/tmp/storage/mmcblk0p2/usr/sbin:/bin:/sbin:/usr/bin:/usr/sbin:$HIWIFI_CRYPTDA
export LD_LIBRARY_PATH=/tmp/storage/mmcblk0p2/lib:/tmp/storage/mmcblk0p2/usr/lib:$HIWIFI_CRYPTDATA/lib:$HIWIFI_CRYPTDATA</code></pre>



这时，就可以用



<pre><code class="language-bash">opkg install python -d usb</code></pre>



将python安装到SD卡了。然后安装setuptools得到easy\_install，接着安装pip，还有requests库，至此，我的python环境配置完成。

### 配置SSH反向代理

接着就想办法穿透内网，让我上学之后在学校也能管理家里的路由器。这方面的方法有很多，比如ngrok（我没试过，但是看教程，感觉配置略微有些麻烦）、n2n（这个基于p2p，我很喜欢，在openwrt也能用，在我学校的路由器配置成功，但是在极路由上运行失败，只能作罢）、花生壳内网版（极路由有花生壳内网版插件可以用，感觉还可以，就是免费版限制太多）、SSH反向代理，无奈前三个都不满意，所以只能SSH反向代理了。

SSH反向代理的原理，是我们的SSH的逆用，跟我们正常登录SSH的顺序刚好相反：

> <strong>正常SSH：</strong>
> 我的电脑 \-\-\-\-\> 内网路由器 \-\-\-\-\> 我的VPS
> 
> 
> 
> <strong>反向代理：</strong>
> 我的电脑 \-\-\-\-\> 你的VPS \-\-\-\-\> 内网路由器

实现很简单，一句代码的事情，而且几乎所有平台都能用（能ssh就行了），当然，准备功夫很多。

使用SSH反向代理，需要有自己的一台有公网IP的VPS，我的是阿里云服务器，学生价9\.9元/月，很实惠。首先，在VPS进行设置，修改/etc/ssh/sshd\_config加入



<pre><code class="language-bash">GatewayPorts yes</code></pre>



跟着/etc/init\.d/ssh restart重启ssh；然后只需要在极路由openwrt上，通过下面代码建立通往VPS的反向隧道



<pre><code class="language-bash">ssh -Nfg -R 11111:192.168.199.1:1022 root@1.1.1.1</code></pre>



跟普通的ssh登陆是一样的，其中1\.1\.1\.1是VPS的公网IP，192\.168\.199\.1是极路由的管理IP，1022是极路由的ssh端口（因为我只是要ssh登录到极路由），登陆成功后，在任何地方地方1\.1\.1\.1:11111，就相当于访问路由器的192\.168\.199\.1:1022了。比如可以ssh root@1\.1\.1\.1 \-p 11111看看能不能登录到路由器。（也可以在VPS上用netstat \-nlp\|grep sshd看一下有没有成功建立端口代理。）至于上述代码的参数的意义，可以通过ssh \-h查看。

到这里，问题已经基本解决，但是有几个问题。第一，ssh是需要交互式地手动输入密码。如果要免密钥登录，需要在本地生成密钥，然后上传到VPS中（即私钥\-公钥机制），具体步骤是先通过以下代码生成公钥（这是openwrt上自带的方案，一般的linux系统应该用ssh\-keygen）



<pre><code class="language-bash">dropbearkey -t rsa -f /etc/dropbear/id_rsa
dropbearkey -y -f /etc/dropbear/id_rsa | grep ssh-rsa &gt; /tmp/id_rsa.pub</code></pre>



接着将/tmp/id\_rsa\.pub的内容，添加到VPS服务器的\~/\.ssh/authorized\_keys中（如果没有就新建一个）,然后通过



<pre><code class="language-bash">ssh -i /etc/dropbear/id_rsa -Nfg -R 11111:192.168.199.1:1022 root@1.1.1.1</code></pre>



进行登录，就会发现不用输入密码了。

第二，ssh本身不是很稳定，长时间没响应，它会自动断开，解决这个问题的一个办法是不用自带的ssh，而用autossh（opkg install autossh \-d usb），它是一个带有监控功能的ssh客户端。autossh的代码是类似的



<pre><code class="language-bash">autossh -i /etc/dropbear/id_rsa -M 11112 -Nfg -R 11111:192.168.199.1:1022 1.1.1.1</code></pre>



它的原理是增加一个11112端口来监听11111端口的连通状态，如果断开自动重连。

第三，如何随路由器启动。路由器随时都可能断电重启，因此需要加入到启动项中，这个也简单，加入到/etc/rc\.local中即可。但有两个小小问题：（1）我的路由器是通过中继上网的，很有可能运行这个脚本的时候，还没有连接上网络，如果没上网连接ssh是没意义的，因此需要判断连上网络了没有，连上网络后再执行（当然，可以通过sleep等待足够多的时间，然后才执行，但不是很靠谱）。我判断网络是否连通的方法很简单，我在自己网站建立了一个test\.html，内容仅仅是一个字符串pass，访问这个页面，如果能够得到pass，就说明网络连通；（2）如果这一次成功建立了隧道，然后路由器重启后，相应的隧道还缓存在VPS中，这样下一次运行上述代码的时候，会默认使用缓存的隧道，依然会显示建立成功，但是如果ssh root@1\.1\.1\.1 \-p 11111就会失败（这时就清空了VPS缓存），因此建立ssh隧道后，需要试用一次，检测是否可用，不可用要重新建立，ssh本身是阻塞运行的，这一点我们可以用上。

代码如下



<pre><code class="language-bash">#!/bin/sh

local_ip=`ifconfig br-lan|grep 'inet addr'|awk '{print $2}'|awk -F: '{print $2}'`
vps_ip=1.1.1.1

while true
do
 wget http://kexue.fm/test.html
 a=`cat test.html`
 rm test.html
 if [ $a = 'pass' ]
    then
        autossh -i /etc/dropbear/id_rsa -M 11112 -Nfg -R 11111:$local_ip:1022 $vps_ip
        ssh -i /etc/dropbear/id_rsa -N $vps_ip -p 11111
    fi
    sleep 10
done</code></pre>



这把LAN口IP检测、网络连通检测、可用检测、自动重连都整合在里边了，将这个代码保存为run\.sh，放在根目录，chmod \+x run\.sh获得执行权，然后在/etc/rc\.local的exit 0前加入



<pre><code class="language-bash">source /etc/profile
/run.sh</code></pre>



第一句是很重要的，因为我的autossh等安装在SD卡，需要加载好系统变量才能正常使用；而运行这个脚本的时候，不一定加载好了系统变量，因此最好手动导入，保证能够正常运行。

至此，终于折腾好了。为了方便访问，我还将自己的一个子域名绑定到VPS中，这样就搭建好了一个比较完美的内网穿透平台了，基本上完胜花生壳内网版。

### 参考链接

[https://yxz\.me/archives/881\.html](<https://yxz.me/archives/881.html>)

[http://blog\.csdn\.net/jk110333/article/details/11920163](<http://blog.csdn.net/jk110333/article/details/11920163>)

[https://blog\.phpgao\.com/ssh\-reverse\-tunnel\.html](<https://blog.phpgao.com/ssh-reverse-tunnel.html>)

[http://www\.cage\.tk/2015/01/22/openwrt\-use\-ssh\-connect\-access/](<http://www.cage.tk/2015/01/22/openwrt-use-ssh-connect-access/>)
