# 进驻中山大学南校区，折腾校园网

> 作者：苏剑林 · 科学空间 · 2016-09-05
>
> 原文：<https://spaces.ac.cn/archives/3936>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

开始研究僧之旅，希望有一天能企及扫地僧的境界。

进入中山大学后，各种郁闷的事情就来了。首先最郁闷的就是开学时间特早，8月26日开学，感觉至少比一般学校早了一星期，开学这么早有意思么～～接着就是感觉中大的管理制度各种混乱，比我本科的华师差多了。好吧，这些琐事先不吐槽，接下来弄校园网，这是作死的开始。

我们是在南校区的，校园网是通过锐捷客户端来认证的，而我是用macbook的，不过中大这边还很人性化地提供了Mac版的锐捷，体积就1M左右，挺好的。但众所周知，macbook并没有有线网卡，每次我上网都得插着个USB网卡然后连着网线，这该有多郁闷。于是想办法通过路由器拨号。我也不算没经验的了，对openwrt这个系统有过一定研究，以前在本科的时候也是锐捷，可以用mentohust替代拨号，很简单。于是我在这里重复这样的过程，发现一直认证失败，按照网上提示的各种方法，都无法解决。

经过研究，我发现在Windows下，这里就只能用官方提供了锐捷4\.90版本，从其他地方下载的更高级或者更低级的锐捷，都无法通过验证。估计就是因为这个机制，导致了mentohust难以通过验证。而且网上流行的mentohust都是基于V2协议的，但4\.90是基于V4的。后来我又去下载了[V4版本](<https://github.com/hyrathb/mentohust>)的进行交叉编译，测试发现还不成功。几近绝望的时候，我发现了[mentohust\-proxy](<https://github.com/updateing/mentohust-proxy>)，一个mentohust的改进版，让我找到了希望。（怎么找到它？我是直接到github搜索了，因为实在没辙了～～）

原理很简单，如果直接通过mentohust无法完成认证，那么就通过代理模式，由电脑来完成认证，而mentohust只需要负责发送心跳包维持联网就行。这是个很折中的方案，但应该说是一个很通用的方案，因为它的成功与否，基本就取决于自己电脑的锐捷客户端而已。看到这个方案，我就知道有戏了，于是赶紧补习了一下交叉编译的知识，最后成功编译好了，并且在路由上成功地完成了认证。

### 交叉编译

我的环境是CentOS 7，首先安装环境依赖，其实这些依赖都是网上找来的，我也不知道全不全，如果不行再留言吧。大家直接搜索一下“openwrt编译”，估计就能找到不少相关链接



<pre><code class="language-bash">sudo yum install autoconf binutils bison bzip2 flex gawk gcc gcc-c++ gettext make ncurses-devel patch unzip wget zlib-devel curl curl-devel openssl-devel perl perl-devel cpio expat-devel gettext-devel xz openssh-clients cmake libtool</code></pre>



下载SDK包到\~目录，我的路由是常见的MT7620n方案，所以下载mt7620的：



<pre><code class="language-bash">cd ~
wget https://downloads.openwrt.org/chaos_calmer/15.05.1/ramips/mt7620/OpenWrt-SDK-15.05.1-ramips-mt7620_gcc-4.8-linaro_uClibc-0.9.33.2.Linux-x86_64.tar.bz2
tar jxvf OpenWrt-SDK-15.05.1-ramips-mt7620_gcc-4.8-linaro_uClibc-0.9.33.2.Linux-x86_64.tar.bz2
mv OpenWrt-SDK-15.05.1-ramips-mt7620_gcc-4.8-linaro_uClibc-0.9.33.2.Linux-x86_64 op
</code></pre>



下载libpcap，这里直接把依赖库libpcap也编译进去，免得要自己安装。



<pre><code class="language-bash">wget http://www.tcpdump.org/release/libpcap-1.7.4.tar.gz
tar zxvf libpcap-1.7.4.tar.gz
</code></pre>



下载mentohust\-proxy。



<pre><code class="language-bash">git clone https://github.com/updateing/mentohust-proxy.git
</code></pre>



配置环境变量（这里的/home/job是我的\~目录）



<pre><code class="language-bash">export PATH=$PATH:/home/job/op/staging_dir/toolchain-mipsel_24kec+dsp_gcc-4.8-linaro_uClibc-0.9.33.2/bin:/home/job/op/staging_dir/host/bin
export STAGING_DIR=/home/job/op/staging_dir
</code></pre>



剩下的过程跟普通编译差不多，就是多了host=mipsel\-openwrt\-linux\-uclibc参数。首先编译libpcap



<pre><code class="language-bash">cd libpcap-1.7.4
./configure --host=mipsel-openwrt-linux-uclibc --with-pcap=linux
make
</code></pre>



然后编译mentohust\-proxy



<pre><code class="language-bash">cd ../mentohust-proxy
sh autogen.sh
./configure --host=mipsel-openwrt-linux-uclibc --disable-encodepass --disable-notify --with-pcap=/home/job/libpcap-1.7.4/libpcap.a
make</code></pre>



在src目录下的mentohust就是最终编译好的文件了。

### 完成认证

把mentohust下载下来，上传到路由器，大家可以先试试直接mentohust可不可以完成认证，如果不行，就使用代理认证。顺便说，这个升级版的mentohust更加人性化了，下面是它的帮助文件。

> \>\>\> mentohust \-h
> 欢迎使用MentoHUST      	版本: 0\.3\.1
> Copyright (C) 2009\-2010 HustMoon Studio
> 人到华中大，有甜亦有辣。明德厚学地，求是创新家。
> 802\.1x proxy and other new features by Hamster Tian 2015\-2016\.
> Bug report to http://code\.google\.com/p/mentohust/issues/list
> 
> 
> 
> 用法:  	\./mentohust &#91;\-选项&#93;&#91;参数&#93; 或 &#91;\-选项&#93; &#91;参数&#93; 或 &#91;\-\-长选项&#93; &#91;参数&#93;
> 选项:  	\-\-help 	\-h 显示本帮助信息
> 	\-\-kill 	\-k \-k(退出程序) 其他(重启程序)
> 	\-\-write	\-w 保存参数到配置文件
> 	\-\-username     	\-u 用户名
> 	\-\-password     	\-p 密码
> 	\-\-nic  	\-n 网卡名
> 	\-\-ip   	\-i IP&#91;默认本机IP&#93;
> 	\-\-mask 	\-m 子网掩码&#91;默认本机掩码&#93;
> 	\-\-gateway      	\-g 网关&#91;默认0\.0\.0\.0&#93;
> 	\-\-dns  	\-s DNS&#91;默认0\.0\.0\.0&#93;
> 	\-\-ping\-host    	\-o Ping主机&#91;默认0\.0\.0\.0，表示关闭该功能&#93;
> 	\-\-auth\-timeout 	\-t 认证超时(秒)&#91;默认8&#93;
> 	\-\-heartbeat    	\-e 心跳间隔(秒)&#91;默认30&#93;
> 	\-\-wait\-after\-fail      	\-r 失败等待(秒)&#91;默认15&#93;
> 	\-\-max\-fail     	\-l 允许失败次数&#91;0表示无限制，默认8&#93;
> 	\-\-no\-auto\-reauth       	\-x 掉线后是否自动重连: 0(不重连) 1(重连) &#91;默认1&#93;
> 	\-\-eap\-bcast\-addr       	\-a 组播地址: 0(标准) 1(锐捷) 2(赛尔) &#91;默认0&#93;
> 	\-\-dhcp\-type    	\-d DHCP方式: 0(不使用) 1(二次认证) 2(认证后) 3(认证前) &#91;默认0&#93;
> 	\-\-daemonize    	\-b 是否后台运行: 0(否) 1(是，关闭输出) 2(是，保留输出) 3(是，输出到文件) &#91;默认0&#93;
> 	\-\-fake\-supplicant\-version      	\-v 客户端版本号&#91;默认0\.00表示兼容xrgsu&#93;
> 	\-\-template\-file	\-f 自定义数据文件&#91;默认不使用&#93;
> 	\-\-dhcp\-script  	\-c DHCP脚本&#91;默认dhclient&#93;
> 	\-\-proxy\-lan\-iface      	\-z 认证代理模式下监听认证数据包的网卡名（设置此项即表示开启代理模式）
> 	\-\-proxy\-require\-success	\-j 认证代理模式下关闭LAN监听线程前需要收到的Success包次数 &#91;默认1&#93;
> 	\-\-decode\-config	\-q 显示SuConfig\.dat的内容(如\-q/path/SuConfig\.dat)
> 	\-\-max\-retries 在得到认证成功或失败的结果前，最多重试的次数，0表示无限重试 &#91;默认0&#93;
> 例如:  	\./mentohust \-u username \-p password \-n eth0 \-i 192\.168\.0\.1 \-m 255\.255\.255\.0 \-g 0\.0\.0\.0 \-s 0\.0\.0\.0 \-o 0\.0\.0\.0 \-t 8 \-e 30 \-r 15 \-a 0 \-d 1 \-b 0 \-v 4\.10 \-f default\.mpf \-c dhclient
> 关于代理模式：此模式下MentoHUST将不会自己发起认证，而是修改LAN内捕获到的认证数据包的源MAC并转发至WAN，使得本机认证通过。
> 代理模式下用户名、密码、掉线重连方式（\-u、\-p、\-x）无效，可不指定。由于Start包为自行构造，仍然需要指定DHCP方式和组播地址（\-d、\-a）。
> 代理模式示例:  	\./mentohust \-\-proxy\-lan\-iface br0 \-\-nic eth0 \-\-eap\-bcast\-addr 1 \-\-dhcp\-type 2 \-proxy\-require\-success 2
> 注意：使用时请确保是以root权限运行！

首先把校园网线接到路由器的wan口，然后电脑接到lan口（用wifi连接也可以），路由器上启动命令



<pre><code class="language-bash">./mentohust --proxy-lan-iface br-lan --nic eth0.2 --eap-bcast-addr 0 --dhcp-type 2</code></pre>



然后在电脑用锐捷进行拨号就行了，可以发现路由器上面显示拨号成功。维持mentohust的运行，可以用screen，也可以直接用\-b参数。好吧，可以享受WiFi了。

### 接下来

接下来可以做的事情就多了，比如内网穿透、SSH代理、资源共享等～

编译好的程序，MTK系列路由器可用：[mentohust\_mtk\.zip](<https://spaces.ac.cn/usr/uploads/2016/11/2403108993.zip>)

### 参考链接

[https://github\.com/updateing/mentohust\-proxy](<https://github.com/updateing/mentohust-proxy>)

[https://www\.yjblog\.net/post/123\.html](<https://www.yjblog.net/post/123.html>)

[http://soundrain\.net/2016/04/25/mentohust\-v4版本编译及ipv6的配置/](<http://soundrain.net/2016/04/25/mentohust-v4%E7%89%88%E6%9C%AC%E7%BC%96%E8%AF%91%E5%8F%8Aipv6%E7%9A%84%E9%85%8D%E7%BD%AE/>)
