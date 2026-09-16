# Openwrt自动扫描WiFi并连接中继

> 作者：苏剑林 · 科学空间 · 2016-03-06
>
> 原文：<https://spaces.ac.cn/archives/3644>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

最近入手了一个非常迷你的路由器——由25 x 25mm的vocore开发板搭建成的超小路由器，配上外壳后，也仅仅是37\.4 x 34 x 25\.9mm，比一个随身WiFi稍大。（[链接](<https://item.taobao.com/item.htm?id=45865192290>)）

[![vocore路由器](<https://spaces.ac.cn/usr/uploads/2016/03/3841953.jpg>)](<https://spaces.ac.cn/usr/uploads/2016/03/3841953.jpg>)

vocore路由器

虽然很小，但是配置却不低

> CPU 处理器：雷凌 RT5350 360MHz MIPS 24KEc
> Memory 内存：32MB 133MHz SDRAM
> Flash 闪存：16MB
> 扩展接口：SPI I2C I2S
> WiFi 无线：802\.11n
> Ethernet 网络：10/100MHz x 2
> GPIO 扩展：28（Reused）
> UART 界面：UART Lite / UART Full
> USB接口：USB 2\.0，up to 480M
> Power Supply 电源：3\.3V \~ 6V

简单来说，就是允许我们装Openwrt，这样一来，可玩性就强了。而且由于它那么小，方便随身携来，用来做随身路由，足够完美呀，说干就干。

<strong>大家都知道，手机或者手提，都会有这样的功能，如果我们在若干个地方连接过若干个WiFi，就会记住相应的信息，如果我们再次到达那个地方，就会自动连接上相应的WiFi，不用人工干预了（除了需要认证的外）。那么Openwrt能不能实现这一点呢？换言之，Openwrt能不能自动扫描WiFi，然后自动连上WiFi，并发射相应的热点给手机或者手提用呢？</strong>

有人就会疑问了，你路由器都能够扫描到WiFi了，那么手机手提肯定可以呀，这样手机手提直接连WiFi不就好了吗，干嘛还要中转一次？正常的WiFi当然是这样，但是像我们校园的WiFi，或者CMCC的WiFi，都是需要认证的，而且一号一机，这样我们一般的解决方法就是用一台电脑连上WiFi，然后用类似随身WiFi的东西发射出来。虽然简单，但是毕竟太普通，而且我也不是随时带着电脑，因此，这个方法自然不为我所取。

说了那么多话，要干活了。我的迷你路由刷了Openwrt 15\.05，首先给出一个完整的中继一个普通WiFi的过程（普通WiFi指的是输入密码就能正常上网的WiFi），直接ssh到路由器，然后输入



<pre><code class="language-bash">#启动WiFi
uci set wireless.@wifi-device[0].disabled=0
uci commit wireless

#创建中继接口
uci set network.wwan=interface
uci set network.wwan.proto=dhcp
uci commit network

#创建热点WiFi
uci set wireless.@wifi-iface[0].device=radio0
uci set wireless.@wifi-iface[0].network=lan
uci set wireless.@wifi-iface[0].ssid=VoCore
uci set wireless.@wifi-iface[0].mode=ap
uci set wireless.@wifi-iface[0].encryption=none
uci commit wireless

#中继热点BoJone
uci add /etc/config/wireless wifi-iface
uci set wireless.@wifi-iface[1].device=radio0
uci set wireless.@wifi-iface[1].network=wwan
uci set wireless.@wifi-iface[1].ssid=BoJone
uci set wireless.@wifi-iface[1].mode=sta
uci set wireless.@wifi-iface[1].encryption=psk2
uci set wireless.@wifi-iface[1].key=12345678
uci commit wireless

#设置防火墙
uci set firewall.@zone[1].forward=ACCEPT
uci set firewall.@zone[1].network='lan wan wwan'
uci commit firewall

#重启网络
/etc/init.d/network restart</code></pre>



这些步骤都是在第一次运行的时候配置的，配置好了之后，如果要改变中继的WiFi，那么只需要更改中继部分，其他都不需要改变。接着，一步一步完成我们的目标，首先是扫描WiFi列表，在Openwrt用的是



<pre><code class="language-bash">iw wlan0 scan</code></pre>



这样会得到一个非常详细的附近可以连接的WiFi列表，包括SSID、信号强度等，为了中继的稳定，我们还需要做中继信号强度的判断。这个过程，基本的工具就是正则。

然后要实现的是自动完成认证，我这里只是自动完成华师校园WiFi的认证，分析它的过程，发现它只是post了账号和密码过去，因此很简单。但是，用什么工具去post呢？很自然我想到了python的requests，但是要装python还要装requests，略显麻烦。后来看到其实wget也支持post数据的（神奇！），直接用wget完成就行了。但是自带的wget并不支持https，需要直接用



<pre><code class="language-bash">opkg update
opkg install wget</code></pre>



安装完整版本的的wget。这样，就可以写出一个shell脚本来：



<pre><code class="language-bash">#!/bin/ash

POST_SCNUNET()
{
    while true
    do
        web=`wget http://kexue.fm -q -O-`
        net_test=`echo $web|grep Scientific`
        if [ ${#net_test} -lt 2 ] ;
        then
                url=`echo $web|awk -F "href=\'" '{print $2}'|awk -F "\'" '{print $1}'`
                url=${url/index.jsp?/userV2.do?method=login&amp;param=ture&amp;}
                wget $url --post-data 'username=XXXXXX&amp;pwd=XXXXXX' --no-check-certificate -q -O-
                sleep 5
        else
                break
        fi
    done
}

uci delete wireless.@wifi-iface[1]
wifis=`iw wlan0 scan`
dbm=`echo $wifis|awk -F 'BoJone' '{print $1}'|awk -F '-' '{print $NF}'|awk -F '.' '{print $1}'`
if [ $dbm -lt 100 ] ;
then
    uci add /etc/config/wireless wifi-iface
    uci set wireless.@wifi-iface[1].device=radio0
    uci set wireless.@wifi-iface[1].network=wwan
    uci set wireless.@wifi-iface[1].ssid=BoJone
    uci set wireless.@wifi-iface[1].mode=sta
    uci set wireless.@wifi-iface[1].encryption=psk2
    uci set wireless.@wifi-iface[1].key=12345678
    uci commit wireless
    /etc/init.d/network restart
    echo 'Connected BoJone' &gt; log.txt
else
    dbm=`echo $wifis|awk -F 'SCNUNET' '{print $1}'|awk -F '-' '{print $NF}'|awk -F '.' '{print $1}'`
    if [ $dbm -lt 100 ] ;
    then
        uci add /etc/config/wireless wifi-iface
        uci set wireless.@wifi-iface[1].device=radio0
        uci set wireless.@wifi-iface[1].network=wwan
        uci set wireless.@wifi-iface[1].ssid=SCNUNET
        uci set wireless.@wifi-iface[1].mode=sta
        uci set wireless.@wifi-iface[1].encryption=none
        uci commit wireless
        /etc/init.d/network restart
        POST_SCNUNET
        echo 'Connected SCNUNET' &gt; log.txt
    else
        uci commit wireless
        /etc/init.d/network restart
        echo 'No Wifi Connected' &gt; log.txt
    fi
fi</code></pre>



这个脚本实现的过程是：<strong>扫描WiFi列表，看看BoJone在不在列表中，信号强度如何，如果足够好，那么就自动连上它；如果不行，就看看SCNUNET在不在其中，信号强度如何，如果足够好，那么就自动连上它，并完成认证，认证过程已经提前分析好，写在POST\_SCNUNET函数中（这个函数首先用wget下载本网站，以判断网络是否正常，如果正常就不用认证了，如果不正常，那么会自动跳转到认证页面的。）；如果都没有，那么就不中继了，只好等待有线接入了。</strong>

其他校园网WiFi或者认证WiFi估计类似，这里仅提供一个参考例子。

最后，就是感觉，实现一些简单的功能，用纯shell脚本也是很爽快呀，尤其在路由器这些低端环境，自然是能用shell就用shell了。有了python的编程基础，这样实现并不是很困难。
