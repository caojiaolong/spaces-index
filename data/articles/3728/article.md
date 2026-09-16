# 【备忘】用树莓派3做无线路由器

> 作者：苏剑林 · 科学空间 · 2016-04-12
>
> 原文：<https://spaces.ac.cn/archives/3728>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

3月初发布的树莓派3自带了WiFi和蓝牙，再加上它本来就有一个网口，因此俨然就是一台无线路由器了。我也忍不住入手了一个，打算用来做路由器和NAS。树莓派做路由器的教程已经有很多了，当然，基本都是基于树莓派2的，3之前的版本都没有自带WiFi，因此需要自己配无线网卡，而3自带了无线网卡，配置就方便多了。参考了两篇外文教程，成功配置，在这里记录一下。

参考教程：
[https://frillip\.com/using\-your\-raspberry\-pi\-3\-as\-a\-wifi\-access\-point\-with\-hostapd/](<https://frillip.com/using-your-raspberry-pi-3-as-a-wifi-access-point-with-hostapd/>)

[https://gist\.github\.com/Lewiscowles1986/fecd4de0b45b2029c390\#file\-rpi3\-ap\-setup\-sh](<https://gist.github.com/Lewiscowles1986/fecd4de0b45b2029c390#file-rpi3-ap-setup-sh>)

### 配置无线热点

主要用到的软件有hostapd、dnsmasq：



<pre><code class="language-bash">sudo apt-get install hostapd dnsmasq</code></pre>



然后在/etc/dnsmasq\.conf末加入（自己修改IP和网段，这个文件是已存在的，很详细的配置文件，但是所有行都加入了\#号注释掉）

> interface=wlan0
> dhcp\-range=10\.0\.0\.2,10\.0\.0\.5,255\.255\.255\.0,12h

然后新建/etc/hostapd/hostapd\.conf，加入

> interface=wlan0
> hw\_mode=g
> channel=10
> auth\_algs=1
> wpa=2
> wpa\_key\_mgmt=WPA\-PSK
> wpa\_pairwise=CCMP
> rsn\_pairwise=CCMP
> wpa\_passphrase=wifi密码
> ssid=wifi名字

接着修改/etc/sysctl\.conf，更改（如果有这一行，把\#号去掉就行）

> net\.ipv4\.ip\_forward=1

最后，将下面脚本加入到/etc/rc\.local的exit 0前：



<pre><code class="language-bash">ifconfig wlan0 down
ifconfig wlan0 10.0.0.1 netmask 255.255.255.0 up
iwconfig wlan0 power off
service dnsmasq restart
hostapd -B /etc/hostapd/hostapd.conf &amp; &gt; /dev/null 2&gt;&amp;1
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE  
sudo iptables -A FORWARD -i eth0 -o wlan0 -m state --state RELATED,ESTABLISHED -j ACCEPT  
sudo iptables -A FORWARD -i wlan0 -o eth0 -j ACCEPT</code></pre>



重启，就可以看到热点了，简单好多～树莓派的wifi信号强度跟当初著名的路由玩具wr703n差不多。

### 跳坑与填坑

顺便还配置了离线下载、NAS、自动云同步什么的。由于对linux不熟悉，跳了好多坑。要提醒大家，树莓派上很多命令都需要sudo开头，而树莓派的sudo并不需要密码。但是有sudo和没sudo完全是两个环境（两个用户），比如sudo screen \-S sync后，在screen \-ls是看不到的，必须要sudo screen \-ls才能看到。另外，如果你把命令加入/etc/rc\.local中运行，默认它是sudo执行的（不管你有没有加sudo），结果我将一个screen任务加入到了这里，启动后用screen \-ls死活看不到，原来要sudo screen \-ls，我晕\.\.\.还有，我运行autossh进行内网穿透，autossh之前一定要加个sleep 5的命令，要不然autossh运行了也没用～～

这些都是跳了一整天的坑啊。
