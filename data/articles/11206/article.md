# 基于树莓派Zero2W搭建一个随身旁路由

> 作者：苏剑林 · 科学空间 · 2025-08-02
>
> 原文：<https://spaces.ac.cn/archives/11206>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

前段时间搞了个很迷你的开发板树莓派Zero2W（下面简称“<strong>Pi</strong>”），还搭配了个USB Key转接板，这几天折腾了一下，用于实现一个随身的旁路由。本文记录一下关键技术点，供有同样需求的读者参考。

[![树莓派Zero2W](<https://spaces.ac.cn/usr/uploads/2025/08/1443796894.jpeg>)](<https://spaces.ac.cn/usr/uploads/2025/08/1443796894.jpeg>)

树莓派Zero2W

## 背景描述

折腾过代理的同学，对“旁路由”这个概念应该不陌生，这里主要将旁路由用作透明代理。现在相关技术已经很成熟，所以有一台小主机的情况下，搭建一个透明代理并不算困难。而本文的主要关注点在于“随身”，这就要求小主机满足以下几个特性：

> 1、小巧（不然不适合随身带）；
> 
> 
> 
> 2、有连接Wi\-Fi功能；
> 
> 
> 
> 3、随时可以切换网络。

这里比较麻烦的是第3点。假设我带着笔记本和小主机来到了一个新环境，知道了当地的Wi\-Fi账号密码，那我怎么才能让小主机连上新Wi\-Fi呢？这里的关键是小主机还没连网时，笔记本怎么连上小主机，如果有网口的情况下，通过网线连接自然可以，但带网口的主机通常不小，而且随身带网线也累赘（Macbook还需要配转接头）。

树莓派Zero2W \+ USB Key正好可以完美实现这个需求！它能够在作为主机运行的同时还模拟一张网卡，通过USB口连接到电脑时，既可以给Zero2W供电，又可以构建一条Zero2W到电脑的直连通道，使得我们可以不经过Wi\-Fi就连接到Zero2W上去。

## 烧录镜像

事实上，这种USB直连Pi Zero的方案由来已久，比如教程[《树莓派 Zero USB/以太网方式连接配置教程（macOS平台）》](<https://shumeipai.nxez.com/2018/02/20/raspberry-pi-zero-usb-ethernet-gadget-tutorial-macos.html>)已经是2018年的了，但里边的配置方式已经过时，这里主要更新一下在[K2](<https://www.kimi.com/>)的帮助下能跑通的最新方案。

首先，准备SD卡，烧录最新的Raspberry Pi OS镜像到SD卡上。现在官方出了Raspberry Pi Imager，烧录镜像变得更简单了，这里烧录镜像的时候要注意几个细节，首先是必须把Wi\-Fi账号密码配置好（后面可以改，第一次配置需要用Wi\-Fi），然后必须开启SSH，其余的按需调整：

[![填好Wi-Fi信息](<https://spaces.ac.cn/usr/uploads/2025/08/2336862975.png>)](<https://spaces.ac.cn/usr/uploads/2025/08/2336862975.png>)

填好Wi\-Fi信息

[![开启SSH](<https://spaces.ac.cn/usr/uploads/2025/08/473887550.png>)](<https://spaces.ac.cn/usr/uploads/2025/08/473887550.png>)

开启SSH

建议把主机名也设置好并记住，这样后面在内网中我们就可以直接通过它来访问Pi，而不需要记IP地址。设置完成后就可以烧录镜像了，烧录结束后，把SD卡继续用读卡器插到电脑上（我的电脑是Macbook Pro，系统是macOS 15\.3\.1，下面都简称“<strong>Mac</strong>”），在SD卡的根目录找到<code>config.txt</code>文件，在末尾加一句（当前末尾是<code>[all]</code>）：



<pre><code class="language-shell">dtoverlay=dwc2</code></pre>



至此，烧录镜像步骤完成。

## 网卡配置

现在我们可以将SD卡插入到Pi上，然后通过USB口将Pi插入到Mac，那么Pi就会通电启动，等到启动完成后，由于刚才已经配置了Wi\-Fi，所以Pi应该已经顺利连上了网络，这时候把Mac也连接到同一个网络，我们就可以通过<code>ssh me@pi.local</code>来SSH上去Pi，当然，也可以在路由器处查找它的IP，通过IP来SSH。

接着我们在Pi端执行



<pre><code class="language-shell">sudo modprobe g_ether
sudo ip link set usb0 up
sudo ip addr add 169.254.7.11/16 dev usb0</code></pre>



执行完后通过<code>ifconfig</code>观察，可以发现多出了个名为<code>usb0</code>的网卡，并且ip设置为了<code>169.254.7.11</code>，这属于内网保留 IP，可以自行设置，没有什么特殊考虑也可以直接复制。

稍等一会后，我们转到Mac端“设置”\-“网络”，正常情况下会多出一个<code>RNDIS/Ethernet Gadget</code>服务，并已经自动连接上，点击“详细信息”可以看到通过DHCP方式分配了IP，我们将它改为手动模式，IP为<code>169.254.7.1</code>，路由留空。这个IP原则上也可以自定义，但没什么特殊考虑也是直接复制就好。

再次转到Pi端，执行<code>ping 169.254.7.1</code>，能ping通，说明Pi和Mac已经通过USB口组了一个小型局域网，Mac的IP是<code>169.254.7.1</code>，Pi的IP是<code>169.254.7.11</code>，在Mac端通过<code>ssh me@169.254.7.11</code>也可以连接到Pi。

## 启动服务

不过现在的网卡配置都是一次性的，重启就没了，我们需要将它配置成开机自动启动的服务，这样以后才能在没有网络的情况下，还能通过USB口以及<code>ssh me@169.254.7.11</code>连接到Pi，这就是我们的最终目的。

在Pi端，新建<code>/usr/local/bin/usb0.sh</code>：



<pre><code class="language-shell">#!/bin/bash
sudo modprobe g_ether
sudo ip link set usb0 up
sudo ip addr add 169.254.7.11/16 dev usb0
nohup ping -c 100 169.254.7.1 &gt; /var/log/usb0_ping.log 2&gt;&amp;1 &amp;
exit 0</code></pre>



然后<code>sudo chmod +x /usr/local/bin/usb0.sh</code>，再新建<code>/etc/systemd/system/usb0.service</code>，写入



<pre><code class="language-shell">[Service]
Type=oneshot
ExecStart=/usr/local/bin/usb0.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target</code></pre>



最后执行



<pre><code class="language-shell">sudo systemctl daemon-reload
sudo systemctl enable usb0.service</code></pre>



就可以了。以后将Pi插入Mac，等开机完成后，就可以通过\`ssh me@169\.254\.7\.11\`连接上去，如果有需要通过<code>sudo raspi-config</code>可以连接/切换Wi\-Fi。

## 文章小结

至于剩下的搭建旁路由/透明代理的步骤，就请大家自行折腾了。本文主要是帮助大家把“随身”这一步跑起来，剩下的看自己发挥了。
