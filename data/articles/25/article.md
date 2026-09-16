# 日全食多路联合直播频道

> 作者：苏剑林 · 科学空间 · 2009-07-18
>
> 原文：<https://spaces.ac.cn/archives/25>
>
> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）](https://creativecommons.org/licenses/by-nc-nd/2.5/cn/)。
>
> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。
>
> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。
>
> 原站转载与引用说明：[科学空间 FAQ](https://spaces.ac.cn/archives/6508#%E6%96%87%E7%AB%A0%E5%A6%82%E4%BD%95%E8%BD%AC%E8%BD%BD/%E5%BC%95%E7%94%A8)

---

<div data-spaces-format="html-v1">
<p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color: rgb(0, 0, 255);&quot;}">正式直播活动计划于北京时间7月22日7时30分开始，11时30分结束，持续约4个小时。</span></strong></p><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;font-size: 16px;&quot;}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color: rgb(255, 0, 0);&quot;}">（观看请安装PPlive插件，只能用IE或者IE内核浏览器观看）</span></span></strong></p><div data-spaces-tag="div" data-spaces-attrs="{&quot;id&quot;: &quot;pplivemain&quot;}"><div data-spaces-tag="div" data-spaces-attrs="{&quot;id&quot;: &quot;ppliveplayer&quot;}"> </div></div><p data-spaces-tag="p" data-spaces-attrs="{}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">简介：</strong></p><p data-spaces-tag="p" data-spaces-attrs="{}"><details data-spaces-opaque="1"><summary>原文嵌入内容（script；预览不执行）</summary><pre><code>&lt;script src="/sci/PPLiveActiveX.js" type="text/javascript"&gt;&lt;/script&gt;</code></pre></details> <details data-spaces-opaque="1"><summary>原文嵌入内容（script；预览不执行）</summary><pre><code>&lt;script type="text/javascript"&gt;

  &lt;!--
	  (function() {
	  // 播放器Id
	  var playerId = 'ewaplayer';

	  // 播放链接
	  var playLink = 'synacast://scSpmqaWopbP0dfT3tXYpvHHqKPNoa/I1Z2hna/GnaTNzKeS0qShoKOeoKadoq2aodSen6viltPVzbOYpKGcmpzT0d3RpqiVoKklYzIvPFw6LTpPOEU0FEAYRiEeDrCXoKWepKiVoKkcUzE/FEEQThzjFFkRZyMeGi0oJrCWoKKkpMnI2dXa3d/L2dPLzu7V3N/eyurO396mmqaYo5bY2N2ioZ6dl6eToaqko66L29+p3trV5KqbmKiXoZ6emaqToqSdl66Wqqicmaag5dTc3bCUn6KdoaSboZ6il6eapKqkmaaVq+XQ2eqfn5+imqSWpaWaoaSXo6qkmaaVq+TP2eqfn5+em6eToqCgl6iZoZ6kobCdoKicpOrI4OSmmKWboZ6dnquTqJ6eobCdoKicpN7Z5ODgo6WUoqKdl6iVpJ6enaeTqKimoaY=';

	  /*****************************************
	  * 播放器准备好事件控制
	  * 参数：
	  *   player: object，PPLive OCX播放器实例
	  *****************************************/
	  function readyHandler(player) {
	  player.Url = playLink;
	  }

	  /*****************************************
	  * 安装进度事件控制
	  * 参数：
	  *   status:  int，安装状态
	  *            1：正在下载
	  *            2：正在安装
	  *            3：安装完成，重启OCX
	  *               注：此处不需要作特别处理，
	  *               但安装完成后并不意味着可以
	  *               立即调用播放器的相关方法，
	  *               需要等到播放器准备好（即需
	  *               要等到触发onReady 事件（播
	  *               放器准备好事件））
	  *   percent: int，下载百分比
	  *            安装完成后这个值会变成0
	  *****************************************/
	  function installingHandler(status, percent) {
	  document.getElementById('installing').innerHTML = '状态：' + status + '  进度：' + percent + '%';
	  }

	  var logoUrl = 'http://res.pplive.com/ikan/0718/player/ikan_logo.jpg';
	  var player_video = {
	  // 播放器属性，默认width和height均为100%，如果使用默认，可以不写
	  // 默认播放器id为：PPLivePlayerActiveX
	  properties: { 'id': playerId, 'width': '100%', 'height': '100%', 'codebase': 'http://dl.pplive.com/PluginSetup.cab' },
	  // 播放器参数，如果使用默认，可以不写
	  params: {
	  'logourl': logoUrl, 				// logo 地址
	  'dbclicktofullscreen': 'true', 	// 双击全屏
	  'showcontextmenu': 'true', 		// 显示上下文菜单（右键菜单）
	  'showstateinfo': 'true', 		// 显示状态信息
	  'showchannelname': 'true', 		// 显示频道名称
	  'showplayerbuffer': 'true', 		// 显示播放器缓冲百分比
	  'showdownloadbuffer': 'true', 	// 显示下载缓冲百分比
	  'showdownloadrate': 'true', 		// 显示下载比特率
	  'showplaycontroller': 'true', 	// 显示播放控制栏
	  'showplayprogress': 'true', 		// 显示播放进度条
	  'showloadingad': 'true', 		// 显示缓冲广告
	  'showadcountdown': 'true', 		// 显示广告倒计时
	  'adcfgurl': 'http://pp1.pplive.com/pp/longshen090211zhu15s.swf', 					// 广告配置地址
	  'enableup&amp;#100;ate': 'true', 			// 允许升级
	  'enableup&amp;#100;atetip': 'true', 		// 启用升级提示
	  'up&amp;#100;ateurl': ''						// 升级地址
	  },
	  // 当播放器ready时触发，默认为空方法，
	  // 建议写上，并在该方法中获取播放器控件，由此可以判断播放器是否可用
	  ready: readyHandler,
	  // 安装进度事件控制
	  installing: installingHandler,
	  // 循环检测间隔，默认间隔3000毫秒，可以不写
	  checkInterval: 3000
	  };

	  // 把播放器写入页面
	  (new PPLiveActiveX(player_video)).write(document.getElementById('ppliveplayer'));
	  })();
	  --&gt;
  &lt;/script&gt;</code></pre></details> <span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color: rgb(255, 153, 0);&quot;}">2009年7月22日我国长江流域将出现罕见的日全食天象，持续时间超过6分钟，堪称21世纪中国人能看到的最壮观的天象之一。届时，中国科学院国家天文台、微软亚洲研究院、中国科学院计算机网络信息中心、中国科学院上海天文台等四家单位将联合进行面向全球公众的日全食直播活动。计划在日食带内设置多个直播点，通过CNGI下一代互联网试验系统将直播点的日食数字高清视频信号传输至北京直播中心，并通过广播通讯卫星、互联网等方式免费向全球发布。届时电视台、网络门户、手机节目服务商、移动电视等各种媒体均可接收该信号源，并可制作成直播节目奉献给全球公众。相信此次日全食直播将成为 2009年最受公众瞩目的天文盛</span><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color: rgb(255, 153, 0);&quot;}">宴。</span></p><p data-spaces-tag="p" data-spaces-attrs="{}"><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color: rgb(255, 153, 0);&quot;}">搜狐、腾讯、网易等国内一流的网络门户将同步播出此次直播的画面。波兰、新加坡、马来西亚、加拿大、英国、乌克兰等国的天文网站也将同步播出直播信号。黑龙江电视台、江苏电视台、北京电视台也正在联系中。</span><span data-spaces-tag="span" data-spaces-attrs="{&quot;style&quot;: &quot;color: rgb(255, 0, 0);&quot;}"><strong data-spaces-tag="strong" data-spaces-attrs="{}">据说，中央电视台也会进行直播。</strong></span></p>
</div>
