# 本地预览与阅读

[文档导航](../README.md) · [日常维护](maintenance.md) · [正文校验与许可](mirror.md)

## 启动预览

```bash
uv sync --locked
# 增量更新后预览
uv run python scripts/update_all.py --serve
# 已有正文，完全离线重建和预览
uv run python scripts/update_all.py --offline --serve
# 只启动已有构建
uv run python scripts/serve.py
```

打开 <http://127.0.0.1:8765/>。产物在 `build/preview/`，正文在 `data/articles/`；服务器仅绑定本机，预览命令不执行部署。

## 阅读与导航

首页、文章探索、主题及系列页的文章标题默认打开站内 Markdown 阅读页，未提供正文的文章仍指向原文。卡片下方与阅读页顶部集中提供“查看原文、下载 Markdown、复制 Markdown”。阅读器提供源码切换、章节目录、进度、字号、深色主题、系列上下篇和站内文章链接。复制和下载保留完整署名、原文链接及许可，不受阅读排版影响。

文章旁的圆形箭头与标题进入同一篇正文，支持键盘操作；“继续阅读”和系列的“继续下一篇”直接打开未读文章。同一系列内切换章节只更新定位与高亮，保留目录状态。首页线框曲面会缓慢旋转、起伏与变形；滚出视口或页面进入后台时自动暂停，并跟随系统减少动态效果设置，不显示暂停按钮。

阅读页使用主站的 `#/article/<id>` 路由，共用导航、页脚、主题、返回顶部和阅读进度。从列表进入后返回，会恢复筛选条件和滚动位置。进度保存在当前浏览器，达到 100% 才计入已读；复制、下载或打开原文不会自动标记已读。旧 `reader.html?id=…` 链接自动跳转到主站文章页。

## 公式选择、复制与预览

完整选中公式主体后按 Ctrl+C（macOS 为 ⌘C），会复制保留定界符的原始 LaTeX，不必选中右侧编号或上下文。仅拖选公式内部的一部分符号时，复制的是可见字符，不保留分式或上下标的二维结构。混合选择文字和公式时会还原公式源码；需要完整原文时使用“复制 Markdown”。会接管复制事件的浏览器扩展可能覆盖该行为，遇到冲突可在本站停用扩展。

悬停公式编号引用或用键盘聚焦，会预览对应公式；点击引用平滑滚动到公式并短暂高亮，右下方出现“返回引用处”，可恢复跳转前的位置及键盘焦点。连续跳转支持逐次返回，切换文章后记录清空。跳转与返回尊重系统减少动态效果设置，触屏直接点击跳转。公式本身不提供点击弹窗。原文中未配对的公式标记继续原样保留。

公式使用本地 MathJax 4.1.3 的 CommonHTML 输出与 TeX 数学字体，保留正常行内基线和文字间距，长公式可横向滚动。组件和字体随网站构建一起提供，排版无需访问原站或外部 CDN，也不修改存储的 LaTeX。

## 图片与图注

正文图片按比例显示，高度不超过当前窗口的一半。点击图片或聚焦后按 Enter 可放大查看，用加减按钮或滚轮缩放，放大后按住拖动；支持适应窗口、原始尺寸和打开原图。按 Esc、关闭按钮或点击遮罩退出，回到原阅读位置。图注根据原文标记显示。

网站优先展示通过哈希校验的本地图片缓存，未缓存和站外图片使用原始链接；Markdown 始终保留原图 URL。“打开原图”也指向源站。缓存与跳过失效子域的规则见 [图片缓存](maintenance.md#图片缓存)。

## 向 AI 提问

默认使用 GPT。点击文章顶部提问按钮旁的下拉箭头，在“选择提问助手”图标菜单中选择 GPT、Claude、Gemini、DeepSeek、Kimi、豆包或 Perplexity。设置保存在当前浏览器，刷新或切换文章后继续使用；全文和选段入口同步显示所选 AI 的名称与图标。菜单支持方向键切换、Enter 确认和 Esc 关闭，适配浅色、深色主题与手机屏幕。切换设置本身不会联系任何 AI 服务。图标随项目保存，[来源与许可证](../../web/vendor/ai-icons/README.md)单独记录。

| 服务 | 传递方式 |
| --- | --- |
| GPT（默认） | 通过兼容的提问链接带入内容，提供复制备用 |
| Perplexity | 使用[官方 OpenSearch 地址](https://www.perplexity.ai/opensearch.xml)带入提问，提供复制备用 |
| Claude、Gemini、DeepSeek、Kimi、豆包 | 显示完整提问面板并尝试复制，然后点击“打开对应 AI”粘贴 |

阅读页顶部的全文提问附文章标题、苏剑林署名、原文链接和 Markdown 文件直链。它请求 AI 使用可用的联网工具获取完整原文，按 UTF-8 读取并保留原始 LaTeX，再等待你的问题。如果链接访问失败或工具拒绝 `text/markdown`，提示词提供 GitHub 原始文件备用地址：`https://raw.githubusercontent.com/caojiaolong/spaces-index/main/data/articles/<id>/article.md`。备用地址直接使用仓库中已提交的正文文件，以 `text/plain` 返回，不额外生成另一份正文。

提示词不能赋予 AI 文件下载或联网能力，也不表示全文已经送达。若两个链接都无法读取，展开按钮旁的菜单选择“复制全文提问”，或下载 Markdown 后上传。全文复制保留完整原文；AI 应明确说明未核对全文，不能用网页摘要或搜索片段代替。

在正文中选择文字后，浮动提问按钮会准备选段及全文链接，默认请所选 AI 解释概念、公式与推导，并在选段后附“请解释这一段文字。”完整公式传递原始 LaTeX；仅选中公式的一部分时保留选中字符，并附完整原式作为上下文。普通复制行为不变。选区取消、按 Esc、进入源码视图或切换文章后隐藏按钮；不接管手机原生选择菜单。

提问 URL 使用编码后 6,000 字符的项目保守阈值，这不是任何 AI 服务的官方上限。超过阈值时显示完整提问内容供复制，不截断选段。浏览器拒绝自动复制时，面板会选中全文供手动复制。只有点击打开入口才访问对应 AI；选择文字不会发起网络请求。本地预览提供公开 GitHub Pages Markdown 地址和 GitHub 原始文件地址，两者对应线上部署或 `main` 中已提交的版本；尚未提交或部署的本地变更需复制内容或上传文件。

提问功能使用自己的 AI 账户，本站不接模型接口。提示词要求区分作者原文与 AI 解释，不冒充作者。[OpenAI 的网页访问说明](https://developers.openai.com/api/docs/bots)也不承诺一次链接跳转就会读取完整正文。服务端登录状态和页面更新可能影响链接提问，若内容未带入，可使用复制面板。

## 验收

公开构建默认包含全站通过校验且未下架、未过期的正文。公式异常、历史 HTML 与旧式嵌入内容的处理见 [正文校验与许可](mirror.md)。

```bash
uv run --with playwright python scripts/check_local_preview.py --channel msedge
uv run --with playwright python scripts/check_reader_math.py --channel msedge
uv run --with playwright python scripts/check_reader_gpt.py --channel msedge
uv run --with playwright python scripts/check_home_interactions.py --channel msedge
```

浏览器验收拦截外部网络，报告和截图在 `.cache/preview-check/`。图片外链可用性不计入离线验收。完整更新、缓存、下架和公开策略见 [维护说明](maintenance.md)。
