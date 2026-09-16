# 全文预览

```bash
# 增量更新后预览
uv run python scripts/update_all.py --serve
# 已有正文，完全离线重建和预览
uv run python scripts/update_all.py --offline --serve
```

打开 <http://127.0.0.1:8765/>。产物在 `build/preview/`，正文在 `data/articles/`；服务器仅绑定本机，预览命令不执行部署。

首页、文章探索、主题及系列页的文章标题默认打开站内 Markdown 阅读页，未提供正文的文章仍指向原文。卡片下方与阅读页顶部集中提供“查看原文、下载 Markdown、复制 Markdown”。阅读器提供源码切换、章节目录、进度、字号、深色主题、系列上下篇和站内文章链接。复制和下载保留完整署名、原文链接及许可，不受阅读排版影响。

文章旁的圆形箭头与标题进入同一篇正文，支持键盘操作；“继续阅读”和系列的“继续下一篇”直接打开未读文章。首页线框曲面会缓慢旋转、起伏与变形，可手动暂停；滚出视口或页面进入后台时自动暂停，并跟随系统减少动态效果设置。

公式内部可以像普通文字一样拖选符号，Ctrl+C（macOS 为 ⌘C）复制选中的可见字符；这是纯文本，不保留分式或上下标的二维结构。选中包含公式的整段文字，或聚焦单个公式后直接复制，仍获得原始 LaTeX。公式不提供点击弹窗。点击编号引用会平滑滚动到对应公式并短暂高亮，右下方出现“返回引用处”，可恢复跳转前的位置及键盘焦点。连续跳转支持逐次返回；切换文章后记录清空。跳转与返回都尊重系统减少动态效果的设置，单独打开编号链接也能定位。原文中未配对的公式标记继续原样保留。

公式使用本地 MathJax 4.1.3 的 CommonHTML 输出与 TeX 数学字体，保留正常行内基线和文字间距，长公式可横向滚动。组件和字体随网站构建一起提供，排版无需访问原站或外部 CDN，也不修改存储的 LaTeX。

正文图片按比例显示，高度不超过当前窗口的一半。点击图片或聚焦后按 Enter 可放大查看，支持适应窗口、原始尺寸和打开原图；按 Esc、关闭按钮或点击遮罩退出，回到原阅读位置。图片仍使用原始外链，图注根据原文标记显示。

阅读页使用主站的 `#/article/<id>` 路由，共用导航、页脚、主题、返回顶部和已读状态。从列表进入后返回，会恢复筛选条件和滚动位置。旧 `reader.html?id=…` 链接自动跳转到主站文章页。

当前 1337 篇正文均通过源内容一致性校验；1256 篇为常规 Markdown，81 篇含保留历史结构的 HTML。验收样例 9119 与 11882 分别保留 142 和 101 段公式。公开构建默认包含全站通过校验且未下架、未过期的正文。原文公式异常、旧式嵌入内容与远程图片的边界见 [正文说明](mirror.md)。

```bash
uv run --with playwright python scripts/check_local_preview.py --channel msedge
uv run --with playwright python scripts/check_reader_math.py --channel msedge
uv run --with playwright python scripts/check_home_interactions.py --channel msedge
```

浏览器验收拦截外部网络，报告和截图在 `.cache/preview-check/`。图片外链可用性不计入离线验收。完整更新、缓存、下架和公开策略见 [维护说明](maintenance.md)。
