# 日常维护

本项目使用 Python 3.11、uv 与原生 HTML/CSS/JavaScript。正文保留原始 LaTeX；主题、系列和短小结元数据沿用既有规则。

## 一个更新入口

```powershell
uv sync --locked
uv run python scripts/update_all.py --serve
```

该命令串行完成：刷新归档 → 新增正文抓取 → 失败自动重试/快照修复 → 有限量的到期复查 → 补齐科学空间托管图片 → 同步元数据 → 分类与系列 → README/docs → 校验和构建 → 在 <http://127.0.0.1:8765/> 启动预览。

不需要预览服务器时去掉 `--serve`。已经完成的数据在中断后保留，重新执行同一命令即可续跑。无需为转换错误重新请求原站；下次更新会使用快照尝试新转换规则。

文章页只经过一个串行抓取器：同一次 HTTP 200 响应同时提取正文、原站分类、标签和短小结，随后丢弃完整页面。元数据先独立保存，正文转换失败不会丢失已解析的元数据。分类和建站只读取本地数据，不再启动第二轮元数据抓取。完整缓存不会为了同步索引重新访问文章；缺少的小结可以从已校验快照补齐，旧快照不会覆盖较新的分类和标签。

归档请求仍有必要：它用于发现新文章，以及更新标题、日期和文章列表。已有正文无法告诉我们原站新增了哪些文章。归档、正文和必要的元数据补取共用限速与重试；不要另外运行一次旧的元数据抓取。`enrich_posts.py` 的命令入口已转入 `update_all.py`，旧 `--input/--output` 参数不再适用。

```powershell
# 前端开发、离线修复和完整重建，不访问原站
uv run python scripts/update_all.py --offline --serve

# 只启动已有构建
uv run python scripts/serve.py

# 查看内容状态
uv run python scripts/extract_articles.py --status

# 原站明确下架后会自动撤回；也可将 ID 写入配置的 withdrawn_ids
# 只强制复查指定文章
uv run python scripts/extract_articles.py --ids 9119 --refresh

# 显式重试失败，仍尊重服务器 Retry-After
uv run python scripts/extract_articles.py --all --retry-failed

# 仅从已有正文重新提取原文的小结摘录，不请求文章页
uv run python scripts/update_all.py --offline --refresh-summaries
```

## 目录与数据责任

| 目录 | 内容 |
| --- | --- |
| `data/articles/<id>/article.md` | 通过一致性校验、附署名和许可的正文 |
| `data/articles/<id>/source.html` | 仅正文的原始 HTML 快照 |
| `data/articles/<id>/snapshot.json` | 来源、时间和 HTTP 缓存信息 |
| `data/articles/<id>/report.json` | 公式、正文、代码、图片位置等校验结果 |
| `data/articles/<id>/images/` | 该文章缓存的原始图片，按 SHA-256 命名 |
| `data/articles/<id>/images.json` | 原图 URL、文件哈希、大小、条件请求及失败重试状态 |
| `data/articles/state.json` | 持久化逐篇状态，支持断点续跑 |
| `data/posts*.json`、`data/overrides.yaml` | 元数据、分类结果与人工覆盖 |
| `config/mirror.json` | 更新周期、撤回列表与过期检查 |
| `web/` | 网站源码和本地公式排版资源 |
| `build/preview/` | 全文预览产物，不纳入版本控制 |
| `_site/` | 包含全部合格正文的全站部署产物 |
| `.cache/` | 运行日志、临时构建、截图和迁移备份 |

正文持久化于 `data/articles/`。保留 `content/articles/` 的旧工作区可运行 `uv run python scripts/migrate_storage.py`，更早的原型可指定 `--source local/articles`。工具只复制、逐文件核对字节，不删除原目录；写入前检查所有目标冲突，不覆盖新数据。新克隆的仓库不需要迁移。

## 更新与发布

请求间隔至少 3 秒，使用串行抓取器，遵守 robots.txt 和 Retry-After，网络请求最多尝试三次。普通抓取失败冷却一天；连续三篇抓取失败或服务器要求延期会停止本批网络访问。

默认正文满 30 天后进入复查队列，每次最多复查 25 篇，按最早检查时间排序；`--max-refresh 0` 可跳过本次到期复查。旧文章优先使用条件请求；缺元数据时需要完整的 200 响应，同步更新正文，不会先请求 304 再补抓。复查失败的正文不进入新网站，也不会通过离线重转旧快照重新开放；404/410 自动撤回。离线转换及转换失败重试都不会延长原站检查时间。元数据失败单独记录并冷却，汇总会显示未修复项。

网站默认全站公开，已移除试点范围、正文总开关和发布确认字段。预览与公开更新都同时维护全站正文和元数据；`--audience public` 将全部通过校验且未下架、未过期的正文构建到 `_site/`。内容验证失败仍会阻止不完整产物替换旧网站。

Actions 先用 `--skip-build` 更新全站正文、图片与索引，再检查变更；需要发布时只构建一次，没有变化时跳过构建和部署。仅图片失败、延期或检查时间变化会保存重试记录，不会重新部署；新增、移除或替换可用图片，以及正文、撤回配置变化仍会构建。`--skip-build` 不能与 `--serve` 同用，本地统一入口默认仍会构建。

图片补齐失败是可降级警告，显示在 Action 注释与运行汇总中；缺失图片继续使用原始链接，不让已验证正文更新报错。正文抓取、转换、元数据和构建错误仍返回非零状态，必要时构建已完成的有效更新和撤回。单独的 `cache_images.py` 命令继续用非零退出码表示未补齐。更新汇总包含抓取、索引和构建耗时，方便定位慢步骤。

同次提取中已验证的正文只把图片 URL 列表交给图片缓存阶段，避免重复解析全站；这不是跨运行的正文验证缓存，发布构建仍独立重验所有正文、公式和图片哈希。Actions 只缓存小型 robots 规则及域名冷却记录，继续遵守有效期与 Retry-After；运行缓存和临时构建不入库。完全不访问网络请使用 `--offline`。

验证器直接查找可能的公式定界符，按原有规则处理转义与嵌套环境；正文顺序检查的结果在同次校验中复用，公式替换一次拼接，避免反复扫描和复制整篇字符串。校验字段、哈希与输出格式保持兼容。元数据完整时不读取正文快照；缓存命中与延期不会逐篇重写整份 `state.json`，状态变化、转换中的 pending 状态和结束检查点仍原子保存，保留断点续跑能力。这些优化不降低发布校验或请求间隔。

作者署名、许可与非官方声明继续保留，维护者的发布决定不代表作者提供了额外授权。本地更新和构建不执行提交、推送或部署；推送到 `main` 会由 GitHub Actions 自动测试、构建并部署，push 触发的流程不抓取原站。

构建先在 `.cache/builds/` 生成并验证完整产物，再切换目录；转换或验证失败不会用半成品覆盖此前的网站。更新汇总保存于 `.cache/update-result.json`。

## 验收与已知边界

```powershell
uv run pytest
uv lock --check
uv run --with playwright python scripts/check_local_preview.py --channel msedge
uv run --with playwright python scripts/check_reader_navigation.py --channel msedge
uv run --with playwright python scripts/check_reader_math.py --channel msedge
uv run --with playwright python scripts/check_reader_progress.py
uv run --with playwright python scripts/check_home_interactions.py --channel msedge
uv run --with playwright python scripts/check_reader_captions.py
git diff --check
```

自动测试不访问原站；浏览器验收拦截外部请求，覆盖桌面与手机、目录、深色主题、正文、公式、表格、复制和下载。截图和报告位于 `.cache/preview-check/`。

同一系列内切换 `chapter` 只更新目录高亮、正文入口的返回链接和滚动位置，不重建页面。导航验收覆盖目录滚动与折叠状态、重复点击当前章节、前进后退、无效章节和进入正文后的返回；系统要求减少动态效果时使用即时定位。

首页交互验收覆盖圆形文章入口的鼠标与键盘操作、返回时恢复位置、系列续读，以及曲面变形边界、系统减少动态效果和离开页面后的动画清理。曲面沿用原有 SVG 线框，滚出视口或切到后台时自动暂停，不显示播放或暂停按钮，不依赖视频或外部绘图库。

图注只根据已校验原文快照中的 `typecho-caption` 结构识别，构建附带图片序号与原文图注。阅读器仅在相邻段落文字匹配时应用图注样式，不根据图片 alt 猜测，不修改持久化 Markdown。生成 README 和主题文档时，字节相同的文件不重复写入，避免刷新修改时间及 Git 状态噪声。

正文图片保持比例，高度不超过半个视口。点击图片进入预览后，可用加减按钮或滚轮缩放，放大后按住拖动；“适应窗口”恢复居中，“原始尺寸”按原始像素显示。支持加减键、画布内方向键、0 键恢复和 Esc 关闭。关闭时返回原阅读位置，“打开原图”继续指向源站。

文章卡片、系列章节和阅读页共用本地阅读进度，记录读到的最远百分比，往回翻和刷新不会降低进度。公式排版后根据正文位置计算，正文末尾进入可见区域且图片加载结束时才到 100%，此时计入已读筛选和系列完成数。阅读源码、复制、下载或打开原文不会自动标记已读；进度不代表逐字理解情况。旧版仅有布尔值的已读记录可能由外链点击产生，保留原存储但不推断为 100%。新记录使用 `spaces-index-reading-progress-v1`，不上传服务器。

悬停正文中的公式编号引用会显示对应公式，键盘聚焦同样可预览；移开鼠标、滚动、按 Esc 或离开阅读页时收起。预览复用已排版公式，不重新运行 MathJax，不修改原始 LaTeX、锚点或阅读进度。点击引用仍平滑跳转，并可返回引用处；触屏维持直接点击跳转。

完整选中公式主体后复制，会得到保留定界符的原始 LaTeX，不必选中右侧编号或上下文；仅选择公式内的一部分字符时保留字符复制。混合选择文字与公式、选区跨出正文边界时也会还原公式源码，并排除隐藏的辅助文本。会接管复制事件的浏览器扩展可能覆盖该行为，发生冲突时应在本站停用扩展。

在线更新自动补齐已缓存正文中的缺失图片，无需重抓文章页。科学空间主域及 `album`、`bbs` 子域的图片保存到各自文章目录；站外图片不下载。已缓存且哈希正确的图片跳过；文章到期复查时，关联图片使用 ETag/Last-Modified 复查。同次更新重复图片 URL 复用下载结果。下载失败独立记录，不让文章正文失效；普通失败冷却一天，404/410 冷却七天，Retry-After 严格遵守。临时复查失败保留旧副本，确认 404/410 后撤掉本地副本。离线模式只使用已有图片，不请求网络。

只补齐图片、不重新请求正文：

```bash
uv run python scripts/cache_images.py --all --sleep 3 --build
# 少量试跑，最多请求 5 个图片 URL
uv run python scripts/cache_images.py --ids 9881 9119 --limit 5
# --build 下载完成后重建预览与公开产物，不部署；随后启动预览
uv run python scripts/serve.py
```

图片单独补齐日志与汇总位于 `.cache/image-cache/`；统一更新的汇总包含图片统计。中断后重复命令可续跑。首次全站补齐约 2000 个图片地址，按 3 秒间隔仅等待时间约 100 分钟，还需网络传输时间。GitHub Actions 沿用统一入口，保存 `data/articles/` 时也包含图片和清单。缓存图片禁用 Git 文本换行转换，确保 SVG 等文件跨系统检出后仍与原始哈希一致。

图片按域名独立退避：robots 检查在有限重试后仍失败，或同一域名连续三次图片请求失败，便暂停该域名，继续其他源站域名，不再被一个失效子域阻断全批。`hosts_paused` 表示本次暂停的域名数，`host_deferred` 表示因此暂缓的图片数；未实际请求的图片不记为下载失败。暂停状态保存在 `.cache/image-cache/hosts-*.json`，默认冷却一天，`--retry-failed` 可重试普通错误；服务器的 Retry-After 对整个域名及其重定向目标生效，跨运行仍须遵守。

`config/mirror.json` 的 `skip_image_hosts` 当前包含持续无法解析的 `album.spaces.ac.cn` 与 `bbs.spaces.ac.cn`。自动更新和手动图片缓存都不再请求这两个子域，也不请求其 robots；重定向到它们的请求同样跳过。正文保留原图 URL，已有可用缓存仍保留。`skipped` 表示按配置主动跳过，不算失败或待重试；`--retry-failed` 不覆盖该配置。子域恢复后，从列表中移除对应域名即可重新启用，既有服务器 Retry-After 仍须到期。

`failed`、`deferred` 或 `pending` 非零表示尚未补齐；旧日志中的 `stopped_early: 1` 也表示未完成。`--build` 仍可用当前缓存构建网站，未缓存图片继续使用原链接；构建完成不等于全部下载完成。图片补齐命令未完成时返回非零退出码。TLS/SSL 失败不会关闭证书校验、改用 HTTP 或绕过 robots 检查，稍后由维护者重新运行同一命令续跑。

SVG 校验兼容 Matplotlib 导出的 RDF、Dublin Core、CC 元数据，以及 draw.io / MathJax 导出的静态文字、XHTML 样式和不执行的 `math/tex` 数据块。CSS 使用解析器检查，允许内部引用和 MathJax 的 `about:blank` 空白字体占位；外部字体、外部资源、可执行脚本和事件处理器仍不缓存。图片保留原始字节，不删除生成工具、日期、署名或 TeX 信息。修复校验规则后，可加 `--retry-failed` 重试先前的误判，无需重新下载成功的图片；正在运行的旧进程需要结束后重新启动才能使用新规则。

部分历史正文使用 Markdown 中的 HTML 保留复杂结构，并独立逐节点比对，避免转换时遗漏。`1023`、`116`、`49` 的原文公式定界符异常保持原样，页面提示并不伪造公式总数。旧式脚本与插件保留位置和原始信息但不执行；远程图片与媒体仍取决于原地址是否可用。
