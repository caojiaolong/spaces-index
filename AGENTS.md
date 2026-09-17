# AGENTS.md

本仓库是面向 [科学空间](https://spaces.ac.cn/) 的非官方、非商业知识库。保持可重复运行、尊重原站、主题与系列索引完整、正文忠于原文。

## 当前范围

- 维护者已授权全站正文提取、修复和网站预览，要求将原型整理为正式项目结构。正文持久化于 `data/articles/`，不再使用 `local/`。
- 维护者明确要求默认全站公开，已移除试点、全站发布及授权确认开关。公开构建包含全部通过校验且未下架、未过期的正文。提交和推送遵循维护者当前指令；推送到 `main` 会触发现有 GitHub Pages 构建与部署。
- 保留既有首页视觉、配色、字体和页面结构，不擅自重新设计风格。Markdown 阅读使用主站文章路由，共用导航、主题、返回顶部和阅读记录，不维护第二套页面框架。
- 作者署名、许可及非官方声明必须保留；项目的发布决定不代表作者提供了额外授权。
- 正文只做必要格式转换，不摘要、润色、翻译或重组；每篇附苏剑林署名、原文链接、CC BY-NC-ND 2.5 CN 和非官方声明。
- 保留原始 LaTeX，独立核对文字、公式、代码、链接、图片位置及顺序。校验失败不提供 Markdown；构建时重新核对快照、内容和哈希。
- 不收录评论、侧栏、推荐、打赏及引用模板。只保存边界已确认的正文快照，不保存完整页面响应。
- 维护者已授权缓存科学空间托管的图片（`spaces.ac.cn`、`album.spaces.ac.cn`、`bbs.spaces.ac.cn`），保存到对应 `data/articles/<id>/images/`，同目录 `images.json` 记录来源、哈希、更新与失败状态。站外图片保留原链接，重定向也不得越过上述域名。Markdown 保留原图链接，网站映射到已校验的本地图片；缓存策略不代表图片权利核验。旧式脚本和插件不得执行，原文错误公式保留并提示。
- 后续知识问答须区分原文与系统解读、提供出处，不冒充作者、不将推断当作作者观点。

## 技术栈与统一入口

Python 3.11、uv、requests、BeautifulSoup、markdown-it-py；前端原生 HTML/CSS/JavaScript，本地 MathJax 排版。

```bash
uv sync --locked
# 增量抓取、修复、有限复查、元数据、分类、索引、构建、预览
uv run python scripts/update_all.py --serve
# 开发时完全离线更新和预览
uv run python scripts/update_all.py --offline --serve
# 只启动已经完成的预览构建
uv run python scripts/serve.py
```

`--serve` 仅绑定 `127.0.0.1:8765`，去掉该参数只更新和构建。默认产物 `build/preview/`，`--audience public` 输出全站部署产物 `_site/`。不自动部署。

## 目录与流程

1. `scripts/extract_articles.py`：归档、串行正文抓取、转换与校验；持久化 `data/articles/<id>/` 及 `state.json`。正文请求同时提取元数据，避免重复访问。
2. `scripts/update_all.py`：统一编排；从同一次文章页响应和本地快照同步 `data/posts_raw.json`、`data/posts.json`，元数据同步本身不得发起 HTTP。归档请求用于发现新文章，仍需保留。
3. `scripts/classify.py`：规则分类与系列识别，不依赖 LLM；`data/overrides.yaml` 优先。
4. `scripts/render_markdown.py`：生成 README 和 docs 主题页。
5. `scripts/build_site.py`：重验正文，先完成临时构建再替换产物；失败构建不得覆盖此前完整网站。

持久路径集中于 `scripts/paths.py`。策略位于 `config/mirror.json`；运行日志、汇总、临时构建和验收截图放 `.cache/`。`scripts/migrate_storage.py` 仅用于旧工作区迁移，按字节核对并拒绝覆盖冲突文件。

网络请求必须串行，使用合理 User-Agent，遵守 robots.txt 和 Retry-After，最小间隔 3 秒。正文默认 30 天复查，每次最多 25 篇，按检查时间轮换；失败自动冷却、重试和断点续跑。404/410 自动撤回，主动下架使用 `withdrawn_ids`。离线转换不得延长原站检查时间。

提取阶段可复用 `.cache/ingestion/verified-bodies.json` 中已校验正文的指纹和图片 URL，必须先核对四个正文文件的字节哈希及校验器代码、依赖版本。发布构建不得使用这份缓存跳过正文、公式或图片校验。Action 汇总区分归档网络、归档解析、正文与图片耗时。

## 分类与生成文件

- 分类规则在 `TOPICS`、`TOPIC_KEYWORDS`、`SOURCE_CATEGORY_TOPICS`；系列规则在 `detect_series_info()` 和 `detect_prefix_series_candidate()`。
- 同一系列以成员主题众数统一 `series_topic`，文章按序号正序排列。
- 错误分类应修改规则或 overrides，不直接修改生成的 JSON。
- `source_summary` 只提取明确“小结 / 文章小结 / 总结 / 结语 / 结束语 / 后记”段落的短摘录，最多 320 字。
- README 顶部保留 motivation；目录合并主题统计并链接主题、系列及非系列文章块；本地命令、流程和详细元数据入口放底部。
- `docs/README.md` 是文档导航；`docs/topics/` 为自动生成的主题索引，保留分类、标签、系列号、系列主题、短小结和备注；`docs/guides/` 为人工维护的使用、维护与许可文档。生成脚本只更新根 README 和 `docs/topics/`。

## 自动更新和验收

GitHub Actions 每天 UTC 02:23 或手动执行 `scripts/update_all.py --audience public --sleep 3`；push 只测试、构建和部署，不抓取。Action 保存全站正文与索引变化并部署，下架也需要重新构建和部署。

正文与索引同属 `data/`，Actions 按明确路径保存元数据和 `data/articles/`，运行缓存和构建产物不得入库。

测试不得访问原站。解析、转换、分类及发布策略使用内联 HTML、样例和临时目录。修改分类规则后运行 classify/render；修改阅读器后执行浏览器验收。

```bash
uv run pytest
uv lock --check
uv run --with playwright python scripts/check_local_preview.py --channel msedge
git diff --check
git status --short
```

确认 `.venv/`、`.pytest_cache/`、`__pycache__/`、`.cache/` 和构建产物未进入版本控制。细节见 [文档导航](docs/README.md)、[维护说明](docs/guides/maintenance.md) 和 [正文校验与许可](docs/guides/mirror.md)。
