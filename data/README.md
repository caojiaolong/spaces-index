# 科学空间数据

`articles/<id>/` 保存逐篇 `article.md`、仅正文的 `source.html`、`snapshot.json` 与 `report.json`。`articles/state.json` 记录断点和检查状态，`articles/archive.json` 保存最近一次归档发现结果。这些是持久数据，不是运行缓存。

`posts_raw.json` 是归档元数据，`posts.json` 合并文章页原站分类、标签及短小结，`posts_classified.json` 是主题和系列分类结果。人工修正写入 `overrides.yaml`，不要直接改生成文件。

日常只运行 `uv run python scripts/update_all.py`：同一次文章页响应同时提供正文和元数据。已有快照用于离线修复，不需要再单独运行元数据抓取器。过程日志和临时文件在 `.cache/`，预览产物在 `build/preview/`。详见 [维护说明](../docs/maintenance.md)。

文章作者为苏剑林，出处为科学空间。每份 Markdown 附原文链接、CC BY-NC-ND 2.5 CN 与非官方声明；源码许可证不覆盖文章和第三方素材。

科学空间托管的图片缓存到 `articles/<id>/images/`，同目录 `images.json` 记录原始地址、SHA-256 和更新状态。在线统一更新自动补齐，离线更新不访问图片服务器；站外图片保留外链。托管域名规则不等于逐图许可证明。

默认提供全站正文，无试点或发布开关。网站构建仍排除已下架、过期或校验失败的正文；`config/mirror.json` 仅管理下架与更新规则。自动更新保存全站正文和元数据，运行缓存与构建产物不入库。作者署名、许可和非官方声明继续保留；发布决定不代表作者提供了额外授权。
