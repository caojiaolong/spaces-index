# 正文提取与失败恢复

[文档导航](../README.md) · [日常维护](maintenance.md) · [正文校验与许可](mirror.md)

日常使用统一入口，完成正文、元数据、分类、索引与网站更新：

```bash
uv run python scripts/update_all.py --serve
```

完整命令、策略和目录见 [维护说明](maintenance.md)。正文已迁移到 `data/articles/`，不再写入原型 `local/articles/`。

## 单独诊断正文

```bash
# 查看保存进度和失败原因
uv run python scripts/extract_articles.py --status
# 少量试跑，已有文章复用缓存
uv run python scripts/extract_articles.py --all --limit 5 --sleep 3
# 全部失败项仅用正文快照重新转换，不访问原站
uv run python scripts/extract_articles.py --all --retry-failed --offline
# 指定文章重新访问原站，尊重服务器要求的等待时间
uv run python scripts/extract_articles.py --ids 9119 --refresh --sleep 3
```

全站范围以归档列出的 ID 为准，不枚举未知数字 ID。每篇完成后保存状态，网络中断后重跑即可恢复。普通更新自动重试冷却期已过的抓取失败；转换失败优先复用快照。显式 `--retry-failed` 提前重试普通失败，但不跳过 Retry-After。

`CONVERSION_FAILED: Unsupported body element: u` 曾由旧转换器缺少下划线标签支持引起。当前已补齐下划线、代码、表格、列表和历史结构；复杂文章通过 Markdown 内嵌 HTML 保留，独立逐节点校验。支持范围之外的结构仍会失败关闭，不自动删除未知内容以制造通过结果。

日志和汇总在 `.cache/ingestion/`，正文、快照及报告在 `data/articles/<id>/`，`state.json` 记录抓取与校验状态。存储正文不等于公开，公开策略见 [正文校验与许可](mirror.md)。
