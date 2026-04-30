# AGENTS.md

本仓库是一个面向 [科学空间](https://spaces.ac.cn/) 的元数据索引项目。后续维护时请优先保持“可重复运行、尊重原站、只保存元数据”的设计。

## 项目边界

- 不要镜像、复制或保存文章全文。
- 可以保存的字段包括：标题、URL、日期、原站分类、原站标签、自动主题、系列名、系列序号、系列主题、备注、少量明确小结短摘录。
- `source_summary` 只从明确的“小结 / 文章小结 / 总结 / 结语 / 结束语 / 后记”等段落提取短摘录，当前限长 320 字。
- 网络抓取必须串行执行，设置合理 User-Agent，并通过 `--sleep` 控制请求间隔。

## 环境与命令

本项目使用 `uv` 管理 Python 3.11 环境。

```bash
uv sync
uv run pytest
uv run python scripts/update_all.py --sleep 0.8 --progress-every 25
```

常用维护命令：

```bash
# 只重新分类和渲染，不访问文章页
uv run python scripts/classify.py
uv run python scripts/render_markdown.py

# 强制重抓所有文章页元数据，慎用
uv run python scripts/update_all.py --force --sleep 0.8

# 只补齐缺少 source_summary 的历史文章，会访问旧文章页
uv run python scripts/update_all.py --refresh-summaries --sleep 0.8
```

## 数据流

1. `scripts/fetch_archive.py`
   - 抓取 `https://spaces.ac.cn/content.html`。
   - 只解析 `id/title/url/date`。
   - 输出 `data/posts_raw.json`。

2. `scripts/enrich_posts.py`
   - 读取 `data/posts_raw.json`。
   - 对照 `data/posts.json` 缓存。
   - 已有 `source_category/source_tags` 的文章默认跳过，不访问文章页。
   - 新文章才访问原文页补元数据。
   - 输出 `data/posts.json`。

3. `scripts/classify.py`
   - 读取 `data/posts.json`。
   - 使用规则分类，不依赖 LLM API。
   - 识别系列名、系列序号，并用同一系列成员主题的众数统一 `series_topic`。
   - `data/overrides.yaml` 优先级最高。
   - 输出 `data/posts_classified.json`。

4. `scripts/render_markdown.py`
   - 读取 `data/posts_classified.json`。
   - 生成 `README.md` 和 `docs/*.md`。
   - README 是主入口；docs 是详细元数据页。

5. `scripts/update_all.py`
   - 串联以上步骤，是本地更新和 GitHub Actions 的入口。

## README 与 docs 约定

- README 顶部保留 motivation，说明本项目解决人工整理帖长期不更新的问题。
- README 的目录合并主题统计，并链接到 README 内部的主题、系列、非系列文章块。
- 系列文章应按序号正序排列。
- 本地运行、更新流程、详细元数据入口放在 README 底部。
- docs 页面保留更完整的元数据：分类、标签、系列号、系列主题、小结摘录、备注。

## 分类与系列维护

- 分类规则主要维护在 `scripts/classify.py` 的 `TOPICS`、`TOPIC_KEYWORDS`、`SOURCE_CATEGORY_TOPICS`。
- 标题序列规则维护在 `detect_series_info()`。
- 无显式序号但共享醒目前缀的系列规则维护在 `detect_prefix_series_candidate()`。
- 如果规则误判，不要硬改生成后的 JSON；优先修规则或在 `data/overrides.yaml` 中覆盖。
- 新增规则后至少运行：

```bash
uv run pytest
uv run python scripts/classify.py
uv run python scripts/render_markdown.py
```

## GitHub Actions

`.github/workflows/update.yml` 每天 UTC 02:23 运行，也支持手动触发。

Action 会：

- `uv sync --locked`
- `uv run python scripts/update_all.py --sleep 0.8 --progress-every 25`
- 如果 `README.md`、`docs/` 或 `data/` 有变化，则自动 commit 并 push。

首次发布时请务必提交 `data/posts.json`。这样日常 Action 会复用缓存，只抓归档页和新增文章页，不会每天全量访问所有文章。

## 测试

测试不应真实访问网络。新增解析、分类、渲染规则时，请使用内联 HTML 或样例字典。

```bash
uv run pytest
```

## 提交前检查

```bash
uv run pytest
uv lock --check
git status --short
```

确认 `.venv/`、`.pytest_cache/`、`__pycache__/` 没有进入版本控制。
