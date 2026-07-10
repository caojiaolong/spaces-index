from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .common import DATA_DIR, DOCS_DIR, ROOT, clean_text, log, read_json, sort_posts_desc
    from .classify import TOPICS
    from .build_site import stable_series_id
except ImportError:  # pragma: no cover - used when running as python scripts/render_markdown.py
    from common import DATA_DIR, DOCS_DIR, ROOT, clean_text, log, read_json, sort_posts_desc
    from classify import TOPICS
    from build_site import stable_series_id


TOPIC_PAGES: dict[str, str] = {
    "深度学习基础": "deep-learning.md",
    "词向量与Embedding": "embeddings.md",
    "大模型与Transformer": "transformer.md",
    "生成模型": "generative-models.md",
    "优化与训练": "optimization.md",
    "数学工具": "math.md",
    "概率统计与信息论": "probability-info.md",
    "几何与方程": "geometry-equations.md",
    "NLP与信息抽取": "nlp.md",
    "工程工具": "engineering.md",
    "天文科普": "astronomy.md",
    "物理化学": "physics-chemistry.md",
    "生物自然": "biology.md",
    "图片摄影": "photography.md",
    "科普问答与百科": "popular-science.md",
    "资源与站务": "resources.md",
    "阅读写作与随笔": "essays.md",
    "其他": "other.md",
}

PAGES_URL = "https://caojiaolong.github.io/spaces-index/"
SOURCE_URL = "https://spaces.ac.cn/"
STAR_HISTORY_TOKEN = (
    "1PtTrJfjwB8TiUNumdr03-qtw6bgfzWVHU7gDNV6rSiul2pnBRigKz4Clvj0JKReYQSx"
    "BrxmBDqLY85wFZ86ySlHRzel8aw7XfnwQ17BcMI46iGX"
)
STAR_HISTORY_CHART_URL = (
    "https://api.star-history.com/chart?repos=caojiaolong/spaces-index"
    f"&type=timeline&legend=top-left&sealed_token={STAR_HISTORY_TOKEN}"
)
STAR_HISTORY_DARK_CHART_URL = (
    "https://api.star-history.com/chart?repos=caojiaolong/spaces-index"
    f"&type=timeline&theme=dark&legend=top-left&sealed_token={STAR_HISTORY_TOKEN}"
)


def escape_md(text: Any) -> str:
    value = clean_text(str(text or ""))
    return value.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def post_link(post: dict[str, Any]) -> str:
    return f"[{escape_md(post.get('title'))}]({post.get('url')})"


def dated_post_link(post: dict[str, Any]) -> str:
    date = escape_md(post.get("date"))
    return f"{date} - {post_link(post)}" if date else post_link(post)


def post_identity(post: dict[str, Any]) -> str:
    return str(post.get("id") or post.get("url") or post.get("title") or "")


def topic_anchor(topic: str) -> str:
    page = TOPIC_PAGES.get(topic, topic)
    return f"topic-{Path(page).stem}"


def stable_anchor(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def series_anchor(topic: str, series: str) -> str:
    return stable_anchor(f"series-{Path(TOPIC_PAGES.get(topic, topic)).stem}", series)


def standalone_anchor(topic: str) -> str:
    return f"series-{Path(TOPIC_PAGES.get(topic, topic)).stem}-standalone"


def series_page_url(series: str) -> str:
    return f"{PAGES_URL}#/series/{stable_series_id(series)}"


def posts_for_topic(posts: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    return sort_posts_desc(
        [post for post in posts if topic in [str(item) for item in post.get("topics", [])]]
    )


def series_post_sort_key(post: dict[str, Any]) -> tuple[int, int, str, str]:
    index = post.get("series_index")
    if isinstance(index, int):
        return (0, index, str(post.get("date") or ""), str(post.get("title") or ""))
    return (1, 0, str(post.get("date") or ""), str(post.get("title") or ""))


def group_series(posts: list[dict[str, Any]]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    standalone: list[dict[str, Any]] = []
    for post in posts:
        series = clean_text(str(post.get("series") or ""))
        if series:
            by_series[series].append(post)
        else:
            standalone.append(post)

    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for series, series_posts in by_series.items():
        if len(series_posts) < 2:
            standalone.extend(series_posts)
            continue
        groups.append((series, sorted(series_posts, key=series_post_sort_key)))

    groups.sort(
        key=lambda item: max(str(post.get("date") or "") for post in item[1]),
        reverse=True,
    )
    return groups, sort_posts_desc(standalone)


def append_post_details(lines: list[str], post: dict[str, Any]) -> None:
    tags = post.get("source_tags") or []
    tag_text = "、".join(escape_md(tag) for tag in tags) if tags else "无"
    lines.append(f"- {dated_post_link(post)}")
    lines.append(f"  - 原站分类：{escape_md(post.get('source_category')) or '无'}")
    lines.append(f"  - 原站标签：{tag_text}")
    if post.get("series"):
        series_text = escape_md(post.get("series"))
        if post.get("series_index") is not None:
            series_text += f" #{post.get('series_index')}"
        lines.append(f"  - 系列：{series_text}")
    if post.get("series_topic"):
        lines.append(f"  - 系列主题：{escape_md(post.get('series_topic'))}")
    if post.get("source_summary"):
        lines.append(f"  - 小结摘录：{escape_md(post.get('source_summary'))}")
    if post.get("notes"):
        lines.append(f"  - 备注：{escape_md(post.get('notes'))}")


def append_grouped_posts(
    lines: list[str],
    posts: list[dict[str, Any]],
    *,
    detailed: bool,
    topic: str,
    back_to_directory: bool = False,
) -> None:
    def group_heading(title: str) -> str:
        heading = f"#### {escape_md(title)}"
        if back_to_directory:
            heading += " [返回目录](#目录)"
        return heading

    groups, standalone = group_series(posts)
    if not groups:
        lines.append(f'<a id="{standalone_anchor(topic)}"></a>')
        lines.extend([group_heading("非系列文章"), ""])
        for post in posts:
            if detailed:
                append_post_details(lines, post)
            else:
                lines.append(f"- {dated_post_link(post)}")
        return

    for series, series_posts in groups:
        lines.append(f'<a id="{series_anchor(topic, series)}"></a>')
        lines.extend([group_heading(series), ""])
        for post in series_posts:
            if detailed:
                append_post_details(lines, post)
            else:
                lines.append(f"- {dated_post_link(post)}")
        lines.append("")

    if standalone:
        lines.append(f'<a id="{standalone_anchor(topic)}"></a>')
        lines.extend([group_heading("非系列文章"), ""])
        for post in standalone:
            if detailed:
                append_post_details(lines, post)
            else:
                lines.append(f"- {dated_post_link(post)}")


def topic_counts(posts: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for post in posts:
        topics = post.get("topics") or ["其他"]
        for topic in topics:
            counts[str(topic)] += 1
    return counts


def latest_post_date(posts: list[dict[str, Any]]) -> str:
    sorted_posts = sort_posts_desc(posts)
    return str(sorted_posts[0].get("date") or "暂无") if sorted_posts else "暂无"


def render_directory(posts: list[dict[str, Any]]) -> str:
    counts = topic_counts(posts)
    lines: list[str] = []
    for topic in TOPICS:
        topic_posts = posts_for_topic(posts, topic)
        lines.append(f"- [{topic}（{counts.get(topic, 0)} 篇）](#{topic_anchor(topic)})")
        groups, standalone = group_series(topic_posts)
        for series, series_posts in groups:
            lines.append(
                f"  - [{escape_md(series)}（{len(series_posts)} 篇）]({series_page_url(series)})"
            )
        if standalone:
            lines.append(f"  - [非系列文章（{len(standalone)} 篇）](#{standalone_anchor(topic)})")
    return "\n".join(lines)


def build_series_link_index(posts: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    links: dict[str, tuple[str, str]] = {}
    for topic in TOPICS:
        topic_posts = posts_for_topic(posts, topic)
        groups, _ = group_series(topic_posts)
        for series, series_posts in groups:
            url = series_page_url(series)
            for post in series_posts:
                identity = post_identity(post)
                if identity and identity not in links:
                    links[identity] = (series, url)
    return links


def recent_post_line(
    post: dict[str, Any],
    series_links: dict[str, tuple[str, str]],
) -> str:
    line = f"- {dated_post_link(post)}"
    series_link = series_links.get(post_identity(post))
    if series_link:
        _, url = series_link
        line += f" - [查看系列]({url})"
    return line


def render_readme(posts: list[dict[str, Any]]) -> str:
    posts = sort_posts_desc(posts)
    series_links = build_series_link_index(posts)
    lines: list[str] = [
        '<p align="center">',
        '  <img src="assets/readme-hero.svg" alt="科学空间文章索引：由曲线、节点与公式组成的抽象数学图形" width="100%">',
        "</p>",
        "",
        '<h1 align="center">科学空间文章索引</h1>',
        "",
        '<p align="center"><strong>在科学与公式之间，找到下一篇值得读的文章。</strong></p>',
        '<p align="center">',
        f'  <a href="{PAGES_URL}"><strong>在线探索</strong></a>',
        "  ·",
        f'  <a href="{SOURCE_URL}">访问科学空间</a>',
        "  ·",
        '  <a href="docs/index.md">详细元数据</a>',
        "</p>",
        '<p align="center"><sub>非官方、持续更新、只保存元数据，不镜像文章正文。</sub></p>',
        "",
        "## 最近更新",
        "",
        "> **2026-07-10 · 交互式 GitHub Pages 体验升级**",
        "",
        f"- **更适合手机浏览**：重做移动端首屏、统计卡片和底部导航，筛选使用底部抽屉，并提供右下角返回顶部按钮。",
        f"- **最近文章更有上下文**：在线首页直接展示文章小结；属于系列的文章可以一键进入完整系列时间线。",
        f"- **探索筛选更完整**：支持搜索、主题、标签、年份、难度、系列文章、非系列文章以及已读/未读组合筛选，筛选状态可以随 URL 分享。",
        f"- **本地阅读进度**：点击原文会自动标为已读，也可以手动切换；记录只保存在当前浏览器，不会上传。",
        "",
        f"[前往在线索引体验这些功能 →]({PAGES_URL})",
        "",
        "## 为什么做这个索引",
        "",
        "苏剑林老师在科学空间积累了大量高质量文章，主题横跨大模型、生成模型、优化、数学、NLP、工程实践和科普随笔。网上也有不少人工整理帖，例如 [这类知乎整理](https://zhuanlan.zhihu.com/p/1935115608074196190)，但人工清单常见的问题是：刚发布时很好用，时间一长就容易停止更新，新文章、系列续篇和分类调整很难持续同步。",
        "",
        "这个仓库的目标是把科学空间的所有文章做成一个持续更新的元数据索引：不复制全文，只保存标题、日期、原文链接、原站分类、标签、自动主题和系列信息，并通过 GitHub Actions 定时更新。这样读者可以直接按主题或系列查找文章，跳转回原站阅读，也不用担心索引长期失修。",
        "",
        "| 文章 | 主题 | 系列 | 最近更新 |",
        "| ---: | ---: | ---: | :---: |",
        f"| {len(posts)} 篇 | {sum(1 for topic in TOPICS if posts_for_topic(posts, topic))} 个 | {len(group_series(posts)[0])} 个 | {latest_post_date(posts)} |",
        "",
        "> 本项目保存标题、链接、日期、分类、标签、自动主题、系列信息和少量小结短摘录；不镜像、复制或保存文章全文。",
        "",
        "## 三种浏览方式",
        "",
        f"- [全文检索]({PAGES_URL}#/explore)：按标题、标签、系列和小结快速搜索，组合主题、年份、系列状态与已读状态等筛选条件。",
        f"- [主题漫游]({PAGES_URL}#/topics)：从大模型、数学、自然科学等方向逐层发现文章。",
        f"- [系列阅读]({PAGES_URL}#/series)：按章节顺序阅读 Transformer、扩散模型等长篇系列。",
        "",
        "## 目录",
        "",
        "<details>",
        "<summary><strong>展开完整主题、系列与非系列目录</strong></summary>",
        "",
        render_directory(posts),
        "",
        "注：系列文章会统一归入该系列的众数主题；非系列文章仍可能属于多个主题，因此目录中的主题数量之和可能大于文章总数。",
        "",
        "</details>",
        "",
        "## 最近 10 篇文章",
        "",
    ]
    if posts:
        for post in posts[:10]:
            lines.append(recent_post_line(post, series_links))
    else:
        lines.append("- 暂无文章数据。")

    lines.extend(["", "## 主题分类", ""])
    for topic in TOPICS:
        topic_posts = posts_for_topic(posts, topic)
        lines.extend(
            [
                f'<a id="{topic_anchor(topic)}"></a>',
                "<details>",
                f"<summary><strong>{escape_md(topic)}</strong> · {len(topic_posts)} 篇</summary>",
                "",
                "[返回目录](#目录)",
                "",
            ]
        )
        if topic_posts:
            append_grouped_posts(
                lines,
                topic_posts,
                detailed=False,
                topic=topic,
                back_to_directory=True,
            )
        else:
            lines.append("- 暂无文章。")
        lines.extend(["", "</details>", ""])

    lines.extend(["", "## 详细元数据", ""])
    for topic in TOPICS:
        page = TOPIC_PAGES.get(topic)
        lines.append(f"- [{topic}](docs/{page})" if page else f"- [{topic}](#其他)")

    lines.extend(
        [
            "",
            "## 本地运行",
            "",
            "```bash",
            "uv sync",
            "uv run python scripts/update_all.py --sleep 0.8 --progress-every 25",
            "uv run python scripts/build_site.py --input data/posts_classified.json --output _site",
            "# 本地预览（保持此命令运行，再打开 http://127.0.0.1:8000/）",
            "uv run python -m http.server 8000 --directory _site",
            "# 可选：补齐历史文章的小结短摘录，会重新访问缺少小结字段的旧文章",
            "uv run python scripts/update_all.py --refresh-summaries --sleep 0.8",
            "```",
            "",
            "> 本地预览必须通过 HTTP 服务访问；不要直接双击 `web/index.html` 或 `_site/index.html`。",
            "",
            "## 更新流程",
            "",
            "- `fetch_archive.py`：从归档页获取文章 `id`、标题、URL、日期。",
            "- `enrich_posts.py`：逐篇访问原文页，只提取原站分类、标签和可选小结短摘录，并写入缓存。",
            "- `classify.py`：根据标题、分类、标签做规则分类，识别系列名与序号，并用系列成员主题众数统一系列主题。",
            "- `render_markdown.py`：稳定生成折叠式 README 和 docs 主题页。",
            "- `build_site.py`：从分类数据构建 `_site/` 静态站点与浏览器端目录数据。",
            "- GitHub Actions：定时更新索引，并将同一次运行生成的静态产物部署到 GitHub Pages。",
            "",
            "## Star History",
            "",
            "如果这个索引对你有帮助，欢迎 Star 支持，后续会通过 GitHub Actions 持续更新。",
            "",
            '<a href="https://www.star-history.com/?repos=caojiaolong%2Fspaces-index&type=timeline&legend=top-left">',
            " <picture>",
            f'   <source media="(prefers-color-scheme: dark)" srcset="{STAR_HISTORY_DARK_CHART_URL}" />',
            f'   <source media="(prefers-color-scheme: light)" srcset="{STAR_HISTORY_CHART_URL}" />',
            f'   <img alt="Star History Chart" src="{STAR_HISTORY_CHART_URL}" />',
            " </picture>",
            "</a>",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_stats_table(posts: list[dict[str, Any]]) -> str:
    counts = topic_counts(posts)
    lines = ["| 主题 | 数量 |", "| --- | ---: |"]
    for topic in TOPICS:
        lines.append(f"| {topic} | {counts.get(topic, 0)} |")
    return "\n".join(lines)


def render_docs_index(posts: list[dict[str, Any]]) -> str:
    counts = topic_counts(posts)
    lines = [
        "# 主题索引",
        "",
        f"文章总数：{len(posts)}",
        "",
        "| 主题 | 数量 | 页面 |",
        "| --- | ---: | --- |",
    ]
    for topic in TOPICS:
        page = TOPIC_PAGES.get(topic)
        target = f"[{page}]({page})" if page else "[README](../README.md#其他)"
        lines.append(f"| {topic} | {counts.get(topic, 0)} | {target} |")
    lines.extend(["", "注：系列文章会统一归入该系列的众数主题；非系列文章仍可能属于多个主题。"])
    return "\n".join(lines).rstrip() + "\n"


def render_topic_page(topic: str, posts: list[dict[str, Any]]) -> str:
    topic_posts = posts_for_topic(posts, topic)
    lines = [
        f"# {topic}",
        "",
        "[返回主题索引](index.md)",
        "",
        f"共 {len(topic_posts)} 篇。",
        "",
    ]
    if not topic_posts:
        lines.append("暂无文章。")
        return "\n".join(lines).rstrip() + "\n"

    append_grouped_posts(lines, topic_posts, detailed=True, topic=topic)
    return "\n".join(lines).rstrip() + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_all(posts: list[dict[str, Any]]) -> None:
    write_text(ROOT / "README.md", render_readme(posts))
    write_text(DOCS_DIR / "index.md", render_docs_index(posts))
    for topic, filename in TOPIC_PAGES.items():
        write_text(DOCS_DIR / filename, render_topic_page(topic, posts))
    log(f"render_markdown: rendered README.md and {len(TOPIC_PAGES) + 1} docs pages")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render README and docs pages.")
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "posts_classified.json"),
        help="Input classified posts JSON path.",
    )
    args = parser.parse_args()

    posts = read_json(Path(args.input), [])
    render_all(posts)


if __name__ == "__main__":
    main()
