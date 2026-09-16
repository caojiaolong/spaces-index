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
    "Mk-_SQRXMESG92vxsl-rENs6bybrSJyqOFx-fGmzdxBlQiWKvWPdKR03AtHLO5zQOgO8u"
    "Jyaj8qJo62vL2-UXF4YyFBdPhpo_wXXEy_FFTqXcBBkPeqaJQ"
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
        '  <img src="assets/readme-hero.svg" alt="科学空间文章索引：由曲面线稿与知识主题组成的视觉横幅" width="100%">',
        "</p>",
        "",
        '<h1 align="center">科学空间文章索引</h1>',
        "",
        '<p align="center"><strong>找到一篇好文章，读懂一条思路。</strong></p>',
        '<p align="center">',
        f'  <a href="{PAGES_URL}"><strong>开始阅读</strong></a>',
        "  ·",
        f'  <a href="{PAGES_URL}#/explore">查找文章</a>',
        "  ·",
        f'  <a href="{PAGES_URL}#/series">系列书架</a>',
        "  ·",
        f'  <a href="{SOURCE_URL}">访问科学空间</a>',
        "  ·",
        '  <a href="#本地运行">本地运行</a>',
        "</p>",
        '<p align="center"><sub>苏剑林「科学空间」的非官方、非商业知识库 · 主题检索 · 系列阅读 · AI 友好的原文 Markdown</sub></p>',
        "",
        "## 为什么做这个索引",
        "",
        "读科学空间时，你或许也有过这样的时刻：记得看过一种推导，却想不起文章标题；从搜索结果点进系列的中间一篇，想把前后的思路串起来；读懂了一个公式，希望连同上下文一起存进自己的笔记。",
        "",
        "**spaces-index 希望把找文章的时间，留给理解文章。** 它将苏剑林老师在科学空间的文章整理为持续更新的主题与系列索引，并提供忠于原文的 Markdown。从 Transformer、扩散模型、优化算法到数学、自然科学和随笔，你可以带着问题来查，也可以沿着一个系列慢慢读。",
        "",
        "另一个核心目的，是做 **AI 友好的原文转换**：把文章的正文、公式和代码保存在结构清晰的 Markdown 中。遇到读不懂的地方，一键复制原文到你选择的 AI，带着完整上下文提问，把精力放在理解推导上。",
        "",
        "| 收录文章 | 知识主题 | 文章系列 | 最新文章 |",
        "| ---: | ---: | ---: | :---: |",
        f"| {len(posts)} 篇 | {sum(1 for topic in TOPICS if posts_for_topic(posts, topic))} 个 | {len(group_series(posts)[0])} 个 | {latest_post_date(posts)} |",
        "",
        "## 从哪里开始",
        "",
        "| 你想做什么 | 从这里开始 |",
        "| --- | --- |",
        f"| 找回一篇文章 | [文章检索]({PAGES_URL}#/explore)：搜索标题、标签、系列与小结，再按主题、年份、难度或已读状态缩小范围。 |",
        f"| 围绕一个方向读下去 | [主题漫游]({PAGES_URL}#/topics)：浏览相关主题，发现它们共同收录的文章。 |",
        f"| 从头读完一个系列 | [系列书架]({PAGES_URL}#/series)：按章节顺序阅读，查看进度，找到下一篇未读文章。 |",
        "",
        f"第一次来？可以从 [RoPE]({PAGES_URL}#/explore?q=RoPE)、[DDPM]({PAGES_URL}#/explore?q=DDPM) 或 [Muon]({PAGES_URL}#/explore?q=Muon) 开始。也可以直接翻阅下方的[完整目录](#目录)。",
        "",
        "## 读不懂的地方，带着原文问 AI",
        "",
        "- **专心读推导。** 点击文章标题进入站内阅读，用章节目录定位内容；悬停公式编号预览公式，点击平滑跳转，看完再点“返回引用处”，接着刚才的思路读。",
        "- **一键复制，带着上下文提问。** 点击“复制 Markdown”，将包含正文、原始 LaTeX、代码与出处的原文粘贴到 AI 对话中，直接问某一步推导、某个假设或一段代码。也可以下载成文件，留在自己的笔记中。",
        "- **按自己的节奏读。** 桌面和手机共用主题、字号与阅读工具，系列可切换上下篇；自动保存阅读百分比，达到 100% 才计入已读。阅读进度只保存在当前浏览器，不会上传。",
        "",
        "遇到没跟上的推导，可以复制 Markdown，连同一个具体问题交给你选择的 AI。例如：",
        "",
        "> 请解释文中从公式 (1) 到公式 (2) 的推导，逐步说明所用假设和中间步骤。请区分作者原文与你补充的说明，原文没有交代的地方请明确指出。",
        "",
        "Markdown 正文只做必要的格式转换，不摘要、润色、翻译或重组。构建时重新核对正文、公式、代码、链接和图片位置，校验失败的内容不提供下载；每篇保留苏剑林署名、原文链接与 **CC BY-NC-ND 2.5 CN** 许可说明。署名与附加许可不代表作者提供了额外授权，详见 [正文校验与许可](docs/mirror.md)。",
        "",
        "## 后续想做的事",
        "",
        "在可靠的正文库之上，继续探索跨文章正文检索、带出处的知识问答，以及从文章中梳理分析和解决问题的方法。这些功能尚未实现；未来的系统解读会与原文区分，不冒充作者，不把推断当作作者观点。",
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
            "uv sync --locked",
            "# 增量更新正文和索引、校验、构建并预览（http://127.0.0.1:8765/）",
            "uv run python scripts/update_all.py --serve",
            "# 前端开发：使用已保存正文离线修复和重建",
            "uv run python scripts/update_all.py --offline --serve",
            "# 构建全站部署产物，包含全部通过校验的正文；不执行部署",
            "uv run python scripts/build_site.py",
            "```",
            "",
            "> 本地预览必须通过 HTTP 服务访问；不要直接双击 `web/index.html` 或 `_site/index.html`。",
            "",
            "## 更新流程",
            "",
            "- `extract_articles.py`：串行发现、抓取和复查文章，同一次响应提取正文、分类、标签与短小结。",
            "- `fetch_archive.py` / `enrich_posts.py`：提供归档与元数据解析函数；无需分别运行，旧元数据命令转入统一更新流程。",
            "- `classify.py`：根据标题、分类、标签做规则分类，识别系列名与序号，并用系列成员主题众数统一系列主题。",
            "- `render_markdown.py`：稳定生成折叠式 README 和 docs 主题页。",
            "- `update_all.py`：一个命令完成归档、正文增量更新、失败恢复、元数据、分类、索引和网站构建；默认构建 `build/preview/`。",
            "- `build_site.py`：独立构建网站；构建完成并校验通过后才替换旧产物。",
            "- Markdown 阅读、校验报告、下架与授权边界见 [docs/mirror.md](docs/mirror.md)。",
            "- `data/articles/`：持久化正文、原始正文快照和逐篇校验报告；`config/mirror.json` 控制更新周期、下架和过期检查。",
            "- 日常命令：`uv run python scripts/update_all.py --serve`；离线更新：`uv run python scripts/update_all.py --offline --serve`。详见 [维护说明](docs/maintenance.md)。",
            "- GitHub Actions：定时更新索引，并将同一次运行生成的静态产物部署到 GitHub Pages。",
            "",
            "## 最近更新",
            "",
            "> **2026-09-15 · 从找到文章，到读懂与保存**",
            "",
            "- 文章标题直达站内 Markdown 阅读，卡片下方与阅读页顶部提供原文、下载和复制入口。",
            "- 公式引用支持平滑跳转与逐次返回；复制含公式的文字时保留原始 LaTeX。",
            "- 一个命令完成增量更新、失败修复、元数据同步、分类和建站；正文持久化于 `data/articles/`。默认提供全站通过校验且未下架、未过期的正文。",
            "",
            "<details>",
            "<summary>2026-09-07 · 现代学术画廊与阅读体验升级</summary>",
            "",
            "- 曲面主视觉、系列书架、主题交汇与手机阅读布局。",
            "- 支持主题、标签、年份、难度、系列文章、非系列文章以及已读/未读组合筛选；筛选条件可以随 URL 分享。",
            "- 小结原地展开，本地保存阅读进度。",
            "",
            "</details>",
            "",
            "## Star History",
            "",
            "如果这里帮你找回了一篇文章，或读完了一个系列，欢迎 Star 留作下次阅读的入口。发现漏篇、分类不准、系列顺序或公式显示问题，也欢迎提交 Issue，并附上文章链接与具体位置。",
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
    encoded = content.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


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
