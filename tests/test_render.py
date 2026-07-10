from scripts.render_markdown import render_readme, render_topic_page


SAMPLE_POSTS = [
    {
        "id": "1",
        "title": "Transformer升级之路：18、RoPE的底数选择原则",
        "url": "https://spaces.ac.cn/archives/1",
        "date": "2024-05-29",
        "source_category": "信息时代",
        "source_tags": ["attention", "位置编码"],
        "topics": ["大模型与Transformer"],
        "series": "Transformer升级之路",
        "series_index": 18,
        "level": "advanced",
        "notes": "重点阅读",
        "source_summary": "这是一段小结短摘录。",
    },
    {
        "id": "2",
        "title": "AdamW的Weight RMS的渐近估计",
        "url": "https://spaces.ac.cn/archives/2",
        "date": "2025-11-17",
        "source_category": "数学研究",
        "source_tags": ["优化器", "梯度"],
        "topics": ["优化与训练", "数学工具"],
        "series": None,
        "series_index": None,
        "level": "advanced",
    },
    {
        "id": "3",
        "title": "Transformer升级之路：17、多模态位置编码的简单思考",
        "url": "https://spaces.ac.cn/archives/3",
        "date": "2024-04-01",
        "source_category": "信息时代",
        "source_tags": ["transformer"],
        "topics": ["大模型与Transformer"],
        "series": "Transformer升级之路",
        "series_index": 17,
        "level": None,
    },
]


def test_readme_contains_topic_links_and_stats():
    readme = render_readme(SAMPLE_POSTS)
    assert "Transformer升级之路：18、RoPE的底数选择原则" in readme
    assert "https://spaces.ac.cn/archives/1" in readme
    assert "2025-11-17" in readme
    assert "## 目录" in readme
    assert "展开完整主题、系列与非系列目录" in readme
    assert "assets/readme-hero.svg" in readme
    assert "https://caojiaolong.github.io/spaces-index/" in readme
    assert "非官方、持续更新、只保存元数据" in readme
    assert "## 最近更新" in readme
    assert "2026-07-10 · 交互式 GitHub Pages 体验升级" in readme
    assert "非系列文章以及已读/未读组合筛选" in readme
    assert "记录只保存在当前浏览器，不会上传" in readme
    assert readme.index("## 最近更新") < readme.index("## 为什么做这个索引")
    assert "## 最近 10 篇文章" in readme
    assert "| 3 篇 | 3 个 | 1 个 | 2025-11-17 |" in readme
    assert "## 三种浏览方式" in readme
    assert "uv run python -m http.server 8000 --directory _site" in readme
    assert "不要直接双击 `web/index.html`" in readme
    assert "- [大模型与Transformer（2 篇）](#topic-transformer)" in readme
    assert (
        "  - [Transformer升级之路（2 篇）]"
        "(https://caojiaolong.github.io/spaces-index/#/series/series-"
        in readme
    )
    assert "<details>" in readme
    assert "<summary><strong>大模型与Transformer</strong> · 2 篇</summary>" in readme
    assert readme.count("<details>") == readme.count("</details>")
    assert "#### Transformer升级之路 [返回目录](#目录)" in readme
    assert "- 2024-05-29 - [Transformer升级之路：18、RoPE的底数选择原则](https://spaces.ac.cn/archives/1)" in readme
    assert (
        "- 2024-05-29 - [Transformer升级之路：18、RoPE的底数选择原则]"
        "(https://spaces.ac.cn/archives/1) - [查看系列]"
        "(https://caojiaolong.github.io/spaces-index/#/series/series-"
        in readme
    )
    assert "## 详细元数据" in readme
    assert readme.index("## 本地运行") > readme.index("## 主题分类")
    assert readme.index("## 更新流程") > readme.index("## 本地运行")
    assert "## Star History" in readme
    assert "如果这个索引对你有帮助，欢迎 Star 支持，后续会通过 GitHub Actions 持续更新" in readme
    assert "api.star-history.com/chart?repos=caojiaolong/spaces-index" in readme
    assert "sealed_token=1PtTrJfjwB8TiUNumdr03-" in readme
    assert 'alt="Star History Chart"' in readme


def test_topic_page_contains_metadata():
    page = render_topic_page("大模型与Transformer", SAMPLE_POSTS)
    assert "# 大模型与Transformer" in page
    assert "2024-05-29 - [Transformer升级之路：18、RoPE的底数选择原则]" in page
    assert "原站分类：信息时代" in page
    assert "原站标签：attention、位置编码" in page
    assert "系列：Transformer升级之路 #18" in page
    assert "小结摘录：这是一段小结短摘录。" in page
    assert "备注：重点阅读" in page
    assert "[返回目录](#目录)" not in page
