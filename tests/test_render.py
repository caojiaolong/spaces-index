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
    assert "- [大模型与Transformer（2 篇）](#topic-transformer)" in readme
    assert "  - [Transformer升级之路（2 篇）](#series-transformer-" in readme
    assert "### 大模型与Transformer" in readme
    assert "#### Transformer升级之路 [返回目录](#目录)" in readme
    assert "- 2024-05-29 - [Transformer升级之路：18、RoPE的底数选择原则](https://spaces.ac.cn/archives/1)" in readme
    assert (
        "- 2024-05-29 - [Transformer升级之路：18、RoPE的底数选择原则]"
        "(https://spaces.ac.cn/archives/1) - [查看系列](#series-transformer-"
        in readme
    )
    assert "## 详细元数据" in readme
    assert readme.index("## 本地运行") > readme.index("## 主题分类")
    assert readme.index("## 更新流程") > readme.index("## 本地运行")


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
