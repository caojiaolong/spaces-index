from scripts.classify import classify_post, classify_posts, detect_prefix_series_candidate, detect_series_info


def assert_topic(title, expected, tags=None, category="信息时代"):
    post = {
        "id": title,
        "title": title,
        "url": f"https://spaces.ac.cn/archives/{abs(hash(title))}",
        "date": "2025-01-01",
        "source_category": category,
        "source_tags": tags or [],
    }
    assert expected in classify_post(post)["topics"]


def test_classifies_transformer_keywords():
    assert_topic("Transformer升级之路：18、RoPE的底数选择原则", "大模型与Transformer")


def test_classifies_generative_keywords():
    assert_topic("生成扩散模型漫谈（二十九）：用DDPM来离散编码", "生成模型")


def test_classifies_optimization_keywords():
    assert_topic("AdamW的Weight RMS的渐近估计", "优化与训练", tags=["优化器", "梯度"])


def test_classifies_math_keywords():
    assert_topic("SVD分解(二)：为什么SVD意味着聚类？", "数学工具", tags=["矩阵"])


def test_classifies_nlp_keywords():
    assert_topic("GlobalPointer：用统一的方式处理NER和关系抽取", "NLP与信息抽取")


def test_classifies_engineering_keywords():
    assert_topic("Cool Papers更新：简单适配Zotero Connector", "工程工具")


def test_classifies_source_category_fallbacks():
    assert_topic("2020年全年天象", "天文科普", category="天文探索")
    assert_topic("关于e是无理数的证明", "数学工具", category="数学研究")


def test_detects_series_index():
    assert detect_series_info("MoE环游记：3、换个思路来分配") == ("MoE环游记", 3)
    assert detect_series_info("生成扩散模型漫谈（二十九）：用DDPM来离散编码") == ("生成扩散模型漫谈", 29)
    assert detect_series_info("“熵”不起：从熵、最大熵原理到最大熵模型（三）") == (
        "“熵”不起：从熵、最大熵原理到最大熵模型",
        3,
    )
    assert detect_series_info("高斯型积分的微扰展开（一）") == ("高斯型积分的微扰展开", 1)
    assert detect_series_info("费曼路径积分思想的发展(四)") == ("费曼路径积分思想的发展", 4)
    assert detect_series_info("《新理解矩阵4》：相似矩阵的那些事儿") == ("新理解矩阵", 4)
    assert detect_series_info("费曼积分法——积分符号内取微分(3)") == (
        "费曼积分法——积分符号内取微分",
        3,
    )
    assert detect_series_info("《自然极值》系列——4.费马点问题") == ("自然极值", 4)


def test_detects_shared_prefix_series_candidate():
    assert detect_prefix_series_candidate("“让Keras更酷一些！”：中间变量、权重滑动和安全生成器") == "“让Keras更酷一些！”"


def test_series_uses_majority_topic():
    posts = [
        {
            "id": "1",
            "title": "示例系列：1、Transformer和Attention",
            "url": "https://spaces.ac.cn/archives/1",
            "date": "2025-01-01",
            "source_category": "信息时代",
            "source_tags": ["attention"],
        },
        {
            "id": "2",
            "title": "示例系列：2、RoPE位置编码",
            "url": "https://spaces.ac.cn/archives/2",
            "date": "2025-01-02",
            "source_category": "信息时代",
            "source_tags": ["transformer"],
        },
        {
            "id": "3",
            "title": "示例系列：3、GAN生成模型",
            "url": "https://spaces.ac.cn/archives/3",
            "date": "2025-01-03",
            "source_category": "信息时代",
            "source_tags": ["GAN"],
        },
    ]
    results = classify_posts(posts)
    assert {post["series_topic"] for post in results} == {"大模型与Transformer"}
    assert {tuple(post["topics"]) for post in results} == {("大模型与Transformer",)}


def test_shared_prefix_series_are_grouped_when_repeated():
    posts = [
        {
            "id": str(index),
            "title": title,
            "url": f"https://spaces.ac.cn/archives/{index}",
            "date": f"2019-01-0{index}",
            "source_category": "信息时代",
            "source_tags": ["Keras"],
        }
        for index, title in enumerate(
            [
                "“让Keras更酷一些！”：小众的自定义优化器",
                "“让Keras更酷一些！”：随意的输出和灵活的归一化",
                "“让Keras更酷一些！”：分层的学习率和自由的梯度",
            ],
            start=1,
        )
    ]
    results = classify_posts(posts)
    assert {post["series"] for post in results} == {"“让Keras更酷一些！”"}


def test_override_has_highest_priority():
    post = {
        "id": "10352",
        "title": "无关标题",
        "url": "https://spaces.ac.cn/archives/10352",
        "date": "2024-09-06",
        "source_category": "信息时代",
        "source_tags": [],
    }
    result = classify_post(
        post,
        {
            "10352": {
                "topics": ["数学工具"],
                "series": "手动系列",
                "level": "advanced",
                "notes": "人工校正",
            }
        },
    )
    assert result["topics"] == ["数学工具"]
    assert result["series"] == "手动系列"
    assert result["series_index"] is None
    assert result["level"] == "advanced"
    assert result["notes"] == "人工校正"
