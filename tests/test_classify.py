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


def test_optimizer_focused_posts_do_not_get_broad_math_topic():
    muon = classify_post(
        {
            "id": "10592",
            "title": "Muon优化器赏析：从向量到矩阵的本质跨越",
            "url": "https://spaces.ac.cn/archives/10592",
            "date": "2024-12-10",
            "source_category": "数学研究",
            "source_tags": ["矩阵", "梯度", "优化器", "谱范数", "muon"],
        }
    )
    hessian = classify_post(
        {
            "id": "10588",
            "title": "从Hessian近似看自适应学习率优化器",
            "url": "https://spaces.ac.cn/archives/10588",
            "date": "2024-11-29",
            "source_category": "数学研究",
            "source_tags": ["优化", "梯度", "学习率", "优化器"],
        }
    )
    assert muon["topics"] == ["优化与训练"]
    assert hessian["topics"] == ["优化与训练"]


def test_classifies_math_keywords():
    assert_topic("SVD分解(二)：为什么SVD意味着聚类？", "数学工具", tags=["矩阵"])


def test_pure_math_convergence_is_not_training_optimization():
    post = classify_post(
        {
            "id": "3194",
            "title": "勒贝格(Lebesgue)控制收敛定理",
            "url": "https://spaces.ac.cn/archives/3194",
            "date": "2015-01-16",
            "source_category": "数学研究",
            "source_tags": ["不等式", "积分"],
        }
    )
    assert "数学工具" in post["topics"]
    assert "优化与训练" not in post["topics"]


def test_dynamics_optimization_series_stays_in_optimization():
    posts = [
        {
            "id": "5655",
            "title": "从动力学角度看优化算法（一）：从SGD到动量加速",
            "url": "https://spaces.ac.cn/archives/5655",
            "date": "2018-06-27",
            "source_category": "数学研究",
            "source_tags": ["微分方程", "动力学", "数值计算", "优化器"],
        },
        {
            "id": "6234",
            "title": "从动力学角度看优化算法（二）：自适应学习率算法",
            "url": "https://spaces.ac.cn/archives/6234",
            "date": "2018-12-20",
            "source_category": "数学研究",
            "source_tags": ["微分方程", "动力学", "数值计算", "优化器"],
        },
    ]
    results = classify_posts(posts)
    assert {post["series_topic"] for post in results} == {"优化与训练"}
    assert {tuple(post["topics"]) for post in results} == {("优化与训练",)}


def test_vq_quantization_is_not_physics_quantum():
    post = classify_post(
        {
            "id": "9862",
            "title": "我在Performer中发现了Transformer-VQ的踪迹",
            "url": "https://spaces.ac.cn/archives/9862",
            "date": "2023-11-29",
            "source_category": "信息时代",
            "source_tags": ["量子化", "语言模型", "attention"],
        }
    )
    assert "大模型与Transformer" in post["topics"]
    assert "物理化学" not in post["topics"]


def test_physics_momentum_is_not_optimizer_momentum():
    post = classify_post(
        {
            "id": "1397",
            "title": "弹簧双体运动",
            "url": "https://spaces.ac.cn/archives/1397",
            "date": "2011-07-10",
            "source_category": "物理化学",
            "source_tags": ["弹性", "能量", "动量"],
        }
    )
    assert post["topics"] == ["物理化学"]


def test_natural_language_is_not_biology_nature():
    embedding = classify_post(
        {
            "id": "4122",
            "title": "词向量与Embedding究竟是怎么回事？",
            "url": "https://spaces.ac.cn/archives/4122",
            "date": "2016-12-03",
            "source_category": "信息时代",
            "source_tags": ["深度学习", "自然语言处理"],
        }
    )
    roformer = classify_post(
        {
            "id": "8998",
            "title": "RoFormerV2：自然语言理解的极限探索",
            "url": "https://spaces.ac.cn/archives/8998",
            "date": "2022-03-21",
            "source_category": "信息时代",
            "source_tags": ["语言模型", "预训练"],
        }
    )
    assert "词向量与Embedding" in embedding["topics"]
    assert "数学工具" not in embedding["topics"]
    assert "生物自然" not in embedding["topics"]
    assert roformer["topics"] == ["大模型与Transformer"]


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
    assert detect_series_info("【搜出来的文本】⋅（四）通过增、删、改来用词造句") == (
        "搜出来的文本",
        4,
    )


def test_detects_shared_prefix_series_candidate():
    assert detect_prefix_series_candidate("“让Keras更酷一些！”：中间变量、权重滑动和安全生成器") == "“让Keras更酷一些！”"
    assert detect_prefix_series_candidate("细水长flow之NICE：流模型的基本概念与实现") == "细水长flow"


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


def test_zhi_prefix_series_are_grouped_when_repeated():
    posts = [
        {
            "id": str(index),
            "title": title,
            "url": f"https://spaces.ac.cn/archives/{index}",
            "date": f"2018-08-0{index}",
            "source_category": "信息时代",
            "source_tags": ["生成模型", "flow"],
        }
        for index, title in enumerate(
            [
                "细水长flow之NICE：流模型的基本概念与实现",
                "细水长flow之RealNVP与Glow：流模型的传承与升华",
                "细水长flow之f-VAEs：Glow与VAEs的联姻",
            ],
            start=1,
        )
    ]
    results = classify_posts(posts)
    assert {post["series"] for post in results} == {"细水长flow"}
    assert {post["series_topic"] for post in results} == {"生成模型"}


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
