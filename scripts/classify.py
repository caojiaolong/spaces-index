from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

try:
    from .common import DATA_DIR, clean_text, log, read_json, sort_posts_desc, write_json
except ImportError:  # pragma: no cover - used when running as python scripts/classify.py
    from common import DATA_DIR, clean_text, log, read_json, sort_posts_desc, write_json


TOPICS = [
    "深度学习基础",
    "词向量与Embedding",
    "大模型与Transformer",
    "生成模型",
    "优化与训练",
    "数学工具",
    "概率统计与信息论",
    "几何与方程",
    "NLP与信息抽取",
    "工程工具",
    "天文科普",
    "物理化学",
    "生物自然",
    "图片摄影",
    "科普问答与百科",
    "资源与站务",
    "阅读写作与随笔",
    "其他",
]

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "深度学习基础": [
        "神经网络",
        "深度学习",
        "激活函数",
        "relu",
        "gelu",
        "swish",
        "squareplus",
        "dropout",
        "baseline",
        "损失函数",
        "多任务学习",
        "自监督",
        "尺度定律",
        "scaling law",
        "keras",
        "tensorflow",
        "pytorch",
        "mixup",
        "正则",
        "数据挖掘",
        "建模",
    ],
    "词向量与Embedding": [
        "word2vec",
        "embedding",
        "词向量",
        "词嵌入",
        "句向量",
        "sentence",
        "cosent",
        "simbert",
        "向量召回",
    ],
    "大模型与Transformer": [
        "transformer",
        "attention",
        "rope",
        "位置编码",
        "长度外推",
        "mla",
        "gqa",
        "mqa",
        "llm",
        "语言模型",
        "大模型",
        "moe",
        "lora",
        "mup",
        "softmax",
        "decoder-only",
        "线性注意力",
        "多模态",
        "deltanet",
        "ssm",
        "hippo",
    ],
    "生成模型": [
        "扩散",
        "ddpm",
        "flow",
        "gan",
        "vq",
        "生成模型",
        "vae",
        "一致性模型",
        "蒸馏",
        "ode采样",
        "文本生成",
    ],
    "优化与训练": [
        "adam",
        "adamw",
        "sgd",
        "muon",
        "lion",
        "tiger",
        "amos",
        "优化器",
        "优化",
        "学习率",
        "权重衰减",
        "梯度",
        "batch size",
        "dropout",
        "训练",
        "收敛",
        "hessian",
        "裁剪",
    ],
    "数学工具": [
        "矩阵",
        "svd",
        "积分",
        "不等式",
        "傅里叶",
        "随机",
        "低秩",
        "范数",
        "正交",
        "谱",
        "数论",
        "极值",
        "级数",
        "函数",
        "素数",
        "复数",
        "向量",
        "竞赛",
        "近似",
        "数学",
    ],
    "概率统计与信息论": [
        "概率",
        "统计",
        "分布",
        "熵",
        "最大熵",
        "最小熵",
        "信息论",
        "贝叶斯",
        "随机",
        "期望",
        "方差",
        "中位数",
        "median",
    ],
    "几何与方程": [
        "几何",
        "黎曼",
        "流形",
        "微分方程",
        "偏微分",
        "方程",
        "曲线",
        "曲面",
        "测地线",
        "外微分",
        "路径积分",
        "特征线",
        "旋转",
    ],
    "NLP与信息抽取": [
        "globalpointer",
        "ner",
        "crf",
        "关系抽取",
        "实体识别",
        "信息抽取",
        "分词",
        "nlp",
        "bert",
        "word2vec",
        "embedding",
        "词向量",
        "文本分类",
        "tokenizer",
        "seq2seq",
        "ocr",
        "clue",
        "kgclue",
        "相似度",
    ],
    "工程工具": [
        "python",
        "rss",
        "搜索",
        "mathjax",
        "cool papers",
        "github",
        "zotero",
        "chrome",
        "浏览器扩展",
        "站内检索",
        "树莓派",
        "网站",
        "智能家居",
        "爬虫",
        "代码",
        "工具",
        "arxiv",
        "linux",
        "openwrt",
        "wifi",
        "校园网",
        "误删",
    ],
    "天文科普": [
        "天文",
        "天象",
        "流星",
        "日食",
        "月食",
        "星云",
        "星系",
        "恒星",
        "行星",
        "月球",
        "火星",
        "木星",
        "望远镜",
        "nasa",
        "apod",
    ],
    "物理化学": [
        "物理",
        "化学",
        "力学",
        "相对论",
        "引力",
        "作用量",
        "光学",
        "磁",
        "量子",
        "热力学",
        "酸",
        "溶液",
        "分子",
        "原子",
    ],
    "生物自然": [
        "生物",
        "自然",
        "植物",
        "动物",
        "生态",
        "演化",
        "基因",
        "病毒",
        "细胞",
        "鱼",
    ],
    "图片摄影": [
        "图片摄影",
        "摄影",
        "照片",
        "相机",
        "镜头",
        "彩虹",
        "拍摄",
        "图像",
        "每日一图",
    ],
    "科普问答与百科": [
        "问题百科",
        "千奇百怪",
        "为什么",
        "是什么",
        "百科",
        "科普",
        "诺贝尔奖",
        "趣闻",
        "漫话模型",
    ],
    "资源与站务": [
        "资源共享",
        "资源",
        "书籍",
        "下载",
        "转载",
        "站务",
        "第1000篇",
        "科学空间",
        "域名",
        "访问量",
        "faq",
    ],
    "阅读写作与随笔": [
        "生活/情感",
        "生活",
        "情感",
        "随笔",
        "杂记",
        "读书",
        "作文",
        "节日",
        "春节",
        "中秋",
        "电影",
        "游记",
    ],
}

SOURCE_CATEGORY_TOPICS: dict[str, list[str]] = {
    "数学研究": ["数学工具"],
    "天文探索": ["天文科普"],
    "物理化学": ["物理化学"],
    "生物自然": ["生物自然"],
    "图片摄影": ["图片摄影"],
    "问题百科": ["科普问答与百科"],
    "千奇百怪": ["科普问答与百科"],
    "资源共享": ["资源与站务"],
    "生活/情感": ["阅读写作与随笔"],
}

BEGINNER_KEYWORDS = ["入门", "基础", "简介", "浅谈", "快速上手", "初探"]
ADVANCED_KEYWORDS = [
    "证明",
    "推导",
    "渐近",
    "谱范数",
    "微分方程",
    "不等式",
    "高阶",
    "定理",
]


def combined_text(post: dict[str, Any]) -> str:
    parts = [
        str(post.get("title") or ""),
        str(post.get("source_category") or ""),
        " ".join(str(tag) for tag in post.get("source_tags") or []),
    ]
    return clean_text(" ".join(parts)).casefold()


def match_topics(post: dict[str, Any]) -> list[str]:
    text = combined_text(post)
    topics = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(keyword.casefold() in text for keyword in keywords)
    ]
    for topic in SOURCE_CATEGORY_TOPICS.get(str(post.get("source_category") or ""), []):
        if topic not in topics:
            topics.append(topic)
    return topics or ["其他"]


CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {"十": 10, "百": 100}
ORDER_WORDS = {"上": 1, "中": 2, "下": 3}


def parse_series_index(value: str | None) -> int | None:
    if not value:
        return None
    value = clean_text(value)
    if value.isdigit():
        return int(value)
    if value in ORDER_WORDS:
        return ORDER_WORDS[value]

    total = 0
    number = 0
    matched = False
    for char in value:
        if char in CHINESE_DIGITS:
            number = CHINESE_DIGITS[char]
            matched = True
        elif char in CHINESE_UNITS:
            unit = CHINESE_UNITS[char]
            total += (number or 1) * unit
            number = 0
            matched = True
    if matched:
        return total + number
    return None


def detect_series_info(title: str) -> tuple[str | None, int | None]:
    patterns = [
        r"^《(?P<series>[^》]*?)(?P<index>\d+)》\s*[:：]",
        r"^《(?P<series>[^》]+)》系列\s*[—\-－-]+\s*(?P<index>[一二三四五六七八九十百零〇两\d]+)\s*[、.．]",
        r"^【(?P<series>[^】]+)】\s*(?P<index>[一二三四五六七八九十百零〇两\d]+)",
        r"^(?P<series>.+?)[（(](?P<index>[一二三四五六七八九十百零〇两\d]+)[）)]\s*[:：]",
        r"^(?P<series>.+?)[:：]\s*(?P<index>[一二三四五六七八九十百零〇两\d]+)\s*[、.．]",
        r"^(?P<series>.+?)[（(](?P<index>[上中下])[）)]",
        r"^(?P<series>.+?)[（(](?P<index>[一二三四五六七八九十百零〇两\d]+)[）)]$",
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            series = clean_text(match.group("series"))
            index = parse_series_index(match.group("index"))
            return series, index
    return None, None


def detect_series(title: str) -> str | None:
    series, _ = detect_series_info(title)
    return series


def detect_series_index(title: str) -> int | None:
    _, index = detect_series_info(title)
    return index


def detect_prefix_series_candidate(title: str) -> str | None:
    patterns = [
        r"^(?P<series>[“\"'][^：:]{3,80}?[”\"'！!？?])[:：]",
        r"^(?P<series>《[^》]{3,80}》)[:：]",
        r"^(?P<series>【[^】]{3,80}】)[:：]",
    ]
    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            return clean_text(match.group("series"))
    return None


def detect_level(post: dict[str, Any]) -> str | None:
    text = combined_text(post)
    if any(keyword.casefold() in text for keyword in BEGINNER_KEYWORDS):
        return "beginner"
    if any(keyword.casefold() in text for keyword in ADVANCED_KEYWORDS):
        return "advanced"
    return None


def normalize_topics(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    raise ValueError(f"topics override must be a string or list, got {type(value).__name__}")


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    if "overrides" in data and isinstance(data["overrides"], dict):
        data = data["overrides"]
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Override for {key!r} must be a mapping.")
        normalized[str(key)] = value
    return normalized


def find_override(post: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [str(post.get("id") or ""), str(post.get("url") or "")]
    for key in candidates:
        if key and key in overrides:
            return overrides[key]
    return {}


def classify_post(
    post: dict[str, Any],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = dict(post)
    result["topics"] = match_topics(post)
    title = str(post.get("title") or "")
    result["series"] = detect_series(title)
    result["series_index"] = detect_series_index(title)
    result["level"] = detect_level(post)

    override = find_override(post, overrides or {})
    if "topics" in override:
        result["topics"] = normalize_topics(override["topics"]) or ["其他"]
    for field in ("series", "series_index", "series_topic", "level", "notes"):
        if field in override:
            result[field] = override[field]
    return result


def topic_rank(topic: str) -> int:
    try:
        return TOPICS.index(topic)
    except ValueError:
        return len(TOPICS)


def choose_series_topic(posts: list[dict[str, Any]]) -> str:
    forced_topics = [
        str(post.get("series_topic"))
        for post in posts
        if post.get("series_topic")
    ]
    if forced_topics:
        return sorted(forced_topics, key=topic_rank)[0]

    counts: Counter[str] = Counter()
    for post in posts:
        for topic in post.get("topics") or ["其他"]:
            topic = str(topic)
            if topic != "其他":
                counts[topic] += 1
    if not counts:
        return "其他"
    return sorted(counts, key=lambda topic: (-counts[topic], topic_rank(topic), topic))[0]


def apply_series_topic_majority(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in posts:
        series = clean_text(str(post.get("series") or ""))
        if series:
            by_series[series].append(post)

    for series_posts in by_series.values():
        if len(series_posts) < 2:
            continue
        series_topic = choose_series_topic(series_posts)
        for post in series_posts:
            post["series_topic"] = series_topic
            post["topics"] = [series_topic]
    return posts


def apply_prefix_series_detection(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in posts:
        if post.get("series"):
            continue
        candidate = detect_prefix_series_candidate(str(post.get("title") or ""))
        if candidate:
            candidates[candidate].append(post)

    for series, series_posts in candidates.items():
        if len(series_posts) < 3:
            continue
        for post in series_posts:
            post["series"] = series
            post["series_index"] = None
    return posts


def classify_posts(
    posts: list[dict[str, Any]],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    classified = [classify_post(post, overrides) for post in posts]
    classified = apply_prefix_series_detection(classified)
    return sort_posts_desc(apply_series_topic_majority(classified))


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify posts into research topics.")
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "posts.json"),
        help="Input enriched posts JSON path.",
    )
    parser.add_argument(
        "--overrides",
        default=str(DATA_DIR / "overrides.yaml"),
        help="Manual overrides YAML path.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "posts_classified.json"),
        help="Output classified posts JSON path.",
    )
    args = parser.parse_args()

    posts = read_json(Path(args.input), [])
    if not posts:
        raise RuntimeError(f"No enriched posts found in {args.input}. Run enrich_posts.py first.")
    overrides = load_overrides(Path(args.overrides))
    classified = classify_posts(posts, overrides)
    write_json(Path(args.output), classified)
    log(f"classify: classified {len(classified)} posts -> {args.output}")


if __name__ == "__main__":
    main()
