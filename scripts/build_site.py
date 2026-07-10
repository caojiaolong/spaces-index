from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "posts_classified.json"
DEFAULT_OUTPUT = ROOT / "_site"
WEB_DIR = ROOT / "web"
SCHEMA_VERSION = 1
GENERATED_SITE_MARKER = ".spaces-index-generated-site"

# This is also the display order used by the generated catalogue. Unknown topics
# are retained and appended alphabetically so that new classifier output is never
# silently dropped.
TOPIC_GROUP_DEFINITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "AI 与机器学习",
        (
            "深度学习基础",
            "词向量与Embedding",
            "大模型与Transformer",
            "生成模型",
            "优化与训练",
            "NLP与信息抽取",
        ),
    ),
    (
        "数学与基础科学",
        (
            "数学工具",
            "概率统计与信息论",
            "几何与方程",
            "天文科普",
            "物理化学",
            "生物自然",
        ),
    ),
    (
        "工具、科普与人文",
        (
            "工程工具",
            "图片摄影",
            "科普问答与百科",
            "资源与站务",
            "阅读写作与随笔",
            "其他",
        ),
    ),
)

KNOWN_TOPICS: tuple[str, ...] = tuple(
    topic for _, group_topics in TOPIC_GROUP_DEFINITIONS for topic in group_topics
)
KNOWN_TOPIC_RANK = {topic: rank for rank, topic in enumerate(KNOWN_TOPICS)}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


def _optional_text(value: Any) -> str | None:
    cleaned = _clean_text(value)
    return cleaned or None


def _optional_summary(value: Any) -> str | None:
    cleaned = _optional_text(value)
    if cleaned is None:
        return None
    normalized = unicodedata.normalize("NFKC", cleaned).casefold()
    return None if normalized in {"null", "none", "undefined", "nan"} else cleaned


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    values: Iterable[Any]
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)

    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = _clean_text(item)
        if cleaned and cleaned not in seen:
            result.append(cleaned)
            seen.add(cleaned)
    return result


def safe_https_url(value: Any) -> str | None:
    """Return a usable HTTPS URL, or ``None`` for unsafe/invalid input."""

    if not isinstance(value, str):
        return None
    url = value.strip()
    if not url or any(character.isspace() for character in url) or "\\" in url:
        return None

    try:
        parsed = urlsplit(url)
        # Accessing port performs range and syntax validation.
        _ = parsed.port
    except ValueError:
        return None

    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    try:
        parsed.hostname.encode("idna")
    except UnicodeError:
        return None
    return url


def stable_series_id(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", _clean_text(name)).casefold()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"series-{digest}"


def _stable_topic_slug(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"topic-{digest}"


def _series_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    return None


def _fallback_post_id(post: dict[str, Any]) -> str:
    basis = "||".join(
        (
            _clean_text(post.get("url")),
            _clean_text(post.get("date")),
            _clean_text(post.get("title")),
        )
    )
    return "post-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _project_post(post: dict[str, Any]) -> dict[str, Any]:
    topics = _string_list(post.get("topics")) or ["其他"]
    series = _optional_text(post.get("series"))
    post_id = _clean_text(post.get("id")) or _fallback_post_id(post)

    # Keep this explicit: data added to the crawler later must not accidentally
    # expose article bodies or other fields in the public Pages artifact.
    return {
        "id": post_id,
        "title": _clean_text(post.get("title")),
        "url": safe_https_url(post.get("url")),
        "date": _clean_text(post.get("date")),
        "sourceCategory": _optional_text(post.get("source_category")),
        "sourceTags": _string_list(post.get("source_tags")),
        "sourceSummary": _optional_summary(post.get("source_summary")),
        "topics": topics,
        "series": series,
        "seriesId": None,
        "seriesIndex": _series_index(post.get("series_index")),
        "level": _optional_text(post.get("level")),
        "seriesTopic": _optional_text(post.get("series_topic")),
        "notes": _optional_text(post.get("notes")),
    }


def _post_sort_key(post: dict[str, Any]) -> tuple[str, int, int, str, str]:
    post_id = post["id"]
    numeric = post_id.isdigit()
    return (
        post["date"],
        int(numeric),
        int(post_id) if numeric else -1,
        post_id,
        post["title"],
    )


def _topic_sort_key(topic: str) -> tuple[int, int | str]:
    if topic in KNOWN_TOPIC_RANK:
        return (0, KNOWN_TOPIC_RANK[topic])
    return (1, topic)


def _choose_series_topic(posts: list[dict[str, Any]]) -> str:
    explicit = Counter(post["seriesTopic"] for post in posts if post["seriesTopic"])
    candidates = explicit or Counter(topic for post in posts for topic in post["topics"])
    if not candidates:
        return "其他"
    return min(candidates, key=lambda topic: (-candidates[topic], _topic_sort_key(topic)))


def _ordered_series_posts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        posts,
        key=lambda post: (
            0 if post["seriesIndex"] is not None else 1,
            post["seriesIndex"] if post["seriesIndex"] is not None else 0,
            not bool(post["date"]),
            post["date"],
            post["id"],
            post["title"],
        ),
    )


def _build_series(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for post in posts:
        if post["series"]:
            groups[post["series"]].append(post)

    result: list[dict[str, Any]] = []
    ids_to_names: dict[str, str] = {}
    for name, members in groups.items():
        if len(members) < 2:
            continue
        series_id = stable_series_id(name)
        if series_id in ids_to_names and ids_to_names[series_id] != name:
            raise ValueError(f"Series id collision: {ids_to_names[series_id]!r} and {name!r}")
        ids_to_names[series_id] = name
        ordered_members = _ordered_series_posts(members)
        dates = [post["date"] for post in members if post["date"]]
        for post in members:
            post["seriesId"] = series_id
        result.append(
            {
                "id": series_id,
                "name": name,
                "topic": _choose_series_topic(members),
                "count": len(members),
                "startDate": min(dates) if dates else "",
                "endDate": max(dates) if dates else "",
                "postIds": [post["id"] for post in ordered_members],
            }
        )

    # Stable sorting keeps names ascending when end dates tie.
    result.sort(key=lambda series: series["name"])
    result.sort(key=lambda series: series["endDate"], reverse=True)
    return result


def _build_topics(
    posts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts = Counter(topic for post in posts for topic in post["topics"])
    names = sorted(counts, key=_topic_sort_key)
    topics = [
        {"name": name, "count": counts[name], "slug": _stable_topic_slug(name)}
        for name in names
    ]

    topic_groups: list[dict[str, Any]] = []
    known = set(KNOWN_TOPICS)
    for group_name, group_topics in TOPIC_GROUP_DEFINITIONS:
        present = [topic for topic in group_topics if topic in counts]
        if present:
            topic_groups.append({"name": group_name, "topics": present})
    unknown = [topic for topic in names if topic not in known]
    if unknown:
        topic_groups.append({"name": "其他主题", "topics": unknown})
    return topics, topic_groups


def build_catalog(raw_posts: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(raw_posts, list) or any(not isinstance(post, dict) for post in raw_posts):
        raise ValueError("The classified posts JSON must be an array of objects")

    posts = [_project_post(post) for post in raw_posts]
    post_ids = [post["id"] for post in posts]
    if len(post_ids) != len(set(post_ids)):
        duplicates = sorted(post_id for post_id, count in Counter(post_ids).items() if count > 1)
        raise ValueError(f"Post ids must be unique; duplicates: {', '.join(duplicates)}")

    series = _build_series(posts)
    topics, topic_groups = _build_topics(posts)
    posts.sort(key=_post_sort_key, reverse=True)
    latest_date = max((post["date"] for post in posts if post["date"]), default="")

    return {
        "schemaVersion": SCHEMA_VERSION,
        "stats": {
            "postCount": len(posts),
            "topicCount": len(topics),
            "seriesCount": len(series),
            "latestDate": latest_date,
        },
        "topicGroups": topic_groups,
        "topics": topics,
        "series": series,
        "posts": posts,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_site(
    input_path: Path,
    output_dir: Path,
    *,
    web_dir: Path = WEB_DIR,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    web_dir = Path(web_dir)
    if not web_dir.is_dir():
        raise FileNotFoundError(f"Web source directory does not exist: {web_dir}")

    resolved_input = input_path.resolve()
    resolved_output = output_dir.resolve()
    resolved_web = web_dir.resolve()
    resolved_root = ROOT.resolve()
    if resolved_output == resolved_root or not _is_relative_to(resolved_output, resolved_root):
        raise ValueError("Output directory must be a child of the repository root")
    if resolved_output == resolved_web or _is_relative_to(resolved_output, resolved_web):
        raise ValueError("Output directory must not be the web source or one of its children")
    if _is_relative_to(resolved_web, resolved_output) or _is_relative_to(resolved_input, resolved_output):
        raise ValueError("Output directory must not contain the input data or web source")

    with input_path.open("r", encoding="utf-8") as handle:
        raw_posts = json.load(handle)
    catalog = build_catalog(raw_posts)

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
        is_default_output = resolved_output == DEFAULT_OUTPUT.resolve()
        has_generated_marker = (output_dir / GENERATED_SITE_MARKER).is_file()
        if not is_default_output and not has_generated_marker:
            raise ValueError(
                "Refusing to replace an existing directory that is not the default "
                "_site output and has no generated-site marker"
            )
        shutil.rmtree(output_dir)
    shutil.copytree(web_dir, output_dir)

    catalog_path = output_dir / "catalog.json"
    with catalog_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(catalog, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (output_dir / GENERATED_SITE_MARKER).write_text(
        "Generated by scripts/build_site.py; safe to replace.\n",
        encoding="utf-8",
        newline="\n",
    )
    return catalog


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the static GitHub Pages site and metadata-only catalogue."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    catalog = build_site(args.input, args.output)
    stats = catalog["stats"]
    print(
        f"Built {args.output} with {stats['postCount']} posts, "
        f"{stats['topicCount']} topics, and {stats['seriesCount']} series."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
