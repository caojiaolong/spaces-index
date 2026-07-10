from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from scripts.build_site import (
    GENERATED_SITE_MARKER,
    ROOT,
    build_catalog,
    build_site,
    stable_series_id,
)


@contextmanager
def writable_test_directory() -> Iterator[Path]:
    """Avoid Windows pytest temp directories whose restrictive ACLs break CI sandboxes."""

    path = Path(__file__).parent / f".tmp-build-site-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


SAMPLE_POSTS = [
    {
        "id": "20",
        "title": "第二章",
        "url": "https://spaces.ac.cn/archives/20",
        "date": "2024-01-01",
        "source_category": "信息时代",
        "source_tags": ["Transformer", "位置编码"],
        "source_summary": "明确的小结短摘录。",
        "topics": ["大模型与Transformer", "数学工具"],
        "series": "示例系列",
        "series_index": 2,
        "series_topic": "大模型与Transformer",
        "level": "advanced",
        "notes": "重点",
        "body": "这一字段模拟正文，绝不能进入构建产物。",
    },
    {
        "id": "10",
        "title": "第一章",
        "url": "https://spaces.ac.cn/archives/10",
        "date": "2024-06-01",
        "source_category": "信息时代",
        "source_tags": ["Transformer"],
        "topics": ["大模型与Transformer"],
        "series": "示例系列",
        "series_index": 1,
        "series_topic": "大模型与Transformer",
    },
    {
        "id": "30",
        "title": "没有小结的独立文章",
        "url": "javascript:alert(1)",
        "date": "2025-02-03",
        "source_category": "数学研究",
        "source_tags": [],
        "topics": ["数学工具"],
        "series": "只有一篇的系列",
        "series_index": None,
    },
]


def test_catalog_projects_only_metadata_and_computes_stats() -> None:
    catalog = build_catalog(SAMPLE_POSTS)

    assert catalog["schemaVersion"] == 1
    assert catalog["stats"] == {
        "postCount": 3,
        "topicCount": 2,
        "seriesCount": 1,
        "latestDate": "2025-02-03",
    }
    assert catalog["topics"] == [
        {
            "name": "大模型与Transformer",
            "count": 2,
            "slug": catalog["topics"][0]["slug"],
        },
        {"name": "数学工具", "count": 2, "slug": catalog["topics"][1]["slug"]},
    ]
    assert catalog["topicGroups"] == [
        {"name": "AI 与机器学习", "topics": ["大模型与Transformer"]},
        {"name": "数学与基础科学", "topics": ["数学工具"]},
    ]

    by_id = {post["id"]: post for post in catalog["posts"]}
    assert by_id["30"]["sourceSummary"] is None
    assert by_id["30"]["url"] is None
    assert by_id["30"]["seriesId"] is None
    assert by_id["20"]["sourceSummary"] == "明确的小结短摘录。"
    assert by_id["20"]["sourceCategory"] == "信息时代"
    assert by_id["20"]["sourceTags"] == ["Transformer", "位置编码"]
    assert set(by_id["20"]) == {
        "id",
        "title",
        "url",
        "date",
        "sourceCategory",
        "sourceTags",
        "sourceSummary",
        "topics",
        "series",
        "seriesId",
        "seriesIndex",
        "level",
        "seriesTopic",
        "notes",
    }
    assert "body" not in by_id["20"]


def test_series_id_is_stable_and_numbered_members_use_index_order() -> None:
    forward = build_catalog(SAMPLE_POSTS)
    reversed_catalog = build_catalog(list(reversed(SAMPLE_POSTS)))

    expected_id = stable_series_id("示例系列")
    assert forward["series"] == reversed_catalog["series"]
    assert forward["series"] == [
        {
            "id": expected_id,
            "name": "示例系列",
            "topic": "大模型与Transformer",
            "count": 2,
            "startDate": "2024-01-01",
            "endDate": "2024-06-01",
            "postIds": ["10", "20"],
        }
    ]
    assert {
        post["seriesId"] for post in forward["posts"] if post["series"] == "示例系列"
    } == {expected_id}


def test_unnumbered_series_members_use_chronological_order() -> None:
    posts = [
        {
            "id": "new",
            "title": "后记",
            "url": "https://spaces.ac.cn/archives/new",
            "date": "2022-09-01",
            "topics": ["阅读写作与随笔"],
            "series": "无编号系列",
        },
        {
            "id": "old",
            "title": "开篇",
            "url": "https://spaces.ac.cn/archives/old",
            "date": "2020-01-01",
            "topics": ["阅读写作与随笔"],
            "series": "无编号系列",
        },
    ]

    assert build_catalog(posts)["series"][0]["postIds"] == ["old", "new"]


def test_mixed_series_orders_numbered_members_then_dated_fallbacks() -> None:
    posts = [
        {
            "id": "fallback-new",
            "title": "无编号后篇",
            "url": "https://spaces.ac.cn/archives/fallback-new",
            "date": "2023-01-01",
            "topics": ["数学工具"],
            "series": "混合系列",
        },
        {
            "id": "numbered-two",
            "title": "第二篇",
            "url": "https://spaces.ac.cn/archives/numbered-two",
            "date": "2024-01-01",
            "topics": ["数学工具"],
            "series": "混合系列",
            "series_index": 2,
        },
        {
            "id": "fallback-old",
            "title": "无编号前篇",
            "url": "https://spaces.ac.cn/archives/fallback-old",
            "date": "2020-01-01",
            "topics": ["数学工具"],
            "series": "混合系列",
        },
        {
            "id": "numbered-one",
            "title": "第一篇",
            "url": "https://spaces.ac.cn/archives/numbered-one",
            "date": "2025-01-01",
            "topics": ["数学工具"],
            "series": "混合系列",
            "series_index": 1,
        },
    ]

    assert build_catalog(posts)["series"][0]["postIds"] == [
        "numbered-one",
        "numbered-two",
        "fallback-old",
        "fallback-new",
    ]


def test_build_is_deterministic_and_replaces_stale_output() -> None:
    with writable_test_directory() as tmp_path:
        input_path = tmp_path / "posts.json"
        web_dir = tmp_path / "web"
        output_dir = tmp_path / "site"
        web_dir.mkdir()
        (web_dir / "index.html").write_text(
            "<!doctype html><title>Index</title>\n", encoding="utf-8"
        )
        (web_dir / "assets").mkdir()
        (web_dir / "assets" / "app.js").write_text(
            "console.log('ok');\n", encoding="utf-8"
        )
        input_path.write_text(json.dumps(SAMPLE_POSTS, ensure_ascii=False), encoding="utf-8")

        build_site(input_path, output_dir, web_dir=web_dir)
        first_catalog = (output_dir / "catalog.json").read_bytes()
        assert (output_dir / "index.html").is_file()
        assert (output_dir / "assets" / "app.js").is_file()
        assert (output_dir / GENERATED_SITE_MARKER).is_file()

        (output_dir / "stale.txt").write_text("remove me", encoding="utf-8")
        input_path.write_text(
            json.dumps(list(reversed(SAMPLE_POSTS)), ensure_ascii=False), encoding="utf-8"
        )
        build_site(input_path, output_dir, web_dir=web_dir)

        assert (output_dir / "catalog.json").read_bytes() == first_catalog
        assert not (output_dir / "stale.txt").exists()


@pytest.mark.parametrize("output_dir", [ROOT, ROOT.parent / "outside-site"])
def test_build_rejects_output_that_is_not_below_repository_root(output_dir: Path) -> None:
    with pytest.raises(ValueError, match="child of the repository root"):
        build_site(ROOT / "missing.json", output_dir, web_dir=ROOT / "tests")


def test_build_refuses_to_replace_an_unmarked_repository_directory() -> None:
    with writable_test_directory() as tmp_path:
        input_path = tmp_path / "posts.json"
        web_dir = tmp_path / "web"
        output_dir = tmp_path / "existing"
        input_path.write_text(json.dumps(SAMPLE_POSTS, ensure_ascii=False), encoding="utf-8")
        web_dir.mkdir()
        (web_dir / "index.html").write_text("<!doctype html>\n", encoding="utf-8")
        output_dir.mkdir()
        sentinel = output_dir / "keep.txt"
        sentinel.write_text("do not delete", encoding="utf-8")

        with pytest.raises(ValueError, match="has no generated-site marker"):
            build_site(input_path, output_dir, web_dir=web_dir)
        assert sentinel.read_text(encoding="utf-8") == "do not delete"
