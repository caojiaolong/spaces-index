from argparse import Namespace

import pytest

from scripts import update_all as pipeline
from scripts.extract_articles import atomic_json
from scripts.common import read_json


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ARTICLES_DIR", tmp_path / "data/articles")
    monkeypatch.setattr(pipeline, "PREVIEW_DIR", tmp_path / "build/preview")
    monkeypatch.setattr(pipeline, "PUBLIC_DIR", tmp_path / "_site")
    monkeypatch.setattr("requests.Session.request", lambda *args, **kwargs: pytest.fail("Unexpected network request"))
    raw = {"id": 12345, "title": "Transformer 学习", "url": "https://spaces.ac.cn/archives/12345", "date": "2026-09-15"}
    atomic_json(pipeline.ARTICLES_DIR / "archive.json", [raw])
    atomic_json(pipeline.ARTICLES_DIR / "12345/snapshot.json", {"metadata": {"source_category": "信息时代", "source_tags": ["Transformer"], "source_summary": None}})
    return tmp_path, raw


def test_metadata_reuses_details_from_article_request(workspace):
    tmp_path, raw = workspace
    result = pipeline.sync_metadata([raw])
    assert result[0]["source_tags"] == ["Transformer"]
    assert read_json(tmp_path / "data/posts.json", []) == result


@pytest.mark.parametrize("offline,audience,expected", [(False, "preview", "--all"), (False, "public", "--all"), (True, "preview", "--offline")])
def test_one_command_connects_ingestion_classification_and_build(workspace, monkeypatch, offline, audience, expected):
    tmp_path, raw = workspace
    steps = []
    monkeypatch.setattr(pipeline, "extract", lambda args: steps.append(("extract", args)) or 0)
    monkeypatch.setattr(pipeline, "render_all", lambda posts: steps.append(("render", posts)))
    def build(path, destination, **kwargs):
        posts = read_json(path, [])
        assert posts[0]["topics"] and posts[0]["source_tags"] == ["Transformer"]
        steps.append(("build", destination))
        return {"mirrors": {}}
    monkeypatch.setattr(pipeline, "build_site", build)
    args = Namespace(offline=offline, audience=audience, sleep=3, max_refresh=2, force=False, refresh_summaries=False, skip_enrich=False)
    result = pipeline.update(args)
    assert [s[0] for s in steps] == ["extract", "render", "build"]
    assert expected in steps[0][1]
    command = steps[0][1]
    assert "--all" in command
    assert "--ids" not in command
    assert result["articles"] == 1


def test_old_snapshot_does_not_replace_fresher_metadata(workspace):
    tmp_path, raw = workspace
    recent = raw | {"source_category": "数学", "source_tags": ["新标签"], "source_summary": "新小结"}
    atomic_json(tmp_path / "data/posts.json", [recent])
    assert pipeline.sync_metadata([raw]) == [recent]


def test_incomplete_metadata_is_reported_without_second_fetcher(workspace):
    tmp_path, raw = workspace
    atomic_json(tmp_path / "data/posts.json", [raw | {"metadata_error": "missing metadata line"}])
    posts = pipeline.sync_metadata([raw])
    assert posts[0]["metadata_error"] == "missing metadata line"
