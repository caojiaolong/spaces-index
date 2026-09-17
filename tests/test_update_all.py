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


def test_ci_updates_index_without_building(workspace, monkeypatch):
    monkeypatch.setattr(pipeline, "extract", lambda args: 0)
    monkeypatch.setattr(pipeline, "render_all", lambda posts: None)
    monkeypatch.setattr(pipeline, "build_site", lambda *a, **kw: pytest.fail("CI must build only after checking changes"))
    args = Namespace(offline=False, audience="public", sleep=3, max_refresh=None,
                     force=False, skip_enrich=False, refresh_summaries=False, skip_build=True)
    result = pipeline.update(args)
    assert result["built"] is False and result["readable_articles"] is None
    assert read_json(workspace[0] / "data/posts_classified.json", [])[0]["topics"]


@pytest.mark.parametrize("counts,stored_failure,expected", [
    ({"images_failed": 1, "images_deferred": 32}, False, 0),
    ({"images_failed": 1, "fetch_failed": 1}, False, 1),
    ({"images_failed": 1, "conversion_failed": 1}, False, 1),
    ({"images_failed": 1, "metadata_failed": 1}, False, 1),
    ({"images_failed": 1, "stopped_early": 1}, False, 1),
    ({"images_failed": 1}, True, 1),
    (None, False, 1),
])
def test_image_warnings_do_not_hide_content_or_early_acquisition_errors(workspace, monkeypatch, capsys, counts, stored_failure, expected):
    root, _ = workspace
    summary = root / ".cache/ingestion/summary.json"
    # A stale image-only summary must not mask today's early archive failure.
    atomic_json(summary, {"this_run": {"images_failed": 1}})
    def ingest(args):
        if counts is not None:
            atomic_json(summary, {"this_run": counts})
        return 1
    monkeypatch.setattr(pipeline, "extract", ingest)
    monkeypatch.setattr(pipeline, "render_all", lambda posts: None)
    if stored_failure:
        atomic_json(pipeline.ARTICLES_DIR / "state.json", {"12345": {"status": "conversion_failed", "error": "bad formula"}})
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(root / "job-summary.md"))
    assert pipeline.main(["--audience", "public", "--skip-build"]) == expected
    result = read_json(root / ".cache/update-result.json", {})
    assert result["exit_code"] == expected
    if counts and counts.get("images_failed"):
        assert "::warning::Image cache incomplete" in capsys.readouterr().out
        assert "Image cache incomplete" in (root / "job-summary.md").read_text(encoding="utf8")
    else:
        assert not result["warnings"]


def test_successful_ingestion_does_not_mask_failed_build(workspace, monkeypatch):
    monkeypatch.setattr(pipeline, "extract", lambda args: 0)
    monkeypatch.setattr(pipeline, "render_all", lambda posts: None)
    def broken_build(*a, **kw):
        raise pipeline.MirrorError("Stored content hash mismatch")
    monkeypatch.setattr(pipeline, "build_site", broken_build)
    assert pipeline.main(["--audience", "public"]) == 1


def test_cannot_serve_an_unbuilt_site():
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["--skip-build", "--serve"])
    assert exc.value.code == 2


def test_plan_only_writes_ci_decision_without_metadata_sync_or_build(workspace, monkeypatch):
    root, _ = workspace
    def extract(args):
        assert '--plan-only' in args and '--refresh-archive' in args
        atomic_json(root / '.cache/ingestion/summary.json', {
            'plan': {'needs_update': False, 'article_ids': [], 'untouched_articles': 1337},
            'timings_seconds': {'planning': 0.1}})
        return 0
    monkeypatch.setattr(pipeline, 'extract', extract)
    monkeypatch.setattr(pipeline, 'sync_metadata', lambda *a, **kw: pytest.fail('Discovery synced index'))
    monkeypatch.setattr(pipeline, 'build_site', lambda *a, **kw: pytest.fail('Discovery built site'))
    monkeypatch.setenv('GITHUB_OUTPUT', str(root / 'output.txt'))
    assert pipeline.main(['--plan-only']) == 0
    assert (root / 'output.txt').read_text() == 'needs_update=false\n'


def test_update_after_discovery_reuses_archive(workspace, monkeypatch):
    commands = []
    monkeypatch.setattr(pipeline, 'extract', lambda args: commands.append(args) or 0)
    monkeypatch.setattr(pipeline, 'render_all', lambda *a: None)
    assert pipeline.main(['--cached-archive', '--skip-build']) == 0
    assert '--refresh-archive' not in commands[0]
    assert '--incremental' in commands[0]
