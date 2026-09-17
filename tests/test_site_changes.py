import json
import subprocess

import pytest

from scripts import check_site_changes as changes

URL = "https://spaces.ac.cn/a.png"
CACHED = {"status": "cached", "file": "a.png", "sha256": "123", "checked_at": "yesterday"}


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    path = tmp_path / "data/articles/123/images.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "images": {URL: CACHED}}), encoding="utf8")
    git("add", ".")
    git("commit", "-qm", "fixture")
    return tmp_path, path, git


@pytest.mark.parametrize("images,deploy", [
    ({URL: CACHED, "https://album.spaces.ac.cn/dead.jpg": {"status": "failed", "error": "DNS failed"}}, False),
    ({URL: CACHED | {"checked_at": "today", "etag": "new", "retry_not_before": "tomorrow", "error": "timeout"}}, False),
    ({URL: CACHED | {"file": "b.png", "sha256": "456"}}, True),
    ({URL: {"status": "failed", "error": "HTTP 404"}}, True),
    ({}, True),
    ({URL: CACHED, "https://spaces.ac.cn/new.png": CACHED}, True),
])
def test_image_retries_are_committed_without_deploying_but_usable_images_are_deployed(repo, images, deploy):
    root, path, git = repo
    path.write_text(json.dumps({"version": 1, "images": images}), encoding="utf8")
    git("add", ".")  # Include staged changes too.
    result = changes.check_changes(root, "schedule")
    assert result["should_commit"] is True
    assert result["should_deploy"] is deploy


@pytest.mark.parametrize("path,content", [
    ("data/articles/123/article.md", "updated body"),
    ("data/articles/123/images/a.png", "updated bytes"),
    ("config/mirror.json", '{"withdrawn_ids":[123]}'),
    ("data/articles/123/images.json", "invalid json"),
])
def test_content_policy_files_and_invalid_manifests_always_build(repo, path, content):
    root, _, _ = repo
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf8")
    assert changes.check_changes(root, "schedule")["should_deploy"]


def test_deleted_manifest_builds_and_clean_push_still_deploys(repo):
    root, manifest, _ = repo
    assert not changes.check_changes(root, "schedule")["should_deploy"]
    assert changes.check_changes(root, "push")["should_deploy"]
    assert not changes.check_changes(root, "push")["should_commit"]
    manifest.unlink()
    assert changes.check_changes(root, "schedule")["should_deploy"]
