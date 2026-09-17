import json
import os
from unittest.mock import Mock

import pytest

from scripts import ingestion_cache as cache
from scripts.extract_articles import atomic_json
from scripts.mirror_content import MirrorError


@pytest.fixture
def audit(tmp_path):
    folder = tmp_path / '123'
    folder.mkdir()
    for name in cache.BODY_FILES:
        (folder / name).write_text('body', encoding='utf8')
    path = tmp_path / 'cache.json'
    verify = Mock(return_value={'tree': []})
    first = cache.VerifiedBodyCache(path)
    assert first.image_urls(folder, verify) == ([], False)
    first.save(atomic_json)
    return folder, path, verify


def test_unchanged_bytes_reuse_audit_across_runs_and_checkout_times(audit):
    folder, path, verify = audit
    for child in folder.iterdir():
        os.utime(child, (1, 1))
    assert cache.VerifiedBodyCache(path).image_urls(folder, verify) == ([], True)
    assert verify.call_count == 1


@pytest.mark.parametrize('name', cache.BODY_FILES)
def test_changed_bytes_invalidate_audit_even_with_same_size_and_mtime(audit, name):
    folder, path, verify = audit
    target = folder / name
    stamp = target.stat()
    target.write_text('edit', encoding='utf8')
    os.utime(target, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert cache.VerifiedBodyCache(path).image_urls(folder, verify) == ([], False)
    assert verify.call_count == 2


def test_changed_validator_invalidates_cached_results(audit, monkeypatch):
    folder, path, verify = audit
    monkeypatch.setattr(cache, 'validator_fingerprint', lambda: 'new-validator')
    assert cache.VerifiedBodyCache(path).image_urls(folder, verify) == ([], False)
    assert verify.call_count == 2


@pytest.mark.parametrize('value', ['invalid', '{}', '[]', '{"validator":null,"entries":null}'])
def test_unreadable_cache_falls_back_to_verification(audit, value):
    folder, path, verify = audit
    path.write_text(value, encoding='utf8')
    assert cache.VerifiedBodyCache(path).image_urls(folder, verify) == ([], False)
    assert verify.call_count == 2


def test_failed_or_interrupted_audits_cannot_be_reused(audit):
    folder, path, verify = audit
    (folder / 'article.md').write_text('changed', encoding='utf8')
    verify.side_effect = MirrorError('invalid body')
    instance = cache.VerifiedBodyCache(path)
    with pytest.raises(MirrorError):
        instance.image_urls(folder, verify)
    instance.save(atomic_json)
    assert not json.loads(path.read_text())['entries']
    (folder / 'source.html').unlink()
    with pytest.raises(FileNotFoundError):
        cache.VerifiedBodyCache(path).image_urls(folder, verify)
