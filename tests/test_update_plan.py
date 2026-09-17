import json

import pytest

from scripts.update_plan import pending_articles
from scripts.mirror_store import now_iso


POST = {"id": "123", "title": "Saved article"}
META = {"source_category": "数学", "source_tags": [], "source_summary": None}
VERIFIED = {"status": "verified", "checked_at": now_iso()}
OLD = "2000-01-01T00:00:00+00:00"
FUTURE = "2099-01-01T00:00:00+00:00"
URL = "https://spaces.ac.cn/a.png"


def plan(root, *, state=None, meta=None, policy=None, due=(), **kwargs):
    return pending_articles([POST], {"123": state if state is not None else VERIFIED},
                            {"123": meta if meta is not None else META}, root, policy or {}, set(due), **kwargs)


def test_completed_article_needs_no_body_or_image_files(tmp_path):
    assert plan(tmp_path) == {}


@pytest.mark.parametrize('state,expected', [
    ({}, 'new'), ({'status': 'pending'}, 'body_repair'),
    ({'status': 'conversion_failed'}, 'body_repair'),
    ({'status': 'fetch_failed', 'checked_at': OLD}, 'body_retry'),
    (VERIFIED | {'images_pending': True}, 'unfinished_images'),
])
def test_new_incomplete_and_retryable_work_is_selected(tmp_path, state, expected):
    assert expected in plan(tmp_path, state=state)['123']


def test_due_refreshes_and_withdrawals_are_not_lost(tmp_path):
    assert plan(tmp_path, due={'123'}) == {'123': ['refresh']}
    assert plan(tmp_path, policy={'withdrawn_ids': [123]}) == {'123': ['withdrawal']}
    assert plan(tmp_path, state={'status': 'withdrawn'}, policy={'withdrawn_ids': [123]}) == {}
    assert plan(tmp_path, state={'status': 'removed'}) == {}
    assert plan(tmp_path, state={'status': 'removed'}, due={'123'}) == {'123': ['refresh']}


def test_body_and_metadata_cooldown_and_server_deferral(tmp_path):
    failed = {'status': 'fetch_failed', 'checked_at': now_iso()}
    assert plan(tmp_path, state=failed, meta={}) == {}
    assert plan(tmp_path, state=failed, retry_failed=True)['123'] == ['body_retry']
    assert plan(tmp_path, state=failed | {'retry_not_before': FUTURE}, retry_failed=True) == {}
    assert plan(tmp_path, meta=META | {'metadata_error': 'failed', 'metadata_checked_at': now_iso()}) == {}
    assert plan(tmp_path, meta=META | {'metadata_error': 'failed', 'metadata_checked_at': OLD}) == {'123': ['metadata_retry']}
    assert plan(tmp_path, meta=META | {'metadata_retry_not_before': FUTURE}, force=True) == {}
    assert plan(tmp_path, meta={}) == {'123': ['metadata']}


@pytest.mark.parametrize('image,policy,expected', [
    ({'status': 'cached'}, {}, False),
    ({'status': 'failed', 'checked_at': OLD}, {}, True),
    ({'status': 'failed', 'checked_at': now_iso()}, {}, False),
    ({'status': 'failed', 'checked_at': OLD, 'retry_not_before': FUTURE}, {}, False),
    ({'status': 'cached', 'error': 'timeout', 'retry_not_before': FUTURE}, {}, False),
    ({'status': 'cached', 'error': 'timeout', 'retry_not_before': OLD}, {}, True),
    ({'status': 'skipped', 'skipped_host': 'album.spaces.ac.cn'}, {'skip_image_hosts': ['album.spaces.ac.cn']}, False),
    ({'status': 'skipped', 'skipped_host': 'album.spaces.ac.cn'}, {}, True),
])
def test_retry_manifest_drives_image_work_without_reading_files(tmp_path, image, policy, expected):
    folder = tmp_path / '123'
    folder.mkdir()
    (folder / 'images.json').write_text(json.dumps({'images': {URL: image}}), encoding='utf8')
    assert bool(plan(tmp_path, policy=policy)) == expected


def test_bad_manifest_requires_repair(tmp_path):
    folder = tmp_path / '123'
    folder.mkdir()
    (folder / 'images.json').write_text('broken', encoding='utf8')
    assert plan(tmp_path) == {'123': ['image_manifest_repair']}
