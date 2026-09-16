import json
from types import SimpleNamespace

import pytest
import requests
from bs4 import BeautifulSoup

from scripts import extract_articles as local
from scripts.common import read_json
from scripts.mirror_content import ALLOWED, MirrorError, convert_article, source_body, validate_body
from scripts.mirror_store import verified_article


def page(post_id, body=None):
    body = body or "<h2>正文</h2><p>完整文字，公式 $x+1$。</p>"
    return (f"<div class='Post'><div class='title-wrap'><h1><a href='https://spaces.ac.cn/archives/{post_id}'>测试文章</a></h1>"
            "<span class='submitted'>苏剑林 | 2026-09-14</span></div>"
            "<span class='cat'>分类：<a href='/category/math/'>数学</a> 标签：<a href='/tag/test/'>测试</a></span>"
            f"<div id='PostContent'>{body}<p>转载到请包括本文地址：原文</p>"
            "<p>更详细的转载事宜请参考：FAQ</p><div id='content_tips'>评论引导</div>"
            "<div id='pay'>打赏</div><div id='how_to_cite'>引用模板</div></div></div>"
            "<div>不收录的评论</div><a href='https://creativecommons.org/licenses/by-nc-nd/2.5/cn/'>许可</a>")


def response(status, content=b"", headers=None):
    r = requests.Response()
    r.status_code, r._content = status, content
    r.headers.update(headers or {})
    return r


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(local, "STORAGE_ROOT", tmp_path / "data")
    output = tmp_path / "data" / "articles"
    calls, responses = [], []
    def get(url, headers=None):
        calls.append((url, headers))
        r = responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r
    fake = SimpleNamespace(get=get, check_robots=lambda: None)
    monkeypatch.setattr(local, "SerialFetcher", lambda *args: fake)
    return output, calls, responses


def test_fetch_then_resume_without_requests(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345").encode(), {"ETag": '"first"'}))
    assert local.main(["--ids", "12345"]) == 0
    assert (output / "12345/article.md").is_file()
    assert "不收录的评论" not in (output / "12345/source.html").read_text(encoding="utf-8")
    assert local.main(["--ids", "12345"]) == 0
    assert len(calls) == 1
    assert read_json(output / "summary.json", {})["this_run"]["cached"] == 1
    assert verified_article(output, "12345")["validation"]["passed"]
    metadata = read_json(output / "metadata.json", [])[0]
    assert metadata["source_category"] == "数学"
    assert metadata["source_tags"] == ["测试"]
    assert "metadata_error" not in metadata


def test_failed_conversion_keeps_source_and_retries_offline(setup, monkeypatch):
    output, calls, responses = setup
    responses.append(response(200, page("12345").encode()))
    real_convert = local.convert_snapshot
    monkeypatch.setattr(local, "convert_snapshot", lambda *args: (_ for _ in ()).throw(MirrorError("sample parser failure")))
    assert local.main(["--ids", "12345"]) == 1
    assert (output / "12345/source.html").is_file()
    assert not (output / "12345/article.md").exists()
    assert read_json(output / "metadata.json", [])[0]["source_tags"] == ["测试"]
    monkeypatch.setattr(local, "convert_snapshot", real_convert)
    assert local.main(["--ids", "12345", "--offline"]) == 0
    assert len(calls) == 1


def test_limit_and_resume_selection(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345", "12346", "--limit", "1"]) == 0
    assert len(calls) == 1
    assert not (output / "12346").exists()


def test_resume_repairs_incomplete_snapshot_installation(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345"]) == 0
    (output / "12345/article.md").unlink()
    (output / "12345/source.html").write_text("partial", encoding="utf-8")
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345"]) == 0
    assert len(calls) == 2
    assert verified_article(output, "12345", local_only=True)["validation"]["passed"]


def test_conditional_refresh_304(setup):
    output, calls, responses = setup
    responses.extend([response(200, page("12345").encode(), {"ETag": '"first"'}), response(304)])
    assert local.main(["--ids", "12345"]) == 0
    before = (output / "12345/article.md").read_bytes()
    assert local.main(["--ids", "12345", "--refresh"]) == 0
    assert calls[-1][1] == {"If-None-Match": '"first"'}
    assert (output / "12345/article.md").read_bytes() == before


def test_404_removes_generated_body(setup):
    output, calls, responses = setup
    responses.extend([response(200, page("12345").encode()), response(404)])
    assert local.main(["--ids", "12345"]) == 0
    assert local.main(["--ids", "12345", "--refresh"]) == 0
    assert not (output / "12345/article.md").exists()
    assert read_json(output / "state.json", {})["12345"]["status"] == "removed"


def test_interrupt_keeps_completed_article_and_resume(setup):
    output, calls, responses = setup
    responses.extend([response(200, page("12345").encode()), KeyboardInterrupt()])
    assert local.main(["--ids", "12345", "12346"]) == 130
    assert (output / "12345/article.md").exists()
    responses.append(response(200, page("12346").encode()))
    assert local.main(["--ids", "12345", "12346"]) == 0
    assert len(calls) == 3


def test_three_acquisition_failures_stop_the_run(setup):
    output, calls, responses = setup
    responses.extend([requests.Timeout("timeout")] * 3)
    assert local.main(["--ids", "12345", "12346", "12347", "12348"]) == 1
    assert len(calls) == 3
    assert "12348" not in read_json(output / "state.json", {})


def test_retry_failed_only(setup):
    output, calls, responses = setup
    responses.extend([response(200, page("12345").encode()), response(403)])
    assert local.main(["--ids", "12345", "12346"]) == 1
    responses.append(response(200, page("12346").encode()))
    assert local.main(["--ids", "12345", "12346", "--retry-failed"]) == 0
    assert [u.rsplit("/", 1)[1] for u, _ in calls] == ["12345", "12346", "12346"]


def test_output_cannot_be_in_public_directories(setup):
    with pytest.raises(SystemExit):
        local.main(["--ids", "12345", "--output", "web/articles"])


def test_offline_missing_and_status_make_no_requests(setup):
    output, calls, responses = setup
    assert local.main(["--ids", "12345", "--offline"]) == 0
    assert local.main(["--status"]) == 0
    assert calls == []


def test_local_run_lock_rejects_parallel_writer(setup):
    output, _, _ = setup
    with local.run_lock(output):
        with pytest.raises(MirrorError, match="Another extractor"):
            with local.run_lock(output):
                pass


def test_exact_code_and_math_are_distinguished():
    body = ('<p>普通文字 (one) [two]，以及公式 $x^2$。<code>echo $HOME</code></p>'
            '<pre><code class="language-mathematica">\\[Theta][t_] = 2;\n  x = "`$";\n</code></pre>'
            '<p>H<sub>2</sub>O，x<sup>2</sup>。</p>'
            '<blockquote>Traceback: <type>: unchanged</type></blockquote>')
    result = convert_article(page("12345", body), "12345")
    assert result["validation"]["passed"]
    assert result["validation"]["formula_count"] == 1
    assert result["validation"]["code_block_count"] == 1
    root, meta = source_body(page("12345", body), "12345")
    for changed in (result["body"].replace('  x =', ' x ='), result["body"].replace('$HOME', '$PATH')):
        assert not validate_body(root, changed, meta["source_url"])["passed"]


def test_underline_preserved_with_math_and_position_audited():
    body = '<p>前<u><strong>重点</strong> $x_1$</u>中<u>重点</u>后重点。</p>'
    result = convert_article(page("12345", body), "12345")
    assert result["validation"]["passed"]
    assert result["validation"]["formula_count"] == 1
    assert '<u><strong>重点</strong> $x_1$</u>' in result["body"]
    root, meta = source_body(page("12345", body), "12345")
    for changed in (
        result["body"].replace('<u>', '').replace('</u>', ''),
        result["body"].replace('<u>重点</u>后重点', '重点后<u>重点</u>'),
        result["body"].replace('$x_1$', '$x_2$'),
    ):
        assert not validate_body(root, changed, meta["source_url"])["passed"]


def test_previously_unsupported_underline_recovers_offline(setup, monkeypatch):
    output, calls, responses = setup
    responses.append(response(200, page("12345", '<p>保留<u>下划线</u>与公式 $x$。</p>').encode()))
    with monkeypatch.context() as old_converter:
        old_converter.setattr("scripts.mirror_content.ALLOWED", ALLOWED - {"u"})
        old_converter.setattr(local, "convert_html_body", lambda *args: (_ for _ in ()).throw(MirrorError("Unsupported body element: u")))
        assert local.main(["--ids", "12345"]) == 1
    assert read_json(output / "state.json", {})["12345"]["error"] == "Unsupported body element: u"
    assert local.main(["--ids", "12345", "--offline"]) == 0
    assert len(calls) == 1
    assert '<u>下划线</u>' in (output / "12345/article.md").read_text(encoding="utf-8")
    assert verified_article(output, "12345", local_only=True)["validation"]["passed"]


def test_atomic_save_retries_temporary_windows_file_lock(tmp_path, monkeypatch):
    from pathlib import Path
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf8")
    original = Path.replace
    attempts = []
    def replace(path, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert target.read_text(encoding="utf8") == "old"
            raise PermissionError("temporarily locked")
        return original(path, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(local.time, "sleep", lambda n: None)
    local.atomic_text(target, "new")
    assert target.read_text(encoding="utf8") == "new"
    assert len(attempts) == 3


def test_scheduled_refresh_is_bounded_and_oldest_first(setup, tmp_path, monkeypatch):
    output, calls, responses = setup
    responses.extend([response(200, page("12345").encode(), {"ETag": '"first"'}), response(200, page("12346").encode(), {"ETag": '"second"'})])
    assert local.main(["--ids", "12345", "12346"]) == 0
    state = read_json(output / "state.json", {})
    state["12345"]["checked_at"] = "2001-01-01T00:00:00+00:00"
    state["12346"]["checked_at"] = "2000-01-01T00:00:00+00:00"
    local.atomic_json(output / "state.json", state)
    policy = tmp_path / "policy.json"
    local.atomic_json(policy, {"refresh_days": 30, "refresh_per_run": 1})
    monkeypatch.setattr(local, "POLICY_PATH", policy)
    responses.append(response(304))
    assert local.main(["--ids", "12345", "12346", "--refresh-due"]) == 0
    assert len(calls) == 3 and calls[-1][0].endswith("/12346")
    assert read_json(output / "state.json", {})["12345"]["checked_at"] == state["12345"]["checked_at"]


def test_offline_conversion_does_not_extend_source_freshness(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345"]) == 0
    state = read_json(output / "state.json", {})
    state["12345"]["checked_at"] = "2000-01-01T00:00:00+00:00"
    local.atomic_json(output / "state.json", state)
    assert local.main(["--ids", "12345", "--offline"]) == 0
    assert read_json(output / "state.json", {})["12345"]["checked_at"] == state["12345"]["checked_at"]
    assert len(calls) == 1


def test_one_response_supplies_body_metadata_and_short_excerpt(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345", '<p>正文 $x$。</p><h2>小结</h2><p>作者自己的小结。</p>').encode()))
    assert local.main(["--ids", "12345"]) == 0
    metadata = read_json(output / "metadata.json", [])[0]
    assert metadata["source_summary"] == "作者自己的小结。"
    assert "作者自己的小结。" in (output / "12345/article.md").read_text(encoding="utf8")
    assert len(calls) == 1


def test_metadata_failure_is_saved_without_second_request_and_can_retry(setup):
    output, calls, responses = setup
    html = page("12345").replace("分类：", "栏目：")
    responses.append(response(200, html.encode(), {"ETag": '"first"'}))
    assert local.main(["--ids", "12345"]) == 1
    assert verified_article(output, "12345", local_only=True)["validation"]["passed"]
    assert read_json(output / "metadata.json", [])[0]["metadata_error"]
    assert local.main(["--ids", "12345"]) == 0
    assert len(calls) == 1  # Failed metadata cools down instead of immediately refetching.
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345", "--retry-failed"]) == 0
    assert calls[-1][1] == {}  # Missing metadata requires the page, not a 304.
    assert "metadata_error" not in read_json(output / "metadata.json", [])[0]


def test_failed_body_boundary_keeps_successfully_parsed_metadata(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345").replace("id='PostContent'", "id='missing'").encode()))
    assert local.main(["--ids", "12345"]) == 1
    meta = read_json(output / "metadata.json", [])[0]
    assert meta["source_category"] == "数学" and "metadata_error" not in meta
    assert not (output / "12345/source.html").exists()


def test_missing_summary_recovered_offline_from_checked_snapshot(setup):
    output, calls, responses = setup
    responses.append(response(200, page("12345", '<h2>小结</h2><p>忠于原文。</p>').encode()))
    assert local.main(["--ids", "12345"]) == 0
    raw = {"id": 12345, "title": "文章", "url": "https://spaces.ac.cn/archives/12345"}
    cached = {"source_category": "数学", "source_tags": [], "source_summary": None}
    result = local.cached_metadata(raw, cached, output, refresh_summary=True)
    assert result["source_summary"] == "忠于原文。"
    assert len(calls) == 1


def test_failed_offline_conversion_then_repair_does_not_extend_freshness(setup, monkeypatch):
    output, calls, responses = setup
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345"]) == 0
    state = read_json(output / "state.json", {})
    state["12345"]["checked_at"] = "2000-01-01T00:00:00+00:00"
    local.atomic_json(output / "state.json", state)
    with monkeypatch.context() as failure:
        failure.setattr(local, "convert_snapshot", lambda *args: (_ for _ in ()).throw(MirrorError("conversion failed")))
        assert local.main(["--ids", "12345", "--offline"]) == 1
    assert local.main(["--ids", "12345", "--offline"]) == 0
    assert read_json(output / "state.json", {})["12345"]["checked_at"] == "2000-01-01T00:00:00+00:00"
    assert len(calls) == 1


def test_failed_network_recheck_cannot_restore_old_body_until_success(setup):
    output, calls, responses = setup
    responses.extend([response(200, page("12345").encode()), requests.Timeout("timeout")])
    assert local.main(["--ids", "12345"]) == 0
    assert local.main(["--ids", "12345", "--refresh"]) == 1
    metadata = read_json(output / "metadata.json", [])[0]
    assert metadata["source_category"] == "数学" and metadata["source_tags"] == ["测试"]
    assert (output / "12345/source.html").exists()
    assert not (output / "12345/article.md").exists()
    assert local.main(["--ids", "12345", "--offline", "--retry-failed"]) == 0
    assert local.main(["--ids", "12345"]) == 0
    assert not (output / "12345/article.md").exists()
    assert len(calls) == 2
    responses.append(response(200, page("12345").encode()))
    assert local.main(["--ids", "12345", "--retry-failed"]) == 0
    assert verified_article(output, "12345", local_only=True)["validation"]["passed"]


def test_removed_article_without_metadata_waits_for_scheduled_recheck(setup):
    output, calls, responses = setup
    responses.append(response(404))
    assert local.main(["--ids", "12345"]) == 0
    assert local.main(["--ids", "12345"]) == 0
    assert len(calls) == 1


def test_unexpected_304_cannot_generate_body(setup):
    output, calls, responses = setup
    responses.append(response(304))
    assert local.main(["--ids", "12345"]) == 1
    assert not (output / "12345/article.md").exists()
    assert "Unexpected 304" in read_json(output / "metadata.json", [])[0]["metadata_error"]
