from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from bs4 import BeautifulSoup

from scripts.common import read_json, write_json
from scripts import mirror_articles as crawler
from scripts.mirror_content import (
    MirrorError, convert_article, convert_body, math_spans, source_body, source_image_captions, validate_body,
)
from scripts.mirror_store import load_config, now_iso, publish_mirrors, verified_article


def page(body=None, post_id="9119"):
    body = body or ("<h2 id='intro'>标题<a href='#intro'> #</a></h2>"
                    "<p>忠于原文的中文说明。" + "这是一段用于离线测试的正文，保留文字与标点。" * 6 +
                    r"<strong>提示：</strong>先看$x_1$，再看$y_2$。<br>"
                    r"\begin{equation}\begin{aligned}a&amp;=b\\<br>c&amp;=d\end{aligned}\end{equation}"
                    "</p><p><a href='/archives/123'>链接</a>以及图片：</p>"
                    "<div><img src='/usr/uploads/example.png' alt='说明-图片'></div>"
                    "<blockquote><p>引用原文。</p></blockquote><h2>结语</h2><p>结束段落。</p>")
    return (f"<div class='Post'><div class='title-wrap'><h1><a href='https://spaces.ac.cn/archives/{post_id}'>测试标题</a></h1>"
            "<span class='submitted'>By 苏剑林 | 2026-09-01</span></div>"
            f"<div id='PostContent'>{body}<!-- this comment is not body -->"
            "<p>转载到请包括本文地址：原文</p><p>更详细的转载事宜请参考：FAQ</p>"
            "<div id='content_tips'>推荐与评论引导</div><div id='pay'>打赏图片</div>"
            "<div id='how_to_cite'>引用模板</div></div></div>"
            "<aside>侧栏内容</aside><div id='comments'>评论内容</div>"
            "<a href='http://creativecommons.org/licenses/by-nc-nd/2.5/cn/'>许可</a>")


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    directory = tmp_path / "mirror"
    directory.mkdir()
    write_json(directory / "config.json", {"withdrawn_ids": [], "refresh_days": 7})
    monkeypatch.setattr(crawler, "MIRROR_DIR", directory)
    monkeypatch.setattr(crawler, "CACHE", tmp_path / "cache")
    monkeypatch.setattr("scripts.image_cache.sync_images", lambda *a, **kw: {})
    return directory


def seed(directory, post_id="9119"):
    report = crawler.save_article(directory, post_id, page(post_id=post_id), now_iso())
    state = read_json(directory / "state.json", {})
    state[post_id] = {"status": "verified", "checked_at": now_iso(),
                      "markdown_sha256": report["markdown_sha256"], "etag": '"v1"'}
    write_json(directory / "state.json", state)
    return report


def test_roundtrip_math_images_and_body_boundary():
    result = convert_article(page(), "9119")
    assert result["validation"]["passed"]
    assert result["validation"]["formula_count"] == 3
    assert result["validation"]["image_count"] == 1
    assert result["validation"]["heading_count"] == 2
    markdown = result["markdown"]
    assert r"\begin{aligned}a&=b\\" in markdown
    assert "<strong>提示：</strong>" in markdown
    for excluded in ("评论内容", "侧栏内容", "推荐与评论引导", "打赏图片", "引用模板", "this comment"):
        assert excluded not in markdown
    for required in ("苏剑林", "https://spaces.ac.cn/archives/9119", "CC BY-NC-ND 2.5 CN", "非官方", "额外授权"):
        assert required in markdown


@pytest.mark.parametrize("change", [
    lambda m: m.replace("忠于原文", "修改观点"),
    lambda m: m.replace("结束段落。", ""),
    lambda m: m.replace("结束段落。", "结束段落。结束段落。"),
    lambda m: m.replace("$x_1$", ""),
    lambda m: m.replace("$x_1$", "$x_1$$x_1$"),
    lambda m: m.replace("$x_1$", "$x_2$"),
    lambda m: m.replace("$x_1$", "$z$").replace("$y_2$", "$x_1$").replace("$z$", "$y_2$"),
    lambda m: m.replace("先看$x_1$", "$x_1$先看"),
    lambda m: m.replace("说明-图片", "说明图片"),
    lambda m: m.replace("/usr/uploads/example.png", "/usr/uploads/other.png"),
    lambda m: m.replace("/archives/123", "/archives/456"),
    lambda m: m.replace("## 结语", "### 结语"),
])
def test_mutations_are_rejected(change):
    root, meta = source_body(page(), "9119")
    markdown = convert_body(root, meta["source_url"])
    try:
        result = validate_body(root, change(markdown), meta["source_url"])
        assert not result["passed"]
    except MirrorError:
        pass


@pytest.mark.parametrize("value", ["$unclosed", r"\begin{equation}missing", r"\begin{a}\end{b}", r"\]orphan"])
def test_broken_latex_rejected(value):
    with pytest.raises(MirrorError):
        math_spans(value)


def test_all_delimiters_and_nested_environments():
    text = r"价格\$5，$a$ $$b$$ \(c\) \[d\] \begin{equation}\begin{aligned}e&=f\end{aligned}\end{equation}"
    assert len(math_spans(text)) == 5


def test_caption_hints_require_explicit_source_structure():
    root = BeautifulSoup("""<div id="PostContent">
      <img src="plain.png" alt="看起来像图注"><p>看起来像图注</p>
      <div class="typecho-caption"><div><a href="one.png"><img src="one.png"></a></div>
        <p class="typecho-caption-text">图一：<em>完整</em>说明 $x_1$</p></div>
      <div class="typecho-caption"><div><img src="old.png"></div><p>旧版图注</p></div>
      <div class="typecho-caption"><img src="empty.png"><p> </p></div>
      <div class="typecho-caption"><img src="ambiguous.png"><p>第一段</p><p>第二段</p></div>
    </div>""", "lxml").select_one("#PostContent")
    assert source_image_captions(root) == [
        {"imageIndex": 1, "text": "图一：完整说明 $x_1$"},
        {"imageIndex": 2, "text": "旧版图注"},
    ]


@pytest.mark.parametrize("change", [
    lambda p: p.replace("id='PostContent'", "id='changed'"),
    lambda p: p.replace("id='content_tips'", "id='different'"),
    lambda p: p.replace("By 苏剑林", "By Someone else"),
    lambda p: p.replace("by-nc-nd/2.5/cn/", "by/4.0/"),
    lambda p: p.replace("<p>结束段落。", "<p onclick='alert(1)'>结束段落。"),
    lambda p: p.replace("<p>结束段落。", "<iframe></iframe><p>结束段落。"),
    lambda p: p.replace("<p>结束段落。", "<script>alert(1)</script><p>结束段落。"),
    lambda p: p.replace("<p>结束段落。", "<table><tr><td>待支持</td></tr></table><p>结束段落。"),
])
def test_changed_layout_or_unsupported_content_fails_closed(change):
    with pytest.raises(MirrorError):
        convert_article(change(page()), "9119")


def test_publication_reaudits_content_and_never_copies_source(mirror, tmp_path):
    seed(mirror)
    output = tmp_path / "site"
    entries = publish_mirrors(output, mirror, allowed_ids={"9119"})
    assert set(entries) == {"9119"}
    assert (output / "mirror/9119/article.md").exists()
    assert not list(output.rglob("*.html"))
    body = mirror / "9119/article.md"
    body.write_text(body.read_text(encoding="utf-8").replace("结束段落。", "篡改段落。"), encoding="utf-8")
    with pytest.raises(MirrorError):
        publish_mirrors(tmp_path / "rejected", mirror, allowed_ids={"9119"})
    assert not list((tmp_path / "rejected").rglob("*.md"))


@pytest.mark.parametrize("setting", ["withdrawn", "withdrawn_numeric", "stale", "failed", "missing_from_catalog"])
def test_publication_gates(mirror, tmp_path, setting):
    seed(mirror)
    config = load_config(mirror)
    state = read_json(mirror / "state.json", {})
    ids = {"9119"}
    if setting == "withdrawn": config["withdrawn_ids"] = ["9119"]
    if setting == "withdrawn_numeric": config["withdrawn_ids"] = [9119]
    if setting == "stale": state["9119"]["checked_at"] = "2000-01-01T00:00:00+00:00"
    if setting == "failed": state["9119"]["status"] = "failed"
    if setting == "missing_from_catalog": ids = set()
    write_json(mirror / "config.json", config)
    write_json(mirror / "state.json", state)
    output = tmp_path / "output"
    assert publish_mirrors(output, mirror, allowed_ids=ids) == {}
    assert not list(output.rglob("*.md"))


def test_maintenance_cli_rejects_invalid_ids(mirror):
    with pytest.raises(SystemExit):
        crawler.main(["--ids", "../12345"])


def response(code, content=b"", headers=None):
    result = requests.Response()
    result.status_code, result._content = code, content
    result.headers.update(headers or {})
    return result


def mock_fetcher(monkeypatch, responses):
    calls = []
    def get(url, headers=None):
        calls.append((url, headers))
        return responses.pop(0)
    fake = SimpleNamespace(check_robots=lambda: None, get=get)
    monkeypatch.setattr(crawler, "SerialFetcher", lambda *args: fake)
    return calls


def test_cached_run_makes_no_requests(mirror, monkeypatch):
    seed(mirror)
    calls = mock_fetcher(monkeypatch, [])
    assert crawler.main(["--ids", "9119"]) == 0
    assert calls == []


def test_304_uses_conditional_headers_and_keeps_valid_body(mirror, monkeypatch):
    seed(mirror)
    before = (mirror / "9119/article.md").read_bytes()
    calls = mock_fetcher(monkeypatch, [response(304)])
    assert crawler.main(["--ids", "9119", "--refresh"]) == 0
    assert calls[0][1] == {"If-None-Match": '"v1"'}
    assert (mirror / "9119/article.md").read_bytes() == before


@pytest.mark.parametrize("code", [404, 410, 403, 500])
def test_source_unavailable_removes_files_and_publication(mirror, monkeypatch, tmp_path, code):
    seed(mirror)
    mock_fetcher(monkeypatch, [response(code)])
    assert crawler.main(["--ids", "9119", "--refresh"]) == (0 if code in {404, 410} else 1)
    assert not (mirror / "9119/article.md").exists()
    assert publish_mirrors(tmp_path / "site", mirror, allowed_ids={"9119"}) == {}


def test_manual_withdrawal_persists_and_skips_network(mirror, monkeypatch):
    seed(mirror)
    assert crawler.main(["--ids", "9119", "--withdraw"]) == 0
    calls = mock_fetcher(monkeypatch, [])
    assert crawler.main(["--ids", "9119", "--refresh"]) == 0
    assert calls == []
    assert "9119" in load_config(mirror)["withdrawn_ids"]


def test_corrupted_cache_is_refetched_without_validators(mirror, monkeypatch):
    seed(mirror)
    (mirror / "9119/article.md").write_text("broken", encoding="utf-8")
    calls = mock_fetcher(monkeypatch, [response(200, page().encode("utf-8"))])
    assert crawler.main(["--ids", "9119", "--refresh"]) == 0
    assert calls[0][1] == {}
    assert verified_article(mirror, "9119")["validation"]["passed"]


def test_transient_retry_is_serial_and_honors_retry_after():
    responses = [response(429, headers={"Retry-After": "17"}), response(200, b"ok")]
    actions = []
    def get(url, **kwargs):
        actions.append(("request", kwargs["allow_redirects"]))
        return responses.pop(0)
    fetcher = crawler.SerialFetcher(session=SimpleNamespace(headers={}, get=get), sleep=lambda n: actions.append(("sleep", n)))
    assert fetcher.get("https://spaces.ac.cn/archives/9119").status_code == 200
    assert actions == [("sleep", 3), ("request", False), ("sleep", 17), ("sleep", 3), ("request", False)]


def test_final_attempt_still_preserves_server_retry_after():
    fetcher = crawler.SerialFetcher(attempts=1, session=SimpleNamespace(
        headers={}, get=lambda *a, **kw: response(429, headers={"Retry-After": "172800"})), sleep=lambda n: None)
    with pytest.raises(crawler.DeferredRetry) as error:
        fetcher.get("https://spaces.ac.cn/archives/9119")
    assert crawler.age_days(error.value.retry_at) < -1.9


def test_redirects_and_foreign_hosts_are_not_followed():
    calls = []
    fetcher = crawler.SerialFetcher(session=SimpleNamespace(headers={}, get=lambda *a, **k: calls.append(k) or response(302, headers={"Location": "https://third.example"})), sleep=lambda n: None)
    assert fetcher.get("https://spaces.ac.cn/usr/uploads/image.png").status_code == 302
    assert calls == [{"headers": {}, "timeout": (10, 45), "allow_redirects": False}]
    with pytest.raises(MirrorError):
        fetcher.get("https://third.example/image.png")


def test_long_retry_after_is_persisted_even_across_manual_refresh(mirror, monkeypatch):
    seed(mirror)
    calls = []
    def get(*args):
        calls.append(args)
        raise crawler.DeferredRetry(172800)
    monkeypatch.setattr(crawler, "SerialFetcher", lambda *args: SimpleNamespace(check_robots=lambda: None, get=get))
    assert crawler.main(["--ids", "9119", "--refresh"]) == 1
    assert read_json(mirror / "state.json", {})["9119"]["retry_not_before"]
    assert crawler.main(["--ids", "9119", "--refresh"]) == 0
    assert len(calls) == 1


def test_full_publication_defaults_work_without_config(mirror, tmp_path):
    seed(mirror, "12345")
    (mirror / "config.json").unlink()
    assert set(publish_mirrors(tmp_path / "site", mirror, allowed_ids={"12345"})) == {"12345"}


def test_full_publication_includes_all_verified_catalog_members(mirror, tmp_path):
    seed(mirror, "9119")
    seed(mirror, "12345")
    assert set(publish_mirrors(tmp_path / "site", mirror, allowed_ids={"9119", "12345"})) == {"9119", "12345"}


def test_failed_article_cannot_partially_publish_full_site(mirror, tmp_path):
    seed(mirror, "9119")
    seed(mirror, "12345")
    (mirror / "12345/article.md").write_text("damaged", encoding="utf8")
    with pytest.raises(MirrorError):
        publish_mirrors(tmp_path / "site", mirror, allowed_ids={"9119", "12345"})
    assert not list((tmp_path / "site").rglob("*.md"))
