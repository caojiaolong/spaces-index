import json

import pytest
from bs4 import BeautifulSoup

from scripts.mirror_content import MirrorError, source_body
from scripts.mirror_html import convert_html_body, validate_html_body, preview_html_body
from scripts.local_preview import preview_posts
from tests.test_mirror import page

BASE = "https://spaces.ac.cn/archives/9119"


def root(body):
    return BeautifulSoup('<div id="PostContent">' + body + '</div>', 'lxml').select_one('#PostContent')


@pytest.mark.parametrize("body", [
    '<p>1) 原序号\n=========\n[REL] <del>删除线</del></p>',
    '<p><trong><div><img src="/a.png" alt="[a,b] *c* \\d" /></div></trong></p>',
    '<table><tr><td rowspan="2">甲 $x_1$</td><td>乙</td></tr><tr><td>丙</td></tr></table>',
    '<p><a href="mailto:test@example.com">邮箱</a><a href="htthttps://broken">坏链接</a></p>',
    '<p><a href="/target"><div>跨块链接</div></a><img /></p>',
    '<video><source src="/a.mp4">Your browser does not support video</source></video>',
    '<p>原文<span style="display:none;">隐藏文本</span><u:0002>文字</u:0002></p>',
    '<hs><pre><code>echo $HOME\n  spaces &amp; tabs\t</code></pre></hs>',
    '<script>document.write("&lt;img&gt;");</script><embed src="/a.swf"/>',
])
def test_legacy_layout_roundtrips_exactly(body):
    source = root(body)
    markdown = convert_html_body(source, BASE)
    assert validate_html_body(source, markdown, BASE)["passed"]
    assert preview_html_body(markdown)


@pytest.mark.parametrize("change", [
    lambda md: md.replace('甲', '改'),
    lambda md: md.replace('$x_1$', '$x_2$'),
    lambda md: md.replace('乙', '乙乙'),
    lambda md: md.replace('rowspan="2"', 'rowspan="1"'),
    lambda md: md.replace('<td ', '<td onclick="alert(1)" ', 1),
    lambda md: md.replace('data-spaces-tag="table"', 'data-spaces-tag="div"'),
])
def test_legacy_mutations_rejected(change):
    source = root('<table><tr><td rowspan="2">甲 $x_1$</td><td>乙</td></tr></table>')
    changed = change(convert_html_body(source, BASE))
    try:
        assert not validate_html_body(source, changed, BASE)["passed"]
    except MirrorError:
        pass


def test_broken_source_formula_is_retained_and_flagged_not_repaired():
    source = root('<p>原文 $x_1 + 2 未闭合</p>')
    markdown = convert_html_body(source, BASE)
    audit = validate_html_body(source, markdown, BASE)
    assert audit["passed"] and audit["formula_count"] is None
    assert audit["source_warnings"] and '$x_1 + 2 未闭合' in markdown
    assert not validate_html_body(source, markdown.replace('$x_1', '$x_2'), BASE)["passed"]


def test_active_embeds_never_execute_and_payload_is_audited():
    source = root('<script>alert("original")</script><iframe src="https://example.com/"></iframe>')
    markdown = convert_html_body(source, BASE)
    assert not BeautifulSoup(markdown, 'lxml').find(['script', 'iframe'])
    assert validate_html_body(source, markdown, BASE)["passed"]
    assert [n['tag'] for n in preview_html_body(markdown)] == ['attachment', 'attachment']
    assert not validate_html_body(source, markdown.replace('original', 'changed'), BASE)["passed"]


def test_unknown_tag_and_active_event_fail_closed():
    for body in ('<unknown>待审核</unknown>', '<p onclick="x()">待审核</p>'):
        with pytest.raises(MirrorError):
            convert_html_body(root(body), BASE)


def test_nested_footer_requires_same_notices_and_no_trailing_body():
    content = page().replace("<div id='PostContent'>", "<div id='PostContent'><blockquote>")
    content = content.replace("</div></div></div><aside>", "</div><div class='clear'></div></blockquote></div></div><aside>")
    extracted, _ = source_body(content, '9119')
    assert '结束段落' in extracted.get_text()
    assert not extracted.find(id='pay')
    changed = content.replace("</blockquote></div></div><aside>", "</blockquote><p>不能丢失的正文</p></div></div><aside>")
    with pytest.raises(MirrorError):
        source_body(changed, '9119')


def test_preview_adds_new_archive_id_without_changing_existing_metadata(tmp_path):
    original = [{"id": 1, "title": "已有元数据", "topics": ["数学工具"]}]
    (tmp_path / 'archive.json').write_text(json.dumps([{"id": 1, "title": "另一标题"},
        {"id": 2, "title": "深度学习基础", "url": "https://spaces.ac.cn/archives/2"}]), encoding='utf8')
    result = preview_posts(original, tmp_path)
    assert result[0] == original[0]
    assert len(result) == 2 and result[1]['topics']
