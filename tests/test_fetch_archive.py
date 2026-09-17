from bs4 import Tag

from scripts.fetch_archive import parse_archive


def test_parse_archive_keeps_article_titles_containing_comment_text():
    html = """
    <div>
      <h3>2013年 (共1篇)</h3>
      <ul>
        <li><span>01月</span>
          <ul>
            <li>
              20日:
              <a href="https://spaces.ac.cn/archives/1884">评论功能修复了</a>
              (<a href="https://spaces.ac.cn/archives/1884#PostComment">2</a>)
            </li>
          </ul>
        </li>
      </ul>
    </div>
    """

    posts = parse_archive(html)

    assert posts == [
        {
            "id": "1884",
            "title": "评论功能修复了",
            "url": "https://spaces.ac.cn/archives/1884",
            "date": "2013-01-20",
        }
    ]


def test_archive_tracks_year_and_month_without_rescanning_prior_elements(monkeypatch):
    def reject_rescan(*args, **kwargs):
        raise AssertionError("Archive discovery must not scan every earlier node")
    monkeypatch.setattr(Tag, "find_all_previous", reject_rescan)
    html = """
    <a href='/archives/9'>导航链接</a>
    <h2>2026年</h2><h3>最新文章</h3>
    <ul><li><span>09月</span><ul>
      <li>17日: <a href='/archives/12'>有公式的标题 $x$</a> (<a href='/archives/12#PostComment'>3</a>)</li>
      <li>17日: <a href='/archives/12'>重复链接</a></li>
    </ul></li><li><span>02月</span><ul>
      <li>30日: <a href='/archives/11'>错误日期</a></li>
      <li>28日: <a href='/archives/10'>另一月份</a></li>
    </ul></li></ul>
    <h4>2025年</h4><ul><li><span>12月</span><ul>
      <li>31日: <a href='/archives/8'>上一年</a></li>
    </ul></li></ul>
    """
    posts = parse_archive(html)
    assert [(p['id'], p['date']) for p in posts] == [
        ('12', '2026-09-17'), ('10', '2026-02-28'), ('8', '2025-12-31')]
    assert posts[0]['title'] == '有公式的标题 $x$'
