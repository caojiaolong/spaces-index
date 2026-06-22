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
