from scripts.enrich_posts import extract_source_summary, parse_post_metadata


def test_parse_post_metadata_prefers_tools_cat_over_body_text():
    html = """
    <html>
      <body>
        <div id="PostContent">
          <h3 id="分类：意味着什么？">分类：意味着什么？</h3>
          <p>这里的正文里也有标签 y 和分类问题。</p>
        </div>
        <div id="tools" class="tools">
          <span class="cat">
            分类：<a href="https://spaces.ac.cn/category/Mathematics">数学研究</a>
            &nbsp; 标签：
            <a href="https://spaces.ac.cn/tag/model/">模型</a>,
            <a href="https://spaces.ac.cn/tag/probability/">概率</a>
          </span>
        </div>
      </body>
    </html>
    """
    metadata = parse_post_metadata(html)
    assert metadata == {
        "source_category": "数学研究",
        "source_tags": ["模型", "概率"],
        "source_summary": None,
    }


def test_extract_source_summary_from_conclusion_heading():
    html = """
    <div id="PostContent">
      <h2>正文</h2>
      <p>很长的正文不应该被提取。</p>
      <h2>文章小结 <a href="#文章小结">#</a></h2>
      <p>这里只保存明确小结部分的短摘录。</p>
      <p>第二段也可以进入短摘录。</p>
      <h2>参考文献</h2>
      <p>参考文献不应该进入小结。</p>
    </div>
    """
    assert extract_source_summary(html) == "这里只保存明确小结部分的短摘录。 第二段也可以进入短摘录。"
