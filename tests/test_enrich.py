from scripts.enrich_posts import (
    extract_source_summary,
    parse_post_metadata,
    summary_needs_refresh,
)


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


def test_extract_source_summary_stops_before_reprint_notice():
    html = """
    <div id="PostContent">
      <h2>小结</h2>
      <p>本文尝试从第一性原理出发，探讨MoE中Router与Gate的设计问题。</p>
      <p>转载到请包括本文地址： https://spaces.ac.cn/archives/11782</p>
      <p>更详细的转载事宜请参考： 《科学空间FAQ》</p>
    </div>
    """

    assert extract_source_summary(html) == "本文尝试从第一性原理出发，探讨MoE中Router与Gate的设计问题。"


def test_summary_needs_refresh_covers_missing_and_dirty_values():
    assert summary_needs_refresh({}) is True
    assert summary_needs_refresh({"source_summary": None}) is False
    assert (
        summary_needs_refresh(
            {"source_summary": "正常小结。 转载到请包括本文地址： https://spaces.ac.cn/archives/11782"}
        )
        is True
    )
    assert summary_needs_refresh({"source_summary": "正常小结。"}) is False


def test_old_metadata_command_delegates_to_unified_updater(monkeypatch):
    import pytest
    from scripts import enrich_posts, update_all
    calls = []
    monkeypatch.setattr(update_all, "main", lambda: calls.append("update") or 0)
    with pytest.raises(SystemExit) as status:
        enrich_posts.main()
    assert status.value.code == 0
    assert calls == ["update"]
