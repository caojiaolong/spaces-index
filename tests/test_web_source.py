from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def test_web_shell_has_all_views_and_only_local_runtime_assets():
    soup = BeautifulSoup((WEB / "index.html").read_text(encoding="utf-8"), "html.parser")
    navigation = {
        link.get("data-nav"): link.get("href")
        for link in soup.select("nav[aria-label='主要导航'] a[data-nav]")
    }
    assert navigation == {
        "home": "#/",
        "explore": "#/explore",
        "topics": "#/topics",
        "series": "#/series",
        "about": "#/about",
    }
    assert soup.select_one("a.skip-link[href='#main']")
    assert not soup.select_one("main").has_attr("aria-live")
    assert soup.select_one("noscript a[href*='github.com/caojiaolong/spaces-index']")
    github_link = soup.select_one("a#github-link")
    assert github_link
    assert github_link.get("href") == "https://github.com/caojiaolong/spaces-index"
    assert github_link.get("target") == "_blank"
    assert github_link.get("rel") == ["noopener", "noreferrer"]
    assert soup.select_one("span#github-stars")
    back_to_top = soup.select_one("button#back-to-top")
    assert back_to_top
    assert back_to_top.get("aria-label") == "返回页面顶部"
    assert back_to_top.get("tabindex") == "-1"

    stylesheets = [link.get("href") for link in soup.select("link[rel='stylesheet']")]
    scripts = [script.get("src") for script in soup.select("script[src]")]
    assert stylesheets == ["./styles.css"]
    assert scripts == ["./app.js"]
    assert not soup.select_one("script[src='./app.js']").has_attr("type")


def test_web_app_uses_safe_dom_and_expected_interaction_contract():
    source = (WEB / "app.js").read_text(encoding="utf-8")
    for unsafe_api in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert unsafe_api not in source

    assert "const PAGE_SIZE = 24" in source
    assert 'fetch("./catalog.json"' in source
    assert 'location.protocol === "file:"' in source
    assert "uv run python -m http.server 8000 --directory _site" in source
    assert 'normalize("NFKC")' in source
    assert "tokenScore += 5" in source
    assert "tokenScore += 3" in source
    assert "tokenScore += 2" in source
    assert "tokenScore += 1" in source
    assert 'target: "_blank", rel: "noopener noreferrer"' in source
    assert 'skipLink.addEventListener("click"' in source
    assert "event.isComposing" in source
    assert "scrollAfterRender" in source
    assert "post.seriesId" in source
    assert '"aria-live": "polite"' in source
    assert "showSummary: true, showActions: true" in source
    assert 'text: "查看系列 →"' in source
    assert 'backToTop.addEventListener("click"' in source
    assert 'backToTop.style.setProperty("--scroll-progress"' in source
    assert 'panel?.classList.add("is-open")' in source
    assert "if (panel) panel.scrollTop = 0" in source
    assert 'const READ_POSTS_KEY = "spaces-index-read-posts-v1"' in source
    assert 'const GITHUB_REPOSITORY_API = "https://api.github.com/repos/caojiaolong/spaces-index"' in source
    assert "repository.stargazers_count" in source
    assert "GITHUB_STARS_CACHE_MS = 30 * 60 * 1000" in source
    assert "void loadGitHubStarCount()" in source
    assert 'label: "仅看非系列文章"' in source
    assert 'label: "只看已读"' in source
    assert 'label: "只看未读"' in source
    assert "localStorage.setItem(READ_POSTS_KEY" in source
    assert 'text: "已读状态只保存在当前浏览器，不上传到服务器。"' in source
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion: reduce" in styles
    assert "width: min(76vw, 280px)" in styles
    assert "grid-template-columns: repeat(2, minmax(0, 1fr))" in styles
    assert "height: 100dvh" in styles
    assert ".filters-panel.is-open" in styles
    assert "-webkit-overflow-scrolling: touch" in styles
    assert "body.drawer-open .view" in styles
    assert "-webkit-backdrop-filter: none" in styles
    assert "width: calc(100% - 20px - env(safe-area-inset-left) - env(safe-area-inset-right))" in styles
    assert "transform: translateX(-50%)" in styles
