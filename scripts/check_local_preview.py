"""Headless acceptance of the full local preview; external requests are blocked."""
from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    site = ROOT / "build/preview"
    results_dir = ROOT / ".cache/preview-check"
    results_dir.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((site / "catalog.json").read_text(encoding="utf8"))
    assert catalog["localPreview"]
    assert set(catalog["mirrors"]) == {p["id"] for p in catalog["posts"]}
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    results = {"catalog_articles": len(catalog["posts"]), "readable_articles": len(catalog["mirrors"]), "samples": {}}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], viewport={"width": 1440, "height": 1000})
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin + "/") else route.abort())
            page = context.new_page()
            page.emulate_media(reduced_motion="reduce")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(origin)
            page.locator('.post-card h3 a[href^="#/article/"]').first.wait_for()
            page.screenshot(path=str(results_dir / "home-desktop.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            page.screenshot(path=str(results_dir / "home-mobile.png"))
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), page.evaluate("[...document.querySelectorAll('body *')].filter(n => n.getBoundingClientRect().right > innerWidth + 1).slice(0, 8).map(n => [n.tagName, n.className, n.getBoundingClientRect().right])")
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.locator('.post-card h3 a[href^="#/article/"]').first.click()
            page.locator("#reader-actions:not([hidden])").wait_for()
            assert "#/article/" in page.url
            for post_id in ("9902", "11885", "9119", "11767", "10922", "3511", "9938", "6877", "391", "25", "49", "1564"):
                page.goto(f"{origin}/index.html#/article/{post_id}")
                page.locator("#reader-status").wait_for()
                page.wait_for_function("document.querySelector('#reader-status').textContent.includes('已排版') || document.querySelector('#reader-status').textContent.includes('无法排版') || document.querySelector('#reader-status').textContent.includes('排版失败')", timeout=60000)
                assert page.locator("#reader-actions").is_visible(), post_id
                assert not page.locator("#article script, #article iframe, #article embed, #article object").count()
                expected = (ROOT / "data/articles" / post_id / "article.md").read_text(encoding="utf8")
                page.get_by_role("button", name="Markdown 源码", exact=True).click()
                assert page.locator("#markdown-source").input_value() == expected
                page.locator("#copy-markdown").click()
                page.wait_for_function("document.querySelector('#reader-copy-status').textContent.includes('已复制')")
                assert page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n") == expected
                with page.expect_download() as download:
                    page.locator("#download-markdown").click()
                assert Path(download.value.path()).read_bytes() == expected.encode("utf8")
                page.get_by_role("button", name="阅读", exact=True).click()
                counts = {"formulas": page.locator("mjx-container").count(), "formula_errors": page.locator('mjx-merror, [data-mml-node="merror"]').count(),
                          "tables": page.locator("#article table").count(), "code_blocks": page.locator("#article pre").count(),
                          "inert_embeds": page.locator(".source-attachment").count()}
                assert counts["formula_errors"] == 0, (post_id, counts)
                if post_id == "3511": assert counts["tables"] > 0
                if post_id == "9938": assert counts["code_blocks"] == 7
                if post_id == "25": assert counts["inert_embeds"] == 2
                if post_id == "49": assert page.locator(".source-warnings").is_visible()
                if post_id == "11885":
                    # View switching must not leave the last, hidden heading active.
                    page.evaluate("scrollTo(0, 0)")
                    page.wait_for_function("document.querySelector('#reader-toc a').getAttribute('aria-current') === 'location'")
                    toc = page.locator("#reader-toc a")
                    assert toc.count() == page.locator("#article h2, #article h3").count()
                    toc.nth(2).click()
                    page.wait_for_function("document.querySelectorAll('#reader-toc a')[2].getAttribute('aria-current') === 'location'")
                    assert int(page.locator("#reading-percent").inner_text().rstrip("%")) > 0
                    page.locator("#font-larger").click()
                    assert page.locator("#article").evaluate("n => getComputedStyle(n).fontSize") == "19px"
                    page.locator("#font-smaller").click()
                    assert page.locator("#series-pagination a").count() > 0
                    assert page.locator('#article a[href^="#/article/"]').count() > 0
                    page.evaluate("scrollTo(0, 0)")
                    while page.locator("html").get_attribute("data-theme") != "dark":
                        page.locator("#theme-toggle").click()
                    assert page.locator("html").get_attribute("data-theme") == "dark"
                    page.screenshot(path=str(results_dir / "reader-dark.png"))
                    page.locator("#theme-toggle").click()
                    results["reading_tools"] = {"toc": True, "progress": True, "font_size": True, "theme": True, "series_navigation": True}
                page.evaluate("scrollTo(0, 0)")
                if post_id in {"9902", "11885", "3511", "9938", "25"}:
                    page.screenshot(path=str(results_dir / f"{post_id}-desktop.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                page.wait_for_function("!document.querySelector('#toc-panel').open")
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth"), post_id
                if post_id in {"9902", "11885", "3511"}:
                    page.screenshot(path=str(results_dir / f"{post_id}-mobile.png"))
                page.set_viewport_size({"width": 1440, "height": 1000})
                results["samples"][post_id] = counts | {"copy_exact": True, "download_exact": True, "mobile_overflow": False}
            # Preserve the original exploration, topic and series page structure.
            for route, name in (("explore?q=Muon", "explore"), ("topics", "topics"), ("series", "series")):
                page.goto(f"{origin}/index.html#/{route}")
                page.locator("#main h1").wait_for()
                page.screenshot(path=str(results_dir / f"{name}-desktop.png"))
            page.goto(f"{origin}/reader.html?id=11885")
            page.locator("#reader-actions:not([hidden])").wait_for()
            page.locator("#series-pagination a").first.click()
            page.wait_for_url(lambda url: "#/article/11882?" in url)
            # Unknown article IDs remain unavailable even though numeric paths
            # are supported; the catalogue is the publication/preview boundary.
            page.goto(f"{origin}/reader.html?id=999999999")
            page.get_by_role("heading", name="Markdown 暂不可用").wait_for()
            assert page.locator("#reader-actions").is_hidden()
            assert page.locator("#reader-original").is_hidden()
            assert errors == [], errors
            results["browser_errors"] = errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (results_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
