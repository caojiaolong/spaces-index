"""Optional local browser acceptance check; all external requests are blocked.

uv run --with playwright python scripts/check_mirror_browser.py --channel msedge
Requires an installed browser (or `playwright install chromium`).
"""
from __future__ import annotations

import argparse
import base64
import json
import re
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
    parser.add_argument("--channel", default="chromium")
    args = parser.parse_args()
    output = ROOT / ".cache" / "mirror" / "browser"
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "_site")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    results = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], viewport={"width": 1440, "height": 1000})
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin + "/") else route.abort())
            # Only the two known image requests receive an inline fixture. No
            # external network is used; production uses the original image URLs.
            pixel = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jPzQAAAAASUVORK5CYII=")
            for image_url in ("https://spaces.ac.cn/usr/uploads/2022/06/2782509104.jpg",
                              "https://spaces.ac.cn/usr/uploads/2022/06/403506617.jpeg"):
                context.route(image_url, lambda route: route.fulfill(body=pixel, content_type="image/png"))
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            for post_id, count in (("9119", 142), ("11882", 101)):
                page.goto(f"{origin}/reader.html?id={post_id}")
                page.locator("#reader-status").wait_for()
                page.wait_for_function("document.querySelector('#reader-status').textContent.includes('公式已排版')", timeout=30000)
                assert page.locator("mjx-container").count() == count
                assert page.locator('mjx-merror, [data-mml-node="merror"]').count() == 0
                for picture in page.locator("#article img").all():
                    picture.scroll_into_view_if_needed()
                    page.wait_for_function("n => n.complete && n.naturalWidth > 0", arg=picture.element_handle(), timeout=10000)
                page.evaluate("scrollTo(0, 0)")
                assert page.locator("#article img").evaluate_all("nodes => nodes.every(n => n.complete && n.naturalWidth > 0)")
                assert page.locator("#article img").count() == (2 if post_id == "9119" else 0)
                page.screenshot(path=str(output / f"{post_id}-desktop.png"))
                page.get_by_role("button", name="Markdown 源码", exact=True).click()
                expected = page.locator("#markdown-source").input_value()
                assert expected == (ROOT / "data/articles" / post_id / "article.md").read_text(encoding="utf-8")
                references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", expected))
                labels = set(page.evaluate("Object.keys(MathJax.startup.document.inputJax[0].parseOptions.tags.allLabels)"))
                assert references <= labels, references - labels
                page.locator("#copy-markdown").click()
                page.wait_for_function("document.querySelector('#reader-copy-status').textContent.includes('已复制')")
                # Windows clipboard transports text with CRLF. Content is identical.
                assert page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n") == expected
                with page.expect_download() as download:
                    page.locator("#download-markdown").click()
                assert Path(download.value.path()).read_bytes() == expected.encode("utf-8")
                page.set_viewport_size({"width": 390, "height": 844})
                page.get_by_role("button", name="阅读", exact=True).click()
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                page.evaluate("scrollTo(0, 0)")
                page.screenshot(path=str(output / f"{post_id}-mobile.png"))
                page.set_viewport_size({"width": 1440, "height": 1000})
                results[post_id] = {"rendered_formulas": count, "formula_errors": 0, "copy_exact": True,
                                    "download_exact": True, "mobile_overflow": False, "unresolved_references": 0}
            # A missing/withdrawn manifest entry must prevent even direct reader URLs.
            page.route("**/catalog.json", lambda route: route.fulfill(json={"mirrors": {}}))
            page.goto(f"{origin}/reader.html?id=9119")
            page.locator("#reader-status").wait_for()
            page.wait_for_function("document.querySelector('#reader-title').textContent === 'Markdown 暂不可用'")
            assert page.locator("#reader-actions").is_hidden()
            assert page.locator("#reader-original").is_hidden()
            assert page.locator("#article").inner_text() == ""
            page.unroute("**/catalog.json")
            # Corrupt the transported Markdown without changing its expected digest.
            def corrupt(route):
                data = route.fetch().json()
                data["markdown"] += "篡改"
                route.fulfill(json=data)
            page.route("**/mirror/9119/article.json", corrupt)
            page.goto(f"{origin}/reader.html?id=9119")
            page.locator("#reader-status").wait_for()
            page.wait_for_function("document.querySelector('#reader-title').textContent === 'Markdown 暂不可用'")
            assert page.locator("#reader-actions").is_hidden()
            assert page.locator("#reader-original").is_hidden()
            assert page.locator("#article").inner_text() == ""
            page.unroute("**/mirror/9119/article.json")
            # Permission-denied clipboard has a usable manual selection fallback.
            page.goto(f"{origin}/reader.html?id=9119")
            page.locator("#reader-status").wait_for()
            page.wait_for_function("document.querySelector('#reader-status').textContent.includes('公式已排版')")
            page.evaluate("() => { navigator.clipboard.writeText = async () => { throw new Error('denied'); }; }")
            page.locator("#copy-markdown").click()
            assert page.locator("#markdown-source").is_visible()
            assert page.locator("#markdown-source").evaluate("n => n.selectionEnd - n.selectionStart === n.value.length")
            page.goto(f"{origin}/index.html#/explore?q=DDPM")
            page.locator('a[href^="#/article/9119?"]').wait_for()
            assert page.locator('a[download="spaces-9119.md"]').count() == 1
            catalog = json.loads((ROOT / "_site/catalog.json").read_text(encoding="utf-8"))
            series = next(post["seriesId"] for post in catalog["posts"] if post["id"] == "9119")
            page.goto(f"{origin}/index.html#/series/{series}")
            page.locator('#chapter-9119 a[href^="#/article/9119?"]').wait_for()
            assert not errors, errors
            results["guards"] = {"withdrawn_hidden": True, "tampered_hidden": True, "clipboard_fallback": True,
                                 "catalog_entry": True, "javascript_errors": errors}
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (output / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
