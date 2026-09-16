"""Exercise persistent reading progress using the preview; block external HTTP."""
from __future__ import annotations

import json
import threading
from urllib.parse import quote
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
KEY = "spaces-index-reading-progress-v1"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    output = ROOT / ".cache/preview-check"
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "build/preview")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors = []
    results = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce",
                                          permissions=["clipboard-read", "clipboard-write"])
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin + "/") else route.abort())
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))

            def ready(article_id):
                page.goto(f"{origin}/#/article/{article_id}?from=%23%2Fexplore")
                page.locator("#reader-progress [role=progressbar]").wait_for()
                page.wait_for_timeout(150)

            def value():
                return int(page.locator("#reader-progress [role=progressbar]").get_attribute("aria-valuenow"))

            def scroll_fraction(fraction):
                page.evaluate("""fraction => {
                    const r = document.querySelector('#article').getBoundingClientRect();
                    const start = Math.max(0, r.top + scrollY - document.querySelector('#site-header').getBoundingClientRect().bottom);
                    const end = r.bottom + scrollY - innerHeight;
                    scrollTo({top: start + (end - start) * fraction, behavior: 'instant'});
                }""", fraction)
                page.wait_for_timeout(120)

            page.goto(origin)
            page.evaluate("localStorage.setItem('spaces-index-read-posts-v1', '[\"9119\"]')")
            ready("9119")
            assert value() == 0 and not page.locator(".post-read-toggle").count()
            scroll_fraction(.4)
            partial_progress = value()
            assert 38 <= partial_progress <= 41, partial_progress
            page.evaluate("scrollTo({top: 0, behavior: 'instant'})")
            page.wait_for_timeout(120)
            assert value() == partial_progress
            page.locator("#show-source").click()
            page.locator("#markdown-source").evaluate("n => { n.scrollTop = n.scrollHeight; }")
            page.wait_for_timeout(120)
            assert value() == partial_progress
            page.locator("#show-reading").click()
            page.reload()
            page.locator("#reader-progress [role=progressbar]").wait_for()
            assert value() == partial_progress
            results["partial_progress_persisted_without_regression"] = partial_progress

            other = context.new_page()
            title = page.locator("#reader-title").inner_text()
            other.goto(origin + "/#/explore?q=" + quote(title))
            other.locator('[data-post-progress="9119"]').wait_for()
            assert other.locator('[data-post-progress="9119"]').get_attribute("aria-valuenow") == str(partial_progress)
            page.bring_to_front()
            scroll_fraction(.999)
            assert value() == 99, value()
            scroll_fraction(1)
            page.wait_for_function("document.querySelector('#reader-progress [role=progressbar]').getAttribute('aria-valuenow') === '100'")
            other.wait_for_function("document.querySelector('[data-post-progress=\"9119\"]').getAttribute('aria-valuenow') === '100'")
            other.goto(origin + "/#/explore?read=read")
            other.locator('[data-post-progress="9119"]').wait_for()
            assert other.locator(".post-card").count() == 1
            other.screenshot(path=str(output / "reading-progress-completed.png"))
            other.close()
            results["only_100_percent_counts_as_read_and_syncs_tabs"] = True

            ready("11882")
            assert value() == 0
            with page.expect_popup() as opened:
                page.locator("#reader-original").click()
            opened.value.close()
            page.locator("#copy-markdown").click()
            with page.expect_download():
                page.locator("#download-markdown").click()
            assert value() == 0
            scroll_fraction(.35)
            assert 33 <= value() <= 36
            page.locator("#reader-back").click()
            indicator = page.locator('[data-post-progress="11882"]')
            indicator.wait_for()
            assert 33 <= int(indicator.get_attribute("aria-valuenow")) <= 36
            assert not indicator.locator("xpath=ancestor::article").evaluate("n => n.classList.contains('is-read')")
            page.screenshot(path=str(output / "reading-progress-partial.png"))
            results["original_copy_download_do_not_mark_read"] = True

            catalog = json.loads((ROOT / "build/preview/catalog.json").read_text(encoding="utf8"))
            series = next(s for s in catalog["series"] if "9119" in s["postIds"])
            page.goto(origin + "/#/series/" + series["id"])
            page.locator('[data-post-progress="9119"]').wait_for()
            assert page.locator(".progress-label").first.inner_text().startswith("已读 1 /")
            results["series_completion_uses_finished_articles"] = True

            page.set_viewport_size({"width": 390, "height": 844})
            ready("11882")
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.screenshot(path=str(output / "reading-progress-mobile.png"))
            results["mobile_no_overflow"] = True
            assert not errors, errors
            results["browser_errors"] = errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    (output / "reading-progress-results.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
