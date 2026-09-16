"""Check article entry buttons, continuation links and the animated home surface.

Run against build/preview; all external requests are blocked.
"""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    site = ROOT / "build/preview"
    output = ROOT / ".cache/preview-check"
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((site / "catalog.json").read_text(encoding="utf8"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="no-preference")
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin + "/") else route.abort())
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))

            def ready():
                page.wait_for_function("document.querySelector('#reader-status')?.textContent.includes('已排版')")

            def shape():
                return page.locator(".sculpture-mesh path").first.get_attribute("d")

            def moving():
                page.wait_for_function("before => document.querySelector('.sculpture-mesh path').getAttribute('d') !== before", arg=shape())

            def still():
                before = shape()
                page.wait_for_timeout(220)
                assert shape() == before

            page.goto(origin)
            page.locator(".atlas-sculpture").wait_for()
            assert page.locator(".sculpture-toggle").count() == 0
            moving()
            node_count = page.locator(".sculpture-mesh path").count()
            page.emulate_media(reduced_motion="reduce")
            page.wait_for_timeout(100)
            still()
            # Sample two minutes of geometry, including the largest deformations.
            bounds = page.evaluate("""() => {
                const svg = document.querySelector('.atlas-sculpture');
                return Array.from({length: 25}, (_, i) => {
                    svg.drawSurface(i * 5);
                    const b = svg.querySelector('.sculpture-mesh').getBBox();
                    return {x: b.x, y: b.y, right: b.x + b.width, bottom: b.y + b.height};
                });
            }""")
            assert all(b["x"] >= 5 and b["right"] <= 515 and b["y"] >= 5 and b["bottom"] <= 365 for b in bounds), bounds
            for second in (0, 14, 28):
                page.locator(".atlas-sculpture").evaluate("(svg, second) => svg.drawSurface(second)", second)
                page.locator(".knowledge-sketch").screenshot(path=str(output / f"surface-{second}.png"))
            assert page.locator(".sculpture-mesh path").count() == node_count == 74
            page.emulate_media(reduced_motion="no-preference")
            moving()
            page.evaluate("scrollTo({top: document.body.scrollHeight, behavior: 'instant'})")
            page.wait_for_timeout(180)
            still()
            page.evaluate("scrollTo({top: 0, behavior: 'instant'})")
            moving()
            # Exercise the browser's visibility event without depending on OS focus.
            page.evaluate("Object.defineProperty(document, 'hidden', {value: true, configurable: true}); document.dispatchEvent(new Event('visibilitychange'))")
            still()
            page.evaluate("delete document.hidden; document.dispatchEvent(new Event('visibilitychange'))")
            moving()
            page.emulate_media(reduced_motion="reduce")
            page.wait_for_timeout(100)
            still()

            card = page.locator(".post-card").first
            arrow = card.locator("a.post-arrow")
            assert arrow.get_attribute("href") == card.locator("h3 a").get_attribute("href")
            assert arrow.get_attribute("aria-label").startswith("阅读文章：")
            assert arrow.evaluate("n => n.tabIndex") == 0
            arrow.scroll_into_view_if_needed()
            position = page.evaluate("scrollY")
            target = arrow.get_attribute("href")
            arrow.click()
            ready()
            assert page.url.endswith(target)
            page.go_back()
            page.locator(".post-card").first.wait_for()
            page.wait_for_function("y => Math.abs(scrollY - y) < 4", arg=position)
            arrow.focus()
            page.keyboard.press("Enter")
            ready()
            assert page.url.endswith(target)

            series = next(s for s in catalog["series"] if "9119" in s["postIds"])
            first_id, next_id = series["postIds"][:2]
            page.evaluate("id => localStorage.setItem('spaces-index-reading-progress-v1', JSON.stringify({[id]:100}))", first_id)
            page.goto(origin + "/#/series/" + series["id"])
            page.reload()
            continuation = page.locator(".continue-link").first
            continuation.wait_for()
            assert continuation.get_attribute("href").startswith(f"#/article/{next_id}?"), continuation.get_attribute("href")
            continuation.click()
            ready()
            assert page.locator("#reader-back").get_attribute("href") == "#/series/" + series["id"]
            page.goto(origin)
            banner_link = page.locator("#continue-reading a.continue-count")
            banner_link.wait_for()
            assert banner_link.get_attribute("href") == page.locator("#continue-reading a").first.get_attribute("href")
            banner_link.click()
            ready()
            assert f"#/article/{next_id}?" in page.url

            page.set_viewport_size({"width": 390, "height": 844})
            page.emulate_media(reduced_motion="no-preference")
            page.goto(origin)
            moving()
            assert page.locator(".sculpture-toggle").count() == 0
            page.screenshot(path=str(output / "surface-mobile.png"))
            page.emulate_media(reduced_motion="reduce")
            arrow.wait_for()
            assert arrow.is_visible()
            assert arrow.evaluate("n => n.getBoundingClientRect().width") >= 40
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            arrow.scroll_into_view_if_needed()
            page.screenshot(path=str(output / "article-arrow-mobile.png"))
            arrow.click()
            ready()
            assert page.locator("#article").is_visible()
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")

            # Missing mirrors retain a real, safely opened original-source link.
            missing_id = catalog["posts"][0]["id"]
            def hide_mirror(route):
                data = route.fetch().json()
                data["mirrors"].pop(missing_id, None)
                route.fulfill(json=data)
            page.route("**/catalog.json", hide_mirror)
            page.goto(origin)
            page.reload()
            arrow.wait_for()
            assert arrow.get_attribute("href") == f"https://spaces.ac.cn/archives/{missing_id}"
            assert arrow.get_attribute("target") == "_blank"
            assert "noopener" in arrow.get_attribute("rel")
            page.unroute("**/catalog.json")

            page.set_viewport_size({"width": 1440, "height": 1000})
            page.emulate_media(reduced_motion="no-preference")
            page.goto(origin)
            moving()
            page.evaluate("window.previousSurface = document.querySelector('.atlas-sculpture')")
            page.locator('.main-nav a[href="#/topics"]').click()
            page.locator(".topic-groups").wait_for()
            old_shape = page.evaluate("previousSurface.querySelector('.sculpture-mesh path').getAttribute('d')")
            page.wait_for_timeout(220)
            assert old_shape == page.evaluate("previousSurface.querySelector('.sculpture-mesh path').getAttribute('d')")
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    result = {"article_arrow_mouse_and_keyboard": True, "mobile_arrow": True, "continuation_opens_article": True,
              "scroll_restored": True, "missing_mirror_original_link": True, "surface_morphing": True,
              "automatic_pause_visibility_and_reduced_motion": True, "surface_stays_in_bounds": True,
              "no_animation_toggle": True,
              "animation_cleanup": True, "browser_errors": errors}
    (output / "home-interactions-results.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
