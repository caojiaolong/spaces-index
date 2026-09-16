"""Check that article reading shares the index shell and navigation state."""
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
    output = ROOT / ".cache/preview-check"
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "build/preview")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin + "/") else route.abort())
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))
            search = "#/explore?q=Muon&size=12&page=2"
            page.goto(origin + "/index.html" + search)
            link = page.locator('.post-card h3 a[href^="#/article/"]').nth(2)
            assert link.get_attribute("target") is None
            card = link.locator("xpath=ancestor::article")
            assert card.locator(".mirror-actions").inner_text().splitlines() == ["查看原文 ↗", "下载 Markdown", "复制 Markdown"]
            link.scroll_into_view_if_needed()
            page.evaluate("window.originalHeader = document.querySelector('#site-header')")
            position = page.evaluate("scrollY")
            link.click()
            page.locator("#reader-status").wait_for()
            page.wait_for_function("document.querySelector('#reader-status').textContent.includes('已排版')")
            assert page.evaluate("originalHeader === document.querySelector('#site-header')")
            assert page.locator('.main-nav a[data-nav="explore"]').get_attribute("aria-current") == "page"
            assert page.locator("#reader-back").get_attribute("href") == search
            assert page.locator(".site-footer").count() == 1
            assert page.locator("#article").is_visible()
            assert page.locator("#markdown-source").is_hidden()
            assert page.locator("#reader-actions .reader-file-actions").evaluate("n => Boolean(n.compareDocumentPosition(document.querySelector('#article')) & Node.DOCUMENT_POSITION_FOLLOWING)")
            page.evaluate("scrollTo(0, 1500)")
            page.locator('#back-to-top[aria-hidden="false"]').wait_for()
            page.screenshot(path=str(output / "reader-with-back-to-top.png"))
            page.locator("#back-to-top").click()
            page.wait_for_function("scrollY < 2")
            page.evaluate("document.querySelector('#article').scrollIntoView({block:'end',behavior:'instant'})")
            page.wait_for_function("document.querySelector('#reader-progress [role=progressbar]').getAttribute('aria-valuenow') === '100'")
            page.locator("#reader-back").click()
            page.locator("#explore-search").wait_for()
            assert page.locator("#explore-search").input_value() == "Muon"
            assert page.url.endswith(search)
            page.wait_for_function("expected => Math.abs(scrollY - expected) < 5", arg=position)
            page.go_back()
            page.locator("#reader-status").wait_for()
            page.wait_for_function("document.querySelector('#reader-status').textContent.includes('已排版')")
            assert page.locator("#reader-progress [role=progressbar]").get_attribute("aria-valuenow") == "100"
            assert page.evaluate("originalHeader === document.querySelector('#site-header')")
            # Consecutive articles share one MathJax instance but reset equation labels.
            page.evaluate("location.hash = '#/article/9119'")
            page.wait_for_function("document.querySelector('#reader-number')?.textContent.endsWith('09119') && document.querySelector('#reader-status').textContent.includes('已排版')")
            assert page.locator("#article mjx-container").count() == 142
            page.evaluate("location.hash = '#/article/11882'")
            page.wait_for_function("document.querySelector('#reader-number')?.textContent.endsWith('11882') && document.querySelector('#reader-status').textContent.includes('已排版')")
            assert page.locator("#article mjx-container").count() == 101
            assert page.locator('mjx-merror, [data-mml-node="merror"]').count() == 0
            assert page.locator('script[src="./vendor/mathjax/tex-chtml-nofont.js"]').count() == 1
            page.set_viewport_size({"width": 390, "height": 844})
            page.locator('.main-nav a[data-nav="series"]').wait_for()
            assert page.locator(".main-nav").evaluate("n => getComputedStyle(n).position") == "fixed"
            page.evaluate("scrollTo(0, 1500)")
            page.locator('#back-to-top[aria-hidden="false"]').wait_for()
            page.screenshot(path=str(output / "reader-mobile-navigation.png"))
            page.locator("#back-to-top").click()
            page.wait_for_function("scrollY < 2")
            page.locator('.main-nav a[data-nav="series"]').click()
            page.get_by_role("heading", name="系列书架", exact=True).wait_for()
            assert page.evaluate("originalHeader === document.querySelector('#site-header')")
            # Chapter query changes reuse the series DOM and retain TOC state.
            # Check the reported long series with motion enabled and reduced motion.
            series_route = '#/series/series-e48ccca641f8'
            for width, motion in ((1440, 'no-preference'), (390, 'reduce')):
                page.set_viewport_size({'width': width, 'height': 1000 if width == 1440 else 844})
                page.emulate_media(reduced_motion=motion)
                page.goto(origin + '/' + series_route + '?chapter=9271')
                page.locator('[data-chapter-post="9271"][aria-current="location"]').wait_for()
                page.wait_for_function("document.activeElement?.id === 'chapter-9271'")
                page.evaluate('''() => {
                  window.seriesView = document.querySelector('#main > .view');
                  window.seriesToc = document.querySelector('.chapter-nav');
                  window.chapterScrollCalls = [];
                  const native = Element.prototype.scrollIntoView;
                  Element.prototype.scrollIntoView = function(options) {
                    if (this.classList.contains('timeline-item')) chapterScrollCalls.push(options.behavior);
                    return native.call(this, options);
                  };
                }''')

                def wait_for_chapter(post_id):
                    page.wait_for_function('''id => {
                      const n = document.getElementById('chapter-' + id);
                      return document.activeElement === n && Math.abs(n.getBoundingClientRect().top - parseFloat(getComputedStyle(n).scrollMarginTop)) < 3;
                    }''', arg=post_id)
                    assert page.evaluate("seriesView === document.querySelector('#main > .view') && seriesToc === document.querySelector('.chapter-nav')")
                    assert page.locator('[data-chapter-post][aria-current="location"]').get_attribute('data-chapter-post') == post_id

                next_link = page.locator('[data-chapter-post="9280"]')
                next_link.scroll_into_view_if_needed()
                toc_scroll = page.locator('.chapter-nav').evaluate('n => n.scrollTop')
                next_link.click()
                wait_for_chapter('9280')
                assert page.locator('.chapter-nav').evaluate('n => n.scrollTop') == toc_scroll
                assert page.evaluate('chapterScrollCalls.at(-1)') == ('smooth' if motion == 'no-preference' else 'instant')
                assert page.url.endswith('?chapter=9280')
                page.go_back()
                wait_for_chapter('9271')
                page.go_forward()
                wait_for_chapter('9280')
                # Selecting the current chapter works even after manually scrolling away.
                page.evaluate('scrollBy(0, 350)')
                next_link.click()
                wait_for_chapter('9280')
                # Returning to an earlier chapter must locate its heading, even
                # if that hash has an old scroll position from manual scrolling.
                page.locator('[data-chapter-post="9271"]').click()
                wait_for_chapter('9271')
                page.locator('[data-chapter-post="9280"]').click()
                wait_for_chapter('9280')
                page.evaluate("seriesToc.open = false; location.hash = '#/series/series-e48ccca641f8?chapter=9305'")
                wait_for_chapter('9305')
                assert not page.locator('.chapter-nav').evaluate('n => n.open')
                # Unknown targets stay harmless and do not rebuild the page.
                page.evaluate("location.hash = '#/series/series-e48ccca641f8?chapter=missing'")
                page.wait_for_function("!document.querySelector('[data-chapter-post][aria-current]')")
                assert page.evaluate("seriesView === document.querySelector('#main > .view')")
                page.go_back()
                wait_for_chapter('9305')
                assert page.locator('.chapter-nav').evaluate('n => n.open') is False
                page.screenshot(path=str(output / f'series-chapter-{width}.png'))
                # Article links need the latest chapter as their return route.
                page.locator('#chapter-9305 h3 a').click()
                page.locator('#reader-actions:not([hidden])').wait_for()
                assert page.locator('#reader-back').get_attribute('href') == series_route + '?chapter=9305'
                page.locator('#reader-back').click()
                page.locator('[data-chapter-post="9305"][aria-current="location"]').wait_for()
                assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
                # Clearing the chapter also keeps the existing series view.
                page.evaluate("window.returnedSeries = document.querySelector('#main > .view'); location.hash = '#/series/series-e48ccca641f8'")
                page.wait_for_function("!document.querySelector('[data-chapter-post][aria-current]')")
                assert page.evaluate("returnedSeries === document.querySelector('#main > .view')")

            # Series chapter titles use the same reader entry as index cards.
            catalog = json.loads((ROOT / "build/preview/catalog.json").read_text(encoding="utf8"))
            series = next(post["seriesId"] for post in catalog["posts"] if post["id"] == "9119")
            page.goto(origin + "/#/series/" + series)
            chapter = page.locator('#chapter-9119 h3 a[href^="#/article/9119?"]')
            chapter.click()
            page.wait_for_function("document.querySelector('#reader-number')?.textContent.endsWith('09119') && document.querySelector('#reader-status').textContent.includes('已排版')")
            assert page.locator("#reader-back").get_attribute("href") == "#/series/" + series
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    result = {"shared_shell": True, "back_to_top_desktop_and_mobile": True, "search_and_scroll_restored": True,
              "read_state_shared": True, "title_opens_reader": True, "top_article_actions": True,
              "series_navigation": True, "formula_state_reset": True, "browser_errors": errors}
    result["series_chapters_reuse_dom_history_repeat_target_and_return_route"] = True
    (output / "navigation-results.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
