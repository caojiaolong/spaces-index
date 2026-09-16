"""Exercise formula copying and equation navigation in the real local reader.

All external requests are blocked. Run after building build/preview/.
"""
from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

try:
    from .mirror_content import math_spans
    from .paths import ROOT, ARTICLES_DIR, PREVIEW_DIR
except ImportError:
    from mirror_content import math_spans
    from paths import ROOT, ARTICLES_DIR, PREVIEW_DIR


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    output = ROOT / ".cache/preview-check"
    output.mkdir(parents=True, exist_ok=True)
    markdown = (ARTICLES_DIR / "11885/article.md").read_text(encoding="utf8")
    original_formulas = [markdown[a:b] for a, b in math_spans(markdown)]
    source = BeautifulSoup((ARTICLES_DIR / "11885/source.html").read_text(encoding="utf8"), "lxml")
    paragraph = source.select_one("#PostContent").find("p").get_text().strip()
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(PREVIEW_DIR)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            context = browser.new_context(permissions=["clipboard-read", "clipboard-write"],
                                          viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            context.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin + "/") else route.abort())
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))

            def ready(post_id):
                page.wait_for_function("id => document.querySelector('#reader-number')?.textContent.endsWith(id) && document.querySelector('#reader-status').textContent.includes('公式已排版')", arg=post_id)

            def clipboard():
                return page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n")

            def target_in_view(target_id):
                page.wait_for_function("id => { const n = document.getElementById(id)?.closest('mjx-container'); if (!n) return false; const y = n.closest('.math-scroll').getBoundingClientRect().top; const bottom = document.querySelector('#site-header').getBoundingClientRect().bottom; return y >= bottom - 1 && y < bottom + 60; }", arg=target_id)

            def returned_to(reference, position):
                page.wait_for_function("y => Math.abs(scrollY - y) < 3", arg=position)
                assert reference.evaluate("n => document.activeElement === n")

            page.goto(origin + "/#/article/11885?from=%23%2F")
            ready("11885")
            original = page.locator("#reader-original")
            assert original.is_visible()
            assert original.get_attribute("href") == "https://spaces.ac.cn/archives/11885"
            assert original.get_attribute("target") == "_blank"
            assert "noopener" in original.get_attribute("rel")
            assert page.locator("#equation-return").is_hidden()
            assert page.locator("#reader-actions .reader-file-actions").inner_text().splitlines() == ["查看原文 ↗", "下载 Markdown ↓", "复制 Markdown ⧉"]
            assert page.locator("#reader-actions").evaluate("n => n.getBoundingClientRect().bottom <= document.querySelector('#article').getBoundingClientRect().top")
            assert page.locator('#article mjx-container[jax="CHTML"]').count() == len(original_formulas)
            assert page.evaluate("[...document.fonts].filter(f => f.status === 'loaded' && f.family.startsWith('MJX')).length") >= 2
            fonts = page.evaluate("performance.getEntriesByType('resource').filter(r => r.name.endsWith('.woff2')).map(r => r.name)")
            assert fonts and all(url.startswith(origin + '/vendor/mathjax/') for url in fonts), fonts
            assert page.locator("#article mjx-container[data-latex]").evaluate_all("nodes => nodes.map(n => n.dataset.latex)") == original_formulas
            inline = page.locator('#article mjx-container:not([display="true"])[data-latex]').first
            assert inline.evaluate("n => parseFloat(getComputedStyle(n).marginInlineStart)") >= 3
            assert inline.evaluate("n => getComputedStyle(n).verticalAlign") != "middle"
            assert inline.evaluate("n => getComputedStyle(n).overflowX") == "visible"
            page.screenshot(path=str(output / "reader-math-toolbar.png"))

            # Formulas behave as readable content, with no click dialog or button.
            inline.click()
            assert page.locator("dialog[open]").count() == 0
            assert inline.get_attribute("role") == "math"
            page.keyboard.press("Control+c")
            assert clipboard() == original_formulas[0]
            inline.focus()
            page.keyboard.press("Enter")
            assert page.locator("dialog[open]").count() == 0

            # Native Ctrl+C preserves formula source in a mixed text selection.
            inline.evaluate("n => { const r = document.createRange(); r.selectNodeContents(n.closest('p')); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }")
            page.keyboard.press("Control+c")
            assert clipboard() == paragraph
            inline.evaluate("n => { const r = document.createRange(); r.selectNode(n); const s = getSelection(); s.removeAllRanges(); s.addRange(r); }")
            page.keyboard.press("Control+c")
            assert clipboard() == original_formulas[0]
            page.evaluate("getSelection().removeAllRanges()")

            equation = page.locator('#article mjx-container[display="true"]').nth(1)
            equation.click()
            equation_source = equation.get_attribute("data-latex")
            assert equation_source in original_formulas and r"\label{leq:base}" in equation_source
            page.keyboard.press("Control+c")
            assert clipboard() == equation_source

            reference = page.locator('#article a[data-equation="mjx-eqn:leq:base"]').nth(2)
            assert reference.get_attribute("aria-label") == "跳转到公式 (2)"
            target_id = reference.get_attribute("data-equation")
            deep_link = reference.get_attribute("href")
            route = page.url
            page.evaluate("window.originalArticle = document.querySelector('#article')")
            reference.scroll_into_view_if_needed()
            page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            reference_position = page.evaluate("scrollY")
            progress_before_preview = page.locator("#reading-percent").inner_text()
            reference.hover()
            preview = page.locator("#equation-preview")
            preview.wait_for(state="visible")
            assert preview.locator("strong").inner_text() == "公式 (2)"
            assert preview.locator("mjx-container").get_attribute("data-latex") == equation_source
            assert preview.locator("[id], [tabindex], a[href]").count() == 0
            assert page.evaluate("scrollY") == reference_position
            assert page.locator("#reading-percent").inner_text() == progress_before_preview
            box = preview.bounding_box()
            assert box['x'] >= 0 and box['x'] + box['width'] <= 1440
            assert preview.locator('.equation-preview-content').evaluate("n => n.scrollWidth <= n.clientWidth + 1")
            preview.hover()
            page.wait_for_timeout(200)
            assert preview.is_visible()
            page.screenshot(path=str(output / "reader-equation-preview.png"))
            page.keyboard.press("Escape")
            assert preview.is_hidden()
            reference.focus()
            page.keyboard.press("Tab")
            page.keyboard.press("Shift+Tab")
            preview.wait_for(state="visible")
            page.keyboard.press("Escape")
            assert preview.is_hidden()
            reference.hover()
            preview.wait_for(state="visible")
            page.emulate_media(reduced_motion="no-preference")
            page.evaluate("""() => {
              window.startMathScrollSampling = () => {
                window.mathScrollPositions = [];
                window.sampleMathScroll = true;
                function sample() {
                    if (!window.sampleMathScroll) return;
                    window.mathScrollPositions.push(scrollY);
                    requestAnimationFrame(sample);
                }
                requestAnimationFrame(sample);
              };
              window.startMathScrollSampling();
            }""")
            reference.click()
            assert preview.is_hidden()
            target_in_view(target_id)
            assert page.evaluate("new Set(mathScrollPositions.map(Math.round)).size") > 3
            page.evaluate("window.sampleMathScroll = false")
            assert page.url == route
            assert page.evaluate("originalArticle === document.querySelector('#article')")
            assert page.locator(".is-equation-target").count() == 1
            assert page.locator("dialog[open]").count() == 0
            page.screenshot(path=str(output / "reader-equation-target.png"))
            page.evaluate("startMathScrollSampling()")
            page.locator("#equation-return").click()
            returned_to(reference, reference_position)
            assert page.evaluate("new Set(mathScrollPositions.map(Math.round)).size") > 3
            page.evaluate("window.sampleMathScroll = false")
            assert page.locator("#equation-return").is_hidden()
            assert page.url == route
            page.emulate_media(reduced_motion="reduce")

            # Keyboard activation and successive references retain a return stack.
            reference.focus()
            reference_position = page.evaluate("scrollY")
            page.keyboard.press("Enter")
            target_in_view(target_id)
            other_reference = page.locator('#article a[data-equation="mjx-eqn:leq:base"]').last
            other_reference.scroll_into_view_if_needed()
            other_position = page.evaluate("scrollY")
            other_reference.click()
            target_in_view(target_id)
            page.locator("#equation-return").click()
            returned_to(other_reference, other_position)
            assert page.locator("#equation-return").is_visible()
            page.locator("#equation-return").click()
            returned_to(reference, reference_position)
            assert page.locator("#equation-return").is_hidden()

            # Reference URLs also work when opened separately or reloaded.
            page.goto(origin + "/" + deep_link)
            ready("11885")
            target_in_view(target_id)
            assert page.locator("#equation-return").is_hidden()
            page.evaluate("location.hash = '#/article/11882'")
            ready("11882")
            assert page.locator("#equation-return").is_hidden()
            second_reference = page.locator("#article a[data-equation]").first
            assert "#/article/11882?" in second_reference.get_attribute("href")
            second_target = second_reference.get_attribute("data-equation")
            second_reference.click()
            target_in_view(second_target)
            assert page.locator('script[src="./vendor/mathjax/tex-chtml-nofont.js"]').count() == 1

            # Native copying works on a touch-sized layout without writeText permission.
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(origin + "/#/article/11885")
            ready("11885")
            assert page.locator("#equation-preview").count() == 1
            assert page.locator("#equation-return").is_hidden()
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.locator("#reader-original").scroll_into_view_if_needed()
            assert page.locator("#reader-original").is_visible()
            page.screenshot(path=str(output / "reader-toolbar-mobile.png"))
            inline.click()
            page.evaluate("() => { navigator.clipboard.writeText = async () => { throw new Error('denied'); }; }")
            assert page.locator("dialog[open]").count() == 0
            page.screenshot(path=str(output / "reader-formula-mobile.png"))
            page.keyboard.press("Control+c")
            assert clipboard() == original_formulas[0]
            reference.scroll_into_view_if_needed()
            mobile_position = page.evaluate("scrollY")
            reference.hover()
            preview.wait_for(state="visible")
            box = preview.bounding_box()
            assert box['x'] >= 0 and box['x'] + box['width'] <= 390
            page.screenshot(path=str(output / "reader-equation-preview-mobile.png"))
            page.mouse.move(1, 1)
            preview.wait_for(state="hidden")
            reference.click()
            target_in_view(target_id)
            assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            page.screenshot(path=str(output / "reader-equation-return-mobile.png"))
            page.locator("#equation-return").click()
            returned_to(reference, mobile_position)
            reference.click()
            target_in_view(target_id)
            page.locator("#show-source").click()
            assert page.locator("#equation-return").is_hidden()
            assert page.locator("#markdown-source").input_value() == markdown
            page.locator("#show-reading").click()
            assert page.locator("#equation-return").is_visible()

            # Real mouse dragging must select characters inside a formula, not
            # just a programmatic Range around the whole MathJax container.
            page.goto(origin + "/#/article/9902")
            ready("09902")
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 1000 if width == 1440 else 844})
                formula = page.locator('#article mjx-container[display="true"]').first
                formula.scroll_into_view_if_needed()
                glyphs = formula.locator('mjx-math mjx-c')
                expected = "".join(glyphs.nth(i).inner_text() for i in range(4))
                assert expected == "\U0001d43f(\U0001d73d)"
                start, end = glyphs.nth(0).bounding_box(), glyphs.nth(3).bounding_box()
                y = start["y"] + start["height"] / 2
                page.mouse.move(start["x"] + 1, y)
                page.mouse.down()
                page.mouse.move(end["x"] + end["width"] - 1, y, steps=15)
                page.mouse.up()
                assert page.evaluate("getSelection().toString()") == expected
                page.keyboard.press("Control+c")
                assert clipboard() == expected
                page.screenshot(path=str(output / f"formula-selection-{width}.png"))
                page.evaluate("getSelection().removeAllRanges()")
                glyphs.nth(2).evaluate("n => {const s=getSelection(), r=document.createRange(); r.selectNodeContents(n); s.removeAllRanges(); s.addRange(r);}")
                page.keyboard.press("Control+c")
                assert clipboard() == "\U0001d73d"
                page.evaluate("getSelection().removeAllRanges()")
                formula.focus()
                page.keyboard.press("Control+c")
                assert clipboard() == formula.get_attribute("data-latex")
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            assert not errors, errors
            page.evaluate("location.hash = '#/explore'")
            page.wait_for_function("!document.querySelector('#equation-preview')")
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    result = {"original_button": True, "formula_sources_exact": len(original_formulas), "inline_spacing": True,
              "inline_and_display_copy": True, "selection_copy_includes_latex": True, "keyboard_and_mobile": True,
              "no_formula_dialog": True, "smooth_equation_jump": True, "smooth_return_to_reference": True,
              "successive_returns_and_focus": True, "top_actions": True, "equation_jump_and_deep_link": True, "article_switch": True,
              "mathjax_chtml_and_local_fonts": True, "markdown_unchanged": True, "browser_errors": errors}
    result["native_formula_partial_selection_desktop_and_mobile"] = True
    result["reference_hover_preview_without_scroll_or_progress_changes"] = True
    (output / "math-results.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
