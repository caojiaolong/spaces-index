"""Check AI handoffs, preferences and formula selections in the local reader.

No external requests are sent: AI provider navigation receives an inert local
response, and all other external requests are blocked. Build preview first.
"""
from __future__ import annotations

import argparse
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

try:
    from .paths import ROOT, ARTICLES_DIR, PREVIEW_DIR
except ImportError:
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
    markdown = (ARTICLES_DIR / "9119/article.md").read_text(encoding="utf8")
    source = BeautifulSoup((ARTICLES_DIR / "9119/source.html").read_text(encoding="utf8"), "lxml")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(PREVIEW_DIR)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors, navigations = [], []

    def route(request):
        url = request.request.url
        if url.startswith(origin + "/"):
            request.continue_()
        elif urlsplit(url).hostname in {"chatgpt.com", "www.perplexity.ai", "claude.ai", "gemini.google.com",
                                      "chat.deepseek.com", "www.kimi.com", "www.doubao.com"}:
            navigations.append(url)
            request.fulfill(content_type="text/html", body="<!doctype html><title>Local handoff check</title><p>No network request sent.</p>")
        else:
            request.abort()

    def question(url):
        return parse_qs(urlsplit(url).query).get("q", [None])[0]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=args.channel, headless=True)
            context = browser.new_context(permissions=["clipboard-read", "clipboard-write"], viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            context.route("**/*", route)
            page = context.new_page()
            page.on("pageerror", lambda error: errors.append(str(error)))

            def ready(article_id):
                page.wait_for_function("id => document.querySelector('#reader-number')?.textContent.endsWith(id) && document.querySelector('#reader-status')?.textContent.includes('公式已排版')", arg=article_id)

            def clipboard():
                return page.evaluate("navigator.clipboard.readText()").replace("\r\n", "\n")

            def select_contents(locator):
                locator.scroll_into_view_if_needed()
                locator.evaluate("n => {const r=document.createRange(), s=getSelection(); r.selectNodeContents(n); s.removeAllRanges(); s.addRange(r);}")
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                page.locator("#ask-gpt-selection").wait_for(state="visible")

            def selected_question():
                ask = page.locator("#ask-gpt-selection")
                ask.wait_for(state="visible")
                prompt = question(ask.get_attribute("href"))
                if prompt is not None:
                    assert len(ask.get_attribute("href")) <= 6000
                    return prompt
                ask.click()
                page.locator("#gpt-handoff[open]").wait_for()
                prompt = page.locator("#gpt-handoff-text").input_value()
                page.locator("#gpt-handoff-close").click()
                return prompt

            def excerpt(prompt):
                return prompt.split("【选段开始】\n\n", 1)[1].split("\n\n【选段结束】", 1)[0]

            def choose_provider(identifier):
                menu = page.locator("#gpt-options")
                if not menu.evaluate("n => n.open"):
                    page.locator("#gpt-options summary").click()
                page.locator(f"#ai-provider [data-ai-provider='{identifier}']").click()
                assert not menu.evaluate("n => n.open")

            page.goto(origin + "/#/article/9119")
            ready("09119")
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "chatgpt"
            assert "全文问 GPT" in page.locator("#ask-gpt").inner_text()
            page.locator("#gpt-options summary").click()
            options = page.locator("#ai-provider [role=radio]")
            assert options.count() == 7
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "chatgpt"
            for icon in page.locator("#ai-provider .ai-provider-icon").all():
                assert "/vendor/ai-icons/" in icon.evaluate("n => getComputedStyle(n).maskImage")
                assert icon.bounding_box()["width"] == 18
            for identifier in ("openai", "claude", "gemini", "deepseek", "kimi", "doubao", "perplexity"):
                response = page.request.get(f"{origin}/vendor/ai-icons/{identifier}.svg")
                assert response.status == 200 and "image/svg+xml" in response.headers["content-type"]
            page.screenshot(path=str(output / "reader-ai-menu-desktop.png"))
            # One tab stop in the radio group; arrows move the remembered choice,
            # Escape restores focus, and Enter confirms and closes the menu.
            options.first.focus()
            page.keyboard.press("ArrowRight")
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "claude"
            assert page.locator("#ai-provider [tabindex='0']").count() == 1
            page.keyboard.press("End")
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "perplexity"
            page.keyboard.press("Home")
            page.keyboard.press("Escape")
            assert not page.locator("#gpt-options").evaluate("n => n.open")
            assert page.locator("#gpt-options summary").evaluate("n => document.activeElement === n")
            page.keyboard.press("Enter")
            options.first.focus()
            page.keyboard.press("Enter")
            assert not page.locator("#gpt-options").evaluate("n => n.open")
            full = page.locator("#ask-gpt")
            full_prompt = question(full.get_attribute("href"))
            assert "https://caojiaolong.github.io/spaces-index/mirror/9119/article.md" in full_prompt
            assert "127.0.0.1" not in full_prompt and "#/article" not in full_prompt
            assert "苏剑林" in full_prompt and "https://spaces.ac.cn/archives/9119" in full_prompt
            assert "无法读取时明确说明" in full_prompt and "等待我的具体问题" in full_prompt
            assert "联网工具获取上面的完整 Markdown 正文" in full_prompt
            assert "https://raw.githubusercontent.com/caojiaolong/spaces-index/main/data/articles/9119/article.md" in full_prompt
            assert "article.txt" not in full_prompt
            assert markdown not in full_prompt
            page.screenshot(path=str(output / "reader-gpt-toolbar.png"))
            with context.expect_page() as opened:
                full.click()
            opened.value.wait_for_load_state()
            assert question(opened.value.url) == full_prompt
            assert opened.value.evaluate("opener === null")
            opened.value.close()

            # Plain text and special characters are copied literally; selecting
            # does not contact ChatGPT or modify ordinary clipboard behavior.
            paragraph = page.locator("#article p").filter(has_text="反复执行").first
            before = len(navigations)
            select_contents(paragraph)
            expected = paragraph.inner_text()
            assert excerpt(selected_question()) == expected
            assert len(navigations) == before
            page.keyboard.press("Control+c")
            assert clipboard() == expected
            select_contents(paragraph)
            page.screenshot(path=str(output / "reader-gpt-selection.png"))
            page.keyboard.press("Escape")
            assert page.locator(".selection-actions").is_hidden()
            select_contents(paragraph)
            with context.expect_page() as opened:
                page.locator("#ask-gpt-selection").click()
            opened.value.wait_for_load_state()
            assert excerpt(question(opened.value.url)) == expected
            opened.value.close()

            # Native keyboard selection and Enter activation use the same
            # prepared excerpt without moving focus or replacing the selection.
            page.bring_to_front()
            paragraph.scroll_into_view_if_needed()
            paragraph.evaluate("n => {const r=document.createRange(),s=getSelection();r.setStart(n.firstChild,0);r.setEnd(n.firstChild,1);s.removeAllRanges();s.addRange(r);}")
            page.keyboard.press("Shift+ArrowRight")
            page.keyboard.press("Shift+ArrowRight")
            page.locator("#ask-gpt-selection").wait_for(state="visible")
            keyboard_text = page.evaluate("getSelection().toString()")
            assert excerpt(selected_question()) == keyboard_text
            page.locator("#ask-gpt-selection").focus()
            with context.expect_page() as opened:
                page.keyboard.press("Enter")
            opened.value.wait_for_load_state()
            assert excerpt(question(opened.value.url)) == keyboard_text
            opened.value.close()

            ddpm = page.locator("#article p").filter(has_text="具体来说，DDPM将“拆楼”的过程建模为").first
            ddpm_text = next(n.get_text() for n in source.select("#PostContent p") if n.get_text().startswith("具体来说，DDPM"))
            select_contents(ddpm)
            assert excerpt(selected_question()).replace("\n", "") == ddpm_text.replace("\n", "")
            formula = ddpm.locator('mjx-container[display="true"]')
            latex = formula.get_attribute("data-latex")
            select_contents(formula)
            prompt = selected_question()
            assert excerpt(prompt) == latex, {"selected": excerpt(prompt), "latex": latex}
            assert prompt.count(latex) == 1
            assert "所属的完整原始公式" not in prompt

            glyph = formula.locator("mjx-math mjx-c").first
            select_contents(glyph)
            prompt = selected_question()
            assert excerpt(prompt) == glyph.inner_text()
            assert "所属的完整原始公式" in prompt and prompt.count(latex) == 1
            select_contents(glyph)
            page.keyboard.press("Control+c")
            assert clipboard() == glyph.inner_text()

            # A selection spanning paragraphs keeps both endpoints and source.
            ddpm.evaluate("""n => {
              const r=document.createRange(), s=getSelection();
              r.setStart(n.firstChild, 4); r.setEnd(n.nextElementSibling.firstChild, 6);
              s.removeAllRanges(); s.addRange(r);
            }""")
            page.locator("#ask-gpt-selection").wait_for(state="visible")
            multi = excerpt(selected_question())
            assert multi.startswith(ddpm_text[4:14]) and latex in multi
            assert multi.endswith(ddpm.evaluate("n => n.nextElementSibling.firstChild.textContent.slice(0,6)"))

            # Full-copy fallback contains the verified Markdown unchanged and
            # permits downloading the same file from the modal.
            page.locator("#gpt-options summary").click()
            page.locator("#copy-gpt-full").click()
            page.wait_for_function("document.querySelector('#gpt-handoff-status').textContent.includes('已复制')")
            whole = page.locator("#gpt-handoff-text").input_value()
            assert whole.endswith(markdown)
            assert clipboard() == whole
            with page.expect_download() as downloaded:
                page.locator("#gpt-handoff-download").click()
            assert Path(downloaded.value.path()).read_bytes() == markdown.encode("utf8")
            page.keyboard.press("Escape")
            assert page.locator("#gpt-handoff").is_hidden()
            assert not page.locator("body").evaluate("n => n.classList.contains('gpt-handoff-open')")

            # The same exact prompt goes to Perplexity's official GET endpoint.
            # Switching preferences itself never contacts a provider.
            before = len(navigations)
            choose_provider("perplexity")
            assert len(navigations) == before
            assert "全文问 Perplexity" in full.inner_text()
            assert question(full.get_attribute("href")) == full_prompt
            with context.expect_page() as opened:
                full.click()
            opened.value.wait_for_load_state()
            assert opened.value.url.startswith("https://www.perplexity.ai/search?")
            assert question(opened.value.url) == full_prompt
            assert opened.value.evaluate("opener === null")
            opened.value.close()
            select_contents(formula)
            assert excerpt(selected_question()) == latex
            assert page.locator("#ask-gpt-selection").get_attribute("href").startswith("https://www.perplexity.ai/search?")

            # Providers without a confirmed webpage prefill offer complete
            # clipboard content and a native link to the chosen website.
            providers = {
                "claude": ("Claude", "https://claude.ai/new"),
                "gemini": ("Gemini", "https://gemini.google.com/app"),
                "deepseek": ("DeepSeek", "https://chat.deepseek.com/"),
                "kimi": ("Kimi", "https://www.kimi.com/"),
                "doubao": ("豆包", "https://www.doubao.com/chat/"),
            }
            for identifier, (name, url) in providers.items():
                before = len(navigations)
                choose_provider(identifier)
                assert name in full.inner_text() and len(navigations) == before
                full.click()
                page.locator("#gpt-handoff[open]").wait_for()
                page.wait_for_function("document.querySelector('#gpt-handoff-status').textContent.includes('已复制')")
                assert page.locator("#gpt-handoff-text").input_value() == full_prompt
                assert clipboard() == full_prompt
                assert name in page.locator("#gpt-handoff-title").inner_text()
                assert page.locator("#gpt-handoff-open").get_attribute("href") == url
                assert len(navigations) == before
                with context.expect_page() as opened:
                    page.locator("#gpt-handoff-open").click()
                opened.value.wait_for_load_state()
                assert opened.value.url == url and opened.value.evaluate("opener === null")
                opened.value.close()
                page.locator("#gpt-handoff-close").click()
                select_contents(formula)
                assert name in page.locator("#ask-gpt-selection").inner_text()
                page.locator("#ask-gpt-selection").click()
                page.wait_for_function("document.querySelector('#gpt-handoff-status').textContent.includes('已复制')")
                copied = page.locator("#gpt-handoff-text").input_value()
                assert excerpt(copied) == latex and copied.count(latex) == 1
                assert copied.endswith("请解释这一段文字。")
                assert clipboard() == copied
                page.locator("#gpt-handoff-close").click()

            choose_provider("gemini")
            page.reload()
            ready("09119")
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "gemini"
            assert "全文问 Gemini" in full.inner_text()
            page.evaluate("location.hash = '#/article/11882'")
            ready("11882")
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "gemini"
            assert "全文问 Gemini" in full.inner_text()
            page.evaluate("localStorage.setItem('spaces-reader-ai','obsolete-provider');location.hash='#/article/9119'")
            ready("09119")
            assert page.locator("#ai-provider [aria-checked=true]").get_attribute("data-ai-provider") == "chatgpt"
            assert options.count() == 7
            assert "全文问 GPT" in full.inner_text()
            choose_provider("chatgpt")

            # Very long selections are not truncated to fit a URL. Simulate
            # denied clipboard permissions and exercise the manual selection.
            page.locator("#article").evaluate("n => {const r=document.createRange(),s=getSelection();r.selectNodeContents(n);s.removeAllRanges();s.addRange(r);}")
            page.locator("#ask-gpt-selection").wait_for(state="visible")
            assert question(page.locator("#ask-gpt-selection").get_attribute("href")) is None
            page.locator("#ask-gpt-selection").click()
            content = page.locator("#gpt-handoff-text").input_value()
            assert latex in content and "文章小结" in content
            assert "没有截断" in page.locator("#gpt-handoff-note").inner_text()
            page.evaluate("() => {navigator.clipboard.writeText = async () => {throw Error('denied');};}")
            page.locator("#gpt-handoff-copy").click()
            page.wait_for_function("document.querySelector('#gpt-handoff-status').textContent.includes('浏览器未允许')")
            assert page.locator("#gpt-handoff-text").evaluate("n => n.selectionStart === 0 && n.selectionEnd === n.value.length")
            page.keyboard.press("Control+c")
            assert clipboard() == content
            page.screenshot(path=str(output / "reader-gpt-long-selection.png"))
            page.locator("#gpt-handoff-close").click()
            select_contents(paragraph)
            page.locator("#show-source").click()
            assert page.locator(".selection-actions").is_hidden()
            page.locator("#show-reading").click()
            select_contents(paragraph)
            page.evaluate("getSelection().removeAllRanges()")
            page.locator(".selection-actions").wait_for(state="hidden")
            page.evaluate("location.hash = '#/article/11882'")
            ready("11882")
            assert page.locator(".selection-actions").count() == 1
            assert page.locator(".selection-actions").is_hidden()
            assert "/mirror/11882/article.md" in question(page.locator("#ask-gpt").get_attribute("href"))
            page.evaluate("location.hash = '#/explore'")
            page.wait_for_function("!document.querySelector('.selection-actions')")

            # Actual touch events activate the frozen selection in a mobile
            # context; none of the context-menu events should be cancelled.
            mobile = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, reduced_motion="reduce", color_scheme="dark")
            mobile.route("**/*", route)
            phone = mobile.new_page()
            phone.on("pageerror", lambda error: errors.append(str(error)))
            phone.goto(origin + "/#/article/9119")
            phone.wait_for_function("document.querySelector('#reader-status')?.textContent.includes('公式已排版')")
            plain = phone.locator("#article p").filter(has_text="反复执行").first
            plain.scroll_into_view_if_needed()
            plain.evaluate("n => {const r=document.createRange(),s=getSelection();r.selectNodeContents(n);s.removeAllRanges();s.addRange(r);}")
            phone.locator("#ask-gpt-selection").wait_for(state="visible")
            box = phone.locator(".selection-actions").bounding_box()
            header = phone.locator("#site-header").bounding_box()
            nav = phone.locator(".main-nav").bounding_box()
            assert box["x"] >= 0 and box["x"] + box["width"] <= 390
            assert box["y"] >= header["y"] + header["height"] and box["y"] + box["height"] <= nav["y"]
            assert plain.evaluate("n => {const e=new MouseEvent('contextmenu',{bubbles:true,cancelable:true});n.dispatchEvent(e);return !e.defaultPrevented;}")
            phone.screenshot(path=str(output / "reader-gpt-selection-mobile.png"))
            with mobile.expect_page() as opened:
                phone.locator("#ask-gpt-selection").tap()
            opened.value.wait_for_load_state()
            assert excerpt(question(opened.value.url)) == plain.inner_text()
            opened.value.close()
            assert not phone.evaluate("document.documentElement.scrollWidth > innerWidth")
            phone.locator("#gpt-options summary").tap()
            phone.locator("#copy-gpt-full").tap()
            phone.locator("#gpt-handoff[open]").wait_for()
            assert phone.locator("#gpt-handoff-text").input_value().endswith(markdown)
            phone.screenshot(path=str(output / "reader-gpt-handoff-mobile.png"))
            phone.locator("#gpt-handoff-close").tap()
            phone.locator("#gpt-options summary").tap()
            phone.locator("#ai-provider [data-ai-provider='deepseek']").tap()
            assert "全文问 DeepSeek" in phone.locator("#ask-gpt").inner_text()
            phone.locator("#gpt-options summary").tap()
            menu_box = phone.locator(".reader-gpt-menu").bounding_box()
            assert menu_box["x"] >= 0 and menu_box["x"] + menu_box["width"] <= 390
            header = phone.locator("#site-header").bounding_box()
            nav = phone.locator(".main-nav").bounding_box()
            assert menu_box["y"] >= header["y"] + header["height"]
            assert menu_box["y"] + menu_box["height"] <= nav["y"]
            assert phone.locator("#ai-provider .ai-provider-name").last.is_visible()
            assert not phone.evaluate("document.documentElement.scrollWidth > innerWidth")
            phone.screenshot(path=str(output / "reader-ai-menu-mobile.png"))
            phone.locator("#gpt-options summary").tap()
            phone.set_viewport_size({"width": 320, "height": 740})
            phone.locator("#gpt-options summary").tap()
            phone.wait_for_function("() => {const r=document.querySelector('.reader-gpt-menu').getBoundingClientRect();return r.left >= 12 && r.right <= innerWidth - 12;}")
            assert not phone.evaluate("document.documentElement.scrollWidth > innerWidth")
            phone.screenshot(path=str(output / "reader-ai-menu-small-mobile.png"))
            phone.locator("#gpt-options summary").tap()
            phone.set_viewport_size({"width": 390, "height": 844})
            plain.scroll_into_view_if_needed()
            plain.evaluate("n => {const r=document.createRange(),s=getSelection();r.selectNodeContents(n);s.removeAllRanges();s.addRange(r);}")
            phone.locator("#ask-gpt-selection").wait_for(state="visible")
            assert "问 DeepSeek" in phone.locator("#ask-gpt-selection").inner_text()
            phone.locator("#ask-gpt-selection").tap()
            phone.locator("#gpt-handoff[open]").wait_for()
            assert excerpt(phone.locator("#gpt-handoff-text").input_value()) == plain.inner_text()
            assert phone.locator("#gpt-handoff-open").get_attribute("href") == "https://chat.deepseek.com/"
            phone.screenshot(path=str(output / "reader-ai-copy-mobile.png"))
            phone.set_viewport_size({"width": 320, "height": 740})
            phone.screenshot(path=str(output / "reader-ai-copy-small-mobile.png"))
            assert phone.locator("#gpt-handoff-title").evaluate("n => n.getBoundingClientRect().height <= parseFloat(getComputedStyle(n).lineHeight) + 1")
            assert not phone.evaluate("document.documentElement.scrollWidth > innerWidth")
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    results = {"public_markdown_links": True, "new_tab_without_opener": True, "plain_and_mixed_selection": True,
               "github_raw_fallback": True,
               "gpt_default_and_provider_preference": True, "perplexity_url_exact": True,
               "all_copy_provider_handoffs": True, "preference_reload_route_and_invalid_storage": True,
               "icon_picker_and_keyboard_navigation": True, "mobile_picker_viewport_bounds": True,
               "full_and_partial_formula_sources": True, "multi_paragraph_selection": True,
               "full_markdown_copy_and_download_exact": True, "long_selection_without_truncation": True,
               "manual_copy_when_denied": True, "source_switch_and_route_cleanup": True,
               "mobile_touch_and_native_context_menu": True, "external_requests_sent": 0, "browser_errors": errors}
    results["native_keyboard_selection_and_activation"] = True
    (output / "gpt-results.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
