"""Check source-backed captions and image zoom/pan offline in the reader."""
from __future__ import annotations

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
    output = ROOT / ".cache/preview-check"
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "build/preview")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    errors, results = [], {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
            # Real cached files must render with every external request blocked.
            cached_article = json.loads((ROOT / 'build/preview/mirror/9881/article.json').read_text(encoding='utf8'))
            if cached_article.get('images'):
                context.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin + '/') else route.abort())
                cached_page = context.new_page()
                cached_page.on('pageerror', lambda e: errors.append(str(e)))
                cached_page.goto(origin + '/#/article/9881')
                cached_picture = cached_page.locator('#article img').first
                cached_picture.scroll_into_view_if_needed()
                cached_page.wait_for_function("n => n.complete && n.naturalWidth > 0", arg=cached_picture.element_handle())
                assert '/mirror/9881/images/' in cached_picture.get_attribute('src')
                cached_picture.click()
                cached_page.wait_for_function("document.querySelector('#image-viewer-image').naturalWidth > 0")
                assert cached_page.locator('#image-viewer-original').get_attribute('href').startswith('https://spaces.ac.cn/')
                cached_page.close()
                context.unroute('**/*')
            # Stand-in image keeps dimensions realistic without contacting an image host.
            fixture = '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="240"><rect width="960" height="240" fill="#dfe4eb"/></svg>'
            tall_fixture = '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1800"><rect width="900" height="1800" fill="#dfe4eb"/><path d="M80 1400Q180 200 820 500M80 1500Q380 550 820 600" stroke="#596bd0" stroke-width="8" fill="none"/></svg>'
            tall_source = 'https://spaces.ac.cn/usr/uploads/2023/12/1641140949.png'
            tall_local = cached_article.get('images', {}).get(tall_source, '').removeprefix('./')
            def route_request(route):
                if route.request.resource_type == "image" and ('/mirror/' in route.request.url or not route.request.url.startswith(origin + "/")):
                    is_tall = '1641140949.png' in route.request.url or (tall_local and route.request.url == origin + '/' + tall_local)
                    route.fulfill(body=tall_fixture if is_tall else fixture, content_type="image/svg+xml")
                elif route.request.url.startswith(origin + "/"):
                    route.continue_()
                else:
                    route.abort()
            context.route("**/*", route_request)
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            for post_id in ("9881", "9119", "10017"):
                page.goto(origin + "/#/article/" + post_id)
                page.wait_for_function("document.querySelector('#reader-status')?.textContent.includes('已排版')")
                article = json.loads((ROOT / "build/preview/mirror" / post_id / "article.json").read_text(encoding="utf8"))
                captions = page.locator("#article .image-caption")
                assert captions.count() == len(article["captions"]) > 0
                assert captions.all_text_contents() == [c["text"] for c in article["captions"]]
                assert captions.evaluate_all("nodes => nodes.every(n => getComputedStyle(n).textAlign === 'center' && parseFloat(getComputedStyle(n).fontSize) < parseFloat(getComputedStyle(document.querySelector('#article')).fontSize))")
                for caption in captions.all():
                    figure = caption.locator(".."); picture = figure.locator("img")
                    assert figure.evaluate("n => n.tagName") == "FIGURE"
                    assert picture.count() == 1
                    assert picture.get_attribute("aria-describedby") == caption.get_attribute("id")
                if post_id == "9881":
                    assert captions.first.inner_text() == "几种扩散ODE-Solver示意图"
                captions.first.scroll_into_view_if_needed()
                picture = captions.first.locator("..").locator("img")
                page.wait_for_function("n => n.complete && n.naturalWidth > 0", arg=picture.element_handle())
                captions.first.scroll_into_view_if_needed()
                assert captions.first.evaluate("n => {const picture=n.parentElement.querySelector('img').getBoundingClientRect();const gap=n.getBoundingClientRect().top-picture.bottom; return gap >= 8 && gap <= 14;}")
                assert picture.evaluate("n => n.getBoundingClientRect().height <= innerHeight / 2 + 1")
                page.screenshot(path=str(output / f"caption-{post_id}-desktop.png"))
                page.set_viewport_size({"width": 390, "height": 844})
                captions.first.scroll_into_view_if_needed()
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
                page.screenshot(path=str(output / f"caption-{post_id}-mobile.png"))
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.get_by_role("button", name="Markdown 源码", exact=True).click()
                assert page.locator("#markdown-source").input_value() == (ROOT / "data/articles" / post_id / "article.md").read_text(encoding="utf8")
                results[post_id] = captions.count()

            # A tall source image stays within half the viewport in the article,
            # then opens larger without navigation or losing the reading position.
            page.goto(origin + "/#/article/9881")
            page.wait_for_function("document.querySelector('#reader-status')?.textContent.includes('已排版')")
            picture = page.locator('#article img').first
            for width in (1440, 390):
                page.set_viewport_size({"width": width, "height": 1000 if width == 1440 else 844})
                picture.scroll_into_view_if_needed()
                page.wait_for_function("n => n.complete && n.naturalWidth > 0", arg=picture.element_handle())
                picture.scroll_into_view_if_needed()
                before = picture.bounding_box()
                assert before['height'] <= page.viewport_size['height'] / 2 + 1
                assert abs(before['width'] / before['height'] - .5) < .01
                route = page.url
                picture.click()
                dialog = page.locator('#image-viewer')
                assert dialog.is_visible() and dialog.evaluate('n => n.open')
                assert page.locator('#image-viewer-caption').inner_text() == '几种扩散ODE-Solver示意图'
                expanded = page.locator('#image-viewer-image')
                page.wait_for_function("n => n.complete && n.naturalWidth > 0", arg=expanded.element_handle())
                assert expanded.bounding_box()['height'] > before['height']
                assert page.locator('#image-viewer-original').get_attribute('href') == picture.get_attribute('data-original-src')
                assert page.url == route
                page.screenshot(path=str(output / f'image-viewer-{width}.png'))
                page.locator('#image-viewer-size').click()
                page.wait_for_function("document.querySelector('#image-viewer-image').getBoundingClientRect().height >= 1799")
                assert page.locator('#image-viewer-scale').inner_text() == '100%'
                stage = page.locator('.image-viewer-stage')
                assert stage.evaluate("n => n.classList.contains('can-pan')")
                original = expanded.bounding_box()
                page.get_by_role('button', name='放大图片', exact=True).click()
                page.wait_for_function("h => Math.abs(document.querySelector('#image-viewer-image').getBoundingClientRect().height / h - 1.25) < .01", arg=original['height'])
                page.get_by_role('button', name='缩小图片', exact=True).click()
                page.wait_for_function("h => Math.abs(document.querySelector('#image-viewer-image').getBoundingClientRect().height - h) < 1", arg=original['height'])
                # Wheel zoom is anchored to the pointer, without scrolling the page.
                box = stage.bounding_box()
                # A fitting axis remains centered; test pointer anchoring on
                # both axes on mobile, and the overflowing vertical axis on desktop.
                pointer = {'x': box['x'] + box['width'] / 2 + (35 if width == 390 else 0),
                           'y': box['y'] + box['height'] / 2 + 45}
                scroll_before = page.evaluate('scrollY')
                page.mouse.move(**pointer)
                page.mouse.wheel(0, -150)
                page.wait_for_function("h => document.querySelector('#image-viewer-image').getBoundingClientRect().height > h * 1.2", arg=original['height'])
                zoomed = expanded.bounding_box()
                for axis, size in [('x', 'width'), ('y', 'height')]:
                    before_point = (pointer[axis] - original[axis]) / original[size]
                    after_point = (pointer[axis] - zoomed[axis]) / zoomed[size]
                    assert abs(before_point - after_point) < .001
                assert page.evaluate('scrollY') == scroll_before
                page.mouse.down()
                page.mouse.move(pointer['x'] + 30, pointer['y'] + 60, steps=8)
                assert stage.evaluate("n => n.classList.contains('is-dragging')")
                page.mouse.up()
                dragged = expanded.bounding_box()
                assert abs(dragged['y'] - zoomed['y'] - 60) < 1
                if width == 390:
                    assert abs(dragged['x'] - zoomed['x'] - 30) < 1
                assert not stage.evaluate("n => n.classList.contains('is-dragging')")
                page.screenshot(path=str(output / f'image-viewer-zoomed-{width}.png'))
                page.mouse.wheel(0, 150)
                page.wait_for_function("h => Math.abs(document.querySelector('#image-viewer-image').getBoundingClientRect().height - h) < 1", arg=original['height'])
                stage.focus()
                y_before_key = expanded.bounding_box()['y']
                page.keyboard.press('ArrowDown')
                page.wait_for_function("y => Math.abs(document.querySelector('#image-viewer-image').getBoundingClientRect().top - y + 40) < 1", arg=y_before_key)
                # Zoom limits remain usable, and fitting resets both scale and pan.
                for _ in range(12):
                    page.keyboard.press('+')
                assert page.get_by_role('button', name='放大图片', exact=True).is_disabled()
                assert page.locator('#image-viewer-scale').inner_text() == '500%'
                for _ in range(24):
                    page.keyboard.press('-')
                assert page.get_by_role('button', name='缩小图片', exact=True).is_disabled()
                page.locator('#image-viewer-size').click()
                page.wait_for_function("h => document.querySelector('#image-viewer-image').getBoundingClientRect().height > h / 2 && document.querySelector('#image-viewer-image').getBoundingClientRect().height <= h - 30", arg=box['height'])
                fitted = expanded.bounding_box()
                assert fitted['height'] <= box['height'] - 30
                assert abs(fitted['x'] + fitted['width'] / 2 - box['x'] - box['width'] / 2) < 1
                assert abs(fitted['y'] + fitted['height'] / 2 - box['y'] - box['height'] / 2) < 1
                assert not stage.evaluate("n => n.classList.contains('can-pan')")
                assert dialog.evaluate('n => n.scrollWidth <= n.clientWidth')
                page.keyboard.press('Escape')
                assert dialog.is_hidden()
                assert picture.evaluate('n => document.activeElement === n')
                # Keep the image's visual position even if delayed layout moves
                # preceding text slightly during a mobile resize.
                page.wait_for_function("expected => Math.abs(document.querySelector('#article img').getBoundingClientRect().top - expected) < 2", arg=before['y'])
                page.keyboard.press('Enter')
                assert dialog.is_visible()
                page.locator('#image-viewer-close').click()
                assert dialog.is_hidden()
                picture.click()
                page.mouse.click(1, 1)
                assert dialog.is_hidden()
                assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
            picture.click()
            page.evaluate("location.hash = '#/topics'")
            page.locator('.topic-groups').wait_for()
            assert page.locator('#image-viewer').count() == 0
            assert not page.evaluate("document.body.classList.contains('image-viewer-open')")

            # Without explicit source hints, an adjacent paragraph stays ordinary body text.
            def no_captions(route):
                data = route.fetch().json(); data["captions"] = []
                route.fulfill(json=data)
            page.route("**/mirror/9881/article.json", no_captions)
            page.goto(origin + "/#/article/9881")
            page.wait_for_function("document.querySelector('#reader-status')?.textContent.includes('已排版')")
            assert page.locator(".image-caption").count() == 0
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown(); server.server_close()
    result = {"captions": results, "source_text_and_markdown_unchanged": True, "no_guessed_captions": True,
              "half_viewport_image_height": True, "image_viewer_desktop_mobile_keyboard_and_cleanup": True,
              "zoom_buttons_wheel_pointer_anchor_drag_limits_and_reset": True,
              "mobile_overflow": False, "browser_errors": errors}
    (output / "caption-results.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
