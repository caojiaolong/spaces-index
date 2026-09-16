"""Lossless HTML-in-Markdown for legacy layouts; executable embeds stay inert.

Every source element's name/attributes and every text node are independently
round-tripped. This is a local conversion format, not a licence to publish.
"""
from __future__ import annotations

import html
import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

try:
    from .mirror_content import MirrorError, digest, math_spans, with_linebreaks, without_code
except ImportError:
    from mirror_content import MirrorError, digest, math_spans, with_linebreaks, without_code

FORMAT = "html-v1"
MARKER = '<div data-spaces-format="html-v1">'
PASSIVE = set("p div span a img br hr strong b em i u del s sup sub font center blockquote h1 h2 h3 h4 h5 h6 pre code table thead tbody tfoot tr th td caption ul ol li dl dt dd video audio source".split())
OPAQUE = {"script", "iframe", "embed", "object", "style"}
LEGACY = {"type", "unk", "u:0002", "font:0017", "hs", "trong", "fong"}
VOID = {"img", "br", "hr", "source"}


def source_url(value: str, base: str) -> str | None:
    if "\\" in value or any(ord(c) < 32 for c in value):
        return None
    target = urljoin(base, value)
    try:
        parts = urlsplit(target)
        if parts.scheme == "mailto" and parts.path and not parts.netloc:
            return target
        if parts.scheme in {"https", "http"} and parts.hostname and not parts.username and not parts.password:
            return target
    except ValueError:
        pass
    return None


def tag_name(name: str) -> str:
    if name in PASSIVE:
        return name
    if name in LEGACY:
        # Keep the original parser category: replacing an unknown inline tag by
        # a div can implicitly close its enclosing paragraph during round trip.
        return name
    raise MirrorError(f"Unreviewed legacy element: {name}")


def attributes(node: Tag, base: str) -> dict:
    result = {"data-spaces-tag": node.name,
              "data-spaces-attrs": json.dumps(node.attrs, ensure_ascii=False, sort_keys=True)}
    for key in ("href", "src"):
        if node.get(key):
            target = source_url(node[key], base)
            if target and (key == "href" or not target.startswith("mailto:")):
                result[key] = target
    for key in ("alt", "title", "colspan", "rowspan", "start", "value", "color"):
        if node.has_attr(key):
            result[key] = str(node[key])
    if node.has_attr("hidden") or "display:none" in node.get("style", "").replace(" ", "").lower():
        result["hidden"] = ""
    if node.name in {"video", "audio"}:
        result.update(controls="", preload="none")
    return result


def convert_html_body(root: Tag, base: str) -> str:
    def render(node):
        if isinstance(node, Comment):
            raise MirrorError("Unexpected comment in extracted source")
        if isinstance(node, NavigableString):
            return html.escape(str(node), quote=False)
        if node.name in OPAQUE:
            # Raw source remains readable/copyable inside escaped code. The
            # preview shows this as a collapsed attachment, never executes it.
            literal = html.escape(str(node), quote=False)
            return ('<details data-spaces-opaque="1"><summary>原文嵌入内容（'
                    + node.name + '；预览不执行）</summary><pre><code>' + literal + '</code></pre></details>')
        name = tag_name(node.name)
        if any(k.lower().startswith("on") for k in node.attrs):
            raise MirrorError("Unreviewed active event in article body")
        attrs = "".join(f' {k}="{html.escape(str(v), quote=True)}"' for k, v in attributes(node, base).items())
        children = "".join(render(child) for child in node.children)
        # lxml may place fallback text inside a void <source> element. Preserve
        # that source tree using a span instead of letting the parser move it.
        if name in VOID and children:
            name = "span"
        return f"<{name}{attrs}>" + ("" if name in VOID else children + f"</{name}>")
    return MARKER + "".join(render(n) for n in root.children) + "</div>\n"


def parse_body(markdown: str) -> Tag:
    if not markdown.startswith(MARKER):
        raise MirrorError("Missing legacy body marker")
    soup = BeautifulSoup(markdown, "lxml")
    roots = soup.select('div[data-spaces-format="html-v1"]')
    if len(roots) != 1 or any(n is not roots[0] and str(n).strip() for n in soup.body.children):
        raise MirrorError("Unexpected content outside legacy body")
    return roots[0]


def validate_html_body(root: Tag, markdown: str, base: str) -> dict:
    rendered = parse_body(markdown)

    def source_record(node):
        if isinstance(node, NavigableString):
            return ("text", str(node))
        if node.name in OPAQUE:
            return ("opaque", str(node))
        return (node.name, node.attrs, [source_record(c) for c in node.children])

    def output_record(node):
        if isinstance(node, NavigableString):
            return ("text", str(node))
        if node.get("data-spaces-opaque") == "1":
            code = node.select_one("pre > code")
            if node.name != "details" or not code:
                raise MirrorError("Changed inert embed")
            raw = code.get_text()
            parsed = BeautifulSoup(raw, "lxml").find(list(OPAQUE))
            if parsed is None or str(parsed) != raw:
                raise MirrorError("Changed embed payload")
            expected = ('<details data-spaces-opaque="1"><summary>原文嵌入内容（' + parsed.name
                        + '；预览不执行）</summary><pre><code>' + html.escape(raw, quote=False) + '</code></pre></details>')
            if str(node) != str(BeautifulSoup(expected, "lxml").find("details")):
                raise MirrorError("Changed embed presentation")
            return ("opaque", raw)
        name = node.get("data-spaces-tag")
        attrs = json.loads(node.get("data-spaces-attrs", "null"))
        if not isinstance(attrs, dict) or name not in PASSIVE | LEGACY:
            raise MirrorError("Unrecognised legacy record")
        expected_name = tag_name(name)
        if expected_name in VOID and node.contents:
            expected_name = "span"
        stub = BeautifulSoup("", "lxml").new_tag(name, attrs=attrs)
        if node.name != expected_name or node.attrs != attributes(stub, base):
            raise MirrorError("Changed or unsafe legacy attributes")
        return (name, attrs, [output_record(c) for c in node.children])

    original = [source_record(n) for n in root.children]
    restored = [output_record(n) for n in rendered.children]
    dump = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    exact = original == restored
    text = without_code(with_linebreaks(root)).get_text()
    warnings = []
    try:
        formulas = [text[a:b] for a, b in math_spans(text)]
        formula_count = len(formulas)
        display_count = sum(f.startswith((r"\begin", "$$", r"\[")) for f in formulas)
        latex_hash = digest("\0".join(formulas))
    except MirrorError as exc:
        # A malformed source formula is retained character for character. Do not
        # repair its delimiters or mislabel a partial count as a complete count.
        formula_count = display_count = None
        latex_hash = digest(text)
        warnings.append("原文公式定界符异常，已逐字保留，未补写或修正：" + str(exc))
    if root.find(list(OPAQUE)):
        warnings.append("原文含旧式或脚本嵌入内容，已保留其位置和源码；预览不执行。")
    if any(n.get(k) and not source_url(n[k], base) for n in root.find_all(True) for k in ("href", "src")):
        warnings.append("原文含格式异常的链接，已保留原始地址但不启用跳转。")
    if any(not n.get("src") for n in root.find_all("img")):
        warnings.append("原文有图片标签缺少地址，保留原位置；无法显示该图片。")
    return {"passed": exact, "checks": {"source_dom_exact": exact, "text_math_image_order": exact,
            "latex_exact_sequence": exact, "attributes_exact": exact, "code_exact_sequence": exact},
            "formula_count": formula_count, "display_formula_count": display_count,
            "image_count": len(root.find_all("img")), "heading_count": len(root.find_all(re.compile(r"^h[1-6]$"))),
            "code_block_count": len(root.find_all("pre")), "source_text_sha256": digest(dump(original)),
            "output_text_sha256": digest(dump(restored)), "latex_sha256": latex_hash,
            "body_sha256": digest(markdown), "converter_version": 1, "format": FORMAT,
            "source_warnings": warnings}


def preview_html_body(markdown: str) -> list:
    root = parse_body(markdown)
    def walk(node):
        if isinstance(node, NavigableString):
            return str(node)
        if node.get("data-spaces-opaque") == "1":
            raw = node.select_one("pre > code").get_text()
            source = BeautifulSoup(raw, "lxml").find(list(OPAQUE))
            return {"tag": "attachment", "attrs": {"kind": source.name, "source": raw,
                    "url": source.get("src", source.get("data", ""))}, "children": []}
        attrs = {k: v for k, v in node.attrs.items() if k in {"href", "src", "alt", "title", "colspan", "rowspan", "start", "value", "color", "hidden", "controls", "preload"}}
        if node.get("data-spaces-tag") == "a" and "href" not in attrs:
            attrs["title"] = "原文链接未启用：" + str(json.loads(node["data-spaces-attrs"]).get("href", ""))
        name = "span" if node.name in LEGACY else node.name
        return {"tag": name, "attrs": attrs, "children": [walk(c) for c in node.children]}
    return [walk(n) for n in root.children]
