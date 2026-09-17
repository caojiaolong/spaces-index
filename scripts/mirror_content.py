"""Faithful HTML -> Markdown conversion, with an independent round trip audit.

Only the explicitly supported source layout is accepted. No heuristic extraction,
LLM, text normalisation, or rendered MathJax DOM is used as the article source.
"""
from __future__ import annotations

import hashlib
import html
import re
from copy import deepcopy
from urllib.parse import quote, urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from markdown_it import MarkdownIt

VERSION = 1
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-nd/2.5/cn/"
FAQ_URL = "https://spaces.ac.cn/archives/6508#文章如何转载/引用"
TOKEN = "SPACESMATHPLACEHOLDER"
CODE_TOKEN = "SPACESCODEPLACEHOLDER"
MD = MarkdownIt("commonmark", {"html": True, "typographer": False})
BLOCKS = {"p", "div", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"}
ALLOWED = BLOCKS | {"a", "img", "br", "strong", "b", "em", "i", "font", "span", "sup", "sub", "u", "pre", "code", "type"}


class MirrorError(ValueError):
    pass


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def absolute_url(value: str, base: str) -> str:
    target = urljoin(base, value)
    parts = urlsplit(target)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise MirrorError("Unsupported or unsafe source URL")
    if "\\" in target or any(ord(c) < 32 for c in target):
        raise MirrorError("Invalid source URL")
    return target


def source_body(page: str, post_id: str, *, check_supported: bool = True) -> tuple[Tag, dict]:
    soup = BeautifulSoup(page, "lxml")
    roots = soup.select(".Post > #PostContent")
    titles = soup.select(".Post .title-wrap h1")
    if len(roots) != 1 or len(titles) != 1:
        raise MirrorError("Expected one article title and one .Post > #PostContent")
    source = f"https://spaces.ac.cn/archives/{post_id}"
    title_link = titles[0].find("a", href=True)
    if not title_link or title_link["href"].rstrip("/") != source:
        raise MirrorError("Article identity mismatch")
    submitted = soup.select_one(".Post .submitted")
    if not submitted or "苏剑林" not in submitted.get_text():
        raise MirrorError("Author attribution not found")
    if not any(a.get("href", "").replace("http://", "https://") == LICENSE_URL
               for a in soup.find_all("a", href=True)):
        raise MirrorError("Expected source license not found; review source terms")
    root = deepcopy(roots[0])
    # These exact template suffixes are outside the article. Any changed layout
    # stops extraction instead of risking silent truncation of the final section.
    tips = root.find_all(id="content_tips")
    if len(tips) != 1:
        raise MirrorError("Unrecognised article footer boundary")
    footer_parent = tips[0].parent
    # Old malformed blockquotes can contain the complete, otherwise identical
    # footer. Accept only verified empty clear divs after it, never body text.
    def clear_only(node):
        return (isinstance(node, NavigableString) and not str(node).strip()) or (
            isinstance(node, Tag) and node.name == "div" and node.get("class") == ["clear"] and not node.contents)
    ancestor = footer_parent
    while ancestor is not root:
        if ancestor is None or ancestor.name not in {"div", "blockquote"}:
            raise MirrorError("Unrecognised article footer ancestry")
        following = list(ancestor.next_siblings)
        if not all(clear_only(n) for n in following):
            raise MirrorError("Unexpected content after nested article footer")
        for node in following:
            node.extract()
        ancestor = ancestor.parent
    children = list(footer_parent.find_all(recursive=False))
    while children and clear_only(children[-1]):
        children.pop().decompose()
    if [n.get("id") for n in children[-3:]] != ["content_tips", "pay", "how_to_cite"]:
        raise MirrorError("Unrecognised article footer boundary")
    notices = children[-5:-3]
    if len(notices) != 2 or not notices[0].get_text().startswith("转载到请包括本文地址：") or not notices[1].get_text().startswith("更详细的转载事宜请参考："):
        raise MirrorError("Source republication notices changed")
    for node in children[-5:]:
        node.decompose()
    for comment in root.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    for heading in root.find_all(re.compile(r"^h[1-6]$")):
        for a in heading.find_all("a", recursive=False):
            if a.get("href") == "#" + heading.get("id", "") and a.get_text().strip() == "#":
                a.decompose()
    if check_supported:
        validate_source(root)
    date = re.search(r"\d{4}-\d{2}-\d{2}", submitted.get_text())
    return root, {"id": post_id, "title": titles[0].get_text(), "author": "苏剑林",
                  "source_url": source, "date": date.group() if date else "",
                  "license": "CC BY-NC-ND 2.5 CN", "license_url": LICENSE_URL}


def validate_source(root: Tag) -> None:
    for node in root.find_all(True):
        if node.name not in ALLOWED:
            raise MirrorError(f"Unsupported body element: {node.name}")
        if any(k.startswith("on") for k in node.attrs) or node.has_attr("hidden"):
            raise MirrorError("Active or hidden body content requires review")
        if "display:none" in node.get("style", "").replace(" ", "").lower():
            raise MirrorError("Hidden body content requires review")
        if node.name == "img" and (not node.get("src") or node.has_attr("srcset") or node.has_attr("data-src")):
            raise MirrorError("Ambiguous image source requires review")
    if (not root.get_text(strip=True) and not root.find("img")) or "\ufffd" in root.get_text() or TOKEN in str(root) or CODE_TOKEN in str(root):
        raise MirrorError("Empty, damaged, or ambiguous source text")


def with_linebreaks(root: Tag) -> Tag:
    root = deepcopy(root)
    for br in root.find_all("br"):
        # A source <br> followed by a literal newline is one line break.
        following = br.next_sibling
        br.replace_with("" if isinstance(following, str) and following.startswith("\n") else "\n")
    return root


_MATH_TOKEN = re.compile(r"\\(?:\\|\$|\(|\)|\[|\]|begin\{[^}]+\}|end\{)|\$\$?")
_MATH_ENV = re.compile(r"\\(begin|end)\{([^}]+)\}")
_MATH_CLOSE = {"$": "$", "$$": "$$", r"\(": r"\)", r"\[": r"\]"}
_MATH_END = {start: re.compile(re.escape(end) + r"|\\[\s\S]?") for start, end in _MATH_CLOSE.items()}


def math_spans(text: str) -> list[tuple[int, int]]:
    """Scan raw TeX, retaining delimiters, environments and exact character order."""
    spans = []
    i = 0
    # Jump between possible delimiters. Scanning every prose character and
    # slicing text[i:] for each one made long articles disproportionately slow.
    while token := _MATH_TOKEN.search(text, i):
        start, i = token.start(), token.end()
        delimiter = token[0]
        if delimiter in (r"\$", r"\\"):
            continue
        if delimiter in _MATH_CLOSE:
            # Match the closing delimiter before escaped pairs, as in the
            # original scanner; an escaped backslash must not hide a later '$'.
            while closing := _MATH_END[delimiter].search(text, i):
                i = closing.end()
                if closing[0] == _MATH_CLOSE[delimiter]:
                    break
            else:
                raise MirrorError("Unclosed math delimiter")
            spans.append((start, i))
            continue
        if delimiter.startswith(r"\begin{"):
            stack = []
            for match in _MATH_ENV.finditer(text, start):
                kind, env = match.groups()
                if kind == "begin":
                    stack.append(env)
                elif not stack or stack.pop() != env:
                    raise MirrorError("Mismatched TeX environments")
                if not stack:
                    i = match.end()
                    break
            else:
                raise MirrorError("Unclosed TeX environment")
            spans.append((start, i))
            continue
        raise MirrorError("Orphaned TeX closing delimiter")
    return spans


def mask_math(text: str) -> tuple[str, list[str]]:
    spans = math_spans(text)
    formulas = [text[a:b] for a, b in spans]
    pieces, end = [], 0
    for index, (a, b) in enumerate(spans):
        pieces.extend((text[end:a], f"{TOKEN}{index}END"))
        end = b
    pieces.append(text[end:])
    return "".join(pieces), formulas


def restore_math(text: str, formulas: list[str]) -> str:
    return re.sub(TOKEN + r"(\d+)END", lambda m: formulas[int(m[1])], text)


def escape_markdown(text: str) -> str:
    # Escaping prose parentheses/brackets as \( / \[ would accidentally create
    # TeX delimiters. Parentheses are literal in prose; use entities for brackets.
    text = re.sub(r"([\\`*_{}<>#+.!|~-])", r"\\\1", text)
    return text.replace("[", "&#91;").replace("]", "&#93;")


def code_nodes(root: Tag):
    return [n for n in root.find_all(["pre", "code"]) if not n.find_parent(["pre", "code"])]


def without_code(root: Tag) -> Tag:
    root = deepcopy(root)
    for index, node in enumerate(code_nodes(root)):
        node.replace_with(f"{CODE_TOKEN}{index}END")
    return root


def convert_body(root: Tag, base: str) -> str:
    prepared = with_linebreaks(root)
    code_fragments = []
    for index, node in enumerate(code_nodes(prepared)):
        literal = html.escape(node.get_text(), quote=False)
        if node.name == "pre":
            code = node.find("code") or node
            language = next((c for c in code.get("class", []) if re.fullmatch(r"language-[\w+-]+", c)), None)
            attribute = f' class="{language}"' if language else ""
            fragment = f"\n\n<pre><code{attribute}>{literal}</code></pre>\n\n"
        else:
            fragment = f"<code>{literal}</code>"
        code_fragments.append(fragment)
        node.replace_with(f"{CODE_TOKEN}{index}END")
    masked, encoded_formulas = mask_math(str(prepared))
    formulas = [html.unescape(f) for f in encoded_formulas]
    # A TeX span crossing formatting markup cannot be safely converted this way.
    if any(re.search(r"</?[A-Za-z][^>]*>", f) for f in encoded_formulas):
        raise MirrorError("TeX crosses unsupported HTML markup")
    protected = BeautifulSoup(masked, "lxml").select_one("#PostContent")

    def render(node) -> str:
        if isinstance(node, NavigableString):
            return escape_markdown(str(node))
        content = "".join(render(child) for child in node.children)
        name = node.name
        if name in {"div", "p"}:
            return "\n\n" + content.strip() + "\n\n" if content.strip() else ""
        if re.fullmatch(r"h[1-6]", name):
            return "\n\n" + "#" * int(name[1]) + " " + content.strip() + "\n\n"
        if name in {"strong", "b"}:
            return "<strong>" + content + "</strong>"
        if name in {"em", "i"}:
            return "<em>" + content + "</em>"
        # A legacy traceback (4231) contains a literal <type> HTML element.
        # Preserve that source markup; do not guess or repair the intended text.
        if name in {"sup", "sub", "u", "type"}:
            return f"<{name}>" + content + f"</{name}>"
        if name == "blockquote":
            return "\n\n" + "\n".join("> " + line for line in content.strip().splitlines()) + "\n\n"
        if name == "a":
            if not node.get("href"):
                return content
            target = quote(absolute_url(node["href"], base), safe="/:#?&=%+@,;~!$'")
            return f"[{content}](<{target}>)"
        if name == "img":
            target = quote(absolute_url(node["src"], base), safe="/:#?&=%+@,;~!$'")
            alt = re.sub(r"([\\\[\]*`])", r"\\\1", node.get('alt', ''))
            return f"![{alt}](<{target}>)"
        if name in {"font", "span"}:
            return content
        raise MirrorError(f"Unsupported conversion element: {name}")

    markdown = re.sub(r"\n{3,}", "\n\n", render(protected)).strip()
    markdown = restore_math(markdown, formulas)
    markdown = re.sub(CODE_TOKEN + r"(\d+)END", lambda m: code_fragments[int(m[1])], markdown)
    return markdown.strip() + "\n"


def markdown_dom(markdown: str) -> tuple[Tag, list[str]]:
    # Literal HTML code is used because fences add a trailing newline and inline
    # backticks collapse whitespace. Neither is acceptable to an exact code audit.
    code_fragments = []
    def protect(match):
        token = f"{CODE_TOKEN}{len(code_fragments)}END"
        code_fragments.append(match[0])
        return token
    markdown = re.sub(r"<pre(?:\s[^>]*)?>[\s\S]*?</pre>|<code(?:\s[^>]*)?>[\s\S]*?</code>", protect, markdown)
    masked, formulas = mask_math(markdown)
    rendered = MD.render(masked)
    for index, fragment in enumerate(code_fragments):
        token = f"{CODE_TOKEN}{index}END"
        if fragment.startswith("<pre"):
            rendered = rendered.replace(f"<p>{token}</p>", fragment)
        rendered = rendered.replace(token, fragment)
    soup = BeautifulSoup(rendered, "lxml")
    return soup.body or soup, formulas


def semantic_stream(root: Tag, formulas: list[str] | None = None) -> str:
    """Independent read of DOM text and image positions; whitespace is collapsed,
    never removed (so an inserted/deleted space inside a word is still detected).
    """
    def walk(node):
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return str(node)
        if node.name == "img":
            return "IMAGE[" + node.get("src", "") + "][" + node.get("alt", "") + "]"
        if node.name in {"pre", "code"}:
            value = f"CODE[{node.name}:{digest(node.get_text())}]"
            return "\n" + value + "\n" if node.name == "pre" else value
        text = "".join(walk(child) for child in node.children)
        if node.name == "u":
            # Retain underline boundaries in the audit, including their position
            # among repeated words. Markdown has no native underline syntax.
            return "UNDERLINE[" + text + "]"
        return "\n" + text + "\n" if node.name in BLOCKS else text
    text = walk(root)
    if formulas is not None:
        text = restore_math(text, formulas)
    # Math is compared exactly in a separate sequence check. Hashes here retain
    # its position relative to every text and image, including repeated formulas.
    spans = math_spans(text)
    pieces, end = [], 0
    for a, b in spans:
        pieces.extend((text[end:a], "MATH[" + digest(text[a:b]) + "]"))
        end = b
    pieces.append(text[end:])
    return " ".join("".join(pieces).split())


def canonical_links(root: Tag, base: str, attr: str, tag: str) -> list[str]:
    return [quote(absolute_url(n[attr], base), safe="/:#?&=%+@,;~!$'")
            for n in root.find_all(tag) if n.get(attr)]


def validate_body(root: Tag, markdown: str, base: str) -> dict:
    source = with_linebreaks(root)
    # The source text audit does not reuse converter output or its DOM walk.
    _, source_math = mask_math(without_code(source).get_text())
    rendered, output_math = markdown_dom(markdown)
    for node in source.find_all("img"):
        node["src"] = quote(absolute_url(node["src"], base), safe="/:#?&=%+@,;~!$'")
    headings = lambda r: [(n.name, " ".join(n.get_text().split())) for n in r.find_all(re.compile(r"^h[1-6]$"))]
    for node in rendered.find_all(string=True):
        if TOKEN in node:
            node.replace_with(restore_math(str(node), output_math))
    source_stream, output_stream = semantic_stream(source), semantic_stream(rendered)
    source_headings, output_headings = headings(source), headings(rendered)
    checks = {
        "text_math_image_order": source_stream == output_stream,
        "latex_exact_sequence": source_math == output_math,
        "headings_exact_sequence": source_headings == output_headings,
        "links_exact_sequence": canonical_links(source, base, "href", "a") == canonical_links(rendered, base, "href", "a"),
        "images_exact_sequence": canonical_links(source, base, "src", "img") == canonical_links(rendered, base, "src", "img"),
        "code_exact_sequence": [(n.name, n.get_text()) for n in code_nodes(source)] == [(n.name, n.get_text()) for n in code_nodes(rendered)],
    }
    return {"passed": all(checks.values()), "checks": checks, "formula_count": len(source_math),
            "display_formula_count": sum(f.startswith((r"\begin", "$$", r"\[")) for f in source_math),
            "image_count": len(source.find_all("img")), "heading_count": len(source_headings),
            "code_block_count": len(source.find_all("pre")),
            "source_text_sha256": digest(source_stream),
            "output_text_sha256": digest(output_stream),
            "latex_sha256": digest("\0".join(source_math)),
            "body_sha256": digest(markdown), "converter_version": VERSION}


def attribution(meta: dict) -> str:
    return (f"# {escape_markdown(meta['title'])}\n\n"
            f"> 作者：{meta['author']} · 科学空间 · {meta['date']}\n>\n"
            f"> 原文：<{meta['source_url']}>\n>\n"
            f"> 原文许可：[CC BY-NC-ND 2.5 CN（署名-非商业性使用-禁止演绎）]({LICENSE_URL})。\n>\n"
            "> 本文件由 spaces-index 非官方、非商业项目进行必要的 HTML → Markdown 格式转换；正文未做摘要、润色、翻译或重组。项目不代表作者，也不表示作者为项目背书。\n>\n"
            "> 署名及附加许可说明不代表已获得作者额外授权。第三方素材权利依其原有声明。\n>\n"
            f"> 原站转载与引用说明：[科学空间 FAQ]({quote(FAQ_URL, safe=':/#')})\n\n---\n\n")


def source_image_captions(root: Tag) -> list[dict]:
    """Presentation hints from explicit source figures; never invent captions."""
    images = root.find_all("img")
    positions = {id(image): index for index, image in enumerate(images)}
    captions = []
    for figure in root.select(".typecho-caption"):
        pictures = figure.find_all("img")
        # Historical typecho captions sometimes omit the paragraph's class.
        paragraphs = figure.find_all("p", recursive=False)
        if len(pictures) != 1 or len(paragraphs) != 1:
            continue
        caption = paragraphs[0]
        text = caption.get_text()
        if text.strip() and caption.find_previous("img") is pictures[0]:
            captions.append({"imageIndex": positions[id(pictures[0])], "text": text})
    return captions


def preview_tree(markdown: str) -> list:
    root, formulas = markdown_dom(markdown)
    def walk(node):
        if isinstance(node, NavigableString):
            return restore_math(str(node), formulas)
        if node.name not in ALLOWED | {"hr"}:
            raise MirrorError(f"Unsupported preview node: {node.name}")
        attrs = {k: node[k] for k in ("href", "src", "alt") if node.has_attr(k)}
        return {"tag": node.name, "attrs": attrs, "children": [walk(c) for c in node.children]}
    return [walk(n) for n in root.children]


def convert_article(page: str, post_id: str) -> dict:
    root, meta = source_body(page.replace("\r\n", "\n").replace("\r", "\n"), post_id)
    body = convert_body(root, meta["source_url"])
    report = validate_body(root, body, meta["source_url"])
    markdown = attribution(meta) + body
    report["markdown_sha256"] = digest(markdown)
    report["source_html_sha256"] = digest(page)
    return {"metadata": meta, "body": body, "markdown": markdown, "validation": report,
            "images": canonical_links(root, meta["source_url"], "src", "img")}
