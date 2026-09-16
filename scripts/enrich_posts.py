"""Pure article metadata parsing; the old CLI delegates to the unified updater."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

try:
    from .common import clean_text
except ImportError:  # pragma: no cover - used when running as python scripts/enrich_posts.py
    from common import clean_text


def _find_meta_container(soup: BeautifulSoup) -> Tag | None:
    preferred = soup.select_one("#tools .cat, .tools .cat, span.cat")
    if isinstance(preferred, Tag):
        text = clean_text(preferred.get_text(" ", strip=True))
        if "分类" in text and "标签" in text:
            return preferred

    for node in soup.find_all(string=re.compile(r"分类\s*[:：]")):
        parent = node.parent
        while isinstance(parent, Tag):
            if parent.find_parent(id="PostContent") or parent.get("id") == "PostContent":
                break
            text = clean_text(parent.get_text(" ", strip=True))
            if "分类" in text and "标签" in text:
                return parent
            parent = parent.parent
    return None


def _links_by_href(container: Tag, marker: str) -> list[str]:
    values: list[str] = []
    for anchor in container.find_all("a", href=True):
        href = urljoin("https://spaces.ac.cn/", str(anchor.get("href") or ""))
        text = clean_text(anchor.get_text(" ", strip=True))
        if marker in href and text and "评论" not in text:
            values.append(text)
    return values


def _fallback_category(text: str) -> str:
    match = re.search(r"分类\s*[:：]\s*(.*?)\s*标签\s*[:：]", text)
    return clean_text(match.group(1)) if match else ""


def _fallback_tags(text: str) -> list[str]:
    match = re.search(r"标签\s*[:：]\s*(.*?)(?:\d+\s*评论|$)", text)
    if not match:
        return []
    raw = match.group(1)
    return [
        clean_text(part)
        for part in re.split(r"[,，、]\s*", raw)
        if clean_text(part) and "评论" not in clean_text(part)
    ]


SUMMARY_HEADINGS = {"小结", "文章小结", "总结", "结语", "结束语", "后记"}
SUMMARY_MAX_CHARS = 320
SUMMARY_STOP_MARKERS = (
    "转载到请包括本文地址",
    "更详细的转载事宜请参考",
    "科学空间FAQ",
)


def _heading_text(heading: Tag) -> str:
    clone = BeautifulSoup(str(heading), "lxml")
    for anchor in clone.find_all("a"):
        anchor.decompose()
    return clean_text(clone.get_text(" ", strip=True)).strip("#").strip()


def _heading_level(heading: Tag) -> int:
    match = re.match(r"h([1-6])", heading.name or "")
    return int(match.group(1)) if match else 6


def _truncate_text(text: str, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _strip_summary_boilerplate(text: str) -> tuple[str, bool]:
    stop_positions = [
        position
        for marker in SUMMARY_STOP_MARKERS
        if (position := text.find(marker)) >= 0
    ]
    if not stop_positions:
        return text, False
    return text[: min(stop_positions)].rstrip(), True


def summary_needs_refresh(post: dict[str, Any]) -> bool:
    summary = post.get("source_summary")
    return (
        "source_summary" not in post
        or (
            isinstance(summary, str)
            and any(marker in summary for marker in SUMMARY_STOP_MARKERS)
        )
    )


def extract_source_summary(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#PostContent")
    if not isinstance(content, Tag):
        return None

    for heading in content.find_all(["h2", "h3", "h4"]):
        title = _heading_text(heading)
        if title not in SUMMARY_HEADINGS:
            continue

        level = _heading_level(heading)
        parts: list[str] = []
        for sibling in heading.find_next_siblings():
            if isinstance(sibling, Tag) and re.match(r"h[1-6]", sibling.name or ""):
                if _heading_level(sibling) <= level:
                    break
            if not isinstance(sibling, Tag):
                continue
            if sibling.name not in {"p", "ul", "ol", "blockquote"}:
                continue
            text = clean_text(sibling.get_text(" ", strip=True))
            text, hit_stop_marker = _strip_summary_boilerplate(text)
            if text:
                parts.append(text)
            if hit_stop_marker:
                break
            if len(" ".join(parts)) >= SUMMARY_MAX_CHARS:
                break

        summary = _truncate_text(" ".join(parts))
        if summary:
            return summary
    return None


def parse_post_metadata(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    container = _find_meta_container(soup)
    if container is None:
        raise ValueError("Could not find article metadata line containing 分类 and 标签.")

    text = clean_text(container.get_text(" ", strip=True))
    categories = _links_by_href(container, "/category/")
    tags = _links_by_href(container, "/tag/")

    source_category = categories[0] if categories else _fallback_category(text)
    source_tags = tags if tags else _fallback_tags(text)
    if not source_category:
        raise ValueError(f"Could not parse source category from metadata line: {text!r}")

    return {
        "source_category": source_category,
        "source_tags": source_tags,
        "source_summary": extract_source_summary(html),
    }


def has_cached_metadata(post: dict[str, Any]) -> bool:
    return bool(post.get("source_category")) and isinstance(post.get("source_tags"), list)


def merge_cached_post(raw: dict[str, Any], cached: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cached)
    merged.update(
        {
            "id": raw.get("id"),
            "title": raw.get("title"),
            "url": raw.get("url"),
            "date": raw.get("date"),
        }
    )
    return merged


def main() -> None:
    # Preserve the old command name without maintaining a second HTTP pipeline.
    try:
        from .update_all import main as update
    except ImportError:
        from update_all import main as update
    print("enrich_posts.py now uses the unified updater; prefer scripts/update_all.py.")
    raise SystemExit(update())


if __name__ == "__main__":
    main()
