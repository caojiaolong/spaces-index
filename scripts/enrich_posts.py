from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

try:
    from .common import (
        DATA_DIR,
        REQUEST_HEADERS,
        clean_text,
        looks_like_gate_page,
        log,
        read_json,
        sort_posts_desc,
        write_json,
    )
except ImportError:  # pragma: no cover - used when running as python scripts/enrich_posts.py
    from common import (
        DATA_DIR,
        REQUEST_HEADERS,
        clean_text,
        looks_like_gate_page,
        log,
        read_json,
        sort_posts_desc,
        write_json,
    )


def fetch_html(session: requests.Session, url: str, timeout: float = 30.0) -> str:
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code == 403 and looks_like_gate_page(response.text):
            response = session.get(url, timeout=timeout)
        elif looks_like_gate_page(response.text):
            response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch article page {url}: {exc}") from exc
    if looks_like_gate_page(response.text):
        raise RuntimeError(
            f"Article page {url} returned only a JavaScript redirect gate after retry."
        )
    return response.text


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
            if text:
                parts.append(text)
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
    return "source_category" in post and isinstance(post.get("source_tags"), list)


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


def enrich_posts(
    raw_posts: list[dict[str, Any]],
    cached_posts: list[dict[str, Any]] | None = None,
    *,
    force: bool = False,
    sleep_seconds: float = 0.5,
    session: requests.Session | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 25,
    progress_every: int = 25,
    refresh_summaries: bool = False,
) -> list[dict[str, Any]]:
    cached_posts = cached_posts or []
    cached_by_url = {str(post.get("url")): post for post in cached_posts if post.get("url")}
    cached_by_id = {str(post.get("id")): post for post in cached_posts if post.get("id")}
    owns_session = session is None
    session = session or requests.Session()
    session.headers.update(REQUEST_HEADERS)

    enriched: list[dict[str, Any]] = []
    try:
        sorted_raw_posts = sort_posts_desc(raw_posts)
        total = len(sorted_raw_posts)
        cached_count = 0
        fetched_count = 0
        log(
            "enrich_posts: "
            f"starting {total} posts; force={force}; "
            f"refresh_summaries={refresh_summaries}; sleep={sleep_seconds}s"
        )
        for index, raw in enumerate(sorted_raw_posts, start=1):
            cached = cached_by_url.get(str(raw.get("url"))) or cached_by_id.get(str(raw.get("id")))
            needs_summary_refresh = (
                refresh_summaries
                and cached is not None
                and has_cached_metadata(cached)
                and "source_summary" not in cached
            )
            if cached and has_cached_metadata(cached) and not force and not needs_summary_refresh:
                enriched.append(merge_cached_post(raw, cached))
                cached_count += 1
                if checkpoint_path and (index % checkpoint_every == 0 or index == total):
                    write_json(checkpoint_path, sort_posts_desc(enriched))
                if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
                    log(
                        "enrich_posts: "
                        f"{index}/{total} cached={cached_count} fetched={fetched_count} "
                        f"current={raw.get('id')} {raw.get('title')}"
                    )
                continue

            url = str(raw.get("url") or "")
            if not url:
                raise ValueError(f"Raw post is missing url: {raw!r}")
            html = fetch_html(session, url)
            try:
                metadata = parse_post_metadata(html)
            except ValueError as exc:
                raise ValueError(f"Failed to parse metadata for {url}: {exc}") from exc
            enriched.append({**raw, **metadata})
            fetched_count += 1
            if checkpoint_path and (index % checkpoint_every == 0 or index == total):
                write_json(checkpoint_path, sort_posts_desc(enriched))
            if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
                log(
                    "enrich_posts: "
                    f"{index}/{total} cached={cached_count} fetched={fetched_count} "
                    f"current={raw.get('id')} {raw.get('title')}"
                )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if owns_session:
            session.close()

    return sort_posts_desc(enriched)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch per-post category and tags.")
    parser.add_argument("--force", action="store_true", help="Ignore cached metadata.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between requests.")
    parser.add_argument(
        "--refresh-summaries",
        action="store_true",
        help="Fetch cached articles that do not yet have source_summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N posts. Set 0 to disable.",
    )
    parser.add_argument(
        "--input",
        default=str(DATA_DIR / "posts_raw.json"),
        help="Input raw posts JSON path.",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "posts.json"),
        help="Output enriched posts JSON path.",
    )
    args = parser.parse_args()

    raw_posts = read_json(Path(args.input), [])
    if not raw_posts:
        raise RuntimeError(f"No raw posts found in {args.input}. Run fetch_archive.py first.")
    cached_posts = read_json(Path(args.output), [])
    posts = enrich_posts(
        raw_posts,
        cached_posts,
        force=args.force,
        sleep_seconds=args.sleep,
        checkpoint_path=Path(args.output),
        progress_every=args.progress_every,
        refresh_summaries=args.refresh_summaries,
    )
    write_json(Path(args.output), posts)
    log(f"enrich_posts: completed {len(posts)} posts -> {args.output}")


if __name__ == "__main__":
    main()
