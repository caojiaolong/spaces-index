from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag

try:
    from .common import (
        ARCHIVE_URL,
        BASE_URL,
        DATA_DIR,
        clean_text,
        fetch_html_with_retries,
        log,
        sort_posts_desc,
        write_json,
    )
except ImportError:  # pragma: no cover - used when running as python scripts/fetch_archive.py
    from common import (
        ARCHIVE_URL,
        BASE_URL,
        DATA_DIR,
        clean_text,
        fetch_html_with_retries,
        log,
        sort_posts_desc,
        write_json,
    )


ARCHIVE_RE = re.compile(r"/archives/(\d+)")


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def fetch_html(url: str, timeout: float = 30.0) -> str:
    session = requests.Session()
    try:
        return fetch_html_with_retries(
            session,
            url,
            page_name="archive page",
            timeout=timeout,
        )
    finally:
        session.close()


def _cached_text(tag: Tag, cache: dict[int, str]) -> str:
    key = id(tag)
    if key not in cache:
        cache[key] = clean_text(tag.get_text(" ", strip=True))
    return cache[key]


def _find_month(anchor: Tag, text_cache: dict[int, str]) -> int | None:
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        text = _cached_text(parent, text_cache)
        match = re.match(r"(\d{1,2})月\b", text)
        if match:
            return int(match.group(1))
    return None


def _find_day(anchor: Tag, text_cache: dict[int, str]) -> int | None:
    parent = anchor.find_parent("li")
    if not parent:
        return None
    text = _cached_text(parent, text_cache)
    match = re.search(r"(\d{1,2})日\s*[:：]", text)
    if match:
        return int(match.group(1))
    return None


def parse_archive(html: str, archive_url: str = ARCHIVE_URL) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "lxml")
    posts_by_url: dict[str, dict[str, Any]] = {}
    year = None
    text_cache: dict[int, str] = {}

    # Walk headings once; scanning every earlier node for each article made
    # archive discovery quadratic as the source library grew.
    for anchor in soup.find_all(["h2", "h3", "h4", "a"]):
        if anchor.name != "a":
            match = re.search(r"(\d{4})年", _cached_text(anchor, text_cache))
            if match:
                year = int(match.group(1))
            continue
        href = str(anchor.get("href") or "")
        match = ARCHIVE_RE.search(href)
        if not match:
            continue

        title = clean_text(anchor.get_text(" ", strip=True))
        if not title or title.isdigit():
            continue

        month = _find_month(anchor, text_cache)
        day = _find_day(anchor, text_cache)
        if not (year and month and day):
            continue

        try:
            parsed_date = date(year, month, day).isoformat()
        except ValueError:
            continue

        url = canonical_url(urljoin(archive_url or BASE_URL, href))
        post_id = match.group(1)
        posts_by_url.setdefault(
            url,
            {
                "id": post_id,
                "title": title,
                "url": url,
                "date": parsed_date,
            },
        )

    return sort_posts_desc(list(posts_by_url.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch spaces.ac.cn archive metadata.")
    parser.add_argument("--url", default=ARCHIVE_URL, help="Archive URL to fetch.")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "posts_raw.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    html = fetch_html(args.url)
    posts = parse_archive(html, args.url)
    if not posts:
        raise RuntimeError(
            f"No posts parsed from {args.url}. The site structure may have changed."
        )
    write_json(Path(args.output), posts)
    log(f"fetch_archive: parsed {len(posts)} posts -> {args.output}")


if __name__ == "__main__":
    main()
