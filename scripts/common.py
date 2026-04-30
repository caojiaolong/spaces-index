from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

BASE_URL = "https://spaces.ac.cn/"
ARCHIVE_URL = "https://spaces.ac.cn/content.html"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "scientific-spaces-index/1.0 metadata-only"
)

REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
    "Cache-Control": "no-cache",
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.replace("\xa0", " ").split())


def looks_like_gate_page(html: str) -> bool:
    compact = clean_text(html)
    return (
        len(compact) < 1000
        and "window.location.href" in compact
        and "<script" in compact.casefold()
    )


def post_sort_key(post: dict[str, Any]) -> tuple[str, int, str]:
    post_id = str(post.get("id") or "")
    numeric_id = int(post_id) if post_id.isdigit() else 0
    return (str(post.get("date") or ""), numeric_id, str(post.get("title") or ""))


def sort_posts_desc(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(posts, key=post_sort_key, reverse=True)


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)
