from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests


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

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_FETCH_ATTEMPTS = 5
DEFAULT_FETCH_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)


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


def _retry_delay(attempt: int, retry_delays: tuple[float, ...]) -> float:
    if not retry_delays:
        return 0.0
    return retry_delays[min(attempt - 1, len(retry_delays) - 1)]


def fetch_html_with_retries(
    session: requests.Session,
    url: str,
    *,
    page_name: str,
    timeout: float = 30.0,
    max_attempts: int = DEFAULT_FETCH_ATTEMPTS,
    retry_delays: tuple[float, ...] = DEFAULT_FETCH_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    session.headers.update(REQUEST_HEADERS)
    last_error = "unknown error"

    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt == max_attempts:
                break
            delay = _retry_delay(attempt, retry_delays)
            log(
                f"fetch: {page_name} {url} attempt {attempt}/{max_attempts} "
                f"failed ({last_error}); retrying in {delay:g}s"
            )
            if delay > 0:
                sleep(delay)
            continue
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch {page_name} {url} after attempt "
                f"{attempt}/{max_attempts}: {type(exc).__name__}: {exc}"
            ) from exc

        if looks_like_gate_page(response.text):
            last_error = "JavaScript redirect gate"
            if attempt == max_attempts:
                break
            delay = _retry_delay(attempt, retry_delays)
            log(
                f"fetch: {page_name} {url} attempt {attempt}/{max_attempts} "
                f"returned {last_error}; retrying in {delay:g}s"
            )
            if delay > 0:
                sleep(delay)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            last_error = f"HTTP {response.status_code}"
            if attempt == max_attempts:
                break
            delay = _retry_delay(attempt, retry_delays)
            log(
                f"fetch: {page_name} {url} attempt {attempt}/{max_attempts} "
                f"returned {last_error}; retrying in {delay:g}s"
            )
            if delay > 0:
                sleep(delay)
            continue

        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch {page_name} {url} after attempt "
                f"{attempt}/{max_attempts}: HTTP {response.status_code}"
            ) from exc
        return response.text

    raise RuntimeError(
        f"Failed to fetch {page_name} {url} after {max_attempts} attempts: {last_error}"
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
