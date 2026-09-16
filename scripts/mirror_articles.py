"""Shared serial HTTP transport and compatibility maintenance for selected articles.

Normal updates use update_all.py and the canonical content library.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests
try:
    from .common import ROOT, REQUEST_HEADERS, read_json, write_json
    from .mirror_content import MirrorError, convert_article, digest, source_body
    from .mirror_store import MIRROR_DIR, age_days, load_config, save_config, now_iso, verified_article
except ImportError:
    from common import ROOT, REQUEST_HEADERS, read_json, write_json
    from mirror_content import MirrorError, convert_article, digest, source_body
    from mirror_store import MIRROR_DIR, age_days, load_config, save_config, now_iso, verified_article

CACHE = ROOT / ".cache" / "mirror"
RETRYABLE = {429, 500, 502, 503, 504}


class DeferredRetry(MirrorError):
    def __init__(self, delay):
        self.retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(timespec="seconds")
        super().__init__(f"Server requested retry after {delay:.0f}s; defer until {self.retry_at}")


class SerialFetcher:
    def __init__(self, interval: float = 3, attempts: int = 3, *, session=None, sleep=time.sleep,
                 host="spaces.ac.cn", max_bytes=None):
        if interval < 3 or not 1 <= attempts <= 3:
            raise MirrorError("Article fetching requires --sleep >= 3 and 1 <= --attempts <= 3")
        self.interval, self.attempts, self.sleep = interval, attempts, sleep
        self.session = session or requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        self.session.headers["User-Agent"] = REQUEST_HEADERS["User-Agent"].replace(
            "metadata-only", "noncommercial-library (+https://github.com/caojiaolong/spaces-index)")
        self.robots = None
        if host not in {"spaces.ac.cn", "album.spaces.ac.cn", "bbs.spaces.ac.cn"}:
            raise MirrorError("Unsupported source host")
        self.host, self.max_bytes = host, max_bytes

    def get(self, url: str, headers: dict | None = None):
        parts = urlsplit(url)
        if (parts.hostname != self.host or parts.scheme != "https" or parts.username or parts.password
                or parts.port not in (None, 443)):
            raise MirrorError(f"Fetcher only requests https://{self.host}")
        if self.robots is not None and not self.robots.can_fetch("scientific-spaces-index", url):
            raise MirrorError("robots.txt disallows this URL")
        for attempt in range(self.attempts):
            self.sleep(self.interval)  # Also separates images, retries and first requests across runs.
            try:
                options = {"stream": True} if self.max_bytes else {}
                response = self.session.get(url, headers=headers or {}, timeout=(10, 45), allow_redirects=False, **options)
                if self.max_bytes:
                    try:
                        chunks, length = [], 0
                        for chunk in response.iter_content(64 * 1024):
                            length += len(chunk)
                            if length > self.max_bytes:
                                raise MirrorError("Image response exceeds size limit")
                            chunks.append(chunk)
                        response._content = b"".join(chunks)
                    finally:
                        response.close()
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt + 1 == self.attempts:
                    raise
                self.sleep(10 * 2**attempt)
                continue
            gate = len(response.content) < 1000 and b"window.location.href" in response.content
            if response.status_code not in RETRYABLE and not gate:
                return response
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError):
                    delay = 0
            # Do not occupy a job for hours; retry on a later invocation instead.
            if delay > 300 or (attempt + 1 == self.attempts and delay > 0):
                raise DeferredRetry(delay)
            if attempt + 1 == self.attempts:
                raise MirrorError(f"Fetch failed after {self.attempts} attempts (HTTP {response.status_code}, gate={gate})")
            self.sleep(max(delay, 10 * 2**attempt))
        raise MirrorError("Unreachable fetch state")

    def check_robots(self):
        cache_path = CACHE / ("robots.json" if self.host == "spaces.ac.cn" else f"robots-{self.host}.json")
        cached = read_json(cache_path, {})
        if not cached or age_days(cached["checked_at"]) >= 7:
            response = self.get(f"https://{self.host}/robots.txt")
            if response.status_code == 404:
                rules = "User-agent: *\nDisallow:\n"
            elif response.status_code == 200:
                rules = response.content.decode("utf-8-sig", errors="strict")
                if "<html" in rules.lower():
                    raise MirrorError("Invalid robots.txt response")
            else:
                raise MirrorError(f"Cannot check robots.txt: HTTP {response.status_code}")
            cached = {"checked_at": now_iso(), "text": rules}
            write_json(cache_path, cached)
        self.robots = RobotFileParser()
        self.robots.parse(cached["text"].splitlines())
        self.interval = max(self.interval, self.robots.crawl_delay("scientific-spaces-index") or 0)


def withdraw(directory: Path, post_id: str):
    # Only delete known generated files below this article, never arbitrary paths.
    folder = (directory / post_id).resolve()
    if not re.fullmatch(r"[1-9]\d*", post_id) or folder.parent != directory.resolve():
        raise MirrorError("Unsafe withdrawal target")
    for name in ("article.md", "source.html", "report.json", "images.json"):
        (folder / name).unlink(missing_ok=True)
    image_dir = folder / "images"
    if image_dir.is_dir():
        if not image_dir.resolve().is_relative_to(folder):
            raise MirrorError("Image directory escapes withdrawal target")
        for file in image_dir.iterdir():
            if file.is_file():
                file.unlink()


def save_article(directory: Path, post_id: str, page: str, fetched_at: str):
    article = convert_article(page, post_id)
    if not article["validation"]["passed"]:
        write_json(CACHE / "quarantine" / f"{post_id}.json", article["validation"])
        raise MirrorError("Conversion validation failed: " + json.dumps(article["validation"]["checks"]))
    root, _ = source_body(page, post_id)
    source = str(root).replace("\r\n", "\n").replace("\r", "\n")
    folder = directory / post_id
    folder.mkdir(parents=True, exist_ok=True)
    report = {"metadata": article["metadata"], "validation": article["validation"],
              "fetched_at": fetched_at, "source_snapshot_sha256": digest(source), "images": {}}
    # State is set to verified only after these files and the independent audit succeed.
    (folder / "source.html").write_text(source, encoding="utf-8", newline="\n")
    (folder / "article.md").write_text(article["markdown"], encoding="utf-8", newline="\n")
    write_json(folder / "report.json", report)
    verified_article(directory, post_id)
    return article["validation"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compatibility maintenance for selected articles; use update_all.py for routine updates")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--sleep", type=float, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--refresh", action="store_true", help="Recheck now, still use conditional HTTP")
    parser.add_argument("--from-html-dir", type=Path, help="Import already fetched article HTML without article requests")
    parser.add_argument("--download-images", action="store_true", help="Compatibility flag; images use the same source-host cache as update_all.py")
    parser.add_argument("--withdraw", action="store_true", help="Remove mirrors and persist withdrawal IDs")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[1-9]\d*", post_id) for post_id in args.ids):
        parser.error("Article IDs must be positive decimal integers")
    config = load_config(MIRROR_DIR)
    state = read_json(MIRROR_DIR / "state.json", {})
    if args.withdraw:
        for post_id in args.ids:
            withdraw(MIRROR_DIR, post_id)
            state[post_id] = {"status": "withdrawn", "checked_at": now_iso()}
        config["withdrawn_ids"] = sorted(set(config.get("withdrawn_ids", [])) | set(args.ids))
        save_config(config, MIRROR_DIR)
        write_json(MIRROR_DIR / "state.json", state)
        print("Withdrawn locally. Rebuild and redeploy to remove previously published files.")
        return 0
    fetcher = SerialFetcher(args.sleep, args.attempts)
    network_ready = False
    failed = False
    for post_id in dict.fromkeys(args.ids):
        prior = state.get(post_id, {})
        if post_id in config.get("withdrawn_ids", []):
            withdraw(MIRROR_DIR, post_id)
            print(f"{post_id}: withdrawn; skipped")
            continue
        if prior.get("retry_not_before") and age_days(prior["retry_not_before"]) < 0:
            print(f"{post_id}: server deferred retry until {prior['retry_not_before']}")
            continue
        if not args.refresh and not args.from_html_dir and prior.get("checked_at"):
            interval = max(1, config.get("refresh_days", 7)) if prior.get("status") == "verified" else 1
            if age_days(prior["checked_at"]) < interval:
                print(f"{post_id}: cached; skipped (next check after {interval} days)")
                continue
        try:
            if (not args.from_html_dir or args.download_images) and not network_ready:
                fetcher.check_robots()
                network_ready = True
            headers = {}
            if not args.from_html_dir:
                if prior.get("status") == "verified":
                    try:
                        verified_article(MIRROR_DIR, post_id)
                        headers = {k: prior[v] for k, v in (("If-None-Match", "etag"), ("If-Modified-Since", "last_modified")) if prior.get(v)}
                    except (MirrorError, OSError, KeyError):
                        prior = {}  # A damaged cache requires a full response, never a 304.
                response = fetcher.get(f"https://spaces.ac.cn/archives/{post_id}", headers)
                if response.status_code in {404, 410}:
                    withdraw(MIRROR_DIR, post_id)
                    state[post_id] = {"status": "withdrawn", "checked_at": now_iso(), "http_status": response.status_code}
                    print(f"{post_id}: source removed")
                    continue
                if response.status_code == 304:
                    if prior.get("status") != "verified":
                        raise MirrorError("304 without verified local content")
                    folder = MIRROR_DIR / post_id
                    report = read_json(folder / "report.json", {})
                    write_json(folder / "report.json", report)
                    prior["checked_at"] = now_iso()
                    state[post_id] = prior
                    print(f"{post_id}: unchanged (304)")
                    continue
                if response.status_code != 200:
                    raise MirrorError(f"HTTP {response.status_code}; no article accepted")
                page = response.content.decode("utf-8-sig", errors="strict")
                validators = {"etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
                fetched_at = now_iso()
            else:
                source_file = args.from_html_dir / f"{post_id}.html"
                page = source_file.read_text(encoding="utf-8")
                fetched_at = datetime.fromtimestamp(source_file.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                validators = {}
            state[post_id] = {"status": "pending", "checked_at": now_iso()}
            write_json(MIRROR_DIR / "state.json", state)
            audit = save_article(MIRROR_DIR, post_id, page, fetched_at)
            state[post_id] = {"status": "verified", "checked_at": fetched_at,
                              "markdown_sha256": audit["markdown_sha256"], **validators}
            print(f"{post_id}: PASS; {audit['formula_count']} formulas, {audit['image_count']} images")
        except (MirrorError, requests.RequestException, OSError, UnicodeError, KeyError) as exc:
            failed = True
            withdraw(MIRROR_DIR, post_id)
            state[post_id] = {"status": "failed", "checked_at": now_iso(), "error": str(exc)}
            if isinstance(exc, DeferredRetry):
                state[post_id]["retry_not_before"] = exc.retry_at
            print(f"{post_id}: FAIL; {exc}")
        finally:
            write_json(MIRROR_DIR / "state.json", state)
    try:
        from .image_cache import sync_images
    except ImportError:
        from image_cache import sync_images
    if not args.from_html_dir or args.download_images:
        ids = [i for i in args.ids if state.get(i, {}).get("status") == "verified" and i not in config.get("withdrawn_ids", [])]
        result = sync_images(MIRROR_DIR, ids, interval=args.sleep, attempts=args.attempts)
        failed = failed or bool(result.get("failed"))
    if hasattr(fetcher, "session"):
        fetcher.session.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
