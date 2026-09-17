"""Resumable, low-frequency article ingestion into the canonical content library.

Successes are Markdown plus auditable body snapshots. Conversion failures retain
only the extracted body snapshot, so parser fixes can be retried without HTTP.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from .common import ROOT, read_json, sort_posts_desc
    from .fetch_archive import parse_archive
    from .mirror_articles import DeferredRetry, SerialFetcher
    from .mirror_content import (VERSION, MirrorError, attribution, convert_body, digest,
                                 source_body, validate_body, validate_source)
    from .mirror_store import MIRROR_DIR, age_days, now_iso, verified_article
    from .mirror_html import convert_html_body, validate_html_body
    from .paths import DATA_DIR, POLICY_PATH
    from .image_cache import sync_images, image_urls, IMAGE_NAME
    from .enrich_posts import parse_post_metadata, has_cached_metadata, merge_cached_post, summary_needs_refresh, extract_source_summary
except ImportError:
    from common import ROOT, read_json, sort_posts_desc
    from fetch_archive import parse_archive
    from mirror_articles import DeferredRetry, SerialFetcher
    from mirror_content import (VERSION, MirrorError, attribution, convert_body, digest,
                                source_body, validate_body, validate_source)
    from mirror_store import MIRROR_DIR, age_days, now_iso, verified_article
    from mirror_html import convert_html_body, validate_html_body
    from paths import DATA_DIR, POLICY_PATH
    from image_cache import sync_images, image_urls, IMAGE_NAME
    from enrich_posts import parse_post_metadata, has_cached_metadata, merge_cached_post, summary_needs_refresh, extract_source_summary

STORAGE_ROOT = DATA_DIR
ARCHIVE_URL = "https://spaces.ac.cn/content.html"


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    # Windows indexers/antivirus can briefly hold the destination without
    # FILE_SHARE_DELETE. Keep the previous atomic file and retry the rename.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * 2 ** attempt)


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def article_folder(output: Path, post_id: str) -> Path:
    if not re.fullmatch(r"[1-9]\d*", post_id):
        raise MirrorError("Article IDs must be positive decimal integers")
    folder = output / post_id
    if folder.resolve().parent != output.resolve():
        raise MirrorError("Article directory escapes the local output")
    return folder


@contextmanager
def run_lock(output: Path):
    """OS releases the lock even after Ctrl+C or process termination."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".extract.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise MirrorError("Another extractor is using this output directory") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def save_snapshot(output: Path, post_id: str, source: str, metadata: dict, *, fetched_at=None, validators=None):
    folder = article_folder(output, post_id)
    if metadata["id"] != post_id or metadata["source_url"] != f"https://spaces.ac.cn/archives/{post_id}":
        raise MirrorError("Snapshot identity mismatch")
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    record = {"metadata": metadata, "fetched_at": fetched_at or now_iso(),
              "source_snapshot_sha256": digest(source), "validators": validators or {}}
    # Invalidate an older generated Markdown before installing a new snapshot.
    for name in ("article.md", "report.json"):
        (folder / name).unlink(missing_ok=True)
    atomic_text(folder / "source.html", source)
    atomic_json(folder / "snapshot.json", record)
    return record


def read_snapshot(output: Path, post_id: str):
    folder = article_folder(output, post_id)
    record = read_json(folder / "snapshot.json", {})
    source = (folder / "source.html").read_text(encoding="utf-8")
    if digest(source) != record.get("source_snapshot_sha256"):
        raise MirrorError("Local snapshot hash mismatch; use --refresh to fetch again")
    meta = record["metadata"]
    if meta["id"] != post_id or meta["source_url"] != f"https://spaces.ac.cn/archives/{post_id}":
        raise MirrorError("Local snapshot identity mismatch")
    return source, record


def convert_snapshot(output: Path, post_id: str) -> dict:
    folder = article_folder(output, post_id)
    source, record = read_snapshot(output, post_id)
    root = BeautifulSoup(source, "lxml").select_one("#PostContent")
    if root is None:
        raise MirrorError("Missing article body")
    meta = record["metadata"]
    try:
        validate_source(root)
        body = convert_body(root, meta["source_url"])
        audit = validate_body(root, body, meta["source_url"])
        if not audit["passed"]:
            raise MirrorError("Standard Markdown round trip changed source structure")
    except MirrorError as exc:
        body = convert_html_body(root, meta["source_url"])
        audit = validate_html_body(root, body, meta["source_url"])
        audit["fallback_reason"] = str(exc)
    if not audit["passed"]:
        atomic_json(folder / "validation-failure.json", audit)
        raise MirrorError("Validation failed: " + ", ".join(k for k, ok in audit["checks"].items() if not ok))
    markdown = attribution(meta) + body
    audit["markdown_sha256"] = digest(markdown)
    report = {"metadata": meta, "validation": audit, "fetched_at": record["fetched_at"],
              "source_snapshot_sha256": record["source_snapshot_sha256"], "images": {}, "publication_policy": "config/mirror.json"}
    atomic_text(folder / "article.md", markdown)
    atomic_json(folder / "report.json", report)
    verified_article(output, post_id, local_only=True)
    (folder / "validation-failure.json").unlink(missing_ok=True)
    return report


def select_posts(posts: list, ids: list[str] | None, limit: int | None):
    known = {str(post["id"]): post for post in posts}
    if any(not re.fullmatch(r"[1-9]\d*", key) for key in known):
        raise MirrorError("Invalid ID in archive")
    if ids is not None:
        selected = [known.get(i, {"id": i, "url": f"https://spaces.ac.cn/archives/{i}", "title": i})
                    for i in dict.fromkeys(ids)]
    else:
        selected = sorted(known.values(), key=lambda p: (str(p.get("date", "")), int(p["id"])), reverse=True)
    for post in selected:
        if not re.fullmatch(r"[1-9]\d*", str(post["id"])):
            raise MirrorError("Invalid ID in archive or --ids")
    return selected[:limit] if limit else selected


def metadata_path(output: Path) -> Path:
    return ROOT / "data/posts.json" if output == MIRROR_DIR else output / "metadata.json"


def cached_metadata(raw: dict, cached: dict, output: Path, *, refresh_summary=False) -> dict:
    """Fill missing legacy fields offline; never overwrite fresher metadata."""
    post = merge_cached_post(raw, cached)
    fields = ("source_category", "source_tags", "source_summary")
    if all(key in post for key in fields) and not refresh_summary and not summary_needs_refresh(post):
        return post
    folder = article_folder(output, str(raw["id"]))
    try:
        record = read_json(folder / "snapshot.json", {})
    except (OSError, ValueError):
        record = {}  # Body ingestion handles incomplete or damaged snapshots.
    for key in fields:
        if key not in post and key in record.get("metadata", {}):
            post[key] = record["metadata"][key]
    if (refresh_summary or summary_needs_refresh(post)) and (folder / "source.html").is_file():
        try:
            source, _ = read_snapshot(output, str(raw["id"]))
            post["source_summary"] = extract_source_summary(source)
        except (OSError, ValueError, KeyError):
            pass  # Damaged snapshots will be repaired by the body pipeline.
    return post


def run(args, output: Path) -> int:
    state_path = output / "state.json"
    state = read_json(state_path, {})
    metadata_file = metadata_path(output)
    metadata_cache = {str(post["id"]): post for post in read_json(metadata_file, [])}
    def save_metadata(post_id, post):
        if metadata_cache.get(post_id) != post:
            metadata_cache[post_id] = dict(post)
            atomic_json(metadata_file, sort_posts_desc(list(metadata_cache.values())))
    counters = Counter()
    started = time.monotonic()
    runtime = ROOT / ".cache/ingestion" if output == MIRROR_DIR else output
    runtime.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f"[{now_iso()}] {message}"
        print(line, flush=True)
        with (runtime / "run.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    def summary():
        result = {"updated_at": now_iso(), "this_run": dict(counters),
                  "stored_statuses": dict(Counter(s.get("status", "unknown") for s in state.values())),
                  "elapsed_seconds": round(time.monotonic() - started, 1), "publishes": False}
        atomic_json(runtime / "summary.json", result)
        log("SUMMARY " + json.dumps(result, ensure_ascii=False))
    fetcher = SerialFetcher(args.sleep, args.attempts)
    ready = False
    def network_ready():
        nonlocal ready
        if not ready:
            fetcher.check_robots()
            ready = True
    if args.refresh_archive:
        network_ready()
        response = fetcher.get(ARCHIVE_URL)
        if response.status_code != 200:
            raise MirrorError(f"Archive HTTP {response.status_code}")
        posts = parse_archive(response.content.decode("utf-8-sig"))
        if not posts:
            raise MirrorError("Empty archive; refusing to continue")
        atomic_json(output / "archive.json", posts)
        log(f"Archive refreshed: {len(posts)} article IDs")
    else:
        posts = read_json(output / "archive.json", None)
        if posts is None:
            posts = read_json(ROOT / "data" / "posts_raw.json", [])
    selected = select_posts(posts, args.ids, None if args.retry_failed else args.limit)
    policy = read_json(POLICY_PATH, {})
    withdrawn = set(map(str, policy.get("withdrawn_ids", [])))
    refresh_ids = set()
    if args.refresh_due and not args.offline:
        due = []
        for post in selected:
            key = str(post["id"])
            if key in withdrawn:
                continue
            previous = state.get(key, {})
            timestamp = previous.get("checked_at") if previous.get("status") in {"verified", "removed"} else None
            if timestamp and age_days(timestamp) >= float(policy.get("refresh_days", 30)):
                due.append((key, timestamp))
        due.sort(key=lambda pair: (pair[1], pair[0]))
        refresh_ids = {key for key, _ in due[:args.max_refresh if args.max_refresh is not None else int(policy.get("refresh_per_run", 25))]}
    if args.retry_failed:
        selected = [p for p in selected if state.get(str(p["id"]), {}).get("status") in {"fetch_failed", "conversion_failed", "pending"}
                    or metadata_cache.get(str(p["id"]), {}).get("metadata_error")
                    or (not args.offline and any(i.get("status") == "failed" or i.get("error") for i in
                        read_json(output / str(p["id"]) / "images.json", {}).get("images", {}).values()))]
        if args.limit:
            selected = selected[:args.limit]
    log(f"Selected {len(selected)} articles. Storage: {output}. Scheduled refreshes: {len(refresh_ids)}.")
    consecutive_network_errors = 0
    interrupted = False
    image_refresh = set()
    # Reuse only the image URLs from bodies validated during this invocation.
    # Publication still performs its own independent full validation.
    verified_image_urls = {}
    try:
        for index, post in enumerate(selected, 1):
            post_id = str(post["id"])
            folder = article_folder(output, post_id)
            prior = state.get(post_id, {})
            label = f"[{index}/{len(selected)}] {post_id}"
            source_ready = False
            meta_post = cached_metadata(post, metadata_cache.get(post_id, {}), output)
            save_metadata(post_id, meta_post)
            metadata_deferred = bool(meta_post.get("metadata_error") and not args.retry_failed and not args.refresh_metadata
                                     and age_days(meta_post.get("metadata_checked_at", now_iso())) < 1)
            need_metadata = not args.skip_metadata_refresh and not metadata_deferred and (
                args.refresh_metadata or not has_cached_metadata(meta_post) or bool(meta_post.get("metadata_error")))
            refresh = args.refresh or post_id in refresh_ids or prior.get("status") == "fetch_failed" or (need_metadata and not args.offline)
            checked_source = False
            metadata_processed = False
            saved_pending = False
            try:
                if post_id in withdrawn:
                    state[post_id] = prior | {"status": "withdrawn"}
                    remove_images(folder)
                    counters["withdrawn"] += 1
                    continue
                if prior.get("retry_not_before") and age_days(prior["retry_not_before"]) < 0:
                    counters["deferred"] += 1
                    log(f"{label} DEFERRED until {prior['retry_not_before']}")
                    continue
                if meta_post.get("metadata_retry_not_before") and age_days(meta_post["metadata_retry_not_before"]) < 0 and not args.offline:
                    counters["deferred"] += 1
                    continue
                if prior.get("status") == "removed" and not (args.refresh or args.refresh_metadata or post_id in refresh_ids):
                    counters["removed_skipped"] += 1
                    continue
                if prior.get("status") == "verified" and not refresh and not args.offline:
                    try:
                        article = verified_article(output, post_id, local_only=True)
                        verified_image_urls[post_id] = image_urls(article["tree"])
                        counters["cached"] += 1
                        log(f"{label} CACHED")
                        continue
                    except (OSError, ValueError, KeyError):
                        pass  # Repair generated files from the local source snapshot.
                if (folder / "snapshot.json").is_file() and not refresh:
                    try:
                        read_snapshot(output, post_id)
                        source_ready = True
                    except (OSError, ValueError, KeyError):
                        log(f"{label} incomplete/damaged snapshot; reacquire if online")
                if not source_ready and not refresh and output != MIRROR_DIR and (MIRROR_DIR / post_id / "report.json").is_file():
                    article = verified_article(MIRROR_DIR, post_id)
                    report = read_json(MIRROR_DIR / post_id / "report.json", {})
                    save_snapshot(output, post_id, (MIRROR_DIR / post_id / "source.html").read_text(encoding="utf-8"),
                                  article["metadata"], fetched_at=report["fetched_at"])
                    source_ready = True
                    counters["imported_snapshot"] += 1
                if not source_ready and args.offline:
                    counters["missing_snapshot"] += 1
                    log(f"{label} SKIP no local snapshot")
                    continue
                if not source_ready:
                    # Generic network failures cool down for a day. Explicit
                    # --retry-failed overrides this, but never Retry-After.
                    if prior.get("status") == "fetch_failed" and not args.retry_failed and not args.refresh and age_days(prior["checked_at"]) < 1:
                        counters["deferred"] += 1
                        log(f"{label} DEFERRED; use --retry-failed for an explicit retry")
                        continue
                    network_ready()
                    validators = {}
                    if refresh and not need_metadata and (folder / "snapshot.json").is_file():
                        try:
                            _, record = read_snapshot(output, post_id)
                            validators = record.get("validators", {})
                        except (OSError, ValueError, KeyError):
                            pass
                    headers = {key: validators[value] for key, value in (("If-None-Match", "etag"), ("If-Modified-Since", "last_modified")) if validators.get(value)}
                    counters["article_requests"] += 1
                    response = fetcher.get(f"https://spaces.ac.cn/archives/{post_id}", headers)
                    if response.status_code in {404, 410}:
                        remove_images(folder)
                        for name in ("article.md", "report.json", "source.html", "snapshot.json", "validation-failure.json"):
                            (folder / name).unlink(missing_ok=True)
                        state[post_id] = {"status": "removed", "checked_at": now_iso(), "http_status": response.status_code}
                        meta_post.pop("metadata_error", None)
                        meta_post.pop("metadata_retry_not_before", None)
                        meta_post["metadata_checked_at"] = now_iso()
                        save_metadata(post_id, meta_post)
                        counters["removed"] += 1
                        consecutive_network_errors = 0
                        log(f"{label} REMOVED HTTP {response.status_code}")
                        continue
                    if response.status_code == 304:
                        if not headers:
                            raise MirrorError("Unexpected 304 without a conditional body request")
                        read_snapshot(output, post_id)
                        source_ready = True
                        checked_source = True
                        counters["not_modified"] += 1
                    elif response.status_code == 200:
                        # Keep only the exact article body, never the full page.
                        page = response.content.decode("utf-8-sig")
                        metadata_processed = True
                        meta_post.pop("metadata_retry_not_before", None)
                        try:
                            parsed_metadata = parse_post_metadata(page)
                            meta_post.update(parsed_metadata)
                            meta_post.pop("metadata_error", None)
                            counters["metadata_updated"] += 1
                        except ValueError as exc:
                            parsed_metadata = {}
                            meta_post["metadata_error"] = str(exc)
                            counters["metadata_failed"] += 1
                            log(f"{label} METADATA_FAILED: {exc}")
                        meta_post["metadata_checked_at"] = now_iso()
                        save_metadata(post_id, meta_post)
                        root, metadata = source_body(page, post_id, check_supported=False)
                        metadata.update(parsed_metadata)
                        save_snapshot(output, post_id, str(root), metadata, validators={
                            "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")})
                        source_ready = True
                        checked_source = True
                        counters["downloaded"] += 1
                    else:
                        raise MirrorError(f"Article HTTP {response.status_code}")
                    consecutive_network_errors = 0
                source_checked_at = now_iso() if checked_source else prior.get("source_checked_at", prior.get("checked_at"))
                state[post_id] = {"status": "pending", "checked_at": now_iso(), "source_checked_at": source_checked_at}
                atomic_json(state_path, state)
                saved_pending = True
                report = convert_snapshot(output, post_id)
                audit = report["validation"]
                checked_at = source_checked_at or report["fetched_at"]
                state[post_id] = {"status": "verified", "checked_at": checked_at, "title": report["metadata"]["title"],
                                  "markdown_sha256": audit["markdown_sha256"], "converter_version": VERSION}
                counters["verified"] += 1
                if checked_source:
                    image_refresh.add(post_id)
                log(f"{label} PASS {audit['formula_count']} formulas, {audit['image_count']} images; {report['metadata']['title']}")
            except (OSError, ValueError, KeyError, requests.RequestException) as exc:
                status = "conversion_failed" if source_ready else "fetch_failed"
                if not source_ready and not metadata_processed:
                    meta_post["metadata_error"] = str(exc)
                    meta_post["metadata_checked_at"] = now_iso()
                    if isinstance(exc, DeferredRetry):
                        meta_post["metadata_retry_not_before"] = exc.retry_at
                    save_metadata(post_id, meta_post)
                for name in ("article.md", "report.json"):
                    (folder / name).unlink(missing_ok=True)
                state[post_id] = {"status": status, "checked_at": now_iso(), "title": post.get("title", post_id),
                                  "error": str(exc), "converter_version": VERSION,
                                  "source_checked_at": now_iso() if checked_source else prior.get("source_checked_at", prior.get("checked_at"))}
                if isinstance(exc, DeferredRetry):
                    state[post_id]["retry_not_before"] = exc.retry_at
                counters[status] += 1
                log(f"{label} {status.upper()}: {exc}")
                if not source_ready:
                    consecutive_network_errors += 1
                if isinstance(exc, DeferredRetry) or consecutive_network_errors >= 3:
                    log("Stopping after server deferral or 3 consecutive acquisition errors; rerun later to resume.")
                    counters["stopped_early"] += 1
                    break
            finally:
                # Cache hits and deferrals do not change persistent state. Avoid
                # serializing and replacing the entire library for every hit.
                if saved_pending or state.get(post_id, {}) != prior:
                    atomic_json(state_path, state)
        if not args.offline and not counters["stopped_early"]:
            image_ids = [str(p["id"]) for p in selected if state.get(str(p["id"]), {}).get("status") == "verified"
                         and str(p["id"]) not in withdrawn]
            image_result = sync_images(output, image_ids, interval=args.sleep, attempts=args.attempts,
                                       refresh_ids=image_refresh, retry_failed=args.retry_failed,
                                       verified_image_urls=verified_image_urls, log=log)
            counters.update({f"images_{key}": value for key, value in image_result.items()})
    except KeyboardInterrupt:
        interrupted = True
        log("Interrupted. Completed articles are saved; rerun the same command to resume.")
    finally:
        atomic_json(state_path, state)
        summary()
        if hasattr(fetcher, "session"):
            fetcher.session.close()
    return 130 if interrupted else (1 if any(counters[k] for k in ("fetch_failed", "conversion_failed", "metadata_failed", "images_failed")) else 0)


def remove_images(folder):
    image_dir = folder / "images"
    if image_dir.exists():
        if not image_dir.resolve().is_relative_to(folder.resolve()):
            raise MirrorError("Image directory escapes article directory")
        for path in image_dir.iterdir():
            if path.is_file() and IMAGE_NAME.fullmatch(path.name):
                path.unlink()
    (folder / "images.json").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract articles and cache source-host images; no publishing")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="All IDs in the local/refreshed archive")
    selection.add_argument("--ids", nargs="+", help="Selected numeric IDs")
    selection.add_argument("--status", action="store_true", help="Print saved progress without network")
    parser.add_argument("--output", type=Path, default=STORAGE_ROOT / "articles")
    parser.add_argument("--refresh-metadata", action="store_true", help="Refresh metadata and body together from one response")
    parser.add_argument("--skip-metadata-refresh", action="store_true", help="Do not request pages solely for missing metadata")
    parser.add_argument("--limit", type=int, help="Limit selected articles for a small trial")
    parser.add_argument("--sleep", type=float, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--refresh-archive", action="store_true", help="Refresh archive IDs into local output only")
    parser.add_argument("--refresh", action="store_true", help="Revalidate source pages with conditional HTTP")
    parser.add_argument("--refresh-due", action="store_true", help="Rotate a bounded number of stale articles using conditional requests")
    parser.add_argument("--max-refresh", type=int, help="Maximum stale articles to revisit this run")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry saved failures; honor Retry-After")
    parser.add_argument("--offline", action="store_true", help="Reconvert local snapshots with no HTTP")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output == STORAGE_ROOT.resolve() or not output.is_relative_to(STORAGE_ROOT.resolve()):
        parser.error("--output must be a directory beneath data/")
    if args.max_refresh is not None and args.max_refresh < 0:
        parser.error("--max-refresh must be nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.offline and (args.refresh or args.refresh_archive):
        parser.error("--offline cannot be combined with network refresh options")
    if args.status:
        state = read_json(output / "state.json", {})
        print(json.dumps({"output": str(output), "counts": dict(Counter(s.get("status", "unknown") for s in state.values())),
                          "failures": {i: s.get("error") for i, s in state.items() if s.get("status", "").endswith("failed")}}, ensure_ascii=False, indent=2))
        return 0
    try:
        with run_lock(output):
            return run(args, output)
    except (MirrorError, requests.RequestException) as exc:
        print(str(exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
