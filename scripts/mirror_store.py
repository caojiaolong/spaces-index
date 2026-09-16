"""Publish only verified, current articles permitted by the central policy."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

try:
    from .common import ROOT, read_json
    from .mirror_content import MirrorError, attribution, digest, preview_tree, source_image_captions, validate_body
    from .mirror_html import FORMAT, validate_html_body, preview_html_body
    from .paths import ARTICLES_DIR, POLICY_PATH
    from .image_cache import cached_image_bytes, hosted_image_url, image_urls
except ImportError:
    from common import ROOT, read_json
    from mirror_content import MirrorError, attribution, digest, preview_tree, source_image_captions, validate_body
    from mirror_html import FORMAT, validate_html_body, preview_html_body
    from paths import ARTICLES_DIR, POLICY_PATH
    from image_cache import cached_image_bytes, hosted_image_url, image_urls

MIRROR_DIR = ARTICLES_DIR


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def age_days(timestamp: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds() / 86400


def load_config(directory: Path = MIRROR_DIR) -> dict:
    config = read_json(POLICY_PATH if directory == ARTICLES_DIR else directory / "config.json", {})
    # JSON editors may store IDs as numbers. Withdrawal must work identically
    # for direct public builds, previews and subsequent ingestion.
    config["withdrawn_ids"] = list(map(str, config.get("withdrawn_ids", [])))
    return config


def save_config(config: dict, directory: Path = MIRROR_DIR):
    path = POLICY_PATH if directory == ARTICLES_DIR else directory / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verified_article(directory: Path, post_id: str, *, local_only: bool = False) -> dict:
    if not re.fullmatch(r"[1-9]\d*", post_id):
        raise MirrorError("Invalid article ID")
    folder = directory / post_id
    report = read_json(folder / "report.json", {})
    markdown = (folder / "article.md").read_text(encoding="utf-8")
    source_html = (folder / "source.html").read_text(encoding="utf-8")
    meta = report["metadata"]
    if meta["id"] != post_id or meta["source_url"] != f"https://spaces.ac.cn/archives/{post_id}":
        raise MirrorError("Stored identity mismatch")
    prefix = attribution(meta)
    if not markdown.startswith(prefix):
        raise MirrorError("Attribution changed or missing")
    source = BeautifulSoup(source_html, "lxml").select_one("#PostContent")
    if source is None:
        raise MirrorError("Missing source body snapshot")
    legacy = report.get("validation", {}).get("format") == FORMAT
    validation = (validate_html_body if legacy else validate_body)(source, markdown[len(prefix):], meta["source_url"])
    if not report.get("validation", {}).get("passed") or not validation["passed"]:
        raise MirrorError("Body or formula validation failed")
    if digest(markdown) != report["validation"]["markdown_sha256"] or digest(source_html) != report["source_snapshot_sha256"]:
        raise MirrorError("Stored content hash mismatch")
    for key in ("latex_sha256", "source_text_sha256", "body_sha256", "converter_version"):
        if validation[key] != report["validation"][key]:
            raise MirrorError(f"Stored validation mismatch: {key}")
    tree = (preview_tree(prefix) + preview_html_body(markdown[len(prefix):])) if legacy else preview_tree(markdown)
    image_manifest = folder / "images.json"
    manifest = read_json(image_manifest, {}).get("images", {})
    images = ({url: info for url, info in manifest.items() if info.get("status") == "cached"}
              if image_manifest.exists() else report.get("images", {}))
    return {"metadata": meta, "markdown": markdown, "validation": validation | {"markdown_sha256": digest(markdown)},
            "tree": tree, "images": images, "captions": source_image_captions(source)}


def prepare_images(directory: Path, post_id: str, article: dict):
    """Both preview and public builds verify and export the same source-host files."""
    urls = set(image_urls(article["tree"]))
    mapping, files = {}, {}
    for source_url, info in article["images"].items():
        if source_url not in urls or not hosted_image_url(source_url):
            continue
        raw = cached_image_bytes(directory / post_id, info)
        name = info["file"]
        files[name] = raw
        mapping[source_url] = f"./mirror/{post_id}/images/{name}"
    article["images"] = mapping
    return files


def publish_mirrors(output: Path, directory: Path = MIRROR_DIR, *, allowed_ids: set[str]) -> dict:
    config = load_config(directory)
    entries = {}
    state = read_json(directory / "state.json", {})
    prepared = []
    for post_id in sorted(allowed_ids):
        status = state.get(post_id, {})
        if post_id in config.get("withdrawn_ids", []) or status.get("status") != "verified":
            continue
        try:
            if age_days(status["checked_at"]) > float(config.get("max_stale_days", 120)):
                continue
            article = verified_article(directory, post_id)
            if article["validation"]["markdown_sha256"] != status.get("markdown_sha256"):
                raise MirrorError("State and Markdown hashes differ")
            target = output / "mirror" / post_id
            image_files = prepare_images(directory, post_id, article)
            prepared.append((target, article, image_files))
            entries[post_id] = {"sha256": status["markdown_sha256"], "checkedAt": status["checked_at"],
                                "formulaCount": article["validation"]["formula_count"],
                                "warnings": article["validation"].get("source_warnings", [])}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Abort the build: a partial output must never be uploaded after an error.
            raise MirrorError(f"Refusing to publish mirror {post_id}: {exc}") from exc
    # No public Markdown is written until every candidate and image has passed.
    for target, article, image_files in prepared:
        target.mkdir(parents=True, exist_ok=True)
        (target / "article.md").write_text(article["markdown"], encoding="utf-8", newline="\n")
        (target / "article.json").write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
        for name, raw in image_files.items():
            (target / "images").mkdir(exist_ok=True)
            (target / "images" / name).write_bytes(raw)
    return entries
