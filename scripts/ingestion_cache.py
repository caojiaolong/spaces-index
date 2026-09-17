"""Avoid repeating unchanged body audits during ingestion, never publication."""
from __future__ import annotations

import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

try:
    from .image_cache import image_urls
    from .mirror_content import MirrorError
except ImportError:
    from image_cache import image_urls
    from mirror_content import MirrorError


BODY_FILES = ("article.md", "source.html", "report.json", "snapshot.json")


def validator_fingerprint() -> str:
    scripts = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("ingestion_cache.py", "extract_articles.py", "mirror_content.py",
                 "mirror_html.py", "mirror_store.py", "image_cache.py", "common.py", "../uv.lock"):
        digest.update(name.encode())
        digest.update((scripts / name).read_bytes().replace(b"\r\n", b"\n"))
    for name in ("beautifulsoup4", "lxml", "markdown-it-py", "tinycss2"):
        digest.update(f"{name}={version(name)}".encode())
    digest.update(str(sys.version_info[:2]).encode())
    return digest.hexdigest()


def body_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for name in BODY_FILES:
        digest.update(name.encode())
        digest.update(hashlib.sha256((folder / name).read_bytes()).digest())
    return digest.hexdigest()


class VerifiedBodyCache:
    def __init__(self, path: Path):
        self.path = path
        self.validator = validator_fingerprint()
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.entries = saved["entries"] if saved["validator"] == self.validator else {}
            if not isinstance(self.entries, dict):
                self.entries = {}
        except (OSError, ValueError, KeyError, TypeError):
            self.entries = {}
        self.dirty = False

    def image_urls(self, folder: Path, verify) -> tuple[list[str], bool]:
        key = folder.name
        fingerprint = body_fingerprint(folder)
        entry = self.entries.get(key)
        if (isinstance(entry, dict) and entry.get("fingerprint") == fingerprint
                and isinstance(entry.get("image_urls"), list)
                and all(isinstance(url, str) for url in entry["image_urls"])):
            return list(entry["image_urls"]), True
        self.entries.pop(key, None)
        self.dirty = True
        urls = list(image_urls(verify()["tree"]))
        if body_fingerprint(folder) != fingerprint:
            raise MirrorError("Body changed during validation")
        self.entries[key] = {"fingerprint": fingerprint, "image_urls": urls}
        return urls, False

    def save(self, atomic_json):
        if self.dirty:
            atomic_json(self.path, {"validator": self.validator, "entries": self.entries})
