"""Canonical project storage. Content is durable; build/cache files are disposable."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTICLES_DIR = DATA_DIR / "articles"
POLICY_PATH = ROOT / "config" / "mirror.json"
PREVIEW_DIR = ROOT / "build" / "preview"
PUBLIC_DIR = ROOT / "_site"
