"""Idempotent copy from earlier storage layouts, with byte-for-byte checks."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .paths import ROOT, ARTICLES_DIR
    from .mirror_content import MirrorError, digest
except ImportError:
    from paths import ROOT, ARTICLES_DIR
    from mirror_content import MirrorError, digest


def migrate(source: Path = ROOT / "content/articles", destination: Path = ARTICLES_DIR) -> int:
    source, destination = source.resolve(), destination.resolve()
    root = ROOT.resolve()
    if (not source.is_relative_to(root) or not destination.is_relative_to(root)
            or source.is_relative_to(destination) or destination.is_relative_to(source)):
        raise MirrorError("Migration paths must be non-overlapping project directories")
    if not source.exists():
        return 0
    copies = []
    for original in source.rglob("*"):
        if not original.is_file() or original.name in {".extract.lock", "run.log", "summary.json"} or original.suffix == ".tmp":
            continue
        target = destination / original.relative_to(source)
        if not original.resolve().is_relative_to(source) or not target.resolve().is_relative_to(destination):
            raise MirrorError("Migration path escapes storage")
        raw = original.read_bytes()
        if target.exists():
            if target.read_bytes() != raw:
                raise MirrorError(f"Conflicting migration file: {target}")
            continue
        copies.append((original, target))
    # Check every existing destination before writing anything; conflicts must
    # not leave a partially merged library.
    for original, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        if digest(target.read_bytes()) != digest(original.read_bytes()):
            raise MirrorError(f"Migration checksum failed: {target}")
    return len(copies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "content/articles")
    args = parser.parse_args()
    print(f"Migrated and verified {migrate(args.source)} files into {ARTICLES_DIR}")
