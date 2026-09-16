"""Build a localhost reading preview from the canonical article library."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .common import ROOT, read_json
    from .classify import classify_post, load_overrides
    from .mirror_content import MirrorError
    from .mirror_store import MIRROR_DIR, verified_article, load_config, prepare_images
except ImportError:
    from common import ROOT, read_json
    from classify import classify_post, load_overrides
    from mirror_content import MirrorError
    from mirror_store import MIRROR_DIR, verified_article, load_config, prepare_images


def preview_posts(posts: list, directory: Path) -> list:
    """Retain existing metadata; classify only newly archived IDs, offline."""
    result = list(posts)
    known = {str(p["id"]) for p in posts}
    overrides = load_overrides(ROOT / "data/overrides.yaml")
    series_topics = {p["series"]: p.get("series_topic") for p in posts if p.get("series")}
    for post in read_json(directory / "archive.json", []):
        if str(post["id"]) not in known:
            classified = classify_post(post, overrides)
            if classified.get("series") in series_topics:
                classified["series_topic"] = series_topics[classified["series"]]
            result.append(classified)
            known.add(str(post["id"]))
    return result


def export_preview(output: Path, directory: Path, allowed_ids: set[str]) -> tuple[dict, dict]:
    state = read_json(directory / "state.json", {})
    policy = load_config()
    withdrawn = set(map(str, policy.get("withdrawn_ids", [])))
    entries, unavailable = {}, {}
    for index, post_id in enumerate(sorted(allowed_ids, key=int), 1):
        status = state.get(post_id, {})
        if post_id in withdrawn or status.get("status") != "verified":
            unavailable[post_id] = "已下架" if post_id in withdrawn else status.get("error", "尚无已校验正文")
            continue
        article = verified_article(directory, post_id, local_only=True)
        validation = article["validation"]
        if validation["markdown_sha256"] != status.get("markdown_sha256"):
            raise MirrorError(f"State and Markdown hashes differ: {post_id}")
        target = output / "mirror" / post_id
        image_files = prepare_images(directory, post_id, article)
        target.mkdir(parents=True)
        (target / "article.md").write_text(article["markdown"], encoding="utf-8", newline="\n")
        (target / "article.json").write_text(json.dumps(article, ensure_ascii=False), encoding="utf-8")
        for name, raw in image_files.items():
            (target / "images").mkdir(exist_ok=True)
            (target / "images" / name).write_bytes(raw)
        entries[post_id] = {"sha256": validation["markdown_sha256"], "checkedAt": status["checked_at"],
                           "formulaCount": validation["formula_count"], "warnings": validation.get("source_warnings", [])}
        if index % 100 == 0:
            print(f"Preview: validated {index}/{len(allowed_ids)} articles", flush=True)
    return entries, unavailable
