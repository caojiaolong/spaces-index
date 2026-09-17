"""One entry point: archive -> articles -> metadata -> classification -> website."""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

try:
    from .paths import ROOT, ARTICLES_DIR, PREVIEW_DIR, PUBLIC_DIR
    from .common import read_json, sort_posts_desc
    from .extract_articles import main as extract, atomic_json, run_lock, cached_metadata
    from .mirror_store import now_iso
    from .mirror_content import MirrorError
    from .classify import classify_posts, load_overrides
    from .render_markdown import render_all
    from .build_site import build_site
except ImportError:
    from paths import ROOT, ARTICLES_DIR, PREVIEW_DIR, PUBLIC_DIR
    from common import read_json, sort_posts_desc
    from extract_articles import main as extract, atomic_json, run_lock, cached_metadata
    from mirror_store import now_iso
    from mirror_content import MirrorError
    from classify import classify_posts, load_overrides
    from render_markdown import render_all
    from build_site import build_site


def sync_metadata(archive: list, *, refresh_summaries=False) -> list:
    """Project already acquired data into the index; never request article pages."""
    path = ROOT / "data/posts.json"
    cached = {str(p["id"]): p for p in read_json(path, [])}
    result = []
    for raw in sort_posts_desc(archive):
        result.append(cached_metadata(raw, cached.get(str(raw["id"]), {}), ARTICLES_DIR,
                                      refresh_summary=refresh_summaries))
    atomic_json(path, result)
    return result


def update(args) -> dict:
    started = time.monotonic()
    summary_path = ROOT / ".cache/ingestion/summary.json"
    # An early acquisition error must not be mistaken for an old image warning.
    summary_path.unlink(missing_ok=True)
    if not args.offline:
        cmd = ["--all", "--refresh-archive", "--refresh-due", "--sleep", str(args.sleep)]
        if args.force:
            cmd += ["--refresh-metadata"]
        if args.skip_enrich:
            cmd += ["--skip-metadata-refresh"]
        if args.max_refresh is not None:
            cmd += ["--max-refresh", str(args.max_refresh)]
        ingestion_status = extract(cmd)
        if ingestion_status == 130:
            raise KeyboardInterrupt
    else:
        ingestion_status = extract(["--all", "--retry-failed", "--offline"])
    ingested_at = time.monotonic()
    ingestion_summary = read_json(summary_path, {})
    counters = ingestion_summary.get("this_run", {})
    archive = read_json(ARTICLES_DIR / "archive.json", read_json(ROOT / "data/posts_raw.json", []))
    if not archive:
        raise MirrorError("No archive available; run an online update first")
    atomic_json(ROOT / "data/posts_raw.json", archive)
    posts = sync_metadata(archive, refresh_summaries=args.refresh_summaries)
    classified = classify_posts(posts, load_overrides(ROOT / "data/overrides.yaml"))
    atomic_json(ROOT / "data/posts_classified.json", classified)
    render_all(classified)
    indexed_at = time.monotonic()
    destination = PREVIEW_DIR if args.audience == "preview" else PUBLIC_DIR
    built = not getattr(args, "skip_build", False)
    catalog = (build_site(ROOT / "data/posts_classified.json", destination,
                          local_articles=ARTICLES_DIR if args.audience == "preview" else None)
               if built else None)
    state = read_json(ARTICLES_DIR / "state.json", {})
    result = {"finished_at": now_iso(), "elapsed_seconds": round(time.monotonic() - started, 1),
              "articles": len(archive), "statuses": dict(Counter(s["status"] for s in state.values())),
              "readable_articles": len(catalog["mirrors"]) if built else None,
              "built": built, "audience": args.audience, "output": str(destination),
              "ingestion_exit_code": ingestion_status,
              "failures": {i: s.get("error", "") for i, s in state.items()
                           if s["status"].endswith("failed")},
              "metadata_failures": {str(p["id"]): p["metadata_error"] for p in posts if p.get("metadata_error")}}
    result["images"] = {key.removeprefix("images_"): value for key, value in counters.items()
                        if key.startswith("images_")}
    image_warning_only = (ingestion_status == 1 and counters.get("images_failed", 0) > 0
                          and not any(counters.get(k) for k in
                                      ("fetch_failed", "conversion_failed", "metadata_failed", "stopped_early")))
    result["exit_code"] = int(bool((ingestion_status and not image_warning_only)
                                   or result["failures"] or result["metadata_failures"]))
    images = result["images"]
    result["warnings"] = ([f"Image cache incomplete: {images.get('failed', 0)} failed, "
                           f"{images.get('deferred', 0)} deferred, {images.get('pending', 0)} pending. "
                           "Uncached images keep their source URLs; see the image log for host errors."]
                          if any(images.get(k) for k in ("failed", "deferred", "pending", "stopped_early")) else [])
    result["timings_seconds"] = {"ingestion": round(ingested_at - started, 1),
                                "metadata_and_index": round(indexed_at - ingested_at, 1),
                                "build": round(time.monotonic() - indexed_at, 1) if built else 0}
    result["ingestion_timings_seconds"] = ingestion_summary.get("timings_seconds", {})
    result["validation_cache_hits"] = counters.get("validation_cache_hits", 0)
    atomic_json(ROOT / ".cache/update-result.json", result)
    return result


def report_ci_result(result):
    """Keep optional image failures visible without labelling a good build broken."""
    for warning in result["warnings"]:
        print(("::warning::" if os.environ.get("GITHUB_ACTIONS") == "true" else "Warning: ") + warning)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        lines = ["### Library update", "", f"- Articles: {result['articles']}",
                 f"- Body failures: {len(result['failures'])}",
                 f"- Metadata failures: {len(result['metadata_failures'])}",
                 f"- Images skipped by policy: {result['images'].get('skipped', 0)}",
                 f"- Built in update step: {result['built']}",
                 f"- Unchanged body audits reused: {result['validation_cache_hits']}",
                 f"- Ingestion detail (seconds): `{json.dumps(result['ingestion_timings_seconds'])}`",
                 f"- Stage durations (seconds): `{json.dumps(result['timings_seconds'])}`"]
        lines.extend(f"- {warning}" for warning in result["warnings"])
        with Path(summary).open("a", encoding="utf8") as handle:
            handle.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Repair saved failures, classify and build without network")
    parser.add_argument("--audience", choices=("preview", "public"), default="preview")
    parser.add_argument("--sleep", type=float, default=3, help="Serial request interval; minimum 3 seconds")
    parser.add_argument("--max-refresh", type=int, help="Bound the number of stale articles revisited")
    parser.add_argument("--serve", action="store_true", help="Serve the completed website on localhost")
    parser.add_argument("--skip-build", action="store_true", help="Update data and index only; CI builds once after checking for changes")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--force", action="store_true", help="Refresh metadata and in-scope bodies together from one page response")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip requests solely for missing metadata; body responses still supply metadata")
    parser.add_argument("--refresh-summaries", action="store_true", help="Re-extract short summary excerpts from saved bodies without extra HTTP")
    parser.add_argument("--progress-every", type=int, default=25, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.skip_build and args.serve:
        parser.error("--skip-build cannot be combined with --serve")
    if args.sleep < 3:
        parser.error("--sleep must be at least 3 seconds")
    if args.max_refresh is not None and args.max_refresh < 0:
        parser.error("--max-refresh must be nonnegative")
    try:
        with run_lock(ROOT / ".cache/update"):
            result = update(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        report_ci_result(result)
        if args.serve:
            try:
                from .serve import serve
            except ImportError:
                from serve import serve
            serve(Path(result["output"]), args.port)
        return result["exit_code"]
    except KeyboardInterrupt:
        print("Interrupted; completed content is saved. Rerun to continue.")
        return 130
    except (OSError, ValueError, KeyError) as exc:
        print(f"Update failed: {exc}. The previous completed website is retained.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
