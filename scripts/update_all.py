from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .common import ROOT, log
except ImportError:  # pragma: no cover - used when running as python scripts/update_all.py
    from common import ROOT, log


def run_step(name: str, args: list[str]) -> None:
    log(f"update_all: starting {name}")
    subprocess.run(args, cwd=ROOT, check=True)
    log(f"update_all: finished {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the full scientific spaces index.")
    parser.add_argument("--force", action="store_true", help="Force per-post metadata refresh.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep seconds between article requests.")
    parser.add_argument("--skip-enrich", action="store_true", help="Skip per-post metadata fetching.")
    parser.add_argument(
        "--refresh-summaries",
        action="store_true",
        help="Fetch cached posts that do not yet have source_summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log enrich progress every N posts. Set 0 to disable.",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    run_step("fetch_archive", [sys.executable, str(scripts_dir / "fetch_archive.py")])

    if not args.skip_enrich:
        enrich_cmd = [
            sys.executable,
            str(scripts_dir / "enrich_posts.py"),
            "--sleep",
            str(args.sleep),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.refresh_summaries:
            enrich_cmd.append("--refresh-summaries")
        if args.force:
            enrich_cmd.append("--force")
        run_step("enrich_posts", enrich_cmd)
    else:
        log("update_all: skipped enrich_posts")

    run_step("classify", [sys.executable, str(scripts_dir / "classify.py")])
    run_step("render_markdown", [sys.executable, str(scripts_dir / "render_markdown.py")])
    log("update_all: done")


if __name__ == "__main__":
    main()
