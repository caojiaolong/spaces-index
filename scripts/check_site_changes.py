"""Decide whether saved update state also changes the published website."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATE_PATHS = ("README.md", "docs", "data", "config")
IMAGE_MANIFEST = re.compile(r"data/articles/[1-9]\d*/images\.json")
IMAGE_RUNTIME_FIELDS = {"checked_at", "etag", "last_modified", "error", "retry_not_before", "server_deferred", "skipped_host"}


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def published_images(raw):
    manifest = json.loads(raw)
    return {"version": manifest.get("version"), "images": {
        url: {k: v for k, v in entry.items() if k not in IMAGE_RUNTIME_FIELDS}
        for url, entry in manifest["images"].items() if entry.get("status") == "cached"
    }}


def changes_site(root, path):
    if not IMAGE_MANIFEST.fullmatch(path):
        return True
    try:
        old = git(root, "show", f"HEAD:{path}")
        new = (root / path).read_bytes()
        return published_images(old) != published_images(new)
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError, AttributeError):
        # Creation, removal, or malformed metadata needs a real build/check.
        return True


def check_changes(root, event):
    changed = git(root, "diff", "--name-only", "-z", "HEAD", "--", *UPDATE_PATHS)
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", *UPDATE_PATHS)
    paths = sorted(set((changed + untracked).decode("utf8").strip("\0").split("\0")) - {""})
    deploy_paths = [path for path in paths if changes_site(root, path)]
    return {"should_commit": event != "push" and bool(paths),
            "should_deploy": event == "push" or bool(deploy_paths),
            "changed_files": len(paths), "site_changed_files": len(deploy_paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "schedule"))
    args = parser.parse_args()
    result = check_changes(ROOT, args.event)
    print(json.dumps(result, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf8") as handle:
            for key in ("should_commit", "should_deploy"):
                handle.write(f"{key}={str(result[key]).lower()}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary).open("a", encoding="utf8") as handle:
            handle.write(f"\n- Changed files: {result['changed_files']}; site changes: {result['site_changed_files']}\n"
                         f"- Build and deploy: {result['should_deploy']}\n")


if __name__ == "__main__":
    main()
