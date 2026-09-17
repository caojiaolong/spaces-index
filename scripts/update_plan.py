"""Select pending work from small records, without opening article bodies/images."""
from __future__ import annotations

from collections import defaultdict
from urllib.parse import urlsplit

try:
    from .common import read_json
    from .enrich_posts import has_cached_metadata, summary_needs_refresh
    from .image_cache import hosted_image_url
    from .mirror_store import age_days
except ImportError:
    from common import read_json
    from enrich_posts import has_cached_metadata, summary_needs_refresh
    from image_cache import hosted_image_url
    from mirror_store import age_days


def waiting(timestamp):
    return bool(timestamp and age_days(timestamp) < 0)


def pending_articles(posts, state, metadata, directory, policy, refresh_ids, *,
                     force=False, skip_metadata=False, retry_failed=False):
    reasons = defaultdict(list)
    withdrawn = set(map(str, policy.get("withdrawn_ids", [])))
    skipped_hosts = set(policy.get("skip_image_hosts", []))
    for post in posts:
        key = str(post["id"])
        saved = state.get(key, {})
        status = saved.get("status")
        meta = metadata.get(key, {})
        if key in withdrawn:
            if status != "withdrawn":
                reasons[key].append("withdrawal")
            continue
        if waiting(saved.get("retry_not_before")) or waiting(meta.get("metadata_retry_not_before")):
            continue
        if force or key in refresh_ids:
            reasons[key].append("refresh")
        elif status == "removed":
            continue
        elif status == "fetch_failed":
            if retry_failed or not saved.get("checked_at") or age_days(saved["checked_at"]) >= 1:
                reasons[key].append("body_retry")
            else:
                continue
        elif status != "verified":
            reasons[key].append("new" if not status else "body_repair")
        if not skip_metadata:
            if meta.get("metadata_error"):
                if retry_failed or not meta.get("metadata_checked_at") or age_days(meta["metadata_checked_at"]) >= 1:
                    reasons[key].append("metadata_retry")
            elif not has_cached_metadata(meta) or summary_needs_refresh(meta):
                reasons[key].append("metadata")
        if status != "verified":
            continue
        if saved.get("images_pending"):
            reasons[key].append("unfinished_images")
        # Read only the small retry manifest, never image bytes or body hashes.
        try:
            images = read_json(directory / key / "images.json", {}).get("images", {})
            for url, image in images.items():
                hosted = hosted_image_url(url)
                if not hosted or urlsplit(hosted).hostname in skipped_hosts or image.get("skipped_host") in skipped_hosts:
                    continue
                if image.get("status") == "cached" and not image.get("error"):
                    continue
                if waiting(image.get("retry_not_before")) and (image.get("server_deferred") or not retry_failed):
                    continue
                if image.get("status") == "failed" and not retry_failed and age_days(image["checked_at"]) < 1:
                    continue
                reasons[key].append("image_retry")
                break
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            reasons[key].append("image_manifest_repair")
    return dict(reasons)
