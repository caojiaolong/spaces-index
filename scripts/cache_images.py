"""Backfill source-host images from verified local bodies, without refetching articles."""
from __future__ import annotations

import argparse
import json
import re

try:
    from .common import read_json
    from .paths import ARTICLES_DIR, ROOT
    from .extract_articles import run_lock, atomic_json
    from .image_cache import sync_images
    from .mirror_store import load_config, now_iso
except ImportError:
    from common import read_json
    from paths import ARTICLES_DIR, ROOT
    from extract_articles import run_lock, atomic_json
    from image_cache import sync_images
    from mirror_store import load_config, now_iso


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true')
    group.add_argument('--ids', nargs='+')
    parser.add_argument('--sleep', type=float, default=3)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Maximum image URLs to request for a trial')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failures now, still honor server Retry-After')
    parser.add_argument('--build', action='store_true', help='Rebuild preview and public artifacts after caching; no deployment')
    args = parser.parse_args(argv)
    if args.sleep < 3 or not 1 <= args.attempts <= 3 or (args.limit is not None and args.limit < 1):
        parser.error('Require --sleep >= 3, --attempts 1..3, and positive --limit')
    if args.ids and any(not re.fullmatch(r'[1-9]\d*', i) for i in args.ids):
        parser.error('Article IDs must be positive decimal integers')
    log_dir = ROOT / '.cache/image-cache'
    log_dir.mkdir(parents=True, exist_ok=True)
    def log(message):
        line = f'[{now_iso()}] {message}'
        print(line, flush=True)
        with (log_dir / 'run.log').open('a', encoding='utf8') as handle:
            handle.write(line + '\n')
    try:
        with run_lock(ARTICLES_DIR):
            state = read_json(ARTICLES_DIR / 'state.json', {})
            withdrawn = set(load_config().get('withdrawn_ids', []))
            candidates = args.ids or sorted(state, key=int, reverse=True)
            ids = [i for i in candidates if i not in withdrawn and state.get(i, {}).get('status') == 'verified']
            log(f'Start {len(ids)} verified articles; interval {args.sleep}s; existing files resume automatically.')
            result = sync_images(ARTICLES_DIR, ids, interval=args.sleep, attempts=args.attempts,
                                 retry_failed=args.retry_failed, max_requests=args.limit, log=log)
            atomic_json(log_dir / 'summary.json', {'finished_at': now_iso(), **result})
            log('SUMMARY ' + json.dumps(result))
            incomplete = any(result.get(key) for key in ('failed', 'stopped_early', 'pending', 'deferred'))
            if incomplete:
                log('Image caching is incomplete; failed, deferred or unvisited images remain. '
                    'Rerun to resume; --retry-failed retries ordinary failures without waiting for cooldown.')
            if args.build:
                try:
                    from .build_site import build_site
                    from .paths import PREVIEW_DIR, PUBLIC_DIR
                except ImportError:
                    from build_site import build_site
                    from paths import PREVIEW_DIR, PUBLIC_DIR
                log('Building preview and public artifacts with cached images; no deployment.')
                build_site(ROOT / 'data/posts_classified.json', PREVIEW_DIR, local_articles=ARTICLES_DIR)
                build_site(ROOT / 'data/posts_classified.json', PUBLIC_DIR)
                log('Both builds completed using currently available images; image caching is still incomplete.'
                    if incomplete else 'Image caching and both builds completed.')
            return 1 if incomplete else 0
    except KeyboardInterrupt:
        log('Interrupted; completed images are saved. Rerun the same command to resume.')
        return 130
    except (OSError, ValueError, KeyError) as exc:
        log(f'Image cache failed: {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
