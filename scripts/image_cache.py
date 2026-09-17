"""Source-host image caching. Hosting scope is not a claim about copyright."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import tinycss2

try:
    from .mirror_content import MirrorError, digest
    from .paths import ROOT
except ImportError:
    from mirror_content import MirrorError, digest
    from paths import ROOT

IMAGE_HOSTS = frozenset({"spaces.ac.cn", "album.spaces.ac.cn", "bbs.spaces.ac.cn"})
IMAGE_CACHE_DIR = ROOT / ".cache" / "image-cache"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
IMAGE_NAME = re.compile(r"[a-f0-9]{64}\.(?:png|jpg|gif|webp|svg|avif|bmp|ico)")
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
SVG_METADATA_NAMESPACES = frozenset({
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://purl.org/dc/elements/1.1/",
    "http://purl.org/dc/terms/",
    "http://creativecommons.org/ns#",
})


def hosted_image_url(url: str) -> str | None:
    try:
        parts = urlsplit(url)
        if (parts.scheme not in {"http", "https"} or parts.hostname not in IMAGE_HOSTS
                or parts.username or parts.password or parts.port not in (None, 80, 443)):
            return None
        return urlunsplit(("https", parts.hostname, parts.path or "/", parts.query, ""))
    except ValueError:
        return None


def image_urls(tree: list) -> list[str]:
    result = []
    def walk(node):
        if isinstance(node, dict):
            if node.get("tag") == "img" and node.get("attrs", {}).get("src"):
                result.append(node["attrs"]["src"])
            for child in node.get("children", []):
                walk(child)
    for node in tree:
        walk(node)
    return list(dict.fromkeys(result))


def validate_svg_css(css: str):
    """Inspect decoded CSS tokens without changing the original stylesheet.

    Exported MathJax uses about:blank for a dummy font; it makes no external
    request. Parsing handles CSS escapes/comments in URLs and @import rules.
    """
    def local_reference(value):
        return value.startswith("#") or value.lower() == "about:blank"

    pending = list(tinycss2.parse_component_value_list(css, skip_comments=True))
    while pending:
        token = pending.pop()
        if token.type == "error":
            raise MirrorError("Malformed SVG style")
        if token.type == "at-keyword" and token.lower_value == "import":
            raise MirrorError("External SVG style resource: @import")
        if token.type == "ident" and token.lower_value in {"behavior", "-moz-binding"}:
            raise MirrorError("Active SVG style")
        if token.type == "url" and not local_reference(token.value):
            raise MirrorError("External SVG style resource: url")
        if token.type == "function":
            args = [item for item in token.arguments if item.type not in {"whitespace", "comment"}]
            if token.lower_name == "url":
                if len(args) != 1 or args[0].type != "string" or not local_reference(args[0].value):
                    raise MirrorError("External SVG style resource: url()")
            if token.lower_name == "expression":
                raise MirrorError("Active SVG style")
            if token.lower_name in {"image", "image-set", "-webkit-image-set"}:
                if any(item.type == "string" and not local_reference(item.value) for item in args):
                    raise MirrorError("External SVG style resource: image()")
            pending.extend(token.arguments)
        pending.extend(getattr(token, "content", ()) or ())


def image_extension(raw: bytes) -> str:
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise MirrorError("Empty or oversized image")
    if raw.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
    if raw.startswith(b"\xff\xd8\xff"): return ".jpg"
    if raw.startswith((b"GIF87a", b"GIF89a")): return ".gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP": return ".webp"
    if raw[4:8] == b"ftyp" and raw[8:12] in (b"avif", b"avis"): return ".avif"
    if raw.startswith(b"BM"): return ".bmp"
    if raw.startswith(b"\x00\x00\x01\x00"): return ".ico"
    # Keep SVG bytes unchanged, but reject active content and external resources.
    if b"<!ENTITY" in raw.upper() or re.search(br"<!DOCTYPE[^>]*\[", raw, re.I):
        raise MirrorError("SVG declarations require review")
    try:
        root = ElementTree.fromstring(raw)
        if root.tag.rsplit("}", 1)[-1].lower() != "svg":
            raise MirrorError("Response is not an image")
        forbidden = {"script", "iframe", "object", "embed", "animate", "set", "animatetransform", "animatemotion"}
        passive_html = {"div", "span", "p", "br", "b", "strong", "i", "em", "u", "s", "sub", "sup", "font", "small", "a", "nobr", "style"}
        # Matplotlib exports RDF/Dublin Core/CC descriptions in metadata.
        # They identify authors, tools and dates, not embedded renderable HTML.
        metadata_nodes = {child for node in root.iter()
                          if node.tag in {"metadata", f"{{{SVG_NAMESPACE}}}metadata"}
                          for child in node.iter()}
        for node in root.iter():
            tag = node.tag.rsplit("}", 1)[-1].lower()
            namespace = node.tag[1:].split("}", 1)[0] if node.tag.startswith("{") else ""
            # MathJax stores original TeX in non-executable XHTML data blocks.
            math_data = (tag == "script" and namespace == XHTML_NAMESPACE
                         and node.get("type", "").split(";", 1)[0].strip().lower() == "math/tex"
                         and not len(node)
                         and not any(key.rsplit("}", 1)[-1].lower() in {"src", "href"} for key in node.attrib))
            if tag in forbidden and not math_data:
                raise MirrorError("Active SVG is not cached")
            # draw.io uses foreignObject with XHTML text labels. Keep these
            # unchanged. Metadata still goes through attribute/resource checks.
            if namespace and namespace != SVG_NAMESPACE:
                passive_label = namespace == XHTML_NAMESPACE and (tag in passive_html or math_data)
                descriptive_metadata = node in metadata_nodes and namespace in SVG_METADATA_NAMESPACES
                if not passive_label and not descriptive_metadata:
                    raise MirrorError(f"Unsupported embedded SVG content: {node.tag[:160]}")
            for key, value in node.attrib.items():
                key = key.rsplit("}", 1)[-1].lower()
                passive_link = tag == "a" and key == "href" and urlsplit(value).scheme in {"http", "https"}
                if key.startswith("on") or (key in {"href", "src"} and value and not value.startswith("#") and not passive_link):
                    raise MirrorError("Active or external SVG resource")
                if key in {"style", "fill", "stroke", "clip-path", "filter", "mask", "marker", "marker-start", "marker-mid", "marker-end", "cursor"}:
                    validate_svg_css(value)
            if tag == "style":
                if len(node):
                    raise MirrorError("Embedded markup in SVG style")
                validate_svg_css(node.text or "")
        return ".svg"
    except ElementTree.ParseError as exc:
        raise MirrorError("Unsupported image response") from exc


def cached_image_bytes(folder: Path, info: dict) -> bytes:
    name = info.get("file", "")
    if not isinstance(name, str) or not IMAGE_NAME.fullmatch(name):
        raise MirrorError("Invalid local image path")
    path = folder / "images" / name
    if not path.resolve().is_relative_to(folder.resolve()):
        raise MirrorError("Image path escapes article directory")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise MirrorError("Oversized cached image")
    raw = path.read_bytes()
    if digest(raw) != info.get("sha256") or name != digest(raw) + image_extension(raw):
        raise MirrorError("Local image hash/type mismatch")
    return raw


class ImageHostPaused(MirrorError):
    def __init__(self, host, failure):
        self.host, self.failure = host, failure
        super().__init__(f"Image host {host} is paused: {failure['error']}")


class ImageHostSkipped(MirrorError):
    def __init__(self, host):
        self.host = host
        super().__init__(f"Image host {host} is skipped by config/mirror.json")


def fetch_image(request, url, headers):
    current = hosted_image_url(url)
    for _ in range(5):
        if not current:
            raise MirrorError("Image redirects outside source hosts")
        response = request(current, headers)
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response, current
        location = response.headers.get("Location")
        if not location:
            raise MirrorError("Image redirect has no location")
        current = hosted_image_url(urljoin(current, location))
        headers = {}
    raise MirrorError("Too many image redirects")


def sync_images(directory, post_ids, *, interval=3, attempts=3, refresh_ids=(), retry_failed=False,
                max_requests=None, verified_image_urls=None, log=print):
    # Imports here avoid cycles with the body extractor and publication verifier.
    try:
        from .common import read_json
        from .extract_articles import atomic_json
        from .mirror_articles import SerialFetcher, DeferredRetry
        from .mirror_store import now_iso, age_days, verified_article, load_config
    except ImportError:
        from common import read_json
        from extract_articles import atomic_json
        from mirror_articles import SerialFetcher, DeferredRetry
        from mirror_store import now_iso, age_days, verified_article, load_config
    import requests
    from collections import Counter
    from datetime import datetime, timedelta, timezone

    counts = Counter()
    skipped_hosts = set(load_config(directory).get("skip_image_hosts", []))
    if not skipped_hosts <= IMAGE_HOSTS:
        raise MirrorError("skip_image_hosts must contain only supported source image hosts")
    if skipped_hosts:
        log("Image hosts skipped by policy: " + ", ".join(sorted(skipped_hosts)))
    transports, ready_hosts, shared, checked_urls = {}, set(), {}, set()
    consecutive_errors = Counter()
    # Host cooldowns are runtime state, scoped to this content directory. In
    # particular, Retry-After applies to every URL on a host, across invocations.
    host_state_path = IMAGE_CACHE_DIR / f"hosts-{digest(str(directory.resolve()))[:16]}.json"
    previous_hosts = read_json(host_state_path, {})
    paused_hosts = {host: info for host, info in previous_hosts.items()
                    if host in IMAGE_HOSTS and info.get("retry_not_before")
                    and age_days(info["retry_not_before"]) < 0
                    and (info.get("server_deferred") or not retry_failed)}
    if previous_hosts != paused_hosts:
        atomic_json(host_state_path, paused_hosts)

    def failure_info(exc):
        if isinstance(exc, ImageHostPaused):
            return dict(exc.failure)
        retry_at = exc.retry_at if isinstance(exc, DeferredRetry) else (
            datetime.now(timezone.utc) + timedelta(days=7 if 'HTTP 404' in str(exc) or 'HTTP 410' in str(exc) else 1)).isoformat(timespec="seconds")
        return {"error": str(exc), "retry_not_before": retry_at,
                "server_deferred": isinstance(exc, DeferredRetry)}

    def pause_host(host, failure):
        if host in paused_hosts:
            return
        paused_hosts[host] = failure
        atomic_json(host_state_path, paused_hosts)
        counts["hosts_paused"] += 1
        log(f"HOST {host} PAUSED until {failure['retry_not_before']}: {failure['error']}. "
            "Skipping further requests to this host; continuing other source hosts.")

    def request(url, headers):
        host = urlsplit(url).hostname
        if host in skipped_hosts:
            raise ImageHostSkipped(host)
        if host in paused_hosts:
            raise ImageHostPaused(host, paused_hosts[host])
        if host not in transports:
            transports[host] = SerialFetcher(interval, attempts, host=host, max_bytes=MAX_IMAGE_BYTES)
        fetcher = transports[host]
        checking_robots = host not in ready_hosts
        try:
            if checking_robots:
                fetcher.check_robots()
                ready_hosts.add(host)
                checking_robots = False
            response = fetcher.get(url, headers)
        except (OSError, ValueError, KeyError, requests.RequestException) as exc:
            network_failure = isinstance(exc, requests.RequestException) or any(
                token in str(exc) for token in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "Fetch failed"))
            consecutive_errors[host] = consecutive_errors[host] + 1 if network_failure else 0
            # The transport already retries each request. A failed robots check
            # blocks the host immediately, rather than retrying it for each image.
            if checking_robots or isinstance(exc, DeferredRetry) or consecutive_errors[host] >= 3:
                pause_host(host, failure_info(exc))
            raise
        consecutive_errors[host] = 0
        return response
    try:
        for post_id in post_ids:
            folder = directory / post_id
            if verified_image_urls is not None and post_id in verified_image_urls:
                # The unified extractor just validated this body. This in-memory
                # handoff avoids parsing all bodies twice before publication.
                source_urls = verified_image_urls[post_id]
            else:
                article = verified_article(directory, post_id, local_only=True)
                source_urls = image_urls(article["tree"])
            path = folder / "images.json"
            previous = read_json(path, {"version": 1, "images": {}})
            old = previous.get("images", {})
            urls = [url for url in source_urls if hosted_image_url(url)]
            if not urls and not path.exists():
                continue
            manifest = {"version": 1, "images": {url: old[url] for url in urls if url in old}}
            for url in urls:
                key = hosted_image_url(url)
                info = manifest["images"].get(url, {})
                valid = False
                if info.get("status") == "cached":
                    try:
                        cached_image_bytes(folder, info)
                        valid = True
                    except (OSError, ValueError, KeyError):
                        pass
                if urlsplit(key).hostname in skipped_hosts or info.get("skipped_host") in skipped_hosts:
                    counts["cached" if valid else "skipped"] += 1
                    continue
                if (info.get("retry_not_before") and age_days(info["retry_not_before"]) < 0
                        and (info.get("server_deferred") or not retry_failed)):
                    counts["deferred"] += 1
                    continue
                if valid and post_id not in refresh_ids and not info.get("error"):
                    counts["cached"] += 1
                    shared.setdefault(key, (folder, info))
                    continue
                if info.get("status") == "failed" and not retry_failed and age_days(info["checked_at"]) < 1:
                    counts["deferred"] += 1
                    continue
                reusable = key in shared and (post_id not in refresh_ids or key in checked_urls)
                if not reusable and urlsplit(key).hostname in paused_hosts:
                    counts["deferred"] += 1
                    counts["host_deferred"] += 1
                    continue
                if not reusable and max_requests is not None and counts["requests"] >= max_requests:
                    counts["pending"] += 1
                    continue
                try:
                    if reusable:
                        other_folder, entry = shared[key]
                        raw = cached_image_bytes(other_folder, entry)
                        entry = dict(entry)
                        counts["reused"] += 1
                    else:
                        headers = {h: info[k] for h, k in (("If-None-Match", "etag"), ("If-Modified-Since", "last_modified")) if valid and info.get(k)}
                        counts["requests"] += 1
                        response, final_url = fetch_image(request, url, headers)
                        if response.status_code == 304 and valid and headers:
                            raw = cached_image_bytes(folder, info)
                            entry = info | {"checked_at": now_iso()}
                            counts["not_modified"] += 1
                        elif response.status_code == 200:
                            raw = response.content
                            name = digest(raw) + image_extension(raw)
                            entry = {"status": "cached", "file": name, "sha256": digest(raw), "bytes": len(raw),
                                     "checked_at": now_iso(), "fetched_url": final_url,
                                     "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
                            counts["downloaded"] += 1
                        else:
                            raise MirrorError(f"Image HTTP {response.status_code}")
                        checked_urls.add(key)
                    entry.pop("retry_not_before", None)
                    entry.pop("error", None)
                    entry.pop("server_deferred", None)
                    entry.pop("skipped_host", None)
                    image_dir = folder / "images"
                    if not image_dir.resolve().is_relative_to(folder.resolve()):
                        raise MirrorError("Image directory escapes article directory")
                    image_dir.mkdir(exist_ok=True)
                    target = image_dir / entry["file"]
                    if not target.exists() or target.read_bytes() != raw:
                        temporary = target.with_suffix(target.suffix + ".tmp")
                        temporary.write_bytes(raw)
                        temporary.replace(target)
                    manifest["images"][url] = entry
                    shared[key] = (folder, entry)
                    log(f"IMAGE {post_id} {counts['requests']} OK {len(raw)} bytes {url}")
                except (OSError, ValueError, KeyError, requests.RequestException) as exc:
                    if isinstance(exc, ImageHostSkipped):
                        # Remember a skipped redirect target too, so the next
                        # run need not request its source URL again.
                        manifest["images"][url] = (info if valid else {"status": "skipped"}) | {"skipped_host": exc.host}
                        counts["skipped"] += 1
                        log(f"IMAGE {post_id} SKIPPED {url}: {exc}")
                        continue
                    # Image failures never invalidate an otherwise verified article.
                    failure = failure_info(exc)
                    if valid and 'HTTP 404' not in str(exc) and 'HTTP 410' not in str(exc):
                        manifest["images"][url] = info | failure
                    else:
                        manifest["images"][url] = {"status": "failed", "checked_at": now_iso(), **failure}
                    if isinstance(exc, ImageHostPaused):
                        counts["deferred"] += 1
                        counts["host_deferred"] += 1
                        log(f"IMAGE {post_id} DEFERRED {url}: {exc}")
                    else:
                        counts["failed"] += 1
                        log(f"IMAGE {post_id} FAILED {url}: {exc}")
                finally:
                    atomic_json(path, manifest)
            if previous != manifest:
                atomic_json(path, manifest)
            # Prune only generated image names, after the new manifest is installed.
            keep = {entry.get("file") for entry in manifest["images"].values() if entry.get("status") == "cached"}
            image_dir = folder / "images"
            if image_dir.exists() and image_dir.resolve().is_relative_to(folder.resolve()):
                for cached in image_dir.iterdir():
                    if cached.is_file() and IMAGE_NAME.fullmatch(cached.name) and cached.name not in keep:
                        cached.unlink()
    finally:
        for fetcher in transports.values():
            fetcher.session.close()
    return dict(counts)
