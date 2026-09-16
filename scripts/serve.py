"""Serve only the generated website, bound to localhost."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from .paths import PREVIEW_DIR
except ImportError:
    from paths import PREVIEW_DIR


def serve(directory: Path = PREVIEW_DIR, port: int = 8765):
    if not (directory / ".spaces-index-generated-site").is_file():
        raise ValueError("Build the website first with scripts/update_all.py")
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"Preview: http://127.0.0.1:{port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--directory", type=Path, default=PREVIEW_DIR)
    args = parser.parse_args()
    serve(args.directory, args.port)
