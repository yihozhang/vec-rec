#!/usr/bin/env python3
"""Serve the VecRec benchmark visualizer with a dropdown of result files.

Usage:
    uv run python benchmarks/serve_visualizer.py [--results-dir DIR] [--port N]

Endpoints:
    GET /            -> visualizer.html
    GET /list        -> JSON [{filename, mtime, timestamp, description, passed, failed}, ...]
                        sorted by mtime descending.
    GET /results/X.json -> the raw result file (X must be in --results-dir).
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.BaseHTTPRequestHandler):
    results_dir: str = "results"

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html", "/visualizer.html"):
            with open(os.path.join(ROOT, "visualizer.html"), "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read())
            return

        if path == "/list":
            self._send(200, "application/json", json.dumps(self._list_results()).encode())
            return

        if path.startswith("/results/"):
            name = path[len("/results/"):]
            # Defend against path traversal: no separators, must end in .json.
            if "/" in name or "\\" in name or ".." in name or not name.endswith(".json"):
                self._send(400, "text/plain", b"bad filename")
                return
            full = os.path.join(self.results_dir, name)
            if not os.path.isfile(full):
                self._send(404, "text/plain", b"not found")
                return
            with open(full, "rb") as f:
                self._send(200, "application/json", f.read())
            return

        self._send(404, "text/plain", b"not found")

    def _list_results(self) -> list[dict[str, object]]:
        entries: list[dict[str, object]] = []
        if not os.path.isdir(self.results_dir):
            return entries
        for name in os.listdir(self.results_dir):
            if not name.endswith(".json"):
                continue
            full = os.path.join(self.results_dir, name)
            if not os.path.isfile(full):
                continue
            info: dict[str, object] = {"filename": name, "mtime": os.path.getmtime(full)}
            try:
                with open(full) as f:
                    payload = json.load(f)
                md = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                info["timestamp"] = md.get("timestamp", "")
                info["description"] = md.get("description", "")
                info["passed"] = md.get("passed")
                info["failed"] = md.get("failed")
                info["n_benchmarks"] = len(payload.get("benchmarks", [])) if isinstance(payload, dict) else 0
            except Exception as exc:
                info["error"] = str(exc)
            entries.append(info)
        entries.sort(key=lambda e: e.get("mtime", 0.0), reverse=True)  # type: ignore[arg-type, return-value]
        return entries

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results", help="Folder of JSON result files (default: results).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    Handler.results_dir = os.path.abspath(args.results_dir)
    print(f"Visualizer:   http://{args.host}:{args.port}/")
    print(f"Results dir:  {Handler.results_dir}")

    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
