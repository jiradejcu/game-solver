#!/usr/bin/env python3
"""Static file server for the solver, plus dev-only sinks for the browser.

The page (see js/devlog.js) POSTs batches of console output to /__log, which
land in dev-console.log. That way the browser's console is readable from the
WSL side -- no Chrome remote-debugging flags, no separate profile.

The "Capture snapshot" button (see js/app.js) POSTs a PNG data URL of the
current camera + overlay to /__snapshot, which lands in snapshot.png -- a way
to actually look at a frame of what the vision pipeline is doing, not just
read numbers about it.

The page also polls GET /__version (see js/livereload.js) and reloads itself
whenever a watched source file's mtime changes, so edits show up without a
manual refresh.

The page separately polls GET /__snapshot-request-version (see js/app.js)
and captures+POSTs a fresh snapshot whenever snapshot-request.flag's mtime
changes -- so an external process (e.g. an agent iterating on vision.js) can
request a snapshot by touching that file, without a human clicking the
button.
"""
import base64
import http.server
import json
import os
import socketserver
import sys
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "dev-console.log")
SNAPSHOT_PATH = os.path.join(ROOT, "snapshot.png")
SNAPSHOT_REQUEST_PATH = os.path.join(ROOT, "snapshot-request.flag")
WATCHED_EXTS = (".html", ".css", ".js")


def file_mtime_or_zero(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def latest_source_mtime():
    latest = 0.0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for name in filenames:
            if name.endswith(WATCHED_EXTS):
                try:
                    latest = max(latest, os.path.getmtime(os.path.join(dirpath, name)))
                except OSError:
                    pass
    return latest


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        # Chrome DevTools auto-probes this on every load looking for a
        # workspace-folder mapping; it's optional and we don't provide one,
        # so answer "no content" instead of letting it 404-spam the log.
        if self.path == "/.well-known/appspecific/com.chrome.devtools.json":
            self.send_response(204)
            self.end_headers()
            return

        if self.path == "/__version":
            body = json.dumps({"version": latest_source_mtime()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/__snapshot-request-version":
            body = json.dumps({"version": file_mtime_or_zero(SNAPSHOT_REQUEST_PATH)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/__snapshot":
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8", errors="ignore")
            prefix = "data:image/png;base64,"
            if raw.startswith(prefix):
                try:
                    with open(SNAPSHOT_PATH, "wb") as fh:
                        fh.write(base64.b64decode(raw[len(prefix):]))
                except (ValueError, OSError):
                    self.send_error(400)
                    return
            self.send_response(204)
            self.end_headers()
            return

        if self.path != "/__log":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            entries = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            entries = []

        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            for entry in entries:
                stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                level = str(entry.get("level", "log")).upper()
                text = entry.get("text", "")
                where = entry.get("where") or ""
                suffix = f"  ({where})" if where else ""
                fh.write(f"[{stamp}] {level:<5} {text}{suffix}\n")

        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        # Always serve fresh files; a stale cached js/ file during debugging
        # wastes more time than the requests save.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Drop the per-request noise, but keep genuine errors visible.
        status = str(args[1]) if len(args) > 1 else ""
        if not status.startswith(("2", "3")):
            super().log_message(fmt, *args)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    # Start each run with a clean log so old output is never mistaken for new.
    open(LOG_PATH, "w").close()
    print(f"Serving {ROOT} on http://localhost:{PORT}")
    print(f"Browser console -> {LOG_PATH}")
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
