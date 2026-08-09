#!/usr/bin/env python3
"""Static file server for the solver, plus a sink for the browser's console.

The page (see js/devlog.js) POSTs batches of console output to /__log, which
land in dev-console.log. That way the browser's console is readable from the
WSL side -- no Chrome remote-debugging flags, no separate profile.
"""
import http.server
import json
import os
import socketserver
import sys
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "dev-console.log")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
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
