#!/usr/bin/env python3
"""
Minimal local telemetry sink (no external deps).
Accepts JSON lines via stdin or appends incoming JSON via a simple HTTP POST handler.
Writes to results/time_telemetry/telemetry_<ts>.jsonl
"""
import http.server
import socketserver
import json
import os
import time

OUT_DIR = os.environ.get("DRIFTLOCK_TEL_DIR", "results/time_telemetry")
os.makedirs(OUT_DIR, exist_ok=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw.decode('utf-8'))
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        path = os.path.join(OUT_DIR, f"ingest_{int(time.time())}.jsonl")
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj) + "\n")
        self.send_response(204)
        self.end_headers()


def main():
    port = int(os.environ.get("DRIFTLOCK_TEL_PORT", "8089"))
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Driftlock telemetry server on http://localhost:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()

