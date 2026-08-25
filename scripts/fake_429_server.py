"""Tiny HTTP server that answers every request with 429. Used by `make fallback-test`."""
from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def _r(self):
        body = b'{"error":{"message":"Rate limit exceeded: free-models-per-day","code":429}}'
        self.send_response(429)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        sys.stderr.write(f"fake-429: {self.command} {self.path} -> 429\n")

    do_GET = do_POST = _r

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8429
    print(f"fake 429 server on 127.0.0.1:{port}", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
