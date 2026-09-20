#!/usr/bin/env python3
"""Tiny echo provider for kev-router live demo: replies with the model it received."""
import http.server
import json
from pathlib import Path

REPLY_FILE = Path(__file__).parent / "echo_reply.json"


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            model = json.loads(body).get("model", "MISSING")
        except Exception:
            model = "PARSE-ERROR"
        reply = json.loads(REPLY_FILE.read_text())
        reply["model"] = f"echo-saw: {model}"
        reply["choices"][0]["message"]["content"] = json.dumps(
            {"received_model": model})
        out = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    http.server.HTTPServer(("127.0.0.1", 8099), H).serve_forever()
