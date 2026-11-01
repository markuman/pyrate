"""fake-mcp-server for tests. stdio (default) oder streamable http (--http).

  python fake_mcp.py                 # stdio
  python fake_mcp.py --crash         # stirbt beim initialize, mit stderr-meldung
  python fake_mcp.py --garbage       # muell auf stdout vor jeder antwort
"""

import argparse
import itertools
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE1 = [
    {"name": "echo", "description": "returns text",
     "inputSchema": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                     "properties": {"text": {"type": "string"}}, "required": ["text"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "add", "description": "a+b", "inputSchema": {"type": "object", "properties": {
        "a": {"type": "number"}, "b": {"type": "number"}}}},
    {"name": "fail", "description": "liefert isError", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "schlaeft", "inputSchema": {"type": "object", "properties": {
        "seconds": {"type": "number"}}}},
]
PAGE2 = [
    {"name": "big", "description": "viel output", "inputSchema": {"type": "object", "properties": {
        "lines": {"type": "integer"}}}},
    {"name": "needs_ping", "description": "pingt erst den client", "inputSchema": {"type": "object"}},
    {"name": "env", "description": "gibt FAKE_TOKEN back", "inputSchema": {"type": "object"}},
    {"name": "weird.name/x", "description": "komischer name", "inputSchema": {"type": "object"}},
    {"name": "structured", "description": "nur structuredContent", "inputSchema": {"type": "object"}},
    {"name": "grow", "description": "fuegt ein tool hinzu + list_changed", "inputSchema": {"type": "object"}},
]


class Server:
    def __init__(self, send):
        self.send = send          # callable(msg)
        self.ids = itertools.count(1000)
        self.waiting = {}         # server-request-id → event+antwort
        self.cancelled = set()
        self.extra_tools = []

    def handle(self, msg):
        if "id" in msg and ("result" in msg or "error" in msg):  # antwort auf unseren ping
            w = self.waiting.get(msg["id"])
            if w:
                w["msg"] = msg
                w["ev"].set()
            return None
        method, rid, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "notifications/cancelled":
            self.cancelled.add(params.get("requestId"))
            return None
        if rid is None:
            return None
        if method == "initialize":
            return self.ok(rid, {"protocolVersion": params.get("protocolVersion"),
                                 "capabilities": {"tools": {"listChanged": True}},
                                 "serverInfo": {"name": "fake-mcp", "version": "1"}})
        if method == "tools/list":
            if params.get("cursor") == "p2":
                return self.ok(rid, {"tools": PAGE2 + self.extra_tools})
            return self.ok(rid, {"tools": PAGE1, "nextCursor": "p2"})
        if method == "tools/call":
            return self.call(rid, params.get("name"), params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown {method}"}}

    def ok(self, rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def text(self, rid, t, error=False):
        return self.ok(rid, {"content": [{"type": "text", "text": t}], "isError": error})

    def call(self, rid, name, a):
        if name == "echo":
            return self.text(rid, a.get("text", ""))
        if name == "add":
            return self.text(rid, str(a.get("a", 0) + a.get("b", 0)))
        if name == "fail":
            return self.text(rid, "broken", error=True)
        if name == "slow":
            end = time.monotonic() + a.get("seconds", 1)
            while time.monotonic() < end:
                if rid in self.cancelled:
                    return None  # laut spec: auf abgebrochene requests nicht mehr antworten
                time.sleep(0.02)
            return self.text(rid, "ausgeschlafen")
        if name == "big":
            return self.text(rid, "\n".join(f"zeile {i}" for i in range(a.get("lines", 10))))
        if name == "needs_ping":
            pid = next(self.ids)
            w = self.waiting[pid] = {"ev": threading.Event(), "msg": None}
            self.send({"jsonrpc": "2.0", "id": pid, "method": "ping"})
            self.send({"jsonrpc": "2.0", "id": next(self.ids), "method": "sampling/createMessage", "params": {}})
            self.send({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "x"}})
            if not w["ev"].wait(5):
                return self.text(rid, "keine ping-antwort", error=True)
            return self.text(rid, f"pong ok: {json.dumps(w['msg'].get('result'))}")
        if name == "env":
            return self.text(rid, os.environ.get("FAKE_TOKEN", "<unset>"))
        if name == "weird.name/x":
            return self.text(rid, "weird ok")
        if name == "structured":
            return self.ok(rid, {"content": [], "structuredContent": {"temp": 21.5, "unit": "C"}})
        if name == "grow":
            self.extra_tools = [{"name": "late", "description": "kam later", "inputSchema": {"type": "object"}}]
            self.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
            return self.text(rid, "gewachsen")
        return self.text(rid, f"unknown tool {name}", error=True)


def run_stdio(args):
    lock = threading.Lock()

    def send(msg):
        with lock:
            if args.garbage:
                sys.stdout.write("das ist kein json\n")
            sys.stdout.write(json.dumps(msg) + "\n")
            sys.stdout.flush()

    srv = Server(send)
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if args.crash and msg.get("method") == "initialize":
            print("fatal: kann config nicht laden", file=sys.stderr, flush=True)
            sys.exit(3)

        def work(m=msg):
            reply = srv.handle(m)
            if reply is not None:
                send(reply)
        threading.Thread(target=work, daemon=True).start()


class HTTPFake:
    """streamable http: requests mit 'echo'/'needs_ping'/'slow' als sse, rest als json."""

    def __init__(self, require_session=True):
        self.requests = []
        self.deleted = []
        self.require_session = require_session
        fake = self
        self.sessions = {}

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_DELETE(self):
                fake.deleted.append(self.headers.get("Mcp-Session-Id"))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"headers": dict(self.headers), "body": msg})
                sid = self.headers.get("Mcp-Session-Id")
                if msg.get("method") == "initialize":
                    sid = f"sess-{len(fake.sessions) + 1}"
                    fake.sessions[sid] = Server(None)
                elif fake.require_session and sid not in fake.sessions:
                    self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                srv = fake.sessions[sid]
                if "id" not in msg or "method" not in msg:  # notification oder antwort
                    srv.handle(msg)
                    self.send_response(202)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                name = (msg.get("params") or {}).get("name")
                if msg["method"] == "tools/call" and name in ("echo", "needs_ping", "slow"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Mcp-Session-Id", sid)
                    self.end_headers()
                    lock = threading.Lock()

                    def send(m):
                        with lock:
                            self.wfile.write(b": kommentar\n")
                            self.wfile.write(f"event: message\ndata: {json.dumps(m)}\n\n".encode())
                            self.wfile.flush()
                    srv.send = send
                    reply = srv.handle(msg)
                    if reply is not None:
                        send(reply)
                    self.close_connection = True
                    return
                reply = srv.handle(msg)
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Mcp-Session-Id", sid)
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.handle_error = lambda *a: None
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/mcp"

    def start(self):
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", action="store_true")
    ap.add_argument("--garbage", action="store_true")
    run_stdio(ap.parse_args())
