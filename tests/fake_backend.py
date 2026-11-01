"""minimaler openai-kompatibler fake-server mit sse. for tests und den egress-check.

script = liste von antworten, jede entweder
  {"text": "..."}                                    oder
  {"tool_calls": [{"name": "read", "arguments": {...}}], "text": "optional"}
  "ollama_style": true streamt tool-calls wie litellm ollama_chat/ (alle mit index 0).
  "deltas": [...] schickt rohe deltas vorneweg (exotische formate).
ist das script leer, kommt "ok" back.
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeBackend:
    def __init__(self, script=None, host="127.0.0.1", port=0):
        self.script = list(script or [])
        self.by_model = {}  # eigene skripte pro modell, z.b. fuers titel-modell (running parallel)
        self.requests = []
        self.lock = threading.Lock()
        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with backend.lock:
                    backend.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                    queue = backend.by_model.get(body.get("model"), backend.script)
                    step = queue.pop(0) if queue else {"text": "ok"}
                if step.get("delay"):  # lahmes backend simulieren
                    time.sleep(step["delay"])
                if step.get("status"):
                    self.send_response(step["status"])
                    for k, v in (step.get("headers") or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(step.get("text", "broken").encode())
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i, chunk in enumerate(backend.chunks(step)):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    if i == 0 and step.get("stall"):  # hanger mitten im stream
                        time.sleep(step["stall"])
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.handle_error = lambda *a: None  # client weg (abbruch-tests) = kein stacktrace
        self.port = self.server.server_address[1]
        self.url = f"http://{host}:{self.port}/v1"

    def chunks(self, step):
        def wrap(delta, finish=None):
            return {"id": "x", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        for delta in step.get("deltas") or []:
            yield wrap(delta)
        if step.get("reasoning"):
            yield wrap({"reasoning_content": step["reasoning"]})
        text = step.get("text") or ""
        for i in range(0, len(text), 5):
            yield wrap({"content": text[i:i + 5]})
        for idx, tc in enumerate(step.get("tool_calls") or []):
            args = json.dumps(tc.get("arguments", {}))
            if step.get("ollama_style"):  # litellm ollama_chat/: ein chunk pro call, index immer 0
                yield wrap({"tool_calls": [{"index": 0, "id": f"uuid-{idx}-{len(self.requests)}", "type": "function",
                                            "function": {"name": tc["name"], "arguments": args}}]})
                continue
            half = len(args) // 2
            yield wrap({"tool_calls": [{"index": idx, "id": f"call_{idx}_{len(self.requests)}", "type": "function",
                                        "function": {"name": tc["name"], "arguments": args[:half]}}]})
            yield wrap({"tool_calls": [{"index": idx, "function": {"arguments": args[half:]}}]})
        yield wrap({}, step.get("finish") or ("tool_calls" if step.get("tool_calls") else "stop"))
        usage = {"prompt_tokens": 1000 + 10 * len(self.requests), "completion_tokens": 42}
        usage.update(step.get("usage") or {})  # z.b. {"prompt_tokens": 90000} for "kontext fast voll"
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        yield {"id": "x", "object": "chat.completion.chunk", "choices": [], "usage": usage}

    def start(self):
        # kurzes poll-intervall, sonst kostet shutdown() pro test 0.5s
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--script", help="json-datei mit antworten")
    a = ap.parse_args()
    script = json.load(open(a.script)) if a.script else []
    fb = FakeBackend(script, port=a.port)
    print(fb.port, flush=True)
    try:
        fb.server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
