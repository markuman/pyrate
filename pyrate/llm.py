"""openai-compatible chat call with sse streaming. stdlib only (urllib)."""

import email.utils
import http.client
import ipaddress
import json
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from . import __version__
from .config import ConfigError
from .oauth import NoRedirect


class LLMError(Exception):
    def __init__(self, message, status=None, body=None, retryable=False, retry_after=None):
        super().__init__(message)
        self.status = status
        self.body = body
        self.retryable = retryable      # transient (429/5xx/connreset), another attempt is worth it
        self.retry_after = retry_after  # seconds per server, or None


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- retry
# only before the first streamed byte: what already reached the user can't be
# taken back. never on auth/quota/billing – waiting doesn't help there.

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
RETRY_MAX = 3      # extra attempts
RETRY_BASE = 2.0   # s, doubles per attempt
RETRY_CAP = 60.0   # s; if the server wants longer, we give up
_NO_RETRY = re.compile(r"quota|billing|insufficient|credit|payment", re.I)
_RETRY_NET = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


def _retry_after(headers):
    raw = (headers or {}).get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        return max(0.0, email.utils.parsedate_to_datetime(raw).timestamp() - time.time())
    except (TypeError, ValueError):
        return None


def retry_delay(err, attempt):
    """wait before attempt attempt+1, or None = give up."""
    if not getattr(err, "retryable", False) or attempt >= RETRY_MAX:
        return None
    if err.retry_after is not None:
        return err.retry_after if err.retry_after <= RETRY_CAP else None
    delay = min(RETRY_CAP, RETRY_BASE * 2 ** attempt)
    return delay * random.uniform(0.75, 1.25)  # jitter, so parallel clients don't sync up and hammer together


# ---------------------------------------------------------------- cancellability
# a blocked recv (waiting for headers or the next sse chunk) doesn't check any
# cancel event. hence a dedicated handler remembers the http connection, and a
# watcher thread calls shutdown() on the socket on cancel. that wakes recv immediately.

CONNECT_TIMEOUT = 15  # s; read timeout stays cfg.request_timeout


class _Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.socks = []

    def add(self, sock):
        with self.lock:
            self.socks.append(sock)

    def kill(self):
        with self.lock:
            socks = list(self.socks)
        for sock in socks:
            try:
                # base method directly: SSLSocket.shutdown() would null _sslobj
                # while the reading thread is still stuck in it
                socket.socket.shutdown(sock, socket.SHUT_RDWR)
            except OSError:
                pass


class _TrackMixin:
    tracker = None

    def do_open(self, http_class, req, **kw):
        tracker = self.tracker

        def factory(*a, **k):
            conn = http_class(*a, **k)
            connect = conn.connect

            def tracked_connect():
                # before connect there is no socket to kill → keep connect (incl. tls handshake)
                # short, then restore the long read timeout
                read_timeout = conn.timeout
                conn.timeout = CONNECT_TIMEOUT if read_timeout is None else min(CONNECT_TIMEOUT, read_timeout)
                try:
                    connect()
                finally:
                    conn.timeout = read_timeout
                conn.sock.settimeout(read_timeout)
                # grab the socket right after connect: urllib sends "Connection: close",
                # so http.client passes it on to the response after getresponse()
                # and sets conn.sock = None. but it stays the same socket object.
                tracker.add(conn.sock)

            conn.connect = tracked_connect
            return conn

        return super().do_open(factory, req, **kw)


class _HTTPHandler(_TrackMixin, urllib.request.HTTPHandler):
    pass


class _HTTPSHandler(_TrackMixin, urllib.request.HTTPSHandler):
    pass


def _is_loopback(url):
    """localhost/127.0.0.0/8/::1 – such backends (ollama, vllm, …) must never go via an
    HTTP(S)_PROXY: with a proxy set and no NO_PROXY entry the request hangs in the proxy
    until the timeout, even though the model is warm."""
    host = (urlparse(url).hostname or "").strip("[]").lower()
    if host == "localhost" or host == "::1":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _opener(tracker, url=""):
    handlers = [_HTTPHandler(), _HTTPSHandler()]
    for h in handlers:
        h.tracker = tracker
    if _is_loopback(url):
        handlers.append(urllib.request.ProxyHandler({}))  # loopback: never via proxy
    # no redirects: urllib would take the api key along (even https → http). for non-loopback
    # the env proxies (HTTP_PROXY/…) stay in effect
    return urllib.request.build_opener(*handlers, NoRedirect())


def _watch(cancel, tracker, done):
    while not done.wait(0.05):
        if cancel.is_set():
            tracker.kill()
            return


def _cancelled(cancel):
    return cancel is not None and cancel.is_set()


def _prepare(backend, payload, session):
    """build headers, run the plugin over them. once per request, not per retry.

    → (headers, url). if the plugin's request() returns a str, that is the
    complete request url (default: f"{backend.base_url}/chat/completions").
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": f"pyrate/{__version__}",
    }
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"
    url = f"{backend.base_url}/chat/completions"
    if backend.plugin:
        try:
            ret = backend.plugin.request(backend, headers, payload, session)
        except Exception as e:
            raise LLMError(f"plugin {backend.plugin.__name__}: {e}") from e
        if isinstance(ret, str) and ret:
            url = ret
    return headers, url


def _request(url, timeout, payload, headers, tracker, cancel):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    n_socks = len(tracker.socks)  # tracker lebt über retries → nur sockets dieses versuchs zählen
    try:
        return _opener(tracker, url).open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:2000]
        raise LLMError(f"http {e.code} von {url}: {body}", status=e.code, body=body,
                       retryable=e.code in RETRY_STATUS and not _NO_RETRY.search(body),
                       retry_after=_retry_after(e.headers)) from e
    except (OSError, http.client.HTTPException) as e:
        # getresponse() errors (e.g. RemoteDisconnected) are not wrapped in URLError by urllib
        if _cancelled(cancel):
            raise Cancelled() from e
        reason = getattr(e, "reason", e)
        # tracker gets the socket only after a successful connect → no new one = connect phase.
        # tells apart "proxy swallowed the request" from "server accepted but never answered"
        phase = "connect" if len(tracker.socks) == n_socks else "read/stream"
        secs = min(CONNECT_TIMEOUT, timeout) if phase == "connect" and timeout else timeout
        where = f"{phase} after {secs}s" if isinstance(reason, (socket.timeout, TimeoutError)) else phase
        # only torn connections: refused (backend down) and timeouts would just make us wait
        raise LLMError(f"backend unreachable ({url}): {where}: {reason}",
                       retryable=isinstance(reason, _RETRY_NET)) from e


def _sse_lines(resp, cancel):
    """yields the json payloads of the `data:` lines."""
    for raw in resp:
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield data


def _split_content(c):
    """delta content → (text, reasoning). usually a string. mistral's reasoning models
    (e.g. mistral-large-4) instead send a list of {"type": "text", "text": …}
    and {"type": "thinking", "thinking": [{"type": "text", "text": …}]}."""
    if isinstance(c, str):
        return c, ""
    text, think = [], []
    for part in c if isinstance(c, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "thinking":
            inner = part.get("thinking")
            think.append(inner if isinstance(inner, str) else _split_content(inner)[0])
        elif isinstance(part.get("text"), str):
            text.append(part["text"])
    return "".join(text), "".join(think)


def stream_chat(cfg, model, messages, tools=None, effort="default", cancel=None, max_tokens=None, session=None,
                timeout=None, temperature=None):
    """generator over events:

    {"type": "text", "text": str}
    {"type": "reasoning", "text": str}
    {"type": "warning", "warning": str}   (e.g. retry after 429/5xx)
    {"type": "done", "message": {...}, "usage": {...} | None, "finish_reason": str}
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    if effort and effort != "default":
        payload["reasoning_effort"] = effort
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    tracker = _Tracker()
    done = threading.Event()
    if cancel is not None:
        threading.Thread(target=_watch, args=(cancel, tracker, done), daemon=True, name="llm-cancel").start()
    try:
        yield from _stream(cfg, payload, tracker, cancel, session, timeout=timeout or cfg.request_timeout)
    finally:
        done.set()


def _stream(cfg, payload, tracker, cancel, session=None, timeout=None):
    text = []
    reasoning = []
    signature = None
    calls = {}
    usage = None
    finish = None

    # "model@backend" → backend + bare model name for the payload
    try:
        backend, name = cfg.resolve(payload["model"])
    except ConfigError as e:
        raise LLMError(str(e)) from e
    payload = {**payload, "model": name}  # copy: plugins may change keys without it leaking to the caller
    headers, url = _prepare(backend, payload, session)
    # optional plugin response translation: raw sse chunk → openai chunk or None
    response_hook = getattr(backend.plugin, "response", None) if backend.plugin else None
    if not callable(response_hook):
        response_hook = None
    attempt = 0
    while True:
        try:
            resp = _request(url, timeout or cfg.request_timeout, payload, headers, tracker, cancel)
            break
        except LLMError as e:
            delay = retry_delay(e, attempt)
            if delay is None:
                raise
            attempt += 1
            what = f"http {e.status}" if e.status else "connection lost"
            yield {"type": "warning", "warning": f"{what}, retry in {delay:.0f}s ({attempt}/{RETRY_MAX})"}
            if cancel is not None and cancel.wait(delay):
                raise Cancelled() from e
            if cancel is None:
                time.sleep(delay)
    try:
        for data in _sse_lines(resp, cancel):
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if response_hook is not None:
                try:
                    chunk = response_hook(chunk)
                except Exception as e:
                    raise LLMError(f"plugin {backend.plugin.__name__} response: {e}") from e
                if chunk is None:
                    continue
            if chunk.get("error"):
                raise LLMError(f"backend error: {chunk['error']}", body=json.dumps(chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                c, think = _split_content(delta.get("content"))
                r = delta.get("reasoning_content") or delta.get("reasoning") or think
                if isinstance(r, str) and r:
                    reasoning.append(r)
                    yield {"type": "reasoning", "text": r}
                for tb in delta.get("thinking_blocks") or []:
                    if isinstance(tb, dict) and tb.get("signature"):
                        signature = tb["signature"]
                if c:
                    text.append(c)
                    yield {"type": "text", "text": c}
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", len(calls))
                    # litellm ollama_chat/: every call arrives complete in its own chunk, all with
                    # index 0 but its own id. new id on an occupied slot = new call, otherwise
                    # "{...}{...}" ends up glued together in one arguments string.
                    if tc.get("id") and calls.get(idx, {}).get("id") not in ("", None, tc["id"]):
                        idx = max(calls) + 1
                    slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    except (OSError, http.client.HTTPException, ValueError) as e:
        if _cancelled(cancel):
            raise Cancelled() from e
        if isinstance(e, ValueError):
            raise
        raise LLMError(f"connection to backend lost: {e}") from e
    finally:
        resp.close()
    if _cancelled(cancel):  # shutdown() can also arrive as a clean EOF
        raise Cancelled()

    message = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = [
            {
                "id": c["id"] or f"call_{i}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
            }
            for i, c in sorted(calls.items())
        ]
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
        if signature:
            message["thinking_blocks"] = [
                {"type": "thinking", "thinking": message["reasoning_content"], "signature": signature}
            ]
    yield {"type": "done", "message": message, "usage": usage, "finish_reason": finish}


def complete(cfg, model, messages, max_tokens=None, cancel=None, session=None, effort="default", timeout=None,
             temperature=None):
    """non-interactive call (e.g. title, mcp_magic router). returns the text."""
    out = []
    for ev in stream_chat(cfg, model, messages, max_tokens=max_tokens, cancel=cancel, session=session,
                          effort=effort, timeout=timeout, temperature=temperature):
        if ev["type"] == "text":
            out.append(ev["text"])
    return "".join(out)
