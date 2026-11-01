"""mcp client: json-rpc 2.0 over stdio or streamable http. stdlib + pyyaml.

  Manager   – alle konfigurierten server, verbindet im hintergrund, liefert die tools
  Client    – ein server: initialize, tools/list (mit pagination), tools/call
  Stdio…    – subprocess, newline-delimited json auf stdin/stdout
  HTTP…     – POST pro nachricht, antwort als json oder sse, Mcp-Session-Id,
              optional bearer-token aus oauth.py (401 → einmal refresh → retry)

die tools tauchen als mcp__<server>__<tool> neben den eingebauten auf. im plan-modus nur die,
die der server mit annotations.readOnlyHint als lesend markiert.
"""

import collections
import hashlib
import itertools
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from . import __version__, oauth, tools

PROTOCOL_VERSION = "2025-06-18"
CONNECT_TIMEOUT = 60  # s for start + initialize + tools/list (uvx/npx download on first use)
MAX_DESCRIPTION = 2000


class MCPError(Exception):
    pass


# ---------------------------------------------------------------- descriptions store
# `pyrate --mcp-setup` schreibt hier name → {description, tools, updated}. der mcp_magic-router
# braucht sie für server, die gerade nicht verbunden sind (enabled: false wird nie gestartet).

def descriptions_path():
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "pyrate" / "mcp-descriptions.yaml"


def load_descriptions():
    try:
        data = yaml.safe_load(descriptions_path().read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _tool_line(t, max_desc=200):
    """"name: description" (description gekürzt) – eine zeile pro tool für den router-prompt
    und den descriptions-store."""
    desc = (t.get("description") or t.get("title") or "").strip()
    desc = " ".join(desc.split())[:max_desc]
    return f"{t['name']}: {desc}" if desc else str(t["name"])


def save_descriptions(data):
    p = descriptions_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- client

class Client:
    def __init__(self, spec, on_change=None):
        self.spec = spec
        self.name = spec.name
        self.on_change = on_change
        self._ids = itertools.count(1)
        self._pending = {}
        self._lock = threading.Lock()
        self.dead = None          # grund, falls der transport weg ist
        self.server_info = {}
        self.capabilities = {}
        self.protocol = None
        self.tools = []
        self.transport = StdioTransport(self) if spec.command else HTTPTransport(self)

    # ------------------------------------------------------------ lifecycle

    def connect(self, timeout=CONNECT_TIMEOUT):
        deadline = time.monotonic() + timeout
        self.transport.start()
        res = self.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "pyrate", "version": __version__},
        }, timeout=timeout)
        self.protocol = res.get("protocolVersion") or PROTOCOL_VERSION
        self.server_info = res.get("serverInfo") or {}
        self.capabilities = res.get("capabilities") or {}
        self.notify("notifications/initialized")
        if "tools" in self.capabilities:
            self.tools = self.list_tools(max(1, deadline - time.monotonic()))

    def list_tools(self, timeout):
        out, cursor = [], None
        for _ in range(100):  # pagination, mit notbremse
            res = self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout=timeout)
            out += [t for t in res.get("tools") or [] if isinstance(t, dict) and t.get("name")]
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return out

    def call_tool(self, name, args, cancel=None):
        return self.request("tools/call", {"name": name, "arguments": args},
                            timeout=self.spec.timeout, cancel=cancel)

    def close(self):
        self.dead = self.dead or "geschlossen"
        self._fail_pending()
        self.transport.close()

    # ------------------------------------------------------------ json-rpc

    def request(self, method, params, timeout, cancel=None):
        if self.dead:
            raise MCPError(f"mcp server '{self.name}' is not connected: {self.dead}")
        rid = next(self._ids)
        q = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[rid] = q
        try:
            self.transport.send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            deadline = time.monotonic() + timeout
            while True:
                try:
                    msg = q.get(timeout=0.05)
                    break
                except queue.Empty:
                    if cancel is not None and cancel.is_set():
                        self._cancel_remote(rid, "cancelled")
                        raise Cancelled() from None
                    if time.monotonic() > deadline:
                        self._cancel_remote(rid, "timeout")
                        raise MCPError(f"mcp-server '{self.name}': timeout nach {timeout:.0f}s ({method})") from None
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        if "error" in msg:
            err = msg["error"] if isinstance(msg["error"], dict) else {"message": str(msg["error"])}
            raise MCPError(f"mcp-server '{self.name}': {err.get('message', '?')} (code {err.get('code', '?')})")
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self.transport.send(msg)

    def _cancel_remote(self, rid, reason):
        try:
            self.notify("notifications/cancelled", {"requestId": rid, "reason": reason})
        except MCPError:
            pass

    def _deliver(self, rid, msg):
        with self._lock:
            q = self._pending.get(rid)
        if q is not None:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    def incoming(self, msg):
        """from the transport: answers, server requests, notifications (also as batch)."""
        if isinstance(msg, list):
            for m in msg:
                self.incoming(m)
            return
        if not isinstance(msg, dict):
            return
        if "id" in msg and ("result" in msg or "error" in msg):
            self._deliver(msg["id"], msg)
        elif "method" in msg and "id" in msg:
            self._answer_server_request(msg)
        elif msg.get("method") == "notifications/tools/list_changed":
            threading.Thread(target=self._refresh_tools, daemon=True).start()
        # we ignore logging & co.

    def _answer_server_request(self, msg):
        # we offer no client capabilities (sampling, roots, elicitation) → ping only
        if msg["method"] == "ping":
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        else:
            reply = {"jsonrpc": "2.0", "id": msg["id"],
                     "error": {"code": -32601, "message": f"method not supported by pyrate: {msg['method']}"}}
        try:
            self.transport.send(reply)
        except MCPError:
            pass

    def _refresh_tools(self):
        try:
            self.tools = self.list_tools(CONNECT_TIMEOUT)
        except (MCPError, Cancelled):
            return
        if self.on_change:
            self.on_change()

    def missing(self, rid, reason):
        """transport delivered nothing for rid (http response without a matching message)."""
        self._deliver(rid, {"id": rid, "error": {"code": -32603, "message": reason}})

    def died(self, reason):
        if self.dead:
            return
        self.dead = reason
        self._fail_pending()
        if self.on_change:
            self.on_change()

    def _fail_pending(self):
        with self._lock:
            rids = list(self._pending)
        for rid in rids:
            self._deliver(rid, {"id": rid, "error": {"code": -32000, "message": self.dead or "verbindung weg"}})


# ---------------------------------------------------------------- stdio

class StdioTransport:
    def __init__(self, client):
        self.client = client
        self.spec = client.spec
        self.proc = None
        self.wlock = threading.Lock()
        self.stderr_tail = collections.deque(maxlen=20)

    def start(self):
        env = {**os.environ, **self.spec.env}
        try:
            self.proc = subprocess.Popen(
                [self.spec.command, *self.spec.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, cwd=self.spec.cwd, start_new_session=True,
            )
        except OSError as e:
            raise MCPError(f"mcp-server '{self.spec.name}': start fehlgeschlagen: {e}") from e
        threading.Thread(target=self._read, daemon=True, name=f"mcp-{self.spec.name}-out").start()
        threading.Thread(target=self._drain_stderr, daemon=True, name=f"mcp-{self.spec.name}-err").start()

    def send(self, msg):
        if self.proc is None or self.proc.poll() is not None:
            raise MCPError(f"mcp server '{self.spec.name}' is not running")
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode()
        try:
            with self.wlock:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
        except (OSError, ValueError) as e:
            raise MCPError(f"mcp-server '{self.spec.name}': schreiben fehlgeschlagen: {e}") from e

    def _read(self):
        for raw in self.proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # garbage on stdout (shouldn't happen, but does)
            self.client.incoming(msg)
        code = self.proc.wait()
        tail = " | ".join(x for x in self.stderr_tail if x)
        self.client.died(f"prozess beendet (exit {code})" + (f": {tail[-500:]}" if tail else ""))

    def _drain_stderr(self):
        for raw in self.proc.stderr:
            self.stderr_tail.append(raw.decode("utf-8", "replace").rstrip())

    def close(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.close()  # polite variant: server sees eof
        except OSError:
            pass
        try:
            self.proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                return
            try:
                self.proc.wait(timeout=2)
                return
            except subprocess.TimeoutExpired:
                continue


# ---------------------------------------------------------------- streamable http

def _sse_messages(resp):
    """data fields of an sse stream, event by event."""
    data = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
    if data:
        yield "\n".join(data)


class HTTPTransport:
    def __init__(self, client):
        self.client = client
        self.spec = client.spec
        self.session = None
        self.closed = False

    def start(self):
        pass  # verbindung entsteht mit dem ersten POST

    def _headers(self, stale=None):
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": f"pyrate/{__version__}",
            **self.spec.headers,
        }
        if self.session:
            h["Mcp-Session-Id"] = self.session
        if self.client.protocol:
            h["MCP-Protocol-Version"] = self.client.protocol
        if self.spec.oauth is not None:
            h["Authorization"] = f"Bearer {self._token(stale)}"
        return h

    def _token(self, stale=None):
        hint = f"pyrate --mcp-login {self.spec.name}"
        try:
            tok = oauth.token(self.spec, stale)
        except oauth.OAuthError as e:
            raise MCPError(f"mcp-server '{self.spec.name}': oauth: {e} – neu einloggen: {hint}") from e
        if not tok:
            raise MCPError(f"mcp server '{self.spec.name}': not logged in – {hint}")
        return tok

    def send(self, msg):
        if self.closed:
            raise MCPError(f"mcp-server '{self.spec.name}' ist geschlossen")
        if "method" in msg and "id" in msg:
            # requests async: the answer arrives via client.incoming, request() waits for it
            threading.Thread(target=self._post, args=(msg,), daemon=True,
                             name=f"mcp-{self.spec.name}-req").start()
        else:
            self._post(msg)

    def _post(self, msg, stale=None):
        rid = msg.get("id") if "method" in msg else None
        try:
            headers = self._headers(stale)
        except MCPError as e:  # oauth: nicht eingeloggt / refresh kaputt
            if rid is not None:
                self.client.missing(rid, str(e))
                return
            raise
        req = urllib.request.Request(self.spec.url, data=json.dumps(msg).encode(),
                                     headers=headers, method="POST")
        try:
            resp = oauth.urlopen(req, timeout=self.spec.timeout + 30)  # ohne redirects: header-leak
        except oauth.RedirectRefused as e:
            reason = str(e.reason)
            if rid is not None:
                self.client.missing(rid, reason)
                return
            raise MCPError(f"mcp-server '{self.spec.name}': {reason}") from e
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 401 and self.spec.oauth is not None and stale is None:
                # token rejected by the server (revoked, expired early): refresh once
                return self._post(msg, stale=headers["Authorization"][len("Bearer "):])
            if e.code == 401 and self.spec.oauth is not None:
                body = f"token abgelehnt – neu einloggen: pyrate --mcp-login {self.spec.name}"
            if e.code == 404 and self.session:
                self.session = None
                self.client.died("session abgelaufen (http 404), neu verbinden mit neustart")
            reason = f"http {e.code}: {body}"
            if rid is not None:
                self.client.missing(rid, reason)
                return
            raise MCPError(f"mcp-server '{self.spec.name}': {reason}") from e
        except (OSError, urllib.error.URLError) as e:
            reason = f"unreachable: {getattr(e, 'reason', e)}"
            if rid is not None:
                self.client.missing(rid, reason)
                return
            raise MCPError(f"mcp-server '{self.spec.name}': {reason}") from e
        with resp:
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self.session = sid
            ctype = (resp.headers.get("Content-Type") or "").lower()
            try:
                if "text/event-stream" in ctype:
                    for data in _sse_messages(resp):
                        try:
                            self.client.incoming(json.loads(data))
                        except ValueError:
                            continue
                elif "json" in ctype:
                    body = resp.read()
                    if body.strip():
                        self.client.incoming(json.loads(body))
            except (OSError, ValueError) as e:
                if rid is not None:
                    self.client.missing(rid, f"broken answer: {e}")
                return
        if rid is not None:
            self.client.missing(rid, "server sent no answer")  # no-op, falls schon da

    def close(self):
        if self.closed:
            return
        self.closed = True
        if not self.session:
            return
        try:
            req = urllib.request.Request(self.spec.url, headers=self._headers(), method="DELETE")
            oauth.urlopen(req, timeout=5).close()
        except (OSError, urllib.error.URLError, MCPError):
            pass


# ---------------------------------------------------------------- tools as pyrate tools

def tool_name(server, tool, taken=()):
    """mcp__<server>__<tool>, nur [A-Za-z0-9_-], max. 64 zeichen (openai-limit).
    ist der name schon vergeben (a.b vs a_b, oder server/tool-grenze verschoben),
    kommt ein hash des originalnamens dran."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", tool)
    name = f"mcp__{server}__{clean}"
    if len(name) > 64 or name in taken:
        h = hashlib.sha1(f"{server}\0{tool}".encode()).hexdigest()[:8]
        name = f"{name[:55]}_{h}"
    return name


def _is_verbatim(server, tool):
    """name passes through unchanged → wins on collisions."""
    return re.fullmatch(r"[A-Za-z0-9_-]+", tool) is not None and len(f"mcp__{server}__{tool}") <= 64


def _schema(raw):
    s = dict(raw) if isinstance(raw, dict) else {}
    s.pop("$schema", None)  # some providers trip over it
    s["type"] = "object"
    s.setdefault("properties", {})
    return s


def format_result(res):
    parts = []
    for c in res.get("content") or []:
        if not isinstance(c, dict):
            continue
        t = c.get("type")
        if t == "text":
            parts.append(str(c.get("text", "")))
        elif t in ("image", "audio"):
            size = len(c.get("data") or "") * 3 // 4
            parts.append(f"[{t}: {c.get('mimeType', '?')}, {size} bytes – not shown]")
        elif t == "resource":
            r = c.get("resource") or {}
            if "text" in r:
                parts.append(f"[resource {r.get('uri', '')}]\n{r['text']}")
            else:
                parts.append(f"[resource {r.get('uri', '')}: binary, {r.get('mimeType', '?')}]")
        elif t == "resource_link":
            parts.append(f"[link: {c.get('name', '')} {c.get('uri', '')}]".replace("  ", " "))
    if not parts and res.get("structuredContent") is not None:
        parts.append(json.dumps(res["structuredContent"], ensure_ascii=False, indent=1))
    return "\n".join(parts)


def _make_func(client, original):
    def func(ctx, args):
        try:
            res = client.call_tool(original, args, cancel=ctx.cancel)
        except Cancelled:
            raise tools.ToolError("cancelled") from None
        except MCPError as e:
            raise tools.ToolError(str(e)) from None
        text = format_result(res)
        if res.get("isError"):
            raise tools.ToolError(tools.limit_text(ctx, text or "(server meldet fehler ohne text)", "mcp"))
        return tools.limit_text(ctx, text or "(leeres ergebnis)", "mcp")
    return func


# ---------------------------------------------------------------- manager

class Manager:
    def __init__(self, servers, on_change=None):
        self.on_change = on_change
        self.clients = {s.name: Client(s, on_change=self._changed) for s in servers}
        # connection state: idle (never connected) | connecting | ready | error
        self.state = {s.name: {"state": "idle", "error": None} for s in servers}
        # on/off for this runtime. off = tools don't go into the request (context!), a
        # an existing connection stays open though → re-enabling is instant
        self.enabled = {s.name: s.enabled for s in servers}
        self._tools = {}
        self._lock = threading.Lock()
        self._threads = []
        self._timeout = CONNECT_TIMEOUT

    def start(self, wait=False, timeout=CONNECT_TIMEOUT):
        """verbindet alle eingeschalteten server im hintergrund."""
        self._timeout = timeout
        for name, on in self.enabled.items():
            if on:
                self._spawn(name)
        if wait:
            self.wait(timeout + 5)

    def _spawn(self, name):
        spec = self.clients[name].spec
        if spec.error:  # config kaputt (z.b. env-variable fehlt): gar nicht erst versuchen
            self.state[name] = {"state": "error", "error": spec.error}
            self._changed()
            return
        if self.state[name]["state"] == "error":  # alter client ist tot, frisch anfangen
            self.clients[name] = Client(spec, on_change=self._changed)
        self.state[name] = {"state": "connecting", "error": None}
        t = threading.Thread(target=self._connect, args=(name, self._timeout), daemon=True,
                             name=f"mcp-{name}-connect")
        self._threads.append(t)
        t.start()

    def set_enabled(self, name, on):
        """enable/disable a server for this runtime. enabling connects on demand
        (auch ein neuer versuch nach einem fehler)."""
        self.enabled[name] = on
        if on and self.state[name]["state"] in ("idle", "error"):
            self._spawn(name)
        self._changed()

    def wait_ready(self, names, timeout, cancel=None):
        """wartet, bis die genannten server nicht mehr 'idle'/'connecting' sind (ready oder
        error). für mcp_magic: frisch eingeschaltete server sollen schon im ersten request
        ihre tools liefern."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                return
            if all(self.state[n]["state"] not in ("idle", "connecting") for n in names):
                return
            time.sleep(0.05)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        for t in list(self._threads):
            t.join(max(0, deadline - time.monotonic()))

    def _connect(self, name, timeout):
        client = self.clients[name]
        self._notify()
        try:
            client.connect(timeout)
            self.state[name] = {"state": "ready", "error": None}
        except (MCPError, Cancelled, OSError) as e:
            self.state[name] = {"state": "error", "error": str(e) or type(e).__name__}
            client.close()
        self._changed()

    def _changed(self):
        with self._lock:
            entries = []
            for idx, (name, client) in enumerate(self.clients.items()):
                if client.dead and self.state[name]["state"] == "ready":
                    self.state[name] = {"state": "error", "error": client.dead}
                if self.state[name]["state"] != "ready" or not self.enabled[name]:
                    continue
                seen = set()
                for t in client.tools:
                    if t["name"] in client.spec.disabled_tools or t["name"] in seen:
                        continue  # doppelte namen vom server: erster gewinnt
                    seen.add(t["name"])
                    entries.append((not _is_verbatim(name, t["name"]), idx, t["name"], name, client, t))
            # stable assignment, independent of the order in which servers deliver tools:
            # unchanged names first, then config order, then alphabetical
            entries.sort(key=lambda e: e[:3])
            new = {}
            for _, _, _, name, client, t in entries:
                exposed = tool_name(name, t["name"], new)
                desc = (t.get("description") or t.get("title") or "").strip()
                if len(desc) > MAX_DESCRIPTION:
                    desc = desc[: MAX_DESCRIPTION - 1] + "…"
                if exposed != f"mcp__{name}__{t['name']}":
                    desc = f"(original: {t['name']}) {desc}".strip()
                readonly = bool((t.get("annotations") or {}).get("readOnlyHint"))
                new[exposed] = tools.Tool(
                    exposed, f"[mcp:{name}] {desc}".rstrip(), {}, _make_func(client, t["name"]),
                    readonly=readonly, self_truncating=True, raw_schema=_schema(t.get("inputSchema")),
                )
            self._tools = new
        self._notify()

    def _notify(self):
        if self.on_change:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001 - ui-callback darf den connect-thread nicht killen
                pass

    def tools(self):
        with self._lock:
            return dict(self._tools)

    def candidates(self, max_tools=12, max_desc=200):
        """[(name, description, [tool lines])] für den mcp_magic-router. tool lines sind
        "name: description". echte tools, solange der server verbunden ist, sonst die
        gespeicherten aus --mcp-setup. description: config gewinnt vor dem store.
        max_tools=None → alle (für den lokalen keyword-fallback, der nichts kostet)."""
        stored = load_descriptions()
        out = []
        for name, client in self.clients.items():
            st = stored.get(name) if isinstance(stored.get(name), dict) else {}
            desc = client.spec.description or st.get("description") or ""
            if self.state[name]["state"] == "ready":
                tl = [_tool_line(t, max_desc) for t in client.tools
                      if t["name"] not in client.spec.disabled_tools]
            else:
                tl = st.get("tools") or []
                tl = [str(t) for t in tl if isinstance(t, str)]
            if max_tools is not None:
                tl = tl[:max_tools]
            out.append((name, str(desc), tl))
        return out

    def summary(self):
        """[(name, state, n_tools, error)] for sidebar/cli. state "off" = disabled
        (n_tools ist dann die bekannte anzahl, falls schon mal verbunden)."""
        out = []
        for name, client in self.clients.items():
            st = self.state[name]
            n = sum(1 for t in client.tools if t["name"] not in client.spec.disabled_tools)
            state = st["state"] if self.enabled[name] else "off"
            out.append((name, state, n if st["state"] == "ready" else 0, st["error"]))
        return out

    def tool_tokens(self, name):
        """grobe token-kosten (chars/4) der tool-schemas eines servers pro request.
        None, solange die tools unbekannt sind."""
        client = self.clients[name]
        if self.state[name]["state"] != "ready":
            return None
        schemas = [{"name": t["name"], "description": t.get("description") or "",
                    "parameters": _schema(t.get("inputSchema"))}
                   for t in client.tools if t["name"] not in client.spec.disabled_tools]
        return len(json.dumps(schemas, ensure_ascii=False)) // 4

    def close(self):
        threads = [threading.Thread(target=c.close, daemon=True) for c in self.clients.values()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(6)
