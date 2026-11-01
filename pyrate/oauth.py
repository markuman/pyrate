"""oauth for http mcp servers per the MCP authorization spec. stdlib only.

  discover  – 401 → protected-resource-metadata → authorization-server-metadata
  login     – dynamic client registration (einmal, gecacht) + authorization code mit PKCE
              via the browser, callback on 127.0.0.1. explicit only: `pyrate --mcp-login NAME`
  token     – valid access token for a server, refreshed when needed. None = never logged in

tokens liegen in ~/.local/share/pyrate/mcp-auth.json (0600), key = mcp-url. zugriffe laufen
under a flock: gitlab rotates refresh tokens, two pyrates must not refresh at the same time
refreshen, sonst entwertet einer den token des anderen.
"""

import base64
import fcntl
import hashlib
import http.server
import json
import os
import re
import secrets
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

from . import __version__

REDIRECT_PORT = 19877   # fest: gitlab vergleicht die redirect-uri exakt mit der registrierten
LOGIN_TIMEOUT = 300     # s for the browser part
HTTP_TIMEOUT = 30
EXPIRY_SKEW = 60        # s: so kurz vor ablauf wird schon refresht
_RESOURCE_METADATA_RE = re.compile(r'resource_metadata="([^"]+)"')
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")
_thread_lock = threading.Lock()


class OAuthError(Exception):
    pass


def redirect_uri():
    return f"http://127.0.0.1:{REDIRECT_PORT}/callback"


# ---------------------------------------------------------------- storage

def store_path():
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "pyrate" / "mcp-auth.json"


@contextmanager
def _locked():
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock, open(path.with_suffix(".lock"), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def _load():
    try:
        data = json.loads(store_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise OAuthError(f"{store_path()}: not readable ({e})") from e
    return data if isinstance(data, dict) else {}


def _save(data):
    path = store_path()
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.chmod(tmp, 0o600)  # falls tmp schon mit anderen rechten herumlag
    os.replace(tmp, path)


# ---------------------------------------------------------------- http

class RedirectRefused(urllib.error.URLError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib keeps all headers on redirect except Content-*, so also Authorization –
    auch bei https → http oder auf einen fremden host. spec-konforme oauth- und mcp-endpoints
    redirecten nicht, also gar nicht folgen."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise RedirectRefused(f"redirect ({code}) von {req.full_url} nach {newurl} verweigert")


_opener = urllib.request.build_opener(NoRedirect)


def urlopen(req, timeout):
    """urlopen without redirects. for anything that sends credentials (also mcp.py, llm.py)."""
    return _opener.open(req, timeout=timeout)


def _check_url(url):
    p = urlparse(url)
    if p.scheme == "https" or (p.scheme == "http" and p.hostname in _LOCAL_HOSTS):
        return
    raise OAuthError(f"{url}: oauth only over https (http only for localhost)")


def _call(url, *, form=None, json_body=None, headers=None):
    """→ (status, json dict). http errors are not an error, only net and broken urls."""
    _check_url(url)
    h = {"Accept": "application/json", "User-Agent": f"pyrate/{__version__}", **(headers or {})}
    data = None
    if form is not None:
        data = urlencode({k: v for k, v in form.items() if v is not None}).encode()
        h["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        data = json.dumps(json_body).encode()
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urlopen(req, timeout=HTTP_TIMEOUT) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except RedirectRefused as e:
        raise OAuthError(e.reason) from e
    except (OSError, urllib.error.URLError) as e:
        raise OAuthError(f"{url}: unreachable ({getattr(e, 'reason', e)})") from e
    try:
        body = json.loads(raw) if raw.strip() else {}
    except ValueError:
        body = {}
    return status, body if isinstance(body, dict) else {}


def _probe(url):
    """unauthenticated mcp request → resource_metadata url from WWW-Authenticate, or None."""
    _check_url(url)
    msg = {"jsonrpc": "2.0", "id": 0, "method": "initialize",
           "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                      "clientInfo": {"name": "pyrate", "version": __version__}}}
    req = urllib.request.Request(url, data=json.dumps(msg).encode(), headers={
        "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
        "User-Agent": f"pyrate/{__version__}"})
    try:
        urlopen(req, timeout=HTTP_TIMEOUT).close()  # kein 401: body egal
        return None
    except urllib.error.HTTPError as e:
        m = _RESOURCE_METADATA_RE.search(e.headers.get("WWW-Authenticate") or "")
        return m.group(1) if e.code == 401 and m else None
    except RedirectRefused as e:
        raise OAuthError(e.reason) from e
    except (OSError, urllib.error.URLError) as e:
        raise OAuthError(f"{url}: unreachable ({getattr(e, 'reason', e)})") from e


def _origin(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}", p.path.rstrip("/")


# ---------------------------------------------------------------- discovery

def discover(url):
    origin, path = _origin(url)
    prm = {}
    for cand in filter(None, (_probe(url), f"{origin}/.well-known/oauth-protected-resource{path}",
                              f"{origin}/.well-known/oauth-protected-resource")):
        status, body = _call(cand)
        if status == 200 and body.get("authorization_servers"):
            prm = body
            break
    resource = prm.get("resource") or url
    if _origin(resource)[0] != origin:
        raise OAuthError(f"protected-resource-metadata nennt fremde resource {resource!r}")
    issuer = (prm.get("authorization_servers") or [origin])[0]  # ohne metadata: alter spec-stand
    as_origin, as_path = _origin(issuer)
    cands = [f"{as_origin}/.well-known/oauth-authorization-server{as_path}",
             f"{as_origin}/.well-known/openid-configuration{as_path}"]
    if as_path:
        cands.append(f"{as_origin}{as_path}/.well-known/openid-configuration")
    for cand in cands:
        status, meta = _call(cand)
        if status == 200 and meta.get("authorization_endpoint") and meta.get("token_endpoint"):
            break
    else:
        raise OAuthError(f"no authorization server metadata for {issuer}")
    if "S256" not in (meta.get("code_challenge_methods_supported") or []):
        raise OAuthError(f"{issuer} does not support PKCE with S256")
    return {
        "issuer": meta.get("issuer") or issuer,
        "authorization_endpoint": meta["authorization_endpoint"],
        "token_endpoint": meta["token_endpoint"],
        "registration_endpoint": meta.get("registration_endpoint"),
        "auth_methods": meta.get("token_endpoint_auth_methods_supported") or [],
        "resource": resource,
        "scopes": prm.get("scopes_supported") or [],
    }


def register(meta, scope):
    """dynamic client registration (RFC 7591). public client, wenn der server das kann."""
    if not meta["registration_endpoint"]:
        raise OAuthError(f"{meta['issuer']} cannot do dynamic client registration")
    methods = meta["auth_methods"]
    method = "none" if not methods or "none" in methods else \
        "client_secret_post" if "client_secret_post" in methods else methods[0]
    status, body = _call(meta["registration_endpoint"], json_body={
        "client_name": "pyrate", "redirect_uris": [redirect_uri()],
        "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
        "token_endpoint_auth_method": method, **({"scope": scope} if scope else {}),
    })
    if status not in (200, 201) or not body.get("client_id"):
        raise OAuthError(f"client-registrierung: {_err(body, status)}")
    return {"client_id": body["client_id"], "client_secret": body.get("client_secret"),
            "auth_method": body.get("token_endpoint_auth_method") or method,
            "redirect_uri": redirect_uri(), "issuer": meta["issuer"]}


def _err(body, status):
    err = body.get("error") or f"http {status}"
    return f"{err}: {body['error_description']}" if body.get("error_description") else err


def _token_request(endpoint, client, form):
    form = {**form, "client_id": client["client_id"]}
    headers = {}
    secret = client.get("client_secret")
    if secret and client.get("auth_method") == "client_secret_basic":
        cred = f"{quote(client['client_id'], safe='')}:{quote(secret, safe='')}"
        headers["Authorization"] = "Basic " + base64.b64encode(cred.encode()).decode()
        form.pop("client_id")
    elif secret:
        form["client_secret"] = secret
    status, body = _call(endpoint, form=form, headers=headers)
    if status != 200 or not isinstance(body.get("access_token"), str) or not body["access_token"]:
        raise OAuthError(f"token-endpoint: {_err(body, status)}")
    return body


def _expires_at(expires_in):
    """expires_in → absoluter zeitpunkt. kaputt/fehlt/<= 0 → None: dann kein proaktiver
    refresh, den regelt der 401 des servers."""
    try:
        secs = int(float(expires_in))
    except (TypeError, ValueError, OverflowError):
        return None
    return int(time.time()) + secs if secs > 0 else None


def _apply(entry, body):
    entry["access_token"] = body["access_token"]
    refresh = body.get("refresh_token")
    entry["refresh_token"] = refresh if isinstance(refresh, str) and refresh else entry.get("refresh_token")
    entry["expires_at"] = _expires_at(body.get("expires_in"))


# ---------------------------------------------------------------- login (browser)

def _wait_for_callback(state, open_url, timeout):
    """einmaliger http-server auf 127.0.0.1 → query-parameter des callbacks."""
    got = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            if u.path != "/callback" or done.is_set():
                self.send_error(404)
                return
            got.update({k: v[0] for k, v in parse_qs(u.query).items()})
            ok = got.get("state") == state and "code" in got
            text = "login ok, back to the terminal." if ok else "login failed, see terminal."
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"pyrate: {text}\n".encode())
            done.set()

    try:
        server = http.server.HTTPServer(("127.0.0.1", REDIRECT_PORT), Handler)
    except OSError as e:
        raise OAuthError(f"port {REDIRECT_PORT} for the callback is taken ({e})") from e
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    t.start()
    try:
        open_url()
        deadline = time.monotonic() + timeout
        while not done.wait(0.5):  # in small steps so ctrl+c gets through
            if time.monotonic() > deadline:
                raise OAuthError(f"no callback after {timeout}s")
    finally:
        server.shutdown()
        server.server_close()
    if "error" in got:
        raise OAuthError(f"login abgelehnt: {_err(got, '?')}")
    if got.get("state") != state:
        raise OAuthError("callback with wrong state – cancelled")
    if not got.get("code"):
        raise OAuthError("callback ohne code")
    return got["code"]


def login(spec, open_browser=webbrowser.open, out=sys.stderr, timeout=LOGIN_TIMEOUT):
    """complete browser login for a server, stores the tokens."""
    meta = discover(spec.url)
    scope = (spec.oauth or {}).get("scope") or " ".join(meta["scopes"]) or None
    with _locked():
        old = _load().get(spec.url)
        client = old.get("client") if isinstance(old, dict) else None
    if not isinstance(client, dict) or not client.get("client_id") or client.get("issuer") != meta["issuer"] or client.get("redirect_uri") != redirect_uri():
        client = register(meta, scope)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)
    endpoint = meta["authorization_endpoint"]
    _check_url(endpoint)
    url = endpoint + ("&" if "?" in endpoint else "?") + urlencode({k: v for k, v in {
        "response_type": "code", "client_id": client["client_id"], "redirect_uri": redirect_uri(),
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
        "resource": meta["resource"], "scope": scope}.items() if v})

    def open_url():
        print(f"login for mcp server '{spec.name}' in the browser. if none opens:\n  {url}", file=out)
        try:
            open_browser(url)
        except Exception:  # noqa: BLE001 - die url steht ja schon da
            pass

    code = _wait_for_callback(state, open_url, timeout)
    body = _token_request(meta["token_endpoint"], client, {
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri(),
        "code_verifier": verifier, "resource": meta["resource"]})
    entry = {"issuer": meta["issuer"], "token_endpoint": meta["token_endpoint"],
             "resource": meta["resource"], "client": client}
    _apply(entry, body)
    with _locked():
        data = _load()
        data[spec.url] = entry
        _save(data)
    return entry


def logout(spec):
    with _locked():
        data = _load()
        if data.pop(spec.url, None) is None:
            return False
        _save(data)
    return True


# ---------------------------------------------------------------- in operation

def token(spec, stale=None):
    """valid access token or None (never logged in). stale = token the server just
    rejected with 401 → force refresh, unless another process was faster.
    refresh-fehler → OAuthError."""
    with _locked():
        data = _load()
        entry = data.get(spec.url)
        if entry is None:
            return None
        broken = f"{store_path()}: entry for {spec.url} broken"
        if not isinstance(entry, dict):
            raise OAuthError(broken)
        tok = entry.get("access_token")
        if not tok:
            return None
        if not isinstance(tok, str):
            raise OAuthError(broken)
        exp = entry.get("expires_at")
        expired = isinstance(exp, (int, float)) and exp - EXPIRY_SKEW < time.time()
        if tok != stale and not expired:
            return tok
        refresh, endpoint, client = entry.get("refresh_token"), entry.get("token_endpoint"), entry.get("client")
        if not refresh:
            raise OAuthError("token expired, no refresh token")
        if not (isinstance(refresh, str) and isinstance(endpoint, str) and endpoint
                and isinstance(client, dict) and client.get("client_id")):
            raise OAuthError(broken)
        body = _token_request(endpoint, client, {
            "grant_type": "refresh_token", "refresh_token": refresh,
            "resource": entry.get("resource")})
        _apply(entry, body)
        _save(data)
        return entry["access_token"]
