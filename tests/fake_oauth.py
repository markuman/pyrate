"""fake for mcp-oauth: ein http-server, der gleichzeitig mcp-server (POST /mcp) und
authorization-server ist (metadata, /register, /authorize, /token). /authorize leitet sofort
per 302 auf die redirect-uri – ein "browser" ist also nur ein urlopen auf die login-url."""

import base64
import hashlib
import itertools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse


class FakeOAuth:
    def __init__(self, ttl=3600, auth_methods=("none",), secret=None):
        self.ttl = ttl                    # expires_in for access-tokens
        self.auth_methods = list(auth_methods)
        self.secret = secret              # client_secret bei der registrierung
        self.ids = itertools.count(1)
        self.clients = {}                 # client_id → registrierung
        self.codes = {}                   # code → authorize-parameter
        self.access = {}                  # token → ablauf
        self.refresh = set()
        self.registrations = 0
        self.refreshes = 0
        self.token_forms = []
        self.mcp_auth = []                # Authorization-header der mcp-requests
        self.deny = None                  # error for /authorize
        self.wrong_state = False
        self.redirect = set()             # pfade, die per 302 nach /leak umleiten
        self.leaks = []                   # Authorization-header, die bei /leak ankamen

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def revoke_all(self):
        self.access.clear()

    def _issue(self):
        n = next(self.ids)
        at, rt = f"at-{n}", f"rt-{n}"
        self.access[at] = time.time() + self.ttl
        self.refresh.add(rt)
        return {"access_token": at, "refresh_token": rt, "expires_in": self.ttl, "token_type": "Bearer"}

    def start(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def _maybe_redirect(self, path):
                if path not in fake.redirect:
                    return False
                self.send_response(302)
                self.send_header("Location", f"{fake.base}/leak")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return True

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/leak":
                    fake.leaks.append(self.headers.get("Authorization"))
                    return self._json(200, {})
                if u.path == "/.well-known/oauth-protected-resource/mcp":
                    return self._json(200, {"resource": fake.url, "authorization_servers": [fake.base],
                                            "scopes_supported": ["mcp"]})
                if u.path == "/.well-known/oauth-authorization-server":
                    return self._json(200, {
                        "issuer": fake.base, "authorization_endpoint": f"{fake.base}/authorize",
                        "token_endpoint": f"{fake.base}/token", "registration_endpoint": f"{fake.base}/register",
                        "code_challenge_methods_supported": ["S256"],
                        "token_endpoint_auth_methods_supported": fake.auth_methods})
                if u.path == "/authorize":
                    q = {k: v[0] for k, v in parse_qs(u.query).items()}
                    assert q["client_id"] in fake.clients and q["code_challenge_method"] == "S256"
                    if fake.deny:
                        back = {"error": fake.deny, "state": q["state"]}
                    else:
                        code = f"code-{next(fake.ids)}"
                        fake.codes[code] = q
                        back = {"code": code, "state": "falsch" if fake.wrong_state else q["state"]}
                    self.send_response(302)
                    self.send_header("Location", q["redirect_uri"] + "?" + urlencode(back))
                    self.end_headers()
                    return None
                return self._json(404, {})

            def do_POST(self):
                u = urlparse(self.path)
                if self._maybe_redirect(u.path):
                    self._body()
                    return None
                if u.path == "/register":
                    reg = json.loads(self._body())
                    cid = f"client-{next(fake.ids)}"
                    fake.clients[cid] = reg
                    fake.registrations += 1
                    out = {"client_id": cid, **reg}
                    if fake.secret:
                        out["client_secret"] = fake.secret
                    return self._json(201, out)
                if u.path == "/token":
                    form = {k: v[0] for k, v in parse_qs(self._body().decode()).items()}
                    fake.token_forms.append({"form": form, "auth": self.headers.get("Authorization")})
                    if fake.secret:
                        basic = self.headers.get("Authorization", "")
                        ok = form.get("client_secret") == fake.secret or \
                            fake.secret in base64.b64decode(basic[6:] or b"").decode()
                        if not ok:
                            return self._json(401, {"error": "invalid_client"})
                    if form["grant_type"] == "authorization_code":
                        q = fake.codes.pop(form.get("code"), None)
                        challenge = base64.urlsafe_b64encode(
                            hashlib.sha256(form.get("code_verifier", "").encode()).digest()).rstrip(b"=").decode()
                        if not q or q["code_challenge"] != challenge or q["redirect_uri"] != form["redirect_uri"]:
                            return self._json(400, {"error": "invalid_grant"})
                        return self._json(200, fake._issue())
                    if form["grant_type"] == "refresh_token":
                        if form.get("refresh_token") not in fake.refresh:
                            return self._json(400, {"error": "invalid_grant", "error_description": "weg"})
                        fake.refresh.discard(form["refresh_token"])  # rotieren wie gitlab
                        fake.refreshes += 1
                        return self._json(200, fake._issue())
                    return self._json(400, {"error": "unsupported_grant_type"})
                if u.path == "/mcp":
                    auth = self.headers.get("Authorization", "")
                    fake.mcp_auth.append(auth)
                    msg = json.loads(self._body())
                    tok = auth[len("Bearer "):] if auth.startswith("Bearer ") else None
                    if tok not in fake.access or fake.access[tok] < time.time():
                        meta = f'Bearer resource_metadata="{fake.base}/.well-known/oauth-protected-resource/mcp"'
                        return self._json(401, {"error": "invalid_token"}, {"WWW-Authenticate": meta})
                    if "id" not in msg:
                        self.send_response(202)
                        self.end_headers()
                        return None
                    res = {"initialize": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                          "serverInfo": {"name": "fake-oauth"}},
                           "tools/list": {"tools": [{"name": "whoami", "inputSchema": {"type": "object"}}]},
                           "tools/call": {"content": [{"type": "text", "text": tok}]}}.get(msg["method"], {})
                    return self._json(200, {"jsonrpc": "2.0", "id": msg["id"], "result": res})
                return self._json(404, {})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
