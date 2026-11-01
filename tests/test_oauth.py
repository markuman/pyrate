import json
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fake_oauth import FakeOAuth

from pyrate import config, mcp, oauth, tools
from pyrate.config import MCPServer


def browser(url):
    """"browser": login-url aufrufen, /authorize leitet auf den callback weiter."""
    try:
        urllib.request.urlopen(url, timeout=10).read()
    except OSError:
        pass  # callback antwortet bei fehlern mit 400, egal


@pytest.fixture
def fake(monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    monkeypatch.setattr(oauth, "REDIRECT_PORT", s.getsockname()[1])
    s.close()
    f = FakeOAuth().start()
    yield f
    f.stop()


def spec(fake, **kw):
    return MCPServer(name="gl", url=fake.url, oauth={"scope": None}, **kw)


def entry(fake):
    return json.loads(oauth.store_path().read_text())[fake.url]


def test_login_flow_and_storage(fake):
    s = spec(fake)
    assert oauth.token(s) is None  # nie eingeloggt
    e = oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    assert e["access_token"] == entry(fake)["access_token"] and fake.registrations == 1
    reg = next(iter(fake.clients.values()))
    assert reg["redirect_uris"] == [oauth.redirect_uri()] and reg["token_endpoint_auth_method"] == "none"
    assert reg["scope"] == "mcp"  # ohne config-scope: scopes_supported aus der resource-metadata
    form = fake.token_forms[0]["form"]
    assert form["resource"] == fake.url and form["code_verifier"]
    assert stat.S_IMODE(oauth.store_path().stat().st_mode) == 0o600
    assert oauth.token(s) == e["access_token"]
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    assert fake.registrations == 1  # client wird wiederverwendet
    assert oauth.logout(s) and not oauth.logout(s) and oauth.token(s) is None


def test_login_rejects_wrong_state_and_denial(fake):
    fake.wrong_state = True
    with pytest.raises(oauth.OAuthError, match="state"):
        oauth.login(spec(fake), open_browser=browser, out=open("/dev/null", "w"))
    fake.wrong_state, fake.deny = False, "access_denied"
    with pytest.raises(oauth.OAuthError, match="access_denied"):
        oauth.login(spec(fake), open_browser=browser, out=open("/dev/null", "w"))
    assert not oauth.store_path().exists()


def test_login_times_out(fake):
    with pytest.raises(oauth.OAuthError, match="no callback after"):
        oauth.login(spec(fake), open_browser=lambda url: None, out=open("/dev/null", "w"), timeout=0.5)


def test_confidential_client_sends_secret(fake):
    fake.auth_methods, fake.secret = ["client_secret_basic", "client_secret_post"], "geheim"
    oauth.login(spec(fake), open_browser=browser, out=open("/dev/null", "w"))
    assert fake.token_forms[0]["form"]["client_secret"] == "geheim"


def test_refresh_when_expired(fake):
    s = spec(fake)
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    data = json.loads(oauth.store_path().read_text())
    data[fake.url]["expires_at"] = int(time.time())  # running gleich ab
    oauth.store_path().write_text(json.dumps(data))
    old_rt = data[fake.url]["refresh_token"]
    tok = oauth.token(s)
    assert fake.refreshes == 1 and tok == entry(fake)["access_token"]
    assert entry(fake)["refresh_token"] != old_rt  # rotierter refresh-token gespeichert
    assert oauth.token(s) == tok and fake.refreshes == 1  # gueltig: kein neuer refresh


def test_transport_uses_token_and_refreshes_on_401(fake, tmp_path):
    s = spec(fake)
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    m = mcp.Manager([s])
    m.start(wait=True, timeout=10)
    try:
        assert m.summary()[0][1] == "ready"
        first = entry(fake)["access_token"]
        assert fake.mcp_auth[-1] == f"Bearer {first}"
        fake.revoke_all()  # server verwirft den token vor ablauf
        out = m.tools()["mcp__gl__whoami"].func(tools.ToolContext(cwd=str(tmp_path)), {})
        assert fake.refreshes == 1 and entry(fake)["access_token"] != first
        assert entry(fake)["access_token"] in str(out)
    finally:
        m.close()


def test_transport_errors_point_to_login(fake):
    s = spec(fake)
    m = mcp.Manager([s])
    m.start(wait=True, timeout=10)
    _, state, _, err = m.summary()[0]
    assert state == "error" and "not logged in" in err and "--mcp-login gl" in err
    assert fake.mcp_auth == []  # ohne token gar nicht erst angefragt
    m.close()
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    fake.revoke_all()
    fake.refresh.clear()  # refresh-token auch tot
    m = mcp.Manager([s])
    m.start(wait=True, timeout=10)
    _, state, _, err = m.summary()[0]
    assert state == "error" and "invalid_grant" in err and "--mcp-login gl" in err
    m.close()


def test_discovery_requires_s256_and_https(fake, monkeypatch):
    with pytest.raises(oauth.OAuthError, match="https"):
        oauth.discover("http://example.com/mcp")
    meta = oauth.discover(fake.url)
    assert meta["resource"] == fake.url and meta["token_endpoint"].endswith("/token")


def test_config_oauth():
    base = {"base_url": "http://x", "model": "m"}
    cfg = config.build({**base, "mcp": {
        "a": {"url": "https://h/mcp", "oauth": True},
        "b": {"url": "https://h/mcp", "oauth": {"scope": "openid offline_access"}},
        "c": {"url": "https://h/mcp", "oauth": False}}})
    assert [s.oauth for s in cfg.mcp] == [{"scope": None}, {"scope": "openid offline_access"}, None]
    for bad, msg in (({"command": "x", "oauth": True}, "only works with 'url'"),
                     ({"url": "https://h", "oauth": True, "headers": {"authorization": "x"}}, "mutually exclusive"),
                     ({"url": "https://h", "oauth": "ja"}, "true/false"),
                     ({"url": "https://h", "oauth": {"client": 1}}, "unknown")):
        with pytest.raises(config.ConfigError, match=msg):
            config.build({**base, "mcp": {"s": bad}})


def test_cli_login_logout(fake, tmp_path, monkeypatch):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(f"base_url: http://127.0.0.1:9/v1\nmodel: m\nmcp:\n"
                       f"  gl: {{url: '{fake.url}', oauth: true, enabled: false}}\n"
                       f"  plain: {{url: '{fake.url}'}}\n")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parent.parent), "HOME": str(tmp_path),
           "XDG_DATA_HOME": str(tmp_path / "xdg-data"), "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
           "BROWSER": "true"}  # webbrowser.open → /bin/true, no callback after
    run = lambda *a: subprocess.run([sys.executable, "-m", "pyrate", "--config", str(cfgfile), *a],  # noqa: E731
                                    capture_output=True, text=True, timeout=30, env=env, cwd=tmp_path)
    r = run("--mcp-login", "nix")
    assert r.returncode == 2 and "oauth servers in cwd: gl" in r.stderr
    r = run("--mcp-login", "plain")
    assert r.returncode == 2 and "no 'oauth'" in r.stderr
    r = run("--mcp-logout", "gl")
    assert r.returncode == 0 and "was not logged in" in r.stderr


def test_token_endpoint_redirect_is_refused(fake):
    """urllib folgt 302 und behaelt dabei Authorization (basic mit client-secret)."""
    fake.auth_methods, fake.secret = ["client_secret_basic"], "geheim"
    fake.redirect.add("/token")
    with pytest.raises(oauth.OAuthError, match="redirect"):
        oauth.login(spec(fake), open_browser=browser, out=open("/dev/null", "w"))
    assert fake.leaks == []


def test_mcp_redirect_is_refused(fake):
    """bearer-token (oauth) bzw. eigener header darf keinem redirect hinterherlaufen."""
    s = spec(fake)
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    fake.redirect.add("/mcp")
    for server in (s, MCPServer(name="hdr", url=fake.url, headers={"Authorization": "Bearer statisch"})):
        m = mcp.Manager([server])
        m.start(wait=True, timeout=10)
        _, state, _, err = m.summary()[0]
        m.close()
        assert state == "error" and "redirect" in err, err
    assert fake.leaks == []


@pytest.mark.parametrize("value, ok", [(3600, True), ("3600", True), ("1e3", True), ("bald", False),
                                       (None, False), (0, False), (-5, False), ([1], False), ("inf", False)])
def test_expires_in_garbage(value, ok):
    at = oauth._expires_at(value)
    assert (at is not None) is ok and (at is None or at > time.time())


def test_refresh_with_garbage_expires_in(fake, monkeypatch):
    s = spec(fake)
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))
    orig = fake._issue
    monkeypatch.setattr(fake, "_issue", lambda: {**orig(), "expires_in": "irgendwann"})
    assert oauth.token(s, stale=entry(fake)["access_token"]) == entry(fake)["access_token"]
    assert entry(fake)["expires_at"] is None


@pytest.mark.parametrize("broken", [
    ["kein", "dict"],
    {"access_token": 42},
    {"access_token": "at", "expires_at": 0, "refresh_token": "rt"},                    # endpoint/client fehlen
    {"access_token": "at", "expires_at": 0, "refresh_token": "rt", "token_endpoint": "https://x/t",
     "client": "kein dict"},
    {"access_token": "at", "expires_at": 0, "refresh_token": "rt", "token_endpoint": "https://x/t",
     "client": {"client_secret": "ohne id"}},
])
def test_broken_store_entry_is_oauth_error(fake, broken):
    s = spec(fake)
    oauth.store_path().parent.mkdir(parents=True, exist_ok=True)
    oauth.store_path().write_text(json.dumps({fake.url: broken}))
    with pytest.raises(oauth.OAuthError, match="broken"):
        oauth.token(s)
    m = mcp.Manager([s])  # im transport: normaler verbindungsfehler mit login-hinweis
    m.start(wait=True, timeout=10)
    _, state, _, err = m.summary()[0]
    m.close()
    assert state == "error" and "broken" in err and "--mcp-login gl" in err
    oauth.login(s, open_browser=browser, out=open("/dev/null", "w"))  # login repariert
    assert oauth.token(s) == entry(fake)["access_token"]


def test_store_odd_but_harmless(fake):
    """kein access_token = nicht eingeloggt; brokenes expires_at = unbekannt, nicht crash."""
    s = spec(fake)
    oauth.store_path().parent.mkdir(parents=True, exist_ok=True)
    oauth.store_path().write_text(json.dumps({fake.url: {"client": 1}}))
    assert oauth.token(s) is None
    oauth.store_path().write_text(json.dumps({fake.url: {"access_token": "at", "expires_at": "morgen"}}))
    assert oauth.token(s) == "at"
