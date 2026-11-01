"""mcp_magic: router-call vor jedem user-prompt, der mcp server ein-/ausschaltet (enable-only).
see agent.route_mcp, mcp.descriptions store, config key mcp_magic."""

import urllib.request

import pytest

from pyrate import config, llm, mcp
from pyrate.agent import Agent, parse_magic
from pyrate.config import MCPServer
from pyrate.db import DB
from pyrate.mcp import Manager


def magic_cfg(make_cfg, **extra):
    return make_cfg(mcp_magic=True, mcp={
        "off": {"url": "http://127.0.0.1:9/mcp", "enabled": False, "description": "off server"},
        "on": {"url": "http://127.0.0.1:8/mcp"},
    }, **extra)


# ---------------------------------------------------------------- config

def test_config_mcp_magic(make_cfg):
    assert make_cfg().mcp_magic is None                      # default: off
    assert make_cfg(mcp_magic=False).mcp_magic is None
    cfg = make_cfg(mcp_magic=True)
    assert cfg.mcp_magic == cfg.model                       # true → small_model, fallback main
    cfg = make_cfg(small_model="fake-model", mcp_magic=True)
    assert cfg.mcp_magic == cfg.small_model
    assert "@" in make_cfg(mcp_magic="fake-model").mcp_magic  # string → qualified
    with pytest.raises(config.ConfigError):
        make_cfg(mcp_magic=42)


def test_config_mcp_description(make_cfg):
    cfg = make_cfg(mcp={"gpx": {"url": "http://x/mcp", "description": "karten"}})
    assert cfg.mcp[0].description == "karten"
    with pytest.raises(config.ConfigError):
        make_cfg(mcp={"gpx": {"url": "http://x/mcp", "description": 5}})


# ---------------------------------------------------------------- descriptions store

def test_descriptions_roundtrip():
    p = mcp.descriptions_path()
    assert not p.exists()
    assert mcp.load_descriptions() == {}
    mcp.save_descriptions({"gpx": {"description": "karten", "tools": ["route"]}})
    assert mcp.load_descriptions() == {"gpx": {"description": "karten", "tools": ["route"]}}
    assert (p.stat().st_mode & 0o777) == 0o600


def test_candidates_stored_vs_live():
    m = Manager([MCPServer(name="gpx", url="http://127.0.0.1:9/mcp", description="cfg-desc")])
    try:
        mcp.save_descriptions({"gpx": {"description": "stored-desc",
                                       "tools": ["route: ziel berechnen", "geocode"]}})
        # nicht verbunden → stored tool lines; config-description gewinnt vor dem store
        assert m.candidates() == [("gpx", "cfg-desc", ["route: ziel berechnen", "geocode"])]
        # verbunden → echte tools mit description; description aus dem store, wenn config keine hat
        m.state["gpx"] = {"state": "ready", "error": None}
        m.clients["gpx"].tools = [{"name": "live_tool", "inputSchema": {},
                                   "description": "macht was  " + "x" * 500}]
        m.clients["gpx"].spec.description = None
        (name, desc, tl), = m.candidates()
        assert name == "gpx" and desc == "stored-desc"
        # whitespace kollabiert, description auf 200 chars gekürzt
        assert tl == ["live_tool: " + "macht was " + "x" * (200 - len("macht was "))]
    finally:
        m.close()


# ---------------------------------------------------------------- parse

def test_parse_magic():
    assert parse_magic('{"enable": ["a", "b"]}') == ["a", "b"]
    assert parse_magic('prose drumherum {"enable": ["a"]} bla') == ["a"]
    assert parse_magic('```json\n{"enable": ["a"]}\n```') == ["a"]
    assert parse_magic('["a", "b"]') == ["a", "b"]           # bare list
    assert parse_magic('Sure! ["gpx"]') == ["gpx"]
    assert parse_magic('{"enable": ["a", 5]}') == ["a"]
    assert parse_magic('{"enable": []}') == []               # gültig: "keiner passt"
    assert parse_magic('{"enable": "a"}') is None            # unparsebar → None (fallback)
    assert parse_magic('{"other": []}') is None
    assert parse_magic("garbage") is None
    assert parse_magic("") is None


# ---------------------------------------------------------------- router

def test_route_mcp_enables(make_cfg, tmp_path, monkeypatch):
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    seen = []

    def fake_complete(cfg_, model, messages, **kw):
        seen.append((model, kw, messages[0]["content"]))
        return 'hier: {"enable": ["OFF", "bogus", "on"]}'

    monkeypatch.setattr(llm, "complete", fake_complete)
    events = []
    agent.route_mcp("wo ist die nächste tanke", events.append, None)
    assert mgr.enabled["off"] is True and mgr.enabled["on"] is True
    # nur bekannte und noch ausgeschaltete; "on" war schon an, "bogus" unbekannt.
    # groß-/kleinschreibung egal ("OFF" → off)
    assert events[0]["type"] == "mcp_magic" and events[0]["enabled"] == ["off"]
    assert "raw" in events[0]
    model, kw, msg = seen[0]
    assert model == cfg.mcp_magic
    assert kw["effort"] == "none" and kw["timeout"] == cfg.mcp_magic_timeout and kw["max_tokens"] == 512
    assert kw["temperature"] == 0.0
    assert "wo ist die nächste tanke" in msg and "- off: off server" in msg
    mgr.close()


def test_route_mcp_retries_without_effort_on_400(make_cfg, tmp_path, monkeypatch):
    """ollama & co. kennen reasoning_effort evtl. nicht → 400. dann nochmal ohne den param."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    calls = []

    def fake_complete(cfg_, model, messages, **kw):
        calls.append(kw)
        if "effort" in kw:  # erster versuch mit effort="none" → 400
            raise llm.LLMError("http 400: unknown field reasoning_effort", status=400)
        return '{"enable": ["off"]}'

    monkeypatch.setattr(llm, "complete", fake_complete)
    events = []
    agent.route_mcp("tanke", events.append, None)
    assert len(calls) == 2 and "effort" not in calls[1] and "temperature" not in calls[1]
    assert mgr.enabled["off"] is True and events[0]["enabled"] == ["off"]
    mgr.close()


def test_route_mcp_skips_when_all_ready(make_cfg, tmp_path, monkeypatch):
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    for n in mgr.enabled:
        mgr.enabled[n] = True
        mgr.state[n] = {"state": "ready", "error": None}
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("router should not run"))
    agent.route_mcp("egal", lambda ev: None, None)
    mgr.close()


def test_route_mcp_fail_open(make_cfg, tmp_path, monkeypatch):
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)

    def boom(*a, **k):
        raise llm.LLMError("backend down")

    monkeypatch.setattr(llm, "complete", boom)
    events = []
    agent.route_mcp("egal", events.append, None)  # darf nicht raisen
    assert mgr.enabled["off"] is False  # fallback findet bei "egal" nichts
    assert events[0]["type"] == "warning" and "mcp_magic" in events[0]["warning"]
    mgr.close()


def test_route_mcp_runtime_toggle(make_cfg, tmp_path, monkeypatch):
    """default magic=False → kein router-call, auch mit mcp_magic in der config; agent.magic=True
    (toggle_magic in der tui) schaltet ihn für die session an."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    assert agent.magic is False  # default: aus
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("router must not run"))
    agent.route_mcp("welche termine habe ich heute", lambda ev: None, None)
    assert mgr.enabled["off"] is False
    agent.magic = True  # an → router läuft
    monkeypatch.setattr(llm, "complete", lambda *a, **k: '{"enable": ["off"]}')
    events = []
    agent.route_mcp("welche termine habe ich heute", events.append, None)
    assert mgr.enabled["off"] is True and events[0]["enabled"] == ["off"]
    mgr.close()


def test_route_mcp_off_without_config(make_cfg, tmp_path, monkeypatch):
    cfg = make_cfg(mcp={"a": {"url": "http://127.0.0.1:9/mcp"}})  # kein mcp_magic
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("router off"))
    agent.route_mcp("egal", lambda ev: None, None)
    mgr.close()


# ---------------------------------------------------------------- loopback proxy bypass

def test_loopback_never_via_proxy(monkeypatch):
    """with HTTP_PROXY set and no NO_PROXY: a loopback backend (ollama) must bypass the proxy,
    otherwise the request hangs in the proxy until the timeout – even with a warm model."""
    from pyrate.llm import _Tracker, _is_loopback, _opener
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("http_proxy", "http://proxy.invalid:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert _is_loopback("http://localhost:11434/v1/chat/completions")
    assert _is_loopback("http://127.0.0.1:11434/v1/chat/completions")
    assert _is_loopback("http://[::1]:11434/v1/chat/completions")
    assert not _is_loopback("https://api.vibe.dev/v1/chat/completions")
    op = _opener(_Tracker(), "http://localhost:11434/v1/chat/completions")
    # loopback: no proxy handler with proxies at all → direct connection
    assert not any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in op.handlers)
    op = _opener(_Tracker(), "https://api.vibe.dev/v1/chat/completions")
    # remote: env proxy stays in effect
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies for h in op.handlers)


# ---------------------------------------------------------------- keyword fallback

def test_magic_keywords_de_en():
    from pyrate.agent import magic_keywords
    servers = [
        ("nextcloud", "files, calendar, contacts", ["list_events: calendar events today",
                                                   "get_calendar: list calendars"]),
        ("gpx", "maps", ["plan_route: route between two points", "geocode: address to coords",
                        "search_poi: find places", "analyze_gpx: stats"]),
    ]
    # deutsches prompt, englische descs → synonym-brücke greift (termine → calendar/event)
    assert magic_keywords("welche termine habe ich heute?", servers) == ["nextcloud"]
    assert magic_keywords("wie weit ist fanningerstr 66 bis corinthstr 41?", servers) == []
    # englisches prompt matcht direkt auf tool-namen (distance/route)
    assert magic_keywords("distance between A and B, plan a route", servers) == ["gpx"]
    # ein einzelnes wort reicht nicht (min_hits=2)
    assert magic_keywords("calendar", servers) == []
    # min_hits=1: ein wort genügt (zweiter fallback-pass)
    assert magic_keywords("calendar", servers, min_hits=1) == ["nextcloud"]


def test_route_mcp_keyword_fallback_when_llm_unparseable(make_cfg, tmp_path, monkeypatch):
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    mcp.save_descriptions({"off": {"description": "calendar events", "tools": ["list_events: termine heute"]}})
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "keine ahnung, sorry")
    events = []
    agent.route_mcp("welche termine habe ich heute", events.append, None)
    assert mgr.enabled["off"] is True
    assert events[0]["enabled"] == ["off"]
    mgr.close()


def test_route_mcp_keyword_fallback_names_only_store(make_cfg, tmp_path, monkeypatch):
    """alter store ohne descriptions (nur tool-namen): fallback greift über die tool-namen."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    mcp.save_descriptions({"off": {"description": "", "tools": ["get_calendar", "list_events"]}})
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "???")
    events = []
    agent.route_mcp("mein kalender heute", events.append, None)
    assert events[0]["enabled"] == ["off"]
    mgr.close()


def test_fallback_sees_all_tools_beyond_cap(make_cfg, tmp_path, monkeypatch):
    """regression: nextcloud hat >12 tools, calendar-tools kommen alphabetisch nach collectives_*.
    der keyword-fallback darf nicht am max_tools-cap hängen bleiben."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    tools = [f"collectives_{c}: page stuff" for c in "abcdefghijklmnop"]  # 16× collectives_* zuerst
    tools += ["calendar_list_events: list calendar events", "calendar_get_event: one event"]
    mcp.save_descriptions({"off": {"description": "nextcloud zeug", "tools": tools}})
    seen = []
    monkeypatch.setattr(llm, "complete",
                        lambda cfg_, model, messages, **kw: seen.append(messages[0]["content"]) or "???")
    events = []
    agent.route_mcp("welche termine habe ich heute", events.append, None)
    assert events[0]["enabled"] == ["off"]  # calendar_* liegt jenseits der ersten 12
    assert seen[0].count("\n    ") == 12    # prompt bekommt trotzdem nur 12 tools
    mgr.close()


def test_route_mcp_waits_for_new_servers(make_cfg, tmp_path, monkeypatch):
    """frisch enabled server: route_mcp wartet bounded, bis er ready/error ist (damit die
    tools schon im ersten llm-request landen), statt sofort weiterzulaufen."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    waited = []
    monkeypatch.setattr(llm, "complete", lambda *a, **k: '{"enable": ["off"]}')
    monkeypatch.setattr(mgr, "wait_ready", lambda names, timeout, cancel=None: waited.append((names, timeout)))
    agent.route_mcp("was auch immer", lambda ev: None, None)
    assert waited == [(["off"], min(15, cfg.mcp_magic_timeout))]
    mgr.close()


def test_prompt_ranks_relevant_tools_first(make_cfg, tmp_path, monkeypatch):
    """bei vielen tools kommen die prompt-relevanten zuerst in den llm-prompt (nicht
    alphabetisch die ersten 12)."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    tools = [f"collectives_{c}: page stuff" for c in "abcdefghijklmnop"]
    tools += ["calendar_list_events: list calendar events", "calendar_get_event: one event"]
    mcp.save_descriptions({"off": {"description": "nc", "tools": tools}})
    seen = []
    monkeypatch.setattr(llm, "complete",
                        lambda cfg_, model, messages, **kw: seen.append(messages[0]["content"]) or '{"enable": []}')
    agent.route_mcp("welche termine heute", lambda ev: None, None)
    msg = seen[0]
    assert msg.index("calendar_list_events") < msg.index("collectives_a")  # calendar zuerst
    assert "collectives_p" not in msg  # nur top 12 tools im prompt
    mgr.close()


# ---------------------------------------------------------------- review-fixes

def test_route_mcp_empty_answer_is_final(make_cfg, tmp_path, monkeypatch):
    """[] vom router ist eine antwort, kein fehler: der keyword-fallback darf nicht drüberraten."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    mcp.save_descriptions({"off": {"description": "calendar events", "tools": ["list_events: termine heute"]}})
    for answer in ('{"enable": []}', '{"enable": ["on"]}'):  # leer bzw. nur schon aktive
        monkeypatch.setattr(llm, "complete", lambda *a, _r=answer, **k: _r)
        events = []
        agent.route_mcp("welche termine habe ich heute", events.append, None)
        assert mgr.enabled["off"] is False, answer
        assert events[0]["enabled"] == []
    mgr.close()


def test_magic_keywords_single_word_not_enough():
    from pyrate.agent import magic_keywords
    servers = [("nc", "files and calendar", ["list_files: list files"])]
    assert magic_keywords("list this please", servers) == []  # ein generisches wort: nix


def test_route_mcp_skips_empty_prompt(make_cfg, tmp_path, monkeypatch):
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("no router for image-only"))
    agent.route_mcp("   ", lambda ev: None, None)
    mgr.close()


def test_route_mcp_skips_when_only_errors(make_cfg, tmp_path, monkeypatch):
    """alles enabled, einer im error: kein router (sonst läuft er bei jedem prompt)."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    for n in mgr.enabled:
        mgr.enabled[n] = True
    mgr.state["off"] = {"state": "error", "error": "down"}
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: pytest.fail("router should not run"))
    agent.route_mcp("egal", lambda ev: None, None)
    mgr.close()


def test_run_cancel_during_routing(make_cfg, tmp_path, monkeypatch):
    """esc während des router-calls → sauberes cancelled-event, keine exception."""
    cfg = magic_cfg(make_cfg)
    mgr = Manager(cfg.mcp)
    agent = Agent(cfg, DB(":memory:"), tmp_path, mcp=mgr)
    agent.magic = True  # router an (default ist aus, siehe test_route_mcp_runtime_toggle)

    def cancelled(*a, **k):
        raise llm.Cancelled()

    monkeypatch.setattr(llm, "complete", cancelled)
    monkeypatch.setattr(llm, "stream_chat", lambda *a, **k: pytest.fail("main loop must not run"))
    events = []
    assert agent.run("tanke", events.append) is None
    assert events[-1]["type"] == "cancelled"
    mgr.close()


def test_mcp_magic_protected_in_project(tmp_path):
    assert "mcp_magic" in config.PROTECTED_KEYS


def test_mcp_setup_keeps_stored_tools_on_error(make_cfg, monkeypatch):
    from pyrate.__main__ import cmd_mcp_setup
    cfg = make_cfg(mcp={"gpx": {"url": "http://127.0.0.1:9/mcp", "enabled": False}})
    cfg.mcp[0].error = "env GPX_TOKEN missing"  # → sofort error, kein connect
    mcp.save_descriptions({"gpx": {"description": "maps", "tools": ["route: a→b"], "updated": "2026-01-01"}})
    assert cmd_mcp_setup(cfg) == 0
    entry = mcp.load_descriptions()["gpx"]
    assert entry["tools"] == ["route: a→b"] and entry["description"] == "maps"
    assert entry["updated"] == "2026-01-01" and "GPX_TOKEN" in entry["error"]
