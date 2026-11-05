"""mcp_tool_search: mcp schemas erst nach search_tools im request (agent.toolset, meta-tool)."""

import sys
from pathlib import Path

import pytest

from pyrate import config, mcp
from pyrate.agent import SEARCH_TOOL, Agent, search_hits
from pyrate.config import MCPServer
from pyrate.db import DB

FAKE = str(Path(__file__).parent / "fake_mcp.py")


@pytest.fixture
def mgr():
    m = mcp.Manager([MCPServer(name="fake", command=sys.executable, args=[FAKE])])
    m.start(wait=True, timeout=10)
    yield m
    m.close()


def sent_tools(req):
    return {t["function"]["name"]: t for t in req["body"]["tools"]}


def test_config(make_cfg):
    cfg = make_cfg()
    assert cfg.mcp_tool_search is False and cfg.mcp_tool_search_min == 20
    cfg = make_cfg(mcp_tool_search=True, mcp_tool_search_min=3)
    assert cfg.mcp_tool_search is True and cfg.mcp_tool_search_min == 3
    assert make_cfg(mcp_tool_search="yes").mcp_tool_search is False  # nur literal true
    with pytest.raises(config.ConfigError):
        make_cfg(mcp_tool_search_min=-1)


def test_search_hits():
    cands = {"mcp__a__route": "plan a route", "mcp__a__geocode": "address to coords",
             "mcp__b__calendar": "calendar events and route planning"}
    assert search_hits("route planning", cands) == ["mcp__b__calendar", "mcp__a__route"]
    assert search_hits("", cands) == []
    assert search_hits("nix", cands) == []


def test_off_and_below_min_unchanged(make_cfg, mgr, tmp_path):
    full = Agent(make_cfg(), DB(":memory:"), tmp_path, mcp=mgr).tool_names()
    assert SEARCH_TOOL not in full and "mcp__fake__echo" in full
    # aktiv, aber weniger mcp tools als min → alles wie gehabt
    names = Agent(make_cfg(mcp_tool_search=True), DB(":memory:"), tmp_path, mcp=mgr).tool_names()
    assert names == full


def test_deferred_manifest_and_load(backend, make_cfg, mgr, tmp_path):
    agent = Agent(make_cfg(mcp_tool_search=True, mcp_tool_search_min=1), DB(":memory:"), tmp_path, mcp=mgr)
    small = agent.estimated_tokens()
    backend.script = [
        # search + direkt im selben step aufrufen
        {"tool_calls": [{"name": SEARCH_TOOL, "arguments": {"query": "a+b"}},
                        {"name": "mcp__fake__add", "arguments": {"a": 40, "b": 2}}]},
        {"text": "42"},
    ]
    events = []
    assert agent.run("rechne", events.append) == "42"
    first, second = backend.requests[0], backend.requests[1]
    t1 = sent_tools(first)
    assert not any(n.startswith("mcp__") for n in t1)  # nur das meta-tool
    assert "mcp__fake__add: a+b" in t1[SEARCH_TOOL]["function"]["description"]
    ends = [e for e in events if e["type"] == "tool_end"]
    assert ends[0]["ok"] and "mcp__fake__add" in ends[0]["result"]
    assert ends[1]["ok"] and ends[1]["result"] == "42"
    t2 = sent_tools(second)
    assert "mcp__fake__add" in t2 and "mcp__fake__echo" not in t2
    assert "mcp__fake__add" not in t2[SEARCH_TOOL]["function"]["description"]  # aus dem manifest raus
    assert agent.estimated_tokens() > small


def test_load_by_name_unknown_and_empty(make_cfg, mgr, tmp_path):
    agent = Agent(make_cfg(mcp_tool_search=True, mcp_tool_search_min=1), DB(":memory:"), tmp_path, mcp=mgr)
    _, extra = agent.toolset()
    func = extra[SEARCH_TOOL].func
    out = func(None, names=["mcp__fake__echo", "bogus"])
    assert "mcp__fake__echo" in out and "unknown: bogus" in out
    assert agent.loaded_tools == {"mcp__fake__echo"}
    with pytest.raises(Exception, match="no matching tools"):
        func(None, query="zzzz")


def test_direct_call_of_unloaded_tool_loads_it(backend, make_cfg, mgr, tmp_path):
    agent = Agent(make_cfg(mcp_tool_search=True, mcp_tool_search_min=1), DB(":memory:"), tmp_path, mcp=mgr)
    backend.script = [{"tool_calls": [{"name": "mcp__fake__echo", "arguments": {"text": "hi"}}]},
                      {"text": "ok"}]
    events = []
    agent.run("x", events.append)
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] and end["result"] == "hi"
    assert "mcp__fake__echo" in agent.loaded_tools


def test_plan_mode_only_readonly(backend, make_cfg, mgr, tmp_path):
    agent = Agent(make_cfg(mcp_tool_search=True, mcp_tool_search_min=1), DB(":memory:"), tmp_path,
                  mcp=mgr, mode="plan")
    _, extra = agent.toolset()
    desc = extra[SEARCH_TOOL].description
    assert "mcp__fake__echo" in desc and "mcp__fake__add" not in desc  # nur readOnlyHint
    backend.script = [{"tool_calls": [{"name": "mcp__fake__add", "arguments": {"a": 1, "b": 1}}]},
                      {"text": "nope"}]
    events = []
    agent.run("x", events.append)
    end = next(e for e in events if e["type"] == "tool_end")
    assert not end["ok"] and "not allowed" in end["result"]
