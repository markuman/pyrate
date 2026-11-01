import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import HAS_LANDLOCK

from pyrate import config, subagent, tools
from pyrate.agent import Agent
from pyrate.db import DB

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------- config

def test_config_subagents(make_cfg):
    assert make_cfg().subagents == 0
    assert make_cfg(subagents=3).subagents == 3
    for bad in (-1, True, "viele"):
        with pytest.raises(config.ConfigError):
            make_cfg(subagents=bad)


def test_config_arg_only_when_explicit(tmp_path, backend):
    f = tmp_path / "c.yaml"
    f.write_text(f"base_url: {backend.url}\nmodel: fake\n")
    assert config.load(tmp_path, str(f)).config_arg == f.absolute()


# ---------------------------------------------------------------- db

def test_child_sessions_hidden(db):
    top = db.create_session("/p", "m")
    child = db.create_session("/p", "m", parent_id=top)
    db.add_message(child, "user", "plan", [("text", {"text": "kind"})])  # kind ist juenger
    assert db.latest_session("/p")["id"] == top
    assert [s["id"] for s in db.list_sessions("/p")] == [top]
    assert [s["id"] for s in db.list_sessions()] == [top]
    assert db.get_session(child)["parent_id"] == top  # per -s weiter erreichbar


def test_migration_from_v1(tmp_path):
    from pyrate.db import MIGRATIONS
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(f"BEGIN;\n{MIGRATIONS[0]}\nPRAGMA user_version=1;\nCOMMIT;")
    conn.execute("INSERT INTO sessions (id, cwd, created_at, updated_at) VALUES ('abc', '/p', 1, 1)")
    conn.commit()
    conn.close()
    d = DB(path)
    assert d.latest_session("/p")["parent_id"] is None
    d.close()


# ---------------------------------------------------------------- tools

def test_bash_env(tmp_path):
    ctx = tools.ToolContext(cwd=str(tmp_path), env={"PYRATE_X": "hallo"})
    assert tools.t_bash(ctx, "echo $PYRATE_X").startswith("hallo")


def test_bash_timeout_sends_sigterm_first(tmp_path):
    ctx = tools.ToolContext(cwd=str(tmp_path))
    out = tools.t_bash(ctx, "trap 'echo aufgeraeumt; exit 0' TERM; sleep 30 & wait", timeout=1)
    assert "aufgeraeumt" in out and "timeout" in out


def test_bash_waits_for_group_cleanup(tmp_path):
    """bash stirbt bei SIGTERM sofort, ein kind (z.b. sub-pyrate) darf trotzdem fertig aufraeumen."""
    script = ("import signal, sys, time\n"
              "def bye(*a):\n"
              "    time.sleep(0.5)\n"
              "    open('aufgeraeumt', 'w').close()\n"
              "    sys.exit(0)\n"
              "signal.signal(signal.SIGTERM, bye)\n"
              "print('running', flush=True)\n"
              "time.sleep(30)\n")
    (tmp_path / "kind.py").write_text(script)
    ctx = tools.ToolContext(cwd=str(tmp_path))
    out = tools.t_bash(ctx, f"{sys.executable} kind.py", timeout=1)
    assert "timeout" in out
    assert (tmp_path / "aufgeraeumt").exists()


def test_bash_sigkill_after_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "TERM_GRACE", 0.3)
    ctx = tools.ToolContext(cwd=str(tmp_path))
    t = time.monotonic()
    out = tools.t_bash(ctx, "trap '' TERM; sleep 30", timeout=1)
    assert time.monotonic() - t < 5 and "timeout" in out


# ---------------------------------------------------------------- agent

def _system(agent):
    return agent.system_prompt()


def test_system_prompt_sections(make_cfg, tmp_path):
    cfg = make_cfg(subagents=2)
    a = Agent(cfg, DB(":memory:"), tmp_path)
    assert "SUBAGENTS:" in _system(a) and "at most 2 per bash call" in _system(a)
    assert "dir=$(mktemp -d)" in _system(a)
    a.mode = "plan"  # ohne bash nuetzt die anleitung nichts
    assert "SUBAGENTS:" not in _system(a)
    a.mode, a.subagents = "build", 0
    assert "SUBAGENTS:" not in _system(a)
    off = Agent(make_cfg(), DB(":memory:"), tmp_path)
    assert "SUBAGENTS:" not in _system(off)
    child = Agent(cfg, DB(":memory:"), tmp_path, parent_id="abc")
    assert child.subagents == 0
    assert "SUBAGENT:" in _system(child) and "SUBAGENTS:" not in _system(child)


def test_tool_env_reaches_bash(backend, make_cfg, tmp_path):
    backend.script = [
        {"tool_calls": [{"name": "bash", "arguments": {"command": "echo $PYRATE_SESSION:$PYRATE_SUBAGENTS"}}]},
        {"text": "ok"},
    ]
    agent = Agent(make_cfg(subagents=3), DB(":memory:"), tmp_path)
    events = []
    agent.run("x", events.append)
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["result"].startswith(f"{agent.session_id}:3")


# ---------------------------------------------------------------- slots

class _Stdin:
    def __init__(self, tty):
        self.tty = tty

    def isatty(self):
        return self.tty


def test_from_env():
    pipe = _Stdin(False)
    assert subagent.from_env({}, pipe) is None
    assert subagent.from_env({"PYRATE_SESSION": "../etc"}, pipe) is None
    assert subagent.from_env({"PYRATE_SESSION": "abc"}, pipe) == ("abc", 0)
    assert subagent.from_env({"PYRATE_SESSION": "abc", "PYRATE_SUBAGENTS": "2"}, pipe) == ("abc", 2)
    assert subagent.from_env({"PYRATE_SESSION": "abc", "PYRATE_SUBAGENTS": "x"}, pipe) == ("abc", 0)
    # mensch am terminal mit geerbter env (tmux aus einem agent-lauf): kein kind
    assert subagent.from_env({"PYRATE_SESSION": "abc", "PYRATE_SUBAGENTS": "2"}, _Stdin(True)) is None


def test_human_with_leaked_env_runs_normally(backend, tmp_path):
    cfgfile = _cfgfile(tmp_path, backend)
    backend.script = [{"text": "hallo mensch"}]
    backend.by_model["tiny"] = [{"text": "Titel"}]  # als mensch running der titel-call parallel mit
    master, slave = os.openpty()
    try:
        r = _cli(tmp_path, cfgfile, "-p", "x", stdin=slave, PYRATE_SESSION="abc", PYRATE_SUBAGENTS="0")
    finally:
        os.close(master)
        os.close(slave)
    assert r.returncode == 0, r.stderr
    assert r.stdout == "hallo mensch\n"


def test_slots_limit_parallelism():
    first = subagent.acquire_slot("abc", 1)
    got = []
    t = threading.Thread(target=lambda: got.append(subagent.acquire_slot("abc", 1, poll=0.02)))
    t.start()
    time.sleep(0.2)
    assert not got  # wartet
    os.close(first)
    t.join(2)
    assert got
    os.close(got[0])
    a, b = subagent.acquire_slot("abc", 2), subagent.acquire_slot("abc", 2)  # zwei parallel ok
    os.close(a)
    os.close(b)


# ---------------------------------------------------------------- cli als kind

def _cfgfile(tmp_path, backend):
    f = tmp_path / "c.yaml"
    f.write_text(json.dumps({
        "backends": [{"name": "fb", "base_url": backend.url, "models": ["fake", "other"]}],
        "model": "fake", "small_model": "tiny", "db_path": str(tmp_path / "db.sqlite"),
        "sandbox": HAS_LANDLOCK,
    }))
    return f


def _cli(tmp_path, cfgfile, *argv, stdin=subprocess.DEVNULL, **env):
    """stdin wie im bash-tool: /dev/null. sonst erbt's pytests terminal und gilt als mensch."""
    base = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "HOME": str(tmp_path),
            "XDG_RUNTIME_DIR": os.environ["XDG_RUNTIME_DIR"]}
    return subprocess.run([sys.executable, "-m", "pyrate", "--config", str(cfgfile), *argv], stdin=stdin,
                          cwd=tmp_path, capture_output=True, text=True, timeout=30, env={**base, **env})


def test_child_prints_only_final_answer(backend, tmp_path):
    (tmp_path / "f.txt").write_text("inhalt")
    cfgfile = _cfgfile(tmp_path, backend)
    backend.script = [
        {"text": "ich schau mal", "tool_calls": [{"name": "read", "arguments": {"path": "f.txt"}}]},
        {"text": "finale antwort"},
    ]
    r = _cli(tmp_path, cfgfile, "-p", "lies f.txt", PYRATE_SESSION="abc123", PYRATE_SUBAGENTS="1",
             PYRATE_MODEL="other@fb")
    assert r.returncode == 0, r.stderr
    assert r.stdout == "finale antwort\n"
    assert "▸" not in r.stderr and "session " in r.stderr
    bodies = [q["body"] for q in backend.requests]
    assert len(bodies) == 2 and all(b["model"] == "other" for b in bodies)  # kein titel-call, modell geerbt
    system = bodies[0]["messages"][0]["content"]
    assert "SUBAGENT:" in system and "PLAN MODE" in system  # default plan
    tool_names = {t["function"]["name"] for t in bodies[0]["tools"]}
    assert "write" not in tool_names and "edit" not in tool_names

    d = DB(tmp_path / "db.sqlite")
    (row,) = d.conn.execute("SELECT * FROM sessions").fetchall()
    assert row["parent_id"] == "abc123" and row["title"] == "lies f.txt"
    d.close()
    r = _cli(tmp_path, cfgfile, "--sessions")
    assert r.stdout == ""  # kind-sessions tauchen nicht auf


def test_child_ask_user_gets_no_human(backend, tmp_path):
    backend.script = [{"tool_calls": [{"name": "ask_user", "arguments": {"question": "welches?"}}]},
                      {"text": "angenommen"}]
    r = _cli(tmp_path, _cfgfile(tmp_path, backend), "-p", "x", PYRATE_SESSION="abc", PYRATE_SUBAGENTS="1")
    assert r.returncode == 0, r.stderr
    tool_msg = backend.requests[1]["body"]["messages"][-1]
    assert "no human reachable" in tool_msg["content"]


def test_child_build_mode_allowed(backend, tmp_path):
    cfgfile = _cfgfile(tmp_path, backend)
    backend.script = [{"text": "ok"}]
    r = _cli(tmp_path, cfgfile, "--mode", "build", "-p", "x", PYRATE_SESSION="abc", PYRATE_SUBAGENTS="1")
    assert r.returncode == 0, r.stderr
    assert "PLAN MODE" not in backend.requests[0]["body"]["messages"][0]["content"]


@pytest.mark.parametrize("argv, env, msg", [
    (["-p", "x"], {"PYRATE_SUBAGENTS": "0"}, "subagents are off"),
    (["-p", "x"], {}, "subagents are off"),
    (["-c", "-p", "x"], {"PYRATE_SUBAGENTS": "1"}, "don't work as subagent"),
    (["-s", "abc", "--compact", "-p", "x"], {"PYRATE_SUBAGENTS": "1"}, "don't work as subagent"),
    (["-s", "abc", "--compact"], {"PYRATE_SUBAGENTS": "1"}, "only with -p"),
    ([], {"PYRATE_SUBAGENTS": "1"}, "only with -p"),
])
def test_child_refused(backend, tmp_path, argv, env, msg):
    r = _cli(tmp_path, _cfgfile(tmp_path, backend), *argv, PYRATE_SESSION="abc", **env)
    assert r.returncode == 2 and msg in r.stderr
    assert backend.requests == []


def test_child_resume_only_siblings(backend, tmp_path):
    cfgfile = _cfgfile(tmp_path, backend)
    d = DB(tmp_path / "db.sqlite")
    parent = d.create_session(str(tmp_path), "fake@fb")
    other = d.create_session(str(tmp_path), "fake@fb")
    sibling = d.create_session(str(tmp_path), "fake@fb", parent_id=parent)
    d.close()
    env = {"PYRATE_SESSION": parent, "PYRATE_SUBAGENTS": "1"}
    for sid in (parent, other):  # weder den aufrufer noch fremde sessions
        r = _cli(tmp_path, cfgfile, "-s", sid, "-p", "x", **env)
        assert r.returncode == 2 and "of the same caller" in r.stderr
    assert backend.requests == []
    backend.script = [{"text": "weiter geht's"}]
    r = _cli(tmp_path, cfgfile, "-s", sibling, "-p", "x", **env)
    assert r.returncode == 0 and r.stdout == "weiter geht's\n", r.stderr


def test_search_marks_child_sessions(tmp_path, backend):
    cfgfile = _cfgfile(tmp_path, backend)
    d = DB(tmp_path / "db.sqlite")
    child = d.create_session(str(tmp_path), "m", parent_id="abc")
    d.add_message(child, "user", "plan", [("text", {"text": "terraform"})])
    d.close()
    r = _cli(tmp_path, cfgfile, "--search", "terraform")
    assert "[subagent of abc]" in r.stdout


def test_project_cannot_set_subagents(tmp_path, backend):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".pyrate.yaml").write_text("subagents: 8\n")
    cfg = config.load(tmp_path, str(_cfgfile(tmp_path, backend)))
    assert cfg.subagents == 0 and any("subagents" in w for w in cfg.warnings)


def test_subagent_cmd_is_own_interpreter(make_cfg, tmp_path):
    from pyrate.agent import subagent_cmd
    assert subagent_cmd().startswith(sys.executable)
    assert subagent_cmd(tmp_path / "a b.yaml").endswith("--config '" + str(tmp_path / "a b.yaml") + "'")


def test_dogfooding_end_to_end(backend, tmp_path):
    """parent-agent ruft per bash ein echtes `pyrate -p` auf, das sieht sich als kind."""
    import shlex
    cfgfile = _cfgfile(tmp_path, backend)
    cfg = config.load(tmp_path, str(cfgfile))
    cfg.subagents = 1
    cmd = (f"PYTHONPATH={shlex.quote(str(ROOT))} {shlex.quote(sys.executable)} -m pyrate "
           f"--config {shlex.quote(str(cfgfile))} -p 'schau dir das an'")
    backend.script = [
        {"tool_calls": [{"name": "bash", "arguments": {"command": cmd, "timeout": 60}}]},
        {"text": "zwischendurch", "tool_calls": [{"name": "read", "arguments": {"path": "."}}]},  # kind
        {"text": "kind-antwort"},  # kind
        {"text": "fertig"},
    ]
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    events = []
    assert agent.run("delegier das", events.append) == "fertig"
    end = next(e for e in events if e["type"] == "tool_end")
    lines = end["result"].splitlines()
    assert lines[0] == "kind-antwort" and lines[1].startswith("session ") and lines[-1] == "[exit 0]"
    assert "zwischendurch" not in end["result"]
    child = backend.requests[1]["body"]["messages"][0]["content"]
    assert "SUBAGENT:" in child and "SUBAGENTS:" not in child


def test_sigterm_exits_cleanly(backend, tmp_path):
    cfgfile = _cfgfile(tmp_path, backend)
    backend.script = [{"delay": 10, "text": "zu spaet"}]
    proc = subprocess.Popen(
        [sys.executable, "-m", "pyrate", "--config", str(cfgfile), "-p", "x"], cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT), "HOME": str(tmp_path)},
    )
    for _ in range(100):  # warten, bis der request draussen ist
        if backend.requests:
            break
        time.sleep(0.05)
    proc.terminate()
    _, err = proc.communicate(timeout=10)
    assert proc.returncode == 130 and "cancelled" in err and "Traceback" not in err
