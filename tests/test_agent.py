import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from pyrate import llm
from pyrate.agent import Agent
from pyrate.db import DB


def test_stream_request_shape(backend, make_cfg):
    cfg = make_cfg()
    backend.script = [{"reasoning": "hmm", "text": "hallo welt"}]
    events = list(llm.stream_chat(cfg, "fake-model", [{"role": "user", "content": "hi"}], effort="high"))
    done = events[-1]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "hallo welt"
    assert done["message"]["reasoning_content"] == "hmm"
    assert done["usage"]["prompt_tokens"] >= 1000
    req = backend.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    assert req["body"]["stream_options"] == {"include_usage": True}
    assert req["body"]["reasoning_effort"] == "high"


def test_mistral_content_chunks(backend, make_cfg):
    """mistrals reasoning-modelle streamen content als liste aus thinking- und text-chunks."""
    think = lambda t: {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": t}]}]}
    backend.script = [{"deltas": [{"content": ""}, think("hm"), think("m"), {"content": [{"type": "text", "text": "o"}]},
                                  {"content": "k"}]}]
    events = list(llm.stream_chat(make_cfg(), "m", [{"role": "user", "content": "hi"}]))
    done = events[-1]["message"]
    assert done["content"] == "ok" and done["reasoning_content"] == "hmm"
    assert [e["text"] for e in events if e["type"] == "reasoning"] == ["hm", "m"]


def test_default_effort_not_sent(backend, make_cfg):
    list(llm.stream_chat(make_cfg(), "m", [{"role": "user", "content": "hi"}]))
    assert "reasoning_effort" not in backend.requests[0]["body"]


def test_effort_defaults_per_model(backend, make_cfg, tmp_path):
    """modell-effort beats backend-effort beats global; modellwechsel setzt den default neu."""
    cfg = make_cfg(models=[{"name": "fake-model", "effort": "high"}, "andere"], backend_effort="low")
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    assert agent.effort == "high"                 # modell-ebene
    agent.set_model("andere")
    assert agent.effort == "low"                  # backend-ebene
    agent.effort = "max"                          # ctrl+t
    agent.set_model("fake-model")
    assert agent.effort == "high"                 # modell-default gewinnt via ctrl+t
    assert Agent(cfg, DB(":memory:"), tmp_path, effort="xhigh").effort == "xhigh"  # explizit bleibt
    assert Agent(cfg, DB(":memory:"), tmp_path, effort="").effort == "high"  # leer = modell-default


def test_http_error(backend, make_cfg):
    backend.script = [{"status": 401, "text": "nope"}]
    try:
        list(llm.stream_chat(make_cfg(), "m", []))
    except llm.LLMError as e:
        assert "401" in str(e) and "nope" in str(e)
    else:
        raise AssertionError("LLMError erwartet")


def test_retry_on_transient_errors(backend, make_cfg, monkeypatch):
    """429/5xx vor dem ersten byte: neuer versuch mit backoff, retry-after hat vorrang."""
    monkeypatch.setattr(llm, "RETRY_BASE", 0.01)
    backend.script = [{"status": 503, "text": "overloaded"},
                      {"status": 429, "text": "slow down", "headers": {"Retry-After": "0"}},
                      {"text": "geht doch"}]
    events = list(llm.stream_chat(make_cfg(), "m", [{"role": "user", "content": "hi"}]))
    warnings = [e["warning"] for e in events if e["type"] == "warning"]
    assert len(warnings) == 2 and "http 503" in warnings[0] and "(2/3)" in warnings[1]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "geht doch"
    assert len(backend.requests) == 3


def test_no_retry_for_auth_quota_or_exhausted(backend, make_cfg, monkeypatch):
    monkeypatch.setattr(llm, "RETRY_BASE", 0.01)
    for step in ({"status": 401, "text": "bad key"},
                 {"status": 429, "text": '{"error": {"code": "insufficient_quota"}}'},
                 {"status": 429, "text": "later", "headers": {"Retry-After": "3600"}}):
        backend.requests.clear()
        backend.script = [step, {"text": "darf nicht kommen"}]
        try:
            list(llm.stream_chat(make_cfg(), "m", []))
        except llm.LLMError as e:
            assert e.status == step["status"]
        else:
            raise AssertionError("LLMError erwartet")
        assert len(backend.requests) == 1
    backend.requests.clear()
    backend.script = [{"status": 502, "text": "x"}] * 4
    try:
        list(llm.stream_chat(make_cfg(), "m", []))
    except llm.LLMError as e:
        assert e.status == 502
    assert len(backend.requests) == 1 + llm.RETRY_MAX


def test_retry_wait_is_cancellable(backend, make_cfg, monkeypatch):
    monkeypatch.setattr(llm, "RETRY_BASE", 30)
    backend.script = [{"status": 503, "text": "x"}]
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    t = time.monotonic()
    try:
        list(llm.stream_chat(make_cfg(), "m", [], cancel=cancel))
    except llm.Cancelled:
        pass
    else:
        raise AssertionError("Cancelled erwartet")
    assert time.monotonic() - t < 5


def test_old_reasoning_dropped_but_kept_in_tool_loop(backend, make_cfg, tmp_path):
    """reasoning nur im laufenden turn zurueckschicken – nach ctrl+p waere es von einem fremden modell."""
    (tmp_path / "f.txt").write_text("x\n")
    backend.script = [
        {"reasoning": "denk 1", "tool_calls": [{"name": "read", "arguments": {"path": "f.txt"}}]},
        {"reasoning": "denk 2", "text": "fertig"},
        {"text": "zweiter turn"},
    ]
    agent = Agent(make_cfg(), DB(":memory:"), tmp_path)
    agent.run("lies f.txt")
    loop_req = backend.requests[1]["body"]["messages"]
    assert next(m for m in loop_req if m["role"] == "assistant")["reasoning_content"] == "denk 1"
    agent.set_model("fake-model")
    agent.run("und jetzt?")
    sent = backend.requests[2]["body"]["messages"]
    assert not any("reasoning_content" in m or "thinking_blocks" in m for m in sent)
    assert any(m.get("reasoning_content") for m in agent.messages)  # verlauf selbst bleibt unangetastet


def test_read_edit_pytest_loop(backend, make_cfg, tmp_path):
    """spec-schritt 2: datei lesen, editieren, pytest laufen lassen."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (proj / "test_calc.py").write_text("from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (proj / "AGENTS.md").write_text("immer pytest -q nutzen")
    backend.script = [
        {"tool_calls": [{"name": "read", "arguments": {"path": "calc.py"}}]},
        {"text": "bug gefunden", "tool_calls": [
            {"name": "edit", "arguments": {"path": "calc.py", "old_string": "a - b", "new_string": "a + b"}}]},
        {"tool_calls": [{"name": "bash", "arguments": {"command": f"{sys.executable} -m pytest -q"}}]},
        {"text": "fertig, tests gruen"},
    ]
    cfg = make_cfg(sandbox_ro=[sys.prefix, sys.base_prefix])  # der venv-python liegt outside von proj
    db = DB(cfg.db_path)
    events = []
    agent = Agent(cfg, db, proj)
    result = agent.run("fix add", events.append)

    assert result == "fertig, tests gruen"
    assert (proj / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    ends = [e for e in events if e["type"] == "tool_end"]
    assert [e["name"] for e in ends] == ["read", "edit", "bash"]
    assert all(e["ok"] for e in ends)
    assert "1 passed" in ends[2]["result"]
    assert [e["n"] for e in events if e["type"] == "step"] == [1, 2, 3, 4]
    assert agent.step == 4
    assert agent.context_tokens > 1000

    first = backend.requests[0]["body"]
    assert "immer pytest -q nutzen" in first["messages"][0]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == {"read", "write", "edit", "bash", "ask_user",
                                                               "grep", "find"}
    last = backend.requests[-1]["body"]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "assistant", "tool", "assistant", "tool",
                                         "assistant", "tool"]

    # fortsetzen aus der db
    again = Agent(cfg, db, proj, session_id=agent.session_id)
    assert len(again.messages) == len(agent.messages)


def test_plan_mode_blocks_writes(backend, make_cfg, tmp_path):
    backend.script = [
        {"tool_calls": [{"name": "write", "arguments": {"path": "x.txt", "content": "boom"}}]},
        {"text": "ok, nur plan"},
    ]
    cfg = make_cfg()
    agent = Agent(cfg, DB(":memory:"), tmp_path, mode="plan")
    events = []
    agent.run("schreib was", events.append)
    assert not (tmp_path / "x.txt").exists()
    end = next(e for e in events if e["type"] == "tool_end")
    assert not end["ok"] and "not allowed" in end["result"]
    body = backend.requests[0]["body"]
    plan_tools = {"read", "ask_user", "grep", "find"} | ({"bash"} if cfg.plan_bash else set())
    assert {t["function"]["name"] for t in body["tools"]} == plan_tools
    assert "PLAN MODE" in body["messages"][0]["content"]


def test_parallel_tool_calls_all_index_zero(backend, make_cfg, tmp_path):
    """litellm ollama_chat/ schickt jeden call mit index 0, aber eigener id – nicht zusammenkleben."""
    (tmp_path / "a.txt").write_text("aaa\n")
    (tmp_path / "b.txt").write_text("bbb\n")
    backend.script = [
        {"ollama_style": True, "tool_calls": [{"name": "read", "arguments": {"path": "a.txt"}},
                                              {"name": "read", "arguments": {"path": "b.txt"}}]},
        {"text": "fertig"},
    ]
    events = []
    Agent(make_cfg(), DB(":memory:"), tmp_path).run("lies beide", events.append)
    ends = [e for e in events if e["type"] == "tool_end"]
    assert len(ends) == 2 and all(e["ok"] for e in ends)
    sent = next(m for m in backend.requests[1]["body"]["messages"] if m.get("tool_calls"))
    assert [json.loads(c["function"]["arguments"]) for c in sent["tool_calls"]] == [{"path": "a.txt"},
                                                                                   {"path": "b.txt"}]
    assert len({c["id"] for c in sent["tool_calls"]}) == 2


def test_missing_tool_args_go_back_to_model(backend, make_cfg, tmp_path):
    backend.script = [{"tool_calls": [{"name": "read", "arguments": {}}]}, {"text": "sorry"}]
    agent = Agent(make_cfg(), DB(":memory:"), tmp_path)
    events = []
    agent.run("x", events.append)
    end = next(e for e in events if e["type"] == "tool_end")
    assert not end["ok"]


def test_title_fallback(backend, make_cfg, tmp_path):
    backend.script = [{"text": ""}]
    agent = Agent(make_cfg(small_model="tiny"), DB(":memory:"), tmp_path)
    agent.ensure_session()
    assert agent.make_title("eins zwei drei vier fuenf sechs sieben acht neun") == "eins zwei drei vier fuenf sechs sieben acht …"
    assert backend.requests[0]["body"]["model"] == "tiny"


def test_print_mode_cli(backend, tmp_path):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(f"base_url: {backend.url}\nmodel: fake\nsmall_model: tiny\n"
                       f"db_path: {tmp_path / 'db.sqlite'}\n")
    backend.script = [{"text": "\n\nantwort aus dem fake"}]
    backend.by_model["tiny"] = [{"text": "Ein Titel"}]
    r = subprocess.run(
        [sys.executable, "-m", "pyrate", "--config", str(cfgfile), "-p", "hallo"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parent.parent), "HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == "antwort aus dem fake\n"
    r = subprocess.run(
        [sys.executable, "-m", "pyrate", "--config", str(cfgfile), "--sessions"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parent.parent), "HOME": str(tmp_path)},
    )
    assert "Ein Titel" in r.stdout


def _cancel_after(cfg, delay):
    """stream_chat starten, nach `delay` s cancel setzen. gibt (exception-typ, dauer, events) back."""
    import threading
    import time
    cancel = threading.Event()
    threading.Timer(delay, cancel.set).start()
    events = []
    t = time.monotonic()
    try:
        for ev in llm.stream_chat(cfg, "m", [{"role": "user", "content": "x"}], cancel=cancel):
            events.append(ev)
        exc = None
    except Exception as e:  # noqa: BLE001
        exc = type(e)
    return exc, time.monotonic() - t, events


def test_cancel_while_waiting_for_headers(backend, make_cfg):
    """esc muss auch greifen, wenn das backend gar nicht antwortet (vorher: erst nach request_timeout)."""
    backend.script = [{"delay": 30, "text": "zu spaet"}]
    exc, took, _ = _cancel_after(make_cfg(), 0.3)
    assert exc is llm.Cancelled
    assert took < 1.5, f"abbruch dauerte {took:.1f}s"


def test_cancel_while_stream_stalls(backend, make_cfg):
    backend.script = [{"stall": 30, "text": "erster chunk und dann stille"}]
    exc, took, events = _cancel_after(make_cfg(), 0.5)
    assert exc is llm.Cancelled
    assert took < 2
    assert events and events[0]["type"] == "text"  # der erste chunk kam noch an


def test_connection_drop_is_llmerror(make_cfg):
    """backend stirbt mitten im stream → LLMError statt roher OSError (sonst 'interner fehler' in der tui)."""
    import socket
    import threading
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     b"10\r\ndata: {\"choices\"")  # chunk kuendigt 16 bytes an, dann tot
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    cfg = make_cfg(base_url=f"http://127.0.0.1:{port}/v1")
    try:
        list(llm.stream_chat(cfg, "m", [{"role": "user", "content": "x"}]))
    except llm.LLMError as e:
        assert "backend lost" in str(e)
    else:
        raise AssertionError("LLMError erwartet")
    finally:
        srv.close()


def test_connect_timeout_short_read_timeout_long(backend, make_cfg):
    """connect kurz (abbrechbarkeit), danach wieder request_timeout – sonst stirbt langsames reasoning."""
    cfg = make_cfg(request_timeout=321)
    tracker = llm._Tracker()
    gen = llm._stream(cfg, {"model": "m", "messages": [], "stream": True}, tracker, None)
    next(gen)
    assert tracker.socks[0].gettimeout() == 321
    gen.close()


def test_print_mode_compact(backend, tmp_path):
    big = "".join(f"zeile {i} " + "x" * 80 + "\n" for i in range(1500))
    (tmp_path / "big.txt").write_text(big)
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(f"base_url: {backend.url}\nmodel: fake\nsmall_model: tiny\ncompact_model: summ\n"
                       f"output_max_bytes: 200000\ndb_path: {tmp_path / 'db.sqlite'}\n")
    backend.by_model["tiny"] = [{"text": "T"}]
    backend.by_model["summ"] = [{"text": "## Goal\n- aus dem print-mode"}]
    backend.script = [{"tool_calls": [{"name": "read", "arguments": {"path": "big.txt"}}]}, {"text": "ok"}]
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parent.parent), "HOME": str(tmp_path)}
    base = [sys.executable, "-m", "pyrate", "--config", str(cfgfile)]
    run = lambda *a: subprocess.run([*base, *a], cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env)  # noqa: E731
    r = run("--compact")
    assert r.returncode == 2 and "session" in r.stderr
    r = run("-p", "/compact")  # / ist nur noch for skills
    assert r.returncode == 2 and "unknown skill /compact" in r.stderr
    assert run("-p", "lies big.txt").returncode == 0
    r = run("-c", "--compact")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "## Goal\n- aus dem print-mode"
    assert "compacted" in r.stderr


def test_routing_across_backends(tmp_path):
    """modellwechsel schickt den request ans richtige backend, mit dessen key und nacktem modellnamen."""
    from fake_backend import FakeBackend

    from pyrate import config
    a, b = FakeBackend().start(), FakeBackend().start()
    try:
        cfg = config.build({"backends": [
            {"name": "eins", "base_url": a.url, "api_key": "key-a", "models": ["m"]},
            {"name": "zwei", "base_url": b.url, "models": ["m", "n"]},
        ], "model": "m@eins", "small_model": "n"})
        agent = Agent(cfg, DB(":memory:"), tmp_path)
        a.script, b.script = [{"text": "von a"}], [{"text": "von b"}]
        assert agent.run("hi") == "von a"
        assert a.requests[0]["body"]["model"] == "m" and a.requests[0]["headers"]["Authorization"] == "Bearer key-a"
        agent.set_model("m@zwei")
        assert agent.run("und?") == "von b"
        assert b.requests[0]["body"]["model"] == "m" and "Authorization" not in b.requests[0]["headers"]
        # session merkt sich das qualifizierte modell
        assert agent.db.get_session(agent.session_id)["model"] == "m@zwei"
        assert Agent(cfg, agent.db, tmp_path, session_id=agent.session_id,
                     model=agent.db.get_session(agent.session_id)["model"]).model == "m@zwei"
    finally:
        a.stop()
        b.stop()


def test_unknown_session_model_falls_back(tmp_path):
    from pyrate import config
    cfg = config.build({"backends": [{"name": "a", "base_url": "http://a/v1", "models": ["m"]},
                                     {"name": "b", "base_url": "http://b/v1", "models": ["n"]}]})
    agent = Agent(cfg, DB(":memory:"), tmp_path, model="weg@gibtsnicht")  # backend aus der config geflogen
    assert agent.model == "m@a"
    assert any("not available" in w for w in agent.context_warnings)
    # alte sessions mit unqualifiziertem namen → qualifiziert
    assert Agent(cfg, DB(":memory:"), tmp_path, model="n").model == "n@b"


def test_cli_models_and_bad_model(tmp_path):
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("backends:\n  - name: api\n    base_url: http://127.0.0.1:9/v1\n    context_limit: 262144\n"
                       "    models: [qwen, big]\n  - base_url: http://localhost:11434/v1\n    models: [qwen3:8b]\n"
                       "model: qwen\nsmall_model: qwen3:8b\n")
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parent.parent), "HOME": str(tmp_path)}
    run = lambda *a: subprocess.run([sys.executable, "-m", "pyrate", "--config", str(cfgfile), *a],  # noqa: E731
                                    cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env)
    r = run("--models")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "qwen@api                  262k  default",
        "big@api                   262k",
        "qwen3:8b@localhost:11434  small",
    ]
    r = run("-m", "gpt-9", "-p", "x")
    assert r.returncode == 2 and "listed in no backend" in r.stderr


def test_set_model_persists_qualified_id(tmp_path):
    """befund: set_model hat den unqualifizierten string gespeichert → resume mehrdeutig."""
    from pyrate import config
    cfg = config.build({"backends": [{"name": "a", "base_url": "http://a/v1", "models": ["m", "x"]},
                                     {"name": "b", "base_url": "http://b/v1", "models": ["n"]}]})
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    agent.ensure_session()
    agent.set_model("x")
    assert agent.db.get_session(agent.session_id)["model"] == "x@a"
    # backend b bekommt later auch ein "x" – resume bleibt trotzdem eindeutig
    cfg.backends[1].models.append(config.Model("x"))
    resumed = Agent(cfg, agent.db, tmp_path, session_id=agent.session_id,
                    model=agent.db.get_session(agent.session_id)["model"])
    assert resumed.model == "x@a" and not resumed.context_warnings


def test_estimate_cache_invalidates(make_cfg, tmp_path):
    agent = Agent(make_cfg(), DB(":memory:"), tmp_path)
    n = agent.estimated_tokens()
    assert agent.estimated_tokens("x" * 400) == n + 100
    agent.messages.append({"role": "user", "content": "y" * 4000})
    assert agent.estimated_tokens() >= n + 1000
    m = agent.estimated_tokens()
    agent.messages = [agent.messages[0]]  # ersetzt (wie bei compaction), gleiche laenge moeglich
    agent.messages[0] = {"role": "user", "content": "kurz"}
    agent.messages = list(agent.messages)
    assert agent.estimated_tokens() < m
    build = agent.estimated_tokens()
    agent.mode = "plan"  # anderer systemprompt + weniger tool-schemas
    assert agent.estimated_tokens() != build


def test_consecutive_user_turns_merged_in_request(backend, make_cfg, tmp_path):
    from pyrate.agent import merge_user_turns
    msgs = [{"role": "user", "content": "summary"}, {"role": "user", "content": "frage"},
            {"role": "assistant", "content": "a"}, {"role": "user", "content": "x"}]
    assert merge_user_turns(msgs) == [{"role": "user", "content": "summary\n\nfrage"}, msgs[2], msgs[3]]
    # abbruch vor der antwort: zweiter prompt folgt direkt auf den ersten
    agent = Agent(make_cfg(), DB(":memory:"), tmp_path)
    agent.ensure_session()
    agent.messages.append({"role": "user", "content": "erster, abgebrochen"})
    backend.script = [{"text": "ok"}]
    agent.run("zweiter")
    sent = backend.requests[0]["body"]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[1]["content"] == "erster, abgebrochen\n\nzweiter"
    assert [m["role"] for m in agent.messages[:2]] == ["user", "user"]  # verlauf unchanged


def test_backend_redirect_is_refused(make_cfg):
    """302 auf den POST: urllib wuerde als GET folgen und den api-key mitnehmen."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    leaks = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(302)
            self.send_header("Location", "/leak")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            leaks.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = make_cfg(base_url=f"http://127.0.0.1:{srv.server_address[1]}/v1")
        events = []
        Agent(cfg, DB(":memory:"), ".").run("hi", events.append)
        errors = [e["error"] for e in events if e["type"] == "error"]
        assert errors and "redirect" in errors[0], events
        assert leaks == []
    finally:
        srv.shutdown()
        srv.server_close()
