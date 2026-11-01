import json

import pytest

from pyrate import compact, llm
from pyrate.agent import Agent
from pyrate.compact import CompactError
from pyrate.db import DB

SUMMARY = "## Goal\n- calc.py fixen\n## Next Steps\n- tests laufen lassen"
def BIG(lines):
    """viele zeilen: read kappt einzelne zeilen bei 2000 zeichen."""
    return "".join(f"zeile {i} " + "x" * 80 + "\n" for i in range(lines))


OVERFLOW = json.dumps({"error": {"message": "This model's maximum context length is 1000 tokens",
                                 "code": "context_length_exceeded"}})


def msg(i, role, *parts):
    return {"id": i, "role": role, "mode": "build", "parts": [{"type": t, "data": d} for t, d in parts]}


def user(i, text):
    return msg(i, "user", ("text", {"text": text}))


def asst_call(i, cid, name, args, text=None):
    parts = [("text", {"text": text})] if text else []
    parts.append(("tool_call", {"id": cid, "name": name, "arguments": json.dumps(args)}))
    return msg(i, "assistant", *parts)


def result(i, cid, content, ok=True):
    return msg(i, "tool", ("tool_result", {"tool_call_id": cid, "name": "x", "content": content, "ok": ok}))


# ---------------------------------------------------------------- unit

def test_find_cut_prefers_user_boundary():
    active = [user(1, "a" * 4000), msg(2, "assistant", ("text", {"text": "b" * 4000})),
              user(3, "c" * 400), msg(4, "assistant", ("text", {"text": "d" * 400}))]
    assert compact.find_cut(active, keep=500) == 2  # schwanz = letzter turn


def test_find_cut_falls_back_to_assistant_never_orphans_tool_result():
    """ein prompt, dann langer agent-lauf: schnitt vor einer assistant-message, nie vor einem tool-result."""
    active = [user(1, "mach mal"),
              asst_call(2, "c1", "read", {"path": "a"}), result(3, "c1", "x" * 8000),
              asst_call(4, "c2", "read", {"path": "b"}), result(5, "c2", "y" * 400)]
    cut = compact.find_cut(active, keep=300)
    assert active[cut]["role"] == "assistant"
    assert cut == 3


def test_find_cut_nothing_fits():
    active = [user(1, "x" * 10000)]
    assert compact.find_cut(active, keep=100) == 1  # alles zusammenfassen


def test_budgets_scale_for_small_windows():
    assert compact.budgets(200000) == (16384, 20000)
    assert compact.budgets(32000) == (8000, 8000)


def test_serialize_truncates_and_drops_reasoning():
    raw = [user(1, "bitte fixen"),
           msg(2, "assistant", ("reasoning", {"text": "GEHEIMES DENKEN"}), ("text", {"text": "les mal"}),
               ("tool_call", {"id": "c1", "name": "read", "arguments": '{"path": "a.py"}'})),
           result(3, "c1", "z" * 10000), result(4, "c2", "broken", ok=False)]
    s = compact.serialize(raw)
    assert "[User]: bitte fixen" in s and "[Assistant]: les mal" in s
    assert '[Assistant tool call]: read {"path": "a.py"}' in s
    assert "GEHEIMES" not in s
    assert "chars cut" in s and len(s) < 3000
    assert "[Tool result (error)]: broken" in s


def test_file_lists(tmp_path):
    raw = [asst_call(1, "a", "read", {"path": "x.py"}), result(2, "a", "..."),
           asst_call(3, "b", "edit", {"path": str(tmp_path / "x.py")}), result(4, "b", "diff"),
           asst_call(5, "c", "read", {"path": "y.py"}), result(6, "c", "..."),
           asst_call(7, "d", "write", {"path": "z.py"}), result(8, "d", "fehler", ok=False)]
    read, modified = compact.file_lists(raw, tmp_path, prev_read=["old.py"])
    assert modified == ["x.py"]          # absolut → relativ, changed beats gelesen
    assert read == ["old.py", "y.py"]    # z.py: write ist fehlgeschlagen → zaehlt nicht


def test_is_overflow():
    assert compact.is_overflow(llm.LLMError("http 400", status=400, body=OVERFLOW))
    assert compact.is_overflow(llm.LLMError("ContextWindowExceededError: prompt is too long"))
    # mistral
    assert compact.is_overflow(llm.LLMError("http 400", status=400, body='{"type":"invalid_request_prompt_too_long",'
                                            '"message":"Prompt 264620 > 262144 maximum context length"}'))
    assert not compact.is_overflow(llm.LLMError("http 401: invalid api key"))


def test_render():
    out = compact.render("## Goal\n- x", ["a.py"], ["b.py"])
    assert out.startswith("<conversation_summary>\n## Goal")
    assert "<read-files>\na.py\n</read-files>" in out and "<modified-files>\nb.py\n</modified-files>" in out


# ---------------------------------------------------------------- agent + db

def _session_with_history(backend, cfg, tmp_path, lines=1500):
    """zwei abgeschlossene turns, der erste mit grossem tool-output."""
    (tmp_path / "big.txt").write_text(BIG(lines))
    backend.script = [
        {"tool_calls": [{"name": "read", "arguments": {"path": "big.txt"}}]},
        {"text": "gelesen, ist gross"},
        {"text": "zweite antwort"},
    ]
    agent = Agent(cfg, DB(cfg.db_path), tmp_path)
    agent.run("lies big.txt")
    agent.run("und jetzt?")
    backend.requests.clear()
    return agent


def test_manual_compact_roundtrip(backend, make_cfg, tmp_path):
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ")
    agent = _session_with_history(backend, cfg, tmp_path)
    backend.by_model["summ"] = [{"text": SUMMARY}]
    before = agent.estimated_tokens()  # fake-usage ist erfunden, daher schaetzung vergleichen
    events = []
    data = agent.compact(events.append)

    assert [e["type"] for e in events if e["type"].startswith("compact_")][0] == "compact_start"
    done = next(e for e in events if e["type"] == "compact_done")
    assert done["after"] < before / 4
    assert data["read_files"] == ["big.txt"]

    # summary-request: kein tools-feld, gekuerzter tool-output, richtiges modell
    req = backend.requests[0]["body"]
    assert req["model"] == "summ" and "tools" not in req
    assert len(req["messages"][1]["content"]) < 10000

    # in-memory und aus der db identisch: summary vorne, letzter turn woertlich
    reloaded = Agent(cfg, DB(cfg.db_path), tmp_path, session_id=agent.session_id)
    assert reloaded.messages == agent.messages
    assert agent.messages[0]["role"] == "user" and SUMMARY in agent.messages[0]["content"]
    assert agent.messages[1] == {"role": "user", "content": "und jetzt?"}
    assert "zeile 1000 " not in json.dumps(agent.messages)

    # danach normal weiter
    backend.script = [{"text": "weiter gehts"}]
    assert agent.run("und weiter") == "weiter gehts"
    sent = backend.requests[-1]["body"]["messages"]
    assert SUMMARY in sent[1]["content"]


def test_second_compaction_updates_previous_summary(backend, make_cfg, tmp_path):
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ")
    agent = _session_with_history(backend, cfg, tmp_path)
    backend.by_model["summ"] = [{"text": SUMMARY}, {"text": SUMMARY + "\n- runde 2"}]
    agent.compact()
    backend.script = [{"tool_calls": [{"name": "read", "arguments": {"path": "big.txt"}}]}, {"text": "nochmal"}]
    agent.run("lies nochmal")
    agent.run("letzte frage")
    agent.compact()
    prompt = backend.requests[-1]["body"]["messages"][1]["content"]
    assert "<previous_summary>" in prompt and "- calc.py fixen" in prompt
    assert "Update the previous summary" in prompt
    assert "runde 2" in agent.messages[0]["content"]
    # kein exists twiceer summary-block, aeltere compaction ist ueberholt
    assert sum("<conversation_summary>" in str(m.get("content")) for m in agent.messages) == 1


def test_length_finish_discards_summary(backend, make_cfg, tmp_path):
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ")
    agent = _session_with_history(backend, cfg, tmp_path)
    backend.by_model["summ"] = [{"text": "## Goal\n- halb", "finish": "length"}]
    msgs_before = list(agent.messages)
    with pytest.raises(CompactError, match="length"):
        agent.compact()
    assert agent.messages == msgs_before
    assert agent.db.load_active(agent.session_id)[0] is None


def test_nothing_to_compact(backend, make_cfg, tmp_path):
    cfg = make_cfg()
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    with pytest.raises(CompactError, match="session"):
        agent.compact()
    backend.script = [{"text": "kurz"}]
    agent.run("hi")
    with pytest.raises(CompactError, match="nothing"):
        agent.compact(reason="auto")


def test_auto_compaction_before_request(backend, make_cfg, tmp_path):
    """usage meldet fast vollen kontext → vor dem naechsten request wird compacted."""
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ", context_limit=100000)
    (tmp_path / "big.txt").write_text(BIG(1500))
    backend.script = [
        {"tool_calls": [{"name": "read", "arguments": {"path": "big.txt"}}], "usage": {"prompt_tokens": 90000}},
        {"text": "fertig"},
    ]
    backend.by_model["summ"] = [{"text": SUMMARY}]
    agent = Agent(cfg, DB(cfg.db_path), tmp_path)
    events = []
    assert agent.run("lies big.txt", events.append) == "fertig"
    types = [e["type"] for e in events]
    assert "compact_start" in types
    start = next(e for e in events if e["type"] == "compact_start")
    assert start["reason"] == "auto"
    # compaction lief nach dem tool-result und vor dem zweiten call
    assert types.index("tool_end") < types.index("compact_start") < types.index("compact_done")
    assert types.index("compact_done") < len(types) - 1 - types[::-1].index("step")
    last = backend.requests[-1]["body"]
    assert last["model"] == "fake-model" and SUMMARY in last["messages"][1]["content"]
    assert "zeile 1000 " not in json.dumps(last["messages"])


def test_auto_compaction_respects_switch(backend, make_cfg, tmp_path):
    cfg = make_cfg(context_limit=100000, auto_compact=False)
    backend.script = [{"text": "a", "usage": {"prompt_tokens": 99000}}, {"text": "b"}]
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    agent.run("eins")
    agent.run("zwei")
    assert all(r["body"].get("model") == "fake-model" for r in backend.requests)
    assert len(backend.requests) == 2


def test_failed_auto_compaction_warns_once_and_continues(backend, make_cfg, tmp_path):
    cfg = make_cfg(context_limit=100000, compact_model="summ")
    backend.script = [
        {"tool_calls": [{"name": "read", "arguments": {"path": "."}}], "usage": {"prompt_tokens": 99000}},
        {"tool_calls": [{"name": "read", "arguments": {"path": "."}}], "usage": {"prompt_tokens": 99000}},
        {"text": "durch"},
    ]
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    events = []
    assert agent.run("x", events.append) == "durch"  # verlauf winzig → nothing to compact
    warnings = [e for e in events if e["type"] == "warning"]
    assert len(warnings) == 1 and "nothing to compact" in warnings[0]["warning"]
    assert not any(e["type"] == "error" for e in events)


def test_overflow_compacts_and_retries_once(backend, make_cfg, tmp_path):
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ")
    agent = _session_with_history(backend, cfg, tmp_path)
    backend.script = [{"status": 400, "text": OVERFLOW}, {"text": "nach compaction geklappt"}]
    backend.by_model["summ"] = [{"text": SUMMARY}]
    events = []
    assert agent.run("noch was", events.append) == "nach compaction geklappt"
    assert next(e for e in events if e["type"] == "compact_start")["reason"] == "overflow"
    assert not any(e["type"] == "error" for e in events)


def test_overflow_twice_gives_up(backend, make_cfg, tmp_path):
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ")
    agent = _session_with_history(backend, cfg, tmp_path)
    backend.script = [{"status": 400, "text": OVERFLOW}, {"status": 400, "text": OVERFLOW}]
    backend.by_model["summ"] = [{"text": SUMMARY}]
    events = []
    agent.run("noch was", events.append)
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1 and "context_length_exceeded" in errors[0]["error"]
    assert sum(e["type"] == "compact_start" for e in events) == 1


def test_compaction_survives_cancelled_tool_batch(backend, make_cfg, tmp_path):
    """repair() + cut: abgebrochene tool-calls im behaltenen schwanz bleiben gueltig."""
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ")
    agent = _session_with_history(backend, cfg, tmp_path)
    agent.db.add_message(agent.session_id, "user", "build", [("text", {"text": "abbruch-turn"})])
    agent.db.add_message(agent.session_id, "assistant", "build",
                         [("tool_call", {"id": "dangling", "name": "bash", "arguments": "{}"})])
    backend.by_model["summ"] = [{"text": SUMMARY}]
    agent.messages = agent.db.load_messages(agent.session_id)
    agent.compact()
    calls = {c["id"] for m in agent.messages for c in m.get("tool_calls") or []}
    results = {m["tool_call_id"] for m in agent.messages if m["role"] == "tool"}
    assert calls <= results


def test_overflow_with_oversized_current_turn_summarizes_everything(backend, make_cfg, tmp_path):
    """passt schon der aktuelle prompt nicht ins behalten-budget, wird alles zusammengefasst.
    der prompt geht dabei komplett ans summary-modell und danach nur noch als summary weiter."""
    cfg = make_cfg(compact_model="summ", context_limit=8000, auto_compact=False)  # keep = 2000 tokens
    prompt = "bitte umsetzen: " + "x" * 12000 + " ENDE-DES-PROMPTS"
    backend.script = [{"status": 400, "text": OVERFLOW}, {"text": "erledigt"}]
    backend.by_model["summ"] = [{"text": SUMMARY}]
    agent = Agent(cfg, DB(cfg.db_path), tmp_path)
    events = []
    assert agent.run(prompt, events.append) == "erledigt"
    done = next(e for e in events if e["type"] == "compact_done")
    assert done["kept"] == 0 and done["summarized"] == 1
    summ_req = next(r["body"] for r in backend.requests if r["body"]["model"] == "summ")
    assert "ENDE-DES-PROMPTS" in summ_req["messages"][1]["content"]  # nichts verloren
    last = backend.requests[-1]["body"]["messages"]
    assert [m["role"] for m in last] == ["system", "user"] and SUMMARY in last[1]["content"]
    assert "ENDE-DES-PROMPTS" not in json.dumps(last)


def test_compaction_passes_session(backend, make_cfg, write_plugin, tmp_path):
    """auch der summary-request kennt die session (z.b. for prompt-caching per plugin)."""
    plugin = write_plugin("""
        def request(backend, headers, payload, session):
            headers["X-Session"] = session
    """)
    cfg = make_cfg(plugin=plugin, compact_model="summ", context_limit=8000, auto_compact=False)
    backend.script = [{"status": 400, "text": OVERFLOW}, {"text": "erledigt"}]
    backend.by_model["summ"] = [{"text": SUMMARY}]
    agent = Agent(cfg, DB(cfg.db_path), tmp_path)
    agent.run("bitte umsetzen: " + "x" * 12000, lambda ev: None)
    summ = next(r for r in backend.requests if r["body"]["model"] == "summ")
    assert summ["headers"]["X-Session"] == agent.session_id
