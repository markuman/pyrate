import os
import subprocess
import sys

from pyrate.db import DB, repair


def test_migration_sets_version(tmp_path):
    d = DB(tmp_path / "x.db")
    assert d.conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert d.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    d.close()
    DB(tmp_path / "x.db").close()  # zweites oeffnen migriert nicht nochmal


def test_roundtrip(db):
    sid = db.create_session("/p", "m")
    db.add_message(sid, "user", "build", [("text", {"text": "hallo pyrate"})])
    db.add_message(sid, "assistant", "build", [
        ("reasoning", {"text": "hm", "signature": "sig"}),
        ("text", {"text": "les mal"}),
        ("tool_call", {"id": "c1", "name": "read", "arguments": '{"path": "x"}'}),
    ])
    db.add_message(sid, "tool", "build", [("tool_result", {"tool_call_id": "c1", "name": "read",
                                                           "content": "inhalt", "ok": True})])
    msgs = db.load_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1]["tool_calls"][0]["function"]["name"] == "read"
    assert msgs[1]["thinking_blocks"][0]["signature"] == "sig"
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "inhalt"}


def test_latest_prefix_search(db):
    a = db.create_session("/p", "m")
    b = db.create_session("/p", "m")
    db.add_message(b, "user", "plan", [("text", {"text": "terraform state broken"})])
    assert db.latest_session("/p")["id"] == b
    assert db.get_session(b[:6])["id"] == b
    hits = db.search("terraform")
    assert [h["id"] for h in hits] == [b]
    assert "[terraform]" in hits[0]["snip"]
    assert db.search('"; DROP TABLE') == []
    db.delete_session(a)
    assert db.get_session(a) is None


def test_repair_dangling_tool_calls():
    msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "user", "content": "weiter"},
    ]
    out = repair(msgs)
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["a", "b"]
    assert out[-1]["content"] == "weiter"


def test_prefix_lookup_rejects_wildcards(db):
    sid = db.create_session("/p", "m")
    assert db.get_session(sid[:4])["id"] == sid
    assert db.get_session("%") is None and db.get_session("_") is None and db.get_session("") is None


def test_empty_xdg_data_home_falls_back_to_home(tmp_path):
    """XDG_DATA_HOME= zaehlt als nicht gesetzt – sonst laege die db relativ im cwd (= im repo)."""
    env = {**os.environ, "XDG_DATA_HOME": "", "HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", "from pyrate import db; print(db.DEFAULT_PATH)"],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(tmp_path / ".local" / "share" / "pyrate" / "pyrate.db")
