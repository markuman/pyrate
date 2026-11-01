"""sessions in sqlite3: sessions, messages, parts + fts5 search. migrations via user_version."""

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .images import image_parts

# `or` instead of a default argument: an empty XDG_DATA_HOME counts as unset (else the db lands relatively in the cwd)
DEFAULT_PATH = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "pyrate" / "pyrate.db"

MIGRATIONS = [
    # v1
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        cwd TEXT NOT NULL,
        model TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE INDEX sessions_cwd ON sessions(cwd, updated_at);
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        mode TEXT,
        created_at REAL NOT NULL
    );
    CREATE INDEX messages_session ON messages(session_id, id);
    CREATE TABLE parts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
        type TEXT NOT NULL,
        data TEXT NOT NULL
    );
    CREATE INDEX parts_message ON parts(message_id, id);
    CREATE VIRTUAL TABLE search USING fts5(text, session_id UNINDEXED, part_id UNINDEXED);
    """,
    # v2: subagent sessions hang off their parent, don't show up in -c/--sessions
    """
    ALTER TABLE sessions ADD COLUMN parent_id TEXT;
    """,
]


class DB:
    def __init__(self, path=None):
        self.path = Path(path) if path else DEFAULT_PATH
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def _migrate(self):
        with self.lock:
            version = self.conn.execute("PRAGMA user_version").fetchone()[0]
            for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
                self.conn.executescript(f"BEGIN;\n{sql}\nPRAGMA user_version={i};\nCOMMIT;")

    def close(self):
        with self.lock:
            self.conn.close()

    # ------------------------------------------------------------ sessions

    def create_session(self, cwd, model, parent_id=None):
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        with self.lock:
            self.conn.execute(
                "INSERT INTO sessions (id, title, cwd, model, created_at, updated_at, parent_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (sid, None, str(cwd), model, now, now, parent_id),
            )
        return sid

    def get_session(self, sid):
        """exact id or unambiguous prefix."""
        with self.lock:
            row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
            if row:
                return dict(row)
            if not re.fullmatch(r"[0-9a-f]+", sid or ""):  # ids are hex; also guards against %/_ in LIKE
                return None
            rows = self.conn.execute("SELECT * FROM sessions WHERE id LIKE ? LIMIT 2", (sid + "%",)).fetchall()
        return dict(rows[0]) if len(rows) == 1 else None

    def latest_session(self, cwd):
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM sessions WHERE cwd = ? AND parent_id IS NULL ORDER BY updated_at DESC LIMIT 1",
                (str(cwd),),
            ).fetchone()
        return dict(row) if row else None

    def list_sessions(self, cwd=None, limit=20):
        """only top-level sessions, subagent runs stay out (still reachable via -s ID)."""
        q = "SELECT * FROM sessions WHERE parent_id IS NULL"
        args = []
        if cwd:
            q += " AND cwd = ?"
            args.append(str(cwd))
        q += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [dict(r) for r in self.conn.execute(q, args)]

    def set_title(self, sid, title):
        with self.lock:
            self.conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, sid))

    def set_model(self, sid, model):
        with self.lock:
            self.conn.execute("UPDATE sessions SET model = ? WHERE id = ?", (model, sid))

    def delete_session(self, sid):
        with self.lock:
            self.conn.execute("BEGIN")
            self.conn.execute("DELETE FROM search WHERE session_id = ?", (sid,))
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            self.conn.execute("COMMIT")

    # ------------------------------------------------------------ messages

    def add_message(self, sid, role, mode, parts):
        """parts: list of (type, data-dict). returns the message id."""
        now = time.time()
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                cur = self.conn.execute(
                    "INSERT INTO messages (session_id, role, mode, created_at) VALUES (?,?,?,?)",
                    (sid, role, mode, now),
                )
                mid = cur.lastrowid
                for ptype, data in parts:
                    pcur = self.conn.execute(
                        "INSERT INTO parts (message_id, type, data) VALUES (?,?,?)",
                        (mid, ptype, json.dumps(data, ensure_ascii=False)),
                    )
                    if ptype == "text" and data.get("text"):
                        self.conn.execute(
                            "INSERT INTO search (text, session_id, part_id) VALUES (?,?,?)",
                            (data["text"], sid, pcur.lastrowid),
                        )
                self.conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, sid))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return mid

    def load_raw(self, sid):
        """[{id, role, mode, parts: [{type, data}]}] in order, complete history."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT m.id, m.role, m.mode, p.type, p.data FROM messages m "
                "LEFT JOIN parts p ON p.message_id = m.id WHERE m.session_id = ? ORDER BY m.id, p.id",
                (sid,),
            ).fetchall()
        out = []
        last = None
        for r in rows:
            if r["id"] != last:
                out.append({"id": r["id"], "role": r["role"], "mode": r["mode"], "parts": []})
                last = r["id"]
            if r["type"]:
                out[-1]["parts"].append({"type": r["type"], "data": json.loads(r["data"])})
        return out

    def load_active(self, sid):
        """(compaction data | None, active messages).

        after a compaction only counts: everything from first_kept_id on (the verbatim
        tail, id-wise *before* the compaction message) plus everything after it. older
        compactions are superseded by the latest one (which continues their summary)."""
        raw = self.load_raw(sid)
        last = next((m for m in reversed(raw) if m["role"] == "compaction"), None)
        if last is None:
            return None, raw
        data = last["parts"][0]["data"]
        start = data.get("first_kept_id") or last["id"]
        return data, [m for m in raw if m["id"] >= start and m["role"] != "compaction"]

    def load_messages(self, sid):
        data, active = self.load_active(sid)
        msgs = to_openai(active)
        if data:
            msgs.insert(0, {"role": "user", "content": data["content"]})
        return msgs

    # ------------------------------------------------------------ search

    def search(self, query, limit=20):
        terms = [t for t in query.split() if t]
        if not terms:
            return []
        fts = " ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.lock:
            rows = self.conn.execute(
                "SELECT s.id, s.title, s.cwd, s.updated_at, s.parent_id, "
                "snippet(search, 0, '[', ']', '…', 12) AS snip "
                "FROM search JOIN sessions s ON s.id = search.session_id "
                "WHERE search MATCH ? ORDER BY rank LIMIT ?",
                (fts, limit * 5),
            ).fetchall()
        seen = {}
        for r in rows:
            if r["id"] not in seen:
                seen[r["id"]] = dict(r)
            if len(seen) >= limit:
                break
        return list(seen.values())


def to_openai(raw):
    msgs = []
    for m in raw:
        texts = [p["data"].get("text", "") for p in m["parts"] if p["type"] == "text"]
        if m["role"] == "user":
            imgs = [p["data"] for p in m["parts"] if p["type"] == "image"]
            if imgs:  # multipart: text + image_url parts (base64 data-url)
                joined = "\n".join(texts)
                content = ([{"type": "text", "text": joined}] if joined else []) + image_parts(imgs)
                msgs.append({"role": "user", "content": content})
            else:
                msgs.append({"role": "user", "content": "\n".join(texts)})
        elif m["role"] == "assistant":
            msg = {"role": "assistant", "content": "\n".join(texts) or None}
            calls = [p["data"] for p in m["parts"] if p["type"] == "tool_call"]
            if calls:
                msg["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c.get("arguments") or "{}"}}
                    for c in calls
                ]
            for p in m["parts"]:
                if p["type"] == "reasoning":
                    msg["reasoning_content"] = p["data"].get("text", "")
                    if p["data"].get("signature"):
                        msg["thinking_blocks"] = [{
                            "type": "thinking", "thinking": msg["reasoning_content"],
                            "signature": p["data"]["signature"],
                        }]
            msgs.append(msg)
        elif m["role"] == "tool":
            for p in m["parts"]:
                if p["type"] == "tool_result":
                    msgs.append({"role": "tool", "tool_call_id": p["data"]["tool_call_id"],
                                 "content": p["data"].get("content", "")})
    return repair(msgs)


def repair(msgs):
    """interrupted tool calls (e.g. ctrl+c mid-run) get a dummy result,
    otherwise most backends reject the next request."""
    out = []
    pending = []
    for m in msgs:
        if m["role"] != "tool" and pending:
            out.extend({"role": "tool", "tool_call_id": cid, "content": "(cancelled)"} for cid in pending)
            pending = []
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append(m)
            pending = [c["id"] for c in m["tool_calls"]]
            continue
        if m["role"] == "tool" and m["tool_call_id"] in pending:
            pending.remove(m["tool_call_id"])
        out.append(m)
    out.extend({"role": "tool", "tool_call_id": cid, "content": "(cancelled)"} for cid in pending)
    return out
