import os
import re
import threading
import time
from pathlib import Path

from pyrate import tools
from pyrate.tools import ToolContext


def ctx(tmp_path, **kw):
    return ToolContext(cwd=str(tmp_path), **kw)


def test_read_offset_limit(tmp_path):
    (tmp_path / "f.txt").write_text("\n".join(f"zeile {i}" for i in range(1, 11)))
    out = tools.t_read(ctx(tmp_path), "f.txt", offset=3, limit=2)
    assert "     3\tzeile 3" in out and "     4\tzeile 4" in out
    assert "zeile 5" not in out
    assert out.endswith("[showing lines 3-4 of 10; use offset=5 to continue]")


def test_read_head_by_lines(tmp_path):
    (tmp_path / "f.txt").write_text("".join(f"z{i}\n" for i in range(1, 5001)))
    out = tools.t_read(ctx(tmp_path), "f.txt")
    assert out.endswith("[showing lines 1-2000 of 5000; use offset=2001 to continue]")
    out = tools.t_read(ctx(tmp_path), "f.txt", offset=4001, limit=99999)  # limit via max → gekappt
    assert "  5000\tz5000" in out and "showing" not in out


def test_read_head_by_bytes_and_continue(tmp_path):
    """footer-offset muss exakt passen: weiterlesen ergibt seamless die ganze datei."""
    lines = [f"zeile {i} " + "x" * 90 for i in range(1, 1001)]
    (tmp_path / "f.txt").write_text("\n".join(lines) + "\n")
    c = ctx(tmp_path)
    seen, offset, calls = [], 1, 0
    while True:
        out = tools.t_read(c, "f.txt", offset=offset)
        calls += 1
        body, _, footer = out.partition("\n\n[showing")
        assert len(body.encode()) <= tools.MAX_BYTES
        seen += [ln.split("\t", 1)[1] for ln in body.splitlines()]
        if not footer:
            break
        offset = int(re.search(r"use offset=(\d+)", footer).group(1))
    assert seen == lines and calls == 3  # ~100kb in 50kb-small steps


def test_read_offset_past_eof(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\n")
    ok, out = tools.run("read", {"path": "f.txt", "offset": 9}, ctx(tmp_path), ["read"])
    assert not ok and "past the end of the file (2 lines)" in out


def test_read_big_dir_listing(tmp_path):
    for i in range(30):
        (tmp_path / f"f{i:02}").touch()
    out = tools.t_read(ctx(tmp_path, max_lines=10), ".")
    assert out.splitlines()[-1] == "[20 more entries not shown]"


def test_read_dir_and_binary(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.bin").write_bytes(b"\0\1\2")
    assert "sub/" in tools.t_read(ctx(tmp_path), ".")
    ok, out = tools.run("read", {"path": "b.bin"}, ctx(tmp_path), ["read"])
    assert not ok and "binary" in out


def test_write_creates_dirs_and_diffs(tmp_path):
    c = ctx(tmp_path)
    assert "created" in tools.t_write(c, "a/b/c.txt", "x\n")
    assert "+y" in tools.t_write(c, "a/b/c.txt", "y\n")


def test_bash_exit_and_output(tmp_path):
    out = tools.t_bash(ctx(tmp_path), "echo hallo; echo fehler >&2; exit 3")
    assert "hallo" in out and "fehler" in out and "[exit 3]" in out


def test_bash_timeout_kills_group(tmp_path):
    t = time.monotonic()
    out = tools.t_bash(ctx(tmp_path), "sleep 30 & sleep 30", timeout=1)
    assert time.monotonic() - t < 5
    assert "timeout" in out


HINT = re.compile(r"^\[showing lines (\d+)-(\d+) of (\d+)(?:, [^;]+)?; full output: ([^,\]]+)(?:, [^\]]+)?\]$", re.M)


def test_bash_short_output_untouched(tmp_path):
    out = tools.t_bash(ctx(tmp_path), "seq 1 100")
    assert "showing lines" not in out
    assert out.splitlines()[0] == "1" and out.splitlines()[-2] == "100"
    assert not (tmp_path / "xdg-data" / "pyrate" / "tool-output").exists()  # nichts auf platte


def test_bash_tail_by_lines_full_text_on_disk(tmp_path):
    """too many zeilen (aber wenig bytes): schwanz zeigen, volltext liegt komplett auf platte."""
    out = tools.t_bash(ctx(tmp_path, max_lines=50), "seq 1 5000")
    m = HINT.search(out)
    assert m, out[:300]
    assert m.group(1, 2, 3) == ("4951", "5000", "5000")
    body = out.splitlines()
    assert body[1] == "4951" and body[-2] == "5000" and body[-1] == "[exit 0]"
    full = Path(m.group(4))
    assert full.read_text() == "".join(f"{i}\n" for i in range(1, 5001))
    assert full.parent == tmp_path / "xdg-data" / "pyrate" / "tool-output"
    assert oct(full.stat().st_mode & 0o777) == "0o600"
    assert oct(full.parent.stat().st_mode & 0o777) == "0o700"


def test_bash_tail_by_bytes_streams_to_disk(tmp_path):
    """viele bytes: speicher bleibt klein (schwanz), volltext wird beim lesen auf platte gestreamt."""
    out = tools.t_bash(ctx(tmp_path, max_bytes=2000), "seq 1 200000")
    m = HINT.search(out)
    first, last, n = map(int, m.group(1, 2, 3))
    assert last == n == 200000 and first > 199000
    shown = out.splitlines()[1:-1]
    assert shown[0] == str(first)  # keine angeschnittene erste zeile
    assert len("\n".join(shown).encode()) <= 2000
    assert Path(m.group(4)).read_text().count("\n") == 200000


def test_bash_giant_single_line(tmp_path):
    out = tools.t_bash(ctx(tmp_path, max_bytes=1000), "head -c 50000 /dev/zero | tr '\\0' x; echo ENDE")
    m = HINT.search(out)
    assert m and "last line clipped at the front" in out
    assert out.splitlines()[1].endswith("xENDE")
    assert len(out) < 1300


def test_bash_spool_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "SPOOL_CAP", 10000)
    out = tools.t_bash(ctx(tmp_path, max_bytes=1000), "seq 1 100000")
    assert ", truncated after 10000 bytes]" in out
    assert Path(HINT.search(out).group(4)).stat().st_size == 10000
    assert "100000" in out  # schwanz stimmt trotzdem


def test_bash_unwritable_output_dir(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir(mode=0o500)
    out = tools.t_bash(ctx(tmp_path, max_lines=10, output_dir=ro / "sub"), "seq 1 100")
    assert "full text not saved" in out and out.splitlines()[-2] == "100"


def test_cleanup_output(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    old, new = d / "20260101-000000-bash-00000001.txt", d / "20260101-000000-bash-00000002.txt"
    old.write_text("a")
    new.write_text("b")
    eight_days = time.time() - 8 * 86400
    os.utime(old, (eight_days, eight_days))
    assert tools.cleanup_output(d) == 1
    assert not old.exists() and new.exists()
    assert tools.cleanup_output(tmp_path / "gibtsnicht") == 0


def test_bash_cancel(tmp_path):
    ev = threading.Event()
    threading.Timer(0.3, ev.set).start()
    out = tools.t_bash(ctx(tmp_path, cancel=ev), "sleep 30")
    assert "cancelled" in out


def test_plan_mode_filter():
    assert tools.for_mode({}, "build") == ["read", "write", "edit", "bash", "ask_user", "grep", "find"]
    assert tools.for_mode({}, "plan") == ["read", "ask_user", "grep", "find"]
    assert tools.for_mode({"grep": False, "find": False, "bash": False}, "plan") == ["read", "ask_user"]


def test_run_guards(tmp_path):
    c = ctx(tmp_path)
    assert tools.run("rm_rf", {}, c, ["read"]) == (False, "unknown tool 'rm_rf'")
    ok, out = tools.run("write", {"path": "x", "content": "y"}, c, ["read"])
    assert not ok and "not allowed" in out
    assert not (tmp_path / "x").exists()


def test_run_unknown_arguments_are_an_error(tmp_path):
    """a typo in the arg name (paht) must not be silently dropped, but returned as an error to the model."""
    c = ctx(tmp_path)
    ok, out = tools.run("write", {"paht": "x.txt", "content": "y"}, c, ["write"])
    assert not ok and "unknown arguments: paht" in out and "allowed: path, content" in out
    assert not (tmp_path / "x.txt").exists()
    # known args pass through normally
    ok, out = tools.run("write", {"path": "x.txt", "content": "y"}, c, ["write"])
    assert ok and (tmp_path / "x.txt").read_text() == "y"


def test_ask_user(tmp_path):
    assert "no human" in tools.t_ask_user(ctx(tmp_path), "?")
    c = ctx(tmp_path, ask_user=lambda q, o: f"{q}->{o[0]}")
    assert tools.t_ask_user(c, "frage", ["ja"]) == "frage->ja"


def test_grep_find(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.py").write_text("def foo(): pass\n")
    out = tools.t_grep(ctx(tmp_path), r"def \w+", include="*.py")
    assert "a.py:1" in out and ".git" not in out
    assert tools.t_find(ctx(tmp_path), "*.py").endswith("a.py")


def test_find_glob_semantics(tmp_path):
    """** = beliebig tief inkl. null, * bleibt im segment, ohne '/' zaehlt der dateiname."""
    for rel in ("calc.py", "pkg/mod.py", "pkg/sub/deep.py", "pkg/sub/notes.txt", "docs/a.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()

    def found(pattern):
        out = tools.t_find(ctx(tmp_path), pattern)
        return sorted(Path(p).relative_to(tmp_path).as_posix() for p in out.splitlines() if p != "no matches")

    every_py = ["calc.py", "pkg/mod.py", "pkg/sub/deep.py"]
    assert found("**/*.py") == every_py          # top-level gehoert dazu
    assert found("*.py") == every_py             # dateiname in jeder tiefe
    assert found("./**/*.py") == every_py
    assert found("pkg/*.py") == ["pkg/mod.py"]   # * geht nicht via /
    assert found("pkg/**/*.py") == ["pkg/mod.py", "pkg/sub/deep.py"]
    assert found("pkg/**") == ["pkg/mod.py", "pkg/sub/deep.py", "pkg/sub/notes.txt"]
    assert found("**/sub/*.txt") == ["pkg/sub/notes.txt"]
    assert found("[cd]*.py") == ["calc.py", "pkg/sub/deep.py"]
    assert found("[!cd]*.py") == ["pkg/mod.py"]
    assert found("calc.p?") == ["calc.py"]
    assert found("*.rs") == []
    # grep-include mit derselben semantik
    (tmp_path / "calc.py").write_text("treffer\n")
    (tmp_path / "pkg/mod.py").write_text("treffer\n")
    out = tools.t_grep(ctx(tmp_path), "treffer", include="**/*.py")
    assert "calc.py:1" in out and "mod.py:1" in out
    assert "calc.py" not in tools.t_grep(ctx(tmp_path), "treffer", include="pkg/*.py")


def test_grep_rejects_huge_regex(tmp_path):
    ok, out = tools.run("grep", {"pattern": "a" * (tools.MAX_REGEX + 1)}, ctx(tmp_path), ["grep"])
    assert not ok and "regex too long" in out


def test_read_streams_and_counts(tmp_path):
    """read laedt nicht die ganze datei, zaehlt aber trotzdem korrekt – auch ohne newline am ende."""
    (tmp_path / "f.txt").write_text("".join(f"z{i}\n" for i in range(1, 100001)) + "letzte")
    out = tools.t_read(ctx(tmp_path), "f.txt", offset=5, limit=2)
    assert out.endswith("[showing lines 5-6 of 100001; use offset=7 to continue]")
    assert tools.t_read(ctx(tmp_path), "f.txt", offset=100001).strip() == "100001\tletzte"
    (tmp_path / "crlf.txt").write_bytes(b"a\r\nb\r\n")
    assert tools.t_read(ctx(tmp_path), "crlf.txt") == "     1\ta\n     2\tb"
    (tmp_path / "leer.txt").write_text("")
    assert tools.t_read(ctx(tmp_path), "leer.txt") == "(empty file)"
    (tmp_path / "nl.txt").write_text("\n")
    assert tools.t_read(ctx(tmp_path), "nl.txt") == "     1\t"


def test_grep_caps_lines_and_bytes(tmp_path):
    (tmp_path / "long.txt").write_text("treffer " + "y" * 2000 + "\n")
    out = tools.t_grep(ctx(tmp_path), "treffer")
    assert out.endswith("y [...]") and len(out.split(": ", 1)[1]) == tools.GREP_LINE + len(" [...]")
    (tmp_path / "many.txt").write_text("".join(f"treffer {i} " + "z" * 100 + "\n" for i in range(150)))
    out = tools.t_grep(ctx(tmp_path, max_bytes=3000), "treffer", include="many.txt")
    assert out.splitlines()[-1].startswith("[cut off after ") and len(out.encode()) < 3200


def test_run_safety_net_only_for_non_truncating_tools(tmp_path):
    """read/bash/grep kuerzen selbst – kein zusaetzlicher mittelschnitt, der den footer zerstoert."""
    (tmp_path / "f.txt").write_text("".join(f"zeile {i} " + "x" * 50 + "\n" for i in range(3000)))
    c = ctx(tmp_path, max_bytes=20000)
    ok, out = tools.run("read", {"path": "f.txt"}, c, ["read"])
    assert ok and "chars cut" not in out and "use offset=" in out
    ok, out = tools.run("write", {"path": "g.txt", "content": "a" * 50000}, c, ["write"])
    ok, out = tools.run("write", {"path": "g.txt", "content": "b" * 50000}, c, ["write"])
    assert ok and "chars cut" in out  # diff: sicherheitsnetz greift


def test_schemas_valid():
    for s in tools.schemas(tools.TOOLS):
        params = s["function"]["parameters"]
        assert set(params["required"]) <= set(params["properties"])
        assert all("optional" not in p for p in params["properties"].values())


def test_cleanup_only_touches_spool_files(tmp_path):
    """befund: ohne namensfilter loescht cleanup_output jede alte datei im verzeichnis."""
    d = tmp_path / "documents"
    d.mkdir()
    old = time.time() - 30 * 86400
    keep = [d / "steuer-2019.pdf", d / "notes.txt", d / "20260101-120000-bash-deadbeef.txt.bak",
            d / "20260101-120000-bash-xyz.txt"]
    spool = [d / "20260101-120000-bash-deadbeef.txt", d / "20260101-120000-mcp-0123abcd.txt"]
    for f in keep + spool:
        f.write_text("x")
        os.utime(f, (old, old))
    (d / "unterordner").mkdir()
    assert tools.cleanup_output(d) == 2
    assert all(f.exists() for f in keep) and not any(f.exists() for f in spool)
    # und was _spool_open erzeugt, passt auch wirklich aufs schema
    path, fh = tools._spool_open(ctx(tmp_path, output_dir=d), "bash")
    fh.close()
    os.utime(path, (old, old))
    assert tools.cleanup_output(d) == 1 and not path.exists()


def test_bash_reader_thread_dead_before_fd_close(tmp_path, monkeypatch):
    """befund: enkel mit eigener session haelt das write-ende offen → lese-thread hing noch in
    os.read, while der fd schon geschlossen (und wiederverwendbar) war."""
    seen = []
    orig = tools._Capture

    class Spy(orig):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen.append(self)

    monkeypatch.setattr(tools, "_Capture", Spy)
    t = time.monotonic()
    out = tools.t_bash(ctx(tmp_path), "echo vorher; setsid sleep 5 >&1 2>&1 & echo nachher")
    took = time.monotonic() - t
    assert "vorher" in out and "nachher" in out and "[exit 0]" in out
    assert not seen[0].thread.is_alive(), "lese-thread lebt nach dem schliessen des fd"
    assert "background process holds stdout/stderr open" in out
    assert took < 2.5
    # normaler befehl: kein hinweis, keine wartezeit
    t = time.monotonic()
    out = tools.t_bash(ctx(tmp_path), "echo x")
    assert "background process" not in out and time.monotonic() - t < 0.4


def test_read_did_you_mean(tmp_path):
    (tmp_path / "pyrate").mkdir()
    (tmp_path / "pyrate" / "tools.py").write_text("x")
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "tools.py").write_text("x")

    def err(path):
        ok, out = tools.run("read", {"path": path}, ctx(tmp_path), ["read"])
        assert not ok and "does not exist" in out
        return out

    assert "did you mean: pyrate/tools.py" in err("pyrate/tool.py")   # tippfehler
    assert "did you mean: README.md" in err("readme.md")              # gross/klein
    out = err("src/lib/tools.py")                                    # falsches verzeichnis
    assert "pyrate/tools.py" in out and "node_modules" not in out
    assert "did you mean" not in err("gibtsnicht_xyz.txt")


def test_edit_did_you_mean(tmp_path):
    (tmp_path / "config.yaml").write_text("a: 1\n")
    ok, out = tools.run("edit", {"path": "config.yml", "old_string": "a", "new_string": "b"},
                        ctx(tmp_path), ["edit"])
    assert not ok and "did you mean: config.yaml" in out and "'write'" in out
