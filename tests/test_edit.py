import pytest

from pyrate.tools import ToolContext, ToolError, apply_edit, t_edit


def test_exact_single():
    assert apply_edit("a = 1\nb = 2\n", "b = 2", "b = 3") == "a = 1\nb = 3\n"


def test_multiple_matches_error_lists_lines():
    with pytest.raises(ToolError, match=r"2x .*lines 1, 3"):
        apply_edit("x\ny\nx\n", "x", "z")


def test_replace_all():
    assert apply_edit("x\ny\nx\n", "x", "z", replace_all=True) == "z\ny\nz\n"


def test_not_found_with_hint():
    with pytest.raises(ToolError, match="closest line: 2"):
        apply_edit("def foo():\n    return compute_value(1)\n", "return compute_valeu(1)", "x")


def test_trailing_whitespace_tolerant():
    src = "def f():   \n    return 1  \n"
    out = apply_edit(src, "def f():\n    return 1", "def f():\n    return 2")
    assert out == "def f():\n    return 2\n"


def test_indentation_tolerant_reindents():
    src = "class A:\n    def f(self):\n        return 1\n"
    out = apply_edit(src, "def f(self):\n    return 1", "def f(self):\n    return 2")
    assert out == "class A:\n    def f(self):\n        return 2\n"


def test_tolerant_ambiguous():
    src = "if a:\n    x = 1\nif b:\n  x = 1\n"
    with pytest.raises(ToolError, match="whitespace-tolerant 2x"):
        apply_edit(src, "x = 1 ", "x = 2")


def test_crlf_preserved():
    assert apply_edit("a\r\nb\r\n", "a\nb", "a\nc") == "a\r\nc\r\n"


def test_empty_and_identical():
    with pytest.raises(ToolError, match="empty"):
        apply_edit("a", "", "b")
    with pytest.raises(ToolError, match="identical"):
        apply_edit("a", "a", "a")


def test_t_edit_returns_diff(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a = 1\n")
    diff = t_edit(ToolContext(cwd=str(tmp_path)), "x.py", "a = 1", "a = 2")
    assert "-a = 1" in diff and "+a = 2" in diff
    assert f.read_text() == "a = 2\n"


def test_t_edit_missing_file(tmp_path):
    with pytest.raises(ToolError, match="write"):
        t_edit(ToolContext(cwd=str(tmp_path)), "nope.py", "a", "b")


# ---------------------------------------------------------------- hardening (befunde #1–#6)

def ctx(tmp_path):
    return ToolContext(cwd=str(tmp_path))


def test_t_edit_crlf_roundtrip(tmp_path):
    """#1: universal newlines haben crlf beim lesen still zu lf gemacht."""
    f = tmp_path / "win.txt"
    f.write_bytes(b"a\r\nb\r\nc\r\n")
    diff = t_edit(ctx(tmp_path), "win.txt", "b", "B")
    assert f.read_bytes() == b"a\r\nB\r\nc\r\n"
    assert "-b" in diff and "+B" in diff and "\r" not in diff


def test_edge_blank_lines_symmetric():
    """#2: leere randzeilen in old werden getrimmt, dann in new genauso viele."""
    assert apply_edit("  foo()  \nnext\n", "foo()\n", "bar()\n") == "  bar()\nnext\n"
    assert apply_edit("  foo()\nnext\n", "\nfoo()\n", "\nbar()\n") == "  bar()\nnext\n"


def test_empty_new_deletes_block():
    assert apply_edit("a\n    foo()\n    bar()\nb\n", "foo()\nbar()\n", "") == "a\nb\n"


def test_reindent_shifts_dedented_closing_brace():
    """#3: schliessende klammer ist flacher als die erste zeile und muss mitwandern."""
    src = "        foo();\n    }\n"
    assert apply_edit(src, "    foo();\n}", "    bar();\n}") == "        bar();\n    }\n"


def test_reindent_shift_left():
    src = "if x:\n  a = 1\n  b = 2\n"
    out = apply_edit(src, "    a = 1\n    b = 2", "    a = 1\n    if b:\n        b = 3")
    assert out == "if x:\n  a = 1\n  if b:\n      b = 3\n"


def test_relative_indent_mismatch_is_error():
    """#4: falsche binnen-einrueckung darf nicht still durchrutschen."""
    src = "if a:\n    x = 1\ny = 2\n"
    with pytest.raises(ToolError, match=r"lines 1–3 match in content.*relative indentation") as e:
        apply_edit(src, "if a:\nx = 1\ny = 2", "if a:\nx = 9\ny = 2")
    assert "'    x = 1'" in str(e.value) and "'x = 1'" in str(e.value)


def test_not_found_hint_names_diverging_line():
    """#5: hint muss die abweichende zeile nennen, nicht nur die erste."""
    src = "def f():\n    a = 1\n    return a\n"
    with pytest.raises(ToolError, match=r"line 1 matches the start, line 3 differs") as e:
        apply_edit(src, "def f():\n    a = 1\n    return b", "x")
    msg = str(e.value)
    assert "file:       '    return a'" in msg
    assert "old_string: '    return b'" in msg


def test_not_found_hint_past_eof():
    with pytest.raises(ToolError, match="<end of file>"):
        apply_edit("a\nb\n", "a\nb\nc", "x")


def test_noop_is_error_and_file_untouched(tmp_path):
    """#6: toleranter treffer, der nichts aendert, ist ein fehler und schreibt nicht."""
    f = tmp_path / "x.py"
    f.write_text("x = 1\n")
    mtime = f.stat().st_mtime_ns
    with pytest.raises(ToolError, match="changes nothing"):
        t_edit(ctx(tmp_path), "x.py", "x = 1 ", "x = 1")
    assert f.read_text() == "x = 1\n"
    assert f.stat().st_mtime_ns == mtime


def test_t_edit_binary_and_non_utf8(tmp_path):
    (tmp_path / "b.bin").write_bytes(b"ab\0cd")
    with pytest.raises(ToolError, match="binary file"):
        t_edit(ctx(tmp_path), "b.bin", "ab", "x")
    (tmp_path / "l1.txt").write_bytes(b"\xff\n")  # invalid utf-8
    with pytest.raises(ToolError, match="not a utf-8 text file"):
        t_edit(ctx(tmp_path), "l1.txt", "gr", "x")


def test_diff_label_relative_and_tolerant_note(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "m.py"
    f.write_text("def f():\n    return 1\n")
    diff = t_edit(ctx(tmp_path), "sub/m.py", "return 1", "return 2")
    assert "--- a/sub/m.py" in diff and "+++ b/sub/m.py" in diff
    assert "hinweis" not in diff
    diff = t_edit(ctx(tmp_path), str(f), "    return 2   ", "    return 3")
    assert diff.startswith("note:")  # old_string war nicht exakt (trailing ws)
    assert f.read_text() == "def f():\n    return 3\n"
    out = tmp_path.parent / f"{tmp_path.name}-out.py"
    out.write_text("a = 1\n")
    try:
        diff = t_edit(ctx(tmp_path), str(out), "a = 1", "a = 2")
        assert f"--- a{out}" in diff and "a//" not in diff
    finally:
        out.unlink()


def test_overlapping_pattern_count_consistent():
    """count und zeilennummern muessen dieselbe (nicht-ueberlappende) zaehlung nutzen."""
    with pytest.raises(ToolError, match=r"2x \(lines 1, 1\)"):
        apply_edit("aaaa\n", "aa", "b")
