"""the tools: read, write, edit, bash, ask_user, grep, find. path boundaries via ToolContext.sandbox."""

import difflib
import json
import os
import re
import secrets
import select
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
                ".pytest_cache", ".tox", "dist", "build", ".terraform"}
MAX_LINE = 2000          # read: single lines are capped at this many chars
GREP_LINE = 500          # grep: same, per hit
MAX_LINES = 2000         # default budget per tool result …
MAX_BYTES = 50 * 1024    # … and in utf-8 bytes
SPOOL_CAP = 100 * 1024 * 1024  # full output on disk at most this big (`yes` would run to timeout otherwise)
OUTPUT_MAX_AGE_DAYS = 7
SPOOL_NAME_RE = re.compile(r"\d{8}-\d{6}-[a-z]+-[0-9a-f]{8}\.txt")  # must match _spool_open
# bash: after SIGTERM wait at most this long for cleanup, then SIGKILL. a sub-pyrate may need
# TERM_GRACE itself for its bash + up to 6s for mcp.close(). normal commands die instantly,
# which ends the wait early.
TERM_GRACE = 10.0


class ToolError(Exception):
    pass


@dataclass
class ToolContext:
    cwd: str
    bash_timeout: int = 120
    max_lines: int = MAX_LINES
    max_bytes: int = MAX_BYTES
    output_dir: object = None  # where full texts of truncated outputs go; None = default_output_dir()
    ask_user: object = None    # callable(question, options) -> str
    cancel: object = None      # threading.Event
    env: dict | None = None    # extra env for bash (PYRATE_SESSION, PYRATE_SUBAGENTS, …)
    sandbox: object = None     # sandbox.Sandbox or None = no path boundaries
    readonly: bool = False     # plan mode: bash may only write in sandbox.scratch
    net: bool = True           # False: no tcp at all (plan mode default, --no-net)


# ---------------------------------------------------------------- full text on disk

def default_output_dir():
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "pyrate" / "tool-output"


def _output_dir(ctx):
    d = Path(ctx.output_dir) if ctx.output_dir else default_output_dir()
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def _spool_open(ctx, kind):
    """new file for a full text, readable only by the user (holds arbitrary command output)."""
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{secrets.token_hex(4)}.txt"
    path = _output_dir(ctx) / name
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return path, os.fdopen(fd, "wb")


def cleanup_output(directory=None, max_age_days=OUTPUT_MAX_AGE_DAYS):
    """delete full texts older than max_age_days. returns the number of deleted files.

    only files whose name exactly matches the _spool_open scheme – if tool_output_dir
    accidentally points at ~ or ~/Documents, everything else stays untouched."""
    d = Path(directory) if directory else default_output_dir()
    if not d.is_dir():
        return 0
    cutoff = time.time() - max_age_days * 86400
    n = 0
    for f in d.iterdir():
        if not SPOOL_NAME_RE.fullmatch(f.name):
            continue
        try:
            if f.is_file() and not f.is_symlink() and f.stat().st_mtime < cutoff:
                f.unlink()
                n += 1
        except OSError:
            continue
    return n


# ---------------------------------------------------------------- truncation

def _nbytes(s):
    return len(s.encode("utf-8", "replace"))


def head_lines(lines, max_lines, max_bytes):
    """as many lines from the top as fit both budgets (at least one)."""
    out, size = [], 0
    for line in lines[:max_lines]:
        size += _nbytes(line) + 1
        if out and size > max_bytes:
            break
        out.append(line)
    return out


def tail_lines(lines, max_lines, max_bytes):
    """same, from the end."""
    out, size = [], 0
    for line in reversed(lines[-max_lines:] if max_lines else []):
        size += _nbytes(line) + 1
        if out and size > max_bytes:
            break
        out.append(line)
    return out[::-1]


def limit_text(ctx, text, kind):
    """head-truncation for finished texts (mcp results): if it doesn't fit the budget, the
    full text goes to disk and the note above says where."""
    lines = text.split("\n")
    if len(lines) <= ctx.max_lines and _nbytes(text) <= ctx.max_bytes:
        return text
    shown = head_lines(lines, ctx.max_lines, ctx.max_bytes)
    clipped = ""
    if len(shown) == 1 and _nbytes(shown[0]) > ctx.max_bytes:
        shown = [_clip_bytes(shown[0], ctx.max_bytes)]
        clipped = ", first line clipped at the end"
    try:
        path, fh = _spool_open(ctx, kind)
        with fh:
            fh.write(text.encode("utf-8", "replace")[:SPOOL_CAP])
        where = f"full output: {path}"
    except OSError as e:
        where = f"full text not saved: {e}"
    return "\n".join(shown) + f"\n\n[showing lines 1-{len(shown)} of {len(lines)}{clipped}; {where}]"


def _clip_bytes(s, max_bytes, keep_end=False):
    """clip a single huge line to max_bytes without cutting a utf-8 char in half."""
    b = s.encode("utf-8", "replace")
    if len(b) <= max_bytes:
        return s
    b = b[-max_bytes:] if keep_end else b[:max_bytes]
    return b.decode("utf-8", "ignore")


@dataclass
class Tool:
    name: str
    description: str
    params: dict
    func: object
    readonly: bool
    default_on: bool = True
    self_truncating: bool = False  # truncates its own output (otherwise truncate() in run() applies)
    raw_schema: dict | None = None  # ready-made json schema (mcp); then func(ctx, args) instead of func(ctx, **args)

    def schema(self):
        if self.raw_schema is not None:
            return {"type": "function",
                    "function": {"name": self.name, "description": self.description,
                                 "parameters": self.raw_schema}}
        props = {k: {kk: vv for kk, vv in v.items() if kk != "optional"} for k, v in self.params.items()}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": [k for k, v in self.params.items() if not v.get("optional")],
                },
            },
        }


def _resolve(ctx, path, write=False):
    if not path:
        raise ToolError("path is missing")
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = Path(ctx.cwd) / p
    p = p.resolve()
    _check(ctx, p, write)
    return p


def _check(ctx, p, write):
    """sandbox path boundary. outside, pyrate asks the user; a grant lasts for the
    session and also applies to bash. exactly p is granted, never the directory above it:
    whoever allows writing ~/notes.txt does not mean all of ~."""
    sb = ctx.sandbox
    if sb is None or sb.allows(p, write):
        return
    what = "write access" if write else "read access"
    kind = "directory (with everything under it)" if p.is_dir() else "file"
    question = f"allow {what} to {kind} {p}? lasts for this session, also for bash"
    answer = ctx.ask_user(question, ["yes", "no"]) if ctx.ask_user else None
    if answer is None:
        raise ToolError(f"{p} is outside the sandbox and nobody answered "
                        f"(no human at the terminal or skipped), {what} stays blocked")
    if answer.strip().lower() not in ("ja", "j", "yes", "y"):
        raise ToolError(f"{what} to {p} denied by the user")
    sb.grant(p, write)


def truncate(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n\n[... {cut} chars cut ...]\n\n{text[-half:]}"


def _is_binary(p):
    with open(p, "rb") as f:
        return b"\0" in f.read(8192)


SIMILAR_MAX = 3
SIMILAR_SCAN = 5000  # at most this many entries scanned in the cwd tree (huge repos, $HOME)


def similar_paths(p, cwd, n=SIMILAR_MAX):
    """up to n existing paths the model probably meant (typo, wrong directory).
    1. in the nearest existing parent dir: similar names
    2. in the cwd tree: same file name (case-insensitive), e.g. src/x.py instead of lib/x.py"""
    name = p.name
    if not name:
        return []
    parent = p.parent
    while not parent.is_dir() and parent != parent.parent:
        parent = parent.parent
    found = []
    try:
        entries = {e.name: e for e in parent.iterdir()}
    except OSError:
        entries = {}
    lower = {k.lower(): v for k, v in entries.items()}
    if name.lower() in lower and lower[name.lower()].name != name:
        found.append(lower[name.lower()])
    found += [entries[m] for m in difflib.get_close_matches(name, list(entries), n=n, cutoff=0.6)]
    if len(found) < n:
        seen = 0
        for root, dirs, files in os.walk(cwd):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
            seen += len(dirs) + len(files)
            found += [Path(root) / f for f in sorted(files) + dirs if f.lower() == name.lower()]
            if seen > SIMILAR_SCAN or len(found) >= 2 * n:
                break
    out = []
    for f in found:
        try:
            shown = str(f.relative_to(cwd))
        except ValueError:
            shown = str(f)
        if shown not in out and Path(f) != p:
            out.append(shown)
    return out[:n]


def _hint(ctx, p):
    """only suggest "did you mean" if the parent dir is readable anyway. otherwise a grant on
    a non-existent file would list the whole directory as a side effect."""
    if ctx.sandbox is not None and not ctx.sandbox.allows(p.parent):
        return []
    return similar_paths(p, ctx.cwd)


# ---------------------------------------------------------------- read

def t_read(ctx, path, offset=1, limit=None):
    """head-truncation: at most max_lines lines and max_bytes, footer with the next offset."""
    p = _resolve(ctx, path)
    if not p.exists():
        hint = _hint(ctx, p)
        raise ToolError(f"{p} does not exist" + (f". did you mean: {', '.join(hint)}" if hint else ""))
    if p.is_dir():
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        names = [e.name + ("/" if e.is_dir() else "") for e in entries]
        shown = head_lines(names, ctx.max_lines, ctx.max_bytes)
        if not names:
            return "(empty directory)"
        more = f"\n[{len(names) - len(shown)} more entries not shown]" if len(shown) < len(names) else ""
        return "\n".join(shown) + more
    if _is_binary(p):
        raise ToolError(f"{p} is a binary file")
    offset = max(1, int(offset or 1))
    max_lines = min(max(1, int(limit)), ctx.max_lines) if limit else ctx.max_lines
    # stream instead of read_text(): only the window lands in memory, the rest is just counted
    # (otherwise a read on a 1gb log eats the ram, even though only 2000 lines are shown)
    numbered, total = [], 0
    with open(p, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            total += 1
            if total < offset:
                continue
            line = line.rstrip("\n")
            if len(line) > MAX_LINE:
                line = line[:MAX_LINE] + " [...]"
            numbered.append(f"{total:6}\t{line}")
            if len(numbered) >= max_lines:
                break
        tail = ""
        while chunk := fh.read(1 << 20):
            total += chunk.count("\n")
            tail = chunk[-1]
        if tail and tail != "\n":
            total += 1  # last line without trailing newline
    if not total:
        return "(empty file)"
    if offset > total:
        raise ToolError(f"offset {offset} is past the end of the file ({total} lines)")
    shown = head_lines(numbered, max_lines, ctx.max_bytes)
    last = offset + len(shown) - 1
    out = "\n".join(shown)
    if last < total:
        out += f"\n\n[showing lines {offset}-{last} of {total}; use offset={last + 1} to continue]"
    return out


# ---------------------------------------------------------------- write

def t_write(ctx, path, content):
    p = _resolve(ctx, path, write=True)
    old = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
    sb = ctx.sandbox
    if not p.parent.exists() and sb is not None and not sb.allows(p.parent, write=True):
        # the grant covers exactly p, never the directory above it (see _check) – without this
        # mkdir would create an empty dir outside the sandbox here
        raise ToolError(f"directory {p.parent} is outside the sandbox and not granted")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    if old is None:
        return f"{p} created ({len(content.splitlines())} lines)"
    return make_diff(old, content, str(p)) or f"{p} unchanged"


# ---------------------------------------------------------------- edit

def make_diff(old, new, label):
    sep = "" if label.startswith("/") else "/"  # absolute pfade ohne a//
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a{sep}{label}", tofile=f"b{sep}{label}", n=3,
    ))


def _indent(line):
    return line[: len(line) - len(line.lstrip())]


def _base_indent(lines):
    """common indentation of all non-empty lines."""
    indents = [_indent(x) for x in lines if x.strip()]
    return os.path.commonprefix(indents) if indents else ""


def _dedent(lines, base):
    return [x[len(base):].rstrip() if x.strip() else "" for x in lines]


def _trim_edges(old_lines, new_lines):
    """drop empty edge lines from old, and from new at most as many (symmetric)."""
    def count(lines, rev):
        n = 0
        for x in (reversed(lines) if rev else lines):
            if x.strip():
                break
            n += 1
        return n

    lead = count(old_lines, False)
    old_lines = old_lines[lead:]
    trail = count(old_lines, True)
    if trail:
        old_lines = old_lines[:-trail]
    nl = min(lead, count(new_lines, False))
    new_lines = new_lines[nl:]
    nt = min(trail, count(new_lines, True))
    if nt:
        new_lines = new_lines[:-nt]
    return old_lines, new_lines


def _reindent(new_lines, o, f):
    """shift new by the base delta o (old) → f (file). empty lines stay as they are."""
    if o == f:
        return new_lines
    out = []
    for line in new_lines:
        if not line.strip():
            out.append(line)
        elif f.startswith(o):              # file deeper: prepend delta
            out.append(f[len(o):] + line)
        elif o.startswith(f):              # file shallower: cut up to delta-length of ws
            cut = min(len(o) - len(f), len(_indent(line)))
            out.append(line[cut:])
        elif line.startswith(o):           # tabs vs. spaces in the base: swap prefix
            out.append(f + line[len(o):])
        else:
            out.append(line)
    return out


def _real_len(lines):
    """line count without the empty element after the last \\n."""
    return len(lines) - 1 if lines and lines[-1] == "" else len(lines)


def _mismatch_hint(lines, old_lines):
    first = old_lines[0].strip()
    n_real = _real_len(lines)
    cands = [i for i in range(n_real) if lines[i].strip() == first]
    if not cands:
        stripped = [x.strip() for x in lines[:n_real]]
        best = difflib.get_close_matches(first, stripped, n=1, cutoff=0.6)
        if not best:
            return ""
        no = stripped.index(best[0]) + 1
        return f"\nclosest line: {no}: {lines[no - 1]!r}"

    def prefix(i):
        k = 0
        while k < len(old_lines) and i + k < n_real and lines[i + k].strip() == old_lines[k].strip():
            k += 1
        return k

    k, i = max((prefix(i), -i) for i in cands)
    i = -i
    if k < len(old_lines):
        got = repr(lines[i + k]) if i + k < n_real else "<end of file>"
        return (f"\nline {i + 1} matches the start, line {i + k + 1} differs:\n"
                f"  file:       {got}\n  old_string: {old_lines[k]!r}")

    window = lines[i:i + len(old_lines)]
    a = _dedent(window, _base_indent(window))
    b = _dedent(old_lines, _base_indent(old_lines))
    for j, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            return (f"\nlines {i + 1}–{i + len(old_lines)} match in content, but the relative indentation "
                    f"differs (line {i + j + 1}): file {window[j]!r} vs old_string {old_lines[j]!r}")
    return ""


def apply_edit(content, old, new, replace_all=False):
    """search/replace. exact first, then whitespace-tolerant. returns the new content."""
    if old == "":
        raise ToolError("old_string is empty. use 'write' for new files")
    if old == new:
        raise ToolError("old_string and new_string are identical")

    crlf = "\r\n" in content
    text = content.replace("\r\n", "\n") if crlf else content
    old = old.replace("\r\n", "\n")
    new = new.replace("\r\n", "\n")

    # count and line numbers from the same (non-overlapping) search
    positions = []
    start = 0
    while (idx := text.find(old, start)) != -1:
        positions.append(idx)
        start = idx + len(old)
    count = len(positions)

    if count == 1 or (count > 1 and replace_all):
        result = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    elif count > 1:
        nos = [text.count("\n", 0, idx) + 1 for idx in positions[:20]]
        raise ToolError(
            f"old_string occurs {count}x (lines {', '.join(map(str, nos))}). "
            "add more context or set replace_all=true"
        )
    else:
        result = _tolerant(text, old, new, replace_all)

    return result.replace("\n", "\r\n") if crlf else result


def _tolerant(text, old, new, replace_all):
    lines = text.split("\n")
    old_lines, new_lines = _trim_edges(old.split("\n"), new.split("\n"))
    if not old_lines:
        raise ToolError("old_string is all whitespace")
    n = len(old_lines)
    o_base = _base_indent(old_lines)
    first = old_lines[0].strip()
    anchors = [i for i in range(len(lines) - n + 1) if lines[i].strip() == first]

    # pass 1: only trailing whitespace ignored
    target = [x.rstrip() for x in old_lines]
    hits = [(i, None) for i in anchors if [x.rstrip() for x in lines[i:i + n]] == target]

    # pass 2: block moved as a whole, relative indentation must match
    if not hits:
        target = _dedent(old_lines, o_base)
        for i in anchors:
            window = lines[i:i + n]
            f_base = _base_indent(window)
            if _dedent(window, f_base) == target:
                hits.append((i, f_base))

    if not hits:
        raise ToolError("old_string not found (not even whitespace-tolerant)."
                        + _mismatch_hint(lines, old_lines))
    if len(hits) > 1 and not replace_all:
        nos = ", ".join(str(h + 1) for h, _ in hits[:20])
        raise ToolError(
            f"old_string matches whitespace-tolerant {len(hits)}x (lines {nos}). "
            "add more context or set replace_all=true"
        )
    for i, f_base in reversed(hits):
        repl = new_lines if f_base is None else _reindent(new_lines, o_base, f_base)
        lines[i:i + n] = repl
    return "\n".join(lines)


def _label(ctx, p):
    cwd = Path(ctx.cwd).resolve()
    return p.relative_to(cwd).as_posix() if p.is_relative_to(cwd) else str(p)


def t_edit(ctx, path, old_string, new_string, replace_all=False):
    p = _resolve(ctx, path, write=True)
    if not p.is_file():
        hint = _hint(ctx, p) if not p.exists() else []
        raise ToolError(f"{p} does not exist" + (f". did you mean: {', '.join(hint)}" if hint else "")
                        + ". use 'write' for new files")
    if _is_binary(p):
        raise ToolError(f"{p} is a binary file")
    try:
        # newline="": read line endings raw, otherwise apply_edit never sees a \r\n
        with open(p, encoding="utf-8", newline="") as fh:
            content = fh.read()
    except UnicodeDecodeError as e:
        raise ToolError(f"{p} is not a utf-8 text file ({e})") from e

    result = apply_edit(content, old_string, new_string, bool(replace_all))
    if result == content:
        raise ToolError("edit changes nothing (after whitespace-tolerance old_string and new_string are equal). "
                        "file untouched")
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(result)

    lf = lambda s: s.replace("\r\n", "\n")  # noqa: E731 - diff ohne \r-rauschen
    diff = make_diff(lf(content), lf(result), _label(ctx, p))
    if lf(old_string) not in lf(content):
        return ("note: old_string was not an exact match (whitespace-tolerant), "
                "next time copy it exactly from read.\n" + diff)
    return diff


# ---------------------------------------------------------------- bash

class _Capture:
    """reads an fd with bounded memory.

    as long as the output fits the byte budget, everything stays in memory. once it grows
    beyond, everything goes to a file (the full text) and only the tail stays in memory.
    at the end text() shows the last max_lines/max_bytes – for bash the important part
    (errors, summary) is almost always at the bottom."""

    def __init__(self, fd, ctx):
        self.fd = fd
        self.ctx = ctx
        self.max_lines = ctx.max_lines
        self.max_bytes = ctx.max_bytes
        self.buf = bytearray()
        self.total = 0
        self.newlines = 0
        self.last = b""
        self.spool = None
        self.spool_path = None
        self.spooled = 0
        self.spool_error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _start_spool(self):
        try:
            self.spool_path, self.spool = _spool_open(self.ctx, "bash")
            self._write(bytes(self.buf))
        except OSError as e:
            self.spool_error = str(e)

    def _write(self, data):
        if self.spool is None or self.spooled >= SPOOL_CAP:
            return
        data = data[: SPOOL_CAP - self.spooled]
        try:
            self.spool.write(data)
            self.spooled += len(data)
        except (OSError, ValueError) as e:  # ValueError: spool already closed (defensive)
            self.spool_error = str(e)
            self.spool = None

    def stop(self, timeout=2):
        """stop reading and wait until the thread is really gone. only then may the fd be
        closed: otherwise the thread may still be stuck in read() and, after the close,
        read from a reused fd number (foreign file/socket)."""
        self.stopping.set()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def _chunks(self):
        """yields read chunks. non-blocking + select, so stop() always works – even when a
        grandchild process (own session, survives killpg) keeps the write end open."""
        os.set_blocking(self.fd, False)
        while True:
            try:
                ready, _, _ = select.select([self.fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not ready:
                if self.stopping.is_set():
                    return  # nothing buffered anymore and stop requested
                continue
            while True:  # drain everything that is there right now
                try:
                    chunk = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                except OSError:
                    return
                if not chunk:
                    return  # eof: all write ends closed
                yield chunk

    def _run(self):
        keep = self.max_bytes + 4096  # tail buffer, a bit more than the budget
        for chunk in self._chunks():
            self.total += len(chunk)
            self.newlines += chunk.count(b"\n")
            self.last = chunk[-1:]
            if self.spool is None and self.spool_error is None and len(self.buf) + len(chunk) > self.max_bytes:
                self._start_spool()
            self._write(chunk)
            self.buf += chunk
            if self.spool is not None or self.spool_error is not None:
                if len(self.buf) > keep:
                    del self.buf[: len(self.buf) - keep]

    def line_count(self):
        return self.newlines + (1 if self.total and self.last != b"\n" else 0)

    def text(self):
        """(visible text, hint line or None). call after the process has ended."""
        n = self.line_count()
        complete = self.total == len(self.buf)  # nothing discarded from the buffer
        if complete and n <= self.max_lines and self.total <= self.max_bytes:
            return self.buf.decode("utf-8", "replace"), None
        if self.spool is None and self.spool_error is None:
            self._start_spool()  # only too many lines: full text is still complete in the buffer
        if self.spool is not None:
            self.spool.close()
        text = self.buf.decode("utf-8", "replace")
        lines = text.split("\n")
        if text.endswith("\n"):
            lines.pop()
        if not complete and len(lines) > 1:
            lines.pop(0)  # first line in the tail buffer is usually cut off
        shown = tail_lines(lines, self.max_lines, self.max_bytes)
        clipped = ""
        if len(shown) == 1 and _nbytes(shown[0]) > self.max_bytes:
            shown = [_clip_bytes(shown[0], self.max_bytes, keep_end=True)]
            clipped = ", last line clipped at the front"
        first = n - len(shown) + 1
        if self.spool_path is not None:
            where = f"full output: {self.spool_path}"
            if self.spooled < self.total:
                mb = self.spooled / (1024 * 1024)
                where += f", truncated after {mb:.0f}MB" if mb >= 1 else f", truncated after {self.spooled} bytes"
        else:
            where = f"full text not saved: {self.spool_error}"
        return "\n".join(shown), f"[showing lines {first}-{n} of {n}{clipped}; {where}]"


def _stop_group(proc):
    """SIGTERM to the process group first, so e.g. a sub-pyrate can clean up its own children
    (own sessions, which a killpg here doesn't reach). then SIGKILL.
    waits for the whole group: bash itself dies instantly, the child is still cleaning up."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + TERM_GRACE
        while time.monotonic() < deadline:
            proc.poll()  # reap bash, otherwise the zombie keeps the group alive
            os.killpg(proc.pid, 0)  # ProcessLookupError = group empty
            time.sleep(0.05)
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def t_bash(ctx, command, timeout=None):
    timeout = int(timeout or ctx.bash_timeout)
    argv = ["bash", "-c", command]
    if ctx.sandbox is not None:
        argv = ctx.sandbox.wrap(argv, readonly=ctx.readonly, net=ctx.net)
    r, w = os.pipe()
    proc = subprocess.Popen(
        argv, cwd=ctx.cwd, stdin=subprocess.DEVNULL,
        stdout=w, stderr=w, start_new_session=True,
        env={**os.environ, **ctx.env} if ctx.env else None,
    )
    os.close(w)
    cap = _Capture(r, ctx)
    deadline = time.monotonic() + timeout
    note = ""
    try:
        while proc.poll() is None:
            if ctx.cancel is not None and ctx.cancel.is_set():
                note = "\n[cancelled]"
                break
            if time.monotonic() > deadline:
                note = f"\n[timeout after {timeout}s, process group killed]"
                break
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            _stop_group(proc)
        proc.wait()
        # wait briefly for eof (direct children are gone, buffered data is read quickly) …
        cap.thread.join(timeout=0.5)
        if cap.thread.is_alive():
            # … no eof: a background process is holding the output open
            note += "\n[background process holds stdout/stderr open, further output is not read]"
        if cap.stop():
            os.close(r)
        # otherwise better to leak the fd than to free it under a still-reading thread
    out, hint = cap.text()
    out = out.rstrip()
    head = f"{hint}\n" if hint else ""
    return f"{head}{out}{note}\n[exit {proc.returncode}]".lstrip("\n")


# ---------------------------------------------------------------- ask_user

def t_ask_user(ctx, question, options=None):
    if ctx.ask_user is None:
        return "(no human reachable. make a reasonable assumption and mention it.)"
    answer = ctx.ask_user(question, list(options or []))
    return answer if answer else "(no answer)"


# ---------------------------------------------------------------- grep / find

def _walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def glob_matcher(pattern):
    """glob → predicate(relpath). without '/' the pattern matches the file name at any depth
    (*.py finds everything), with '/' the path relative to the start dir: * and ? stay within
    one path segment, ** spans any number (including zero) of directories."""
    pattern = pattern.strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    out, i, n = [], 0, len(pattern)
    while i < n:
        c = pattern[i]
        if pattern.startswith("**", i):
            i += 2
            if pattern.startswith("/", i):
                out.append("(?:.*/)?")
                i += 1
            else:
                out.append(".*")
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[" and (j := pattern.find("]", i + 2)) != -1:
            body = pattern[i + 1:j].replace("\\", "\\\\")
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = j + 1
            continue
        else:
            out.append(re.escape(c))
        i += 1
    rx = re.compile("".join(out) + r"\Z", re.DOTALL)
    if "/" in pattern:
        return lambda rel: rx.match(rel) is not None
    return lambda rel: rx.match(rel.rpartition("/")[2]) is not None


MAX_REGEX = 500  # python-re has no timeout; long model regexes are the classic redos candidate


def t_grep(ctx, pattern, path=".", include=None):
    if len(pattern) > MAX_REGEX:
        raise ToolError(f"regex too long ({len(pattern)} > {MAX_REGEX} chars), simplify it")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"invalid regex: {e}") from e
    root = _resolve(ctx, path)
    files = [root] if root.is_file() else _walk(root)
    match = glob_matcher(include) if include else None
    hits, size = [], 0
    max_hits = min(200, ctx.max_lines)
    for f in files:
        if match and not match(f.name if f == root else f.relative_to(root).as_posix()):
            continue
        if ctx.sandbox is not None and f.is_symlink() and not ctx.sandbox.allows(f.resolve()):
            continue  # symlink leading out of the sandbox: don't follow it
        try:
            if f.stat().st_size > 2_000_000 or _is_binary(f):
                continue
            with open(f, encoding="utf-8", errors="replace") as fh:
                for no, line in enumerate(fh, 1):
                    if not rx.search(line):
                        continue
                    line = line.rstrip()
                    if len(line) > GREP_LINE:
                        line = line[:GREP_LINE] + " [...]"
                    hit = f"{f}:{no}: {line}"
                    size += _nbytes(hit) + 1
                    if len(hits) >= max_hits or size > ctx.max_bytes:
                        return "\n".join(hits) + f"\n[cut off after {len(hits)} hits, narrow the search]"
                    hits.append(hit)
        except OSError:
            continue
    return "\n".join(hits) or "no matches"


def t_find(ctx, pattern, path="."):
    root = _resolve(ctx, path)
    match = glob_matcher(pattern)
    out = []
    for f in _walk(root):
        if match(f.relative_to(root).as_posix()):
            out.append(str(f))
            if len(out) >= 500:
                out.append("(cut off after 500 matches)")
                break
    return "\n".join(out) or "no matches"


# ---------------------------------------------------------------- registry

def _p(type_, desc, optional=False, **extra):
    d = {"type": type_, "description": desc, **extra}
    if optional:
        d["optional"] = True
    return d


TOOLS = {t.name: t for t in [
    Tool("read", "read a file with line numbers (or list a directory). at most 2000 lines / 50kb "
                 "per call; longer files get a footer telling you the next offset.", {
        "path": _p("string", "path, relative to the cwd or absolute"),
        "offset": _p("integer", "start line, 1-based", optional=True),
        "limit": _p("integer", "max. lines (default and upper bound 2000)", optional=True),
    }, t_read, readonly=True, self_truncating=True),
    Tool("write", "write a whole file / create it. for changes prefer 'edit'.", {
        "path": _p("string", "path"),
        "content": _p("string", "complete new content"),
    }, t_write, readonly=False),
    Tool("edit", "search/replace in a file. old_string must occur exactly once "
                 "(otherwise use replace_all). returns a diff. copy old_string exactly from read "
                 "(without line numbers); only trailing whitespace and a uniform indentation shift "
                 "of the whole block are tolerated.", {
        "path": _p("string", "path"),
        "old_string": _p("string", "text to replace, with enough context"),
        "new_string": _p("string", "replacement text"),
        "replace_all": _p("boolean", "replace all occurrences", optional=True),
    }, t_edit, readonly=False),
    Tool("bash", "run a shell command in the cwd (bash -c). stdout+stderr combined. long output "
                 "is tail-truncated (2000 lines / 50kb); the full text is in the file from the hint "
                 "and can be read with read (offset).", {
        "command": _p("string", "command"),
        "timeout": _p("integer", "timeout in seconds", optional=True),
    }, t_bash, readonly=False, self_truncating=True),
    Tool("ask_user", "ask the user a question when something is genuinely unclear.", {
        "question": _p("string", "the question"),
        "options": _p("array", "optional answer suggestions", optional=True, items={"type": "string"}),
    }, t_ask_user, readonly=True),
    Tool("grep", "regex search over file contents (python-re).", {
        "pattern": _p("string", "regex"),
        "path": _p("string", "directory or file (default cwd)", optional=True),
        "include": _p("string", "file-name glob, e.g. *.py", optional=True),
    }, t_grep, readonly=True, self_truncating=True),
    Tool("find", "find files by glob, e.g. **/*.py or *.yaml.", {
        "pattern": _p("string", "glob"),
        "path": _p("string", "start directory (default cwd)", optional=True),
    }, t_find, readonly=True),
]}


def enabled(cfg_tools):
    """all tools enabled via config (independent of mode)."""
    return [n for n, t in TOOLS.items() if cfg_tools.get(n, t.default_on)]


def for_mode(cfg_tools, mode, plan_bash=False):
    """plan_bash: bash in plan mode too – only if the sandbox can lock it read-only."""
    names = enabled(cfg_tools)
    if mode == "plan":
        names = [n for n in names if TOOLS[n].readonly or (n == "bash" and plan_bash)]
    return names


def _lookup(name, extra):
    return TOOLS.get(name) or (extra or {}).get(name)


def schemas(names, extra=None):
    """extra: additional tools (mcp), name → Tool."""
    return [_lookup(n, extra).schema() for n in names]


def short(obj, n=120):
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def tool_label(name, args):
    """one-liner for display: 'bash: pytest -q'."""
    if isinstance(args, dict):
        for key in ("command", "path", "pattern", "question"):
            if key in args:
                return f"{name}: {short(args[key], 100)}"
    return f"{name}: {short(args, 100)}"


def run(name, args, ctx, allowed, extra=None):
    """runs a tool. returns (ok, text), never raises."""
    tool = _lookup(name, extra)
    if tool is None:
        return False, f"unknown tool '{name}'"
    if name not in allowed:
        return False, f"tool '{name}' is not allowed in the current mode (plan = read-only)"
    if not isinstance(args, dict):
        return False, "arguments must be a json object"
    try:
        if tool.raw_schema is not None:
            out = str(tool.func(ctx, args))
        else:
            unknown = sorted(k for k in args if k not in tool.params)
            if unknown:  # a typo in the arg name must be an error, not a silent drop
                raise ToolError(f"unknown arguments: {', '.join(unknown)} "
                                f"(allowed: {', '.join(tool.params)})")
            out = str(tool.func(ctx, **args))
        # read/bash/grep truncate themselves sensibly (head/tail + hint). for the rest
        # (diffs, find, …) only a coarse safety net.
        return True, out if tool.self_truncating else truncate(out, ctx.max_bytes)
    except ToolError as e:
        return False, f"error: {e}"
    except TypeError as e:
        return False, f"error: wrong arguments: {e}"
    except Exception as e:  # noqa: BLE001 - tool errors go to the model, not into the stacktrace
        return False, f"error: {type(e).__name__}: {e}"
