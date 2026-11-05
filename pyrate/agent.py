"""the loop: call → run tool_calls → until none are left. headless, without tui."""

import json
import os
import platform
import re
import shlex
import sys
import time
from pathlib import Path

from . import compact, context, llm, podman, subagent, tools
from . import images as images_mod  # alias: run(prompt, images=…) would otherwise shadow the module
from .compact import CompactError, estimate
from .config import ConfigError
from .db import DEFAULT_PATH, repair
from .sandbox import SYSTEM_RO, SYSTEM_RW, SYSTEM_SOCK, Sandbox

SYSTEM = """You are pyrate, a coding agent in the user's terminal.
cwd: {cwd} | os: {os} | date: {date}
Use the tools to inspect and change files and run commands. Read before you edit.
Small, targeted edits. Concise; no filler."""

PLAN = """PLAN MODE: read-only tools only. Investigate, then propose a concrete plan.
Do not claim to have changed anything."""

PLAN_BASH = """bash works in plan mode, but the filesystem is read-only (only $TMPDIR is writable) and
there is no network. use it to inspect: git diff/log/status, rg, ls, tests that don't write into the cwd."""

PLAN_BASH_NET = """bash works in plan mode, but the filesystem is read-only (only $TMPDIR is writable).
tcp is enabled for this session: curl, ssh, git push etc. work, but nothing may be written outside $TMPDIR."""

BUILD_NO_NET = """tcp is off for this session (ctrl+n turns it on): curl, ssh, git push etc. fail with
connect errors. the filesystem is fully writable."""

SANDBOX = """SANDBOX: tools and bash only reach the cwd{tmp}, plus system dirs read-only.
everything else, including $HOME, is blocked (bash: permission denied). need another path? use read
(or write) on it: the user gets asked, and once granted it works in bash too."""

SANDBOX_TMP = """scratch files go to $TMPDIR, never /tmp; it is wiped when pyrate exits."""

PODMAN = """podman works in build mode (remote, CONTAINER_HOST is set), not in plan mode. the user's own
containers are visible too: leave them alone unless asked. bind-mount only the cwd or $TMPDIR."""

PODMAN_OFF = """podman and docker don't work in this sandbox (no mount, no uid_map). don't try to get them
running; tell the user. they can enable podman outside the sandbox: `podman: true` in the global config."""

PODMAN_DOWN = """podman and docker don't work in this sandbox (no mount, no uid_map), and pyrate's podman
service outside of it is down in this session. don't try to get them running; tell the user."""

SUBAGENTS = """SUBAGENTS: delegate self-contained subtasks to fresh pyrate instances via bash:
  {cmd} -p - <<'EOF'
  <task>
  EOF
- the subagent sees nothing of this conversation: put everything it needs (goal, paths, constraints) into its prompt.
- it runs in plan mode (read-only) by default. add `--mode build` only if it must change files.
- only its final answer reaches stdout; ask for a concise report. `session <id>` on stderr; `-s <id>` continues it.
- parallel: at most {n} per bash call. `dir=$(mktemp -d)`, start each with
  `{cmd} -p - > "$dir/1.md" <<'EOF' &`, then `wait`, then read the files. never let two edit the same files.
- set the bash timeout generously (e.g. 900): it covers the whole call, not each subagent.
- for broad research or independent work packages, not for a single lookup."""

SUBAGENT_CHILD = """SUBAGENT: another pyrate agent started you for one task. nobody can answer questions.
only your final message goes back: a complete, concise report
(findings, exact file paths, what you changed, what is still open)."""

TITLE_PROMPT = """Summarize the request below as a short title (3-6 words) describing its concrete topic.
Use the language of the request. Reply with the title only: no quotes, no punctuation at the end.

<request>
{prompt}
</request>"""

MAGIC_MAX_TOKENS = 512    # thinking tokens count against max_tokens on some backends – 200 can come back empty
MAGIC_SERVERS_MAX = 6000  # chars cap for the server list in the llm prompt
MAGIC_PROMPT_TOOL_LINE = 80  # chars pro tool-zeile im llm-prompt (name + gekürzte desc)
MAGIC_PROMPT = """Available MCP servers (name: description, then one line per tool):
{servers}

User message:
<message>
{prompt}
</message>

Enable the servers whose tools can help with this message. Reply with JSON only, nothing else,
no markdown: {{"enable": ["name", ...]}} – names exactly as listed above, [] if none fit.
Example: message "how far is A from B?" with a maps server "gpx" → {{"enable": ["gpx"]}}"""


# kleine de→en-brücke für den keyword-fallback: deutsche prompts, englische tool-descs
MAGIC_SYNONYMS = {
    "termin": "calendar event appointment", "termine": "calendar event appointment",
    "kalender": "calendar event appointment",
    "wetter": "weather forecast",
    "karte": "map route geocode poi", "karten": "map route geocode poi",
    "route": "route", "entfernung": "distance route", "adresse": "geocode address",
    "ticket": "issue ticket", "fehler": "issue bug error", "bug": "issue bug error",
    "datei": "file", "suche": "search", "suchen": "search",
}


def _prompt_words(prompt):
    """prompt-wörter (>=4 zeichen, lowercased) + de→en-synonyme – für ranking und fallback."""
    words = {w for w in re.findall(r"[a-zäöüß]+", (prompt or "").lower()) if len(w) >= 4}
    for w in list(words):
        words.update(MAGIC_SYNONYMS.get(w, "").split())
    return words


def magic_keywords(prompt, servers, min_hits=2):
    """deterministischer fallback, wenn der router ausfällt oder unparsebar antwortet:
    prompt-wörter gegen name+description+tool-lines jedes servers. erst ab min_hits treffern
    → server an, sonst schaltet jedes "list"/"file" alles ein (enable-only, bleibt an)."""
    words = _prompt_words(prompt)
    hits = []
    for name, desc, tl in servers:
        text = f"{name} {desc} {' '.join(tl)}".lower()
        n = sum(1 for w in words if w in text)
        if n >= min_hits:
            hits.append((n, name))
    hits.sort(reverse=True)
    return [name for _, name in hits]


def parse_magic(raw):
    """router-antwort → liste von server-namen ([] = "keiner passt", gültige antwort).
    robust gegen prose/markdown drumherum, auch eine bare list ["a", "b"] geht.
    nichts parsebares → None (dann greift der keyword-fallback)."""
    for m in re.finditer(r"\{.*\}|\[.*\]", raw or "", re.S):
        try:
            data = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(data, dict):
            data = data.get("enable")
        if isinstance(data, list):
            return [str(x) for x in data if isinstance(x, str)]
    return None


SEARCH_TOOL = "search_tools"
SEARCH_LOAD_MAX = 8    # query: so viele treffer werden geladen
SEARCH_DESC = 150      # chars pro treffer-zeile


def search_hits(query, candidates):
    """query-wörter gegen name + description, sortiert nach trefferzahl, dann name.
    candidates: {name: description}. keine wörter → []."""
    words = [w for w in (query or "").lower().split() if w]
    scored = []
    for name, desc in candidates.items():
        text = f"{name} {desc}".lower()
        n = sum(1 for w in words if w in text)
        if n:
            scored.append((-n, name))
    return [name for _, name in sorted(scored)]


def _short_desc(desc, n):
    desc = " ".join((desc or "").split())
    return desc if len(desc) <= n else desc[: n - 1] + "…"


def subagent_cmd(config_arg=None):
    """how pyrate calls itself: same interpreter = guaranteed same code. a `pyrate` from PATH may
    be an older copy that knows nothing about subagents. the package is importable in all venvs
    (make venv/install/install-link install it)."""
    cmd = [sys.executable, "-m", "pyrate"]
    if config_arg:
        cmd += ["--config", str(config_arg)]
    return shlex.join(cmd)


def fallback_title(prompt):
    """title without llm: the first words of the prompt."""
    words = prompt.split()
    return " ".join(words[:8]) + (" …" if len(words) > 8 else "")


def _as_parts(content):
    """user-content → list of content parts. string → one text part (empty → none)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []


def merge_user_turns(messages):
    """merge consecutive user messages into one. happens after compaction
    (summary + next prompt) or after a cancel before the first answer; strict
    backends (anthropic & proxies) reject two user turns in a row.
    with images: parts of both turns concatenated in order (label stays before its image).
    request only – history and db stay as they are."""
    out = []
    for m in messages:
        prev = out[-1] if out else None
        if prev is not None and m.get("role") == "user" and prev.get("role") == "user":
            a, b = prev.get("content"), m.get("content")
            if isinstance(a, str) and isinstance(b, str):
                content = f"{a}\n\n{b}"
            else:
                content = _as_parts(a) + _as_parts(b)
            out[-1] = {**prev, "content": content}
        else:
            out.append(m)
    return out


REASONING_KEYS = ("reasoning_content", "thinking_blocks")


def drop_old_reasoning(messages):
    """send reasoning/thinking blocks only in the current turn (after the last user message).
    older ones may come from a different model (ctrl+p) – foreign signatures and
    reasoning fields get some backends to answer 400. they are only needed within
    a tool loop; anthropic & co. drop thinking from earlier turns anyway.
    the model only changes between runs, and every run starts with a user message."""
    last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    out = []
    for i, m in enumerate(messages):
        if i < last_user and m.get("role") == "assistant" and any(k in m for k in REASONING_KEYS):
            m = {k: v for k, v in m.items() if k not in REASONING_KEYS}
        out.append(m)
    return out


class Agent:
    def __init__(self, cfg, db, cwd, session_id=None, model=None, mode=None, effort=None, ask_user=None,
                 mcp=None, parent_id=None, tmp=None, podman=None):
        self.cfg = cfg
        self.db = db
        self.cwd = str(cwd)
        self.tmp = str(tmp) if tmp else None  # this run's own tmp (main.make_tmp), None = none
        self.podman = podman                  # podman.Service or None
        self.session_id = session_id
        self.parent_id = parent_id  # set = we are a subagent ourselves
        # max. parallel children, switchable at runtime (ctrl+s). depth 1: children don't spawn
        self.subagents = 0 if parent_id else cfg.subagents
        self.mode = mode or cfg.mode_default
        # tcp on/off, switchable at runtime (ctrl+n in the tui), in both modes. default per mode:
        # plan = off (that's the point of plan), build = on
        self.net = self.mode == "build"
        # mcp_magic router on/off at runtime (toggle_magic/ctrl+g in der tui). default: aus,
        # erst der toggle schaltet ihn für die session an
        self.magic = False
        self.effort = effort or None    # None = default from config (model → backend → global)
        self.step = 0               # current step in the running/last run (0 = none yet)
        self.ask_user = ask_user
        self.mcp = mcp              # mcp.Manager oder None
        self.loaded_tools = set()   # mcp_tool_search: per search_tools geladene schemas (session-lang)
        self._estimate_cache = None
        self.messages = db.load_messages(session_id) if session_id else []
        self.context_warnings = []
        self.sandbox = self.make_sandbox() if cfg.sandbox else None
        self.reload_context()
        self.model = cfg.model
        if model:
            # e.g. from an old session: unqualified or backend gone meanwhile
            try:
                self.model = cfg.qualify(model)
            except ConfigError as e:
                self.context_warnings.append(f"model '{model}' not available ({e}), using {cfg.model}")
        if self.effort is None:
            self.effort = cfg.effort_for(self.model)
        self.context_tokens = None  # real value from usage, None = no request yet
        self.last_usage = None
        self._usage_len = 0         # len(messages) at the time of the last usage

    def reload_context(self):
        """re-read the AGENTS.md hierarchy and skills (global + project)."""
        warnings = []
        self.instructions = context.load_instructions(self.cwd, self.cfg.global_dir, warnings)
        self.skills = context.discover_skills(self.cwd, self.cfg.global_dir, self.cfg.skill_dirs, warnings)
        self.context_warnings = warnings
        if self.sandbox is not None:  # the prompt says "read SKILL.md" with absolute path, scripts/ included
            for s in self.skills:
                self.sandbox.grant(s.path.parent)

    def make_sandbox(self):
        """cwd + tmp + sandbox_rw writable, system + sandbox_ro readable, full texts of truncated
        outputs readable. tmp goes in explicitly: without it there is none, never silently /tmp."""
        out_dir = self.cfg.tool_output_dir or tools.default_output_dir()
        # git aborts hard ("unknown error reading config") if the global config exists but is
        # unreadable – without it there is no git in the sandbox at all
        git = [Path.home() / ".gitconfig",
               Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "git"]
        # podman dir rw: build = connect to the socket ok, plan = ro = no connect
        pod_ro, pod_rw = ([podman.config_dir()], [self.podman.path]) if self.podman else ([], [])
        return Sandbox(ro=[*SYSTEM_RO, *git, *pod_ro, *self.cfg.sandbox_ro, out_dir],
                       rw=[self.cwd, *self.cfg.sandbox_rw, *pod_rw],
                       scratch=[self.tmp, *SYSTEM_RW] if self.tmp else SYSTEM_RW,
                       sock=SYSTEM_SOCK)

    def allow_subagents(self):
        """what a child pyrate needs in bash: interpreter, code, config, db, slots, full texts.
        per step, because ctrl+s enables subagents at runtime. lasts until the session ends."""
        if self.sandbox is None or not self.subagents:
            return
        for p in (sys.prefix, sys.base_prefix, Path(__file__).parent, self.cfg.global_dir, self.cfg.config_arg):
            self.sandbox.grant(p)
        for p in (self.cfg.tool_output_dir or tools.default_output_dir(), DEFAULT_PATH.parent,
                  (self.cfg.db_path or DEFAULT_PATH).parent, subagent.run_dir()):
            self.sandbox.grant(p, write=True)

    # ------------------------------------------------------------ prompt & tools

    def system_prompt(self):
        parts = [SYSTEM.format(cwd=self.cwd, os=platform.system().lower(), date=time.strftime("%Y-%m-%d"))]
        if self.sandbox is not None:
            parts.append(SANDBOX.format(tmp=f" and $TMPDIR ({self.tmp})" if self.tmp else ""))
            if self.tmp:
                parts.append(SANDBOX_TMP)
            if "bash" in self.tool_names():
                if self.podman_ok():
                    parts.append(PODMAN)
                elif self.cfg.podman:  # configured, but the service didn't come up
                    parts.append(PODMAN_DOWN)
                elif self.cfg.container_cli:  # installed, but off: otherwise the model wastes time on it
                    parts.append(PODMAN_OFF)
        if self.mode == "plan":
            parts.append(PLAN)
            if "bash" in self.tool_names():
                parts.append(PLAN_BASH_NET if self.net else PLAN_BASH)
        elif not self.net and "bash" in self.tool_names():
            parts.append(BUILD_NO_NET)
        if self.parent_id:
            parts.append(SUBAGENT_CHILD)
        # build + net only: in plan mode a child can neither write to the db nor reach the llm,
        # and without tcp (ctrl+n off) it can't reach the llm either
        elif self.subagents > 0 and self.mode == "build" and self.net \
                and "bash" in tools.for_mode(self.cfg.tools, self.mode):
            parts.append(SUBAGENTS.format(cmd=subagent_cmd(self.cfg.config_arg), n=self.subagents))
        if self.skills:
            parts.append(context.render_skills(self.skills))
        if self.instructions:  # last: the most specific instruction ends up closest to the history
            parts.append(context.render_instructions(self.instructions))
        return "\n\n".join(parts)

    def toolset(self):
        """(allowed names, extra tools) – a snapshot, mcp tools can come and go.
        mcp_tool_search: ab mcp_tool_search_min mcp tools gehen nur die geladenen schemas in den
        request, der rest steht als manifest im meta-tool search_tools. extra behält alle,
        damit ein direkter call auf ein ungeladenes tool trotzdem aufgelöst werden kann."""
        extra = self.mcp.tools() if self.mcp else {}
        names = tools.for_mode(self.cfg.tools, self.mode, self.cfg.plan_bash)
        visible = sorted(n for n, t in extra.items() if self.mode != "plan" or t.readonly)
        deferred = self.deferred_tools(visible)
        if deferred:
            extra = {**extra, SEARCH_TOOL: self._search_tool(deferred, extra)}
            names.append(SEARCH_TOOL)
        names += [n for n in visible if n not in deferred]
        return names, extra

    def deferred_tools(self, visible):
        """mcp tools, deren schema (noch) nicht im request steht. leer = tool search aus."""
        if not self.cfg.mcp_tool_search or len(visible) < max(1, self.cfg.mcp_tool_search_min):
            return []
        return [n for n in visible if n not in self.loaded_tools]

    def _search_tool(self, deferred, extra):
        """meta-tool: manifest (name + kurze description) in der description, laden per
        query (wort-treffer) oder exakten namen. geladen = ab dem nächsten call im request."""
        # "[mcp:server] " weg: der server steht schon im namen, spart tokens pro zeile
        cands = {n: re.sub(r"^\[mcp:[^\]]*\]\s*", "", extra[n].description) for n in deferred}
        lines = [f"{n}: {_short_desc(cands[n], 80)}" for n in deferred]

        def func(ctx, query=None, names=None):
            picked = [n for n in (names or []) if isinstance(n, str) and n in cands]
            unknown = [n for n in (names or []) if n not in cands and n not in self.loaded_tools]
            if query:
                picked += [n for n in search_hits(query, cands) if n not in picked][:SEARCH_LOAD_MAX]
            if not picked:
                hint = f" unknown: {', '.join(map(str, unknown))}." if unknown else ""
                raise tools.ToolError(f"no matching tools.{hint} use other words or exact names "
                                      "from the list in the search_tools description")
            self.loaded_tools.update(picked)
            out = [f"loaded {len(picked)} tool(s), callable now:"]
            out += [f"- {n}: {_short_desc(cands[n], SEARCH_DESC)}" for n in picked]
            if unknown:
                out.append(f"unknown: {', '.join(map(str, unknown))}")
            return "\n".join(out)

        desc = ("load mcp tools on demand. their schemas are not sent until loaded. "
                "pass a keyword query and/or exact names; loaded tools are callable right after.\n"
                "available (not loaded yet):\n" + "\n".join(lines))
        return tools.Tool(SEARCH_TOOL, desc, {
            "query": tools._p("string", "keywords matched against tool names and descriptions", optional=True),
            "names": tools._p("array", "exact tool names to load", optional=True, items={"type": "string"}),
        }, func, readonly=True)

    def tool_names(self):
        return self.toolset()[0]

    def request_messages(self):
        msgs = drop_old_reasoning(merge_user_turns(self.messages))
        return [{"role": "system", "content": self.system_prompt()}, *msgs]

    def estimated_tokens(self, extra=""):
        """chars/4 estimate. the tui asks this on every keypress → cache the base.
        messages are only appended or replaced as a whole list (repair, compaction)."""
        names, ext = self.toolset()
        key = (id(self.messages), len(self.messages), self.system_prompt(), tuple(names))
        if self._estimate_cache is None or self._estimate_cache[0] != key:
            base = estimate(self.request_messages()) + estimate(tools.schemas(names, ext))
            self._estimate_cache = (key, base)
        return self._estimate_cache[1] + len(extra) // 4

    def context_limit(self):
        return self.cfg.limit_for(self.model)

    def current_tokens(self):
        """best estimate for the next request: last real usage + what was added since."""
        if self.context_tokens is None:
            return self.estimated_tokens()
        return self.context_tokens + estimate(self.messages[self._usage_len:])

    def needs_compaction(self):
        limit = self.context_limit()
        if not limit or not self.cfg.auto_compact:
            return False
        reserve, _ = compact.budgets(limit)
        return self.current_tokens() > limit - reserve

    # ------------------------------------------------------------ compaction

    def compact(self, emit=None, cancel=None, reason="manual"):
        """summarize the old history. raises CompactError, LLMError, Cancelled."""
        emit = emit or (lambda ev: None)
        if not self.session_id:
            raise CompactError("no session yet, nothing to compact")
        prev, active = self.db.load_active(self.session_id)
        limit = self.context_limit()
        keep = compact.budgets(limit)[1] if limit else compact.KEEP
        cut = compact.find_cut(active, keep)
        if cut == 0:
            if reason != "manual" or not active:
                raise CompactError("nothing to compact, the history fits entirely into the keep budget")
            cut = len(active)  # explicitly requested: then summarize everything
        old = active[:cut]
        model = self.cfg.compact_model or self.model
        before = self.current_tokens()
        emit({"type": "compact_start", "reason": reason, "tokens": before})

        summary = compact.summarize(self.cfg, model, old, previous=(prev or {}).get("summary"),
                                    session=self.session_id,
                                    emit=emit, cancel=cancel)
        read, modified = compact.file_lists(old, self.cwd, (prev or {}).get("read_files", ()),
                                            (prev or {}).get("modified_files", ()))
        data = {
            "summary": summary,
            "content": compact.render(summary, read, modified),
            "first_kept_id": active[cut]["id"] if cut < len(active) else None,
            "read_files": read,
            "modified_files": modified,
            "reason": reason,
            "model": model,
            "tokens_before": before,
        }
        self.db.add_message(self.session_id, "compaction", self.mode, [("compaction", data)])
        self.messages = self.db.load_messages(self.session_id)
        self.context_tokens = None
        self._usage_len = 0
        emit({"type": "compact_done", "reason": reason, "before": before, "after": self.estimated_tokens(),
              "summary": summary, "summarized": cut, "kept": len(active) - cut})
        return data

    def run_compact(self, on_event=None, cancel=None):
        """manual /compact. errors go out as events, like in run()."""
        emit = on_event or (lambda ev: None)
        try:
            self.compact(emit, cancel, reason="manual")
        except llm.Cancelled:
            emit({"type": "cancelled"})
            return False
        except (CompactError, llm.LLMError) as e:
            emit({"type": "error", "error": f"compaction failed: {e}"})
            return False
        emit({"type": "done"})
        return True

    # ------------------------------------------------------------ session

    def ensure_session(self):
        if self.session_id is None:
            self.session_id = self.db.create_session(self.cwd, self.model, parent_id=self.parent_id)
            return True
        return False

    def tool_env(self):
        """env for bash. a `pyrate -p` inside the bash tool recognizes from this that it is a subagent."""
        env = {
            "PYRATE_SESSION": self.session_id or "",
            "PYRATE_SUBAGENTS": str(self.subagents),
            "PYRATE_MODEL": self.model,
        }
        if self.tmp:
            env["TMPDIR"] = self.tmp  # mktemp & co. land in our own tmp, a child pyrate creates its own inside
        if self.mode == "plan":
            env["GIT_OPTIONAL_LOCKS"] = "0"  # git status/diff must not try to write .git/index
        if self.podman is not None and self.sandbox is not None:
            # also in plan mode and with a dead service: podman stays remote and never touches the
            # local state. in plan mode landlock blocks the connect
            env.update(self.podman.env(self.tmp))
        return env

    def podman_ok(self):
        return self.podman is not None and self.podman.error is None

    def set_model(self, model):
        self.model = self.cfg.qualify(model)
        # model default wins: whoever set ctrl+t manually sets it again after the switch
        self.effort = self.cfg.effort_for(self.model)
        self.context_tokens = None
        if self.session_id:
            self.db.set_model(self.session_id, self.model)  # qualified, else ambiguous on resume

    def make_title(self, prompt, cancel=None):
        """title via small_model (fallback main model). blocking, so call it in a thread.
        cancel set → llm.Cancelled propagates."""
        # reasoning models like to burn the whole budget on thinking, hence generous + fallback
        try:
            text = llm.complete(
                self.cfg, self.cfg.title_model(),
                [{"role": "user", "content": TITLE_PROMPT.format(prompt=prompt[:2000])}],
                max_tokens=2000, cancel=cancel, session=self.session_id,
            )
        except llm.LLMError:
            text = ""
        title = " ".join(text.strip().strip("\"'`#*").split())[:80] or fallback_title(prompt)
        if title and self.session_id:
            self.db.set_title(self.session_id, title)
        return title

    # ------------------------------------------------------------ mcp_magic router

    def route_mcp(self, prompt, emit, cancel):
        """mcp_magic: kleines modell entscheidet anhand des letzten user-prompts, welche mcp
        server eingeschaltet werden (enable-only, manuelles ctrl+o/--mcp gewinnt). nur der
        letzte prompt geht an den router, nicht die history. fail-open: bei fehler bleibt
        der zustand, wie er ist (plus keyword-fallback). Cancelled geht an den caller.
        default ist der router aus (self.magic=False); toggle_magic in der tui schaltet ihn an."""
        if not self.magic or not self.cfg.mcp_magic or self.mcp is None or not (prompt or "").strip():
            return  # magic aus (default) / kein mcp_magic / kein mcp / leerer prompt
        if all(self.mcp.enabled.values()):
            return  # nichts aus → nichts einzuschalten (enabled, aber im error: kein router-job)
        servers_all = self.mcp.candidates(max_tools=None)  # alle tools (fallback + prompt-ranking)
        if not servers_all:
            return
        # prompt-relevante tools zuerst in den llm-prompt: bei 141 tools (nextcloud) landen
        # sonst alphabetisch nur collectives_* in den ersten 12, calendar_* nie
        words = _prompt_words(prompt)

        def kw_rank(text):
            return -sum(1 for w in words if w in text.lower())

        lines = []
        for name, desc, tl in sorted(servers_all, key=lambda e: kw_rank(f"{e[0]} {e[1]}")):
            tl = sorted(tl, key=kw_rank)[:12]  # top 12 nach prompt-relevanz
            lines.append(f"- {name}: {desc}" if desc else f"- {name}:")
            lines += [f"    {t if len(t) <= MAGIC_PROMPT_TOOL_LINE else t[:MAGIC_PROMPT_TOOL_LINE - 1] + '…'}"
                      for t in tl]
        servers_text = "\n".join(lines)
        if len(servers_text) > MAGIC_SERVERS_MAX:
            servers_text = servers_text[:MAGIC_SERVERS_MAX] + "\n    … (truncated)"
        msg = [{"role": "user", "content": MAGIC_PROMPT.format(servers=servers_text, prompt=prompt[:2000])}]

        raw, names = "", None
        # 1. versuch mit reasoning_effort none + temperature 0; 400 (backend kennt einen der
        # params nicht) → nochmal ganz ohne beide
        for extra in ({"effort": "none", "temperature": 0.0}, {}):
            try:
                raw = llm.complete(self.cfg, self.cfg.mcp_magic, msg, max_tokens=MAGIC_MAX_TOKENS,
                                   cancel=cancel, timeout=self.cfg.mcp_magic_timeout,
                                   session=self.session_id, **extra)
                names = parse_magic(raw)
                break
            except llm.LLMError as e:
                if e.status == 400 and extra:
                    continue
                emit({"type": "warning", "warning": f"mcp_magic router failed, keyword fallback: {e}"})
                break

        if names is None:  # router kaputt oder antwort unparsebar → deterministischer fallback
            names = magic_keywords(prompt, servers_all)
        known = {n.lower(): n for n in self.mcp.enabled}
        turned = []
        for n in names:  # eine gültige antwort ([] oder nur schon aktive server) wird respektiert
            real = known.get(n.lower())
            if real and not self.mcp.enabled[real] and real not in turned:
                turned.append(real)
        for n in turned:
            self.mcp.set_enabled(n, True)
        if turned:
            # kurz warten, bis die neuen server verbunden sind, damit ihre tool-schemas
            # schon im ersten request landen (sonst sieht das hauptmodell sie erst ab dem
            # zweiten prompt). bounded, fail-open: bei timeout geht's ohne sie weiter
            self.mcp.wait_ready(turned, min(15, self.cfg.mcp_magic_timeout), cancel)
        emit({"type": "mcp_magic", "enabled": turned, "raw": " ".join(raw.split())[:300]})

    # ------------------------------------------------------------ loop

    def run(self, prompt, on_event=None, cancel=None, images=None):
        """images: [{path, mime, data(base64)}] from images.scan() – become image_url parts
        in the user message (multipart content) and part "image" in the db."""
        emit = on_event or (lambda ev: None)
        if self.ensure_session():
            emit({"type": "session", "id": self.session_id, "new": True, "prompt": prompt})

        self.messages = repair(self.messages)
        content = prompt
        if images:
            content = [{"type": "text", "text": prompt}, *images_mod.image_parts(images)]
        self.messages.append({"role": "user", "content": content})
        parts = [("text", {"text": prompt})]
        if images:
            parts += [("image", im) for im in images]
        self.db.add_message(self.session_id, "user", self.mode, parts)

        try:
            self.route_mcp(prompt, emit, cancel)  # vor dem loop: pro user-message einmal
        except llm.Cancelled:
            emit({"type": "cancelled"})
            return None

        ctx = tools.ToolContext(
            cwd=self.cwd, bash_timeout=self.cfg.bash_timeout,
            max_lines=self.cfg.output_max_lines, max_bytes=self.cfg.output_max_bytes,
            output_dir=self.cfg.tool_output_dir, ask_user=self.ask_user, cancel=cancel,
            sandbox=self.sandbox,
        )

        auto_ok = True            # after a failed auto-compaction don't retry at every step
        overflow_retried = False  # on overflow: compact exactly once and retry
        self.step = 0
        try:
            for step in range(1, self.cfg.max_steps + 1):
                self.step = step
                allowed, extra = self.toolset()
                if auto_ok and self.needs_compaction():
                    try:
                        self.compact(emit, cancel, reason="auto")
                    except (CompactError, llm.LLMError) as e:
                        auto_ok = False
                        emit({"type": "warning", "warning": f"auto-compaction skipped: {e}"})
                emit({"type": "step", "n": step})
                try:
                    message, usage = self._call(allowed, extra, emit, cancel)
                except llm.LLMError as e:
                    if overflow_retried or not compact.is_overflow(e):
                        raise
                    overflow_retried = True
                    try:
                        self.compact(emit, cancel, reason="overflow")
                    except (CompactError, llm.LLMError) as ce:
                        raise llm.LLMError(f"{e}\ncontext full and compaction failed too: {ce}. "
                                           "restart pyrate for a fresh session") from ce
                    emit({"type": "step", "n": step})
                    message, usage = self._call(allowed, extra, emit, cancel)
                self._store_assistant(message)

                if usage:
                    self.last_usage = usage
                    self.context_tokens = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
                    self._usage_len = len(self.messages)
                    emit({"type": "usage", "usage": usage, "context": self.context_tokens})

                calls = message.get("tool_calls") or []
                if not calls:
                    emit({"type": "done"})
                    return message.get("content") or ""

                ctx.env = self.tool_env()  # per step: ctrl+s/ctrl+n/ctrl+p/tab also work mid-run
                ctx.readonly = self.mode == "plan"
                ctx.net = self.net  # ctrl+n, in plan und build wirksam
                self.allow_subagents()
                results = []
                try:
                    for call in calls:
                        if cancel is not None and cancel.is_set():
                            raise llm.Cancelled()
                        # pro call neu: search_tools im selben step lädt, der nächste call darf's nutzen
                        allowed, extra = self.toolset()
                        results.append(self._exec(call, ctx, allowed, extra, emit))
                finally:
                    self._store_results(results)
            emit({"type": "error", "error": f"max_steps ({self.cfg.max_steps}) reached, aborting"})
        except llm.Cancelled:
            emit({"type": "cancelled"})
        except llm.LLMError as e:
            emit({"type": "error", "error": str(e)})
        return None

    def _call(self, allowed, extra, emit, cancel):
        message, usage = None, None
        for ev in llm.stream_chat(
            self.cfg, self.model, self.request_messages(), tools=tools.schemas(allowed, extra),
            effort=self.effort, cancel=cancel, session=self.session_id,
        ):
            if ev["type"] == "done":
                message, usage = ev["message"], ev["usage"]
            else:
                emit(ev)
        emit({"type": "assistant_done", "message": message})
        return message, usage

    def _exec(self, call, ctx, allowed, extra, emit):
        name = call["function"]["name"]
        raw = call["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            args = None
            ok, result = False, f"error: arguments are not valid json ({e}): {raw[:500]}"
        emit({"type": "tool_start", "id": call["id"], "name": name, "args": args if args is not None else raw})
        if args is not None:
            if SEARCH_TOOL in allowed and name in extra and name not in allowed:
                # direkter call auf ein ungeladenes mcp tool (name aus dem manifest): laden statt
                # fehler. toolset() entscheidet weiter über den modus (plan = nur readonly)
                self.loaded_tools.add(name)
                allowed = self.toolset()[0]
            ok, result = tools.run(name, args, ctx, allowed, extra)
        emit({"type": "tool_end", "id": call["id"], "name": name, "ok": ok, "result": result})
        return {"tool_call_id": call["id"], "name": name, "content": result, "ok": ok}

    def _store_assistant(self, message):
        self.messages.append(message)
        parts = []
        if message.get("reasoning_content"):
            sig = None
            for tb in message.get("thinking_blocks") or []:
                sig = tb.get("signature") or sig
            parts.append(("reasoning", {"text": message["reasoning_content"], "signature": sig}))
        if message.get("content"):
            parts.append(("text", {"text": message["content"]}))
        for c in message.get("tool_calls") or []:
            parts.append(("tool_call", {"id": c["id"], "name": c["function"]["name"],
                                        "arguments": c["function"]["arguments"]}))
        self.db.add_message(self.session_id, "assistant", self.mode, parts)

    def _store_results(self, results):
        if not results:
            return
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": r["content"]})
        self.db.add_message(self.session_id, "tool", self.mode, [("tool_result", r) for r in results])
