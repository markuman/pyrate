"""textual tui. just a client for agent.py."""

import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from urllib.parse import unquote, urlparse

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.markup import MarkupError
from textual.message import Message
from textual.screen import ModalScreen
from textual.screen import Screen
from textual.widgets import Collapsible, Footer, Input, Label, Markdown, OptionList, Static, TextArea
from textual.widgets.markdown import MarkdownBlock
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from . import __version__, config, context, images, tools
from .agent import Agent
from .tools import short, tool_label


# ---------------------------------------------------------------- helpers

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
CTX_WARN = 75     # % of context window → yellow, compact (ctrl+k) recommended
CTX_CRIT = 85     # % → red
STALE_AFTER = 15  # s without signs of life → yellow
DEAD_AFTER = 60   # s without signs of life → red
VIEWS = ("normal", "half", "chat")  # ctrl+v: eingabe 15% → 50:50 → ausgeblendet


def esc(s):
    """embed arbitrary text literally in textual markup. do NOT use rich.markup.escape:
    textual treats every '[' as a tag and every '[/' as a closer, rich escapes only real tags.
    a trailing '\\' would escape the next tag ('[b]a\\[/]') → space after it."""
    s = str(s).replace("[", "\\[")
    return s + " " if s.endswith("\\") else s

def dropped_paths(text):
    """drag & drop from the file manager arrives as paste: file:// uris or (quoted) absolute paths.
    returns absolute paths, or None if it wasn't a drop."""
    text = text.strip()
    if not text:
        return None
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if all(x.startswith("file://") for x in lines):
        tokens = [unquote(urlparse(x).path) for x in lines]
    else:
        try:
            tokens = shlex.split(text)
        except ValueError:
            return None
        tokens = [unquote(urlparse(t).path) if t.startswith("file://") else t for t in tokens]
    if not tokens:
        return None
    paths = []
    for t in tokens:
        p = Path(t).expanduser()
        if not p.is_absolute() or not p.exists():
            return None
        paths.append(str(p.resolve()))
    return paths


def clipboard_cmd(primary=False):
    """command for the system clipboard, or None. primary = select → middle mouse button."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return ["wl-copy", *(["--primary"] if primary else [])]
    if os.environ.get("DISPLAY") and shutil.which("xclip"):
        return ["xclip", "-selection", "primary" if primary else "clipboard"]
    if os.environ.get("DISPLAY") and shutil.which("xsel"):
        return ["xsel", "--primary" if primary else "--clipboard", "--input"]
    return None


def system_copy(text, primary=False):
    """textual only copies via osc 52, which e.g. vte terminals (gnome-terminal, ptyxis)
    can't do. hence additionally via the local clipboard tool. all tools fork themselves into
    the background and hold the selection; stdout/stderr closed, so nothing writes into the tui terminal."""
    cmd = clipboard_cmd(primary)
    if cmd is None:
        return False
    try:
        subprocess.run(cmd, input=text.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=2, start_new_session=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def md_table(widget):
    """the markdown table the widget is in (cell, row, table itself), else None."""
    for w in (widget, *widget.ancestors):
        if isinstance(w, MarkdownBlock) and w.name == "table_open":  # name = token-typ
            return w
    return None


def selected_text(selections):
    """like Screen.get_selected_text, but markdown tables come as source text
    (`| a | b |` …) instead of one cell per line. half selected = whole table: without header row
    and `|---|` separator it's no longer valid gfm, and cutting cell-precise would be fiddly.
    the empty line before the table is needed by gfm too, else it sticks to the paragraph before."""
    if not selections:
        return None
    parts = []
    tables = set()
    for widget, selection in selections.items():
        if not widget.is_attached:
            continue
        table = md_table(widget)
        if table is not None and table.source:
            if table not in tables:
                tables.add(table)
                before = "".join(parts)
                if before and not before.endswith("\n\n"):  # leerzeile davor, sonst kein gfm
                    parts.append("\n" if before.endswith("\n") else "\n\n")
                parts.append(table.source.rstrip("\n") + "\n\n")
            continue
        got = widget.get_selection(selection)
        if got is not None:
            parts.extend(got)
    return "".join(parts).rstrip("\n")


def human(n):
    if n is None:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


# ---------------------------------------------------------------- widgets

class PromptInput(TextArea):
    """ctrl+j submits, enter inserts a newline.
    ctrl+j arrives as its own control byte (\x0a) in every terminal; enter-variants
    like shift/ctrl/alt+enter are indistinguishable from enter in legacy terminals."""

    class Submitted(Message):
        def __init__(self, text):
            super().__init__()
            self.text = text

    NEWLINE_KEYS = ("enter", "newline")  # textual meldet enter je nach terminal auch als "newline"
    SUBMIT_KEY = "ctrl+j"
    submit_key = SUBMIT_KEY  # instance: aus config `keys:`, gesetzt in PyrateApp.on_mount

    # callable(key) -> bool, e.g. for the skill popup. True = key swallowed.
    intercept = None

    # careful: textual calls handlers along the whole MRO, TextArea._on_key/_on_paste run
    # after this automatically – unless you call prevent_default(). no super() call, else twice.

    async def _on_key(self, event: events.Key) -> None:
        if self.intercept is not None and self.intercept(event.key):
            event.stop()  # stop: auch keine bindings (TextArea-pfeile, app-escape) mehr
            event.prevent_default()
            return
        if event.key == self.submit_key:
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in self.NEWLINE_KEYS or event.key == self.SUBMIT_KEY:
            # SUBMIT_KEY auch als newline, falls submit per config woanders liegt
            event.stop()
            event.prevent_default()
            self.insert("\n")

    async def _on_paste(self, event: events.Paste) -> None:
        paths = dropped_paths(event.text)
        if paths is None:
            return
        event.stop()
        event.prevent_default()
        start, _ = self.selection
        before = self.document.get_text_range((start[0], 0), start) if start[1] else ""
        sep = " " if before and not before.endswith(" ") else ""
        self.insert(sep + " ".join(shlex.quote(p) for p in paths) + " ")


class SkillPopup(OptionList):
    """suggestions while typing "/…" above the input field. never takes focus,
    controlled from PromptInput (arrows, tab/enter, esc)."""

    can_focus = False


class ToolBlock(Collapsible):
    def __init__(self, label):
        self.label = label
        self.body = Static("", markup=False, classes="tool-body")
        super().__init__(self.body, title=f"⋯ {esc(label)}", collapsed=True, classes="tool running")

    def finish(self, ok, result):
        self.title = f"{'✓' if ok else '✗'} {esc(self.label)}"
        self.body.update(result)
        self.remove_class("running")
        self.add_class("ok" if ok else "err")


class MainScreen(Screen):
    """only so ctrl+c (and the primary selection) yields tables as markdown."""

    def get_selected_text(self):
        return selected_text(self.selections)


# ---------------------------------------------------------------- modals

class PickScreen(ModalScreen):
    """list with filter: typing narrows the list (substring, case-insensitive),
    ↑/↓ + enter operate the filtered list, mouse click works too."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", "cancel"),
        Binding("up", "move(-1)", show=False),
        Binding("down", "move(1)", show=False),
    ]

    def __init__(self, title, items, current=None):
        super().__init__()
        self.title_text = title
        self.items = items
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            yield Input(placeholder="filter …", id="filter")
            yield OptionList(id="opts")

    def on_mount(self):
        self._refilter("")
        self.query_one("#filter", Input).focus()

    def _option(self, item):
        return Option(Text(("● " if item == self.current else "  ") + item), id=item)

    def _refilter(self, needle):
        shown = [i for i in self.items if needle.lower() in i.lower()]
        ol = self.query_one("#opts", OptionList)
        keep = ol.highlighted
        keep_id = ol.get_option_at_index(keep).id if keep is not None and keep < ol.option_count else None
        ol.clear_options()
        ol.add_options([self._option(i) for i in shown])
        if keep_id in shown:
            ol.highlighted = shown.index(keep_id)  # highlight survives filtering
        elif shown:
            ol.highlighted = 0

    @on(Input.Changed, "#filter")
    def _on_filter(self, event):
        self._refilter(event.value)

    @on(Input.Submitted, "#filter")
    def _on_submit(self, event):
        self.action_pick()

    def action_move(self, delta):
        ol = self.query_one("#opts", OptionList)
        if ol.option_count:
            ol.highlighted = (ol.highlighted + delta) % ol.option_count

    def action_pick(self):
        ol = self.query_one("#opts", OptionList)
        if ol.highlighted is not None and ol.highlighted < ol.option_count:
            self.dismiss(ol.get_option_at_index(ol.highlighted).id)

    @on(OptionList.OptionSelected)
    def picked(self, event):
        self.dismiss(event.option.id)


class McpScreen(ModalScreen):
    """toggle mcp servers for this runtime. off = tool schemas no longer go into the request."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", "close"),
        Binding("space", "toggle", "on/off", show=False),
    ]

    def __init__(self, manager, on_toggle):
        super().__init__()
        self.manager = manager
        self.on_toggle = on_toggle

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("mcp servers · enter/space = on/off · esc = close", classes="dialog-title")
            yield OptionList(*[Option(self._prompt(n), id=n) for n in self.manager.clients])
            yield Static(id="mcp-total", classes="mcp-total")

    def on_mount(self):
        self.query_one(OptionList).focus()
        self.refresh_prompts()

    def _prompt(self, name):
        st = {n: (s, k, e) for n, s, k, e in self.manager.summary()}[name]
        state, n, error = st
        on = self.manager.enabled[name]
        tok = self.manager.tool_tokens(name)
        box = Text("[x] " if on else "[ ] ", style="bold green" if on else "dim")
        line = Text.assemble(box, Text(f"{name:<16}", style="bold" if on else "dim"))
        conn = self.manager.state[name]["state"]
        if conn == "error":
            line.append(f"✗ {short(error or 'error', 70)}", style="red" if on else "dim red")
        elif conn == "connecting":
            line.append("connecting …", style="yellow")
        elif conn == "ready":
            info = f"{n} tools · ~{human(tok)} tokens"
            line.append(info if on else f"off · {info}", style="" if on else "dim")
        else:
            line.append("off · not connected yet", style="dim")
        return line

    def refresh_prompts(self):
        ol = self.query_one(OptionList)
        for name in self.manager.clients:
            ol.replace_option_prompt(name, self._prompt(name))
        total = sum(self.manager.tool_tokens(n) or 0 for n in self.manager.clients if self.manager.enabled[n])
        self.query_one("#mcp-total", Static).update(
            Text(f"active tool schemas: ~{human(total)} tokens per request", style="dim"))

    def action_toggle(self):
        opt = self.query_one(OptionList).highlighted_option
        if opt is not None:
            self._toggle(opt.id)

    @on(OptionList.OptionSelected)
    def picked(self, event):
        self._toggle(event.option.id)

    def _toggle(self, name):
        self.on_toggle(name, not self.manager.enabled[name])
        self.refresh_prompts()


class AskScreen(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss(None)", "skip")]

    def __init__(self, question, options):
        super().__init__()
        self.question = question
        self.options = options

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label("question from the model", classes="dialog-title")
            yield Static(self.question, markup=False, classes="question")
            if self.options:
                yield OptionList(*[Option(Text(str(o)), id=str(i)) for i, o in enumerate(self.options)])
            yield Input(placeholder="own answer, enter submits")

    def on_mount(self):
        (self.query_one(OptionList) if self.options else self.query_one(Input)).focus()

    @on(OptionList.OptionSelected)
    def picked(self, event):
        self.dismiss(self.options[int(event.option.id)])

    @on(Input.Submitted)
    def typed(self, event):
        self.dismiss(event.value.strip() or None)


# ---------------------------------------------------------------- app

class PyrateApp(App):
    TITLE = "pyrate"
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #left { width: 1fr; }
    #chat { height: 1fr; padding: 0 1; scrollbar-size-vertical: 1; }
    #input { height: 15vh; min-height: 4; border: round $success; }
    #activity { height: 1; padding: 0 1; display: none; color: $text-muted; }
    #activity.busy { display: block; }
    #activity.stale { color: $warning; }
    #activity.dead { color: $error; text-style: bold; }
    #skillpop { display: none; height: auto; max-height: 10; border: round $accent; background: $panel;
                scrollbar-size-vertical: 1; }
    #skillpop.open { display: block; }
    Screen.view-half #input { height: 1fr; }
    Screen.view-chat #input, Screen.view-chat #skillpop { display: none; }
    #searchbar { height: 1; display: none; background: $panel; }
    #searchbar.open { display: block; }
    #search { width: 1fr; height: 1; border: none; padding: 0 1; background: $panel; }
    #searchinfo { width: auto; padding: 0 1; color: $text-muted; }
    .hit { background: $warning 30%; }
    Screen.mode-plan #input { border: round $warning; }
    #right { width: 34; border-left: solid $success; }
    Screen.mode-plan #right { border-left: solid $warning; }
    #sidebarscroll { height: 1fr; scrollbar-size-vertical: 1; }
    #sidebar { padding: 0 1; }
    #version { height: 1; padding: 0 1; color: $text-muted; }
    #bottombar { dock: bottom; height: 1; }
    #bottombar Footer { dock: none; width: 1fr; }
    #modebadge { width: auto; padding: 0 1; background: $success; color: $background; text-style: bold; }
    Screen.mode-plan #modebadge { background: $warning; }
    .user { margin: 1 0 0 0; padding: 0 1; background: $boost; border-left: thick $accent; }
    .reasoning { color: $text-muted; text-style: italic; max-height: 8; overflow-y: auto; margin: 1 0 0 0; }
    Markdown { margin: 1 0 0 0; padding: 0; }
    .tool { margin: 0; padding: 0; border: none; background: transparent; }
    .tool CollapsibleTitle { color: $text-muted; padding: 0; }
    .tool.err CollapsibleTitle { color: $error; }
    .tool.ok CollapsibleTitle { color: $success-darken-1; }
    .compaction { margin: 1 0 0 0; }
    .compaction CollapsibleTitle { color: $accent; text-style: bold; }
    .tool-body { color: $text-muted; max-height: 30; overflow-y: auto; }
    .error { color: $error; margin: 1 0 0 0; }
    .warning { color: $warning; margin: 1 0 0 0; }
    .note { color: $text-muted; margin: 1 0 0 0; }
    PickScreen, AskScreen, McpScreen { align: center middle; }
    McpScreen .dialog { width: 90; }
    .mcp-total { margin-top: 1; }
    .dialog { width: 70; height: auto; max-height: 80%; border: round $accent; background: $panel; padding: 1 2; }
    .dialog-title { text-style: bold; margin-bottom: 1; }
    .dialog Input { margin-bottom: 1; }
    .dialog OptionList { height: auto; max-height: 20; }
    .question { margin-bottom: 1; }
    """

    ENABLE_COMMAND_PALETTE = False  # ctrl+p belongs to model switching

    # defaults aus config.KEYS; per `keys:` in der config umbelegbar → _apply_keys in __init__.
    # priority: TextArea/Input binden ctrl+w/ctrl+v/ctrl+k selbst. submit: nur das footer-label,
    # die taste selbst behandelt PromptInput.
    BINDINGS = [
        Binding(key, "noop" if action == "submit" else action, label, show=show, priority=prio)
        for action, (key, label, show, prio) in config.KEYS.items()
    ]

    def __init__(self, cfg, db, cwd, session=None, args=None, mcp=None, tmp=None, pod=None):
        super().__init__()
        self.cfg = cfg
        self.keys = dict(cfg.keys)  # action → key, aus `keys:` in der config
        self._apply_keys()
        self.db = db
        self.cwd = cwd
        self.args = args
        self.tmp = tmp
        self.pod = pod
        self.mcp = mcp
        self._mcp_errors_shown = set()
        self.session_row = session
        self.agent = self._new_agent(session)
        self.busy = False
        self.cancel_event = None
        self.closing = threading.Event()  # app closing → also cancel title calls
        self.stream = None
        self.md = None
        self.reasoning = None
        self.reasoning_text = ""
        self.blocks = {}
        self.active_tool = None
        # activity display: phase + timestamp of the last sign of life from the agent thread
        self.phase = ""
        self.run_started = 0.0
        self.last_activity = 0.0
        self.phase_started = 0.0
        self.ticker = None
        self.view = VIEWS[0]
        self.hit = None          # current search hit (widget)
        self.hit_needle = None   # the needle for it, new term = from the start

    def get_default_screen(self):
        return MainScreen(id="_default")

    def _apply_keys(self):
        """`keys:` aus der config auf die bindings anwenden. die defaults stehen schon in
        BINDINGS – pro action wird nur der default-key gegen den konfigurierten getauscht."""
        km = self._bindings.key_to_bindings
        for action, (dflt, label, show, prio) in config.KEYS.items():
            key = self.keys[action]
            if key == dflt:
                continue
            bound = "noop" if action == "submit" else action
            if dflt in km:  # nur unser default-binding dieser action entfernen (app-defaults wie ctrl+q bleiben)
                km[dflt] = [b for b in km[dflt] if b.action != bound]
                if not km[dflt]:
                    del km[dflt]
            km[key] = [*km.get(key, []), Binding(key, bound, label, show=show, priority=prio)]  # kopieren: listen sind shared mit _merged_bindings

    def _new_agent(self, session):
        a = self.args
        return Agent(
            self.cfg, self.db, self.cwd,
            session_id=session["id"] if session else None,
            model=(a.model if a else None) or (session or {}).get("model"),
            mode=a.mode if a else None,
            effort=a.effort if a else None,
            ask_user=self.ask_user_blocking,
            mcp=self.mcp,
            tmp=self.tmp,
            podman=self.pod,
        )

    # ------------------------------------------------------------ layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield VerticalScroll(id="chat")
                yield Static(id="activity")
                with Horizontal(id="searchbar"):
                    yield Input(id="search", placeholder="search chat · enter/ctrl+w = next (older) · esc = close")
                    yield Static(id="searchinfo")
                yield SkillPopup(id="skillpop")
                yield PromptInput(id="input", soft_wrap=True, show_line_numbers=False,
                                  tab_behavior="focus", placeholder="prompt · ctrl+j submits · enter = newline · /skill")
            with Vertical(id="right"):
                with VerticalScroll(id="sidebarscroll"):
                    yield Static(id="sidebar")
                yield Static(f"pyrate {__version__}", id="version")
        with Horizontal(id="bottombar"):
            yield Static(id="modebadge")
            yield Footer()

    def on_mount(self):
        self.chat = self.query_one("#chat", VerticalScroll)
        self.input = self.query_one("#input", PromptInput)
        self.input.submit_key = self.keys["submit"]
        self.input.intercept = self.skillpop_key
        self.skillpop = self.query_one("#skillpop", SkillPopup)
        self.sidebar = self.query_one("#sidebar", Static)
        self.badge = self.query_one("#modebadge", Static)
        self.activity = self.query_one("#activity", Static)
        self.searchbar = self.query_one("#searchbar", Horizontal)
        self.search = self.query_one("#search", Input)
        self.searchinfo = self.query_one("#searchinfo", Static)
        self.chat.anchor()
        for w in self.cfg.warnings + self.agent.context_warnings:
            self.notify(w, severity="warning", timeout=10, markup=False)
        if self.session_row:
            self.render_history()
        self.apply_mode()
        if self.mcp is not None:
            self.mcp.on_change = self.mcp_changed_from_thread
            self.mcp_changed()  # was vor dem mount schon passiert ist (z.b. server sofort gescheitert)
        self.input.focus()

    def on_unmount(self):
        """app is closing (ctrl+d, crash, …): cancel running threads. otherwise python waits on
        exit for the agent – which continues headless with tool calls without ui or hangs in
        ask_user forever."""
        self.closing.set()
        if self.cancel_event is not None:
            self.cancel_event.set()

    # ------------------------------------------------------------ sidebar & mode

    def apply_mode(self):
        plan = self.agent.mode == "plan"
        self.screen.set_class(plan, "mode-plan")
        self.screen.set_class(not plan, "mode-build")
        self.badge.update(self.agent.mode.upper())
        self.refresh_sidebar()

    def refresh_sidebar(self):
        a = self.agent
        title = self.session_row.get("title") if self.session_row else None
        limit = a.context_limit()
        if a.context_tokens is None:
            used, prefix = a.estimated_tokens(self.input.text if hasattr(self, "input") else ""), "~"
        else:
            used, prefix = a.context_tokens, ""
        ctx = f"{prefix}{human(used)}"
        if limit:
            pct = min(100, used * 100 // limit)
            bar_w = 20
            filled = bar_w * pct // 100
            color = "green" if pct < CTX_WARN else "yellow" if pct < CTX_CRIT else "red"
            ctx = (f"[{color}]{ctx} / {human(limit)} ({pct}%)[/]\n"
                   f"[{color}]{'█' * filled}[/][dim]{'░' * (bar_w - filled)}[/]")
            if pct >= CTX_WARN:
                ctx += f"\n[{color}]→ ctrl+k (compact) recommended[/]"
        allowed = set(a.tool_names())
        tool_lines = []
        for name in tools.enabled(self.cfg.tools):
            if name in allowed:
                tool_lines.append(f"[green]●[/] {name}")
            else:
                tool_lines.append(f"[dim]○ {name}[/]")
        mcp_lines = []
        if self.mcp is not None:
            extra = self.mcp.tools()
            for name, state, n, _ in self.mcp.summary():
                if state == "ready":
                    usable = sum(1 for t in extra if t.startswith(f"mcp__{name}__") and t in allowed)
                    count = f"{usable}/{n}" if usable != n else str(n)
                    mcp_lines.append(f"[green]●[/] {esc(name)} [dim]({count})[/]")
                elif state == "off":
                    mcp_lines.append(f"[dim]○ {esc(name)} (off)[/]")
                elif state == "error":
                    mcp_lines.append(f"[red]✗[/] {esc(name)}")
                else:
                    mcp_lines.append(f"[yellow]⋯[/] {esc(name)}")
        # zauberstab hinter mcp, solange der mcp_magic-router an ist (toggle_magic schaltet um)
        mcp_head = "[b]mcp[/]" + (" 🪄" if self.cfg.mcp_magic and self.agent.magic else "")
        if self.active_tool:
            status = f"[yellow]⋯[/] {esc(self.active_tool)}"
        elif self.busy:
            status = f"[yellow]{esc(self.phase or 'running')} …[/]"
        else:
            status = "[dim]idle[/]"
        max_steps = self.cfg.max_steps
        steps = f"{a.step}/{max_steps}"
        if not self.busy:
            steps = f"[dim]{steps}[/]"  # state from the last run
        elif a.step * 100 >= max_steps * 80:
            steps = f"[yellow]{steps}[/]"
        lines = [
            f"[b]{esc(title or 'new session')}[/]",
            f"[dim]{esc(a.session_id or '-')}[/]",
            f"[dim]{esc(self.cwd)}[/]",
            "",
            "[b]model[/]",
            esc(a.model),
            "",
            f"[b]{'mode':<10}[/]{'[yellow]plan[/]' if a.mode == 'plan' else '[green]build[/]'}",
            f"[b]{'effort':<10}[/]{esc(a.effort)}",
            f"[b]{'subagents':<10}[/]{a.subagents if a.subagents else '[dim]off[/]'}"
            f"{' [dim](build only)[/]' if a.subagents and a.mode == 'plan' else ''}",
            f"[b]{'tcp':<10}[/]{'[green]on[/]' if a.net else '[dim]off[/]'} [dim](ctrl+n)[/]",
            f"[b]{'steps':<10}[/]{steps}",
            "",
            "[b]context[/]",
            ctx,
            "",
            "[b]tools[/]",
            *tool_lines,
            *(["", mcp_head, *mcp_lines] if mcp_lines else []),
            "",
            "[b]status[/]",
            status,
        ]
        extras = []
        for ins in a.instructions:
            extras.append(f"[dim]{esc(ins.path.name)} ({ins.scope})[/]")
        if a.skills:
            n_proj = sum(1 for s in a.skills if s.scope == "project")
            extras.append(f"[dim]{len(a.skills)} skills" + (f", {n_proj} project" if n_proj else "") + "[/]")
        if a.podman is not None and not a.podman_ok():
            extras.append("[red]✗ podman[/] [dim](service dead)[/]")
        elif a.podman is not None:
            extras.append("[yellow]podman[/] [dim](outside the sandbox" + (", build only)" if a.mode == "plan" else ")") + "[/]")
        if extras:
            lines += ["", "[b]context[/]", *extras]
        markup = "\n".join(lines)
        try:
            self.sidebar.update(markup)
        except MarkupError:  # thanks to esc() should never happen – but never kill the session again
            self.sidebar.update(Text(markup))

    def mcp_changed_from_thread(self):
        try:
            self.call_from_thread(self.mcp_changed)
        except RuntimeError:
            pass  # app beendet sich gerade (oder wir sind schon im ui-thread)

    def mcp_changed(self):
        for name, state, _, error in self.mcp.summary():
            if state == "error" and name not in self._mcp_errors_shown:
                self._mcp_errors_shown.add(name)
                self.notify(f"mcp {name}: {error}", severity="error", timeout=15, markup=False)
        self.refresh_sidebar()
        if isinstance(self.screen, McpScreen):
            self.screen.refresh_prompts()  # catch up "connecting …" → "6 tools" live

    @on(TextArea.Changed, "#input")
    def draft_changed(self, _):
        self.update_skillpop()
        if self.agent.context_tokens is None:
            self.refresh_sidebar()

    # ------------------------------------------------------------ skill-popup

    @property
    def skillpop_open(self):
        return self.skillpop.has_class("open")

    def update_skillpop(self):
        """runs only on text change. so esc closes until the next keypress."""
        matches = context.complete_skill(self.input.text, self.agent.skills)
        if not matches:
            self.skillpop.remove_class("open")
            return
        self.skillpop.clear_options()
        self.skillpop.add_options([self._skillpop_option(s) for s in matches])
        self.skillpop.highlighted = 0
        self.skillpop.add_class("open")

    @staticmethod
    def _skillpop_option(s):
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(f"/{s.name}", style="bold")
        if s.scope == "project":
            line.append(" (project)", style="dim italic")
        line.append(f"  {short(s.description, 200)}", style="dim")
        return Option(line, id=s.name)

    def skillpop_key(self, key):
        """keys in the input field while the popup is open. True = swallowed."""
        if not self.skillpop_open:
            return False
        pop = self.skillpop
        if key in ("up", "down"):
            pop.action_cursor_up() if key == "up" else pop.action_cursor_down()
            return True
        if key == "escape":
            pop.remove_class("open")
            return True
        if key == PromptInput.SUBMIT_KEY:
            pop.remove_class("open")
            return False  # submit geht durch
        if key == "enter":
            opt = pop.highlighted_option
            if opt is None or self.input.text == f"/{opt.id}":
                pop.remove_class("open")
                return False  # already typed out → submit normally
            self.accept_skill(opt.id)
            return True
        return False

    def accept_skill(self, name):
        self.input.clear()
        self.input.insert(f"/{name} ")
        self.skillpop.remove_class("open")
        self.input.focus()

    @on(OptionList.OptionSelected, "#skillpop")
    def skillpop_clicked(self, event):
        event.stop()
        self.accept_skill(event.option.id)

    # ------------------------------------------------------------ actions

    def action_toggle_mode(self):
        if isinstance(self.screen, ModalScreen):
            self.screen.focus_next()
            return
        if self.skillpop_open and self.focused is self.input:
            opt = self.skillpop.highlighted_option
            if opt is not None:
                self.accept_skill(opt.id)
            return
        self.agent.mode = "build" if self.agent.mode == "plan" else "plan"
        self.apply_mode()

    def action_pick_model(self):
        if isinstance(self.screen, ModalScreen):
            return

        def done(model):
            if model and model != self.agent.model:
                self.agent.set_model(model)
                self.notify(f"model: {model}", markup=False)
                self.refresh_sidebar()

        self.push_screen(PickScreen("pick model", self.cfg.model_ids(), self.agent.model), done)

    def check_action(self, action, parameters):
        if action == "mcp_panel" and self.mcp is None:
            return False  # without mcp server: key out of the footer
        if action == "toggle_magic" and (self.mcp is None or not self.cfg.mcp_magic):
            return False  # without mcp_magic in the config: key out of the footer
        return True

    def action_mcp_panel(self):
        if isinstance(self.screen, ModalScreen) or self.mcp is None:
            return
        self.push_screen(McpScreen(self.mcp, self.toggle_mcp))

    def toggle_mcp(self, name, on):
        self._mcp_errors_shown.discard(name)  # report errors again after a new attempt
        self.mcp.set_enabled(name, on)
        self.notify(f"mcp {name}: {'on' if on else 'off'}", timeout=2, markup=False)
        self.refresh_sidebar()

    def action_cycle_effort(self):
        if isinstance(self.screen, ModalScreen):
            return
        efforts = config.EFFORTS
        self.agent.effort = efforts[(efforts.index(self.agent.effort) + 1) % len(efforts)]
        self.notify(f"reasoning effort: {self.agent.effort}", timeout=2)
        self.refresh_sidebar()

    def action_toggle_subagents(self):
        """off ↔ on. on = configured value, with config 0 just 1."""
        if isinstance(self.screen, ModalScreen):
            return
        a = self.agent
        a.subagents = 0 if a.subagents else max(self.cfg.subagents, 1)
        self.notify(f"subagents: {a.subagents or 'off'}", timeout=2)
        self.refresh_sidebar()

    def action_toggle_net(self):
        """tcp on/off, in plan and build mode. default per mode: plan off, build on."""
        if isinstance(self.screen, ModalScreen):
            return
        a = self.agent
        a.net = not a.net
        self.notify(f"tcp: {'on' if a.net else 'off'}", timeout=2)
        self.refresh_sidebar()

    def action_toggle_magic(self):
        """mcp_magic router on/off for this session (only when mcp_magic is configured)."""
        if isinstance(self.screen, ModalScreen):
            return
        self.agent.magic = not self.agent.magic
        self.notify(f"mcp_magic: {'on' if self.agent.magic else 'off'}", timeout=2, markup=False)
        self.refresh_sidebar()

    def action_noop(self):
        pass

    def action_cycle_view(self):
        if isinstance(self.screen, ModalScreen):
            return
        self.view = VIEWS[(VIEWS.index(self.view) + 1) % len(VIEWS)]
        for v in VIEWS:
            self.screen.set_class(v == self.view, f"view-{v}")
        if self.focused is not self.search:
            (self.chat if self.view == "chat" else self.input).focus()

    # ------------------------------------------------------------ search (ctrl+w, like nano)
    # dumb: on every enter search everything again, no index. hit = whole paragraph/table/
    # codeblock or prompt or tool line, not the single word. bottom to top.

    def action_search(self):
        if isinstance(self.screen, ModalScreen):
            return
        if self.focused is self.search:
            self.search_next()
            return
        self.searchbar.add_class("open")
        self.search.focus()
        self.search.select_all()  # old term still there: enter repeats, typing replaces

    def search_targets(self):
        """(widget, text) for everything in the chat, oldest first. tool outputs stay out."""
        for w in self.chat.children:
            if isinstance(w, Markdown):
                for block in w.children:
                    yield block, getattr(block, "source", None) or ""
            elif isinstance(w, ToolBlock):
                yield w, w.label
            elif isinstance(w, Collapsible):
                yield w, str(w.title)
            else:
                yield w, str(w.render())

    def search_next(self):
        needle = self.search.value.strip().lower()
        if not needle:
            return
        hits = [w for w, text in self.search_targets() if needle in text.lower()]
        hits.reverse()  # newest first
        if not hits:
            self.mark_hit(None)
            self.searchinfo.update("no hit")
            return
        i = 0
        if needle == self.hit_needle and self.hit in hits:
            i = (hits.index(self.hit) + 1) % len(hits)
        self.hit_needle = needle
        self.mark_hit(hits[i])
        self.searchinfo.update(f"{i + 1}/{len(hits)}")

    def mark_hit(self, widget):
        if self.hit is not None:
            self.hit.remove_class("hit")
        self.hit = widget
        if widget is not None:
            widget.add_class("hit")
            self.chat.release_anchor()  # else new output pulls down again immediately
            self.chat.scroll_to_widget(widget, animate=False, top=True)

    def close_search(self):
        self.searchbar.remove_class("open")
        self.searchinfo.update("")
        self.mark_hit(None)
        self.hit_needle = None
        (self.chat if self.view == "chat" else self.input).focus()

    @on(Input.Submitted, "#search")
    def search_submitted(self, event):
        event.stop()
        self.search_next()

    # ------------------------------------------------------------ clipboard
    # select with the mouse (chat, sidebar) or shift+arrows in the input field, then ctrl+c.

    def copy_to_clipboard(self, text):
        super().copy_to_clipboard(text)  # osc 52: works over ssh too, but not in every terminal
        if system_copy(text):
            self.notify(f"copied ({len(text)} chars)", timeout=1.5)
        else:  # whether osc 52 arrived, you never learn
            self.notify(f"copied via osc 52 only ({len(text)} chars) – not every terminal can", timeout=3)

    @on(events.TextSelected)
    def text_selected(self, _):
        """selection finished → primary selection, middle mouse button pastes then (like in the terminal)."""
        text = self.screen.get_selected_text()
        if text:
            system_copy(text, primary=True)

    def action_cancel(self):
        if self.focused is self.search:
            self.close_search()
            return
        if self.busy and self.cancel_event:
            self.cancel_event.set()
            self.notify("cancelling …", timeout=2)

    # ------------------------------------------------------------ chat-rendering

    async def mount_chat(self, widget):
        await self.chat.mount(widget)

    def render_history(self):
        results = {}
        raw = self.db.load_raw(self.agent.session_id)
        for m in raw:
            for p in m["parts"]:
                if p["type"] == "tool_result":
                    results[p["data"]["tool_call_id"]] = p["data"]
        widgets = []
        for m in raw:
            for p in m["parts"]:
                d = p["data"]
                if p["type"] == "compaction":
                    widgets.append(self.compaction_block(d["summary"], d.get("reason", "?"),
                                                         d.get("tokens_before")))
                elif m["role"] == "user" and p["type"] == "text":
                    shown = context.skill_label(d["text"]) or d["text"]  # don't show skill content again
                    widgets.append(Static(Text(shown), classes="user"))
                elif m["role"] == "user" and p["type"] == "image":
                    widgets.append(Static(Text(f"📷 {d.get('path', '?')}"), classes="user"))
                elif p["type"] == "text" and d.get("text", "").strip():
                    widgets.append(Markdown(d["text"]))
                elif p["type"] == "tool_call":
                    try:
                        args = json.loads(d.get("arguments") or "{}")
                    except ValueError:
                        args = d.get("arguments")
                    block = ToolBlock(tool_label(d["name"], args))
                    widgets.append(block)
                    r = results.get(d["id"])
                    if r:
                        block.call_after_refresh(block.finish, r.get("ok", True), r.get("content", ""))
        if widgets:
            self.chat.mount_all(widgets)
            self.chat.mount(Static(f"— session {self.agent.session_id} continued —", classes="note"))

    @staticmethod
    def compaction_block(summary, reason, before=None):
        label = {"auto": "auto", "overflow": "after overflow", "manual": "manual"}.get(reason, reason)
        size = f", before ~{human(before)} tokens" if before else ""
        return Collapsible(Markdown(summary), title=f"⇊ context compacted ({esc(label)}{size})",
                           collapsed=True, classes="tool compaction")

    @on(PromptInput.Submitted)
    async def submitted(self, event):
        text = event.text.strip()
        if not text:
            return
        if self.busy:
            self.notify("still running. esc cancels.", severity="warning")
            return
        try:
            prompt = context.expand_skill(text, self.agent.skills)
        except context.SkillError as e:
            self.notify(str(e), severity="error", timeout=8, markup=False)
            return
        imgs, skipped = images.scan(text)  # image paths in the prompt → attach to the model
        for path, reason in skipped:
            self.notify(f"image not attached: {path} ({reason})", severity="warning", timeout=8)
        self.input.clear()
        self.chat.scroll_end(animate=False)  # after search/scrolling up: let the answer follow again
        shown = text + (f"\n📷 {len(imgs)} image(s) attached" if imgs else "")
        await self.mount_chat(Static(Text(shown), classes="user"))
        self.busy = True
        self.cancel_event = threading.Event()
        self.start_activity()
        self.refresh_sidebar()
        self.run_worker(partial(self.agent.run, prompt, self.emit_from_thread, self.cancel_event,
                                images=imgs),
                        thread=True, group="agent", name="agent")

    def action_compact(self):
        if isinstance(self.screen, ModalScreen):
            return
        if self.busy:
            self.notify("still running. esc cancels.", severity="warning")
            return
        if not self.agent.session_id:
            self.notify("no session yet – nothing to compact", severity="warning")
            return
        self.busy = True
        self.cancel_event = threading.Event()
        self.start_activity()
        self.set_phase("compacting context")
        self.refresh_sidebar()
        self.run_worker(partial(self.agent.run_compact, self.emit_from_thread, self.cancel_event),
                        thread=True, group="agent", name="agent")

    def emit_from_thread(self, ev):
        try:
            self.call_from_thread(self.handle_event, ev)
        except RuntimeError:
            pass  # app wird gerade beendet

    async def close_stream(self):
        if self.stream is not None:
            await self.stream.stop()
            self.stream = None
        self.md = None

    # ------------------------------------------------------------ activity display
    # the spinner runs in the ui loop, independent of the backend. "last sign of life" counts:
    # how long the agent thread hasn't reported anything – that's how you see dead vs. slow.

    def start_activity(self):
        now = time.monotonic()
        self.run_started = self.last_activity = self.phase_started = now
        self.phase = "waiting for backend"
        self.activity.add_class("busy")
        if self.ticker is None:
            self.ticker = self.set_interval(0.1, self.tick_activity)
        self.tick_activity()

    def stop_activity(self):
        if self.ticker is not None:
            self.ticker.stop()
            self.ticker = None
        self.activity.remove_class("busy", "stale", "dead")

    def set_phase(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.phase_started = time.monotonic()
            self.refresh_sidebar()

    def tick_activity(self):
        now = time.monotonic()
        frame = SPINNER[int(now * 10) % len(SPINNER)]
        total = now - self.run_started
        stale = dead = False
        if self.phase.startswith("tool"):
            # silence here comes from the tool (e.g. long pytest), not the backend
            text = f"{frame} {self.phase} · running for {now - self.phase_started:.0f}s · total {total:.0f}s"
        else:
            idle = now - self.last_activity
            stale, dead = idle >= STALE_AFTER, idle >= DEAD_AFTER
            text = f"{frame} {self.phase} · {total:.1f}s · last sign of life {idle:.1f}s ago"
            if dead:
                text += " · backend not answering"
            elif stale:
                text += " · backend sluggish"
        self.activity.set_class(stale and not dead, "stale")
        self.activity.set_class(dead, "dead")
        self.activity.update(Text(text + " · esc cancels"))

    PHASES = {"step": "waiting for backend", "reasoning": "thinking", "text": "writing",
              "tool_end": "waiting for backend", "compact_start": "compacting context",
              "compact_delta": "compacting context"}

    async def handle_event(self, ev):
        t = ev["type"]
        self.last_activity = time.monotonic()
        if t in self.PHASES:
            self.set_phase(self.PHASES[t])
        elif t == "tool_start":
            self.set_phase(f"tool {tool_label(ev['name'], ev['args'])}")
        if t == "session":
            self.session_row = self.db.get_session(ev["id"])
            self.refresh_sidebar()
            prompt = context.skill_label(ev["prompt"]) or ev["prompt"]  # title without skill content
            self.run_worker(partial(self.title_thread, self.agent, prompt), thread=True, group="title")
        elif t == "step":
            await self.close_stream()
            self.reasoning = None
            self.reasoning_text = ""
            self.refresh_sidebar()
        elif t == "reasoning":
            if self.reasoning is None:
                self.reasoning = Static("", markup=False, classes="reasoning")
                await self.mount_chat(self.reasoning)
            self.reasoning_text += ev["text"]
            self.reasoning.update(self.reasoning_text[-4000:])
        elif t == "text":
            if self.md is None:
                chunk = ev["text"].lstrip()
                if not chunk:
                    return
                self.md = Markdown("")
                await self.mount_chat(self.md)
                self.stream = Markdown.get_stream(self.md)
                await self.stream.write(chunk)
            else:
                await self.stream.write(ev["text"])
        elif t == "assistant_done":
            await self.close_stream()
        elif t == "tool_start":
            label = tool_label(ev["name"], ev["args"])
            block = ToolBlock(label)
            self.blocks[ev["id"]] = block
            self.active_tool = label
            await self.mount_chat(block)
            self.refresh_sidebar()
        elif t == "tool_end":
            block = self.blocks.pop(ev["id"], None)
            if block:
                block.finish(ev["ok"], ev["result"])
            self.active_tool = None
            self.refresh_sidebar()
        elif t == "usage":
            self.refresh_sidebar()
        elif t == "compact_done":
            await self.close_stream()
            await self.mount_chat(self.compaction_block(ev["summary"], ev["reason"], ev["before"]))
            self.notify(f"context compacted: ~{human(ev['before'])} → ~{human(ev['after'])} tokens", timeout=4)
            self.refresh_sidebar()
        elif t == "mcp_magic":
            if ev["enabled"]:  # kein match → still, sonst nervt's bei jedem prompt (raw: headless -v)
                self.notify(f"mcp_magic: enabled {', '.join(ev['enabled'])}", timeout=4, markup=False)
                self.refresh_sidebar()
        elif t == "warning":
            await self.mount_chat(Static(Text(f"warning: {ev['warning']}"), classes="warning"))
        elif t == "error":
            await self.mount_chat(Static(Text(f"error: {ev['error']}"), classes="error"))
        elif t == "cancelled":
            await self.mount_chat(Static("— cancelled —", classes="note"))

    def title_thread(self, agent, prompt):
        try:
            title = agent.make_title(prompt, cancel=self.closing)
        except Exception:  # noqa: BLE001 – also llm.Cancelled on shutdown
            return
        if agent is self.agent and title:
            def apply():
                if self.session_row:
                    self.session_row["title"] = title
                self.refresh_sidebar()
            try:
                self.call_from_thread(apply)
            except RuntimeError:
                pass  # app is shutting down

    @on(Worker.StateChanged)
    async def worker_changed(self, event):
        if event.worker.name != "agent":
            return
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            await self.close_stream()
            self.busy = False
            self.active_tool = None
            self.stop_activity()
            if event.state == WorkerState.ERROR:
                await self.mount_chat(Static(Text(f"internal error: {event.worker.error!r}"), classes="error"))
            self.refresh_sidebar()

    # ------------------------------------------------------------ ask_user (from the agent thread)

    def ask_user_blocking(self, question, options):
        """None = no answer (skipped, cancelled, app closing)."""
        done = threading.Event()
        box = {}
        screen = AskScreen(question, options)

        def cb(result):
            box["answer"] = result
            done.set()

        def close():
            if screen.is_active:  # maybe just answered or app is shutting down → don't pop anything foreign
                self.pop_screen()

        try:
            self.call_from_thread(self.push_screen, screen, cb)
            while not done.wait(0.2):
                if self.cancel_event is not None and self.cancel_event.is_set():
                    self.call_from_thread(close)
                    return None
        except RuntimeError:  # app is shutting down
            return None
        return box.get("answer")


TERM_TITLE = "Ay! pyrate, a hacker's best wingman 🏴‍☠️☠️"


@contextlib.contextmanager
def terminal_title(title, stream=None):
    """set the terminal window title (osc 2) for the duration. first push the old one onto the
    title stack (xterm `CSI 22;2t`), restore it afterwards (`CSI 23;2t`). terminals without a
    stack ignore that; then the shell prompt usually sets the title again anyway."""
    stream = stream or sys.__stdout__
    tty = stream is not None and stream.isatty()
    if tty:
        stream.write(f"\x1b[22;2t\x1b]2;{title}\x07")
        stream.flush()
    try:
        yield
    finally:
        if tty:
            stream.write("\x1b[23;2t")
            stream.flush()


def run_tui(cfg, db, cwd, session, args, mcp=None, tmp=None, pod=None):
    app = PyrateApp(cfg, db, cwd, session, args, mcp, tmp, pod)
    with terminal_title(TERM_TITLE):
        app.run()
    if app.agent.session_id:
        print(f"session {app.agent.session_id} · resume with: pyrate -c")
    return 0
