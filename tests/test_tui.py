import asyncio
import threading
from pathlib import Path

from textual import events
from textual.markup import to_content
from textual.widgets import Markdown, OptionList

from pyrate.db import DB
from pyrate.tui import PickScreen, PromptInput, PyrateApp, dropped_paths, esc


def test_terminal_title():
    import io

    from pyrate.tui import terminal_title

    class Tty(io.StringIO):
        def isatty(self):
            return True

    out = Tty()
    with terminal_title("ahoi", out):
        assert out.getvalue() == "\x1b[22;2t\x1b]2;ahoi\x07"
    assert out.getvalue().endswith("\x1b[23;2t")
    pipe = io.StringIO()  # kein terminal: keine escapes in umgeleitete ausgabe
    with terminal_title("ahoi", pipe):
        pass
    assert pipe.getvalue() == ""


def test_dropped_paths(tmp_path):
    a = tmp_path / "a b.txt"
    a.write_text("x")
    b = tmp_path / "c.txt"
    b.write_text("x")
    assert dropped_paths(f"'{a}' {b}") == [str(a), str(b)]
    assert dropped_paths(str(a).replace(" ", "\\ ")) == [str(a)]
    assert dropped_paths(f"file://{str(a).replace(' ', '%20')}\nfile://{b}") == [str(a), str(b)]
    assert dropped_paths("einfach text") is None
    assert dropped_paths("c.txt") is None  # relativ = kein drop
    assert dropped_paths(f"{a}.gibtsnicht") is None


async def _wait_idle(app, pilot, secs=10):
    for _ in range(secs * 20):
        await pilot.pause(0.05)
        if not app.busy:
            return


def test_tui_flow(backend, make_cfg, tmp_path):
    (tmp_path / "f.txt").write_text("inhalt")
    backend.script = [
        {"reasoning": "denk", "tool_calls": [{"name": "read", "arguments": {"path": "f.txt"}}]},
        {"text": "# antwort\n\ndie datei sagt **inhalt**"},
    ]
    backend.by_model["tiny"] = [{"text": "Ein Titel"}]
    cfg = make_cfg(models=["fake-model", "other-model"], small_model="tiny")

    async def run():
        app = PyrateApp(cfg, DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert "~" in str(app.sidebar.render())  # schaetzung vor dem ersten request
            assert app.screen.has_class("mode-build")

            await pilot.press("tab")
            assert app.agent.mode == "plan" and app.screen.has_class("mode-plan")
            await pilot.press("tab")
            assert app.agent.mode == "build"

            await pilot.press("ctrl+t")
            assert app.agent.effort == "low"
            await pilot.press("ctrl+t", "ctrl+t", "ctrl+t", "ctrl+t", "ctrl+t")
            assert app.agent.effort == "default"

            assert app.agent.subagents == 0  # config-default
            await pilot.press("ctrl+s")
            assert app.agent.subagents == 1  # config 0 → an heisst 1
            assert "SUBAGENTS:" in app.agent.system_prompt()
            await pilot.press("ctrl+s")
            assert app.agent.subagents == 0

            await pilot.press("ctrl+p")
            await pilot.pause()
            assert isinstance(app.screen, PickScreen)  # nicht die command-palette
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.agent.model.startswith("other-model@127.0.0.1:")  # label = modell@backend

            inp = app.query_one(PromptInput)
            await pilot.press("h", "enter", "i")
            assert inp.text == "h\ni"  # enter = newline
            inp.clear()

            # eingabe hat feste hoehe: 15% von 40 zeilen = 6, auch mit viel text
            assert inp.outer_size.height == 6
            inp.insert("\n".join(f"zeile {i}" for i in range(30)))
            await pilot.pause()
            assert inp.outer_size.height == 6
            inp.clear()

            inp.post_message(events.Paste(str(tmp_path / "f.txt")))
            await pilot.pause()
            assert inp.text.strip() == str(tmp_path / "f.txt")
            inp.clear()

            # wie echt vom terminal: paste geht an die app, die forwardet ans fokussierte widget
            inp.focus()
            app.post_message(events.Paste("normaler text"))  # darf nicht exists twice landen
            await pilot.pause()
            assert inp.text == "normaler text"
            inp.clear()

            inp.insert("lies f.txt")
            await pilot.press("ctrl+j")
            await _wait_idle(app, pilot)
            assert inp.text == ""
            md = app.query(Markdown).last()
            assert "die datei sagt **inhalt**" in md.source
            blocks = list(app.query("ToolBlock"))
            assert blocks and blocks[0].title.startswith("✓ read")
            assert "other-model" in [r["body"]["model"] for r in backend.requests]
            await pilot.pause(0.5)
            side = str(app.sidebar.render())
            assert "~" not in side.split("context")[1].split("tools")[0]  # jetzt echter wert
            assert "Ein Titel" in side
            title_req = next(r for r in backend.requests if r["body"]["model"] == "tiny")
            assert "lies f.txt" in title_req["body"]["messages"][-1]["content"]
    asyncio.run(run())


def test_prompt_enter_newline_ctrl_j_submit(make_cfg, tmp_path):
    """enter = newline im input, ctrl+j = Submitted (submit)."""

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            inp = app.query_one(PromptInput)
            got = []
            orig_post = inp.post_message

            def spy(msg, *a, **k):  # Submitted abfangen, damit kein agent-run startet
                if isinstance(msg, PromptInput.Submitted):
                    got.append(msg.text)
                    return True
                return orig_post(msg, *a, **k)

            inp.post_message = spy
            await pilot.press("h", "enter", "i")
            assert inp.text == "h\ni" and not got  # enter = newline, kein submit
            await pilot.press("ctrl+j")
            assert got == ["h\ni"] and inp.text == "h\ni"  # text bleibt bis zum handler
    asyncio.run(run())


def test_tui_resume_renders_history(backend, make_cfg, tmp_path):
    cfg = make_cfg()
    db = DB(":memory:")
    sid = db.create_session(str(tmp_path), "fake-model")
    db.add_message(sid, "user", "build", [("text", {"text": "alte frage"})])
    db.add_message(sid, "assistant", "build", [("text", {"text": "alte antwort"})])

    async def run():
        app = PyrateApp(cfg, db, str(tmp_path), db.get_session(sid))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.query(Markdown).last().source == "alte antwort"
            assert len(app.agent.messages) == 2
    asyncio.run(run())


def test_activity_indicator(backend, make_cfg, tmp_path, monkeypatch):
    """spinner + 'letztes lebenszeichen' sichtbar, solange die antwort aussteht; eskaliert bei stille."""
    import pyrate.tui as tui
    monkeypatch.setattr(tui, "STALE_AFTER", 0.3)
    monkeypatch.setattr(tui, "DEAD_AFTER", 0.8)
    backend.script = [{"delay": 1.5, "text": "endlich"}]
    backend.by_model["tiny"] = [{"text": "t"}]
    cfg = make_cfg(small_model="tiny")

    async def run():
        app = PyrateApp(cfg, DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert not app.activity.has_class("busy")
            app.input.insert("hallo")
            await pilot.press("ctrl+j")
            await pilot.pause(0.15)
            first = str(app.activity.render())
            assert app.activity.has_class("busy")
            assert "waiting for backend" in first and "sign of life" in first
            await pilot.pause(0.4)
            assert app.activity.has_class("stale")
            assert str(app.activity.render()) != first  # spinner/zeit bewegen sich
            await pilot.pause(0.6)
            assert app.activity.has_class("dead")
            assert "not answering" in str(app.activity.render())
            await _wait_idle(app, pilot)
            assert not app.activity.has_class("busy")
            assert not app.activity.has_class("dead")
            assert app.query(Markdown).last().source == "endlich"
    asyncio.run(run())


def test_esc_cancels_hanging_backend(backend, make_cfg, tmp_path):
    """esc bei totem backend: tui ist sofort wieder frei, nicht erst nach request_timeout."""
    backend.script = [{"delay": 30, "text": "zu spaet"}]
    backend.by_model["tiny"] = [{"text": "t"}]
    cfg = make_cfg(small_model="tiny")

    async def run():
        app = PyrateApp(cfg, DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            app.input.insert("hallo?")
            await pilot.press("ctrl+j")
            await pilot.pause(0.3)
            assert app.busy
            await pilot.press("escape")
            await _wait_idle(app, pilot, secs=2)
            assert not app.busy, "tui hangs nach esc noch"
            assert not app.activity.has_class("busy")
            notes = [str(w.render()) for w in app.query(".note")]
            assert any("cancelled" in n for n in notes)
            # und danach geht's normal weiter
            backend.script = [{"text": "wieder da"}]
            app.input.insert("nochmal")
            await pilot.press("ctrl+j")
            await _wait_idle(app, pilot)
            assert app.query(Markdown).last().source == "wieder da"
    asyncio.run(run())


def test_tui_slash_compact_and_resume(backend, make_cfg, tmp_path):
    from textual.widgets import Collapsible
    big = "".join(f"zeile {i} " + "x" * 80 + "\n" for i in range(1500))
    (tmp_path / "big.txt").write_text(big)
    cfg = make_cfg(output_max_bytes=200000, compact_model="summ", small_model="tiny")
    backend.by_model["tiny"] = [{"text": "T"}]
    backend.by_model["summ"] = [{"text": "## Goal\n- big.txt verstehen"}]
    backend.script = [{"tool_calls": [{"name": "read", "arguments": {"path": "big.txt"}}]},
                      {"text": "gelesen"}, {"text": "zweiter turn"}]
    db = DB(cfg.db_path)

    async def run():
        app = PyrateApp(cfg, db, str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+k")
            await pilot.pause()
            assert not app.busy  # ohne session: nur notify
            app.input.insert("/compact")  # kein kommando mehr: unbekannter skill, bleibt im input
            await pilot.press("ctrl+j")
            await pilot.pause()
            assert not app.busy and app.input.text == "/compact"
            app.input.clear()
            for prompt in ("lies big.txt", "und?"):
                app.input.insert(prompt)
                await pilot.press("ctrl+j")
                await _wait_idle(app, pilot)
            app.input.insert("entwurf")
            await pilot.press("ctrl+k")
            await _wait_idle(app, pilot)
            assert app.input.text == "entwurf"  # ctrl+k lets den entwurf stehen
            blocks = [c for c in app.query(Collapsible) if c.has_class("compaction")]
            assert len(blocks) == 1 and "manual" in blocks[0].title
            assert "~" in str(app.sidebar.render()).split("context")[1]  # wieder schaetzung
            sid = app.agent.session_id

        # fortsetzen: compaction-block in der historie, agent sieht summary statt big.txt
        app = PyrateApp(cfg, db, str(tmp_path), db.get_session(sid))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert any(c.has_class("compaction") for c in app.query(Collapsible))
            assert "big.txt verstehen" in app.agent.messages[0]["content"]
    asyncio.run(run())


def test_tui_sidebar_mcp_and_context(backend, make_cfg, tmp_path):
    import sys

    from pyrate import mcp
    from pyrate.config import MCPServer
    fake = str(Path(__file__).parent / "fake_mcp.py")
    (tmp_path / "AGENTS.md").write_text("regel")
    sk = tmp_path / ".pyrate" / "skills" / "deploy"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: deploy\ndescription: so wird deployt\n---\n")
    m = mcp.Manager([MCPServer(name="fake", command=sys.executable, args=[fake]),
                     MCPServer(name="weg", command="/gibts/nicht")])
    m.start()  # bewusst ohne warten: die tui muss nachziehen

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path), mcp=m)
        async with app.run_test(size=(120, 50)) as pilot:
            for _ in range(100):
                await pilot.pause(0.05)
                side = str(app.sidebar.render())
                if "fake (10)" in side and "✗ weg" in side:
                    break
            assert "● fake (10)" in side and "✗ weg" in side
            assert "AGENTS.md (project)" in side and "1 skills, 1 project" in side
            await pilot.press("tab")  # plan: nur readOnlyHint-tools nutzbar
            await pilot.pause()
            assert "fake (1/10)" in str(app.sidebar.render())
    try:
        asyncio.run(run())
    finally:
        m.close()


def test_tui_toggle_magic_key_and_wand(backend, make_cfg, tmp_path):
    """default-key für toggle_magic (ctrl+g) schaltet den router an/aus (default: aus),
    sidebar zeigt 🪄 nur während an."""
    import sys

    from pyrate import mcp
    from pyrate.config import MCPServer
    fake = str(Path(__file__).parent / "fake_mcp.py")
    m = mcp.Manager([MCPServer(name="fake", command=sys.executable, args=[fake])])

    async def run():
        app = PyrateApp(make_cfg(mcp_magic=True), DB(":memory:"), str(tmp_path), mcp=m)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.keys["toggle_magic"] == "ctrl+g"
            assert app.agent.magic is False  # default: aus
            assert "🪄" not in str(app.sidebar.render())
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert app.agent.magic is True
            assert "mcp 🪄" in str(app.sidebar.render())
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert app.agent.magic is False
            assert "🪄" not in str(app.sidebar.render())
    try:
        asyncio.run(run())
    finally:
        m.close()


def test_tui_tool_search_magnifier(backend, make_cfg, tmp_path):
    """sidebar zeigt 🔍 hinter mcp, solange mcp_tool_search in der config an ist."""
    import sys

    from pyrate import mcp
    from pyrate.config import MCPServer
    fake = str(Path(__file__).parent / "fake_mcp.py")

    async def run(cfg, m):
        app = PyrateApp(cfg, DB(":memory:"), str(tmp_path), mcp=m)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            return str(app.sidebar.render())

    for on in (True, False):
        m = mcp.Manager([MCPServer(name="fake", command=sys.executable, args=[fake])])
        try:
            text = asyncio.run(run(make_cfg(mcp_tool_search=on), m))
        finally:
            m.close()
        assert ("mcp 🔍" in text) is on


def test_tui_toggle_tool_search_key(backend, make_cfg, tmp_path):
    """ctrl+f schaltet tool search für die session um, 🔍 folgt dem laufzeit-zustand."""
    import sys

    from pyrate import mcp
    from pyrate.config import MCPServer
    fake = str(Path(__file__).parent / "fake_mcp.py")
    m = mcp.Manager([MCPServer(name="fake", command=sys.executable, args=[fake])])

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path), mcp=m)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.keys["toggle_tool_search"] == "ctrl+f"
            assert app.agent.tool_search is False  # default aus config
            await pilot.press("ctrl+f")
            await pilot.pause()
            assert app.agent.tool_search is True
            assert "mcp 🔍" in str(app.sidebar.render())
            await pilot.press("ctrl+f")
            await pilot.pause()
            assert app.agent.tool_search is False
            assert "🔍" not in str(app.sidebar.render())
    try:
        asyncio.run(run())
    finally:
        m.close()


def test_tui_toggle_magic_hidden_without_magic_config(backend, make_cfg, tmp_path):
    """ohne mcp_magic in der config: key ist aus dem footer, der shortcut macht nichts."""
    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.check_action("toggle_magic", []) is False
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert app.agent.magic is False  # unverändert (default aus)
    asyncio.run(run())


def test_tui_custom_keys_from_config(backend, make_cfg, tmp_path):
    """`keys:` in der config belegt den shortcut um; der default-key feuert nicht mehr."""
    import sys

    from pyrate import mcp
    from pyrate.config import MCPServer
    fake = str(Path(__file__).parent / "fake_mcp.py")
    m = mcp.Manager([MCPServer(name="fake", command=sys.executable, args=[fake])])

    async def run():
        app = PyrateApp(make_cfg(mcp_magic=True, keys={"toggle_magic": "ctrl+e", "submit": "ctrl+enter"}),
                        DB(":memory:"), str(tmp_path), mcp=m)
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert app.input.submit_key == "ctrl+enter"
            assert app.agent.magic is False  # default: aus
            await pilot.press("ctrl+g")  # default-key ist umbelegt → nichts
            await pilot.pause()
            assert app.agent.magic is False
            await pilot.press("ctrl+e")
            await pilot.pause()
            assert app.agent.magic is True
            # submit per config: ctrl+enter reicht, ctrl+j macht wieder newline
            await pilot.press("h", "ctrl+j", "i")
            assert app.input.text == "h\ni"
            app.input.clear()
            await pilot.press("h", "ctrl+enter")
            await _wait_idle(app, pilot)
            assert app.input.text == ""  # submitted
    try:
        asyncio.run(run())
    finally:
        m.close()


def test_context_bar_thresholds(make_cfg, tmp_path):
    cfg = make_cfg(context_limit=1000)

    async def run():
        app = PyrateApp(cfg, DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 50)) as pilot:
            for used, color, hint in ((740, "green", False), (750, "yellow", True),
                                      (849, "yellow", True), (850, "red", True)):
                app.agent.context_tokens = used
                app.refresh_sidebar()
                await pilot.pause()
                markup = app.sidebar.content if hasattr(app.sidebar, "content") else str(app.sidebar.renderable)
                ctx = str(markup).split("[b]context[/]")[1].split("[b]tools[/]")[0]
                assert f"[{color}]" in ctx, (used, ctx)
                assert ("ctrl+k (compact) recommended" in ctx) is hint, (used, ctx)
    asyncio.run(run())


NASTY = ["x d[k] y[/", "[/]", "[/x]", "[b]fett?[/b]", "a\\", "a\\[b]", "for f in ['host_vars/a.yml']:", "[", "$x [@click=quit]"]


def test_esc_is_literal_in_textual_markup():
    for s in NASTY:
        for tpl in ("[b]{}[/]", "[yellow]⋯[/] {}"):
            plain = to_content(tpl.format(esc(s))).plain
            assert plain.rstrip() == tpl.replace("[b]", "").replace("[yellow]", "").replace("[/]", "").format(s)


def test_tool_label_with_markup_chars_does_not_crash(make_cfg, tmp_path):
    """regression: bash-label mit '[k] … [/' hat refresh_sidebar mit MarkupError abgeschossen."""
    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(160, 50)) as pilot:
            for i, s in enumerate(NASTY):
                await app.handle_event({"type": "tool_start", "id": f"c{i}", "name": "bash", "args": {"command": s}})
                await pilot.pause()
                assert s in to_content(app.sidebar.content).plain
                await app.handle_event({"type": "tool_end", "id": f"c{i}", "ok": True, "result": s})
                await pilot.pause()
    asyncio.run(run())


def test_tui_mcp_panel_toggles(make_cfg, tmp_path):
    import sys

    from pyrate import mcp
    from pyrate.config import MCPServer
    from pyrate.tui import McpScreen
    fake = str(Path(__file__).parent / "fake_mcp.py")
    m = mcp.Manager([MCPServer(name="an", command=sys.executable, args=[fake]),
                     MCPServer(name="aus", command=sys.executable, args=[fake], enabled=False)])
    m.start(wait=True, timeout=10)

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path), mcp=m)
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            assert "○ aus (off)" in str(app.sidebar.render())
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, McpScreen)
            ol = app.screen.query_one("OptionList")
            assert "[x] an" in str(ol.get_option("an").prompt) and "tokens" in str(ol.get_option("an").prompt)
            assert "not connected yet" in str(ol.get_option("aus").prompt)
            await pilot.press("down", "enter")  # "aus" einschalten → verbindet im hintergrund
            for _ in range(100):
                await pilot.pause(0.05)
                if "tools ·" in str(ol.get_option("aus").prompt):
                    break
            assert "[x] aus" in str(ol.get_option("aus").prompt)
            assert "mcp__aus__echo" in app.agent.tool_names()
            await pilot.press("up", "space")  # "an" ausschalten
            await pilot.pause()
            assert not any(n.startswith("mcp__an__") for n in app.agent.tool_names())
            assert "[ ] an" in str(ol.get_option("an").prompt)
            await pilot.press("escape")
            await pilot.pause()
            side = str(app.sidebar.render())
            assert "○ an (off)" in side and "● aus (10)" in side
    try:
        asyncio.run(run())
    finally:
        m.close()


def test_mcp_key_hidden_without_servers(make_cfg, tmp_path):
    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test() as pilot:
            assert app.check_action("mcp_panel", ()) is False
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert not isinstance(app.screen, ModalScreen)
    from textual.screen import ModalScreen
    asyncio.run(run())


def test_tui_skill_command(backend, make_cfg, tmp_path):
    sk = tmp_path / ".pyrate" / "skills" / "deploy"
    sk.mkdir(parents=True)
    (sk / "SKILL.md").write_text("---\nname: deploy\ndescription: so wird deployt\n---\nerst testen\n")
    backend.script = [{"text": "ok"}]
    db = DB(":memory:")

    async def run():
        app = PyrateApp(make_cfg(), db, str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            app.input.insert("/deploy prod")
            await pilot.press("ctrl+j")
            await _wait_idle(app, pilot)
            sent = [r["body"]["messages"][-1]["content"] for r in backend.requests]
            main = [s for s in sent if s.startswith("<skill")]
            assert len(main) == 1 and "erst testen" in main[0] and main[0].endswith("prod")
            assert not any("erst testen" in s for s in sent if s not in main)  # titel ohne skill
            assert str(app.query(".user").first().render()) == "/deploy prod"
            sid = app.agent.session_id
        app = PyrateApp(make_cfg(), db, str(tmp_path), db.get_session(sid))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert str(app.query(".user").first().render()) == "/deploy prod"  # historie kompakt
    asyncio.run(run())


def test_tui_skill_popup(backend, make_cfg, tmp_path):
    for name, desc in [("deploy", "so wird deployt"), ("debug", "fehler jagen"), ("review", "quelltext lesen")]:
        sk = tmp_path / ".pyrate" / "skills" / name
        sk.mkdir(parents=True)
        (sk / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody {name}\n")
    backend.script = [{"text": "ok"}]

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            names = lambda: [app.skillpop.get_option_at_index(i).id for i in range(app.skillpop.option_count)]
            await pilot.press("/")
            assert app.skillpop_open and names() == ["debug", "deploy", "review"]
            await pilot.press("d", "e")
            assert names() == ["debug", "deploy"]
            await pilot.press("down", "tab")  # tab vervollstaendigt statt plan/build
            assert app.input.text == "/deploy " and not app.skillpop_open
            assert app.agent.mode == "build"
            app.input.clear()
            await pilot.press("/", "r", "e", "v")
            await pilot.press("escape")
            assert not app.skillpop_open and app.input.text == "/rev"
            await pilot.press("i")
            assert app.skillpop_open  # nach esc + weitertippen wieder da
            await pilot.press("enter")  # enter vervollstaendigt erstmal
            assert app.input.text == "/review " and not app.busy
            app.input.clear()
            await pilot.press("/", "x", "y", "z")
            assert not app.skillpop_open  # no hit, kein popup
            app.input.clear()
            await pilot.press("/", "d", "e", "b", "u", "g")
            await pilot.press("ctrl+j")  # schon ausgeschrieben → direkt abschicken
            await _wait_idle(app, pilot)
            assert backend.requests and app.input.text == ""
    asyncio.run(run())


def test_exit_cancels_pending_ask_user(make_cfg, tmp_path):
    """regression: app zu while ask_user-modal → agent-thread wartete ewig, prozess ging nie aus."""
    from pyrate.tui import AskScreen
    box = {}

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            app.cancel_event = threading.Event()  # wie bei einem laufenden run
            t = threading.Thread(target=lambda: box.update(answer=app.ask_user_blocking("na?", ["a"])),
                                 daemon=True)  # daemon: bei regression hangs wenigstens pytest nicht
            t.start()
            for _ in range(30):
                await pilot.pause(0.1)
                if isinstance(app.screen, AskScreen):
                    break
            assert isinstance(app.screen, AskScreen)
        for _ in range(30):  # loop am leben lassen, call_from_thread braucht ihn evtl. noch
            if not t.is_alive():
                break
            await asyncio.sleep(0.1)
        assert not t.is_alive(), "ask_user hangs nach dem beenden"
        assert box == {"answer": None}
        assert app.closing.is_set()  # titel-call wird auch abgebrochen
    asyncio.run(run())


def test_copy_uses_system_clipboard(make_cfg, tmp_path, monkeypatch):
    """osc 52 allein reicht in vte nicht: ctrl+c und markieren gehen zusaetzlich an wl-copy/xclip."""
    from pyrate import tui
    log = tmp_path / "clip.log"
    fake = tmp_path / "bin" / "wl-copy"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\nprintf "%s|" "$*" >> {log}\ncat >> {log}\necho >> {log}\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:/usr/bin:/bin")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-test")
    assert tui.clipboard_cmd(primary=True) == ["wl-copy", "--primary"]

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            app.input.insert("kopier mich")
            app.input.select_all()
            await pilot.press("ctrl+c")
            await pilot.pause()
            app.copy_to_clipboard("direkt")
    asyncio.run(run())
    assert log.read_text().splitlines() == ["|kopier mich", "|direkt"]
    assert tui.system_copy("sel", primary=True)
    assert log.read_text().splitlines()[-1] == "--primary|sel"


def test_sidebar_scrolls_and_shows_version(backend, make_cfg, tmp_path):
    """kleines terminal: sidebar running via → scrollbar, version bleibt unten stehen."""
    import pyrate

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(100, 15)) as pilot:
            await pilot.pause()
            box = app.query_one("#sidebarscroll")
            assert box.max_scroll_y > 0
            version = app.query_one("#version")
            assert f"pyrate {pyrate.__version__}" in str(version.render())
            assert version.region.bottom == box.region.bottom + 1  # direkt unter der scrollbox
            box.scroll_end(animate=False)
            await pilot.pause()
            assert box.scroll_y == box.max_scroll_y

    asyncio.run(run())


def test_view_cycle(make_cfg, tmp_path):
    """ctrl+v: eingabe 15% → 50:50 → ausgeblendet → wieder 15%."""
    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.input.outer_size.height == 6
            await pilot.press("ctrl+v")
            await pilot.pause()
            assert app.screen.has_class("view-half")
            assert abs(app.input.outer_size.height - app.chat.outer_size.height) <= 1
            assert app.focused is app.input
            await pilot.press("ctrl+v")
            await pilot.pause()
            assert not app.input.display and app.focused is app.chat
            await pilot.press("ctrl+v")
            await pilot.pause()
            assert app.input.display and app.input.outer_size.height == 6
            assert app.focused is app.input
    asyncio.run(run())


SEARCH_MD = "erst ein absatz\n\n| a | b |\n|---|---|\n| WELT | 2 |\n\nende welt\n"


def test_search_chat(make_cfg, tmp_path):
    """ctrl+w: case insensitive, neueste zuerst, enter/ctrl+w = naechster, esc = zu."""
    from textual.widgets import Static

    from pyrate.tui import ToolBlock

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await app.mount_chat(Static("hallo Welt", classes="user"))
            md = Markdown(SEARCH_MD)
            await app.mount_chat(md)
            await app.mount_chat(ToolBlock("read welt.txt"))
            await pilot.pause()
            await pilot.press("ctrl+w")
            assert app.searchbar.has_class("open") and app.focused is app.search
            await pilot.press(*"wElT", "enter")
            await pilot.pause()
            seen = [app.hit]
            assert str(app.searchinfo.render()) == "1/4" and isinstance(app.hit, ToolBlock)
            for key in ("enter", "ctrl+w", "enter"):
                await pilot.press(key)
                seen.append(app.hit)
            assert str(app.searchinfo.render()) == "4/4"
            assert seen[1].source.startswith("ende welt") and "| WELT |" in seen[2].source
            assert seen[3].has_class("user")
            assert [w for w in app.query(".hit")] == [seen[3]]
            await pilot.press("enter")
            assert app.hit is seen[0]  # wieder von vorn
            app.search.value = "gibtsnicht"
            await pilot.press("enter")
            assert app.hit is None and str(app.searchinfo.render()) == "no hit"
            await pilot.press("escape")
            assert not app.searchbar.has_class("open") and app.focused is app.input
            assert not app.query(".hit")
            await pilot.press("ctrl+w")
            assert app.search.value == "gibtsnicht"  # alter begriff bleibt stehen
    asyncio.run(run())


def test_copy_table_as_markdown(make_cfg, tmp_path):
    """markierte tabellen landen als markdown-quelltext im clipboard, der rest bleibt text."""
    from textual.selection import SELECT_ALL

    md_text = "intro **fett**\n\n| name | wert |\n|------|-----:|\n| a | `1` |\n| b | **2** |\n\noutro\n"
    copied = []

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            app.copy_to_clipboard = copied.append
            md = Markdown("")
            await app.mount_chat(md)
            stream = Markdown.get_stream(md)  # wie im echten betrieb: gestreamt
            for i in range(0, len(md_text), 7):
                await stream.write(md_text[i:i + 7])
            await stream.stop()
            await pilot.pause()
            widgets = [w for w in md.query("*") if w.get_selection(SELECT_ALL)]
            app.screen.selections = {w: SELECT_ALL for w in widgets}
            await pilot.press("ctrl+c")
            await pilot.pause()
            # nur ein teil der tabelle markiert → trotzdem ganze tabelle
            cell = next(w for w in widgets if "row2" in w.classes)
            app.screen.selections = {cell: SELECT_ALL}
            await pilot.press("ctrl+c")
            await pilot.pause()
    asyncio.run(run())
    table = "| name | wert |\n|------|-----:|\n| a | `1` |\n| b | **2** |"
    assert copied == [f"intro fett\n\n{table}\n\noutro", table]


def test_pick_model_filter(backend, make_cfg, tmp_path):
    """ctrl+p: tippen filtert die modelle (substring, case-insensitive), enter waehlt den highlight."""
    cfg = make_cfg(models=["qwen3:8b", "qwen3:32b", "claude-sonnet"])

    async def run():
        app = PyrateApp(cfg, DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await pilot.press("ctrl+p")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, PickScreen)
            ol = screen.query_one("#opts", OptionList)
            assert ol.option_count == 4  # 3 modelle + "fake-model" (default-model des fixtures)

            await pilot.press(*"QWEN")  # case-insensitive
            await pilot.pause()
            assert ol.option_count == 2
            ids = [ol.get_option_at_index(i).id for i in range(ol.option_count)]
            assert all("qwen" in i for i in ids)

            await pilot.press("backspace", "backspace", "backspace", "backspace")
            await pilot.press(*"32b")
            await pilot.pause()
            assert ol.option_count == 1
            await pilot.press("enter")
            await pilot.pause()
            assert app.agent.model.startswith("qwen3:32b@")

            # esc schliesst ohne wahl
            await pilot.press("ctrl+p")
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()
            assert app.agent.model.startswith("qwen3:32b@")
    asyncio.run(run())


def test_mousedown_on_stale_widget_does_not_crash(make_cfg, tmp_path):
    # streaming swaps markdown blocks; till the next repaint the compositor still maps the
    # removed block → MouseDown hit a widget with parent None and textual crashed.
    from textual.geometry import Offset
    from textual.widgets import Static

    async def run():
        app = PyrateApp(make_cfg(), DB(":memory:"), str(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            stale = Static("weg")  # never mounted = same state as a just-removed block
            screen.get_widget_and_offset_at = lambda x, y: (stale, Offset(0, 0))
            ev = events.MouseDown(None, 5, 5, 0, 0, 1, False, False, False, screen_x=5, screen_y=5)
            screen._forward_event(ev)  # used to raise AttributeError: 'NoneType' ... 'region'
            assert screen._select_state is None
    asyncio.run(run())
