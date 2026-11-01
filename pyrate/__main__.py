"""cli entry point: tui (default) or print mode (-p)."""

import argparse
import os
import shutil
import signal
import sys
import tempfile
import threading
import time

from . import __version__, config, context, images, podman, subagent
from .tools import cleanup_output, short, tool_label
from .db import DB
from .mcp import CONNECT_TIMEOUT, Manager

_EXIT_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="pyrate", description="minimal coding harness 🏴‍☠️")
    ap.add_argument("-p", "--print", dest="prompt", metavar="PROMPT",
                    help="headless: run prompt, answer on stdout. '-' reads from stdin")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="continue the last session in the cwd")
    ap.add_argument("-s", "--session", metavar="ID", help="continue a specific session (id or prefix)")
    ap.add_argument("--compact", action="store_true",
                    help="compact the session (with -c/-s), summary on stdout")
    ap.add_argument("-m", "--model", help="override model: model@backend (or just model, if unambiguous)")
    ap.add_argument("--models", action="store_true", help="list configured models")
    ap.add_argument("--mcp", metavar="NAME", action="append", default=[],
                    help="enable an mcp server with enabled: false for this run (repeatable or a,b)")
    ap.add_argument("--mcp-login", metavar="NAME", help="oauth login for an mcp server (browser)")
    ap.add_argument("--mcp-logout", metavar="NAME", help="delete stored oauth tokens of an mcp server")
    ap.add_argument("--mcp-setup", action="store_true",
                    help="connect all mcp servers once (even disabled ones) and store their tools in the "
                         "state dir (mcp-descriptions.yaml) – used by mcp_magic routing")
    ap.add_argument("--mode", choices=config.MODES, help="plan or build")
    ap.add_argument("--effort", choices=config.EFFORTS, help="reasoning effort")
    ap.add_argument("--sessions", action="store_true", help="list sessions in the cwd")
    ap.add_argument("--all", action="store_true", help="with --sessions: all directories")
    ap.add_argument("--search", metavar="QUERY", help="full-text search across old sessions")
    ap.add_argument("--config", metavar="PATH", help="alternative global config")
    ap.add_argument("-v", "--verbose", action="store_true", help="print mode: reasoning + tool output on stderr")
    ap.add_argument("--version", action="version", version=f"pyrate {__version__}")
    return ap.parse_args(argv)


def _fmt_time(ts):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def cmd_models(cfg):
    roles = {cfg.model: "default"}
    if cfg.small_model:
        roles[cfg.small_model] = (roles.get(cfg.small_model, "") + " small").strip()
    if cfg.compact_model:
        roles[cfg.compact_model] = (roles.get(cfg.compact_model, "") + " compact").strip()
    if cfg.mcp_magic:
        roles[cfg.mcp_magic] = (roles.get(cfg.mcp_magic, "") + " magic").strip()
    ids = cfg.model_ids()
    ids += [r for r in roles if r not in ids]  # small/compact may be unlisted
    width = max(len(i) for i in ids)
    for mid in ids:
        limit = cfg.limit_for(mid)
        extra = "  ".join(x for x in (f"{limit // 1000}k" if limit else "", roles.get(mid, "")) if x)
        print(f"{mid:<{width}}  {extra}".rstrip())
    return 0


def cmd_sessions(db, cwd, all_dirs):
    rows = db.list_sessions(None if all_dirs else cwd, limit=50)
    if not rows:
        print("no sessions", file=sys.stderr)
        return 0
    for r in rows:
        where = f"  {r['cwd']}" if all_dirs else ""
        print(f"{r['id']}  {_fmt_time(r['updated_at'])}  {r['title'] or '(no title)'}{where}")
    return 0


def cmd_search(db, query):
    rows = db.search(query)
    if not rows:
        print("nothing found", file=sys.stderr)
        return 1
    for r in rows:
        sub = f"  [subagent of {r['parent_id']}]" if r["parent_id"] else ""
        print(f"{r['id']}  {_fmt_time(r['updated_at'])}  {r['title'] or '(no title)'}  {r['cwd']}{sub}")
        print(f"    {' '.join(r['snip'].split())}")
    return 0


def cmd_mcp_auth(cfg, args):
    from . import oauth
    from .mcp import Client, MCPError

    name = args.mcp_login or args.mcp_logout
    spec = next((s for s in cfg.mcp if s.name == name), None)
    if spec is None:
        known = ", ".join(s.name for s in cfg.mcp if s.oauth) or "none"
        print(f"pyrate: no mcp server '{name}' here (oauth servers in cwd: {known}). "
              f"folders servers only from the matching directory", file=sys.stderr)
        return 2
    if spec.oauth is None:
        print(f"pyrate: mcp server '{name}' has no 'oauth' in the config", file=sys.stderr)
        return 2
    if args.mcp_logout:
        print(f"{name}: {'logged out' if oauth.logout(spec) else 'was not logged in'}", file=sys.stderr)
        return 0
    try:
        oauth.login(spec)
    except oauth.OAuthError as e:
        print(f"pyrate: login {name}: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\npyrate: cancelled", file=sys.stderr)
        return 130
    client = Client(spec)
    try:
        client.connect()
    except MCPError as e:
        print(f"pyrate: logged in, but connect failed: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()
    print(f"{name}: logged in, {len(client.tools)} tools", file=sys.stderr)
    return 0


def cmd_mcp_setup(cfg):
    """alle server einmal verbinden (auch enabled: false) und name → {description, tools}
    in mcp-descriptions.yaml speichern. basis für den mcp_magic-router, der server sehen
    muss, die gerade nicht laufen."""
    from .mcp import Manager, _tool_line, descriptions_path, load_descriptions, save_descriptions
    if not cfg.mcp:
        print("pyrate: no mcp servers configured", file=sys.stderr)
        return 2
    old = load_descriptions()
    data = {}
    mcp = Manager(cfg.mcp)
    try:
        for name in mcp.enabled:
            mcp.enabled[name] = True  # setup connects even disabled servers
        mcp.start(wait=True, timeout=CONNECT_TIMEOUT)
        for name, state, n, error in mcp.summary():
            spec = mcp.clients[name].spec
            if state == "ready":
                data[name] = {"description": spec.description or "", "updated": time.strftime("%Y-%m-%d"),
                              "tools": sorted(_tool_line(t) for t in mcp.clients[name].tools
                                              if t["name"] not in spec.disabled_tools)}
                print(f"{name}: {n} tools", file=sys.stderr)
            else:
                # server gerade down: alten eintrag (tools!) behalten, nur den fehler vermerken
                prev = old.get(name) if isinstance(old.get(name), dict) else {}
                data[name] = {"description": spec.description or prev.get("description") or "",
                              **{k: v for k, v in prev.items() if k in ("tools", "updated")},
                              "error": error or state}
                kept = " (kept stored tools)" if prev.get("tools") else ""
                print(f"{name}: {error or state}{kept}", file=sys.stderr)
    finally:
        mcp.close()
    save_descriptions(data)
    print(f"saved: {descriptions_path()}", file=sys.stderr)
    return 0


def resolve_session(db, args, cwd):
    if args.session:
        s = db.get_session(args.session)
        if not s:
            raise SystemExit(f"pyrate: session '{args.session}' not found (or ambiguous prefix)")
        return s
    if args.cont:
        s = db.latest_session(cwd)
        if not s:
            print("pyrate: no session in cwd, starting fresh", file=sys.stderr)
        return s
    return None


def run_print(cfg, db, args, cwd, session, mcp=None, parent=None, tmp=None, pod=None):
    """parent set: we are a subagent → stdout gets only the final answer, stderr only
    problems + session id. the caller is a model that dumps all of it into its context."""
    from .agent import Agent, fallback_title

    if args.compact:
        prompt = ""
    else:
        prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
        if not prompt.strip():
            print("pyrate: empty prompt", file=sys.stderr)
            return 2

    def ask_user(question, options):
        if not sys.stdin.isatty():
            return None
        print(f"\n? {question}", file=sys.stderr)
        for i, o in enumerate(options, 1):
            print(f"  {i}) {o}", file=sys.stderr)
        try:
            answer = input("> ").strip()
        except EOFError:
            return None
        if answer.isdigit() and 0 < int(answer) <= len(options):
            return options[int(answer) - 1]
        return answer

    agent = Agent(cfg, db, cwd, session_id=session["id"] if session else None,
                  model=args.model or (session or {}).get("model"), mode=args.mode,
                  # child: without callback ask_user says "no human reachable, make an assumption"
                  effort=args.effort, ask_user=None if parent else ask_user, mcp=mcp, parent_id=parent,
                  tmp=tmp, podman=pod)
    err = sys.stderr
    tty = err.isatty()
    dim = (lambda s: f"\x1b[2m{s}\x1b[0m") if tty else (lambda s: s)
    for w in agent.context_warnings:
        err.write(f"pyrate: warning: {w}\n")
    if mcp is not None:
        mcp.wait(CONNECT_TIMEOUT + 5)  # print mode: only start once all servers are up (or failed)
        for name, state, n, error in mcp.summary():
            if state == "ready":
                err.write(dim(f"mcp {name}: {n} tools\n") if args.verbose else "")
            elif state == "off":
                err.write(dim(f"mcp {name}: off (--mcp {name} enables it)\n") if args.verbose else "")
            else:
                err.write(f"pyrate: warning: mcp {name}: {error or state}\n")
    state = {"text_open": False, "failed": False}
    title_thread = None

    def on_event_child(ev):
        t = ev["type"]
        if t == "session" and ev.get("new"):  # no title call per child, hardly anyone sees those sessions
            db.set_title(agent.session_id, fallback_title(prompt))
        elif t == "warning":
            err.write(f"pyrate: warning: {ev['warning']}\n")
        elif t == "error":
            err.write(f"pyrate: {ev['error']}\n")
            state["failed"] = True
        elif t == "cancelled":
            err.write("pyrate: cancelled\n")
            state["failed"] = True

    def on_event(ev):
        nonlocal title_thread
        t = ev["type"]
        if t == "session" and ev.get("new"):
            title_thread = threading.Thread(target=_safe_title, args=(agent, prompt), daemon=True)
            title_thread.start()
        elif t == "text":
            chunk = ev["text"] if state["text_open"] else ev["text"].lstrip()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                state["text_open"] = True
        elif t == "reasoning" and args.verbose:
            err.write(dim(ev["text"]))
            err.flush()
        elif t == "assistant_done":
            if state["text_open"]:
                if not (ev["message"].get("content") or "").endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
                state["text_open"] = False
        elif t == "tool_start":
            err.write(dim(f"▸ {tool_label(ev['name'], ev['args'])}\n"))
        elif t == "tool_end":
            if not ev["ok"]:
                err.write(f"  ✗ {short(ev['result'], 300)}\n")
            elif args.verbose:
                err.write(dim("  " + ev["result"].replace("\n", "\n  ") + "\n"))
        elif t == "compact_start":
            err.write(dim(f"⇊ compacting context ({ev['reason']}, ~{ev['tokens']} tokens) …\n"))
        elif t == "compact_done":
            err.write(dim(f"⇊ compacted: ~{ev['before']} → ~{ev['after']} tokens "
                          f"({ev['summarized']} messages summarized, {ev['kept']} kept verbatim)\n"))
        elif t == "mcp_magic" and args.verbose:
            err.write(dim(f"mcp_magic: enabled [{', '.join(ev['enabled'])}] raw: {ev.get('raw', '')!r}\n"))
        elif t == "warning":
            err.write(f"pyrate: warning: {ev['warning']}\n")
        elif t == "error":
            err.write(f"pyrate: {ev['error']}\n")
            state["failed"] = True
        elif t == "cancelled":
            err.write("pyrate: cancelled\n")
            state["failed"] = True

    if args.compact:
        if not agent.session_id:
            print("pyrate: --compact needs a session (-c or -s ID)", file=err)
            return 2
        try:
            ok = agent.run_compact(on_event)
        except KeyboardInterrupt:
            err.write("\npyrate: cancelled\n")
            return 130
        if ok:
            prev, _ = db.load_active(agent.session_id)
            print(prev["summary"])
        return 0 if ok else 1

    try:
        expanded = context.expand_skill(prompt, agent.skills)
    except context.SkillError as e:
        print(f"pyrate: {e}", file=err)
        return 2
    imgs, skipped = images.scan(prompt)  # image paths in the prompt → attach to the model
    for path, reason in skipped:
        print(f"pyrate: image not attached: {path} ({reason})", file=err)
    try:
        if parent:
            answer = agent.run(expanded, on_event_child, images=imgs)
            if answer:
                sys.stdout.write(answer.strip() + "\n")
                sys.stdout.flush()
        else:
            agent.run(expanded, on_event, images=imgs)
    except KeyboardInterrupt:
        err.write("\npyrate: cancelled\n")
        return 130  # title thread is a daemon: don't wait up to 15s for it on cancel
    if title_thread is not None:
        title_thread.join(timeout=15)
    if agent.session_id:
        err.write(dim(f"session {agent.session_id}\n"))
    return 1 if state["failed"] else 0


def _safe_title(agent, prompt):
    try:
        agent.make_title(prompt)
    except Exception:  # noqa: BLE001 - title is nice-to-have
        pass


def main(argv=None):
    args = parse_args(argv)
    cwd = os.getcwd()
    try:
        cfg = config.load(cwd, args.config)
    except config.ConfigError as e:
        print(f"pyrate: config: {e}", file=sys.stderr)
        return 2
    for w in cfg.warnings:
        print(f"pyrate: warning: {w}", file=sys.stderr)
    if args.models:
        return cmd_models(cfg)
    if args.mcp_login or args.mcp_logout:
        return cmd_mcp_auth(cfg, args)
    if args.mcp_setup:
        return cmd_mcp_setup(cfg)
    want_mcp = [n.strip() for arg in args.mcp for n in arg.split(",") if n.strip()]
    unknown = sorted(set(want_mcp) - {s.name for s in cfg.mcp})
    if unknown:
        print(f"pyrate: --mcp: unknown: {', '.join(unknown)} (configured: "
              f"{', '.join(s.name for s in cfg.mcp) or 'none'})", file=sys.stderr)
        return 2
    headless = args.prompt is not None or args.compact
    # inside another pyrate's bash tool? then we are a subagent (not for --sessions/--search)
    child = None if args.sessions or args.search else subagent.from_env()
    if child is not None:
        rc = _check_child(cfg, args, *child)
        if rc:
            return rc
    if args.model:  # given explicitly: better complain now than later in the request
        try:
            args.model = cfg.qualify(args.model)
        except config.ConfigError as e:
            print(f"pyrate: -m: {e}", file=sys.stderr)
            return 2
    if child is None:  # with N parallel children it's enough if the parent cleans up
        cleanup_output(cfg.tool_output_dir)  # full texts older than 7 days
    if headless:  # SIGTERM (e.g. timeout in the parent's bash tool) → clean up like ctrl+c
        for sig in _EXIT_SIGNALS:
            signal.signal(sig, _on_sigterm)

    db = DB(cfg.db_path)
    mcp = None
    slot = None
    tmp = None
    pod = None
    try:
        if args.sessions:
            return cmd_sessions(db, cwd, args.all)
        if args.search:
            return cmd_search(db, args.search)
        tmp = make_tmp()
        if cfg.podman:
            pod = start_podman(cfg, child is not None, headless)
        if not headless:  # the tui has no KeyboardInterrupt cleanup: tmp away, then out
            def bye(signum, frame):
                if pod is not None:
                    pod.close(timeout=1)  # own session, wouldn't die with us otherwise. in handler: don't dawdle
                shutil.rmtree(tmp, ignore_errors=True)
                os._exit(128 + signum)
            for sig in _EXIT_SIGNALS:
                signal.signal(sig, bye)
        session = resolve_session(db, args, cwd)
        if child is not None:
            parent, slots = child
            if session and session.get("parent_id") != parent:
                print("pyrate: with -s a subagent can only continue earlier subagents of the same caller",
                      file=sys.stderr)
                return 2
            try:
                slot = subagent.acquire_slot(parent, slots)  # waits while `slots` children are already running
            except OSError as e:
                print(f"pyrate: subagent slot: {e}", file=sys.stderr)
                return 1
        if cfg.mcp:
            mcp = Manager(cfg.mcp)
            for name in want_mcp:
                mcp.enabled[name] = True
            mcp.start()  # in the background; print mode waits in run_print, the tui doesn't
        if headless:
            return run_print(cfg, db, args, cwd, session, mcp, parent=child[0] if child else None, tmp=tmp,
                             pod=pod)
        from .tui import run_tui
        return run_tui(cfg, db, cwd, session, args, mcp, tmp=tmp, pod=pod)
    except KeyboardInterrupt:  # only outside agent.run (which catches itself), e.g. while waiting for a slot
        print("\npyrate: cancelled", file=sys.stderr)
        return 130
    finally:
        if mcp is not None:
            mcp.close()
        if pod is not None:
            pod.close()
        db.close()
        if slot is not None:
            os.close(slot)
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def make_tmp():
    """own tmp per run: mkdtemp (0700, random name, no symlink risk) under $TMPDIR.
    passed explicitly to the agent, which sets TMPDIR only for bash (a child pyrate creates
    its tmp inside ours). deleted completely on exit."""
    return tempfile.mkdtemp(prefix="pyrate-")


def start_podman(cfg, is_child, headless):
    """service outside the sandbox. a child already sits in the parent's sandbox: started
    there, the service itself would be confined (see podman.py) → the child takes the parent's socket."""
    if is_child:
        return podman.Service.inherited()
    try:
        svc = podman.Service.start()
    except OSError as e:
        svc, msg = None, f"podman service won't start: {e}"
    else:
        # stays in even when dead: env stays remote, otherwise podman would fall back to local mode
        msg = svc.ready() and f"{svc.error}. podman doesn't work in this session"
    if msg and headless:
        print(f"pyrate: warning: {msg}", file=sys.stderr)
    elif msg:  # stderr would immediately overwrite the tui → as a toast on mount
        cfg.warnings.append(msg)
    return svc


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt


def _check_child(cfg, args, parent, slots):
    """rules for subagents. returns an exit code, or None if it may proceed."""
    if slots <= 0:
        # also covers recursion: a child passes PYRATE_SUBAGENTS=0 to its children
        print("pyrate: subagents are off (config 'subagents: N', ctrl+s in the tui). as agent: "
              "do the task yourself. as human with inherited env (tmux etc. from an agent run): "
              "env -u PYRATE_SESSION pyrate …", file=sys.stderr)
        return 2
    if args.prompt is None:  # tui needs a terminal, --compact is maintenance, nothing for children
        print("pyrate: as subagent only with -p", file=sys.stderr)
        return 2
    if args.cont or args.compact:
        print("pyrate: -c/--compact don't work as subagent, use -s ID", file=sys.stderr)
        return 2
    if args.mode is None:
        args.mode = "plan"
    model = os.environ.get("PYRATE_MODEL")
    if not args.model and not args.session and model:  # inherit the parent's model (ctrl+p)
        try:
            args.model = cfg.qualify(model)
        except config.ConfigError:
            pass
    return None


if __name__ == "__main__":
    sys.exit(main())
