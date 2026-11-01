import subprocess
import sys
from pathlib import Path

import pytest
from conftest import HAS_LANDLOCK

from pyrate import config, sandbox, tools
from pyrate.sandbox import Sandbox
from pyrate.tools import ToolContext

landlock = pytest.mark.skipif(not HAS_LANDLOCK, reason="kernel ohne landlock")
landlock_net = pytest.mark.skipif(sandbox.abi() < 4, reason="kernel ohne landlock-netzregeln (< 6.7)")


@pytest.fixture
def box(tmp_path):
    """cwd = tmp/in, daneben tmp/out mit einem geheimnis."""
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "geheim").write_text("42\n")
    sb = Sandbox(ro=sandbox.SYSTEM_RO, rw=[tmp_path / "in", *sandbox.SYSTEM_RW])
    return tmp_path, sb


def ctx(tmp_path, sb, answer=None):
    ask = (lambda q, o: answer) if answer is not None else None
    return ToolContext(cwd=str(tmp_path / "in"), sandbox=sb, ask_user=ask)


def test_allows(tmp_path):
    (tmp_path / "rw").mkdir()
    (tmp_path / "ro").mkdir()
    (tmp_path / "rw" / "link").symlink_to(tmp_path / "ro")
    sb = Sandbox(ro=[tmp_path / "ro"], rw=[tmp_path / "rw"])
    assert sb.allows(tmp_path / "rw" / "x", write=True)
    assert sb.allows(tmp_path / "ro" / "x") and not sb.allows(tmp_path / "ro" / "x", write=True)
    assert not sb.allows(tmp_path / "rwx")  # praefix-falle
    assert not sb.allows(tmp_path)
    assert Path(sb.wrap(["bash"])[3]).name == "sandbox.py"


def test_file_tools_outside_without_human(box):
    tmp, sb = box
    ok, out = tools.run("read", {"path": "../out/geheim"}, ctx(tmp, sb), ["read"])
    assert not ok and "outside the sandbox" in out and "nobody answered" in out
    ok, out = tools.run("write", {"path": str(tmp / "out" / "neu"), "content": "x"}, ctx(tmp, sb), ["write"])
    assert not ok and not (tmp / "out" / "neu").exists()
    # headless ohne tty: callback da, liefert aber None → kein "vom user verweigert"
    c = ToolContext(cwd=str(tmp / "in"), sandbox=sb, ask_user=lambda q, o: None)
    ok, out = tools.run("read", {"path": "../out/geheim"}, c, ["read"])
    assert not ok and "nobody answered" in out and "denied" not in out


def test_write_grant_is_the_file_not_the_dir(box):
    """~/notes.txt schreiben gibt nicht ganz ~ frei."""
    tmp, sb = box
    asked = []
    c = ToolContext(cwd=str(tmp / "in"), sandbox=sb, ask_user=lambda q, o: asked.append(q) or "ja")
    ok, out = tools.run("write", {"path": str(tmp / "out" / "notes.txt"), "content": "x"}, c, ["write"])
    assert ok, out
    assert "file" in asked[0] and str(tmp / "out" / "notes.txt") in asked[0]
    assert sb.allows(tmp / "out" / "notes.txt", write=True)
    assert not sb.allows(tmp / "out" / "anderes.txt", write=True) and not sb.allows(tmp / "out" / "geheim")


def test_write_into_missing_dir_needs_dir_grant(box):
    """a grant for exactly the file is not enough when the parent dir is missing: mkdir must
    not create an empty dir outside the sandbox."""
    tmp, sb = box
    c = ToolContext(cwd=str(tmp / "in"), sandbox=sb, ask_user=lambda q, o: "ja")
    ok, out = tools.run("write", {"path": str(tmp / "out" / "neu" / "f.txt"), "content": "x"}, c, ["write"])
    assert not ok and "outside the sandbox and not granted" in out and str(tmp / "out" / "neu") in out
    assert not (tmp / "out" / "neu").exists()
    # grant for the directory itself → creating it is ok
    sb.grant(tmp / "out" / "neu", write=True)
    ok, out = tools.run("write", {"path": str(tmp / "out" / "neu" / "f.txt"), "content": "x"}, c, ["write"])
    assert ok, out and (tmp / "out" / "neu" / "f.txt").read_text() == "x"


def test_no_hint_from_outside(box):
    """grant auf eine nicht existierende datei listet nicht nebenbei den ordner drueber."""
    tmp, sb = box
    ok, out = tools.run("read", {"path": "../out/geheimm"}, ctx(tmp, sb, "ja"), ["read"])
    assert not ok and "does not exist" in out and "did you mean" not in out


def test_file_tools_ask_and_grant(box):
    tmp, sb = box
    ok, out = tools.run("read", {"path": "../out/geheim"}, ctx(tmp, sb, "nein"), ["read"])
    assert not ok and "denied" in out
    ok, out = tools.run("read", {"path": "../out/geheim"}, ctx(tmp, sb, "ja"), ["read"])
    assert ok and "42" in out
    assert sb.allows(tmp / "out" / "geheim") and not sb.allows(tmp / "out" / "geheim", write=True)
    # lesen freigegeben heisst nicht schreiben
    ok, out = tools.run("edit", {"path": "../out/geheim", "old_string": "42", "new_string": "43"},
                        ctx(tmp, sb), ["edit"])
    assert not ok and "outside" in out


def test_grep_skips_symlinks_out_of_sandbox(box):
    tmp, sb = box
    (tmp / "in" / "a.txt").write_text("42\n")
    (tmp / "in" / "leak").symlink_to(tmp / "out" / "geheim")
    out = tools.t_grep(ctx(tmp, sb), "42")
    assert "a.txt" in out and "leak" not in out


@landlock
def test_bash_is_locked_in(box):
    tmp, sb = box
    c = ctx(tmp, sb)
    out = tools.t_bash(c, "echo ok > drin; cat drin; cat ../out/geheim; echo x > ../out/neu; ls ..")
    assert out.splitlines()[0] == "ok"
    assert out.count("Permission denied") + out.count("Keine Berechtigung") == 3, out
    assert not (tmp / "out" / "neu").exists()


@landlock
def test_grant_reaches_bash(box):
    tmp, sb = box
    tools.run("read", {"path": "../out"}, ctx(tmp, sb, "ja"), ["read"])
    out = tools.t_bash(ctx(tmp, sb), "cat ../out/geheim")
    assert "42" in out and "[exit 0]" in out


@landlock
def test_child_cannot_escape(box):
    """landlock vererbt sich: ein sandbox-aufruf im sandbox-bash wird nur enger, nie weiter."""
    tmp, sb = box
    for p in (sys.prefix, sys.base_prefix, Path(sandbox.__file__).parent):  # innerer wrapper muss starten
        sb.grant(p)
    inner = " ".join(Sandbox(ro=["/"], rw=["/"]).wrap(["cat", str(tmp / "out" / "geheim")]))
    out = tools.t_bash(ctx(tmp, sb), inner)
    assert out.startswith("cat: ") and out.endswith("[exit 1]"), out


def test_config_sandbox_keys_protected(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox, "abi", lambda: 9)
    glob = tmp_path / "g.yaml"
    glob.write_text("base_url: http://x\nmodel: m\nsandbox_ro: [~/.gitconfig]\nsandbox_rw: [~/.cache/uv]\n")
    (tmp_path / ".pyrate.yaml").write_text("sandbox: false\nsandbox_ro: [/]\nsandbox_rw: [/]\n")
    cfg = config.load(tmp_path, glob)
    assert cfg.sandbox is True
    assert cfg.sandbox_ro == [Path("~/.gitconfig").expanduser()]
    assert cfg.sandbox_rw == [Path("~/.cache/uv").expanduser()]
    assert sum("sandbox" in w for w in cfg.warnings) == 3


def test_sandbox_rw_reaches_agent(make_cfg, tmp_path, db):
    from pyrate.agent import Agent
    (tmp_path / "cache").mkdir()
    agent = Agent(make_cfg(sandbox=True, sandbox_rw=[str(tmp_path / "cache")]), db, tmp_path / "cache")
    assert agent.sandbox.allows(tmp_path / "cache" / "x", write=True)


def test_without_landlock_bash_is_off_with_warning(monkeypatch):
    monkeypatch.setattr(sandbox, "abi", lambda: 0)
    base = {"backends": [{"base_url": "http://x", "models": ["m"]}]}
    cfg = config.build(base)
    assert cfg.tools["bash"] is False and "bash" not in tools.for_mode(cfg.tools, "build")
    assert any("no landlock" in w for w in cfg.warnings)
    cfg = config.build({**base, "sandbox": False})  # bewusst ohne sandbox: bash bleibt, keine warnung
    assert "bash" in tools.for_mode(cfg.tools, "build") and not cfg.warnings
    cfg = config.build({**base, "tools": {"bash": False}})  # bash eh aus: nichts zu meckern
    assert not any("landlock" in w for w in cfg.warnings)


def test_tmp_wiped_on_exit(backend, tmp_path):
    """eigenes tmp pro lauf unter $TMPDIR, TMPDIR zeigt fuers bash dorthin, danach ist alles weg."""
    base = tmp_path / "tmp"
    base.mkdir()
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text(f"base_url: {backend.url}\nmodel: fake\nsmall_model: tiny\ndb_path: {tmp_path / 'db'}\n"
                       f"sandbox: {str(HAS_LANDLOCK).lower()}\n")
    backend.script = [{"tool_calls": [{"name": "bash", "arguments": {"command": "echo x > $TMPDIR/f; echo $TMPDIR"}}]},
                      {"text": "fertig"}]
    backend.by_model["tiny"] = [{"text": "T"}]
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).parent.parent), "HOME": str(tmp_path),
           "XDG_DATA_HOME": str(tmp_path / "xdg"), "TMPDIR": str(base)}
    r = subprocess.run([sys.executable, "-m", "pyrate", "--config", str(cfgfile), "-p", "los"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == 0, r.stderr
    result = [x for x in backend.requests if x["body"]["model"] == "fake"][-1]["body"]["messages"][-1]["content"]
    used = Path(result.splitlines()[0])
    assert used.parent == base and used.name.startswith("pyrate-")
    assert list(base.iterdir()) == []


# ---------------------------------------------------------------- plan-modus: bash read-only, ohne tcp

def test_wrap_readonly_and_no_net(tmp_path):
    sb = Sandbox(ro=[tmp_path / "ro"], rw=[tmp_path / "rw"], scratch=[tmp_path / "tmp"])
    args = sb.wrap(["bash"])
    assert "--no-net" not in args and args[args.index(str(tmp_path / "rw")) - 1] == "--rw"
    args = sb.wrap(["bash"], readonly=True)  # plan-modus: read-only, netz getrennt schaltbar
    assert "--no-net" not in args
    assert args[args.index(str(tmp_path / "rw")) - 1] == "--ro"  # cwd & co. nur noch lesbar …
    assert args[args.index(str(tmp_path / "tmp")) - 1] == "--rw"  # … scratch bleibt schreibbar
    args = sb.wrap(["bash"], readonly=True, net=False)  # plan-default: zusaetzlich kein tcp
    assert "--no-net" in args
    assert sb.allows(tmp_path / "tmp" / "x", write=True)


@pytest.fixture
def listener():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    yield s.getsockname()[1]
    s.close()


@landlock_net
def test_plan_bash_readonly_without_tcp(box, listener):
    tmp, sb = box
    (tmp / "scratch").mkdir()
    sb.scratch.append((tmp / "scratch").resolve())
    (tmp / "in" / "a.txt").write_text("hallo\n")
    c = ctx(tmp, sb)
    c.readonly, c.net = True, False  # plan-default: read-only + kein tcp
    out = tools.t_bash(c, f"cat a.txt; echo x > b.txt; echo y > ../scratch/s; cat ../scratch/s; "
                          f"exec 3<>/dev/tcp/127.0.0.1/{listener} && echo verbunden")
    assert out.splitlines()[0] == "hallo" and "y" in out.splitlines()
    assert not (tmp / "in" / "b.txt").exists()
    assert "verbunden" not in out
    c.net = True  # plan + ctrl+n: tcp offen, fs weiter read-only
    out = tools.t_bash(c, f"echo x > b.txt; exec 3<>/dev/tcp/127.0.0.1/{listener} && echo verbunden")
    assert "verbunden" in out and not (tmp / "in" / "b.txt").exists()
    c.readonly = False  # build: netz offen
    assert "verbunden" in tools.t_bash(c, f"exec 3<>/dev/tcp/127.0.0.1/{listener} && echo verbunden")


@landlock_net
def test_plan_agent_git_diff(backend, make_cfg, tmp_path, db):
    import shutil
    from pyrate.agent import Agent
    (tmp_path / "tmp").mkdir()
    git = shutil.which("git")
    if not git:
        pytest.skip("kein git")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run([git, "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=repo,  # noqa: E731
                                    check=True, capture_output=True)
    (repo / "f.txt").write_text("alt\n")
    run("init", "-q")
    run("add", ".")
    run("commit", "-qm", "init")
    (repo / "f.txt").write_text("neu\n")
    backend.script = [{"tool_calls": [{"name": "bash", "arguments": {"command": "git diff; touch x"}}]},
                      {"text": "ok"}]
    agent = Agent(make_cfg(), db, repo, mode="plan", tmp=tmp_path / "tmp")
    events = []
    agent.run("was hat sich changed?", events.append)
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["ok"] and "-alt" in end["result"] and "+neu" in end["result"], end["result"]
    assert not (repo / "x").exists()
    system = backend.requests[0]["body"]["messages"][0]["content"]
    assert "bash works in plan mode" in system and "no network" in system
    agent.net = True  # ctrl+n: tcp im plan-modus an, prompt wechselt
    system = agent.system_prompt()
    assert "bash works in plan mode" in system and "tcp is enabled" in system and "no network" not in system


def test_net_toggle_both_modes(make_cfg, tmp_path, db):
    """ctrl+n gilt in plan und build. default je modus: plan aus, build an."""
    from pyrate.agent import Agent
    a = Agent(make_cfg(), db, tmp_path, mode="plan")
    assert a.net is False and "no network" in a.system_prompt()
    a.net = True
    assert "tcp is enabled" in a.system_prompt()
    b = Agent(make_cfg(), db, tmp_path, mode="build")
    assert b.net is True and "tcp is off" not in b.system_prompt()
    b.net = False  # build + ctrl+n aus: prompt warnt, subagents-hinweis faellt weg (kein llm ohne tcp)
    assert "tcp is off for this session" in b.system_prompt()
    c = Agent(make_cfg(subagents=2), db, tmp_path, mode="build")
    assert "SUBAGENTS:" in c.system_prompt()
    c.net = False
    assert "SUBAGENTS:" not in c.system_prompt()


def test_plan_bash_only_with_net_rules(monkeypatch):
    """bash im plan-modus nur, wenn landlock fs read-only *und* tcp aus kann (abi >= 4)."""
    base = {"backends": [{"base_url": "http://x", "models": ["m"]}]}
    for abi, want in ((3, False), (4, True)):
        monkeypatch.setattr(sandbox, "abi", lambda abi=abi: abi)
        cfg = config.build(base)
        assert cfg.plan_bash is want and ("bash" in tools.for_mode(cfg.tools, "plan", cfg.plan_bash)) is want
    assert config.build({**base, "sandbox": False}).plan_bash is False  # ohne sandbox nie


def test_scratch_is_explicit(make_cfg, tmp_path, db):
    """ohne tmp gibt's kein scratch – nie stillschweigend ganz /tmp."""
    from pyrate.agent import Agent
    agent = Agent(make_cfg(sandbox=True), db, tmp_path / "x")
    assert not agent.sandbox.allows("/tmp/irgendwas", write=True)
    assert "TMPDIR" not in agent.tool_env() and "$TMPDIR" not in agent.system_prompt()
    agent = Agent(make_cfg(sandbox=True), db, tmp_path / "x", tmp=tmp_path / "t")
    assert agent.sandbox.allows(tmp_path / "t" / "f", write=True)
    assert agent.tool_env()["TMPDIR"] == str(tmp_path / "t")


def test_skills_are_readable(make_cfg, tmp_path, db):
    """der prompt nennt SKILL.md mit absolutem pfad – das darf keine rueckfrage ausloesen."""
    from pyrate.agent import Agent
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: test\n---\nmach was\n")
    (tmp_path / "proj").mkdir()
    agent = Agent(make_cfg(sandbox=True, skill_dirs=[str(tmp_path / "skills")]), db, tmp_path / "proj")
    assert agent.skills and agent.sandbox.allows(skill / "scripts" / "run.sh")
    assert not agent.sandbox.allows(skill / "SKILL.md", write=True)
    assert not agent.sandbox.allows(tmp_path / "skills")  # nur die gefundenen skills, nicht der ganze ordner


@landlock
def test_exec_failure_without_traceback(box):
    tmp, sb = box
    out = subprocess.run(sb.wrap(["/gibts/nicht"]), capture_output=True, text=True)
    assert out.returncode == 126 and "pyrate-sandbox: cannot start /gibts/nicht" in out.stderr
    assert "Traceback" not in out.stderr


# ---------------------------------------------------------------- unix-sockets

landlock_unix = pytest.mark.skipif(not sandbox.restricts_unix(), reason="kernel ohne landlock RESOLVE_UNIX")
SYS_PY = "/usr/bin/python3"  # das venv-python liegt nicht in SYSTEM_RO


def test_wrap_sock(tmp_path):
    sb = Sandbox(sock=[tmp_path / "dns"])
    socks = lambda args: [args[i + 1] for i, a in enumerate(args) if a == "--sock"]  # noqa: E731
    assert socks(sb.wrap(["bash"])) == socks(sb.wrap(["bash"], readonly=True)) == [str(tmp_path / "dns")]


@landlock_unix
@pytest.mark.skipif(not Path(SYS_PY).exists(), reason="kein /usr/bin/python3")
def test_unix_sockets_locked(box):
    """connect nur unter rw (build) und --sock (immer). sonst: systemd-run --user via den
    session-bus = ausbruch. podman hangs genau an der rw-semantik: plan = ro = kein connect."""
    import socket
    tmp, sb = box
    (tmp / "sock").mkdir()
    servers = []
    for name in ("out/s", "in/s", "sock/s"):
        s = socket.socket(socket.AF_UNIX)
        s.bind(str(tmp / name))
        s.listen()
        servers.append(s)
    sb.sock.append((tmp / "sock").resolve())
    probe = ("import socket,sys\nfor p in sys.argv[1:]:\n s=socket.socket(socket.AF_UNIX)\n"
             " try: s.connect(p); print(p, 'ok')\n except OSError as e: print(p, 'zu')\n")
    (tmp / "in" / "probe.py").write_text(probe)
    cmd = f"{SYS_PY} probe.py {tmp / 'out/s'} {tmp / 'in/s'} {tmp / 'sock/s'}"
    c = ctx(tmp, sb)
    out = tools.t_bash(c, cmd)
    assert f"{tmp / 'out/s'} zu" in out and f"{tmp / 'in/s'} ok" in out and f"{tmp / 'sock/s'} ok" in out, out
    c.readonly = True  # plan: cwd nur lesbar, also auch kein connect. --sock bleibt
    out = tools.t_bash(c, cmd)
    assert f"{tmp / 'in/s'} zu" in out and f"{tmp / 'sock/s'} ok" in out, out
    for s in servers:
        s.close()


@pytest.mark.parametrize("argv", [["--bogus", "x", "--", "true"], ["--ro"], ["--ro", "/usr"], ["--"], []])
def test_main_bad_args_no_traceback(argv):
    r = subprocess.run([sys.executable, "-I", "-S", sandbox.__file__, *argv], capture_output=True, text=True)
    assert r.returncode == 126 and "usage:" in r.stderr and "Traceback" not in r.stderr, r.stderr
