import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import HAS_LANDLOCK

from pyrate import __main__ as main
from pyrate import config, podman, sandbox
from pyrate.agent import PODMAN, PODMAN_DOWN, PODMAN_OFF, Agent

BASE = {"backends": [{"base_url": "http://x", "models": ["m"]}]}


@pytest.fixture
def can_podman(monkeypatch):
    monkeypatch.setattr(sandbox, "restricts_unix", lambda: True)
    monkeypatch.setattr(config.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_config_off_by_default_and_checks(monkeypatch, can_podman):
    assert config.build(BASE).podman is False
    assert config.build({**BASE, "podman": True}).podman is True
    assert config.build({**BASE, "podman": True, "sandbox": False}).podman is False  # ohne sandbox: normales podman
    for not_true in ("nein", "yes", "true", 1):  # nur das literal true macht das loch auf
        assert config.build({**BASE, "podman": not_true}).podman is False
    monkeypatch.setattr(config.shutil, "which", lambda name: None)
    cfg = config.build({**BASE, "podman": True})
    assert cfg.podman is False and any("not installed" in w for w in cfg.warnings)
    monkeypatch.setattr(sandbox, "restricts_unix", lambda: False)
    cfg = config.build({**BASE, "podman": True})
    assert cfg.podman is False and any("unix sockets" in w for w in cfg.warnings)


def test_config_project_cannot_enable(tmp_path, can_podman):
    glob = tmp_path / "g.yaml"
    glob.write_text("base_url: http://x\nmodel: m\n")
    (tmp_path / ".pyrate.yaml").write_text("podman: true\n")
    cfg = config.load(tmp_path, glob)
    assert cfg.podman is False and any("podman" in w for w in cfg.warnings)


def test_agent_wiring(make_cfg, tmp_path, db):
    (tmp_path / "proj").mkdir()
    (tmp_path / "t").mkdir()
    (tmp_path / "pod").mkdir()
    svc = podman.Service(tmp_path / "pod")  # wie geerbt: kein eigener prozess
    agent = Agent(make_cfg(sandbox=True), db, tmp_path / "proj", tmp=tmp_path / "t", podman=svc)
    env = agent.tool_env()
    assert env["CONTAINER_HOST"] == f"unix://{tmp_path / 'pod' / podman.SOCKET}"
    assert env["REGISTRY_AUTH_FILE"] == str(tmp_path / "t" / "podman-auth.json")  # nicht die echten logins
    assert env[podman.ENV_VAR] == str(tmp_path / "pod")
    assert agent.sandbox.allows(podman.config_dir() / "registries.conf")
    assert not agent.sandbox.allows(podman.config_dir() / "x", write=True)
    # dienst-dir ist rw: build → --rw (connect ok), plan → --ro (kein connect)
    for readonly, flag in ((False, "--rw"), (True, "--ro")):
        args = agent.sandbox.wrap(["bash"], readonly=readonly)
        assert args[args.index(str(tmp_path / "pod")) - 1] == flag
    agent.mode = "plan"
    assert "CONTAINER_HOST" in agent.tool_env()  # bleibt remote, landlock blockt das connect
    if "bash" in agent.tool_names():
        assert PODMAN in agent.system_prompt()
    svc.error = "podman-dienst beendet (exit 125)"
    assert not agent.podman_ok() and PODMAN not in agent.system_prompt()
    assert "CONTAINER_HOST" in agent.tool_env()  # auch tot: nie in den lokalen modus
    plain = Agent(make_cfg(sandbox=True), db, tmp_path / "proj", tmp=tmp_path / "t")
    assert "CONTAINER_HOST" not in plain.tool_env() and PODMAN not in plain.system_prompt()


@pytest.mark.skipif(not HAS_LANDLOCK, reason="ohne landlock kein bash, ohne bash kein hinweis")
def test_prompt_when_podman_is_off_or_down(make_cfg, tmp_path, db, can_podman):
    """ohne hinweis mueht sich das modell mit podman in der sandbox ab (eigene storage-dirs,
    unshare, newuidmap), obwohl das nie gehen kann. can_podman: podman + docker „im PATH“."""
    (tmp_path / "proj").mkdir()
    (tmp_path / "pod").mkdir()

    def prompt(cfg, svc=None, mode=None):
        return Agent(cfg, db, tmp_path / "proj", mode=mode, podman=svc).system_prompt()

    off = make_cfg()
    assert off.container_cli and PODMAN_OFF in prompt(off)
    assert PODMAN not in prompt(off) and PODMAN_DOWN not in prompt(off)
    if off.plan_bash:  # mit bash im plan-modus gilt's da genauso
        assert PODMAN_OFF in prompt(off, mode="plan")
    off.container_cli = False  # gar nicht installiert: kein wort above it
    assert PODMAN_OFF not in prompt(off)

    on = make_cfg(podman=True)
    svc = podman.Service(tmp_path / "pod")
    assert PODMAN in prompt(on, svc) and PODMAN_OFF not in prompt(on, svc)
    svc.error = "podman-dienst beendet (exit 125)"
    assert PODMAN_DOWN in prompt(on, svc) and PODMAN not in prompt(on, svc)
    assert PODMAN_DOWN in prompt(on)  # dienst startete gar nicht erst (OSError)

    bare = make_cfg(sandbox=False)  # ohne sandbox geht podman ganz normal
    assert not any(p in prompt(bare) for p in (PODMAN, PODMAN_OFF, PODMAN_DOWN))
    no_bash = make_cfg(tools={"bash": False})
    assert not any(p in prompt(no_bash) for p in (PODMAN, PODMAN_OFF, PODMAN_DOWN))


def test_inherited(monkeypatch, tmp_path):
    monkeypatch.delenv(podman.ENV_VAR, raising=False)
    assert podman.Service.inherited() is None
    monkeypatch.setenv(podman.ENV_VAR, str(tmp_path))
    svc = podman.Service.inherited()
    assert svc.path == str(tmp_path) and svc.proc is None
    svc.close()  # nicht unser dienst: nichts killen, nichts delete
    assert tmp_path.exists()


def fake_podman(monkeypatch, tmp_path, body):
    """fake-`podman system service unix://…` im PATH, dienst-dirs unter tmp_path."""
    fake = tmp_path / "bin" / "podman"
    fake.parent.mkdir(exist_ok=True)
    fake.write_text(f"#!{sys.executable}\nimport socket, sys, time\n"
                    "assert sys.argv[1:4] == ['system', 'service', '--time=0']\n"
                    "path = sys.argv[4][len('unix://'):]\n" + body)
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake.parent}:{os.environ['PATH']}")
    monkeypatch.setattr(podman.tempfile, "tempdir", str(tmp_path))


LISTEN = "s = socket.socket(socket.AF_UNIX); s.bind(path); s.listen()\ntime.sleep(60)\n"


def test_service_lifecycle(monkeypatch, tmp_path):
    fake_podman(monkeypatch, tmp_path, LISTEN)
    svc = podman.Service.start()
    try:
        # socket-dir unter XDG_RUNTIME_DIR, nicht in $TMPDIR (das ist scratch und im plan-modus rw)
        assert Path(svc.path).parent == Path(os.environ["XDG_RUNTIME_DIR"])
        assert Path(svc.path).name.startswith("pyrate-podman-")
        assert svc.ready() is None and svc.error is None
    finally:
        svc.close()
    assert svc.proc.poll() is not None and not os.path.exists(svc.path)


def test_start_podman_warns_when_service_dies(monkeypatch, tmp_path, make_cfg, capsys):
    """stirbt der dienst gleich: warnung mit log-zeilen (tui: toast, headless: stderr), der dienst
    bleibt als Service drin – env bleibt remote, sonst fiele podman in den lokalen modus back."""
    fake_podman(monkeypatch, tmp_path, "print('Error: storage lock broken', file=sys.stderr); sys.exit(125)\n")
    cfg = make_cfg()
    svc = main.start_podman(cfg, is_child=False, headless=False)
    try:
        assert svc is not None and svc.error
        w = next(w for w in cfg.warnings if "podman" in w)
        assert "exit 125" in w and "storage lock broken" in w and "doesn't work in this session" in w, w
        assert capsys.readouterr().err == ""  # stderr wuerde die tui uebermalen
    finally:
        svc.close()
    cfg = make_cfg()
    main.start_podman(cfg, is_child=False, headless=True).close()
    assert "storage lock broken" in capsys.readouterr().err and not cfg.warnings


def test_start_podman_ok_is_quiet(monkeypatch, tmp_path, make_cfg, capsys):
    fake_podman(monkeypatch, tmp_path, LISTEN)
    cfg = make_cfg()
    svc = main.start_podman(cfg, is_child=False, headless=True)
    try:
        assert svc.error is None and not cfg.warnings and capsys.readouterr().err == ""
    finally:
        svc.close()


def test_start_failure_reaches_tui(monkeypatch, make_cfg, capsys):
    def boom(cls):
        raise OSError("broken")
    monkeypatch.setattr(podman.Service, "start", classmethod(boom))
    cfg = make_cfg()
    assert main.start_podman(cfg, is_child=False, headless=False) is None
    assert any("podman service won't start: broken" in w for w in cfg.warnings)
    assert capsys.readouterr().err == ""


def test_socket_file_is_not_ready(monkeypatch, tmp_path):
    """socket bound, but no listen(): file exists, but not ready."""
    fake_podman(monkeypatch, tmp_path, "s = socket.socket(socket.AF_UNIX); s.bind(path)\ntime.sleep(60)\n")
    svc = podman.Service.start()
    try:
        for _ in range(100):
            if os.path.exists(svc.sock):
                break
            time.sleep(0.02)
        assert os.path.exists(svc.sock)
        err = svc.ready(timeout=0.3)
        assert err and "not answer" in err
    finally:
        svc.close()


def test_service_dies_with_pyrate(monkeypatch, tmp_path):
    """SIGKILL auf pyrate: PR_SET_PDEATHSIG nimmt den dienst mit."""
    fake_podman(monkeypatch, tmp_path, LISTEN)
    script = (f"import sys, tempfile, time; sys.path.insert(0, {str(Path(__file__).parent.parent)!r})\n"
              f"tempfile.tempdir = {str(tmp_path)!r}\n"
              "from pyrate import podman\nsvc = podman.Service.start()\nprint(svc.proc.pid, flush=True)\n"
              "time.sleep(60)\n")
    owner = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    svc_pid = int(owner.stdout.readline())
    os.kill(owner.pid, signal.SIGKILL)
    owner.wait()
    for _ in range(100):
        if not _alive(svc_pid):
            break
        time.sleep(0.05)
    assert not _alive(svc_pid), "dienst lebt nach SIGKILL weiter"


def _alive(pid):
    """running noch (zombie zaehlt als tot: der init-reaper holt ihn ab)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_start_podman_child_does_not_spawn(monkeypatch, tmp_path):
    monkeypatch.setattr(podman.Service, "start", classmethod(lambda cls: pytest.fail("kind startet keinen dienst")))
    monkeypatch.setenv(podman.ENV_VAR, str(tmp_path))
    assert main.start_podman(None, is_child=True, headless=False).path == str(tmp_path)
