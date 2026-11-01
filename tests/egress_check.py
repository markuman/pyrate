"""egress-check: `strace -f -e trace=connect` via einen kompletten print-mode-lauf.
erlaubt sind nur inet-connects zum backend (+ dns auf port 53 im --real-modus).

  python tests/egress_check.py           # gegen lokales fake-backend (default, offline)
  python tests/egress_check.py --real    # gegen das backend aus der echten config
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
INET_RE = re.compile(r'sa_family=AF_INET6?, sin6?_port=htons\((\d+)\).*?(?:inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")')
UNIX_RE = re.compile(r'sa_family=AF_UNIX, sun_path=@?"([^"]*)"')


def run_traced(cmd, cwd, env):
    log = Path(tempfile.mkstemp(prefix="pyrate-strace-")[1])
    r = subprocess.run(["strace", "-f", "-qq", "-e", "trace=connect", "-o", str(log), *cmd],
                       cwd=cwd, env=env, capture_output=True, text=True, timeout=300)
    text = log.read_text()
    log.unlink()
    return r, text


def parse(text):
    inet, unix = [], set()
    for line in text.splitlines():
        if m := INET_RE.search(line):
            inet.append((m.group(2) or m.group(3), int(m.group(1))))
        elif m := UNIX_RE.search(line):
            unix.add(m.group(1))
    return inet, unix


def fake_setup(tmp):
    script = [
        {"tool_calls": [{"name": "bash", "arguments": {"command": "echo egress"}}]},
        {"text": "fertig"},
    ]
    (tmp / "script.json").write_text(json.dumps(script))
    proc = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_backend.py"), "--script", str(tmp / "script.json")],
                            stdout=subprocess.PIPE, text=True)
    port = int(proc.stdout.readline())
    cfg = tmp / "config.yaml"
    cfg.write_text(f"backends:\n  - base_url: http://127.0.0.1:{port}/v1\n    models: [fake]\n"
                   f"small_model: fake-small\ndb_path: {tmp / 'db.sqlite'}\n")
    return proc, ["--config", str(cfg)], {("127.0.0.1", port)}, False


def real_setup(tmp):
    """erlaubt sind alle konfigurierten backends (der lauf nutzt default- und small-modell)."""
    sys.path.insert(0, str(ROOT))
    from pyrate import config
    cfg = config.load(os.getcwd())
    addrs = set()
    for be in cfg.backends:
        u = urlparse(be.base_url)
        port = u.port or (443 if u.scheme == "https" else 80)
        addrs |= {(ai[4][0], port) for ai in socket.getaddrinfo(u.hostname, port, proto=socket.IPPROTO_TCP)}
    import yaml
    dbcfg = tmp / "override.yaml"
    # echte config, aber wegwerf-db. mcp-server raus: das sind eigene, bewusst konfigurierte
    # egress-quellen (remote-server, lokale prozesse mit eigenem netz) – geprueft wird pyrate selbst
    data = yaml.safe_load((config.global_dir() / "config.yaml").read_text()) or {}
    if data.pop("mcp", None):
        print("note: mcp-server aus der config for den check ignored")
    data["db_path"] = str(tmp / "db.sqlite")
    dbcfg.write_text(yaml.safe_dump(data))
    return None, ["--config", str(dbcfg)], addrs, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    if not shutil.which("strace"):
        sys.exit("egress: strace fehlt (pacman -S strace / apt install strace)")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        proc, extra, allowed, allow_dns = (real_setup if a.real else fake_setup)(tmp)
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        try:
            r, text = run_traced([sys.executable, "-m", "pyrate", *extra, "-p", "sag nur: ok"], tmp, env)
        finally:
            if proc:
                proc.terminate()
        if r.returncode != 0:
            print(r.stdout, r.stderr, sep="\n")
            sys.exit(f"egress: pyrate-lauf fehlgeschlagen (exit {r.returncode})")

    inet, unix = parse(text)
    bad = [(h, p) for h, p in inet if (h, p) not in allowed and not (allow_dns and p == 53)]
    print(f"erlaubt: {sorted(allowed)}")
    print(f"inet-connects: {len(inet)}  →  {sorted(set(inet))}")
    if unix:
        print(f"lokale unix-sockets (kein egress): {sorted(unix)}")
    if not inet:
        sys.exit("egress: keine inet-connects gesehen – strace-parsing broken?")
    if bad:
        sys.exit(f"egress: UNERLAUBTE verbindungen: {sorted(set(bad))}")
    print("egress: ok 🏴‍☠️")


if __name__ == "__main__":
    main()
