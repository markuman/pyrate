"""subagents = dog-fooding: the agent calls `pyrate -p` via the bash tool. the bash tool sets
PYRATE_SESSION/PYRATE_SUBAGENTS (agent.tool_env), the child recognizes itself from that. here:
detect child and limit the number of parallel children per parent via flock slots."""

import fcntl
import os
import re
import sys
import time
from pathlib import Path

_SID_RE = re.compile(r"[0-9a-f]{1,64}")  # session ids are hex, they end up in the path


def from_env(environ=None, stdin=None):
    """(parent session, max. parallel children) or None if we are not a child.

    the bash tool passes stdin=/dev/null. if stdin is a terminal, a human is sitting in front
    of it with only an inherited env (tmux/daemon that an agent started once) → not a child."""
    env = os.environ if environ is None else environ
    stdin = sys.stdin if stdin is None else stdin
    try:
        if stdin.isatty():
            return None
    except (AttributeError, ValueError):  # stdin closed/None
        pass
    parent = env.get("PYRATE_SESSION") or ""
    if not _SID_RE.fullmatch(parent):
        return None
    try:
        n = max(0, int(env.get("PYRATE_SUBAGENTS") or 0))
    except ValueError:
        n = 0
    return parent, n


def run_dir():
    """tmpfs per user (XDG_RUNTIME_DIR), otherwise next to the db. no /tmp: a foreign
    user could have placed a symlink there beforehand."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base and Path(base).is_dir():
        path = Path(base) / "pyrate"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "pyrate" / "run"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def acquire_slot(parent, n, poll=0.2):
    """blocks until one of n slots is free. returns the fd; the lock holds until close()
    or process end – even after SIGKILL the kernel cleans up, nothing gets stuck."""
    d = run_dir() / "slots" / parent
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    while True:
        for i in range(n):
            fd = os.open(d / f"slot-{i}", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                os.close(fd)
        time.sleep(poll)
