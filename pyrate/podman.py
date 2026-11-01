"""podman for the bash tool. inside the sandbox podman doesn't work: landlock forbids mount/pivot_root
(crun can't build a container) and ptrace outside – podman then considers the pause process dead
and removes pid files of running containers.

therefore: `podman system service` runs outside the sandbox, bash talks to it via CONTAINER_HOST
in remote mode. the socket lives in its own directory: mkdtemp(prefix="pyrate-podman-", 0700) under
$XDG_RUNTIME_DIR, not in $TMPDIR – $TMPDIR is scratch and stays writable in plan mode too.
so the directory is rw only in build mode (connect ok); in plan mode it becomes ro,
and ro also means no connect.

note, intentional: the service is not confined. `podman run -v ~:/h` reaches everything the
user can, and the model sees/controls the normal containers. hence opt-in (`podman: true`).

if pyrate dies hard (SIGKILL, oom), the kernel sends the service SIGTERM (PR_SET_PDEATHSIG)."""

import ctypes
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

SOCKET = "podman.sock"
LOG = "service.log"
ENV_VAR = "PYRATE_PODMAN"  # socket dir for child pyrates, which don't start their own service
_PR_SET_PDEATHSIG = 1


def config_dir():
    """registries.conf & co., needed by the remote client for image resolution."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "containers"


def _die_with_parent():
    """runs in the child before exec: pyrate gone → SIGTERM, even on SIGKILL to pyrate.
    applies to the thread that forks – start() runs in the main thread."""
    ctypes.CDLL(None).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)


class Service:
    def __init__(self, path, proc=None):
        self.path = str(path)
        self.proc = proc    # None = inherited from the parent, not our service
        self.error = None   # set by ready(): service dead or silent

    @property
    def sock(self):
        return os.path.join(self.path, SOCKET)

    @classmethod
    def start(cls):
        """own service for this run. stderr goes to the log in the service dir, not the tui terminal."""
        # not $TMPDIR: that is scratch and stays rw for bash in plan mode too (connect to the
        # socket would then be allowed). $XDG_RUNTIME_DIR (/run/user/UID) is unreachable for bash.
        d = tempfile.mkdtemp(prefix="pyrate-podman-", dir=os.environ.get("XDG_RUNTIME_DIR") or None)
        try:
            with open(os.path.join(d, LOG), "wb") as log:
                proc = subprocess.Popen(
                    ["podman", "system", "service", "--time=0", f"unix://{os.path.join(d, SOCKET)}"],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                    start_new_session=True,  # ctrl+c in the terminal must not kill it before cleanup
                    preexec_fn=_die_with_parent)
        except OSError:
            shutil.rmtree(d, ignore_errors=True)
            raise
        return cls(d, proc)

    @classmethod
    def inherited(cls):
        """child pyrate: socket from the parent, if it passed it along."""
        d = os.environ.get(ENV_VAR)
        return cls(d) if d and os.path.isdir(d) else None

    def _accepts(self):
        s = socket.socket(socket.AF_UNIX)
        try:
            s.connect(self.sock)  # the file exists before listen(), only connect says "ready"
            return True
        except OSError:
            return False
        finally:
            s.close()

    def ready(self, timeout=5.0):
        """wait until the service accepts. error text (with the last log lines) or None."""
        deadline = time.monotonic() + timeout
        while not self._accepts():
            if self.proc.poll() is not None:
                self.error = f"podman service exited (exit {self.proc.returncode})"
                break
            if time.monotonic() > deadline:
                self.error = f"podman service did not answer within {timeout:.0f}s"
                break
            time.sleep(0.05)
        if self.error:
            tail = self._log_tail()
            if tail:
                self.error += f": {tail}"
        return self.error

    def _log_tail(self, n=3):
        try:
            with open(os.path.join(self.path, LOG), errors="replace") as f:
                return " | ".join(x.strip() for x in deque(f, n) if x.strip())
        except OSError:
            return ""

    def env(self, tmp=None):
        """CONTAINER_HOST switches podman to remote. own auth file: the user's registry logins
        (/run/user/…/containers/auth.json) stay outside, a `podman login` lands in tmp."""
        env = {"CONTAINER_HOST": f"unix://{self.sock}", ENV_VAR: self.path}
        if tmp:
            env["REGISTRY_AUTH_FILE"] = os.path.join(tmp, "podman-auth.json")
        return env

    def close(self, timeout=5):
        """stop the service. containers started with -d keep running (conmon isn't attached)."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        shutil.rmtree(self.path, ignore_errors=True)
