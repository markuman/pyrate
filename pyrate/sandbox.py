"""path boundaries for the tools, enforced via landlock (linux ≥5.13), no external deps.

two halves:
- Sandbox: which paths are readable/writable. tools._resolve checks read/write/edit/grep/find
  against it, wrap() builds the bash call from it. grants apply to both.
- main(): runs as its own script (`python -I -S sandbox.py --ro … --rw … [--no-net] -- bash -c …`),
  arms landlock and execs the command. landlock is inherited by all children and can only be
  tightened further: even a sub-pyrate cannot get out.

stdlib only, no imports from pyrate: the script runs with -S (no site, faster start)."""

import ctypes
import os
import struct
import sys
from pathlib import Path

# what bash needs to live. if a path doesn't exist, it is skipped.
# target platform: arch, at most debian. no snap, no nix.
# /run not completely: usb sticks (/run/media), fuse mounts (/run/user/*/gvfs) and
# registry logins (/run/user/*/containers/auth.json) live there. only dns is needed
# (resolv.conf often points at systemd-resolved).
SYSTEM_RO = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/opt", "/proc", "/sys",
             "/var", "/run/systemd/resolve")
SYSTEM_RW = ("/dev",)  # /dev/null, /dev/tty, /dev/shm
# unix sockets bash may always reach (nss-resolve via varlink). all others are closed,
# otherwise `systemd-run --user` via the session bus is the escape hatch (also ssh-/gpg-agent, docker.sock).
SYSTEM_SOCK = ("/run/systemd/resolve",)

# linux/landlock.h – the syscall numbers are the same on all archs
_SYS_CREATE, _SYS_ADD_RULE, _SYS_RESTRICT = 444, 445, 446
_CREATE_VERSION = 1
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1, 2, 4, 8
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
RESOLVE_UNIX = 1 << 16  # connect() on path unix sockets, new (abi 9). older kernels: not covered
NET_ALL = 1 | 2  # bind_tcp | connect_tcp, since abi 4
RO = EXECUTE | READ_FILE | READ_DIR
# a rule on a file must not grant more than this (RESOLVE_UNIX: e.g. sandbox_rw on a single agent socket)
FILE_ONLY = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV | RESOLVE_UNIX


class Sandbox:
    """readable (ro) and writable (rw) roots, plus scratch: writable in plan mode too
    ($TMPDIR, /dev). paths resolved, so the comparison with tools._resolve matches."""

    def __init__(self, ro=(), rw=(), scratch=(), sock=()):
        self.ro, self.rw = [], []
        self.scratch = [Path(p).resolve() for p in scratch]
        self.sock = [Path(p).resolve() for p in sock]  # only connect() on sockets under these, in both modes
        for p in ro:
            self.grant(p)
        for p in rw:
            self.grant(p, write=True)

    def grant(self, path, write=False):
        if not path:
            return
        p = Path(path).expanduser().resolve()
        roots = self.rw if write else self.ro
        if p not in roots:
            roots.append(p)

    def allows(self, path, write=False):
        roots = self.scratch + self.rw + ([] if write else self.ro)
        return any(Path(path).is_relative_to(r) for r in roots)

    def wrap(self, argv, readonly=False, net=True):
        """readonly (plan mode): rw becomes ro, only scratch stays writable.
        net=False: additionally no tcp (--no-net). the two are independent."""
        ro, rw = (self.ro + self.rw, self.scratch) if readonly else (self.ro, self.scratch + self.rw)
        args = [sys.executable, "-I", "-S", os.path.abspath(__file__)]
        for p in ro:
            args += ["--ro", str(p)]
        for p in rw:
            args += ["--rw", str(p)]
        for p in self.sock:
            args += ["--sock", str(p)]
        if not net:
            args.append("--no-net")
        return [*args, "--", *argv]


# ---------------------------------------------------------------- landlock

def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def _call(fn, *args):
    r = fn(*[ctypes.c_long(a) if isinstance(a, int) else a for a in args])
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return r


def abi():
    """landlock version of the kernel, 0 = none."""
    if not sys.platform.startswith("linux"):
        return 0  # syscall 444 means something else elsewhere
    try:
        return _call(_libc().syscall, _SYS_CREATE, None, 0, _CREATE_VERSION)
    except OSError:
        return 0


def _handled(version):
    """all filesystem rights this kernel knows. anything no rule allows is forbidden."""
    bits = (1 << 13) - 1  # abi 1: execute … make_sym
    if version >= 2:
        bits |= REFER
    if version >= 3:
        bits |= TRUNCATE
    if version >= 5:
        bits |= IOCTL_DEV
    if version >= 5 and _accepts(bits | RESOLVE_UNIX):  # don't hardcode the abi number: ask
        bits |= RESOLVE_UNIX
    return bits


def _accepts(fs_bits):
    """does the kernel know all these rights? (unknown bits → EINVAL on creation)"""
    attr = ctypes.create_string_buffer(struct.pack("=QQ", fs_bits, 0), 16)
    try:
        os.close(_call(_libc().syscall, _SYS_CREATE, attr, 16, 0))
    except OSError:
        return False
    return True


def restricts_unix():
    """does landlock block connect() on unix sockets here? without it bash can reach e.g. the session bus."""
    version = abi()
    return version >= 1 and bool(_handled(version) & RESOLVE_UNIX)


def restrict(ro, rw, net=True, sock=()):
    """restrict this process (and all future children) to ro/rw. net=False: tcp
    (bind + connect) fully forbidden, udp is not covered by landlock. unix sockets (where the kernel
    can): connect only under rw and sock."""
    libc = _libc()
    version = abi()
    if version < 1:
        raise OSError("landlock not available (kernel < 5.13 or lsm not active)")
    if not net and version < 4:
        raise OSError("landlock cannot block net here (kernel < 6.7)")
    handled = _handled(version)
    # struct landlock_ruleset_attr {u64 fs; u64 net}. older kernels also take the 16 bytes as long as net is 0
    attr = ctypes.create_string_buffer(struct.pack("=QQ", handled, 0 if net else NET_ALL), 16)
    ruleset = _call(libc.syscall, _SYS_CREATE, attr, 16, 0)
    try:
        for paths, access in ((ro, RO), (rw, handled), (sock, handled & RESOLVE_UNIX)):
            for p in paths:
                allowed = access if os.path.isdir(p) else access & FILE_ONLY
                if not allowed:
                    continue  # e.g. sock on a kernel without RESOLVE_UNIX: everything is open there anyway
                try:
                    fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
                except OSError:
                    continue
                try:
                    rule = ctypes.create_string_buffer(struct.pack("=Qi", allowed, fd), 12)
                    _call(libc.syscall, _SYS_ADD_RULE, ruleset, _RULE_PATH_BENEATH, rule, 0)
                finally:
                    os.close(fd)
        _call(libc.prctl, _PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        _call(libc.syscall, _SYS_RESTRICT, ruleset, 0)
    finally:
        os.close(ruleset)


def main(argv):
    """--ro PFAD … --rw PFAD … --sock PFAD … [--no-net] -- befehl …"""
    usage = "usage: sandbox.py [--ro PFAD] [--rw PFAD] [--sock PFAD] [--no-net] -- befehl …"
    lists = {"--ro": [], "--rw": [], "--sock": []}
    net, i = True, 0
    while i < len(argv) and argv[i] != "--":
        if argv[i] == "--no-net":
            net, i = False, i + 1
            continue
        if argv[i] not in lists or i + 1 >= len(argv):
            print(f"pyrate-sandbox: {argv[i]!r}?\n{usage}", file=sys.stderr)
            return 126
        lists[argv[i]].append(argv[i + 1])
        i += 2
    if i + 1 >= len(argv):  # no "--" or nothing after it: never start anything unsandboxed
        print(f"pyrate-sandbox: no command\n{usage}", file=sys.stderr)
        return 126
    try:
        restrict(lists["--ro"], lists["--rw"], net, lists["--sock"])
    except OSError as e:
        print(f"pyrate-sandbox: {e}. command not executed", file=sys.stderr)
        return 126
    cmd = argv[i + 1:]
    try:
        os.execvp(cmd[0], cmd)
    except OSError as e:  # e.g. bash under /home/linuxbrew: no ro root → no stacktrace for the model
        print(f"pyrate-sandbox: cannot start {cmd[0]}: {e}", file=sys.stderr)
        return 126


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
