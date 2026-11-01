"""load config: global yaml + folders blocks + project override, ${ENV} interpolation, validation."""

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import yaml

from . import plugins, sandbox


def global_dir():
    """~/.config/pyrate – determined at runtime, so XDG_CONFIG_HOME (e.g. in tests) applies."""
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "pyrate"


PROJECT_NAME = ".pyrate.yaml"

MODES = ("plan", "build")
EFFORTS = ("default", "low", "medium", "high", "xhigh", "max")

# tui keybindings: action → (default key, footer label, show in footer, priority binding).
# rebindable via `keys: {action: key}` in the config (global, folders and project – like `model`
# it grants no capability, so it's not in PROTECTED_KEYS). ctrl+m is deliberately unused: in
# legacy terminals it arrives as enter (\r) and would insert a newline instead of firing.
KEYS = {
    "toggle_mode":      ("tab",    "plan/build", True,  True),
    "pick_model":       ("ctrl+p", "model",      True,  True),
    "cycle_effort":     ("ctrl+t", "effort",     True,  True),
    "toggle_subagents": ("ctrl+s", "subagents",  True,  True),
    "toggle_net":       ("ctrl+n", "tcp",        True,  True),
    "mcp_panel":        ("ctrl+o", "mcp",        True,  True),
    "toggle_magic":     ("ctrl+g", "magic",      True,  True),
    "compact":          ("ctrl+k", "compact",    True,  True),
    "search":           ("ctrl+w", "search",     False, True),
    "cycle_view":       ("ctrl+v", "view",       False, True),
    "submit":           ("ctrl+j", "submit",     True,  False),  # footer label only, PromptInput handles the key
    "cancel":           ("escape", "cancel",     True,  False),
    "quit":             ("ctrl+d", "quit",       True,  True),
}
DEFAULT_KEYS = {action: spec[0] for action, spec in KEYS.items()}

_KEY_RE = re.compile(r"^(?:(?:ctrl|alt|shift|meta|super)\+)*[a-z0-9][a-z0-9_\-]*$"
                     r"|^(?:f(?:[1-9]|1[0-9]|2[0-4])|space|tab|escape|backspace|delete|up|down|left|right|"
                     r"home|end|pageup|pagedown|insert)$")
# enter/newline belong to the input field; ctrl+m IS enter (\r) in legacy terminals
_RESERVED_KEYS = ("enter", "newline", "ctrl+m")

# keys a project override may NOT set:
# - backends (or legacy base_url/api_key): otherwise a foreign repo could redirect the backend and
#   grab the key, or load arbitrary code via `plugin:`
# - mcp: stdio servers are arbitrary commands → code execution just by starting in a repo
# - tool_output_dir: gets cleaned at startup → a repo could point it at ~/Documents
# - db_path: sessions (incl. tool outputs) would land in the repo and get committed
# - skill_dirs: arbitrary directories as instruction source; project skills go via .pyrate/skills
# - folders: blocks there may set everything above → from a repo, `folders: {/: ...}` would be the bypass
# - subagents: no capability gain (bash already gives the model that), but a cost knob (fan-out)
# - sandbox, sandbox_ro, sandbox_rw: otherwise a foreign repo disables the sandbox or grants itself ~
# - podman: the service runs outside the sandbox, bind mounts reach everything
# - mcp_magic: starts servers the user deliberately left at enabled: false (egress, write access)
# model/small_model/compact_model may: they only select among the globally configured backends.
PROTECTED_KEYS = ("backends", "base_url", "api_key", "mcp", "mcp_magic", "tool_output_dir", "db_path", "skill_dirs",
                  "folders", "subagents", "sandbox", "sandbox_ro", "sandbox_rw", "podman")

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_MCP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


class ConfigError(Exception):
    pass


@dataclass
class Model:
    name: str
    context_limit: int | None = None
    effort: str | None = None        # reasoning-effort default for this model


@dataclass
class MCPServer:
    name: str
    command: str | None = None          # stdio
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    url: str | None = None              # streamable http
    headers: dict[str, str] = field(default_factory=dict)
    timeout: int = 120                  # s per tool call
    disabled_tools: list[str] = field(default_factory=list)
    enabled: bool = True                # false: don't start, enable at runtime (ctrl+o)
    oauth: dict | None = None           # {"scope": str|None}, login via `pyrate --mcp-login`
    description: str | None = None      # what the server does – for the mcp_magic router prompt
    error: str | None = None            # config error (e.g. env missing) → never connect


@dataclass
class Backend:
    id: str                 # name from the config, else host[:port] of the base_url
    base_url: str
    api_key: str = ""
    models: list[Model] = field(default_factory=list)
    context_limit: int | None = None  # default for models of this backend without their own limit
    effort: str | None = None         # default for models of this backend without their own effort
    plugin: object = field(default=None, repr=False)  # loaded plugin module (plugins.py) or None


@dataclass
class Config:
    backends: list[Backend]
    model: str                        # qualified: "<model>@<backend-id>"
    small_model: str | None = None    # qualified, None = model
    context_limit: int | None = None  # global fallback
    mode_default: str = "build"
    effort_default: str = "default"
    tools: dict[str, bool] = field(default_factory=dict)
    bash_timeout: int = 120
    output_max_lines: int = 2000
    output_max_bytes: int = 50 * 1024
    tool_output_dir: Path | None = None  # None = ~/.local/share/pyrate/tool-output
    max_steps: int = 50
    request_timeout: int = 600
    db_path: Path | None = None
    compact_model: str | None = None  # None = current main model
    mcp_magic: str | None = None      # qualified model for mcp routing before each prompt, None = off
    mcp_magic_timeout: int = 20       # s per router call; cold ollama (model load) can take a while
    auto_compact: bool = True
    skill_dirs: list[Path] = field(default_factory=list)  # extra skill directories
    mcp: list[MCPServer] = field(default_factory=list)
    subagents: int = 0                # max. parallel child pyrates (dog-fooding via bash), 0 = off
    sandbox: bool = True              # tools only in cwd + $TMPDIR, bash via landlock
    sandbox_ro: list[Path] = field(default_factory=list)  # additionally readable, e.g. ~/.gitconfig
    sandbox_rw: list[Path] = field(default_factory=list)  # additionally writable, e.g. ~/.cache/uv
    podman: bool = False              # bash: podman via a service outside the sandbox (build only)
    plan_bash: bool = False           # derived: landlock can do read-only + no tcp (abi >= 4)
    container_cli: bool = False       # derived: podman/docker in PATH, see agent.PODMAN_OFF
    global_dir: Path | None = None    # dir of the global config (AGENTS.md, skills/)
    config_arg: Path | None = None    # --config, absolute; subagents need the same one
    keys: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_KEYS))  # tui shortcuts, action → key
    warnings: list[str] = field(default_factory=list)

    def title_model(self):
        return self.small_model or self.model

    # ------------------------------------------------------------ model resolution
    # spellings: "model@backend" (always unambiguous) or just "model". without backend the
    # model must be listed in exactly one backend – or there is only one backend.
    # a model only needs to be listed for the picker; with @backend any works.

    def _backend(self, bid):
        return next((b for b in self.backends if b.id == bid), None)

    def resolve(self, ref):
        """(Backend, model name). raises ConfigError."""
        ref = str(ref or "").strip()
        if not ref:
            raise ConfigError("no model given")
        name, sep, bid = ref.rpartition("@")
        if sep and name:
            be = self._backend(bid)
            if be is not None:
                return be, name
            # "@" can also be part of the model name – then try the whole thing below
        hits = [b for b in self.backends if any(m.name == ref for m in b.models)]
        if len(hits) == 1:
            return hits[0], ref
        if len(hits) > 1:
            opts = ", ".join(f"{ref}@{b.id}" for b in hits)
            raise ConfigError(f"model '{ref}' exists in multiple backends, be specific: {opts}")
        if len(self.backends) == 1:
            return self.backends[0], ref
        known = ", ".join(b.id for b in self.backends)
        raise ConfigError(f"model '{ref}' is listed in no backend. "
                          f"use @backend (backends: {known})")

    def qualify(self, ref):
        be, name = self.resolve(ref)
        return f"{name}@{be.id}"

    def model_ids(self):
        """all listed models as "model@backend", in config order."""
        return [f"{m.name}@{b.id}" for b in self.backends for m in b.models]

    def limit_for(self, ref):
        try:
            be, name = self.resolve(ref)
        except ConfigError:
            return self.context_limit
        for m in be.models:
            if m.name == name and m.context_limit:
                return m.context_limit
        return be.context_limit or self.context_limit

    def effort_for(self, ref):
        """reasoning effort for a model: model level → backend level → effort_default."""
        try:
            be, name = self.resolve(ref)
        except ConfigError:
            return self.effort_default
        for m in be.models:
            if m.name == name and m.effort:
                return m.effort
        return be.effort or self.effort_default


def interpolate(value):
    """replace ${VAR} / ${VAR:-default} recursively in all strings."""
    if isinstance(value, str):
        def repl(m):
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable ${{{name}}} is not set")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate(v) for v in value]
    return value


def _load_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: broken yaml: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _apply_folders(data, cwd, warnings):
    """`folders: {path: block}` from the global config: every block whose path contains cwd
    is merged on top – most specific last. comes from the global config, so it may
    also set PROTECTED_KEYS (mcp per directory tree, independent of the git root)."""
    folders = data.pop("folders", None)
    if folders is None:
        return data
    if not isinstance(folders, dict):
        raise ConfigError("folders must be a mapping path → config block")
    cwd = Path(cwd).resolve()
    hits = []
    for key, block in folders.items():
        path = Path(str(key)).expanduser()
        if not path.is_absolute():
            raise ConfigError(f"folders: {key!r} must be absolute or start with ~")
        if not isinstance(block, dict):
            raise ConfigError(f"folders.{key} must be a mapping")
        path = path.resolve()
        if cwd.is_relative_to(path):
            hits.append((len(path.parts), key, block))
    for _, key, block in sorted(hits, key=lambda h: h[0]):
        if "folders" in block:
            warnings.append(f"folders.{key}: nested 'folders' ignored")
            block = {k: v for k, v in block.items() if k != "folders"}
        data = _merge(data, block)
    return data


def _find_project_file(cwd):
    cwd = Path(cwd).resolve()
    for d in (cwd, *cwd.parents):
        p = d / PROJECT_NAME
        if p.is_file():
            return p
        if (d / ".git").exists():
            break
    return None


def _parse_models(raw, where):
    if raw is not None and not isinstance(raw, list):
        raise ConfigError(f"{where}.models must be a list")
    models = []
    seen = set()
    for item in raw or []:
        if isinstance(item, str):
            m = Model(item)
        elif isinstance(item, dict) and "name" in item:
            m = Model(str(item["name"]),
                      _opt_int(item.get("context_limit"), f"{where}.models[].context_limit"),
                      _effort(item.get("effort"), f"{where}.models[].effort"))
        else:
            raise ConfigError(f"{where}.models: entry {item!r} must be a string or {{name: ...}}")
        if m.name not in seen:
            models.append(m)
            seen.add(m.name)
    return models


def _backend_id(base_url, name, where):
    if name:
        bid = str(name).strip()
    else:
        bid = urlparse(base_url).netloc.rpartition("@")[2]  # drop possible user:pass@
    if not bid or "@" in bid or any(c.isspace() for c in bid):
        raise ConfigError(f"{where}: invalid backend name {bid!r} (no @, no spaces)")
    return bid


def _parse_backends(raw, warnings):
    """backends list. env vars per backend: if a key is missing, only that backend is
    skipped (with a warning) instead of breaking the whole config."""
    if not isinstance(raw, list) or not raw:
        raise ConfigError("backends must be a non-empty list")
    backends = []
    for i, spec in enumerate(raw):
        where = f"backends[{i}]"
        if not isinstance(spec, dict):
            raise ConfigError(f"{where} must be a mapping")
        if spec.get("enabled", True) is False:
            continue
        try:
            spec = interpolate(spec)
        except ConfigError as e:
            warnings.append(f"{where} ({spec.get('name') or spec.get('base_url')}) skipped: {e}")
            continue
        base_url = str(spec.get("base_url") or "").rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(f"{where}.base_url missing or doesn't start with http(s)://")
        bid = _backend_id(base_url, spec.get("name"), where)
        where = f"backend '{bid}'"
        if any(b.id == bid for b in backends):
            raise ConfigError(f"{where} exists twice – set 'name' for the same host")
        plugin = None
        if spec.get("plugin"):
            try:
                plugin = plugins.load(spec["plugin"])
            except plugins.PluginError as e:
                warnings.append(f"{where} skipped: {e}")
                continue
        backends.append(Backend(
            id=bid, base_url=base_url, api_key=str(spec.get("api_key") or ""),
            models=_parse_models(spec.get("models"), where),
            context_limit=_opt_int(spec.get("context_limit"), f"{where}.context_limit"),
            effort=_effort(spec.get("effort"), f"{where}.effort"),
            plugin=plugin,
        ))
    if not backends:
        raise ConfigError("no backend available (all disabled or skipped)")
    return backends


def _legacy_backends(data, warnings):
    """legacy format (base_url/api_key/models at top level) → one backend."""
    warnings.append("config: base_url/api_key/models at top level are deprecated, "
                    "please switch to 'backends:' (see README)")
    return [{"base_url": data.get("base_url"), "api_key": data.get("api_key"),
             "models": data.get("models") or []}]


def _pick_model(backends, ref, what):
    """ref → (Backend, model name), error prefixed with the field name."""
    try:
        return Config(backends=backends, model="").resolve(ref)
    except ConfigError as e:
        raise ConfigError(f"{what}: {e}") from e


def _effort(v, name):
    if v is None or v == "":
        return None
    v = str(v)
    if v not in EFFORTS:
        raise ConfigError(f"{name} must be one of {EFFORTS}, not {v!r}")
    return v


def _opt_int(v, name):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{name}: {v!r} is not a number") from e


def _non_neg_int(v, name):
    if isinstance(v, bool):  # yaml `true` would otherwise silently become 1
        raise ConfigError(f"{name}: {v!r} is not a number")
    n = _opt_int(v, name) or 0
    if n < 0:
        raise ConfigError(f"{name} must be >= 0, not {n}")
    return n


def _str_list(v, name):
    if v is None:
        return []
    if not isinstance(v, list) or not all(isinstance(x, (str, int, float)) for x in v):
        raise ConfigError(f"{name} must be a list of strings")
    return [str(x) for x in v]


def _str_map(v, name):
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ConfigError(f"{name} must be a mapping")
    return {str(k): str(x) for k, x in v.items()}


def _parse_mcp(raw, warnings):
    """mcp section. `enabled: false` servers stay in (enable at runtime).
    env vars per server: if one is missing, only that server gets an error instead
    of breaking the whole config."""
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ConfigError("mcp must be a mapping name → server")
    servers = []
    for name, spec in raw.items():
        where = f"mcp.{name}"
        if not _MCP_NAME_RE.match(str(name)):
            raise ConfigError(f"{where}: name only a-z, 0-9, _ and -, max. 32 chars")
        if not isinstance(spec, dict):
            raise ConfigError(f"{where} must be a mapping")
        enabled = spec.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"{where}.enabled must be true or false")
        error = None
        try:
            spec = interpolate(spec)
        except ConfigError as e:
            error = str(e)  # keep parsing raw, such a server never connects
            if enabled:
                warnings.append(f"{where}: {e} – server stays off")
        command, url = spec.get("command"), spec.get("url")
        if bool(command) == bool(url):
            raise ConfigError(f"{where}: exactly one of 'command' (stdio) or 'url' (http)")
        if command is not None and not isinstance(command, str):
            raise ConfigError(f"{where}.command must be a string (arguments go in 'args')")
        if url and not str(url).startswith(("http://", "https://")):
            raise ConfigError(f"{where}.url must start with http:// or https://")
        headers = _str_map(spec.get("headers"), f"{where}.headers")
        oauth = _parse_oauth(spec.get("oauth"), where)
        if oauth is not None and not url:
            raise ConfigError(f"{where}.oauth only works with 'url' (http)")
        if oauth is not None and any(k.lower() == "authorization" for k in headers):
            raise ConfigError(f"{where}: oauth and a custom Authorization header are mutually exclusive")
        cwd = spec.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ConfigError(f"{where}.cwd must be a path (string)")
        description = spec.get("description")
        if description is not None and not isinstance(description, str):
            raise ConfigError(f"{where}.description must be a string")
        servers.append(MCPServer(
            name=str(name),
            command=command,
            args=_str_list(spec.get("args"), f"{where}.args"),
            env=_str_map(spec.get("env"), f"{where}.env"),
            cwd=str(Path(cwd).expanduser()) if cwd else None,
            url=str(url) if url else None,
            headers=headers,
            timeout=_opt_int(spec.get("timeout"), f"{where}.timeout") or 120,
            disabled_tools=_str_list(spec.get("disabled_tools"), f"{where}.disabled_tools"),
            enabled=enabled,
            oauth=oauth,
            description=description,
            error=error,
        ))
    return servers


def _parse_oauth(raw, where):
    """oauth: true | false | {scope: "a b"}. without scope the login takes the server's scopes."""
    if raw is None or raw is False:
        return None
    if raw is True:
        return {"scope": None}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}.oauth must be true/false or {{scope: ...}}")
    unknown = set(raw) - {"scope"}
    if unknown:
        raise ConfigError(f"{where}.oauth: unknown: {', '.join(sorted(map(str, unknown)))}")
    scope = raw.get("scope")
    if scope is not None and not isinstance(scope, str):
        raise ConfigError(f"{where}.oauth.scope must be a string (space-separated)")
    return {"scope": scope or None}


def _parse_keys(raw):
    """`keys:` mapping action → key, merged over DEFAULT_KEYS. validation: known action, key
    format, no reserved keys (enter/newline/ctrl+m), no bare keys without ctrl/alt/meta (those
    would swallow typing in the input), no key bound twice."""
    keys = dict(DEFAULT_KEYS)
    if raw is None:
        return keys
    if not isinstance(raw, dict):
        raise ConfigError("keys must be a mapping action → key, e.g. {toggle_magic: ctrl+g}")
    for raw_action, key in raw.items():
        action = str(raw_action).strip().lower()
        if action not in DEFAULT_KEYS:
            raise ConfigError(f"keys: unknown action {raw_action!r} (known: {', '.join(DEFAULT_KEYS)})")
        key = str(key).strip().lower()
        if key in _RESERVED_KEYS:
            raise ConfigError(f"keys.{action}: {key!r} is reserved (enter/newline belong to the input, "
                              "ctrl+m is enter in legacy terminals)")
        if not _KEY_RE.match(key):
            raise ConfigError(f"keys.{action}: invalid key {key!r} (e.g. ctrl+g, alt+x, f5, shift+tab)")
        mods = key.split("+")[:-1]
        bare_ok = key in ("tab", "escape", "shift+tab") or key.startswith("f")  # f-tasten: drucken kein zeichen
        if not any(m in ("ctrl", "alt", "meta", "super") for m in mods) and not bare_ok:
            raise ConfigError(f"keys.{action}: {key!r} needs ctrl/alt/meta/super – a bare key would "
                              "swallow typing in the input field")
        keys[action] = key
    seen = {}
    for action, key in keys.items():
        if key in seen:
            raise ConfigError(f"keys: {key!r} is bound twice ({seen[key]} and {action})")
        seen[key] = action
    return keys


def build(data, warnings=None, global_dir=None):
    data = dict(data)
    warnings = list(warnings or [])
    mcp_raw = data.pop("mcp", None)
    backends_raw = data.pop("backends", None)
    if backends_raw is None and "base_url" in data:
        backends_raw = _legacy_backends(data, warnings)
    elif backends_raw is not None and "base_url" in data:
        warnings.append("config: top-level base_url is ignored, there is 'backends:'")
    if backends_raw is None:
        raise ConfigError("required field 'backends' missing (see README)")
    for key in ("base_url", "api_key", "models"):
        data.pop(key, None)
    backends = _parse_backends(backends_raw, warnings)
    data = interpolate(data)

    # default model: 'model' or the first listed model of the first backend
    model_ref = data.get("model")
    if not model_ref:
        first = next((b for b in backends if b.models), None)
        if first is None:
            raise ConfigError("no model: set 'model' or list 'models' in a backend")
        model_ref = f"{first.models[0].name}@{first.id}"
    be, name = _pick_model(backends, model_ref, "model")
    if not any(m.name == name for m in be.models):
        be.models.insert(0, Model(name))  # default should show up in the picker
    model = f"{name}@{be.id}"
    small = compact = None
    if data.get("small_model"):
        sbe, sname = _pick_model(backends, data["small_model"], "small_model")
        small = f"{sname}@{sbe.id}"
    if data.get("compact_model"):
        cbe, cname = _pick_model(backends, data["compact_model"], "compact_model")
        compact = f"{cname}@{cbe.id}"
    magic = None
    raw_magic = data.get("mcp_magic", False)
    if raw_magic is True:
        magic = small or model  # true → small_model (fallback main model)
    elif isinstance(raw_magic, str) and raw_magic:
        mbe, mname = _pick_model(backends, raw_magic, "mcp_magic")
        magic = f"{mname}@{mbe.id}"
    elif raw_magic not in (None, False):
        raise ConfigError(f"mcp_magic must be true, false or model@backend, not {raw_magic!r}")

    mode = data.get("mode_default", "build")
    if mode not in MODES:
        raise ConfigError(f"mode_default must be one of {MODES}, not {mode!r}")
    effort = str(data.get("effort_default", "default"))
    if effort not in EFFORTS:
        raise ConfigError(f"effort_default must be one of {EFFORTS}, not {effort!r}")
    tools = data.get("tools") or {}
    if not isinstance(tools, dict):
        raise ConfigError("tools must be a mapping, e.g. {grep: false}")
    use_sandbox = bool(data.get("sandbox", True))
    landlock = sandbox.abi() if use_sandbox else 0
    if use_sandbox and tools.get("bash", True) and landlock < 1:
        # better loud at startup than exit 126 on every bash call, which only the model sees
        warnings.append("sandbox: no landlock (linux >= 5.13 required), bash is off. "
                        "'sandbox: false' in the global config disables the sandbox entirely")
        tools = {**tools, "bash": False}
    # only the literal true: an intentional hole, `podman: "no"` must not open it.
    # without sandbox podman works normally anyway
    use_podman = data.get("podman") is True and use_sandbox
    if use_podman and not sandbox.restricts_unix():
        # without RESOLVE_UNIX plan mode could also reach the socket → podman run -v $PWD:… would write after all
        warnings.append("podman: landlock cannot block unix sockets here (kernel too old), podman stays off")
        use_podman = False
    if use_podman and not shutil.which("podman"):
        warnings.append("podman: not installed, stays off")
        use_podman = False
    ctx = _opt_int(data.get("context_limit"), "context_limit")
    db_path = data.get("db_path")
    out_dir = data.get("tool_output_dir")
    max_bytes = _opt_int(data.get("output_max_bytes"), "output_max_bytes")
    if "output_limit" in data:  # old name, was a char limit for the whole tool result
        warnings.append("output_limit is deprecated, now output_max_bytes (plus output_max_lines)")
        max_bytes = max_bytes or _opt_int(data.get("output_limit"), "output_limit")
    return Config(
        backends=backends,
        model=model,
        small_model=small,
        context_limit=ctx,
        mode_default=mode,
        effort_default=effort,
        tools={str(k): bool(v) for k, v in tools.items()},
        bash_timeout=_opt_int(data.get("bash_timeout"), "bash_timeout") or 120,
        output_max_lines=_opt_int(data.get("output_max_lines"), "output_max_lines") or 2000,
        output_max_bytes=max_bytes or 50 * 1024,
        tool_output_dir=Path(out_dir).expanduser() if out_dir else None,
        max_steps=_opt_int(data.get("max_steps"), "max_steps") or 50,
        request_timeout=_opt_int(data.get("request_timeout"), "request_timeout") or 600,
        db_path=Path(db_path).expanduser() if db_path else None,
        compact_model=compact,
        mcp_magic=magic,
        mcp_magic_timeout=_opt_int(data.get("mcp_magic_timeout"), "mcp_magic_timeout") or 20,
        auto_compact=bool(data.get("auto_compact", True)),
        skill_dirs=[Path(p).expanduser() for p in _str_list(data.get("skill_dirs"), "skill_dirs")],
        mcp=_parse_mcp(mcp_raw, warnings),
        subagents=_non_neg_int(data.get("subagents"), "subagents"),
        sandbox=use_sandbox,
        sandbox_ro=[Path(p).expanduser() for p in _str_list(data.get("sandbox_ro"), "sandbox_ro")],
        sandbox_rw=[Path(p).expanduser() for p in _str_list(data.get("sandbox_rw"), "sandbox_rw")],
        podman=use_podman,
        # bash in plan mode only if landlock can do fs read-only *and* no tcp. otherwise not,
        # whoever wants bash with net uses build
        plan_bash=landlock >= 4,
        # once here instead of in agent.system_prompt(): that runs on every keypress
        container_cli=any(shutil.which(c) for c in ("podman", "docker")),
        global_dir=Path(global_dir) if global_dir else None,
        keys=_parse_keys(data.get("keys")),
        warnings=warnings,
    )


def load(cwd=None, global_path=None):
    cwd = cwd or os.getcwd()
    explicit = bool(global_path)
    global_path = Path(global_path) if global_path else global_dir() / "config.yaml"
    data = {}
    warnings = []
    if global_path.is_file():
        data = _load_yaml(global_path)
    data = _apply_folders(data, cwd, warnings)
    project = _find_project_file(cwd)
    if project and project.resolve() != global_path.resolve():
        override = _load_yaml(project)
        for key in PROTECTED_KEYS:
            if key in override:
                warnings.append(f"{project}: '{key}' ignored in project override (global only)")
                override.pop(key)
        data = _merge(data, override)
    if not data:
        raise ConfigError(f"no config found. create {global_path} (see README)")
    # AGENTS.md and skills/ live next to the global config (with --config, right there)
    cfg = build(data, warnings, global_dir=global_path.parent)
    if explicit:
        cfg.config_arg = global_path.absolute()
    return cfg
