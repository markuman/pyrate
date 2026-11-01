from pathlib import Path

import pytest

from pyrate import config, sandbox


@pytest.fixture(autouse=True)
def _kernel_independent(monkeypatch):
    """hier werden warnings und tools exakt verglichen, egal ob der kernel landlock kann
    (sonst kommt "no landlock, bash is off" dazu)."""
    monkeypatch.setattr(sandbox, "abi", lambda: 9)


def write(path, text):
    path.write_text(text)
    return path


TWO = """
backends:
  - name: api
    base_url: https://api.example.com/v1/
    api_key: ${LITELLM_API_KEY}
    context_limit: 262144
    models:
      - openrouter/qwen3.8-27b
      - name: openrouter/claude-sonnet
        context_limit: 200000
  - base_url: http://localhost:11434/v1
    models: [qwen3:8b, openrouter/qwen3.8-27b]
model: openrouter/qwen3.8-27b@api
small_model: qwen3:8b
"""


def two(monkeypatch, tmp_path, extra=""):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-geheim")
    return config.load(tmp_path, write(tmp_path / "g.yaml", TWO + extra))


def test_backends_and_ids(monkeypatch, tmp_path):
    cfg = two(monkeypatch, tmp_path)
    api, local = cfg.backends
    assert (api.id, api.base_url, api.api_key) == ("api", "https://api.example.com/v1", "sk-geheim")
    assert local.id == "localhost:11434" and local.api_key == ""  # id = host:port, wenn kein name
    assert cfg.model_ids() == [
        "openrouter/qwen3.8-27b@api", "openrouter/claude-sonnet@api",
        "qwen3:8b@localhost:11434", "openrouter/qwen3.8-27b@localhost:11434",
    ]
    assert cfg.model == "openrouter/qwen3.8-27b@api"
    assert cfg.small_model == "qwen3:8b@localhost:11434"  # unqualifiziert, aber eindeutig
    assert cfg.title_model() == cfg.small_model
    assert cfg.warnings == []


def test_resolve_rules(monkeypatch, tmp_path):
    cfg = two(monkeypatch, tmp_path)
    be, name = cfg.resolve("openrouter/claude-sonnet")
    assert (be.id, name) == ("api", "openrouter/claude-sonnet")
    assert cfg.qualify("irgendwas@localhost:11434") == "irgendwas@localhost:11434"  # ungelistet mit @ ok
    with pytest.raises(config.ConfigError, match="multiple backends.*@api.*@localhost:11434"):
        cfg.resolve("openrouter/qwen3.8-27b")
    with pytest.raises(config.ConfigError, match="listed in no backend"):
        cfg.resolve("gpt-9")
    # "@" im modellnamen, wenn dahinter no backend available steht
    single = config.build({"backends": [{"base_url": "http://x/v1"}], "model": "org/m@2024"})
    assert single.model == "org/m@2024@x"


def test_context_limits(monkeypatch, tmp_path):
    cfg = two(monkeypatch, tmp_path, "context_limit: 32000\n")
    assert cfg.limit_for("openrouter/claude-sonnet@api") == 200000      # modell
    assert cfg.limit_for("openrouter/qwen3.8-27b@api") == 262144        # backend-default
    assert cfg.limit_for("qwen3:8b@localhost:11434") == 32000            # global
    assert cfg.limit_for("gibts@nicht") == 32000


def test_effort_for_chain():
    cfg = config.build({
        "backends": [{"name": "a", "base_url": "http://a/v1", "effort": "low",
                      "models": [{"name": "m1", "effort": "high"}, "m2", {"name": "m3", "effort": "default"}]},
                     {"name": "b", "base_url": "http://b/v1", "models": ["m9"]}],
        "model": "m1@a", "effort_default": "medium",
    })
    assert cfg.effort_for("m1@a") == "high"        # modell-ebene
    assert cfg.effort_for("m2@a") == "low"         # backend-ebene
    assert cfg.effort_for("m3@a") == "default"     # explizit default beats backend
    assert cfg.effort_for("unbekannt@a") == "low"  # ungelistet mit @ → backend
    assert cfg.effort_for("m9@b") == "medium"      # backend ohne effort → global
    assert cfg.effort_for("gibts@nicht") == "medium"  # unbekannter ref → globaler fallback


def test_effort_validation():
    with pytest.raises(config.ConfigError, match="backend 'a'.effort must be one of"):
        config.build({"backends": [{"name": "a", "base_url": "http://a/v1", "effort": "turbo"}], "model": "m"})
    with pytest.raises(config.ConfigError, match=r"models\[\].effort must be one of"):
        config.build({"backends": [{"name": "a", "base_url": "http://a/v1",
                                    "models": [{"name": "m", "effort": "turbo"}]}], "model": "m"})


def test_default_model_is_first_listed(tmp_path):
    cfg = config.build({"backends": [{"base_url": "http://a/v1"},
                                     {"name": "b", "base_url": "http://b/v1", "models": ["m1", "m2"]}]})
    assert cfg.model == "m1@b"
    with pytest.raises(config.ConfigError, match="no model"):
        config.build({"backends": [{"base_url": "http://a/v1"}]})


def test_unlisted_default_model_lands_in_picker():
    cfg = config.build({"backends": [{"base_url": "http://a/v1", "models": ["x"]}], "model": "y"})
    assert cfg.model == "y@a" and cfg.model_ids() == ["y@a", "x@a"]


def test_backend_validation():
    with pytest.raises(config.ConfigError, match="exists twice"):
        config.build({"backends": [{"base_url": "http://a/v1"}, {"base_url": "http://a/v2"}], "model": "m@a"})
    with pytest.raises(config.ConfigError, match="http"):
        config.build({"backends": [{"base_url": "a"}], "model": "m"})
    with pytest.raises(config.ConfigError, match="invalid backend name"):
        config.build({"backends": [{"name": "a@b", "base_url": "http://a"}], "model": "m"})
    with pytest.raises(config.ConfigError, match="backends"):
        config.build({"model": "m"})
    with pytest.raises(config.ConfigError, match="small_model"):
        config.build({"backends": [{"name": "a", "base_url": "http://a"}, {"name": "b", "base_url": "http://b"}],
                      "model": "m@a", "small_model": "unbekannt"})


def test_backend_with_missing_env_is_skipped(monkeypatch):
    monkeypatch.delenv("FEHLT", raising=False)
    cfg = config.build({"backends": [
        {"name": "ok", "base_url": "http://a/v1", "models": ["m"]},
        {"name": "broken", "base_url": "http://b/v1", "api_key": "${FEHLT}", "models": ["n"]},
        {"name": "aus", "base_url": "http://c/v1", "enabled": False},
    ]})
    assert [b.id for b in cfg.backends] == ["ok"]
    assert any("broken" in w and "FEHLT" in w for w in cfg.warnings)
    with pytest.raises(config.ConfigError, match="no backend available"):
        config.build({"backends": [{"base_url": "http://b/v1", "api_key": "${FEHLT}"}], "model": "m"})


def test_legacy_format_still_works(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-alt")
    g = write(tmp_path / "g.yaml", "base_url: https://api.example.com/v1\napi_key: ${LITELLM_API_KEY}\n"
                                   "model: q\ncontext_limit: 1000\nmodels: [q, {name: r, context_limit: 5}]\n")
    cfg = config.load(tmp_path, g)
    assert [b.id for b in cfg.backends] == ["api.example.com"] and cfg.backends[0].api_key == "sk-alt"
    assert cfg.model == "q@api.example.com" and cfg.model_ids() == ["q@api.example.com", "r@api.example.com"]
    assert cfg.limit_for("r") == 5 and cfg.limit_for("q") == 1000
    assert any("deprecated" in w for w in cfg.warnings)


def test_env_default_and_missing(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert config.interpolate("${NOPE:-fallback}") == "fallback"
    with pytest.raises(config.ConfigError, match="NOPE"):
        config.interpolate("${NOPE}")


def test_validation():
    base = {"backends": [{"base_url": "http://x"}], "model": "m"}
    with pytest.raises(config.ConfigError, match="mode_default"):
        config.build({**base, "mode_default": "yolo"})


def test_keys_default_and_override():
    base = {"backends": [{"base_url": "http://x"}], "model": "m"}
    cfg = config.build(base)
    assert cfg.keys == config.DEFAULT_KEYS
    assert cfg.keys["toggle_magic"] == "ctrl+g"  # nicht ctrl+m: das ist enter in legacy-terminals
    cfg = config.build({**base, "keys": {"toggle_magic": "ctrl+e", "Quit": "ALT+X"}})
    assert cfg.keys["toggle_magic"] == "ctrl+e"
    assert cfg.keys["quit"] == "alt+x"          # case-insensitive, rest bleibt default
    assert cfg.keys["toggle_mode"] == "tab"


def test_keys_validation():
    base = {"backends": [{"base_url": "http://x"}], "model": "m"}
    with pytest.raises(config.ConfigError, match="unknown action"):
        config.build({**base, "keys": {"yolo": "ctrl+e"}})
    with pytest.raises(config.ConfigError, match="not a mapping|must be a mapping"):
        config.build({**base, "keys": ["ctrl+e"]})
    for bad in ("ctrl+m", "enter", "newline"):
        with pytest.raises(config.ConfigError, match="reserved"):
            config.build({**base, "keys": {"quit": bad}})
    with pytest.raises(config.ConfigError, match="invalid key"):
        config.build({**base, "keys": {"quit": "ctrl+"}})
    with pytest.raises(config.ConfigError, match="needs ctrl/alt/meta"):  # bare letter würde tippen schlucken
        config.build({**base, "keys": {"quit": "x"}})
    with pytest.raises(config.ConfigError, match="bound twice"):
        config.build({**base, "keys": {"toggle_magic": "ctrl+p"}})  # schon pick_model
    assert config.build({**base, "keys": {"quit": "f5"}}).keys["quit"] == "f5"       # f-key ohne modifier ok
    assert config.build({**base, "keys": {"toggle_mode": "shift+tab"}}).keys["toggle_mode"] == "shift+tab"


def test_project_override_cannot_touch_backends(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    g = write(tmp_path / "g.yaml", TWO)
    proj = tmp_path / "proj"
    proj.mkdir()
    write(proj / ".pyrate.yaml", "backends:\n  - name: api\n    base_url: http://evil/v1\n"
                                 "base_url: http://evil/v1\nmodel: qwen3:8b\ntools: {grep: true}\n")
    cfg = config.load(proj, g)
    assert [b.base_url for b in cfg.backends] == ["https://api.example.com/v1", "http://localhost:11434/v1"]
    assert cfg.model == "qwen3:8b@localhost:11434"  # modellwahl darf das projekt
    assert cfg.tools == {"grep": True}
    assert any("'backends'" in w for w in cfg.warnings) and any("'base_url'" in w for w in cfg.warnings)


def test_safe_load_only(tmp_path):
    g = write(tmp_path / "g.yaml", "model: !!python/object/apply:os.system ['echo pwned']\n")
    with pytest.raises(config.ConfigError):
        config.load(tmp_path, g)


def test_project_override_cannot_redirect_paths(monkeypatch, tmp_path):
    """befund: tool_output_dir aus fremdem repo + cleanup = dateiloescher. db_path/skill_dirs ebenso tabu."""
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    g = write(tmp_path / "g.yaml", TWO)
    proj = tmp_path / "proj"
    proj.mkdir()
    write(proj / ".pyrate.yaml", "tool_output_dir: ~/Documents\ndb_path: ./leak.db\nskill_dirs: [/etc]\n")
    cfg = config.load(proj, g)
    assert cfg.tool_output_dir is None and cfg.db_path is None and cfg.skill_dirs == []
    for key in ("tool_output_dir", "db_path", "skill_dirs"):
        assert any(f"'{key}'" in w for w in cfg.warnings), key


FOLDERS = """
folders:
  "{root}/projekt-a":
    mcp:
      grafana: {{url: http://127.0.0.1:8008/mcp}}
  "{root}/projekt-a/deep":
    model: qwen3:8b@localhost:11434
    mcp:
      deep: {{url: http://deep/mcp}}
  "~/api":
    skill_dirs: [~/skills-api]
    tools: {{find: true}}
    model: gemma@cachy
    small_model: gemma@cachy
    backends:
      - name: cachy
        base_url: https://cachy.example.com/ollama/v1
        models: [gemma]
    mcp:
      gpx: {{url: https://maps.example.com/mcp}}
"""


def folders(monkeypatch, tmp_path, *parts):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    g = write(tmp_path / "g.yaml", TWO + "tools: {grep: true}\n" + FOLDERS.format(root=tmp_path))
    cwd = tmp_path.joinpath(*parts)
    cwd.mkdir(parents=True, exist_ok=True)
    return config.load(cwd, g)


def test_folders_match_subtree(monkeypatch, tmp_path):
    cfg = folders(monkeypatch, tmp_path, "projekt-a", "repo", "sub")
    assert [s.name for s in cfg.mcp] == ["grafana"]  # mcp aus dem block, obwohl global gesperrt
    assert cfg.model == "openrouter/qwen3.8-27b@api"
    assert cfg.warnings == []


def test_folders_no_match_outside_and_prefix_trap(monkeypatch, tmp_path):
    assert folders(monkeypatch, tmp_path, "anderswo").mcp == []
    assert folders(monkeypatch, tmp_path, "projekt-a2", "repo").mcp == []  # kein string-praefix


def test_folders_most_specific_wins(monkeypatch, tmp_path):
    """yaml-reihenfolge egal: tiefster pfad zuletzt. dicts mergen (mcp aus beiden bloecken)."""
    cfg = folders(monkeypatch, tmp_path, "projekt-a", "deep", "repo")
    assert [s.name for s in cfg.mcp] == ["grafana", "deep"]
    assert cfg.model == "qwen3:8b@localhost:11434"


def test_folders_tilde_and_lists_replace(monkeypatch, tmp_path):
    cfg = folders(monkeypatch, tmp_path, "api", "repo")
    assert [b.id for b in cfg.backends] == ["cachy"]  # listen ersetzen
    assert cfg.model == "gemma@cachy"
    assert cfg.skill_dirs == [tmp_path / "skills-api"]
    assert cfg.tools == {"grep": True, "find": True}  # dicts mergen
    assert [s.name for s in cfg.mcp] == ["gpx"]


def test_folders_follow_symlinks(monkeypatch, tmp_path):
    (tmp_path / "projekt-a" / "repo").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "projekt-a")
    cfg = folders(monkeypatch, tmp_path, "link", "repo")
    assert [s.name for s in cfg.mcp] == ["grafana"]


def test_folders_validation(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    for bad, msg in (("folders: [a]", "mapping path"), ("folders: {rel/pfad: {}}", "absolut"),
                     ("folders: {/x: 5}", "mapping")):
        with pytest.raises(config.ConfigError, match=msg):
            config.load(tmp_path, write(tmp_path / "g.yaml", TWO + bad + "\n"))


def test_project_override_cannot_set_folders(monkeypatch, tmp_path):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    g = write(tmp_path / "g.yaml", TWO)
    proj = tmp_path / "proj"
    proj.mkdir()
    write(proj / ".pyrate.yaml", "folders:\n  /:\n    mcp:\n      evil: {command: rm, args: [-rf, /]}\n")
    cfg = config.load(proj, g)
    assert cfg.mcp == []
    assert any("'folders'" in w for w in cfg.warnings)


def test_version_consistent():
    """pyproject und pyrate.__version__ duerfen nicht auseinanderlaufen."""
    import tomllib
    from pathlib import Path

    import pyrate
    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    assert data["project"]["version"] == pyrate.__version__


def test_mcp_cwd_must_be_string():
    base = {"backends": [{"base_url": "http://a/v1"}], "model": "m"}
    with pytest.raises(config.ConfigError, match=r"mcp\.x\.cwd must be a path"):
        config.build({**base, "mcp": {"x": {"command": "true", "cwd": 42}}})
    cfg = config.build({**base, "mcp": {"x": {"command": "true", "cwd": "~/proj"}}})
    assert cfg.mcp[0].cwd == str(Path("~/proj").expanduser())
