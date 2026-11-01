import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fake_backend import FakeBackend  # noqa: E402

from pyrate import config, sandbox  # noqa: E402
from pyrate.db import DB  # noqa: E402

# mit landlock laufen alle agent-tests echt in der sandbox, ohne (macos, alte kernel) eben ohne
HAS_LANDLOCK = sandbox.abi() > 0


@pytest.fixture(autouse=True)
def _isolated_data_home(tmp_path, tmp_path_factory, monkeypatch):
    """volltexte (tool-output) landen nie im echten ~/.local/share, und die echte
    ~/.config/pyrate (AGENTS.md, skills, mcp) sickert nicht in tests."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    # subagent-slots nicht im echten runtime-dir; und tests, die selbst in einem pyrate-bash
    # laufen (dog-fooding), sollen sich nicht for subagents halten
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path_factory.mktemp("xdg-run")))  # nicht in tmp_path, das listen tests
    for var in ("PYRATE_SESSION", "PYRATE_SUBAGENTS", "PYRATE_MODEL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def backend():
    fb = FakeBackend().start()
    yield fb
    fb.stop()


@pytest.fixture
def make_cfg(backend, tmp_path):
    def make(**extra):
        be = {"base_url": extra.pop("base_url", backend.url), "api_key": extra.pop("api_key", "sk-test"),
              "models": extra.pop("models", ["fake-model"]), "plugin": extra.pop("plugin", None)}
        if (eff := extra.pop("backend_effort", None)):
            be["effort"] = eff
        data = {"backends": [be], "model": "fake-model", "context_limit": 100000,
                "db_path": str(tmp_path / "t.db"), "sandbox": HAS_LANDLOCK, **extra}
        return config.build(data)
    return make


@pytest.fixture
def write_plugin(tmp_path):
    """plugin-quelltext nach tmp_path/<name>.py schreiben, gibt den pfad back."""
    def write(source, name="plug"):
        path = tmp_path / f"{name}.py"
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return str(path)
    return write


@pytest.fixture
def db():
    d = DB(":memory:")
    yield d
    d.close()
