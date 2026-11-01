"""plugins: one python file, attached to a backend via `plugin: ~/path/file.py`.

global config only (backends are locked in the project override), otherwise a foreign
repo could load arbitrary code.

interface – one required function, one optional:

    def request(backend, headers, payload, session):
        once before every request to this backend (retries send the same thing again).
        - headers: dict, modify directly (e.g. Authorization)
        - payload: the json body (model, messages, tools, reasoning_effort, …). setting
          or deleting keys is ok. but messages belong to the agent: don't modify in place,
          assign a new list instead
        - backend: config.Backend (id, base_url, api_key, …), session: session id or None
        - return value: a str replaces the whole request url
          (default: f"{backend.base_url}/chat/completions") – e.g. for providers with a
          different endpoint like anthropic's /v1/messages
        an exception aborts the request, its text lands as an error message for the user.

    def response(chunk):  # optional
        called for every sse chunk of the answer after json.loads. returns the chunk
        in openai format (pyrate reads choices[].delta.content / .tool_calls /
        .reasoning_content, finish_reason, usage) or None to discard it. this also maps
        the answer of a foreign protocol onto the openai format.
        an exception aborts the stream.
"""

import importlib.util
from pathlib import Path

_loaded = {}  # path → module: backends with the same file share one module (and its cache)


class PluginError(Exception):
    pass


def load(path):
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        raise PluginError(f"plugin {path!r} must be absolute or start with ~")
    key = str(p.resolve())
    if key not in _loaded:
        spec = importlib.util.spec_from_file_location(p.stem, key)
        if spec is None:
            raise PluginError(f"plugin {path}: not a .py file")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:  # file missing, syntax error, broken import, …
            raise PluginError(f"plugin {path}: {type(e).__name__}: {e}") from e
        if not callable(getattr(mod, "request", None)):
            raise PluginError(f"plugin {path}: function request(backend, headers, payload, session) missing")
        _loaded[key] = mod
    return _loaded[key]
