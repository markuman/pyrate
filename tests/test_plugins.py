"""plugin-schnittstelle: laden per `plugin:` am backend, request-hook, fehlerfaelle."""

import pytest

from pyrate import config, llm, plugins

HI = [{"role": "user", "content": "hi"}]


def test_request_hook(backend, make_cfg, write_plugin, monkeypatch):
    """headers und payload change, einmal pro request – der retry schickt dasselbe nochmal."""
    monkeypatch.setattr(llm, "RETRY_BASE", 0.01)
    path = write_plugin("""
        CALLS = []

        def request(backend, headers, payload, session):
            CALLS.append(session)
            headers["X-Test"] = f"{backend.id}|{headers['Authorization']}"
            payload["temperature"] = 0.5
    """)
    cfg = make_cfg(plugin=path)
    backend.script = [{"status": 503, "text": "busy"}, {"text": "ok"}]
    list(llm.stream_chat(cfg, "fake-model", HI, session="s1"))
    assert plugins.load(path).CALLS == ["s1"]
    for r in backend.requests:
        assert r["headers"]["X-Test"] == f"{cfg.backends[0].id}|Bearer sk-test"
        assert r["body"]["temperature"] == 0.5


def test_hook_error_aborts_request(backend, make_cfg, write_plugin):
    path = write_plugin("""
        def request(backend, headers, payload, session):
            raise RuntimeError("kein key")
    """)
    with pytest.raises(llm.LLMError, match="plugin plug: kein key"):
        list(llm.stream_chat(make_cfg(plugin=path), "fake-model", HI))
    assert not backend.requests


@pytest.mark.parametrize("plugin", ["fehlt.py", "/gibts/nicht.py", "SYNTAX", "OHNE_REQUEST"])
def test_broken_plugin_skips_backend(plugin, write_plugin):
    if plugin == "SYNTAX":
        plugin = write_plugin("def broken(:")
    elif plugin == "OHNE_REQUEST":
        plugin = write_plugin("def headers(backend, headers, session): pass")  # alte schnittstelle
    cfg = config.build({"backends": [{"name": "a", "base_url": "http://a/v1", "plugin": plugin},
                                     {"name": "b", "base_url": "http://b/v1"}], "model": "m@b"})
    assert [b.id for b in cfg.backends] == ["b"]
    assert any("backend 'a' skipped: plugin" in w for w in cfg.warnings)


def test_request_return_overrides_url(backend, make_cfg, write_plugin):
    """str-rueckgabewert von request() ersetzt die komplette url (z.b. /v1/messages)."""
    path = write_plugin("""
        def request(backend, headers, payload, session):
            return backend.base_url + "/messages?beta=true"
    """)
    backend.script = [{"text": "ok"}]
    list(llm.stream_chat(make_cfg(plugin=path), "fake-model", HI))
    assert backend.requests[0]["path"] == "/v1/messages?beta=true"


def test_default_url_unchanged(backend, make_cfg, write_plugin):
    path = write_plugin("def request(backend, headers, payload, session): pass")
    backend.script = [{"text": "ok"}]
    list(llm.stream_chat(make_cfg(plugin=path), "fake-model", HI))
    assert backend.requests[0]["path"] == "/v1/chat/completions"


def test_response_hook(backend, make_cfg, write_plugin):
    """optionale response-funktion bildet rohe chunks aufs openai-format ab, None verwirft."""
    path = write_plugin("""
        def request(backend, headers, payload, session):
            pass

        def response(chunk):
            # tut so, als kaeme ein fremdes format: text landet im reasoning-feld
            for c in chunk.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content"):
                    return {"choices": [{"delta": {"reasoning_content": delta["content"]}}]}
                if c.get("finish_reason"):
                    return {"choices": [{"delta": {}, "finish_reason": c["finish_reason"]}]}
            return None  # usage-chunk etc. verwerfen
    """)
    backend.script = [{"text": "hallo welt"}]
    events = list(llm.stream_chat(make_cfg(plugin=path), "fake-model", HI))
    reasoning = "".join(e["text"] for e in events if e["type"] == "reasoning")
    assert reasoning == "hallo welt"  # fake streamt in 5er-chunks, daher mehrere events
    assert not any(e["type"] == "text" for e in events)
    done = next(e for e in events if e["type"] == "done")
    assert done["message"]["reasoning_content"] == "hallo welt"
    assert done["message"]["content"] is None
    assert done["usage"] is None          # vom hook verworfen
    assert done["finish_reason"] == "stop"


def test_response_hook_error_aborts_stream(backend, make_cfg, write_plugin):
    path = write_plugin("""
        def request(backend, headers, payload, session):
            pass

        def response(chunk):
            raise RuntimeError("broken")
    """)
    backend.script = [{"text": "ok"}]
    with pytest.raises(llm.LLMError, match="plugin plug response: broken"):
        list(llm.stream_chat(make_cfg(plugin=path), "fake-model", HI))


def test_loaded_once(write_plugin):
    path = write_plugin("def request(*a): pass")
    assert plugins.load(path) is plugins.load(path)
