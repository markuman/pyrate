import base64

from pyrate import compact, images
from pyrate.agent import Agent, merge_user_turns
from pyrate.db import DB

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def test_find_and_load(tmp_path):
    img = tmp_path / "bild.png"
    img.write_bytes(PNG_BYTES)
    (tmp_path / "notes.txt").write_text("x")
    found = images.find(f"schau {img} und {tmp_path / 'notes.txt'} und {tmp_path / 'weg.png'}")
    assert found == [img.resolve()]
    imgs, skipped = images.load(found)
    assert skipped == []
    assert imgs[0]["mime"] == "image/png"
    assert imgs[0]["path"] == str(img.resolve())
    assert base64.b64decode(imgs[0]["data"]) == PNG_BYTES


def test_find_quoted_and_dedup(tmp_path):
    img = tmp_path / "a b.png"  # leerzeichen im namen
    img.write_bytes(PNG_BYTES)
    found = images.find(f"'{img}' {img}")  # quoted + unquoted, gleicher pfad
    assert found == [img.resolve()]


def test_load_limits(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"0" * (images.MAX_BYTES + 1))
    imgs, skipped = images.load([big])
    assert imgs == []
    assert "too large" in skipped[0][1]
    viele = []
    for i in range(images.MAX_IMAGES + 2):
        p = tmp_path / f"{i}.png"
        p.write_bytes(b"x")
        viele.append(p)
    imgs, skipped = images.load(viele)
    assert len(imgs) == images.MAX_IMAGES
    assert len(skipped) == 2
    assert "too many" in skipped[0][1]


def test_scan_leere_eingabe():
    assert images.scan("nichts besonderes hier") == ([], [])


def test_image_parts():
    parts = images.image_parts([{"path": "/x.png", "mime": "image/png", "data": "QUJD"}])
    assert parts == [{"type": "text", "text": images.label("/x.png")},
                     {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]
    assert "/x.png" in parts[0]["text"]  # pfad im label: modell verbindet prompt-pfad und bild


def test_merge_user_turns_multipart():
    a = {"role": "user", "content": [{"type": "text", "text": "eins"},
                                    {"type": "text", "text": images.label("/a.png")},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}
    b = {"role": "user", "content": "zwei"}
    out = merge_user_turns([a, b])
    assert len(out) == 1
    c = out[0]["content"]
    # reihenfolge bleibt: label direkt vor seinem bild
    assert [p.get("text") or p["type"] for p in c] == ["eins", images.label("/a.png"), "image_url", "zwei"]
    # str + str bleibt str (abwaertskompatibel)
    out = merge_user_turns([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert out[0]["content"] == "a\n\nb"


def test_db_roundtrip_image(db):
    sid = db.create_session("/p", "m")
    db.add_message(sid, "user", "build", [("text", {"text": "was ist das?"}),
                                          ("image", {"path": "/x.png", "mime": "image/png", "data": "QUJD"})])
    msgs = db.load_messages(sid)
    assert msgs[0]["content"][0] == {"type": "text", "text": "was ist das?"}
    assert msgs[0]["content"][1] == {"type": "text", "text": images.label("/x.png")}
    assert msgs[0]["content"][2]["image_url"]["url"] == "data:image/png;base64,QUJD"
    assert db.search("QUJD") == []  # base64 landet nicht in der fts


def test_estimate_bild_pauschal():
    """base64 darf nicht mit chars/4 gerechnet werden (~5x zu hoch)."""
    part = {"type": "image", "data": {"path": "/x.png", "mime": "image/png", "data": "A" * 40000}}
    with_img = compact.estimate([part])
    assert 1000 < with_img < 2000  # ≈ IMAGE_TOKENS, statt 10000
    plain = compact.estimate([{"type": "text", "data": {"text": "A" * 40000}}])
    assert plain > 9000  # text bleibt chars/4 (plus json-overhead)


def test_serialize_image():
    raw = [{"id": 1, "role": "user", "mode": "build",
            "parts": [{"type": "image", "data": {"path": "/x.png", "mime": "image/png", "data": "QUJD"}}]}]
    assert "[User]: [image attached: /x.png]" in compact.serialize(raw)


def test_agent_run_sends_multipart(backend, make_cfg, tmp_path):
    backend.script = [{"text": "ein bild"}]
    agent = Agent(make_cfg(), DB(":memory:"), tmp_path)
    (tmp_path / "x.png").write_bytes(PNG_BYTES)
    imgs, skipped = images.load([tmp_path / "x.png"])
    assert skipped == []
    agent.run("was siehst du?", images=imgs)
    user_msg = backend.requests[0]["body"]["messages"][-1]
    assert user_msg["content"][0] == {"type": "text", "text": "was siehst du?"}
    assert str(tmp_path / "x.png") in user_msg["content"][1]["text"]
    assert user_msg["content"][2]["type"] == "image_url"
    assert user_msg["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
    # beim reload aus der db kommt derselbe multipart-content back
    agent2 = Agent(make_cfg(), agent.db, tmp_path, session_id=agent.session_id)
    reloaded = next(m for m in agent2.messages if m["role"] == "user")
    assert reloaded["content"] == user_msg["content"]


def test_agent_run_without_images_stays_string(backend, make_cfg, tmp_path):
    backend.script = [{"text": "ok"}]
    agent = Agent(make_cfg(), DB(":memory:"), tmp_path)
    agent.run("nur text")
    user_msg = backend.requests[0]["body"]["messages"][-1]
    assert user_msg["content"] == "nur text"
