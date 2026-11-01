"""images from the prompt to the model: multipart content with base64 data-url.

the user types or drops a path – pyrate reads the file itself (user context, not the
model, hence no sandbox check) and attaches it to the user message. the model gets
text + image in one message."""

import base64
import shlex
from pathlib import Path

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp"}
MAX_IMAGES = 5
MAX_BYTES = 10 * 1024 * 1024  # per image; many apis cap the whole request at ~20 MB


def find(text):
    """image paths in the text: token with a known extension that exists as a file.
    shlex, so quoted paths with spaces work."""
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    seen, out = set(), []
    for tok in tokens:
        p = Path(tok)
        if p.suffix.lower() not in MIME:
            continue
        try:
            if not p.is_file():
                continue
            p = p.resolve()
        except OSError:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load(paths):
    """→ (images, skipped). images: [{path, mime, data(base64)}]; skipped: [(path, reason)]."""
    images, skipped = [], []
    for p in paths[:MAX_IMAGES]:
        try:
            raw = p.read_bytes()
        except OSError as e:
            skipped.append((str(p), f"not readable ({e})"))
            continue
        if len(raw) > MAX_BYTES:
            skipped.append((str(p), f"too large ({len(raw) // 1048576} MB, max {MAX_BYTES // 1048576} MB)"))
            continue
        images.append({"path": str(p), "mime": MIME[p.suffix.lower()],
                       "data": base64.b64encode(raw).decode()})
    skipped += [(str(p), f"too many images (max {MAX_IMAGES})") for p in paths[MAX_IMAGES:]]
    return images, skipped


def scan(text):
    """find + load combined, for the prompt window / headless."""
    return load(find(text))


def label(path):
    return f"[image attached: {path} – attached directly to this message, no tool needed]"


def image_parts(images):
    """image dicts → openai content parts: per image a text label with the path, then the data-url.
    without the label some models (mistral) don't connect path in the prompt with the image and
    try to read the file via a tool."""
    out = []
    for im in images:
        out.append({"type": "text", "text": label(im["path"])})
        out.append({"type": "image_url", "image_url": {"url": f"data:{im['mime']};base64,{im['data']}"}})
    return out
