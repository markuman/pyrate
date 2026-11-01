"""compaction: summarize the old history via llm, keep the last ~20k tokens verbatim.

flow (agent.compact):
  1. find_cut()    – where to cut? never between a tool call and its result
  2. serialize()   – old part as text, tool outputs truncated, reasoning dropped
  3. summarize()   – llm call without tools, previous summary is continued
  4. file_lists()  – <read-files>/<modified-files> deterministically from the tool calls
  5. render()      – ready user-message content, stored as part "compaction" in the db
"""

import json
from pathlib import Path

from . import llm, tools

RESERVE = 16384           # auto-compaction when context > limit - RESERVE
KEEP = 20000              # this many tokens stay verbatim at the end
TOOL_RESULT_CHARS = 2000  # tool outputs in the summary input
TOOL_ARGS_CHARS = 500
SUMMARY_MAX_TOKENS = 16384  # generous: reasoning models think first

OVERFLOW_PATTERNS = (
    "context_length_exceeded", "context length", "context window", "contextwindowexceeded",
    "maximum context", "too many tokens", "prompt is too long", "input is too long",
    "reduce the length", "exceeds the context",
)

SYSTEM = ("You summarize a conversation between a user and a coding agent, so that the agent "
          "can continue the work with a fresh context window. You only write the summary.")

PROMPT = """{previous}<conversation>
{conversation}
</conversation>

{task}

Use exactly this structure:

## Goal
## Constraints & Preferences
## Progress
### Done
### In Progress
### Blocked
## Key Decisions
## Next Steps
## Critical Context

Rules:
- keep exact file paths, function names, error messages and commands
- dense bullet points, no prose
- same language as the user"""

TASK_NEW = "Summarize the conversation above."
TASK_UPDATE = ("Update the previous summary with the conversation above: keep what is still true, "
               "merge the new progress, drop what became obsolete. Output the complete updated summary.")

WRAPPER = """<conversation_summary>
{summary}
{files}</conversation_summary>
Earlier parts of this conversation were compacted into the summary above. Continue the work from there; \
re-read files before you edit them."""


class CompactError(Exception):
    pass


IMAGE_TOKENS = 1500  # flat rate per image; base64 via chars/4 would be ~5x too high


def _deimage(obj):
    """replace image parts with a placeholder so estimate() doesn't count the base64 blob."""
    if isinstance(obj, dict):
        if obj.get("type") in ("image", "image_url") and (obj.get("data") or obj.get("image_url")):
            return {"type": obj["type"], "data": "x" * (IMAGE_TOKENS * 4)}
        return {k: _deimage(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deimage(v) for v in obj]
    return obj


def estimate(obj):
    """rough token estimate: chars/4. images flat (IMAGE_TOKENS)."""
    return len(json.dumps(_deimage(obj), ensure_ascii=False)) // 4


def budgets(limit):
    """(reserve, keep) – proportionally smaller for small context windows, else the budget eats everything."""
    return min(RESERVE, limit // 4), min(KEEP, limit // 4)


def is_overflow(err):
    text = f"{err} {getattr(err, 'body', '') or ''}".lower()
    return any(p in text for p in OVERFLOW_PATTERNS)


# ---------------------------------------------------------------- cut

def find_cut(active, keep):
    """index of the first entry kept verbatim in `active` (db raw format).

    prefers a user boundary. if not even the current turn fits the budget (long
    agent run after a prompt), then before an assistant message – that one brings its
    tool results along, so a call/result pair is never torn apart.
    len(active) = keep nothing.
    """
    tail = 0
    user_cut = assistant_cut = None
    for i in range(len(active) - 1, -1, -1):
        tail += estimate(active[i]["parts"])
        if tail > keep:
            break
        if active[i]["role"] == "user":
            user_cut = i
        elif active[i]["role"] == "assistant":
            assistant_cut = i
    if user_cut is not None:
        return user_cut
    if assistant_cut is not None:
        return assistant_cut
    return len(active)


# ---------------------------------------------------------------- serialization

def serialize(raw):
    out = []
    for m in raw:
        who = "User" if m["role"] == "user" else "Assistant"
        for p in m["parts"]:
            d, t = p["data"], p["type"]
            if t == "text" and d.get("text", "").strip():
                out.append(f"[{who}]: {d['text'].strip()}")
            elif t == "image":
                out.append(f"[{who}]: [image attached: {d.get('path', '?')}]")
            elif t == "tool_call":
                args = tools.truncate(d.get("arguments") or "{}", TOOL_ARGS_CHARS)
                out.append(f"[Assistant tool call]: {d['name']} {args}")
            elif t == "tool_result":
                status = "" if d.get("ok", True) else " (error)"
                out.append(f"[Tool result{status}]: {tools.truncate(d.get('content', ''), TOOL_RESULT_CHARS)}")
            # reasoning and everything else is dropped
    return "\n\n".join(out)


# ---------------------------------------------------------------- file lists

def _norm(cwd, path):
    try:
        p = tools._resolve(tools.ToolContext(cwd=cwd), path)
    except (tools.ToolError, OSError, ValueError):
        return None
    return p.relative_to(cwd).as_posix() if p.is_relative_to(cwd) else str(p)


def file_lists(raw, cwd, prev_read=(), prev_modified=()):
    """read and modified files from successful tool calls, cumulative with the previous summary.
    what was modified is not listed under read again."""
    cwd = Path(cwd).resolve()
    ok = {}
    for m in raw:
        for p in m["parts"]:
            if p["type"] == "tool_result":
                ok[p["data"].get("tool_call_id")] = p["data"].get("ok", True)
    read = dict.fromkeys(prev_read)
    modified = dict.fromkeys(prev_modified)
    for m in raw:
        for p in m["parts"]:
            d = p["data"]
            if p["type"] != "tool_call" or not ok.get(d.get("id")):
                continue
            try:
                args = json.loads(d.get("arguments") or "{}")
            except ValueError:
                continue
            path = _norm(cwd, args.get("path")) if isinstance(args, dict) and args.get("path") else None
            if not path:
                continue
            if d["name"] == "read":
                read[path] = None
            elif d["name"] in ("write", "edit"):
                modified[path] = None
    return [p for p in read if p not in modified], list(modified)


def render(summary, read, modified):
    files = ""
    if read:
        files += "\n<read-files>\n" + "\n".join(read) + "\n</read-files>\n"
    if modified:
        files += "\n<modified-files>\n" + "\n".join(modified) + "\n</modified-files>\n"
    return WRAPPER.format(summary=summary.strip(), files=files)


# ---------------------------------------------------------------- llm call

def summarize(cfg, model, raw, previous=None, emit=None, cancel=None, session=None):
    """returns the summary text. finish_reason=length → CompactError (summary discarded)."""
    emit = emit or (lambda ev: None)
    prev = f"<previous_summary>\n{previous.strip()}\n</previous_summary>\n\n" if previous else ""
    prompt = PROMPT.format(previous=prev, conversation=serialize(raw),
                           task=TASK_UPDATE if previous else TASK_NEW)
    text, finish = [], None
    for ev in llm.stream_chat(cfg, model, [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": prompt}],
                              max_tokens=SUMMARY_MAX_TOKENS, cancel=cancel, session=session):
        if ev["type"] == "done":
            finish = ev["finish_reason"]
        elif ev["type"] == "warning":
            emit(ev)
        else:
            if ev["type"] == "text":
                text.append(ev["text"])
            emit({"type": "compact_delta"})  # heartbeat for the activity display
    summary = "".join(text).strip()
    if finish == "length":
        raise CompactError("summary was cut off (finish_reason=length) and discarded")
    if not summary:
        raise CompactError("model returned an empty summary")
    return summary
