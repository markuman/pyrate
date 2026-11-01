# pyrate – implementation battle plan 🏴‍☠️

**tl;dr:** python coding harness with tui. stdlib where possible. external deps only for the tui (`textual`) and yaml (`pyyaml`). talks to exactly one host.

## guiding principles

- **kiss:** one loop and four tools. everything else has to earn its place
- **deps:** `textual`, `pyyaml`, nothing else. http via `urllib.request` (read sse line by line, in the textual worker thread). `httpx` only if urllib really annoys
- **privacy:** zero calls except to the configured backend. no model fetches, no telemetry, no fallback providers
- **headless first:** the agent loop runs without tui (`pyrate -p "prompt"`). the tui is just a client

## layout

```
pyrate/
  __main__.py   # cli entry: tui or -p (print mode)
  config.py     # load yaml, interpolate ${ENV}, validate dataclasses
  llm.py        # openai-compatible chat call, sse streaming, usage
  tools.py      # read, write, edit, bash (+ grep/find read-only)
  agent.py      # loop: call → run tool_calls → until none are left
  db.py         # sqlite3: sessions, messages, parts, fts5
  tui.py        # textual app
```

## config (yaml)

- paths: `~/.config/pyrate/config.yaml`, on top `./.pyrate.yaml` as project override
- `yaml.safe_load`, **never** `yaml.load`
- api keys as `${LITELLM_API_KEY}`, never in plaintext
- `backends:` list of `base_url`, `api_key`, `models: []` (optional `name`, `context_limit`, `effort`) –
  multiple backends in parallel, models are called `model@backend`
- reasoning effort per model: `effort` in `models[]` beats `effort` in the backend, which beats `effort_default` globally
- fields: `model` (default), `small_model` (title, falls back to the main model), `context_limit` (optional), `mode_default`
- all models of all backends are switchable anytime without restart (picker shows `model@backend`)

## sessions (sqlite3, stdlib)

- `~/.local/share/pyrate/pyrate.db`, `PRAGMA journal_mode=WAL`
- tables:
  - `sessions` (id, title, cwd, model, created_at, updated_at)
  - `messages` (id, session_id, role, mode, created_at)
  - `parts` (id, message_id, type: text/tool_call/tool_result, data json)
- `fts5` for searching old sessions
- migrations via `PRAGMA user_version`, no alembic

## tools

- `read`, `write`, `edit`, `bash`. `grep`/`find` are read-only and off by default
  - must also include `ask_user`
- **harden edit first:** search/replace with a clear error on 0 or more than one match, whitespace tolerance, diff as return value
- `bash` via `subprocess.run` with timeout and output limit
- load `AGENTS.md` from the cwd into the system prompt. keep the system prompt tiny

## tui (textual)

- mouse-wheel scrolling, resize and clicks come for free
- streaming via `Markdown.get_stream()`
- **tab:** switches plan/build. needs `Binding("tab", ..., priority=True)`
  - plan = read-only tools only, own prompt addition
  - build = everything
  - make the mode visible via the border color and in the footer
- sidebar: context display and tool status, visible **before** the first prompt
- ctrl+p switches the model (ctrl+m is enter in the terminal; command palette is off for that)
- ctrl+t switches the reasoning effort (default, low, medium, high, xhigh, max)
- the effort default comes per model from the config (model → backend → global); model switch (ctrl+p) resets it, ctrl+t overrides until the next switch
- enter inserts a newline, ctrl+j submits the prompt (ctrl+j arrives as its own control byte in every terminal; shift/ctrl/alt+enter are indistinguishable from enter in legacy terminals without the kitty keyboard protocol)
- prompt input has a fixed height (15 % of the screen) and scrolls internally
- while the answer is pending: animation + time since the last sign of life, so you can tell dead from slow
- drag & drop of files from the file manager results in absolute paths to the files in the prompt

## context window

- send `stream_options: {"include_usage": true}`, the real values are in the last chunk
- before the first request a rough estimate (chars/4) with `~`, after that the real value

## order

1. **config + llm.py:** a streamed chat against litellm in the terminal
2. **agent.py + tools.py:** print mode reads a file, edits it and runs `pytest`
3. **db.py:** persist sessions and continue them (`-c`)
4. **tui.py:** chat, streaming, mouse scroll
5. **plan/build** via tab
6. **context display** + title via `small_model`
7. **egress test in ci:** `strace -e trace=connect -f` may only show the backend host
   (with multiple `backends`: only the configured hosts)

## later (not v1)

- ~~mcp~~ → implemented (`mcp.py`); every configured mcp server is deliberately extra egress
- ~~compaction on full context~~ → implemented (`compact.py`)
- confirmation dialog for bash in plan mode
