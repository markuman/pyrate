"""what additionally goes into the system prompt: instruction files (AGENTS.md) and skills.

both exist globally (next to the config, ~/.config/pyrate/) and at project level.
project level = all directories from the git root down to the cwd; without git just the cwd
(otherwise e.g. a ~/AGENTS.md would leak into every project).
the most specific one comes last and wins.
"""

import re
from dataclasses import dataclass
from html import escape
from pathlib import Path

import yaml

INSTRUCTION_FILES = ("AGENTS.md", "CLAUDE.md")  # per directory the first match counts
PROJECT_SKILL_DIR = Path(".pyrate") / "skills"
MAX_INSTRUCTION_BYTES = 50 * 1024
MAX_DESCRIPTION = 1024

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)


@dataclass
class Instruction:
    path: Path
    text: str
    scope: str  # "global" | "project"


@dataclass
class Skill:
    name: str
    description: str
    path: Path   # the SKILL.md
    scope: str   # "extra" | "global" | "project"


def git_root(cwd):
    cwd = Path(cwd).resolve()
    for d in (cwd, *cwd.parents):
        if (d / ".git").exists():
            return d
    return None


def project_dirs(cwd):
    """[git-root, …, cwd] or [cwd] without git."""
    cwd = Path(cwd).resolve()
    root = git_root(cwd)
    if root is None:
        return [cwd]
    dirs = [cwd]
    for d in cwd.parents:
        if not d.is_relative_to(root):
            break
        dirs.append(d)
    return dirs[::-1]


# ---------------------------------------------------------------- instructions

def _read_instruction(path, scope, warnings):
    try:
        raw = path.read_bytes()
    except OSError as e:
        warnings.append(f"{path}: not readable ({e})")
        return None
    if len(raw) > MAX_INSTRUCTION_BYTES:
        warnings.append(f"{path}: larger than {MAX_INSTRUCTION_BYTES // 1024}kb, truncated")
        raw = raw[:MAX_INSTRUCTION_BYTES]
    text = raw.decode("utf-8", "replace").strip()
    return Instruction(path, text, scope) if text else None


def load_instructions(cwd, global_dir=None, warnings=None):
    """global first, then git-root → cwd. per directory the first file from INSTRUCTION_FILES."""
    warnings = warnings if warnings is not None else []
    out = []
    seen = set()
    candidates = [(Path(global_dir), "global")] if global_dir else []
    candidates += [(d, "project") for d in project_dirs(cwd)]
    for d, scope in candidates:
        for name in INSTRUCTION_FILES:
            p = d / name
            if p.is_file():
                real = p.resolve()
                if real not in seen:  # e.g. CLAUDE.md as a symlink to AGENTS.md
                    seen.add(real)
                    ins = _read_instruction(p, scope, warnings)
                    if ins:
                        out.append(ins)
                break
    return out


def render_instructions(instructions):
    return "\n\n".join(
        f'<project_instructions path="{escape(str(i.path))}">\n{i.text}\n</project_instructions>'
        for i in instructions
    )


# ---------------------------------------------------------------- skills

def parse_skill(path, scope):
    """SKILL.md with yaml frontmatter (name, description). returns Skill or error text."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"not readable ({e})"
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return "no yaml frontmatter (--- name/description ---)"
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as e:
        return f"broken frontmatter: {e}"
    if not isinstance(meta, dict):
        return "frontmatter must be a mapping"
    name = str(meta.get("name") or path.parent.name).strip()
    desc = " ".join(str(meta.get("description") or "").split())
    if not desc:
        return "description missing"
    if len(desc) > MAX_DESCRIPTION:
        desc = desc[: MAX_DESCRIPTION - 1] + "…"
    return Skill(name, desc, path.resolve(), scope)


def _scan(root, scope, warnings):
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        res = parse_skill(skill_md, scope)
        if isinstance(res, Skill):
            found.append(res)
        else:
            warnings.append(f"skill {skill_md}: {res}")
    return found


def discover_skills(cwd, global_dir=None, extra_dirs=(), warnings=None):
    """order = ascending priority: extra < global < project (git-root … cwd).
    on equal names the later one wins."""
    warnings = warnings if warnings is not None else []
    roots = [(Path(d), "extra") for d in extra_dirs]
    if global_dir:
        roots.append((Path(global_dir) / "skills", "global"))
    roots += [(d / PROJECT_SKILL_DIR, "project") for d in project_dirs(cwd)]
    by_name = {}
    for root, scope in roots:
        for s in _scan(root, scope, warnings):
            by_name[s.name] = s
    return sorted(by_name.values(), key=lambda s: s.name)


SKILLS_INTRO = """# Skills
Skills are instructions for specific tasks. When a task matches a skill's description, read its \
SKILL.md with the read tool before you start; relative paths in it refer to the skill's directory. \
Only load skills that fit the task."""


class SkillError(Exception):
    pass


# first word "/name", then whitespace or end. "/etc/hosts" is a path, not a command.
_SKILL_CMD_RE = re.compile(r"\A/([^\s/]+)(?:\s+(.*))?\Z", re.S)
_SKILL_TAG_RE = re.compile(r'\A<skill name="([^"]*)"[^>]*>.*?\n</skill>\n\n(.*)\Z', re.S)
SKILL_DEFAULT_TASK = "Apply this skill."


def invocable(skill):
    """invocable via `/name`? "foo bar" or "a/b" are valid skills (the model reads them via the
    path), but don't fit _SKILL_CMD_RE – popup and error text don't offer them."""
    return bool(re.fullmatch(r"[^\s/]+", skill.name))


def expand_skill(prompt, skills):
    """`/name rest` → SKILL.md (without frontmatter) inline + rest. otherwise prompt unchanged.
    unknown name → SkillError, instead of silently sending it to the model."""
    m = _SKILL_CMD_RE.match(prompt.strip())
    if not m:
        return prompt
    name, rest = m.group(1), (m.group(2) or "").strip()
    skill = next((s for s in skills if s.name == name), None)
    if skill is None:
        known = ", ".join(f"/{s.name}" for s in skills if invocable(s)) or "none"
        raise SkillError(f"unknown skill /{name} (available: {known})")
    try:
        text = skill.path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise SkillError(f"skill /{name}: not readable ({e})") from e
    body = _FRONTMATTER_RE.sub("", text, count=1).strip()
    return (f'<skill name="{escape(skill.name)}" location="{escape(str(skill.path))}">\n'
            f"Relative paths in this skill refer to {escape(str(skill.path.parent))}.\n\n"
            f"{body}\n</skill>\n\n{rest or SKILL_DEFAULT_TASK}")


_SKILL_PARTIAL_RE = re.compile(r"\A/([^\s/]*)\Z")


def complete_skill(text, skills):
    """autocomplete: input is (still) just "/partial" → matching skills, prefix hits
    first, then substring in name or description. otherwise None (no popup)."""
    m = _SKILL_PARTIAL_RE.match(text)
    if not m:
        return None
    q = m.group(1).lower()
    skills = [s for s in skills if invocable(s)]
    prefix = [s for s in skills if s.name.lower().startswith(q)]
    infix = [s for s in skills if s not in prefix and q in s.name.lower()]
    desc = [s for s in skills if s not in prefix and s not in infix and q in s.description.lower()]
    return prefix + infix + desc


def skill_label(prompt):
    """inverse for display: expanded prompt → "/name rest", otherwise None."""
    m = _SKILL_TAG_RE.match(prompt)
    if not m:
        return None
    rest = m.group(2)
    return f"/{m.group(1)}" + ("" if rest == SKILL_DEFAULT_TASK else f" {rest}")


def render_skills(skills):
    if not skills:
        return ""
    items = "\n".join(
        f"  <skill>\n    <name>{escape(s.name)}</name>\n    <description>{escape(s.description)}</description>\n"
        f"    <location>{escape(str(s.path))}</location>\n  </skill>"
        for s in skills
    )
    return f"{SKILLS_INTRO}\n<available_skills>\n{items}\n</available_skills>"
