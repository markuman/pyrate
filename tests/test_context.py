from pyrate import config, context
from pyrate.agent import Agent
from pyrate.db import DB


def skill(root, name, desc, body="mach es so", extra=""):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n{extra}---\n\n{body}\n")
    return d / "SKILL.md"


def repo(tmp_path):
    """git-repo mit unterordner: root/.git, root/sub/deeper = cwd."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    cwd = root / "sub" / "deeper"
    cwd.mkdir(parents=True)
    return root, cwd


def test_project_dirs(tmp_path):
    root, cwd = repo(tmp_path)
    assert context.project_dirs(cwd) == [root, root / "sub", cwd]
    lone = tmp_path / "ohne-git"
    lone.mkdir()
    assert context.project_dirs(lone) == [lone]  # nicht bis / hochlaufen


def test_instruction_hierarchy(tmp_path):
    glob = tmp_path / "global"
    glob.mkdir()
    (glob / "AGENTS.md").write_text("global: antworte kurz")
    root, cwd = repo(tmp_path)
    (root / "AGENTS.md").write_text("repo: ansible-stil")
    (root / "CLAUDE.md").write_text("ignored, AGENTS.md hat vorrang")
    (root / "sub" / "CLAUDE.md").write_text("sub: nur claude.md")
    (cwd / "AGENTS.md").write_text("   ")  # empty → skipped
    (tmp_path / "AGENTS.md").write_text("outside the repo")

    ins = context.load_instructions(cwd, glob)
    assert [(i.path.name, i.scope, i.text) for i in ins] == [
        ("AGENTS.md", "global", "global: antworte kurz"),
        ("AGENTS.md", "project", "repo: ansible-stil"),
        ("CLAUDE.md", "project", "sub: nur claude.md"),
    ]
    out = context.render_instructions(ins)
    assert out.index("global:") < out.index("repo:") < out.index("sub:")
    assert f'<project_instructions path="{root / "AGENTS.md"}">' in out


def test_instruction_symlink_dedup_and_size_cap(tmp_path):
    root, cwd = repo(tmp_path)
    (root / "AGENTS.md").write_text("x" * 60000)
    warnings = []
    ins = context.load_instructions(root, None, warnings)
    assert len(ins[0].text) == context.MAX_INSTRUCTION_BYTES and "truncated" in warnings[0]
    (cwd / "AGENTS.md").symlink_to(root / "AGENTS.md")
    assert len(context.load_instructions(cwd)) == 1


def test_skills_discovery_and_precedence(tmp_path):
    extra, glob = tmp_path / "extra", tmp_path / "global"
    root, cwd = repo(tmp_path)
    skill(extra, "deploy", "from extra")
    skill(extra, "nur-extra", "stays")
    skill(glob / "skills", "deploy", "global beats extra")
    skill(glob / "skills", "review", "global review")
    skill(root / ".pyrate" / "skills", "review", "project beats global")
    skill(cwd / ".pyrate" / "skills", "lokal", "only in the subdir")
    bad = cwd / ".pyrate" / "skills" / "broken"
    bad.mkdir()
    (bad / "SKILL.md").write_text("kein frontmatter")
    nodesc = skill(cwd / ".pyrate" / "skills", "empty", "")

    warnings = []
    skills = context.discover_skills(cwd, glob, [extra], warnings)
    got = {s.name: (s.description, s.scope) for s in skills}
    assert got == {
        "deploy": ("global beats extra", "global"),
        "lokal": ("only in the subdir", "project"),
        "nur-extra": ("stays", "extra"),
        "review": ("project beats global", "project"),
    }
    assert [s.name for s in skills] == sorted(got)
    assert any("no yaml frontmatter" in w for w in warnings)
    assert any(str(nodesc) in w and "description missing" in w for w in warnings)


def test_skill_frontmatter_details(tmp_path):
    p = skill(tmp_path, "ordnername", "mehrzeilig\n  fortgesetzt", extra="")
    p.write_text("---\ndescription: >\n  over multiple\n  lines <&>\n---\nbody\n")
    s = context.parse_skill(p, "global")
    assert s.name == "ordnername" and s.description == "over multiple lines <&>"
    out = context.render_skills([s])
    assert "<description>over multiple lines &lt;&amp;&gt;</description>" in out
    assert f"<location>{p.resolve()}</location>" in out


def test_agent_system_prompt(tmp_path, monkeypatch):
    glob = tmp_path / "cfgdir"
    glob.mkdir()
    (glob / "config.yaml").write_text("base_url: http://x\nmodel: m\n")
    (glob / "AGENTS.md").write_text("GLOBAL-REGEL")
    skill(glob / "skills", "weather", "wetterfragen")
    root, cwd = repo(tmp_path)
    (root / "AGENTS.md").write_text("PROJEKT-REGEL")
    skill(root / ".pyrate" / "skills", "deploy", "deploy-ablauf")
    cfg = config.load(cwd, glob / "config.yaml")
    assert cfg.global_dir == glob

    agent = Agent(cfg, DB(":memory:"), cwd)
    sp = agent.system_prompt()
    assert sp.index("<available_skills>") < sp.index("GLOBAL-REGEL") < sp.index("PROJEKT-REGEL")
    assert "<name>deploy</name>" in sp and "<name>weather</name>" in sp
    assert "read its SKILL.md" in sp

    # neue datei im projekt → reload_context nimmt sie mit
    skill(cwd / ".pyrate" / "skills", "neu", "frisch")
    agent.reload_context()
    assert "<name>neu</name>" in agent.system_prompt()


def test_no_context_without_files(tmp_path):
    cfg = config.build({"base_url": "http://x", "model": "m"})
    agent = Agent(cfg, DB(":memory:"), tmp_path)
    sp = agent.system_prompt()
    assert "available_skills" not in sp and "project_instructions" not in sp


def test_expand_skill(tmp_path):
    import pytest
    path = skill(tmp_path, "zusammenfassen", "kurz", body="radikal kurz.\nsiehe ref/x.md")
    skills = [context.parse_skill(path, "global")]
    assert context.expand_skill("hallo /welt", skills) == "hallo /welt"  # nur am anfang
    assert context.expand_skill("/etc/hosts ist broken", skills) == "/etc/hosts ist broken"  # pfad
    out = context.expand_skill("  /zusammenfassen den\nthread  ", skills)
    assert out.startswith('<skill name="zusammenfassen"') and "radikal kurz." in out
    assert "description:" not in out  # frontmatter raus
    assert f"refer to {path.parent.resolve()}" in out
    assert out.endswith("</skill>\n\nden\nthread")
    assert context.skill_label(out) == "/zusammenfassen den\nthread"
    bare = context.expand_skill("/zusammenfassen", skills)
    assert bare.endswith(context.SKILL_DEFAULT_TASK) and context.skill_label(bare) == "/zusammenfassen"
    assert context.skill_label("normaler prompt") is None
    with pytest.raises(context.SkillError, match="unknown skill /nix.*available.*zusammenfassen"):
        context.expand_skill("/nix da", skills)


def test_complete_skill(tmp_path):
    skills = [context.Skill("debug", "fehler jagen", tmp_path, "global"),
              context.Skill("deploy", "ausrollen", tmp_path, "global"),
              context.Skill("redeploy", "nochmal", tmp_path, "global"),
              context.Skill("review", "deploy-diff lesen", tmp_path, "project")]
    names = lambda t: [s.name for s in context.complete_skill(t, skills)]
    assert names("/") == ["debug", "deploy", "redeploy", "review"]
    assert names("/DEP") == ["deploy", "redeploy", "review"]  # prefix, substring, description
    assert names("/xyz") == []
    for t in ("", "hallo", "/deploy ", "/etc/hosts", "/de\nx"):
        assert context.complete_skill(t, skills) is None


def test_uninvocable_names_not_offered(tmp_path):
    """"foo bar" ist ein valid skill for the model, but /foo bar not invocable:
    popup and error text must not offer it."""
    import pytest
    skills = [context.Skill("foo bar", "mit leerzeichen", tmp_path, "global"),
              context.Skill("a/b", "mit slash", tmp_path, "global"),
              context.Skill("foo", "normal", tmp_path, "global")]
    assert [s.name for s in context.complete_skill("/", skills)] == ["foo"]
    assert [s.name for s in context.complete_skill("/leerz", skills)] == []  # not even via the description
    with pytest.raises(context.SkillError) as e:
        context.expand_skill("/nix", skills)
    assert "/foo" in str(e.value) and "foo bar" not in str(e.value) and "a/b" not in str(e.value)
