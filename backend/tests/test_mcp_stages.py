"""#358: the MCP server's ?toolset=core listing keeps every tool AGENTS.md tells an agent to call."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _meta() -> dict[str, tuple[str, str]]:
    src = (ROOT / "mcp" / "src" / "stages.ts").read_text(encoding="utf-8")
    return {n: (st, tier) for n, st, tier in
            re.findall(r'^  ([a-z_]+): \{ stage: "([a-z]+)", tier: "([a-z]+)"', src, re.M)}


def _mcp_tools() -> set[str]:
    tools = (ROOT / "mcp" / "src" / "tools.ts").read_text(encoding="utf-8")
    views = (ROOT / "mcp" / "src" / "views.ts").read_text(encoding="utf-8")
    return set(re.findall(r'^    name: "([a-z_]+)",', tools, re.M)) | {
        n for n in re.findall(r'name: "cortex\.([a-z_]+)"', views)}


def test_every_tool_agents_md_names_is_in_the_core_toolset():
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    tools = _mcp_tools()
    named = {n for n in re.findall(r"`([a-z_]+)`", agents) if n in tools}
    assert len(named) > 10  # the parse found the path
    meta = _meta()
    hidden = sorted(n for n in named if meta.get(n, ("manual", "advanced"))[1] not in ("golden", "custom"))
    assert not hidden, f"?toolset=core hides {hidden}: give them a golden or custom META entry in stages.ts"
    unstaged = sorted(n for n in named if meta.get(n, ("manual",))[0] == "manual")
    assert not unstaged, f"?project= hides {unstaged}"


def test_meta_names_real_tools():
    assert not set(_meta()) - _mcp_tools()
