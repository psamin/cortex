"""The agent-facing index (web/public/llms*.txt, .well-known/mcp.json, server card) matches the real tool registry,
and the landing page's worked session only uses tools and arguments that exist."""

import importlib.util
import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cortex.services.agent_orchestrator import TOOLS

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "web" / "public"

_spec = importlib.util.spec_from_file_location("gen_agent_index", ROOT / "scripts" / "gen_agent_index.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


@pytest.fixture(autouse=True)
def default_site(monkeypatch):
    monkeypatch.delenv("CORTEX_SITE_URL", raising=False)


def test_generated_files_are_current():
    stale = [rel for rel, text in gen.render().items()
             if not (PUBLIC / rel).exists() or (PUBLIC / rel).read_text(encoding="utf-8") != text]
    assert not stale, f"run scripts/gen_agent_index.py (stale: {stale})"


def test_every_registered_tool_is_listed():
    full = (PUBLIC / "llms-full.txt").read_text(encoding="utf-8")
    card = json.loads((PUBLIC / ".well-known" / "mcp.json").read_text(encoding="utf-8"))
    listed = {t["name"] for t in card["tools"]}
    for name, tool in TOOLS.items():
        assert f"\n### {name}\n" in full, name
        assert tool.description in full, name
        assert name in listed, name
    assert listed == set(TOOLS)


def test_every_mcp_tool_maps_to_the_registry():
    names = gen.mcp_names()
    assert set(names) <= set(TOOLS), set(names) - set(TOOLS)
    src = (ROOT / "mcp" / "src" / "tools.ts").read_text(encoding="utf-8")
    declared = {f"cortex.{n}" for n in re.findall(r'^    name: "([a-z_]+)",', src, re.M)}
    assert declared == {n for v in names.values() for n in v}
    llms = (PUBLIC / "llms.txt").read_text(encoding="utf-8")
    assert llms.startswith("# Cortex\n\n> ")  # llms.txt: H1 then a blockquote summary


def test_server_card_follows_sep_2127():
    card = json.loads((PUBLIC / "server-card.json").read_text(encoding="utf-8"))
    assert card["$schema"] == "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json"
    assert re.fullmatch(r"[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+", card["name"])
    assert 1 <= len(card["description"]) <= 100 and card["version"]
    assert card["remotes"] == [{"type": "streamable-http", "url": "http://localhost:3000/mcp"}]
    assert "tools" not in card  # SEP-2127 deliberately excludes primitives
    cat = json.loads((PUBLIC / ".well-known" / "ai-catalog.json").read_text(encoding="utf-8"))
    assert cat["entries"][0]["type"] == "application/mcp-server-card+json"


def test_recipes_and_faq_name_real_mcp_tools():
    mcp = {n for v in gen.mcp_names().values() for n in v}
    for title, tools, note in gen.RECIPES:
        assert set(tools) <= mcp, (title, set(tools) - mcp)
        assert set(re.findall(r"cortex\.[a-z_]+", note)) <= mcp, title
    for f in gen.load_faq():
        named = {n for n in re.findall(r"cortex\.[a-z_]+", f["a"]) if not n.endswith("_")}  # skip "cortex.robot_*"
        assert named <= mcp, (f["id"], named - mcp)
        assert f["q"] in (PUBLIC / "llms.txt").read_text(encoding="utf-8")


def _backend_name(mcp_tool: str) -> str:
    return next(k for k, v in gen.mcp_names().items() if mcp_tool in v)


def test_worked_session_uses_real_tools_and_args():
    for step in gen.load_snippets()["session"]:
        name = _backend_name(step["tool"])
        model = TOOLS[name].args
        assert set(step["args"]) <= set(model.model_fields), (step["tool"], set(step["args"]) - set(model.model_fields))
        model.model_validate(step["args"])  # placeholder ids still have to fit the schema (e.g. ck_ prefix)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def test_worked_session_intake_steps_run(client):
    s = {x["tool"]: x["args"] for x in gen.load_snippets()["session"]}
    r = client.post("/api/tools/start_intake", json=s["cortex.start_intake"], headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 200, r.text
    pid = r.json()["project_id"]
    args = {**s["cortex.update_intake"], "project_id": pid}
    r = client.post("/api/tools/update_intake", json=args, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["spec"]["task"]["pick"] == "small_water_bottle" and st["spec"]["task"]["place"] == ["mat"]
