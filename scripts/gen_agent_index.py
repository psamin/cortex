"""Generate the agent-facing index files in web/public from the real tool registry.

    backend/.venv/Scripts/python scripts/gen_agent_index.py      (or: cd web && pnpm gen:agents)

Re-run it whenever a tool, its arguments or the connect snippets change;
backend/tests/test_agent_index.py fails while the committed files are stale.

Writes:
  web/public/llms.txt                   llms.txt convention (https://llmstxt.org/)
  web/public/llms-full.txt              every tool: name, description, arguments, JSON schema
  web/public/.well-known/mcp.json       Cortex discovery document (our format, documented in its "$comment")
  web/public/server-card.json           MCP Server Card, SEP-2127 (no tools, by design of that spec)
  web/public/.well-known/ai-catalog.json  AI Catalog entry pointing at the server card (SEP-2127 discovery)
  web/public/robots.txt, web/public/sitemap.xml

Sources: cortex.services.agent_orchestrator.TOOLS (names, descriptions, pydantic arg models),
mcp/src/tools.ts (the cortex.* names the MCP server exposes), mcp/package.json (version) and
web/src/landing/connect/snippets.json (client configs and the worked session shown on the landing page).
Set CORTEX_SITE_URL to the public origin when deploying (default http://localhost:5173).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from cortex.services.agent_orchestrator import NEEDS_GUI_APPROVAL, TOOLS  # noqa: E402

PUBLIC = ROOT / "web" / "public"
REPO = "https://github.com/psamin/speculativespeculativeagentaction"
NAME = "Cortex"
TAGLINE = "Build intelligence for the physical world"
SUMMARY = (
    "Your coding agent describes the robot arm and the task over the Model Context Protocol; Cortex builds a MuJoCo simulation, collects simulator-verified demonstrations, trains an ACT or SmolVLA "
    "policy and evaluates it, with a human approving every step that costs money or moves hardware."
)
DEFINITION = (
    "Cortex is an open-source (MIT) MCP server and studio for training robot policies with an AI agent, built on "
    "LeRobot, MuJoCo, ACT (Action Chunking Transformer) and the SO-101 (SO-ARM101) arm."
)
CARD_DESCRIPTION = "Robot-learning studio: simulate, collect demos, train and evaluate policies, human-approved."


def site_url() -> str:
    return os.environ.get("CORTEX_SITE_URL", "http://localhost:5173").rstrip("/")


def load_snippets() -> dict[str, Any]:
    return json.loads((ROOT / "web" / "src" / "landing" / "connect" / "snippets.json").read_text(encoding="utf-8"))


def mcp_version() -> str:
    return json.loads((ROOT / "mcp" / "package.json").read_text(encoding="utf-8"))["version"]


def mcp_names() -> dict[str, list[str]]:
    """Registry tool -> the cortex.* MCP tools that call it (robot_act is split into one MCP tool per robot action)."""
    src = (ROOT / "mcp" / "src" / "tools.ts").read_text(encoding="utf-8")
    body = src[src.index("const TOOLS: Def[] = ["):]
    out: dict[str, list[str]] = {}
    blocks = re.split(r"\n  \{\n", body)[1:]
    for b in blocks:
        m = re.search(r'^\s*name: "([a-z_]+)",', b, re.M)
        if not m:
            continue
        name = m.group(1)
        if re.search(r"backend: act\(", b):
            target = "robot_act"
        else:
            t = re.search(r'backend: \(i\) => \["([a-z_]+)"', b)
            target = t.group(1) if t else name
        out.setdefault(target, []).append(f"cortex.{name}")
    return out


# ----------------------------------------------------------------------------- schema rendering

def _type(s: dict[str, Any], defs: dict[str, Any]) -> str:
    if "$ref" in s:
        name = s["$ref"].rsplit("/", 1)[-1]
        d = defs.get(name, {})
        return _type(d, defs) if "enum" in d else name
    if "const" in s:
        return json.dumps(s["const"])
    if "enum" in s:
        return " | ".join(json.dumps(v) for v in s["enum"])
    if "anyOf" in s:
        parts = [_type(x, defs) for x in s["anyOf"]]
        return " | ".join(dict.fromkeys(parts))
    t = s.get("type", "any")
    if t == "array":
        return f"list[{_type(s.get('items', {}), defs)}]"
    if t == "object" and "additionalProperties" in s and isinstance(s["additionalProperties"], dict):
        return f"dict[string, {_type(s['additionalProperties'], defs)}]"
    return t


def _constraints(s: dict[str, Any]) -> str:
    bits = []
    for k, sym in (("minimum", ">="), ("exclusiveMinimum", ">"), ("maximum", "<="), ("maxLength", "max length")):
        if k in s:
            bits.append(f"{sym} {s[k]}")
    if "pattern" in s:
        bits.append(f"matches `{s['pattern']}`")
    return ", ".join(bits)


def arg_lines(schema: dict[str, Any], indent: str = "", defs: dict[str, Any] | None = None) -> list[str]:
    defs = defs if defs is not None else schema.get("$defs", {})
    req = set(schema.get("required", []))
    lines = []
    for k, s in schema.get("properties", {}).items():
        meta = [_type(s, defs)]
        meta.append("required" if k in req else f"default {json.dumps(s['default'])}" if "default" in s else "optional")
        if c := _constraints(s):
            meta.append(c)
        desc = f": {s['description']}" if s.get("description") else ""
        lines.append(f"{indent}- `{k}` ({', '.join(meta)}){desc}")
    return lines


def compact(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def strip_titles(s: Any) -> Any:
    """Drop pydantic's auto-generated 'title' keys (noise for agents) but keep property names."""
    if isinstance(s, dict):
        return {k: strip_titles(v) for k, v in s.items() if not (k == "title" and isinstance(v, str))}
    if isinstance(s, list):
        return [strip_titles(v) for v in s]
    return s


def tool_records() -> list[dict[str, Any]]:
    names = mcp_names()
    recs = []
    for t in TOOLS.values():
        recs.append({
            "name": t.name,
            "description": t.description,
            "mcp": names.get(t.name, []),
            "schema": strip_titles(t.args.model_json_schema()),
            "agent_callable": t.agent_callable,
            "needs_gui_approval": t.name in NEEDS_GUI_APPROVAL or t.name == "launch_training_job",
        })
    return recs


# ----------------------------------------------------------------------------- documents

# Task recipes for llms.txt: MCP tool sequences, in order. backend/tests/test_agent_index.py checks every name.
RECIPES = [
    ("Train a pick-and-place policy (SO-101, MuJoCo, ACT)", [
        "cortex.start_intake", "cortex.update_intake", "cortex.get_intake_status", "cortex.build_simulation",
        "cortex.capture_frame", "cortex.start_agent_collection", "cortex.get_collection_progress",
        "cortex.create_dataset_version", "cortex.propose_training_job", "cortex.launch_training_job",
        "cortex.get_project_state"],
     "Start from a photo of the real setup (image_path), the person's description, or scene_source='default'; ask "
     "them which. Answer intake questions with the patch each one carries until complete. Only simulator-verified successes "
     "become LeRobotDataset v3 episodes. Stop after propose_training_job and ask the person to approve it in the "
     "studio; never invent an approval token. A checkpoint (ck_...) appears in get_project_state when training ends."),
    ("Collect demonstrations by driving the arm yourself", [
        "cortex.robot_start_episode", "cortex.robot_observe", "cortex.robot_propose_grasp", "cortex.robot_gripper",
        "cortex.robot_move", "cortex.robot_verify_grasp", "cortex.robot_place", "cortex.robot_check_success",
        "cortex.robot_finish_episode"],
     "You point at objects in the overhead image; Cortex validates workspace, reach, joint limits and collisions and "
     "executes with IK. The simulator, not you, decides success."),
    ("Train from existing LeRobot data", [
        "cortex.create_project", "cortex.import_demonstrations", "cortex.get_job", "cortex.create_dataset_version",
        "cortex.propose_training_job"],
     "Imports episodes from a Hugging Face Hub dataset; they stay separate from simulation demos."),
    ("Evaluate a policy in simulation", [
        "cortex.evaluate_learned_policy", "cortex.get_job", "cortex.get_project_state", "cortex.compare_experiments"],
     "Three seeded MuJoCo branches (baseline, shifted object, dim lighting) on held-out seeds; success rates with "
     "95% Wilson intervals; compare_experiments gives deltas and p-values."),
    ("Improve a policy", [
        "cortex.improve_policy", "cortex.get_job", "cortex.launch_training_job", "cortex.evaluate_learned_policy"],
     "Targets the weakest measured condition with new demonstrations in fresh seeds, then proposes retraining "
     "(needs approval). Also useful: cortex.analyze_dataset, cortex.get_failure_clusters (VLM clusters are "
     "hypotheses), cortex.recommend_demonstrations, cortex.get_next_actions."),
    ("One-call start", ["cortex.create_policy_project", "cortex.get_collection_progress",
                        "cortex.train_from_agent_demonstrations"],
     "Creates the project and starts autonomous agent collection in one call, then builds the dataset and proposes "
     "ACT training."),
]


def load_faq() -> list[dict[str, str]]:
    return json.loads((ROOT / "web" / "src" / "landing" / "faq" / "faq.json").read_text(encoding="utf-8"))["items"]


def _mcp_label(rec: dict[str, Any]) -> str:
    return ", ".join(f"`{n}`" for n in rec["mcp"]) or "REST only"


def llms_txt(recs: list[dict[str, Any]], snip: dict[str, Any]) -> str:
    site = site_url()
    out = [f"# {NAME}", "", f"> {DEFINITION} {SUMMARY}", ""]
    out += [
        f"Cortex runs locally. MCP server (streamable HTTP): {snip['mcp_url']}, tools named `cortex.<name>`. "
        f"REST mirror: `GET {snip['rest_url']}` lists every tool with its JSON schema; `POST {snip['rest_url']}/<name>` "
        f"calls one with the arguments as the JSON body. Studio (where people approve training): {site}/studio.",
        "",
        "Rules for agents: use ids from tool results, never invent them; agents never send motor commands "
        "(Cortex validates targets and executes with IK); training starts only after a human approves it in the studio; "
        "report only what the simulator measured.",
        "",
        "**Connect** (pick your client):",
        "",
    ]
    for c in snip["clients"]:
        where = "" if c["file"] == "Terminal" else f" (`{c['file']}`)"
        out.append(f"{c['label']}{where}:")
        out += ["", "```" + ("sh" if c["file"] == "Terminal" else "json"), c["code"], "```", ""]
    out += ["**Worked session** (MCP tool calls, ids abbreviated):", ""]
    for i, s in enumerate(snip["session"], 1):
        out.append(f"{i}. `{s['tool']}` `{compact(s['args'])}`: {s['returns']}.")
        if s.get("alt"):
            out.append(f"   {s['alt']}")
    out += ["", "**Recipes by task** (MCP tools in order):", ""]
    for title, tools, note in RECIPES:
        out.append(f"- {title}: {' → '.join(f'`{t}`' for t in tools)}. {note}")
    out += ["", "**Questions agents ask:**", ""]
    for f in load_faq():
        out += [f"- {f['q']} {f['a']}"]
    out += [
        "",
        "## Docs",
        "",
        f"- [Full tool reference]({site}/llms-full.txt): all {len(recs)} registry tools with descriptions, arguments and JSON schemas",
        f"- [MCP discovery document]({site}/.well-known/mcp.json): endpoint, transport, REST mirror and tool list",
        f"- [MCP Server Card]({site}/server-card.json): SEP-2127 server card",
        f"- [MCP server README]({REPO}/blob/main/mcp/README.md): tool table, approvals, long-running jobs",
        f"- [Studio]({site}/studio): the web app where people review data and approve training",
        "",
        "## Optional",
        "",
        f"- [Source code]({REPO}): FastAPI backend, React studio, MCP server",
        f"- [Landing page]({site}/): the same content for people",
        "",
    ]
    return "\n".join(out)


def llms_full_txt(recs: list[dict[str, Any]], snip: dict[str, Any]) -> str:
    out = [
        f"# {NAME}: full tool reference",
        "",
        f"> Every tool in the Cortex registry, generated by scripts/gen_agent_index.py from "
        f"cortex.services.agent_orchestrator.TOOLS. {len(recs)} tools. Do not edit by hand.",
        "",
        f"{DEFINITION} {SUMMARY}",
        "",
        f"- MCP: {snip['mcp_url']} (streamable HTTP). Tools are named `cortex.<name>`. The MCP server splits `robot_act` "
        "into one tool per robot action (`cortex.robot_observe`, `cortex.robot_move`, ...).",
        f"- REST: `GET {snip['rest_url']}` returns this list as JSON; `POST {snip['rest_url']}/<name>` with the "
        "arguments as the JSON body calls a tool. Remote callers need `Authorization: Bearer <CORTEX_MCP_KEY>`.",
        "- Every result has `summary` and `ref_ids`, plus tool-specific data. Long jobs return a `job_id`; poll `get_job`.",
        "- `agent_callable: false` means the studio's built-in agent does not get the tool; MCP clients can still call it.",
        "- `needs approval` means a person must approve in the studio before work starts.",
        "",
        "## Tools",
        "",
    ]
    for r in recs:
        out.append(f"### {r['name']}")
        out.append("")
        flags = [f"MCP: {_mcp_label(r)}", f"REST: `POST /api/tools/{r['name']}`"]
        if not r["agent_callable"]:
            flags.append("agent_callable: false")
        if r["needs_gui_approval"]:
            flags.append("needs approval")
        out += [" · ".join(flags), "", r["description"], "", "Arguments:", ""]
        out += arg_lines(r["schema"]) or ["- none"]
        for dname, d in r["schema"].get("$defs", {}).items():
            if not d.get("properties"):
                continue
            out += ["", f"{dname}:", ""] + arg_lines(d, "  ", r["schema"]["$defs"])
        out += ["", "JSON schema:", "", "```json", compact(r["schema"]), "```", ""]
    return "\n".join(out)


def mcp_json(recs: list[dict[str, Any]], snip: dict[str, Any]) -> dict[str, Any]:
    site = site_url()
    return {
        "$comment": (
            "Cortex MCP discovery document, generated by scripts/gen_agent_index.py. Our own format: no standard yet "
            "defines a tool list at /.well-known/mcp.json. The standard pre-connection metadata is the SEP-2127 MCP "
            "Server Card at serverCard (linked from /.well-known/ai-catalog.json); tools here mirror GET /api/tools."
        ),
        "name": "cortex",
        "title": NAME,
        "description": f"{DEFINITION} {SUMMARY}",
        "version": mcp_version(),
        "transport": {"type": "streamable-http", "url": snip["mcp_url"]},
        "rest": {"list": snip["rest_url"], "call": snip["rest_url"] + "/{name}", "method": "POST"},
        "serverCard": f"{site}/server-card.json",
        "documentation": {"llms": f"{site}/llms.txt", "llmsFull": f"{site}/llms-full.txt", "repository": REPO},
        "tools": [
            {"name": r["name"], "mcpTools": r["mcp"], "description": r["description"], "inputSchema": r["schema"],
             "agentCallable": r["agent_callable"], "needsApproval": r["needs_gui_approval"]}
            for r in recs
        ],
    }


def server_card(snip: dict[str, Any]) -> dict[str, Any]:
    """SEP-2127 MCP Server Card (schema: experimental-ext-server-card/schema.ts). Tools are deliberately excluded."""
    return {
        "$schema": "https://static.modelcontextprotocol.io/schemas/v1/server-card.schema.json",
        "name": "io.github.psamin/cortex",
        "version": mcp_version(),
        "title": NAME,
        "description": CARD_DESCRIPTION,
        "websiteUrl": site_url() + "/",
        "repository": {"url": REPO, "source": "github", "subfolder": "mcp"},
        "remotes": [{"type": "streamable-http", "url": snip["mcp_url"]}],
    }


def ai_catalog() -> dict[str, Any]:
    host = re.sub(r"^https?://", "", site_url()).split("/")[0].split(":")[0]
    return {
        "specVersion": "1.0",
        "entries": [{
            "identifier": f"urn:air:{host}:mcp:cortex",
            "type": "application/mcp-server-card+json",
            "url": f"{site_url()}/server-card.json",
        }],
    }


def robots_txt() -> str:
    return (f"# Agents: start at {site_url()}/llms.txt\nUser-agent: *\nAllow: /\n\nSitemap: {site_url()}/sitemap.xml\n")


def sitemap_xml() -> str:
    urls = ["/", "/studio", "/llms.txt", "/llms-full.txt"]
    items = "\n".join(f"  <url><loc>{site_url()}{u}</loc></url>" for u in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{items}\n</urlset>\n')


def render() -> dict[str, str]:
    """Relative path under web/public -> file content."""
    recs, snip = tool_records(), load_snippets()
    j = lambda o: json.dumps(o, indent=2, ensure_ascii=False) + "\n"
    return {
        "llms.txt": llms_txt(recs, snip),
        "llms-full.txt": llms_full_txt(recs, snip),
        ".well-known/mcp.json": j(mcp_json(recs, snip)),
        ".well-known/ai-catalog.json": j(ai_catalog()),
        "server-card.json": j(server_card(snip)),
        "robots.txt": robots_txt(),
        "sitemap.xml": sitemap_xml(),
    }


def main() -> None:
    for rel, text in render().items():
        p = PUBLIC / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote web/public/{rel} ({len(text.encode())} bytes)")


if __name__ == "__main__":
    main()
