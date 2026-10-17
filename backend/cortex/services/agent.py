"""The studio's agent panel.

With a Model Gateway key: an LLM (OpenRouter, OpenAI-compatible tool calling)
plans over the same tool registry the MCP server exposes. Without one: a small
deterministic intent planner covers the core flow. Either way the agent never
launches gated work itself; it emits `approval_required` cards the human clicks.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..context import Ctx
from . import agent_orchestrator as orch
from . import autopilot

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_TURNS = 8
HISTORY: dict[str, list[dict[str, Any]]] = {}

SYSTEM = """You are Cortex, an experiment orchestrator for robot learning on LeRobot (SO-101 arm, MuJoCo sim).
Project: {project}
Act, don't ask: call tools to make progress, then explain in 2-4 short sentences what you did and what the evidence shows.
Rules:
- Distinguish measured results (sim evaluations, quality checks) from VLM hypotheses (unverified).
- Never invent numbers; only cite tool outputs.
- Evaluations and training proposals are surfaced as approval cards for the human; call the tool and say approval is pending.
- To run the whole loop unattended, call start_autopilot.
Current state summary: {state}"""

EXTRA_TOOLS = {
    "start_autopilot": "Run the full loop (collect, analyze, version, train, evaluate, recommend, repeat) under the "
                       "project's standing approvals.",
}


def tool_specs() -> list[dict[str, Any]]:
    specs = []
    for t in orch.TOOLS.values():
        if not t.agent_callable:
            continue
        schema = t.args.model_json_schema()
        schema.get("properties", {}).pop("project_id", None)
        if "required" in schema:
            schema["required"] = [r for r in schema["required"] if r != "project_id"]
        specs.append({"type": "function", "function": {"name": t.name, "description": t.description,
                                                       "parameters": schema}})
    for name, desc in EXTRA_TOOLS.items():
        specs.append({"type": "function", "function": {"name": name, "description": desc,
                                                       "parameters": {"type": "object", "properties": {}}}})
    return specs


async def run_tool(ctx: Ctx, pid: str, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], dict | None]:
    """Returns (result, approval card or None)."""
    if name == "start_autopilot":
        run = await autopilot.start(ctx, await orch.get_project(ctx, pid))
        return {"summary": f"Autopilot {run.id} started", "ref_ids": [run.id]}, None
    if name == "launch_simulation_evaluation":
        card = {"kind": "evaluation", "ref_id": pid, "summary": (
            f"Evaluate {args.get('policy', 'scripted_expert')} on baseline / shifted object / dim lighting, "
            f"{args.get('episodes_per_branch', 10)} seeds each"), "args": {**args, "project_id": pid}}
        return {"summary": "Evaluation proposed; waiting for your approval", "ref_ids": []}, card
    res = await orch.call_tool(ctx, name, {**args, "project_id": pid} if "project_id" in
                               orch.TOOLS[name].args.model_fields else args, actor="agent")
    card = None
    if name == "propose_training_job":
        tj = res["training_job"]
        card = {"kind": "training", "ref_id": tj["id"], "summary": res["summary"], "args": {}}
    return res, card


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def chat(ctx: Ctx, pid: str, message: str) -> AsyncIterator[str]:
    hist = HISTORY.setdefault(pid, [])
    hist.append({"role": "user", "content": message})
    await ctx.audit("gui", "agent_message", pid, {"message": message[:500]}, "ok", "user message")
    if ctx.openrouter_key:
        async for chunk in _llm(ctx, pid, hist):
            yield chunk
    else:
        async for chunk in _rules(ctx, pid, message, hist):
            yield chunk
    yield _sse("done", {})


async def _llm(ctx: Ctx, pid: str, hist: list[dict[str, Any]]) -> AsyncIterator[str]:
    project = await orch.get_project(ctx, pid)
    st = await orch.project_state(ctx, project)
    sysmsg = SYSTEM.format(project=f"{project.name}: {project.spec.goal} ({project.mode})",
                           state=json.dumps({"episodes": st.episode_counts, "evaluations": st.evaluations,
                                             "next": [n.action for n in st.next_actions]}, default=str))
    msgs: list[dict[str, Any]] = [{"role": "system", "content": sysmsg}, *hist[-20:]]
    async with httpx.AsyncClient(timeout=120) as http:
        for _ in range(MAX_TURNS):
            r = await http.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {ctx.openrouter_key}"}, json={
                "model": ctx.settings.agent_model, "messages": msgs, "tools": tool_specs(), "max_tokens": 1200})
            if r.status_code >= 400:
                yield _sse("text", {"delta": f"Model Gateway error {r.status_code}: {r.text[:200]}"})
                return
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if msg.get("content"):
                yield _sse("text", {"delta": msg["content"]})
            msgs.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not calls:
                hist.append({"role": "assistant", "content": msg.get("content") or ""})
                return
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                yield _sse("tool_call", {"tool": name, "args": args})
                try:
                    res, card = await run_tool(ctx, pid, name, args)
                    out = {"summary": res.get("summary"), "ref_ids": res.get("ref_ids", [])}
                    if card:
                        yield _sse("approval_required", card)
                except Exception as e:
                    res = {"error": f"{type(e).__name__}: {e}"}
                    out = {"summary": res["error"], "ref_ids": []}
                yield _sse("tool_result", {"tool": name, **out})
                msgs.append({"role": "tool", "tool_call_id": c["id"],
                             "content": json.dumps(res, default=str)[:6000]})


INTENTS: list[tuple[str, str]] = [
    (r"autopilot|do (it|everything)|automat|end.to.end|whole loop", "start_autopilot"),
    (r"record|collect|demo", "record_demonstrations"),
    (r"analy[sz]|quality|diagnos|why.*fail", "analyze_dataset"),
    (r"recommend|what (should|next)|suggest", "recommend_demonstrations"),
    (r"dataset|version|snapshot|augment", "create_dataset_version"),
    (r"train", "propose_training_job"),
    (r"eval|test|benchmark", "launch_simulation_evaluation"),
    (r"compar", "compare_experiments"),
    (r"status|state|progress", "get_project_state"),
]


async def _rules(ctx: Ctx, pid: str, message: str, hist: list[dict[str, Any]]) -> AsyncIterator[str]:
    m = message.lower()
    name = next((t for pat, t in INTENTS if re.search(pat, m)), "get_next_actions")
    project = await orch.get_project(ctx, pid)
    st = await orch.project_state(ctx, project)
    args: dict[str, Any] = {}
    if name == "record_demonstrations":
        nxt = next((n for n in st.next_actions if n.action == "record_demonstrations"), None)
        args = dict(nxt.args) if nxt else {"conditions": [{"cube_region": "center", "lighting": "normal", "count": 3}]}
        args.pop("project_id", None)
    elif name == "create_dataset_version" and "augment" in m:
        args = {"augmentation_preset": "lighting_robustness"}
    elif name == "propose_training_job":
        if not st.dataset_versions:
            yield _sse("text", {"delta": "There is no dataset version yet, so I'll snapshot the accepted demos first."})
            name = "create_dataset_version"
        else:
            args = {"dataset_version_id": st.dataset_versions[-1]}
    elif name == "launch_simulation_evaluation":
        from ..models import Checkpoint
        ck = await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)
        args = {"policy": ck[-1].id if ck else "scripted_expert"}
    yield _sse("tool_call", {"tool": name, "args": args})
    try:
        res, card = await run_tool(ctx, pid, name, args)
        if card:
            yield _sse("approval_required", card)
        yield _sse("tool_result", {"tool": name, "summary": res.get("summary"), "ref_ids": res.get("ref_ids", [])})
        text = res.get("summary", "")
        if name == "get_next_actions":
            text = "Here's what I'd do next: " + "; ".join(f"{n.action} ({n.reason})" for n in st.next_actions[:4])
    except Exception as e:
        text = f"That failed: {type(e).__name__}: {e}"
    note = " (rule-based planner: add an InsForge key to enable the LLM agent)"
    yield _sse("text", {"delta": text + note})
    hist.append({"role": "assistant", "content": text})
