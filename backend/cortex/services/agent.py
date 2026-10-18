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


async def engine(ctx: Ctx) -> str:
    """Which agent answers the studio chat: the user's Claude Code, a model route, or rules."""
    from . import claude_code_agent as cc

    want = ctx.settings.agent_engine
    if want in ("auto", "claude_code") and cc.available() is None and await cc.mcp_up():
        return "claude_code"
    if want in ("auto", "model") and ctx.model:
        return "model"
    return "rules"


def _log_path(ctx: Ctx, pid: str):
    return ctx.data_dir / "agent_sessions" / pid / "chat.jsonl"


def load_log(ctx: Ctx, pid: str) -> list[dict[str, Any]]:
    """The full chat as the UI shows it: user messages, assistant text, tool calls/results, approvals."""
    p = _log_path(ctx, pid)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _append_log(ctx: Ctx, pid: str, entry: dict[str, Any]) -> None:
    from ..models import now

    p = _log_path(ctx, pid)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({**entry, "at": now().isoformat()}, default=str) + "\n")


async def chat(ctx: Ctx, pid: str, message: str) -> AsyncIterator[str]:
    """Stream one agent turn and record it to the project's chat log."""
    _append_log(ctx, pid, {"kind": "user", "text": message})
    text: list[str] = []
    async for chunk in _chat(ctx, pid, message):
        ev, _, data = chunk.partition("\ndata: ")
        ev = ev.removeprefix("event: ")
        try:
            d = json.loads(data)
        except ValueError:
            d = {}
        if ev == "text":
            text.append(d.get("delta", ""))
        elif ev in ("tool_call", "tool_result", "approval_required"):
            if text:
                _append_log(ctx, pid, {"kind": "assistant", "text": "".join(text)})
                text = []
            # keep the card's own kind (e.g. "training") under card_kind; `kind` is the row type
            _append_log(ctx, pid, {**d, "card_kind": d.get("kind"), "kind": ev})
        elif ev == "done":
            if text:
                _append_log(ctx, pid, {"kind": "assistant", "text": "".join(text)})
                text = []
            _append_log(ctx, pid, {"kind": "done", **d})
        yield chunk


async def _chat(ctx: Ctx, pid: str, message: str) -> AsyncIterator[str]:
    hist = HISTORY.setdefault(pid, [])
    hist.append({"role": "user", "content": message})
    await ctx.audit("gui", "agent_message", pid, {"message": message[:500]}, "ok", "user message")
    eng = await engine(ctx)
    if eng == "claude_code":
        from . import claude_code_agent as cc

        text = []
        async for chunk in cc.chat(ctx, pid, message, _sse):
            if chunk.startswith("event: text"):
                text.append(json.loads(chunk.split("data: ", 1)[1])["delta"])
            yield chunk
        hist.append({"role": "assistant", "content": "".join(text)})
        return
    if ctx.model:
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
            route = ctx.model
            assert route is not None
            r = await http.post(route.chat_url, headers={"Authorization": f"Bearer {route.api_key}"}, json={
                "model": route.agent_model, "messages": msgs, "tools": tool_specs(), "max_tokens": 1200})
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
    note = " (rule-based planner: configure a model endpoint to enable the LLM agent)"
    yield _sse("text", {"delta": text + note})
    hist.append({"role": "assistant", "content": text})


# ------------------------------------------------- deterministic job status lines in the chat

JOB_LABELS = {"record": "Recording demonstrations", "agent_collect": "Demonstration collection",
              "analyze": "Quality analysis", "dataset": "Dataset build", "evaluate": "Evaluation",
              "train": "Training", "train_from_agent": "Dataset + training proposal",
              "improve_policy": "Improvement round", "import": "Demonstration import"}


async def post_job_done(ctx: Ctx, job) -> None:
    """When a job finishes, Cortex (not the LLM) posts a fixed status line into the chat."""
    from ..models import AgentCollection, DatasetVersion

    label = JOB_LABELS.get(job.kind, job.kind)
    detail = ""
    try:
        if job.status == "succeeded" and job.result_ref:
            ref = job.result_ref
            if ref.startswith("ac_"):
                c = await ctx.repo.get("agent_collections", ref, AgentCollection)
                if c:
                    t = c.totals
                    detail = f"{t.get('successes', 0)}/{t.get('attempts', 0)} demos succeeded in simulation"
            elif ref.startswith("ds_"):
                d = await ctx.repo.get("dataset_versions", ref, DatasetVersion)
                if d:
                    detail = f"{d.name}: {d.num_source_episodes} demos, {len(d.links)} LeRobot episodes"
            elif ref.startswith("ev_"):
                from . import simulation_runner
                ev = await simulation_runner.get_evaluation(ctx, ref)
                if ev:
                    detail = ", ".join(f"{b.spec.name} {b.result.successes}/{b.result.n}" for b in ev["branches"]
                                       if b.result)
            elif ref.startswith("an_"):
                from . import analysis_service
                an = await analysis_service.latest_analysis(ctx, job.project_id)
                detail = an.summary if an else ""
    except Exception:
        detail = ""
    text = {"succeeded": f"{label} finished" + (f": {detail}" if detail else ""),
            "failed": f"{label} failed: {job.error or job.message or 'unknown error'}",
            "cancelled": f"{label} stopped"}.get(job.status, f"{label}: {job.status}")
    entry = {"kind": "job_done", "job_id": job.id, "job_kind": job.kind, "status": job.status,
             "text": text, "ref_ids": [r for r in (job.result_ref or "").split(",") if r][:12]}
    _append_log(ctx, job.project_id, entry)
    ctx.bus.publish(job.project_id, "chat", entry)


def install_job_listener(ctx: Ctx) -> None:
    import asyncio

    def on_finished(job) -> None:
        if job.kind in JOB_LABELS:
            asyncio.get_running_loop().create_task(post_job_done(ctx, job))
    ctx.jobs.on_finished.append(on_finished)
