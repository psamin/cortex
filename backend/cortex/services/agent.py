"""The studio's agent panel.

With a Model Gateway key: an LLM (OpenRouter, OpenAI-compatible tool calling)
plans over the same tool registry the MCP server exposes. Without one: a small
deterministic intent planner covers the core flow. Either way the agent never
approves gated work itself: a human clicks an `approval_required` card, or, in
autonomous mode, the project's standing approval covers it (services/autonomy.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..context import Ctx
from . import agent_orchestrator as orch
from . import autonomy, autopilot
from .narration import tool_line

MAX_TURNS = 8
MODEL_TIMEOUT_S = 60  # one model reply; past it the turn falls back or stops, it never hangs
HISTORY: dict[str, list[dict[str, Any]]] = {}

log = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """The model route failed before doing anything, so the built-in planner answers this turn instead."""

SYSTEM = """You are Cortex, an experiment orchestrator for robot learning on LeRobot (SO-101 arm, MuJoCo sim).
Project: {project}
Act, don't ask: call tools to make progress, then explain in 2-4 short sentences what you did and what the evidence shows.
Rules:
- Distinguish measured results (sim evaluations, quality checks) from VLM hypotheses (unverified).
- Never invent numbers; only cite tool outputs.
{mode_rules}
- Work in this project's sim scene. When the user describes a different task or different objects, rebuild it:
  start_intake with their words, update_intake with the suggested patch for anything they did not say, then
  build_simulation.
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
    if name == "launch_simulation_evaluation" and not autonomy.is_autonomous(await orch.get_project(ctx, pid)):
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


RULES_LABEL = "built-in planner"


async def engine_info(ctx: Ctx) -> dict[str, Any]:
    """Which agent answers the studio chat (the user's Claude Code, a model route, or the built-in planner),
    its label, and `note`: what is missing for a better one, in plain words (None when nothing is)."""
    from . import claude_code_agent as cc

    want = ctx.settings.agent_engine
    missing: list[str] = []

    def using(label: str) -> str | None:
        note = f"{' and '.join(missing)}: using {label}" if missing else None
        return note[0].upper() + note[1:] if note else None

    if want in ("auto", "claude_code"):
        why = cc.available()
        if why is None and not await cc.mcp_up():
            why = "the Cortex MCP server is not running (start it with scripts/dev.sh)"
        if why is None:
            return {"engine": "claude_code", "label": "Claude Code (your agent)", "note": None}
        missing.append(why)
    if want in ("auto", "model"):
        if ctx.model:
            return {"engine": "model", "label": ctx.model.label, "note": using(ctx.model.label)}
        missing.append("no model key is set")
    return {"engine": "rules", "label": RULES_LABEL, "note": using(f"the {RULES_LABEL}")}


async def engine(ctx: Ctx) -> str:
    return (await engine_info(ctx))["engine"]


async def load_log(ctx: Ctx, pid: str, tid: str | None = None) -> list[dict[str, Any]]:
    """One chat thread as the UI shows it (the most recent one unless `tid` is given)."""
    from . import chat_store

    items = await chat_store.threads(ctx, pid)
    if not items:
        return []
    return tidy_job_rows(await chat_store.load(ctx, pid, tid or items[0]["id"]))


def tidy_job_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Job rows in words people can act on, also for rows written before that rule existed; a
    failure followed by a success of the same kind is marked superseded (it stays in history)."""
    from ..errors import friendly

    later_ok: set[str] = set()
    for e in reversed(entries):
        if e.get("kind") != "job_done":
            continue
        if e.get("status") == "failed":
            label = JOB_LABELS.get(e.get("job_kind", ""), e.get("job_kind", "Job"))
            if "detail" not in e:  # old row: "<label> failed: <raw error>"
                raw = str(e.get("text", "")).split(" failed: ", 1)[-1]
                msg, detail = friendly(raw)
                e["text"], e["detail"] = f"{label} failed: {msg}", detail
            e["superseded"] = e.get("job_kind") in later_ok
        elif e.get("status") == "succeeded":
            later_ok.add(e.get("job_kind", ""))
    return entries


async def _append_log(ctx: Ctx, pid: str, entry: dict[str, Any], tid: str | None = None) -> None:
    from . import chat_store

    t = await chat_store.current(ctx, pid, tid)
    await chat_store.append(ctx, pid, t["id"], entry)


async def chat(ctx: Ctx, pid: str, message: str, tid: str | None = None) -> AsyncIterator[str]:
    """Stream one user turn and record it to the chat thread (the latest unless `tid` is given).
    Turns in a project run one at a time: a message sent during an automatic step waits for it."""
    async with autonomy.lock(pid):
        async for chunk in turn(ctx, pid, message, tid):
            yield chunk


async def turn(ctx: Ctx, pid: str, message: str, tid: str | None = None, auto: str | None = None) -> AsyncIterator[str]:
    """One agent turn; the caller holds autonomy.lock(pid). `auto`: Cortex woke the agent (autonomous mode)
    because this work finished; the row shows that instead of a user message."""
    import time

    from . import chat_store, project_cleanup

    tid = (await chat_store.current(ctx, pid, tid))["id"]
    yield _sse("thread", {"id": tid})
    await project_cleanup.restore(ctx, pid)  # a chat is history: an empty project deleted earlier comes back
    autonomy.turn_started(ctx, pid, auto=bool(auto))
    if auto:
        row = autonomy.notice("auto", f"Continuing on my own: {auto}")
        await _append_log(ctx, pid, row, tid)
        yield _sse("notice", row)
    else:
        await _append_log(ctx, pid, {"kind": "user", "text": message}, tid)
    before = await _counts(ctx, pid)
    text: list[str] = []
    last_result_at = time.time()
    async for chunk in _ends_with_done(_chat(ctx, pid, message, tid, actor="cortex" if auto else "gui")):
        ev, _, data = chunk.partition("\ndata: ")
        ev = ev.removeprefix("event: ")
        try:
            d = json.loads(data)
        except ValueError:
            d = {}
        if ev == "text":
            text.append(d.get("delta", ""))
        elif ev == "todos":
            await _append_log(ctx, pid, {"kind": "todos", "items": d.get("items", [])}, tid)
        elif ev == "notice":  # Cortex's own line, e.g. which engine answered and why
            await _append_log(ctx, pid, d, tid)
        elif ev in ("tool_call", "tool_result", "approval_required"):
            if text:
                await _append_log(ctx, pid, {"kind": "assistant", "text": "".join(text)}, tid)
                text = []
            # keep the card's own kind (e.g. "training") under card_kind; `kind` is the row type
            await _append_log(ctx, pid, {**d, "card_kind": d.get("kind"), "kind": ev}, tid)
        elif ev == "done":
            if text:
                await _append_log(ctx, pid, {"kind": "assistant", "text": "".join(text)}, tid)
                text = []
            changes = _diff(before, await _counts(ctx, pid))
            if changes:  # what this turn actually changed, measured from the project, not the model's words
                await _append_log(ctx, pid, {"kind": "changes", "items": changes}, tid)
                yield _sse("changes", {"items": changes})
            await _append_log(ctx, pid, {"kind": "done", **d}, tid)
        yield chunk
        if ev in ("tool_result", "approval_required"):
            last_result_at = time.time()
            # autonomous mode: training the agent proposed starts now if the standing approval covers it
            for ref in {r for r in [*d.get("ref_ids", []), d.get("ref_id") or ""] if r.startswith("tj_")}:
                if row := await autonomy.approve_if_covered(ctx, pid, ref):
                    await _append_log(ctx, pid, row, tid)
                    yield _sse("notice", row)
        elif ev == "done":
            await autonomy.turn_ended(ctx, pid, tid, last_result_at)


async def _ends_with_done(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    """Every turn ends with `done`, also when an engine raises, so the studio never waits on a dead stream."""
    done = False
    try:
        async for chunk in chunks:
            done = done or chunk.startswith("event: done")
            yield chunk
    except Exception as e:
        log.exception("studio agent turn failed")
        yield _sse("text", {"delta": f"\n\nThis turn failed ({type(e).__name__}: {str(e)[:200]}). "
                                     "The backend log has the details."})
        if not done:
            yield _sse("done", {"is_error": True})
        return
    if not done:
        yield _sse("done", {})


CHANGE_LABELS = {"episodes": ("demo", "demos"), "dataset_versions": ("dataset version", "dataset versions"),
                 "training_jobs": ("training job", "training jobs"), "checkpoints": ("policy", "policies"),
                 "evaluation_runs": ("evaluation", "evaluations"), "agent_collections": ("collection", "collections")}


async def _counts(ctx: Ctx, pid: str) -> dict[str, int]:
    out = {}
    for table in CHANGE_LABELS:
        try:
            out[table] = len(await ctx.store.list(table, project_id=pid))
        except Exception:  # a store without the table counts as unchanged
            out[table] = 0
    return out


def _diff(before: dict[str, int], after: dict[str, int]) -> list[dict[str, Any]]:
    items = []
    for table, (one, many) in CHANGE_LABELS.items():
        n = after.get(table, 0) - before.get(table, 0)
        if n:
            items.append({"what": table, "delta": n, "text": f"{'+' if n > 0 else ''}{n} {one if abs(n) == 1 else many}"})
    return items


async def _chat(ctx: Ctx, pid: str, message: str, tid: str | None = None, actor: str = "gui") -> AsyncIterator[str]:
    hist = HISTORY.setdefault(f"{pid}:{tid}", [])
    hist.append({"role": "user", "content": message})
    await ctx.audit(actor, "agent_message", pid, {"message": message[:500]}, "ok",
                    "user message" if actor == "gui" else "autonomous mode: continue after background work")
    scene: list[str] = []
    project = await orch.get_project(ctx, pid)
    if actor == "gui" and _wants_scene(project, message):
        async for chunk in _scene_from_prompt(ctx, pid, message, project.mode.value, scene):
            yield chunk
    # the agents hear what Cortex just built, so they work in that scene instead of rebuilding or using the cube
    prompt = message + (f"\n\n[Cortex built this project's sim scene from that message: {scene[0]}]" if scene else "")
    info = await engine_info(ctx)
    eng = info["engine"]
    if info["note"] and not await _said(ctx, pid, tid, info["note"]):  # what is missing, once per chat
        yield _sse("notice", autonomy.notice("engine", info["note"]))
    if eng == "claude_code":
        from . import claude_code_agent as cc

        text = []
        try:
            async for chunk in cc.chat(ctx, pid, prompt, _sse, tid):
                if chunk.startswith("event: text"):
                    text.append(json.loads(chunk.split("data: ", 1)[1])["delta"])
                yield chunk
            hist.append({"role": "assistant", "content": "".join(text)})
            return
        except cc.Unavailable as e:  # it did nothing yet, so the next engine answers this turn
            eng = "model" if ctx.model and ctx.settings.agent_engine == "auto" else "rules"
            nxt = ctx.model.label if eng == "model" and ctx.model else f"the {RULES_LABEL}"
            yield _sse("notice", autonomy.notice("engine", f"{_cap(str(e))}: {nxt} answered instead."))
    if eng == "model":
        hist[-1]["content"] = prompt
        try:
            async for chunk in _llm(ctx, pid, hist):
                yield chunk
        except ModelUnavailable as e:
            eng = "rules"
            yield _sse("notice", autonomy.notice("engine", f"{_cap(str(e))}: the {RULES_LABEL} answered instead."))
    if eng == "rules":
        async for chunk in _rules(ctx, pid, message, hist, scene[0] if scene else None):
            yield chunk
    yield _sse("done", {})


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


async def _said(ctx: Ctx, pid: str, tid: str | None, text: str) -> bool:
    from . import chat_store

    tid = tid or (await chat_store.current(ctx, pid))["id"]
    return any(e.get("kind") == "notice" and e.get("text") == text for e in await chat_store.load(ctx, pid, tid))


# ------------------------------------------------- a new task prompt becomes this project's own scene (#192)

# where things get put, in the user's words: the place zone is named after the last one the prompt mentions
PLACES = ("bin", "bowl", "box", "basket", "tray", "plate", "mat", "coaster", "bucket", "container", "dish", "crate",
          "shelf", "drawer")


def _wants_scene(project, message: str) -> bool:
    """The message names catalogue objects this project's sim doesn't have. The cube alone keeps the built-in
    tabletop task (the fast path)."""
    from ..intake import catalogue

    named = set(catalogue.mentioned(message))
    if project.intake is None:
        return bool(named - {"cube"})
    return bool(named - {catalogue.lookup(o.name.replace("_", " ")) for o in project.intake.objects})


def _merge_patch(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _merge_patch(out[k], v)
        elif isinstance(out.get(k), list) and isinstance(v, list):
            out[k] = out[k] + v  # intake lists merge by name, so two answers' items both land
        else:
            out[k] = v
    return out


def _default_answers(st: dict[str, Any], message: str) -> dict[str, Any]:
    """One patch answering every open required question with its suggested value."""
    from ..intake.service import DEFAULT_ZONES

    patch: dict[str, Any] = {}
    for q in st.get("questions", []):
        if q.get("required") and q.get("patch"):
            patch = _merge_patch(patch, q["patch"])
    hits = [(m.start(), w) for w in PLACES for m in re.finditer(rf"\b{w}\b", message.lower())]
    if patch.get("zones") == DEFAULT_ZONES and hits:
        patch["zones"] = [{**DEFAULT_ZONES[0], "name": max(hits)[1]}, *DEFAULT_ZONES[1:]]
    return patch


def _describe_scene(p, feasibility: str) -> str:
    objs = ", ".join(f"{o.name.replace('_', ' ')} ({o.shape}, {' x '.join(f'{s:g}' for s in o.size)} m, {o.mass_kg:g} kg)"
                     for o in p.intake.objects)
    zone = (p.intake.task.place or ["?"])[0].replace("_", " ")
    check = ("The arm can reach and grasp everything." if feasibility == "feasible" else
             f"The feasibility check flags {feasibility.removeprefix('not feasible: ').replace('_', ' ')}: "
             "the arm may not grasp or reach these as described.")
    return (f"{objs}; place zone '{zone}'. {check} Sizes, weights and positions are defaults: "
            "say what differs in your setup and it gets rebuilt.")


async def _scene_from_prompt(ctx: Ctx, pid: str, message: str, mode: str, out: list[str]) -> AsyncIterator[str]:
    """start_intake with the prompt on this project, the suggested answer to every open question, build_simulation.
    Appends a one-line description of the built scene to `out`; a failed step is shown and the turn goes on."""
    name, args = "start_intake", {"prompt": message, "mode": mode}
    try:
        for _ in range(5):  # start, then answers (a later question only gets a suggestion once earlier ones exist)
            yield _sse("tool_call", {"tool": name, "args": args})
            yield _sse("status", {"state": "running", "tool": name, "text": tool_line(name, args)})
            st, _ = await run_tool(ctx, pid, name, args)
            yield _sse("tool_result", {"tool": name, "summary": st.get("summary"), "ref_ids": st.get("ref_ids", [])})
            if st.get("complete") or not (patch := _default_answers(st, message)):
                break
            name, args = "update_intake", {"patch": patch}
        if not st.get("complete"):
            return
        name, args = "build_simulation", {}
        yield _sse("tool_call", {"tool": name, "args": args})
        yield _sse("status", {"state": "running", "tool": name, "text": tool_line(name, args)})
        res, _ = await run_tool(ctx, pid, name, args)
        yield _sse("tool_result", {"tool": name, "summary": res.get("summary"), "ref_ids": res.get("ref_ids", [])})
        out.append(_describe_scene(await orch.get_project(ctx, pid), res["feasibility"].get("summary", "")))
    except Exception as e:
        from ..errors import friendly

        yield _sse("tool_result", {"tool": name, "summary": friendly(str(e))[0], "ref_ids": [], "ok": False})


async def _llm(ctx: Ctx, pid: str, hist: list[dict[str, Any]]) -> AsyncIterator[str]:
    project = await orch.get_project(ctx, pid)
    st = await orch.project_state(ctx, project)
    sysmsg = SYSTEM.format(project=f"{project.name}: {project.spec.goal} ({project.mode})",
                           mode_rules=autonomy.mode_rules(project),
                           state=json.dumps({"episodes": st.episode_counts, "evaluations": st.evaluations,
                                             "next": [n.action for n in st.next_actions]}, default=str))
    msgs: list[dict[str, Any]] = [{"role": "system", "content": sysmsg}, *hist[-20:]]
    route = ctx.model
    assert route is not None
    async with httpx.AsyncClient(timeout=httpx.Timeout(MODEL_TIMEOUT_S, connect=10)) as http:
        for step in range(MAX_TURNS):
            yield _sse("status", {"state": "thinking", "text": "Thinking…"})
            try:
                async with asyncio.timeout(MODEL_TIMEOUT_S):  # the whole reply, also when it trickles in
                    r = await http.post(route.chat_url, headers={"Authorization": f"Bearer {route.api_key}"}, json={
                        "model": route.agent_model, "messages": msgs, "tools": tool_specs(), "max_tokens": 1200})
                if r.status_code >= 400:
                    raise ModelUnavailable(f"{route.label} refused the request (HTTP {r.status_code})")
                msg = r.json()["choices"][0]["message"]
            except Exception as e:
                why = (str(e) if isinstance(e, ModelUnavailable)
                       else f"{route.label} did not answer within {MODEL_TIMEOUT_S:g} s"
                       if isinstance(e, TimeoutError | httpx.TimeoutException)
                       else f"cannot reach {route.label}" if isinstance(e, httpx.HTTPError)
                       else f"{route.label} sent a reply Cortex could not read")
                if step == 0:  # nothing happened yet: another engine can answer
                    raise ModelUnavailable(why) from e
                yield _sse("text", {"delta": f"\n\n{_cap(why)}, so this turn stops here."})
                return
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
                yield _sse("status", {"state": "running", "tool": name, "text": tool_line(name, args)})
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


async def _rules(ctx: Ctx, pid: str, message: str, hist: list[dict[str, Any]],
                 scene: str | None = None) -> AsyncIterator[str]:
    m = message.lower()
    name = next((t for pat, t in INTENTS if re.search(pat, m)), None)
    if scene and name is None:  # the message described a new task: the scene built from it is the answer
        text = f"I built the simulation from your description: {scene} Next, ask me to collect demonstrations."
        yield _sse("text", {"delta": text})
        hist.append({"role": "assistant", "content": text})
        return
    name = name or "get_next_actions"
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
    yield _sse("status", {"state": "running", "tool": name, "text": tool_line(name, args)})
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
    yield _sse("text", {"delta": text})  # which engine answered, and why, is the engine notice (#192)
    hist.append({"role": "assistant", "content": text})


# ------------------------------------------------- deterministic job status lines in the chat

JOB_LABELS = {"record": "Recording demonstrations", "agent_collect": "Demonstration collection",
              "analyze": "Quality analysis", "dataset": "Dataset build", "evaluate": "Evaluation",
              "train": "Training", "train_from_agent": "Dataset + training proposal",
              "improve_policy": "Improvement round", "import": "Demonstration import"}


async def post_job_done(ctx: Ctx, job) -> None:
    """When a job finishes, Cortex (not the LLM) posts a fixed status line into the chat."""
    from ..models import AgentCollection, DatasetVersion
    from . import project_cleanup

    await project_cleanup.restore(ctx, job.project_id)  # a job ran: an empty project deleted earlier comes back

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
             "text": text, "detail": getattr(job, "error_detail", None),
             "ref_ids": [r for r in (job.result_ref or "").split(",") if r][:12]}
    tid = autonomy.armed_thread(ctx, job.project_id)  # an autonomous chain keeps its results in its own chat
    await _append_log(ctx, job.project_id, entry, tid)
    ctx.bus.publish(job.project_id, "chat", entry)
    if row := await autonomy.job_done(ctx, job, entry):  # autonomous mode: start covered training, wake the agent
        await _append_log(ctx, job.project_id, row, tid)
        ctx.bus.publish(job.project_id, "chat", row)


def install_job_listener(ctx: Ctx) -> None:
    import asyncio

    def on_finished(job) -> None:
        if job.kind in JOB_LABELS:
            asyncio.get_running_loop().create_task(post_job_done(ctx, job))
    ctx.jobs.on_finished.append(on_finished)
