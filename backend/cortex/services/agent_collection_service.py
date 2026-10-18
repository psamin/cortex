"""Autonomous demonstration collection.

Collectors:
  agent   a vision-language planner (our self-hosted Qwen3-VL, or GPT-6 Astra via OpenRouter)
          plans through the validated tool API; IK executes in MuJoCo
  oracle  the existing scripted IK expert (privileged state) on the same scenes

Every attempt is persisted (agent_attempts + trace/video/trajectory in storage).
Only attempts the simulator marks successful become Episodes, which carry
provenance; failed attempts are diagnostics and can never enter a dataset.
Planner spend is capped per episode, per collection, and by the project's
standing model budget; going beyond it needs a human to raise the budget.
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from ..context import Ctx
from ..errors import NotFound
from ..models import (
    AgentAttempt, AgentCollection, Condition, Episode, EpisodeSource, EpisodeStatus, Project, Scenario,
    StartAgentCollectionRequest,
)
from ..sim.so101 import REGION_XY
from .jobs import Job
from .recording_service import AUTO_ACCEPT_MIN_SCORE, dataset_blur_reference, live_key, list_episodes, quality_for

REGIONS = ["left", "center", "right"]
LIGHTS = ["normal", "dim", "bright"]
AGENT_SEED_BASE = 20_000  # disjoint from oracle demos (10_000+) and evaluation seeds (1_000+)
MAX_CUBE_Y = 0.045  # keep the cube's start clear of the box


class PlannerUnavailable(RuntimeError):
    """The requested planner model is not configured (HTTP 503, not an approval problem)."""

    code = "planner_unavailable"


class PlannerBudgetExceeded(PermissionError):
    code = "planner_budget"


class SimulationOnly(PermissionError):
    code = "hardware_requires_operator"

# 4. live replay: recent steps per collection (in-process; full traces are persisted per attempt)
LIVE: dict[str, dict[str, Any]] = {}


# ------------------------------------------------------------------ planner routing

def resolve_planner(ctx: Ctx) -> dict[str, Any] | None:
    """The configured model route (self-hosted first, else OpenRouter). Never substitutes
    another model or provider when the configured one is unavailable."""
    r = ctx.model
    if r is None:
        return None
    return {"provider": r.provider, "model": r.planner_model, "label": r.label, "api_key": r.api_key,
            "endpoint": r.chat_url, "point_format": r.point_format, "openrouter_extras": r.openrouter_extras,
            "price_per_m": r.price_per_m}


async def selfhosted_reachable(ctx: Ctx) -> str | None:
    """None if the self-hosted endpoint answers /models, else the reason it doesn't."""
    import httpx

    r = ctx.model
    if r is None or r.provider != "selfhosted":
        return None
    try:
        async with httpx.AsyncClient(timeout=4) as c:
            resp = await c.get(r.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {r.api_key}"})
        if resp.status_code >= 400:
            return f"self-hosted endpoint {r.base_url} answered {resp.status_code}"
        served = [m.get("id") for m in resp.json().get("data", [])]
        if r.planner_model not in served:
            return f"self-hosted endpoint serves {served}, not {r.planner_model}"
        return None
    except Exception as e:
        return (f"self-hosted endpoint {r.base_url} is not reachable ({type(e).__name__}); start it with "
                "scripts/vlm_futurama.sh up (needs the GT VPN)")


async def openrouter_balance_problem(ctx: Ctx) -> str | None:
    """None if the OpenRouter account behind the route has credit left, else why not."""
    import httpx

    r = ctx.model
    if r is None or r.provider not in ("openrouter", "insforge"):
        return None
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            resp = await c.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {r.api_key}"})
        if resp.status_code >= 400:
            return None  # can't tell (some keys can't read credits); let the call itself report
        d = resp.json().get("data", {})
        left = float(d.get("total_credits", 0)) - float(d.get("total_usage", 0))
        return None if left > 0.05 else f"the OpenRouter account has ${max(left, 0):.2f} of credit; add credits to use {r.planner_model}"
    except Exception:
        return None


async def planner_status(ctx: Ctx) -> dict[str, Any]:
    from . import claude_code_agent as cc

    if ctx.settings.agent_engine in ("auto", "claude_code") and cc.available() is None and await cc.mcp_up():
        return {"available": True, "model": ctx.settings.claude_code_model or "claude-code",
                "provider": "claude_code", "route": "Claude Code (your agent) via MCP", "point_format": "pixels",
                "pricing_per_m_tokens": "your Claude Code subscription (not charged by Cortex)"}
    p = resolve_planner(ctx)
    if not p:
        reason = (f"InsForge Model Gateway: {ctx.gateway_error}" if ctx.gateway_error else
                  "No model endpoint: set PLANNER_BASE_URL (self-hosted) or an InsForge/OpenRouter key.")
        return {"available": False, "model": None, "provider": None, "reason": reason}
    down = await selfhosted_reachable(ctx) or await openrouter_balance_problem(ctx)
    if down:
        return {"available": False, "model": p["model"], "provider": p["provider"], "route": p["label"],
                "reason": down}
    return {"available": True, "model": p["model"], "provider": p["provider"], "route": p["label"],
            "point_format": p["point_format"],
            "pricing_per_m_tokens": p["price_per_m"] or "self-hosted: no per-token charge"}


# ------------------------------------------------------------------ scenarios

def plan_scenarios(project: Project, req: StartAgentCollectionRequest, base_seed: int) -> list[Scenario]:
    """Deterministic in (request, base_seed): matched scenes for A/B runs."""
    rng = random.Random(base_seed)
    if req.conditions:
        cells = list(req.conditions)
    elif req.diversity == "plan":
        cells = [c.condition for c in project.plan.cells for _ in range(max(1, c.target))]
    else:
        cells = [Condition(cube_region=r, lighting=lt) for r in REGIONS for lt in LIGHTS]  # type: ignore[arg-type]
    rng.shuffle(cells)
    span = {"plan": 0.0, "medium": 0.012, "high": 0.025}[req.diversity]
    out = []
    for i in range(req.num_demos):
        c = cells[i % len(cells)]
        dx, dy = rng.uniform(-span, span), rng.uniform(-span, span)
        dy = min(dy, MAX_CUBE_Y - REGION_XY[c.cube_region][1])
        box = (rng.uniform(-0.02, 0.02), rng.uniform(0.0, 0.02)) if req.diversity == "high" else (0.0, 0.0)
        out.append(Scenario(cube_region=c.cube_region, lighting=c.lighting, cube_offset=(round(dx, 4), round(dy, 4)),
                            box_offset=(round(box[0], 4), round(box[1], 4))))
    return out


# ------------------------------------------------------------------ budget

async def spent_usd(ctx: Ctx, project_id: str) -> float:
    rows = await ctx.repo.list("agent_attempts", AgentAttempt, project_id=project_id)
    return round(sum(float(a.usage.get("cost_usd", 0.0)) for a in rows), 4)


async def check_budget(ctx: Ctx, project: Project, req: StartAgentCollectionRequest) -> float:
    spent = await spent_usd(ctx, project.id)
    remaining = round(project.autopilot.model_budget_usd - spent, 4)
    cap = req.max_total_usd if req.max_total_usd is not None else min(remaining, req.num_demos * req.max_usd_per_episode)
    if cap <= 0 or cap > remaining + 1e-9:
        raise PlannerBudgetExceeded(f"Planner budget: ${spent:.2f} of the project's ${project.autopilot.model_budget_usd:.2f} "
                              f"standing approval is spent; this collection asks for up to ${cap:.2f}. "
                              f"A human must raise model_budget_usd in the studio.")
    return cap


# ------------------------------------------------------------------ start

async def start_collection(ctx: Ctx, project: Project, req: StartAgentCollectionRequest,
                           actor: str = "gui") -> AgentCollection:
    if project.mode != "sim":
        raise SimulationOnly("Autonomous collection runs in simulation only; physical robots need the local "
                              "hardware adapter with an operator present.")
    if req.planner == "auto":  # the project's planner choice from the studio's setup
        req = req.model_copy(update={"planner": project.infra.planner})
    # the scripted oracle always reads simulator object poses, so it is privileged by construction
    mode = "privileged" if req.collector == "oracle" else (req.mode or project.autopilot.planning_mode)
    if req.collector == "agent" and await use_claude_code(ctx, req):
        return await _start_claude_code_collection(ctx, project, req, actor)
    planner = resolve_planner(ctx) if req.collector == "agent" else None
    if req.collector == "agent" and planner is None:
        raise PlannerUnavailable("No agent planner is configured (self-hosted endpoint or InsForge/OpenRouter key). "
                                 "Cortex does not substitute another model; use collector='oracle' for "
                                 "scripted-oracle demos.")
    if req.collector == "agent" and (down := await selfhosted_reachable(ctx) or await openrouter_balance_problem(ctx)):
        raise PlannerUnavailable(down)
    paid = bool(planner and planner["price_per_m"])
    cap = await check_budget(ctx, project, req) if req.collector == "agent" and paid else 0.0
    prior = await ctx.repo.list("agent_attempts", AgentAttempt, project_id=project.id)
    base_seed = req.base_seed if req.base_seed is not None else AGENT_SEED_BASE + len(prior)
    scenarios = plan_scenarios(project, req, base_seed)
    col = AgentCollection(project_id=project.id, collector=req.collector, request=req, mode=mode,
                          provider=planner["label"] if planner else None,
                          model=planner["model"] if planner else "scripted_ik_oracle",
                          scenarios=[{"seed": base_seed + i, **s.model_dump()} for i, s in enumerate(scenarios)],
                          reason=req.reason, totals=_empty_totals(cap))
    attempts = [AgentAttempt(project_id=project.id, collection_id=col.id, index=i, collector=req.collector,
                             seed=base_seed + i, scenario=s, status="failed", mode=mode,
                             provider=col.provider, model=col.model) for i, s in enumerate(scenarios)]
    await ctx.repo.save("agent_collections", col)
    await ctx.audit(actor, "start_agent_collection", project.id,
                    {**req.model_dump(mode="json"), "budget_cap_usd": cap, "provider": col.provider,
                     "model": col.model, "mode": mode}, "ok", f"{col.id}: {req.num_demos} {req.collector} attempts")

    async def run(job: Job) -> str:
        col.status, col.job_id = "running", job.id
        await _save(ctx, col)
        blur_ref = await dataset_blur_reference(ctx, project.id)
        live_dir = ctx.local(live_key(project.id))

        async def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type")
            if t == "episode_start":  # oracle worker: same shape as the agent's attempt_start
                a = attempts[ev["index"]]
                ev = {"type": "attempt_start", "index": ev["index"], "total": ev["total"], "seed": a.seed,
                      "scenario": a.scenario.model_dump()}
                t = "attempt_start"
            if t in ("attempt_start", "agent_step", "agent_thought"):
                live = LIVE.setdefault(col.id, {"index": None, "start": None, "steps": []})
                if t == "attempt_start":
                    live.update(index=ev["index"], start=ev, steps=[])
                else:
                    live["steps"] = [*live["steps"][-200:], ev]
                ctx.bus.publish(project.id, "agent_step", {"collection_id": col.id, **ev})
                if t == "attempt_start":
                    ctx.jobs.update(job, message=f"{req.collector} attempt {ev['index'] + 1}/{len(attempts)}",
                                    progress=ev["index"] / len(attempts))
            elif t in ("attempt", "episode"):
                await _record_attempt(ctx, project, col, attempts[ev["index"]], ev, blur_ref)
                ctx.jobs.update(job, progress=(ev["index"] + 1) / len(attempts))
            elif t == "budget_exhausted":
                col.status = "budget_exhausted"
                col.totals["stopped_reason"] = f"collection budget reached after ${ev['spent_usd']:.2f}"

        if req.collector == "agent":
            p = planner or {}
            from .intake_service import scene_payload

            scene = scene_payload(ctx, project)
            cfg = {"task": req.task or (scene or {}).get("task") or project.spec.goal, "mode": mode,
                   "live_dir": str(live_dir), "scene": scene,
                   "max_total_usd": cap if paid else 1e9,
                   "planner": {"provider": p["provider"], "model": p["model"], "endpoint": p["endpoint"],
                               "point_format": p["point_format"], "openrouter_extras": p["openrouter_extras"],
                               "price_per_m": p["price_per_m"],
                               "max_calls": req.max_calls_per_episode, "max_usd": req.max_usd_per_episode,
                               "reasoning_effort": "low"},
                   "attempts": [{"seed": a.seed, "scenario": a.scenario.model_dump(),
                                 "out_dir": str(ctx.local(attempt_key(a)))} for a in attempts]}
            await ctx.jobs.run_worker(job, "cortex.workers.agent_worker", cfg,
                                      ctx.local(f"projects/{project.id}/jobs/{job.id}.json"), on_event,
                                      env={"CORTEX_PLANNER_API_KEY": planner["api_key"]})  # type: ignore[index]
        else:
            from .intake_service import scene_payload

            cfg = {"mode": "record", "max_steps": 240, "live_dir": str(live_dir), "scene": scene_payload(ctx, project),
                   "episodes": [{"seed": a.seed, "condition": a.scenario.condition.model_dump(),
                                 "scenario": a.scenario.model_dump(), "out_dir": str(ctx.local(attempt_key(a)))}
                                for a in attempts]}
            await ctx.jobs.run_worker(job, "cortex.workers.sim_worker", cfg,
                                      ctx.local(f"projects/{project.id}/jobs/{job.id}.json"), on_event)
        if col.status == "running":
            col.status = "succeeded"
        await _save(ctx, col)
        await ctx.audit(ACTOR_SYSTEM, "agent_collection_finished", project.id, {"collection_id": col.id}, "ok",
                        _summary(col))
        return col.id

    job = ctx.jobs.submit("agent_collect", project.id, run, idempotency_key=f"collect:{col.id}")
    col.job_id = job.id
    await _save(ctx, col)
    return col


ACTOR_SYSTEM = "system"


async def use_claude_code(ctx: Ctx, req: StartAgentCollectionRequest) -> bool:
    from . import claude_code_agent as cc

    if req.planner == "model":
        return False
    ok = cc.available() is None and await cc.mcp_up()
    if req.planner == "claude_code" and not ok:
        raise PlannerUnavailable("Claude Code planner requested but the `claude` CLI or the Cortex MCP server "
                                 "(http://localhost:3000/mcp) is not available.")
    return ok and ctx.settings.agent_engine in ("auto", "claude_code")


async def _start_claude_code_collection(ctx: Ctx, project: Project, req: StartAgentCollectionRequest,
                                        actor: str) -> AgentCollection:
    """The user's Claude Code plans every attempt (one headless run per attempt)."""
    from . import claude_code_collector as ccc

    prior = await ctx.repo.list("agent_attempts", AgentAttempt, project_id=project.id)
    base_seed = req.base_seed if req.base_seed is not None else AGENT_SEED_BASE + len(prior)
    scenarios = plan_scenarios(project, req, base_seed)
    model = ctx.settings.claude_code_model or "claude-code"
    col = AgentCollection(project_id=project.id, collector="agent", request=req, mode="camera",
                          provider="Claude Code (your agent) via MCP", model=model,
                          scenarios=[{"seed": base_seed + i, **s.model_dump()} for i, s in enumerate(scenarios)],
                          reason=req.reason or "agent collection", totals=_empty_totals(0))
    await ctx.repo.save("agent_collections", col)
    await ctx.audit(actor, "start_agent_collection", project.id,
                    {**req.model_dump(mode="json"), "planner": "claude_code"}, "ok",
                    f"{col.id}: {req.num_demos} attempts planned by Claude Code")

    async def run(job: Job) -> str:
        col.status, col.job_id = "running", job.id
        await _save(ctx, col)
        try:
            for i, sc in enumerate(scenarios):
                ctx.jobs.update(job, message=f"Claude Code attempt {i + 1}/{len(scenarios)}",
                                progress=i / len(scenarios))
                out = await ccc.run_attempt(ctx, project, col, sc, base_seed + i, i, ctx.settings.claude_code_model)
                fresh = await get_collection(ctx, col.id)  # totals were updated by the session finish
                fresh.totals["model_calls"] = fresh.totals.get("model_calls", 0) + int(out.get("turns") or 0)
                col.totals = fresh.totals
                await _save(ctx, col)
        except BaseException as e:  # stopped by the user or an attempt crashed: never leave it "running"
            col.status = "cancelled" if isinstance(e, asyncio.CancelledError) else "failed"
            col.error = None if col.status == "cancelled" else f"{type(e).__name__}: {e}"
            await _save(ctx, col)
            raise
        col.status = "succeeded"
        await _save(ctx, col)
        await ctx.audit(ACTOR_SYSTEM, "agent_collection_finished", project.id, {"collection_id": col.id}, "ok",
                        _summary(col))
        return col.id

    job = ctx.jobs.submit("agent_collect", project.id, run, idempotency_key=f"collect:{col.id}")
    col.job_id = job.id
    await _save(ctx, col)
    return col


def attempt_key(a: AgentAttempt, name: str = "") -> str:
    return f"projects/{a.project_id}/attempts/{a.id}/{name}".rstrip("/")


def _empty_totals(cap: float) -> dict[str, Any]:
    return {"attempts": 0, "successes": 0, "failures": 0, "model_calls": 0, "cost_usd": 0.0, "recoveries": 0,
            "tool_errors": 0, "sim_seconds": 0.0, "wall_s": 0.0, "cells": {}, "budget_cap_usd": cap}


def _summary(col: AgentCollection) -> str:
    t = col.totals
    return (f"{col.id}: {t['successes']}/{t['attempts']} successful {col.collector} demos, "
            f"{t['model_calls']} model calls, ${t['cost_usd']:.2f}, {t['recoveries']} recoveries")


async def _record_attempt(ctx: Ctx, project: Project, col: AgentCollection, a: AgentAttempt, ev: dict[str, Any],
                          blur_ref: float | None) -> None:
    success = bool(ev.get("success"))
    a.status = "success" if success else "failed"
    a.usage = ev.get("usage") or {}
    a.recoveries = int(ev.get("recoveries") or 0)
    a.counters = ev.get("counters") or {}
    a.num_frames = int(ev.get("num_frames") or 0)
    a.wall_s = float(ev.get("wall_s") or ev.get("wall_s", 0.0))
    a.model_claimed_success = ev.get("model_claimed_success")
    a.stop_reason = ev.get("stop_reason")
    a.error = ev.get("error")
    a.trace_len = int(ev.get("trace_len") or 0)
    a.endpoint = ev.get("endpoint")
    if a.collector == "oracle":
        (ctx.local(attempt_key(a)) / "trace.json").write_text(json.dumps(
            {"planner": "scripted_ik_oracle", "seed": a.seed, "scenario": a.scenario.model_dump(), "trace": [],
             "note": "the scripted oracle has no tool trace; it follows fixed IK waypoints from simulator state"}))
    for kind, name in (("episode", "episode.npz"), ("front", "front.mp4"), ("thumb", "thumb.jpg"),
                       ("trace", "trace.json")):
        if ctx.local(attempt_key(a, name)).exists():
            a.artifacts[kind] = attempt_key(a, name)
            await ctx.publish_file(a.artifacts[kind])
    if success and "episode" in a.artifacts and a.num_frames:
        eps = await list_episodes(ctx, project.id)
        ep = Episode(project_id=project.id, index=len(eps), condition=a.scenario.condition, seed=a.seed,
                     source=EpisodeSource.sim_agent if a.collector == "agent" else EpisodeSource.sim_scripted,
                     num_frames=a.num_frames, fps=30, success=True,
                     artifacts={k: v for k, v in a.artifacts.items() if k in ("episode", "front", "thumb")},
                     provenance={"planner": a.model if a.collector == "agent" else "scripted_ik_oracle",
                                 "provider": a.provider, "model": a.model, "mode": a.mode,
                                 "attempt_id": a.id, "collection_id": col.id, "scenario": a.scenario.model_dump()})
        ep.quality = quality_for(ctx, ep, blur_ref)
        if ep.quality.score >= AUTO_ACCEPT_MIN_SCORE:
            ep.status = EpisodeStatus.accepted
        await ctx.repo.save("episodes", ep)
        a.episode_id = ep.id
        ctx.changed(project.id, "episodes", ep.id)
    await ctx.repo.save("agent_attempts", a)
    t = col.totals
    t["attempts"] += 1
    t["successes" if success else "failures"] += 1
    t["model_calls"] += int(a.usage.get("calls", 0))
    t["cost_usd"] = round(t["cost_usd"] + float(a.usage.get("cost_usd", 0.0)), 4)
    t["recoveries"] += a.recoveries
    t["tool_errors"] += int(a.counters.get("tool_errors", 0))
    t["sim_seconds"] = round(t["sim_seconds"] + a.num_frames / 30, 2)
    t["wall_s"] = round(t["wall_s"] + a.wall_s, 2)
    cell = t["cells"].setdefault(a.scenario.condition.key, {"attempts": 0, "successes": 0})
    cell["attempts"] += 1
    cell["successes"] += int(success)
    await _save(ctx, col)
    ctx.bus.publish(project.id, "agent_attempt", {"collection_id": col.id, **a.model_dump(mode="json")})


async def _save(ctx: Ctx, col: AgentCollection) -> None:
    await ctx.repo.save("agent_collections", col)
    ctx.changed(col.project_id, "agent_collections", col.id)


async def list_collections(ctx: Ctx, project_id: str) -> list[AgentCollection]:
    return await ctx.repo.list("agent_collections", AgentCollection, project_id=project_id)


async def get_collection(ctx: Ctx, collection_id: str) -> AgentCollection:
    c = await ctx.repo.get("agent_collections", collection_id, AgentCollection)
    if c is None:
        raise NotFound(collection_id)
    return c


async def list_attempts(ctx: Ctx, project_id: str, collection_id: str | None = None) -> list[AgentAttempt]:
    rows = await ctx.repo.list("agent_attempts", AgentAttempt, project_id=project_id)
    return [a for a in rows if collection_id is None or a.collection_id == collection_id]


async def attempt_trace(ctx: Ctx, attempt_id: str) -> dict[str, Any]:
    a = await ctx.repo.get("agent_attempts", attempt_id, AgentAttempt)
    if a is None:
        raise NotFound(attempt_id)
    trace = {}
    if "trace" in a.artifacts:
        trace = json.loads((await ctx.ensure_local(a.artifacts["trace"])).read_text())
    return {"attempt": a, "trace": trace.get("trace", []), "model_summary": trace.get("model_summary"),
            "media": {k: await ctx.media_url(v) for k, v in a.artifacts.items() if k in ("front", "thumb")}}


def compare_collectors(attempts: list[AgentAttempt]) -> dict[str, Any]:
    """Per-collector metrics over matched scenes (no claims beyond the counts)."""
    out: dict[str, Any] = {}
    for coll in ("agent", "oracle"):
        rows = [a for a in attempts if a.collector == coll]
        if not rows:
            continue
        n = len(rows)
        ok = sum(a.status == "success" for a in rows)
        out[coll] = {"attempts": n, "successes": ok, "success_rate": round(ok / n, 3),
                     "recoveries": sum(a.recoveries for a in rows),
                     "mean_sim_seconds": round(sum(a.num_frames for a in rows) / n / 30, 2),
                     "mean_wall_s": round(sum(a.wall_s for a in rows) / n, 2),
                     "model_calls": sum(int(a.usage.get("calls", 0)) for a in rows),
                     "cost_usd": round(sum(float(a.usage.get("cost_usd", 0.0)) for a in rows), 4),
                     "cells_covered": len({a.scenario.condition.key for a in rows if a.status == "success"})}
    return out
