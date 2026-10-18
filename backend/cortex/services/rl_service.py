"""RL sandboxes as a service (#80): weaknesses -> proposals with measured pilots -> accepted runs -> results.

Each sandbox runs ``cortex.workers.rl_worker`` as a background job (first ``pilot``, then ``run`` once accepted)
and is persisted in ``rl_sandboxes``. Worker events are relayed on the project bus as ``rl`` events (sim_stats
throttled) so the studio can draw a live box per sandbox. Local CPU sandboxes cost $0 and an agent may accept
them within the project's standing budget; GPU or paid sandboxes need a human approval.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..context import Ctx
from ..models import FailureCluster, Project, new_id, now
from ..rl import Pilot, RLSandbox, SandboxBudget, estimate, find_weaknesses, propose
from ..rl.types import IterationStats, SandboxResult, Weakness
from . import simulation_runner
from .training_service import ApprovalRequired

LIVE_STATS_S = 0.25  # at most ~4 sim_stats events per second per sandbox on the bus
ACTIVE = ("piloting", "accepted", "running")
_slots: dict[str, asyncio.Semaphore] = {}
_live: dict[str, RLSandbox] = {}  # sandboxes with a job in flight: the job's copy is the source of truth


class RLBudgetExceeded(PermissionError):
    code = "rl_budget"


class RLRound(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rlr"))
    project_id: str
    sandbox_id: str
    weakness_key: str
    condition: str
    skill_gain: float  # measured: tuned - base skill success on matched held-out seeds
    predicted_policy_uplift: float | None = None  # the estimate at acceptance
    realised_uplift: float | None = None  # measured once the retrained ACT is re-evaluated (closing the loop)
    status: str = "skill_measured"
    created_at: datetime = Field(default_factory=now)


# ------------------------------------------------------------------ reads
async def weaknesses(ctx: Ctx, project: Project, checkpoint_id: str | None = None,
                     target: float = 0.8) -> list[Weakness]:
    evals = await simulation_runner.list_evaluations(ctx, project.id)
    clusters = await ctx.repo.list("failure_clusters", FailureCluster, project_id=project.id)
    return find_weaknesses(evals, clusters, target=target, checkpoint_id=checkpoint_id)


async def get(ctx: Ctx, sandbox_id: str) -> RLSandbox:
    sb = _live.get(sandbox_id) or await ctx.repo.get("rl_sandboxes", sandbox_id, RLSandbox)
    if sb is None:
        raise KeyError(f"RL sandbox {sandbox_id}")
    return sb


async def list_sandboxes(ctx: Ctx, project_id: str) -> list[RLSandbox]:
    return [_live.get(sb.id, sb) for sb in await ctx.repo.list("rl_sandboxes", RLSandbox, project_id=project_id)]


async def history(ctx: Ctx, project_id: str) -> list[tuple[float, float]]:
    rounds = await ctx.repo.list("rl_rounds", RLRound, project_id=project_id)
    return [(r.skill_gain, r.realised_uplift) for r in rounds if r.realised_uplift is not None]


async def _save(ctx: Ctx, sb: RLSandbox) -> None:
    await ctx.repo.save("rl_sandboxes", sb)
    ctx.changed(sb.project_id, "rl_sandboxes", sb.id)


def _scene(ctx: Ctx, project: Project) -> dict[str, Any] | None:
    from . import intake_service

    payload = getattr(intake_service, "scene_payload", None)  # #99; until then sandboxes use the tabletop
    return payload(ctx, project) if payload else None


# ------------------------------------------------------------------ worker jobs
def _relay(ctx: Ctx, sb: RLSandbox, mode: str):
    last = [0.0]

    async def on_event(ev: dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "sim_stats":
            if time.monotonic() - last[0] < LIVE_STATS_S:
                return
            last[0] = time.monotonic()
        if t != "result":
            ctx.bus.publish(sb.project_id, "rl", {"sandbox_id": sb.id, **ev})
        if t == "rl_phase":
            sb.phase = ev["phase"]
            await _save(ctx, sb)
        elif t == "rl_episode":
            sb.episodes_done += 1
        elif t == "rl_iteration" and mode == "run":  # the pilot's short CEM is reported in its estimate only
            st = IterationStats.model_validate(ev)
            sb.progress.append(st)
            if st.env is not None:
                sb.env = st.env
            await _save(ctx, sb)
    return on_event


def _spawn(ctx: Ctx, sb: RLSandbox, mode: str, scene: dict[str, Any] | None, on_done) -> str:
    base = ctx.local(f"projects/{sb.project_id}/rl/{sb.id}")
    cfg = {"mode": mode, "sandbox": sb.model_dump(mode="json"), "out_dir": str(base / mode),
           "live_dir": str(base / "live"), "stop_file": str(base / "STOP"), "seed_base": 1, "scene": scene}
    slots = _slots.setdefault(sb.project_id, asyncio.Semaphore(sb.budget.max_parallel))

    async def run(job) -> str:
        holder: dict[str, Any] = {}
        relay = _relay(ctx, sb, mode)

        async def on_event(ev):
            if ev.get("type") in ("rl_pilot", "rl_result"):
                holder.update(ev)
            await relay(ev)
        async with slots:
            sb.started_at = sb.started_at or now()
            try:
                await ctx.jobs.run_worker(job, "cortex.workers.rl_worker", cfg, base / f"{mode}.json", on_event)
                await on_done(holder)
            except asyncio.CancelledError:
                sb.status, sb.reason = "stopped", f"stopped by {sb.stop_requested_by or 'user'} during the {mode}"
                await _save(ctx, sb)
                raise
            except Exception as e:
                sb.status, sb.reason = "failed", f"{mode} failed: {e}"[:500]
                await _save(ctx, sb)
                raise
            finally:
                _live.pop(sb.id, None)
        return sb.id

    _live[sb.id] = sb
    job = ctx.jobs.submit(f"rl_{mode}", sb.project_id, run, idempotency_key=f"rl:{mode}:{sb.id}")
    return job.id


# ------------------------------------------------------------------ propose / accept / stop
async def propose_sandboxes(ctx: Ctx, project: Project, budget: SandboxBudget | None = None,
                            checkpoint_id: str | None = None, max_sandboxes: int = 3,
                            actor: str = "gui") -> list[RLSandbox]:
    """One sandbox per top weakness; each runs a measured pilot (base skill + short CEM) as a job."""
    scene = _scene(ctx, project)
    ws = await weaknesses(ctx, project, checkpoint_id)
    if not ws:
        return []
    digest = (scene or {}).get("info", {}).get("digest")
    hist = await history(ctx, project.id)
    out = propose(project.id, ws, budget, scene_digest=digest, max_sandboxes=max_sandboxes)
    for sb in out:
        sb.status, sb.phase = "piloting", "pilot"
        await _save(ctx, sb)

        async def done(ev, sb=sb):
            sb.estimate = estimate(sb, Pilot.model_validate(ev["pilot"]), ev["sim_steps_per_s"], hist,
                                   steps_per_episode=ev.get("steps_per_episode", 300),
                                   record_steps_per_s=ev.get("record_steps_per_s"))
            sb.status, sb.phase, sb.episodes_done = "ready", None, 0
            await _save(ctx, sb)
        sb.job_id = _spawn(ctx, sb, "pilot", scene, done)
        await _save(ctx, sb)
    await ctx.audit(actor, "propose_rl_sandboxes", project.id, {"checkpoint_id": checkpoint_id}, "ok",
                    f"{len(out)} sandboxes piloting: {', '.join(s.weakness.key for s in out)}")
    return out


async def _remaining_usd(ctx: Ctx, project: Project) -> float:
    from .agent_collection_service import spent_usd

    return round(project.autopilot.model_budget_usd - await spent_usd(ctx, project.id), 4)


async def accept(ctx: Ctx, project: Project, ids: list[str], budget_cap_usd: float = 0.0,
                 max_cpu_hours: float | None = None, actor: str = "gui") -> list[RLSandbox]:
    sbs = [await get(ctx, i) for i in ids]
    for sb in sbs:
        if sb.project_id != project.id:
            raise KeyError(f"RL sandbox {sb.id} in project {project.id}")
        if sb.status != "ready":
            raise ValueError(f"{sb.id} is '{sb.status}'; only sandboxes with a finished pilot can be accepted")
        if sb.budget.executor != "local_cpu":
            if actor != "gui":
                raise ApprovalRequired(f"{sb.id} runs on {sb.budget.executor}: GPU or paid sandboxes need a human "
                                       "approval in the Cortex studio")
            raise ValueError("GPU sandboxes (residual_rl on Futurama) aren't built yet; use local_cpu")
    cost = [sb.estimate.cost for sb in sbs]
    usd = sum(c.usd for c in cost if c)
    cpu_h = sum(c.cpu_hours for c in cost if c)
    remaining = await _remaining_usd(ctx, project)
    if usd > budget_cap_usd + 1e-9 or usd > remaining + 1e-9:
        raise RLBudgetExceeded(f"these sandboxes cost ${usd:.2f}; the cap is ${budget_cap_usd:.2f} and the project's "
                               f"standing budget has ${remaining:.2f} left. A human must approve more.")
    if max_cpu_hours is not None and cpu_h > max_cpu_hours + 1e-9:
        raise RLBudgetExceeded(f"these sandboxes need ~{cpu_h:.2f} CPU hours (estimate); the cap is {max_cpu_hours}")
    scene = _scene(ctx, project)
    for sb in sbs:
        sb.status, sb.accepted_by, sb.started_at = "accepted", actor, None

        async def done(ev, sb=sb):
            res = SandboxResult.model_validate(ev["result"])
            sb.result, sb.finished_at, sb.phase = res, now(), None
            stopped = res.stop_reason == "stopped"
            sb.status = "stopped" if stopped else "succeeded"
            sb.reason = (f"stopped by {sb.stop_requested_by or 'user'}; evaluated what it had" if stopped
                         else f"CEM stopped on '{res.stop_reason}'")
            await _save(ctx, sb)
            up = sb.estimate.predicted_policy_uplift
            rnd = RLRound(project_id=sb.project_id, sandbox_id=sb.id, weakness_key=sb.weakness.key,
                          condition=sb.weakness.condition.key, skill_gain=round(res.tuned_success - res.base_success, 4),
                          predicted_policy_uplift=up.mean if up else None)
            await ctx.repo.save("rl_rounds", rnd)
        sb.job_id = _spawn(ctx, sb, "run", scene, done)
        sb.status = "running"
        await _save(ctx, sb)
    await ctx.audit(actor, "accept_rl_sandboxes", project.id, {"ids": ids, "budget_cap_usd": budget_cap_usd}, "ok",
                    f"{len(sbs)} sandboxes running locally (${usd:.2f}, ~{cpu_h:.2f} CPU h estimated)")
    return sbs


async def stop(ctx: Ctx, sandbox_id: str, actor: str = "gui") -> RLSandbox:
    sb = await get(ctx, sandbox_id)
    sb.stop_requested_by = actor
    if sb.status in ("proposed", "ready"):
        sb.status, sb.reason = "dismissed", f"dismissed by {actor}"
    elif sb.status == "piloting" and sb.job_id:
        await _save(ctx, sb)
        await ctx.jobs.cancel(sb.job_id)  # the job marks it stopped
        return await get(ctx, sandbox_id)
    elif sb.status in ("accepted", "running"):  # graceful: the worker evaluates what it has, then reports
        stop_file = ctx.local(f"projects/{sb.project_id}/rl/{sb.id}/STOP")
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text(actor)
        sb.reason = f"stop requested by {actor}"
    await _save(ctx, sb)
    await ctx.audit(actor, "stop_rl_sandbox", sb.project_id, {"id": sandbox_id}, "ok", sb.reason)
    return sb


# ------------------------------------------------------------------ views (what the agent and the studio see)
def proposal_view(sb: RLSandbox) -> dict[str, Any]:
    e = sb.estimate
    c = e.cost
    return {
        "id": sb.id, "status": sb.status, "weakness_id": sb.weakness.key, "branch_id": sb.weakness.branch_id,
        "condition": sb.weakness.condition.key, "policy_success": sb.weakness.success_rate,
        "dominant_failure": sb.weakness.dominant_failure, "method": sb.method, "params": [p.name for p in sb.params],
        "reward_weights": sb.reward_weights, "kind": e.kind,
        "pilot": e.pilot.model_dump(include={"n", "successes", "wilson_95", "cem_iterations", "cem_best_successes",
                                             "cem_best_n"}) if e.pilot else None,
        "predicted_sandbox_success": e.predicted_sandbox_success and e.predicted_sandbox_success.model_dump(),
        "predicted_policy_uplift": e.predicted_policy_uplift and e.predicted_policy_uplift.model_dump(),
        "predicted_regression_elsewhere": e.predicted_regression_elsewhere
        and e.predicted_regression_elsewhere.model_dump(),
        "cost": c and {"episodes": c.episodes, "cpu_hours": c.cpu_hours, "gpu_hours": c.gpu_hours,
                       "wall_clock_est_min": c.wall_clock_est_min, "usd": c.usd, "sim_steps_per_s": c.sim_steps_per_s},
        "uplift_per_hour": e.uplift_per_hour, "p_improve_ge_delta": e.p_improve_ge_delta, "delta": e.delta,
        "demos_expected": e.demos_expected, "executor": sb.budget.executor, "assumptions": e.assumptions,
        "reason": sb.reason, "job_id": sb.job_id,
    }


def status_view(sb: RLSandbox) -> dict[str, Any]:
    last = sb.progress[-1] if sb.progress else None
    end = sb.finished_at or now()
    elapsed = (end - sb.started_at).total_seconds() if sb.started_at else 0.0
    return {
        "id": sb.id, "status": sb.status, "phase": sb.phase, "reason": sb.reason, "weakness_id": sb.weakness.key,
        "method": sb.method, "iteration": len(sb.progress), "max_iterations": sb.budget.iterations,
        "episodes": sb.episodes_done,
        "success": last and {"rate": last.success_rate, "ci95": last.ci95, "elite_rate": last.elite_success_rate},
        "mean_return": last and last.mean_return, "elite_params": last and last.best_params,
        "curve": [{"iteration": s.iteration, "success_rate": s.success_rate, "ci95": s.ci95,
                   "elite_success_rate": s.elite_success_rate, "mean_return": s.mean_return} for s in sb.progress],
        "current_ranges": sb.env.model_dump(include={"cube_offset", "offset_jitter_m", "yaw_jitter_rad",
                                                     "light_scales", "friction_jitter", "mass_jitter", "widened"}),
        "elapsed_s": round(elapsed, 1),
        "cost_so_far": {"cpu_hours": round(elapsed / 3600, 4), "usd": 0.0, "measured": True},
        "estimate": proposal_view(sb), "result": sb.result and sb.result.model_dump(mode="json"),
        "live_frame": f"/api/rl/sandboxes/{sb.id}/live.jpg", "job_id": sb.job_id,
    }
