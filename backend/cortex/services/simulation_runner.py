"""Parallel simulation evaluation across condition branches.

Each branch is an independent, reproducible condition (variation + explicit
seed list + pinned env revision). Branches run concurrently as separate local
worker processes (one MuJoCo/GL context each).
"""

from __future__ import annotations

import logging

import asyncio
import json
import math
from typing import Any

from ..context import Ctx
from ..errors import NotFound
from ..models import (
    BranchResult,
    BranchSpec,
    Checkpoint,
    Condition,
    EpisodeResult,
    EvaluationRun,
    JobStatus,
    LaunchEvaluationRequest,
    Project,
    SimulationBranch,
)
from ..sim.so101 import ENVHUB_REVISION, ENVHUB_URI
from .jobs import TERMINAL, Job

DEFAULT_BRANCHES = [
    BranchSpec(name="baseline", condition=Condition(cube_region="center", lighting="normal")),
    BranchSpec(name="shifted_object", condition=Condition(cube_region="center", lighting="normal"),
               cube_offset=(0.04, 0.06)),
    BranchSpec(name="changed_lighting", condition=Condition(cube_region="center", lighting="dim")),
]


log = logging.getLogger(__name__)
INTERRUPTED = "interrupted by a backend restart"
START_TIMEOUT_S = 120.0  # launch -> each branch on its way to its worker; past it the branch never will (#125)
_WATCHDOGS: set[asyncio.Task] = set()  # running start watchdogs (the loop only keeps weak references)


class BranchNeverStarted(RuntimeError):
    """A branch that did not get to its worker within START_TIMEOUT_S. The message is written for people."""

    user_facing = True


def never_started() -> str:
    return (f"Never started: no simulation worker was launched within {START_TIMEOUT_S:.0f} s, so Cortex stopped "
            "waiting for it. Run the evaluation again.")


async def sweep_interrupted(ctx: Ctx) -> int:
    """At startup: no evaluation job survives a restart, so any run or branch still queued/running was cut
    off. Mark them failed instead of leaving them "running" forever (#125). Returns how many runs."""
    n = 0
    for status in (JobStatus.queued, JobStatus.running):
        for b in await ctx.repo.list("simulation_branches", SimulationBranch, status=status):
            b.status, b.error = JobStatus.failed, INTERRUPTED
            await _save_or_skip(ctx, "simulation_branches", b, project_id=b.project_id)
        for run in await ctx.repo.list("evaluation_runs", EvaluationRun, status=status):
            run.status = JobStatus.failed
            if await _save_or_skip(ctx, "evaluation_runs", run):
                n += 1
    return n


async def _save_or_skip(ctx: Ctx, table: str, row, **kw) -> bool:
    """One unsavable row (e.g. an orphan whose project is gone) must not stop the backend from starting."""
    try:
        await ctx.repo.save(table, row, **kw)
        return True
    except Exception as e:  # noqa: BLE001 - logged and skipped by design
        log.warning("startup sweep skipped %s %s: %s", table, getattr(row, "id", "?"), e)
        return False


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(episodes: list[EpisodeResult], policy_name: str, wall_s: float) -> BranchResult:
    k = sum(e.success for e in episodes)
    n = len(episodes)
    return BranchResult(policy_name=policy_name, wall_s=wall_s, n=n, successes=k,
                        success_rate=k / n if n else 0.0, ci95=wilson(k, n),
                        mean_reward=sum(e.reward for e in episodes) / n if n else 0.0, episodes=episodes)


def branch_dir(project_id: str, eval_id: str, branch_id: str) -> str:
    return f"projects/{project_id}/evaluations/{eval_id}/{branch_id}"


async def resolve_policy(ctx: Ctx, req: LaunchEvaluationRequest) -> tuple[dict[str, Any], str]:
    if req.policy.kind == "scripted_expert":
        return {"kind": "scripted_expert"}, "scripted expert (oracle reference, privileged state)"
    ck = await ctx.repo.get("checkpoints", req.policy.checkpoint_id or "", Checkpoint)
    if ck is None:
        raise NotFound(f"checkpoint {req.policy.checkpoint_id} not found")
    from .training_service import local_checkpoint
    path = await local_checkpoint(ctx, ck)
    # #region agent log
    from ..context import agent_debug
    agent_debug("C", "simulation_runner.py:resolve_policy", "resolve checkpoint path", {
        "id": ck.id, "stored_path": ck.path, "resolved": str(path),
        "exists": (path / "config.json").exists(), "store": ctx.store.name,
        "inside_data_dir": path.resolve().is_relative_to(ctx.data_dir.resolve()) if path.exists() else None,
    })
    # #endregion
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"checkpoint files missing at {path}")
    return {"kind": "checkpoint", "checkpoint_path": str(path)}, f"{ck.policy_type} checkpoint {ck.id} (step {ck.step})"


async def launch_evaluation(ctx: Ctx, project: Project, req: LaunchEvaluationRequest,
                            actor: str = "gui") -> EvaluationRun:
    from .intake_service import scene_payload

    policy_cfg, label = await resolve_policy(ctx, req)
    scene = scene_payload(ctx, project)  # evaluate in the project's intake scene when it has one
    executor = "local"
    specs = req.branches or DEFAULT_BRANCHES
    run = EvaluationRun(project_id=project.id, policy=req.policy, policy_label=label, executor=executor,
                        name=req.name or f"{label.split(' ')[0]} × {len(specs)} conditions",
                        recommendation_id=req.recommendation_id, status=JobStatus.running)
    branches = [SimulationBranch(evaluation_id=run.id, project_id=project.id, spec=s, executor=executor,
                                 seeds=[req.base_seed + i for i in range(req.episodes_per_branch)])
                for s in specs]
    run.branch_ids = [b.id for b in branches]
    await ctx.repo.save("evaluation_runs", run)
    for b in branches:
        await ctx.repo.save("simulation_branches", b, project_id=project.id)
    ctx.changed(project.id, "evaluation_runs", run.id)
    await ctx.audit(actor, "launch_simulation_evaluation", project.id,
                    {"policy": req.policy.model_dump(), "branches": [s.name for s in specs],
                     "episodes_per_branch": req.episodes_per_branch, "base_seed": req.base_seed,
                     "executor": executor, "env": ENVHUB_URI}, "ok", f"evaluation {run.id} started")
    began = asyncio.Event()  # the job got to its branches; the start watchdog waits for this

    async def run_all(job: Job) -> str:
        began.set()
        try:
            results = await asyncio.gather(*(run_branch(ctx, job, run, b, policy_cfg, scene) for b in branches),
                                           return_exceptions=True)
        except asyncio.CancelledError:  # stopped, or the backend is shutting down: never leave it "running"
            run.status = JobStatus.failed if ctx.jobs.closing else JobStatus.cancelled
            for b in branches:  # a branch cancelled before it started never reached its own handler
                if b.status in (JobStatus.queued, JobStatus.running):
                    b.status, b.error = run.status, INTERRUPTED if ctx.jobs.closing else "stopped by user"
                    await ctx.repo.save("simulation_branches", b, project_id=project.id)
            await ctx.repo.save("evaluation_runs", run)
            ctx.changed(project.id, "evaluation_runs", run.id)
            raise
        failed = [r for r in results if isinstance(r, BaseException)]
        if ctx.jobs.stop_requested(job.id) and any(b.status == JobStatus.cancelled for b in branches):
            run.status = JobStatus.cancelled
        else:
            run.status = JobStatus.failed if len(failed) == len(branches) else JobStatus.succeeded
        await ctx.repo.save("evaluation_runs", run)
        ctx.changed(project.id, "evaluation_runs", run.id)
        if failed and len(failed) == len(branches):
            raise failed[0]
        return run.id

    job = ctx.jobs.submit("evaluate", project.id, run_all, idempotency_key=f"eval:{run.id}")
    watchdog = asyncio.create_task(watch_start(ctx, job, run, branches, began))
    _WATCHDOGS.add(watchdog)
    watchdog.add_done_callback(_WATCHDOGS.discard)
    run.job_id = job.id
    await ctx.repo.save("evaluation_runs", run)
    return run


async def watch_start(ctx: Ctx, job: Job, run: EvaluationRun, branches: list[SimulationBranch],
                      began: asyncio.Event) -> None:
    """The start watchdog (#125). It runs beside the job, not in it, since the job's own task is what stopped there:
    a job that has not got to its branches within START_TIMEOUT_S never will, so its branches, the evaluation and
    the job are failed with the reason instead of staying queued and running forever."""
    try:
        await asyncio.wait_for(began.wait(), START_TIMEOUT_S)
        return
    except TimeoutError:
        pass
    if ctx.jobs.closing:
        return  # the startup sweep ends it
    msg = never_started()
    log.error("evaluation %s: job %s never reached its branches in %.0f s; failing it", run.id, job.id,
              START_TIMEOUT_S)
    for b in branches:
        if b.status in (JobStatus.queued, JobStatus.running):
            b.status, b.error = JobStatus.failed, msg
            await ctx.repo.save("simulation_branches", b, project_id=run.project_id)
            ctx.changed(run.project_id, "simulation_branches", b.id)
    run.status = JobStatus.failed
    await ctx.repo.save("evaluation_runs", run)
    ctx.changed(run.project_id, "evaluation_runs", run.id)
    if job.status not in TERMINAL:
        ctx.jobs.update(job, status="failed", error=msg)


async def _start(ctx: Ctx, run: EvaluationRun, b: SimulationBranch) -> None:
    """Record the branch as running, its last step before the worker, within START_TIMEOUT_S: a store call that
    never returns fails the branch with the reason instead of leaving it queued forever (#125)."""
    try:
        async with asyncio.timeout(START_TIMEOUT_S):
            b.status = JobStatus.running
            await ctx.repo.save("simulation_branches", b, project_id=run.project_id)
    except TimeoutError:
        raise BranchNeverStarted(never_started()) from None
    ctx.changed(run.project_id, "simulation_branches", b.id)


async def run_branch(ctx: Ctx, job: Job, run: EvaluationRun, b: SimulationBranch,
                     policy_cfg: dict[str, Any], scene: dict[str, Any] | None = None) -> SimulationBranch:
    key = branch_dir(run.project_id, run.id, b.id)
    cfg = {
        "mode": "evaluate", "seeds": b.seeds, "policy": policy_cfg, "max_steps": 300, "max_videos": 2,
        "variation": {**b.spec.condition.model_dump(), "cube_offset": list(b.spec.cube_offset),
                      "light_scale": b.spec.light_scale},
        "env_revision": ENVHUB_REVISION,
        "scene": scene,
    }
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    try:
        await _start(ctx, run, b)  # inside the try, so a failure here is recorded on the branch, not lost
        cfg["out_dir"] = str(ctx.local(key))

        def on_event(ev: dict[str, Any]) -> None:
            if ev.get("type") == "episode":
                ctx.jobs.log(job, f"[{b.spec.name}] seed {ev['seed']}: "
                                  f"{'success' if ev['success'] else ev.get('failure_mode')}")
                ctx.changed(run.project_id, "simulation_branches", b.id)

        await ctx.jobs.run_worker(job, "cortex.workers.sim_worker", cfg, ctx.local(f"{key}/config.json"), on_event)
        res = json.loads(ctx.local(f"{key}/result.json").read_text())
        eps = []
        for e in res["episodes"]:
            vk = f"{key}/{e['video']}" if e.get("video") else None
            if vk:
                await ctx.publish_file(vk)
            eps.append(EpisodeResult(seed=e["seed"], success=e["success"], reward=e["reward"], steps=e["steps"],
                                     max_cube_z=e["max_cube_z"], final_cube_box_dist=e["final_cube_box_dist"],
                                     failure_mode=e.get("failure_mode"), video_key=vk))
        b.result = summarize(eps, res.get("policy", ""), loop.time() - t0)
        b.status = JobStatus.succeeded
    except asyncio.CancelledError:
        b.status = JobStatus.failed if ctx.jobs.closing else JobStatus.cancelled
        b.error = INTERRUPTED if ctx.jobs.closing else "stopped by user"
        raise
    except Exception as e:
        if ctx.jobs.stop_requested(job.id):  # the stop ended the worker before the task saw the cancel
            b.status, b.error = JobStatus.cancelled, "stopped by user"
        else:
            b.status, b.error = JobStatus.failed, (str(e) if isinstance(e, BranchNeverStarted)
                                                   else f"{type(e).__name__}: {e}")[:500]
        raise
    finally:
        await ctx.repo.save("simulation_branches", b, project_id=run.project_id)
        ctx.changed(run.project_id, "simulation_branches", b.id)
    return b


async def get_evaluation(ctx: Ctx, eval_id: str) -> dict[str, Any] | None:
    run = await ctx.repo.get("evaluation_runs", eval_id, EvaluationRun)
    if run is None:
        return None
    branches = [await ctx.repo.get("simulation_branches", bid, SimulationBranch) for bid in run.branch_ids]
    return {"run": run, "branches": [b for b in branches if b]}


async def list_evaluations(ctx: Ctx, project_id: str) -> list[dict[str, Any]]:
    runs = await ctx.repo.list("evaluation_runs", EvaluationRun, project_id=project_id)
    out = []
    for r in runs:
        ev = await get_evaluation(ctx, r.id)
        if ev:
            out.append(ev)
    return out
