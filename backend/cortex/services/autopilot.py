"""Autopilot: the closed improvement loop as an explicit, resumable state machine.

  TASK_REQUESTED -> SCENARIO_PLANNED -> AGENT_COLLECTING -> EPISODES_VALIDATED
  -> DATASET_VERSIONED -> TRAINING_REQUESTED -> POLICY_TRAINED -> POLICY_EVALUATED
  -> FAILURES_ANALYZED -> NEXT_EXPERIMENT_SELECTED -> SCENARIO_PLANNED -> ...  -> DONE

Demonstrations come from the agent planner (self-hosted Qwen3-VL, or GPT-6 Astra) when it is available,
otherwise from the scripted IK oracle, and the run records which one it used.
After each evaluation the weakest measured condition becomes the next targeted
collection, in fresh simulator seeds (evaluation seeds stay disjoint).

Humans appear only where the project's standing AutopilotPolicy does not cover an
action: planner spend above model_budget_usd, paid GPUs, GPU jobs above the cap,
and anything on physical hardware (refused). The current state and every id it
waits on are persisted, so a backend restart resumes the run where it stopped.
"""

from __future__ import annotations

import asyncio
import logging

from ..context import Ctx
from ..models import (
    AutopilotRun, AutopilotStep, Checkpoint, Condition, CreateDatasetVersionRequest, DatasetVersion,
    EpisodeStatus, JobStatus, LaunchEvaluationRequest, Mode, PolicyRef, Project,
    StartAgentCollectionRequest, TrainingJob,
)
from . import (
    agent_collection_service, analysis_service, dataset_service, experiment_comparison, recommendation_service,
    recording_service, simulation_runner, training_service,
)
from .augment import PRESETS

log = logging.getLogger(__name__)
ACTOR = "autopilot"
MIN_TRAIN_EPISODES = 10
_running: dict[str, asyncio.Task] = {}

STATES = ["TASK_REQUESTED", "SCENARIO_PLANNED", "AGENT_COLLECTING", "EPISODES_VALIDATED", "DATASET_VERSIONED",
          "TRAINING_REQUESTED", "POLICY_TRAINED", "POLICY_EVALUATED", "FAILURES_ANALYZED",
          "NEXT_EXPERIMENT_SELECTED", "DONE"]


class Stop(Exception):
    pass


# ------------------------------------------------------------------ lifecycle

async def start(ctx: Ctx, project: Project) -> AutopilotRun:
    if project.mode != Mode.sim:
        raise PermissionError("Autopilot only runs simulation projects; real-robot actions need an operator.")
    cur = await current(ctx, project.id)
    if cur and cur.status in ("running", "waiting_approval") and project.id in _running:
        return cur
    collector = project.autopilot.collector
    if collector == "agent" and agent_collection_service.resolve_planner(ctx) is None:
        collector = "oracle"
    run = AutopilotRun(project_id=project.id, collector=collector)
    await _save(ctx, run)
    if collector != project.autopilot.collector:
        await _step(ctx, run, "plan", "No agent planner is configured; this run uses the scripted IK oracle for "
                    "demonstrations (no other model is substituted).", status="skipped")
    _running[project.id] = asyncio.create_task(_loop(ctx, project.id, run))
    await ctx.audit(ACTOR, "autopilot_start", project.id, {**project.autopilot.model_dump(), "collector": collector},
                    "ok", run.id)
    return run


async def stop(ctx: Ctx, project_id: str) -> AutopilotRun | None:
    t = _running.pop(project_id, None)
    if t:
        t.cancel()
    run = await current(ctx, project_id)
    if run:
        for key in ("collection_job_id", "dataset_job_id", "evaluation_job_id"):
            if run.context.get(key):
                await ctx.jobs.cancel(run.context[key])
        if run.status in ("running", "waiting_approval"):
            run.status = "stopped"
            await _save(ctx, run)
    return run


async def resume_all(ctx: Ctx) -> int:
    """Called at startup: re-attach to runs that were in flight when the backend stopped."""
    n = 0
    for status in ("running", "waiting_approval"):
        for row in await ctx.store.list("autopilot_runs", status=status):
            run = AutopilotRun.model_validate(row["data"])
            if run.project_id in _running:
                continue
            if "state" not in row["data"]:  # created before the state machine: nothing safe to resume from
                run.status = "stopped"
                await _step(ctx, run, run.phase, "Stopped on restart: this run predates resumable state; "
                            "start Autopilot again", status="skipped")
                continue
            await _step(ctx, run, run.phase, f"Resumed after backend restart in state {run.state}")
            _running[run.project_id] = asyncio.create_task(_loop(ctx, run.project_id, run))
            n += 1
    return n


async def current(ctx: Ctx, project_id: str) -> AutopilotRun | None:
    runs = await ctx.repo.list("autopilot_runs", AutopilotRun, project_id=project_id)
    return runs[-1] if runs else None


async def _save(ctx: Ctx, run: AutopilotRun) -> None:
    await ctx.repo.save("autopilot_runs", run)
    ctx.bus.publish(run.project_id, "autopilot", run.model_dump(mode="json"))


async def _step(ctx: Ctx, run: AutopilotRun, phase: str, message: str, ref_ids: list[str] | None = None,
                status: str = "done") -> None:
    run.phase = phase
    run.steps.append(AutopilotStep(phase=phase, message=message, ref_ids=[r for r in (ref_ids or []) if r],
                                   status=status))  # type: ignore[arg-type]
    await _save(ctx, run)


async def _goto(ctx: Ctx, run: AutopilotRun, state: str) -> None:
    run.state = state
    await _save(ctx, run)


async def _wait_job(ctx: Ctx, job_id: str | None) -> str:
    """Wait for an in-process job. Returns its status, or 'lost' if it did not survive a restart."""
    if not job_id or ctx.jobs.get(job_id) is None:
        return "lost"
    task = ctx.jobs.tasks.get(job_id)
    if task:
        await asyncio.shield(task)
    return ctx.jobs.get(job_id).status  # type: ignore[union-attr]


# ------------------------------------------------------------------ the loop

async def _loop(ctx: Ctx, project_id: str, run: AutopilotRun) -> None:
    from .agent_orchestrator import get_project

    c = run.context
    try:
        while run.state != "DONE":
            project = await get_project(ctx, project_id)
            pol = project.autopilot
            st = run.state

            if st == "TASK_REQUESTED":
                run.iteration = max(run.iteration, 1)
                await _step(ctx, run, "plan", f"Task: {project.spec.goal}. Success = simulator check that the cube "
                            f"rests in the box. Collector: {_collector_label(ctx, run)}.")
                await _goto(ctx, run, "SCENARIO_PLANNED")

            elif st == "SCENARIO_PLANNED":
                if run.collector == "agent" and (agent_collection_service.resolve_planner(ctx) is None or
                                                 await agent_collection_service.selfhosted_reachable(ctx)):
                    run.collector = "oracle"
                    await _step(ctx, run, "plan", "The agent planner is not available; switching this run to the "
                                "scripted IK oracle (no other model is substituted).", status="skipped")
                targeted = c.get("next_conditions")
                conds = [Condition(**x) for x in targeted] if targeted else None
                n = pol.demos_per_round if targeted else max(pol.demos_per_round, project.plan.budget_total)
                req = StartAgentCollectionRequest(
                    num_demos=n, conditions=conds, diversity="plan" if not targeted else "medium",
                    collector=run.collector, reason=c.get("next_reason", "initial plan"))  # type: ignore[arg-type]
                col = await agent_collection_service.start_collection(ctx, project, req, ACTOR)
                c.update(collection_id=col.id, collection_job_id=col.job_id)
                where = ", ".join(sorted({x.key for x in conds})) if conds else "the initial plan"
                await _step(ctx, run, "collect", f"{_collector_label(ctx, run)} generating {n} demonstrations for "
                            f"{where} (new seeds)", [col.id], status="running")
                await _goto(ctx, run, "AGENT_COLLECTING")

            elif st == "AGENT_COLLECTING":
                status = await _wait_job(ctx, c.get("collection_job_id"))
                col = await agent_collection_service.get_collection(ctx, c["collection_id"])
                if status == "lost" and col.status == "running":
                    col.status = "failed"
                    col.error = "interrupted by backend restart; kept the attempts recorded before it"
                    await ctx.repo.save("agent_collections", col)
                t = col.totals
                await _step(ctx, run, "collect", f"{t.get('successes', 0)}/{t.get('attempts', 0)} attempts succeeded "
                            f"in simulation; {t.get('model_calls', 0)} model calls, ${t.get('cost_usd', 0):.2f}"
                            + (f" ({col.error})" if col.error else ""), [col.id])
                await _goto(ctx, run, "EPISODES_VALIDATED")

            elif st == "EPISODES_VALIDATED":
                job = await analysis_service.start_analysis(ctx, project, use_vlm=False, actor=ACTOR)
                await _wait_job(ctx, job.id)
                eps = await recording_service.list_episodes(ctx, project.id)
                accepted = [e for e in eps if e.status == EpisodeStatus.accepted]
                if len(accepted) < MIN_TRAIN_EPISODES:
                    if c.get("top_up_rounds", 0) >= 2:
                        raise Stop(f"only {len(accepted)} accepted demonstrations after repeated collection")
                    c["top_up_rounds"] = c.get("top_up_rounds", 0) + 1
                    c.pop("next_conditions", None)
                    await _step(ctx, run, "validate", f"{len(accepted)} accepted demos (< {MIN_TRAIN_EPISODES}); "
                                "collecting more before training")
                    await _goto(ctx, run, "SCENARIO_PLANNED")
                    continue
                augment = bool(c.get("augment_next"))
                job = await dataset_service.create_version(ctx, project, CreateDatasetVersionRequest(
                    name=f"autopilot-it{run.iteration}", augmentations=PRESETS["lighting_robustness"] if augment else []),
                    actor=ACTOR)
                c["dataset_job_id"] = job.id
                await _step(ctx, run, "validate", f"{len(accepted)} accepted demos (simulator-verified, quality "
                            f"checked); versioning dataset" + (" with lighting augmentation" if augment else ""))
                await _goto(ctx, run, "DATASET_VERSIONED")

            elif st == "DATASET_VERSIONED":
                status = await _wait_job(ctx, c.get("dataset_job_id"))
                if status != "succeeded":
                    await _goto(ctx, run, "EPISODES_VALIDATED")  # rebuild (also covers a lost job)
                    continue
                c["dataset_id"] = ctx.jobs.get(c["dataset_job_id"]).result_ref  # type: ignore[union-attr]
                ds = await ctx.repo.get("dataset_versions", c["dataset_id"], DatasetVersion)
                await _step(ctx, run, "dataset", f"Dataset {ds.id} ({ds.num_source_episodes} source demos, "  # type: ignore[union-attr]
                            f"{len(ds.links)} LeRobot episodes) stored", [ds.id])  # type: ignore[union-attr]
                tj = await training_service.propose(ctx, project, c["dataset_id"], "act", actor=ACTOR)
                c["training_job_id"] = tj.id
                await _goto(ctx, run, "TRAINING_REQUESTED")

            elif st == "TRAINING_REQUESTED":
                ck = await _train(ctx, run, project, c["training_job_id"])
                c["checkpoint_id"] = ck.id
                await _goto(ctx, run, "POLICY_TRAINED")

            elif st == "POLICY_TRAINED":
                ev = None
                if c.get("evaluation_id"):
                    ev = await simulation_runner.get_evaluation(ctx, c["evaluation_id"])
                if not ev or (ev["run"].status not in (JobStatus.succeeded,) and
                              ctx.jobs.get(c.get("evaluation_job_id", "")) is None):
                    run_ = await simulation_runner.launch_evaluation(ctx, project, LaunchEvaluationRequest(
                        policy=PolicyRef(kind="checkpoint", checkpoint_id=c["checkpoint_id"]),
                        episodes_per_branch=pol.episodes_per_branch, name=f"autopilot it{run.iteration}"), actor=ACTOR)
                    c.update(evaluation_id=run_.id, evaluation_job_id=run_.job_id)
                    await _step(ctx, run, "evaluate", f"Evaluating learned policy {c['checkpoint_id']} on baseline / "
                                "shifted object / dim lighting (held-out evaluation seeds)", [run_.id], status="running")
                await _wait_job(ctx, c.get("evaluation_job_id"))
                await _goto(ctx, run, "POLICY_EVALUATED")

            elif st == "POLICY_EVALUATED":
                ev = await simulation_runner.get_evaluation(ctx, c["evaluation_id"])
                rates = {b.spec.name: (b.result.success_rate, b) for b in ev["branches"] if b.result}  # type: ignore[index]
                if not rates:
                    raise Stop(_no_results(ev))
                cmp = await experiment_comparison.compare(ctx, project.id)
                await _step(ctx, run, "evaluate", "Measured: " + ", ".join(
                    f"{k} {b.result.successes}/{b.result.n}" for k, (_, b) in rates.items()) + f". {cmp.summary}",
                    [c["evaluation_id"]])
                worst = min((r for r, _ in rates.values()), default=0.0)
                if not run.best or worst > run.best.get("min_rate", -1):
                    run.best = {"checkpoint_id": c["checkpoint_id"], "evaluation_id": c["evaluation_id"],
                                "rates": {k: r for k, (r, _) in rates.items()}, "min_rate": worst}
                if rates and worst >= pol.target_success_rate:
                    await _step(ctx, run, "done", f"Target {pol.target_success_rate:.0%} reached on every condition")
                    run.status = "succeeded"
                    await _goto(ctx, run, "DONE")
                    continue
                await _goto(ctx, run, "FAILURES_ANALYZED")

            elif st == "FAILURES_ANALYZED":
                job = await analysis_service.start_analysis(ctx, project, use_vlm=True, actor=ACTOR)
                await _wait_job(ctx, job.id)
                an = await analysis_service.latest_analysis(ctx, project.id)
                await _step(ctx, run, "analyze", an.summary if an else "analysis finished", [an.id if an else ""])
                await _goto(ctx, run, "NEXT_EXPERIMENT_SELECTED")

            elif st == "NEXT_EXPERIMENT_SELECTED":
                if run.iteration >= pol.max_iterations:
                    await _step(ctx, run, "done", f"Stopped after {pol.max_iterations} iterations; best checkpoint "
                                f"{run.best.get('checkpoint_id')} {run.best.get('rates')}")
                    run.status = "succeeded"
                    await _goto(ctx, run, "DONE")
                    continue
                ev = await simulation_runner.get_evaluation(ctx, c["evaluation_id"])
                rec = await recommendation_service.recommend(ctx, project, actor=ACTOR)
                c["augment_next"] = any(r.action == "augment" for r in rec)
                measured = [b for b in ev["branches"] if b.result]  # type: ignore[index]
                if not measured:  # e.g. resumed into this state after the branches failed
                    raise Stop(_no_results(ev))
                worst_b = min(measured, key=lambda b: b.result.success_rate)  # type: ignore[union-attr]
                region = recommendation_service.region_of(worst_b.spec.condition.cube_region, worst_b.spec.cube_offset)
                cell = Condition(cube_region=region, lighting=worst_b.spec.condition.lighting)  # type: ignore[arg-type]
                r = worst_b.result
                cells = [cell]
                gap = next((x for x in rec if x.action == "collect" and x.condition and x.support == "evidence_backed"
                            and x.condition.key != cell.key), None)
                if gap:
                    cells.append(gap.condition)
                c["next_conditions"] = [x.model_dump() for x in cells]
                c["next_reason"] = (f"ACT failed {r.n - r.successes} of {r.n} trials in '{worst_b.spec.name}' "
                                    f"-> generate {pol.demos_per_round} agent demonstrations in {cell.key}"
                                    + (f" and {gap.condition.key} (coverage gap: {gap.title})" if gap else ""))
                await _step(ctx, run, "select", c["next_reason"] + (" (+ lighting augmentation: measured lighting gap)"
                                                                     if c["augment_next"] else ""), [worst_b.id])
                run.iteration += 1
                for k in ("collection_id", "collection_job_id", "dataset_job_id", "dataset_id", "training_job_id",
                          "checkpoint_id", "evaluation_id", "evaluation_job_id", "top_up_rounds"):
                    c.pop(k, None)
                await _goto(ctx, run, "SCENARIO_PLANNED")
    except Stop as e:
        run.status, run.error = "failed", str(e)
        await _step(ctx, run, run.phase, f"Stopped: {e}", status="failed")
    except asyncio.CancelledError:
        run.status = "stopped"
        await _save(ctx, run)
        raise
    except Exception as e:
        log.exception("autopilot failed")
        run.status, run.error = "failed", f"{type(e).__name__}: {e}"
        await _step(ctx, run, run.phase, f"Error: {run.error}", status="failed")
    finally:
        _running.pop(project_id, None)


def _no_results(ev: dict | None) -> str:
    """Why an evaluation can't steer the next round: no branch produced a measured result."""
    if not ev:
        return "evaluation not found; nothing measured to choose the next experiment from"
    errors = "; ".join(f"{b.spec.name}: {b.error or b.status}" for b in ev["branches"]) or "no branches"
    return f"evaluation {ev['run'].id} produced no results ({errors})"


def _collector_label(ctx: Ctx, run: AutopilotRun) -> str:
    if run.collector == "agent":
        p = agent_collection_service.resolve_planner(ctx)
        return f"agent planner {p['model']} ({p['label']})" if p else "agent planner (not configured)"
    return "scripted IK oracle"


async def _train(ctx: Ctx, run: AutopilotRun, project: Project, tj_id: str) -> Checkpoint:
    pol = project.autopilot
    tj = await ctx.repo.get("training_jobs", tj_id, TrainingJob)
    assert tj
    if tj.status == JobStatus.proposed:
        covered = (not tj.estimate.paid or pol.allow_paid_jobs) and tj.estimate.gpu_hours <= pol.max_gpu_hours_per_job
        if tj.backend == "manual":
            run.status = "waiting_approval"
            await _step(ctx, run, "train", f"No GPU backend reachable. Run {tj.id}'s command bundle on a GPU and "
                        "register the checkpoint; autopilot continues when it appears.", [tj.id], status="waiting")
        elif covered and pol.auto_approve_free_training:
            await training_service.approve(ctx, tj.id, "standing approval (autopilot policy)", ACTOR, None)
            await _step(ctx, run, "train", f"Auto-approved {tj.id}: {tj.estimate.gpu}, ~{tj.estimate.gpu_hours} "
                        f"GPU-h, ${tj.estimate.est_cost_usd} — inside the standing approval", [tj.id])
        else:
            run.status = "waiting_approval"
            await _step(ctx, run, "train", f"{tj.id} needs your approval (~{tj.estimate.gpu_hours} GPU-h, "
                        f"${tj.estimate.est_cost_usd}); outside the standing approval", [tj.id], status="waiting")
    while True:  # the training job is durable; polling the DB survives restarts
        cur = await ctx.repo.get("training_jobs", tj_id, TrainingJob)
        assert cur
        if cur.status == JobStatus.approved:
            await training_service.launch(ctx, cur.id, ACTOR)
            run.status = "running"
            await _step(ctx, run, "train", f"Training {cur.id} on {cur.backend}", [cur.id], status="running")
        elif cur.status == JobStatus.succeeded and cur.checkpoint_ids:
            ck = await ctx.repo.get("checkpoints", cur.checkpoint_ids[-1], Checkpoint)
            run.status = "running"
            await _step(ctx, run, "train", f"Checkpoint {ck.id} ready", [ck.id])  # type: ignore[union-attr]
            return ck  # type: ignore[return-value]
        elif cur.status in (JobStatus.failed, JobStatus.cancelled):
            raise Stop(f"training {cur.id} {cur.status}: {cur.error}")
        await asyncio.sleep(10)

