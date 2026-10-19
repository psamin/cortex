"""RL sandboxes as a service (#80): weaknesses -> proposals with measured pilots -> accepted runs -> results.

Each sandbox runs ``cortex.workers.rl_worker`` as a background job (first ``pilot``, then ``run`` once accepted)
and is persisted in ``rl_sandboxes``. Worker events are relayed on the project bus as ``rl`` events (sim_stats
throttled) so the studio can draw a live box per sandbox. Local CPU sandboxes cost $0 and an agent may accept
them within the project's standing budget; GPU or paid sandboxes need a human approval.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..context import Ctx
from ..errors import NotFound
from ..models import (
    BranchSpec,
    Checkpoint,
    Condition,
    CreateDatasetVersionRequest,
    Episode,
    EpisodeSource,
    EpisodeStatus,
    FailureCluster,
    JobStatus,
    LaunchEvaluationRequest,
    PolicyRef,
    Project,
    new_id,
    now,
)
from ..rl import Pilot, RLSandbox, SandboxBudget, estimate, find_weaknesses, propose
from ..rl.proposals import transfer_fit
from ..rl.rounds import ConditionComparison, needs_human_demos, paired_compare, select_targeted
from ..rl.types import IterationStats, SandboxResult, Weakness
from . import simulation_runner
from .training_service import ApprovalRequired

HELDOUT_EVAL_SEEDS = 900_000  # re-evaluation seeds: disjoint from demo, sandbox and default evaluation seeds

LIVE_STATS_S = 0.25  # at most ~4 sim_stats events per second per sandbox on the bus
ACTIVE = ("piloting", "accepted", "running")
_slots: dict[str, asyncio.Semaphore] = {}
_live: dict[str, RLSandbox] = {}  # sandboxes with a job in flight: the job's copy is the source of truth


class RLBudgetExceeded(PermissionError):
    code = "rl_budget"


RoundStatus = Literal["skill_measured", "dataset_building", "retrain_proposed", "reevaluating", "realised",
                      "no_demos"]


class RLRound(BaseModel):
    """One sandbox's trip around the loop (docs/RL_DESIGN.md section 7). ``status`` moves
    skill_measured -> dataset_building -> retrain_proposed -> reevaluating -> realised (or no_demos when the sandbox
    produced no demo that fits). Only the paired re-evaluation of the retrained checkpoint is a measured policy
    improvement; the predicted uplift is the estimate logged at acceptance."""

    id: str = Field(default_factory=lambda: new_id("rlr"))
    project_id: str
    sandbox_id: str
    weakness_key: str
    condition: str
    skill_gain: float  # measured: tuned - base skill success on matched held-out seeds
    predicted_policy_uplift: float | None = None  # the estimate at acceptance
    realised_uplift: float | None = None  # measured: new - old checkpoint on the weak condition, matched seeds
    status: RoundStatus = "skill_measured"
    # closing the loop
    demo_episode_ids: list[str] = []  # the sandbox's demos registered as episodes (provenance rl_sandbox:<id>)
    targeted: dict[str, Any] = {}  # the cap check: offered, added, train share
    parent_dataset_version_id: str | None = None
    dataset_job_id: str | None = None
    dataset_version_id: str | None = None
    training_job_id: str | None = None  # proposed only; a human approves it
    checkpoint_before: str | None = None
    checkpoint_after: str | None = None
    evaluation_before: str | None = None
    evaluation_after: str | None = None
    heldout_base_seed: int | None = None
    episodes_per_condition: int | None = None
    comparisons: list[ConditionComparison] = []
    p_value: float | None = None  # McNemar on the weak condition
    regressions: list[str] = []  # conditions that got significantly worse
    possible_regressions: list[str] = []  # drops of >= 10 pp that are not significant at this n
    needs_human_demos: bool = False  # no paired improvement on this condition in 2 consecutive rounds
    transfer: dict[str, Any] | None = None  # the transfer curve refitted after this round (labelled estimate)
    note: str = ""
    created_at: datetime = Field(default_factory=now)
    updated_at: datetime | None = None


# ------------------------------------------------------------------ reads
async def weaknesses(ctx: Ctx, project: Project, checkpoint_id: str | None = None,
                     target: float = 0.8) -> list[Weakness]:
    evals = await simulation_runner.list_evaluations(ctx, project.id)
    clusters = await ctx.repo.list("failure_clusters", FailureCluster, project_id=project.id)
    return find_weaknesses(evals, clusters, target=target, checkpoint_id=checkpoint_id)


async def get(ctx: Ctx, sandbox_id: str) -> RLSandbox:
    sb = _live.get(sandbox_id) or await ctx.repo.get("rl_sandboxes", sandbox_id, RLSandbox)
    if sb is None:
        raise NotFound(f"RL sandbox {sandbox_id}")
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
            raise NotFound(f"RL sandbox {sb.id} in project {project.id}")
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


# ------------------------------------------------------------------ closing the loop (docs/RL_DESIGN.md section 7)
async def rounds(ctx: Ctx, project_id: str) -> list[RLRound]:
    return await ctx.repo.list("rl_rounds", RLRound, project_id=project_id)


async def round_for(ctx: Ctx, sandbox_id: str) -> RLRound:
    sb = await get(ctx, sandbox_id)
    for r in await rounds(ctx, sb.project_id):
        if r.sandbox_id == sandbox_id:
            return r
    raise ValueError(f"{sandbox_id} is '{sb.status}': a round starts once the sandbox has finished with a "
                     "measured held-out result")


async def _save_round(ctx: Ctx, rnd: RLRound, msg: str | None = None) -> None:
    rnd.updated_at = now()
    if msg:
        rnd.note = msg
    await ctx.repo.save("rl_rounds", rnd)
    ctx.changed(rnd.project_id, "rl_rounds", rnd.id)
    ctx.bus.publish(rnd.project_id, "rl", {"type": "rl_round", "sandbox_id": rnd.sandbox_id, "round_id": rnd.id,
                                           "status": rnd.status, "note": rnd.note})


def demo_dir(sb: RLSandbox, name: str) -> str:
    """Where the worker wrote a demo (``out_dir/demos/<name>`` of the run)."""
    return f"projects/{sb.project_id}/rl/{sb.id}/run/demos/{name}"


async def register_demos(ctx: Ctx, sb: RLSandbox, rnd: RLRound) -> list[Episode]:
    """The sandbox's successful demos as episodes (idempotent). They are oracle demos (the tuned skill reads
    privileged sim state) with provenance ``rl_sandbox:<id>``; each passes the same quality gate as other demos."""
    from .recording_service import AUTO_ACCEPT_MIN_SCORE, HD_FILES, list_episodes, quality_for

    eps = await list_episodes(ctx, sb.project_id)
    have = {e.provenance.get("demo"): e for e in eps if e.provenance.get("sandbox_id") == sb.id}
    out = []
    for name in (sb.result.demo_episode_ids if sb.result else []):
        if name in have:
            out.append(have[name])
            continue
        base = demo_dir(sb, name)
        meta_p = ctx.local(f"{base}/result.json")
        if not meta_p.exists():
            continue
        meta = json.loads(meta_p.read_text())
        files = (("episode", "episode.npz"), ("front", "front.mp4"), ("thumb", "thumb.jpg"), *HD_FILES)
        artifacts = {kind: f"{base}/{f}" for kind, f in files if ctx.local(f"{base}/{f}").exists()}
        if "episode" not in artifacts or not meta.get("success"):
            continue
        ep = Episode(project_id=sb.project_id, index=len(eps) + len(out), source=EpisodeSource.sim_scripted,
                     condition=Condition.model_validate(meta.get("condition") or sb.env.condition.model_dump()),
                     seed=int(meta["seed"]), num_frames=int(meta["num_frames"]), fps=int(meta.get("fps", 30)),
                     success=True, artifacts=artifacts,
                     provenance={**meta.get("provenance", {}), "sandbox_id": sb.id, "round_id": rnd.id, "demo": name,
                                 "weakness": sb.weakness.key, "oracle": "privileged sim state"})
        ep.quality = quality_for(ctx, ep, None)
        if ep.quality.score >= AUTO_ACCEPT_MIN_SCORE:
            ep.status = EpisodeStatus.accepted
        for k in artifacts.values():
            await ctx.publish_file(k)
        await ctx.repo.save("episodes", ep)
        ctx.changed(sb.project_id, "episodes", ep.id)
        out.append(ep)
    return out


async def _start_dataset(ctx: Ctx, project: Project, sb: RLSandbox, rnd: RLRound, actor: str) -> None:
    """A new version of the project's latest dataset with the demos that fit under the targeted-data cap."""
    from . import dataset_service
    from .recording_service import list_episodes

    versions = await dataset_service.list_versions(ctx, project.id)
    if not versions:
        raise ValueError("no dataset version to extend: build one (create_dataset_version) and train on it first")
    latest = versions[-1]
    demos = [e for e in await register_demos(ctx, sb, rnd) if e.status == EpisodeStatus.accepted]
    rnd.demo_episode_ids = [e.id for e in demos]
    by_id = {e.id: e for e in await list_episodes(ctx, project.id)}
    base = [by_id[i] for i in dict.fromkeys(ln.episode_id for ln in latest.links) if i in by_id]
    in_base = {e.id for e in base}
    new = [e for e in demos if e.id not in in_base]
    chosen, rnd.targeted = select_targeted(base, new, latest.eval_fraction, latest.split_seed)
    rnd.parent_dataset_version_id = latest.id
    if not chosen:
        recorded = len(sb.result.demo_episode_ids) if sb.result else 0
        rnd.status = "no_demos"
        await _save_round(ctx, rnd, f"no demo to add: {recorded} recorded, {len(demos)} passed quality, {len(new)} "
                                    f"new; the targeted-data cap ({rnd.targeted['cap']:.0%} of the train split) "
                                    "left room for none")
        return
    eps = base + chosen
    imported = any(e.source == EpisodeSource.imported for e in eps)
    req = CreateDatasetVersionRequest(
        name=f"{latest.name}+rl-{sb.weakness.key}", parent_id=latest.id, episode_ids=[e.id for e in eps],
        sources=sorted({e.source for e in eps}) if imported else None,
        segments=sorted({e.segment for e in eps} | {"demo"}), augmentations=latest.augmentations,
        eval_fraction=latest.eval_fraction, split_seed=latest.split_seed)
    job = await dataset_service.create_version(ctx, project, req, actor)
    rnd.dataset_job_id, rnd.status = job.id, "dataset_building"
    await _save_round(ctx, rnd, f"{len(chosen)}/{len(new)} new demos added to {latest.name} (targeted share of "
                                f"train {rnd.targeted['targeted_share']:.0%}, cap {rnd.targeted['cap']:.0%}); "
                                f"building the new version (job {job.id})")


async def _propose_retrain(ctx: Ctx, project: Project, rnd: RLRound, actor: str) -> None:
    from . import training_service

    job = ctx.jobs.get(rnd.dataset_job_id or "")
    if job is None or job.status in ("failed", "cancelled"):  # lost on a restart, or failed: rebuild next call
        rnd.status, rnd.dataset_job_id = "skill_measured", None
        why = f"failed: {job.error or job.status}" if job else "was lost"
        await _save_round(ctx, rnd, f"the dataset build {why}; call again to rebuild it")
        return
    if job.status != "succeeded":
        return
    rnd.dataset_version_id = job.result_ref
    before = await ctx.repo.get("checkpoints", rnd.checkpoint_before or "", Checkpoint)
    policy = before.policy_type if before and before.policy_type in ("act", "smolvla") else "act"
    tj = await training_service.propose(ctx, project, rnd.dataset_version_id, policy, actor=actor)
    rnd.training_job_id, rnd.status = tj.id, "retrain_proposed"
    await _save_round(ctx, rnd, f"retraining {tj.id} proposed on {rnd.dataset_version_id}; a human approves it in "
                                "the Cortex studio")


async def _launch_reevaluation(ctx: Ctx, project: Project, sb: RLSandbox, rnd: RLRound, actor: str,
                               episodes: int | None) -> None:
    """Old vs new checkpoint on every condition of the evaluation the weakness came from, on the same fresh seeds."""
    cks = [c for c in await ctx.repo.list("checkpoints", Checkpoint, project_id=project.id)
           if c.training_job_id and c.training_job_id == rnd.training_job_id]
    if not cks:
        return
    if not rnd.checkpoint_before:
        raise ValueError(f"{sb.id}'s weakness has no checkpoint to compare against")
    rnd.checkpoint_after = max(cks, key=lambda c: (c.step, c.created_at)).id
    src = await simulation_runner.get_evaluation(ctx, sb.weakness.evaluation_id)
    branches = src["branches"] if src else []
    specs = [b.spec for b in branches]
    if not any(s.name == sb.weakness.key for s in specs):
        specs.append(BranchSpec(name=sb.weakness.key, condition=sb.weakness.condition,
                                cube_offset=sb.weakness.cube_offset, light_scale=sb.weakness.light_scale))
    n = episodes or max([b.result.n for b in branches if b.result] or [20])
    done = sum(r.heldout_base_seed is not None for r in await rounds(ctx, project.id) if r.id != rnd.id)
    rnd.heldout_base_seed = HELDOUT_EVAL_SEEDS + 1000 * done
    rnd.episodes_per_condition = n
    runs = {}
    for side, ck in (("before", rnd.checkpoint_before), ("after", rnd.checkpoint_after)):
        runs[side] = await simulation_runner.launch_evaluation(ctx, project, LaunchEvaluationRequest(
            policy=PolicyRef(kind="checkpoint", checkpoint_id=ck), branches=specs, episodes_per_branch=n,
            base_seed=rnd.heldout_base_seed, executor="local", name=f"RL round {rnd.id} ({side}): {ck}"), actor)
    rnd.evaluation_before, rnd.evaluation_after = runs["before"].id, runs["after"].id
    rnd.status = "reevaluating"
    await _save_round(ctx, rnd, f"paired re-evaluation: {rnd.checkpoint_before} vs {rnd.checkpoint_after} on "
                                f"{len(specs)} conditions x {n} matched held-out seeds from {rnd.heldout_base_seed}")


async def _realise(ctx: Ctx, project: Project, sb: RLSandbox, rnd: RLRound, actor: str) -> None:
    evs = [await simulation_runner.get_evaluation(ctx, i or "") for i in (rnd.evaluation_before, rnd.evaluation_after)]
    if any(e is None for e in evs):
        rnd.status = "retrain_proposed"
        await _save_round(ctx, rnd, "a re-evaluation run is missing; call again to relaunch both")
        return
    if any(e["run"].status not in (JobStatus.succeeded, JobStatus.failed) for e in evs):
        return
    before, after = evs
    comps = paired_compare(before["branches"], after["branches"], target_name=sb.weakness.key)
    target = next((c for c in comps if c.is_target), None)
    if target is None:
        why = next((b.error for e in evs for b in e["branches"] if b.error), None)  # e.g. no GPU here (#290)
        rnd.status = "retrain_proposed"
        await _save_round(ctx, rnd, "the weak condition has no finished paired result"
                          + (f" ({why})" if why else "") + "; call again to relaunch")
        return
    rnd.comparisons = comps
    rnd.realised_uplift, rnd.p_value = target.delta, target.p_value
    rnd.regressions = [c.name for c in comps if c.verdict == "regressed"]
    rnd.possible_regressions = [c.name for c in comps if c.possible_regression]
    rnd.status = "realised"
    earlier = sorted((r for r in await rounds(ctx, project.id) if r.status == "realised" and r.id != rnd.id
                      and r.weakness_key == rnd.weakness_key), key=lambda r: r.created_at)
    verdicts = [next((c.verdict for c in r.comparisons if c.is_target), "no_significant_change") for r in earlier]
    rnd.needs_human_demos = needs_human_demos(verdicts + [target.verdict])
    await ctx.repo.save("rl_rounds", rnd)  # so the refit below includes this round
    rnd.transfer = transfer_fit(await history(ctx, project.id))
    pred = f"{rnd.predicted_policy_uplift:+.0%}" if rnd.predicted_policy_uplift is not None else "n/a"
    msg = (f"{sb.weakness.key}: {target.old_rate:.0%} -> {target.new_rate:.0%} on {target.n} matched seeds "
           f"(realised {target.delta:+.0%}, predicted {pred}, McNemar p={target.p_value:.3g}, {target.verdict})"
           + (f"; regressions: {', '.join(rnd.regressions)}" if rnd.regressions else "")
           + ("; no paired improvement twice running: flag this condition for human or teleop demos"
              if rnd.needs_human_demos else "")
           + f"; transfer curve now {rnd.transfer['source']} ({rnd.transfer['rounds_used']} rounds)")
    await _save_round(ctx, rnd, msg)
    await ctx.audit(actor, "close_rl_round", project.id, {"sandbox_id": sb.id, "round_id": rnd.id}, "ok", msg)


async def close_round(ctx: Ctx, project: Project, sandbox_id: str, actor: str = "gui",
                      episodes_per_condition: int | None = None) -> RLRound:
    """Advance a finished sandbox's round as far as it can go now; call again to continue. Never approves training:
    the retrain is only proposed, and the round waits until a human approves it and its checkpoint exists."""
    sb = await get(ctx, sandbox_id)
    if sb.project_id != project.id:
        raise NotFound(f"RL sandbox {sandbox_id} in project {project.id}")
    rnd = await round_for(ctx, sandbox_id)
    rnd.checkpoint_before = rnd.checkpoint_before or sb.weakness.checkpoint_id
    if rnd.status == "skill_measured":
        await _start_dataset(ctx, project, sb, rnd, actor)
    if rnd.status == "dataset_building":
        await _propose_retrain(ctx, project, rnd, actor)
    if rnd.status == "retrain_proposed":
        await _launch_reevaluation(ctx, project, sb, rnd, actor, episodes_per_condition)
    if rnd.status == "reevaluating":
        await _realise(ctx, project, sb, rnd, actor)
    return rnd


def waiting_on(rnd: RLRound) -> dict[str, Any] | None:
    """What the round is blocked on right now: a job to poll, or a human approval."""
    if rnd.status == "dataset_building":
        return {"kind": "job", "job_id": rnd.dataset_job_id, "what": "the dataset version build"}
    if rnd.status == "retrain_proposed":
        return {"kind": "human_approval", "training_job_id": rnd.training_job_id,
                "what": "a human approves and launches the retraining in the Cortex studio; the round resumes once "
                        "its checkpoint is registered"}
    if rnd.status == "reevaluating":
        return {"kind": "evaluations", "evaluation_ids": [rnd.evaluation_before, rnd.evaluation_after],
                "what": "the paired re-evaluation of the old and new checkpoints"}
    return None


def round_view(rnd: RLRound) -> dict[str, Any]:
    out = rnd.model_dump(mode="json")
    out["predicted_vs_realised"] = {
        "predicted_policy_uplift": {"value": rnd.predicted_policy_uplift, "kind": "estimate"},
        "realised_uplift": {"value": rnd.realised_uplift, "kind": "measured", "p_value": rnd.p_value,
                            "n": rnd.episodes_per_condition}}
    out["waiting_on"] = waiting_on(rnd)
    return out


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
