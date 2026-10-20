"""Real trials and sim <-> real (docs/STACK_SPEC.md 2.2).

A real trial is one rollout of a checkpoint on the physical arm, judged by a person. Trials sit next to the sim
evaluation of the same checkpoint and condition, so each checkpoint shows its real rate, its sim rate and the gap,
and the project shows how well sim ranks checkpoints the way the real arm does. Sim evaluation is only worth
trusting where that agreement has been measured; below three checkpoints with both, the report says so instead
of guessing.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from ..context import Ctx
from ..errors import NotFound
from ..models import Checkpoint, Condition, JobStatus, Project, new_id, now
from . import recommendation_service
from .simulation_runner import list_evaluations, wilson

MIN_CHECKPOINTS = 3  # rank agreement needs at least this many checkpoints with both real and sim results
TABLE = "real_trials"


class RealTrial(BaseModel):
    id: str = Field(default_factory=lambda: new_id("rt"))
    project_id: str
    checkpoint_id: str
    condition: Condition = Field(default_factory=Condition)
    success: bool
    failure_mode: str | None = None
    judged_by: str  # the person who watched the rollout
    notes: str = ""
    source: Literal["studio", "agent"] = "studio"  # agent: an agent recorded what a person reported
    voided: bool = False  # undone in the studio (a mis-click); kept for the audit trail, left out of every rate
    created_at: datetime = Field(default_factory=now)


class TrialIn(BaseModel):
    checkpoint_id: str = Field(pattern=r"^ck_", description="the checkpoint that ran (ck_...)")
    success: bool = Field(description="did the task succeed, as the person watching judged it")
    condition: Condition = Field(default_factory=Condition, description="where the object was and the lighting")
    failure_mode: str | None = Field(None, max_length=80, description="e.g. missed_grasp, dropped, collision")
    notes: str = Field("", max_length=500)


class Rate(BaseModel):
    n: int
    successes: int
    rate: float | None
    ci95: tuple[float, float]


class SimRealRow(BaseModel):
    checkpoint_id: str
    condition: str  # Condition.key, or "all"
    real: Rate
    agent_logged: int = 0  # of the real trials, how many an agent recorded from a person's report
    sim: Rate | None  # None: no finished sim evaluation of this checkpoint covers the condition
    gap: float | None  # real - sim, in rate points


class SimRealReport(BaseModel):
    kind: Literal["measured"] = "measured"
    checkpoints: list[SimRealRow]  # one row per checkpoint, all conditions pooled
    rows: list[SimRealRow]  # per checkpoint and condition
    spearman: float | None  # rank agreement of sim and real success across checkpoints
    checkpoints_compared: int
    summary: str


def _rate(k: int, n: int) -> Rate:
    return Rate(n=n, successes=k, rate=(k / n if n else None), ci95=wilson(k, n))


def spearman(x: list[float], y: list[float]) -> float | None:
    """Rank correlation with average ranks for ties; None when either side has no spread."""
    def ranks(v: list[float]) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        order = a.argsort()
        r = np.empty(len(a))
        r[order] = np.arange(len(a), dtype=float)
        for val in np.unique(a):
            idx = a == val
            r[idx] = r[idx].mean()
        return r
    if len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


async def log_trials(ctx: Ctx, project: Project, trials: list[TrialIn], judged_by: str,
                     source: Literal["studio", "agent"], actor: str) -> list[RealTrial]:
    """Record trials a person judged. Every checkpoint must be this project's; nothing is saved otherwise."""
    if not judged_by.strip():
        raise ValueError("judged_by must name the person who watched the trials")
    for t in trials:
        ck = await ctx.repo.get("checkpoints", t.checkpoint_id, Checkpoint)
        if ck is None or ck.project_id != project.id:
            raise NotFound(f"checkpoint {t.checkpoint_id} in project {project.id}")
    out = []
    for t in trials:
        rt = RealTrial(project_id=project.id, judged_by=judged_by.strip()[:80], source=source, **t.model_dump())
        await ctx.repo.save(TABLE, rt, status="success" if rt.success else "failure")
        out.append(rt)
    ctx.changed(project.id, TABLE, out[-1].id if out else "")
    k = sum(t.success for t in out)
    await ctx.audit(actor, "log_real_trials", project.id, {"n": len(out), "judged_by": judged_by, "source": source},
                    "ok", f"{len(out)} real trial(s), {k} success(es), judged by {judged_by}")
    return out


async def void_trial(ctx: Ctx, trial_id: str, actor: str) -> RealTrial:
    """Undo a mis-click in the studio. Only a person can withdraw a judgement; the row stays, marked void."""
    if actor != "gui":
        err = PermissionError("only the studio can withdraw a real trial")
        err.approval_required = False  # type: ignore[attr-defined] - no approval unlocks this
        raise err
    t = await ctx.repo.get(TABLE, trial_id, RealTrial)
    if t is None:
        raise NotFound(trial_id)
    t.voided = True
    await ctx.repo.save(TABLE, t, status="voided")
    ctx.changed(t.project_id, TABLE, trial_id)
    await ctx.audit(actor, "void_real_trial", t.project_id, {"id": trial_id}, "ok", f"withdrew {trial_id}")
    return t


async def list_trials(ctx: Ctx, project_id: str) -> list[RealTrial]:
    return [t for t in await ctx.repo.list(TABLE, RealTrial, project_id=project_id) if not t.voided]


async def _sim_by_checkpoint(ctx: Ctx, project_id: str) -> dict[str, dict[str, tuple[int, int]]]:
    """{checkpoint: {condition key: (successes, n)}} from each checkpoint's latest finished sim evaluation. A
    branch's effective condition places a shifted object in the region it ends up in, like improve_policy does."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for e in await list_evaluations(ctx, project_id):  # oldest first: later runs replace earlier ones
        run = e["run"]
        if run.policy.kind != "checkpoint" or not run.policy.checkpoint_id or run.status != JobStatus.succeeded:
            continue
        cells: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for b in e["branches"]:
            if not b.result:
                continue
            region = recommendation_service.region_of(b.spec.condition.cube_region, b.spec.cube_offset)
            key = Condition(cube_region=region, lighting=b.spec.condition.lighting).key  # type: ignore[arg-type]
            cells[key][0] += b.result.successes
            cells[key][1] += b.result.n
        out[run.policy.checkpoint_id] = {k: (v[0], v[1]) for k, v in cells.items()}
    return out


async def report(ctx: Ctx, project: Project) -> SimRealReport:
    trials = await list_trials(ctx, project.id)
    sim = await _sim_by_checkpoint(ctx, project.id)
    by: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for t in trials:
        cell = by[t.checkpoint_id][t.condition.key]
        cell[0] += int(t.success)
        cell[1] += 1
        cell[2] += int(t.source == "agent")
    rows, per_ck = [], []
    pairs: list[tuple[float, float]] = []
    for ck, conds in by.items():
        sim_ck = sim.get(ck, {})
        rk = rn = sk = sn = ra = 0
        for key, (k, n, by_agent) in sorted(conds.items()):
            s = sim_ck.get(key)
            rows.append(SimRealRow(checkpoint_id=ck, condition=key, real=_rate(k, n), agent_logged=by_agent,
                                   sim=_rate(*s) if s else None, gap=(k / n - s[0] / s[1]) if s and s[1] else None))
            rk, rn, ra = rk + k, rn + n, ra + by_agent
            if s:
                sk, sn = sk + s[0], sn + s[1]
        real = _rate(rk, rn)
        simr = _rate(sk, sn) if sn else None
        per_ck.append(SimRealRow(checkpoint_id=ck, condition="all", real=real, sim=simr, agent_logged=ra,
                                 gap=(real.rate - simr.rate) if simr and real.rate is not None else None))
        if simr and real.rate is not None:  # only the conditions both measured: the same tasks on both sides
            pairs.append((simr.rate, real.rate))  # type: ignore[arg-type]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= MIN_CHECKPOINTS else None
    return SimRealReport(checkpoints=per_ck, rows=rows, spearman=rho, checkpoints_compared=len(pairs),
                         summary=_summary(len(trials), len(by), len(pairs), rho))


def _summary(n_trials: int, n_ck: int, n_pairs: int, rho: float | None) -> str:
    if not n_trials:
        return "No real trials yet: run a checkpoint on the arm and log each rollout."
    head = f"{n_trials} real trial{'s' if n_trials != 1 else ''} on {n_ck} checkpoint{'s' if n_ck != 1 else ''}."
    if rho is None:
        need = MIN_CHECKPOINTS - n_pairs
        return (f"{head} Sim-to-real agreement needs {need} more checkpoint{'s' if need != 1 else ''} with both "
                "real trials and a finished sim evaluation." if need > 0 else
                f"{head} Sim and real rates don't vary enough to rank yet.")
    word = "agree" if rho >= 0.7 else "partly agree" if rho >= 0.3 else "disagree"
    return f"{head} Sim and real rankings {word} (Spearman {rho:+.2f} over {n_pairs} checkpoints)."


def as_dict(rep: SimRealReport) -> dict[str, Any]:
    return rep.model_dump(mode="json")
