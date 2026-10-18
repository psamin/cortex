"""Read models for the studio's viewer, pipeline and metrics panels.

Everything here is derived from stored entities (episodes, attempts, checkpoints,
training jobs, evaluations, collections); nothing is synthesized. Iterations exist
only once a policy has been trained (or a training job is in flight).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..context import Ctx
from ..models import (
    AgentAttempt, AgentCollection, Checkpoint, DatasetVersion, Episode, EvaluationRun, SimulationBranch, TrainingJob,
)
from .recording_service import npz_key


async def rollouts(ctx: Ctx, pid: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in await ctx.repo.list("episodes", Episode, project_id=pid):
        if "front" not in e.artifacts:
            continue
        out.append({"kind": "demo", "id": e.id, "label": f"Demo #{e.index}", "success": e.success,
                    "status": str(e.status), "condition": e.condition.key, "seed": e.seed,
                    "planner": e.provenance.get("planner", "scripted_ik_oracle"),
                    "video_url": await ctx.media_url(e.artifacts["front"]),
                    "cameras": ["presentation", "front", "wrist"], "created_at": e.created_at})
    for a in await ctx.repo.list("agent_attempts", AgentAttempt, project_id=pid):
        if a.status == "failed" and "front" in a.artifacts:
            out.append({"kind": "attempt", "id": a.id, "label": f"Failed attempt ({a.collector})", "success": False,
                        "condition": a.scenario.condition.key, "seed": a.seed, "planner": a.model,
                        "failure": a.error or a.stop_reason, "video_url": await ctx.media_url(a.artifacts["front"]),
                        "cameras": ["presentation"], "created_at": a.created_at})
    versions = await policy_versions(ctx, pid)
    for run in await ctx.repo.list("evaluation_runs", EvaluationRun, project_id=pid):
        policy = versions.get(run.policy.checkpoint_id or "", "oracle reference")
        for bid in run.branch_ids:
            b = await ctx.repo.get("simulation_branches", bid, SimulationBranch)
            if not b or not b.result:
                continue
            for ep in b.result.episodes:
                if not ep.video_key:
                    continue
                out.append({"kind": "eval", "id": f"{run.id}/{b.id}/seed{ep.seed}",
                            "label": f"{policy} · {b.spec.name} · seed {ep.seed}", "success": ep.success,
                            "failure_mode": ep.failure_mode, "condition": b.spec.condition.key, "branch": b.spec.name,
                            "seed": ep.seed, "policy": policy, "evaluation_id": run.id,
                            "video_url": await ctx.media_url(ep.video_key), "cameras": ["presentation"],
                            "created_at": run.created_at})
    out.sort(key=lambda r: str(r["created_at"]), reverse=True)
    return out


async def policy_versions(ctx: Ctx, pid: str) -> dict[str, str]:
    cks = await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)
    return {ck.id: f"v{i + 1}" for i, ck in enumerate(cks)}


async def iterations(ctx: Ctx, pid: str) -> list[dict[str, Any]]:
    cks = await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)
    tjs = {t.id: t for t in await ctx.repo.list("training_jobs", TrainingJob, project_id=pid)}
    runs = await ctx.repo.list("evaluation_runs", EvaluationRun, project_id=pid)
    cols = await ctx.repo.list("agent_collections", AgentCollection, project_id=pid)
    out: list[dict[str, Any]] = []
    for i, ck in enumerate(cks):
        tj = tjs.get(ck.training_job_id or "")
        ds = await ctx.repo.get("dataset_versions", tj.dataset_version_id, DatasetVersion) if tj else None
        ev_runs = [r for r in runs if r.policy.checkpoint_id == ck.id]
        ev = await _eval_summary(ctx, ev_runs[-1]) if ev_runs else None
        nxt = None
        if ev_runs:
            after = [c for c in cols if c.created_at > ev_runs[-1].created_at and c.reason]
            if after:
                c = after[0]
                nxt = {"reason": c.reason, "collection_id": c.id, "status": c.status,
                       "conditions": [x.key for x in (c.request.conditions or [])],
                       "successes": c.totals.get("successes", 0)}
        out.append({"iteration": i + 1, "policy_version": f"v{i + 1}", "checkpoint_id": ck.id,
                    "demos_accepted": ds.num_source_episodes if ds else None, "dataset_id": ds.id if ds else None,
                    "training": _training(tj), "evaluation": ev, "next": nxt, "created_at": ck.created_at})
    # training in flight that has not produced a checkpoint yet
    done = {ck.training_job_id for ck in cks}
    for tj in tjs.values():
        if tj.id not in done and str(tj.status) in ("approved", "queued", "running", "proposed"):
            ds = await ctx.repo.get("dataset_versions", tj.dataset_version_id, DatasetVersion)
            out.append({"iteration": len(out) + 1, "policy_version": f"v{len(out) + 1} (pending)",
                        "checkpoint_id": None, "demos_accepted": ds.num_source_episodes if ds else None,
                        "dataset_id": tj.dataset_version_id, "training": _training(tj), "evaluation": None,
                        "next": None, "created_at": tj.created_at})
    return out


def _training(tj: TrainingJob | None) -> dict[str, Any] | None:
    if tj is None:
        return None
    return {"id": tj.id, "status": str(tj.status), "backend": tj.backend, "policy_type": tj.policy_type,
            "external_id": tj.external_id, "gpu_hours_est": tj.estimate.gpu_hours,
            "loss_curve": tj.metrics.get("loss_curve", []), "error": tj.error}


async def _eval_summary(ctx: Ctx, run: EvaluationRun) -> dict[str, Any]:
    branches = []
    for bid in run.branch_ids:
        b = await ctx.repo.get("simulation_branches", bid, SimulationBranch)
        if not b:
            continue
        r = b.result
        branches.append({"name": b.spec.name, "status": str(b.status), "successes": r.successes if r else None,
                         "n": r.n if r else None, "success_rate": r.success_rate if r else None,
                         "ci95": r.ci95 if r else None, "mean_steps": (float(np.mean([e.steps for e in r.episodes]))
                                                                      if r and r.episodes else None),
                         "failure_modes": dict(Counter(e.failure_mode for e in r.episodes if e.failure_mode))
                         if r else {}})
    rates = [b["success_rate"] for b in branches if b["success_rate"] is not None]
    return {"id": run.id, "status": str(run.status), "branches": branches,
            "overall_success_rate": float(np.mean(rates)) if rates else None}


async def camera_video(ctx: Ctx, episode_id: str, camera: str) -> Path:
    """The robot's own camera (what the policy sees), rendered from the recorded trajectory."""
    from ..workers.sim_worker import write_mp4

    if camera not in ("front", "wrist"):
        raise ValueError("camera must be front or wrist")
    ep = await ctx.repo.get("episodes", episode_id, Episode)
    if ep is None:
        raise KeyError(episode_id)
    src = await ctx.ensure_local(npz_key(ep))
    out = src.parent / f"cam_{camera}.mp4"
    if not out.exists():
        with np.load(src) as z:
            frames = z[camera]
        big = [np.repeat(np.repeat(f, 4, axis=0), 4, axis=1) for f in frames]  # 96x128 -> 384x512, pixel-true
        write_mp4(big, out)
    return out
