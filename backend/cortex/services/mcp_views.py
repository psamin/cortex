"""Data for the MCP Apps views (#119): snapshots the MCP server renders inside the agent's chat.

Read-only snapshots (collection, evaluation, training review) plus one human-only path: a training
review issues a single-use approval nonce that the MCP server hands only to the view (result
`_meta`, which hosts keep out of the model's context). The view's Approve button sends it back.
The model never sees the nonce, so it cannot approve training through this path.
"""

from __future__ import annotations

import base64
import secrets
import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..context import Ctx
from ..errors import NotFound
from ..models import DatasetVersion, JobStatus
from . import agent_collection_service as acs
from . import recording_service, simulation_runner, training_service

NONCE_TTL_S = 600
_NONCES: dict[str, tuple[str, float]] = {}  # nonce -> (training job id, expiry)


def issue_nonce(tj_id: str) -> str:
    now = time.monotonic()
    for k in [k for k, (_, exp) in _NONCES.items() if exp < now]:
        del _NONCES[k]
    nonce = secrets.token_urlsafe(24)
    _NONCES[nonce] = (tj_id, now + NONCE_TTL_S)
    return nonce


def redeem_nonce(nonce: str, tj_id: str) -> bool:
    """Single use: a nonce is consumed by its first redemption attempt, right or wrong."""
    entry = _NONCES.pop(nonce, None)
    return entry is not None and entry[0] == tj_id and entry[1] >= time.monotonic()


class ViewArgs:
    class Collection(BaseModel):
        project_id: str
        collection_id: str | None = Field(None, description="defaults to the project's latest collection")

    class Evaluation(BaseModel):
        project_id: str
        evaluation_id: str | None = Field(None, description="defaults to the project's latest evaluation")

    class Review(BaseModel):
        project_id: str
        training_job_id: str | None = Field(None, description="defaults to the latest proposed training job")

    class Approve(BaseModel):
        project_id: str
        training_job_id: str
        nonce: str = Field(description="single-use approval nonce given only to the review view")
        decision: Literal["approve"] = "approve"


def _live_frame(ctx: Ctx, project_id: str) -> tuple[str | None, float | None]:
    path = ctx.local(f"{recording_service.live_key(project_id)}/live.jpg")
    try:
        data, age = path.read_bytes(), time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None, None
    return base64.b64encode(data).decode(), round(age, 1)


async def collection_snapshot(ctx: Ctx, a: ViewArgs.Collection, actor: str) -> dict[str, Any]:
    cols = await acs.list_collections(ctx, a.project_id)
    if a.collection_id:
        col = await acs.get_collection(ctx, a.collection_id)
    elif cols:
        col = max(cols, key=lambda c: c.created_at)
    else:
        raise NotFound(f"no collection in project {a.project_id}")
    rows = sorted(await acs.list_attempts(ctx, a.project_id, col.id), key=lambda r: r.index)
    frame, age = _live_frame(ctx, a.project_id)
    t = col.totals
    return {
        "summary": acs._summary(col),
        "ref_ids": [col.id],
        "project_id": a.project_id,
        "collection_id": col.id,
        "status": col.status,
        "collector": str(col.collector),
        "model": col.model,
        "provider": col.provider,
        "planned": len(col.scenarios),
        "attempts": int(t.get("attempts", 0)),
        "successes": int(t.get("successes", 0)),
        "model_calls": int(t.get("model_calls", 0)),
        "cost_usd": float(t.get("cost_usd", 0.0)),
        "budget_cap_usd": float(t.get("budget_cap_usd", 0.0)),
        "reason": col.reason,
        "error": col.error,
        "recent": [{"index": r.index, "status": r.status, "seed": r.seed,
                    "condition": f"{r.scenario.cube_region}/{r.scenario.lighting}",
                    "model_calls": r.usage.get("calls"), "cost_usd": r.usage.get("cost_usd"),
                    "stop_reason": r.stop_reason, "recoveries": r.recoveries} for r in rows[-12:]],
        "live_frame_b64": frame,
        "live_frame_age_s": age,
        "running": col.status in ("queued", "running"),
    }


def weakest_branch(done: list[dict[str, Any]]) -> str | None:
    """The finished branch with the lowest success rate, or None when every branch did equally well
    (all 3/3 has no weakest condition)."""
    if not done:
        return None
    rates = [b["success_rate"] for b in done]
    return None if min(rates) == max(rates) else min(done, key=lambda b: b["success_rate"])["name"]


async def evaluation_report(ctx: Ctx, a: ViewArgs.Evaluation, actor: str) -> dict[str, Any]:
    if a.evaluation_id:
        ev = await simulation_runner.get_evaluation(ctx, a.evaluation_id)
        if ev is None:
            raise NotFound(a.evaluation_id)
    else:
        evs = await simulation_runner.list_evaluations(ctx, a.project_id)
        if not evs:
            raise NotFound(f"no evaluation in project {a.project_id}")
        ev = max(evs, key=lambda e: e["run"].created_at)
    run = ev["run"]
    branches = []
    for b in ev["branches"]:
        r = b.result
        modes: dict[str, int] = {}
        for e in (r.episodes if r else []):
            if not e.success and e.failure_mode:
                modes[e.failure_mode] = modes.get(e.failure_mode, 0) + 1
        branches.append({"name": b.spec.name, "condition": b.spec.condition.model_dump(mode="json"),
                         "status": str(b.status), "n": r.n if r else 0, "successes": r.successes if r else 0,
                         "success_rate": r.success_rate if r else None,
                         "ci95": list(r.ci95) if r else None, "failure_modes": modes})
    done = [b for b in branches if b["n"]]
    weakest = weakest_branch(done)
    br = ", ".join(f"{b['name']} {b['successes']}/{b['n']}" for b in done) or "no finished branches"
    return {"summary": f"Evaluation {run.id} ({run.policy_label}): {br}", "ref_ids": [run.id],
            "project_id": a.project_id, "evaluation_id": run.id, "policy_label": run.policy_label,
            "policy_kind": run.policy.kind, "status": str(run.status),
            "running": run.status in (JobStatus.queued, JobStatus.running),
            "branches": branches, "weakest": weakest}


async def review_training_job(ctx: Ctx, a: ViewArgs.Review, actor: str) -> dict[str, Any]:
    from ..models import TrainingJob

    if a.training_job_id:
        tj = await training_service._get(ctx, a.training_job_id)
    else:
        jobs = await ctx.repo.list("training_jobs", TrainingJob, project_id=a.project_id)
        proposed = [j for j in jobs if j.status == JobStatus.proposed]
        if not proposed:
            raise NotFound(f"no proposed training job in project {a.project_id}")
        tj = max(proposed, key=lambda j: j.created_at)
    ds = await ctx.repo.get("dataset_versions", tj.dataset_version_id, DatasetVersion)
    awaiting = tj.status == JobStatus.proposed
    out = {
        "summary": (f"{tj.id}: {tj.policy_type.upper()} on {tj.backend}, ~{tj.estimate.gpu_hours} GPU-h, "
                    f"${tj.estimate.est_cost_usd}. " + ("Waiting for a human to approve it." if awaiting
                                                         else f"Status: {tj.status}.")),
        "ref_ids": [tj.id],
        "project_id": tj.project_id,
        "training_job_id": tj.id,
        "status": str(tj.status),
        "awaiting_approval": awaiting,
        "policy_type": tj.policy_type,
        "backend": tj.backend,
        "rationale": tj.rationale,
        "estimate": tj.estimate.model_dump(mode="json"),
        "steps": tj.config.get("steps"),
        "approved_by": tj.approved_by,
        "dataset": None if ds is None else {
            "id": ds.id, "name": ds.name, "source_episodes": ds.num_source_episodes, "frames": ds.num_frames,
            "train_episodes": sum(1 for ln in ds.links if ln.split == "train"),
            "eval_episodes": sum(1 for ln in ds.links if ln.split == "eval"),
            "augmentations": [x.type for x in ds.augmentations]},
    }
    if awaiting:
        # The MCP server moves this into view-only result _meta; it never reaches the model.
        out["view_approval_nonce"] = issue_nonce(tj.id)
    return out


async def approve_from_view(ctx: Ctx, a: ViewArgs.Approve, actor: str) -> dict[str, Any]:
    if not redeem_nonce(a.nonce, a.training_job_id):
        await ctx.audit(actor, "approve_training_from_view", a.project_id, {"id": a.training_job_id}, "denied",
                        "invalid, used or expired view approval nonce")
        raise training_service.ApprovalRequired(
            "This approval link is invalid, used or expired. Open the training review again and click Approve.")
    tj = await training_service.approve(ctx, a.training_job_id, "human click in the MCP App view", actor, None,
                                        human_verified=True)
    return {"summary": f"{tj.id} approved by a human in the MCP App view. Launch it with launch_training_job.",
            "ref_ids": [tj.id], "training_job_id": tj.id, "status": str(tj.status)}


def register(tools: dict[str, Any], tool_cls: Any) -> None:
    tools.update({t.name: t for t in [
        tool_cls("collection_snapshot", "Live snapshot of an agent demonstration collection for the MCP App view: "
                 "status, successes/attempts, model calls and spend, recent attempts and the latest sim frame.",
                 ViewArgs.Collection, collection_snapshot),
        tool_cls("evaluation_report", "Evaluation of a policy per condition, with Wilson 95% intervals, measured "
                 "failure modes and the weakest condition.", ViewArgs.Evaluation, evaluation_report),
        tool_cls("review_training_job", "Show a proposed training job (cost, GPU hours, dataset, rationale) for a "
                 "human to review and approve. Does not approve anything.", ViewArgs.Review, review_training_job),
        tool_cls("approve_training_from_view", "Human-only: approve a training job with the single-use nonce from "
                 "the review view. Agents never have the nonce.", ViewArgs.Approve, approve_from_view,
                 agent_callable=False),
    ]})
