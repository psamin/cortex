"""The experiment lineage graph: demos -> datasets -> training -> checkpoints ->
evaluations -> recommendations -> new demos. Built from the persisted rows
(InsForge Postgres, or the SQLite fallback), so every node points at the table
row and storage object it came from."""

from __future__ import annotations

from typing import Any

from ..context import Ctx
from ..models import (
    Analysis, AutopilotRun, Checkpoint, DatasetVersion, Episode, EvaluationRun, FailureCluster, Recommendation,
    SimulationBranch, TrainingJob,
)
from .agent_orchestrator import get_project


async def build_graph(ctx: Ctx, project_id: str) -> dict[str, Any]:
    p = await get_project(ctx, project_id)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    def node(id: str, type: str, label: str, table: str | None, status: str | None = None, **meta: Any) -> None:
        nodes[id] = {"id": id, "type": type, "label": label, "status": status,
                     "source": {"backend": ctx.store.name, "table": table, "row_id": id if table else None},
                     "meta": meta}

    def edge(a: str, b: str, kind: str) -> None:
        if a in nodes and b in nodes:
            edges.append({"source": a, "target": b, "kind": kind})

    node(p.id, "project", p.name, "projects", "active", goal=p.spec.goal, mode=p.mode)
    for cell in p.plan.cells:
        cid = f"cond:{cell.condition.key}"
        node(cid, "condition", cell.condition.key, None, target=cell.target)
        edge(p.id, cid, "plans")

    eps = await ctx.repo.list("episodes", Episode, project_id=project_id)
    for e in eps:
        cid = f"cond:{e.condition.key}"
        if cid not in nodes:
            node(cid, "condition", e.condition.key, None, target=0)
            edge(p.id, cid, "plans")
        node(e.id, "episode", f"demo #{e.index}", "episodes", str(e.status), seed=e.seed, success=e.success,
             quality=e.quality.score if e.quality else None,
             flags=[f.code for f in e.quality.flags] if e.quality else [],
             storage={"bucket": ctx.blobs.name, "keys": list(e.artifacts.values())})
        edge(cid, e.id, "recorded_in")

    for an in await ctx.repo.list("analyses", Analysis, project_id=project_id):
        node(an.id, "analysis", "analysis", "analyses", an.status, summary=an.summary, vlm=an.vlm_provider)
        edge(p.id, an.id, "analyzed")
    for cl in await ctx.repo.list("failure_clusters", FailureCluster, project_id=project_id, status="current"):
        node(cl.id, "cluster", cl.category, "failure_clusters", cl.source, uncertainty=cl.uncertainty,
             verified=cl.verified_by_human, evidence=cl.evidence[:3])
        edge(cl.analysis_id, cl.id, "found")
        for ref in cl.episode_ids:
            edge(ref, cl.id, "member_of")
            if "/" in ref:  # eval episode ref: evid/branch/seedN -> link the branch
                edge(ref.split("/")[1], cl.id, "member_of")

    for ds in await ctx.repo.list("dataset_versions", DatasetVersion, project_id=project_id, status="ready"):
        n_aug = sum(1 for ln in ds.links if ln.augmentation)
        node(ds.id, "dataset", ds.name, "dataset_versions", "ready", source_episodes=ds.num_source_episodes,
             lerobot_episodes=len(ds.links), augmented=n_aug, frames=ds.num_frames,
             augmentations=[a.type for a in ds.augmentations],
             storage={"bucket": ctx.blobs.name, "prefix": ds.storage_prefix})
        for eid in {ln.episode_id for ln in ds.links}:
            edge(eid, ds.id, "in_dataset")
        if ds.parent_id:
            edge(ds.parent_id, ds.id, "derived_from")

    for tj in await ctx.repo.list("training_jobs", TrainingJob, project_id=project_id):
        node(tj.id, "training", f"{tj.policy_type.upper()} training", "training_jobs", str(tj.status),
             backend=tj.backend, slurm_job=tj.external_id, gpu_hours=tj.estimate.gpu_hours,
             approved_by=tj.approved_by)
        edge(tj.dataset_version_id, tj.id, "trains")
    for ck in await ctx.repo.list("checkpoints", Checkpoint, project_id=project_id):
        node(ck.id, "checkpoint", f"{ck.policy_type} @ {ck.step}", "checkpoints", "ready", path=ck.path)
        if ck.training_job_id:
            edge(ck.training_job_id, ck.id, "produced")

    for ev in await ctx.repo.list("evaluation_runs", EvaluationRun, project_id=project_id):
        node(ev.id, "evaluation", ev.name or "evaluation", "evaluation_runs", str(ev.status), policy=ev.policy_label)
        src = ev.policy.checkpoint_id if ev.policy.kind == "checkpoint" else "policy:expert"
        if src == "policy:expert" and src not in nodes:
            node(src, "checkpoint", "scripted expert (oracle)", None, "reference")
            edge(p.id, src, "reference")
        edge(src, ev.id, "evaluated_by")  # type: ignore[arg-type]
        for bid in ev.branch_ids:
            b = await ctx.repo.get("simulation_branches", bid, SimulationBranch)
            if not b:
                continue
            r = b.result
            node(b.id, "branch", b.spec.name, "simulation_branches", str(b.status),
                 success=f"{r.successes}/{r.n}" if r else None, success_rate=r.success_rate if r else None,
                 seeds=b.seeds)
            edge(ev.id, b.id, "branch")

    for rec in await ctx.repo.list("recommendations", Recommendation, project_id=project_id):
        if rec.status == "dismissed":
            continue
        node(rec.id, "recommendation", rec.title[:60], "recommendations", rec.status, action=rec.action,
             support=rec.support, score=rec.score)
        for evd in rec.evidence:
            for ref in evd.ref_ids[:8]:
                edge(ref.split("/")[-1] if ref.startswith("ev_") else ref, rec.id, f"evidence:{evd.kind}")
        if rec.condition:
            edge(rec.id, f"cond:{rec.condition.key}", "targets")

    for ap in await ctx.repo.list("autopilot_runs", AutopilotRun, project_id=project_id):
        node(ap.id, "autopilot", f"autopilot it{ap.iteration}", "autopilot_runs", ap.status, phase=ap.phase)
        edge(p.id, ap.id, "automates")

    seen = set()
    uniq = []
    for e in edges:
        k = (e["source"], e["target"], e["kind"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return {"nodes": list(nodes.values()), "edges": uniq,
            "backend": {"db": ctx.store.name, "storage": ctx.blobs.name}}
