"""Dataset analysis: quality checks, measured failure modes, VLM hypotheses,
and failure clusters. Evidence sources are kept separate and labelled:
  quality     deterministic checks on recorded data (measured)
  evaluation  failure modes from simulator telemetry (measured)
  vlm         model hypotheses from keyframes (unverified until a human confirms)
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from ..context import Ctx
from ..models import Analysis, EpisodeDiagnosis, FailureCluster, Project
from . import recording_service as rec
from . import simulation_runner as sim
from . import vlm_diagnostics as vlm
from .jobs import Job

TELEMETRY_LABEL = {
    "grasp_failed": "cube never lifted (grasp failed)",
    "dropped_before_target": "cube lifted but dropped before the box",
    "released_outside_target": "cube released near but outside the box",
}


async def start_analysis(ctx: Ctx, project: Project, use_vlm: bool = True, max_vlm: int = 6,
                         actor: str = "gui") -> Job:
    async def run(job: Job) -> str:
        an = Analysis(project_id=project.id, vlm_provider=ctx.vlm_source if use_vlm else None)
        await ctx.repo.save("analyses", an)
        eps = await rec.list_episodes(ctx, project.id)
        an.episode_count = len(eps)

        # 1) quality (recomputed against the current dataset blur reference)
        ctx.jobs.update(job, message="quality checks", progress=0.05)
        blur_ref = await rec.dataset_blur_reference(ctx, project.id)
        for i, ep in enumerate(eps):
            ep.quality = await asyncio.to_thread(rec.quality_for, ctx, ep, blur_ref)
            await ctx.repo.save("episodes", ep)
            an.quality[ep.id] = ep.quality
            ctx.jobs.update(job, progress=0.05 + 0.35 * (i + 1) / max(1, len(eps)))

        # 2) measured failures from the latest evaluation of each policy
        evals = await sim.list_evaluations(ctx, project.id)
        failures: list[dict[str, Any]] = []
        for ev in evals:
            for b in ev["branches"]:
                if not b.result:
                    continue
                for e in b.result.episodes:
                    if not e.success:
                        failures.append({"eval_id": ev["run"].id, "branch": b, "ep": e})

        # 3) VLM hypotheses on failed eval episodes with video, then failed/flagged demos
        if use_vlm and ctx.openrouter_key:
            targets: list[tuple[str, list, dict[str, Any]]] = []
            for f in failures:
                if f["ep"].video_key and len(targets) < max_vlm:
                    frames = await asyncio.to_thread(vlm.keyframes_from_video, ctx.local(f["ep"].video_key))
                    targets.append((f"{f['eval_id']}/{f['branch'].id}/seed{f['ep'].seed}", frames, {
                        "outcome": "failure", "condition": f["branch"].spec.name,
                        "cube_max_height_m": round(f["ep"].max_cube_z, 3),
                        "final_cube_to_box_m": round(f["ep"].final_cube_box_dist, 3)}))
            for ep in eps:
                if len(targets) >= max_vlm:
                    break
                if ep.success is False or (ep.quality and ep.quality.flags):
                    z = await asyncio.to_thread(rec.load_npz, ctx, ep)
                    targets.append((ep.id, vlm.keyframes_from_array(z["front"]), {
                        "outcome": "success" if ep.success else "failure",
                        "quality_flags": [fl.code for fl in ep.quality.flags] if ep.quality else []}))
            ctx.jobs.update(job, message=f"VLM review of {len(targets)} episodes", progress=0.5)
            diags = await asyncio.gather(*(vlm.diagnose(ctx.openrouter_key, ctx.settings.vlm_model, ref, fr,
                                                         project.spec.goal, tel) for ref, fr, tel in targets))
            an.diagnoses = list(diags)
        elif use_vlm:
            an.summary += "VLM skipped: no Model Gateway key configured. "

        # 4) clusters
        ctx.jobs.update(job, message="clustering failures", progress=0.9)
        clusters = build_clusters(project.id, an, eps, failures)
        for c in clusters:
            await ctx.repo.save("failure_clusters", c, status="current")
        for old in await ctx.repo.list("failure_clusters", FailureCluster, project_id=project.id):
            if old.analysis_id != an.id:
                await ctx.repo.save("failure_clusters", old, status="superseded")
        an.status = "succeeded"
        n_flag = sum(1 for q in an.quality.values() if q.flags)
        an.summary += (f"{len(eps)} episodes checked, {n_flag} with quality flags; "
                       f"{len(failures)} failed evaluation episodes; "
                       f"{sum(1 for d in an.diagnoses if d.hypotheses)} VLM reviews; {len(clusters)} clusters.")
        await ctx.repo.save("analyses", an)
        ctx.changed(project.id, "analyses", an.id)
        await ctx.audit(actor, "analyze_dataset", project.id, {"use_vlm": use_vlm}, "ok", an.summary)
        return an.id

    return ctx.jobs.submit("analyze", project.id, run)


def build_clusters(project_id: str, an: Analysis, eps, failures) -> list[FailureCluster]:
    out: list[FailureCluster] = []
    by_flag: dict[str, list] = defaultdict(list)
    for ep in eps:
        for fl in (ep.quality.flags if ep.quality else []):
            by_flag[fl.code].append((ep, fl))
    for code, items in by_flag.items():
        out.append(FailureCluster(
            project_id=project_id, analysis_id=an.id, category=f"quality:{code}", source="quality",
            episode_ids=[e.id for e, _ in items], conditions=sorted({e.condition.key for e, _ in items}),
            evidence=[f"{e.id}: {fl.message}" for e, fl in items[:5]], uncertainty=0.0))

    by_mode: dict[tuple[str, str], list] = defaultdict(list)
    for f in failures:
        by_mode[(f["ep"].failure_mode or "unknown", f["branch"].spec.name)].append(f)
    for (mode, branch), items in by_mode.items():
        b = items[0]["branch"]
        out.append(FailureCluster(
            project_id=project_id, analysis_id=an.id, category=f"eval:{mode}", source="evaluation",
            episode_ids=[f"{f['eval_id']}/{f['branch'].id}/seed{f['ep'].seed}" for f in items],
            conditions=[b.spec.condition.key + ("+shift" if b.spec.cube_offset != (0.0, 0.0) else "")],
            evidence=[f"{len(items)} failed trials in '{branch}': {TELEMETRY_LABEL.get(mode, mode)}"],
            uncertainty=0.0))

    by_cat: dict[str, list[tuple[EpisodeDiagnosis, float]]] = defaultdict(list)
    for d in an.diagnoses:
        if d.hypotheses:
            top = max(d.hypotheses, key=lambda h: h.confidence)
            by_cat[top.category].append((d, top.confidence))
    for cat, items in by_cat.items():
        if cat == "no_failure_visible":
            continue
        mean_conf = sum(c for _, c in items) / len(items)
        out.append(FailureCluster(
            project_id=project_id, analysis_id=an.id, category=f"vlm:{cat}", source="vlm",
            episode_ids=[d.episode_id for d, _ in items], conditions=[],
            evidence=[next(h.rationale for h in d.hypotheses if h.category == cat) for d, _ in items[:3]],
            uncertainty=round(1 - mean_conf, 3)))
    return out


async def label_diagnosis(ctx: Ctx, analysis_id: str, ref_id: str, status: str, label: str | None,
                          actor: str = "gui") -> EpisodeDiagnosis:
    an = await ctx.repo.get("analyses", analysis_id, Analysis)
    if an is None:
        raise KeyError(analysis_id)
    for d in an.diagnoses:
        if d.episode_id == ref_id:
            d.status = status  # type: ignore[assignment]
            d.human_label = label
            await ctx.repo.save("analyses", an)
            for c in await ctx.repo.list("failure_clusters", FailureCluster, project_id=an.project_id):
                if c.analysis_id == an.id and c.source == "vlm" and ref_id in c.episode_ids:
                    c.verified_by_human = status == "human_confirmed"
                    await ctx.repo.save("failure_clusters", c, status="current")
            ctx.changed(an.project_id, "analyses", an.id)
            await ctx.audit(actor, "label_diagnosis", an.project_id,
                            {"analysis_id": analysis_id, "ref": ref_id, "status": status, "label": label}, "ok",
                            f"{ref_id}: {status}")
            return d
    raise KeyError(ref_id)


async def latest_analysis(ctx: Ctx, project_id: str) -> Analysis | None:
    rows = await ctx.repo.list("analyses", Analysis, project_id=project_id, status="succeeded")
    return rows[-1] if rows else None


async def current_clusters(ctx: Ctx, project_id: str) -> list[FailureCluster]:
    return await ctx.repo.list("failure_clusters", FailureCluster, project_id=project_id, status="current")
