"""Gather evidence from the database and produce ranked recommendations."""

from __future__ import annotations

import math

from ..context import Ctx
from ..models import (
    Checkpoint,
    Condition,
    DatasetVersion,
    EpisodeStatus,
    Evidence,
    Project,
    Recommendation,
)
from ..sim.so101 import REGION_XY
from . import analysis_service, simulation_runner
from .augment import ASSUMED_DIM_RATIO, LIGHTING_GAP, AugmentEvidence, describe, plan_augmentation
from .recommendation_engine import CellEvidence, cells_from_plan, recommend_demonstrations
from .recording_service import AUTO_ACCEPT_MIN_SCORE, list_episodes

MIN_TRAIN_EPISODES = 10


def region_of(base: str, offset: tuple[float, float]) -> str:
    """Nearest collection region to a (possibly shifted) cube position."""
    x, y = REGION_XY[base][0] + offset[0], REGION_XY[base][1] + offset[1]
    return min(REGION_XY, key=lambda r: math.dist((x, y), REGION_XY[r]))


async def recommend(ctx: Ctx, project: Project, budget: int | None = None, actor: str = "gui") -> list[Recommendation]:
    eps = await list_episodes(ctx, project.id)
    cells = cells_from_plan(project.plan.cells)
    for ep in eps:
        c = cells.get(ep.condition.key)
        if c is None or ep.status != EpisodeStatus.accepted or ep.source == "imported":
            continue  # imported real-world demos do not cover simulation conditions
        w = ep.quality.score if ep.quality else 1.0
        c.accepted += 1
        c.effective += 1.0 if w >= AUTO_ACCEPT_MIN_SCORE else w
        c.episode_ids.append(ep.id)
        if w < AUTO_ACCEPT_MIN_SCORE:
            c.low_quality_ids.append(ep.id)

    evals = await simulation_runner.list_evaluations(ctx, project.id)
    learned = [e for e in evals if e["run"].policy.kind == "checkpoint"]
    latest_learned = learned[-1] if learned else None
    if latest_learned:  # evidence comes from the most recent learned policy only
        for b in latest_learned["branches"]:
            if not b.result:
                continue
            region = region_of(b.spec.condition.cube_region, b.spec.cube_offset)
            key = Condition(cube_region=region, lighting=b.spec.condition.lighting).key  # type: ignore[arg-type]
            if key not in cells:
                cells[key] = CellEvidence(condition=Condition.model_validate(
                    {"cube_region": region, "lighting": b.spec.condition.lighting}), target=0)
            cells[key].eval_successes += b.result.successes
            cells[key].eval_trials += b.result.n
            cells[key].eval_ids.append(f"{latest_learned['run'].id}/{b.id}")

    for cl in await analysis_service.current_clusters(ctx, project.id):
        if cl.source == "vlm":
            for key in cl.conditions:
                if key in cells:
                    cells[key].vlm_mentions += len(cl.episode_ids)
                    cells[key].vlm_episode_ids += cl.episode_ids

    remaining = budget if budget is not None else max(0, project.plan.budget_total + 12 -
                                                      sum(1 for e in eps if e.status == EpisodeStatus.accepted))
    recs = recommend_demonstrations(project.id, list(cells.values()), remaining)
    recs += await pipeline_recommendations(ctx, project, eps, latest_learned)

    for old in await ctx.repo.list("recommendations", Recommendation, project_id=project.id, status="proposed"):
        old.status = "dismissed"
        await ctx.repo.save("recommendations", old)
    for r in recs:
        await ctx.repo.save("recommendations", r)
    ctx.changed(project.id, "recommendations", project.id)
    await ctx.audit(actor, "recommend_demonstrations", project.id, {"budget": remaining}, "ok",
                    f"{len(recs)} recommendations")
    return recs


def dim_brightness_ratio(eps) -> float | None:
    """Median frame brightness of recorded dim-light demos over normal-light ones (measured)."""
    def med(light: str) -> float | None:
        v = sorted(e.quality.metrics["brightness_mean"] for e in eps if e.condition.lighting == light
                   and e.quality and e.quality.metrics.get("brightness_mean"))
        return v[len(v) // 2] if v else None
    dim, normal = med("dim"), med("normal")
    return dim / normal if dim and normal else None


async def augment_recommendation(ctx: Ctx, project: Project, eps, latest_learned) -> Recommendation | None:
    """Augmentation chosen and sized from measured evidence (#102): a lighting gap in the latest
    learned-policy evaluation, the measured dimming of recorded frames, and human-confirmed VLM
    visibility clusters. Unconfirmed VLM hypotheses alone never produce an evidence-backed plan."""
    ev, evidence = AugmentEvidence(), []
    if latest_learned:
        res = {b.spec.name: b.result for b in latest_learned["branches"] if b.result}
        base, dim = res.get("baseline"), res.get("changed_lighting")
        if base and dim and base.n >= 5 and dim.n >= 5:
            ev.lighting_gap = base.success_rate - dim.success_rate
            if ev.lighting_gap >= LIGHTING_GAP:
                evidence.append(Evidence(kind="evaluation", ref_ids=[latest_learned["run"].id], value=ev.lighting_gap,
                                         description=f"baseline {base.successes}/{base.n} vs dim {dim.successes}/"
                                                     f"{dim.n} successes (95% CI dim {dim.ci95[0]:.0%}–{dim.ci95[1]:.0%})"))
    vis = [c for c in await analysis_service.current_clusters(ctx, project.id) if c.category == "vlm:poor_visibility"]
    ev.visibility_confirmed = any(c.verified_by_human for c in vis)
    if vis:
        evidence.append(Evidence(kind="vlm", ref_ids=[x for c in vis for x in c.episode_ids][:10],
                                 description=f"{sum(len(c.episode_ids) for c in vis)} rollouts with a poor-visibility "
                                             + ("diagnosis (confirmed by a human)" if ev.visibility_confirmed
                                                else "hypothesis (unverified)")))
    if ev.lighting_gap >= LIGHTING_GAP:
        ev.dim_brightness_ratio = dim_brightness_ratio(eps)
        evidence.append(Evidence(kind="quality", value=ev.dim_brightness_ratio, description=(
            f"recorded dim-light frames are {ev.dim_brightness_ratio:.0%} as bright as normal ones (measured)"
            if ev.dim_brightness_ratio else f"no dim-light demos to measure darkening; assumed {ASSUMED_DIM_RATIO:.0%}")))
    plan = plan_augmentation(ev)
    if not plan and not vis:
        return None
    backed = bool(plan)  # lighting gap (measured) or human-confirmed visibility
    what = (f"success drops {ev.lighting_gap:.0%} under dim light" if ev.lighting_gap >= LIGHTING_GAP
            else "poor visibility in failed rollouts")
    return Recommendation(
        project_id=project.id, action="augment", support="evidence_backed" if backed else "hypothesis",
        title=(f"Train on an augmented dataset version ({what}): {describe(plan)}" if backed else
               "Possible visibility problem (unverified VLM hypothesis): confirm it in Analyze to get an augmentation plan"),
        score=round(max(ev.lighting_gap, 0.3 if ev.visibility_confirmed else 0.1), 3),
        components=ev.as_components(), evidence=evidence)


async def pipeline_recommendations(ctx: Ctx, project: Project, eps, latest_learned) -> list[Recommendation]:
    out: list[Recommendation] = []
    accepted = [e for e in eps if e.status == EpisodeStatus.accepted]
    datasets = await ctx.repo.list("dataset_versions", DatasetVersion, project_id=project.id)
    ckpts = await ctx.repo.list("checkpoints", Checkpoint, project_id=project.id)

    rec = await augment_recommendation(ctx, project, eps, latest_learned)
    if rec:
        out.append(rec)

    if len(accepted) >= MIN_TRAIN_EPISODES:
        newest = datasets[-1] if datasets else None
        covered = {ln.episode_id for ln in newest.links} if newest else set()
        new_eps = [e for e in accepted if e.id not in covered]
        if not newest or new_eps:
            out.append(Recommendation(
                project_id=project.id, action="train",
                title=(f"Snapshot {len(accepted)} accepted demos as a dataset version and train ACT"
                       if not ckpts else f"Retrain ACT: {len(new_eps)} new accepted demos since the last dataset"),
                score=0.5, components={"new_episodes": float(len(new_eps))},
                evidence=[Evidence(kind="coverage", description=f"{len(accepted)} accepted demos, "
                                   f"{len(new_eps)} not yet in any dataset version",
                                   ref_ids=[e.id for e in new_eps[:20]])],
                support="evidence_backed"))
    evaluated = {e["run"].policy.checkpoint_id for e in ([latest_learned] if latest_learned else [])}
    for ck in ckpts:
        if ck.id not in evaluated and not any(
                e for e in await simulation_runner.list_evaluations(ctx, project.id)
                if e["run"].policy.checkpoint_id == ck.id):
            out.append(Recommendation(
                project_id=project.id, action="evaluate",
                title=f"Evaluate checkpoint {ck.id} under baseline, shifted-object and dim-lighting conditions",
                score=0.6, components={}, support="evidence_backed",
                evidence=[Evidence(kind="coverage", description="checkpoint has no simulation evaluation yet",
                                   ref_ids=[ck.id])]))
    return out


async def open_recommendations(ctx: Ctx, project_id: str) -> list[Recommendation]:
    return await ctx.repo.list("recommendations", Recommendation, project_id=project_id, status="proposed")


async def set_status(ctx: Ctx, rec_id: str, status: str, actor: str = "gui") -> Recommendation:
    r = await ctx.repo.get("recommendations", rec_id, Recommendation)
    if r is None:
        raise KeyError(rec_id)
    r.status = status  # type: ignore[assignment]
    await ctx.repo.save("recommendations", r)
    ctx.changed(r.project_id, "recommendations", r.id)
    await ctx.audit(actor, "set_recommendation_status", r.project_id, {"id": rec_id, "status": status}, "ok", r.title)
    return r
