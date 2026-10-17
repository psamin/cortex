"""Demonstration collection: run the sim recorder, store assets, register episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..context import Ctx
from ..models import Condition, Episode, EpisodeSource, EpisodeStatus, Project
from .jobs import Job
from .quality_analysis import analyze_episode

SEED_BASE = 10_000
AUTO_ACCEPT_MIN_SCORE = 0.6


def episode_key(project_id: str, episode_id: str, name: str = "") -> str:
    return f"projects/{project_id}/episodes/{episode_id}/{name}".rstrip("/")


def live_key(project_id: str) -> str:
    return f"projects/{project_id}/live"


async def list_episodes(ctx: Ctx, project_id: str) -> list[Episode]:
    return await ctx.repo.list("episodes", Episode, project_id=project_id)


async def start_recording(ctx: Ctx, project: Project, conditions: list[Condition],
                          seeds: list[int] | None = None, auto_accept: bool = False,
                          actor: str = "gui") -> Job:
    if project.mode != "sim":
        raise PermissionError("Real-robot recording runs through the local LeRobot bridge and needs operator approval.")
    existing = await list_episodes(ctx, project.id)
    start = len(existing)
    if seeds is None:
        seeds = [SEED_BASE + start + i for i in range(len(conditions))]
    specs = []
    for i, (cond, seed) in enumerate(zip(conditions, seeds)):
        ep = Episode(project_id=project.id, index=start + i, source=EpisodeSource.sim_scripted,
                     condition=cond, seed=seed, num_frames=0, fps=30, success=None,
                     provenance={"planner": "scripted_ik_oracle", "mode": "privileged",
                                 "note": "scripted IK expert reading simulator object poses"})
        specs.append(ep)
    key = f"{project.id}:record:" + ",".join(f"{c.key}@{s}" for c, s in zip(conditions, seeds))

    async def run(job: Job) -> str | None:
        by_index = {i: ep for i, ep in enumerate(specs)}
        done: list[str] = []
        blur_ref = await dataset_blur_reference(ctx, project.id)

        async def on_event(ev: dict[str, Any]) -> None:
            if ev.get("type") == "episode_start":
                ctx.jobs.update(job, message=f"recording demo {ev['index'] + 1}/{ev['total']} "
                                f"({ev['condition']['cube_region']}, {ev['condition']['lighting']})")
            elif ev.get("type") == "progress":
                frac = (len(done) + ev["step"] / ev["max_steps"]) / len(specs)
                ctx.jobs.update(job, progress=min(0.99, frac))
            elif ev.get("type") == "episode":  # register each demo as soon as it is written
                ep = by_index[ev["index"]]
                await finalize_episode(ctx, ep, ev, blur_ref, auto_accept)
                done.append(ep.id)

        cfg = {
            "mode": "record", "max_steps": 240, "live_dir": str(ctx.local(live_key(project.id))),
            "episodes": [{"seed": ep.seed, "condition": ep.condition.model_dump(),
                          "out_dir": str(ctx.local(episode_key(project.id, ep.id)))} for ep in specs],
        }
        await ctx.jobs.run_worker(job, "cortex.workers.sim_worker", cfg,
                                  ctx.local(f"projects/{project.id}/jobs/{job.id}.json"), on_event)
        await ctx.audit(actor, "record_demonstrations", project.id,
                        {"conditions": [c.key for c in conditions], "seeds": seeds}, "ok",
                        f"recorded {len(done)} sim demos")
        return ",".join(done)

    return ctx.jobs.submit("record", project.id, run, idempotency_key=key)


async def finalize_episode(ctx: Ctx, ep: Episode, ev: dict[str, Any], blur_ref: float | None,
                           auto_accept: bool) -> Episode:
    ep.num_frames = int(ev["num_frames"])
    ep.fps = int(ev["fps"])
    ep.success = bool(ev["success"])
    ep.quality = quality_for(ctx, ep, blur_ref)
    ep.artifacts = {name.split(".")[0]: episode_key(ep.project_id, ep.id, name)
                    for name in ("front.mp4", "thumb.jpg", "episode.npz")}
    for k in ep.artifacts.values():
        await ctx.publish_file(k)
    if auto_accept and ep.success and ep.quality.score >= AUTO_ACCEPT_MIN_SCORE:
        ep.status = EpisodeStatus.accepted
    await ctx.repo.save("episodes", ep)
    for kind, k in ep.artifacts.items():
        p = ctx.local(k)
        await ctx.store.put("episode_artifacts", {
            "id": f"{ep.id}_{kind}", "project_id": ep.project_id, "status": "stored",
            "data": {"id": f"{ep.id}_{kind}", "episode_id": ep.id, "kind": kind, "storage_key": k,
                     "bytes": p.stat().st_size if p.exists() else None, "storage": ctx.blobs.name}})
    ctx.changed(ep.project_id, "episodes", ep.id)
    return ep


def npz_key(ep: Episode) -> str:
    """Oracle demos live under episodes/<id>/, agent demos under attempts/<id>/."""
    return ep.artifacts.get("episode") or episode_key(ep.project_id, ep.id, "episode.npz")


def load_npz(ctx: Ctx, ep: Episode) -> dict[str, np.ndarray]:
    with np.load(ctx.local(npz_key(ep))) as z:
        return {k: z[k] for k in z.files}


def quality_for(ctx: Ctx, ep: Episode, blur_ref: float | None):
    z = load_npz(ctx, ep)
    return analyze_episode(z["front"], z["timestamp"].astype(np.float64), z["action"], ep.fps,
                           z["action_low"], z["action_high"], blur_reference=blur_ref)


async def dataset_blur_reference(ctx: Ctx, project_id: str) -> float | None:
    vals = [e.quality.metrics.get("blur_laplacian_var") for e in await list_episodes(ctx, project_id)
            if e.quality and e.condition.lighting == "normal"]
    vals = [v for v in vals if v]
    return float(np.median(vals)) if vals else None


async def set_status(ctx: Ctx, episode_id: str, status: EpisodeStatus, actor: str = "gui") -> Episode:
    ep = await ctx.repo.get("episodes", episode_id, Episode)
    if ep is None:
        raise KeyError(episode_id)
    ep.status = status
    await ctx.repo.save("episodes", ep)
    ctx.changed(ep.project_id, "episodes", ep.id)
    await ctx.audit(actor, "set_episode_status", ep.project_id, {"episode_id": ep.id, "status": status}, "ok",
                    f"{ep.id} -> {status}")
    return ep
