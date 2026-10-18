"""Demonstration collection: run the sim recorder, store assets, register episodes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..context import Ctx
from ..errors import NotFound
from ..models import Condition, Episode, EpisodeSource, EpisodeStatus, Project, new_id
from .jobs import Job
from .quality_analysis import analyze_episode

SEED_BASE = 10_000
AUTO_ACCEPT_MIN_SCORE = 0.6
# display-only 640x480 copies of the policy cameras (cortex.sim.viewer.PolicyCameras), when the recorder wrote them
HD_FILES = (("front_hd", "front_hd.mp4"), ("wrist_hd", "wrist_hd.mp4"))


def episode_key(project_id: str, episode_id: str, name: str = "") -> str:
    return f"projects/{project_id}/episodes/{episode_id}/{name}".rstrip("/")


def live_key(project_id: str) -> str:
    return f"projects/{project_id}/live"


async def list_episodes(ctx: Ctx, project_id: str) -> list[Episode]:
    return await ctx.repo.list("episodes", Episode, project_id=project_id)


async def start_recording(ctx: Ctx, project: Project, conditions: list[Condition],
                          seeds: list[int] | None = None, auto_accept: bool = False,
                          actor: str = "gui", cycles: int = 0, reset_policy: str | None = None) -> Job:
    """Record oracle demos: one per condition, or with ``cycles`` the cycle protocol on the intake scene
    (demo, reset and recover segments, each successful segment its own tagged episode)."""
    if project.mode != "sim":
        raise PermissionError("Real-robot recording runs through the local LeRobot bridge and needs operator approval.")
    if cycles:
        return await start_cycle_recording(ctx, project, conditions[0] if conditions else Condition(), cycles,
                                           seeds[0] if seeds else None, auto_accept, actor, reset_policy)
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

        from .intake_service import scene_payload

        cfg = {
            "mode": "record", "max_steps": 240, "live_dir": str(ctx.local(live_key(project.id))),
            "scene": scene_payload(ctx, project),
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
                           auto_accept: bool, base_key: str | None = None) -> Episode:
    ep.num_frames = int(ev["num_frames"])
    ep.fps = int(ev["fps"])
    ep.success = bool(ev["success"])

    def key(name: str) -> str:
        return f"{base_key}/{name}" if base_key else episode_key(ep.project_id, ep.id, name)

    ep.artifacts = {name.split(".")[0]: key(name) for name in ("front.mp4", "thumb.jpg", "episode.npz")}
    ep.artifacts.update({kind: key(name) for kind, name in HD_FILES if ctx.local(key(name)).exists()})
    ep.quality = quality_for(ctx, ep, blur_ref)
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


# ------------------------------------------------------------------ cycle protocol (PRODUCT_SPEC 3.2)

def cycle_key(project_id: str, run_id: str, index: int | None = None) -> str:
    base = f"projects/{project_id}/cycles/{run_id}"
    return base if index is None else f"{base}/seg_{index:03d}"


def segment_provenance(run_id: str, ev: dict[str, Any], reset_policy: str) -> dict[str, Any]:
    """What a cycle segment's episode records about where it came from (cycle_id, index, start and end pose)."""
    return {"run_id": run_id, "cycle_id": f"{run_id}/{ev['cycle']}", "cycle": ev["cycle"],
            "segment": ev["segment"], "segment_index": ev["segment_index"], "goal": ev.get("goal"),
            "target": ev.get("target"), "start_pose": ev.get("start_pose"), "end_pose": ev.get("end_pose"),
            "reset_policy": reset_policy, "steps": ev.get("steps")}


def reset_policy_for(project: Project, override: str | None = None) -> str:
    return override or (project.intake.task.reset_policy if project.intake else "return_to_origin")


async def start_cycle_recording(ctx: Ctx, project: Project, cond: Condition, cycles: int, seed: int | None,
                                auto_accept: bool, actor: str, reset_policy: str | None = None) -> Job:
    """The scripted oracle runs ``cycles`` cycles in one sim (no teleport between them)."""
    from .intake_service import scene_payload

    scene = scene_payload(ctx, project)
    if scene is None:
        raise ValueError("the cycle protocol needs the project's intake scene: run build_simulation first")
    policy = reset_policy_for(project, reset_policy)
    existing = await list_episodes(ctx, project.id)
    seed = SEED_BASE + len(existing) if seed is None else seed
    run_id = new_id("cyc")
    key = f"{project.id}:cycles:{cond.key}@{seed}x{cycles}:{policy}"

    async def run(job: Job) -> str | None:
        done: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        expected = 2 * cycles
        blur_ref = await dataset_blur_reference(ctx, project.id)

        async def on_event(ev: dict[str, Any]) -> None:
            t = ev.get("type")
            if t == "segment_start":
                ctx.jobs.update(job, message=f"cycle {ev['cycle'] + 1}/{cycles}: {ev['segment']} segment")
            elif t == "progress":
                frac = (len(done) + len(diagnostics) + ev["step"] / ev["max_steps"]) / expected
                ctx.jobs.update(job, progress=min(0.99, frac))
            elif t == "segment" and ev["success"]:
                n = len(await list_episodes(ctx, project.id))
                ep = Episode(project_id=project.id, index=n, source=EpisodeSource.sim_scripted, condition=cond,
                             seed=seed, num_frames=0, fps=30, success=None, segment=ev["segment"],
                             provenance={"planner": "scripted_ik_oracle", "mode": "privileged",
                                         "note": "scripted IK skill reading simulator object poses",
                                         **segment_provenance(run_id, ev, policy)})
                await finalize_episode(ctx, ep, ev, blur_ref, auto_accept,
                                       base_key=cycle_key(project.id, run_id, ev["segment_index"]))
                done.append(ep.id)
            elif t == "segment":  # failed segments are kept as diagnostics (video + result.json), not episodes
                diagnostics.append({k: ev.get(k) for k in ("cycle", "segment", "segment_index", "failure_reason")})
            elif t in ("operator_reset_required", "cycles_done") and ev.get("stop_reason", ev.get("reason")):
                ctx.jobs.update(job, message=ev.get("stop_reason") or ev.get("reason"))

        cfg = {"mode": "record", "cycles": cycles, "reset_policy": policy, "seed": seed,
               "condition": cond.model_dump(), "max_steps": 240, "scene": scene,
               "live_dir": str(ctx.local(live_key(project.id))), "out_root": str(ctx.local(cycle_key(project.id, run_id)))}
        await ctx.jobs.run_worker(job, "cortex.workers.sim_worker", cfg,
                                  ctx.local(f"projects/{project.id}/jobs/{job.id}.json"), on_event)
        await ctx.audit(actor, "record_demonstrations", project.id,
                        {"cycles": cycles, "reset_policy": policy, "seed": seed, "run_id": run_id,
                         "condition": cond.key, "failed_segments": diagnostics}, "ok",
                        f"{cycles} oracle cycles: {len(done)} segments recorded, {len(diagnostics)} failed")
        return ",".join(done)

    return ctx.jobs.submit("record", project.id, run, idempotency_key=key)


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
        raise NotFound(episode_id)
    ep.status = status
    await ctx.repo.save("episodes", ep)
    ctx.changed(ep.project_id, "episodes", ep.id)
    await ctx.audit(actor, "set_episode_status", ep.project_id, {"episode_id": ep.id, "status": status}, "ok",
                    f"{ep.id} -> {status}")
    return ep
