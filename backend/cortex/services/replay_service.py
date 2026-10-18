"""Policy replays (#179): trained policies run open loop on held-out recorded demos.

Two producers: a Futurama replay job (infra/futurama/replay_policy.py writes ep<N>.mp4 + metrics.json,
imported with scripts/import_replay.py) and a live GPU session streamed to the studio (recorded by
live_policy_service). Both measure joint error against the human's actions on episodes the policy never
trained on. Neither moves an arm, so neither claims a success rate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..context import Ctx
from ..errors import NotFound
from ..models import REPLAY_NOTE, Checkpoint, PolicyReplay, Project, ReplayEpisode

TABLE = "policy_replays"
CAMERA = {"id": "presentation", "label": "Recorded camera + joint targets",
          "detail": "A held-out demo's recorded camera, with the policy's joint targets drawn against the "
                    "human's. Open loop: the arm did not move."}


def file_key(pid: str, replay_id: str, name: str) -> str:
    return f"projects/{pid}/replays/{replay_id}/{name}"


async def save(ctx: Ctx, rp: PolicyReplay, actor: str = "cortex") -> PolicyReplay:
    await ctx.repo.save(TABLE, rp)
    ctx.changed(rp.project_id, TABLE, rp.id)
    err = f"{rp.joint_mae_deg:.2f} deg mean joint error" if rp.joint_mae_deg is not None else "no joint error"
    await ctx.audit(actor, f"record_{rp.kind}", rp.project_id, {"task": rp.task, "checkpoint": rp.checkpoint_name},
                    "ok", f"{rp.id}: {len(rp.episodes)} held-out episode(s), {err} ({rp.source})")
    return rp


async def list_replays(ctx: Ctx, pid: str) -> list[PolicyReplay]:
    return await ctx.repo.list(TABLE, PolicyReplay, project_id=pid)


async def store_file(ctx: Ctx, key: str, src: Path) -> str:
    """Copy a file into the data dir under `key` and upload it to durable storage."""
    dst = ctx.local(key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    await ctx.publish_file(key)
    return key


async def checkpoint_named(ctx: Ctx, pid: str, name: str) -> Checkpoint | None:
    """The project's checkpoint whose directory has this name (e.g. jenga_010000), newest first."""
    cks = await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)
    return next((c for c in reversed(cks) if Path(c.path.split(":", 1)[-1]).name == name), None)


def mean_or_none(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(float(np.mean(vals)), 3) if vals else None


async def import_replay_dir(ctx: Ctx, pid: str, folder: Path, source: str, actor: str = "cortex") -> PolicyReplay:
    """Import a replay_policy.py output folder (metrics.json + ep<N>.mp4) into a project.

    Re-importing the same source updates the earlier record instead of adding a second one.
    """
    if await ctx.repo.get("projects", pid, Project) is None:
        raise NotFound(pid)
    m = json.loads((folder / "metrics.json").read_text())
    ck_name = Path(str(m.get("checkpoint", ""))).name
    ck = await checkpoint_named(ctx, pid, ck_name) if ck_name else None
    prior = [r for r in await list_replays(ctx, pid) if r.kind == "held_out_replay" and r.source == source]
    rp = PolicyReplay(project_id=pid, kind="held_out_replay", task=str(m.get("task", "")), episodes=[],
                      checkpoint_id=ck.id if ck else None, checkpoint_name=ck_name,
                      policy_type=ck.policy_type if ck else "smolvla", source=source, note=REPLAY_NOTE)
    if prior:
        rp.id, rp.created_at = prior[-1].id, prior[-1].created_at
    for ep, e in sorted(m.get("episodes", {}).items(), key=lambda kv: int(kv[0])):
        video = folder / f"ep{ep}.mp4"
        key = await store_file(ctx, file_key(pid, rp.id, video.name), video) if video.exists() else None
        rp.episodes.append(ReplayEpisode(episode=int(ep), instruction=e.get("task", ""), frames=int(e["frames"]),
                                         total_frames=int(e["frames"]), joint_mae_deg=e.get("joint_mae_deg"),
                                         gripper_mae=e.get("gripper_mae"), video_key=key))
    rp.joint_mae_deg = m.get("overall_joint_mae_deg", mean_or_none([e.joint_mae_deg for e in rp.episodes]))
    return await save(ctx, rp, actor)


# ------------------------------------------------------------------ read models
def _metric(e: ReplayEpisode) -> str | None:
    parts = []
    if e.joint_mae_deg is not None:
        parts.append(f"{e.joint_mae_deg:.2f}° joint error")
    if e.inference_ms.get("mean") is not None:
        parts.append(f"{e.inference_ms['mean']:.0f} ms/frame")
    return " · ".join(parts) or None


async def rollout_rows(ctx: Ctx, pid: str, versions: dict[str, str]) -> list[dict[str, Any]]:
    """Viewer rows for every replayed episode that has a video. success is None: nothing was judged."""
    out = []
    for rp in await list_replays(ctx, pid):
        live = rp.kind == "live_inference"
        policy = versions.get(rp.checkpoint_id or "") or rp.checkpoint_name or "policy"
        for e in rp.episodes:
            if not e.video_key:
                continue
            out.append({"kind": "replay", "id": f"{rp.id}/ep{e.episode}", "replay_id": rp.id, "replay_kind": rp.kind,
                        "label": f"{'Live GPU' if live else 'Held-out replay'} · ep {e.episode}",
                        "success": None, "policy": policy, "metric": _metric(e), "joint_mae_deg": e.joint_mae_deg,
                        "inference_ms": e.inference_ms.get("mean"), "episode": e.episode,
                        "instruction": e.instruction, "frames": e.frames, "partial": rp.status == "partial",
                        "note": rp.note, "video_url": await ctx.media_url(e.video_key), "cameras": [CAMERA],
                        "created_at": rp.created_at})
    return out


def summary(rps: list[PolicyReplay]) -> dict[str, Any] | None:
    """A checkpoint's replays in one line: the latest held-out replay and the live GPU sessions."""
    if not rps:
        return None
    held = [r for r in rps if r.kind == "held_out_replay"]
    live = [r for r in rps if r.kind == "live_inference"]
    last = held[-1] if held else None
    ms = [e.inference_ms.get("mean") for r in live for e in r.episodes]
    return {"id": last.id if last else None, "episodes": len(last.episodes) if last else 0,
            "joint_mae_deg": last.joint_mae_deg if last else None, "live_sessions": len(live),
            "inference_ms": mean_or_none(ms), "note": REPLAY_NOTE}


def brief(rp: PolicyReplay) -> dict[str, Any]:
    """For get_project_state: enough for an agent to report it correctly."""
    return {"id": rp.id, "kind": rp.kind, "task": rp.task, "checkpoint_id": rp.checkpoint_id,
            "checkpoint": rp.checkpoint_name, "episodes": [e.episode for e in rp.episodes],
            "joint_mae_deg": rp.joint_mae_deg, "status": rp.status, "note": rp.note}
