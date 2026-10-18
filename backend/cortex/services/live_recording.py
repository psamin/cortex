"""Record live GPU inference sessions into their project (#179).

The studio streams /api/live-policy/stream/<task>/<ep> (a policy on a held-out real demo) and
/api/live-policy/sim/<task>/<seed> (a sim-trained policy closed loop in MuJoCo), MJPEG from serve_policy.py.
As a stream passes through the API, StreamRecorder splits it into frames, keeps every other JPEG and every
frame's measurements, which the server sends as part headers: X-Frame, X-Inference-Ms, X-Joint-Err-Deg and
X-Gripper-Err (held-out demos), X-Success and X-Policy (sim). Nothing is encoded while streaming. When the
stream ends (finished, or the viewer left) the session is encoded to mp4 in a thread, uploaded and stored as
a PolicyReplay (live_inference, or live_sim) in the task's project from infra/futurama/projects.json
(<task>, or sim_<task>). A live sim run whose task has no project (e.g. a sim skill the GPU server serves that
was never imported) is kept as a Physical Skills replay instead: <data dir>/skill_replays/<task>/live_*.mp4.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np

from ..context import Ctx
from ..models import LIVE_SIM_NOTE, PolicyReplay, ReplayEpisode
from . import replay_service, skills_service

log = logging.getLogger(__name__)
BOUNDARY = b"--frame"
MIN_FRAMES = 30  # about a second at 30 fps: shorter visits are clicks, not runs
KEEP_EVERY = 2  # the server streams at 30 fps; the saved video plays at 15 fps, in real time
MAX_KEPT = 1800  # 2 minutes of video at most in memory
KEEP_LIVE = 20  # live_*.mp4 kept per task in skill_replays; other files there are curated and never deleted
_TASKS: set[asyncio.Task] = set()


class StreamRecorder:
    """Splits an MJPEG stream into frames as it passes through, without decoding anything."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.jpegs: list[bytes] = []
        self.ms: list[float] = []
        self.err: list[float] = []
        self.grip: list[float] = []
        self.frames = 0
        self.last_index: int | None = None
        self.total: int | None = None
        self.instruction = ""
        self.success_frames: list[int] = []
        self.policy = ""  # sim streams: X-Policy, <variant>@<step>
        self.finished = False  # set by the proxy when the upstream connection closed cleanly

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        while (part := self._next_part()) is not None:
            self._frame(*part)

    def close(self) -> None:
        """The stream ended: a final part without Content-Length runs to the end of the buffer."""
        start = self.buf.find(BOUNDARY)
        end = self.buf.find(b"\r\n\r\n", start) if start >= 0 else -1
        if end >= 0:
            headers = _headers(self.buf[start + len(BOUNDARY):end])
            body = bytes(self.buf[end + 4:]).rstrip(b"\r\n-")
            if "content-length" not in headers and body.endswith(b"\xff\xd9"):
                self._frame(headers, body)
        self.buf.clear()

    def _next_part(self) -> tuple[dict[str, str], bytes] | None:
        start = self.buf.find(BOUNDARY)
        if start < 0:
            return None
        end = self.buf.find(b"\r\n\r\n", start)
        if end < 0:
            return None
        headers, body_at = _headers(self.buf[start + len(BOUNDARY):end]), end + 4
        if not headers.get("content-length", "").isdigit():
            headers.pop("content-length", None)
        if "content-length" in headers:
            n = int(headers["content-length"])
            if len(self.buf) < body_at + n:
                return None
            body = bytes(self.buf[body_at:body_at + n])
            del self.buf[:body_at + n]
        else:  # older servers: the part runs to the next boundary
            nxt = self.buf.find(b"\r\n" + BOUNDARY, body_at)
            if nxt < 0:
                return None
            body = bytes(self.buf[body_at:nxt])
            del self.buf[:nxt + 2]
        return headers, body

    def _frame(self, headers: dict[str, str], jpeg: bytes) -> None:
        if not jpeg:
            return
        if self.frames % KEEP_EVERY == 0 and len(self.jpegs) < MAX_KEPT:
            self.jpegs.append(jpeg)
        self.frames += 1
        for key, out in (("x-inference-ms", self.ms), ("x-joint-err-deg", self.err), ("x-gripper-err", self.grip)):
            if key in headers:
                try:
                    out.append(float(headers[key]))
                except ValueError:
                    pass
        if "/" in headers.get("x-frame", ""):
            i, n = headers["x-frame"].split("/", 1)
            if i.strip().isdigit() and n.strip().isdigit():
                self.last_index, self.total = int(i), int(n)
        if not self.instruction and headers.get("x-instruction"):
            self.instruction = unquote(headers["x-instruction"])
        if headers.get("x-success") in ("0", "1"):  # sim streams: the simulator's verdict so far
            self.success_frames.append(int(headers["x-success"]))
        if headers.get("x-policy"):
            self.policy = headers["x-policy"]

    @property
    def success(self) -> bool | None:
        """Sim streams: the verdict at the last frame received."""
        return bool(self.success_frames[-1]) if self.success_frames else None

    @property
    def complete(self) -> bool:
        """The run reached its end: the episode's last frame (X-Frame i/n) or, in sim, the simulator's verified
        success (the server stops shortly after it). A closed connection proves nothing by itself: a dropped
        tunnel or Stop GPU looks the same as a finished stream. Servers without X-Frame: the stream closed."""
        if self.total is not None and self.last_index is not None:
            return self.last_index + 1 >= self.total or self.success is True
        return self.finished


def _headers(raw: bytes | bytearray) -> dict[str, str]:
    out = {}
    for line in bytes(raw).decode("latin-1").split("\r\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs)
    return {"mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}


def encode_mp4(jpegs: list[bytes], path: Path, fps: int) -> None:
    import av
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        s = None
        for jpg in jpegs:
            img = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
            if s is None:
                s = out.add_stream("libx264", rate=fps)
                s.width, s.height = img.shape[1] // 2 * 2, img.shape[0] // 2 * 2
                s.pix_fmt = "yuv420p"
                s.options = {"crf": "23", "preset": "faster", "movflags": "+faststart"}
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(img[:s.height, :s.width]), format="rgb24")
            for pkt in s.encode(frame):
                out.mux(pkt)
        if s is not None:
            for pkt in s.encode():
                out.mux(pkt)


def keep_skill_replay(data_dir: Path, task: str, seed: int, jpegs: list[bytes]) -> Path:
    """Encode a live sim run into the Physical Skills replay folder of `task`, newest first in GET /api/skills.
    Written under a hidden name and renamed, so a half-written file is never listed. Keeps the newest KEEP_LIVE
    live_*.mp4; never touches the other (curated) replays."""
    if not skills_service.TASK.match(task):
        raise ValueError(f"bad task name {task!r}")
    d = data_dir / skills_service.REPLAYS / task
    name = f"live_{datetime.now(UTC):%Y%m%d-%H%M%S}_seed{seed}.mp4"
    tmp = d / f".{name}"  # replay_urls and replay_file skip dot files
    try:
        encode_mp4(jpegs, tmp, 30 // KEEP_EVERY)
        tmp.replace(d / name)
    finally:
        tmp.unlink(missing_ok=True)
    live = sorted(d.glob("live_*.mp4"), key=lambda f: (f.stat().st_mtime, f.name), reverse=True)
    for old in live[KEEP_LIVE:]:
        old.unlink(missing_ok=True)
    return d / name


async def save_session(ctx: Ctx, task: str, episode: int, rec: StreamRecorder, server: dict[str, Any],
                       job: str | None = None, kind: str = "live_inference") -> PolicyReplay | None:
    """Store a finished (or abandoned) live session in the task's project. None when there is nothing to keep.

    kind live_inference: /live/<task>/<episode> on a held-out real demo (project key <task>).
    kind live_sim: /sim/<task>/<seed>, closed loop in MuJoCo (project key sim_<task>; episode is the seed).
    A live_sim run with no project is kept as a Physical Skills replay instead (returns None)."""
    rec.close()
    sim = kind == "live_sim"
    pid = await replay_service.project_for(ctx, f"sim_{task}" if sim else task)
    if pid is None:
        if sim and rec.frames >= MIN_FRAMES and rec.jpegs:
            p = await asyncio.to_thread(keep_skill_replay, ctx.data_dir, task, episode, rec.jpegs)
            log.info("live sim %s seed %s kept as a Physical Skills replay %s (no project)", task, episode, p.name)
        else:
            log.info("live %s %s/%s not saved: no project for this task in %s", kind, task, episode,
                     ctx.settings.futurama_projects_file)
        return None
    if rec.frames < MIN_FRAMES or not rec.jpegs:
        return None
    if sim:
        s = (server.get("sim") or {}).get(task, {})
        ck_name = rec.policy or (f"{s['variant']}@{s.get('step', '')}" if s.get("variant") else "")
        ck = await replay_service.checkpoint_for_policy(ctx, pid, ck_name)
    else:
        ck_name = str((server.get("policies") or {}).get(task, {}).get("checkpoint", ""))
        ck = await replay_service.checkpoint_named(ctx, pid, ck_name) if ck_name else None
    what = f"/sim/{task}/{episode}" if sim else f"/live/{task}/{episode}"
    rp = PolicyReplay(project_id=pid, kind=kind, status="complete" if rec.complete else "partial",
                      checkpoint_id=ck.id if ck else None, checkpoint_name=ck_name,
                      policy_type=ck.policy_type if ck else ck_name.split("@")[0] or "smolvla", task=task,
                      gpu=server.get("gpu"), episodes=[],
                      source=f"live GPU stream {what}" + (f", Slurm job {job}" if job else ""))
    if sim:
        rp.note = LIVE_SIM_NOTE
    name = f"seed{episode}" if sim else f"ep{episode}"
    video = replay_service.file_key(pid, rp.id, f"{name}.mp4")
    await asyncio.to_thread(encode_mp4, rec.jpegs, ctx.local(video), 30 // KEEP_EVERY)
    await ctx.publish_file(video)
    trace = replay_service.file_key(pid, rp.id, f"{name}_frames.json")
    ctx.local(trace).write_text(json.dumps({"inference_ms": rec.ms, "joint_err_deg": rec.err,
                                            "gripper_err": rec.grip, "success": rec.success_frames,
                                            "frames": rec.frames, "total": rec.total}))
    await ctx.publish_file(trace)
    joint = replay_service.mean_or_none(rec.err)
    rp.episodes.append(ReplayEpisode(episode=episode, instruction=rec.instruction, frames=rec.frames,
                                     total_frames=rec.total, joint_mae_deg=joint,
                                     gripper_mae=replay_service.mean_or_none(rec.grip), inference_ms=_stats(rec.ms),
                                     video_key=video, trace_key=trace,
                                     success=rec.success if sim and rec.complete else None,
                                     steps=(rec.last_index + 1 if rec.last_index is not None else rec.frames)
                                     if sim else None))
    rp.joint_mae_deg = joint
    return await replay_service.save(ctx, rp, actor="studio")


def save_soon(ctx: Ctx, task: str, episode: int, rec: StreamRecorder, server: dict[str, Any],
              job: str | None = None, kind: str = "live_inference") -> None:
    """Schedule save_session after the stream ends; failures are logged, never raised into the response."""
    async def run() -> None:
        try:
            rp = await save_session(ctx, task, episode, rec, server, job, kind)
            if rp:
                log.info("saved live session %s (%s ep %s, %d frames)", rp.id, task, episode, rec.frames)
        except Exception:
            log.exception("could not save the live session %s/%s", task, episode)

    t = asyncio.get_running_loop().create_task(run())
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)
