"""Record live GPU inference sessions into their project (#179).

The studio's GPU tab streams /api/live-policy/stream/<task>/<ep> (MJPEG from serve_policy.py). As the
stream passes through the API, StreamRecorder splits it into frames, keeps every other JPEG and every
frame's measurements (serve_policy.py sends them as part headers: X-Inference-Ms, X-Joint-Err-Deg,
X-Gripper-Err, X-Frame). Nothing is encoded while streaming. When the stream ends (finished, or the viewer
left) the session is encoded to mp4 in a thread, uploaded and stored as a PolicyReplay of kind
live_inference in the task's project (infra/futurama/projects.json).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np

from ..context import Ctx
from ..models import PolicyReplay, ReplayEpisode
from . import replay_service

log = logging.getLogger(__name__)
BOUNDARY = b"--frame"
MIN_FRAMES = 30  # about a second at 30 fps: shorter visits are clicks, not runs
KEEP_EVERY = 2  # the server streams at 30 fps; the saved video plays at 15 fps, in real time
MAX_KEPT = 1800  # 2 minutes of video at most in memory
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

    @property
    def complete(self) -> bool:
        return self.total is not None and self.last_index is not None and self.last_index + 1 >= self.total


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


async def save_session(ctx: Ctx, task: str, episode: int, rec: StreamRecorder, server: dict[str, Any],
                       job: str | None = None) -> PolicyReplay | None:
    """Store a finished (or abandoned) live session in the task's project. None when there is nothing to keep."""
    rec.close()
    pid = replay_service.project_for_task(ctx, task)
    if pid is None:
        log.info("live session %s/%s not saved: no project for this task in %s", task, episode,
                 ctx.settings.futurama_projects_file)
        return None
    if rec.frames < MIN_FRAMES or not rec.jpegs:
        return None
    pol = (server.get("policies") or {}).get(task, {})
    ck_name = str(pol.get("checkpoint", ""))
    ck = await replay_service.checkpoint_named(ctx, pid, ck_name) if ck_name else None
    rp = PolicyReplay(project_id=pid, kind="live_inference", status="complete" if rec.complete else "partial",
                      checkpoint_id=ck.id if ck else None, checkpoint_name=ck_name,
                      policy_type=ck.policy_type if ck else "smolvla", task=task, gpu=server.get("gpu"), episodes=[],
                      source=f"live GPU stream /live/{task}/{episode}" + (f", Slurm job {job}" if job else ""))
    video = replay_service.file_key(pid, rp.id, f"ep{episode}.mp4")
    await asyncio.to_thread(encode_mp4, rec.jpegs, ctx.local(video), 30 // KEEP_EVERY)
    await ctx.publish_file(video)
    trace = replay_service.file_key(pid, rp.id, f"ep{episode}_frames.json")
    ctx.local(trace).write_text(json.dumps({"inference_ms": rec.ms, "joint_err_deg": rec.err,
                                            "gripper_err": rec.grip, "frames": rec.frames, "total": rec.total}))
    await ctx.publish_file(trace)
    joint = replay_service.mean_or_none(rec.err)
    rp.episodes.append(ReplayEpisode(episode=episode, instruction=rec.instruction, frames=rec.frames,
                                     total_frames=rec.total, joint_mae_deg=joint,
                                     gripper_mae=replay_service.mean_or_none(rec.grip), inference_ms=_stats(rec.ms),
                                     video_key=video, trace_key=trace))
    rp.joint_mae_deg = joint
    return await replay_service.save(ctx, rp, actor="studio")


def save_soon(ctx: Ctx, task: str, episode: int, rec: StreamRecorder, server: dict[str, Any],
              job: str | None = None) -> None:
    """Schedule save_session after the stream ends; failures are logged, never raised into the response."""
    async def run() -> None:
        try:
            rp = await save_session(ctx, task, episode, rec, server, job)
            if rp:
                log.info("saved live session %s (%s ep %s, %d frames)", rp.id, task, episode, rec.frames)
        except Exception:
            log.exception("could not save the live session %s/%s", task, episode)

    t = asyncio.get_running_loop().create_task(run())
    _TASKS.add(t)
    t.add_done_callback(_TASKS.discard)
