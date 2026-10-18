"""Real SO-101 runs from the studio (spec section 5): leader-arm teleop recording.

A run needs every setup gate (CORTEX_ENABLE_HARDWARE=1, the follower plugged in and calibrated,
an armed operator), the leader arm for teleop, and the operator's confirmation for this run;
it is never started by a standing policy or an agent (these are not tools). The studio relays
the operator's held deadman as heartbeats; the arm worker stops within one control tick when
they lapse. Saved teleop episodes are imported as human demonstrations recorded on the real arm.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..context import Ctx
from ..hardware import calibrate
from ..hardware import setup as hw
from ..models import Project, new_id
from .jobs import BACKEND_DIR


class TeleopRequest(BaseModel):
    episodes: int = Field(5, ge=1, le=50)
    episode_s: float = Field(30.0, ge=3, le=300)
    task: str | None = Field(None, description="defaults to the project's task text")
    operator_confirmed: bool = Field(description="the operator is at the arm with the e-stop in reach, for this run")


@dataclass
class Run:
    id: str
    project_id: str
    kind: str
    proc: asyncio.subprocess.Process
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))
    state: dict[str, Any] | None = None
    holding: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    import_job_id: str | None = None
    started_at: float = field(default_factory=time.time)


RUN: Run | None = None


def active() -> bool:
    return RUN is not None and RUN.proc.returncode is None


def worker_cmd(cfg_path: str) -> list[str]:
    return [sys.executable, "-m", "cortex.hardware.arm_worker", cfg_path]


async def start_teleop(ctx: Ctx, project: Project, req: TeleopRequest) -> dict[str, Any]:
    global RUN
    if not req.operator_confirmed:
        raise hw.HardwareRefused("Confirm you are at the arm with the e-stop in reach to start this run.")
    st = hw.status(ctx.data_dir, ctx.settings.cortex_enable_hardware)
    missing = st["missing"] + st["teleop_missing"]
    if missing:
        raise hw.HardwareRefused("Not ready for real motion: " + "; ".join(missing) + ".")
    if active() or calibrate.busy():
        raise hw.HardwareRefused("Another hardware run is in progress; stop it first.")
    cfg = hw.load_config(ctx.data_dir)
    run_id = new_id("hr")
    base = ctx.local(f"projects/{project.id}/hardware/{run_id}")
    base.mkdir(parents=True, exist_ok=True)
    conf = {"run": {"mode": "teleop", "episodes": req.episodes, "episode_s": req.episode_s,
                    "max_step_deg": cfg.max_step_deg},
            **cfg.model_dump(), "cameras": [c.model_dump() for c in cfg.cameras], "record": True,
            "dataset_root": str(base / "lerobot"), "repo_id": f"cortex/{run_id}",
            "task": req.task or project.task_prompt}
    (base / "config.json").write_text(json.dumps(conf))
    proc = await asyncio.create_subprocess_exec(*worker_cmd(str(base / "config.json")), cwd=str(BACKEND_DIR),
                                                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    RUN = Run(id=run_id, project_id=project.id, kind="teleop", proc=proc)
    asyncio.get_running_loop().create_task(_pump(ctx, project, RUN))
    await ctx.audit("gui", "hardware_teleop", project.id, req.model_dump(), "ok",
                    f"{run_id}: {req.episodes} teleop episode(s) of {req.episode_s:.0f} s, operator confirmed")
    return snapshot()


async def _pump(ctx: Ctx, project: Project, run: Run) -> None:
    assert run.proc.stdout
    async for raw in run.proc.stdout:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "state":
            run.state = ev
        else:
            run.events.append(ev)
        if t == "holding":
            run.holding = ev
        elif t == "result":
            run.result = ev
        ctx.bus.publish(project.id, "hardware", {"run_id": run.id, **ev})
    await run.proc.wait()
    res = run.result or {}
    saved = res.get("episodes_saved") or []
    if res.get("ok") and saved and res.get("dataset_root"):
        from . import import_service

        job = await import_service.start_import(ctx, project, {"path": res["dataset_root"]}, len(saved), "gui",
                                                planner="human teleop (real SO-101)", label=f"teleop:{run.id}")
        run.import_job_id = job.id


async def send(op: Literal["heartbeat", "end_episode", "stop", "release"]) -> None:
    if not active():
        raise hw.HardwareRefused("No hardware run is in progress.")
    assert RUN and RUN.proc.stdin
    RUN.proc.stdin.write((json.dumps({"op": op}) + "\n").encode())
    await RUN.proc.stdin.drain()


def snapshot() -> dict[str, Any] | None:
    if RUN is None:
        return None
    state = "running" if active() and not RUN.holding else "holding" if active() else "finished"
    return {"run_id": RUN.id, "project_id": RUN.project_id, "kind": RUN.kind, "state": state,
            "live": RUN.state, "holding": RUN.holding, "events": list(RUN.events)[-20:], "result": RUN.result,
            "import_job_id": RUN.import_job_id, "started_at": RUN.started_at}
