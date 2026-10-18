"""Real SO-101 runs from the studio (spec section 5): leader-arm teleop recording, and a trained
policy (a project checkpoint) driving the arm.

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
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..context import Ctx
from ..hardware import calibrate
from ..hardware import setup as hw
from ..models import Checkpoint, DatasetVersion, Episode, Project, TrainingJob, new_id
from .jobs import BACKEND_DIR


class TeleopRequest(BaseModel):
    episodes: int = Field(5, ge=1, le=50)
    episode_s: float = Field(30.0, ge=3, le=300)
    task: str | None = Field(None, description="defaults to the project's task text")
    operator_confirmed: bool = Field(description="the operator is at the arm with the e-stop in reach, for this run")


class PolicyRunRequest(BaseModel):
    checkpoint_id: str
    episodes: int = Field(1, ge=1, le=20)
    episode_s: float = Field(20.0, ge=3, le=120)
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
    warnings: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)


RUN: Run | None = None


def active() -> bool:
    return RUN is not None and RUN.proc.returncode is None


def worker_cmd(cfg_path: str) -> list[str]:
    return [sys.executable, "-m", "cortex.hardware.arm_worker", cfg_path]


def _check_ready(ctx: Ctx, confirmed: bool, teleop: bool) -> None:
    if not confirmed:
        raise hw.HardwareRefused("Confirm you are at the arm with the e-stop in reach to start this run.")
    st = hw.status(ctx.data_dir, ctx.settings.cortex_enable_hardware)
    missing = st["missing"] + (st["teleop_missing"] if teleop else [])
    if missing:
        raise hw.HardwareRefused("Not ready for real motion: " + "; ".join(missing) + ".")
    if active() or calibrate.busy():
        raise hw.HardwareRefused("Another hardware run is in progress; stop it first.")


async def _launch(ctx: Ctx, project: Project, kind: str, run_conf: dict[str, Any], extra: dict[str, Any],
                  warnings: list[str] | None = None) -> Run:
    global RUN
    cfg = hw.load_config(ctx.data_dir)
    run_id = new_id("hr")
    base = ctx.local(f"projects/{project.id}/hardware/{run_id}")
    base.mkdir(parents=True, exist_ok=True)
    conf = {"run": {**run_conf, "max_step_deg": cfg.max_step_deg}, **cfg.model_dump(),
            "cameras": [c.model_dump() for c in cfg.cameras], **extra}
    if extra.get("record"):
        conf.update(dataset_root=str(base / "lerobot"), repo_id=f"cortex/{run_id}")
    (base / "config.json").write_text(json.dumps(conf))
    proc = await asyncio.create_subprocess_exec(*worker_cmd(str(base / "config.json")), cwd=str(BACKEND_DIR),
                                                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    RUN = Run(id=run_id, project_id=project.id, kind=kind, proc=proc, warnings=warnings or [])
    asyncio.get_running_loop().create_task(_pump(ctx, project, RUN))
    return RUN


async def start_teleop(ctx: Ctx, project: Project, req: TeleopRequest) -> dict[str, Any]:
    _check_ready(ctx, req.operator_confirmed, teleop=True)
    run = await _launch(ctx, project, "teleop", {"mode": "teleop", "episodes": req.episodes,
                                                 "episode_s": req.episode_s},
                        {"record": True, "task": req.task or project.task_prompt})
    await ctx.audit("gui", "hardware_teleop", project.id, req.model_dump(), "ok",
                    f"{run.id}: {req.episodes} teleop episode(s) of {req.episode_s:.0f} s, operator confirmed")
    return snapshot()


async def start_policy(ctx: Ctx, project: Project, req: PolicyRunRequest) -> dict[str, Any]:
    _check_ready(ctx, req.operator_confirmed, teleop=False)
    ck = await ctx.repo.get("checkpoints", req.checkpoint_id, Checkpoint)
    if ck is None or ck.project_id != project.id:
        raise KeyError(f"checkpoint {req.checkpoint_id}")
    path = Path(ck.path) if Path(ck.path).is_absolute() else ctx.local(ck.path)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"checkpoint files for {ck.id} are not on this machine ({path})")
    run = await _launch(ctx, project, "policy", {"mode": "policy", "episodes": req.episodes,
                                                 "episode_s": req.episode_s},
                        {"record": False, "checkpoint_path": str(path)}, await _transfer_warnings(ctx, ck))
    await ctx.audit("gui", "hardware_policy_run", project.id, req.model_dump(), "ok",
                    f"{run.id}: checkpoint {ck.id} on the real arm, {req.episodes} x {req.episode_s:.0f} s, "
                    f"operator confirmed")
    return snapshot()


async def _transfer_warnings(ctx: Ctx, ck: Checkpoint) -> list[str]:
    """A policy trained only on simulator demos rarely works on the real arm: say so up front."""
    tj = await ctx.repo.get("training_jobs", ck.training_job_id or "", TrainingJob)
    ds = await ctx.repo.get("dataset_versions", tj.dataset_version_id, DatasetVersion) if tj else None
    if ds is None:
        return ["Couldn't tell what this policy was trained on."]
    sources = set()
    for link in ds.links:
        ep = await ctx.repo.get("episodes", link.episode_id, Episode)
        if ep:
            sources.add(str(ep.source))
    if sources and sources <= {"sim_scripted", "sim_agent", "sim_teleop"}:
        return ["This policy was trained only on simulator demos. Expect it to behave differently on the real arm; "
                "keep a hand on the deadman."]
    return []


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
    return {"run_id": RUN.id, "project_id": RUN.project_id, "kind": RUN.kind, "state": state, "warnings": RUN.warnings,
            "live": RUN.state, "holding": RUN.holding, "events": list(RUN.events)[-20:], "result": RUN.result,
            "import_job_id": RUN.import_job_id, "started_at": RUN.started_at}
