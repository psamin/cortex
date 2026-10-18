import asyncio
import sys

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.hardware.setup import HardwareRefused
from cortex.models import CreateProjectRequest
from cortex.services import agent_orchestrator as orch
from cortex.services import hardware_service as hs

# The real control loop and LeRobot recorder, with a fake arm in place of the USB devices.
FAKE_WORKER = r"""
import json, sys, threading
from pathlib import Path
import numpy as np
from cortex.hardware import arm_worker as aw
from cortex.hardware.so101_bridge import SafetyController, SafetyLimits

class FakeArm:
    name = "so101_follower"
    def __init__(self): self.q = np.array([0, -60, 60, 35, 0, 30.0])
    observation_features = {**{k: float for k in aw.KEYS}, "front": (48, 64, 3)}
    action_features = {k: float for k in aw.KEYS}
    def get_observation(self):
        return {**{k: float(v) for k, v in zip(aw.KEYS, self.q)}, "front": np.full((48, 64, 3), 90, np.uint8)}
    def send_action(self, a): self.q = np.array([a[k] for k in aw.KEYS])
    def disconnect(self): pass

c = json.loads(Path(sys.argv[1]).read_text())
cfg = aw.RunConfig(**c["run"], first_deadman_wait_s=10)
ctl = aw.Controls(aw.Deadman())
threading.Thread(target=ctl.read, args=(sys.stdin,), daemon=True).start()
arm = FakeArm()
rec = aw.Recorder(Path(c["dataset_root"]), c["repo_id"], cfg.fps, arm, c["task"])
aw.emit(type="ready", mode="teleop", cameras=["front"])
safety = SafetyController(SafetyLimits(max_deg_per_tick=cfg.max_step_deg * 1.001))
out = aw.control(arm, lambda obs: aw.joints(obs) + np.r_[0.5, 0, 0, 0, 0, 0], ctl, cfg, safety, rec)
aw.emit(type="result", **out, dataset_root=c["dataset_root"])
"""


@pytest.fixture
def ready(monkeypatch):
    monkeypatch.setattr(hs.hw, "status", lambda data_dir, enabled: {"missing": [], "teleop_missing": []})
    monkeypatch.setattr(hs, "worker_cmd", lambda cfg_path: [sys.executable, "-c", FAKE_WORKER, cfg_path])
    hs.RUN = None


async def until(cond, timeout=60.0, every=0.05, tick=None):
    t0 = asyncio.get_running_loop().time()
    while not cond():
        assert asyncio.get_running_loop().time() - t0 < timeout, hs.snapshot()
        if tick:
            await tick()
        await asyncio.sleep(every)


def test_teleop_run_records_and_imports_real_demos(tmp_path, ready):
    async def run():
        ctx = await build_ctx(Settings(data_dir=tmp_path, cortex_enable_hardware=True, _env_file=None))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
        with pytest.raises(HardwareRefused, match="Confirm"):
            await hs.start_teleop(ctx, p, hs.TeleopRequest(operator_confirmed=False))
        short = hs.TeleopRequest.model_construct(episodes=2, episode_s=0.4, task=None, operator_confirmed=True)
        snap = await hs.start_teleop(ctx, p, short)  # real runs are validated to >= 3 s episodes
        assert snap["state"] == "running"
        with pytest.raises(HardwareRefused, match="in progress"):
            await hs.start_teleop(ctx, p, hs.TeleopRequest(operator_confirmed=True))
        # the operator holds the deadman (the studio relays ~20 Hz heartbeats) until the run holds
        await until(lambda: hs.snapshot()["state"] == "holding", tick=lambda: hs.send("heartbeat"))
        assert hs.snapshot()["holding"]["reason"] == "run complete"
        await hs.send("release")
        await until(lambda: hs.snapshot()["import_job_id"])
        job = ctx.jobs.get(hs.snapshot()["import_job_id"])
        await until(lambda: job.status in ("succeeded", "failed"), timeout=120)
        assert job.status == "succeeded", job.error_detail
        from cortex.services.recording_service import list_episodes
        eps = await list_episodes(ctx, p.id)
        assert len(eps) == 2 and all(e.provenance["planner"] == "human teleop (real SO-101)" for e in eps)
        assert all(e.provenance["source"].startswith("teleop:hr_") for e in eps)
        from cortex.services.studio_service import cameras_for
        assert cameras_for("demo", eps[0].provenance)[0]["label"] == "front camera (recorded)"
    asyncio.run(run())


def test_not_ready_lists_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hs.hw, "status", lambda d, e: {"missing": ["follower arm 'a' is not calibrated"],
                                                       "teleop_missing": ["leader arm port not chosen"]})
    hs.RUN = None

    async def run():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube"))
        with pytest.raises(HardwareRefused, match="not calibrated; leader arm port not chosen"):
            await hs.start_teleop(ctx, p, hs.TeleopRequest(operator_confirmed=True))
        with pytest.raises(HardwareRefused, match="No hardware run"):
            await hs.send("heartbeat")
    asyncio.run(run())
