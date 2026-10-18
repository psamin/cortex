"""#225: a collection whose first attempts all fail stops with a clear message instead of burning the rest."""

import asyncio
import json
from types import SimpleNamespace

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Condition, CreateProjectRequest, StartAgentCollectionRequest
from cortex.services import agent_collection_service as acs
from cortex.services import agent_orchestrator as orch
from cortex.services import recording_service as rec
from cortex.workers.sim_worker import EARLY_STOP_AFTER, early_stop_reason


def test_only_a_run_of_first_failures_stops_a_collection():
    fail = (False, "grasp_failed")
    assert early_stop_reason([fail] * (EARLY_STOP_AFTER - 1)) is None
    assert early_stop_reason([(True, None)] + [fail] * (EARLY_STOP_AFTER - 1)) is None
    why = early_stop_reason([fail, fail, (False, "dropped_before_target")])
    assert why.startswith("Stopped early: the first 3 attempts all failed")
    assert "grasp_failed x2" in why and "dropped_before_target x1" in why
    assert early_stop_reason([fail] * (EARLY_STOP_AFTER + 1)) is None  # decided once, after the first few


class Idle:
    """Holds the arm still: every episode is a genuine grasp_failed in a few steps (real sim)."""

    name = "idle (test)"
    needs_pixels = False

    def reset(self, env, obs, seed):
        self.ex = SimpleNamespace(done=True)
        self.q = obs["agent_pos"].copy()

    def act(self, obs):
        return self.q


def test_the_oracle_recorder_stops_after_the_first_failures(tmp_path, monkeypatch, capsys):
    from cortex.sim import env_factory
    from cortex.workers import sim_worker

    monkeypatch.setattr(env_factory, "oracle", lambda scene: Idle())
    cfg = {"live_dir": str(tmp_path / "live"), "max_steps": 60,
           "episodes": [{"seed": 10_000 + i, "condition": {"cube_region": "center", "lighting": "normal"},
                         "out_dir": str(tmp_path / f"ep{i}")} for i in range(6)]}
    res = sim_worker.cmd_record(cfg)
    assert len(res["episodes"]) == EARLY_STOP_AFTER and not any(e["success"] for e in res["episodes"])
    assert "grasp_failed x3" in res["stopped_early"]
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    stop = [e for e in events if e["type"] == "stopped_early"]
    assert len(stop) == 1 and stop[0]["index"] == EARLY_STOP_AFTER - 1 and stop[0]["total"] == 6
    assert not (tmp_path / f"ep{EARLY_STOP_AFTER}").exists()  # the rest never ran


REASON = "Stopped early: the first 3 attempts all failed (grasp_failed x3), so the rest were not run."


async def _setup(tmp_path, monkeypatch):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))

    async def worker(job, module, cfg, cfg_path, on_event=None, env=None):
        await on_event({"type": "stopped_early", "reason": REASON, "index": 2, "total": 6})
    monkeypatch.setattr(ctx.jobs, "run_worker", worker)
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    return ctx, p


async def _finish(ctx, job_id):
    await asyncio.wait_for(asyncio.shield(ctx.jobs.tasks[job_id]), 10)
    return ctx.jobs.get(job_id)


async def test_record_demonstrations_fails_with_the_reason(tmp_path, monkeypatch):
    ctx, p = await _setup(tmp_path, monkeypatch)
    job = await rec.start_recording(ctx, p, [Condition()] * 6)
    j = await _finish(ctx, job.id)
    assert j.status == "failed" and j.error == REASON


async def test_an_oracle_collection_fails_with_the_reason(tmp_path, monkeypatch):
    ctx, p = await _setup(tmp_path, monkeypatch)
    col = await acs.start_collection(ctx, p, StartAgentCollectionRequest(num_demos=6, collector="oracle"))
    j = await _finish(ctx, col.job_id)
    col = await acs.get_collection(ctx, col.id)
    assert col.status == "failed" and col.error == REASON and col.totals["stopped_reason"] == REASON
    assert j.status == "failed" and j.error == REASON


async def test_a_claude_code_collection_stops_after_the_first_failures(tmp_path, monkeypatch):
    from cortex.services import claude_code_collector as ccc

    ctx, p = await _setup(tmp_path, monkeypatch)
    calls = []

    async def attempt(ctx, project, col, sc, seed, i, model=None):
        calls.append(i)
        return {"attempt_id": f"at_{i}", "finished_by_agent": i != 1, "turns": 3, "status": "failed"}

    async def yes(ctx, req):
        return True
    monkeypatch.setattr(ccc, "run_attempt", attempt)
    monkeypatch.setattr(acs, "use_claude_code", yes)
    col = await acs.start_collection(ctx, p, StartAgentCollectionRequest(num_demos=6))
    j = await _finish(ctx, col.job_id)
    col = await acs.get_collection(ctx, col.id)
    assert calls == [0, 1, 2]
    assert col.status == "failed" and col.error.startswith("Stopped early: the first 3 attempts all failed")
    assert "agent stopped without finishing x1" in col.error
    assert j.status == "failed" and j.error == col.error
