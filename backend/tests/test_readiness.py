"""The readiness check (docs/STACK_SPEC.md 2.1): each finding trips on the problem it names and stays quiet on a
clean demo."""

import numpy as np
import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (
    Condition,
    CreateProjectRequest,
    Episode,
    EpisodeSource,
    EpisodeStatus,
)
from cortex.services import agent_orchestrator as orch
from cortex.services import readiness as rd
from cortex.services.recording_service import npz_key

FPS = 30


def demo(n=150, lead=0, tail=0, gripper=True, seed=0, scale=1.0) -> tuple[np.ndarray, np.ndarray]:
    """A smooth reach-grasp-lift: 5 joints moving, gripper closing mid-way. Returns (action, state)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    joints = np.stack([np.sin(t * np.pi * (1 + k * 0.3)) * 0.6 + rng.normal(0, 0.002, n) for k in range(5)], 1)
    g = np.where(t > 0.5, 0.0, 0.9) if gripper else np.full(n, 0.9)
    a = np.concatenate([joints, g[:, None]], 1) * scale
    a = np.concatenate([np.repeat(a[:1], lead, 0), a, np.repeat(a[-1:], tail, 0)])
    state = np.concatenate([a[:1].repeat(2, 0), a[:-2]])  # state follows the command by 2 frames
    return a, state


def codes(findings):
    return {f.code for f in findings}


def test_a_clean_demo_has_no_findings():
    a, s = demo()
    assert rd.episode_findings("ep", a, s, FPS, gripper=5) == []


def test_idle_edges_are_found():
    a, s = demo(lead=45, tail=60)
    f = {x.code: x for x in rd.episode_findings("ep", a, s, FPS, gripper=5)}
    assert f["idle_head"].evidence["seconds"] >= 1.4 and f["idle_tail"].evidence["seconds"] >= 1.9


def test_an_unused_gripper_is_found_only_when_checked():
    a, s = demo(gripper=False)
    assert "gripper_unused" in codes(rd.episode_findings("ep", a, s, FPS, gripper=5))
    assert "gripper_unused" not in codes(rd.episode_findings("ep", a, s, FPS, gripper=None))


def test_a_glitch_step_is_found():
    a, s = demo()
    a[80, 1] += 1.5
    assert "action_jump" in codes(rd.episode_findings("ep", a, s, FPS, gripper=5))


def test_actions_recorded_late_are_found():
    a, _ = demo()
    late = np.concatenate([a[5:], a[-1:].repeat(5, 0)])  # the "state" runs 5 frames ahead of the actions
    f = {x.code: x for x in rd.episode_findings("ep", a, late, FPS, gripper=5)}
    assert f["action_state_lag"].evidence["lag_frames"] <= -3


def _ep(i, source=EpisodeSource.sim_scripted, fps=FPS, cond=Condition()):
    return Episode(id=f"ep_{i}", project_id="p", index=i, source=source, condition=cond, seed=i, num_frames=10,
                   fps=fps, success=True, status=EpisodeStatus.accepted)


def test_dataset_level_errors_block():
    items = [(_ep(0), demo()[0]), (_ep(1, fps=15), demo()[0]), (_ep(2), demo()[0][:, :5])]
    found = {f.code: f for f in rd.dataset_findings(items, None)}
    assert found["fps_mismatch"].episode_ids == ["ep_1"] and found["shape_mismatch"].episode_ids == ["ep_2"]
    assert rd.verdict_of(list(found.values())) == "blocked"


def test_degrees_mixed_with_radians_is_found():
    sim = [(_ep(i), demo(seed=i)[0]) for i in range(3)]
    real = [(_ep(10 + i, EpisodeSource.imported), demo(seed=i, scale=57.3)[0]) for i in range(3)]
    found = {f.code: f for f in rd.dataset_findings(sim + real, None)}
    assert "unit_mismatch" in found and set(found["unit_mismatch"].episode_ids) == {"ep_10", "ep_11", "ep_12"}
    assert "unit_mismatch" not in codes(rd.dataset_findings(sim + [(_ep(9, EpisodeSource.imported), demo()[0])], None))


def test_thin_conditions_and_idle_data_warn():
    items = [(_ep(i, EpisodeSource.sim_teleop), demo(seed=i, lead=200)[0]) for i in range(3)]
    found = {f.code: f for f in rd.dataset_findings(items, {"center/normal": 3, "left/dim": 2})}
    assert "mostly_idle_data" in found
    assert found["few_demos"].evidence == {"left/dim": {"have": 0, "planned": 2}}  # center/normal met its target


def test_planner_holds_are_not_idle_data():
    """A scripted oracle settles before each grasp; that's half its steps and is meant to be there."""
    items = [(_ep(i), demo(seed=i, lead=200)[0]) for i in range(3)]
    assert "mostly_idle_data" not in codes(rd.dataset_findings(items, None))


def test_teleop_tremor_is_not_motion_or_a_glitch():
    """A teleop arm held still reads 0.1 degree of jitter; that's idle, and real motion isn't a glitch against it."""
    a, s = demo(n=300, lead=60, scale=57.3)  # degrees
    a = a + np.random.default_rng(3).normal(0, 0.1, a.shape)
    f = {x.code: x for x in rd.episode_findings("ep", a, s, FPS, gripper=5)}
    assert "action_jump" not in f
    assert f["idle_head"].evidence["seconds"] >= 1.5
    a[200, 1] += 40  # a real glitch still shows
    assert "action_jump" in codes(rd.episode_findings("ep", a, s, FPS, gripper=5))


async def test_check_reports_on_a_projects_episodes(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    for i in range(4):
        ep = _ep(i)
        ep.project_id = p.id
        a, s = demo(seed=i, lead=60 if i == 0 else 0)
        f = ctx.local(npz_key(ep))
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, action=a.astype(np.float32), state=s.astype(np.float32))
        await ctx.repo.save("episodes", ep)
    missing = _ep(9)
    missing.project_id = p.id
    await ctx.repo.save("episodes", missing)

    rep = await rd.check(ctx, p)
    found = {f.code: f for f in rep.findings}
    assert rep.episodes_checked == 5 and rep.verdict == "blocked"  # the unreadable file is an error
    assert found["unreadable"].episode_ids == ["ep_9"] and found["idle_head"].episode_ids == ["ep_0"]
    assert rep.summary.startswith("Blocked: 5 episodes checked")

    out = await orch.call_tool(ctx, "check_dataset_readiness",
                               {"project_id": p.id, "episode_ids": ["ep_1", "ep_2", "ep_3"]}, "mcp")
    assert out["readiness"]["verdict"] in ("ready", "ready_with_warnings") and out["summary"]


async def test_nothing_to_check_is_blocked(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube"))
    rep = await rd.check(ctx, p)
    assert rep.verdict == "blocked" and rep.findings[0].code == "no_episodes"


@pytest.mark.parametrize("goal,expected", [("pick up the cube", 5), ("wave hello", None)])
def test_the_gripper_is_checked_for_object_tasks(goal, expected):
    from types import SimpleNamespace

    assert rd._gripper_index(SimpleNamespace(spec=SimpleNamespace(goal=goal)), 6) == expected


async def test_a_proposal_carries_the_datasets_readiness(tmp_path):
    from cortex.models import DatasetEpisodeLink, DatasetVersion
    from cortex.services import training_service as ts

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    links = []
    for i in range(3):
        ep = _ep(i)
        ep.project_id = p.id
        a, s = demo(seed=i, gripper=False)  # nobody closed the gripper
        f = ctx.local(npz_key(ep))
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, action=a.astype(np.float32), state=s.astype(np.float32))
        await ctx.repo.save("episodes", ep)
        links.append(DatasetEpisodeLink(episode_id=ep.id, split="train", lerobot_episode_index=i))
    ds = DatasetVersion(project_id=p.id, name="v1", links=links, lerobot_root=f"projects/{p.id}/datasets/v1/lerobot")
    await ctx.repo.save("dataset_versions", ds, status="ready")
    tj = await ts.propose(ctx, p, ds.id)
    assert tj.readiness["verdict"] == "ready_with_warnings" and tj.readiness["warnings"] >= 1
    assert "gripper never moves (3 episodes)" in tj.rationale


def test_the_readiness_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api import app as api

    with TestClient(api.app) as c:
        pid = c.portal.call(orch.create_project, api.CTX, CreateProjectRequest(task_prompt="pick up the cube")).id
        r = c.post(f"/api/projects/{pid}/readiness", json={})
        assert r.status_code == 200 and r.json()["verdict"] == "blocked"
        assert c.get("/api/datasets/ds_nope/readiness").status_code == 404
    settings.cache_clear()


def test_the_summary_counts_episodes_not_findings():
    f = [rd.Finding(code="idle_head", severity="warn", message="", episode_ids=["a", "b", "c"]),
         rd.Finding(code="few_demos", severity="warn", message=""),
         rd.Finding(code="length_outlier", severity="info", message="", episode_ids=["a"])]
    assert rd.summarize("ready_with_warnings", 9, f) == (
        "Ready, with warnings: 9 episodes checked; idle start (3 episodes), thin conditions")


async def test_odd_files_are_findings_not_crashes(tmp_path):
    """A flat (1-D) action array and a file without actions are reported, never a 500."""
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube"))
    for i, arrays in enumerate([{"action": np.linspace(0, 1, 90)}, {"state": np.zeros((90, 6))},
                                {"action": demo()[0], "front": np.zeros((150, 8, 8, 3), np.uint8)}]):
        ep = _ep(i)
        ep.project_id = p.id
        f = ctx.local(npz_key(ep))
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, **arrays)
        await ctx.repo.save("episodes", ep)
    rep = await rd.check(ctx, p)
    found = {f.code: f for f in rep.findings}
    assert rep.episodes_checked == 3 and "shape_mismatch" in found


async def test_episodes_in_a_cloud_store_are_fetched_first(tmp_path, monkeypatch):
    """With InsForge storage the npz may only be in the bucket; readiness fetches it rather than calling it missing."""
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube"))
    ep = _ep(0)
    ep.project_id = p.id
    await ctx.repo.save("episodes", ep)
    a, s = demo()

    async def fetch(key):
        f = ctx.local(key)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, action=a, state=s)
        return f
    monkeypatch.setattr(ctx, "ensure_local", fetch)
    rep = await rd.check(ctx, p)
    assert "unreadable" not in codes(rep.findings)
