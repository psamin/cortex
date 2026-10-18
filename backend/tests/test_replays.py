"""Policy replays (#179): held-out replays from Futurama are stored per project and shown, labelled as joint
error rather than success."""

import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Checkpoint, CreateProjectRequest
from cortex.services import agent_orchestrator, project_service, replay_service, studio_service

METRICS = {"task": "jenga", "checkpoint": "/nethome/pi34/cortex-jenga/output/jenga/jenga_010000",
           "episodes": {"58": {"frames": 241, "joint_mae_deg": 0.1, "gripper_mae": 0.15,
                               "task": "push brown wooden block without tipping tower"},
                        "132": {"frames": 206, "joint_mae_deg": 0.12, "gripper_mae": 0.13,
                                "task": "pull a wooden block out of the Jenga tower"}},
           "overall_joint_mae_deg": 0.11}


@pytest.fixture
async def setup(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None))
    p = project_service.create_project(CreateProjectRequest(task_prompt="Pull a block out of a Jenga tower"))
    await ctx.repo.save("projects", p, status="active")
    ck = Checkpoint(project_id=p.id, training_job_id=None, policy_type="smolvla", step=10000,
                    path="futurama:/nethome/pi34/cortex-jenga/output/jenga/jenga_010000")
    await ctx.repo.save("checkpoints", ck, status="ready")
    folder = tmp_path / "replay" / "jenga"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text(json.dumps(METRICS))
    (folder / "ep58.mp4").write_bytes(b"mp4-58")
    (folder / "ep132.mp4").write_bytes(b"mp4-132")
    return ctx, p, ck, folder


async def test_import_links_the_checkpoint_and_stores_the_videos(setup):
    ctx, p, ck, folder = setup
    rp = await replay_service.import_replay_dir(ctx, p.id, folder, "futurama:~/cortex-jenga/replay/jenga")
    assert rp.kind == "held_out_replay" and rp.checkpoint_id == ck.id and rp.checkpoint_name == "jenga_010000"
    assert [e.episode for e in rp.episodes] == [58, 132] and rp.joint_mae_deg == 0.11
    assert rp.episodes[0].instruction.startswith("push") and rp.episodes[1].gripper_mae == 0.13
    assert ctx.local(rp.episodes[0].video_key).read_bytes() == b"mp4-58"
    assert "not a success rate" in rp.note

    again = await replay_service.import_replay_dir(ctx, p.id, folder, "futurama:~/cortex-jenga/replay/jenga")
    assert again.id == rp.id and len(await replay_service.list_replays(ctx, p.id)) == 1  # updated, not duplicated


async def test_viewer_and_pipeline_show_joint_error_never_success(setup):
    ctx, p, ck, folder = setup
    rp = await replay_service.import_replay_dir(ctx, p.id, folder, "futurama:~/cortex-jenga/replay/jenga")
    rows = [r for r in await studio_service.rollouts(ctx, p.id) if r["kind"] == "replay"]
    assert {r["id"] for r in rows} == {f"{rp.id}/ep58", f"{rp.id}/ep132"}
    assert all(r["success"] is None and r["policy"] == "v1" for r in rows)
    assert {r["metric"] for r in rows} == {"0.10° joint error", "0.12° joint error"}
    it = (await studio_service.iterations(ctx, p.id))[0]
    assert it["checkpoint_id"] == ck.id and it["evaluation"] is None
    assert it["replay"]["joint_mae_deg"] == 0.11 and it["replay"]["episodes"] == 2

    st = await agent_orchestrator.project_state(ctx, p)
    assert st.replays[0]["id"] == rp.id and st.replays[0]["episodes"] == [58, 132]
    assert st.evaluations == []  # a replay is not an evaluation: no success rate is claimed


async def test_a_replay_of_an_unregistered_checkpoint_keeps_its_name(setup):
    ctx, p, _, folder = setup
    m = {**METRICS, "checkpoint": "/x/output/suture/needle_001500"}
    (folder / "metrics.json").write_text(json.dumps(m))
    rp = await replay_service.import_replay_dir(ctx, p.id, folder, "futurama:~/cortex-jenga/replay/suture_v1")
    assert rp.checkpoint_id is None and rp.checkpoint_name == "needle_001500"
    rows = [r for r in await studio_service.rollouts(ctx, p.id) if r["kind"] == "replay"]
    assert rows and all(r["policy"] == "needle_001500" for r in rows)


async def test_a_live_sim_run_cut_off_early_is_not_a_failure(setup):
    """Recorded before the fix: a partial run stored the last frame's "not yet" as success=False."""
    from cortex.models import PolicyReplay, ReplayEpisode

    ctx, p, _, _ = setup
    rp = PolicyReplay(project_id=p.id, kind="live_sim", status="partial", task="jenga", episodes=[ReplayEpisode(
        episode=7, frames=140, total_frames=900, success=False, steps=140, video_key="v.mp4")])
    await replay_service.save(ctx, rp)
    row = [r for r in await studio_service.rollouts(ctx, p.id) if r["kind"] == "replay"][0]
    assert row["success"] is None and row["metric"] == "cut off, not judged · 140 steps"
    assert replay_service.brief(rp)["success"] is None
