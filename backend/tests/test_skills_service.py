"""Physical Skills (#276): every trained policy across projects, with an honest score and how to run it live."""

import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.errors import NotFound
from cortex.models import (
    CreateProjectRequest, JobStatus, Mode, PolicyReplay, ReplayEpisode, ResourceEstimate, TrainingJob,
)
from cortex.services import (
    live_policy_service, project_cleanup, project_service, sim_import_service, skills_service, training_service,
)
from fake_futurama import FakeServe, free_port

REAL_CK = "futurama:~/cortex-jenga/output/jenga/jenga_010000"
SIM_SRC = "futurama:~/cortex-sim/unscrew"


def _write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _mirror(root):
    """<job_dir>/unscrew as sim_task.sbatch leaves it: ACT trained, step 30000 evaluated on 4 unseen seeds."""
    _write(root / "dataset" / "meta" / "info.json", {"total_frames": 900})
    _write(root / "dataset" / "meta" / "cortex_sim.json", {"instruction": "unscrew the bolt", "kept_seeds": [0, 1],
                                                           "failed_seeds": [], "oracle_success_rate": 1.0})
    for step in ("015000", "030000"):
        pm = root / "runs" / "act" / "checkpoints" / step / "pretrained_model"
        _write(pm / "config.json", {"type": "act"})
    ev = root / "runs" / "act" / "eval" / "030000"
    (ev / "seed100000_ok.mp4").parent.mkdir(parents=True, exist_ok=True)
    (ev / "seed100000_ok.mp4").write_bytes(b"mp4")
    res = [{"seed": 100000 + i, "success": i != 3, "steps": 200, "video": "seed100000_ok.mp4" if i == 0 else None}
           for i in range(4)]
    _write(ev / "metrics.json", {"policy": "act", "seeds": "100000..100003", "success_rate": 0.75, "results": res})
    return root


@pytest.fixture(autouse=True)
def _stopped():
    """Reading the GPU server's status marks it ready; leave the module state as other tests expect it."""
    yield
    live_policy_service.STATE.update(phase="stopped", step=None, detail=None, error=None)


@pytest.fixture
async def world(tmp_path):
    """A real-data SmolVLA trained on Futurama (weights on the cluster) with a held-out replay and a live run,
    an imported sim task with an evaluation and a live sim run, and things that are not skills."""
    port = free_port()  # nothing listens here: the GPU server is down
    mapping = tmp_path / "projects.json"
    ctx = await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None, futurama_projects_file=mapping,
                                   live_policy_url=f"http://127.0.0.1:{port}"))
    real = project_service.create_project(CreateProjectRequest(task_prompt="Pull a block out of a Jenga tower",
                                                               mode=Mode.real))
    await ctx.repo.save("projects", real, status="active")
    mapping.write_text(json.dumps({"jenga": real.id}))
    tj = TrainingJob(project_id=real.id, dataset_version_id="ds_x", policy_type="smolvla", backend="futurama_slurm",
                     config={"steps": 10000, "robot_type": "so100_follower"}, status=JobStatus.succeeded,
                     estimate=ResourceEstimate(gpu="1x H200 (Futurama general partition)", gpu_hours=0,
                                               est_cost_usd=0, paid=False, notes=""), rationale="")
    await ctx.repo.save("training_jobs", tj)
    ck = await training_service.register_remote_checkpoint(ctx, real.id, tj.id, REAL_CK, {"type": "smolvla"}, 10000)
    held = PolicyReplay(project_id=real.id, kind="held_out_replay", task="jenga", checkpoint_id=ck.id,
                        checkpoint_name="jenga_010000", joint_mae_deg=0.11, source="futurama:~/cortex-jenga/replay",
                        episodes=[ReplayEpisode(episode=58, frames=241, joint_mae_deg=0.1, video_key="r/ep58.mp4"),
                                  ReplayEpisode(episode=132, frames=206, joint_mae_deg=0.12)])
    live = PolicyReplay(project_id=real.id, kind="live_inference", task="jenga", checkpoint_id=ck.id,
                        checkpoint_name="jenga_010000", gpu="NVIDIA H200", source="live GPU stream /live/jenga/58",
                        episodes=[ReplayEpisode(episode=58, frames=200, joint_mae_deg=0.5, video_key="r/live58.mp4",
                                                inference_ms={"mean": 41.0})])
    for rp in (held, live):
        await ctx.repo.save("policy_replays", rp)

    sim = await sim_import_service.import_task(ctx, "unscrew", _mirror(tmp_path / "mirror"), SIM_SRC, [])
    run = PolicyReplay(project_id=sim["project_id"], kind="live_sim", task="unscrew",
                       checkpoint_id=sim["variants"][0]["checkpoints"]["030000"], checkpoint_name="act@30000",
                       policy_type="act", source="live GPU stream /sim/unscrew/123456",
                       episodes=[ReplayEpisode(episode=123456, frames=300, success=True, steps=300,
                                               video_key="s/seed123456.mp4")])
    await ctx.repo.save("policy_replays", run)

    # not skills: a training job that never finished, and a trained policy in a deleted project
    await ctx.repo.save("training_jobs", tj.model_copy(update={"id": "tj_failed", "status": JobStatus.failed,
                                                              "checkpoint_ids": []}))
    gone = project_service.create_project(CreateProjectRequest(task_prompt="Stack cups"))
    await ctx.repo.save("projects", gone, status=project_cleanup.DELETED)
    await ctx.repo.save("training_jobs", tj.model_copy(update={"id": "tj_gone", "project_id": gone.id}))
    await training_service.register_remote_checkpoint(ctx, gone.id, "tj_gone", "futurama:~/x/cups_1", {"type": "act"})
    return ctx, real, tj, ck, sim


async def test_every_trained_policy_is_a_skill_with_plain_names(world):
    ctx, real, tj, ck, sim = world
    skills = await skills_service.list_skills(ctx)
    by_name = {s["name"]: s for s in skills}
    assert set(by_name) == {"Pull a Jenga block", "Unscrew a bolt with the gripper"}
    j = by_name["Pull a Jenga block"]
    assert j["id"] == f"sk_{tj.id}" and j["project_id"] == real.id and j["checkpoint_id"] == ck.id
    assert j["robot"] == "SO-100" and j["policy"] == "SmolVLA" and j["data"] == "real"
    assert j["trained_where"] == "Futurama H200" and j["weights"] == "on the cluster" and j["steps"] == 10000
    u = by_name["Unscrew a bolt with the gripper"]
    assert u["policy"] == "ACT" and u["data"] == "sim" and u["robot"] == "SO-101" and u["steps"] == 30000
    assert u["project_id"] == sim["project_id"]


async def test_scores_are_labelled_honestly(world):
    ctx, *_ = world
    by_name = {s["name"]: s for s in await skills_service.list_skills(ctx)}
    j = by_name["Pull a Jenga block"]["score"]
    assert j["kind"] == "joint_error" and j["value"] == 0.11 and j["n"] == 2
    assert j["label"] == "0.11° mean joint error on 2 held-out real demos"
    assert "success" not in j["label"].lower() and "not a success rate" in j["detail"]
    u = by_name["Unscrew a bolt with the gripper"]["score"]
    assert u["kind"] == "sim_success" and u["value"] == 0.75 and u["n"] == 4
    assert u["label"] == "75% success on 4 unseen sim layouts" and u["ci95"]


async def test_past_live_runs_and_a_thumbnail(world):
    ctx, *_ = world
    by_name = {s["name"]: s for s in await skills_service.list_skills(ctx)}
    j = by_name["Pull a Jenga block"]
    assert [r["kind"] for r in j["runs"]] == ["live_inference"]
    r = j["runs"][0]
    assert r["video_url"].endswith("/files/r/live58.mp4") and r["joint_mae_deg"] == 0.5 and r["success"] is None
    assert j["thumbnail_url"] == r["video_url"]
    u = by_name["Unscrew a bolt with the gripper"]
    assert [(r["kind"], r["success"]) for r in u["runs"]] == [("live_sim", True)]
    assert "not part of the score" in u["runs"][0]["note"]


async def test_how_to_run_it_when_the_gpu_server_is_down(world):
    ctx, *_ = world
    by_name = {s["name"]: s for s in await skills_service.list_skills(ctx)}
    run = by_name["Pull a Jenga block"]["run"]
    assert run == {"kind": "real", "task": "jenga", "episodes": [58, 132], "live_now": False,
                   "note": "The GPU server isn't running. Request a GPU to run it live."}
    assert by_name["Unscrew a bolt with the gripper"]["run"]["kind"] == "sim"
    assert by_name["Unscrew a bolt with the gripper"]["run"]["live_now"] is False


async def test_with_the_gpu_server_up_skills_run_live_and_unimported_sim_policies_appear(world):
    ctx, real, tj, *_ = world
    srv = FakeServe()
    srv.info["sim"] = {"unscrew": {"variant": "act", "step": 30000, "success_rate": 0.8, "eval_episodes": 50,
                                   "eval_seeds": "100000..100049", "instruction": "unscrew the bolt"},
                       "conveyor": {"variant": "smolvla", "step": 20000, "success_rate": None, "eval_episodes": 0,
                                    "eval_seeds": "", "instruction": "sort the parts"}}
    ctx.settings.live_policy_url = f"http://127.0.0.1:{srv.port}"
    try:
        by_name = {s["name"]: s for s in await skills_service.list_skills(ctx)}
    finally:
        srv.close()
    j = by_name["Pull a Jenga block"]["run"]
    assert j["live_now"] and j["episodes"] == [58, 59] and j["note"] is None  # the server's held-out episodes
    u = by_name["Unscrew a bolt with the gripper"]
    assert u["run"]["live_now"] and u["score"]["n"] == 4  # the local evaluation of that checkpoint wins
    c = by_name["Pick a moving part off a conveyor"]
    assert c["id"] == "sk_served_sim_conveyor" and c["data"] == "sim" and c["run"]["live_now"]
    assert c["score"]["kind"] == "pending" and c["score"]["label"] == "Score pending"
    assert c["trained_where"] == "Futurama" and c["policy"] == "SmolVLA"


async def test_a_served_checkpoint_that_is_not_this_one_is_not_run_as_this_skill(world):
    ctx, *_ = world
    srv = FakeServe()
    srv.info["policies"]["jenga"]["checkpoint"] = "jenga_020000"
    ctx.settings.live_policy_url = f"http://127.0.0.1:{srv.port}"
    try:
        j = {s["name"]: s for s in await skills_service.list_skills(ctx)}["Pull a Jenga block"]
    finally:
        srv.close()
    assert j["run"]["live_now"] is False and "jenga_020000" in j["run"]["note"]


async def test_get_skill(world):
    ctx, real, tj, *_ = world
    s = await skills_service.get_skill(ctx, f"sk_{tj.id}")
    assert s["name"] == "Pull a Jenga block"
    with pytest.raises(NotFound):
        await skills_service.get_skill(ctx, "sk_nope")


def test_the_api_lists_and_gets_skills(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LIVE_POLICY_URL", f"http://127.0.0.1:{free_port()}")
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    try:
        with TestClient(app) as c:
            assert c.get("/api/skills").json() == []
            assert c.get("/api/skills/sk_nope").status_code == 404
    finally:
        settings.cache_clear()
