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


def _mirror(root, practice: int | None = None):
    """<job_dir>/unscrew as sim_task.sbatch leaves it: ACT trained, step 30000 evaluated on 4 unseen seeds (or on 4
    runs over `practice` layouts it trained on: LAYOUTS=N)."""
    _write(root / "dataset" / "meta" / "info.json", {"total_frames": 900})
    _write(root / "dataset" / "meta" / "cortex_sim.json", {"instruction": "unscrew the bolt", "kept_seeds": [0, 1],
                                                           "failed_seeds": [], "oracle_success_rate": 1.0})
    for step in ("015000", "030000"):
        pm = root / "runs" / "act" / "checkpoints" / step / "pretrained_model"
        _write(pm / "config.json", {"type": "act"})
    ev = root / "runs" / "act" / "eval" / "030000"
    (ev / "seed100000_ok.mp4").parent.mkdir(parents=True, exist_ok=True)
    (ev / "seed100000_ok.mp4").write_bytes(b"mp4")
    res = [{"seed": i % practice if practice else 100000 + i, "success": i != 3, "steps": 200,
            "video": "seed100000_ok.mp4" if i == 0 else None} for i in range(4)]
    _write(ev / "metrics.json", {"policy": "act", "seeds": "100000..100003", "success_rate": 0.75, "results": res,
                                 "layouts": practice})
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
    skills = await skills_service.list_skills(ctx, real=True)
    by_name = {s["name"]: s for s in skills}
    assert set(by_name) == {"Pull a Jenga block", "Unscrew a bolt with the gripper"}
    j = by_name["Pull a Jenga block"]
    assert j["id"] == f"sk_{tj.id}" and j["project_id"] == real.id and j["checkpoint_id"] == ck.id
    assert j["robot"] == "SO-100" and j["policy"] == "SmolVLA" and j["data"] == "real"
    assert j["trained_where"] == "Futurama H200" and j["weights"] == "on the cluster" and j["steps"] == 10000
    assert j["instruction"] == "Pull a block out of a Jenga tower"  # the prompt, not the default cube goal
    u = by_name["Unscrew a bolt with the gripper"]
    assert u["policy"] == "ACT" and u["data"] == "sim" and u["robot"] == "SO-101" and u["steps"] == 30000
    assert u["project_id"] == sim["project_id"] and u["instruction"] == "Unscrew a hex bolt from a fixture with the SO-101"


async def test_scores_are_labelled_honestly(world):
    ctx, *_ = world
    by_name = {s["name"]: s for s in await skills_service.list_skills(ctx, real=True)}
    j = by_name["Pull a Jenga block"]["score"]
    assert j["kind"] == "joint_error" and j["value"] == 0.11 and j["n"] == 2
    assert j["label"] == "0.11° mean joint error on 2 held-out real demos"
    assert "success" not in j["label"].lower() and "not a success rate" in j["detail"]
    u = by_name["Unscrew a bolt with the gripper"]["score"]
    assert u["kind"] == "sim_success" and u["value"] == 0.75 and u["n"] == 4
    assert u["label"] == "75% success on 4 unseen sim layouts" and u["ci95"]


async def test_past_live_runs_and_a_thumbnail(world):
    ctx, *_ = world
    by_name = {s["name"]: s for s in await skills_service.list_skills(ctx, real=True)}
    j = by_name["Pull a Jenga block"]
    assert [r["kind"] for r in j["runs"]] == ["live_inference"]
    r = j["runs"][0]
    assert r["video_url"].endswith("/files/r/live58.mp4") and r["joint_mae_deg"] == 0.5 and r["success"] is None
    assert j["thumbnail_url"] == r["video_url"]
    u = by_name["Unscrew a bolt with the gripper"]
    assert [(r["kind"], r["label"], r["success"]) for r in u["runs"]] == [("live_sim", "Live sim run", True)]
    assert "not part of the score" in u["runs"][0]["note"]


async def test_how_to_run_it_when_the_gpu_server_is_down(world):
    ctx, *_ = world
    by_name = {s["name"]: s for s in await skills_service.list_skills(ctx, real=True)}
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
        by_name = {s["name"]: s for s in await skills_service.list_skills(ctx, real=True)}
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
        j = {s["name"]: s for s in await skills_service.list_skills(ctx, real=True)}["Pull a Jenga block"]
    finally:
        srv.close()
    assert j["run"]["live_now"] is False and "jenga_020000" in j["run"]["note"]


async def test_only_policies_trained_in_our_sim_by_default(world):
    """User rule (#298): the studio and agents see sim-trained policies; real-demo ones only when asked for."""
    ctx, *_ = world
    assert [s["name"] for s in await skills_service.list_skills(ctx)] == ["Unscrew a bolt with the gripper"]
    assert {s["data"] for s in await skills_service.list_skills(ctx, real=True)} == {"sim", "real"}


async def test_practice_layouts_are_never_called_unseen(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None, futurama_projects_file=tmp_path / "x",
                                   live_policy_url=f"http://127.0.0.1:{free_port()}"))
    await sim_import_service.import_task(ctx, "unscrew", _mirror(tmp_path / "m", practice=3), SIM_SRC, [])
    [u] = await skills_service.list_skills(ctx)
    assert u["score"]["label"] == "75% success on 4 runs over 3 practice layouts it trained on"
    assert u["layouts"] == 3 and "not a test on new layouts" in u["score"]["detail"]
    assert u["evidence"][0]["label"] == "Practice layout 0"

    srv = FakeServe()
    srv.info["sim"] = {"conveyor": {"variant": "act", "step": 9000, "success_rate": 0.9, "eval_episodes": 10,
                                    "eval_seeds": "practice layouts 0..4", "instruction": "", "layouts": 5}}
    ctx.settings.live_policy_url = f"http://127.0.0.1:{srv.port}"
    try:
        c = {s["name"]: s for s in await skills_service.list_skills(ctx)}["Pick a moving part off a conveyor"]
    finally:
        srv.close()
    assert c["score"]["label"] == "90% success on 10 runs over 5 practice layouts it trained on" and c["layouts"] == 5


async def test_get_skill(world):
    ctx, real, tj, *_ = world
    s = await skills_service.get_skill(ctx, f"sk_{tj.id}", real=True)
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


def _replay(ctx, task, name, mtime):
    import os

    p = ctx.data_dir / "skill_replays" / task / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"mp4")
    os.utime(p, (mtime, mtime))


async def test_sim_skills_stay_listed_with_their_replays_while_the_gpu_server_is_down(world):
    """#308: a sim policy the live server served stays listed after it goes down; known skills are listed only
    with a replay; replays are newest first."""
    ctx, *_ = world
    _replay(ctx, "unscrew", "old.mp4", 1000)
    _replay(ctx, "unscrew", "new.mp4", 2000)
    _replay(ctx, "pick_place", "pp.mp4", 1000)
    (ctx.data_dir / "skill_replays" / "unscrew" / "notes.txt").write_text("not a video")
    srv = FakeServe()
    srv.info["sim"] = {"conveyor": {"variant": "smolvla", "step": 20000, "success_rate": None, "eval_episodes": 0,
                                    "eval_seeds": "", "instruction": "sort the parts"}}
    down = ctx.settings.live_policy_url
    ctx.settings.live_policy_url = f"http://127.0.0.1:{srv.port}"
    try:
        live = {s["task"]: s for s in await skills_service.list_skills(ctx)}
    finally:
        srv.close()
    assert live["conveyor"]["live"] is True and live["unscrew"]["live"] is False
    ctx.settings.live_policy_url = down  # the server is gone

    by_task = {s["task"]: s for s in await skills_service.list_skills(ctx)}
    assert set(by_task) == {"unscrew", "conveyor", "pick_place"}  # no jenga/unscrew_black: no replay, never seen
    c = by_task["conveyor"]
    assert c["id"] == "sk_served_sim_conveyor" and c["live"] is False and c["policy"] == "SmolVLA"
    assert c["run"] == {"kind": "sim", "task": "conveyor", "episodes": [], "live_now": False,
                        "note": skills_service.DOWN}
    assert c["replays"] == []
    u = by_task["unscrew"]  # the imported skill, not a second seeded one
    assert u["name"] == "Unscrew a bolt with the gripper" and u["id"].startswith("sk_tj")
    assert u["replays"] == ["/api/skills/unscrew/replays/new.mp4", "/api/skills/unscrew/replays/old.mp4"]
    p = by_task["pick_place"]
    assert p["name"] == "Pick and place" and p["live"] is False and p["run"]["note"] == skills_service.DOWN
    assert p["thumbnail_url"] == "/api/skills/pick_place/replays/pp.mp4" and p["score"]["kind"] == "none"


async def test_a_malformed_skills_seen_entry_is_skipped_not_fatal(world):
    """#319: one entry missing keys (a hand edit, an older file) made every GET /api/skills fail with KeyError."""
    ctx, *_ = world
    ok = skills_service._seeded("jenga_big")
    (ctx.data_dir / skills_service.SEEN).write_text(json.dumps({
        "jenga": {"id": "sk_served_sim_jenga"}, "palletize": "not a skill", "unscrew_black": {**ok, "score": None},
        "conveyor": {**ok, "task": "suture"}, "jenga_big": ok}))
    by_task = {s["task"]: s for s in await skills_service.list_skills(ctx)}
    assert set(by_task) == {"unscrew", "jenga_big"}
    assert by_task["jenga_big"]["run"]["note"] == skills_service.DOWN


async def test_a_replay_file_never_leaves_its_folder(world):
    ctx, *_ = world
    _replay(ctx, "unscrew", "run 1.mp4", 1000)
    (ctx.data_dir / "secret.mp4").write_bytes(b"x")
    assert skills_service.replay_file(ctx, "unscrew", "run 1.mp4").read_bytes() == b"mp4"
    for task, name in [("unscrew", "../../secret.mp4"), ("..", "secret.mp4"), ("unscrew", "..%2Fsecret.mp4"),
                       ("unscrew", "/etc/passwd"), ("unscrew", "missing.mp4"), ("unscrew", ".hidden.mp4"),
                       ("../unscrew", "run 1.mp4"), ("unscrew", "run 1.txt")]:
        with pytest.raises(NotFound):
            skills_service.replay_file(ctx, task, name)


def test_the_api_serves_replays_and_passes_the_camera_view_on(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    srv = FakeServe()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LIVE_POLICY_URL", f"http://127.0.0.1:{srv.port}")
    monkeypatch.setenv("FUTURAMA_PROJECTS_FILE", str(tmp_path / "projects.json"))
    (tmp_path / "skill_replays" / "jenga").mkdir(parents=True)
    (tmp_path / "skill_replays" / "jenga" / "a.mp4").write_bytes(b"mp4data")
    (tmp_path / "x.mp4").write_bytes(b"secret")
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    try:
        with TestClient(app) as c:
            [j] = c.get("/api/skills").json()
            assert j["task"] == "jenga" and j["live"] is True and j["replays"] == ["/api/skills/jenga/replays/a.mp4"]
            r = c.get(j["replays"][0])
            assert r.status_code == 200 and r.content == b"mp4data" and r.headers["content-type"] == "video/mp4"
            assert c.get("/api/skills/jenga/replays/..%2F..%2Fx.mp4").status_code == 404
            assert c.get("/api/skills/jenga/replays/nope.mp4").status_code == 404
            assert c.get("/api/live-policy/sim/jenga/7?view=wrist").status_code == 200
            assert c.get("/api/live-policy/sim/jenga/8").status_code == 200
            assert c.get("/api/live-policy/sim/jenga/9?view=side").status_code == 422
            assert srv.started == ["/sim/jenga/7?view=wrist", "/sim/jenga/8"]
    finally:
        srv.close()
        settings.cache_clear()
