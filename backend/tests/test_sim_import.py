"""Importing sim_task.sbatch results (#179), on fixtures in backend/cortex/sim/tasks/run.py's output format."""

import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Checkpoint, DatasetVersion, EvaluationRun, Project, TrainingJob
from cortex.services import project_cleanup, sim_import_service, simulation_runner, studio_service

SRC = "futurama:~/cortex-sim/jenga"


def _write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data) if not isinstance(data, str) else data)


def _ck(run, step, ptype):
    pm = run / "checkpoints" / step / "pretrained_model"
    _write(pm / "config.json", {"type": ptype})
    _write(pm / "train_config.json", {"steps": 30000, "batch_size": 32})


def _eval(run, step, results):
    d = run / "eval" / step
    for r in results:
        if r.get("video"):
            (d / r["video"]).parent.mkdir(parents=True, exist_ok=True)
            (d / r["video"]).write_bytes(b"mp4")
    _write(d / "metrics.json", {"task": "jenga", "checkpoint": f"/x/{step}/pretrained_model", "policy": "act",
                                "episodes": len(results), "success_rate": sum(r["success"] for r in results) / len(results),
                                "seeds": "100000..100001", "results": results})


@pytest.fixture
def job(tmp_path):
    """<job_dir>/jenga as sim_task.sbatch leaves it: demos done, act trained and evaluated, smolvla training."""
    t = tmp_path / "mirror"
    _write(t / "dataset" / "meta" / "info.json", {"total_frames": 1234, "total_episodes": 3, "fps": 30})
    _write(t / "dataset" / "meta" / "cortex_sim.json", {
        "task": "jenga", "instruction": "push the marked block out of the tower", "size": "240x320",
        "kept_seeds": [0, 1, 3], "failed_seeds": [2], "oracle_success_rate": 0.75})
    for s in (0, 1):
        (t / "examples").mkdir(exist_ok=True)
        (t / "examples" / f"demo_seed{s}.mp4").write_bytes(b"mp4")
    act = t / "runs" / "act"
    _ck(act, "015000", "act")
    _ck(act, "030000", "act")
    _eval(act, "015000", [{"seed": 100000, "success": True, "steps": 250, "wall_s": 3.0, "video": "seed100000_ok.mp4"},
                          {"seed": 100001, "success": False, "steps": 400, "wall_s": 4.0}])
    _eval(act, "030000", [{"seed": 100000, "success": True, "steps": 240, "wall_s": 3.0},
                          {"seed": 100001, "success": True, "steps": 260, "wall_s": 3.0}])
    (t / "runs" / "smolvla").mkdir(parents=True)
    logs = tmp_path / "logs"
    _write(logs / "slurm-sim-demos-jenga-10590.out", "[sim] node=a gpu=H200 mode=demos task=jenga\n"
           '{"seed": 0, "success": true, "steps": 210, "kept": 1}\n{"seed": 2, "success": false, "steps": 600, "kept": 1}\n')
    _write(logs / "slurm-sim-act-10591.out", "[sim] node=a gpu=H200 mode=train task=jenga\n'job_name': 'sim_jenga_act'\n"
           "step:200 smpl:6K ep:20 epch:0.5 loss:0.512 grdn:3.1\nstep:400 smpl:13K ep:40 epch:1 loss:0.301 grdn:2\n"
           "[sim] TRAINING_DONE\n")
    _write(logs / "slurm-sim-act-ens-10593.out", "[sim] mode=train task=jenga\n'job_name': 'sim_jenga_act_ensemble'\n"
           "step:200 smpl:6K ep:20 epch:0.5 loss:0.9 grdn:3.1\n")
    _write(logs / "slurm-sim-smolvla-10592.out", "[sim] mode=train task=jenga\n'job_name': 'sim_jenga_smolvla'\n"
           "step:200 smpl:6K ep:20 epch:0.5 loss:1.2 grdn:3.1\n")
    return t, sorted(logs.glob("*.out"))


@pytest.fixture
async def ctx(tmp_path):
    return await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None))


async def test_a_task_run_becomes_a_project_with_data_training_and_evaluations(ctx, job):
    mirror, logs = job
    out = await sim_import_service.import_task(ctx, "jenga", mirror, SRC, logs)
    pid = out["project_id"]
    assert out["project"] == "Jenga (sim)" and out["demos"] == 3 and out["oracle_success_rate"] == 0.75

    ds = (await ctx.repo.list("dataset_versions", DatasetVersion, project_id=pid))[0]
    assert ds.imported_from == f"{SRC}/dataset" and ds.num_frames == 1234 and "3 scripted-oracle demos" in ds.name

    evs = {e["run"].name: e for e in await simulation_runner.list_evaluations(ctx, pid)}
    oracle = evs["oracle demos"]
    assert oracle["run"].policy.kind == "scripted_expert"
    r = oracle["branches"][0].result
    assert (r.n, r.successes) == (4, 3)  # the simulator's verdict on every seed the oracle tried
    by_seed = {e.seed: e for e in r.episodes}
    assert by_seed[0].video_key and by_seed[3].video_key is None and by_seed[2].steps == 600

    tjs = {t.config["variant"]: t for t in await ctx.repo.list("training_jobs", TrainingJob, project_id=pid)}
    act = tjs["act"]
    assert act.status == "succeeded" and act.external_id == "10591"  # not the act_ensemble log
    assert act.metrics["loss_curve"] == [[200, 0.512], [400, 0.301]] and act.config["steps"] == 30000
    assert tjs["smolvla"].status == "running" and tjs["smolvla"].metrics["loss_curve"] == [[200, 1.2]]

    cks = await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)
    assert [c.path for c in cks] == [f"{SRC}/runs/act/checkpoints/{s}/pretrained_model" for s in ("015000", "030000")]
    e15, e30 = evs["act · step 15000"], evs["act · step 30000"]
    assert e15["run"].policy.checkpoint_id == cks[0].id and e15["branches"][0].result.success_rate == 0.5
    assert e30["branches"][0].result.successes == 2 and e30["branches"][0].executor == "futurama_slurm"

    its = await studio_service.iterations(ctx, pid)
    assert [i["evaluation"]["overall_success_rate"] for i in its[:2]] == [0.5, 1.0]
    assert its[2]["policy_version"].endswith("(pending)")  # smolvla still training
    rows = [x for x in await studio_service.rollouts(ctx, pid) if x["kind"] == "eval"]
    assert rows and all(x["condition"] is None for x in rows)  # no cube/lighting cell in these tasks


async def test_re_importing_updates_instead_of_duplicating(ctx, job):
    mirror, logs = job
    first = await sim_import_service.import_task(ctx, "jenga", mirror, SRC, logs)
    _ck(mirror / "runs" / "smolvla", "030000", "smolvla")  # the next run landed
    second = await sim_import_service.import_task(ctx, "jenga", mirror, SRC, logs)
    pid = first["project_id"]
    assert second["project_id"] == pid
    assert len(await ctx.repo.list("dataset_versions", DatasetVersion, project_id=pid)) == 1
    assert len(await ctx.repo.list("training_jobs", TrainingJob, project_id=pid)) == 2
    assert len(await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)) == 3
    assert len(await ctx.repo.list("evaluation_runs", EvaluationRun, project_id=pid)) == 3
    smol = [t for t in await ctx.repo.list("training_jobs", TrainingJob, project_id=pid) if t.policy_type == "smolvla"]
    assert smol[0].status == "succeeded" and len(smol[0].checkpoint_ids) == 1


async def test_a_deleted_project_of_that_name_comes_back(ctx, job):
    mirror, logs = job
    first = await sim_import_service.import_task(ctx, "jenga", mirror, SRC, logs)
    p = await ctx.repo.get("projects", first["project_id"], Project)
    await ctx.repo.save("projects", p, status=project_cleanup.DELETED)
    again = await sim_import_service.import_task(ctx, "jenga", mirror, SRC, logs)
    assert again["project_id"] == p.id and (await ctx.store.get("projects", p.id))["status"] == "active"


async def test_unfinished_demos_create_nothing(ctx, tmp_path):
    out = await sim_import_service.import_task(ctx, "unscrew", tmp_path / "empty", "futurama:~/cortex-sim/unscrew", [])
    assert "not finished" in out["skipped"] and await ctx.store.list("projects") == []
    assert sim_import_service.project_name("unscrew") == "Screwdriver (sim)"
