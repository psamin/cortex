"""Copying Futurama results (#179): only the light files come along; tasks map to their projects."""

import json
from types import SimpleNamespace

import pytest

from cortex.services import futurama_files, replay_service


def test_fetch_copies_only_the_included_files(tmp_path):
    src = tmp_path / "replay" / "jenga"
    src.mkdir(parents=True)
    for name in ("metrics.json", "ep58.mp4", "model.safetensors"):
        (src / name).write_bytes(b"x")
    (src / "sub").mkdir()
    (src / "sub" / "optimizer.pt").write_bytes(b"x")
    dst = futurama_files.fetch(str(src), tmp_path / "copy", ["metrics.json", "*.mp4"])
    assert sorted(p.name for p in dst.rglob("*")) == ["ep58.mp4", "metrics.json"]


def test_remote_sources_and_labels(tmp_path):
    assert futurama_files.is_remote("futurama:~/cortex-jenga/replay/jenga")
    assert futurama_files.is_remote("pi34@host.example:/nethome/pi34/x")
    assert not futurama_files.is_remote(str(tmp_path))
    assert not futurama_files.is_remote("replay/jenga")
    assert futurama_files.local_label("futurama:~/cortex-jenga/replay/suture_v1") == "cortex-jenga_replay_suture_v1"


def test_tasks_map_to_projects(tmp_path):
    f = tmp_path / "projects.json"
    f.write_text(json.dumps({"_note": "x", "jenga": "prj_a8be717dad10", "bad": "nope"}))
    ctx = SimpleNamespace(settings=SimpleNamespace(futurama_projects_file=f))
    assert replay_service.project_for_task(ctx, "jenga") == "prj_a8be717dad10"
    assert replay_service.project_for_task(ctx, "bad") is None and replay_service.project_for_task(ctx, "x") is None
    ctx.settings.futurama_projects_file = tmp_path / "missing.json"
    assert replay_service.project_for_task(ctx, "jenga") is None


def test_the_shipped_mapping_names_the_three_real_demo_projects():
    from cortex.config import Settings

    mapping = json.loads(Settings(_env_file=None).futurama_projects_file.read_text())
    assert {k: v for k, v in mapping.items() if not k.startswith("_")} == {
        "jenga": "prj_a8be717dad10", "conveyor": "prj_b39b49bf6c75", "suture": "prj_25dcd4dd3804"}


async def test_a_checkpoint_can_stay_on_the_cluster(tmp_path):
    from cortex.config import Settings
    from cortex.context import build_ctx
    from cortex.models import (
        CreateProjectRequest, JobStatus, LaunchEvaluationRequest, PolicyRef, ResourceEstimate, TrainingJob,
    )
    from cortex.services import project_service, simulation_runner, training_service

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = project_service.create_project(CreateProjectRequest(task_prompt="Pull a block out of a Jenga tower"))
    await ctx.repo.save("projects", p, status="active")
    tj = TrainingJob(project_id=p.id, dataset_version_id="ds_x", policy_type="smolvla", backend="futurama_slurm",
                     config={}, estimate=ResourceEstimate(gpu="h200", gpu_hours=0, est_cost_usd=0, paid=False, notes=""),
                     rationale="", status=JobStatus.succeeded)
    await ctx.repo.save("training_jobs", tj)
    remote = "futurama:~/cortex-jenga/output/jenga/jenga_010000"
    ck = await training_service.register_remote_checkpoint(ctx, p.id, tj.id, remote, {"type": "smolvla"}, 10000)
    assert ck.path == remote and ck.policy_type == "smolvla" and ck.step == 10000
    assert (await ctx.repo.get("training_jobs", tj.id, TrainingJob)).checkpoint_ids == [ck.id]
    with pytest.raises(FileNotFoundError, match="on the cluster"):  # never read as a path on this machine
        await simulation_runner.resolve_policy(ctx, LaunchEvaluationRequest(
            policy=PolicyRef(kind="checkpoint", checkpoint_id=ck.id)))
    with pytest.raises(ValueError):
        await training_service.register_remote_checkpoint(ctx, p.id, None, str(tmp_path), {}, 1)
