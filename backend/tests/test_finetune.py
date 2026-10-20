import asyncio
import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Checkpoint, CreateProjectRequest, DatasetVersion, ResourceEstimate, TrainingJob
from cortex.services import agent_orchestrator as orch
from cortex.services import training_service as ts


def dataset(tmp_path, robot):
    root = tmp_path / f"ds_{robot}"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"robot_type": robot}))
    return root


async def setup(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ds = DatasetVersion(project_id=p.id, name="real", links=[], lerobot_root=str(dataset(tmp_path, "so101_follower")))
    await ctx.repo.save("dataset_versions", ds, status="ready")
    base_dir = tmp_path / "ck"
    base_dir.mkdir()
    base_tj = TrainingJob(project_id=p.id, dataset_version_id=ds.id, policy_type="smolvla", backend="manual",
                          config={"robot_type": "so100"}, rationale="", estimate=ResourceEstimate(
                              gpu="", gpu_hours=0, est_cost_usd=0, paid=False, notes=""))
    await ctx.repo.save("training_jobs", base_tj)
    ck = Checkpoint(project_id=p.id, training_job_id=base_tj.id, policy_type="smolvla", step=7500, path=str(base_dir))
    await ctx.repo.save("checkpoints", ck)
    return ctx, p, ds, ck


def test_continue_fine_tuning_records_lineage_and_robot(tmp_path):
    async def go():
        ctx, p, ds, ck = await setup(tmp_path)
        tj = await ts.propose(ctx, p, ds.id, "act", 3000, base_checkpoint_id=ck.id)
        c = tj.config
        assert tj.policy_type == "smolvla"  # follows the base policy, not the default
        assert c["policy.path"] == ck.path and c["base_checkpoint_id"] == ck.id
        assert c["robot_type"] == "so101_follower" and c["base_robot_type"] == "so100" and c["cross_embodiment"]
        assert "different robot" in tj.rationale and "rename_map" not in c
        with pytest.raises(KeyError):
            await ts.propose(ctx, p, ds.id, base_checkpoint_id="ck_missing")
        fresh = await ts.propose(ctx, p, ds.id, "smolvla", 3000)
        assert fresh.config["policy.path"] == "lerobot/smolvla_base"
        assert fresh.config["rename_map"] == {"observation.images.front": "observation.images.camera1",
                                              "observation.images.wrist": "observation.images.camera2"}
    asyncio.run(go())


def test_slurm_submit_ships_the_base_and_the_rename_map(tmp_path, monkeypatch):
    calls = []

    async def fake_sh(*args, timeout=600, cwd=None, input=None):
        calls.append(" ".join(args))
        return 0, "12345\n" if "sbatch --parsable" in " ".join(args) else ""

    async def no_poll(ctx, tj, job):
        return "done"
    monkeypatch.setattr(ts, "_sh", fake_sh)
    monkeypatch.setattr(ts, "poll_slurm", no_poll)

    async def go():
        ctx, p, ds, ck = await setup(tmp_path)
        (tmp_path / ds.lerobot_root).mkdir(exist_ok=True)
        for tj in (await ts.propose(ctx, p, ds.id, base_checkpoint_id=ck.id), await ts.propose(ctx, p, ds.id, "smolvla")):
            tj.config["dataset.root"] = ds.lerobot_root
            job = ctx.jobs.submit("train", p.id, lambda j: asyncio.sleep(0))
            await ts._run_slurm(ctx, tj, job)
        return calls
    calls = asyncio.run(go())
    sbatch = [c for c in calls if "sbatch --parsable" in c]
    assert any(c.endswith("/base/") and c.startswith("rsync") for c in calls)
    assert "POLICY_PATH=base" in sbatch[0] and "POLICY_PATH=lerobot/smolvla_base" in sbatch[1]
    assert any(c.startswith("ssh") and c.endswith("cat > ~/cortex/jobs/" + c.split("/jobs/")[1].split("/")[0] + "/rename_map.json")
               for c in calls if "/jobs/" in c)


def test_a_cortex_built_dataset_records_its_robot(tmp_path):
    """#369: lerobot_root is relative to the data dir; the robot type was read relative to the cwd, so never found."""
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
        rel = f"projects/{p.id}/datasets/v1/lerobot"
        (tmp_path / rel / "meta").mkdir(parents=True)
        (tmp_path / rel / "meta" / "info.json").write_text(json.dumps({"robot_type": "so101_follower"}))
        ds = DatasetVersion(project_id=p.id, name="v1", links=[], lerobot_root=rel)
        await ctx.repo.save("dataset_versions", ds, status="ready")
        assert ts.robot_type_of(ctx, ds) == "so101_follower"
        tj = await ts.propose(ctx, p, ds.id)
        assert tj.config["robot_type"] == "so101_follower"
    asyncio.run(go())
