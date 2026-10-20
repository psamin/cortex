"""#152 finding 21: lerobot-train only sees the train split; the eval episodes stay held out."""

import asyncio

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, DatasetEpisodeLink, DatasetVersion
from cortex.services import agent_orchestrator as orch
from cortex.services import training_service as ts

# LeRobot episodes 1 and 3 are eval; 4 is an augmented copy of train episode 0.
SPLITS = ["train", "eval", "train", "eval", "train"]
TRAIN = [0, 2, 4]


async def setup(tmp_path, links=True):
    ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ds = DatasetVersion(project_id=p.id, name="v1", lerobot_root=f"projects/{p.id}/datasets/v1/lerobot",
                        links=[DatasetEpisodeLink(episode_id=f"ep_{i}", split=s, lerobot_episode_index=i)  # type: ignore[arg-type]
                               for i, s in enumerate(SPLITS)] if links else [])
    await ctx.repo.save("dataset_versions", ds, status="ready")
    return ctx, p, ds


def test_the_manual_command_trains_on_train_episodes_only(tmp_path):
    async def go():
        ctx, p, ds = await setup(tmp_path)
        tj = await ts.propose(ctx, p, ds.id)
        assert tj.config["dataset.episodes"] == TRAIN
        cmd = ts.manual_command(ctx, tj)
        assert "--dataset.episodes=[0,2,4] " in cmd
    asyncio.run(go())


def test_a_dataset_without_split_links_is_not_filtered(tmp_path):
    async def go():
        ctx, p, ds = await setup(tmp_path, links=False)
        tj = await ts.propose(ctx, p, ds.id)
        assert tj.config["dataset.episodes"] is None
        assert "--dataset.episodes" not in ts.manual_command(ctx, tj)
    asyncio.run(go())


def test_a_job_proposed_before_the_filter_gets_it_at_launch(tmp_path):
    async def go():
        ctx, p, ds = await setup(tmp_path)
        tj = await ts.propose(ctx, p, ds.id)
        tj.backend = "manual"
        del tj.config["dataset.episodes"]
        await ctx.repo.save("training_jobs", tj)
        await ts.approve(ctx, tj.id, "tester", "gui", None)
        out = await ts.launch(ctx, tj.id)
        assert out.config["dataset.episodes"] == TRAIN
        cmd = (ctx.local(f"projects/{p.id}/training/{tj.id}") / "command.sh").read_text()
        assert "--dataset.episodes=[0,2,4] " in cmd
    asyncio.run(go())


def test_slurm_submit_ships_the_train_episode_list(tmp_path, monkeypatch):
    shipped: dict[str, str] = {}
    calls: list[str] = []

    async def fake_sh(*args, timeout=600, cwd=None, input=None):
        calls.append(" ".join(args))
        if args[0] == "ssh" and args[-1].endswith("episodes.txt"):  # shipped through ssh's stdin (#346)
            shipped[args[-1]] = input.decode()
        return 0, "12345\n" if "sbatch --parsable" in " ".join(args) else ""

    async def no_poll(ctx, tj, job):
        return "done"
    monkeypatch.setattr(ts, "_sh", fake_sh)
    monkeypatch.setattr(ts, "poll_slurm", no_poll)

    async def go():
        ctx, p, ds = await setup(tmp_path)
        tj = await ts.propose(ctx, p, ds.id)
        job = ctx.jobs.submit("train", p.id, lambda j: asyncio.sleep(0))
        await ts._run_slurm(ctx, tj, job)
    asyncio.run(go())
    assert list(shipped.values()) == ["[0,2,4]"]
    # the list never goes through sbatch --export (which splits on commas)
    assert not any("episodes" in c for c in calls if "sbatch --parsable" in c)
