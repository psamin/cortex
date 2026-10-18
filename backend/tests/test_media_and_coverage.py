import asyncio

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Condition, CreateProjectRequest, Episode, EpisodeSource, EpisodeStatus
from cortex.services import agent_orchestrator as orch
from cortex.services.import_service import dedup_key


class SigningBlobs:
    """Stands in for InsForge storage: every url() call signs a new URL."""

    def __init__(self):
        self.calls = 0

    async def url(self, key, expires_s=3600):
        self.calls += 1
        return f"https://cdn.example/{key}?sig={self.calls}"


def test_signed_media_urls_are_reused_until_near_expiry(tmp_path):
    async def run():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        ctx.blobs = SigningBlobs()
        a, b = await ctx.media_url("v.mp4"), await ctx.media_url("v.mp4")
        assert a == b and ctx.blobs.calls == 1  # polling keeps the same src: the <video> doesn't reload
        url, _ = ctx.signed_urls["v.mp4"]
        ctx.signed_urls["v.mp4"] = (url, 0.0)  # pretend it is about to expire
        assert await ctx.media_url("v.mp4") != a and ctx.blobs.calls == 2
    asyncio.run(run())


def test_import_dedup_is_by_repo_or_archive_bytes():
    assert dedup_key("p", {"hub": "lerobot/x"}, 5) == dedup_key("p", {"hub": "lerobot/x"}, 5)
    assert dedup_key("p", {"hub": "lerobot/x"}, 5) != dedup_key("p", {"hub": "lerobot/x", "revision": "v2"}, 5)
    same_folder_a = {"path": "/u/upl_1/raw/lerobot", "sha256": "aa"}
    same_folder_b = {"path": "/u/upl_2/raw/lerobot", "sha256": "bb"}
    assert dedup_key("p", same_folder_a, 5) != dedup_key("p", same_folder_b, 5)  # different archives both import
    assert dedup_key("p", same_folder_a, 5) == dedup_key("p", {**same_folder_b, "sha256": "aa"}, 5)


def test_imported_demos_do_not_count_toward_sim_coverage(tmp_path):
    async def run():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
        cell = p.plan.cells[0].condition
        for i, src in enumerate([EpisodeSource.imported, EpisodeSource.imported, EpisodeSource.sim_scripted]):
            await ctx.repo.save("episodes", Episode(project_id=p.id, index=i, source=src, seed=i, num_frames=10,
                                                    fps=30, success=True, status=EpisodeStatus.accepted,
                                                    condition=Condition(cube_region=cell.cube_region,
                                                                        lighting=cell.lighting)))
        st = await orch.project_state(ctx, p)
        assert next(c for c in st.project.plan.cells if c.condition.key == cell.key).collected == 1
    asyncio.run(run())
