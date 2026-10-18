"""The viewer's front and wrist views (#214): display-only 640x480 copies of the policy cameras.

Policy observations and datasets keep the env's 96x128 frames; only the videos people watch change.
"""

import asyncio
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import Condition, Episode, EpisodeSource
from cortex.sim.so101 import IMG_H, IMG_W, PickPlaceEnv, Variation, configure_gl
from cortex.sim.viewer import CAM_EVERY, CAM_H, CAM_W, PolicyCameras
from cortex.workers import sim_worker


@pytest.fixture(scope="module")
def env():
    configure_gl()
    e = PickPlaceEnv(episode_steps=100)
    e.reset(1, Variation())
    yield e
    e.close()


def probe(path: Path) -> tuple[int, int, int]:
    """(width, height, frames) of an mp4."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        w, h = s.codec_context.width, s.codec_context.height
        return w, h, sum(1 for _ in c.decode(s))


class Hold:
    """Holds the start pose; reads pixels like a learned policy would."""
    name, needs_pixels = "hold", True

    def reset(self, env, obs, seed):
        self.q = obs["agent_pos"].copy()

    def act(self, obs):
        return self.q


def test_display_copies_are_the_policy_cameras_at_640x480(env):
    m = env.world.model
    q, g = m.vis.quality, m.vis.global_
    before = (q.offsamples, q.shadowsize, g.offwidth, g.offheight)
    cams = PolicyCameras(m, env.env.cameras)
    try:
        frames = cams.render(env.world.data)
    finally:
        cams.close()
    assert (q.offsamples, q.shadowsize, g.offwidth, g.offheight) == before  # observation renderers unchanged
    assert set(frames) == {"front", "wrist"}
    assert all(f.shape == (CAM_H, CAM_W, 3) and f.dtype == np.uint8 for f in frames.values())
    obs = env._obs()["pixels"]
    for k, f in frames.items():  # same camera, scene and lighting as the observation, just sharper
        assert obs[k].shape == (IMG_H, IMG_W, 3)
        small = np.asarray(Image.fromarray(f).resize((IMG_W, IMG_H), Image.BILINEAR), np.float32)
        assert np.abs(small - obs[k].astype(np.float32)).mean() < 20, k


def test_recorded_demo_writes_hd_videos_and_keeps_the_dataset_frames(env, tmp_path):
    preview = sim_worker.Preview(env, tmp_path, live=False, policy_cams=True)
    try:
        obs = env.reset(2, Variation())
        ep = sim_worker.rollout(env, Hold(), obs, 2, 6, preview, record=True)
    finally:
        preview.close()
    buf = ep.pop("buf")
    assert len(buf["front"]) == 6 and buf["front"][0].shape == (IMG_H, IMG_W, 3)
    assert {k: len(v) for k, v in buf["hd"].items()} == {"front": 6 // CAM_EVERY, "wrist": 6 // CAM_EVERY}
    sim_worker.save_npz(env, buf, tmp_path)
    sim_worker.save_video(ep.pop("video"), tmp_path, hd=buf["hd"])
    with np.load(tmp_path / "episode.npz") as z:
        assert z["front"].shape == z["wrist"].shape == (6, IMG_H, IMG_W, 3)
    assert probe(tmp_path / "front_hd.mp4") == probe(tmp_path / "wrist_hd.mp4") == (CAM_W, CAM_H, 6 // CAM_EVERY)
    assert probe(tmp_path / "front.mp4")[:2] == (1280, 960)  # the studio view is unchanged


def test_previews_without_policy_cams_record_no_copies(env, tmp_path):
    preview = sim_worker.Preview(env, tmp_path, live=False)
    try:
        obs = env.reset(3, Variation())
        ep = sim_worker.rollout(env, Hold(), obs, 3, 4, preview, record=True)
    finally:
        preview.close()
    assert not ep["buf"].get("hd")


def test_agent_sessions_save_hd_copies(env, tmp_path):
    from cortex.sim.manipulation import ManipulationSession
    from cortex.workers.agent_worker import save_attempt

    cams = PolicyCameras(env.world.model, env.env.cameras)
    try:
        s = ManipulationSession(env, mode="privileged", cam_renderer=cams)
        s.reset_environment(4, Variation())
        for _ in range(4):
            s._step(s.cmd)
    finally:
        cams.close()
    assert len(s.rec.front) == 4 and len(s.rec.hd["wrist"]) == 4 // CAM_EVERY
    save_attempt(s, env, tmp_path, {})
    assert probe(tmp_path / "wrist_hd.mp4") == (CAM_W, CAM_H, 4 // CAM_EVERY)
    with np.load(tmp_path / "episode.npz") as z:
        assert z["wrist"].shape == (4, IMG_H, IMG_W, 3)


def checker(n: int = 4) -> np.ndarray:
    c = (np.indices((IMG_H, IMG_W)).sum(0) // 8 % 2 * 255).astype(np.uint8)
    return np.repeat(c[None, ..., None], 3, -1).repeat(n, 0)


def test_old_episodes_are_upscaled_smoothly_not_blocky():
    from cortex.services.studio_service import upscale

    up = upscale(checker(1)[0])
    assert up.shape == (2 * IMG_H, 2 * IMG_W, 3)
    edges = (up > 20) & (up < 235)
    assert edges.mean() > 0.02  # nearest-neighbour would leave only 0 and 255


def test_viewer_serves_hd_copies_and_falls_back_for_old_episodes(tmp_path):
    from cortex.services import studio_service
    from cortex.services.recording_service import episode_key

    async def run():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        ep = Episode(project_id="p", index=0, source=EpisodeSource.sim_scripted, condition=Condition(), seed=1,
                     num_frames=4, fps=30, success=True)
        d = ctx.local(episode_key("p", ep.id))
        d.mkdir(parents=True)
        np.savez_compressed(d / "episode.npz", front=checker(), wrist=checker())
        ep.artifacts = {"episode": episode_key("p", ep.id, "episode.npz")}
        await ctx.repo.save("episodes", ep)
        old = await studio_service.camera_video(ctx, ep.id, "wrist")
        assert probe(old) == (2 * IMG_W, 2 * IMG_H, 4)

        sim_worker.write_mp4([np.zeros((CAM_H, CAM_W, 3), np.uint8)] * 2, d / "wrist_hd.mp4", fps=15)
        ep.artifacts["wrist_hd"] = episode_key("p", ep.id, "wrist_hd.mp4")
        await ctx.repo.save("episodes", ep)
        assert await studio_service.camera_video(ctx, ep.id, "wrist") == d / "wrist_hd.mp4"
    asyncio.run(run())


def test_recorded_episodes_register_their_hd_videos(tmp_path):
    from cortex.services.recording_service import episode_key, finalize_episode

    async def run():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        ep = Episode(project_id="p", index=0, source=EpisodeSource.sim_scripted, condition=Condition(), seed=1,
                     num_frames=4, fps=30, success=None)
        d = ctx.local(episode_key("p", ep.id))
        d.mkdir(parents=True)
        np.savez_compressed(d / "episode.npz", front=checker(), wrist=checker(), timestamp=np.arange(4) / 30,
                            action=np.zeros((4, 6), np.float32), action_low=-np.ones(6, np.float32),
                            action_high=np.ones(6, np.float32))
        (d / "front_hd.mp4").write_bytes(b"x")
        ev = {"num_frames": 4, "fps": 30, "success": True}
        ep = await finalize_episode(ctx, ep, ev, None, False)
        assert ep.artifacts["front_hd"] == episode_key("p", ep.id, "front_hd.mp4")
        assert "wrist_hd" not in ep.artifacts  # only files that were written
    asyncio.run(run())
