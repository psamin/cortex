"""The studio's live view streams projects/<pid>/live/live.jpg: a new sim run never shows the last run's frame,
and two runs in one project never fail each other over it."""

import asyncio
import os

import numpy as np

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.services.recording_service import fresh_live_dir, live_key
from cortex.workers import livefile


def test_a_new_sim_run_never_streams_the_last_runs_frame(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None, agent_engine="rules"))
        old = ctx.local(live_key("prj_x"))
        old.mkdir(parents=True)
        (old / "live.jpg").write_bytes(b"the cube scene")
        return fresh_live_dir(ctx, "prj_x"), fresh_live_dir(ctx, "prj_new")  # a project's first run too

    d, new = asyncio.run(go())
    assert d.is_dir() and not (d / "live.jpg").exists() and new.is_dir()


def test_two_runs_writing_live_frames_never_fail_each_other(tmp_path, monkeypatch):
    """A recording failed with "No such file or directory: live.tmp.jpg" when another run swapped it away."""
    frame = np.zeros((8, 8, 3), np.uint8)

    def gone(*a):
        raise FileNotFoundError(a[0])

    monkeypatch.setattr(livefile.os, "replace", gone)
    assert livefile.write_live_jpg(frame, tmp_path) is False  # skipped, not raised
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.undo()
    swapped = []
    real = os.replace
    monkeypatch.setattr(livefile.os, "replace", lambda a, b: (swapped.append(str(a)), real(a, b)))
    assert livefile.write_live_jpg(frame, tmp_path)
    assert f".{os.getpid()}." in swapped[0]  # each process its own temp file
    assert (tmp_path / "live.jpg").exists() and not list(tmp_path.glob("*.tmp"))
