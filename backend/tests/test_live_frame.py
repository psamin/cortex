"""#364: on Windows a live.jpg held open by the studio's live view must not fail a run."""

from pathlib import Path

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.services import recording_service as rec


async def test_a_locked_live_frame_does_not_fail_the_run(tmp_path, monkeypatch):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    d = ctx.local(rec.live_key("prj_1"))
    d.mkdir(parents=True)
    (d / "live.jpg").write_bytes(b"old frame")
    real = Path.unlink

    def locked(self, missing_ok=False):
        if self.name == "live.jpg":
            raise PermissionError(13, "The process cannot access the file because it is being used")
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    assert rec.fresh_live_dir(ctx, "prj_1") == d  # no exception: the new run overwrites the frame
