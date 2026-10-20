"""#367: blob keys use '/' on every OS."""

from pathlib import PureWindowsPath

from cortex.config import Settings
from cortex.context import build_ctx


def test_a_windows_relative_path_becomes_a_forward_slash_key():
    data_dir = PureWindowsPath(r"C:\Users\anees\Cortex\.cortex-data")
    f = data_dir / "projects" / "prj_x" / "datasets" / "ds_y" / "lerobot" / "meta" / "info.json"
    assert f.relative_to(data_dir).as_posix() == "projects/prj_x/datasets/ds_y/lerobot/meta/info.json"


async def test_publish_file_normalises_backslash_keys(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    (tmp_path / "projects" / "p").mkdir(parents=True)
    (tmp_path / "projects" / "p" / "a.json").write_text("{}")
    got: list[str] = []

    class Blobs:  # a remote store (not LocalBlobStore)
        async def put_bytes(self, key, data, ctype):
            got.append(key)
    ctx.blobs = Blobs()  # type: ignore[assignment]
    await ctx.publish_file("projects\\p\\a.json")
    assert got == ["projects/p/a.json"]
