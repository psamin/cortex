"""Copying Futurama results (#179): only the light files come along; tasks map to their projects."""

import json
from types import SimpleNamespace

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
