"""Import a held-out policy replay (infra/futurama/replay_policy.py output) into its project.

    uv run python scripts/import_replay.py futurama:~/cortex-jenga/replay/jenga
    uv run python scripts/import_replay.py futurama:~/cortex-jenga/replay/suture_v1 --project prj_25dcd4dd3804
    uv run python scripts/import_replay.py ./some/local/replay/dir --project prj_x

Copies only metrics.json and the ep<N>.mp4 videos (a few MB), uploads the videos to storage and records a
PolicyReplay: the policy run open loop on held-out real demos, scored by joint error against the human.
That is not a success rate, and none is recorded. The project defaults to the task's entry in
infra/futurama/projects.json; the checkpoint is linked when one with the same name is registered there.
Re-running the same source updates the record instead of adding another.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from cortex.config import settings
from cortex.context import build_ctx
from cortex.services import futurama_files, replay_service


async def main(a: argparse.Namespace) -> None:
    ctx = await build_ctx(settings())
    if futurama_files.is_remote(a.source):
        folder = ctx.local(f"imports/replay/{futurama_files.local_label(a.source)}")
        futurama_files.fetch(a.source, folder, ["metrics.json", "*.mp4"])
    else:
        folder = Path(a.source).resolve()
    task = json.loads((folder / "metrics.json").read_text()).get("task", "")
    pid = a.project or replay_service.project_for_task(ctx, task)
    if not pid:
        raise SystemExit(f"No project for task '{task}': pass --project or add it to "
                         f"{ctx.settings.futurama_projects_file}")
    rp = await replay_service.import_replay_dir(ctx, pid, folder, a.source, actor="import_replay")
    ck = rp.checkpoint_id or f"{rp.checkpoint_name} (not registered in {pid})"
    print(f"{pid}: replay {rp.id}, {len(rp.episodes)} held-out episodes, "
          f"{rp.joint_mae_deg} deg mean joint error, checkpoint {ck}, "
          f"{sum(1 for e in rp.episodes if e.video_key)} videos stored ({ctx.blobs.name})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="replay folder: local path or host:path (e.g. futurama:~/cortex-jenga/replay/jenga)")
    p.add_argument("--project", help="project id (default: infra/futurama/projects.json by task)")
    asyncio.run(main(p.parse_args()))
