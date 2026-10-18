"""Import a hand-built sim task's Futurama results (infra/futurama/sim_task.sbatch) into its project.

    uv run python scripts/import_sim_task.py futurama:~/cortex-sim jenga
    uv run python scripts/import_sim_task.py futurama:~/cortex-sim unscrew
    uv run python scripts/import_sim_task.py /path/to/local/job_dir suture --project prj_x

The project is "Jenga (sim)", "Screwdriver (sim)" (unscrew) or "Suture (sim)", reused by name or created;
--project picks one explicitly. Copies only light files: metrics.json, configs, the example and evaluation
videos, the Slurm logs, the dataset if it is under --dataset-max-gb, and checkpoint weights only when under
--weights-max-mb (otherwise the checkpoint is registered at its Futurama path). Then records the dataset
version, the oracle's demo run, a training job per variant, a checkpoint per saved step and an evaluation
per evaluated step (see backend/cortex/services/sim_import_service.py). Safe to re-run while runs land:
it updates what it recorded before. The project id is added to infra/futurama/projects.json as sim_<task>.
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from cortex.config import settings
from cortex.context import build_ctx
from cortex.services import futurama_files, sim_import_service

LIGHT = ["*.json", "*.mp4"]
HEAVY_DIRS = ("dataset/data/", "dataset/videos/", "dataset/images/")


def _small_weights(src_task: str, max_mb: int) -> list[str]:
    """Remote model.safetensors files under max_mb, relative to src_task."""
    if max_mb <= 0:
        return []
    host, path = src_task.split(":", 1)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                        f"cd {path} 2>/dev/null && find runs -name model.safetensors -size -{max_mb}M"],
                       capture_output=True, text=True, timeout=120)
    return [ln.strip() for ln in r.stdout.splitlines() if ln.strip().endswith("model.safetensors")]


async def main(a: argparse.Namespace) -> None:
    ctx = await build_ctx(settings())
    src_task = f"{a.job_dir.rstrip('/')}/{a.task}"
    if futurama_files.is_remote(a.job_dir):
        base = ctx.local(f"imports/sim/{futurama_files.local_label(a.job_dir)}")
        mirror = futurama_files.fetch(src_task, base / a.task, LIGHT, exclude=HEAVY_DIRS)
        for rel in _small_weights(src_task, a.weights_max_mb):
            parent, name = rel.rsplit("/", 1)
            futurama_files.fetch(f"{src_task}/{parent}", mirror / parent, [name], recursive=False)
        logs_dir = futurama_files.fetch(a.job_dir, base / "logs", ["slurm-*.out"], recursive=False)
        dataset_root = None
        size = futurama_files.remote_size_bytes(f"{src_task}/dataset")
        if size is not None and size <= a.dataset_max_gb * 1024 ** 3:
            dataset_root = futurama_files.fetch(f"{src_task}/dataset", mirror / "dataset", ["*"])
        elif size is not None:
            print(f"dataset is {size / 1024 ** 3:.1f} GB: kept on Futurama only ({src_task}/dataset)")
    else:
        mirror, logs_dir = Path(src_task).resolve(), Path(a.job_dir).resolve()
        dataset_root = mirror / "dataset" if (mirror / "dataset" / "meta" / "info.json").exists() else None
    logs = sorted(logs_dir.glob("slurm-*.out"))
    out = await sim_import_service.import_task(ctx, a.task, mirror, src_task, logs, dataset_root, a.project)
    if out.get("project_id"):
        f = ctx.settings.futurama_projects_file
        mapping = json.loads(f.read_text()) if f.exists() else {}
        if f"sim_{a.task}" not in mapping:  # add only: never repoint an existing entry
            mapping[f"sim_{a.task}"] = out["project_id"]
            f.write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("job_dir", help="sim_task.sbatch's <job_dir>: local path or host:path (futurama:~/cortex-sim)")
    p.add_argument("task", help="jenga, unscrew, suture, ...")
    p.add_argument("--project", help="import into this project instead of the one named for the task")
    p.add_argument("--dataset-max-gb", type=float, default=1.0, help="copy the dataset when at most this size")
    p.add_argument("--weights-max-mb", type=int, default=50, help="copy checkpoint weights under this size")
    asyncio.run(main(p.parse_args()))
