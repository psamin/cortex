"""Register a policy trained outside the studio (e.g. a Futurama SmolVLA fine-tune) in a project.

    uv run python scripts/register_external_policy.py --project prj_x --checkpoint <dir with model.safetensors>
        --log slurm-cortex-jenga-10560.out --job-id 10560 --sources jakmilller/jenga-v2,Mahanthesh0r/jenga_pull
        --steps 10000 --batch 32 [--note "..."] [--dataset ds_x]

--checkpoint and --log may also be host:path (e.g. futurama:~/cortex-jenga/output/jenga/jenga_010000). Then
only the checkpoint's JSON files and the log are copied; the weights stay on the cluster and the checkpoint
is registered at that location.

Records a TrainingJob (status succeeded, backend futurama_slurm) with the real config and the loss
curve parsed from the Slurm log, then registers the checkpoint, so it shows as the project's next
policy version. Nothing is invented: the numbers come from the log; no evaluation is claimed.
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
from cortex.models import DatasetVersion, JobStatus, ResourceEstimate, TrainingJob
from cortex.services import futurama_files, training_service


def _remote_has_weights(remote: str) -> bool:
    host, path = remote.split(":", 1)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host,
                        f"test -s {path}/model.safetensors && echo present"], capture_output=True, text=True, timeout=60)
    return "present" in r.stdout


async def main(a: argparse.Namespace) -> None:
    ctx = await build_ctx(settings())
    dss = await ctx.repo.list("dataset_versions", DatasetVersion, project_id=a.project)
    if not dss:
        raise SystemExit(f"{a.project} has no dataset version: import the demos first")
    ds = next((d for d in dss if d.id == a.dataset), None) if a.dataset else dss[-1]
    if ds is None:
        raise SystemExit(f"{a.dataset} is not a dataset version of {a.project}")
    remote = futurama_files.is_remote(a.checkpoint)
    ck_dir = Path(a.checkpoint)
    if remote:
        if not _remote_has_weights(a.checkpoint):
            raise SystemExit(f"no model.safetensors at {a.checkpoint} (is the GT VPN on?)")
        ck_dir = futurama_files.fetch(a.checkpoint, ctx.local(f"imports/checkpoints/{futurama_files.local_label(a.checkpoint)}"),
                                      ["*.json"])
    log_path = Path(a.log)
    if futurama_files.is_remote(a.log):
        host, path = a.log.split(":", 1)
        parent, name = path.rsplit("/", 1)
        log_path = futurama_files.fetch(f"{host}:{parent}", ctx.local("imports/logs"), [name]) / name
    log = log_path.read_text(errors="replace")
    cfg = json.loads((ck_dir / "config.json").read_text())
    tj = TrainingJob(
        project_id=a.project, dataset_version_id=ds.id, policy_type=cfg.get("type", "smolvla"),
        backend="futurama_slurm", external_id=a.job_id, status=JobStatus.succeeded, approved_by="owner (chat)",
        config={"policy.path": "lerobot/smolvla_base", "steps": a.steps, "batch_size": a.batch,
                "trained_on": a.sources.split(","), "note": a.note,
                "held_out": "last episodes of each source (see split.json)", "lerobot": "0.6.1",
                **({"checkpoint_location": a.checkpoint} if remote else {})},
        estimate=ResourceEstimate(gpu="1x H200 (Futurama general partition)", gpu_hours=0.0, est_cost_usd=0.0,
                                  paid=False, notes="ran on Futurama; see the Slurm log"),
        rationale=f"SmolVLA (~450M) fine-tuned from lerobot/smolvla_base on real SO-100/101 demos: {a.sources}",
        metrics={"loss_curve": training_service.parse_loss(log)},
    )
    await ctx.repo.save("training_jobs", tj)
    if remote:
        ck = await training_service.register_remote_checkpoint(ctx, a.project, tj.id, a.checkpoint, cfg, a.steps,
                                                                actor="register_external_policy")
    else:
        ck = await training_service.register_checkpoint(ctx, a.project, tj.id, str(ck_dir.resolve()), a.steps)
    print(f"{a.project}: training job {tj.id} ({len(tj.metrics['loss_curve'])} loss points), checkpoint {ck.id}"
          f" at {ck.path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("project", "checkpoint", "log", "job-id", "sources"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--note", default="")
    p.add_argument("--dataset", help="dataset version the job is recorded against (default: the newest)")
    asyncio.run(main(p.parse_args()))
