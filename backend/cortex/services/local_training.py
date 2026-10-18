"""An explicit, labelled CPU smoke run of ``lerobot-train`` on this machine.

It is not a substitute for the approved GPU training job: it runs a few steps to prove the dataset
trains end to end and yields a real (but untrained-for-the-task) checkpoint. The checkpoint is
registered without a training job, so the GPU job is never marked succeeded by it, and it carries
``metrics.cpu_smoke_run = 1`` so every view can label it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from ..context import Ctx
from ..models import DatasetVersion, Project
from . import training_service

SMOKE_STEPS = 20


def _exe() -> Path:
    scripts = Path(sys.executable).parent
    return scripts / ("lerobot-train.exe" if sys.platform == "win32" else "lerobot-train")


def available() -> str | None:
    """None if a smoke run can start here, else why not."""
    return None if _exe().exists() else f"lerobot-train is not installed next to {sys.executable}"


def command(ctx: Ctx, ds: DatasetVersion, out_dir: Path) -> list[str]:
    return [str(_exe()), f"--dataset.repo_id=cortex/{ds.project_id}_{ds.id}", f"--dataset.root={ctx.local(ds.lerobot_root or '')}",
            "--policy.type=act", "--policy.device=cpu", "--policy.push_to_hub=false", f"--steps={SMOKE_STEPS}",
            "--batch_size=2", "--num_workers=0", f"--save_freq={SMOKE_STEPS}", "--log_freq=5", "--seed=1000",
            f"--output_dir={out_dir}", f"--job_name=cortex_smoke_{ds.id}", "--wandb.enable=false"]


async def start_smoke(ctx: Ctx, p: Project, dataset_version_id: str):
    if (why := available()) is not None:
        raise RuntimeError(why)
    ds = await ctx.repo.get("dataset_versions", dataset_version_id, DatasetVersion)
    if ds is None or not ds.lerobot_root:
        raise FileNotFoundError(f"dataset {dataset_version_id} has no local LeRobot directory")
    base = ctx.local(f"projects/{p.id}/training/smoke_{ds.id}")
    base.mkdir(parents=True, exist_ok=True)
    out_dir = base / f"run{len(list(base.iterdir()))}"  # lerobot-train refuses an existing output_dir
    cmd = command(ctx, ds, out_dir)

    async def run(job) -> str:
        ctx.jobs.log(job, "$ " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, cwd=str(base))
        tail: list[str] = []
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            tail = [*tail[-30:], line]
            ctx.jobs.log(job, line)
            if "step:" in line:
                ctx.jobs.update(job, message=line[-120:])
        rc = await proc.wait()
        if rc != 0:
            raise RuntimeError(f"lerobot-train exited {rc}: {' | '.join(tail[-4:])[-400:]}")
        ck_dir = out_dir / "checkpoints" / "last" / "pretrained_model"
        ck = await training_service.register_checkpoint(ctx, p.id, None, str(ck_dir), SMOKE_STEPS, actor="gui")
        ck.metrics = {**ck.metrics, "cpu_smoke_run": 1.0}
        await ctx.repo.save("checkpoints", ck, status="ready")
        return ck.id

    return ctx.jobs.submit("train_smoke", p.id, run, idempotency_key=f"smoke:{out_dir}")


