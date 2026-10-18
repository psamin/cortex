"""An explicit, labelled CPU smoke run of ``lerobot-train`` on this machine.

It is not a substitute for the approved GPU training job: it runs a few steps to prove the dataset
trains end to end and yields a real (but untrained-for-the-task) checkpoint. The checkpoint is
registered without a training job, so the GPU job is never marked succeeded by it, and it carries
``metrics.cpu_smoke_run = 1`` so every view can label it.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
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
    base = ctx.local(f"projects/{p.id}/smoke")  # short: Windows MAX_PATH applies to the files lerobot writes
    base.mkdir(parents=True, exist_ok=True)
    dest = base / str(len(list(base.iterdir()))) / "pretrained_model"
    # lerobot-train writes optimizer state several levels deep; a short scratch dir keeps Windows paths under
    # MAX_PATH. Only pretrained_model/ is kept, under the project.
    scratch = Path(tempfile.mkdtemp(prefix="cx_smoke_"))
    out_dir = scratch / "out"  # lerobot-train refuses an existing output_dir
    cmd = command(ctx, ds, out_dir)

    async def run(job) -> str:
        ctx.jobs.log(job, "$ " + " ".join(cmd))
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT, cwd=str(scratch))
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
        try:
            ckpts = sorted((out_dir / "checkpoints").glob("[0-9]*"))
            done = bool(ckpts) and (ckpts[-1] / "pretrained_model" / "model.safetensors").exists()
            # On Windows without Developer Mode, lerobot-train saves the checkpoint and then fails to create the
            # checkpoints/last symlink (WinError 1314). The checkpoint itself is complete, so keep it.
            symlink_only = done and any("symlink" in ln or "WinError 1314" in ln for ln in tail[-6:])
            if rc != 0 and not symlink_only:
                raise RuntimeError(f"lerobot-train exited {rc}: {' | '.join(tail[-4:])[-400:]}")
            if symlink_only:
                ctx.jobs.log(job, "note: checkpoint saved; only the 'last' symlink failed (Windows privilege)")
            if not ckpts:
                raise RuntimeError("lerobot-train finished without writing a checkpoint")
            shutil.copytree(ckpts[-1] / "pretrained_model", dest)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        ck = await training_service.register_checkpoint(ctx, p.id, None, str(dest), SMOKE_STEPS, actor="gui")
        ck.metrics = {**ck.metrics, "cpu_smoke_run": 1.0}
        await ctx.repo.save("checkpoints", ck, status="ready")
        return ck.id

    return ctx.jobs.submit("train_smoke", p.id, run, idempotency_key=f"smoke:{dest}")


