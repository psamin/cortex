"""Policy training: propose -> human approval -> launch -> stream logs -> checkpoint.

Backends:
  futurama_slurm  ACT via `lerobot-train` on the GT Futurama cluster, submitted over
                  SSH (needs GT VPN + key auth for FUTURAMA_SSH_HOST). Free to the
                  user but still requires approval.
  manual          Cortex writes the exact command bundle; the user runs it on any GPU
                  and registers the resulting pretrained_model/ directory.
Nothing is ever marked succeeded without a checkpoint on disk.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable
from pathlib import Path

from ..context import Ctx
from ..models import (
    GPU_CHOICES,
    Checkpoint,
    DatasetVersion,
    JobStatus,
    Project,
    ResourceEstimate,
    TrainingJob,
    now,
)
from ..sim.so101 import FPS

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[3]
SBATCH = REPO / "infra" / "futurama" / "train_act.sbatch"
POLL_S = 30


class ApprovalRequired(PermissionError):
    code = "training_approval"


def estimate(policy_type: str, steps: int, backend: str, n_frames: int, gres: str = "gpu:l40s:1") -> ResourceEstimate:
    # ACT (~52M params, 2x 96x128 cams, batch 8): ~12 it/s measured class on L40S-class GPUs is
    # an assumption until our first run reports it; SmolVLA is ~5x slower per step.
    its = 12.0 if policy_type == "act" else 2.5
    hours = steps / its / 3600 + 0.15  # + env setup
    paid = backend == "hf_jobs"
    return ResourceEstimate(
        gpu=f"{GPU_CHOICES.get(gres, gres)} (Futurama general partition)" if backend == "futurama_slurm"
        else "1x GPU (your choice)",
        gpu_hours=round(hours, 2), est_cost_usd=round(hours * 1.8, 2) if paid else 0.0, paid=paid,
        notes=(f"{steps} steps, batch 8 over {n_frames} frames; throughput is an estimate until measured. "
               + ("Free GT allocation; queue wait not included." if backend == "futurama_slurm" else "")))


async def propose(ctx: Ctx, project: Project, dataset_version_id: str, policy_type: str = "act",
                  steps: int | None = None, actor: str = "gui") -> TrainingJob:
    ds = await ctx.repo.get("dataset_versions", dataset_version_id, DatasetVersion)
    if ds is None:
        raise KeyError(dataset_version_id)
    infra = project.infra
    backend = infra.training_backend if infra.training_backend != "auto" else (
        "futurama_slurm" if ctx.settings.futurama_ssh_host else "manual")
    if backend == "futurama_slurm" and not ctx.settings.futurama_ssh_host:
        raise ValueError("Training on Futurama was chosen, but FUTURAMA_SSH_HOST is not set in .env.")
    steps = steps or (20_000 if policy_type == "act" else 20_000)
    cfg = {
        "policy.type": policy_type, "steps": steps, "batch_size": 8, "save_freq": steps // 2,
        "dataset.repo_id": f"cortex/{project.id}_{ds.id}", "dataset.root": ds.lerobot_root,
        "fps": FPS, "seed": 1000, "lerobot": "0.6.1",
        "cameras": ["observation.images.front", "observation.images.wrist"],
        "slurm": {"partition": ctx.settings.futurama_partition, "gres": infra.training_gres,
                  "time_min": infra.training_walltime_min},
    }
    if policy_type == "smolvla":
        cfg["policy.path"] = "lerobot/smolvla_base"
    n_train = sum(1 for ln in ds.links if ln.split == "train")
    why = (f"ACT imitation learning on {ds.num_source_episodes} source demos ({n_train} train LeRobot episodes "
           f"incl. augmented views): it trains in minutes, fits 30 fps joint control and needs no language "
           f"conditioning for a single task.") if policy_type == "act" else (
           "SmolVLA fine-tuning: language-conditioned and pretrained, but ~5x slower per step.")
    tj = TrainingJob(project_id=project.id, dataset_version_id=ds.id, policy_type=policy_type,  # type: ignore[arg-type]
                     backend=backend, config=cfg, rationale=why,  # type: ignore[arg-type]
                     estimate=estimate(policy_type, steps, backend, ds.num_frames, infra.training_gres))
    await ctx.repo.save("training_jobs", tj)
    ctx.changed(project.id, "training_jobs", tj.id)
    await ctx.audit(actor, "propose_training_job", project.id, {"dataset_version_id": ds.id, "policy": policy_type},
                    "ok", f"{tj.id} proposed ({backend})")
    return tj


async def approve(ctx: Ctx, tj_id: str, approver: str, actor: str, token: str | None,
                  human_verified: bool = False) -> TrainingJob:
    tj = await _get(ctx, tj_id)
    # gui = a human click; autopilot = the project's standing approval, checked by the caller;
    # human_verified = a human click in an MCP App view, proven by a single-use nonce (mcp_views).
    if actor not in ("gui", "autopilot") and not human_verified:
        secret = ctx.settings.cortex_approval_secret
        if not secret or token != secret:
            await ctx.audit(actor, "approve_training_job", tj.project_id, {"id": tj_id}, "denied",
                            "agent approval refused: approve in the Cortex studio")
            raise ApprovalRequired("Training must be approved by a human in the Cortex studio "
                                   "(or with the operator's approval token).")
    if tj.status != JobStatus.proposed:
        return tj
    tj.status, tj.approved_by, tj.updated_at = JobStatus.approved, approver, now()
    await ctx.repo.save("training_jobs", tj)
    ctx.changed(tj.project_id, "training_jobs", tj.id)
    await ctx.audit(actor, "approve_training_job", tj.project_id, {"id": tj_id, "approver": approver}, "ok",
                    f"approved by {approver}", approved=True)
    return tj


async def launch(ctx: Ctx, tj_id: str, actor: str = "gui") -> TrainingJob:
    tj = await _get(ctx, tj_id)
    if tj.status in (JobStatus.queued, JobStatus.running, JobStatus.succeeded):
        return tj  # idempotent
    if tj.status != JobStatus.approved:
        raise ApprovalRequired(f"{tj.id} is '{tj.status}'; it must be approved before launch")
    job_dir = ctx.local(f"projects/{tj.project_id}/training/{tj.id}")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "command.sh").write_text(manual_command(ctx, tj))
    if tj.backend == "manual":
        tj.status = JobStatus.queued
        tj.error = None
        await _save(ctx, tj, "manual: run command.sh on a GPU machine, then register the checkpoint")
    else:
        tj.status = JobStatus.queued
        await _save(ctx, tj, "submitting to Futurama")
        ctx.jobs.submit("train", tj.project_id, lambda job: _tracked(ctx, tj, _run_slurm(ctx, tj, job)),
                        idempotency_key=f"train:{tj.id}")
    await ctx.audit(actor, "launch_training_job", tj.project_id, {"id": tj.id, "backend": tj.backend}, "ok",
                    f"{tj.id} launched on {tj.backend}", approved=True)
    return tj


def manual_command(ctx: Ctx, tj: TrainingJob) -> str:
    c = tj.config
    return (f"# Cortex training job {tj.id} (approved by {tj.approved_by})\n"
            f"pip install 'lerobot[dataset,training]==0.6.1'\n"
            f"lerobot-train --dataset.repo_id={c['dataset.repo_id']} --dataset.root={ctx.local(c['dataset.root'])} "
            f"--policy.type={c['policy.type']} --policy.device=cuda --policy.push_to_hub=false "
            f"--steps={c['steps']} --batch_size={c['batch_size']} --save_freq={c['save_freq']} --seed={c['seed']} "
            f"--output_dir=outputs/{tj.id} --job_name=cortex_{tj.id} --wandb.enable=false\n"
            f"# then: POST /api/projects/{tj.project_id}/checkpoints "
            f"{{\"training_job_id\": \"{tj.id}\", \"path\": \"outputs/{tj.id}/checkpoints/last/pretrained_model\"}}\n")


async def _sh(*args: str, timeout: float = 600) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(p.communicate(), timeout)
    except TimeoutError:
        p.kill()
        return 124, "timeout"
    return p.returncode or 0, out.decode(errors="replace")


async def _tracked(ctx: Ctx, tj: TrainingJob, work: Awaitable[str]) -> str:
    """Run Slurm work so the durable TrainingJob always ends terminal: a failed ssh/rsync/sbatch or a
    stop must not leave it 'queued' (Autopilot waits on the stored status and would wait forever)."""
    try:
        return await work
    except BaseException as e:
        cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob) or tj
        if cur.status not in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            stopped = isinstance(e, asyncio.CancelledError)
            cur.status = JobStatus.cancelled if stopped else JobStatus.failed
            cur.error = "stopped by user" if stopped else f"{type(e).__name__}: {e}"[:500]
            await _save(ctx, cur, cur.status)
        raise


async def _run_slurm(ctx: Ctx, tj: TrainingJob, job) -> str:
    host = ctx.settings.futurama_ssh_host
    assert host
    remote = f"{ctx.settings.futurama_repo_dir}/jobs/{tj.id}"
    ds_local = ctx.local(tj.config["dataset.root"])
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host]
    ctx.jobs.log(job, f"rsync dataset -> {host}:{remote}")
    for cmd in (ssh + [f"mkdir -p {remote}"],
                ["rsync", "-az", "--delete", f"{ds_local}/", f"{host}:{remote}/dataset/"],
                ["rsync", "-az", str(SBATCH), f"{host}:{remote}/train_act.sbatch"]):
        rc, out = await _sh(*cmd)
        if rc != 0:
            raise RuntimeError(f"{cmd[0]} failed ({rc}): {out[-400:]} — is the GT VPN up?")
    c = tj.config
    export = (f"ALL,STEPS={c['steps']},BATCH={c['batch_size']},SAVE={c['save_freq']},"
              f"REPO_ID={c['dataset.repo_id']}")
    slurm = c.get("slurm") or {}  # chosen in the studio; older jobs fall back to the .env defaults
    rc, out = await _sh(*ssh, f"cd {remote} && sbatch --parsable -p {slurm.get('partition', ctx.settings.futurama_partition)} "
                              f"--gres={slurm.get('gres', ctx.settings.futurama_gres)} "
                              f"-t {int(slurm.get('time_min', 120))} --export={export} train_act.sbatch {remote}")
    m = re.search(r"^(\d+)", out.strip().splitlines()[-1] if out.strip() else "")
    if rc != 0 or not m:
        raise RuntimeError(f"sbatch failed: {out[-400:]}")
    tj.external_id = m.group(1)
    await _save(ctx, tj, f"slurm job {tj.external_id} submitted")
    ctx.jobs.log(job, f"submitted slurm job {tj.external_id}")
    return await poll_slurm(ctx, tj, job)


def parse_loss(log: str) -> list[list[float]]:
    """[[step, loss], ...] from lerobot-train log lines like 'step:2K smpl:16K ... loss:0.231 grdn:...'."""
    pts = []
    for m in re.finditer(r"step:(\d+(?:\.\d+)?)([KM]?)\b[^\n]*?\bloss:([0-9.]+(?:e-?\d+)?)", log):
        step = float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6}[m.group(2)]
        pts.append([int(step), float(m.group(3))])
    return pts


async def poll_slurm(ctx: Ctx, tj: TrainingJob, job) -> str:
    host = ctx.settings.futurama_ssh_host
    remote = f"{ctx.settings.futurama_repo_dir}/jobs/{tj.id}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host]
    seen = 0
    while True:
        rc, state = await _sh(*ssh, f"squeue -h -j {tj.external_id} -o %T", timeout=60)
        state = state.strip()
        rc2, logtxt = await _sh(*ssh, f"cat {remote}/slurm-{tj.external_id}.out 2>/dev/null | tail -n +{seen + 1}",
                                timeout=60)
        if rc2 == 0 and logtxt:
            lines = logtxt.splitlines()
            seen += len(lines)
            for ln in lines[-40:]:
                ctx.jobs.log(job, ln)
            step = re.findall(r"step:(\d+[KM]?)", logtxt)
            if step:
                ctx.jobs.update(job, message=f"training step {step[-1]}")
            pts = parse_loss(logtxt)
            if pts:
                curve = tj.metrics.get("loss_curve", [])
                tj.metrics["loss_curve"] = (curve + pts)[-500:]
                await _save(ctx, tj, f"loss {pts[-1][1]:.4f} @ step {pts[-1][0]}")
        if rc == 0 and state in ("PENDING", "CONFIGURING"):
            if tj.status != JobStatus.queued:
                tj.status = JobStatus.queued
                await _save(ctx, tj, "pending in queue")
        elif rc == 0 and state in ("RUNNING", "COMPLETING"):
            if tj.status != JobStatus.running:
                tj.status = JobStatus.running
                await _save(ctx, tj, "running")
        elif rc == 0 and state == "":
            break  # left the queue
        await asyncio.sleep(POLL_S)
    _, tail = await _sh(*ssh, f"tail -n 5 {remote}/slurm-{tj.external_id}.out", timeout=60)
    if "TRAINING_DONE" not in tail:
        tj.status, tj.error = JobStatus.failed, f"slurm job ended without completing: {tail[-400:]}"
        await _save(ctx, tj, "failed")
        raise RuntimeError(tj.error)
    dest = ctx.local(f"projects/{tj.project_id}/training/{tj.id}/pretrained_model")
    rc, out = await _sh("rsync", "-az", f"{host}:{remote}/output/checkpoints/last/pretrained_model/", f"{dest}/")
    if rc != 0:
        raise RuntimeError(f"checkpoint fetch failed: {out[-300:]}")
    await register_checkpoint(ctx, tj.project_id, tj.id, str(dest.relative_to(ctx.data_dir)), int(tj.config["steps"]))
    return tj.id


async def register_checkpoint(ctx: Ctx, project_id: str, training_job_id: str | None, path: str,
                              step: int | None = None, actor: str = "gui") -> Checkpoint:
    p = Path(path) if Path(path).is_absolute() else ctx.local(path)
    if not (p / "config.json").exists() or not (p / "model.safetensors").exists():
        raise FileNotFoundError(f"{p} is not a LeRobot pretrained_model directory")
    import json

    ptype = json.loads((p / "config.json").read_text()).get("type", "unknown")
    ck = Checkpoint(project_id=project_id, training_job_id=training_job_id, policy_type=ptype,
                    step=step or 0, path=str(p))
    await ctx.repo.save("checkpoints", ck, status="ready")
    if training_job_id:
        tj = await ctx.repo.get("training_jobs", training_job_id, TrainingJob)
        if tj:
            tj.checkpoint_ids.append(ck.id)
            tj.status = JobStatus.succeeded
            await _save(ctx, tj, f"checkpoint {ck.id} registered")
    ctx.changed(project_id, "checkpoints", ck.id)
    stored = await _publish_checkpoint(ctx, p)
    await ctx.audit(actor, "register_checkpoint", project_id, {"path": str(p)}, "ok", f"{ck.id} ({ptype}); {stored}")
    return ck


async def _publish_checkpoint(ctx: Ctx, p: Path) -> str:
    """Copy a checkpoint inside the data dir to durable storage (the S3 gateway takes the ~200 MB weights).
    The local copy stays registered either way; a failed upload is reported, not fatal."""
    if not p.resolve().is_relative_to(ctx.data_dir.resolve()):
        return "kept at its own path (outside the data dir)"
    try:
        files = sorted(f for f in p.rglob("*") if f.is_file())
        for f in files:
            await ctx.publish_file(str(f.resolve().relative_to(ctx.data_dir.resolve()).as_posix()))
        return f"{len(files)} files stored ({ctx.blobs.name})"
    except Exception as e:
        log.warning("checkpoint %s kept locally only: %s", p, e)
        return f"kept locally only, upload failed: {e}"[:300]


async def resume_pollers(ctx: Ctx) -> None:
    """Re-attach to Slurm jobs that were in flight when the backend restarted."""
    if not ctx.settings.futurama_ssh_host:
        return
    for status in (JobStatus.queued, JobStatus.running):
        for tj in await ctx.repo.list("training_jobs", TrainingJob, status=status):
            if tj.backend == "futurama_slurm" and tj.external_id:
                ctx.jobs.submit("train", tj.project_id, lambda job, tj=tj: _tracked(ctx, tj, poll_slurm(ctx, tj, job)),
                                idempotency_key=f"train:{tj.id}")


async def _get(ctx: Ctx, tj_id: str) -> TrainingJob:
    tj = await ctx.repo.get("training_jobs", tj_id, TrainingJob)
    if tj is None:
        raise KeyError(tj_id)
    return tj


async def _save(ctx: Ctx, tj: TrainingJob, msg: str) -> None:
    tj.updated_at = now()
    await ctx.repo.save("training_jobs", tj)
    ctx.changed(tj.project_id, "training_jobs", tj.id)
    log.info("%s: %s", tj.id, msg)
