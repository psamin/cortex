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
import hmac
import json
import logging
import re
import shlex
import shutil
from collections.abc import Awaitable
from pathlib import Path

from ..context import Ctx
from ..errors import NotFound
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
MAX_SSH_FAILURES = 20  # consecutive failed squeue calls (~10 min at POLL_S) before the poller gives up


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


def robot_type_of(ds: DatasetVersion) -> str | None:
    """The robot a dataset was recorded on (LeRobot info.json), e.g. so101_follower or so100."""
    import json

    info = Path(ds.lerobot_root or "") / "meta" / "info.json"
    return json.loads(info.read_text()).get("robot_type") if ds.lerobot_root and info.exists() else None


def train_episodes(ds: DatasetVersion) -> list[int] | None:
    """The LeRobot episode indices lerobot-train may use: the train split only. The dataset also holds the
    held-out eval episodes, and training on them would leak the evaluation set. None when the dataset has no
    split links (an imported dataset), which trains on all of it."""
    if not ds.links:
        return None
    eps = sorted(ln.lerobot_episode_index for ln in ds.links if ln.split == "train")
    if not eps:
        raise ValueError(f"dataset {ds.id} has no train-split episodes to train on")
    return eps


def episodes_arg(eps: list[int]) -> str:
    """``[0,2,5]``: the lerobot-train --dataset.episodes value, without spaces so it survives a shell."""
    return "[" + ",".join(str(i) for i in eps) + "]"


async def propose(ctx: Ctx, project: Project, dataset_version_id: str, policy_type: str = "act",
                  steps: int | None = None, actor: str = "gui", base_checkpoint_id: str | None = None) -> TrainingJob:
    """Propose training a policy. With `base_checkpoint_id` the run continues fine-tuning that policy (from any
    project, so a policy can be carried to a new task or a specific arm) instead of starting from scratch."""
    ds = await ctx.repo.get("dataset_versions", dataset_version_id, DatasetVersion)
    if ds is None:
        raise NotFound(dataset_version_id)
    base = await ctx.repo.get("checkpoints", base_checkpoint_id, Checkpoint) if base_checkpoint_id else None
    if base_checkpoint_id and base is None:
        raise NotFound(f"checkpoint {base_checkpoint_id}")
    if base is not None:
        policy_type = base.policy_type if base.policy_type in ("act", "smolvla") else policy_type
    infra = project.infra
    backend = infra.training_backend if infra.training_backend != "auto" else (
        "futurama_slurm" if ctx.settings.futurama_ssh_host else "manual")
    if backend == "futurama_slurm" and not ctx.settings.futurama_ssh_host:
        raise ValueError("Training on Futurama was chosen, but FUTURAMA_SSH_HOST is not set in .env.")
    steps = steps or (20_000 if policy_type == "act" else 20_000)
    cfg = {
        "policy.type": policy_type, "steps": steps, "batch_size": 8, "save_freq": steps // 2,
        "dataset.repo_id": f"cortex/{project.id}_{ds.id}", "dataset.root": ds.lerobot_root,
        "dataset.episodes": train_episodes(ds),
        "fps": FPS, "seed": 1000, "lerobot": "0.6.1",
        "cameras": ["observation.images.front", "observation.images.wrist"],
        "slurm": {"partition": ctx.settings.futurama_partition, "gres": infra.training_gres,
                  "time_min": infra.training_walltime_min},
    }
    robot = robot_type_of(ds)
    cfg["robot_type"] = robot
    if base is not None:
        base_tj = await ctx.repo.get("training_jobs", base.training_job_id or "", TrainingJob)
        base_robot = (base_tj.config.get("robot_type") if base_tj else None)
        cfg.update({"policy.path": base.path, "base_checkpoint_id": base.id, "base_robot_type": base_robot,
                    "cross_embodiment": bool(robot and base_robot and robot != base_robot)})
    elif policy_type == "smolvla":
        cfg["policy.path"] = "lerobot/smolvla_base"
        # smolvla_base expects observation.images.camera1..3; map this dataset's cameras onto them in order
        cfg["rename_map"] = {c: f"observation.images.camera{i + 1}" for i, c in enumerate(cfg["cameras"][:3])}
    n_train = sum(1 for ln in ds.links if ln.split == "train")
    why = (f"ACT imitation learning on {ds.num_source_episodes} source demos ({n_train} train LeRobot episodes "
           f"incl. augmented views): it trains in minutes, fits 30 fps joint control and needs no language "
           f"conditioning for a single task.") if policy_type == "act" else (
           "SmolVLA fine-tuning: language-conditioned and pretrained, but ~5x slower per step.")
    if base is not None:
        why = (f"Continue fine-tuning {base.policy_type} checkpoint {base.id} (step {base.step}) on "
               f"{ds.num_source_episodes} source demos" + (f" recorded on {robot}" if robot else "") +
               (": a different robot than the base was trained on, so expect a larger adaptation" if
                cfg["cross_embodiment"] else "") + ".")
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
        if not secret or not token or not hmac.compare_digest(token, secret):
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
    if tj.status in RETRYABLE:
        _preflight(tj)  # before the reset: a check that fails must leave the job failed, with its reason
        tj = await _reset_for_retry(ctx, tj, actor)
    if tj.status != JobStatus.approved:
        raise ApprovalRequired(f"{tj.id} is '{tj.status}'; it must be approved before launch")
    _preflight(tj)
    if "dataset.episodes" not in tj.config:  # proposed before the train-only filter existed
        ds = await ctx.repo.get("dataset_versions", tj.dataset_version_id, DatasetVersion)
        tj.config["dataset.episodes"] = train_episodes(ds) if ds is not None else None
        await ctx.repo.save("training_jobs", tj)
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
        ctx.jobs.submit("train", tj.project_id, lambda job: _tracked(ctx, tj, _run_slurm(ctx, tj, job), job),
                        idempotency_key=f"train:{tj.id}")
    await ctx.audit(actor, "launch_training_job", tj.project_id, {"id": tj.id, "backend": tj.backend}, "ok",
                    f"{tj.id} launched on {tj.backend}", approved=True)
    return tj


RETRYABLE = (JobStatus.failed, JobStatus.cancelled)


def retry_needs_human(tj: TrainingJob) -> bool:
    """A job that reached the cluster already spent GPU time, so running it again needs a fresh human click.
    So does a job someone stopped: the stop was a decision, and only a person reverses it. One that failed before
    Slurm accepted it (VPN down, a missing tool, a restart mid-upload) cost nothing and may be retried by anyone."""
    return bool(tj.external_id) or tj.status == JobStatus.cancelled


def _preflight(tj: TrainingJob) -> None:
    """Local checks a cluster launch needs; they raise before anything is changed."""
    if tj.backend == "futurama_slurm":
        from ..subprocs import unavailable_reason
        if why := unavailable_reason():  # ssh/rsync/sbatch would all fail in the background (#347)
            raise ValueError(why)
        preflight_transfer_tools()


async def _reset_for_retry(ctx: Ctx, tj: TrainingJob, actor: str) -> TrainingJob:
    """Put a failed or stopped job back to approved so it can run again, without re-proposing it."""
    if not tj.approved_by:
        raise ApprovalRequired(f"{tj.id} was never approved; approve it in the Cortex studio first")
    if actor != "gui" and retry_needs_human(tj):
        why = (f"already ran on the cluster (Slurm job {tj.external_id}), so running it again spends GPU time again"
               if tj.external_id else "was stopped by a person")
        raise ApprovalRequired(f"{tj.id} {why}: ask the human to click Retry on the job in the Cortex studio.")
    prev = f"{tj.status}: {tj.error}" if tj.error else str(tj.status)
    if tj.external_id and not await scancel(ctx, tj.external_id):
        # "lost contact" failures say the Slurm job may still run: it must be ended first, or the retry would share
        # its job directory (and its output) and spend the GPU twice. If Slurm can't be reached, retry later.
        raise ValueError(f"couldn't cancel {tj.id}'s earlier Slurm job {tj.external_id}, which may still be "
                         "running: retry once the cluster is reachable")
    tj.status, tj.error, tj.external_id = JobStatus.approved, None, None
    tj.metrics = {k: v for k, v in tj.metrics.items() if k != "loss_curve"}
    await _save(ctx, tj, f"retrying after {prev}")
    await ctx.audit(actor, "retry_training_job", tj.project_id, {"id": tj.id, "previous": prev[:300]}, "ok",
                    f"{tj.id} reset for another run", approved=actor == "gui")
    return tj


def manual_output_dir(ctx: Ctx, tj: TrainingJob) -> Path:
    """Where a manual job's command writes its run: an absolute path in the data dir, so the checkpoint can be
    registered from there whatever directory the command was run in."""
    return ctx.local(f"projects/{tj.project_id}/training/{tj.id}/outputs").resolve()


def manual_checkpoint_path(ctx: Ctx, tj: TrainingJob) -> Path:
    return manual_output_dir(ctx, tj) / "checkpoints" / "last" / "pretrained_model"


def manual_command(ctx: Ctx, tj: TrainingJob) -> str:
    c = tj.config
    eps = c.get("dataset.episodes")
    held_out = "# trains on the train split only; the dataset's eval episodes are held out\n" if eps else ""
    return (f"# Cortex training job {tj.id} (approved by {tj.approved_by})\n{held_out}"
            f"pip install 'lerobot[dataset,training]==0.6.1'\n"
            f"lerobot-train --dataset.repo_id={c['dataset.repo_id']} --dataset.root={ctx.local(c['dataset.root'])} "
            + (f"--dataset.episodes={episodes_arg(eps)} " if eps else "") +
            f"--policy.type={c['policy.type']} --policy.device=cuda --policy.push_to_hub=false "
            f"--steps={c['steps']} --batch_size={c['batch_size']} --save_freq={c['save_freq']} --seed={c['seed']} "
            f"--output_dir={manual_output_dir(ctx, tj).as_posix()} --job_name=cortex_{tj.id} --wandb.enable=false\n"
            f"# on another machine: copy that output folder back to the same path here\n"
            f"# then: POST /api/projects/{tj.project_id}/checkpoints "
            f"{{\"training_job_id\": \"{tj.id}\", \"path\": \"{manual_checkpoint_path(ctx, tj).as_posix()}\"}}\n")


async def _sh(*args: str, timeout: float = 600, cwd: Path | None = None,
              input: bytes | None = None) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*args, cwd=str(cwd) if cwd else None,
                                             stdin=asyncio.subprocess.PIPE if input is not None else None,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(p.communicate(input), timeout)
    except TimeoutError:
        p.kill()
        return 124, "timeout"
    return p.returncode or 0, out.decode(errors="replace")


SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def _remote_q(path: str) -> str:
    """Quote a remote path for the remote shell, keeping a leading ~/ expandable."""
    return "~/" + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)


def _safe_job_dir(path: str) -> str:
    """rm -rf only ever runs on a job's own directory under the configured jobs root, never on a short or empty
    path (an unset FUTURAMA_REPO_DIR must not turn into rm -rf ~/)."""
    parts = [p for p in path.replace("~/", "", 1).split("/") if p]
    if len(parts) < 3 or "jobs" not in parts or ".." in parts:
        raise ValueError(f"refusing to delete remote path {path!r}: not a job directory")
    return path


def preflight_transfer_tools() -> None:
    """Refuse a cluster launch up front when this machine can't reach the cluster at all (no ssh, or neither
    rsync nor scp), instead of failing later in the background."""
    if not shutil.which("ssh"):
        raise ValueError("Training on Futurama needs the ssh client on this machine, and none was found on PATH.")
    if not (shutil.which("rsync") or shutil.which("scp")):
        raise ValueError("Training on Futurama needs rsync or scp on this machine to copy the dataset, and neither "
                         "was found on PATH.")


# Local paths are always passed to rsync/scp relative to a working directory. An absolute Windows path
# (C:\Users\...) has a colon before its first slash, so both tools read it as host "C" (#346).

async def _push_dir(host: str, local: Path, remote_dir: str) -> None:
    """Make host:remote_dir an exact copy of the local directory."""
    if shutil.which("rsync"):
        cmd = ["rsync", "-az", "--delete", f"{local.name}/", f"{host}:{remote_dir}/"]
    else:  # e.g. Windows, which ships OpenSSH's scp but no rsync
        rc, out = await _sh("ssh", *SSH_OPTS, host, f"rm -rf {_remote_q(_safe_job_dir(remote_dir))}", timeout=120)
        if rc != 0:
            raise RuntimeError(f"ssh failed ({rc}): {out[-400:]} — is the GT VPN up?")
        cmd = ["scp", *SSH_OPTS, "-rq", local.name, f"{host}:{remote_dir}"]
    rc, out = await _sh(*cmd, cwd=local.parent)
    if rc != 0:
        raise RuntimeError(f"{cmd[0]} of {local.name} failed ({rc}): {out[-400:]} — is the GT VPN up?")


async def _pull_dir(host: str, remote_dir: str, local: Path) -> None:
    """Copy host:remote_dir to the local directory (replacing it)."""
    local.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        cmd = ["rsync", "-az", f"{host}:{remote_dir}/", f"{local.name}/"]
    else:
        shutil.rmtree(local, ignore_errors=True)
        cmd = ["scp", *SSH_OPTS, "-rq", f"{host}:{remote_dir}", local.name]
    rc, out = await _sh(*cmd, cwd=local.parent)
    if rc != 0:
        raise RuntimeError(f"checkpoint fetch failed ({rc}): {out[-300:]}")


async def _ship_text(host: str, remote_path: str, text: str, what: str) -> None:
    """Write a small text file on the cluster through ssh's stdin: no local temp file, no rsync path parsing."""
    rc, out = await _sh("ssh", *SSH_OPTS, host, f"cat > {_remote_q(remote_path)}", timeout=120,
                        input=text.replace("\r\n", "\n").encode())
    if rc != 0:
        raise RuntimeError(f"copying the {what} failed ({rc}): {out[-400:]} — is the GT VPN up?")


async def _tracked(ctx: Ctx, tj: TrainingJob, work: Awaitable[str], job=None) -> str:
    """Run Slurm work so the durable TrainingJob always ends terminal: a failed ssh/rsync/sbatch or a
    stop must not leave it 'queued' (Autopilot waits on the stored status and would wait forever).
    A backend shutdown is not a stop: a submitted Slurm job keeps running and ``resume_pollers`` re-attaches
    to it on the next start. A user stop cancels the Slurm job too."""
    try:
        return await work
    except BaseException as e:
        cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob) or tj
        cur.external_id = cur.external_id or tj.external_id
        if cur.status in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled):
            raise
        cancelled = isinstance(e, asyncio.CancelledError)
        if cancelled and ctx.jobs.closing:
            if cur.external_id:
                raise  # stays queued/running with its Slurm id; the next start polls it again
            cur.status, cur.error = JobStatus.failed, "interrupted by a backend restart before Slurm accepted it"
            await _save(ctx, cur, cur.status)
            raise
        stopped = cancelled or (job is not None and ctx.jobs.stop_requested(job.id))
        if stopped and cur.external_id:
            await scancel(ctx, cur.external_id)
        cur.status = JobStatus.cancelled if stopped else JobStatus.failed
        cur.error = "stopped by user" if stopped else f"{type(e).__name__}: {e}"[:500]
        await _save(ctx, cur, cur.status)
        raise


async def scancel(ctx: Ctx, external_id: str) -> bool:
    """Best effort for a stop: the stop is recorded either way, and a failure is logged for the operator.
    Returns whether Slurm confirmed it (a retry needs that, see _reset_for_retry)."""
    host = ctx.settings.futurama_ssh_host
    if not host:
        return False
    rc, out = await _sh("ssh", *SSH_OPTS, host, f"scancel {external_id}", timeout=60)
    if rc != 0 and "Invalid job id" in out:  # Slurm forgot it: it ended long ago, which is as good as cancelled
        return True
    if rc != 0:
        log.warning("scancel %s failed (%s): %s", external_id, rc, out[-300:])
    return rc == 0


async def _run_slurm(ctx: Ctx, tj: TrainingJob, job) -> str:
    host = ctx.settings.futurama_ssh_host
    assert host
    remote = f"{ctx.settings.futurama_repo_dir}/jobs/{tj.id}"
    ds_local = ctx.local(tj.config["dataset.root"])
    ssh = ["ssh", *SSH_OPTS, host]
    ctx.jobs.log(job, f"copy dataset -> {host}:{remote}")
    rc, out = await _sh(*ssh, f"mkdir -p {_remote_q(remote)}")
    if rc != 0:
        raise RuntimeError(f"ssh failed ({rc}): {out[-400:]} — is the GT VPN up?")
    await _push_dir(host, ds_local, f"{remote}/dataset")
    # the job script as text: a Windows checkout with CRLF line endings would otherwise break bash on the cluster
    await _ship_text(host, f"{remote}/train_act.sbatch", SBATCH.read_text(), "training job script")
    c = tj.config
    export = (f"ALL,STEPS={c['steps']},BATCH={c['batch_size']},SAVE={c['save_freq']},"
              f"REPO_ID={c['dataset.repo_id']}")
    base = c.get("policy.path")
    if base and Path(base).is_dir():  # a project checkpoint: ship it next to the dataset
        await _push_dir(host, Path(base), f"{remote}/base")
        export += ",POLICY_PATH=base"
    elif base:  # a hub id such as lerobot/smolvla_base
        export += f",POLICY_PATH={base}"
    if c.get("rename_map"):  # JSON has commas, which sbatch --export would split: ship it as a file
        await _ship_text(host, f"{remote}/rename_map.json", json.dumps(c["rename_map"]), "rename map")
    if c.get("dataset.episodes"):  # a list has commas too: ship it as a file the sbatch script reads
        await _ship_text(host, f"{remote}/episodes.txt", episodes_arg(c["dataset.episodes"]), "train episode list")
    slurm = c.get("slurm") or {}  # chosen in the studio; older jobs fall back to the .env defaults
    rc, out = await _sh(*ssh, f"cd {_remote_q(remote)} && sbatch --parsable -p {slurm.get('partition', ctx.settings.futurama_partition)} "
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
    ssh = ["ssh", *SSH_OPTS, host]
    seen, misses = 0, 0
    while True:
        rc, state = await _sh(*ssh, f"squeue -h -j {tj.external_id} -o %T", timeout=60)
        state = state.strip()
        if "Invalid job id" in state:
            break  # Slurm already purged the finished job: it left the queue
        misses = misses + 1 if rc != 0 else 0
        if misses >= MAX_SSH_FAILURES:
            tj.status, tj.error = JobStatus.failed, (f"lost contact with {host}: squeue failed {misses} times in a "
                                                     f"row ({state[-200:]}); slurm job {tj.external_id} may still run")
            await _save(ctx, tj, "failed")
            raise RuntimeError(tj.error)
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
    await _pull_dir(host, f"{remote}/output/checkpoints/last/pretrained_model", dest)
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


async def register_manual_checkpoint(ctx: Ctx, project_id: str, training_job_id: str, path: str,
                                     step: int | None = None, actor: str = "mcp") -> Checkpoint:
    """The agent-callable way to finish a manual job: only for a launched (so human-approved) manual job of this
    project, and only from that job's own output folder. Anything else goes through the studio."""
    tj = await _get(ctx, training_job_id)
    if tj.project_id != project_id:
        raise ValueError(f"{training_job_id} belongs to another project")
    if tj.backend != "manual" or tj.status != JobStatus.queued:
        raise ValueError(f"{training_job_id} is a {tj.backend} job that is '{tj.status}': only a launched manual "
                         "job takes a checkpoint this way")
    own = manual_output_dir(ctx, tj)  # this job's own run folder: not another project's weights, nothing outside
    p = Path(path) if Path(path).is_absolute() else ctx.local(path)
    p = p.resolve()
    if not p.is_relative_to(own):
        raise ValueError(f"the checkpoint must be inside this job's output folder ({own.as_posix()}); expected "
                         f"{manual_checkpoint_path(ctx, tj).as_posix()}")
    return await register_checkpoint(ctx, project_id, training_job_id, str(p), step, actor)


async def register_remote_checkpoint(ctx: Ctx, project_id: str, training_job_id: str | None, remote: str,
                                     config: dict, step: int | None = None, actor: str = "gui") -> Checkpoint:
    """A checkpoint whose weights stay where they were trained, e.g. futurama:~/cortex-jenga/output/jenga/jenga_010000.

    Only its config was read (the caller fetched config.json). Evaluating it here or running it on an arm
    needs a local copy, registered with register_checkpoint."""
    from .futurama_files import is_remote

    if not is_remote(remote):
        raise ValueError(f"{remote} is not a host:path location")
    ck = Checkpoint(project_id=project_id, training_job_id=training_job_id, policy_type=config.get("type", "unknown"),
                    step=step or 0, path=remote)
    await ctx.repo.save("checkpoints", ck, status="ready")
    if training_job_id:
        tj = await ctx.repo.get("training_jobs", training_job_id, TrainingJob)
        if tj:
            tj.checkpoint_ids.append(ck.id)
            tj.status = JobStatus.succeeded
            await _save(ctx, tj, f"checkpoint {ck.id} registered (weights on the cluster)")
    ctx.changed(project_id, "checkpoints", ck.id)
    await ctx.audit(actor, "register_checkpoint", project_id, {"path": remote}, "ok",
                    f"{ck.id} ({ck.policy_type}); weights kept at {remote}, not copied")
    return ck


def checkpoint_dir(ctx: Ctx, ck: Checkpoint) -> Path:
    """The checkpoint's directory on this machine. Raises when its weights live only on the cluster."""
    from .futurama_files import is_remote

    if is_remote(ck.path):
        raise FileNotFoundError(f"{ck.id}'s weights are on the cluster ({ck.path}), not on this machine. Copy the "
                                f"directory here with rsync and register the copy to use it locally.")
    return Path(ck.path) if Path(ck.path).is_absolute() else ctx.local(ck.path)


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
                ctx.jobs.submit("train", tj.project_id, lambda job, tj=tj: _tracked(ctx, tj, poll_slurm(ctx, tj, job), job),
                                idempotency_key=f"train:{tj.id}")


async def _get(ctx: Ctx, tj_id: str) -> TrainingJob:
    tj = await ctx.repo.get("training_jobs", tj_id, TrainingJob)
    if tj is None:
        raise NotFound(tj_id)
    return tj


# (kind, patterns in the stored error), first match wins. The studio maps each kind to a plain sentence.
FAILURE_KINDS: list[tuple[str, tuple[str, ...]]] = [
    # Cortex's own prefixes first: they name where it failed, while the log tail after them can contain anything
    # (a crash log saying "Read timed out" is still a crash, not an unreachable cluster).
    ("interrupted", ("interrupted by a backend restart",)),
    ("training_crashed", ("ended without completing",)),
    ("checkpoint_missing", ("checkpoint fetch failed", "is not a LeRobot pretrained_model")),
    ("cluster_rejected", ("sbatch failed", "sbatch: error")),
    ("cluster_unreachable", ("lost contact with",)),
    ("subprocesses_unavailable", ("NotImplementedError", "can't start child processes", "cannot start child processes")),
    ("tool_missing", ("No such file or directory: 'rsync'", "No such file or directory: 'ssh'",
                      "No such file or directory: 'scp'", "rsync: command not found", "is not installed")),
    ("cluster_unreachable", ("(255)", "Could not resolve hostname", "Connection timed out", "Connection refused",
                             "No route to host", "Permission denied (publickey", "timed out")),
    ("upload_failed", ("rsync", "scp", "failed (")),
]


def failure_kind(tj: TrainingJob) -> str | None:
    """A one-word cause for a failed job; None for anything that didn't fail (a user stop is 'cancelled')."""
    if tj.status != JobStatus.failed:
        return None
    err = tj.error or ""
    for kind, pats in FAILURE_KINDS:
        if any(p in err for p in pats):
            return kind
    return "other"


async def _save(ctx: Ctx, tj: TrainingJob, msg: str) -> None:
    tj.failure_kind = failure_kind(tj)
    tj.updated_at = now()
    await ctx.repo.save("training_jobs", tj)
    ctx.changed(tj.project_id, "training_jobs", tj.id)
    log.info("%s: %s", tj.id, msg)
