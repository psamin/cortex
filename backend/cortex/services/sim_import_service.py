"""Import a hand-built sim task's Futurama runs (infra/futurama/sim_task.sbatch) into a project (#179).

sim_task.sbatch writes, under <job_dir>/<task>/:
    dataset/                       LeRobotDataset v3 of scripted-oracle demos; meta/cortex_sim.json has the
                                   kept and failed seeds and the oracle's success rate
    examples/demo_seed<N>.mp4      the first kept demos, rendered for people
    runs/<variant>/checkpoints/<step>/pretrained_model/
    runs/<variant>/eval/<step>/metrics.json + seed<N>_ok|fail.mp4   closed-loop evaluation on unseen seeds
and Slurm logs slurm-*.out in <job_dir>.

This works on a local mirror of those light files (scripts/import_sim_task.py makes it) and records:
the project (reused by name), a dataset version, the oracle's demo run as a scripted-expert evaluation,
one training job per variant (loss curve from its log), one checkpoint per saved step (weights stay at
their Futurama path unless copied) and one evaluation per evaluated step, with videos uploaded. Every
number comes from those files; success is the simulator's verdict. Re-running updates, never duplicates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..context import Ctx
from ..models import (
    BranchSpec, Checkpoint, Condition, CreateProjectRequest, DatasetVersion, EpisodeResult, EvaluationRun, JobStatus,
    Mode, PolicyRef, Project, ResourceEstimate, SimulationBranch, TrainingJob,
)
from . import project_cleanup, project_service, replay_service, training_service
from .simulation_runner import summarize

NAMES = {"jenga": "Jenga (sim)", "screwdriver": "Screwdriver (sim)", "palletize": "Palletizing (sim)",
         "suture": "Suture (sim)", "unscrew": "Unscrew bolt (sim)"}  # two tasks must never share a project name
PROMPTS = {"jenga": "Push a block out of a Jenga tower without toppling it (MuJoCo sim)",
           "screwdriver": "Pick up a screwdriver and unscrew a screw with it (MuJoCo sim)",
           "palletize": "Move boxes off a forklift onto a pallet (MuJoCo sim)",
           "unscrew": "Unscrew a hex bolt from a fixture with the SO-101 (MuJoCo sim)",
           "suture": "Pass a suture needle through a tissue pad (MuJoCo sim, rigid proxy)"}
OBJECTS = {"jenga": "block", "screwdriver": "screwdriver", "palletize": "box", "unscrew": "bolt",
           "suture": "needle"}  # the object the project's header names (it appears in the task prompt)
EXECUTOR = "futurama_slurm"
ORACLE_RUN = "oracle demos"


def project_name(task: str) -> str:
    return NAMES.get(task, f"{task.replace('_', ' ').capitalize()} task (sim)")


async def find_or_create_project(ctx: Ctx, task: str, instruction: str | None, name: str | None = None) -> Project:
    """The project with this name (default: named for the task; a deleted one is brought back), else a new one.
    Its spec describes the task itself: the studio's header shows spec.goal, and the default spec is the cube
    tabletop ("pick up the cube and place it in the bowl ... sim has cube and bowl only")."""
    name = name or project_name(task)
    rows = [r for r in await ctx.store.list("projects") if r["data"].get("name") == name]
    rows.sort(key=lambda r: (r.get("status") == project_cleanup.DELETED, str(r.get("created_at"))))
    if rows:
        await project_cleanup.restore(ctx, rows[0]["id"])
        p = Project.model_validate(rows[0]["data"])
    else:
        p = project_service.create_project(CreateProjectRequest(
            task_prompt=PROMPTS.get(task) or instruction or f"{task} (MuJoCo sim)", name=name, mode=Mode.sim))
    p.spec.goal = (instruction or PROMPTS.get(task) or task).strip().rstrip(".")
    p.spec.objects = [OBJECTS.get(task, task)]
    await ctx.repo.save("projects", p, status="active")
    return p


def _job_id(log: Path) -> str | None:
    m = re.search(r"-(\d+)\.out$", log.name)
    return m.group(1) if m else None


def find_log(logs: list[Path], task: str, variant: str | None) -> Path | None:
    """The newest Slurm log of this task's demos (variant None) or of this training variant."""
    hits = []
    for f in logs:
        text = f.read_text(errors="replace")
        if variant is None:
            ok = f"mode=demos task={task}" in text
        else:
            ok = f"mode=train task={task}" in text and re.search(rf"sim_{re.escape(task)}_{re.escape(variant)}\b", text)
        if ok:
            hits.append(f)
    return max(hits, key=lambda f: int(_job_id(f) or 0)) if hits else None


async def _upload(ctx: Ctx, pid: str, rel: str, src: Path) -> str:
    return await replay_service.store_file(ctx, f"projects/{pid}/sim/{rel}", src)


async def _save_eval(ctx: Ctx, pid: str, name: str, policy: PolicyRef, label: str, ds_id: str | None,
                     branch: str, episodes: list[EpisodeResult], policy_name: str, wall_s: float) -> EvaluationRun:
    prior = [r for r in await ctx.repo.list("evaluation_runs", EvaluationRun, project_id=pid)
             if r.executor == EXECUTOR and r.name == name]
    run = prior[-1] if prior else EvaluationRun(project_id=pid, name=name, policy=policy)
    run.policy, run.policy_label, run.executor, run.dataset_version_id = policy, label, EXECUTOR, ds_id
    run.status = JobStatus.succeeded
    b = SimulationBranch(evaluation_id=run.id, spec=BranchSpec(name=branch, condition=Condition()),
                         seeds=[e.seed for e in episodes], executor=EXECUTOR, project_id=pid,
                         status=JobStatus.succeeded, result=summarize(episodes, policy_name, wall_s))
    if run.branch_ids:
        b.id = run.branch_ids[0]
    run.branch_ids = [b.id]
    await ctx.repo.save("simulation_branches", b)
    await ctx.repo.save("evaluation_runs", run)
    ctx.changed(pid, "evaluation_runs", run.id)
    return run


async def _dataset(ctx: Ctx, pid: str, task: str, src: str, meta: dict, info: dict,
                   dataset_root: Path | None) -> DatasetVersion:
    where = f"{src}/dataset"
    prior = [d for d in await ctx.repo.list("dataset_versions", DatasetVersion, project_id=pid)
             if d.imported_from == where]
    ds = prior[-1] if prior else DatasetVersion(project_id=pid, name="", links=[], imported_from=where)
    kept = meta.get("kept_seeds", [])
    ds.name = f"{task} sim: {len(kept)} scripted-oracle demos (Futurama)"
    ds.eval_fraction, ds.num_frames = 0.0, int(info.get("total_frames", 0))
    if dataset_root is not None and dataset_root.resolve().is_relative_to(ctx.data_dir.resolve()):
        ds.lerobot_root = str(dataset_root.resolve().relative_to(ctx.data_dir.resolve()))
    await ctx.repo.save("dataset_versions", ds, status="ready")
    ctx.changed(pid, "dataset_versions", ds.id)
    return ds


async def _oracle(ctx: Ctx, pid: str, task: str, mirror: Path, meta: dict, logs: list[Path],
                  ds_id: str) -> EvaluationRun:
    """The demo run is a measured oracle run: every seed tried, the simulator judged each one."""
    steps: dict[int, int] = {}
    log = find_log(logs, task, None)
    for ln in (log.read_text(errors="replace").splitlines() if log else []):
        if ln.startswith('{"seed"'):
            try:
                r = json.loads(ln)
                steps[int(r["seed"])] = int(r["steps"])
            except (ValueError, KeyError):
                pass
    videos = {int(m.group(1)): f for f in (mirror / "examples").glob("demo_seed*.mp4")
              if (m := re.fullmatch(r"demo_seed(\d+)\.mp4", f.name))}
    eps = []
    kept = set(meta.get("kept_seeds", []))
    for seed in sorted(kept | set(meta.get("failed_seeds", []))):
        key = await _upload(ctx, pid, f"{task}/examples/{videos[seed].name}", videos[seed]) if seed in videos else None
        ok = seed in kept
        eps.append(EpisodeResult(seed=seed, success=ok, reward=float(ok), steps=steps.get(seed, 0), video_key=key))
    return await _save_eval(ctx, pid, ORACLE_RUN, PolicyRef(kind="scripted_expert"),
                            "scripted oracle (privileged state), demo recording", ds_id, "demo seeds", eps,
                            "scripted oracle", 0.0)


def _step_dirs(root: Path) -> list[Path]:
    return sorted((d for d in root.glob("[0-9]*") if d.is_dir()), key=lambda d: int(d.name)) if root.exists() else []


async def _variant(ctx: Ctx, pid: str, task: str, src: str, run_dir: Path, logs: list[Path], ds: DatasetVersion,
                   actor: str) -> dict[str, Any]:
    variant = run_dir.name
    where = f"{src}/runs/{variant}"
    cks = [d for d in _step_dirs(run_dir / "checkpoints") if (d / "pretrained_model" / "config.json").exists()]
    log = find_log(logs, task, variant)
    text = log.read_text(errors="replace") if log else ""
    last = cks[-1] / "pretrained_model" if cks else None
    cfg = json.loads((last / "config.json").read_text()) if last else {}
    tcfg = json.loads((last / "train_config.json").read_text()) if last and (last / "train_config.json").exists() else {}
    ptype = cfg.get("type") or {"act_ensemble": "act"}.get(variant, variant)
    if ptype not in ("act", "smolvla", "diffusion"):
        return {"variant": variant, "skipped": f"policy type {ptype} is not tracked"}
    prior = [t for t in await ctx.repo.list("training_jobs", TrainingJob, project_id=pid)
             if t.config.get("source") == where]
    tj = prior[-1] if prior else TrainingJob(
        project_id=pid, dataset_version_id=ds.id, policy_type=ptype, backend="futurama_slurm", config={},
        estimate=ResourceEstimate(gpu="1x GPU (Futurama general partition)", gpu_hours=0.0, est_cost_usd=0.0,
                                  paid=False, notes="ran on Futurama with infra/futurama/sim_task.sbatch"),
        rationale=f"{variant} trained on the {task} sim task's scripted-oracle demos", approved_by="owner (sbatch)")
    tj.policy_type, tj.dataset_version_id, tj.external_id = ptype, ds.id, _job_id(log) if log else tj.external_id
    tj.config = {**tj.config, "variant": variant, "source": where, "dataset": ds.imported_from, "lerobot": "0.6.1",
                 **({"steps": tcfg["steps"]} if "steps" in tcfg else {}),
                 **({"batch_size": tcfg["batch_size"]} if "batch_size" in tcfg else {})}
    tj.metrics = {**tj.metrics, "loss_curve": training_service.parse_loss(text) or tj.metrics.get("loss_curve", [])}
    tj.status = (JobStatus.succeeded if cks else JobStatus.failed if "Traceback" in text[-4000:]
                 else JobStatus.running)
    await ctx.repo.save("training_jobs", tj)
    ctx.changed(pid, "training_jobs", tj.id)

    have = {c.path: c for c in await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)}
    by_step: dict[str, Checkpoint] = {}
    for d in cks:
        pm, remote = d / "pretrained_model", f"{where}/checkpoints/{d.name}/pretrained_model"
        local = str(pm.resolve()) if (pm / "model.safetensors").exists() else None
        ck = have.get(local or remote)
        if ck is None and local:
            ck = await training_service.register_checkpoint(ctx, pid, tj.id, local, int(d.name), actor)
        elif ck is None:
            ck = await training_service.register_remote_checkpoint(
                ctx, pid, tj.id, remote, json.loads((pm / "config.json").read_text()), int(d.name), actor)
        by_step[d.name] = ck

    evals = []
    for d in _step_dirs(run_dir / "eval"):
        if not (d / "metrics.json").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        eps = []
        for r in m.get("results", []):
            vid = d / r["video"] if r.get("video") else None
            key = await _upload(ctx, pid, f"{task}/{variant}/eval/{d.name}/{vid.name}", vid) if vid and vid.exists() \
                else None
            eps.append(EpisodeResult(seed=int(r["seed"]), success=bool(r["success"]), reward=float(bool(r["success"])),
                                     steps=int(r.get("steps", 0)), video_key=key))
        ck = by_step.get(d.name)
        run = await _save_eval(
            ctx, pid, f"{variant} · step {int(d.name)}",
            PolicyRef(kind="checkpoint", checkpoint_id=ck.id) if ck else PolicyRef(kind="checkpoint"),
            f"{variant} step {int(d.name)}", ds.id, f"unseen seeds {m.get('seeds', '')}".strip(), eps,
            str(m.get("policy", variant)), sum(float(r.get("wall_s", 0)) for r in m.get("results", [])))
        evals.append({"id": run.id, "step": int(d.name), "success_rate": m.get("success_rate"), "n": len(eps)})
    return {"variant": variant, "training_job": tj.id, "status": str(tj.status), "slurm_job": tj.external_id,
            "loss_points": len(tj.metrics["loss_curve"]), "checkpoints": {s: c.id for s, c in by_step.items()},
            "evaluations": evals}


async def import_task(ctx: Ctx, task: str, mirror: Path, src: str, logs: list[Path],
                      dataset_root: Path | None = None, project_id: str | None = None,
                      actor: str = "import_sim_task", name: str | None = None) -> dict[str, Any]:
    """Record a sim task's Futurama results from `mirror` (a local copy of <job_dir>/<task>'s light files).

    `src` is where they live (e.g. futurama:~/cortex-sim/jenga): it is recorded as each item's source and as
    the location of checkpoints whose weights were not copied. `name` overrides the project name."""
    meta_f, info_f = mirror / "dataset" / "meta" / "cortex_sim.json", mirror / "dataset" / "meta" / "info.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else None
    if project_id:
        p = await ctx.repo.get("projects", project_id, Project)
        if p is None:
            raise ValueError(f"no project {project_id}")
    elif meta is None:
        return {"task": task, "skipped": "the demos are not finished (no dataset/meta/cortex_sim.json yet)"}
    else:
        p = await find_or_create_project(ctx, task, meta.get("instruction"), name)
    out: dict[str, Any] = {"task": task, "project_id": p.id, "project": p.name}
    if meta is None:
        return {**out, "skipped": "the demos are not finished (no dataset/meta/cortex_sim.json yet)"}
    info = json.loads(info_f.read_text()) if info_f.exists() else {}
    ds = await _dataset(ctx, p.id, task, src, meta, info, dataset_root)
    oracle = await _oracle(ctx, p.id, task, mirror, meta, logs, ds.id)
    out.update(dataset_version=ds.id, dataset_local=ds.lerobot_root, oracle_evaluation=oracle.id,
               oracle_success_rate=meta.get("oracle_success_rate"), demos=len(meta.get("kept_seeds", [])))
    out["variants"] = [await _variant(ctx, p.id, task, src, d, logs, ds, actor)
                       for d in sorted((mirror / "runs").glob("*")) if d.is_dir()]
    await ctx.audit(actor, "import_sim_task", p.id, {"task": task, "source": src}, "ok",
                    f"{out['demos']} demos, {len(out['variants'])} training variant(s) from {src}")
    return out
