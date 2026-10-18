"""Physical Skills (#276, #285): every trained policy across all projects, in one list.

A skill is one trained policy: a finished training job's checkpoint (including Futurama runs whose weights stay
on the cluster, and imported sim-task runs), or a sim-trained policy the live GPU server serves that was never
imported here. Each carries its measured score, labelled for what was measured: success on unseen sim layouts
(the simulator's verdict), success on the few practice layouts it trained on (never called unseen), "score
pending" while its evaluation runs, or mean joint error against held-out real human demos, which measures
imitation and is never called a success rate. It also says how to run it live on the GPU server and lists its
past live runs.

The studio's Physical Skills page (GET /api/skills) and the MCP tools read this one list. It holds only policies
trained in our physics sim (user rule, #298: the GPU server serves real-demo policies only with SERVE_REAL=1);
list_skills(ctx, real=True) adds the real-demo ones.

Skills stay listed while no GPU server is up (#308): the sim policies a live server served are kept in
<data dir>/skills_seen.json, and each sim skill carries its recorded replays, the mp4s in
<data dir>/skill_replays/<task>/ (newest first). Known sim skills (SEEDED) are listed before any server has been
seen, but only when they have a replay: never a skill with no trained policy behind it.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..context import Ctx
from ..errors import NotFound
from ..models import (
    REPLAY_NOTE, Checkpoint, EvaluationRun, JobStatus, Mode, PolicyReplay, Project, SimulationBranch, TrainingJob,
)
from . import live_policy_service, project_cleanup, sim_import_service
from .futurama_files import is_remote
from .simulation_runner import wilson

log = logging.getLogger(__name__)

# What each known task does, in plain words. Real-data tasks are the Futurama SmolVLA fine-tunes.
REAL_NAMES = {"jenga": "Pull a Jenga block", "conveyor": "Sort parts on a conveyor",
              "suture": "Pass a needle through eyelets"}
SIM_NAMES = {"jenga": "Push a block out of a Jenga tower", "unscrew": "Unscrew a bolt with the gripper",
             "screwdriver": "Unscrew a screw with a screwdriver", "palletize": "Move boxes onto a pallet",
             "suture": "Pass a needle through a tissue pad", "conveyor": "Pick a moving part off a conveyor",
             "unscrew_black": "Unscrewing policy 2", "pick_place": "Pick and place"}
POLICY = {"act": "ACT", "smolvla": "SmolVLA", "diffusion": "Diffusion"}
GPUS = ("GH200", "H200", "H100", "A100", "L40S", "L40", "L4", "V100", "P40")
UNSEEN_FROM = 100_000  # cortex/sim/tasks/run.py EVAL_SEED0: layouts from here up were never used in training
SIM_DETAIL = "Closed-loop runs in MuJoCo on layouts it never trained on; the simulator judged each one."
PRACTICE_DETAIL = ("Closed-loop runs in MuJoCo on the few practice layouts it trained on (picked for reliable demos), "
                   "so this is not a test on new layouts; the simulator judged each one.")
PENDING_DETAIL = "Its evaluation is still running, so no score is claimed yet."
NONE_DETAIL = "No evaluation or held-out replay has been run for this policy yet."
DOWN = "The GPU server isn't running. Request a GPU to run it live."
SEEN = "skills_seen.json"  # under the data dir: the sim skills a live GPU server last served (#308)
REPLAYS = "skill_replays"  # under the data dir: <task>/*.mp4, recorded replays of each sim skill (#308)
SEEDED = {"unscrew": "act", "unscrew_black": "", "jenga": "", "pick_place": ""}  # known sim skills -> policy type
TASK = re.compile(r"^[A-Za-z0-9_-]+$")


def replay_urls(ctx: Ctx, task: str | None) -> list[str]:
    """URLs of the task's recorded replays, newest first."""
    d = ctx.data_dir / REPLAYS / (task or "")
    if not task or not TASK.match(task) or not d.is_dir():
        return []
    files = sorted((f for f in d.glob("*.mp4") if f.is_file() and not f.name.startswith(".")),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return [f"/api/skills/{task}/replays/{quote(f.name)}" for f in files]


def replay_file(ctx: Ctx, task: str, name: str) -> Path:
    """The replay file `name` of `task`, never anything outside <data dir>/skill_replays/<task>/."""
    root = (ctx.data_dir / REPLAYS).resolve()
    if TASK.match(task) and name.lower().endswith(".mp4") and not name.startswith(".") \
            and not any(c in name for c in "/\\\0"):
        p = (root / task / name).resolve()
        if p.parent == root / task and p.is_file():
            return p
    raise NotFound(f"No replay {name!r} for {task!r}.")


def _read_seen(ctx: Ctx) -> dict[str, dict[str, Any]]:
    try:
        d = json.loads((ctx.data_dir / SEEN).read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in d.items() if isinstance(v, dict)} if isinstance(d, dict) else {}


def _seeded(task: str) -> dict[str, Any]:
    v = SEEDED[task]
    return {"id": f"sk_served_sim_{task}", "name": _name(task, True, None), "task": task, "data": "sim",
            "instruction": "", "robot": "SO-101", "project_id": None, "project_name": None, "policy_type": v,
            "policy": _policy(v), "variant": v, "steps": None, "trained_where": None, "trained_at": None,
            "weights": "on the cluster", "checkpoint_id": None, "checkpoint": v, "training_job_id": None,
            "dataset_version_id": None, "slurm_job": None, "layouts": None,
            "score": {"kind": "none", "value": None, "n": None, "label": "Not measured yet", "detail": NONE_DETAIL},
            "runs": [], "evidence": [], "thumbnail_url": None}


def _policy(ptype: str) -> str:
    return POLICY.get(ptype, ptype.upper() if len(ptype) <= 4 else ptype.capitalize())


def _mapping(ctx: Ctx) -> dict[str, str]:
    """infra/futurama/projects.json: task (or sim_<task>) -> project id."""
    try:
        m = json.loads(ctx.settings.futurama_projects_file.read_text())
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in m.items() if not k.startswith("_") and isinstance(v, str)}


def _task_of(p: Project, keys: dict[str, str]) -> tuple[str | None, bool | None]:
    """(task key, is sim) of a project: from projects.json and the live server's map, or the sim importer's
    project names ("Unscrew bolt (sim)")."""
    key = next((k for k, pid in keys.items() if pid == p.id), None)
    if key:
        return (key.removeprefix("sim_"), True) if key.startswith("sim_") else (key, False)
    sim = next((t for t, n in sim_import_service.NAMES.items() if n == p.name), None)
    return (sim, True) if sim else (None, None)


def _prompt(p: Project) -> str:
    """The task as the person (or the sim importer) typed it, without the importer's "(MuJoCo sim)" tag. Not
    spec.goal: a project made from a prompt alone keeps the default cube goal there."""
    return re.sub(r"\s*\((MuJoCo )?sim[^)]*\)", "", p.task_prompt).strip()


def _name(task: str | None, sim: bool, p: Project | None) -> str:
    names = SIM_NAMES if sim else REAL_NAMES
    if task in names:
        return names[task]
    text = _prompt(p) if p else (task or "").replace("_", " ")
    text = re.split(r"(?<=[.!?])\s", text)[0].rstrip(".")
    return (text[:1].upper() + text[1:])[:80] or "Trained policy"


def _robot(tj: TrainingJob | None) -> str:
    m = re.search(r"so[-_]?(\d{3})", str((tj.config.get("robot_type") if tj else None) or ""), re.I)
    return f"SO-{m.group(1)}" if m else "SO-101"


def _where(tj: TrainingJob | None) -> str | None:
    if tj is None:
        return None
    gpu = next((g for g in GPUS if re.search(rf"\b{g}\b", tj.estimate.gpu)), None)
    if tj.backend == "futurama_slurm":
        return f"Futurama {gpu}" if gpu else "Futurama"
    if tj.backend == "hf_jobs":
        return f"Hugging Face Jobs {gpu}" if gpu else "Hugging Face Jobs"
    return "Your own GPU"


def _ck_name(ck: Checkpoint) -> str:
    """The checkpoint's directory name as the GPU server names it (jenga_010000), or its step directory."""
    parts = Path(ck.path.split(":", 1)[-1]).parts
    return parts[-2] if parts and parts[-1] == "pretrained_model" and len(parts) > 1 else (parts[-1] if parts else "")


def _variant(tj: TrainingJob | None, ck: Checkpoint) -> str:
    return str((tj.config.get("variant") if tj else None) or ck.policy_type)


def _eval_score(run: EvaluationRun, branches: dict[str, SimulationBranch]) -> dict[str, Any] | None:
    bs = [branches[b] for b in run.branch_ids if b in branches and branches[b].result and branches[b].result.n]
    if not bs:
        return None
    n = sum(b.result.n for b in bs)
    k = sum(b.result.successes for b in bs)
    seeds = {e.seed for b in bs for e in b.result.episodes}
    imported = all(b.executor == "futurama_slurm" for b in bs)  # sim_task.sbatch evaluations
    layouts = len(seeds) if imported and any(x < UNSEEN_FROM for x in seeds) else None  # practice layouts
    where = (f"on {n} runs over {layouts} practice layouts it trained on" if layouts else
             f"on {n} unseen sim layouts" if imported else f"in {n} sim test episodes")
    lo, hi = wilson(k, n)
    return {"kind": "sim_success", "value": round(k / n, 4), "n": n, "successes": k, "ci95": [lo, hi],
            "layouts": layouts, "label": f"{round(100 * k / n)}% success {where}",
            "detail": f"{PRACTICE_DETAIL if layouts else SIM_DETAIL} {k} of {n} succeeded "
                      f"(95% interval {round(100 * lo)}–{round(100 * hi)}%).",
            "evaluation_id": run.id}


def _server_score(s: dict[str, Any]) -> dict[str, Any]:
    if s.get("success_rate") is None:
        return {"kind": "pending", "value": None, "n": None, "label": "Score pending", "detail": PENDING_DETAIL}
    n, rate, layouts = int(s.get("eval_episodes") or 0), float(s["success_rate"]), s.get("layouts") or None
    where = f"on {n} runs over {layouts} practice layouts it trained on" if layouts else f"on {n} unseen sim layouts"
    return {"kind": "sim_success", "value": rate, "n": n, "successes": round(rate * n), "ci95": None,
            "layouts": layouts, "label": f"{round(100 * rate)}% success {where}",
            "detail": f"{PRACTICE_DETAIL if layouts else SIM_DETAIL} Reported by the GPU server "
                      f"(seeds {s.get('eval_seeds') or 'not given'})."}


def _joint_score(rp: PolicyReplay) -> dict[str, Any]:
    n = len(rp.episodes)
    return {"kind": "joint_error", "value": rp.joint_mae_deg, "n": n,
            "label": f"{rp.joint_mae_deg:.2f}° mean joint error on {n} held-out real demo{'s' if n != 1 else ''}",
            "detail": "Run open loop on recorded human demos it never trained on: how far its joint targets are "
                      f"from what the person did, in degrees ({REPLAY_NOTE}). The arm did not move."}


async def _runs(ctx: Ctx, rps: list[PolicyReplay]) -> list[dict[str, Any]]:
    """Past live runs, newest first, each with its video."""
    out = []
    for rp in sorted((r for r in rps if r.kind in ("live_inference", "live_sim")), key=lambda r: r.created_at,
                     reverse=True):
        sim, partial = rp.kind == "live_sim", rp.status == "partial"
        for e in rp.episodes:
            note = ("Cut off before the end, so not judged." if sim and partial else
                    "A single closed-loop run in MuJoCo, judged by the simulator; not part of the score."
                    if sim else "Open loop on a held-out real demo: joint error vs the human, not a success rate.")
            out.append({"id": f"{rp.id}/{e.episode}", "kind": rp.kind,
                        "label": "Live sim run" if sim else f"Held-out demo {e.episode}",
                        "when": rp.created_at.isoformat(), "partial": partial,
                        "success": e.success if sim and not partial else None, "joint_mae_deg": e.joint_mae_deg,
                        "steps": e.steps, "inference_ms": e.inference_ms.get("mean"), "gpu": rp.gpu,
                        "video_url": await ctx.media_url(e.video_key) if e.video_key else None, "note": note})
    return out


async def _evidence(ctx: Ctx, score: dict[str, Any], branches: dict[str, SimulationBranch],
                    runs: dict[str, EvaluationRun], held: PolicyReplay | None) -> list[dict[str, Any]]:
    """Videos behind the score: the evaluation's episodes, or the held-out replay's."""
    out: list[dict[str, Any]] = []
    if (run := runs.get(score.get("evaluation_id") or "")) is not None:
        for bid in run.branch_ids:
            for e in (branches[bid].result.episodes if bid in branches and branches[bid].result else []):
                where = "Unseen layout" if e.seed >= UNSEEN_FROM else "Practice layout"
                out.append({"label": f"{where} {e.seed}", "success": e.success, "steps": e.steps,
                            "video_url": await ctx.media_url(e.video_key) if e.video_key else None})
    elif score["kind"] == "joint_error" and held is not None:
        for e in held.episodes:
            out.append({"label": f"Held-out demo {e.episode}", "joint_mae_deg": e.joint_mae_deg,
                        "video_url": await ctx.media_url(e.video_key) if e.video_key else None})
    return out


def _run_real(task: str | None, ck: Checkpoint, held: PolicyReplay | None, live: dict | None) -> dict[str, Any]:
    if not task:
        return {"kind": None, "task": None, "episodes": [], "live_now": False,
                "note": "This policy isn't on the GPU server, so it can't run live yet."}
    served = ((live or {}).get("policies") or {}).get(task)
    episodes = [e.episode for e in held.episodes] if held else []
    if live is None:
        return {"kind": "real", "task": task, "episodes": episodes, "live_now": False, "note": DOWN}
    if not served:
        return {"kind": "real", "task": task, "episodes": episodes, "live_now": False,
                "note": "The GPU server is up but doesn't have this policy loaded (real-demo policies load only "
                        "when it starts with SERVE_REAL=1)."}
    if served.get("checkpoint") != _ck_name(ck):
        return {"kind": "real", "task": task, "episodes": episodes, "live_now": False,
                "note": f"The GPU server is running a different {task} policy ({served.get('checkpoint')})."}
    return {"kind": "real", "task": task, "episodes": list(served.get("held_out") or episodes), "live_now": True,
            "note": None}


def _run_sim(task: str | None, variant: str, live: dict | None) -> dict[str, Any]:
    if not task:
        return {"kind": None, "task": None, "episodes": [], "live_now": False,
                "note": "This policy isn't on the GPU server, so it can't run live yet."}
    served = ((live or {}).get("sim") or {}).get(task)
    if live is None:
        return {"kind": "sim", "task": task, "episodes": [], "live_now": False, "note": DOWN}
    if not served:
        return {"kind": "sim", "task": task, "episodes": [], "live_now": False,
                "note": "The GPU server is up but doesn't have this policy loaded."}
    if served.get("variant") != variant:
        return {"kind": "sim", "task": task, "episodes": [], "live_now": False,
                "note": f"The GPU server is running the {_policy(str(served.get('variant')))} policy for this task."}
    return {"kind": "sim", "task": task, "episodes": [], "live_now": True, "note": None}


async def list_skills(ctx: Ctx, real: bool = False) -> list[dict[str, Any]]:
    """Every policy trained in our physics sim (with real=True, also the real-demo ones), newest first. Reads the
    live GPU server's /policies.json (3 s timeout) to say which can run live now."""
    st = await live_policy_service.status(ctx)
    live = st if st.get("available") else None
    keys = {**(st.get("projects") or {}), **_mapping(ctx)}
    projects = {r["id"]: Project.model_validate(r["data"]) for r in await ctx.store.list("projects")
                if r.get("status") != project_cleanup.DELETED}
    tjs = {t.id: t for t in await ctx.repo.list("training_jobs", TrainingJob)}
    branches = {b.id: b for b in await ctx.repo.list("simulation_branches", SimulationBranch)}
    runs = {r.id: r for r in await ctx.repo.list("evaluation_runs", EvaluationRun)}
    evals: dict[str, dict[str, Any]] = {}  # checkpoint id -> the score of its latest evaluation with results
    for r in sorted(runs.values(), key=lambda r: r.created_at):
        if r.policy.kind == "checkpoint" and r.policy.checkpoint_id and r.status == JobStatus.succeeded \
                and (sc := _eval_score(r, branches)):
            evals[r.policy.checkpoint_id] = sc
    replays: dict[str, list[PolicyReplay]] = defaultdict(list)
    for rp in await ctx.repo.list("policy_replays", PolicyReplay):
        replays[rp.project_id].append(rp)
    groups: dict[str, list[Checkpoint]] = defaultdict(list)  # training job (or the checkpoint itself) -> checkpoints
    for ck in await ctx.repo.list("checkpoints", Checkpoint):
        if ck.project_id not in projects:
            continue
        tj = tjs.get(ck.training_job_id or "")
        if tj is not None and tj.status != JobStatus.succeeded:
            continue
        groups[tj.id if tj else ck.id].append(ck)

    out: list[dict[str, Any]] = []
    served_sims: set[str] = set()
    for gid, cks in groups.items():
        tj, p = tjs.get(gid), projects[cks[0].project_id]
        task, sim = _task_of(p, keys)
        names = {_ck_name(c) for c in cks}
        variant = _variant(tj, cks[0])
        # its replays and live runs; ones recorded before the checkpoint was registered match by name (live sim
        # streams name the policy <variant>@<step>)
        mine = [r for r in replays[p.id] if r.checkpoint_id in {c.id for c in cks} or r.checkpoint_id is None and (
            r.checkpoint_name in names or r.kind == "live_sim" and r.checkpoint_name.split("@")[0] == variant)]
        held = next((r for r in sorted(mine, key=lambda r: r.created_at, reverse=True)
                     if r.kind == "held_out_replay" and r.joint_mae_deg is not None), None)
        if sim is None:
            sim = not (p.mode == Mode.real or held is not None)
        served = (((live or {}).get("sim") or {}).get(task or "") if sim else None) or {}
        if served.get("variant") != variant:
            served = {}
        # the checkpoint this skill stands for: the one the GPU server runs, else the best evaluated, else the newest
        rep = next((c for c in cks if served and c.step == int(served.get("step") or -1)), None) or max(
            cks, key=lambda c: (evals[c.id]["value"] if c.id in evals else -1, c.step, str(c.created_at)))
        if served:
            served_sims.add(task or "")
        score = (evals[rep.id] if rep.id in evals else _server_score(served) if served
                 else _joint_score(held) if held else
                 {"kind": "none", "value": None, "n": None, "label": "Not measured yet", "detail": NONE_DETAIL})
        past = await _runs(ctx, mine)
        evidence = await _evidence(ctx, score, branches, runs, held)
        out.append({
            "id": f"sk_{gid}", "name": _name(task, sim, p), "task": task, "data": "sim" if sim else "real",
            "instruction": _prompt(p), "robot": _robot(tj),
            "project_id": p.id, "project_name": p.name, "policy_type": rep.policy_type,
            "policy": _policy(rep.policy_type), "variant": variant, "steps": rep.step,
            "trained_where": _where(tj), "trained_at": rep.created_at.isoformat(),
            "weights": "on the cluster" if is_remote(rep.path) else "on this machine",
            "checkpoint_id": rep.id, "checkpoint": _ck_name(rep), "training_job_id": tj.id if tj else None,
            "dataset_version_id": tj.dataset_version_id if tj else None, "slurm_job": tj.external_id if tj else None,
            "layouts": score.get("layouts"),
            "score": score, "run": _run_sim(task, variant, live) if sim else _run_real(task, rep, held, live),
            "runs": past, "evidence": evidence,
            "thumbnail_url": next((x["video_url"] for x in [*past, *sorted(evidence, key=lambda e: not e.get(
                "success"))] if x.get("video_url")), None),
        })

    # sim policies the GPU server runs that were never imported here (e.g. fresh from training, score pending)
    for task, s in ((live or {}).get("sim") or {}).items():
        if task in served_sims:
            continue
        p = projects.get(keys.get(f"sim_{task}", ""))
        variant = str(s.get("variant") or "")
        out.append({
            "id": f"sk_served_sim_{task}", "name": _name(task, True, p), "task": task, "data": "sim",
            "instruction": str(s.get("instruction") or ""), "robot": "SO-101", "project_id": p.id if p else None,
            "project_name": p.name if p else None, "policy_type": variant, "policy": _policy(variant),
            "variant": variant, "steps": s.get("step"), "trained_where": "Futurama", "trained_at": None,
            "weights": "on the cluster", "checkpoint_id": None, "checkpoint": f"{variant}@{s.get('step')}",
            "training_job_id": None, "dataset_version_id": None, "slurm_job": None, "score": _server_score(s),
            "layouts": s.get("layouts") or None,
            "run": _run_sim(task, variant, live), "runs": [], "evidence": [], "thumbnail_url": None,
        })

    # sim skills a GPU server served before, and known ones with a recorded replay: listed while it's down (#308)
    seen = _read_seen(ctx)
    if fresh := {s["task"]: s for s in out if s["id"].startswith("sk_served_sim_")}:
        seen = {**seen, **fresh}
        try:
            (ctx.data_dir / SEEN).write_text(json.dumps(seen))
        except OSError:
            log.exception("could not save the served sim skills")
    have = {s["task"] for s in out if s["data"] == "sim"}
    for task, s in {**{t: _seeded(t) for t in SEEDED}, **seen}.items():
        if task not in have and (task in seen or replay_urls(ctx, task)):
            out.append({**s, "run": _run_sim(task, str(s.get("variant") or ""), live)})
    for s in out:
        s["replays"] = replay_urls(ctx, s["task"]) if s["data"] == "sim" else []
        s["live"] = bool(s["run"]["live_now"])
        s["thumbnail_url"] = s["thumbnail_url"] or next(iter(s["replays"]), None)
    out.sort(key=lambda s: s["trained_at"] or "", reverse=True)
    return out if real else [s for s in out if s["data"] == "sim"]


async def get_skill(ctx: Ctx, skill_id: str, real: bool = False) -> dict[str, Any]:
    for s in await list_skills(ctx, real):
        if s["id"] == skill_id:
            return s
    raise NotFound(f"No physical skill {skill_id}.")
