"""The guided planner: walks a person from "what should the robot learn?" to a trained, evaluated policy.

It is deterministic and needs no model key. It reads the project's state, decides the one next step
of the pipeline (spec 3.6), runs it through the same tool registry an MCP agent uses, and yields chat
events: ``text``, ``tool_call``, ``tool_result``, ``approval_required``, ``step`` and ``card``.

    intake questions -> build_simulation (preview + feasibility) -> collect demos (Astra if a planner
    is configured, otherwise the labelled scripted oracle) -> dataset -> propose training -> a human
    approves in the chat -> launch -> evaluate -> results -> improve_policy

Gated work (training, evaluation) is only ever proposed here; the human clicks the approval card.
The LLM path gets the same plan through :func:`guidance`.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ..context import Ctx
from ..intake import catalogue, status
from ..models import Checkpoint, EpisodeStatus, JobStatus, Project, TrainingJob
from . import agent_orchestrator as orch
from . import dataset_service, local_training, recording_service, simulation_runner, training_service
from .guide_answers import answer, catalogue_values, is_yes, option_buttons
from .recommendation_service import MIN_TRAIN_EPISODES, region_of

Event = tuple[str, dict[str, Any]]
STEPS = ["intake", "simulate", "collect", "train", "evaluate"]
STEP_LABELS = {"intake": "Describe", "simulate": "Simulate", "collect": "Collect", "train": "Train",
               "evaluate": "Evaluate"}
JOB_STEP = {"agent_collect": "collect", "record": "collect", "dataset": "train", "train_from_agent": "train",
            "train": "train", "train_smoke": "train", "evaluate": "evaluate", "improve_policy": "collect",
            "analyze": "collect"}
JOB_TITLES = {"agent_collect": "Collecting demonstrations", "dataset": "Building the dataset",
              "train": "Training on the cluster", "train_smoke": "CPU smoke training run",
              "evaluate": "Evaluating in simulation", "improve_policy": "Improving the policy",
              "record": "Recording demonstrations", "analyze": "Analyzing demonstrations"}
STOCK_SCENE_NOTE = ("Demonstrations and evaluations still run in the stock SO-101 tabletop scene (a cube into a box); "
                    "running them in the scene built from your description arrives with roadmap P5/P6.")
PATH_RE = re.compile(r"([A-Za-z]:[\\/][^\s\"']+|~?[./]?[\w.\-]+(?:[\\/][\w.\-]+)+)")

GUIDANCE = """You are guiding a person through the Cortex pipeline, one step at a time:
intake questions -> build_simulation (show the preview and the feasibility report; if anything is infeasible,
offer its suggestions as patches) -> collect demonstrations (start_agent_collection with collector 'agent' when an
Astra planner is configured, otherwise collector 'oracle'; say plainly which one you used) -> create_dataset_version
-> propose_training_job (a human approves it in the chat; you cannot) -> evaluate the checkpoint (needs approval)
-> summarize the success rates with their 95% CIs -> offer improve_policy.
With no GPU cluster the training backend is 'manual': say training needs a GPU, give the command, and ask for the
checkpoint path when it is done. Never claim training ran when it did not.
Planner's next step: {next}"""


@dataclass
class Plan:
    step: str
    action: str
    reason: str
    data: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ state -> plan

def _active_jobs(ctx: Ctx, pid: str) -> list[Any]:
    return [j for j in ctx.jobs.list(pid) if j.status in ("queued", "running")]


async def plan(ctx: Ctx, p: Project) -> Plan:
    if p.intake is None:
        return Plan("intake", "no_intake", "this project has no intake (legacy project)")
    st = status(p.intake)
    open_q = [q for q in st.questions if q.required]
    if open_q:
        return Plan("intake", "ask", open_q[0].prompt, {"question": open_q[0]})
    if p.scene is None:
        return Plan("simulate", "build", "the intake is complete but the sim scene is not built")
    feas = p.scene.get("feasibility") or {}
    if not feas.get("ok", False):
        return Plan("simulate", "infeasible", feas.get("summary", "not feasible"), {"feasibility": feas})
    jobs = _active_jobs(ctx, p.id)
    if jobs:
        return Plan(JOB_STEP.get(jobs[0].kind, "collect"), "wait", f"{jobs[0].kind} is {jobs[0].status}",
                    {"jobs": jobs})
    eps = await recording_service.list_episodes(ctx, p.id)
    accepted = sum(1 for e in eps if e.status == EpisodeStatus.accepted)
    if accepted < MIN_TRAIN_EPISODES:
        return Plan("collect", "collect", f"{accepted} accepted demonstrations; training needs {MIN_TRAIN_EPISODES}",
                    {"accepted": accepted, "attempted": len(eps)})
    dss = await dataset_service.list_versions(ctx, p.id)
    if not dss:
        return Plan("train", "dataset", f"{accepted} accepted demonstrations are ready to snapshot")
    ds = dss[-1]
    if accepted > ds.num_source_episodes:  # an improve round added demos: snapshot them before retraining
        return Plan("train", "dataset", f"{accepted - ds.num_source_episodes} new accepted demonstrations since {ds.id}")
    # Only work on the latest dataset counts; earlier rounds stay in the lineage graph.
    ckpts = [c for c in await ctx.repo.list("checkpoints", Checkpoint, project_id=p.id) if c.created_at >= ds.created_at]
    tjs = [t for t in await ctx.repo.list("training_jobs", TrainingJob, project_id=p.id) if t.dataset_version_id == ds.id]
    tj = tjs[-1] if tjs else None
    if not ckpts:
        if tj is None or tj.status in (JobStatus.failed, JobStatus.cancelled):
            why = "no training job yet" if tj is None else f"{tj.id} {tj.status}: {tj.error or ''}"
            return Plan("train", "propose", why, {"dataset_version_id": ds.id})
        if tj.status == JobStatus.proposed:
            return Plan("train", "await_approval", f"{tj.id} is waiting for a human approval", {"tj": tj})
        if tj.status == JobStatus.approved:
            return Plan("train", "launch", f"{tj.id} is approved", {"tj": tj})
        return Plan("train", "manual" if tj.backend == "manual" else "wait_train",
                    f"{tj.id} is {tj.status} on {tj.backend}", {"tj": tj})
    ck = ckpts[-1]
    runs = [e for e in await simulation_runner.list_evaluations(ctx, p.id)
            if e["run"].policy.kind == "checkpoint" and e["run"].policy.checkpoint_id == ck.id]
    done = [e for e in runs if e["run"].status == JobStatus.succeeded and any(b.result for b in e["branches"])]
    if done:
        return Plan("evaluate", "results", f"checkpoint {ck.id} is evaluated", {"ck": ck, "eval": done[-1]})
    if runs and runs[-1]["run"].status == JobStatus.failed:
        j = ctx.jobs.get(runs[-1]["run"].job_id or "")
        return Plan("evaluate", "eval_failed", (j.error if j and j.error else "the evaluation failed"), {"ck": ck})
    return Plan("evaluate", "evaluate", f"checkpoint {ck.id} has not been evaluated", {"ck": ck})


async def guidance(ctx: Ctx, p: Project) -> str:
    pl = await plan(ctx, p)
    return GUIDANCE.format(next=f"{pl.step}: {pl.action} ({pl.reason})")


# ------------------------------------------------------------------ cards

def step_event(step: str) -> Event:
    i = STEPS.index(step)
    return "step", {"step": step, "index": i + 1, "label": STEP_LABELS[step], "steps": STEPS}


def question_card(p: Project, q: Any) -> dict[str, Any]:
    return {"type": "question", "path": q.path, "prompt": q.prompt, "why": q.why, "question_type": q.type,
            "options": option_buttons(p.intake, q), "allow_text": True}  # type: ignore[arg-type]


def feasibility_card(feas: dict[str, Any]) -> dict[str, Any]:
    sugg = []
    for o in feas.get("objects", []):
        s = o.get("suggestion") or ""
        key = s.removeprefix("try a ").strip()
        if not o["ok"] and key in catalogue.CATALOGUE:
            sugg.append({"label": f"Use a {key} instead of {o['name']}",
                         "patch": {"objects": [{"name": o["name"], **catalogue_values(key)}]}})
    for pl in feas.get("places", []):
        if not pl["ok"] and pl.get("suggestion"):
            x, y = pl["suggestion"]
            patch = ({"objects": [{"name": pl["name"], "start": [x, y]}]} if pl["kind"] == "start"
                     else {"zones": [{"name": pl["name"], "center": [x, y]}]})
            sugg.append({"label": f"Move {pl['name']} to ({x:.2f}, {y:.2f}) m", "patch": patch})
    return {"type": "feasibility", "ok": feas.get("ok", False), "summary": feas.get("summary", ""),
            "jaw_opening_m": feas.get("jaw_opening_m"),
            "objects": [{k: o.get(k) for k in ("name", "ok", "grasp", "payload", "tall", "grasp_mode")}
                        for o in feas.get("objects", [])],
            "places": feas.get("places", []), "suggestions": sugg}


def image_card(res: dict[str, Any], camera: str = "top", label: str = "") -> dict[str, Any]:
    prev = res.get("preview") or res
    return {"type": "image", "label": label or f"FIG.02 SCENE · {camera.upper()} CAMERA", "camera": camera,
            "url": prev.get("url"), "key": prev.get("key"), "scene": prev.get("scene"),
            "image_jpeg_base64": res.get("image_jpeg_base64")}


def job_card(j: Any, title: str | None = None) -> dict[str, Any]:
    d = j.model_dump() if hasattr(j, "model_dump") else dict(j)
    return {"type": "job", "job_id": d["id"], "kind": d["kind"], "title": title or JOB_TITLES.get(d["kind"], d["kind"]),
            "job": d}


def results_card(ev: dict[str, Any], ck: Checkpoint) -> dict[str, Any]:
    run = ev["run"]
    rows = [{"name": b.spec.name, "n": b.result.n, "successes": b.result.successes,
             "success_rate": b.result.success_rate, "ci95": list(b.result.ci95)}
            for b in ev["branches"] if b.result]
    return {"type": "results", "evaluation_id": run.id, "policy": run.policy_label, "checkpoint_id": ck.id,
            "checkpoint_step": ck.step, "smoke": bool(ck.metrics.get("cpu_smoke_run")), "branches": rows}


def actions_card(*pairs: tuple[str, str]) -> dict[str, Any]:
    return {"type": "actions", "actions": [{"label": lab, "message": msg} for lab, msg in pairs]}


def cards_for_tool(name: str, res: dict[str, Any]) -> list[dict[str, Any]]:
    """Rich cards for a tool result (used by the LLM path too)."""
    out: list[dict[str, Any]] = []
    if name == "build_simulation":
        out += [image_card(res), feasibility_card(res.get("feasibility") or {})]
    elif name == "capture_frame":
        out.append(image_card(res, res.get("camera", "top"), f"FIG.03 SCENE · {res.get('camera', 'top').upper()} CAMERA"))
    elif res.get("job"):
        out.append(job_card(res["job"]))
    return out


# ------------------------------------------------------------------ running steps

async def _tool(ctx: Ctx, name: str, args: dict[str, Any], out: dict[str, Any]) -> AsyncIterator[Event]:
    """Call a registry tool, yielding tool_call/tool_result; the result lands in ``out['res']``."""
    yield "tool_call", {"tool": name, "args": {k: v for k, v in args.items() if k != "project_id"}}
    try:
        res = await orch.call_tool(ctx, name, args, actor="agent")
    except Exception as e:  # reported to the person, never swallowed
        out["error"] = f"{type(e).__name__}: {e}"
        yield "tool_result", {"tool": name, "summary": out["error"], "ref_ids": []}
        return
    out["res"] = res
    yield "tool_result", {"tool": name, "summary": res.get("summary", ""), "ref_ids": res.get("ref_ids", [])}


def text(s: str) -> Event:
    return "text", {"delta": s}


def card(c: dict[str, Any]) -> Event:
    return "card", c


async def respond(ctx: Ctx, pid: str, message: str, *, applied: bool = False,
                  kickoff: bool = False) -> AsyncIterator[Event]:
    """One guided turn. ``applied``: the message came with a patch that agent.chat already applied.
    ``kickoff``: the first message of a chat (the task prompt itself), so it is not parsed as an answer."""
    p = await orch.get_project(ctx, pid)
    pl = await plan(ctx, p)
    msg = message.strip()

    if pl.action == "ask" and not (applied or kickoff):
        q = pl.data["question"]
        patch = answer(p.intake, q, msg)  # type: ignore[arg-type]
        if patch is None:
            yield step_event("intake")
            yield text("I couldn't match that to the open question. Pick one of the options, or answer in your "
                       "own words (sizes like \"4 cm\", masses like \"50 g\", or \"yes\" to accept).")
            yield card(question_card(p, q))
            return
        out: dict[str, Any] = {}
        async for ev in _tool(ctx, "update_intake", {"project_id": pid, "patch": patch}, out):
            yield ev
        if "error" in out:
            yield text(f"Cortex refused that answer: {out['error']}")
            yield card(question_card(p, q))
            return
        p = await orch.get_project(ctx, pid)
        pl = await plan(ctx, p)

    if pl.action == "ask":
        q = pl.data["question"]
        yield step_event("intake")
        n = sum(1 for x in status(p.intake).questions if x.required)  # type: ignore[arg-type]
        lead = "Let's describe the task so Cortex can build it in simulation. " if kickoff else ""
        yield text(f"{lead}{n} question{'s' if n != 1 else ''} left.")
        yield card(question_card(p, q))
        return

    if pl.action in ("build", "infeasible") and (pl.action == "build" or applied):
        async for ev in _build(ctx, pid):
            yield ev
        return
    if pl.action == "infeasible":
        yield step_event("simulate")
        yield text(f"The scene is built but {pl.reason}. Apply a suggestion to fix it, or describe the change.")
        yield card(feasibility_card(pl.data["feasibility"]))
        return

    async for ev in _pipeline(ctx, p, pl, msg):
        yield ev


async def _build(ctx: Ctx, pid: str) -> AsyncIterator[Event]:
    yield step_event("simulate")
    yield text("The description is complete. Building the MuJoCo scene and checking grasp, payload and reach.")
    out: dict[str, Any] = {}
    async for ev in _tool(ctx, "build_simulation", {"project_id": pid}, out):
        yield ev
    if "error" in out:
        yield text(f"The scene build failed: {out['error']}")
        return
    for c in cards_for_tool("build_simulation", out["res"]):
        yield card(c)
    feas = out["res"]["feasibility"]
    if feas["ok"]:
        yield text("Everything is feasible: the arm can grasp, carry and reach every object and zone. "
                   "Next, collect demonstrations.")
        yield card(actions_card(("Collect demonstrations", "Collect demonstrations")))
    else:
        yield text(f"Not feasible yet: {feas['summary']}. Each suggestion below is a one-click fix.")


async def _pipeline(ctx: Ctx, p: Project, pl: Plan, msg: str) -> AsyncIterator[Event]:
    pid = p.id
    low = msg.lower()
    tj: TrainingJob | None = pl.data.get("tj")

    # Explicit requests that do not follow the default next step.
    if pl.action == "manual" and tj and (m := PATH_RE.search(msg)) and not is_yes(msg):
        yield step_event("train")
        try:
            ck = await training_service.register_checkpoint(ctx, pid, tj.id, m.group(1).strip(), actor="gui")
        except FileNotFoundError as e:
            yield text(f"I couldn't register that checkpoint: {e}. Point me at the pretrained_model directory "
                       "(it contains config.json and model.safetensors).")
            return
        yield text(f"Registered checkpoint {ck.id} ({ck.policy_type}). Next, evaluate it in simulation.")
        yield "approval_required", _eval_approval(ck)
        return
    if "smoke" in low and pl.step in ("train", "evaluate"):
        async for ev in _smoke(ctx, p):
            yield ev
        return
    if "improve" in low and pl.action == "results":
        async for ev in _improve(ctx, p, pl):
            yield ev
        return
    yield step_event(pl.step)
    if pl.action == "wait":
        names = ", ".join(JOB_TITLES.get(j.kind, j.kind).lower() for j in pl.data["jobs"])
        yield text(f"Still working: {names}. Progress updates live below; continue when it finishes.")
        for j in pl.data["jobs"]:
            yield card(job_card(j))
        return
    if pl.action == "collect":
        async for ev in _collect(ctx, p, pl):
            yield ev
        return
    if pl.action == "dataset":
        out: dict[str, Any] = {}
        async for ev in _tool(ctx, "create_dataset_version", {"project_id": pid}, out):
            yield ev
        if "error" in out:
            yield text(f"The dataset build failed to start: {out['error']}")
            return
        yield text(f"{pl.reason}. Building a versioned LeRobot dataset (train/eval split) from them.")
        yield card(job_card(out["res"]["job"]))
        return
    if pl.action == "propose":
        out = {}
        args = {"project_id": pid, "dataset_version_id": pl.data["dataset_version_id"]}
        async for ev in _tool(ctx, "propose_training_job", args, out):
            yield ev
        if "error" in out:
            yield text(f"Training could not be proposed: {out['error']}")
            return
        tj = TrainingJob.model_validate(out["res"]["training_job"])
        yield text(_proposal_text(tj))
        yield "approval_required", _train_approval(tj)
        return
    if pl.action == "await_approval" and tj:
        yield text(f"Training {tj.id} is waiting for you. Approving is a click on the card; agents cannot approve.")
        yield "approval_required", _train_approval(tj)
        return
    if pl.action == "launch" and tj:
        tj = await training_service.launch(ctx, tj.id, actor="gui")
        pl = Plan("train", "manual" if tj.backend == "manual" else "wait_train", "", {"tj": tj})
    if pl.action == "manual" and tj:
        yield text("Training needs a GPU, and no cluster is configured, so this job runs on the manual backend. "
                   "Run the command below on any GPU machine, then paste the checkpoint path here "
                   "(the pretrained_model directory) and I'll register it. Nothing is marked trained until then.")
        yield card(_manual_card(ctx, tj))
        return
    if pl.action == "wait_train" and tj:
        yield text(f"Training {tj.id} is {tj.status} on {tj.backend}. Logs stream into the job card.")
        for j in ctx.jobs.list(pid):
            if j.kind == "train":
                yield card(job_card(j))
        return
    if pl.action == "evaluate":
        ck: Checkpoint = pl.data["ck"]
        yield text(f"Checkpoint {ck.id} is ready. Evaluating it runs parallel MuJoCo branches (baseline, shifted "
                   f"object, dim lighting); it needs your approval. {STOCK_SCENE_NOTE}")
        yield "approval_required", _eval_approval(ck)
        return
    if pl.action == "eval_failed":
        ck = pl.data["ck"]
        yield text(f"The last evaluation of {ck.id} failed: {pl.reason}. No results were recorded. Approve to run it again.")
        yield "approval_required", _eval_approval(ck)
        return
    if pl.action == "results":
        ck = pl.data["ck"]
        rc = results_card(pl.data["eval"], ck)
        parts = [f"{b['name']} {b['successes']}/{b['n']} ({b['ci95'][0]:.0%}–{b['ci95'][1]:.0%} 95% CI)"
                 for b in rc["branches"]]
        target = p.intake.task.sim_target if p.intake else 0.7
        best = max((b["success_rate"] for b in rc["branches"]), default=0.0)
        verdict = "meets" if best >= target else "is below"
        smoke = " This is a CPU smoke checkpoint, so near-zero success is expected." if rc["smoke"] else ""
        yield text(f"Measured in simulation: {'; '.join(parts)}. The best branch {verdict} the {target:.0%} target."
                   f"{smoke}")
        yield card(rc)
        yield card(actions_card(("Improve the weakest condition", "Improve the policy"), ("Show status", "status")))


async def _collect(ctx: Ctx, p: Project, pl: Plan) -> AsyncIterator[Event]:
    from . import agent_collection_service as acs

    planner = await acs.planner_status(ctx)
    collector = "agent" if planner["available"] else "oracle"
    accepted, attempted = pl.data["accepted"], pl.data["attempted"]
    per_round = p.intake.task.demos_per_round if p.intake else 10
    n = min(50, max(per_round, MIN_TRAIN_EPISODES + 2) if attempted == 0 else max(4, MIN_TRAIN_EPISODES + 2 - accepted))
    out: dict[str, Any] = {}
    args = {"project_id": p.id, "num_demos": n, "collector": collector, "diversity": "medium"}
    async for ev in _tool(ctx, "start_agent_collection", args, out):
        yield ev
    if "error" in out:
        yield text(f"Collection did not start: {out['error']}")
        return
    who = (f"GPT-6 Astra ({planner['model']}) plans from the camera and validated IK executes" if collector == "agent"
           else f"No Astra planner is available ({planner['reason']}), so I'm using the scripted IK oracle. It reads "
                "the simulator's object pose (privileged), and every episode it records is labelled sim_scripted")
    more = f"Only {accepted} demonstrations were accepted so far, so this collects more. " if attempted else ""
    yield text(f"{more}Collecting {n} demonstrations: {who}. Simulator-verified successes become LeRobot episodes. "
               f"{STOCK_SCENE_NOTE}")
    j = ctx.jobs.get(out["res"].get("job_id") or "")
    if j:
        yield card(job_card(j))


async def _improve(ctx: Ctx, p: Project, pl: Plan) -> AsyncIterator[Event]:
    """improve_policy when Astra is available; otherwise the same targeting with labelled oracle demos."""
    from . import agent_collection_service as acs

    yield step_event("collect")
    out: dict[str, Any] = {}
    if (await acs.planner_status(ctx))["available"]:
        async for ev in _tool(ctx, "improve_policy", {"project_id": p.id}, out):
            yield ev
        if "error" in out:
            yield text(f"improve_policy failed: {out['error']}")
            return
        yield text(out["res"]["summary"] + " The retraining proposal will need your approval again.")
        if out["res"].get("job_id") and (j := ctx.jobs.get(out["res"]["job_id"])):
            yield card(job_card(j))
        return
    scored = [(b.result.success_rate, b) for b in pl.data["eval"]["branches"] if b.result]
    rate, b = min(scored, key=lambda t: t[0])
    cell = {"cube_region": region_of(b.spec.condition.cube_region, b.spec.cube_offset),
            "lighting": b.spec.condition.lighting}
    args = {"project_id": p.id, "num_demos": 6, "collector": "oracle", "conditions": [cell]}
    async for ev in _tool(ctx, "start_agent_collection", args, out):
        yield ev
    if "error" in out:
        yield text(f"Collection did not start: {out['error']}")
        return
    yield text(f"The weakest measured branch is {b.spec.name} ({rate:.0%}), so I'm collecting 6 more demonstrations in "
               f"the {cell['cube_region']} / {cell['lighting']} cell. No Astra planner is configured, so improve_policy "
               "(which uses Astra) is unavailable; these come from the labelled scripted oracle. When they finish, "
               "continue: Cortex snapshots a new dataset version and proposes retraining for your approval.")
    if (j := ctx.jobs.get(out["res"].get("job_id") or "")):
        yield card(job_card(j))


async def _smoke(ctx: Ctx, p: Project) -> AsyncIterator[Event]:
    yield step_event("train")
    st = await orch.project_state(ctx, p)
    if not st.dataset_versions:
        yield text("A smoke run needs a dataset version first.")
        return
    try:
        job = await local_training.start_smoke(ctx, p, st.dataset_versions[-1])
    except (RuntimeError, FileNotFoundError) as e:
        yield text(f"The CPU smoke run is not available here: {e}")
        return
    yield text(f"Started a CPU smoke run: lerobot-train for {local_training.SMOKE_STEPS} steps on this machine. "
               "It proves the dataset trains end to end and gives a real checkpoint, but at this length the policy "
               "will not succeed at the task. It is labelled as a smoke checkpoint everywhere.")
    yield card(job_card(job))


def _proposal_text(tj: TrainingJob) -> str:
    e = tj.estimate
    gap = (" No GPU cluster is configured, so after you approve, Cortex gives you the exact command to run on any "
           "GPU and waits for the checkpoint." if tj.backend == "manual" else "")
    return (f"Proposed {tj.policy_type.upper()} training {tj.id} on {tj.backend}: {tj.config.get('steps')} steps, "
            f"about {e.gpu_hours} GPU-hours (${e.est_cost_usd:.2f}). {tj.rationale}{gap}")


def _train_approval(tj: TrainingJob) -> dict[str, Any]:
    return {"kind": "training", "ref_id": tj.id, "summary": _proposal_text(tj),
            "args": {"backend": tj.backend, "steps": tj.config.get("steps"), "policy_type": tj.policy_type,
                     "gpu_hours": tj.estimate.gpu_hours, "est_cost_usd": tj.estimate.est_cost_usd}}


def _eval_approval(ck: Checkpoint) -> dict[str, Any]:
    return {"kind": "evaluation", "ref_id": ck.id, "args": {"policy": ck.id, "episodes_per_branch": 10},
            "summary": f"Evaluate checkpoint {ck.id} ({ck.policy_type}, step {ck.step}) on baseline / shifted "
                       "object / dim lighting, 10 seeds each."}


def _manual_card(ctx: Ctx, tj: TrainingJob) -> dict[str, Any]:
    return {"type": "training_manual", "training_job_id": tj.id, "backend": tj.backend,
            "command": training_service.manual_command(ctx, tj),
            "checkpoint_hint": f"outputs/{tj.id}/checkpoints/last/pretrained_model",
            "smoke_available": local_training.available() is None, "smoke_reason": local_training.available(),
            "smoke_steps": local_training.SMOKE_STEPS}
