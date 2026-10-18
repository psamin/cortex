"""Project state, next-action planning, and the typed tool registry.

The registry is the single application API for agents: the MCP server (via
POST /api/tools/{name}) and the in-studio agent both dispatch through
`call_tool`, which invokes the same domain services the GUI's REST routes use
and writes an audit row for every call. Physical-robot actions and paid or
cluster jobs are never executable by an agent without a human approval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..context import Ctx
from ..models import (
    Collector,
    StartAgentCollectionRequest,
    Augmentation,
    Checkpoint,
    Condition,
    CreateDatasetVersionRequest,
    CreateProjectRequest,
    EpisodeStatus,
    JobStatus,
    LaunchEvaluationRequest,
    Mode,
    NextAction,
    PolicyRef,
    Project,
    ProjectState,
    TrainingJob,
)
from . import (
    analysis_service,
    dataset_service,
    experiment_comparison,
    project_service,
    recommendation_service,
    recording_service,
    simulation_runner,
    training_service,
)
from .augment import PRESETS

# ------------------------------------------------------------------ projects

async def get_project(ctx: Ctx, project_id: str) -> Project:
    p = await ctx.repo.get("projects", project_id, Project)
    if p is None:
        raise KeyError(f"project {project_id} not found")
    return p


async def create_project(ctx: Ctx, req: CreateProjectRequest, actor: str = "gui") -> Project:
    p = project_service.create_project(req)
    await ctx.repo.save("projects", p, status="active")
    await ctx.store.put("robot_configs", {
        "id": f"rc_{p.id}", "project_id": p.id, "status": "sim" if p.mode == Mode.sim else "real",
        "data": {"id": f"rc_{p.id}", "robot": "so101_follower", "mode": p.mode,
                 "sim_env": "lerobot/so101-tabletop-mujoco@12e504e8", "units": "degrees, gripper 0-100",
                 "fps": 30, "hardware_commands_require_approval": True}})
    await ctx.audit(actor, "create_project", p.id, req.model_dump(mode="json"), "ok", p.name)
    return p


async def project_state(ctx: Ctx, project: Project) -> ProjectState:
    eps = await recording_service.list_episodes(ctx, project.id)
    counts = {s.value: sum(1 for e in eps if e.status == s) for s in EpisodeStatus}
    counts["total"] = len(eps)
    accepted_by_cell: dict[str, int] = {}
    for e in eps:
        if e.status == EpisodeStatus.accepted and e.source != "imported":  # real demos don't cover sim cells
            accepted_by_cell[e.condition.key] = accepted_by_cell.get(e.condition.key, 0) + 1
    for cell in project.plan.cells:
        cell.collected = accepted_by_cell.get(cell.condition.key, 0)
    an = await analysis_service.latest_analysis(ctx, project.id)
    dss = await dataset_service.list_versions(ctx, project.id)
    tjs = await ctx.repo.list("training_jobs", TrainingJob, project_id=project.id)
    evals = await simulation_runner.list_evaluations(ctx, project.id)
    recs = await recommendation_service.open_recommendations(ctx, project.id)
    state = ProjectState(
        project=project, episode_counts=counts, latest_analysis_id=an.id if an else None,
        dataset_versions=[d.id for d in dss],
        training_jobs=[{"id": t.id, "status": t.status, "policy_type": t.policy_type, "backend": t.backend,
                        "dataset_version_id": t.dataset_version_id, "checkpoint_ids": t.checkpoint_ids}
                       for t in tjs],
        evaluations=[{"id": e["run"].id, "name": e["run"].name, "status": e["run"].status,
                      "policy": e["run"].policy_label,
                      "branches": {b.spec.name: (f"{b.result.successes}/{b.result.n}" if b.result else b.status)
                                   for b in e["branches"]}} for e in evals],
        open_recommendations=recs, next_actions=[], storage_backend=ctx.blobs.name, db_backend=ctx.store.name)
    state.next_actions = await next_actions(ctx, project, state)
    return state


async def next_actions(ctx: Ctx, project: Project, st: ProjectState) -> list[NextAction]:
    pid = project.id
    out: list[NextAction] = []
    c = st.episode_counts
    running = [j for j in ctx.jobs.list(pid) if j.status in ("queued", "running")]
    if running:
        out.append(NextAction(action="get_job", reason=f"{len(running)} job(s) in progress ({running[0].kind})",
                              args={"job_id": running[0].id}))
    if c["total"] == 0:
        conds = [{"cube_region": cell.condition.cube_region, "lighting": cell.condition.lighting,
                  "count": cell.target} for cell in project.plan.cells]
        out.append(NextAction(action="record_demonstrations", reason="No demonstrations yet: record the initial plan "
                              "in simulation", args={"project_id": pid, "conditions": conds, "auto_accept": True}))
        return out
    if c.get("recorded", 0):
        out.append(NextAction(action="review_episodes", reason=f"{c['recorded']} recorded demos await accept/reject"))
    if not st.latest_analysis_id:
        out.append(NextAction(action="analyze_dataset", reason="Episodes have not been quality-checked yet",
                              args={"project_id": pid}))
    for r in st.open_recommendations[:2]:
        if r.action == "collect" and r.condition:
            out.append(NextAction(action="record_demonstrations", reason=r.title, args={
                "project_id": pid, "auto_accept": True,
                "conditions": [{**r.condition.model_dump(), "count": r.count}], "recommendation_id": r.id}))
    accepted = c.get("accepted", 0)
    if accepted >= recommendation_service.MIN_TRAIN_EPISODES and not st.dataset_versions:
        out.append(NextAction(action="create_dataset_version", reason=f"{accepted} accepted demos are ready to snapshot",
                              args={"project_id": pid}))
    if st.dataset_versions and not st.training_jobs:
        out.append(NextAction(action="propose_training_job", reason="A dataset version exists but no policy is trained",
                              args={"project_id": pid, "dataset_version_id": st.dataset_versions[-1]}))
    for t in st.training_jobs:
        if t["status"] == JobStatus.proposed:
            out.append(NextAction(action="approve_training_job", reason=f"{t['id']} is waiting for human approval",
                                  requires_approval=True, args={"training_job_id": t["id"]}))
        if t["status"] == JobStatus.approved:
            out.append(NextAction(action="launch_training_job", reason=f"{t['id']} is approved",
                                  args={"training_job_id": t["id"]}))
    ckpts = await ctx.repo.list("checkpoints", Checkpoint, project_id=pid)
    evaluated = {e["run"].policy.checkpoint_id for e in await simulation_runner.list_evaluations(ctx, pid)}
    for ck in ckpts:
        if ck.id not in evaluated:
            out.append(NextAction(action="launch_simulation_evaluation",
                                  reason=f"Checkpoint {ck.id} has not been evaluated in simulation",
                                  requires_approval=True, args={"project_id": pid, "policy": ck.id}))
    if not st.evaluations:
        out.append(NextAction(action="launch_simulation_evaluation",
                              reason="Run the 3-condition evaluation with the scripted expert as a reference ceiling",
                              requires_approval=True, args={"project_id": pid, "policy": "scripted_expert"}))
    elif len(st.evaluations) >= 2:
        out.append(NextAction(action="compare_experiments", reason="Compare the latest runs branch by branch",
                              args={"project_id": pid}))
    if not st.open_recommendations:
        out.append(NextAction(action="recommend_demonstrations", reason="Refresh ranked recommendations from current evidence",
                              args={"project_id": pid}))
    return out


# --------------------------------------------------------------- tool registry

class ConditionCount(BaseModel):
    cube_region: Literal["left", "center", "right"] = "center"
    lighting: Literal["dim", "normal", "bright"] = "normal"
    count: int = Field(1, ge=1, le=20)


class Args:
    class CreateProject(BaseModel):
        task_prompt: str = Field(description="What the robot should learn, in plain language")
        mode: Mode = Field(Mode.sim, description="'sim' (default) or 'real'")
        demo_budget: int = 24

    class ProjectRef(BaseModel):
        project_id: str

    class ListEpisodes(BaseModel):
        project_id: str
        status: EpisodeStatus | None = None

    class Record(BaseModel):
        project_id: str
        conditions: list[ConditionCount]
        auto_accept: bool = Field(True, description="accept demos that succeed and pass quality checks")
        recommendation_id: str | None = None
        wait_s: int = Field(0, ge=0, le=600, description="block up to this many seconds for completion")

    class Analyze(BaseModel):
        project_id: str
        use_vlm: bool = True
        wait_s: int = Field(0, ge=0, le=600)

    class Recommend(BaseModel):
        project_id: str
        budget: int | None = None

    class DatasetVersion(BaseModel):
        project_id: str
        augmentation_preset: Literal["lighting_robustness"] | None = Field(
            None, description="brightness x2, color temperature and shadow views of each train episode")
        augmentations: list[Augmentation] = []
        eval_fraction: float = 0.2
        name: str | None = None
        wait_s: int = Field(0, ge=0, le=600)

    class ProposeTraining(BaseModel):
        project_id: str
        dataset_version_id: str
        policy_type: Literal["act", "smolvla"] = "act"
        steps: int | None = None

    class LaunchTraining(BaseModel):
        training_job_id: str
        approval_token: str | None = Field(None, description="operator token; otherwise approve in the studio")

    class LaunchEval(BaseModel):
        project_id: str
        policy: str = Field("scripted_expert", description="'scripted_expert' or a checkpoint id (ck_...)")
        episodes_per_branch: int = Field(10, ge=1, le=50)
        base_seed: int = 1000
        wait_s: int = Field(0, ge=0, le=900)

    class Compare(BaseModel):
        project_id: str
        evaluation_ids: list[str] | None = None

    class JobRef(BaseModel):
        job_id: str


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[Ctx, Any, str], Awaitable[dict[str, Any]]]
    # False hides the tool from the studio's own agent (actor "agent") and `call_tool` refuses it
    # for that actor. MCP clients still reach it on purpose: they drive the simulated arm step by
    # step (robot_*), and launch_training_job stays behind the human approval token either way.
    agent_callable: bool = True


async def _wait(ctx: Ctx, job_id: str, wait_s: int) -> dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + wait_s
    while wait_s and asyncio.get_running_loop().time() < deadline:
        j = ctx.jobs.get(job_id)
        if j and j.status in ("succeeded", "failed", "cancelled"):
            break
        await asyncio.sleep(1)
    j = ctx.jobs.get(job_id)
    return j.model_dump() if j else {"id": job_id, "status": "unknown"}


async def t_create_project(ctx, a, actor):
    p = await create_project(ctx, CreateProjectRequest(task_prompt=a.task_prompt, mode=a.mode,
                                                       demo_budget=a.demo_budget), actor)
    notes = project_service.unsupported_notes(CreateProjectRequest(task_prompt=a.task_prompt, mode=a.mode))
    return {"summary": f"Created project {p.id} '{p.name}' ({p.mode})", "ref_ids": [p.id],
            "project": p.model_dump(mode="json"), "notes": notes}


async def t_state(ctx, a, actor):
    st = await project_state(ctx, await get_project(ctx, a.project_id))
    return {"summary": f"{st.episode_counts['total']} episodes, {len(st.dataset_versions)} datasets, "
                       f"{len(st.evaluations)} evaluations", "ref_ids": [a.project_id],
            "state": st.model_dump(mode="json")}


async def t_list_episodes(ctx, a, actor):
    eps = await recording_service.list_episodes(ctx, a.project_id)
    if a.status:
        eps = [e for e in eps if e.status == a.status]
    rows = [{"id": e.id, "index": e.index, "condition": e.condition.key, "seed": e.seed, "success": e.success,
             "status": e.status, "frames": e.num_frames,
             "quality_score": e.quality.score if e.quality else None,
             "flags": [f.code for f in e.quality.flags] if e.quality else []} for e in eps]
    return {"summary": f"{len(rows)} episodes", "ref_ids": [r["id"] for r in rows], "episodes": rows}


async def t_record(ctx, a, actor):
    p = await get_project(ctx, a.project_id)
    if p.mode != Mode.sim:
        raise PermissionError("Physical robot recording requires the local LeRobot bridge and operator approval; "
                              "agents cannot start it.")
    conds = [Condition.model_validate({"cube_region": c.cube_region, "lighting": c.lighting})
             for c in a.conditions for _ in range(c.count)]
    job = await recording_service.start_recording(ctx, p, conds, auto_accept=a.auto_accept, actor=actor)
    if a.recommendation_id:
        await recommendation_service.set_status(ctx, a.recommendation_id, "accepted", actor)
    j = await _wait(ctx, job.id, a.wait_s)
    return {"summary": f"Recording {len(conds)} sim demos (job {job.id}: {j['status']})", "ref_ids": [job.id], "job": j}


async def t_analyze(ctx, a, actor):
    job = await analysis_service.start_analysis(ctx, await get_project(ctx, a.project_id), a.use_vlm, actor=actor)
    j = await _wait(ctx, job.id, a.wait_s)
    out = {"summary": f"Analysis job {job.id}: {j['status']}", "ref_ids": [job.id], "job": j}
    if j["status"] == "succeeded":
        an = await analysis_service.latest_analysis(ctx, a.project_id)
        out["summary"] = an.summary if an else out["summary"]
        out["analysis_id"] = an.id if an else None
    return out


async def t_clusters(ctx, a, actor):
    cl = await analysis_service.current_clusters(ctx, a.project_id)
    return {"summary": f"{len(cl)} failure clusters", "ref_ids": [c.id for c in cl],
            "clusters": [c.model_dump(mode="json") for c in cl],
            "note": "source=vlm clusters are unverified hypotheses unless verified_by_human"}


async def t_recommend(ctx, a, actor):
    recs = await recommendation_service.recommend(ctx, await get_project(ctx, a.project_id), a.budget, actor)
    return {"summary": "; ".join(f"[{r.support}] {r.title}" for r in recs) or "No recommendations",
            "ref_ids": [r.id for r in recs], "recommendations": [r.model_dump(mode="json") for r in recs]}


async def t_dataset(ctx, a, actor):
    augs = list(a.augmentations) + (PRESETS.get(a.augmentation_preset, []) if a.augmentation_preset else [])
    job = await dataset_service.create_version(ctx, await get_project(ctx, a.project_id), CreateDatasetVersionRequest(
        name=a.name, augmentations=augs, eval_fraction=a.eval_fraction), actor)
    j = await _wait(ctx, job.id, a.wait_s)
    return {"summary": f"Dataset build job {job.id}: {j['status']}", "ref_ids": [job.id, *(
        [j["result_ref"]] if j.get("result_ref") else [])], "job": j}


async def t_propose(ctx, a, actor):
    tj = await training_service.propose(ctx, await get_project(ctx, a.project_id), a.dataset_version_id,
                                        a.policy_type, a.steps, actor)
    return {"summary": f"Proposed {tj.policy_type} training {tj.id} on {tj.backend}: "
                       f"~{tj.estimate.gpu_hours} GPU-h, ${tj.estimate.est_cost_usd}. Needs human approval.",
            "ref_ids": [tj.id], "training_job": tj.model_dump(mode="json"), "requires_approval": True}


async def t_launch_training(ctx, a, actor):
    tj = await training_service._get(ctx, a.training_job_id)
    if tj.status == JobStatus.proposed:
        tj = await training_service.approve(ctx, tj.id, f"{actor}-token", actor, a.approval_token)
    tj = await training_service.launch(ctx, tj.id, actor)
    return {"summary": f"{tj.id} is {tj.status} on {tj.backend}", "ref_ids": [tj.id],
            "training_job": tj.model_dump(mode="json")}


async def t_eval(ctx, a, actor):
    pol = PolicyRef(kind="scripted_expert") if a.policy == "scripted_expert" else PolicyRef(
        kind="checkpoint", checkpoint_id=a.policy)
    run = await simulation_runner.launch_evaluation(ctx, await get_project(ctx, a.project_id), LaunchEvaluationRequest(
        policy=pol, episodes_per_branch=a.episodes_per_branch, base_seed=a.base_seed), actor)
    if a.wait_s:
        deadline = asyncio.get_running_loop().time() + a.wait_s
        while asyncio.get_running_loop().time() < deadline:
            ev = await simulation_runner.get_evaluation(ctx, run.id)
            if ev and ev["run"].status in (JobStatus.succeeded, JobStatus.failed):
                break
            await asyncio.sleep(2)
    ev = await simulation_runner.get_evaluation(ctx, run.id)
    br = {b.spec.name: (f"{b.result.successes}/{b.result.n}" if b.result else str(b.status)) for b in ev["branches"]}
    j = ctx.jobs.get(run.job_id) if run.job_id else None
    return {"summary": f"Evaluation {run.id} ({run.policy_label}): {br}", "ref_ids": [run.id],
            "job": j.model_dump() if j else None,
            "evaluation": {"run": ev["run"].model_dump(mode="json"),
                           "branches": [b.model_dump(mode="json") for b in ev["branches"]]}}


async def t_compare(ctx, a, actor):
    cmp = await experiment_comparison.compare(ctx, a.project_id, a.evaluation_ids)
    return {"summary": cmp.summary, "ref_ids": cmp.evaluation_ids, "comparison": cmp.model_dump(mode="json")}


async def t_next(ctx, a, actor):
    st = await project_state(ctx, await get_project(ctx, a.project_id))
    return {"summary": "; ".join(n.action for n in st.next_actions), "ref_ids": [],
            "next_actions": [n.model_dump(mode="json") for n in st.next_actions]}


async def t_job(ctx, a, actor):
    j = ctx.jobs.get(a.job_id)
    if j is None:
        raise KeyError(a.job_id)
    return {"summary": f"{j.kind} {j.status} {j.progress:.0%} {j.message}", "ref_ids": [j.id], "job": j.model_dump()}


# ------------------------------------------------- autonomous demonstrations (agent-facing)

class Cell(BaseModel):
    cube_region: Literal["left", "center", "right"]
    lighting: Literal["dim", "normal", "bright"] = "normal"


class AgentArgs:
    class CreatePolicyProject(BaseModel):
        task_prompt: str = Field(description="e.g. 'Train an SO-101 policy to pick up a cube and place it in a bowl'")
        num_demos: int = Field(20, ge=1, le=50, description="autonomous demonstration attempts to run")
        start_collection: bool = True
        diversity: Literal["plan", "medium", "high"] = "medium"
        model_budget_usd: float | None = Field(None, gt=0, description="planner spend cap; can only lower the "
                                                                       "server default, never raise it")

    class StartCollection(BaseModel):
        project_id: str
        num_demos: int = Field(6, ge=1, le=50)
        diversity: Literal["plan", "medium", "high"] = "medium"
        conditions: list[Cell] | None = Field(None, description="target these cells only (e.g. a weak condition)")
        collector: Collector = "agent"
        max_total_usd: float | None = Field(None, gt=0)
        mode: Literal["camera", "privileged"] | None = None

    class CollectionRef(BaseModel):
        collection_id: str

    class DemoResults(BaseModel):
        project_id: str
        collection_id: str | None = None

    class TrainFromAgent(BaseModel):
        project_id: str
        sources: list[Literal["sim_agent", "sim_scripted"]] = ["sim_agent"]
        augmentation_preset: Literal["lighting_robustness"] | None = None
        policy_type: Literal["act", "smolvla"] = "act"

    class EvalLearned(BaseModel):
        project_id: str
        checkpoint_id: str = Field(pattern=r"^ck_", description="a trained checkpoint (not the scripted oracle)")
        episodes_per_branch: int = Field(10, ge=1, le=50)

    class Improve(BaseModel):
        project_id: str
        max_new_demos: int = Field(5, ge=1, le=20)
        max_usd: float | None = Field(None, gt=0)


async def _await_job(ctx: Ctx, job_id: str | None) -> Any:
    task = ctx.jobs.tasks.get(job_id or "")
    if task:
        await asyncio.shield(task)
    j = ctx.jobs.get(job_id or "")
    if not j or j.status != "succeeded":
        raise RuntimeError(f"job {job_id} {j.status if j else 'missing'}: {j.error if j else ''}")
    return j


async def t_create_policy_project(ctx, a, actor):
    from ..models import AutopilotPolicy
    from . import agent_collection_service as acs

    default = AutopilotPolicy().model_budget_usd
    budget = min(a.model_budget_usd or default, default) if actor != "gui" else (a.model_budget_usd or default)
    p = await create_project(ctx, CreateProjectRequest(task_prompt=a.task_prompt, autopilot=AutopilotPolicy(
        collector="agent", model_budget_usd=budget)), actor)
    out: dict[str, Any] = {"summary": f"Created {p.id} '{p.name}' (planner budget ${budget:.2f})", "ref_ids": [p.id],
                           "project_id": p.id, "spec": p.spec.model_dump(), "planner": await acs.planner_status(ctx)}
    if a.start_collection:
        try:
            col = await acs.start_collection(ctx, p, StartAgentCollectionRequest(
                num_demos=a.num_demos, diversity=a.diversity, reason="initial collection"), actor)
            out.update(collection_id=col.id, job_id=col.job_id)
            out["summary"] += f"; started {col.id} ({a.num_demos} agent attempts with {col.model}, job {col.job_id})"
        except (PermissionError, acs.PlannerUnavailable) as e:
            out["collection_not_started"] = str(e)
    return out


async def t_start_collection(ctx, a, actor):
    from . import agent_collection_service as acs

    p = await get_project(ctx, a.project_id)
    col = await acs.start_collection(ctx, p, StartAgentCollectionRequest(
        num_demos=a.num_demos, diversity=a.diversity, collector=a.collector, max_total_usd=a.max_total_usd,
        mode=a.mode, conditions=[Condition(**c.model_dump()) for c in a.conditions] if a.conditions else None), actor)
    return {"summary": f"{col.id} started: {a.num_demos} {a.collector} attempts via {col.provider or 'scripted oracle'}"
                       f" (budget cap ${col.totals.get('budget_cap_usd', 0):.2f})",
            "ref_ids": [col.id, col.job_id], "collection_id": col.id, "job_id": col.job_id,
            "scenarios": col.scenarios}


async def t_progress(ctx, a, actor):
    from . import agent_collection_service as acs

    col = await acs.get_collection(ctx, a.collection_id)
    j = ctx.jobs.get(col.job_id or "")
    t = col.totals
    return {"summary": f"{col.status}: {t.get('successes', 0)} successful / {t.get('attempts', 0)} attempts of "
                       f"{len(col.scenarios)}, {t.get('model_calls', 0)} model calls, ${t.get('cost_usd', 0):.2f}",
            "ref_ids": [col.id], "collection": col.model_dump(mode="json"), "job": j.model_dump() if j else None}


async def t_results(ctx, a, actor):
    from . import agent_collection_service as acs

    rows = await acs.list_attempts(ctx, a.project_id, a.collection_id)
    return {"summary": f"{sum(r.status == 'success' for r in rows)}/{len(rows)} attempts succeeded "
                       "(successes are demonstrations; failures are kept as diagnostics only)",
            "ref_ids": [r.id for r in rows],
            "attempts": [{"id": r.id, "collector": r.collector, "status": r.status, "episode_id": r.episode_id,
                          "seed": r.seed, "scenario": r.scenario.model_dump(), "recoveries": r.recoveries,
                          "model_calls": r.usage.get("calls"), "cost_usd": r.usage.get("cost_usd"),
                          "provider": r.provider, "model": r.model, "mode": r.mode, "stop_reason": r.stop_reason,
                          "error": r.error} for r in rows],
            "by_collector": acs.compare_collectors(rows)}


async def t_train_from_agent(ctx, a, actor):
    from ..models import EpisodeSource

    p = await get_project(ctx, a.project_id)
    augs = PRESETS.get(a.augmentation_preset, []) if a.augmentation_preset else []

    async def chain(job):
        ds_job = await dataset_service.create_version(ctx, p, CreateDatasetVersionRequest(
            name=f"{'+'.join(a.sources)}-demos", sources=[EpisodeSource(s) for s in a.sources], augmentations=augs),
            actor)
        ctx.jobs.update(job, message=f"building dataset ({ds_job.id})")
        ds = await _await_job(ctx, ds_job.id)
        tj = await training_service.propose(ctx, p, ds.result_ref, a.policy_type, actor=actor)
        ctx.jobs.update(job, message=f"proposed {tj.id}; waiting for human approval")
        return tj.id

    job = ctx.jobs.submit("train_from_agent", p.id, chain)
    return {"summary": f"Building a dataset from {a.sources} demos then proposing {a.policy_type.upper()} training "
                       f"(job {job.id}). The proposal needs human approval (or the project's standing approval) "
                       "before it runs.", "ref_ids": [job.id], "job_id": job.id, "requires_approval": True}


async def t_eval_learned(ctx, a, actor):
    return await t_eval(ctx, Args.LaunchEval(project_id=a.project_id, policy=a.checkpoint_id,
                                             episodes_per_branch=a.episodes_per_branch), actor)


async def t_improve(ctx, a, actor):
    from . import agent_collection_service as acs

    p = await get_project(ctx, a.project_id)
    evals = [e for e in await simulation_runner.list_evaluations(ctx, p.id) if e["run"].policy.kind == "checkpoint"]
    target, evidence = None, ""
    if evals:
        scored = [(b.result.success_rate, b) for b in evals[-1]["branches"] if b.result]
        if scored:
            rate, b = min(scored, key=lambda t: t[0])
            region = recommendation_service.region_of(b.spec.condition.cube_region, b.spec.cube_offset)
            target = Condition(cube_region=region, lighting=b.spec.condition.lighting)  # type: ignore[arg-type]
            evidence = (f"weakest branch of {evals[-1]['run'].id}: {b.spec.name} {b.result.successes}/{b.result.n} "
                        f"(measured) -> cell {target.key}")
    if target is None:
        recs = [r for r in await recommendation_service.recommend(ctx, p, actor=actor) if r.action == "collect"]
        if not recs:
            return {"summary": "Nothing to improve: no learned-policy evaluation and no coverage gap.", "ref_ids": []}
        target, evidence = recs[0].condition, f"top recommendation ({recs[0].support}): {recs[0].title}"
    col = await acs.start_collection(ctx, p, StartAgentCollectionRequest(
        num_demos=a.max_new_demos, conditions=[target] * a.max_new_demos, max_total_usd=a.max_usd,
        reason=evidence), actor)

    async def chain(job):
        ctx.jobs.update(job, message=f"collecting targeted demos ({col.id})")
        await _await_job(ctx, col.job_id)
        ds_job = await dataset_service.create_version(ctx, p, CreateDatasetVersionRequest(name="improve"), actor)
        ds = await _await_job(ctx, ds_job.id)
        tj = await training_service.propose(ctx, p, ds.result_ref, "act", actor=actor)
        ctx.jobs.update(job, message=f"proposed retraining {tj.id}; needs approval")
        return tj.id

    job = ctx.jobs.submit("improve_policy", p.id, chain)
    return {"summary": f"Targeting {target.key}: {evidence}. Collecting {a.max_new_demos} new agent demos in fresh "
                       f"seeds ({col.id}), then a new dataset version and a retraining proposal (job {job.id}).",
            "ref_ids": [col.id, job.id], "collection_id": col.id, "job_id": job.id, "target": target.model_dump(),
            "evidence": evidence, "requires_approval_for_training": True}


# ------------------------------------------------- robot sessions (the user's agent drives the arm)

RobotToolName = Literal["observe_scene", "get_robot_state", "propose_grasp", "move_end_effector", "set_gripper",
                        "verify_grasp", "place_object", "check_task_success"]


class RobotArgs:
    class Start(BaseModel):
        project_id: str
        cube_region: Literal["left", "center", "right"] | None = None
        lighting: Literal["dim", "normal", "bright"] | None = None
        seed: int | None = None

    class Act(BaseModel):
        session_id: str
        tool: RobotToolName
        args: dict[str, Any] = {}

    class Finish(BaseModel):
        session_id: str
        believes_success: bool
        summary: str = Field("", max_length=400)


async def t_robot_start(ctx, a, actor):
    from . import robot_session_service as rs

    out = await rs.start(ctx, await get_project(ctx, a.project_id), a.cube_region, a.lighting, a.seed, actor=actor)
    return {"summary": f"Episode ready: session {out['session_id']}, cube {out['scenario']['cube_region']}, "
                       f"{out['scenario']['lighting']} light, seed {out['seed']}. Start with observe_scene.",
            "ref_ids": [out["session_id"], out["attempt_id"]], **out}


async def t_robot_act(ctx, a, actor):
    from . import robot_session_service as rs

    out = await rs.act(ctx, a.session_id, a.tool, a.args)
    r = out["result"]
    summary = (f"{out['phase']}: ok" if r.get("ok") else
               f"{out['phase']}: rejected ({r.get('error', {}).get('code')}: {r.get('error', {}).get('message')})")
    return {"summary": summary, "ref_ids": [a.session_id], **out}


async def t_robot_finish(ctx, a, actor):
    from . import robot_session_service as rs

    out = await rs.finish(ctx, a.session_id, a.believes_success, a.summary, actor)
    verdict = "SUCCESS" if out["success"] else "FAILED"
    return {"summary": f"{verdict} according to the simulator ({out['frames_recorded']} frames recorded)"
                       + (f"; saved as demonstration {out['episode_id']}" if out["episode_id"] else ""),
            "ref_ids": [x for x in (out["attempt_id"], out["episode_id"]) if x], **out}


class ImportArgs(BaseModel):
    project_id: str
    repo_id: str = Field(description="Hugging Face dataset repo id, e.g. 'lerobot/svla_so101_pickplace' or the "
                                     "user's own 'user/my_so101_demos'")
    max_episodes: int = Field(10, ge=1, le=50)


class InfraArgs(BaseModel):
    project_id: str


class SetInfraArgs(BaseModel):
    project_id: str
    planner: Literal["auto", "claude_code", "model"] | None = None
    training_backend: Literal["auto", "futurama_slurm", "manual"] | None = None
    training_gres: Literal["gpu:l40s:1", "gpu:l40:1", "gpu:l4:1", "gpu:h200:1"] | None = None
    training_walltime_min: int | None = Field(None, ge=15, le=24 * 60)


def _infra_summary(out: dict[str, Any]) -> str:
    c = out["choice"]
    lines = [f"Chosen: planner {c['planner']}, training {c['training_backend']} on {c['training_gres']} for "
             f"{c['training_walltime_min']} min."]
    for group in ("planner", "training_backend", "evaluation", "robot"):
        for o in out[group]:
            lines.append(f"- {group} / {o['label']}: " + ("available" if o["available"] else f"unavailable ({o['reason']})"))
    return "\n".join(lines)


async def t_get_infra(ctx, a, actor):
    from . import infra_service

    out = await infra_service.options(ctx, await get_project(ctx, a.project_id))
    return {"summary": _infra_summary(out), "ref_ids": [], **out}


async def t_set_infra(ctx, a, actor):
    from ..models import InfraChoice
    from . import infra_service

    p = await get_project(ctx, a.project_id)
    update = {k: v for k, v in a.model_dump().items() if k != "project_id" and v is not None}
    await infra_service.set_choice(ctx, p, InfraChoice.model_validate({**p.infra.model_dump(), **update}), actor)
    out = await infra_service.options(ctx, p)
    return {"summary": "Saved. " + _infra_summary(out), "ref_ids": [], **out}


async def t_import(ctx, a, actor):
    from . import import_service

    job = await import_service.start_import(ctx, await get_project(ctx, a.project_id), {"hub": a.repo_id},
                                            a.max_episodes, actor)
    return {"summary": f"Importing up to {a.max_episodes} episodes from {a.repo_id} (job {job.id}). They are kept "
                       "separate from simulation demos and can be trained on directly.",
            "ref_ids": [job.id], "job_id": job.id}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("create_project", "Create a robot-learning project from a natural-language task; returns task spec, "
         "success criteria and an initial demonstration plan.", Args.CreateProject, t_create_project),
    Tool("get_project_state", "Counts, datasets, training jobs, evaluations, open recommendations and next actions.",
         Args.ProjectRef, t_state),
    Tool("list_episodes", "List recorded demonstrations with condition, seed, success and quality flags.",
         Args.ListEpisodes, t_list_episodes),
    Tool("record_demonstrations", "Record scripted-expert demonstrations in MuJoCo for the given conditions "
         "(simulation only).", Args.Record, t_record),
    Tool("analyze_dataset", "Run quality checks, telemetry failure analysis and VLM hypotheses; builds failure "
         "clusters.", Args.Analyze, t_analyze),
    Tool("get_failure_clusters", "Current failure clusters with source (quality/evaluation/vlm), episodes and "
         "uncertainty.", Args.ProjectRef, t_clusters),
    Tool("recommend_demonstrations", "Rank which demonstrations to collect next (and augment/train/evaluate "
         "suggestions) with evidence and support level.", Args.Recommend, t_recommend),
    Tool("create_dataset_version", "Snapshot accepted episodes as a versioned LeRobotDataset v3 with lineage, "
         "train/eval split and optional image-only augmentation.", Args.DatasetVersion, t_dataset),
    Tool("propose_training_job", "Propose an ACT/SmolVLA training job with config and resource estimate "
         "(does not start it).", Args.ProposeTraining, t_propose),
    Tool("launch_training_job", "Launch an approved training job. Refused unless a human approved it in the "
         "studio or a valid operator approval_token is given.", Args.LaunchTraining, t_launch_training,
         agent_callable=False),
    Tool("launch_simulation_evaluation", "Evaluate a policy in parallel MuJoCo branches: baseline, shifted object, "
         "dim lighting. Records seeds, successes, rewards and videos.", Args.LaunchEval, t_eval),
    Tool("compare_experiments", "Branch-by-branch comparison of evaluation runs with deltas and p-values.",
         Args.Compare, t_compare),
    Tool("get_next_actions", "What to do next for this project, and why.", Args.ProjectRef, t_next),
    Tool("get_job", "Status/progress of a background job.", Args.JobRef, t_job),
    Tool("create_policy_project", "Create a project for a manipulation policy and (by default) start autonomous "
         "agent demonstration collection (our self-hosted VLM, or GPT-6 Astra) within the planner budget.",
         AgentArgs.CreatePolicyProject, t_create_policy_project),
    Tool("start_agent_collection", "Generate demonstrations autonomously: the agent planner (self-hosted VLM or GPT-6 Astra) plans from camera images, "
         "validated IK executes in MuJoCo, simulator-verified successes become LeRobot episodes. Returns job_id.",
         AgentArgs.StartCollection, t_start_collection),
    Tool("get_collection_progress", "Status, successes, failures, model calls and spend of a collection.",
         AgentArgs.CollectionRef, t_progress),
    Tool("get_demonstration_results", "Per-attempt outcomes with provenance (agent planner model vs scripted oracle), usage and "
         "recoveries, plus per-collector metrics.", AgentArgs.DemoResults, t_results),
    Tool("train_from_agent_demonstrations", "Build a dataset version from agent-generated demos and propose ACT "
         "training (training itself needs approval). Returns job_id.", AgentArgs.TrainFromAgent, t_train_from_agent),
    Tool("evaluate_learned_policy", "Evaluate a trained checkpoint (never the oracle) in the three parallel MuJoCo "
         "conditions.", AgentArgs.EvalLearned, t_eval_learned),
    Tool("import_demonstrations", "Import existing human/robot demonstrations from a LeRobot dataset on the Hugging "
         "Face Hub (the user's own SO-101 recordings or public ones). Returns job_id.", ImportArgs, t_import),
    Tool("get_infrastructure", "Where this project's stages run: planner, training backend (Futurama GPU or manual), "
         "GPU and walltime, evaluation and robot, each with live availability and the reason when unavailable.",
         InfraArgs, t_get_infra),
    Tool("set_infrastructure", "Change this project's infrastructure choices (planner, training backend, GPU, "
         "walltime). Only stores the choice: training still needs the human's approval.", SetInfraArgs, t_set_infra),
    Tool("robot_start_episode", "Start a simulated SO-101 episode you will control yourself (fresh seed; optional cube "
         "region/lighting). Returns session_id.", RobotArgs.Start, t_robot_start, agent_callable=False),
    Tool("robot_act", "Run one validated robot tool in your session (observe_scene, get_robot_state, propose_grasp, "
         "move_end_effector, set_gripper, verify_grasp, place_object, check_task_success).", RobotArgs.Act,
         t_robot_act, agent_callable=False),
    Tool("robot_finish_episode", "End your episode; the simulator judges success and successes become demonstrations.",
         RobotArgs.Finish, t_robot_finish, agent_callable=False),
    Tool("improve_policy", "Find the weakest measured condition, collect targeted agent demos there in fresh seeds, "
         "build a new dataset version and propose retraining. Returns job_id.", AgentArgs.Improve, t_improve),
]}

# ------------------------------------------------------------------ intake (spec v2)

class IntakeArgs:
    class Start(BaseModel):
        prompt: str = Field("", description="The task in plain language, e.g. 'pick up the water bottle and put it "
                                            "on the coaster'. Objects named here are pre-filled from a catalogue.")
        mode: Mode = Field(Mode.sim, description="'sim' (the MVP runs everything in physics sim) or 'real'")

    class Update(BaseModel):
        project_id: str
        patch: dict[str, Any] = Field(description="Partial intake spec to merge. Each open question carries a "
                                                  "ready-to-send patch; lists of objects/zones/cameras/obstacles merge "
                                                  "by name.")

    class Frame(BaseModel):
        project_id: str
        camera: Literal["top", "front", "wrist_cam"] = "top"
        seed: int = Field(0, description="scene randomization seed")


async def t_start_intake(ctx, a, actor):
    from . import intake_service

    p, st = await intake_service.start_intake(ctx, a.prompt, a.mode.value, actor)
    n = sum(1 for q in st["questions"] if q["required"])
    return {"summary": f"Project {p.id}: intake started, {n} questions open", "ref_ids": [p.id], "project_id": p.id,
            **st}


async def t_update_intake(ctx, a, actor):
    from . import intake_service

    st = await intake_service.update_intake(ctx, await get_project(ctx, a.project_id), a.patch, actor)
    n = sum(1 for q in st["questions"] if q["required"])
    done = "complete; call build_simulation" if st["complete"] else f"{n} questions open"
    return {"summary": f"Intake {done}. Changed: {', '.join(st['changed']) or 'nothing'}", "ref_ids": [a.project_id],
            **st}


async def t_intake_status(ctx, a, actor):
    from ..intake import status

    p = await get_project(ctx, a.project_id)
    if p.intake is None:
        raise ValueError("this project has no intake; call start_intake")
    from .intake_service import scene_stale

    st = status(p.intake).model_dump(mode="json")
    st["scene_built"], st["scene_stale"] = p.scene is not None, scene_stale(p)
    st["scene_digest"] = (p.scene or {}).get("digest")
    return {"summary": "complete" if st["complete"] else "incomplete", "ref_ids": [p.id], **st}


async def t_build_sim(ctx, a, actor):
    from . import intake_service

    out = await intake_service.build_simulation(ctx, await get_project(ctx, a.project_id), actor)
    f = out["feasibility"]
    return {"summary": f"Scene {out['scene']['digest']} built; {f['summary']}", "ref_ids": [a.project_id],
            "image_b64": out["preview"].pop("image_b64", None), **out}


async def t_capture_frame(ctx, a, actor):
    from . import intake_service

    out = await intake_service.capture_frame(ctx, await get_project(ctx, a.project_id), a.camera, a.seed)
    return {"summary": f"{a.camera} camera, scene {out['scene']} (sim render)", "ref_ids": [a.project_id], **out}


TOOLS.update({t.name: t for t in [
    Tool("start_intake", "Start describing a robot-learning task: creates a draft project and returns typed "
         "questions (each with a ready-to-send patch) about the arm, cameras, objects, place zones, environment and "
         "task.", IntakeArgs.Start, t_start_intake),
    Tool("update_intake", "Answer intake questions by merging a partial spec. Returns the merged spec, what changed, "
         "what is still assumed, and the remaining questions.", IntakeArgs.Update, t_update_intake),
    Tool("get_intake_status", "The intake spec, open questions, and whether the sim scene is built.",
         Args.ProjectRef, t_intake_status),
    Tool("build_simulation", "Build the MuJoCo scene from the completed intake, then check feasibility (jaw opening, "
         "payload torque, reach to every start and zone, with alternatives). Returns a top-camera preview.",
         Args.ProjectRef, t_build_sim),
    Tool("capture_frame", "Render a scene camera (top, front, wrist_cam) so you can check the sim matches the "
         "real setup you described.", IntakeArgs.Frame, t_capture_frame),
]})

# Tools that start work the agent should not start without a human click.
NEEDS_GUI_APPROVAL = {"launch_simulation_evaluation", "propose_training_job"}


class NotAgentCallable(PermissionError):
    """The studio agent asked for a tool reserved for humans or MCP clients. No approval card fixes it."""

    code = "not_agent_callable"
    approval_required = False


async def call_tool(ctx: Ctx, name: str, raw_args: dict[str, Any], actor: str) -> dict[str, Any]:
    tool = TOOLS.get(name)
    if tool is None:
        raise KeyError(f"unknown tool {name}")
    pid = raw_args.get("project_id")
    if actor == "agent" and not tool.agent_callable:
        await ctx.audit(actor, name, pid, raw_args, "refused", "not callable by the studio agent")
        raise NotAgentCallable(f"{name} is not available to the studio agent; a human runs it from the studio")
    args = tool.args.model_validate(raw_args)
    try:
        result = await tool.fn(ctx, args, actor)
    except Exception as e:
        await ctx.audit(actor, name, pid, raw_args, "error", f"{type(e).__name__}: {e}"[:300])
        raise
    if name in ("get_project_state", "list_episodes", "get_failure_clusters", "get_next_actions", "get_job",
                "compare_experiments", "get_intake_status", "capture_frame"):
        await ctx.audit(actor, name, pid, raw_args, "ok", result.get("summary", "")[:300])
    return result
