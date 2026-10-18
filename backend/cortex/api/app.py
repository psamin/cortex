"""FastAPI app: REST for the studio, SSE/MJPEG for live views, and
POST /api/tools/{name} for the MCP server (same domain services)."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile

from ..config import settings
from ..context import Ctx, build_ctx
from ..errors import NotFound, describe
from ..models import (
    Analysis,
    Checkpoint,
    Condition,
    CreateDatasetVersionRequest,
    CreateProjectRequest,
    Episode,
    EpisodeStatus,
    LaunchEvaluationRequest,
    Project,
    StartAgentCollectionRequest,
    TrainingJob,
)
from ..services import (
    agent,
    analysis_service,
    autopilot,
    dataset_service,
    experiment_comparison,
    recommendation_service,
    recording_service,
    simulation_runner,
    training_service,
)
from ..services import agent_orchestrator as orch

log = logging.getLogger("cortex")
CTX: Ctx | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CTX
    logging.basicConfig(level=logging.INFO)
    CTX = await build_ctx(settings())
    for n in CTX.notes:
        log.warning(n)
    await training_service.resume_pollers(CTX)
    agent.install_job_listener(CTX)
    # no job survives a restart (JobManager marks them all interrupted), so neither does any collection
    from ..models import AgentCollection
    for status in ("running", "queued"):
        for col in await CTX.repo.list("agent_collections", AgentCollection, status=status):
            col.status, col.error = "failed", "interrupted by a backend restart; recorded attempts are kept"
            await CTX.repo.save("agent_collections", col)
    if swept := await simulation_runner.sweep_interrupted(CTX):
        log.info("marked %d interrupted evaluation(s) failed", swept)
    resumed = await autopilot.resume_all(CTX)
    if resumed:
        log.info("resumed %d autopilot run(s)", resumed)
    app.mount("/files", StaticFiles(directory=CTX.data_dir, check_dir=False), name="files")
    yield
    from ..services import live_policy_service

    live_policy_service.close_tunnel()  # a tunnel left running would hold its local port against the next one
    await CTX.jobs.shutdown()
    from ..services import intake_service

    intake_service.shutdown()  # GL contexts must be released on the thread that made them


app = FastAPI(title="Cortex", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings().cors_origins.split(","), allow_methods=["*"],
                   allow_headers=["*"])


def ctx() -> Ctx:
    assert CTX is not None
    return CTX


# ------------------------------------------------------------------ guard
LOCAL = {"127.0.0.1", "::1", "localhost", "testclient"}


@app.middleware("http")
async def guard(request: Request, call_next):
    """When the backend is tunneled for a hosted MCP server, only the tool
    endpoint (with the MCP key) and health are reachable from outside."""
    path = request.url.path
    remote = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    is_local = request.client is not None and request.client.host in LOCAL and not remote
    if not is_local and path not in ("/api/health",):
        key = settings().cortex_mcp_key
        presented = request.headers.get("authorization") or ""
        ok = path.startswith("/api/tools") and bool(key) and hmac.compare_digest(presented, f"Bearer {key}")
        if not ok:
            return JSONResponse({"detail": "Remote access is limited to /api/tools with the MCP key"}, 403)
    return await call_next(request)


@app.exception_handler(NotFound)  # a plain KeyError is a bug and falls through to _500
async def _404(_, e: NotFound):
    return JSONResponse({"detail": f"not found: {e.args[0] if e.args else e}"}, 404)


@app.exception_handler(PermissionError)
async def _403(_, e: PermissionError):
    return JSONResponse({"detail": str(e), "code": getattr(e, "code", "approval_required"),
                         "approval_required": getattr(e, "approval_required", True)}, 403)


@app.exception_handler(RuntimeError)
async def _503(_, e: RuntimeError):
    from ..services.agent_collection_service import PlannerUnavailable
    if isinstance(e, PlannerUnavailable):
        return JSONResponse({"detail": str(e), "code": e.code, "planner_unavailable": True}, 503)
    return _internal(e)


def _internal(e: Exception) -> JSONResponse:
    """People get a sentence; the raw error goes to `error_detail` and the log, never the headline."""
    msg, detail = describe(e)
    if detail is not None:
        log.error("request failed: %s", detail, exc_info=e)
    return JSONResponse({"detail": msg, "error_detail": detail, "code": "internal_error" if detail else None},
                        500 if detail else 400)


@app.exception_handler(Exception)
async def _500(_, e: Exception):
    return _internal(e)


@app.exception_handler(ValueError)
async def _400(_, e: ValueError):
    if isinstance(e, ValidationError) or describe(e)[1] is None:  # bad input, or raised on purpose by Cortex
        return JSONResponse({"detail": str(e)}, 400)
    return _internal(e)


@app.exception_handler(FileNotFoundError)
async def _409(_, e: FileNotFoundError):
    return JSONResponse({"detail": str(e)}, 409)


def actor(request: Request) -> str:
    from ..services.agent_obs import has_agent_headers

    if has_agent_headers(request.headers):  # an agent key or session is never a human click (#105)
        return "mcp"
    a = request.headers.get("x-cortex-actor", "gui")
    return a if a in ("gui", "agent", "mcp") else "gui"


# ------------------------------------------------------------------ health
@app.get("/api/health")
async def health():
    c = ctx()
    s = c.settings
    try:
        from importlib.metadata import version
        lr = version("lerobot")
    except Exception:
        lr = None
    from ..sim.so101 import ENVHUB_URI
    return {
        "db": {"provider": c.store.name, "fallback": not s.insforge_enabled},
        "storage": {"provider": c.blobs.name, "fallback": not s.insforge_enabled},
        "vlm": {"provider": c.vlm_source, "model": c.model.vlm_model if c.model else None,
                "fallback": not c.model},
        "agent": {"provider": c.model.label if c.model else "rule-based planner",
                  "model": c.model.agent_model if c.model else None, "fallback": not c.model},
        "executor": {"provider": "parallel local worker processes", "fallback": False},
        "training": {"provider": f"futurama slurm ({s.futurama_ssh_host})" if s.futurama_ssh_host
                     else "manual command bundle", "fallback": not s.futurama_ssh_host},
        "lerobot_version": lr, "sim_env": ENVHUB_URI, "notes": c.notes,
    }


# ------------------------------------------------------------------ projects
@app.post("/api/projects")
async def create_project(req: CreateProjectRequest, request: Request) -> Project:
    p = await orch.create_project(ctx(), req, actor(request))
    if req.start_autopilot:
        await autopilot.start(ctx(), p)
    return p


@app.get("/api/projects")
async def list_projects(history: bool = False) -> list[Project]:
    """Newest first. `history=true`: only projects with history (Home); empty ones past their grace period are
    deleted on the way (services/project_cleanup.py)."""
    from ..services import project_cleanup
    projects = await project_cleanup.listed(ctx())
    return await project_cleanup.with_history(ctx(), projects) if history else projects


@app.get("/api/projects/{pid}")
async def get_project(pid: str) -> Project:
    return await orch.get_project(ctx(), pid)


@app.get("/api/projects/{pid}/state")
async def state(pid: str):
    return await orch.project_state(ctx(), await orch.get_project(ctx(), pid))


# ------------------------------------------------------------------ intake (spec v2)

class IntakeStartBody(BaseModel):
    prompt: str = ""
    mode: str = "sim"


@app.post("/api/intake")
async def intake_start(body: IntakeStartBody, request: Request):
    from ..services import intake_service

    p, st = await intake_service.start_intake(ctx(), body.prompt, body.mode, actor(request))
    return {"project_id": p.id, **st}


@app.get("/api/projects/{pid}/intake")
async def intake_get(pid: str):
    return await orch.call_tool(ctx(), "get_intake_status", {"project_id": pid}, "gui")


@app.patch("/api/projects/{pid}/intake")
async def intake_patch(pid: str, patch: dict[str, Any], request: Request):
    from ..services import intake_service

    return await intake_service.update_intake(ctx(), await orch.get_project(ctx(), pid), patch, actor(request))


@app.post("/api/projects/{pid}/simulation/build")
async def simulation_build(pid: str, request: Request):
    from ..services import intake_service

    out = await intake_service.build_simulation(ctx(), await orch.get_project(ctx(), pid), actor(request))
    out["preview"].pop("image_b64", None)
    return out


@app.get("/api/projects/{pid}/simulation/frame")
async def simulation_frame(pid: str, camera: str = "top", seed: int = 0):
    import base64

    from ..services import intake_service

    out = await intake_service.capture_frame(ctx(), await orch.get_project(ctx(), pid), camera, seed)
    return Response(base64.b64decode(out["image_b64"]), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{pid}/events")
async def events(pid: str, request: Request):
    async def gen():
        yield "event: hello\ndata: {}\n\n"
        q = ctx().bus.subscribe(pid)
        try:
            while True:
                try:
                    ev, data = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: {ev}\ndata: {json.dumps(data, default=str)}\n\n"
                except TimeoutError:
                    if await request.is_disconnected():
                        return
                    yield ": ping\n\n"
        finally:
            ctx().bus.unsubscribe(pid, q)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/projects/{pid}/jobs")
async def jobs(pid: str):
    return [j.model_dump() for j in ctx().jobs.list(pid)]


@app.get("/api/jobs/{job_id}")
async def job(job_id: str):
    j = ctx().jobs.get(job_id)
    if not j:
        raise NotFound(job_id)
    return j.model_dump()


@app.get("/api/projects/{pid}/audit")
async def audit(pid: str):
    rows = await ctx().store.list("tool_audit_logs", project_id=pid)
    return [{**r["data"], "status": r["status"]} for r in reversed(rows)][:300]


# ------------------------------------------------------------------ collect
_render_lock = threading.Lock()
_preview_env: dict[str, Any] = {}


def _render_png(region: str, lighting: str, seed: int) -> bytes:
    from PIL import Image

    from ..sim.so101 import PickPlaceEnv, Variation
    with _render_lock:
        from ..sim.viewer import DemoCamera
        env = _preview_env.get("env")
        if env is None:
            env = _preview_env["env"] = PickPlaceEnv()
        env.reset(seed, Variation(cube_region=region, lighting=lighting))
        cam = _preview_env.get("cam")
        if cam is None:
            cam = _preview_env["cam"] = DemoCamera(env.world.model)
        img = cam.render(env.world.data)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


@app.get("/api/projects/{pid}/sim/frame")
async def sim_frame(pid: str, cube_region: str = "center", lighting: str = "normal", seed: int = 0):
    from ..services import intake_service

    png = await intake_service.on_gl(_render_png, cube_region, lighting, seed)  # GL contexts are thread-bound
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/projects/{pid}/sim/live")
async def sim_live(pid: str, request: Request):
    path = ctx().local(f"{recording_service.live_key(pid)}/live.jpg")

    async def gen():
        last = 0.0
        idle = 0
        while not await request.is_disconnected():
            try:
                m = path.stat().st_mtime
                if m != last:
                    last, idle = m, 0
                    data = path.read_bytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                           str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            except FileNotFoundError:
                pass
            idle += 1
            if idle > 20 * 60 * 5:
                return
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


class RecordReq(BaseModel):
    condition: Condition
    seed: int | None = None
    auto_accept: bool = False


class RecordBatchReq(BaseModel):
    conditions: list[Condition]
    per_condition: int = 1
    base_seed: int | None = None
    auto_accept: bool = True


@app.post("/api/projects/{pid}/recordings")
async def record(pid: str, req: RecordReq, request: Request):
    p = await orch.get_project(ctx(), pid)
    job = await recording_service.start_recording(ctx(), p, [req.condition],
                                                  [req.seed] if req.seed is not None else None,
                                                  req.auto_accept, actor(request))
    return job.model_dump()


@app.post("/api/projects/{pid}/recordings/batch")
async def record_batch(pid: str, req: RecordBatchReq, request: Request):
    p = await orch.get_project(ctx(), pid)
    conds = [c for c in req.conditions for _ in range(req.per_condition)]
    seeds = [req.base_seed + i for i in range(len(conds))] if req.base_seed is not None else None
    job = await recording_service.start_recording(ctx(), p, conds, seeds, req.auto_accept, actor(request))
    return job.model_dump()


@app.post("/api/recordings/{job_id}/stop")
async def stop_recording(job_id: str):
    j = await ctx().jobs.cancel(job_id)
    if not j:
        raise NotFound(job_id)
    return j.model_dump()


@app.get("/api/projects/{pid}/episodes")
async def episodes(pid: str) -> list[Episode]:
    return await recording_service.list_episodes(ctx(), pid)


class StatusReq(BaseModel):
    status: str


@app.patch("/api/episodes/{eid}")
async def patch_episode(eid: str, req: StatusReq, request: Request) -> Episode:
    return await recording_service.set_status(ctx(), eid, EpisodeStatus(req.status), actor(request))


@app.post("/api/projects/{pid}/episodes/accept-passing")
async def accept_passing(pid: str, request: Request):
    n = 0
    for e in await recording_service.list_episodes(ctx(), pid):
        if e.status == EpisodeStatus.recorded and e.success and e.quality and \
                e.quality.score >= recording_service.AUTO_ACCEPT_MIN_SCORE:
            await recording_service.set_status(ctx(), e.id, EpisodeStatus.accepted, actor(request))
            n += 1
    return {"accepted": n}


@app.get("/api/episodes/{eid}/media")
async def episode_media(eid: str):
    e = await ctx().repo.get("episodes", eid, Episode)
    if not e:
        raise NotFound(eid)
    return {"video_url": await ctx().media_url(e.artifacts["front"]) if "front" in e.artifacts else None,
            "thumb_url": await ctx().media_url(e.artifacts["thumb"]) if "thumb" in e.artifacts else None,
            "frames": e.num_frames}


@app.get("/api/media")
async def media(key: str):
    return {"url": await ctx().media_url(key)}


# ------------------------------------------------------------------ analyze
class AnalyzeReq(BaseModel):
    use_vlm: bool = True
    max_vlm_episodes: int = 6


@app.post("/api/projects/{pid}/analyses")
async def analyze(pid: str, req: AnalyzeReq, request: Request):
    job = await analysis_service.start_analysis(ctx(), await orch.get_project(ctx(), pid), req.use_vlm,
                                                req.max_vlm_episodes, actor(request))
    return job.model_dump()


@app.get("/api/projects/{pid}/analyses/latest")
async def latest_analysis(pid: str) -> Analysis | None:
    return await analysis_service.latest_analysis(ctx(), pid)


@app.get("/api/analyses/{aid}")
async def get_analysis(aid: str) -> Analysis:
    a = await ctx().repo.get("analyses", aid, Analysis)
    if not a:
        raise NotFound(aid)
    return a


@app.get("/api/projects/{pid}/failure-clusters")
async def clusters(pid: str):
    return await analysis_service.current_clusters(ctx(), pid)


class LabelReq(BaseModel):
    status: str
    human_label: str | None = None


@app.post("/api/analyses/{aid}/diagnoses/{ref:path}")
async def label(aid: str, ref: str, req: LabelReq, request: Request):
    return await analysis_service.label_diagnosis(ctx(), aid, ref, req.status, req.human_label, actor(request))


# ------------------------------------------------------------------ improve
class RecReq(BaseModel):
    budget: int | None = None


@app.post("/api/projects/{pid}/recommendations")
async def recommend(pid: str, req: RecReq, request: Request):
    return await recommendation_service.recommend(ctx(), await orch.get_project(ctx(), pid), req.budget,
                                                  actor(request))


@app.get("/api/projects/{pid}/recommendations")
async def open_recs(pid: str):
    return await recommendation_service.open_recommendations(ctx(), pid)


@app.patch("/api/recommendations/{rid}")
async def patch_rec(rid: str, req: StatusReq, request: Request):
    return await recommendation_service.set_status(ctx(), rid, req.status, actor(request))


@app.post("/api/projects/{pid}/datasets")
async def create_dataset(pid: str, req: CreateDatasetVersionRequest, request: Request):
    job = await dataset_service.create_version(ctx(), await orch.get_project(ctx(), pid), req, actor(request))
    return job.model_dump()


@app.get("/api/projects/{pid}/datasets")
async def datasets(pid: str):
    return await dataset_service.list_versions(ctx(), pid)


class ProposeReq(BaseModel):
    dataset_version_id: str
    policy_type: str = "act"
    steps: int | None = None
    base_checkpoint_id: str | None = None  # continue fine-tuning this policy


@app.post("/api/projects/{pid}/training/proposals")
async def propose(pid: str, req: ProposeReq, request: Request) -> TrainingJob:
    return await training_service.propose(ctx(), await orch.get_project(ctx(), pid), req.dataset_version_id,
                                          req.policy_type, req.steps, actor(request), req.base_checkpoint_id)


class ApproveReq(BaseModel):
    approver: str = "studio user"
    approval_token: str | None = None


@app.post("/api/training/{tid}/approve")
async def approve(tid: str, req: ApproveReq, request: Request) -> TrainingJob:
    return await training_service.approve(ctx(), tid, req.approver, actor(request), req.approval_token)


@app.post("/api/training/{tid}/launch")
async def launch(tid: str, request: Request) -> TrainingJob:
    return await training_service.launch(ctx(), tid, actor(request))


@app.get("/api/projects/{pid}/training")
async def training(pid: str) -> list[TrainingJob]:
    return await ctx().repo.list("training_jobs", TrainingJob, project_id=pid)


@app.get("/api/training/{tid}/command")
async def training_command(tid: str):
    tj = await training_service._get(ctx(), tid)
    return {"command": training_service.manual_command(ctx(), tj)}


class CheckpointReq(BaseModel):
    path: str
    training_job_id: str | None = None
    step: int | None = None


@app.post("/api/projects/{pid}/checkpoints")
async def register_checkpoint(pid: str, req: CheckpointReq, request: Request) -> Checkpoint:
    return await training_service.register_checkpoint(ctx(), pid, req.training_job_id, req.path, req.step,
                                                      actor(request))


@app.get("/api/projects/{pid}/checkpoints")
async def checkpoints(pid: str) -> list[Checkpoint]:
    c = ctx()
    rows = await c.repo.list("checkpoints", Checkpoint, project_id=pid)
    # #region agent log
    from ..context import agent_debug
    unfiltered = await c.store.list("checkpoints")
    agent_debug("D", "app.py:checkpoints", "list checkpoints", {
        "project_id": pid, "store": c.store.name, "matched": len(rows),
        "unfiltered": len(unfiltered),
        "matched_paths": [{"id": ck.id, "path": ck.path, "exists": Path(ck.path).joinpath("config.json").exists()} for ck in rows[:8]],
        "other_projects": sorted({r.get("project_id") for r in unfiltered if r.get("project_id") != pid})[:8],
    })
    # #endregion
    return rows


@app.post("/api/projects/{pid}/evaluations")
async def evaluate(pid: str, req: LaunchEvaluationRequest, request: Request):
    run = await simulation_runner.launch_evaluation(ctx(), await orch.get_project(ctx(), pid), req, actor(request))
    return await simulation_runner.get_evaluation(ctx(), run.id)


@app.get("/api/evaluations/{evid}")
async def get_eval(evid: str):
    ev = await simulation_runner.get_evaluation(ctx(), evid)
    if not ev:
        raise NotFound(evid)
    return ev


@app.get("/api/projects/{pid}/evaluations")
async def evals(pid: str):
    return await simulation_runner.list_evaluations(ctx(), pid)


class CompareReq(BaseModel):
    evaluation_ids: list[str] | None = None


@app.post("/api/projects/{pid}/comparisons")
async def compare(pid: str, req: CompareReq):
    return await experiment_comparison.compare(ctx(), pid, req.evaluation_ids)


@app.get("/api/projects/{pid}/next-actions")
async def next_actions(pid: str):
    st = await orch.project_state(ctx(), await orch.get_project(ctx(), pid))
    return st.next_actions


# ------------------------------------------------------------------ import demonstrations
class HubImportReq(BaseModel):
    repo_id: str
    max_episodes: int = 10
    revision: str | None = None


@app.post("/api/projects/{pid}/imports/hub")
async def import_hub(pid: str, req: HubImportReq, request: Request):
    from ..services import import_service
    src = {"hub": req.repo_id, **({"revision": req.revision} if req.revision else {})}
    job = await import_service.start_import(ctx(), await orch.get_project(ctx(), pid), src,
                                            max(1, min(req.max_episodes, 50)), actor(request))
    return job.model_dump()


@app.post("/api/projects/{pid}/imports/upload")
async def import_upload(pid: str, request: Request, max_episodes: int = 10):
    """Body: multipart form with `file` = .zip / .tar.gz of a LeRobot dataset folder."""
    from ..services import import_service
    form = await request.form()
    up = form.get("file")
    if not isinstance(up, UploadFile):
        raise ValueError("send the dataset archive as multipart field 'file'")
    if (up.size or 0) > import_service.MAX_UPLOAD_BYTES:
        raise ValueError("archive larger than 2 GB; use a Hugging Face repo id instead")
    # streamed from Starlette's spooled temp file to disk off the event loop, never held in memory
    src = await asyncio.to_thread(import_service.save_upload, ctx(), pid, up.filename or "upload.zip", up.file)
    job = await import_service.start_import(ctx(), await orch.get_project(ctx(), pid), src,
                                            max(1, min(max_episodes, 50)), actor(request))
    return job.model_dump()


# ------------------------------------------------------------------ real SO-101 arm (local studio only)
def _local_human(request: Request) -> None:
    """Hardware endpoints are never tools: agents and remote callers cannot reach them."""
    from ..hardware import setup as hw_setup

    remote = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    if remote or request.client is None or request.client.host not in LOCAL:
        raise hw_setup.HardwareRefused("Real-robot controls are only available in the local studio.")
    from ..services.agent_obs import has_agent_headers

    if has_agent_headers(request.headers) or request.headers.get("x-cortex-actor") in ("mcp", "agent"):
        raise hw_setup.HardwareRefused("Real-robot controls are a human click in the studio; agents cannot use them.")


@app.get("/api/hardware")
async def hardware_status():
    from ..hardware import setup as hw
    return hw.status(ctx().data_dir, settings().cortex_enable_hardware)


@app.put("/api/hardware/config")
async def hardware_config(request: Request):
    from ..hardware import setup as hw
    _local_human(request)
    hw.save_config(ctx().data_dir, hw.HardwareConfig.model_validate(await request.json()))
    await ctx().audit(actor(request), "hardware_config", None, {}, "ok", "hardware setup saved; disarmed")
    return hw.status(ctx().data_dir, settings().cortex_enable_hardware)


@app.post("/api/hardware/arm")
async def hardware_arm(request: Request):
    from ..hardware import setup as hw
    _local_human(request)
    a = hw.arm(settings().cortex_enable_hardware, hw.ArmRequest.model_validate(await request.json()))
    await ctx().audit(actor(request), "hardware_arm", None, {"operator": a.operator}, "ok",
                      f"armed until {a.expires_at.isoformat()}")
    return hw.status(ctx().data_dir, settings().cortex_enable_hardware)


@app.post("/api/hardware/disarm")
async def hardware_disarm(request: Request):
    from ..hardware import setup as hw
    _local_human(request)
    hw.disarm()
    await ctx().audit(actor(request), "hardware_disarm", None, {}, "ok", "disarmed")
    return hw.status(ctx().data_dir, settings().cortex_enable_hardware)


@app.get("/api/skills")
async def skills():
    """Physical Skills (#276): every trained policy across projects, its measured score and how to run it live."""
    from ..services import skills_service
    return await skills_service.list_skills(ctx())


@app.get("/api/skills/{sid}")
async def skill(sid: str):
    from ..services import skills_service
    return await skills_service.get_skill(ctx(), sid)


@app.get("/api/live-policy")
async def live_policy_status():
    from ..services import live_policy_service as lp
    return await lp.status(ctx())


@app.post("/api/live-policy/start")
async def live_policy_start(request: Request):
    from ..services import live_policy_service as lp
    try:
        res = await lp.start(ctx())
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e)) from e
    await ctx().audit(actor(request), "live_policy_start", None, {"job": res.get("job")}, "ok",
                      "GPU inference server requested")
    return res


@app.post("/api/live-policy/stop")
async def live_policy_stop(request: Request):
    from ..services import live_policy_service as lp
    res = await lp.stop(ctx())
    await ctx().audit(actor(request), "live_policy_stop", None, {}, "ok", "GPU inference server stopped")
    return res


@app.get("/api/live-policy/stream/{task}/{episode}")
async def live_policy_stream(task: str, episode: int):
    """The policy running on the GPU over a held-out episode, as MJPEG (proxied through the tunnel).
    The session is recorded as it passes and saved to the task's project when the stream ends (#179)."""
    from ..services import live_policy_service as lp
    from ..services import live_recording
    st = await lp.status(ctx())
    if not st["available"]:
        raise HTTPException(503, st["reason"])
    if task not in st["policies"] or episode not in st["policies"][task]["held_out"]:
        raise NotFound(f"No live policy '{task}' with held-out episode {episode}.")
    rec = live_recording.StreamRecorder()

    async def gen():
        try:
            async for chunk in lp.relay(ctx(), f"/live/{task}/{episode}", rec):
                yield chunk
        finally:  # finished, or the viewer left: encode and store in the background
            live_recording.save_soon(ctx(), task, episode, rec, st, lp.STATE.get("job"))

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/live-policy/sim/{task}/{seed}")
async def live_policy_sim(task: str, seed: int):
    """A sim-trained policy running closed loop in MuJoCo on the GPU, on an unseen layout, as MJPEG.
    The run is recorded as it passes and saved to the sim_<task> project when the stream ends (#179)."""
    from ..services import live_policy_service as lp
    from ..services import live_recording
    st = await lp.status(ctx())
    if not st["available"]:
        raise HTTPException(503, st["reason"])
    if task not in (st.get("sim") or {}):
        raise NotFound(f"No sim-trained policy for '{task}' on the GPU server yet.")
    rec = live_recording.StreamRecorder()

    async def gen():
        try:
            async for chunk in lp.relay(ctx(), f"/sim/{task}/{seed}", rec):
                yield chunk
        finally:  # finished, or the viewer left: encode and store in the background
            live_recording.save_soon(ctx(), task, seed, rec, st, lp.STATE.get("job"), kind="live_sim")

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


class CalibrateReq(BaseModel):
    device: Literal["follower", "leader"]


class ProcessInputReq(BaseModel):
    text: Literal["", "c"] = ""


@app.post("/api/hardware/calibrate")
async def hardware_calibrate(req: CalibrateReq, request: Request):
    from ..hardware import calibrate, setup as hw
    _local_human(request)
    out = await calibrate.start_calibration(req.device, hw.load_config(ctx().data_dir),
                                            settings().cortex_enable_hardware)
    await ctx().audit(actor(request), "hardware_calibrate", None, {"device": req.device}, "ok", "calibration started")
    return out


@app.get("/api/hardware/process")
async def hardware_process():
    from ..hardware import calibrate
    return calibrate.snapshot()


@app.post("/api/hardware/process/input")
async def hardware_process_input(req: ProcessInputReq, request: Request):
    from ..hardware import calibrate
    _local_human(request)
    return await calibrate.send_input(req.text)


@app.post("/api/hardware/process/stop")
async def hardware_process_stop(request: Request):
    from ..hardware import calibrate
    _local_human(request)
    return await calibrate.stop()


@app.post("/api/projects/{pid}/hardware/teleop")
async def hardware_teleop(pid: str, request: Request):
    from ..services import hardware_service as hs
    _local_human(request)
    req = hs.TeleopRequest.model_validate(await request.json())
    return await hs.start_teleop(ctx(), await orch.get_project(ctx(), pid), req)


@app.post("/api/projects/{pid}/hardware/policy")
async def hardware_policy(pid: str, request: Request):
    from ..services import hardware_service as hs
    _local_human(request)
    req = hs.PolicyRunRequest.model_validate(await request.json())
    return await hs.start_policy(ctx(), await orch.get_project(ctx(), pid), req)


@app.get("/api/hardware/run")
async def hardware_run():
    from ..services import hardware_service as hs
    return hs.snapshot()


@app.post("/api/hardware/run/{op}")
async def hardware_run_op(op: Literal["heartbeat", "end_episode", "stop", "release"], request: Request):
    """heartbeat: the operator is holding the deadman (send ~10 Hz while held; lapses stop the arm)."""
    from ..services import hardware_service as hs
    _local_human(request)
    await hs.send(op)
    return {"ok": True} if op == "heartbeat" else hs.snapshot()


# ------------------------------------------------------------------ studio read models
@app.get("/api/projects/{pid}/iterations")
async def iterations(pid: str):
    from ..services import studio_service
    return await studio_service.iterations(ctx(), pid)


@app.get("/api/projects/{pid}/rollouts")
async def rollouts(pid: str):
    from ..services import studio_service
    return await studio_service.rollouts(ctx(), pid)


@app.get("/api/projects/{pid}/replays")
async def replays(pid: str):
    """Policies run open loop on held-out recorded demos (Futurama replays and live GPU sessions)."""
    from ..services import replay_service
    return await replay_service.list_replays(ctx(), pid)


@app.get("/api/episodes/{eid}/camera/{camera}.mp4")
async def episode_camera(eid: str, camera: str):
    from fastapi.responses import FileResponse

    from ..services import studio_service
    path = await studio_service.camera_video(ctx(), eid, camera)
    return FileResponse(path, media_type="video/mp4")


# ------------------------------------------------------------------ agent demonstrations
@app.get("/api/planner")
async def planner():
    from ..services.agent_collection_service import planner_status
    return await planner_status(ctx())


@app.post("/api/projects/{pid}/agent-collections")
async def start_agent_collection(pid: str, req: StartAgentCollectionRequest, request: Request):
    from ..services import agent_collection_service as acs
    return await acs.start_collection(ctx(), await orch.get_project(ctx(), pid), req, actor(request))


@app.get("/api/projects/{pid}/agent-collections")
async def agent_collections(pid: str):
    from ..services import agent_collection_service as acs
    return await acs.list_collections(ctx(), pid)


@app.get("/api/agent-collections/{cid}")
async def agent_collection(cid: str):
    from ..services import agent_collection_service as acs
    col = await acs.get_collection(ctx(), cid)
    job = ctx().jobs.get(col.job_id or "")
    return {"collection": col, "attempts": await acs.list_attempts(ctx(), col.project_id, cid),
            "job": job.model_dump() if job else None}


@app.post("/api/agent-collections/{cid}/stop")
async def stop_agent_collection(cid: str, request: Request):
    from ..services import agent_collection_service as acs
    col = await acs.get_collection(ctx(), cid)
    if col.job_id:
        await ctx().jobs.cancel(col.job_id)
    if col.status not in ("queued", "running"):
        return col  # already finished: keep its real outcome
    col.status = "cancelled"
    await ctx().repo.save("agent_collections", col)
    await ctx().audit(actor(request), "stop_agent_collection", col.project_id, {"collection_id": cid}, "ok", cid)
    return col


@app.get("/api/agent-collections/{cid}/live")
async def agent_collection_live(cid: str):
    """Steps of the attempt in progress (for reconnecting viewers); finished attempts have full traces."""
    from ..services import agent_collection_service as acs
    await acs.get_collection(ctx(), cid)
    return acs.LIVE.get(cid, {"index": None, "start": None, "steps": []})


@app.get("/api/projects/{pid}/planner-budget")
async def planner_budget(pid: str):
    from ..services import agent_collection_service as acs
    p = await orch.get_project(ctx(), pid)
    spent = await acs.spent_usd(ctx(), pid)
    return {"budget_usd": p.autopilot.model_budget_usd, "spent_usd": spent,
            "remaining_usd": round(p.autopilot.model_budget_usd - spent, 4)}


@app.get("/api/projects/{pid}/agent-attempts")
async def agent_attempts(pid: str):
    from ..services import agent_collection_service as acs
    return await acs.list_attempts(ctx(), pid)


@app.get("/api/agent-attempts/{aid}")
async def agent_attempt(aid: str):
    from ..services import agent_collection_service as acs
    return await acs.attempt_trace(ctx(), aid)


@app.get("/api/projects/{pid}/collector-comparison")
async def collector_comparison(pid: str):
    from ..services import agent_collection_service as acs
    return acs.compare_collectors(await acs.list_attempts(ctx(), pid))


# ------------------------------------------------------------------ autopilot
@app.post("/api/projects/{pid}/autopilot/start")
async def ap_start(pid: str):
    return await autopilot.start(ctx(), await orch.get_project(ctx(), pid))


@app.post("/api/projects/{pid}/autopilot/stop")
async def ap_stop(pid: str):
    return await autopilot.stop(ctx(), pid)


@app.get("/api/projects/{pid}/autopilot")
async def ap_get(pid: str):
    return await autopilot.current(ctx(), pid)


@app.get("/api/projects/{pid}/infra")
async def infra_get(pid: str):
    from ..services import infra_service
    return await infra_service.options(ctx(), await orch.get_project(ctx(), pid))


@app.put("/api/projects/{pid}/infra")
async def infra_put(pid: str, request: Request):
    from ..models import InfraChoice
    from ..services import infra_service
    p = await orch.get_project(ctx(), pid)
    choice = InfraChoice.model_validate({**p.infra.model_dump(), **(await request.json())})
    await infra_service.set_choice(ctx(), p, choice, actor(request))
    return await infra_service.options(ctx(), p)


@app.put("/api/projects/{pid}/autopilot/policy")
async def ap_policy(pid: str, policy: dict, request: Request):
    p = await orch.get_project(ctx(), pid)
    p.autopilot = p.autopilot.model_validate({**p.autopilot.model_dump(), **policy})
    await ctx().repo.save("projects", p, status="active")
    await ctx().audit(actor(request), "set_autopilot_policy", pid, policy, "ok", "policy updated")
    return p.autopilot


# ------------------------------------------------------------------ agent
class ChatReq(BaseModel):
    message: str
    thread: str | None = None  # continue this chat thread; default: the latest


@app.post("/api/projects/{pid}/agent")
async def chat(pid: str, req: ChatReq):
    return StreamingResponse(agent.chat(ctx(), pid, req.message, req.thread), media_type="text/event-stream")


@app.post("/api/projects/{pid}/agent/stop")
async def agent_stop(pid: str):
    """Stop the autonomous chain: no more automatic steps, and the one running now (if any) ends."""
    from ..services import autonomy
    return {"stopped": autonomy.stop(ctx(), pid)}


@app.get("/api/projects/{pid}/agent/history")
async def history(pid: str, thread: str | None = None):
    """One chat thread (the latest unless `thread` is given):
    [{kind: user|assistant|tool_call|tool_result|approval_required|done|job_done, ..., at}]."""
    return await agent.load_log(ctx(), pid, thread)


@app.get("/api/projects/{pid}/agent/threads")
async def chat_threads(pid: str):
    """Past chats, newest first: [{id, title, created_at, updated_at}]."""
    from ..services import chat_store
    return [{k: t[k] for k in ("id", "title", "created_at", "updated_at")} for t in await chat_store.threads(ctx(), pid)]


@app.post("/api/projects/{pid}/agent/threads")
async def chat_new_thread(pid: str):
    from ..services import chat_store
    t = await chat_store.new_thread(ctx(), pid)
    return {k: t[k] for k in ("id", "title", "created_at", "updated_at")}


@app.get("/api/agent/engine")
async def agent_engine():
    """Which agent answers the studio chat, and `note`: what is missing for a better one (None if nothing)."""
    return {**await agent.engine_info(ctx()), "model": ctx().settings.claude_code_model}


@app.post("/api/projects/{pid}/agent/reset")
async def agent_reset(pid: str):
    """'New chat': start a new thread; earlier chats stay in the history."""
    from ..services import chat_store
    t = await chat_store.new_thread(ctx(), pid)
    return {"reset": True, "thread": t["id"]}


# ------------------------------------------------------------------ tools (MCP)
@app.get("/api/tools")
async def list_tools():
    return [{"name": t.name, "description": t.description, "input_schema": t.args.model_json_schema(),
             "agent_callable": t.agent_callable} for t in orch.TOOLS.values()]


@app.post("/api/tools/{name}")
async def call_tool(name: str, args: dict[str, Any], request: Request):
    from ..services import agent_obs

    who = "mcp" if request.headers.get("x-cortex-actor", "mcp") != "agent" else "agent"
    caller = await agent_obs.resolve(ctx(), request.headers, who)  # agent key, MCP session, client (#105)
    try:
        return await orch.call_tool(ctx(), name, args, who, caller)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)) from e


# ------------------------------------------------------------------ RL sandboxes (#80)
class RLProposeReq(BaseModel):
    checkpoint_id: str | None = None
    max_sandboxes: int = 3
    budget: dict[str, Any] | None = None


class RLAcceptReq(BaseModel):
    ids: list[str]
    budget_cap: float = 0.0
    max_cpu_hours: float | None = None


@app.get("/api/projects/{pid}/rl/weaknesses")
async def rl_weaknesses(pid: str, checkpoint_id: str | None = None, target: float = 0.8):
    from ..services import rl_service
    ws = await rl_service.weaknesses(ctx(), await orch.get_project(ctx(), pid), checkpoint_id, target)
    return [w.model_dump(mode="json") for w in ws]


@app.post("/api/projects/{pid}/rl/sandboxes")
async def rl_propose(pid: str, req: RLProposeReq, request: Request):
    from ..rl import SandboxBudget
    from ..services import rl_service
    budget = SandboxBudget.model_validate(req.budget) if req.budget else None
    sbs = await rl_service.propose_sandboxes(ctx(), await orch.get_project(ctx(), pid), budget, req.checkpoint_id,
                                             req.max_sandboxes, actor(request))
    return [rl_service.proposal_view(s) for s in sbs]


@app.post("/api/projects/{pid}/rl/sandboxes/accept")
async def rl_accept(pid: str, req: RLAcceptReq, request: Request):
    from ..services import rl_service
    sbs = await rl_service.accept(ctx(), await orch.get_project(ctx(), pid), req.ids, req.budget_cap,
                                  req.max_cpu_hours, actor(request))
    return [rl_service.status_view(s) for s in sbs]


@app.get("/api/projects/{pid}/rl/sandboxes")
async def rl_list(pid: str):
    from ..services import rl_service
    return [rl_service.status_view(s) for s in await rl_service.list_sandboxes(ctx(), pid)]


@app.get("/api/rl/sandboxes/{sid}")
async def rl_get(sid: str):
    from ..services import rl_service
    return rl_service.status_view(await rl_service.get(ctx(), sid))


@app.post("/api/rl/sandboxes/{sid}/stop")
async def rl_stop(sid: str, request: Request):
    from ..services import rl_service
    return rl_service.status_view(await rl_service.stop(ctx(), sid, actor(request)))


@app.get("/api/rl/sandboxes/{sid}/live.jpg")
async def rl_live(sid: str):
    from ..services import rl_service
    sb = await rl_service.get(ctx(), sid)
    path = ctx().local(f"projects/{sb.project_id}/rl/{sb.id}/live/live.jpg")
    if not path.exists():
        raise HTTPException(404, "no live frame yet")
    return Response(path.read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


class RLCloseReq(BaseModel):
    episodes_per_condition: int | None = None


@app.post("/api/rl/sandboxes/{sid}/close")
async def rl_close(sid: str, req: RLCloseReq, request: Request):
    """Advance a finished sandbox's round one step (demos -> dataset version -> proposed retrain -> paired re-eval)."""
    from ..services import rl_service
    sb = await rl_service.get(ctx(), sid)
    rnd = await rl_service.close_round(ctx(), await orch.get_project(ctx(), sb.project_id), sid, actor(request),
                                       req.episodes_per_condition)
    return rl_service.round_view(rnd)


@app.get("/api/projects/{pid}/rl/rounds")
async def rl_rounds(pid: str):
    from ..rl.proposals import transfer_fit
    from ..services import rl_service
    rows = await rl_service.rounds(ctx(), pid)
    return {"rounds": [rl_service.round_view(r) for r in rows],
            "transfer": transfer_fit(await rl_service.history(ctx(), pid))}


# ------------------------------------------------------------------ agent observability (#105)
from .agents import router as agents_router  # noqa: E402

app.include_router(agents_router)
