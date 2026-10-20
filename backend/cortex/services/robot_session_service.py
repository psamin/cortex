"""Robot sessions driven by an external agent (the user's Claude Code, via MCP).

start  -> a worker process with a reset MuJoCo episode (new seed, chosen scenario)
act    -> one validated tool call (observe, propose_grasp, move, gripper, verify, place, check)
finish -> recorded episode; the simulator decides success; successes become Episodes
          with provenance {planner: "claude-code"}, failures stay diagnostics

Every action is streamed to the studio (agent_step events) so the live robot view and
the action timeline show what the agent is doing. Sessions idle for 10 minutes close.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from ..context import Ctx
from ..errors import NotFound, UserError
from ..models import AgentAttempt, AgentCollection, Condition, Project, Scenario, StartAgentCollectionRequest, new_id
from . import agent_collection_service as acs
from .jobs import BACKEND_DIR
from .recording_service import dataset_blur_reference, fresh_live_dir

IDLE_TIMEOUT_S = 600
PHASES = {"observe_scene": "Observing scene", "get_robot_state": "Reading robot state",
          "propose_grasp": "Selecting grasp", "verify_grasp": "Verifying grasp",
          "place_object": "Moving to bowl and releasing", "check_task_success": "Verifying task success"}


@dataclass
class RobotSession:
    id: str
    project_id: str
    collection: AgentCollection
    attempt: AgentAttempt
    proc: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.time)
    holding: bool = False
    steps: int = 0


SESSIONS: dict[str, RobotSession] = {}


RPC_TIMEOUT_S = 180


class SessionBroken(UserError, RuntimeError):
    """The worker stopped answering in step: the session is closed, so start a new episode. A UserError, so the
    agent gets this message (HTTP 400), not a generic internal error."""

    user_facing = True


async def _rpc(s: RobotSession, req: dict[str, Any]) -> dict[str, Any]:
    """One request, one reply. The worker answers strictly in order, so a reply that arrives after we gave up would
    be read as the answer to the *next* request (#356). Any timeout or I/O error therefore ends the session: it is
    dropped and its worker killed, and the caller is told to start a new episode."""
    async with s.lock:
        try:
            assert s.proc.stdin and s.proc.stdout
            s.proc.stdin.write((json.dumps(req) + "\n").encode())
            await s.proc.stdin.drain()
            while True:
                line = await asyncio.wait_for(s.proc.stdout.readline(), timeout=RPC_TIMEOUT_S)
                if not line:
                    raise SessionBroken("the robot session's simulator exited; start a new episode")
                try:
                    out = json.loads(line)
                except json.JSONDecodeError:
                    continue  # library noise on stdout
                s.last_used = time.time()
                return out
        except BaseException as e:
            _kill(s)
            if isinstance(e, TimeoutError):
                raise SessionBroken(f"the robot session didn't answer {req.get('op')!r} within {RPC_TIMEOUT_S} s, "
                                    "so it was closed; start a new episode") from e
            if isinstance(e, (OSError, AssertionError)):
                raise SessionBroken(f"lost the robot session's simulator ({type(e).__name__}); "
                                    "start a new episode") from e
            raise


def _kill(s: RobotSession) -> None:
    """Drop a session whose worker can't be trusted any more and kill the worker (no close handshake)."""
    SESSIONS.pop(s.id, None)
    if s.proc.returncode is None:
        try:
            s.proc.kill()
        except ProcessLookupError:
            pass


def _phase(tool: str, args: dict[str, Any], holding: bool) -> str:
    if tool == "set_gripper":
        return "Closing gripper" if args.get("state") == "closed" else "Opening gripper"
    if tool == "move_end_effector":
        return "Approaching object" if not holding else (
            "Lifting" if (args.get("target") or {}).get("z", 0) >= 0.06 else "Moving toward bowl")
    return PHASES.get(tool, tool)


async def start(ctx: Ctx, project: Project, cube_region: str | None = None, lighting: str | None = None,
                seed: int | None = None, collection_id: str | None = None, agent_label: str = "claude-code",
                actor: str = "mcp", scenario: Scenario | None = None, attempt_index: int | None = None
                ) -> dict[str, Any]:
    if project.mode != "sim":
        raise PermissionError("Physical robots need the local hardware adapter with an operator present.")
    await _reap_idle(ctx)
    if collection_id:
        col = await acs.get_collection(ctx, collection_id)
    else:
        col = AgentCollection(project_id=project.id, collector="agent", provider=f"{agent_label} via MCP",
                              model=agent_label, mode="camera", status="running", reason="interactive agent session",
                              request=StartAgentCollectionRequest(num_demos=1, collector="agent"),
                              totals=acs._empty_totals(0))
        await ctx.repo.save("agent_collections", col)
    prior = await ctx.repo.list("agent_attempts", AgentAttempt, project_id=project.id)
    seed = seed if seed is not None else acs.AGENT_SEED_BASE + len(prior)
    if scenario is None and (cube_region or lighting):
        cond = Condition(cube_region=cube_region or "center", lighting=lighting or "normal")  # type: ignore[arg-type]
        scenario = Scenario(cube_region=cond.cube_region, lighting=cond.lighting)
    elif scenario is None:
        scenario = acs.plan_scenarios(project, StartAgentCollectionRequest(num_demos=1, diversity="medium"), seed)[0]
    index = attempt_index if attempt_index is not None else col.totals.get("attempts", 0)
    attempt = AgentAttempt(project_id=project.id, collection_id=col.id, index=index,
                           collector="agent", seed=seed, scenario=scenario, status="failed", mode="camera",
                           provider=col.provider, model=col.model)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "cortex.workers.session_worker", cwd=str(BACKEND_DIR),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        limit=32 * 1024 * 1024)
    s = RobotSession(id=new_id("rs"), project_id=project.id, collection=col, attempt=attempt, proc=proc)
    SESSIONS[s.id] = s
    from .intake_service import scene_payload

    try:
        out = await _rpc(s, {"op": "reset", "seed": seed, "scenario": scenario.model_dump(), "mode": "camera",
                             "live_dir": str(fresh_live_dir(ctx, project.id)), "scene": scene_payload(ctx, project)})
    except BaseException:  # the agent never got this session's id: don't leave its worker running (#356)
        _kill(s)
        raise
    if not out.get("ok"):
        await close(s.id)
        raise RuntimeError(out.get("error", "reset failed"))
    ctx.bus.publish(project.id, "agent_step", {"collection_id": col.id, "type": "attempt_start", "index": attempt.index,
                                               "total": attempt.index + 1, "seed": seed,
                                               "scenario": scenario.model_dump()})
    await ctx.audit(actor, "robot_start_episode", project.id, {"session_id": s.id, "seed": seed,
                                                                "scenario": scenario.model_dump()}, "ok", s.id)
    return {"session_id": s.id, "attempt_id": attempt.id, "collection_id": col.id, "seed": seed,
            "scenario": scenario.model_dump(), "image_px": [480, 480], "task": out.get("task"),
            "frames": "robot base frame: +x forward, +y left, z up (metres); overhead image up=+x, left=+y"}


async def act(ctx: Ctx, session_id: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if s is None:
        raise NotFound(f"robot session {session_id} (closed or expired; start a new episode)")
    if tool in ("finish_episode", "reset_environment"):
        raise ValueError("use robot_finish_episode / robot_start_episode")
    phase = _phase(tool, args, s.holding)
    out = await _rpc(s, {"op": "call", "tool": tool, "args": args})
    if "result" not in out:
        raise RuntimeError(out.get("error", "tool call failed"))
    s.holding, s.steps = bool(out.get("holding")), int(out.get("sim_steps", s.steps))
    res = out["result"]
    ctx.bus.publish(s.project_id, "agent_step", {
        "collection_id": s.collection.id, "type": "agent_step", "index": s.attempt.index, "tool": tool, "args": args,
        "phase": phase, "ok": bool(res.get("ok")), "result": res, "sim_step": s.steps, "model_call": None})
    return {"result": res, "image_b64": out.get("image_b64"), "image_kind": out.get("image_kind"), "phase": phase}


async def finish(ctx: Ctx, session_id: str, believes_success: bool, summary: str, actor: str = "mcp") -> dict[str, Any]:
    s = SESSIONS.get(session_id)
    if s is None:
        raise NotFound(session_id)
    a = s.attempt
    out_dir = ctx.local(acs.attempt_key(a))
    res = await _rpc(s, {"op": "finish", "out_dir": str(out_dir), "meta": {
        "seed": a.seed, "scenario": a.scenario.model_dump(), "mode": "camera", "provider": a.provider,
        "model_requested": a.model, "model_claimed_success": believes_success, "model_summary": summary}})
    if not res.get("ok"):
        raise RuntimeError(res.get("error", "finish failed"))
    ev = {"success": res["success"], "usage": {"calls": 0, "cost_usd": 0.0,
                                               "cost_source": "the user's own agent (Claude Code); calls not metered by Cortex"},
          "recoveries": 0, "counters": res.get("counters", {}), "num_frames": res.get("num_frames", 0),
          "wall_s": res.get("wall_s", 0.0), "model_claimed_success": believes_success,
          "stop_reason": None, "error": None, "trace_len": res.get("trace_len", 0), "endpoint": "mcp"}
    blur_ref = await dataset_blur_reference(ctx, s.project_id)
    project = await ctx.repo.get("projects", s.project_id, Project)
    await acs._record_attempt(ctx, project, s.collection, a, ev, blur_ref)  # type: ignore[arg-type]
    if s.collection.reason == "interactive agent session":  # batch collections are closed by their runner
        s.collection.status = "succeeded"
        await ctx.repo.save("agent_collections", s.collection)
    await close(session_id)
    await ctx.audit(actor, "robot_finish_episode", s.project_id, {"session_id": session_id}, "ok",
                    f"{'success' if res['success'] else 'failure'} (simulator); agent claimed {believes_success}")
    return {"success": res["success"], "judged_by": "simulator (cube resting inside the box)",
            "agent_claimed_success": believes_success, "attempt_id": a.id, "episode_id": a.episode_id,
            "frames_recorded": res.get("num_frames", 0),
            "note": "successful episodes become LeRobot demonstrations; failures are kept as diagnostics only"}


CLOSE_TIMEOUT_S = 20  # how long a worker gets to exit cleanly after the close op


async def close(session_id: str) -> None:
    s = SESSIONS.pop(session_id, None)
    if s is None:
        return
    try:
        if s.proc.returncode is None and s.proc.stdin:
            s.proc.stdin.write(b'{"op":"close"}\n')
            await s.proc.stdin.drain()
            # GL teardown can take a while on a busy Windows machine; 5 s killed healthy workers mid-exit
            await asyncio.wait_for(s.proc.wait(), CLOSE_TIMEOUT_S)
    except Exception:  # noqa: BLE001 - closing is best effort; it must never fail the episode or collection
        try:
            s.proc.kill()
        except ProcessLookupError:  # it exited on its own between the wait and the kill
            pass


async def _reap_idle(ctx: Ctx) -> None:
    for sid, s in list(SESSIONS.items()):
        if time.time() - s.last_used > IDLE_TIMEOUT_S:
            await close(sid)
