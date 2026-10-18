"""Batch demonstration collection with the user's own Claude Code as the planner.

For each planned scenario, Cortex opens a robot session (exact scene + seed, joined
to the collection), then runs `claude -p` headless with ONLY the robot act/finish
tools allowed and the session id in the prompt. Claude looks at the camera images
and drives the arm; Cortex validates and executes every command, records the
trajectory, and the simulator judges success. If Claude stops without finishing,
Cortex closes the episode itself as a failed attempt (never a demonstration).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..context import Ctx
from ..models import AgentAttempt, AgentCollection, Project, Scenario
from . import robot_session_service as rs
from .claude_code_agent import MCP_URL, executable

ROBOT_TOOLS = ["robot_observe", "robot_state", "robot_propose_grasp", "robot_move", "robot_gripper",
               "robot_verify_grasp", "robot_place", "robot_check_success", "robot_finish_episode"]
MAX_TURNS = 30

PROMPT = """You are generating one robot demonstration for Cortex. Robot session {sid} is already started
({setting}). Use ONLY the cortex robot tools with session_id="{sid}":
1. robot_observe (overhead) and find {obj} and {target} in the image (pixel grid every 40 px).
2. robot_propose_grasp with image_point at the centre of {obj}. It returns the approach for this object
   ({grasp}), with pregrasp and grasp poses.
3. robot_gripper open -> robot_move to the returned pregrasp -> robot_move to the returned grasp -> robot_gripper closed.
4. robot_verify_grasp. If not holding: open, observe again, re-propose and retry once.
5. Lift straight up, robot_observe again, then robot_place {place}.
6. robot_go_home if available, robot_check_success, then robot_finish_episode with your honest belief.
If a command is rejected, read the error code and adjust. Be brief; do not explain at length."""


def _prompt(sid: str, scenario: Scenario, task: dict[str, Any] | None) -> str:
    """The demo prompt for this session's actual task (an intake scene), else the built-in cube task."""
    t = task or {}
    if t.get("scene") == "intake" and t.get("object"):
        zone = t.get("target_zone") or ""
        return PROMPT.format(sid=sid, setting=f"{scenario.lighting} light", obj=t.get("object_text") or f"the {t['object']}",
                             target=f"the '{zone}' zone" if zone else "the target zone", grasp=t.get("grasp_mode", "top"),
                             place=f"with zone='{zone}'" if zone else "with image_point at the target zone's centre")
    return PROMPT.format(sid=sid, setting=f"cube {scenario.cube_region}, {scenario.lighting} light", obj="the red cube",
                         target="the green box", grasp="top", place="with image_point at the green box's centre")


async def run_attempt(ctx: Ctx, project: Project, col: AgentCollection, scenario: Scenario, seed: int, index: int,
                      model: str | None = None) -> dict[str, Any]:
    started = await rs.start(ctx, project, scenario=scenario, seed=seed, collection_id=col.id,
                             attempt_index=index, actor="claude-code-batch")
    sid = started["session_id"]
    workdir = ctx.data_dir / "agent_sessions" / project.id / "batch"
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = workdir / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"cortex": {"type": "http", "url": MCP_URL}}}))
    allowed = ",".join(f"mcp__cortex__cortex_{t}" for t in ROBOT_TOOLS)
    exe = executable() or "claude"  # full path so Windows finds claude.cmd
    cmd = [exe, "-p", _prompt(sid, scenario, started.get("task")),
           "--output-format", "stream-json", "--verbose", "--mcp-config", str(cfg), "--strict-mcp-config",
           "--allowedTools", allowed, "--max-turns", str(MAX_TURNS)]
    if model:
        cmd += ["--model", model]
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(workdir), stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL, limit=16 * 1024 * 1024)
    result: dict[str, Any] = {}
    assert proc.stdout
    try:
        async for raw in proc.stdout:
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        ctx.bus.publish(project.id, "agent_step", {"collection_id": col.id, "type": "agent_thought",
                                                                   "index": index, "text": block["text"][:500]})
            elif ev.get("type") == "result":
                result = ev
        await proc.wait()
    except BaseException:  # job cancelled: stop Claude and release the simulator
        if proc.returncode is None:
            proc.kill()
        await rs.close(sid)
        raise
    finished_by_agent = sid not in rs.SESSIONS
    if not finished_by_agent:  # Claude stopped early: close it ourselves, judged by the simulator
        await rs.finish(ctx, sid, believes_success=False,
                        summary=f"agent stopped without finishing ({result.get('subtype') or 'no result'})",
                        actor="claude-code-batch")
    a = await ctx.repo.get("agent_attempts", started["attempt_id"], AgentAttempt)
    if a is not None:  # record what Claude Code reported for this attempt
        a.usage = {**a.usage, "calls": result.get("num_turns") or 0, "cost_usd": 0.0,
                   "claude_code_reported_cost_usd": result.get("total_cost_usd"),
                   "duration_ms": result.get("duration_ms"),
                   "cost_source": "the user's Claude Code subscription (reported estimate, not charged by Cortex)"}
        a.stop_reason = None if finished_by_agent else "agent stopped without finishing"
        await ctx.repo.save("agent_attempts", a)
    return {"attempt_id": started["attempt_id"], "finished_by_agent": finished_by_agent,
            "turns": result.get("num_turns"), "status": a.status if a else None}
