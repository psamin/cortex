"""Vision-language manipulation planner over the validated tool API.

The model chooses semantic actions and waypoints (observe, point at the object, move,
grip, verify, place, go home); ManipulationSession validates and executes them in MuJoCo
with IK. The model never emits motor commands. Each episode is bounded by a
model-call budget, a dollar budget and the session's step limit.

It talks to any OpenAI-compatible chat-completions endpoint. Cortex's route for it is the opt-in
self-hosted model (e.g. Qwen3-VL on Futurama via vLLM, points in relative 0-1000 coordinates, no
per-token cost); the default agent planner is the user's Claude Code (services/claude_code_collector.py).
The endpoint and model are always given explicitly and recorded on every attempt; no other model is
substituted when the configured one is unavailable.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from ..sim.manipulation import (
    PLAN_PX, ManipulationSession, annotate_overhead, from_model_point, to_data_url, tool_schemas,
)


SYSTEM = """You control a simulated SO-101 robot arm through tools. Task: {task}
You never send motor commands; you choose actions and targets, and a local IK controller executes them safely.

Frames and units:
- Robot base frame in metres: +x forward (away from the base), +y to the robot's left, z = height above the table.
- {image_text} Image up = +x, image left = +y.
- {scene_text}

Planning mode: {mode_text}

Typical plan (observe -> grasp -> place -> go home): {plan_text}
Rules:
- Point at the object's centre in the most recent overhead image. {point_text}
- If a tool returns ok=false, read the error code and adjust (different point, lower/closer target) instead of repeating it.
- If verify_grasp says holding_object=false: open the gripper, observe again, re-propose the grasp and retry.
- Call one tool at a time. Be brief in text. Finish when check_task_success is true or you cannot make progress."""

CUBE_SCENE = 'The red cube is 3 cm wide. The green open box is the "bowl": place the cube inside it.'
CUBE_PLAN = """observe_scene(overhead) -> propose_grasp(image_point on the red cube) -> set_gripper(open) ->
move_end_effector(pregrasp) -> move_end_effector(grasp) -> set_gripper(closed) -> verify_grasp ->
move_end_effector(lift to z≈0.075) -> place_object(image_point at the centre of the green box) -> go_home ->
check_task_success -> finish_episode."""
TOP_PLAN = """observe_scene(overhead) -> propose_grasp(image_point on the {obj}) -> set_gripper(open) ->
move_end_effector(pregrasp) -> move_end_effector(grasp) -> set_gripper(closed) -> verify_grasp ->
move_end_effector(lift: the grasp pose raised 6 cm) -> place_object(zone='{zone}') -> go_home -> check_task_success
-> finish_episode."""
SIDE_PLAN = """observe_scene(overhead) -> propose_grasp(image_point on the {obj}) -> set_gripper(open) ->
move_end_effector(transit) -> move_end_effector(pregrasp) -> move_end_effector(grasp) -> move_end_effector(slide)
-> set_gripper(closed) -> verify_grasp -> move_end_effector(lift) -> place_object(zone='{zone}') -> go_home ->
check_task_success -> finish_episode. It is a side grasp: pass each pose exactly as propose_grasp returns it,
including its approach axis."""


def scene_prompt(ctx: dict[str, Any] | None) -> tuple[str, str]:
    """(scene text, typical plan) naming the session's real object and zones; the cube text on the tabletop."""
    if not ctx or ctx.get("scene") != "intake":
        return CUBE_SCENE, CUBE_PLAN
    obj = str(ctx["object"]).replace("_", " ")
    zones = ", ".join(f"'{z}'" for z in ctx.get("zones") or [])
    scene = (f"The object to pick is {ctx['object_text']}. Place zones on the table: {zones or 'none'}; "
             f"place it in zone '{ctx['target_zone']}' (green marker).")
    plan = (SIDE_PLAN if ctx.get("grasp_mode") == "side" else TOP_PLAN).format(obj=obj, zone=ctx["target_zone"])
    return scene, plan


MODE_TEXT = {
    "camera": "camera-grounded. You only get camera images and robot proprioception; locate objects by pointing at "
              "pixels in the overhead image.",
    "privileged": "PRIVILEGED-STATE PLANNING. observe_scene also returns simulator ground-truth object poses; "
                  "you may use them directly.",
}

POINT_TEXT = {
    "pixels": ("The overhead image is {px}x{px} pixels with a labelled grid every 40 px.",
               "Image points are pixel coordinates (u = column, v = row)."),
    "norm1000": ("The overhead image uses relative coordinates from 0 to 1000 on both axes (grid labelled every 100).",
                 "Image points are relative coordinates 0-1000 (u = x from the left edge, v = y from the top edge)."),
}

PHASES = {
    "observe_scene": "Observing scene", "get_robot_state": "Reading robot state",
    "propose_grasp": "Selecting grasp", "set_gripper": None, "verify_grasp": "Verifying grasp",
    "place_object": "Moving to bowl and releasing", "check_task_success": "Verifying task success",
    "go_home": "Returning home", "finish_episode": "Finishing",
}


def phase_for(tool: str, args: dict[str, Any], holding: bool) -> str:
    if tool == "set_gripper":
        return "Closing gripper" if args.get("state") == "closed" else "Opening gripper"
    if tool == "move_end_effector":
        if not holding:
            return "Approaching object"
        z = (args.get("target") or {}).get("z", 0)
        return "Lifting" if z >= 0.06 else "Moving toward bowl"
    return PHASES.get(tool) or tool


@dataclass
class PlannerConfig:
    provider: str  # selfhosted
    api_key: str
    model: str = ""
    mode: str = "camera"
    max_calls: int = 16
    max_usd: float = 1.50
    reasoning_effort: str = "low"
    timeout_s: float = 120.0
    max_images: int = 4  # images kept in each request; vLLM serves Qwen3-VL with --limit-mm-per-prompt image=8
    endpoint: str = ""  # chat/completions URL; recorded on every attempt
    point_format: str = "pixels"  # how the model reports image points: pixels | norm1000
    price_per_m: dict[str, float] | None = None  # None = no per-token charge (self-hosted)

    def __post_init__(self) -> None:
        if not (self.endpoint and self.model):
            raise ValueError(f"the {self.provider} planner needs an endpoint and a model")


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_source: str = "computed from token counts"
    latency_s: float = 0.0
    models_reported: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 4), "cost_source": self.cost_source,
                "latency_s": round(self.latency_s, 2), "models_reported": sorted(self.models_reported)}


class BudgetExceeded(Exception):
    pass


OMITTED = {"type": "text", "text": "[earlier image omitted; the latest images are below]"}


def window_images(messages: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """A copy of `messages` with only the newest `keep` images; older ones become a text note.
    The conversation itself (and so the recorded trace) keeps every image."""
    seen, out = 0, []
    for m in reversed(messages):
        content = m.get("content")
        if isinstance(content, list):
            parts = []
            for part in reversed(content):
                if part.get("type") == "image_url":
                    seen += 1
                    part = part if seen <= keep else OMITTED
                parts.append(part)
            m = {**m, "content": parts[::-1]}
        out.append(m)
    return out[::-1]


class AstraPlanner:
    def __init__(self, cfg: PlannerConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.http = httpx.Client(timeout=cfg.timeout_s, transport=transport)
        self.ctx: dict[str, Any] | None = None  # the current session's task object and zones

    def _complete(self, messages: list[dict[str, Any]], usage: Usage) -> dict[str, Any]:
        if usage.calls >= self.cfg.max_calls:
            raise BudgetExceeded(f"model-call budget of {self.cfg.max_calls} reached")
        if usage.cost_usd >= self.cfg.max_usd:
            raise BudgetExceeded(f"episode cost budget ${self.cfg.max_usd:.2f} reached")
        body: dict[str, Any] = {"model": self.cfg.model, "messages": window_images(messages, self.cfg.max_images),
                                "tools": tool_schemas(self.cfg.point_format, self.ctx), "tool_choice": "auto",
                                "parallel_tool_calls": False}
        body["max_tokens"] = 600  # tool calls are ~100-300 tokens
        body["temperature"] = 0.0
        t0 = time.time()
        r = self.http.post(self.cfg.endpoint, json=body,
                           headers={"Authorization": f"Bearer {self.cfg.api_key}"})
        usage.latency_s += time.time() - t0
        usage.calls += 1
        if r.status_code >= 400:
            raise RuntimeError(f"{self.cfg.provider} {r.status_code}: {r.text[:300]}")
        data = r.json()
        u = data.get("usage") or {}
        pin, pout = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
        usage.input_tokens += pin
        usage.output_tokens += pout
        if "cost" in u:
            usage.cost_usd += float(u["cost"])
            usage.cost_source = "provider-reported (usage.cost)"
        elif self.cfg.price_per_m is None:
            usage.cost_source = "self-hosted: no per-token charge (GPU time on our own cluster)"
        else:
            cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            p = self.cfg.price_per_m
            usage.cost_usd += ((pin - cached) * p["input"] + cached * p.get("cached", p["input"])
                               + pout * p["output"]) / 1e6
        if data.get("model"):
            usage.models_reported.add(data["model"])
        return data["choices"][0]["message"]

    def run_episode(self, session: ManipulationSession, task: str,
                    emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        usage = Usage()
        trace: list[dict[str, Any]] = []
        self.ctx = session.task_context()
        scene_text, plan_text = scene_prompt(self.ctx)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM.format(
                task=task, mode_text=MODE_TEXT[self.cfg.mode], scene_text=scene_text, plan_text=plan_text,
                image_text=POINT_TEXT[self.cfg.point_format][0].format(px=PLAN_PX),
                point_text=POINT_TEXT[self.cfg.point_format][1])},
            {"role": "user", "content": "Begin. The arm is at its home pose with the gripper half open."},
        ]
        finished, claim, summary, stop_reason, recoveries = False, None, "", None, 0
        last_failed: set[str] = set()
        while not finished:
            try:
                msg = self._complete(messages, usage)
            except BudgetExceeded as e:
                stop_reason = str(e)
                break
            calls = msg.get("tool_calls") or []
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if msg.get("content"):
                emit({"type": "agent_thought", "text": str(msg["content"])[:500]})
            if not calls:
                messages.append({"role": "user", "content": "Continue with a tool call, or call finish_episode."})
                continue
            images: list[tuple[str, np.ndarray]] = []
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                phase = phase_for(name, args, session.holding)
                if name == "finish_episode":
                    finished, claim, summary = True, bool(args.get("believes_success")), str(args.get("summary", ""))
                    result: dict[str, Any] = {"ok": True}
                    img = None
                else:
                    executed = dict(args)
                    if isinstance(args.get("image_point"), dict):
                        executed["image_point"] = from_model_point(args["image_point"], self.cfg.point_format)
                    result, img = session.call(name, executed)
                ok = bool(result.get("ok"))
                if ok and name in last_failed:
                    recoveries += 1
                    last_failed.discard(name)
                if not ok:
                    last_failed.add(name)
                if name == "verify_grasp" and ok and not result.get("holding_object"):
                    last_failed.add("verify_grasp_retry")
                if name == "verify_grasp" and ok and result.get("holding_object") and "verify_grasp_retry" in last_failed:
                    recoveries += 1
                    last_failed.discard("verify_grasp_retry")
                step = {"i": len(trace), "tool": name, "args": args, "phase": phase, "ok": ok,
                        **({"args_executed": executed} if name != "finish_episode" and executed != args else {}),
                        "result": {k: v for k, v in result.items() if k not in ("wall_ms",)},
                        "sim_step": len(session.rec), "model_call": usage.calls}
                trace.append(step)
                emit({"type": "agent_step", **step})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)[:4000]})
                if img is not None:
                    images.append((name, img))
            for name, img in images:
                shown = annotate_overhead(img, self.cfg.point_format) if session.last_overhead is img else img
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"Image returned by {name} ({img.shape[1]}x{img.shape[0]} px)."},
                    {"type": "image_url", "image_url": {"url": to_data_url(shown)}}]})
        return {"trace": trace, "usage": usage.as_dict(), "finished": finished, "model_claimed_success": claim,
                "model_summary": summary, "stop_reason": stop_reason, "recoveries": recoveries,
                "counters": dict(session.counters)}
