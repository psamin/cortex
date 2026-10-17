"""GPT-6 Astra as a manipulation planner over the validated tool API.

Astra chooses semantic actions and waypoints (observe, point at the cube, move,
grip, verify, place); ManipulationSession validates and executes them in MuJoCo
with IK. The model never emits motor commands. Each episode is bounded by a
model-call budget, a dollar budget and the session's step limit.

Provider routing (decided by the backend, recorded on every attempt): the InsForge
Model Gateway's OpenRouter key, model "openai/gpt-6-astra" ("openrouter" when a
self-hosted InsForge supplies OPENROUTER_API_KEY itself). No other model or route is
substituted: without a gateway key, collection refuses to start.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np

from ..sim.manipulation import PLAN_PX, ManipulationSession, annotate_overhead, to_data_url, tool_schemas

PRICES_PER_M = {"gpt-6-astra": {"input": 10.0, "cached": 1.0, "output": 50.0}}  # developers.openai.com, Oct 2026
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
ENDPOINTS = {"insforge": OPENROUTER, "openrouter": OPENROUTER}
MODEL_IDS = {"insforge": "openai/gpt-6-astra", "openrouter": "openai/gpt-6-astra"}

SYSTEM = """You control a simulated SO-101 robot arm through tools. Task: {task}
You never send motor commands; you choose actions and targets, and a local IK controller executes them safely.

Frames and units:
- Robot base frame in metres: +x forward (away from the base), +y to the robot's left, z = height above the table.
- The overhead image is {px}x{px} pixels with a labelled grid every 40 px. Image up = +x, image left = +y.
- The red cube is 3 cm wide. The green open box is the "bowl": place the cube inside it.

Planning mode: {mode_text}

Typical plan: observe_scene(overhead) -> propose_grasp(image_point on the red cube) -> set_gripper(open) ->
move_end_effector(pregrasp) -> move_end_effector(grasp) -> set_gripper(closed) -> verify_grasp ->
move_end_effector(lift to z≈0.075) -> place_object(image_point at the centre of the green box) -> check_task_success
-> finish_episode.
Rules:
- Point at the object's centre in the most recent overhead image. Pixel coordinates are (u = column, v = row).
- If a tool returns ok=false, read the error code and adjust (different point, lower/closer target) instead of repeating it.
- If verify_grasp says holding_object=false: open the gripper, observe again, re-propose the grasp and retry.
- Call one tool at a time. Be brief in text. Finish when check_task_success is true or you cannot make progress."""

MODE_TEXT = {
    "camera": "camera-grounded. You only get camera images and robot proprioception; locate objects by pointing at "
              "pixels in the overhead image.",
    "privileged": "PRIVILEGED-STATE PLANNING. observe_scene also returns simulator ground-truth object poses; "
                  "you may use them directly.",
}

PHASES = {
    "observe_scene": "Observing scene", "get_robot_state": "Reading robot state",
    "propose_grasp": "Selecting grasp", "set_gripper": None, "verify_grasp": "Verifying grasp",
    "place_object": "Moving to bowl and releasing", "check_task_success": "Verifying task success",
    "finish_episode": "Finishing",
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
    provider: str  # insforge (gateway-provisioned OpenRouter key) | openrouter (self-hosted InsForge key)
    api_key: str
    model: str = ""
    mode: str = "camera"
    max_calls: int = 16
    max_usd: float = 1.50
    reasoning_effort: str = "low"
    timeout_s: float = 120.0
    endpoint: str = ""  # test hook: CORTEX_PLANNER_BASE_URL; recorded on every attempt

    def __post_init__(self) -> None:
        self.model = self.model or MODEL_IDS[self.provider]
        self.endpoint = self.endpoint or ENDPOINTS[self.provider]


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


class AstraPlanner:
    def __init__(self, cfg: PlannerConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.http = httpx.Client(timeout=cfg.timeout_s, transport=transport)

    def _complete(self, messages: list[dict[str, Any]], usage: Usage) -> dict[str, Any]:
        if usage.calls >= self.cfg.max_calls:
            raise BudgetExceeded(f"model-call budget of {self.cfg.max_calls} reached")
        if usage.cost_usd >= self.cfg.max_usd:
            raise BudgetExceeded(f"episode cost budget ${self.cfg.max_usd:.2f} reached")
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "tools": tool_schemas(),
                                "tool_choice": "auto", "parallel_tool_calls": False,
                                "max_completion_tokens": 600}  # tool calls are ~100-300 tokens; OpenRouter reserves credit for the max
        body["reasoning"] = {"effort": self.cfg.reasoning_effort}  # OpenRouter's unified reasoning param
        body["usage"] = {"include": True}  # provider-reported cost
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
            usage.cost_source = "provider-reported (OpenRouter usage.cost)"
        else:
            cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
            p = PRICES_PER_M["gpt-6-astra"]
            usage.cost_usd += ((pin - cached) * p["input"] + cached * p["cached"] + pout * p["output"]) / 1e6
        if data.get("model"):
            usage.models_reported.add(data["model"])
        return data["choices"][0]["message"]

    def run_episode(self, session: ManipulationSession, task: str,
                    emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        usage = Usage()
        trace: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM.format(task=task, px=PLAN_PX, mode_text=MODE_TEXT[self.cfg.mode])},
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
                    result, img = session.call(name, args)
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
                        "result": {k: v for k, v in result.items() if k not in ("wall_ms",)},
                        "sim_step": len(session.rec), "model_call": usage.calls}
                trace.append(step)
                emit({"type": "agent_step", **step})
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)[:4000]})
                if img is not None:
                    images.append((name, img))
            for name, img in images:
                shown = annotate_overhead(img) if session.last_overhead is img else img
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"Image returned by {name} ({img.shape[1]}x{img.shape[0]} px)."},
                    {"type": "image_url", "image_url": {"url": to_data_url(shown)}}]})
        return {"trace": trace, "usage": usage.as_dict(), "finished": finished, "model_claimed_success": claim,
                "model_summary": summary, "stop_reason": stop_reason, "recoveries": recoveries,
                "counters": dict(session.counters)}
