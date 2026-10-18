"""One short, human line for what the studio agent is doing right now (the chat's live status line).

Built from what the agent actually does (tool calls, its plan, its thinking), so a person can see a
turn is moving. Lines are short and end with an ellipsis; they never carry ids or JSON.
"""

from __future__ import annotations

import re
from typing import Any

MAX = 110

_FIXED = {
    "create_project": "Setting up the project",
    "get_project_state": "Reading the project",
    "get_next_actions": "Reading the project",
    "list_episodes": "Looking at the demos",
    "analyze_dataset": "Checking demo quality",
    "get_failure_clusters": "Grouping the failures",
    "recommend_demonstrations": "Ranking what to collect next",
    "create_dataset_version": "Building the dataset",
    "launch_training_job": "Starting training",
    "compare_experiments": "Comparing runs",
    "get_job": "Checking progress",
    "get_collection_progress": "Checking progress",
    "get_demonstration_results": "Reading the demo results",
    "import_demonstrations": "Importing demonstrations",
    "get_infrastructure": "Checking where training runs",
    "set_infrastructure": "Choosing where training runs",
    "start_intake": "Describing the task and the scene",
    "update_intake": "Describing the task and the scene",
    "get_intake_status": "Checking the task description",
    "build_simulation": "Building the simulation",
    "get_policy_weaknesses": "Finding where the policy is weak",
    "propose_rl_sandboxes": "Proposing RL sandboxes",
    "accept_rl_sandboxes": "Starting RL sandboxes",
    "get_rl_sandbox_status": "Checking the RL sandboxes",
    "stop_rl_sandbox": "Stopping an RL sandbox",
    "get_session_link": "Getting the session link",
    "robot_start_episode": "Driving the sim arm: starting an episode",
    "robot_act": "Driving the sim arm",
    "robot_finish_episode": "Driving the sim arm: finishing the episode",
    "start_autopilot": "Starting autopilot",
}


def tool_line(tool: str, args: dict[str, Any] | None = None) -> str:
    """What a tool call is doing, in words, e.g. "Collecting 20 demos in sim…"."""
    a = args or {}
    if tool == "start_agent_collection":
        line = f"Collecting {a.get('num_demos', 20)} demos in sim"
    elif tool == "create_policy_project":
        line = f"Setting up the project and collecting {a.get('num_demos', 6)} demos in sim"
    elif tool == "record_demonstrations":
        n = sum(int(c.get("count", 1)) for c in a.get("conditions") or [] if isinstance(c, dict))
        line = f"Recording {n} demos in sim" if n else "Recording demos in sim"
    elif tool in ("propose_training_job", "train_from_agent_demonstrations"):
        policy = str(a.get("policy_type", "act")).upper()
        line = (f"Building the dataset and preparing {policy} training" if tool.startswith("train_from")
                else f"Preparing {policy} training")
    elif tool in ("launch_simulation_evaluation", "evaluate_learned_policy"):
        line = f"Testing the policy in 3 conditions, {a.get('episodes_per_branch', 10)} runs each"
    elif tool == "improve_policy":
        line = f"Targeting the weakest condition with {a.get('max_new_demos', 5)} new demos"
    elif tool == "capture_frame":
        line = f"Looking at the {str(a.get('camera', 'top')).replace('_cam', '')} camera"
    else:
        line = _FIXED.get(tool) or tool.replace("_", " ").capitalize()
    return line + "…"


def todo_line(todos: list[dict[str, Any]]) -> str | None:
    """The step the agent's own checklist says it is on."""
    cur = next((t for t in todos if t.get("status") == "in_progress"), None)
    if cur:
        return _clip(str(cur.get("activeForm") or cur.get("content") or "")) + "…"
    if todos and all(t.get("status") == "completed" for t in todos):
        return "Wrapping up…"
    if todos:
        return f"Planned {len(todos)} steps…"
    return None


def thought_line(text: str) -> str | None:
    """The first sentence of the agent's thinking, as a short "what I'm planning" line."""
    text = " ".join(str(text).split())
    if not text:
        return None
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0].rstrip(".!?:")
    return _clip(first) + "…"


def _clip(s: str) -> str:
    s = s.strip()
    return s if len(s) <= MAX else s[: MAX - 1].rsplit(" ", 1)[0]
