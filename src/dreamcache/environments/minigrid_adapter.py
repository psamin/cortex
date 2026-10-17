"""Environment construction and per-environment metadata."""
from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid envs)

ACTION_NAMES = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}

# Actions an agent may sensibly choose. drop/done are excluded for DoorKey because
# they are never useful there; this is stated in the report.
LEGAL_ACTIONS = {
    "dynamic": [0, 1, 2],
    "doorkey": [0, 1, 2, 3, 5],
}


def env_family(env_id: str) -> str:
    if "Dynamic-Obstacles" in env_id:
        return "dynamic"
    if "DoorKey" in env_id:
        return "doorkey"
    raise ValueError(f"unsupported env {env_id}")


def legal_actions(env_id: str) -> list[int]:
    return LEGAL_ACTIONS[env_family(env_id)]


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    return env, obs


def is_success(reward: float, terminated: bool) -> bool:
    return bool(terminated and reward > 0)


def is_failure(reward: float, terminated: bool) -> bool:
    """Dynamic-Obstacles: collision gives reward -1 and terminates."""
    return bool(terminated and reward < 0)
