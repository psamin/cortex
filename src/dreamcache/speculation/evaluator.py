"""Rollout utility functions.

env      : success_bonus*success - failure_penalty*failure - step_cost*steps + sum(env reward)
           Unfinished rollouts get terminal value 0. Uses no privileged knowledge beyond
           what the sandbox simulator itself emits.
shaped   : env + progress terms. SIMULATOR-ASSISTED: progress is the decrease in BFS
           remaining-plan distance (on the full grid), plus one-off DoorKey
           subgoal bonuses (key picked up during the rollout, door opened during the
           rollout). Unfinished rollouts get terminal value -dist_coef * remaining distance.
           Key possession is rewarded only at the moment of acquisition, never per step,
           so it cannot be farmed by looping.
"""
from __future__ import annotations

from dataclasses import dataclass

from dreamcache.policies.heuristic import SimBFSPolicy, _bfs


@dataclass
class EvalConfig:
    kind: str = "shaped"          # "env" | "shaped"
    success_bonus: float = 10.0
    failure_penalty: float = 10.0
    step_cost: float = 0.01
    progress_coef: float = 0.2
    dist_coef: float = 0.1
    key_bonus: float = 1.0
    door_bonus: float = 2.0


_planner = SimBFSPolicy()


def _free_fn(u):
    def free(p):
        x, y = p
        if not (0 <= x < u.width and 0 <= y < u.height):
            return False
        c = u.grid.get(x, y)
        return c is None or c.type in ("goal", "ball") or (c.type == "door" and c.is_open)
    return free


def subgoal_distance(sim) -> float:
    """Remaining plan length through all outstanding subgoals (agent -> key -> door -> goal).

    Phase-consistent: picking up the key or opening the door never increases it, so the
    progress term cannot penalize completing a subgoal.
    """
    u = sim.unwrapped
    free = _free_fn(u)
    goal, _ = _planner._find(u, "goal")
    door_pos, door = _planner._find(u, "door")
    key_pos, _ = _planner._find(u, "key")
    carrying = u.carrying is not None and u.carrying.type == "key"
    waypoints = []
    if door is not None and not door.is_open:
        if not carrying and key_pos is not None:
            waypoints.append(key_pos)
        waypoints.append(door_pos)
    waypoints.append(goal)
    total, cur = 0.0, tuple(int(v) for v in u.agent_pos)
    for w in waypoints:
        path = _bfs(cur, w, lambda p, w=w: free(p) or p == door_pos)
        total += (len(path) - 1) if path else 50.0
        cur = w
    return total


def snapshot(sim) -> dict:
    u = sim.unwrapped
    door = next((c for c in u.grid.grid if c is not None and c.type == "door"), None)
    return {"carrying_key": u.carrying is not None and u.carrying.type == "key",
            "door_open": bool(door.is_open) if door is not None else True,
            "dist": None}


def utility(cfg: EvalConfig, start: dict, end_sim, total_reward: float, steps: int,
            success: bool, failure: bool, d0: float | None) -> float:
    u = (cfg.success_bonus * success - cfg.failure_penalty * failure
         - cfg.step_cost * steps + total_reward)
    if cfg.kind == "env":
        return u
    end = snapshot(end_sim)
    if end["carrying_key"] and not start["carrying_key"]:
        u += cfg.key_bonus
    if end["door_open"] and not start["door_open"]:
        u += cfg.door_bonus
    if not (success or failure):
        d1 = subgoal_distance(end_sim)
        u += cfg.progress_coef * ((d0 or 0.0) - d1) - cfg.dist_coef * d1
    return u
