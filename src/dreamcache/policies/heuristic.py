"""Non-LLM policies.

- SimBFSPolicy        SIMULATOR-ASSISTED: BFS on the full grid (privileged). Used as the
                      sandbox rollout policy, the oracle reference policy, and baseline I-sim.
- ObsBFSPolicy        OBSERVATION-ONLY: BFS inside the 7x7 egocentric view + exploration.
                      Deployable; baseline I-obs.
- RandomPolicy        uniform over legal actions.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from dreamcache.environments.observations import AX, AY, cell

DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
LEFT, RIGHT, FORWARD, PICKUP, TOGGLE = 0, 1, 2, 3, 5


def _turn_toward(cur_dir: int, need_dir: int) -> int:
    if cur_dir == need_dir:
        return FORWARD
    return LEFT if need_dir == (cur_dir - 1) % 4 else RIGHT


def _bfs(start, goal, passable):
    """Shortest path start->goal on 4-connected cells. goal is always enterable."""
    prev = {start: None}
    q = deque([start])
    while q:
        p = q.popleft()
        if p == goal:
            path = []
            while p is not None:
                path.append(p)
                p = prev[p]
            return path[::-1]
        for dx, dy in DIR_VEC.values():
            n = (p[0] + dx, p[1] + dy)
            if n not in prev and (n == goal or passable(n)):
                prev[n] = p
                q.append(n)
    return None


class RandomPolicy:
    name = "random"

    def __init__(self, legal, seed=0):
        self.legal = legal
        self.rng = np.random.default_rng(seed)

    def act_sim(self, sim) -> int:
        return int(self.rng.choice(self.legal))


class SimBFSPolicy:
    """Privileged shortest-path planner over the full simulator state."""

    name = "sim_bfs"
    privileged = True

    def _find(self, u, typ):
        for x in range(u.width):
            for y in range(u.height):
                c = u.grid.get(x, y)
                if c is not None and c.type == typ:
                    return (x, y), c
        return None, None

    def target(self, u):
        goal, _ = self._find(u, "goal")
        door_pos, door = self._find(u, "door")
        if door is None or door.is_open:
            return goal, None
        carrying_key = u.carrying is not None and u.carrying.type == "key"
        if carrying_key:
            return door_pos, TOGGLE
        key_pos, _ = self._find(u, "key")
        return key_pos, PICKUP

    def act_sim(self, sim) -> int:
        u = sim.unwrapped
        tgt, interact = self.target(u)
        start = tuple(int(v) for v in u.agent_pos)

        def free(p, avoid_balls=True):
            x, y = p
            if not (0 <= x < u.width and 0 <= y < u.height):
                return False
            c = u.grid.get(x, y)
            if c is None or c.type == "goal":
                return True
            if c.type == "door":
                return c.is_open
            if c.type == "ball":
                return not avoid_balls
            return False

        path = _bfs(start, tgt, free) or _bfs(start, tgt, lambda p: free(p, False))
        fx, fy = u.front_pos
        front = u.grid.get(fx, fy)
        if path is None or len(path) < 2:
            return LEFT
        nxt = path[1]
        if nxt == tgt and interact is not None:
            if (fx, fy) == tgt:
                return interact
        need = next(d for d, v in DIR_VEC.items() if (start[0] + v[0], start[1] + v[1]) == nxt)
        a = _turn_toward(int(u.agent_dir), need)
        if a == FORWARD and front is not None and front.type not in ("goal",) and not (
                front.type == "door" and front.is_open):
            return RIGHT  # never walk into a blocking cell
        return a


class ObsBFSPolicy:
    """Observation-only heuristic: plan inside the visible 7x7 window, else explore."""

    name = "obs_bfs"
    privileged = False

    def __init__(self, seed=0, explore_forward_p=0.75):
        self.rng = np.random.default_rng(seed)
        self.p = explore_forward_p

    def act(self, obs) -> int:
        img = obs["image"]
        carrying = cell(img, AX, AY)[0]
        targets = []
        door_locked = door_open = False
        for x in range(7):
            for y in range(7):
                t, _, s = cell(img, x, y)
                if t == "door":
                    door_locked, door_open = s == 2, s == 0
        for x in range(7):
            for y in range(7):
                t, _, s = cell(img, x, y)
                if t == "goal":
                    targets.append(((x, y), None, 0))
                elif t == "door" and s != 0 and carrying == "key":
                    targets.append(((x, y), TOGGLE, 1))
                elif t == "key" and carrying != "key" and not door_open:
                    targets.append(((x, y), PICKUP, 2 if door_locked else 3))
        targets.sort(key=lambda t: t[2])

        def free(p):
            x, y = p
            if not (0 <= x < 7 and 0 <= y < 7):
                return False
            t, _, s = cell(img, x, y)
            return t in ("empty", "floor", "goal") or (t == "door" and s == 0)

        start = (AX, AY)
        for tgt, interact, _ in targets:
            path = _bfs(start, tgt, free)
            if path is None or len(path) < 2:
                continue
            nxt = path[1]
            if nxt == tgt and interact is not None and nxt == (AX, AY - 1):
                return interact
            # view frame: "up" (0,-1) is the facing direction (dir index 3 here)
            need = next(d for d, v in DIR_VEC.items() if (AX + v[0], AY + v[1]) == nxt)
            return _turn_toward(3, need)
        # explore
        if free((AX, AY - 1)) and self.rng.random() < self.p:
            return FORWARD
        return int(self.rng.choice([LEFT, RIGHT]))
