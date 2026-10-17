"""Observation-only views of MiniGrid state.

Everything here reads ONLY the agent's partial egocentric observation
(obs["image"], obs["direction"], obs["mission"]) plus the agent's own action
history. Deployable policies and the difficulty detector use these functions,
never the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT

from dreamcache.environments.minigrid_adapter import ACTION_NAMES

# Egocentric view: agent sits at (3, 6) facing "up" (towards y = 0).
AX, AY = 3, 6
CHARS = {"unseen": "?", "empty": ".", "wall": "#", "floor": ".", "goal": "G",
         "ball": "O", "key": "K", "box": "B", "lava": "L", "agent": "A"}
DIRS = {0: "east", 1: "south", 2: "west", 3: "north"}
ACTION_HELP = {
    "left": "turn 90 degrees left (stay in place)",
    "right": "turn 90 degrees right (stay in place)",
    "forward": "move one cell in the facing direction",
    "pickup": "pick up the object directly ahead",
    "toggle": "open/unlock the door directly ahead (needs matching key if locked)",
}
BLOCKING = {"wall", "ball", "box", "lava"}


def cell(img: np.ndarray, x: int, y: int) -> tuple[str, int, int]:
    t, c, s = img[x, y]
    return IDX_TO_OBJECT[int(t)], int(c), int(s)


def cell_char(img, x, y) -> str:
    t, _, s = cell(img, x, y)
    if t == "door":
        return {0: "_", 1: "d", 2: "D"}[s]
    return CHARS.get(t, "?")


@dataclass
class ObsFeatures:
    front: str            # type of the cell directly ahead
    front_blocked: bool   # forward would walk into wall/obstacle/closed door
    left: str
    right: str
    nearby_obstacles: int  # visible balls within Manhattan distance 2 of agent
    obstacles_near_front: int  # visible balls adjacent to the front cell
    safe_exits: int       # of {front, left, right} cells that are free
    goal_visible: bool
    goal_rel: tuple[int, int] | None  # (ahead, right)


def features(obs) -> ObsFeatures:
    img = obs["image"]
    front_t, _, front_s = cell(img, AX, AY - 1)
    blocked = front_t in BLOCKING or (front_t == "door" and front_s != 0) or front_t == "key"
    left_t = cell(img, AX - 1, AY)[0]
    right_t = cell(img, AX + 1, AY)[0]
    near, near_front, goal = 0, 0, None
    for x in range(7):
        for y in range(7):
            t = cell(img, x, y)[0]
            if t == "ball":
                if abs(x - AX) + abs(y - AY) <= 2:
                    near += 1
                if abs(x - AX) + abs(y - (AY - 1)) == 1:
                    near_front += 1
            if t == "goal":
                goal = (AY - y, x - AX)
    free = lambda t: t in ("empty", "floor", "goal")  # noqa: E731
    exits = sum(free(t) for t in (front_t, left_t, right_t))
    return ObsFeatures(front_t, blocked, left_t, right_t, near, near_front, exits,
                       goal is not None, goal)


def describe(obs, history: list[int], legal: list[int]) -> str:
    """Text prompt body describing the current observation."""
    img = obs["image"]
    rows = []
    for y in range(7):
        row = "".join("A" if (x, y) == (AX, AY) else cell_char(img, x, y) for x in range(7))
        rows.append(row)
    objs = []
    for x in range(7):
        for y in range(7):
            t, _, s = cell(img, x, y)
            if t in ("goal", "ball", "key", "door"):
                ahead, right = AY - y, x - AX
                side = f"{abs(right)} {'right' if right > 0 else 'left'}" if right else "0 sideways"
                extra = {0: " (open)", 1: " (closed)", 2: " (locked)"}[s] if t == "door" else ""
                name = "moving obstacle" if t == "ball" else t
                objs.append(f"- {name}{extra}: {ahead} ahead, {side}")
    f = features(obs)
    hist = ", ".join(ACTION_NAMES[a] for a in history[-6:]) or "none"
    acts = "\n".join(f"- {ACTION_NAMES[a]}: {ACTION_HELP[ACTION_NAMES[a]]}" for a in legal)
    return (
        f"Mission: {obs['mission']}\n"
        f"You are facing {DIRS[int(obs['direction'])]}.\n"
        "Your 7x7 view (A = you at bottom center, facing up the page; rows go far -> near):\n"
        + "\n".join(rows) + "\n"
        "Legend: # wall, . empty, ? not visible, G goal, O moving obstacle, K key, "
        "D locked door, d closed door, _ open door\n"
        "Visible objects (cells ahead, cells sideways):\n"
        + ("\n".join(objs) if objs else "- none") + "\n"
        f"Directly ahead: {f.front}. Left of you: {f.left}. Right of you: {f.right}.\n"
        f"Your recent actions: {hist}\n"
        f"Available actions:\n{acts}\n"
    )


# ---- BALROG-style text observation (prompt v4) ---------------------------------------
# Adapted from BartekCupial/Minigrid (Apache-2.0) `MiniGridEnv.gen_graph`, English path, as
# used by BALROG's BabyAI wrapper (balrog-ai/BALROG, MIT). Rewritten to take only the
# egocentric observation (no env access): walls straight ahead/left/right, then every
# visible non-empty, non-wall object with "N steps left/right and M steps forward".
_COLORS = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
_STATES = {0: "open", 1: "closed", 2: "locked"}


def balrog_descriptions(obs) -> list[str]:
    img = obs["image"]
    out = []
    t_carry, c_carry, _ = img[AX, AY]
    if IDX_TO_OBJECT[int(t_carry)] not in ("empty", "unseen", "agent", "wall", "floor"):
        out.append(f"You carry a {_COLORS[int(c_carry)]} {IDX_TO_OBJECT[int(t_carry)]}")

    def first_wall(cells, label):
        for dist, (x, y) in enumerate(cells, start=1):
            t = int(img[x, y][0])
            if t not in (0, 1):            # not unseen / empty
                if t == 2:
                    out.append(f"You see a wall {dist} step{'s' if dist > 1 else ''} {label}")
                return

    first_wall([(AX, y) for y in range(AY - 1, -1, -1)], "forward")
    first_wall([(x, AY) for x in range(AX - 1, -1, -1)], "left")
    first_wall([(x, AY) for x in range(AX + 1, 7)], "right")
    for x in range(7):
        for y in range(7):
            t, c, s = (int(v) for v in img[x, y])
            if t in (0, 1, 2) or (x, y) == (AX, AY):
                continue
            name = IDX_TO_OBJECT[t]
            if t == 4:
                art = "an" if _STATES[s] == "open" else "a"
                desc = f"You see {art} {_STATES[s]} {_COLORS[c]} {name} "
            else:
                desc = f"You see a {_COLORS[c]} {name} "
            dx, dy = x - AX, AY - y
            parts = []
            if dx == 0:
                parts.append((dy, "forward"))
            elif dy == 0:
                parts.append((abs(dx), "right" if dx > 0 else "left"))
            else:
                parts += [(abs(dx), "right" if dx > 0 else "left"), (dy, "forward")]
            desc += " and ".join(f"{d} step{'s' if d > 1 else ''} {lab}" for d, lab in parts)
            out.append(desc)
    return out


BALROG_ACTIONS = {"left": "turn to the left", "right": "turn to the right", "forward": "take one step forward",
                  "pickup": "pick up the object below you", "toggle": "manipulate the object in front of you"}


def balrog_prompt(obs, history, legal) -> str:
    """BALROG BabyAI instruction prompt (balrog/environments/babyai_text/__init__.py) + observation."""
    acts = ",\n".join(f"{ACTION_NAMES[a]}: {BALROG_ACTIONS[ACTION_NAMES[a]]}" for a in legal)
    hist = ", ".join(ACTION_NAMES[a] for a in history[-6:]) or "none"
    return (f"You are an agent playing a simple navigation game. Your goal is to {obs['mission']}. "
            "The following are the possible actions you can take in the game, followed by a short "
            f"description of each action:\n\n{acts}.\n\nTips:\n- Once the desired object you want to "
            "interact or pickup in front of you, you can use the 'toggle' action to interact with it.\n"
            "- It doesn't make sense to repeat the same action over and over if the observation doesn't change.\n\n"
            f"Observation:\n" + ("\n".join(balrog_descriptions(obs)) or "You see nothing special.") +
            f"\nYour recent actions: {hist}\n")
