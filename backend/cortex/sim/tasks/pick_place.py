"""Cube pick-and-place as a hand-built sim task (issue #309).

The scene is the existing SO-101 tabletop (``PickPlaceEnv``: the EnvHub cube and box), so nothing is built here.
The seed picks the cube's start region (left / center / right, in turn), its jitter and yaw, and the lighting.
The oracle is the scripted cube expert (``ScriptedExpert``) on privileged state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..expert import ScriptedExpert
from ..so101 import IMG_H, IMG_W, REGION_XY, PickPlaceEnv, Variation
from .base import LIGHT_RANGE

REGIONS = list(REGION_XY)  # left, center, right


class Env(PickPlaceEnv):
    name = "pick_place"
    instruction = "Pick up the cube and place it in the box"
    max_steps = 400

    def __init__(self, height: int = IMG_H, width: int = IMG_W):
        super().__init__(height=height, width=width, episode_steps=self.max_steps, instruction=self.instruction)

    def reset(self, seed: int, var: Variation | None = None) -> dict[str, Any]:
        if var is None:
            var = Variation(cube_region=REGIONS[seed % len(REGIONS)],
                            light_scale=float(np.random.default_rng(seed).uniform(*LIGHT_RANGE)))
        return super().reset(seed, var)

    def success(self) -> bool:
        return self.in_box()

    def metrics(self) -> dict[str, Any]:
        return {"lifted": bool(self.max_cube_z > 0.045)}  # the reward's lift threshold


class Oracle:
    """``ScriptedExpert`` on the oracle protocol. Its plan ends with release and retreat above the box; it commands
    absolute joint targets from the plan, so run.py's DART noise on the executed command doesn't stall it."""

    name = "scripted_expert (oracle, privileged state)"
    needs_pixels = False

    def reset(self, env, obs, seed):
        self.ex = ScriptedExpert(env, np.random.default_rng(seed))
        self.ex.plan(np.asarray(obs["agent_pos"], dtype=np.float64))

    def act(self, obs):
        return self.ex.act()

    @property
    def done(self) -> bool:
        return self.ex.done
