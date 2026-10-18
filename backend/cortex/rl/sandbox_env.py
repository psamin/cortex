"""A sandbox: domain randomization centred on one weak condition, and the shaped return.

Each seed draws its own object offset around the weak branch's offset, its own light scale,
and friction/mass jitter on the task object (restored afterwards), so RL can't overfit one
scene. The return is a weighted sum of measured terms:
  success        1 if the simulator's success test passes
  lifted         1 if the object rose above 4.5 cm
  held_at_carry  1 if it rose above 6 cm (it survived the lift into the carry)
  place_dist     1 - min(1, final object-target distance / 0.3 m)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.so101 import PickPlaceEnv, Variation
from .types import SandboxEnv


@dataclass
class Draw:
    var: Variation
    friction_scale: float
    mass_scale: float


def draw(env_spec: SandboxEnv, seed: int) -> Draw:
    rng = np.random.default_rng(seed)
    j = env_spec.offset_jitter_m
    ox = env_spec.cube_offset[0] + rng.uniform(-j, j)
    oy = env_spec.cube_offset[1] + rng.uniform(-j, j)
    lo, hi = env_spec.light_scales
    var = Variation(cube_region=env_spec.condition.cube_region, lighting=env_spec.condition.lighting,
                    cube_offset=(float(ox), float(oy)), light_scale=float(rng.uniform(lo, hi)))
    f = 1 + rng.uniform(-env_spec.friction_jitter, env_spec.friction_jitter)
    m = 1 + rng.uniform(-env_spec.mass_jitter, env_spec.mass_jitter)
    return Draw(var, float(f), float(m))


class Physics:
    """Applies and restores the friction/mass draw on the task object's geoms and body."""

    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.saved: tuple[np.ndarray, float] | None = None

    def apply(self, d: Draw) -> None:
        w = self.env.world
        geoms = np.flatnonzero(w.model.geom_bodyid == self.env.cube)
        if self.saved is None:
            self.saved = (w.model.geom_friction[geoms].copy(), float(w.model.body_mass[self.env.cube]))
        fr, mass = self.saved
        w.model.geom_friction[geoms] = fr * d.friction_scale
        w.model.body_mass[self.env.cube] = mass * d.mass_scale


def episode_return(ep: dict, weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    terms = {
        "success": float(ep["success"]),
        "lifted": float(ep["max_cube_z"] > 0.045),
        "held_at_carry": float(ep["max_cube_z"] > 0.06),
        "place_dist": float(1 - min(1.0, ep["final_cube_box_dist"] / 0.3)),
    }
    total = sum(weights.get(k, 0.0) * v for k, v in terms.items())
    return total / max(sum(weights.values()), 1e-9), terms
