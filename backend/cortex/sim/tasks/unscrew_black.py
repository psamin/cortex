"""Unscrewing policy 2 (issue #161): the claw-unscrew task with the bolt always in black oxide.

Same scene, physics, oracle and per-seed variation as ``unscrew``; only the bolt finish is fixed, so it stands
out against the brushed-aluminium fixture for people watching and looks the same in every demo for the policy.
"""

from __future__ import annotations

from . import unscrew

FINISH = "black oxide"


class Env(unscrew.Env):
    name = "unscrew_black"

    def reset_task(self, rng):
        super().reset_task(rng)  # keeps every other per-seed variation and the rng sequence
        rgb, spec, shine = unscrew.FINISHES[FINISH]
        m = self.world.model
        m.mat_rgba[self.finish_mat] = (*rgb, 1)
        m.mat_specular[self.finish_mat] = spec
        m.mat_shininess[self.finish_mat] = shine
        self.finish = FINISH


Oracle = unscrew.Oracle
