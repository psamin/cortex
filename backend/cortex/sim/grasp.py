"""How the SO-101 grasps an object of a given height, shared by the scene, feasibility and tools.

Short objects are grasped from above. Tall ones (bottles, cans) are grasped from the side:
a top-down approach over them is beyond the arm's reach. Side grasps only reach far from
the base. Measured on the Menagerie SO-101 (12 mm IK tolerance): about r >= 0.30 m at
z = 0.07 and r >= 0.26 m at z >= 0.10. So tall objects need starts and zones out there.
"""

from __future__ import annotations

import numpy as np

SIDE_GRASP_MIN_H = 0.07  # objects taller than this are grasped from the side
FINGER_DEPTH = 0.04  # how far down an object the fingertips reach from above
SIDE_MIN_RADIUS = 0.30  # default starts for side grasps sit at least this far from the base


def grasp_mode(height: float) -> str:
    return "side" if height > SIDE_GRASP_MIN_H else "top"


def grasp_height(height: float) -> float:
    """Gripper-site height for the grasp. A top grasp holds near the object's top. A side grasp holds
    above mid height, which is easier to reach, capped at 0.10 m."""
    if grasp_mode(height) == "side":
        return float(np.clip(height * 0.6, 0.03, 0.10))
    return max(height / 2, height - FINGER_DEPTH)


def ik_seeds(xyz: np.ndarray, home: np.ndarray) -> list[np.ndarray]:
    """Initial joint guesses that cover elbow-up and elbow-down solutions, aimed at the target."""
    pan = float(np.rad2deg(np.arctan2(xyz[1], xyz[0])))
    seeds = [np.asarray(home, dtype=np.float64)]
    seeds += [np.array([pan, a, e, w, 0.0, 30.0]) for a in (-60, -20, 20, 60) for e in (-60, 0, 60)
              for w in (-60, 0, 60)]
    return seeds
