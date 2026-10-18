"""How the SO-101 grasps an object of a given height, shared by the scene, feasibility and tools.

Short objects are grasped from above. Tall ones (bottles, cans) are grasped from the side:
a top-down approach over them is beyond the arm's reach. A horizontal side grasp only reaches far
from the base: measured on the Menagerie SO-101, about r >= 0.30 m at z = 0.07 and
r >= 0.26 m at z >= 0.10. Tilting the approach 40 degrees down brings it in to about
r = 0.20 m, so side grasps search tilts of 0, 20 and 40 degrees.
"""

from __future__ import annotations

import numpy as np

from .expert import side_approach

SIDE_GRASP_MIN_H = 0.07  # objects taller than this are grasped from the side
TOP_MARGIN = 0.025  # side grasps hold this far below the object's top
FINGER_DEPTH = 0.04  # how far down an object the fingertips reach from above
SIDE_MIN_RADIUS = 0.30  # purely horizontal side grasps reach from here outward
# Side-grasp approach tilts tried in order: tilting down brings the grasp in to r ~0.20 m.
SIDE_PITCHES = (0.0, 20.0, 40.0)


def grasp_mode(height: float) -> str:
    return "side" if height > SIDE_GRASP_MIN_H else "top"


def grasp_height(height: float) -> float:
    """Gripper-site height for the grasp. A top grasp holds near the object's top. A side grasp holds
    TOP_MARGIN below the top, capped at 0.12 m."""
    if grasp_mode(height) == "side":
        # Near the top: the wrist camera mount sits above the jaws, so little of the object may stick
        # out above them; the centre of mass then hangs below the grasp, which carries steadily.
        return float(np.clip(height - TOP_MARGIN, 0.03, 0.12))
    return max(height / 2, height - FINGER_DEPTH)


def ik_seeds(xyz: np.ndarray, home: np.ndarray) -> list[np.ndarray]:
    """Initial joint guesses that cover elbow-up and elbow-down solutions, aimed at the target."""
    pan = float(np.rad2deg(np.arctan2(xyz[1], xyz[0])))
    seeds = [np.asarray(home, dtype=np.float64)]
    seeds += [np.array([pan, a, e, w, 0.0, 30.0]) for a in (-60, -20, 20, 60) for e in (-60, 0, 60)
              for w in (-60, 0, 60)]
    return seeds


def approaches(xy: np.ndarray, mode: str) -> list[np.ndarray | None]:
    """Approach axes to try for a grasp at xy: straight down (top), or the side tilts."""
    return [None] if mode == "top" else [side_approach(xy, p) for p in SIDE_PITCHES]
