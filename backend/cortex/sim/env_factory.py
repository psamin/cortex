"""One place that decides which simulation a worker runs: the project's intake scene or the upstream tabletop.

Services put ``scene_payload(...)`` (see services/intake_service.py) into a worker's config under
``"scene"``. Workers call ``make_env(cfg.get("scene"), ...)`` and ``oracle(cfg.get("scene"))``.
Without a scene payload, everything behaves exactly as before (the cube task on the tabletop).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .so101 import PickPlaceEnv

# The intake skill (side grasps especially) needs far more control steps than the cube oracle's 240.
SCENE_MAX_STEPS = 2400


def make_env(scene: dict[str, Any] | None, episode_steps: int) -> PickPlaceEnv:
    if not scene:
        return PickPlaceEnv(episode_steps=episode_steps)
    from .scene import SceneInfo
    from .scene_env import SceneEnv

    info = SceneInfo.from_dict(scene["info"], Path(scene["data_dir"]))
    return SceneEnv(info, episode_steps=max(episode_steps, scene.get("max_steps", SCENE_MAX_STEPS)))


def max_steps(scene: dict[str, Any] | None, default: int) -> int:
    return max(default, scene.get("max_steps", SCENE_MAX_STEPS)) if scene else default


def oracle(scene: dict[str, Any] | None):
    """The scripted demonstrator for this simulation (privileged state; labelled an oracle)."""
    if not scene:
        from .policies import ExpertPolicy

        return ExpertPolicy()
    from .pick_place_skill import PickPlacePolicy

    obj = scene.get("object", {})
    return PickPlacePolicy(height=obj.get("height"), grasp_width=obj.get("grasp_width"))


def digest(scene: dict[str, Any] | None) -> str | None:
    return scene["info"].get("digest") if scene else None
