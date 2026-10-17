"""Observable risk: uses only the egocentric observation and the proposed action."""
from dreamcache.environments.observations import features

FORWARD = 2


def observable_risk(obs, fast_action: int, family: str) -> float:
    f = features(obs)
    r = 0.0
    if fast_action == FORWARD and f.front_blocked:
        # Dynamic-Obstacles: walking into a wall/obstacle ends the episode.
        r = 1.0 if family == "dynamic" else 0.3
    if f.nearby_obstacles or f.obstacles_near_front:
        r = max(r, min(1.0, 0.35 * f.nearby_obstacles + 0.3 * f.obstacles_near_front))
        r = max(r, 0.5 * (1 - f.safe_exits / 3))
    return r
