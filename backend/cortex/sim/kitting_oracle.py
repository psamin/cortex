"""Scripted oracle demonstrations for the kitting benchmark (stages 1-2).

The oracle is ``PickPlaceSkill`` (home -> pick -> place -> home) aimed at the episode's component and slot.
It reads privileged simulator state, so every episode it produces is labelled PROVENANCE. Episodes run
through ``sim_worker.run_episode`` and are saved in the sim worker's layout (episode.npz, front.mp4,
thumb.jpg, result.json), so the dataset builder reads them like any other oracle demo.

    python -m cortex.sim.kitting_oracle measure --stage 1 --seeds 0:20 --out DIR   # headless success rate
    python -m cortex.sim.kitting_oracle record --stage 2 --seeds 0:10 --out DIR    # rendered demos
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

from .kitting import COMPONENTS, KittingEnv, check_success
from .pick_place_skill import PickPlacePolicy
from .so101 import FPS, configure_gl

PROVENANCE = "scripted oracle (privileged state)"
MAX_STEPS = 1500


class KittingOracle(PickPlacePolicy):
    """``run_episode``-compatible oracle that sizes the skill to the episode's component."""

    name = PROVENANCE

    def __init__(self) -> None:
        # release_z 0 lowers to the skill's floor (object height + 1 cm), the shortest drop into the slot.
        super().__init__(params={"release_z": 0.0})

    def reset(self, env: KittingEnv, obs, seed):
        o = COMPONENTS[env.pick]
        self.height, self.width = o.height, o.grasp_width
        super().reset(env, obs, seed)


def provenance(env: KittingEnv) -> dict[str, Any]:
    t = env.task
    assert t is not None
    return {"planner": PROVENANCE, "mode": "privileged", "skill": "pick_place_skill", "benchmark": "kitting",
            "stage": t.stage, "kit": t.kit, "component": t.component, "slot": t.slot, "instruction": t.instruction,
            "lighting": t.lighting}


def _episode(env: KittingEnv, seed: int, max_steps: int, preview, record: bool) -> dict[str, Any]:
    from ..workers.sim_worker import run_episode

    ep = run_episode(env, KittingOracle(), seed, None, max_steps, preview, record)  # type: ignore[arg-type]
    _, reason = check_success(env, env.task)  # type: ignore[arg-type]  # same final state as ep["success"]
    return {**ep, "reason": reason, "provenance": provenance(env)}


def measure(stage: int, seeds: list[int], scene_dir: Path, max_steps: int = MAX_STEPS) -> list[dict[str, Any]]:
    """Headless oracle episodes (no rendering): seed, success, reason, steps per episode."""
    env = KittingEnv(stage, scene_dir, episode_steps=max_steps)
    out = []
    try:
        for seed in seeds:
            with redirect_stdout(StringIO()):  # run_episode prints worker progress lines
                ep = _episode(env, seed, max_steps, None, record=False)
            out.append({k: ep[k] for k in ("seed", "success", "reason", "steps", "failure_mode")} |
                       {"kit": ep["provenance"]["kit"]})
    finally:
        env.close()
    return out


def record(stage: int, seeds: list[int], out_dir: Path, max_steps: int = MAX_STEPS) -> list[dict[str, Any]]:
    """Rendered demos, one directory per seed, in the sim worker's episode layout."""
    from PIL import Image

    from ..workers.sim_worker import Preview, write_mp4

    out_dir.mkdir(parents=True, exist_ok=True)
    env = KittingEnv(stage, out_dir / "scene", episode_steps=max_steps)
    env.reset(seeds[0])
    preview = Preview(env, out_dir, live=False)
    results = []
    try:
        for seed in seeds:
            out = out_dir / f"stage{stage}_seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            ep = _episode(env, seed, max_steps, preview, record=True)
            buf, video = ep.pop("buf"), ep.pop("video")
            n = len(buf["action"])
            np.savez_compressed(  # same arrays as sim_worker.cmd_record
                out / "episode.npz",
                front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
                state=np.stack(buf["state"]).astype(np.float32), action=np.stack(buf["action"]).astype(np.float32),
                reward=np.asarray(buf["reward"], np.float32), timestamp=(np.arange(n) / FPS).astype(np.float32),
                action_low=np.r_[np.rad2deg(env.world.lo[:5]), 0].astype(np.float32),
                action_high=np.r_[np.rad2deg(env.world.hi[:5]), 100].astype(np.float32))
            write_mp4(video, out / "front.mp4")
            Image.fromarray(video[min(len(video) - 1, 45)]).save(out / "thumb.jpg", quality=85)
            res = {**ep, "num_frames": n, "fps": FPS, "out_dir": str(out)}
            (out / "result.json").write_text(json.dumps(res))
            results.append(res)
    finally:
        preview.close()
        env.close()
    return results


def _seeds(s: str) -> list[int]:
    a, _, b = s.partition(":")
    return list(range(int(a), int(b))) if b else [int(a)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("measure", "record"))
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--seeds", default="0:20", help="a:b (half-open range) or one seed")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    configure_gl()
    if a.cmd == "record":
        res = record(a.stage, _seeds(a.seeds), a.out)
        res = [{k: r[k] for k in ("seed", "success", "reason", "num_frames", "out_dir")} for r in res]
    else:
        res = measure(a.stage, _seeds(a.seeds), a.out)
    print(json.dumps({"type": "result", "stage": a.stage, "episodes": res,  # one JSON line, like the workers
                      "success_rate": sum(r["success"] for r in res) / max(1, len(res))}))


if __name__ == "__main__":
    main()
