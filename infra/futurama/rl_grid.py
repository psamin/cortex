"""Massively parallel RL in sim, as a video: CEM tunes the pick-and-place skill's corrections with 50 rollouts
per iteration running at once (one process per CPU core, MuJoCo physics on the CPU, EGL rendering on the GPU).

    python rl_grid.py <out_dir> [iterations] [population]

Same CEM update as cortex.rl.learner.cem (elites 20%, smoothing 0.7, std floor 2% of the range), with the
candidates of an iteration evaluated in parallel instead of one after another. Parameters and bounds are
cortex.rl.proposals.PARAMS; the skill is cortex.rl.skill.TunedSkill. The search starts from a deliberately
mis-calibrated skill, so what the video shows is the search fixing it. Everything drawn is measured here.
"""

import io
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TILE_W, TILE_H = 192, 144
BAD_INIT = {"grasp_dx": 0.011, "grasp_dy": -0.011, "grasp_dz": 0.007, "yaw_offset": 0.28, "release_dx": 0.018}
_ENV = None


def _init():
    global _ENV
    from cortex.sim.tasks import make

    _ENV, _ = make("pick_place", TILE_H, TILE_W)


def rollout(job):
    """One episode of the skill with these corrections: (jpeg frames, success, score, steps)."""
    from PIL import Image

    from cortex.rl.skill import TunedSkill

    params, seed = job
    env = _ENV
    obs = env.reset(seed)
    sk = TunedSkill(env, params)
    sk.plan(obs["agent_pos"])
    frames, after = [], 0
    for step in range(env.max_steps):
        if step % 2 == 0:
            buf = io.BytesIO()
            Image.fromarray(obs["pixels"]["front"]).save(buf, format="JPEG", quality=80)
            frames.append(buf.getvalue())
        obs, _, _, _, _ = env.step(sk.act())
        if sk.done:
            after += 1
            if after > 20:  # let the cube settle after the release
                break
    ok = bool(env.success())
    d = float(np.linalg.norm((env.cube_pos() - env.box_pos())[:2]))
    return frames, ok, (1.0 if ok else 0.0) - d, step + 1


def main():
    from cortex.rl.proposals import PARAMS

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    names = list(PARAMS)
    lo = np.array([PARAMS[n].lo for n in names])
    hi = np.array([PARAMS[n].hi for n in names])
    mean = np.array([np.clip(BAD_INIT.get(n, PARAMS[n].init), PARAMS[n].lo, PARAMS[n].hi) for n in names])
    std, floor = (hi - lo) / 4, (hi - lo) * 0.02
    n_elite = max(2, round(pop * 0.2))
    rng = np.random.default_rng(0)
    cores = len(os.sched_getaffinity(0))
    procs = min(pop, cores)
    t_all = time.time()
    with get_context("spawn").Pool(procs, initializer=_init) as pool:
        for it in range(iters):
            cands = np.clip(rng.normal(mean, std, size=(pop, len(names))), lo, hi)
            cands[0] = mean
            jobs = [(dict(zip(names, c.tolist())), 1000 + it * 1000 + k) for k, c in enumerate(cands)]
            t0 = time.time()
            res = pool.map(rollout, jobs, chunksize=1)
            wall = time.time() - t0
            scores = np.array([r[2] for r in res])
            wins = [r[1] for r in res]
            elite = cands[np.argsort(scores)[::-1][:n_elite]]
            mean = 0.7 * elite.mean(axis=0) + 0.3 * mean
            std = np.maximum(0.7 * elite.std(axis=0) + 0.3 * std, floor)
            stats = {"iteration": it + 1, "success": int(sum(wins)), "population": pop,
                     "mean_score": round(float(scores.mean()), 4), "rollout_wall_s": round(wall, 2),
                     "steps": int(sum(r[3] for r in res)), "mean": dict(zip(names, np.round(mean, 5).tolist()))}
            print("[rl]", json.dumps(stats), flush=True)
            np.save(out / f"iter{it + 1}.npy", np.array([{"frames": r[0], "ok": r[1], "score": r[2]} for r in res],
                                                        dtype=object), allow_pickle=True)
            (out / f"iter{it + 1}.json").write_text(json.dumps(stats))
    meta = {"iterations": iters, "population": pop, "processes": procs, "cores": cores,
            "gpu": os.environ.get("GPU_NAME", "GPU"), "rl_wall_s": round(time.time() - t_all, 1),
            "bad_init": BAD_INIT, "params": names}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("[rl] DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
