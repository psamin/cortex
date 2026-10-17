"""Agent demonstration worker: GPT-6 Astra plans, MuJoCo + IK executes, we record.

    CORTEX_PLANNER_API_KEY=... python -m cortex.workers.agent_worker <config.json>

config: {task, mode, planner{provider, model?, max_calls, max_usd, reasoning_effort},
         max_total_usd, live_dir, attempts: [{seed, scenario{cube_region, lighting, cube_offset, box_offset},
         out_dir}]}
For every attempt it writes episode.npz (same layout as oracle demos: synchronized
front/wrist frames, joint state, joint-target actions), front.mp4, thumb.jpg and
trace.json, then emits {"type": "attempt", ...}. Success is the simulator's verdict.
The API key comes from the environment, never from the config file.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.manipulation import ManipulationSession
from ..sim.so101 import FPS, PickPlaceEnv, Variation, configure_gl
from .sim_worker import PREVIEW_H, PREVIEW_W, emit, write_mp4


def live_writer(live_dir: Path):
    from PIL import Image

    live_dir.mkdir(parents=True, exist_ok=True)

    def write(frame: np.ndarray) -> None:
        tmp = live_dir / "live.tmp.jpg"
        Image.fromarray(frame).save(tmp, quality=80)
        os.replace(tmp, live_dir / "live.jpg")
    return write


def save_attempt(s: ManipulationSession, env: PickPlaceEnv, out: Path, meta: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    r = s.rec
    n = len(r)
    if n:
        np.savez_compressed(
            out / "episode.npz", front=np.stack(r.front), wrist=np.stack(r.wrist),
            state=np.stack(r.state).astype(np.float32), action=np.stack(r.action).astype(np.float32),
            reward=np.asarray(r.reward, np.float32), timestamp=(np.arange(n) / FPS).astype(np.float32),
            action_low=np.r_[np.rad2deg(env.world.lo[:5]), 0].astype(np.float32),
            action_high=np.r_[np.rad2deg(env.world.hi[:5]), 100].astype(np.float32))
    if r.video:
        write_mp4(r.video, out / "front.mp4")
        Image.fromarray(r.video[min(len(r.video) - 1, len(r.video) // 2)]).save(out / "thumb.jpg", quality=85)
    (out / "trace.json").write_text(json.dumps(meta, default=str))
    return {"num_frames": n, "fps": FPS}


def main() -> None:
    import mujoco

    from ..agentic.astra import AstraPlanner, PlannerConfig

    configure_gl()
    cfg = json.loads(Path(sys.argv[1]).read_text())
    key = os.environ.get("CORTEX_PLANNER_API_KEY", "")
    pc = cfg["planner"]
    planner = AstraPlanner(PlannerConfig(provider=pc["provider"], api_key=key, model=pc.get("model", ""),
                                         mode=cfg.get("mode", "camera"), max_calls=pc.get("max_calls", 16),
                                         max_usd=pc.get("max_usd", 1.5),
                                         reasoning_effort=pc.get("reasoning_effort", "low"),
                                         endpoint=os.environ.get("CORTEX_PLANNER_BASE_URL", "")))
    env = PickPlaceEnv(episode_steps=10_000)
    on_frame = live_writer(Path(cfg["live_dir"]))
    total_cost, results = 0.0, []
    for i, a in enumerate(cfg["attempts"]):
        if total_cost >= cfg.get("max_total_usd", 1e9):
            emit(type="budget_exhausted", index=i, spent_usd=round(total_cost, 4))
            break
        sc = a["scenario"]
        var = Variation(cube_region=sc["cube_region"], lighting=sc["lighting"],
                        cube_offset=tuple(sc.get("cube_offset", (0, 0))), box_offset=tuple(sc.get("box_offset", (0, 0))))
        env.reset(a["seed"], var)
        renderer = mujoco.Renderer(env.world.model, PREVIEW_H, PREVIEW_W)
        s = ManipulationSession(env, mode=cfg.get("mode", "camera"), on_frame=on_frame, video_renderer=renderer)
        s.reset_environment(a["seed"], var)
        emit(type="attempt_start", index=i, total=len(cfg["attempts"]), seed=a["seed"], scenario=sc)
        t0 = time.time()

        def relay(ev: dict[str, Any], i: int = i) -> None:
            emit(**ev, index=i)
        try:
            run = planner.run_episode(s, cfg["task"], relay)
            error = None
        except Exception as e:  # provider/network failures end this attempt, recorded as such
            run, error = {"trace": [], "usage": {}, "recoveries": 0, "counters": dict(s.counters)}, \
                f"{type(e).__name__}: {e}"[:500]
        success = bool(env.in_box())  # simulator ground truth, independent of the model's claim
        meta = {"seed": a["seed"], "scenario": sc, "mode": cfg.get("mode", "camera"),
                "provider": pc["provider"], "model_requested": planner.cfg.model,
                "endpoint": planner.cfg.endpoint, **run,
                "success": success, "error": error, "wall_s": round(time.time() - t0, 2),
                "max_cube_z": env.max_cube_z}
        info = save_attempt(s, env, Path(a["out_dir"]), meta)
        renderer.close()
        total_cost += float(run.get("usage", {}).get("cost_usd", 0.0))
        res = {**{k: v for k, v in meta.items() if k != "trace"}, **info, "out_dir": a["out_dir"],
               "trace_len": len(run.get("trace", []))}
        results.append(res)
        emit(type="attempt", index=i, **res)
    env.close()
    Path(cfg["live_dir"], "result.json").write_text(json.dumps({"attempts": results}, default=str))
    emit(type="result", ok=True, spent_usd=round(total_cost, 4))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        emit(type="result", ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
        sys.exit(1)
