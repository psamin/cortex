"""Simulation worker process: records demonstrations or evaluates a policy.

    python -m cortex.workers.sim_worker <config.json>

Runs in its own process (one MuJoCo/GL context each) locally, or unchanged inside
a remote worker host. Emits JSON lines on stdout ({"type": "progress"|"log"|"result"})
and writes artifacts to config["out_dir"]. The latest preview frame is written
atomically to out_dir/live.jpg so the API can stream it as MJPEG.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import worker_error

from ..sim.so101 import FPS, PickPlaceEnv, Variation, configure_gl
from ..sim.viewer import VIEW_H as PREVIEW_H  # noqa: F401  (presentation video size, re-exported)
from ..sim.viewer import VIEW_W as PREVIEW_W  # noqa: F401
SETTLE_STEPS = 10


def emit(**kw: Any) -> None:
    print(json.dumps(kw), flush=True)


class Preview:
    """The presentation camera (cortex.sim.viewer) for the live view and episode videos.
    Cosmetic only: policies are trained on the env's own observation cameras."""

    def __init__(self, env: PickPlaceEnv, out_dir: Path, live: bool):
        from ..sim.viewer import DemoCamera

        self.r = DemoCamera(env.world.model)
        self.env, self.out_dir, self.live = env, out_dir, live

    def grab(self) -> np.ndarray:
        img = self.r.render(self.env.world.data)
        if self.live:
            from PIL import Image

            tmp = self.out_dir / "live.tmp.jpg"
            Image.fromarray(img).save(tmp, quality=80)
            os.replace(tmp, self.out_dir / "live.jpg")
        return img

    def close(self) -> None:
        self.r.close()


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = FPS) -> None:
    import av

    with av.open(str(path), "w") as out:
        s = out.add_stream("libx264", rate=fps)
        s.width, s.height = frames[0].shape[1], frames[0].shape[0]
        s.pix_fmt = "yuv420p"
        s.options = {"crf": "26", "preset": "veryfast", "movflags": "+faststart"}
        for f in frames:
            for pkt in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(pkt)
        for pkt in s.encode():
            out.mux(pkt)


def failure_mode(success: bool, max_z: float, steps: int, max_steps: int, final_dist: float) -> str | None:
    """Measured from simulator telemetry (not a model guess)."""
    if success:
        return None
    if max_z < 0.03:
        return "grasp_failed"  # cube never left the table
    if final_dist > 0.06:
        return "dropped_before_target"
    return "released_outside_target"


def run_episode(env: PickPlaceEnv, policy, seed: int, var: Variation, max_steps: int,
                preview: Preview | None, record: bool) -> dict[str, Any]:
    obs = env.reset(seed, var)
    policy.reset(env, obs, seed)
    buf: dict[str, list] = {"front": [], "wrist": [], "state": [], "action": [], "reward": []}
    video: list[np.ndarray] = []
    success, settle, reward, t0 = False, 0, 0.0, time.time()
    for step in range(max_steps):
        action = policy.act(obs)
        if record:
            buf["front"].append(obs["pixels"]["front"])
            buf["wrist"].append(obs["pixels"]["wrist"])
            buf["state"].append(obs["agent_pos"])
            buf["action"].append(action)
        obs, reward, success, _, _ = env.step(action)
        if record:
            buf["reward"].append(reward)
        if preview is not None:
            video.append(preview.grab())
        if step % 10 == 0:
            emit(type="progress", step=step, max_steps=max_steps, seed=seed)
        ex = getattr(policy, "ex", None)
        finished = ex.done if ex is not None else success
        if finished:
            settle += 1
            if settle >= SETTLE_STEPS:
                break
    steps = step + 1
    c, b = env.cube_pos(), env.box_pos()
    final_dist = float(np.linalg.norm(c[:2] - b[:2]))
    return {
        "seed": seed, "success": bool(success), "reward": float(reward), "steps": steps,
        "max_cube_z": env.max_cube_z, "final_cube_box_dist": final_dist,
        "failure_mode": failure_mode(success, env.max_cube_z, steps, max_steps, final_dist),
        "wall_s": time.time() - t0, "buf": buf, "video": video,
    }


def cmd_record(cfg: dict[str, Any]) -> dict[str, Any]:
    """cfg: {episodes: [{seed, condition, out_dir}], live_dir, max_steps}. One process, many demos."""
    from PIL import Image

    from ..sim.policies import ExpertPolicy

    live_dir = Path(cfg["live_dir"])
    live_dir.mkdir(parents=True, exist_ok=True)
    max_steps = cfg.get("max_steps", 240)
    env = PickPlaceEnv(episode_steps=max_steps)
    preview: Preview | None = None
    results = []
    for i, spec in enumerate(cfg["episodes"]):
        out = Path(spec["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        cond = spec["condition"]
        sc = spec.get("scenario") or {}
        var = Variation(cube_region=cond["cube_region"], lighting=cond["lighting"],
                        cube_offset=tuple(sc.get("cube_offset", (0.0, 0.0))),
                        box_offset=tuple(sc.get("box_offset", (0.0, 0.0))))
        if preview is None:
            env.reset(spec["seed"], var)
            preview = Preview(env, live_dir, live=True)
        emit(type="episode_start", index=i, total=len(cfg["episodes"]), seed=spec["seed"], condition=cond)
        ep = run_episode(env, ExpertPolicy(), spec["seed"], var, max_steps, preview, record=True)
        buf = ep.pop("buf")
        n = len(buf["action"])
        np.savez_compressed(
            out / "episode.npz",
            front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
            state=np.stack(buf["state"]).astype(np.float32), action=np.stack(buf["action"]).astype(np.float32),
            reward=np.asarray(buf["reward"], np.float32), timestamp=(np.arange(n) / FPS).astype(np.float32),
            action_low=np.r_[np.rad2deg(env.world.lo[:5]), 0].astype(np.float32),
            action_high=np.r_[np.rad2deg(env.world.hi[:5]), 100].astype(np.float32),
        )
        video = ep.pop("video")
        write_mp4(video, out / "front.mp4")
        Image.fromarray(video[min(len(video) - 1, 45)]).save(out / "thumb.jpg", quality=85)
        res = {**ep, "num_frames": n, "fps": FPS, "out_dir": str(out)}
        (out / "result.json").write_text(json.dumps(res))
        emit(type="episode", index=i, **res)
        results.append(res)
    if preview:
        preview.close()
    env.close()
    return {"episodes": results}


def cmd_evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    from ..sim.policies import load_policy

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    v = cfg["variation"]
    var = Variation(cube_region=v["cube_region"], lighting=v["lighting"],
                    cube_offset=tuple(v.get("cube_offset", (0.0, 0.0))), light_scale=v.get("light_scale", 1.0))
    pol = cfg["policy"]
    policy = load_policy(pol["kind"], pol.get("checkpoint_path"))
    max_steps = cfg.get("max_steps", 300)
    env = PickPlaceEnv(episode_steps=max_steps)
    env.reset(cfg["seeds"][0], var)
    preview = Preview(env, out, live=False)
    videos_left = cfg.get("max_videos", 2)
    results = []
    for i, seed in enumerate(cfg["seeds"]):
        ep = run_episode(env, policy, seed, var, max_steps, preview if videos_left > 0 else None, record=False)
        ep.pop("buf")
        video = ep.pop("video")
        if video and (i == 0 or not ep["success"]):
            name = f"seed{seed}.mp4"
            write_mp4(video, out / name)
            ep["video"] = name
            videos_left -= 1
        results.append(ep)
        emit(type="episode", **ep)
    preview.close()
    env.close()
    return {"policy": policy.name, "episodes": results}


def main() -> None:
    configure_gl()
    cfg = json.loads(Path(sys.argv[1]).read_text())
    try:
        res = cmd_record(cfg) if cfg["mode"] == "record" else cmd_evaluate(cfg)
        (Path(cfg.get("out_dir") or cfg["live_dir"]) / "result.json").write_text(json.dumps(res))
        emit(type="result", ok=True)
    except Exception as e:  # report, then fail the process

        emit(type="result", ok=False, **worker_error(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
