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
    return rollout(env, policy, obs, seed, max_steps, preview, record)


def rollout(env: PickPlaceEnv, policy, obs: dict[str, Any], seed: int, max_steps: int,
            preview: Preview | None, record: bool, settle_max: int | None = None) -> dict[str, Any]:
    """Run ``policy`` from the env's current state (no reset). After the policy finishes it holds for
    SETTLE_STEPS, or with ``settle_max`` until the simulator reports success (at most that many steps)."""
    render = record or getattr(policy, "needs_pixels", True)  # oracles read sim state, not the cameras
    policy.reset(env, obs, seed)
    buf: dict[str, list] = {"front": [], "wrist": [], "state": [], "action": [], "reward": []}
    video: list[np.ndarray] = []
    success, settle, reward, t0 = False, 0, 0.0, time.time()
    env.max_cube_z = 0.0
    for step in range(max_steps):
        action = policy.act(obs)
        if record:
            buf["front"].append(obs["pixels"]["front"])
            buf["wrist"].append(obs["pixels"]["wrist"])
            buf["state"].append(obs["agent_pos"])
            buf["action"].append(action)
        nxt, reward, success, _, _ = env.step(action, render=render)
        obs = nxt if nxt is not None else obs
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
            if (settle_max is None and settle >= SETTLE_STEPS) or (
                    settle_max is not None and (success or settle >= settle_max)):
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


def save_npz(env: PickPlaceEnv, buf: dict[str, list], out: Path) -> int:
    n = len(buf["action"])
    np.savez_compressed(
        out / "episode.npz",
        front=np.stack(buf["front"]), wrist=np.stack(buf["wrist"]),
        state=np.stack(buf["state"]).astype(np.float32), action=np.stack(buf["action"]).astype(np.float32),
        reward=np.asarray(buf["reward"], np.float32), timestamp=(np.arange(n) / FPS).astype(np.float32),
        action_low=np.r_[np.rad2deg(env.world.lo[:5]), 0].astype(np.float32),
        action_high=np.r_[np.rad2deg(env.world.hi[:5]), 100].astype(np.float32),
    )
    return n


def save_video(video: list[np.ndarray], out: Path, thumb_at: int = 45) -> None:
    from PIL import Image

    if not video:
        return
    write_mp4(video, out / "front.mp4")
    Image.fromarray(video[min(len(video) - 1, thumb_at)]).save(out / "thumb.jpg", quality=85)


def cmd_record(cfg: dict[str, Any]) -> dict[str, Any]:
    """cfg: {episodes: [{seed, condition, out_dir}], live_dir, max_steps}. One process, many demos.
    With ``cycles`` it runs the cycle protocol instead (see ``cmd_cycles``)."""
    from ..sim import env_factory

    if cfg.get("cycles"):
        return cmd_cycles(cfg)

    live_dir = Path(cfg["live_dir"])
    live_dir.mkdir(parents=True, exist_ok=True)
    scene = cfg.get("scene")
    max_steps = env_factory.max_steps(scene, cfg.get("max_steps", 240))
    env = env_factory.make_env(scene, max_steps)
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
        ep = run_episode(env, env_factory.oracle(scene), spec["seed"], var, max_steps, preview, record=True)
        n = save_npz(env, ep.pop("buf"), out)
        save_video(ep.pop("video"), out)
        res = {**ep, "num_frames": n, "fps": FPS, "out_dir": str(out)}
        (out / "result.json").write_text(json.dumps(res))
        emit(type="episode", index=i, **res)
        results.append(res)
    if preview:
        preview.close()
    env.close()
    return {"episodes": results}


# ------------------------------------------------------------------ cycle protocol (PRODUCT_SPEC 3.2)
#
# A cycle is a DEMO segment (home -> pick from its start -> place in the task zone -> home) and then a
# RESET segment (home -> pick from the zone -> place at the next start -> home). The next demo starts
# wherever the reset left the object: only the very first placement teleports (labelled as the operator
# reset). After a failed segment a RECOVER segment puts the object back at a valid start; two failed
# recoveries end the run. Every successful segment is its own episode; failed ones are diagnostics.

HOME_TOL_DEG = 3.0  # a segment must end at the home pose
RECOVER_TRIES = 2
SEGMENT_SETTLE_MAX = 40  # steps held at home for the object to come to rest
START_HALF = (0.03, 0.03)  # tolerance box around a start spot (reset and recover targets)
RECOVERY_FAILED = "recovery failed twice: the object needs an operator reset"


def object_pose(env) -> dict[str, Any]:
    p = env.cube_pos()
    return {"xyz": [round(float(v), 4) for v in p], "yaw": round(float(env.cube_yaw()), 4),
            "upright": bool(env.upright()) if hasattr(env, "upright") else None,
            "arm": [round(float(v), 2) for v in env.world.joint_state()]}


def at_home(env) -> bool:
    err = np.abs(np.asarray(env.world.joint_state())[:5] - np.asarray(env.module.HOME)[:5]).max()
    return bool(err < HOME_TOL_DEG)


class CycleScheduler:
    """Where each reset puts the object, from the intake's ``reset_policy``:

    * return_to_origin: back to the object's nominal start;
    * cycle_zones: the extra place zones (``task.place[1:]``) in turn, else the origin;
    * random_zone: a seeded random extra zone, else a seeded random reachable spot 3-5 cm from the origin.
    """

    def __init__(self, env, reset_policy: str, seed: int):
        from ..sim.grasp import SIDE_MIN_RADIUS, grasp_mode

        info = env.info
        self.env, self.policy = env, reset_policy
        self.origin = tuple(float(v) for v in info.starts[env.pick])
        self.task_zone = info.place[0] if info.place else None
        self.extra = list(info.place[1:])
        self.rng = np.random.default_rng(seed + 7919)
        self.r_min = SIDE_MIN_RADIUS if grasp_mode(env.obj_h) == "side" else 0.14

    def origin_target(self) -> dict[str, Any]:
        return {"xy": list(self.origin), "label": "origin"}

    def next_start(self, cycle: int) -> dict[str, Any]:
        if self.policy == "cycle_zones" and self.extra:
            return {"zone": self.extra[cycle % len(self.extra)]}
        if self.policy == "random_zone":
            if self.extra:
                return {"zone": str(self.rng.choice(self.extra))}
            return self._random_spot()
        return self.origin_target()

    def _random_spot(self) -> dict[str, Any]:
        (zx, zy), (hx, hy) = self.env.info.zones[self.task_zone] if self.task_zone else ((9.0, 9.0), (0, 0))
        for _ in range(50):
            r, a = self.rng.uniform(0.03, 0.05), self.rng.uniform(0, 2 * np.pi)
            x, y = self.origin[0] + r * np.cos(a), self.origin[1] + r * np.sin(a)
            far_from_zone = abs(x - zx) > hx + 0.05 or abs(y - zy) > hy + 0.05
            if self.r_min <= np.hypot(x, y) <= 0.35 and 0.10 <= x <= 0.36 and far_from_zone:
                return {"xy": [round(float(x), 4), round(float(y), 4)], "label": "random"}
        return self.origin_target()

    def aim(self, target: dict[str, Any]) -> None:
        if "zone" in target:
            self.env.set_target(zone=target["zone"])
        else:
            self.env.set_target(xy=tuple(target["xy"]), half=START_HALF)


def segment_goal(kind: str, obj: str, target: dict[str, Any], task: str | None = None) -> str:
    """The text a planner gets for a segment (the oracle ignores it; it is recorded either way)."""
    where = (f"the {target['zone'].replace('_', ' ')}" if "zone" in target
             else "its start" if target.get("label") == "origin" else f"the new start at {target['xy']}")
    if kind == "demo":
        return task or f"demo: put the {obj} in {where}, then go home"
    if kind == "reset":
        return f"reset: put the {obj} back at {where}, then go home"
    return f"recover: find the {obj}, pick it up, put it at {where}, then go home"


def run_cycles(env, cfg: dict[str, Any], run_segment, initial_reset, keep_going=lambda: True) -> dict[str, Any]:
    """The cycle loop shared by the oracle (``cmd_cycles``) and the agent collector (agent_worker).

    ``run_segment(kind, goal, index, out_dir) -> {success, steps, num_frames, failure_reason, ...}`` runs one
    segment from the current state; ``initial_reset()`` teleports the object to its first start."""
    if not hasattr(env, "set_target"):
        raise ValueError("the cycle protocol needs an intake scene (build_simulation first); "
                         "the upstream cube tabletop records single demos")
    n_cycles = int(cfg["cycles"])
    sched = CycleScheduler(env, cfg.get("reset_policy", "return_to_origin"), int(cfg.get("seed", 0)))
    obj = env.pick.replace("_", " ")
    out_root = Path(cfg["out_root"])
    segments: list[dict[str, Any]] = []
    initial_reset()
    emit(type="operator_reset", reason="initial placement: the sim teleport stands in for the operator's reset",
         pose=object_pose(env))
    start = sched.origin_target()  # where the object is expected to be for the next demo

    def do(kind: str, cycle: int, target: dict[str, Any]) -> bool:
        k = len(segments)
        sched.aim(target)
        goal = segment_goal(kind, obj, target, cfg.get("task") if kind == "demo" else None)
        out = out_root / f"seg_{k:03d}"
        out.mkdir(parents=True, exist_ok=True)
        head = {"cycle": cycle, "segment": kind, "segment_index": k, "goal": goal, "target": target}
        emit(type="segment_start", **head, total_cycles=n_cycles, start_pose=object_pose(env))
        t0 = time.time()
        res = run_segment(kind, goal, k, out)
        rec = {**res, **head, "success": bool(res["success"]), "recorded": bool(res["success"]),
               "end_pose": object_pose(env), "wall_s": round(time.time() - t0, 2), "out_dir": str(out)}
        (out / "result.json").write_text(json.dumps(rec, default=str))
        segments.append(rec)
        emit(type="segment", **rec)
        return rec["success"]

    def recover(cycle: int) -> bool:
        for _ in range(RECOVER_TRIES):
            if not keep_going():
                return False
            if do("recover", cycle, start):
                return True
        return False

    stop_reason = None
    for cycle in range(n_cycles):
        if not keep_going():
            stop_reason = "budget exhausted or stop requested"
            break
        if sched.task_zone is None:
            stop_reason = "the intake task has no place zone"
            break
        if not do("demo", cycle, {"zone": sched.task_zone}):
            if not recover(cycle):
                stop_reason = RECOVERY_FAILED
                break
            continue
        nxt = sched.next_start(cycle)
        if do("reset", cycle, nxt):
            start = nxt
        elif not recover(cycle):
            stop_reason = RECOVERY_FAILED
            break
    if stop_reason == RECOVERY_FAILED:
        emit(type="operator_reset_required", reason=stop_reason, pose=object_pose(env))
    ok = [s for s in segments if s["success"]]
    summary = {"cycles": n_cycles, "segments": len(segments), "recorded": len(ok),
               "by_segment": {k: sum(s["segment"] == k for s in ok) for k in ("demo", "reset", "recover")},
               "stop_reason": stop_reason, "reset_policy": sched.policy}
    emit(type="cycles_done", **summary)
    return {**summary, "segment_results": segments}


def cmd_cycles(cfg: dict[str, Any]) -> dict[str, Any]:
    """Oracle cycles. cfg: {cycles, reset_policy, seed, condition, scene, live_dir, out_root, max_steps}."""
    from ..sim import env_factory

    scene = cfg.get("scene")
    if not scene:
        raise ValueError("the cycle protocol needs an intake scene (build_simulation first)")
    live_dir = Path(cfg["live_dir"])
    live_dir.mkdir(parents=True, exist_ok=True)
    max_steps = env_factory.max_steps(scene, cfg.get("max_steps", 240))
    env = env_factory.make_env(scene, max_steps)
    cond = cfg.get("condition") or {}
    var = Variation(lighting=cond.get("lighting", "normal"))
    seed = int(cfg.get("seed", 0))
    preview: list[Preview] = []

    def initial_reset() -> None:
        env.reset(seed, var)
        preview.append(Preview(env, live_dir, live=True))

    def run_segment(kind: str, goal: str, k: int, out: Path) -> dict[str, Any]:
        start_pose = object_pose(env)
        policy = env_factory.oracle(scene)
        ep = rollout(env, policy, env._obs(), seed, max_steps, preview[0], record=True,
                     settle_max=SEGMENT_SETTLE_MAX)
        ex = policy.ex
        home = at_home(env)
        success = bool(ep["success"]) and ex.failed is None and ex.finished and home
        reason = None if success else (ex.failed or ("timed out" if not ex.finished else None)
                                       or ("did not end at home" if not home else None)
                                       or ep["failure_mode"] or "not in the target")
        buf, video = ep.pop("buf"), ep.pop("video")
        n = save_npz(env, buf, out) if success else 0  # failed segments are diagnostics, never episodes
        save_video(video, out, thumb_at=len(video) // 2)
        return {**ep, "success": success, "failure_reason": reason, "num_frames": n, "fps": FPS,
                "start_pose": start_pose, "planner": "scripted_ik_oracle"}

    try:
        res = run_cycles(env, {**cfg, "seed": seed}, run_segment, initial_reset)
    finally:
        for p in preview:
            p.close()
        env.close()
    return res


def cmd_evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    from ..sim import env_factory
    from ..sim.policies import load_policy

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    v = cfg["variation"]
    var = Variation(cube_region=v["cube_region"], lighting=v["lighting"],
                    cube_offset=tuple(v.get("cube_offset", (0.0, 0.0))), light_scale=v.get("light_scale", 1.0))
    pol = cfg["policy"]
    scene = cfg.get("scene")
    policy = (env_factory.oracle(scene) if pol["kind"] == "scripted_expert" and scene
              else load_policy(pol["kind"], pol.get("checkpoint_path")))
    max_steps = env_factory.max_steps(scene, cfg.get("max_steps", 300))
    env = env_factory.make_env(scene, max_steps)
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
