"""Real SO-101 control loop under the safety rules of spec section 5.

    python -m cortex.hardware.arm_worker <config.json>     (JSON lines on stdin and stdout)

stdin:  {"op": "heartbeat"}    the operator's held deadman, relayed by the studio (~10 Hz)
        {"op": "end_episode"}  {"op": "stop"}  {"op": "release"}
stdout: {"type": "ready" | "state" | "tripped" | "episode_saved" | "holding" | "result", ...}

Each tick reads the follower arm, takes a target (the leader arm's joints for teleop; a trained
policy's action, added with the service), clips it to the joint limits, rate-limits it to
`max_step_deg` per tick, checks it with `SafetyController` (and the gripper against the table
floor), and sends it only while the deadman is live and the watchdog has not tripped. The
watchdog runs on its own thread: a lapsed deadman or a stalled loop trips it, and the loop
re-checks the deadman just before every send, so motion stops within one control tick. Once
stopped, nothing more is sent and the servos hold their last goal with torque on. Torque is
released only when the operator presses Release (support the arm first), or 30 s after the
studio goes away. Nothing moves until the operator first holds the deadman.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import worker_error
from .so101_bridge import JOINTS, SafetyController, SafetyLimits, SafetyViolation

KEYS = [f"{j}.pos" for j in JOINTS]
STUDIO_GONE_RELEASE_S = 30.0


def emit(**kw: Any) -> None:
    print(json.dumps(kw, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), flush=True)


class Deadman:
    def __init__(self) -> None:
        self._t: float | None = None
        self._lock = threading.Lock()

    def beat(self) -> None:
        with self._lock:
            self._t = time.monotonic()

    def age(self) -> float:
        with self._lock:
            return float("inf") if self._t is None else time.monotonic() - self._t


class Watchdog(threading.Thread):
    """Own thread. The control loop only reports ticks; it never feeds the deadman."""

    def __init__(self, deadman: Deadman, deadman_timeout: float, stale_timeout: float):
        super().__init__(daemon=True)
        self.deadman, self.deadman_timeout, self.stale_timeout = deadman, deadman_timeout, stale_timeout
        self.tripped = threading.Event()
        self.reason: str | None = None
        self.last_tick = time.monotonic()
        self._halt = threading.Event()

    def tick(self) -> None:
        self.last_tick = time.monotonic()

    def trip(self, reason: str) -> None:
        if not self.tripped.is_set():
            self.reason = reason
            self.tripped.set()

    def run(self) -> None:
        while not self._halt.wait(0.005):
            if self.deadman.age() > self.deadman_timeout:
                self.trip("deadman released")
            elif time.monotonic() - self.last_tick > self.stale_timeout:
                self.trip("control loop stalled")

    def halt(self) -> None:
        self._halt.set()


@dataclass
class Controls:
    """Operator commands, set by the stdin reader thread."""

    deadman: Deadman
    end_episode: bool = False
    stop: bool = False
    release: bool = False
    studio_gone_at: float | None = None

    def read(self, stream) -> None:
        for line in stream:
            try:
                op = json.loads(line).get("op")
            except (json.JSONDecodeError, AttributeError):
                continue
            if op == "heartbeat":
                self.deadman.beat()
            elif op in ("end_episode", "stop", "release"):
                setattr(self, op, True)
        self.studio_gone_at = time.monotonic()


def limit(target: np.ndarray, current: np.ndarray, max_step: float, safety: SafetyController) -> np.ndarray:
    """Clip to the joint limits, rate-limit toward the target, then run the safety checks."""
    target = np.asarray(target, float)
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        raise SafetyViolation("malformed", "target must be 6 finite joint values")
    current = np.asarray(current, float)
    target = np.clip(target, safety.limits.low, safety.limits.high)
    step = np.r_[np.full(5, max_step), 2 * max_step]  # gripper may move twice as fast
    safe = current + np.clip(target - current, -step, step)
    safety.check(safe, current)
    return safe


def joints(obs: dict[str, Any]) -> np.ndarray:
    return np.array([float(obs[k]) for k in KEYS])


@dataclass
class RunConfig:
    mode: str = "teleop"  # teleop | policy
    fps: int = 30
    episodes: int = 1
    episode_s: float = 30.0
    max_step_deg: float = 2.0
    deadman_timeout_s: float = 0.5
    stale_timeout_s: float = 0.5
    first_deadman_wait_s: float = 60.0


def control(robot, source: Callable[[dict[str, Any]], np.ndarray], ctl: Controls, cfg: RunConfig,
            safety: SafetyController, recorder=None, reset_source: Callable[[], None] | None = None) -> dict:
    """Run episodes, then hold until released. Returns the result summary."""
    t_wait = time.monotonic()
    while ctl.deadman.age() > cfg.deadman_timeout_s:
        if ctl.stop or ctl.studio_gone_at or time.monotonic() - t_wait > cfg.first_deadman_wait_s:
            robot.disconnect()
            return {"ok": False, "error": "the deadman was never held; nothing moved", "ticks": 0}
        time.sleep(0.01)
    wd = Watchdog(ctl.deadman, cfg.deadman_timeout_s, cfg.stale_timeout_s)
    wd.start()
    ticks, saved, last_state = 0, [], 0.0
    period = 1.0 / cfg.fps
    for ep in range(cfg.episodes):
        if reset_source:
            reset_source()
        end = time.monotonic() + cfg.episode_s
        while time.monotonic() < end and not (ctl.stop or wd.tripped.is_set()):
            t0 = time.monotonic()
            wd.tick()
            if ctl.end_episode:
                ctl.end_episode = False
                break
            obs = robot.get_observation()
            current = joints(obs)
            try:
                safe = limit(source(obs), current, cfg.max_step_deg, safety)
            except SafetyViolation as e:
                wd.trip(f"{e.code}: {e}")
                break
            if ctl.deadman.age() > cfg.deadman_timeout_s:  # re-checked right before every send
                wd.trip("deadman released")
            if wd.tripped.is_set():
                break
            robot.send_action(dict(zip(KEYS, safe.tolist(), strict=True)))
            ticks += 1
            if recorder is not None:
                recorder.add(obs, safe)
            if t0 - last_state > 0.2:
                emit(type="state", episode=ep, joints=current.round(2), target=safe.round(2), ticks=ticks)
                last_state = t0
            time.sleep(max(0.0, period - (time.monotonic() - t0)))
        cut_short = ctl.stop or wd.tripped.is_set()
        if recorder is not None and recorder.frames:
            if cut_short:  # an interrupted demonstration is not a demonstration
                emit(type="episode_discarded", frames=recorder.discard(), reason=wd.reason or "stopped")
            else:
                n = recorder.save_episode()
                saved.append(n)
                emit(type="episode_saved", index=len(saved) - 1, frames=n)
        if cut_short:
            break
    wd.halt()
    reason = wd.reason or ("stopped by the operator" if ctl.stop else "run complete")
    if wd.tripped.is_set():
        emit(type="tripped", reason=reason)
    if recorder is not None:
        recorder.finalize()
    emit(type="holding", reason=reason, note="Servos hold their last goal. Support the arm, then Release.")
    while not ctl.release:  # hold with torque on; nothing is sent
        if ctl.studio_gone_at and time.monotonic() - ctl.studio_gone_at > STUDIO_GONE_RELEASE_S:
            break
        time.sleep(0.02)
    robot.disconnect()  # disable_torque_on_disconnect: the arm goes limp now
    return {"ok": True, "reason": reason, "tripped": wd.tripped.is_set(), "ticks": ticks, "episodes_saved": saved}


class Recorder:
    """Writes a LeRobot v3 dataset of what the follower saw and the action actually sent."""

    def __init__(self, root: Path, repo_id: str, fps: int, robot, task: str):
        from lerobot.datasets import LeRobotDataset
        from lerobot.utils.feature_utils import hw_to_dataset_features

        self.features = {**hw_to_dataset_features(robot.action_features, "action", True),
                         **hw_to_dataset_features(robot.observation_features, "observation", True)}
        self.ds = LeRobotDataset.create(repo_id, fps, root=root, robot_type=robot.name, features=self.features,
                                        use_videos=True)
        self.task, self.frames = task, 0

    def add(self, obs: dict[str, Any], action: np.ndarray) -> None:
        from lerobot.utils.feature_utils import build_dataset_frame

        frame = {**build_dataset_frame(self.features, obs, "observation"),
                 **build_dataset_frame(self.features, dict(zip(KEYS, action.tolist(), strict=True)), "action"),
                 "task": self.task}
        self.ds.add_frame(frame)
        self.frames += 1

    def save_episode(self) -> int:
        n, self.frames = self.frames, 0
        self.ds.save_episode(parallel_encoding=False)
        return n

    def discard(self) -> int:
        n, self.frames = self.frames, 0
        self.ds.clear_episode_buffer()
        return n

    def finalize(self) -> None:
        self.ds.finalize()


def connect_follower(c: dict[str, Any], max_step: float):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig

    cams = {x["name"]: OpenCVCameraConfig(index_or_path=x["index_or_path"], width=x["width"], height=x["height"],
                                          fps=x["fps"]) for x in c.get("cameras", [])}
    robot = SO101Follower(SOFollowerRobotConfig(port=c["follower_port"], id=c["follower_id"], cameras=cams,
                                                max_relative_target=max_step, use_degrees=True,
                                                disable_torque_on_disconnect=True))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:  # connect(calibrate=False) would otherwise carry on uncalibrated
        robot.disconnect()
        raise RuntimeError(f"follower '{c['follower_id']}' is not calibrated: calibrate it in the studio first")
    return robot


def connect_leader(c: dict[str, Any]):
    from lerobot.teleoperators.so_leader import SO101Leader, SOLeaderTeleopConfig

    leader = SO101Leader(SOLeaderTeleopConfig(port=c["leader_port"], id=c["leader_id"], use_degrees=True))
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        leader.disconnect()
        raise RuntimeError(f"leader '{c['leader_id']}' is not calibrated: calibrate it in the studio first")
    return leader


def floor_fk(floor_z: float = 0.005) -> Callable[[np.ndarray], np.ndarray]:
    """Gripper position from the EnvHub SO-101 model (so101_follower degrees, base frame, table at z=0)."""
    import mujoco

    from ..sim.so101 import PickPlaceEnv, Variation, configure_gl

    configure_gl()
    env = PickPlaceEnv()
    env.reset(0, Variation())
    m = env.world.model
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in JOINTS[:5]]

    def fk(q: np.ndarray) -> np.ndarray:
        d.qpos[qadr] = np.deg2rad(np.asarray(q, float)[:5])
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[env.site].copy()

    return fk


def main() -> None:
    c = json.loads(Path(sys.argv[1]).read_text())
    cfg = RunConfig(**c["run"])
    limits = SafetyLimits(max_deg_per_tick=cfg.max_step_deg * 1.001, watchdog_timeout_s=cfg.deadman_timeout_s)
    safety = SafetyController(limits, fk=floor_fk() if c.get("workspace_check", True) else None)
    ctl = Controls(Deadman())
    threading.Thread(target=ctl.read, args=(sys.stdin,), daemon=True).start()
    robot = connect_follower(c, cfg.max_step_deg)
    if cfg.mode != "teleop":
        robot.disconnect()
        raise ValueError(f"unsupported mode {cfg.mode}")
    leader = connect_leader(c)
    recorder = Recorder(Path(c["dataset_root"]), c["repo_id"], cfg.fps, robot, c.get("task", "")) \
        if c.get("record") else None
    emit(type="ready", mode=cfg.mode, cameras=[x["name"] for x in c.get("cameras", [])])
    try:
        out = control(robot, lambda obs: joints(leader.get_action()), ctl, cfg, safety, recorder)
    finally:
        leader.disconnect()
    emit(type="result", **out, dataset_root=c.get("dataset_root") if recorder else None)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit(type="result", ok=False, **worker_error(e))
        sys.exit(1)
