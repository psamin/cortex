"""Local SO-101 hardware adapter (opt-in, supervised only). NOT exposed to the API or MCP.

The same planner-facing joint-target interface as the simulator session, executed
through LeRobot's SO-101 follower driver (`lerobot.robots.so_follower`, so101_follower
units: degrees, gripper 0-100) behind a local safety controller:

  * operator authorization, a confirmed independent physical e-stop, and a calibration file
  * per-command joint-limit, per-tick velocity and Cartesian workspace checks
  * a watchdog: no heartbeat within the timeout -> hold position and refuse further motion
  * supervised validation: every motion is confirmed by the local operator; unattended
    control is not implemented in this build and raises
  * CORTEX_ENABLE_HARDWARE=1 must be set on the machine wired to the robot

Cloud model output can only ever reach `execute`, which runs every target through
`SafetyController.check` first. Successful MuJoCo trajectories are not physical
validation; they must be re-validated here under supervision.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class SafetyViolation(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class HardwareNotEnabled(PermissionError):
    pass


@dataclass
class SafetyLimits:
    # Joint limits in so101_follower degrees (from the Menagerie model's ctrlrange), gripper 0-100.
    low: np.ndarray = field(default_factory=lambda: np.array([-110, -100, -96.8, -95, -157.2, 0.0]))
    high: np.ndarray = field(default_factory=lambda: np.array([110, 100, 96.8, 95, 162.8, 100.0]))
    max_deg_per_tick: float = 3.0  # at 30 Hz: 90 deg/s, below the sim expert's 120 deg/s
    workspace_min: np.ndarray = field(default_factory=lambda: np.array([0.05, -0.25, 0.005]))
    workspace_max: np.ndarray = field(default_factory=lambda: np.array([0.38, 0.25, 0.30]))
    watchdog_timeout_s: float = 0.5


class SafetyController:
    def __init__(self, limits: SafetyLimits, fk: Callable[[np.ndarray], np.ndarray] | None = None):
        self.limits, self.fk = limits, fk
        self.last_heartbeat = time.monotonic()
        self.tripped: str | None = None

    def heartbeat(self) -> None:
        self.last_heartbeat = time.monotonic()

    def watchdog_ok(self) -> bool:
        if time.monotonic() - self.last_heartbeat > self.limits.watchdog_timeout_s:
            self.tripped = "watchdog"
        return self.tripped is None

    def check(self, target: np.ndarray, current: np.ndarray) -> None:
        L = self.limits
        if self.tripped:
            raise SafetyViolation("tripped", f"safety controller tripped ({self.tripped}); operator reset required")
        target = np.asarray(target, float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise SafetyViolation("malformed", "target must be 6 finite joint values")
        bad = np.nonzero((target < L.low) | (target > L.high))[0]
        if len(bad):
            raise SafetyViolation("joint_limit", f"joint limit on {[JOINTS[i] for i in bad]}")
        step = np.abs(target[:5] - np.asarray(current, float)[:5]).max()
        if step > L.max_deg_per_tick:
            raise SafetyViolation("velocity", f"{step:.1f} deg in one tick exceeds {L.max_deg_per_tick}")
        if self.fk is not None:
            p = self.fk(target)
            if np.any(p < L.workspace_min) or np.any(p > L.workspace_max):
                raise SafetyViolation("workspace", f"end effector {np.round(p, 3).tolist()} leaves the workspace")


@dataclass
class OperatorGate:
    operator_authorized: bool = False
    estop_confirmed: bool = False  # operator confirms the independent physical e-stop is in reach and tested
    calibration_path: Path | None = None
    confirm_motion: Callable[[str], bool] | None = None  # local operator prompt, e.g. a keypress

    def unmet(self) -> list[str]:
        missing = []
        if os.environ.get("CORTEX_ENABLE_HARDWARE") != "1":
            missing.append("CORTEX_ENABLE_HARDWARE=1 not set on this machine")
        if not self.operator_authorized:
            missing.append("operator authorization")
        if not self.estop_confirmed:
            missing.append("independent physical e-stop confirmed")
        if not (self.calibration_path and Path(self.calibration_path).exists()):
            missing.append("LeRobot calibration file")
        if self.confirm_motion is None:
            missing.append("supervised validation: a local operator confirmation callback")
        return missing


class SO101HardwareBridge:
    def __init__(self, port: str, gate: OperatorGate, safety: SafetyController):
        self.port, self.gate, self.safety = port, gate, safety
        self.robot = None

    def connect(self, unattended: bool = False) -> None:
        if unattended:
            raise HardwareNotEnabled("Unattended physical control is not implemented; supervised mode only.")
        missing = self.gate.unmet()
        if missing:
            raise HardwareNotEnabled("Hardware preflight failed: " + "; ".join(missing))
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig  # local, opt-in dependency

        self.robot = SOFollower(SOFollowerRobotConfig(port=self.port, use_degrees=True,
                                                      max_relative_target=self.safety.limits.max_deg_per_tick))
        self.robot.connect(calibrate=False)

    def current(self) -> np.ndarray:
        obs = self.robot.get_observation()  # type: ignore[union-attr]
        return np.array([obs[f"{j}.pos"] for j in JOINTS], float)

    def execute(self, trajectory: np.ndarray, summary: str) -> int:
        """Run a joint-target trajectory (N x 6) at 30 Hz under supervision. Returns ticks executed."""
        if self.robot is None:
            raise HardwareNotEnabled("not connected")
        if not self.gate.confirm_motion(summary):  # type: ignore[misc]
            raise HardwareNotEnabled("operator declined the motion")
        ticks = 0
        for target in np.asarray(trajectory, float):
            self.safety.heartbeat()
            if not self.safety.watchdog_ok():
                break
            self.safety.check(target, self.current())
            self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, target)})
            ticks += 1
            time.sleep(1 / 30)
        return ticks
