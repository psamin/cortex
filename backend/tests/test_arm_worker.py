import threading
import time

import numpy as np
import pytest

from cortex.hardware import arm_worker as aw
from cortex.hardware.so101_bridge import SafetyController, SafetyLimits, SafetyViolation

HOME = np.array([0, -60, 60, 35, 0, 30.0])


class FakeArm:
    """Stands in for LeRobot's SO101Follower: ideal servos, optional camera, records every command."""

    name = "so101_follower"

    def __init__(self, camera=False):
        self.q, self.sent, self.disconnected, self.camera = HOME.copy(), [], False, camera

    @property
    def observation_features(self):
        return {**{k: float for k in aw.KEYS}, **({"front": (48, 64, 3)} if self.camera else {})}

    @property
    def action_features(self):
        return {k: float for k in aw.KEYS}

    def get_observation(self):
        obs = {k: float(v) for k, v in zip(aw.KEYS, self.q, strict=True)}
        if self.camera:
            obs["front"] = np.full((48, 64, 3), 100, np.uint8)
        return obs

    def send_action(self, a):
        self.q = np.array([a[k] for k in aw.KEYS])
        self.sent.append((time.monotonic(), self.q.copy()))

    def disconnect(self):
        self.disconnected = True


def safety(step=2.0, fk=None):
    return SafetyController(SafetyLimits(max_deg_per_tick=step * 1.001), fk=fk)


def cfg(**kw):
    return aw.RunConfig(**{"episode_s": 5.0, "deadman_timeout_s": 0.2, "stale_timeout_s": 0.3,
                           "first_deadman_wait_s": 0.3, **kw})


def beat(ctl, seconds, period=0.04):
    """The operator holds the deadman for `seconds`. Returns when they let go."""
    stamps = []

    def run():
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            ctl.deadman.beat()
            stamps.append(time.monotonic())
            time.sleep(period)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return stamps


def release_on_hold(ctl, after=0.6):
    threading.Timer(after, lambda: setattr(ctl, "release", True)).start()


def reach_up(obs):  # keep asking for a far-away pose
    return HOME + np.r_[40, 30, -30, 0, 0, 0]


def test_limit_clips_rate_limits_and_rejects_garbage():
    s = safety(step=2.0)
    out = aw.limit(HOME + np.r_[50, -50, 0, 0, 0, 50], HOME, 2.0, s)
    assert np.allclose(out - HOME, [2, -2, 0, 0, 0, 4])  # gripper may move twice as fast
    out = aw.limit(np.r_[500, -60, 60, 35, 0, 30], np.r_[109, -60, 60, 35, 0, 30], 2.0, s)
    assert out[0] == 110  # clipped to the joint limit, then rate-limited
    with pytest.raises(SafetyViolation, match="finite"):
        aw.limit(np.r_[HOME[:5], np.nan], HOME, 2.0, s)


def test_nothing_moves_until_the_deadman_is_held():
    arm, ctl = FakeArm(), aw.Controls(aw.Deadman())
    out = aw.control(arm, reach_up, ctl, cfg(), safety())
    assert out["ok"] is False and "never held" in out["error"]
    assert arm.sent == [] and arm.disconnected


def test_releasing_the_deadman_stops_within_one_tick_and_holds():
    arm, ctl = FakeArm(), aw.Controls(aw.Deadman())
    stamps = beat(ctl, 0.5)
    release_on_hold(ctl, after=1.2)
    c = cfg()
    out = aw.control(arm, reach_up, ctl, c, safety())
    assert out["tripped"] and out["reason"] == "deadman released"
    last_beat, last_send = stamps[-1], arm.sent[-1][0]
    assert last_send <= last_beat + c.deadman_timeout_s + 1 / c.fps + 0.02  # stopped within one control tick
    assert len(arm.sent) == out["ticks"] > 5 and arm.disconnected  # torque released only after Release


def test_a_stalled_loop_trips_the_watchdog_and_nothing_more_is_sent():
    arm, ctl = FakeArm(), aw.Controls(aw.Deadman())
    beat(ctl, 3.0)
    calls = []

    def stalling(obs):
        calls.append(1)
        if len(calls) == 6:
            time.sleep(0.6)  # e.g. a hung camera read or a slow policy
        return reach_up(obs)
    release_on_hold(ctl, after=1.2)
    out = aw.control(arm, stalling, ctl, cfg(), safety())
    assert out["reason"] == "control loop stalled" and len(arm.sent) == 5


def test_the_table_floor_trips_before_the_gripper_goes_below_it():
    fk = lambda q: np.array([0.25, 0.0, 0.10 - 0.01 * max(0.0, q[1] + 60)])  # noqa: E731  lower as the shoulder lifts
    arm, ctl = FakeArm(), aw.Controls(aw.Deadman())
    beat(ctl, 3.0)
    release_on_hold(ctl, after=1.5)
    out = aw.control(arm, reach_up, ctl, cfg(), safety(fk=fk))
    assert out["tripped"] and out["reason"].startswith("workspace")
    assert all(fk(q)[2] >= 0.005 for _, q in arm.sent)


def test_stop_discards_the_partial_episode_and_holds_until_release(tmp_path, monkeypatch):
    monkeypatch.setattr(aw, "STUDIO_GONE_RELEASE_S", 0.1)
    arm, ctl = FakeArm(camera=True), aw.Controls(aw.Deadman())
    rec = aw.Recorder(tmp_path / "ds", "local/teleop_test", 30, arm, "pick up the cube")
    beat(ctl, 3.0)  # after the (slow on Windows) recorder is built
    threading.Timer(0.3, lambda: setattr(ctl, "stop", True)).start()
    threading.Timer(0.5, lambda: setattr(ctl, "studio_gone_at", time.monotonic())).start()  # studio closed
    out = aw.control(arm, lambda obs: HOME, ctl, cfg(episodes=2), safety(), recorder=rec)
    assert out["reason"] == "stopped by the operator" and out["episodes_saved"] == [] and arm.disconnected


def test_a_missing_video_encoder_names_ffmpeg(monkeypatch):
    from cortex.errors import UserError

    def boom():
        raise RuntimeError("Could not load libtorchcodec. FFmpeg is not properly installed")

    monkeypatch.setattr(aw, "_load_torchcodec", boom)
    with pytest.raises(UserError, match="FFmpeg"):
        aw.require_video_encoder()


def test_a_slow_video_save_is_not_a_stalled_control_loop():
    class SlowSave:
        frames = 0

        def add(self, obs, action):
            self.frames += 1

        def save_episode(self):
            time.sleep(0.6)  # longer than stale_timeout_s; encoding, not the control loop
            n, self.frames = self.frames, 0
            return n

        def discard(self):
            n, self.frames = self.frames, 0
            return n

        def finalize(self):
            pass

    arm, ctl = FakeArm(camera=True), aw.Controls(aw.Deadman())
    beat(ctl, 3.0)
    release_on_hold(ctl, after=2.0)
    out = aw.control(arm, lambda obs: HOME, ctl, cfg(episodes=1, episode_s=0.3), safety(), recorder=SlowSave())
    assert out["reason"] == "run complete" and out["episodes_saved"] == [out["ticks"]] and arm.disconnected


def test_teleop_recording_writes_a_lerobot_dataset(tmp_path):
    from cortex.errors import UserError
    from lerobot.datasets import LeRobotDataset

    arm, ctl = FakeArm(camera=True), aw.Controls(aw.Deadman())
    rec = aw.Recorder(tmp_path / "ds", "local/teleop_test", 30, arm, "pick up the cube")  # slow on Windows: build first
    beat(ctl, 3.0)
    release_on_hold(ctl, after=1.5)
    try:
        out = aw.control(arm, lambda obs: HOME + np.r_[1, 0, 0, 0, 0, 0], ctl, cfg(episodes=2, episode_s=0.3),
                         safety(), recorder=rec)
    except UserError as e:
        assert "FFmpeg" in str(e) and arm.disconnected
        pytest.skip(str(e))
    assert out["reason"] == "run complete" and len(out["episodes_saved"]) == 2
    try:
        ds = LeRobotDataset("local/teleop_test", root=tmp_path / "ds")
    except (ImportError, OSError, RuntimeError) as e:  # Windows without FFmpeg shared DLLs (Gyan.FFmpeg.Shared)
        if "torchcodec" not in str(e).lower():
            raise
        pytest.skip(f"torchcodec can't load here; recording itself passed ({type(e).__name__})")
    assert ds.meta.total_episodes == 2 and ds.meta.total_frames == sum(out["episodes_saved"])
    assert "observation.images.front" in ds.meta.features and ds.meta.robot_type == "so101_follower"


def test_a_trained_policy_drives_the_arm_under_the_same_rules(tmp_path):
    from tiny_act import make_tiny_act

    from cortex.errors import UserError

    ck = make_tiny_act(tmp_path / "ck", cameras=("front",))
    with pytest.raises(UserError, match=r"needs camera\(s\) \['front'\].*\['wrist'\]"):
        aw.PolicySource(str(ck), ["wrist"])  # refused before the arm is touched
    src = aw.PolicySource(str(ck), ["front"])
    arm, ctl = FakeArm(camera=True), aw.Controls(aw.Deadman())  # 48x64 camera, resized to the policy's 96x128
    a = src(arm.get_observation())
    assert a.shape == (6,) and np.all(np.isfinite(a))
    beat(ctl, 0.6)
    release_on_hold(ctl, after=1.0)
    out = aw.control(arm, src, ctl, cfg(), safety(), reset_source=src.reset)
    assert out["ticks"] > 5 and out["reason"] == "deadman released"
    steps = np.abs(np.diff(np.array([HOME] + [q for _, q in arm.sent]), axis=0))
    assert steps[:, :5].max() <= 2.0 + 1e-6  # the policy's targets are rate-limited like any other source
