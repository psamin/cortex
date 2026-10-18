import asyncio
import sys

import pytest

from cortex.hardware import calibrate as cal
from cortex.hardware.setup import HardwareConfig, HardwareRefused

# Mimics lerobot-calibrate's console flow: input() prompts without a newline, then a live joint
# table redrawn with cursor-up escape codes until Enter arrives (select on stdin, like LeRobot).
FAKE = r'''
import select, sys, time
x = input("Press ENTER to use provided calibration file associated with the id a1, or type 'c' and press ENTER to run calibration: ")
print("choice", repr(x))
input("Move so_follower to the middle of its range of motion and press ENTER....")
print("Move all joints sequentially through their entire ranges of motion.\nRecording positions. Press ENTER to stop...")
while True:
    print("\n-------------------------------------------\nNAME            |    MIN |    POS |    MAX")
    print("shoulder_pan    |    812 |   2047 |   3301")
    print("\033[4A", end="", flush=True)
    if select.select([sys.stdin], [], [], 0.05)[0] and sys.stdin.readline().strip() == "":
        break
print("Calibration saved to /tmp/a1.json")
'''
CFG = HardwareConfig(follower_port="/dev/fake", follower_id="a1")


async def until(cond, timeout=10.0):
    t = asyncio.get_running_loop().time()
    while not cond():
        assert asyncio.get_running_loop().time() - t < timeout, cal.snapshot()
        await asyncio.sleep(0.02)


@pytest.mark.skipif(sys.platform == "win32", reason="LeRobot reads Enter from the console on Windows")
def test_calibration_prompts_are_relayed_and_answered(monkeypatch):
    monkeypatch.setattr(cal, "calibrate_cmd", lambda device, cfg: [sys.executable, "-c", FAKE])

    async def run():
        snap = await cal.start_calibration("follower", CFG, enabled=True)
        assert snap["kind"] == "calibrate_follower"
        await until(lambda: "type 'c'" in (cal.snapshot()["prompt"] or ""))
        assert cal.snapshot()["state"] == "waiting_for_operator"
        with pytest.raises(HardwareRefused, match="running"):
            await cal.start_calibration("leader", CFG.model_copy(update={"leader_port": "/dev/x"}), enabled=True)
        await cal.send_input("c")
        await until(lambda: "middle of its range" in (cal.snapshot()["prompt"] or ""))
        await cal.send_input("")
        await until(lambda: "Press ENTER to stop" in (cal.snapshot()["prompt"] or "")
                    and cal.snapshot()["joint_ranges"])
        assert cal.snapshot()["joint_ranges"]["shoulder_pan"] == {"min": 812, "pos": 2047, "max": 3301}
        await cal.send_input("")
        await until(lambda: cal.snapshot()["state"] == "succeeded")
        s = cal.snapshot()
        assert "choice 'c'" in s["lines"] and any("Calibration saved" in x for x in s["lines"])
        assert not any("|" in x for x in s["lines"])  # table rows are parsed, not dumped into the log
        assert s["prompt"] is None

    asyncio.run(run())


def test_calibration_preconditions():
    async def run():
        with pytest.raises(HardwareRefused, match="CORTEX_ENABLE_HARDWARE"):
            await cal.start_calibration("follower", CFG, enabled=False)
        with pytest.raises(HardwareRefused, match="leader arm's port"):
            await cal.start_calibration("leader", CFG, enabled=True)

    asyncio.run(run())


def test_calibration_is_refused_on_windows_with_the_terminal_command(monkeypatch):
    """LeRobot reads Enter from the console on Windows, so the studio can't relay prompts there."""
    monkeypatch.setattr(cal.platform, "system", lambda: "Windows")
    with pytest.raises(HardwareRefused, match="run `lerobot-calibrate` in a terminal") as e:
        asyncio.run(cal.start_calibration("follower", CFG, enabled=True))
    assert "--robot.port=/dev/fake" in str(e.value) and "--robot.id=a1" in str(e.value)
    assert not cal.busy()


@pytest.mark.skipif(sys.platform == "win32", reason="LeRobot reads Enter from the console on Windows")
def test_calibration_can_be_stopped(monkeypatch):
    async def run():
        monkeypatch.setattr(cal, "calibrate_cmd", lambda d, c: [sys.executable, "-c", "input('press ENTER: ')"])
        await cal.start_calibration("follower", CFG, enabled=True)
        await until(lambda: cal.snapshot()["prompt"])
        assert (await cal.stop())["state"] == "stopped"
        assert not cal.busy()

    asyncio.run(run())


def test_failure_is_summarised_without_the_traceback():
    lines = ["Traceback (most recent call last):", '  File "x.py", line 1', "ConnectionError:",
             "Could not connect on port '/dev/x'. Make sure you are using the correct port.",
             "Try running `lerobot-find-port`"]
    assert cal._error(lines) == ("ConnectionError: Could not connect on port '/dev/x'. Make sure you are using "
                                 "the correct port. Try running `lerobot-find-port`")


def test_real_command_targets_so101_with_the_configured_ids():
    f = cal.calibrate_cmd("follower", CFG)
    assert f[1:] == ["-m", "lerobot.scripts.lerobot_calibrate", "--robot.type=so101_follower",
                     "--robot.port=/dev/fake", "--robot.id=a1"]
    leader = cal.calibrate_cmd("leader", HardwareConfig(leader_port="/dev/l", leader_id="l1"))
    assert "--teleop.type=so101_leader" in leader and "--teleop.id=l1" in leader
