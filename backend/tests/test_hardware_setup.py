from datetime import UTC, datetime, timedelta

import pytest
from starlette.requests import Request

from cortex.hardware import setup as hw


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    import lerobot.utils.constants as c

    monkeypatch.setattr(c, "HF_LEROBOT_CALIBRATION", tmp_path / "calibration")
    monkeypatch.setattr(hw, "list_ports", lambda: [{"device": "/dev/tty.usbmodem1", "usb": True}])
    hw.disarm()
    yield
    hw.disarm()


def calibrate(kind, dev):
    f = hw.calibration_file(kind, dev)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{}")
    return f


def test_calibration_paths_match_lerobot(tmp_path):
    assert hw.calibration_file("follower", "a1") == tmp_path / "calibration/robots/so_follower/a1.json"
    assert hw.calibration_file("leader", "l1") == tmp_path / "calibration/teleoperators/so_leader/l1.json"


def test_not_ready_until_every_gate_is_met(tmp_path):
    st = hw.status(tmp_path, enabled=False)
    assert not st["ready_for_motion"]
    assert any("CORTEX_ENABLE_HARDWARE" in m for m in st["missing"])
    assert any("port not chosen" in m for m in st["missing"])
    hw.save_config(tmp_path, hw.HardwareConfig(follower_port="/dev/tty.usbmodem1"))
    calibrate("follower", "cortex_follower")
    st = hw.status(tmp_path, enabled=True)
    assert st["missing"] == ["not armed: an operator must confirm the e-stop in the studio"]
    hw.arm(True, hw.ArmRequest(operator="pi", estop_confirmed=True))
    st = hw.status(tmp_path, enabled=True)
    assert st["ready_for_motion"] and st["armed"]["operator"] == "pi"
    assert st["teleop_missing"] == ["leader arm port not chosen", "leader arm 'cortex_leader' is not calibrated"]


def test_unplugged_port_blocks_motion(tmp_path):
    hw.save_config(tmp_path, hw.HardwareConfig(follower_port="/dev/tty.usbmodem9"))
    assert any("not connected" in m for m in hw.status(tmp_path, enabled=True)["missing"])


def test_arming_rules_expiry_and_setup_change(tmp_path):
    with pytest.raises(hw.HardwareRefused, match="CORTEX_ENABLE_HARDWARE"):
        hw.arm(False, hw.ArmRequest(operator="pi", estop_confirmed=True))
    with pytest.raises(hw.HardwareRefused, match="e-stop"):
        hw.arm(True, hw.ArmRequest(operator="pi", estop_confirmed=False))
    a = hw.arm(True, hw.ArmRequest(operator="pi", estop_confirmed=True))
    assert hw.current_arming() is a
    a.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert hw.current_arming() is None
    hw.arm(True, hw.ArmRequest(operator="pi", estop_confirmed=True))
    hw.save_config(tmp_path, hw.HardwareConfig(max_step_deg=1.0))
    assert hw.current_arming() is None  # a changed setup must be re-armed
    with pytest.raises(ValueError):
        hw.HardwareConfig(max_step_deg=10)  # never faster than 3 deg per tick


def test_hardware_endpoints_refuse_forwarded_and_remote_callers():
    from cortex.api.app import _local_human

    def req(host, headers=()):
        return Request({"type": "http", "client": (host, 1), "headers": [(k.encode(), v.encode()) for k, v in headers]})

    _local_human(req("127.0.0.1"))
    with pytest.raises(PermissionError):
        _local_human(req("10.0.0.5"))
    with pytest.raises(PermissionError):
        _local_human(req("127.0.0.1", [("x-forwarded-for", "8.8.8.8")]))  # tunneled
