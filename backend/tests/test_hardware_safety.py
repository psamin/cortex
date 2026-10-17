import time

import numpy as np
import pytest

from cortex.hardware.so101_bridge import (
    HardwareNotEnabled, OperatorGate, SafetyController, SafetyLimits, SafetyViolation, SO101HardwareBridge,
)

HOME = np.array([0, -60, 60, 35, 0, 30.0])


def ctrl(fk=None):
    return SafetyController(SafetyLimits(), fk=fk)


def test_safety_rejects_limits_velocity_workspace_and_malformed():
    s = ctrl(fk=lambda q: np.array([0.2, 0.0, 0.05]) if q[0] < 50 else np.array([0.5, 0.0, 0.05]))
    s.check(HOME + np.r_[1, 0, 0, 0, 0, 0], HOME)  # small step is fine
    with pytest.raises(SafetyViolation, match="joint limit"):
        s.check(np.r_[HOME[:5], 150.0], HOME)
    with pytest.raises(SafetyViolation, match="exceeds"):
        s.check(HOME + np.r_[10, 0, 0, 0, 0, 0], HOME)
    far = HOME.copy()
    far[0] = 51
    with pytest.raises(SafetyViolation, match="workspace"):
        s.check(far, far)
    with pytest.raises(SafetyViolation, match="finite"):
        s.check(np.r_[HOME[:5], np.nan], HOME)


def test_watchdog_trips_and_latches():
    s = SafetyController(SafetyLimits(watchdog_timeout_s=0.01))
    time.sleep(0.03)
    assert not s.watchdog_ok()
    with pytest.raises(SafetyViolation, match="tripped"):
        s.check(HOME, HOME)


def test_bridge_refuses_without_preflight_and_never_unattended(monkeypatch, tmp_path):
    b = SO101HardwareBridge("/dev/null", OperatorGate(), ctrl())
    with pytest.raises(HardwareNotEnabled, match="preflight"):
        b.connect()
    cal = tmp_path / "cal.json"
    cal.write_text("{}")
    monkeypatch.setenv("CORTEX_ENABLE_HARDWARE", "1")
    gate = OperatorGate(operator_authorized=True, estop_confirmed=True, calibration_path=cal,
                        confirm_motion=lambda s: True)
    assert gate.unmet() == []
    with pytest.raises(HardwareNotEnabled, match="Unattended"):
        SO101HardwareBridge("/dev/null", gate, ctrl()).connect(unattended=True)
