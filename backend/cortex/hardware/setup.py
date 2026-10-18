"""Real SO-101 setup for the studio: what is plugged in, what is configured, and whether
an operator has armed the hardware.

Real motion needs all of (spec section 5): CORTEX_ENABLE_HARDWARE=1 on this machine, a local
caller, a calibration file for the follower arm, and an operator who armed the hardware in
the studio and confirmed the physical e-stop is in reach. Arming lives in memory only and
expires, so a backend restart or a forgotten session disarms. Motion itself also needs a
live deadman and a per-run human approval; those are enforced by the arm worker and service.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ARM_TTL = timedelta(minutes=30)


class HardwareRefused(PermissionError):
    """A hardware precondition is unmet. Not something an approval card can fix."""

    code = "hardware_not_ready"
    approval_required = False


class CameraSpec(BaseModel):
    name: str = Field("front", pattern=r"^[a-z][a-z0-9_]{0,31}$",
                      description="policy key: observation.images.<name>")
    index_or_path: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30


class HardwareConfig(BaseModel):
    follower_port: str | None = None
    follower_id: str = Field("cortex_follower", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    leader_port: str | None = None
    leader_id: str = Field("cortex_leader", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    cameras: list[CameraSpec] = Field(default_factory=list)
    # Real joint speed cap, below the sim expert's 120 deg/s (spec 5.3): 2 deg per tick at 30 Hz.
    max_step_deg: float = Field(2.0, gt=0, le=3.0)


class Arming(BaseModel):
    operator: str
    estop_confirmed: bool
    armed_at: datetime
    expires_at: datetime


class ArmRequest(BaseModel):
    operator: str = Field(min_length=1, max_length=80)
    estop_confirmed: bool = Field(description="the independent physical e-stop is in reach and was tested")


ARMING: dict[str, Arming] = {}  # in memory only: a restart disarms


def config_path(data_dir: Path) -> Path:
    return data_dir / "hardware.json"


def load_config(data_dir: Path) -> HardwareConfig:
    p = config_path(data_dir)
    return HardwareConfig.model_validate_json(p.read_text()) if p.exists() else HardwareConfig()


def save_config(data_dir: Path, cfg: HardwareConfig) -> HardwareConfig:
    config_path(data_dir).write_text(cfg.model_dump_json(indent=2))
    ARMING.clear()  # a changed setup must be re-armed
    return cfg


def calibration_file(kind: Literal["follower", "leader"], device_id: str) -> Path:
    """Where LeRobot stores the calibration for an SO-101 follower or leader with this id."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    sub = ("robots", "so_follower") if kind == "follower" else ("teleoperators", "so_leader")
    return HF_LEROBOT_CALIBRATION / sub[0] / sub[1] / f"{device_id}.json"


def list_ports() -> list[dict[str, Any]]:
    """Serial ports, with USB devices (the SO-101 controller boards) marked as likely arms."""
    from serial.tools import list_ports as lp

    out = []
    for p in lp.comports():
        out.append({"device": p.device, "description": p.description or "", "usb": p.vid is not None,
                    "vid": f"{p.vid:04x}" if p.vid is not None else None,
                    "pid": f"{p.pid:04x}" if p.pid is not None else None, "serial_number": p.serial_number})
    return sorted(out, key=lambda x: (not x["usb"], x["device"]))


def driver_problem() -> str | None:
    try:
        import scservo_sdk  # noqa: F401
        from lerobot.robots.so_follower import SO101Follower  # noqa: F401
    except ImportError as e:
        return f"SO-101 driver missing ({e.name}): run `uv sync` in backend/ (needs lerobot[feetech])"
    return None


def current_arming() -> Arming | None:
    a = ARMING.get("operator")
    if a and a.expires_at <= datetime.now(UTC):
        ARMING.clear()
        return None
    return a


def arm(enabled: bool, req: ArmRequest) -> Arming:
    if not enabled:
        raise HardwareRefused("Hardware is disabled on this machine: set CORTEX_ENABLE_HARDWARE=1 in .env on the "
                              "computer wired to the arm, then restart the backend.")
    if not req.estop_confirmed:
        raise HardwareRefused("Confirm the independent physical e-stop is in reach and was tested before arming.")
    now = datetime.now(UTC)
    ARMING["operator"] = Arming(operator=req.operator, estop_confirmed=True, armed_at=now, expires_at=now + ARM_TTL)
    return ARMING["operator"]


def disarm() -> None:
    ARMING.clear()


def status(data_dir: Path, enabled: bool) -> dict[str, Any]:
    cfg = load_config(data_dir)
    drv = driver_problem()
    ports = [] if drv else list_ports()
    cal = {}
    for kind, dev in (("follower", cfg.follower_id), ("leader", cfg.leader_id)):
        f = calibration_file(kind, dev) if not drv else None  # type: ignore[arg-type]
        cal[kind] = {"id": dev, "path": str(f) if f else None, "exists": bool(f and f.exists())}
    armed = current_arming()
    missing = []
    if not enabled:
        missing.append("CORTEX_ENABLE_HARDWARE=1 is not set on this machine")
    if drv:
        missing.append(drv)
    if not cfg.follower_port:
        missing.append("follower arm port not chosen")
    elif not drv and cfg.follower_port not in {p["device"] for p in ports}:
        missing.append(f"follower port {cfg.follower_port} is not connected")
    if not cal["follower"]["exists"]:
        missing.append(f"follower arm '{cfg.follower_id}' is not calibrated")
    if armed is None:
        missing.append("not armed: an operator must confirm the e-stop in the studio")
    teleop_missing = []
    if not cfg.leader_port:
        teleop_missing.append("leader arm port not chosen")
    if not cal["leader"]["exists"]:
        teleop_missing.append(f"leader arm '{cfg.leader_id}' is not calibrated")
    return {"enabled": enabled, "driver_error": drv, "ports": ports, "config": json.loads(cfg.model_dump_json()),
            "calibration": cal, "armed": json.loads(armed.model_dump_json()) if armed else None,
            "ready_for_motion": not missing, "missing": missing, "teleop_missing": teleop_missing,
            "note": "Every real run also needs a per-run human approval and a held deadman in the studio."}
