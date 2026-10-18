"""Calibrate an SO-101 follower or leader arm with LeRobot's own `lerobot-calibrate`, its prompts
relayed to the studio.

Torque is off during calibration: the operator moves the joints by hand and presses Continue
(Enter) at each step, and LeRobot writes the calibration file. One hardware process runs at a
time on this machine. On Windows LeRobot reads Enter from the console (msvcrt), not from a pipe,
so calibration there must be run in a terminal.
"""

from __future__ import annotations

import asyncio
import os
import platform
import re
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from .setup import HardwareConfig, HardwareRefused

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
TABLE_ROW = re.compile(r"^\s*([a-z_]+)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+)\s*\|\s*(-?\d+)\s*$")


@dataclass
class HardwareProcess:
    kind: str
    proc: asyncio.subprocess.Process
    lines: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    prompt: str | None = None
    table: dict[str, dict[str, int]] = field(default_factory=dict)  # live joint ranges while recording
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    stopped: bool = False
    partial: str = ""  # output after the last newline (an input() prompt has none)

    @property
    def state(self) -> str:
        if self.proc.returncode is None:
            return "waiting_for_operator" if self.prompt else "running"
        if self.stopped:
            return "stopped"
        return "succeeded" if self.proc.returncode == 0 else "failed"


CURRENT: HardwareProcess | None = None


def busy() -> bool:
    return CURRENT is not None and CURRENT.proc.returncode is None


def calibrate_cmd(device: Literal["follower", "leader"], cfg: HardwareConfig) -> list[str]:
    base = [sys.executable, "-m", "lerobot.scripts.lerobot_calibrate"]
    if device == "follower":
        return [*base, "--robot.type=so101_follower", f"--robot.port={cfg.follower_port}",
                f"--robot.id={cfg.follower_id}"]
    return [*base, "--teleop.type=so101_leader", f"--teleop.port={cfg.leader_port}", f"--teleop.id={cfg.leader_id}"]


async def start_calibration(device: Literal["follower", "leader"], cfg: HardwareConfig, enabled: bool) -> dict:
    global CURRENT
    if not enabled:
        raise HardwareRefused("Hardware is disabled on this machine: set CORTEX_ENABLE_HARDWARE=1 in .env and "
                              "restart the backend.")
    port = cfg.follower_port if device == "follower" else cfg.leader_port
    if not port:
        raise HardwareRefused(f"Choose the {device} arm's port in the hardware setup first.")
    if platform.system() == "Windows":
        raise HardwareRefused("On Windows, LeRobot reads Enter from the console: run `lerobot-calibrate` in a "
                              "terminal. " + " ".join(calibrate_cmd(device, cfg)[3:]))
    if busy():
        raise HardwareRefused(f"Another hardware process is running ({CURRENT.kind}); stop it first.")  # type: ignore[union-attr]
    proc = await asyncio.create_subprocess_exec(
        *calibrate_cmd(device, cfg), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    CURRENT = HardwareProcess(kind=f"calibrate_{device}", proc=proc)
    asyncio.get_running_loop().create_task(_pump(CURRENT))
    return snapshot()


async def _pump(hp: HardwareProcess) -> None:
    """Relay output; `input()` prompts have no trailing newline, so read chunks, not lines."""
    assert hp.proc.stdout
    while chunk := await hp.proc.stdout.read(4096):
        hp.partial += ANSI.sub("", chunk.decode(errors="replace"))
        *done, hp.partial = hp.partial.split("\n")
        for line in (x.rstrip() for x in done):
            _take(hp, line)
        tail = hp.partial.strip()  # a pending input() prompt, not a half-received table row
        if tail and ("enter" in tail.lower() or tail.endswith((":", "..."))):
            hp.prompt = tail
    _take(hp, hp.partial.rstrip())
    await hp.proc.wait()
    hp.prompt, hp.finished_at = None, time.time()


def _take(hp: HardwareProcess, line: str) -> None:
    if not line.strip() or set(line.strip()) <= {"-"} or line.strip().startswith("NAME"):
        return
    m = TABLE_ROW.match(line)
    if m:
        hp.table[m[1]] = {"min": int(m[2]), "pos": int(m[3]), "max": int(m[4])}
        return
    hp.lines.append(line)
    if "press enter" in line.lower():
        hp.prompt = line.strip()


async def send_input(text: Literal["", "c"]) -> dict:
    if not busy():
        raise HardwareRefused("No hardware process is waiting for input.")
    assert CURRENT and CURRENT.proc.stdin
    if CURRENT.partial.strip():  # the answered prompt becomes its own log line
        _take(CURRENT, CURRENT.partial.rstrip())
        CURRENT.partial = ""
    CURRENT.prompt = None
    CURRENT.lines.append(f"> {'Enter' if text == '' else text}")
    CURRENT.proc.stdin.write((text + "\n").encode())
    await CURRENT.proc.stdin.drain()
    return snapshot()


async def stop() -> dict:
    if busy():
        assert CURRENT
        CURRENT.stopped = True
        CURRENT.proc.send_signal(signal.SIGINT)  # LeRobot disconnects cleanly on KeyboardInterrupt
        try:
            await asyncio.wait_for(CURRENT.proc.wait(), 5)
        except TimeoutError:
            CURRENT.proc.kill()
            await CURRENT.proc.wait()
    return snapshot()


def _error(lines: list[str]) -> str:
    """The final exception and its message, without the traceback."""
    starts = [i for i, x in enumerate(lines) if re.match(r"^[A-Za-z_.]*(Error|Exception)\b", x)]
    tail = lines[starts[-1]:] if starts else lines[-2:]
    return " ".join(x.strip() for x in tail if x.strip())[:500]


def snapshot() -> dict[str, Any] | None:
    hp = CURRENT
    if hp is None:
        return None
    return {"kind": hp.kind, "state": hp.state, "prompt": hp.prompt, "lines": list(hp.lines)[-40:],
            "error": _error(list(hp.lines)) if hp.state == "failed" else None,
            "joint_ranges": hp.table, "returncode": hp.proc.returncode, "started_at": hp.started_at,
            "finished_at": hp.finished_at}
