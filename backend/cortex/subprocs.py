"""Can this backend start child processes? Training (ssh/rsync/sbatch), the sim workers and the studio's chat
engine all need asyncio subprocesses.

On Windows only the Proactor event loop supports them. uvicorn picks the Selector loop on Windows whenever it
runs the app in a child process (`--reload`, or `--workers` > 1), and then every create_subprocess_exec raises
NotImplementedError (#347).
"""

from __future__ import annotations

import asyncio
import sys

WHY = ("This backend can't start child processes: on Windows, uvicorn's --reload (or --workers) switches to an "
       "event loop without subprocess support. Restart it without --reload; training, simulations and the chat "
       "agent need it.")


def unavailable_reason(loop: asyncio.AbstractEventLoop | None = None) -> str | None:
    """None when asyncio subprocesses work on `loop` (default: the running loop), else why not."""
    if sys.platform != "win32":
        return None
    loop = loop or asyncio.get_running_loop()
    proactor = getattr(asyncio, "ProactorEventLoop", None)
    return None if proactor is not None and isinstance(loop, proactor) else WHY
