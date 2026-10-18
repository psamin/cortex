"""Turning failures into messages people can act on.

An error Cortex raises on purpose (`UserError`, or a ValueError / PermissionError / KeyError raised
from Cortex's own code) is written for people and shown as-is. Known library and system failures
get a plain sentence. Anything else is an internal error: people see a short notice, and the raw
text goes in `detail`, never in the headline.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path

CORTEX_DIR = str(Path(__file__).resolve().parent)
INTERNAL = "Cortex hit an internal error. The technical details are attached."
DELIBERATE = (ValueError, PermissionError, LookupError, FileNotFoundError)
WRAPPERS = re.compile(r"^(?:[A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Refused|Exceeded|Unavailable)\s*:\s*)+")

KNOWN: list[tuple[str, str]] = [
    (r"interrupted by a backend restart", "It stopped because Cortex restarted. Run it again."),
    (r"Could not connect on port '([^']+)'",
     "Couldn't reach the arm on {0}. Check the USB cable and the port in the hardware setup."),
    (r"RepositoryNotFoundError|Repository Not Found|401 Client Error|404 Client Error.*huggingface",
     "That Hugging Face dataset wasn't found. Check the repo id; private datasets need HF_TOKEN in .env."),
    (r"ConnectError|ConnectTimeout|Temporary failure in name resolution|nodename nor servname|Network is unreachable",
     "Cortex couldn't reach the network. Check your connection and try again."),
    (r"No space left on device", "The disk is full. Free some space and run it again."),
    (r"CUDA out of memory|OutOfMemoryError", "The GPU ran out of memory."),
]


class UserError(ValueError):
    """A problem the user can fix. The message is written for people."""


def _known(raw: str) -> str | None:
    for pat, msg in KNOWN:
        m = re.search(pat, raw)
        if m:
            return msg.format(*m.groups())
    return None


def is_deliberate(exc: BaseException) -> bool:
    """Raised on purpose by Cortex code (not by a library deep inside a call)?"""
    if isinstance(exc, UserError):
        return True
    if not isinstance(exc, DELIBERATE):
        return False
    tb = traceback.extract_tb(exc.__traceback__)
    return bool(tb) and tb[-1].filename.startswith(CORTEX_DIR)


def describe(exc: BaseException) -> tuple[str, str | None]:
    """(message for people, technical detail or None)."""
    raw = str(exc) if hasattr(exc, "user_facing") else f"{type(exc).__name__}: {exc}"
    known = _known(raw)
    if known:
        return known, raw
    if getattr(exc, "user_facing", False) or is_deliberate(exc):
        msg = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
        return msg or type(exc).__name__, None
    return INTERNAL, raw


def friendly(raw: str | None) -> tuple[str, str | None]:
    """Same, from stored text only (old chat rows, worker output without a flag)."""
    raw = (raw or "").strip()
    if not raw:
        return "Unknown error.", None
    known = _known(raw)
    if known:
        return known, raw
    if not WRAPPERS.match(raw) or WRAPPERS.match(raw).group(0).rstrip().endswith("UserError:"):  # type: ignore[union-attr]
        return WRAPPERS.sub("", raw), None
    return INTERNAL, raw


class WorkerFailed(RuntimeError):
    """A worker subprocess failed. `user_facing` says whether its message was written for people."""

    def __init__(self, message: str, user_facing: bool = False):
        super().__init__(message)
        self.user_facing = user_facing


def worker_error(exc: BaseException) -> dict[str, object]:
    """What a worker emits on failure: its message, whether people can act on it, and the trace."""
    msg, detail = describe(exc)
    return {"error": msg if detail is None else f"{type(exc).__name__}: {exc}", "user_facing": detail is None,
            "trace": traceback.format_exc()[-2000:]}
